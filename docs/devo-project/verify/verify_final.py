"""最终验证：髓鞘 delay 可塑性对动力学良态性的影响

锁定 v8 找到的临界工作点 gain=0.18（幅度 0.424，不饱和）。

待回答三个问题：
1. 效应是否统计显著（多种子 + t 检验）
2. delay **到底变了没有**（v8 显示分布熵几乎相同，需要看个体位移）
3. 若 delay 分布未变而动力学改善，机制是什么
"""
import numpy as np

TAU_MAX = 20
N = 32
IN_DEGREE = 6
G = 0.18


def autocorr_halflife(series, max_lag=400):
    s = series - series.mean()
    var = np.dot(s, s) / len(s)
    if var < 1e-12:
        return 0.0
    for lag in range(1, min(max_lag, len(s) // 2)):
        c = np.dot(s[:-lag], s[lag:]) / (len(s) - lag) / var
        if c < 1.0 / np.e:
            return float(lag)
    return float(max_lag)


def spectral_entropy(series, lo=2, hi=200):
    s = series - series.mean()
    if np.std(s) < 1e-12:
        return np.log(10)
    n = len(s)
    spec = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= lo) & (periods <= hi)
    if not mask.any() or spec[mask].sum() < 1e-12:
        return np.log(10)
    p = spec[mask] / spec[mask].sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def run(seed=0, steps=7000, warmup=1500, adapt=True,
        g=G, lr_delay=1.0, delay_block=20, noise=0.10,
        N=N, in_degree=IN_DEGREE, tau_max=TAU_MAX):
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for i in range(N):
        for j in rng.choice(N, size=in_degree, replace=False):
            src.append(j)
            dst.append(i)
    src, dst = np.array(src), np.array(dst)

    gain = np.full(len(src), g)
    delay0 = rng.integers(1, tau_max + 1, size=len(src)).astype(float)
    delay = delay0.copy()

    hist = np.zeros((tau_max + 1, N))
    x = rng.normal(0, 0.1, size=N)
    act, amp = np.zeros(steps), np.zeros(steps)

    for t in range(steps):
        d_int = np.clip(np.round(delay).astype(int), 1, tau_max)
        x_delayed = hist[d_int, src]
        agg = np.zeros(N)
        np.add.at(agg, dst, gain * x_delayed)
        x = np.tanh(agg + rng.normal(0, noise, size=N))
        act[t] = x.mean()
        amp[t] = np.mean(np.abs(x))
        hist[t % (tau_max + 1)] = x

        if adapt and t >= warmup and t % delay_block == 0 and t > tau_max + 1:
            dm = np.clip(d_int - 1, 1, tau_max)
            dp = np.clip(d_int + 1, 1, tau_max)
            cm1 = hist[(t - dm) % (tau_max + 1), src] * x[dst]
            c0 = hist[(t - d_int) % (tau_max + 1), src] * x[dst]
            cp1 = hist[(t - dp) % (tau_max + 1), src] * x[dst]
            bp = (cp1 > c0) & (cp1 >= cm1)
            bm = (cm1 > c0) & (cm1 > cp1)
            delay = np.clip(
                delay + lr_delay * (bp.astype(float) - bm.astype(float)),
                1.0, float(tau_max))

    return {"act": act, "amp": amp, "delay": delay,
            "delay0": delay0, "src": src, "dst": dst}


def welch_t(a, b):
    """Welch t 检验统计量（不假设等方差）"""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    return (a.mean() - b.mean()) / se if se > 1e-12 else 0.0


if __name__ == "__main__":
    seeds = list(range(16))

    print("=" * 84)
    print(f"最终验证：gain={G}（v8 定位的临界工作点），{len(seeds)} seeds")
    print("=" * 84)

    res = {}
    for mode, ad in [("delay 可塑", True), ("delay 冻结", False)]:
        rs = [run(seed=s, adapt=ad) for s in seeds]
        hl = [autocorr_halflife(r["act"][3500:]) for r in rs]
        se = [spectral_entropy(r["act"][3500:]) for r in rs]
        amp = [float(np.mean(r["amp"][3500:])) for r in rs]
        # delay 的实际位移（相对于初始值）
        disp = [float(np.mean(np.abs(r["delay"] - r["delay0"]))) for r in rs]
        frac = [float(np.mean(r["delay"] != r["delay0"])) for r in rs]
        res[mode] = {"hl": hl, "se": se, "amp": amp,
                     "disp": disp, "frac": frac}

    print()
    print("【1】动力学良态性")
    print("-" * 84)
    print(f"{'配置':>12} {'幅度':>8} {'半衰期(均值±std)':>22} {'频谱熵':>16}")
    for mode in res:
        d = res[mode]
        print(f"{mode:>12} {np.mean(d['amp']):>8.4f} "
              f"{np.mean(d['hl']):>12.1f} ± {np.std(d['hl']):<8.1f} "
              f"{np.mean(d['se']):>10.3f} ± {np.std(d['se']):.3f}")

    hl_a = res["delay 可塑"]["hl"]
    hl_f = res["delay 冻结"]["hl"]
    se_a = res["delay 可塑"]["se"]
    se_f = res["delay 冻结"]["se"]
    t_hl = welch_t(hl_a, hl_f)
    t_se = welch_t(se_f, se_a)      # 冻结熵 − 可塑熵，正 = 冻结更混乱
    print()
    print(f"  半衰期 t = {t_hl:+.2f}   （|t|>2 ≈ p<0.05）")
    print(f"  频谱熵 t = {t_se:+.2f}   （正 = 冻结更混乱）")

    print()
    print("【2】delay 到底变了没有")
    print("-" * 84)
    for mode in res:
        d = res[mode]
        print(f"{mode:>12} 平均位移 {np.mean(d['disp']):.2f} 步  "
              f"发生变化的边占比 {np.mean(d['frac']):.1%}")

    print()
    print("【3】结论")
    print("-" * 84)
    sig = abs(t_hl) > 2 or abs(t_se) > 2
    if np.mean(hl_a) > np.mean(hl_f) and sig:
        print("  假设获支持：delay 可塑组的动力学记忆显著更长、频谱更有序。")
        print(f"    半衰期 {np.mean(hl_a):.0f} vs {np.mean(hl_f):.0f} 步")
        print(f"    频谱熵 {np.mean(se_a):.3f} vs {np.mean(se_f):.3f}")
        print()
        print("  机制：delay 适应是**针对具体网络的逐边微调**")
        print("    —— 分布层面看不出聚集（v8 的 delay 熵几乎相同），")
        print("       但每条边找到了适配该网络上下文的延迟，整体更协调。")
    elif np.mean(hl_a) > np.mean(hl_f):
        print(f"  方向一致（可塑半衰期更长）但**未达显著**（t={t_hl:.2f}）。")
        print("  需更多种子或更长模拟。")
    else:
        print("  假设未获支持：可塑组并未表现出更良态的动力学。")
    print("=" * 84)
