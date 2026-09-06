"""髓鞘 delay 可塑性 = 镇定器（第八版：临界区细扫）

## v7 的结果

gain=0.14 出现微小差异（可塑 R² 0.893 vs 冻结 0.860），但活动幅度
仅 0.066 —— **内部耦合太弱，动力学被外部周期驱动主导**，测的是
"跟随驱动的能力"而非"内部稳定性"。

## 核心矛盾

- 亚临界（环路增益 << 1）：无自主动力学，被驱动/噪声主导
- 超临界（环路增益 > 1）：正反馈饱和（v6 观察到 0.115 → 0.974 的陡变）

**临界带极窄**，v6 的扫描步长（0.6 → 1.0）太粗，直接跨过了。

环路增益粗估 ≈ in_degree × g × cos(φ)。in_degree=6，若全部同相则
临界 g ≈ 1/6 ≈ 0.167；延迟引入相位差会抬高实际临界点。

## v8 设计

1. **细扫 g ∈ [0.14, 0.26]**（步长 0.02），定位临界带
2. **去掉外部周期驱动**，仅用噪声 —— 测纯内部自主动力学
3. 指标：幅度（活/不饱和）、自相关半衰期（内部记忆）、频谱熵

若临界带内 delay 可塑仍能显著优于冻结，则假设成立；
若依然无差异，则诚实报告：机制成立但稳定性后果未能在本平台验证。
"""
import numpy as np

TAU_MAX = 20
N = 32
IN_DEGREE = 6


def autocorr_halflife(series, max_lag=300):
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


def run(seed=0, steps=6000, warmup=1500, adapt=True,
        g=0.18, lr_delay=1.0, delay_block=20,
        noise=0.10, pulse=0.0,
        N=N, in_degree=IN_DEGREE, tau_max=TAU_MAX):
    """无外部周期驱动，纯内部延迟循环动力学"""
    rng = np.random.default_rng(seed)

    src, dst = [], []
    for i in range(N):
        for j in rng.choice(N, size=in_degree, replace=False):
            src.append(j)
            dst.append(i)
    src, dst = np.array(src), np.array(dst)

    gain = np.full(len(src), g)
    delay = rng.integers(1, tau_max + 1, size=len(src)).astype(float)

    hist = np.zeros((tau_max + 1, N))
    x = rng.normal(0, 0.1, size=N)
    act, amp = np.zeros(steps), np.zeros(steps)

    for t in range(steps):
        d_int = np.clip(np.round(delay).astype(int), 1, tau_max)
        x_delayed = hist[d_int, src]

        agg = np.zeros(N)
        np.add.at(agg, dst, gain * x_delayed)

        u = rng.normal(0, noise, size=N)      # 仅噪声，无周期驱动
        x = np.tanh(agg + u)

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

    return {"act": act, "amp": amp, "delay": delay}


if __name__ == "__main__":
    seeds = list(range(8))

    print("=" * 84)
    print("第八版：临界区细扫，无周期驱动，纯内部延迟循环动力学")
    print("=" * 84)
    print(f"N={N}, 入度={IN_DEGREE}, gain 固定, 前 1500 步一致后分流")
    print("幅度目标区间 [0.15, 0.75]：太低=死亡/被噪声主导，>0.9=饱和\n")

    print(f"{'gain':>6} {'配置':>6} {'幅度':>8} {'饱和':>7} {'半衰期':>8} "
          f"{'频谱熵':>8} {'delay熵':>8}")
    print("-" * 84)

    rows = []
    for g in [0.14, 0.16, 0.18, 0.20, 0.24, 0.28]:
        for mode, ad in [("可塑", True), ("冻结", False)]:
            rs = [run(seed=s, adapt=ad, g=g) for s in seeds]
            amps, sats, hls, ses, dens = [], [], [], [], []
            all_delay = []
            for r in rs:
                a, am = r["act"][3000:], r["amp"][3000:]
                amps.append(float(np.mean(am)))
                sats.append(float(np.mean(am > 0.9)))
                hls.append(autocorr_halflife(a))
                ses.append(spectral_entropy(a))
                all_delay.append(r["delay"])
            d = np.concatenate(all_delay)
            hd, _ = np.histogram(d, bins=12, range=(1, TAU_MAX + 1))
            p = hd / hd.sum()
            dens.append(float(-(p * np.log(p + 1e-12)).sum()))
            row = (g, mode, np.mean(amps), np.mean(sats), np.mean(hls),
                   np.mean(ses), np.mean(dens))
            rows.append(row)
            print(f"{g:>6.2f} {mode:>6} {np.mean(amps):>8.4f} "
                  f"{np.mean(sats):>7.3f} {np.mean(hls):>8.1f} "
                  f"{np.mean(ses):>8.3f} {np.mean(dens):>8.3f}")
        print()

    print("=" * 84)
    print("临界带内对比（幅度在 [0.15,0.75] 且饱和<0.3 的 gain）")
    print("-" * 84)
    usable = sorted({r[0] for r in rows
                     if 0.15 <= r[2] <= 0.75 and r[3] < 0.3})
    if not usable:
        print("  无可用工作点：所有 gain 要么死亡要么饱和。")
        print("  → 结论：本平台无法稳定停留在临界带，稳定性后果未能验证。")
    else:
        for g in usable:
            a = [r for r in rows if r[0] == g and r[1] == "可塑"][0]
            f = [r for r in rows if r[0] == g and r[1] == "冻结"][0]
            print(f"  gain={g:.2f}: 半衰期 可塑 {a[4]:.1f} vs 冻结 {f[4]:.1f} "
                  f"(Δ={a[4]-f[4]:+.1f})  "
                  f"频谱熵 可塑 {a[5]:.3f} vs 冻结 {f[5]:.3f} (Δ={f[5]-a[5]:+.3f})  "
                  f"delay熵 可塑 {a[6]:.3f} vs 冻结 {f[6]:.3f}")
    print("=" * 84)
