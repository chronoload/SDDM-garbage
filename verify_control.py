"""关键对照：效应来自"适应性"还是"扰动本身"？

v_final 显示 delay 可塑组动力学显著更良态（半衰期 t=3.98，频谱熵 t=11.01）。
但存在一个混淆：**可塑组的 delay 一直在动，冻结组完全静止**。
效应可能只是"任何扰动都打破初始随机配置"，而非"适应性爬山有效"。

三配置对照：
- ``adapt``   ：爬山法（向更高对齐度移动）
- ``random``  ：随机游走（同等幅度的扰动，但无方向性）
- ``frozen``  ：完全冻结

判定：
- adapt > random ≈ frozen → 效应来自**适应性** ✓
- adapt ≈ random > frozen → 效应只是**扰动**，非适应 ✗
- 三者无差异 → 原结果是噪声
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


def run(seed=0, steps=7000, warmup=1500, mode="adapt",
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

        if mode != "frozen" and t >= warmup \
                and t % delay_block == 0 and t > tau_max + 1:
            if mode == "adapt":
                # 爬山：向对齐度更高的方向移动
                dm = np.clip(d_int - 1, 1, tau_max)
                dp = np.clip(d_int + 1, 1, tau_max)
                cm1 = hist[(t - dm) % (tau_max + 1), src] * x[dst]
                c0 = hist[(t - d_int) % (tau_max + 1), src] * x[dst]
                cp1 = hist[(t - dp) % (tau_max + 1), src] * x[dst]
                bp = (cp1 > c0) & (cp1 >= cm1)
                bm = (cm1 > c0) & (cm1 > cp1)
                step = lr_delay * (bp.astype(float) - bm.astype(float))
            else:  # random：同等幅度的无方向扰动
                # 匹配爬山的移动概率：爬山每步以约 2/3 概率移动 ±1
                mv = rng.random(len(src)) < (2.0 / 3.0)
                step = mv * rng.choice([-1.0, 1.0], size=len(src)) * lr_delay
            delay = np.clip(delay + step, 1.0, float(tau_max))

    return {"act": act, "amp": amp, "delay": delay, "delay0": delay0}


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    return (a.mean() - b.mean()) / se if se > 1e-12 else 0.0


if __name__ == "__main__":
    seeds = list(range(14))

    print("=" * 88)
    print("对照实验：适应性爬山 vs 随机扰动 vs 完全冻结")
    print("=" * 88)
    print(f"gain={G}, {len(seeds)} seeds, 前 1500 步一致后分流\n")

    res = {}
    for mode in ["adapt", "random", "frozen"]:
        rs = [run(seed=s, mode=mode) for s in seeds]
        res[mode] = {
            "hl": [autocorr_halflife(r["act"][3500:]) for r in rs],
            "se": [spectral_entropy(r["act"][3500:]) for r in rs],
            "amp": [float(np.mean(r["amp"][3500:])) for r in rs],
            "disp": [float(np.mean(np.abs(r["delay"] - r["delay0"])))
                     for r in rs],
        }

    names = {"adapt": "适应性爬山", "random": "随机扰动", "frozen": "完全冻结"}
    print(f"{'配置':>12} {'幅度':>8} {'半衰期':>16} {'频谱熵':>16} "
          f"{'delay位移':>10}")
    print("-" * 88)
    for mode in ["adapt", "random", "frozen"]:
        d = res[mode]
        print(f"{names[mode]:>12} {np.mean(d['amp']):>8.4f} "
              f"{np.mean(d['hl']):>9.1f} ± {np.std(d['hl']):<5.1f} "
              f"{np.mean(d['se']):>9.3f} ± {np.std(d['se']):<5.3f} "
              f"{np.mean(d['disp']):>10.2f}")

    print()
    print("显著性检验（Welch t，|t|>2 ≈ p<0.05）")
    print("-" * 88)
    print(f"  爬山 vs 冻结   半衰期 t={welch_t(res['adapt']['hl'], res['frozen']['hl']):+.2f}"
          f"   频谱熵 t={welch_t(res['frozen']['se'], res['adapt']['se']):+.2f}")
    print(f"  爬山 vs 随机   半衰期 t={welch_t(res['adapt']['hl'], res['random']['hl']):+.2f}"
          f"   频谱熵 t={welch_t(res['random']['se'], res['adapt']['se']):+.2f}")
    print(f"  随机 vs 冻结   半衰期 t={welch_t(res['random']['hl'], res['frozen']['hl']):+.2f}"
          f"   频谱熵 t={welch_t(res['frozen']['se'], res['random']['se']):+.2f}")

    print()
    print("=" * 88)
    hl = {m: np.mean(res[m]["hl"]) for m in res}
    se = {m: np.mean(res[m]["se"]) for m in res}
    t_ar = welch_t(res["adapt"]["hl"], res["random"]["hl"])
    t_rf = welch_t(res["random"]["hl"], res["frozen"]["hl"])

    if hl["adapt"] > hl["random"] and abs(t_ar) > 2:
        print("结论：效应来自**适应性** ✓")
        print(f"  爬山 ({hl['adapt']:.0f}) > 随机扰动 ({hl['random']:.0f}) > "
              f"冻结 ({hl['frozen']:.0f})")
        print("  随机扰动不足以产生效应，方向性（向对齐度更高的方向）是关键。")
    elif abs(t_rf) > 2 and abs(t_ar) <= 2:
        print("结论：效应只是**扰动**，非适应 ✗")
        print(f"  爬山 ({hl['adapt']:.0f}) ≈ 随机扰动 ({hl['random']:.0f}) > "
              f"冻结 ({hl['frozen']:.0f})")
        print("  → delay 适应的**方向性**没有价值，任何扰动都行。")
        print("  → 那么'髓鞘 delay 是镇定器'的强主张不成立，")
        print("     应弱化为'维持 delay 的可变性本身有价值'。")
    else:
        print(f"结论：三者关系不明确（爬山 {hl['adapt']:.0f} / "
              f"随机 {hl['random']:.0f} / 冻结 {hl['frozen']:.0f}）")
    print("=" * 88)
