"""补充：平滑对齐度能否让抛物线插值优于爬山？

## 动机

上一轮对比显示抛物线插值不优于离散爬山（无预算 t=-1.36，方向不利；
预算受限 t=+0.10，无差异）。

推测原因：对齐度是**瞬时**计算的（单步 x_delayed × x_dst），噪声极大。
抛物线插值对噪声敏感 —— 三点拟合的极值位置会被噪声带偏；
而离散爬山只比较大小关系，对噪声更鲁棒。

若此推测成立，则**先平滑对齐度**（EMA）再插值，应能恢复抛物线的优势。

## 实验

四配置对比（无预算）：
1. 爬山 + 瞬时
2. 爬山 + EMA
3. 抛物线 + 瞬时
4. 抛物线 + EMA

若 4 显著优于 2，则采用"EMA + 抛物线"；
否则维持当前的爬山（瞬时或 EMA，取更优者）。
"""
import numpy as np

TAU_MAX = 20
N = 32
IN_DEGREE = 6
G = 0.18


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


def run(seed=0, steps=6000, warmup=1500, strategy="hill",
        ema=0.0, g=G, w=1.0, delay_block=20, noise=0.10,
        N=N, in_degree=IN_DEGREE, tau_max=TAU_MAX):
    """ema=0 表示瞬时；ema>0 表示对该对齐度做 EMA 平滑"""
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for i in range(N):
        for j in rng.choice(N, size=in_degree, replace=False):
            src.append(j)
            dst.append(i)
    src, dst = np.array(src), np.array(dst)
    E = len(src)

    gain = np.full(E, g)
    delay = rng.integers(1, tau_max + 1, size=E).astype(float)
    e_m1 = np.zeros(E)
    e_0 = np.zeros(E)
    e_p1 = np.zeros(E)
    init = False

    hist = np.zeros((tau_max + 1, N))
    x = rng.normal(0, 0.1, size=N)
    act = np.zeros(steps)

    for t in range(steps):
        d_int = np.clip(np.round(delay).astype(int), 1, tau_max)
        agg = np.zeros(N)
        np.add.at(agg, dst, gain * hist[d_int, src])
        x = np.tanh(agg + rng.normal(0, noise, size=N))
        act[t] = x.mean()
        hist[t % (tau_max + 1)] = x

        if t >= warmup and t % delay_block == 0 and t > tau_max + 1:
            dm = np.clip(d_int - 1, 1, tau_max)
            dp = np.clip(d_int + 1, 1, tau_max)
            cm1 = hist[(t - dm) % (tau_max + 1), src] * x[dst]
            c0 = hist[(t - d_int) % (tau_max + 1), src] * x[dst]
            cp1 = hist[(t - dp) % (tau_max + 1), src] * x[dst]

            if ema > 0:
                if not init:
                    e_m1, e_0, e_p1 = cm1.copy(), c0.copy(), cp1.copy()
                    init = True
                else:
                    e_m1 = ema * e_m1 + (1 - ema) * cm1
                    e_0 = ema * e_0 + (1 - ema) * c0
                    e_p1 = ema * e_p1 + (1 - ema) * cp1
                a_m1, a_0, a_p1 = e_m1, e_0, e_p1
            else:
                a_m1, a_0, a_p1 = cm1, c0, cp1

            if strategy == "hill":
                bp = (a_p1 > a_0) & (a_p1 >= a_m1)
                bm = (a_m1 > a_0) & (a_m1 > a_p1)
                delta = w * (bp.astype(float) - bm.astype(float))
            else:  # parabolic
                denom = a_m1 - 2.0 * a_0 + a_p1
                concave = denom < -1e-12
                with np.errstate(divide="ignore", invalid="ignore"):
                    xstar = np.where(concave,
                                     w * (a_m1 - a_p1) / (2.0 * denom), 0.0)
                xstar = np.clip(np.nan_to_num(xstar), -w, w)
                bp = (a_p1 > a_0) & (a_p1 >= a_m1)
                bm = (a_m1 > a_0) & (a_m1 > a_p1)
                hill = w * (bp.astype(float) - bm.astype(float))
                delta = np.where(concave, xstar, hill)

            delay = np.clip(delay + delta, 1.0, float(tau_max))

    return {"act": act, "delay": delay}


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 1e-12 else 0.0


if __name__ == "__main__":
    seeds = list(range(10))

    print("=" * 76)
    print("平滑对齐度 × 探索策略（无预算，10 seeds）")
    print("=" * 76)
    print("判据：频谱熵（低 = 更有序）\n")
    print(f"  {'配置':>20} {'频谱熵':>18}")
    print("  " + "-" * 40)

    res = {}
    for strat, sname in [("hill", "爬山"), ("parabolic", "抛物线")]:
        for ema, ename in [(0.0, "瞬时"), (0.9, "EMA 0.9")]:
            ses = []
            for s in seeds:
                r = run(seed=s, strategy=strat, ema=ema)
                ses.append(spectral_entropy(r["act"][3000:]))
            key = f"{sname}+{ename}"
            res[key] = np.array(ses)
            print(f"  {key:>20} {np.mean(ses):>10.3f} ± {np.std(ses):<5.3f}")

    print("  " + "-" * 40)
    print("显著性（正 = 前者更差，即后者更优）")
    print(f"  抛物线+EMA  vs  爬山+瞬时    "
          f"t={welch_t(res['抛物线+瞬时'], res['抛物线+EMA 0.9']):+.2f}"
          f"  ← 平滑是否救活抛物线")
    print(f"  抛物线+EMA  vs  爬山+EMA     "
          f"t={welch_t(res['爬山+EMA 0.9'], res['抛物线+EMA 0.9']):+.2f}")
    print(f"  爬山+EMA    vs  爬山+瞬时    "
          f"t={welch_t(res['爬山+瞬时'], res['爬山+EMA 0.9']):+.2f}"
          f"  ← 平滑对爬山的影响")
    print()
    print("=" * 76)
    best = min(res, key=lambda k: res[k].mean())
    print(f"最优配置：{best}  (频谱熵 {res[best].mean():.3f})")
    print("=" * 76)
