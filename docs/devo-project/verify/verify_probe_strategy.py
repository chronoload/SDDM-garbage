"""对比三种 delay 探索策略：随机 / 离散爬山 / 抛物线插值

## 背景

判决实验已确立：
- 起作用的是 delay 的**位置**而非游走（冻在终态 ≈ 可塑 ≪ 冻结）
- 随机扰动与不探索无差异（t=-1.24 / -1.69，不显著）

而可公度性实验进一步发现：可塑组 delay **边缘分布仍接近均匀**
（std 5.37 vs 均匀分布 5.77），说明不是收敛到离散值，而是
**高维向量的逐坐标微调** —— 每维移动方向不同，边缘分布不变。

## 待检验的改进

当前 `probe_delay` 用离散爬山：向对齐度更高的一侧跳 ±window。
这在"高维逐坐标微调"的场景下过于粗糙 —— 每次只能跳固定步长，
无法精细定位，且容易在最优值附近震荡。

**抛物线插值**：用三个采样点 (d-w, d, d+w) 的对齐度拟合二次曲线，
直接跳到估计的极值点：

    denom = f(-w) - 2f(0) + f(+w)
    x* = w·(f(-w) - f(+w)) / (2·denom)      [denom < 0，即凹函数时有效]
    denom ≥ 0（单调或凸）时退化为爬山

这是标准三点极值搜索，与爬山**计算成本相同**（同样采样三个点），
但移动更精准。

## 实验

三策略 × 两种预算：
- 无预算：每 block 自由更新
- 距离预算：累计移动距离上限（模拟 MAX_DELAY_PROBES 的资源约束）

判据：频谱熵（低 = 动力学更有序）
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
        budget=None, g=G, w=1.0, delay_block=20, noise=0.10,
        N=N, in_degree=IN_DEGREE, tau_max=TAU_MAX):
    """strategy: 'random' | 'hill' | 'parabolic'
    budget: 每条边累计移动距离上限（None = 无限制）
    """
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
    moved = np.zeros(E)          # 累计移动距离

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

            if strategy == "random":
                # 匹配爬山的移动频率（约 2/3 的边会移动）
                mv = rng.random(E) < (2.0 / 3.0)
                delta = mv * rng.choice([-1.0, 1.0], size=E) * w
            elif strategy == "hill":
                bp = (cp1 > c0) & (cp1 >= cm1)
                bm = (cm1 > c0) & (cm1 > cp1)
                delta = w * (bp.astype(float) - bm.astype(float))
            else:  # parabolic
                denom = cm1 - 2.0 * c0 + cp1
                concave = denom < -1e-12
                with np.errstate(divide="ignore", invalid="ignore"):
                    xstar = np.where(concave,
                                     w * (cm1 - cp1) / (2.0 * denom), 0.0)
                xstar = np.clip(np.nan_to_num(xstar), -w, w)
                bp = (cp1 > c0) & (cp1 >= cm1)
                bm = (cm1 > c0) & (cm1 > cp1)
                hill = w * (bp.astype(float) - bm.astype(float))
                delta = np.where(concave, xstar, hill)

            # 预算约束
            if budget is not None:
                room = np.maximum(0.0, budget - moved)
                delta = np.clip(delta, -room, room)

            delay = np.clip(delay + delta, 1.0, float(tau_max))
            moved += np.abs(delta)

    return {"act": act, "delay": delay, "moved": moved}


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 1e-12 else 0.0


if __name__ == "__main__":
    seeds = list(range(10))

    print("=" * 80)
    print("delay 探索策略对比：随机 / 离散爬山 / 抛物线插值")
    print("=" * 80)
    print(f"N={N}, 入度={IN_DEGREE}, gain={G}, {len(seeds)} seeds")
    print("判据：频谱熵（低 = 动力学更有序）\n")

    for budget in [None, 4.0]:
        label = "无预算" if budget is None else f"距离预算 {budget:.0f} 步"
        print(f"【{label}】")
        print(f"  {'策略':>10} {'频谱熵':>16} {'终态std':>9} {'平均移动':>9}")
        print("  " + "-" * 50)

        scores = {}
        for strat in ["random", "hill", "parabolic"]:
            ses, stds, mvds = [], [], []
            for s in seeds:
                r = run(seed=s, strategy=strat, budget=budget)
                ses.append(spectral_entropy(r["act"][3000:]))
                stds.append(float(np.std(r["delay"])))
                mvds.append(float(np.mean(r["moved"])))
            scores[strat] = np.array(ses)
            name = {"random": "随机", "hill": "离散爬山",
                    "parabolic": "抛物线插值"}[strat]
            print(f"  {name:>10} {np.mean(ses):>10.3f} ± {np.std(ses):<5.3f} "
                  f"{np.mean(stds):>9.3f} {np.mean(mvds):>9.2f}")

        print(f"  {'-' * 50}")
        t_pr = welch_t(scores["parabolic"], scores["random"])
        t_ph = welch_t(scores["hill"], scores["parabolic"])  # 正=抛物线更低(更好)
        t_hr = welch_t(scores["hill"], scores["random"])
        print(f"  抛物线 vs 随机   t={welch_t(scores['random'], scores['parabolic']):+.2f}"
              f"  (正=抛物线更优)")
        print(f"  抛物线 vs 爬山   t={t_ph:+.2f}  (正=抛物线更优)")
        print(f"  爬山   vs 随机   t={welch_t(scores['random'], scores['hill']):+.2f}"
              f"  (正=爬山更优)")
        print()

    print("=" * 80)
    print("注：|t| > 2 ≈ p < 0.05")
    print("=" * 80)
