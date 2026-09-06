"""按环路长度分组检验 Coombes 2026 的可公度性预言 τ = pT

## 背景与待解释的谜题

Coombes 等 2026（arXiv 2606.21508，已核实）：活动依赖的髓鞘可塑性会
**选择可公度的延迟-周期关系**，把异质延迟组织成**离散的延迟-周期类**。

但 v8 实验发现：可塑组 delay 的**边缘分布熵几乎不变**（2.434 vs 冻结
2.436），而 94.9% 的边都移动了平均 6.08 步，动力学显著更有序。

判决实验进一步显示：把 delay 冻在可塑终态，有序性完全保留
（5.742 ≈ 可塑 5.803 ≪ 冻结 6.505）→ 起作用的是**位置**不是游走。

**待检验的解读**：结构是**条件分布**而非边缘分布。不同环长的环需要
不同的最优延迟；各自收敛后，混合的边缘分布仍保持分散。

## 本实验

- **检验 A（不依赖周期）**：按边所在最短环的长度分组，组内 delay 集中度
  预测：可塑组**组内** std 显著小于冻结组，且组内比值 > 全局比值
- **检验 B（Coombes 原预言）**：环总延迟 Στ 对主导周期 T 取模后的
  相位聚集度（Rayleigh R）。物理依据：信号绕环一圈回来应同相
- **诊断优先**：B 依赖明确的主导周期 T，先测主频显著性
"""
import numpy as np
from collections import deque

TAU_MAX = 20
N = 32
IN_DEGREE = 6
G = 0.18


# ---------------------------------------------------------------------------
# 模拟
# ---------------------------------------------------------------------------

def run(seed=0, steps=6000, warmup=1500, adapt=True,
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
    delay = rng.integers(1, tau_max + 1, size=len(src)).astype(float)

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

    return {"act": act, "delay": delay, "src": src, "dst": dst}


# ---------------------------------------------------------------------------
# 图论
# ---------------------------------------------------------------------------

def bfs_dist(adj, start, target, N):
    """从 start 到 target 的最短有向距离；不可达返回 -1"""
    if start == target:
        return 0
    dist = [-1] * N
    dist[start] = 0
    q = deque([start])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y == target:
                return dist[x] + 1
            if dist[y] < 0:
                dist[y] = dist[x] + 1
                q.append(y)
    return -1


def edge_cycle_lengths(src, dst, N):
    """每条边所在的**最短**有向环长度；不在任何环上则为 0

    对边 (u→v)：环长 = 1 + 从 v 回到 u 的最短距离
    """
    adj = [[] for _ in range(N)]
    for u, v in zip(src, dst):
        adj[u].append(v)
    res = np.zeros(len(src), dtype=int)
    for i, (u, v) in enumerate(zip(src, dst)):
        d = bfs_dist(adj, v, u, N)
        if d >= 0:
            res[i] = 1 + d
    return res


def shortest_cycle_from(src, dst, N, start, max_len=10):
    """找从 start 回到 start 的最短有向环，返回边索引列表"""
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(zip(src, dst)):
        adj[u].append((v, i))

    pnode = [-1] * N
    pedge = [-1] * N
    dist = [-1] * N
    dist[start] = 0
    q = deque([start])

    while q:
        x = q.popleft()
        if dist[x] >= max_len:
            continue
        for (y, eidx) in adj[x]:
            if y == start:
                edges = [eidx]
                cur = x
                while cur != start:
                    edges.append(pedge[cur])
                    cur = pnode[cur]
                return edges[::-1]
            if dist[y] < 0:
                dist[y] = dist[x] + 1
                pnode[y] = x
                pedge[y] = eidx
                q.append(y)
    return None


def all_shortest_cycles(src, dst, N, max_len=10):
    """每个节点一个最短环，去重"""
    cycles, seen = [], set()
    for s in range(N):
        c = shortest_cycle_from(src, dst, N, s, max_len=max_len)
        if c is None:
            continue
        key = tuple(sorted(c))
        if key in seen:
            continue
        seen.add(key)
        cycles.append(c)
    return cycles


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def dominant_period_and_power(series, lo=4, hi=200):
    """(主导周期, 主频功率占带内总功率比例)"""
    s = series - series.mean()
    if np.std(s) < 1e-12:
        return 0.0, 0.0
    n = len(s)
    spec = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= lo) & (periods <= hi)
    if not mask.any() or spec[mask].sum() < 1e-12:
        return 0.0, 0.0
    band, p = spec[mask], periods[mask]
    idx = int(np.argmax(band))
    return float(p[idx]), float(band[idx] / band.sum())


def rayleigh_R(phases):
    """圆形集中度 R ∈ [0,1] 与 Rayleigh 统计量 Z = n·R²"""
    n = len(phases)
    if n < 3:
        return 0.0, 0.0
    C = float(np.mean(np.cos(phases)))
    S = float(np.mean(np.sin(phases)))
    R = float(np.sqrt(C ** 2 + S ** 2))
    return R, n * R ** 2


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seeds = list(range(10))

    print("=" * 78)
    print("按环路长度分组检验：Coombes 可公度性预言 τ = pT")
    print("=" * 78)

    data = {}
    for mode, ad in [("可塑", True), ("冻结", False)]:
        per, pow_, elens, delays, cycles = [], [], [], [], []
        for s in seeds:
            r = run(seed=s, adapt=ad)
            p, pw = dominant_period_and_power(r["act"][3000:])
            per.append(p)
            pow_.append(pw)
            el = edge_cycle_lengths(r["src"], r["dst"], N)
            m = el > 0
            elens.append(el[m])
            delays.append(r["delay"][m])
            for c in all_shortest_cycles(r["src"], r["dst"], N):
                cycles.append((len(c), float(r["delay"][c].sum())))
        data[mode] = {
            "period": np.array(per), "power": np.array(pow_),
            "elen": np.concatenate(elens),
            "edelay": np.concatenate(delays),
            "cycles": cycles,
        }

    # ---------- 诊断 ----------
    print("\n【诊断】主导周期可靠性")
    print("-" * 78)
    for mode in data:
        d = data[mode]
        print(f"  {mode}: 周期 {np.mean(d['period']):.1f} ± "
              f"{np.std(d['period']):.1f} 步   主频功率占比 "
              f"{np.mean(d['power']):.3f}")
    pw = np.mean(data["可塑"]["power"])
    cv = np.std(data["可塑"]["period"]) / max(1e-9, np.mean(data["可塑"]["period"]))
    reliable = pw > 0.15 and cv < 0.5
    print(f"  → 主频功率 {pw:.3f}、周期变异系数 {cv:.2f}  "
          f"{'→ 周期可靠' if reliable else '→ 周期不可靠'}")

    # ---------- 检验 A ----------
    print("\n【检验 A】按环长分组：组内 delay 标准差（越小越集中）")
    print("-" * 78)
    print(f"{'环长':>6} {'可塑n':>7} {'可塑std':>9} {'冻结n':>7} "
          f"{'冻结std':>9} {'std比':>8}")
    print("-" * 78)
    lens = sorted(set(data["可塑"]["elen"]) & set(data["冻结"]["elen"]))
    for L in lens:
        a = data["可塑"]["edelay"][data["可塑"]["elen"] == L]
        f = data["冻结"]["edelay"][data["冻结"]["elen"] == L]
        if len(a) < 15 or len(f) < 15:
            continue
        sa, sf = float(np.std(a)), float(np.std(f))
        print(f"{L:>6} {len(a):>7} {sa:>9.3f} {len(f):>7} {sf:>9.3f} "
              f"{sf / max(1e-9, sa):>8.2f}×")

    sa_all = float(np.std(data["可塑"]["edelay"]))
    sf_all = float(np.std(data["冻结"]["edelay"]))
    print("-" * 78)
    print(f"{'全局':>6} {len(data['可塑']['edelay']):>7} {sa_all:>9.3f} "
          f"{len(data['冻结']['edelay']):>7} {sf_all:>9.3f} "
          f"{sf_all / max(1e-9, sa_all):>8.2f}×  ← 不分组")

    # 组内加权平均 std 比
    num = den = 0
    for L in lens:
        a = data["可塑"]["edelay"][data["可塑"]["elen"] == L]
        f = data["冻结"]["edelay"][data["冻结"]["elen"] == L]
        if len(a) < 15 or len(f) < 15:
            continue
        num += np.std(a) * len(a)
        den += len(a)
    within_a = num / den if den else np.nan
    within_ratio = sf_all / max(1e-9, within_a) if within_a == within_a else np.nan

    # ---------- 检验 B ----------
    if reliable:
        print("\n【检验 B】环总延迟 Στ 对周期 T 取模的 Rayleigh 集中度")
        print("-" * 78)
        T = float(np.mean(data["可塑"]["period"]))
        print(f"  使用 T = {T:.1f} 步")
        for mode in ["可塑", "冻结"]:
            ph = np.array([2 * np.pi * (sig % T) / T
                           for (_, sig) in data[mode]["cycles"]])
            R, Z = rayleigh_R(ph)
            print(f"  {mode}: n={len(ph)}  R={R:.3f}  Z={Z:.1f}  "
                  f"p≈{np.exp(-Z):.2e}")
    else:
        print("\n【检验 B】跳过 —— 主导周期不可靠，Στ mod T 无明确含义")
        # 退而求其次：看 Στ 的分布是否比"随机和"更集中
        print("  退化为：Στ 的分布熵 vs 随机基线（同长度随机 delay 的和）")
        rng0 = np.random.default_rng(0)
        for mode in ["可塑", "冻结"]:
            sigs = np.array([sig for (_, sig) in data[mode]["cycles"]])
            h, _ = np.histogram(sigs, bins=12)
            p = h / h.sum()
            ent = float(-(p * np.log(p + 1e-12)).sum())
            # 基线：同样本量、同环长分布，delay 均匀随机
            lens_arr = np.array([l for (l, _) in data[mode]["cycles"]])
            base = np.array([rng0.integers(1, TAU_MAX + 1, size=L).sum()
                             for L in lens_arr])
            hb, _ = np.histogram(base, bins=12)
            pb = hb / hb.sum()
            ent_b = float(-(pb * np.log(pb + 1e-12)).sum())
            print(f"  {mode}: Στ 熵 {ent:.3f}   随机基线熵 {ent_b:.3f}   "
                  f"Δ={ent_b - ent:+.3f}（正=比随机更集中）")

    # ---------- 结论 ----------
    print("\n" + "=" * 78)
    print("结论")
    print("-" * 78)
    print(f"  全局 std 比（冻结/可塑）      = {sf_all / sa_all:.2f}×")
    global_ratio = sf_all / sa_all
    if within_ratio == within_ratio:
        print(f"  组内加权 std 比（冻结/可塑） = {within_ratio:.2f}×")
        # 必须显著超过才构成"分组揭示结构"；1.09 vs 1.09 属同值，不是支持
        MARGIN = 0.05
        if within_ratio > global_ratio + MARGIN:
            print(f"  → 组内比值显著高于全局（>{MARGIN}）：支持'条件分布'解读")
        else:
            print(f"  → 组内比值与全局**实质相同**（差 "
                  f"{within_ratio - global_ratio:+.3f} < {MARGIN}）")
            print("    分组**未**揭示额外结构 —— '按环长收敛到不同延迟'的")
            print("    解读被否定。可塑组 delay 边缘分布仍接近均匀")
            print(f"    （std {sa_all:.2f}，均匀分布理论值 "
                  f"{np.std(np.arange(1, TAU_MAX + 1)):.2f}），")
            print("    说明并非收敛到离散值，而是**高维向量的逐坐标微调**：")
            print("    各维移动方向不同，边缘分布不变，整体落到更优区域。")
    if not reliable:
        print()
        print("  ⚠ Coombes 的 τ=pT 预言**无法在本平台检验**：")
        print(f"    主频功率占比仅 {pw:.3f}（阈值 0.15），系统无明确周期振荡。")
        print("    该预言在脉冲网络锁相语境下推导，需要明确发放周期 T；")
        print("    本系统为连续值 + 噪声驱动的衰减振荡，**前提缺失**。")
    print("=" * 78)
