"""跨模态几何分析：把"超模态编码层的几何意义"变成可计算量

## 几何的三层来源（设计主张）

- 物理层 = 坐标系：各模态原始信号是坐标轴（裸编码，身体形态学）。
- 通路层 = 投影：信号在系统内的"坐标"是它激活的通路集合；
  模态 A→B 的联通 = 传递算子 M^(A→B)（W 子块 × 髓鞘 gain 复合）。
- 确认层 = 度量：距离由共发射统计与确认度定义（因果度量）。

## 三个可计算量

1. **传递算子** M^(A→B) = Σ_n activity_n · W_n[B_rows, A_cols]
   —— A 的方差有多少能到达 B。
2. **参与率**（effective rank，participation ratio）
   (Σs)²/Σs² —— 共享子空间的维数估计（binding 的定量）。
3. **循环一致性** M^(A→B)·M^(B→A) 的首奇异值占界比值
   —— 循环上守恒的部分 = 超模态不变量（共享结构）；
   循环中消失的部分 = 模态私有成分（区分）。
   "区分与连接不重复"的判据：重复的通路会在循环里表现为
   同一方向的冗余缩放，被用进废退的生存竞争去重。
"""
from __future__ import annotations

import numpy as np


def _to_np(x):
    """shim Tensor / torch Tensor / ndarray → ndarray"""
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy(), dtype=float)
    if hasattr(x, "a"):
        return np.asarray(x.a, dtype=float)
    return np.asarray(x, dtype=float)


def modal_transfer_matrices(hb, modality_channels=None):
    """提取各模态对之间的有效传递算子（边复合定义）

    ⚠ 定义修正（实测教训）：第一版用 W_n[B_rows, A_cols]（W 的目标模态
    行块）——结果 vis→lang 恒为零，而系统明明在正确发声。原因：真实的
    跨模态传递不走 W 的目标模态行，走**边的复合**：

        payload(e: n:src_ch → ·:dst_ch) = gain_e · W_n[src_ch_rows, :] @ x

    即源模态行的完整输出，经髓鞘 gain 散射到目标模态位置。故：

        M^{A→B}[B_col, input_col] = Σ_e act_{src(e)} · gain_e · W_{src(e)}[A_rows, input_col]

    活动度取 metabolism（存活账本：被世界反复使用的神经元权重大）。

    Returns:
        {(A, B): ndarray(sizeB, global_dim)}
    """
    layout = hb.port_layout
    chs = modality_channels or list(layout.keys())
    global_dim = sum(s for _, s in layout.values())
    mats = {(A, B): np.zeros((layout[B][1], global_dim))
            for A in chs for B in chs if A != B}
    for (sn, sc, dn, dc), conn in hb.sheath_registry._connections.items():
        if sc not in chs or dc not in chs or sc == dc:
            continue
        sh = hb.sheath_registry._sheaths.get((sn, sc, dn, dc))
        gain = float(conn.effective_gain(sh))
        n = hb.ecosystem.neurons.get(sn)
        W = getattr(n, "W", None)
        if W is None or not getattr(n, "unfolded", None):
            continue
        if sc not in n.unfolded:
            continue
        Wn = _to_np(W)
        bounds = {c: (s0, e0) for c, s0, e0 in n._channel_bounds()}
        if sc not in bounds:
            continue
        # 局部列 → 全局输入位置 + 维度增益（输入侧用进废退的执行点）
        col_global, col_gain = [], []
        for c2, s2, e2 in n._channel_bounds():
            off2, _ = layout[c2]
            g2 = float(getattr(n.unfolded[c2], "gain", 1.0))
            for k in range(s2, e2):
                col_global.append(off2 + (k - s2))
                col_gain.append(g2)
        s0, e0 = bounds[sc]
        Wblk = Wn[s0:e0, :len(col_global)]
        Wg = Wblk * np.asarray(col_gain, dtype=float)[None, :]
        act = float(getattr(n, "metabolism", 0.5))
        offB, sizeB = layout[dc]
        M = np.zeros((sizeB, global_dim))
        M[:, col_global] += act * gain * Wg
        mats[(sc, dc)] += M
    return mats


def _participation_ratio(s):
    """奇异谱的参与率 = (Σs)²/Σs²，有效秩（共享子空间维数估计）"""
    s = np.asarray(s, dtype=float)
    if s.sum() <= 1e-12:
        return 0.0
    return float((s.sum() ** 2) / max(1e-12, (s ** 2).sum()))


def analyze_pair(M):
    """单个传递算子的几何分析"""
    S = np.linalg.svd(M, compute_uv=False)
    energy = float((S ** 2).sum())
    return {
        "energy": energy,
        "top_sv": float(S[0]) if len(S) else 0.0,
        "participation": _participation_ratio(S),
    }


def cycle_coherence(M_ab, M_ba, off_a=0, size_a=None, k=4):
    """循环一致性：M^{A→B}·M^{B→A} 限制在 A 的输入列上

    首奇异值接近 √(‖M_ab‖·‖M_ba‖) 的比例越高，
    说明 A↔B 之间共享的不变量子空间越大（那个既被看见又被听到的
    物体）。私有成分在循环中衰减消失。
    """
    if size_a is not None:
        M_ba = M_ba[:, off_a:off_a + size_a]
    if M_ab.size == 0 or M_ba.size == 0 or M_ab.shape[1] != M_ba.shape[0]:
        return {"top": 0.0, "ratio": 0.0, "spectrum": []}
    C = M_ab @ M_ba
    s_c = np.linalg.svd(C, compute_uv=False)
    s_a = np.linalg.svd(M_ab, compute_uv=False)
    s_b = np.linalg.svd(M_ba, compute_uv=False)
    bound = float(np.sqrt((s_a ** 2).sum() * (s_b ** 2).sum()))
    top = float(s_c[0]) if len(s_c) else 0.0
    return {
        "top": top,
        "bound": bound,
        "ratio": top / bound if bound > 1e-12 else 0.0,
        "spectrum": [float(x) for x in s_c[:k]],
    }


def modal_geometry_report(hb, modality_channels=None):
    """完整几何报告：传递能量 / 参与率 / 循环一致性"""
    mats = modal_transfer_matrices(hb, modality_channels)
    pairs = {}
    for (A, B), M in mats.items():
        if A == B:
            continue
        pairs[f"{A}→{B}"] = analyze_pair(M)
    cyc = {}
    chs = modality_channels or list(hb.port_layout.keys())
    for i, A in enumerate(chs):
        offA, sizeA = hb.port_layout[A]
        for B in chs[i + 1:]:
            cyc[f"{A}↔{B}"] = cycle_coherence(
                mats[(A, B)], mats[(B, A)], off_a=offA, size_a=sizeA)
    return {"pairs": pairs, "cycles": cyc}
