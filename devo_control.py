"""发育式髓鞘网络在具身控制任务上的检验（numpy，无 torch）

设计原则（针对上一版实验的缺陷修正）：

上一版实验用"线性高斯 + 边免费"的环境检验"边有成本、结构应稀疏"的机制，
结论必然是机制无用——因为在该环境下全连接就是最优解，稀疏化只是纯损失。
本次修正三处：

1. **成本真实存在**：噪声维度通过固定反射增益直接驱动肢体 → 动作抖动 →
   被能耗项惩罚。误用噪声有即时代价，稀疏化因此有真实收益。
2. **环境有真实稀疏结构**：12 个观测维度中只有 5 个携带状态信息（分布在
   3 个异构信道上），其余 7 个是纯传感器噪声。
3. **任务是具身控制而非拟合**：指标是控制代价（状态误差 + 动作能耗），
   不是预测 MSE。预测（自回归）是**手段**，控制是目的。

架构对应用户设计：
- 爬虫脑：固定反射 K_refl（2×12），对所有维度一视同仁（含噪声维）——
  先天、一刀切、定义了"肢体可行动作面"。
- 高层脑：发育式髓鞘通路矩阵 G（12×12），学习从含噪观测重建可信表征
  ô，供同一条反射使用。控制律不变，改进的是**输入质量**。
- 学习信号：自回归残差（下一帧真实观测 − 预测），无外部标注、无反向传播。
- 选择：持续衰减 −ρ（用进废退的"废退"侧）+ 共发射成边（"用"侧）。

关键预期（可证伪）：
- 噪声维不可预测 → 其通路贡献是零均值随机涨落 → 需 −ρ 清理才能淘汰；
  ρ 太小则噪声通路靠随机游走堆积，ρ 太大则误杀弱信号通路 → ρ 有最优值。
- 真实维跨 3 个信道且同源（都来自状态 s）→ 跨模态成边应有正收益。
"""
import numpy as np

D = 12
# 信道划分（超模态：异构信号统一编码）
CHANNELS = {"vision": (0, 6), "proprio": (6, 10), "event": (10, 12)}
REAL_DIMS = [0, 1, 6, 7, 10]          # 跨 3 信道的真实维度（携带状态信息）
NOISE_DIMS = [2, 3, 4, 5, 8, 9, 11]   # 纯传感器噪声维度

T = 3000
A_MAT = 1.1 * np.eye(2)               # **不稳定**：不控制即发散，杜绝躺平解
B_MAT = np.eye(2)                     # 动作直接作用于状态
PROC_NOISE = 0.02
LAM_A = 0.1                           # 动作能耗惩罚系数
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


def make_brain(seed):
    """固定反射 K_refl —— 先天、一刀切、非最优

    对真实维（0,1）增益大致合理（0.6，最优约 0.9）；
    对噪声维也全非零（±0.1）——这是它的缺陷，且它无法自我修正。
    """
    rng = np.random.default_rng(seed)
    K = np.zeros((2, D))
    K[0, 0] = 0.6
    K[1, 1] = 0.6
    for j in NOISE_DIMS:
        K[:, j] = 0.1 * rng.normal(size=2)
    return K


def observe(s, s_prev, rng):
    """超模态观测：3 个异构信道，尺度与稀疏度各异

    视觉 (0-5)   : 0,1 = 位置（真实）; 2-5 = 传感器噪声（std 1.5，大）
    本体 (6-9)   : 6,7 = 速度（真实）; 8,9 = 噪声（std 1.0）
    事件 (10-11) : 10  = 位置的冗余观测（真实，尺度 0.5）; 11 = 噪声
    """
    o = np.zeros(D)
    vel = s - s_prev
    o[0:2] = s + 0.05 * rng.normal(size=2)
    o[2:6] = 1.5 * rng.normal(size=4)
    o[6:8] = vel + 0.05 * rng.normal(size=2)
    o[8:10] = 1.0 * rng.normal(size=2)
    o[10] = 0.5 * s[0] + 0.05 * rng.normal()
    o[11] = 0.8 * rng.normal()
    return o


def run(seed, rho=0.01, lr=0.3, allow_cross=True, wire=True,
        death=0.02, wire_confirm=10, norm=True, warmup=200,
        self_proof=0.0, verbose=False):
    """单次运行。高层轨迹与爬虫脑基线并行模拟，共享过程噪声以保证公平。

    返回：控制代价、结构指标、预测误差。
    """
    rng = np.random.default_rng(seed)
    K_refl = make_brain(seed)

    # 通路矩阵：初始恒等 = 蒸馏起点（直通观测，不加工）
    G = np.eye(D) * 1.0
    M = np.eye(D).astype(bool)      # 通路存在掩码
    co_fire = np.zeros((D, D), dtype=int)
    # 超模态统一编码：逐信道运行 RMS（对应感受野增益控制）
    rms = np.ones(D)

    s_h = np.array([1.0, -0.5])
    s_r = s_h.copy()
    s_prev_h, s_prev_r = s_h.copy(), s_r.copy()

    ohat_next = None
    o_prev = None
    costs_h, costs_r, pred_mse, n_paths = [], [], [], []

    for t in range(T):
        o_h = observe(s_h, s_prev_h, rng)
        o_r = observe(s_r, s_prev_r, rng)

        # ---- 超模态统一编码 ----
        # 异构信道尺度差异巨大（视觉 std 1.5 / 本体 1.0 / 事件 0.5，
        # 真实维仅 0.05+状态幅度）。若不归一化，通路增益的稳态由信道
        # **尺度**而非**可预测性**决定：噪声通路靠大幅度随机游走抵抗衰减
        # 而存活，弱信号通路反而跌破死亡阈值被误杀（实测 ρ=0.2 时真实维
        # 对角通路剩 3.0 条、噪声维剩 4.6 条）。
        # 归一化后，通路存亡才由"是否可预测"决定。
        rms = np.sqrt(0.99 * rms ** 2 + 0.01 * o_h ** 2)
        on_h = o_h / (rms + 1e-6) if norm else o_h
        on_r = o_r / (rms + 1e-6) if norm else o_r

        # 高层：预测在归一化空间，反归一化后交给同一条反射（控制律不变）
        if ohat_next is not None:
            a_h = -K_refl @ (ohat_next * (rms + 1e-6) if norm else ohat_next)
        else:
            a_h = -K_refl @ o_h
        # 爬虫脑：直接用当前含噪观测（反应式，无滤波）
        a_r = -K_refl @ o_r

        w = PROC_NOISE * rng.normal(size=2)
        s_next_h = A_MAT @ s_h + B_MAT @ a_h + w
        s_next_r = A_MAT @ s_r + B_MAT @ a_r + w
        # 裁剪：失控时防止数值溢出污染统计（发散本身由代价反映）
        s_next_h = np.clip(s_next_h, -1e3, 1e3)
        s_next_r = np.clip(s_next_r, -1e3, 1e3)

        costs_h.append(float(np.sum(s_next_h ** 2) + LAM_A * np.sum(a_h ** 2)))
        costs_r.append(float(np.sum(s_next_r ** 2) + LAM_A * np.sum(a_r ** 2)))

        # ---- 学习：自回归残差驱动，无标注、无反向传播 ----
        if ohat_next is not None and o_prev is not None:
            r_vec = on_h - ohat_next                # 环境即时反馈（归一化空间）
            pred_mse.append(float(np.mean(r_vec ** 2)))
            # 局部梯度：∂‖r‖²/∂G[i,j] = −2·r[j]·on_prev[i]（NLMS 归一化）
            denom = float(np.dot(o_prev, o_prev)) + 1e-8
            # 预热期只估计 rms，不更新权重
            if t >= warmup:
                G += M * (lr * np.outer(o_prev, r_vec) / denom)
            # 用进废退：持续衰减（唯一的选择压力）
            #
            # **相对衰减** G *= (1-ρ) 而非绝对衰减 G -= ρ。
            # 绝对衰减的病：控制收敛后状态变小 → 真实通路的增量 increment
            # 同步变小，但 −ρ 是恒定绝对值 → 衰减压倒学习 → 必要通路被误杀
            # （实测 ρ=0.01 时真实维对角通路从 5 条掉到 2 条，系统失控发散）。
            # 相对衰减下稳态为 lr·increment = ρ·G*，即 G* ∝ increment/ρ，
            # 只要增量非零就能维持，不会归零。
            G -= M * (rho * G)
            dead = M & (G < death)
            G[dead] = 0.0
            M[dead] = False

            # 共发射成边：反复同时激活的维度对 → 新通路（沉默突触 gain=0）
            if wire:
                active = np.where(np.abs(o_prev) > 0.5)[0]   # 归一化空间统一阈值
                for i in active:
                    for j in active:
                        if not allow_cross and i != j:
                            continue
                        co_fire[i, j] += 1
                        if (not M[i, j]) and co_fire[i, j] >= wire_confirm:
                            # ---- 先自证，再连接 ----
                            # 无差别成边的问题：噪声维的通路增益是零均值随机
                            # 游走，衰减只能拉回 0 附近却杀不死（波动幅度
                            # ∝ σ/√ρ，实测可达 0.2）。结果 41 条有效通路里
                            # 29 条涉及噪声维。
                            # 但**对角自预测增益能区分真假**（真实维
                            # 0.13~0.32，噪声维 0.00~0.08）——"该维度是否
                            # 可预测"这个信息已在系统内，只是没用于约束成边。
                            # 因此：向外出边前，来源维度必须先自证可预测。
                            if i != j and self_proof > 0.0:
                                if not (M[i, i] and abs(G[i, i]) >= self_proof):
                                    continue
                            M[i, j] = True
                            G[i, j] = 0.0

        n_paths.append(int(M.sum()))
        o_prev = on_h
        ohat_next = G.T @ on_h
        s_prev_h, s_prev_r = s_h, s_r
        s_h, s_r = s_next_h, s_next_r

    tail = slice(2 * T // 3, None)
    ch, cr = np.mean(costs_h[tail]), np.mean(costs_r[tail])
    inv = M & (G > 0.05)
    # 细分结构指标：对角=自预测，非对角=跨模态组合
    diag_real = sum(1 for i in REAL_DIMS if M[i, i])
    diag_noise = sum(1 for i in NOISE_DIMS if M[i, i])
    cross = sum(1 for i in range(D) for j in range(D)
                if M[i, j] and i != j)
    cross_real = sum(1 for i in REAL_DIMS for j in REAL_DIMS
                     if M[i, j] and i != j)
    return {
        "cost_high": ch,
        "cost_reflex": cr,
        "improve": (cr - ch) / max(cr, 1e-8),
        "n_paths": int(M.sum()),
        "n_eff": int(inv.sum()),
        "diag_real": diag_real,
        "diag_noise": diag_noise,
        "cross": cross,
        "cross_real": cross_real,
        "final_state": float(np.linalg.norm(s_h)),
        "pred_mse": float(np.mean(pred_mse[tail])) if pred_mse else float("nan"),
    }


def agg(**kw):
    keys = ("cost_high", "cost_reflex", "improve", "n_paths", "n_eff",
            "diag_real", "diag_noise", "cross", "cross_real", "final_state", "pred_mse")
    out = {}
    for s in SEEDS:
        r = run(s, **kw)
        for k in keys:
            out.setdefault(k, []).append(r[k])
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))))
            for k, v in out.items()}


def table(title, rows):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print(f"{'配置':<22}{'代价(高层)':>11}{'代价(反射)':>11}{'改进%':>8}"
          f"{'通路':>7}{'对角真':>7}{'对角噪':>7}{'跨模态':>7}{'状态范数':>9}")
    print("-" * 96)
    for name, r in rows:
        print(f"{name:<22}{r['cost_high'][0]:>11.4f}{r['cost_reflex'][0]:>11.4f}"
              f"{r['improve'][0]*100:>7.1f}%{r['n_paths'][0]:>7.1f}"
              f"{r['diag_real'][0]:>7.1f}{r['diag_noise'][0]:>7.1f}"
              f"{r['cross'][0]:>7.1f}{r['final_state'][0]:>9.3f}")


def main():
    print(f"配置：D={D} 真实维={REAL_DIMS} 噪声维={NOISE_DIMS} "
          f"T={T} seeds={len(SEEDS)} λ_a={LAM_A}")

    # ---- 1. ρ 扫描：用进废退的衰减强度 ----
    rows = []
    for rho in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
        rows.append((f"ρ={rho}", agg(rho=rho)))
    table("1. 衰减率 ρ 扫描（跨模态成边开启）", rows)

    # ---- 2. 跨模态成边的价值 ----
    rows = [
        ("跨模态成边 ON", agg(rho=0.05, allow_cross=True, wire=True)),
        ("跨模态成边 OFF", agg(rho=0.05, allow_cross=False, wire=True)),
        ("完全不成边", agg(rho=0.05, wire=False)),
    ]
    table("2. 超模态混合成边的价值（ρ=0.05）", rows)

    # ---- 3. 最优 ρ 下的完整对照 ----
    rows = [
        ("ρ=0（无废退）", agg(rho=0.0)),
        ("ρ=0.01", agg(rho=0.01)),
        ("ρ=0.05", agg(rho=0.05)),
        ("ρ=0.2", agg(rho=0.2)),
        ("ρ=0.4", agg(rho=0.4)),
    ]
    table("3. 完整对照（含标准差）", rows)
    for name, r in rows:
        print(f"  {name:<18} 改进 {r['improve'][0]*100:>6.1f}% "
              f"± {r['improve'][1]*100:>4.1f}%   通路 {r['n_paths'][0]:>5.1f}   "
              f"对角(真/噪) {r['diag_real'][0]:.1f}/{r['diag_noise'][0]:.1f}   "
              f"跨模态 {r['cross'][0]:.1f}")


if __name__ == "__main__":
    main()
