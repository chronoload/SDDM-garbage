"""验证四个设计选择的后果（numpy 原型，无 torch 依赖）

针对用户明确选择的四项：
1. 反馈 = 自回归 + 蒸馏（双目标）
2. 竞争 = 只有衰减，无硬约束（自适应预算）
3. 髓鞘 = delay + gain 双算子
4. 分化触发 = 新奇性 + 输出髓鞘化稳定反应

本脚本重点检验 **1** 与 **2**——它们是唯一有实验数据可能反对的选择。

环境设定（对应"环境提供噪声和有用信号"）：
- 前 4 维由 2 个隐特征驱动，AR(1) 结构 → **可预测**
- 后 4 维为纯白噪声，逐帧独立 → **不可预测**

爬虫脑反射（对应"固有反射基线"）：
- y_reflex = α · x_t（延续/平滑），α = 0.9
- 对可预测维：合理（AR 系数恰为 0.9）
- 对噪声维：**比什么都不做更差**——预测 x_t 的噪声，而真值是新噪声，
  误差 2σ² > 预测 0 的 σ²
- 反射是先天、一刀切的，**无法学会不去预测噪声**

高层脑（对应"髓鞘网络"）：
- 通路 (i→j)，增益可塑；y_high[j] = Σ_i gain[(i,j)] · x_t[i]
- 用进废退：Δgain = a·c − ρ，c ∝ ∂‖residual‖²/∂gain（基于残差，故自限）
- 共发射成边：同时激活的维度对 → 创建新通路（超模态混合）

关键推导（说明机制为何有效）：
    对噪声维 j：E[residual[j]·x_t[j]] = E[x_{t+1}[j]·x_t[j]] − gain·E[x_t[j]²]
                                     = 0 − gain·σ²  < 0  (当 gain > 0)
    → 贡献恒负 → gain 单调衰减 → 通路被淘汰 → 该维输出归零 → **优于反射**

    对可预测维 j：residual[j] = (0.9 − gain)·x_t[j]
    → gain < 0.9 时贡献正、> 0.9 时贡献负 → 收敛到 gain = 0.9
"""
import numpy as np

D = 8                 # 总维度
N_USEFUL = 4          # 可预测维（前 4）
N_NOISE = 4           # 纯噪声维（后 4）
T = 4000
ALPHA = 0.9           # 隐特征 AR 系数 = 反射的平滑系数
NOISE_SIGMA = 1.0
SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------

CROSS_COEF = 0.5      # 跨模态耦合强度：x_next[j] 依赖 x_t[k]


def make_env(seed: int, cross: float = 0.0):
    """生成含噪声信道的环境序列。返回 X (T, D)。

    Args:
        cross: 跨模态耦合系数。>0 时前 4 维之间存在真实的跨维度预测结构
               （x_next[j] 依赖 x_t[k], k≠j），用于检验"超模态混合"成边
               的价值。cross=0 时最优解是对角的，跨模态通路纯属噪声——
               此时"关闭跨模态成边"更好只是任务假象，不能据此否定超模态。
    """
    rng = np.random.default_rng(seed)
    f = np.zeros((T, 2))
    for t in range(1, T):
        f[t] = ALPHA * f[t - 1] + 0.3 * rng.normal(size=2)
    W = rng.normal(size=(N_USEFUL, 2))
    X = np.zeros((T, D))
    X[:, :N_USEFUL] = f @ W.T
    X[:, N_USEFUL:] = NOISE_SIGMA * rng.normal(size=(T, N_NOISE))
    if cross > 0.0:
        # 环状耦合：维 j 的下一帧额外依赖维 (j+1) % N_USEFUL 的当前帧
        raw = X[:, :N_USEFUL].copy()
        for t in range(T - 1):
            for j in range(N_USEFUL):
                k = (j + 1) % N_USEFUL
                X[t + 1, j] += cross * raw[t, k]
    X += 0.02 * rng.normal(size=(T, D))
    return X


# ---------------------------------------------------------------------------
# 高层网络：共发射成边 + 用进废退（只有衰减，无硬约束）
# ---------------------------------------------------------------------------

class DevoNet:
    def __init__(self, D, rho, lr=0.5, death=0.0, init_gain=0.9,
                 wire_thresh=0.5, allow_cross=True,
                 wire_confirm=1, new_gain=0.0, protect_steps=0):
        self.D = D
        self.rho = rho            # 衰减率 ρ —— 唯一的选择压力来源
        self.lr = lr              # 局部梯度步长
        self.death = death        # gain 死亡阈值
        self.wire_thresh = wire_thresh
        self.allow_cross = allow_cross
        self.wire_confirm = wire_confirm   # 成边所需的最少共发射次数
        self.new_gain = new_gain           # 新通路初始增益（沉默突触）
        self.protect_steps = protect_steps   # 新生保护期长度（对应 protection 系数）
        # 初始：对角通路，增益 = 反射系数（蒸馏的起点）
        self.gain = {(j, j): init_gain for j in range(D)}
        self.co_fire_count = {}
        self.age = {(j, j): 10**9 for j in range(D)}  # 初始通路无保护
        self.born = D
        self.died = 0

    def forward(self, x):
        y = np.zeros(self.D)
        for (i, j), g in self.gain.items():
            y[j] += g * x[i]
        return y

    def co_fire_wire(self, x):
        """共发射成边：反复同时激活的维度对 → 建立新通路（超模态混合）

        **沉默突触**：新通路的初始增益为 0，结构上存在但功能上沉默，
        需经后续选择增厚才产生作用。

        若新通路一建立就带非零增益（本实验最初用 0.2），则每步新建的随机
        通路会立即污染输出，随即被淘汰、再被重建——实测形成 10870 次新建 /
        10830 次淘汰的空转循环，高层性能反低于反射基线。

        ``wire_confirm`` 要求反复共发射才成边，对应突触形成需要强激活，
        并抑制上述空转。
        """
        active = np.where(np.abs(x) > self.wire_thresh)[0]
        created = 0
        for i in active:
            for j in active:
                if not self.allow_cross and i != j:
                    continue
                key = (i, j)
                self.co_fire_count[key] = self.co_fire_count.get(key, 0) + 1
                if (key not in self.gain
                        and self.co_fire_count[key] >= self.wire_confirm):
                    self.gain[key] = self.new_gain
                    self.age[key] = 0
                    self.born += 1
                    created += 1
        return created

    def select(self, x, residual):
        """用进废退：局部梯度 + 持续衰减，无硬约束

        更新规则是 MSE 的**局部梯度**：

            ∂‖x_next − y‖² / ∂gain[(i,j)] = −2 · residual[j] · x[i]
            Δgain = lr · residual[j] · x[i] − ρ

        这是局部的、逐通路可微的——**不需要**链式法则穿过层级，因此仍属
        "无反向传播"范畴；但它给出了正确的信用分配，而早期版本用的全局
        归一化（除以 ‖residual‖·‖x‖）会稀释贡献，使有用通路的增益学不到
        正确量级（实测有用维增益只到 0.08，远低于应有的 0.9）。

        衰减项 ρ 造成稳态偏差：
            gain* = 0.9 − ρ · E[‖x‖²] / (lr · E[x[j]²])
        因此 ρ 必须远小于 lr·E[x[j]²]/E[‖x‖²]，否则衰减会压垮学习。

        **为什么用 NLMS 归一化**：朴素梯度在数十条通路并行更新时会严重过冲
        ——每条通路都用同一个 residual 更新，而 residual 同时被所有通路改变。
        稳定步长取决于总 Hessian 的谱范数（≈ 通路数 × E[x²]），通路数越多
        所需步长越小。除以 ‖x‖² 使步长自适应，lr 可取到接近 1。
        """
        nx2 = float(np.dot(x, x)) + 1e-8
        dead = []
        for (i, j), g in list(self.gain.items()):
            # 新生保护期：新通路的衰减按 protection 缩放，随年龄线性消退。
            # 对应生物学中新生突触的易化期——真实但微弱的结构需要窗口期
            # 才能积累到足以抵抗持续衰减的强度。
            prot = 0.0
            if self.protect_steps > 0:
                prot = max(0.0, 1.0 - self.age.get((i, j), 0) / self.protect_steps)
            self.age[(i, j)] = self.age.get((i, j), 0) + 1
            eff_rho = self.rho * (1.0 - prot)
            new_g = g + (self.lr * residual[j] * x[i] / nx2) - eff_rho
            if new_g < self.death:
                dead.append((i, j))
            else:
                self.gain[(i, j)] = new_g
        for k in dead:
            del self.gain[k]
        self.died += len(dead)
        return len(dead)

    def effective_paths(self, thresh=0.1):
        """有效通路数（增益高于阈值的）——排除沉默/濒死通路"""
        return sum(1 for g in self.gain.values() if g > thresh)

    def noise_path_ratio(self):
        """噪声维度相关通路占比——越低说明越学会"不去预测噪声" """
        if not self.gain:
            return 0.0
        n_noise = sum(
            1 for (i, j) in self.gain
            if i >= N_USEFUL or j >= N_USEFUL
        )
        return n_noise / len(self.gain)


# ---------------------------------------------------------------------------
# 单次运行
# ---------------------------------------------------------------------------

def run(seed, rho=0.01, mode="dual", allow_cross=True, lr=0.5,
        wire_confirm=20, new_gain=0.0, wire=True, cross=0.0,
        protect_steps=0):
    """mode:
      - "reflex"       仅反射基线
      - "distill_only" 高层以反射为目标（蒸馏上限 = 追平反射）
      - "dual"         高层以 x_next 为目标，λ 由 progress 门控
    """
    X = make_env(seed, cross=cross)
    net = DevoNet(D, rho=rho, lr=lr, allow_cross=allow_cross,
                  wire_confirm=wire_confirm, new_gain=new_gain,
                  protect_steps=protect_steps)
    progress = 0.0
    errs_sys, errs_reflex = [], []
    n_path_hist = []

    for t in range(T - 1):
        x, x_next = X[t], X[t + 1]
        y_reflex = ALPHA * x                      # 先天、一刀切
        y_high = net.forward(x)

        # 系统输出 = λ 混合（λ 由毕业度门控）
        z = max(-60.0, min(60.0, 8.0 * progress))
        lam = 1.0 / (1.0 + np.exp(-z)) if mode == "dual" else 1.0
        y_sys = (1 - lam) * y_reflex + lam * y_high

        err_reflex = float(np.mean((x_next - y_reflex) ** 2))
        err_high = float(np.mean((x_next - y_high) ** 2))
        errs_reflex.append(err_reflex)
        errs_sys.append(float(np.mean((x_next - y_sys) ** 2)))
        n_path_hist.append(len(net.gain))

        # 毕业度：高层相对反射的自回归改进（EMA）
        raw = (err_reflex - err_high) / max(err_reflex, 1e-8)
        progress = 0.95 * progress + 0.05 * raw

        if mode == "dual":
            residual = x_next - y_high            # 自回归：环境即时反馈
        elif mode == "distill_only":
            residual = y_reflex - y_high          # 蒸馏：以反射为示范
        else:
            continue

        if wire:
            net.co_fire_wire(x)
        net.select(x, residual)

    tail = slice(2 * (T - 1) // 3, None)
    return {
        "mse_sys": float(np.mean(errs_sys[tail])),
        "mse_reflex": float(np.mean(errs_reflex[tail])),
        "n_paths": len(net.gain),
        "n_eff": net.effective_paths(),
        "n_paths_peak": int(np.max(n_path_hist)),
        "noise_ratio": net.noise_path_ratio(),
        "progress": progress,
        "born": net.born,
        "died": net.died,
    }


def sweep_rho():
    """ρ 敏感性：验证"只有衰减"下的自适应预算行为"""
    rows = []
    for rho in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        agg = {k: [] for k in
               ("mse_sys", "mse_reflex", "n_paths", "n_eff",
                "n_paths_peak", "noise_ratio", "progress")}
        for s in SEEDS:
            r = run(s, rho=rho, mode="dual")
            for k in agg:
                agg[k].append(r[k])
        rows.append({
            "rho": rho,
            "mse_sys": np.mean(agg["mse_sys"]),
            "mse_reflex": np.mean(agg["mse_reflex"]),
            "n_paths": np.mean(agg["n_paths"]),
            "n_eff": np.mean(agg["n_eff"]),
            "n_paths_peak": np.mean(agg["n_paths_peak"]),
            "noise_ratio": np.mean(agg["noise_ratio"]),
            "progress": np.mean(agg["progress"]),
        })
    return rows


def main():
    print("=" * 78)
    print("配置对照（5 seeds 均值，后 1/3 步）")
    print("=" * 78)
    configs = [
        ("反射基线", dict(mode="reflex"), None),
        ("纯蒸馏（上限=追平反射）", dict(mode="distill_only", rho=0.001), None),
        ("双目标 ρ=0.001", dict(mode="dual", rho=0.001), None),
        ("双目标 ρ=0.005", dict(mode="dual", rho=0.005), None),
        ("双目标 ρ=0.01（推荐）", dict(mode="dual", rho=0.01), None),
        ("双目标 ρ=0.02", dict(mode="dual", rho=0.02), None),
        ("双目标 无跨模态成边", dict(mode="dual", rho=0.01, allow_cross=False), None),
        ("双目标 无成边(仅选择)", dict(mode="dual", rho=0.01, wire=False), None),
    ]
    print(f"{'配置':<24}{'MSE(系统)':>11}{'MSE(反射)':>11}"
          f"{'通路数':>8}{'有效':>7}{'毕业度':>9}")
    print("-" * 78)
    for name, kw, _ in configs:
        agg = {k: [] for k in
               ("mse_sys", "mse_reflex", "n_paths", "n_eff",
                "n_paths_peak", "noise_ratio", "progress")}
        for s in SEEDS:
            r = run(s, **kw)
            for k in agg:
                agg[k].append(r[k])
        print(f"{name:<24}{np.mean(agg['mse_sys']):>11.4f}"
              f"{np.mean(agg['mse_reflex']):>11.4f}"
              f"{np.mean(agg['n_paths']):>8.1f}"
              f"{np.mean(agg['n_eff']):>7.1f}"
              f"{np.mean(agg['progress']):>9.2f}")

    print()
    print("=" * 78)
    print("ρ 敏感性：'只有衰减、无硬约束'的容量行为（双目标，5 seeds）")
    print("=" * 78)
    print(f"{'ρ':>8}{'MSE(系统)':>11}{'MSE(反射)':>11}{'终态通路':>10}"
          f"{'有效通路':>10}{'峰值':>8}{'毕业度':>9}")
    print("-" * 78)
    for r in sweep_rho():
        print(f"{r['rho']:>8.4f}{r['mse_sys']:>11.4f}{r['mse_reflex']:>11.4f}"
              f"{r['n_paths']:>10.1f}{r['n_eff']:>10.1f}"
              f"{r['n_paths_peak']:>8.1f}{r['progress']:>9.2f}")

    print()
    print("最大通路数上界（全连接，含自环）= {}".format(D * D))


def cross_modal_experiment():
    """检验超模态混合成边的价值：环境含真实跨维度结构时，成边是否必要"""
    print()
    print("=" * 78)
    print("超模态混合成边的价值（环境含跨模态耦合 cross=0.5，5 seeds）")
    print("=" * 78)
    print(f"{'配置':<24}{'MSE(系统)':>11}{'MSE(反射)':>11}"
          f"{'通路数':>8}{'有效':>7}{'毕业度':>9}")
    print("-" * 78)
    for name, kw in [
        ("反射基线", dict(mode="reflex", cross=0.5)),
        ("允许跨模态成边", dict(mode="dual", rho=0.01, cross=0.5,
                            allow_cross=True, wire=True)),
        ("禁止跨模态成边", dict(mode="dual", rho=0.01, cross=0.5,
                            allow_cross=False, wire=True)),
        ("完全不成边", dict(mode="dual", rho=0.01, cross=0.5,
                          wire=False)),
    ]:
        agg = {k: [] for k in
               ("mse_sys", "mse_reflex", "n_paths", "n_eff", "progress")}
        for s_ in SEEDS:
            r = run(s_, **kw)
            for k in agg:
                agg[k].append(r[k])
        print(f"{name:<24}{np.mean(agg['mse_sys']):>11.4f}"
              f"{np.mean(agg['mse_reflex']):>11.4f}"
              f"{np.mean(agg['n_paths']):>8.1f}"
              f"{np.mean(agg['n_eff']):>7.1f}"
              f"{np.mean(agg['progress']):>9.2f}")


def protection_experiment():
    """检验新生保护期：弱而真实的跨模态结构能否在持续衰减下存活"""
    print()
    print("=" * 78)
    print("新生保护期的价值（环境含跨模态耦合 cross=0.5，ρ=0.01，5 seeds）")
    print("=" * 78)
    print(f"{'保护期':<12}{'MSE(系统)':>11}{'MSE(反射)':>11}"
          f"{'通路数':>8}{'有效':>7}{'毕业度':>9}")
    print("-" * 78)
    for ps in (0, 50, 200, 500, 1000):
        agg = {k: [] for k in
               ("mse_sys", "mse_reflex", "n_paths", "n_eff", "progress")}
        for s_ in SEEDS:
            r = run(s_, mode="dual", rho=0.01, cross=0.5,
                    allow_cross=True, wire=True, protect_steps=ps)
            for k in agg:
                agg[k].append(r[k])
        print(f"{ps:<12}{np.mean(agg['mse_sys']):>11.4f}"
              f"{np.mean(agg['mse_reflex']):>11.4f}"
              f"{np.mean(agg['n_paths']):>8.1f}"
              f"{np.mean(agg['n_eff']):>7.1f}"
              f"{np.mean(agg['progress']):>9.2f}")


if __name__ == "__main__":
    main()
    cross_modal_experiment()
    protection_experiment()
