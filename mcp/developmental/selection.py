"""选择信号与双目标调度 —— 环境反馈如何变成"用进废退"的判据

这是本系统**唯一的目标来源**。设计上刻意避开外部标注与反向传播，
只使用环境自身的即时反馈，分两条时间尺度：

======================  ==========================  ========================
目标                     反馈信号                     对应时间尺度
======================  ==========================  ========================
蒸馏（distillation）     ``y_reflex`` 爬虫脑反射      **紧急**：即时可得
自回归（autoregressive） ``x_{t+1}`` 下一帧环境信号   **长期**：延迟一步
======================  ==========================  ========================

**为什么必须是两个而不是一个：**

纯蒸馏的上限是**追平反射**——反射是先天、一刀切、对所有维度一视同仁的，
它无法学会不去预测噪声信道。追平反射意味着继承了反射的全部盲区。

自回归才能裁定反射本身是否正确，从而**超越**基线。但自回归需要等
``x_{t+1}`` 到达，紧急情形下等不起——此时反射是唯一可靠的响应。

**因此：用自回归裁定蒸馏。**

    progress = (E_reflex_ar − E_high_ar) / E_reflex_ar
    λ = (1 − urgency) · σ(β · (progress − threshold))

- 高层只有在自回归上**真的**超过反射时才逐步接管（λ→1）；
- 紧急时 λ→0，反射直接接管，不等 ``x_{t+1}``；
- 环境持续变化导致高层退步时，λ 自动回落——接管是可逆的。

这同时给出"以爬虫脑固有反射为基线学习，然后蒸馏"的**可测量进度条**，
也是高层脑的毕业准则。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
from torch import Tensor

from mcp.developmental.myelin import SignalEvent


# ---------------------------------------------------------------------------
# 形状对齐
# ---------------------------------------------------------------------------

def align_to(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """把两个张量对齐到共同长度（尾部截断 / 零填充）

    髓鞘传输的分量与残差来自不同信道，长度未必一致。对齐到较短者的长度，
    避免广播错误，同时保证余弦相似度在共同子空间上计算。
    """
    n = min(a.numel(), b.numel())
    if n == 0:
        return a.new_zeros(0), b.new_zeros(0)
    return a.reshape(-1)[:n], b.reshape(-1)[:n]


def cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> float:
    """余弦相似度，零向量返回 0.0（视为无关，纯衰减）"""
    x, y = align_to(a, b)
    if x.numel() == 0:
        return 0.0
    nx, ny = x.norm().item(), y.norm().item()
    if nx < eps or ny < eps:
        return 0.0
    return float((x @ y).item() / (nx * ny + eps))


# ---------------------------------------------------------------------------
# 贡献：用进废退的判据
# ---------------------------------------------------------------------------

def pathway_contributions(
    events: list[SignalEvent],
    residual: Tensor,
    input_energy: Optional[float] = None,
    eps: float = 1e-8,
) -> dict[tuple[int, str, int, str], float]:
    """计算每条髓鞘的贡献 —— 用进废退的判据

    贡献定义为 MSE 的**局部梯度**（NLMS 归一化）：

        ∂‖target − y‖² / ∂gain_j = −2 · ⟨transmitted_j, residual⟩
        c_j = ⟨transmitted_j, residual⟩ / (‖x‖² + ε)

    这是局部的、逐通路可微的——**不需要**链式法则穿过层级，因此仍属
    "无反向传播"范畴，但给出了正确的信用分配。

    三种结果：

    - ``c > 0``：该通路在把输出推向目标 → 用进（增厚）
    - ``c < 0``：该通路在把输出推离目标 → 废退（加速衰减，比"没用"更该退）
    - ``c ≈ 0``：与目标无关 → 纯衰减（自然淘汰）

    **为什么用残差而不是目标**：基于残差使系统自限。当输出已经超过目标时，
    同一通路的贡献自动转负，无需任何硬约束就能防止无界增长。这正是
    "只有衰减、没有硬约束"仍能维持有界容量的原因（实测：ρ=0 时通路数
    膨胀至接近全连接，ρ=0.02 时收敛到 29%）。

    **为什么必须按 ‖x‖² 归一化（NLMS）**：朴素梯度在数十条通路并行更新时
    会严重过冲——每条通路都用同一个 residual 更新，而 residual 同时被所有
    通路改变，稳定步长取决于总 Hessian 谱范数（≈ 通路数 × E[x²]）。
    实测朴素梯度在 64 条通路下数值爆炸。除以 ‖x‖² 使步长自适应。

    Args:
        events: 本步分发产生的信号事件（含 sheath_key 与传输分量）
        residual: ``target − current_output``，期望修正方向
        input_energy: 输入信号能量 ‖x‖²。None → 退化为按 residual 范数归一化

    Returns:
        ``{sheath_key: contribution}``。未出现在 events 中的髓鞘不出现在
        结果里 → 被 :meth:`apply_global_decay` 视为 contribution=0（纯衰减）。
    """
    out: dict[tuple[int, str, int, str], float] = {}
    if residual is None or residual.numel() == 0:
        return out

    denom = input_energy if (input_energy and input_energy > eps) else None
    if denom is None:
        rn = float(residual.reshape(-1).norm().item())
        denom = rn * rn + eps

    for e in events:
        if e.sheath_key is None:
            continue
        x, r = align_to(e.data, residual)
        if x.numel() == 0:
            continue
        c = float((x @ r).item()) / denom
        # 同一髓鞘可能承载多个事件（多信道），取最大绝对值以保留最强信号
        prev = out.get(e.sheath_key)
        if prev is None or abs(c) > abs(prev):
            out[e.sheath_key] = c
    return out


# ---------------------------------------------------------------------------
# 双目标调度器
# ---------------------------------------------------------------------------

@dataclass
class DualTargetFeedback:
    """一步环境反馈的结算结果"""
    distill_error: float          # ||y_high − y_reflex||²：蒸馏差距
    ar_error_reflex: float        # ||y_reflex − x_next||²：反射的自回归误差
    ar_error_high: float          # ||y_high − x_next||²：高层的自回归误差
    progress: float               # 相对改进（EMA 平滑）
    target: Tensor                # 本步应逼近的目标
    residual: Tensor              # target − y_high，供贡献计算
    settled: bool                 # 是否完成了自回归结算



# ============================================================================
# 驱动满足度：λ 的裁定权归爬虫脑
# ============================================================================


class DriveSatisfaction:
    """**内在驱动满足度** —— 裁定 λ 的标量，属爬虫脑而非感知通道

    ⚠ 为什么不能由预测误差裁定 λ（实测陷阱）
    ------------------------------------------

    在反应式控制 loop 里，"下一帧观测"取决于系统**自己的输出**。
    于是出现自证式陷阱：

        高层输出错误 → 世界无响应 → 观测变全零
        → 恒零预测误差极低 → 自回归 progress 上升
        → λ 升高 → 更彻底接管 → 世界彻底死寂

    实测（baby_loop.py，3 seeds × 1500 步）：

    ============  =========
    控制者         满足率
    ============  =========
    反射           0.941
    高层           0.001
    λ 自适应       0.003
    ============  =========

    λ 自适应把 94% 的能力毁成了随机的 1/12 以下 —— 它以为自己在进步。

    **"预测得准" ≠ "控制得好"。** 控制权的裁定必须来自**反应侧**：
    "我的输出是否让世界朝想要的方向变化"。

    生物学定位
    ----------

    多巴胺能系统（VTA / 黑质致密部）是**皮层下**结构。它不感知世界，
    它裁定"刚才那个行动好不好"，并据此调节皮层对行为的控制权重
    （基底节的 Go/NoGo 门控）。

    这与本报告 §1.2 的架构主张一致：满足度是**爬虫脑的内在信号**，
    作用在学习与门控侧，**不进入高层脑的输入通道**。

    裁定方式：交替试验
    ------------------

    影子期高层不接管，但也无从知道自己的能力。因此以小概率 ε 让高层
    **真正接管**（探索性试验），观测结果，累积证据：

        competence = (sat_high − sat_reflex) / max(sat_reflex, ε)

    只有 competence 显著为正，才允许 λ 上升。样本不足时保持 λ=0。
    """

    def __init__(
        self,
        eps: float = 0.1,
        momentum: float = 0.95,
        min_samples: int = 20,
        eps_background: float = 0.02,
    ):
        self.eps = eps                  # 证据不足时的探索率（较高）
        self.eps_background = eps_background   # 证据充分后的背景探索率
        self.momentum = momentum
        self.min_samples = min_samples
        self._sat_reflex_ema: Optional[float] = None
        self._sat_high_ema: Optional[float] = None
        self._n_reflex: int = 0
        self._n_high: int = 0
        self._last_controller: str = "reflex"
        # --- 咨询台账：按**输出来源**（器官级 provenance）分账 ---
        #
        # 与 reflex/higher 二分臂正交：二分臂回答"影子期 vs 接管期谁
        # 控制得好"，台账回答"哪个**器官**的行为满足了驱动"。
        # 键来自输出信号的 metadata["source"]（如反射工具的 "babbling"、
        # 融合框架下 LLM 教师器官的 "teacher"）。
        #
        # 融合框架的核心曲线就从这里读：**教师咨询的满足率 × 咨询次数
        # 随髓鞘化下降**——"咨询费只付一次"若成立，teacher 的调用
        # 占比应随发育单调衰减，而其满足率不应显著劣于习惯通路。
        # {source: {"n": 次数, "sat_ema": 满足度 EMA, "sat_sum": 累计}}
        self._by_source: dict[str, dict[str, float]] = {}

    # ---------- 探索调度 ----------

    def should_explore(self, rng=None) -> bool:
        """本步是否让高层真正接管（探索性试验）

        ⚠ 两档探索率，不能只有一档（实测缺陷）：

        初版写成"证据不足才探索，够了就停"。后果是**探索在第 20 步
        就永久停止**（min_samples=20 即满足），而此时高层还很弱
        （实测 sat_high 仅 0.124）。此后即使高层通过学习变强，也
        永远没有机会再证明自己 —— 被锁死在影子模式里。

        修正为标准的 ε-greedy 两档：
          · 证据不足 → 高探索率 eps（快速积累初始证据）
          · 证据充分 → 低背景探索率 eps_background（持续监控，
            高层变强后能被发现，环境变化后也能被察觉）
        """
        rate = self.eps if not self.has_sufficient_evidence else self.eps_background
        if rate <= 0.0:
            return False
        if rng is None:
            import random
            rng = random.Random()
        return rng.random() < rate

    # ---------- 观测 ----------

    def observe(self, satisfaction: float, controller: str) -> None:
        """记录一次满足度观测

        Args:
            satisfaction: 本步内在驱动被满足的程度 ∈ [0, 1]
            controller: 本步实际的控制者，``"reflex"`` 或 ``"higher"``
        """
        if satisfaction is None:
            return
        m = self.momentum
        if controller == "higher":
            self._n_high += 1
            self._sat_high_ema = (
                satisfaction if self._sat_high_ema is None
                else m * self._sat_high_ema + (1 - m) * satisfaction
            )
        else:
            self._n_reflex += 1
            self._sat_reflex_ema = (
                satisfaction if self._sat_reflex_ema is None
                else m * self._sat_reflex_ema + (1 - m) * satisfaction
            )
        self._last_controller = controller
        # 咨询台账：无论控制臂如何，都按输出来源记一笔
        self.observe_source(controller, satisfaction)

    def observe_source(self, source: str, satisfaction: float) -> None:
        """按**输出来源**（器官级 provenance）记一笔满足度

        与 :meth:`observe` 的 reflex/higher 二分臂正交。同一器官的
        输出可能在两臂中都出现（影子期反射全权 vs 接管期高层主导），
        二分臂度量的是"控制权归属"，台账度量的是"器官贡献"。

        Args:
            source: 来源标签（输出信号 metadata["source"]，如
                ``"babbling"`` / ``"teacher"``）
            satisfaction: 本步内在驱动被满足的程度 ∈ [0, 1]
        """
        if satisfaction is None or source is None:
            return
        m = self.momentum
        entry = self._by_source.setdefault(
            str(source), {"n": 0.0, "sat_ema": 0.0, "sat_sum": 0.0})
        entry["n"] += 1
        entry["sat_sum"] += float(satisfaction)
        entry["sat_ema"] = (
            float(satisfaction) if entry["n"] == 1
            else m * entry["sat_ema"] + (1 - m) * float(satisfaction)
        )

    def source_report(self) -> dict[str, dict[str, float]]:
        """咨询台账快照：{source: {n, sat_ema, sat_mean}}"""
        return {
            src: {
                "n": int(e["n"]),
                "sat_ema": e["sat_ema"],
                "sat_mean": e["sat_sum"] / e["n"] if e["n"] else 0.0,
            }
            for src, e in self._by_source.items()
        }

    # ---------- 裁定 ----------

    @property
    def has_sufficient_evidence(self) -> bool:
        """双方是否都累积了足够的试验样本"""
        return (self._n_reflex >= self.min_samples
                and self._n_high >= self.min_samples)

    @property
    def competence(self) -> float:
        """高层相对反射的**胜任度**

        > 0 表示高层让世界更接近想要的状态，是高层接管的**唯一凭据**。
        证据不足时返回 0（不允许接管）。
        """
        if not self.has_sufficient_evidence:
            return 0.0
        base = max(self._sat_reflex_ema, 1e-6)
        return (self._sat_high_ema - self._sat_reflex_ema) / base

    def lambda_gate(
        self,
        urgency: float = 0.0,
        beta: float = 8.0,
        threshold: float = 0.0,
    ) -> float:
        """由胜任度裁定 λ ∈ [0,1]

        λ = (1 − urgency) · σ(β · (competence − θ))

        - 证据不足（样本 < min_samples）→ competence=0 → λ=σ(−βθ)，
          θ=0 时为 0.5。**调用方应先用 shadow_mode 把 λ 压到 0**，
          不要依赖 gate 自己兜底 —— 证据不足时任何非零 λ 都是冒险。
        - 紧急 → λ→0，反射接管（等不到满足度回流的下一帧）
        - 高层胜任 → λ→1
        - 高层退步 → competence 下降，λ 自动回落（**接管可逆**）
        """
        z = beta * (self.competence - threshold)
        sig = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
        return float((1.0 - max(0.0, min(1.0, urgency))) * sig)

    def report(self) -> dict:
        """诊断快照"""
        return {
            "sat_reflex": self._sat_reflex_ema,
            "sat_high": self._sat_high_ema,
            "n_reflex": self._n_reflex,
            "n_high": self._n_high,
            "competence": self.competence,
            "sufficient": self.has_sufficient_evidence,
            "by_source": self.source_report(),
        }

class DualTargetScheduler:
    """自回归 + 蒸馏的双目标调度器

    用法（每个系统步）：

    1. ``begin_step(x_t, y_reflex_t, y_high_t)`` — 记录本步，给出即时目标
    2. 用返回的 ``target`` / ``residual`` 计算贡献，驱动选择动力学
    3. 下一步开始时 ``settle(x_next)`` — 结算自回归误差，更新毕业进度

    之所以要延迟一步结算：自回归目标 ``x_{t+1}`` 在本步尚不存在。
    """

    def __init__(
        self,
        beta: float = 8.0,
        progress_threshold: float = 0.0,
        ema_momentum: float = 0.95,
        eps: float = 1e-8,
    ):
        self.beta = beta
        self.progress_threshold = progress_threshold
        self.ema_momentum = ema_momentum
        self.eps = eps
        # 待结算的上一帧
        self._pending: Optional[tuple[Tensor, Tensor, Tensor]] = None
        # 自回归误差的 EMA
        self._ar_reflex_ema: Optional[float] = None
        self._ar_high_ema: Optional[float] = None
        self._progress: float = 0.0
        self._last_distill: float = 0.0
        self._settled_steps: int = 0

    # ---------- 状态 ----------

    @property
    def progress(self) -> float:
        """高层相对反射的自回归改进度（EMA），∈ (-∞, 1]

        > 0 表示高层已超越反射基线，是高层脑的**毕业准则**。
        """
        return self._progress

    @property
    def last_distill_error(self) -> float:
        """最近一步的蒸馏误差（高层输出与反射的差距）"""
        return self._last_distill

    def report(self) -> dict:
        """诊断快照"""
        return {
            "progress": self._progress,
            "ar_reflex_ema": self._ar_reflex_ema,
            "ar_high_ema": self._ar_high_ema,
            "distill_error": self._last_distill,
            "settled_steps": self._settled_steps,
        }

    # ---------- 主流程 ----------

    def begin_step(
        self,
        y_reflex: Tensor,
        y_high: Tensor,
        urgency: float = 0.0,
    ) -> DualTargetFeedback:
        """开始一步：给出当前应逼近的目标

        目标随紧急度在两条时间尺度间切换：

        - ``urgency → 1``：目标 = 反射（紧急，等不到 ``x_{t+1}``）
        - ``urgency → 0``：目标 = 反射与高层自回归预测的**加权混合**，
          权重由已毕业程度（progress）决定

        注意混合目标不是"预测下一帧"——自回归项在结算时才使用。
        此处用于在从容情形下把高层已验证的优势掺入即时目标。

        Args:
            y_reflex: 爬虫脑反射输出（基线、先天、即时）
            y_high: 高层脑当前输出
            urgency: 紧急度 ∈ [0, 1]

        Returns:
            :class:`DualTargetFeedback`
        """
        distill_error = self._sq_dist(y_reflex, y_high)

        # 紧急 → 完全以反射为目标；从容 → 按毕业度掺入高层自身
        w_own = (1.0 - urgency) * max(0.0, min(1.0, self._progress))
        r, h = align_to(y_reflex, y_high)
        target = (1.0 - w_own) * r + w_own * h

        residual = target - h
        self._last_distill = distill_error
        # 存入待结算：下一帧到达时用 x_next 裁定
        self._pending = (
            r.detach().clone(), h.detach().clone(),
            y_reflex.detach().clone(),
        )
        return DualTargetFeedback(
            distill_error=distill_error,
            ar_error_reflex=float('nan'),
            ar_error_high=float('nan'),
            progress=self._progress,
            target=target,
            residual=residual,
            settled=False,
        )

    def settle(self, x_next: Tensor) -> DualTargetFeedback:
        """结算：下一帧环境信号到达，裁定反射与高层谁更接近

        这是**环境即时反馈**的落点，也是唯一能裁定"反射是否正确"的机制。

        Args:
            x_next: 下一帧环境信号

        Returns:
            :class:`DualTargetFeedback`，含两个自回归误差与更新后的 progress。
        """
        if self._pending is None:
            return DualTargetFeedback(
                distill_error=self._last_distill,
                ar_error_reflex=float('nan'),
                ar_error_high=float('nan'),
                progress=self._progress,
                target=x_next,
                residual=torch.zeros_like(x_next),
                settled=False,
            )

        r, h, y_reflex_raw = self._pending
        self._pending = None

        # 自回归误差：谁的上一帧输出更接近这一帧真实到达的信号
        xn_r, rr = align_to(x_next, r)
        xn_h, hh = align_to(x_next, h)
        e_reflex = float(((xn_r - rr) ** 2).sum().item())
        e_high = float(((xn_h - hh) ** 2).sum().item())

        # EMA 平滑，避免单步噪声主导毕业判定
        m = self.ema_momentum
        self._ar_reflex_ema = (
            e_reflex if self._ar_reflex_ema is None
            else m * self._ar_reflex_ema + (1 - m) * e_reflex
        )
        self._ar_high_ema = (
            e_high if self._ar_high_ema is None
            else m * self._ar_high_ema + (1 - m) * e_high
        )

        denom = max(self._ar_reflex_ema, self.eps)
        raw_progress = (self._ar_reflex_ema - self._ar_high_ema) / denom
        self._progress = m * self._progress + (1 - m) * raw_progress
        self._settled_steps += 1

        # ⚠ 原实现返回 ``torch.zeros_like(x_next.reshape(-1)[:1])`` ——
        # 一个**长度为 1 的零向量**，不是自回归残差。
        # 后果：自回归误差只用于计算 progress（一个标量），
        # 从未被用于更新权重 → 系统学不到"输入→输出映射"。
        # 实测（真实系统 600 步）：lang 段余弦为负、top-1 低于随机基线。
        #
        # 修正：返回真正的自回归残差 = 目标 − 高层输出（对齐后）。
        # 这是环境即时反馈，无需外部标注，是 delta rule 的训练信号。
        residual = xn_h - hh
        return DualTargetFeedback(
            distill_error=self._last_distill,
            ar_error_reflex=e_reflex,
            ar_error_high=e_high,
            progress=self._progress,
            target=x_next,
            residual=residual,
            settled=True,
        )

    def lambda_gate(self, urgency: float = 0.0) -> float:
        """连续干预强度 λ ∈ [0, 1]

        ``λ = (1 − urgency) · σ(β · (progress − threshold))``

        - λ→0：反射接管（紧急，或高层尚未毕业）
        - λ→1：高层接管（从容，且高层已在自回归上超越反射）

        接管是**可逆的**：环境变化导致高层退步时，λ 自动回落。
        """
        import math
        z = self.beta * (self._progress - self.progress_threshold)
        sig = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
        return float((1.0 - max(0.0, min(1.0, urgency))) * sig)

    # ---------- 工具 ----------

    def _sq_dist(self, a: Tensor, b: Tensor) -> float:
        x, y = align_to(a, b)
        if x.numel() == 0:
            return 0.0
        return float(((x - y) ** 2).sum().item())
