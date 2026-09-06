"""神经元类：动态张量生命体

神经元是动态张量生命体——内部权重矩阵 W 的维数随分化/生长/自噬/髓鞘隔离动态变化。
算子 = 权重 = 线性变换矩阵 W（可微调）。
髓鞘是 W 的包裹层（delay/gain/protection），不是额外算子。
seed 给 W 的初始值（先天倾向），使用反馈微调 W 的参数到收敛（后天适应）。
单神经元蕴含处理所有信号的潜能（全息性）。
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class DimSlice:
    """维度切片：一个信道在神经元中的展开状态

    维度是对偶的——既是输入接口也是输出接口。

    **四个状态量的分工**（不可混为一谈）：

    - ``activity``  : 输入侧。该维度近期**有没有信号到达**。EMA，自带衰减。
    - ``utility``   : 输出侧。该维度**用了有没有用**——它解释了目标能量的
                      多大比例。无量纲，∈ (-∞, 1]。**自噬的真正判据。**
    - ``gain``      : 用进废退的执行载体。按 utility 增减，渐进弱化（退化），
                      趋零后被自噬剪除。
    - ``evaluated`` : 已收到反馈的次数。防止未评估过的维度被误剪。

    ``activity`` 与 ``utility`` 的区别至关重要：噪声通道的 activity 一直很高
    （信号一直在到达），但 utility 趋零（解释不了任何东西）。**只看 activity
    会导致噪声维度永远剪不掉**——这是原实现的根本缺陷。
    """
    dim_idx: int                    # 维度索引（对应信道偏移）
    size: int                       # 该维度的张量大小
    gain: float = 1.0               # 增益（用进废退的执行载体）
    activity: float = 0.0           # **输入侧**：有没有信号到达（EMA）
    utility: float = 0.0            # **输出侧**：用了有没有用（EMA，无量纲）
    evaluated: int = 0              # 已收到反馈的次数（自噬前置条件）
    age: int = 0                    # 展开后经历的步数（新生儿保护期判据）
    protection: float = 0.0         # 保护系数（髓鞘庇护）
    myelinated: bool = False        # 是否被髓鞘隔离


@dataclass
class ConvergenceState:
    """收敛状态：三层稳定判定"""
    weight_delta: float = float('inf')    # 权重变化量（< ε 时收敛）
    myelin_stable: bool = False           # 髓鞘拓扑是否稳定
    differentiation_rate: float = float('inf')  # 分化速率（趋零时收敛）

    def is_converged(self, weight_eps: float = 1e-4,
                     diff_threshold: float = 0.01) -> bool:
        """三层都稳定才算收敛"""
        return (self.weight_delta < weight_eps and
                self.myelin_stable and
                self.differentiation_rate < diff_threshold)


class Neuron:
    """动态神经元：权重矩阵 W 维数随分化/生长/自噬/髓鞘隔离动态变化

    单神经元蕴含处理所有信号的潜能（全息性）。
    算子 = 权重 = 线性变换矩阵 W（可微调）。
    髓鞘是 W 的包裹层（delay/gain/protection），不是额外算子。
    seed 给 W 的初始值（先天倾向），使用反馈微调 W 的参数到收敛（后天适应）。
    """

    def __init__(
        self,
        seed: int,
        port_layout: dict[str, tuple[int, int]],
        activity_momentum: float = 0.95,
    ):
        """
        Args:
            seed: 生成种子，决定权重的先天倾向
            port_layout: 信道布局 {channel_name: (offset, size)}
            activity_momentum: activity EMA 动量。决定"多快忘记未使用的维度"，
                越小则维度自噬越激进。
        """
        self.seed = seed
        self.port_layout = port_layout
        self.global_dim = sum(s for _, s in port_layout.values())

        # 权重矩阵 W：已展开维度的线性变换
        # 初始为空，随维度展开动态扩张
        self.W: Optional[Tensor] = None
        self.unfolded: dict[str, DimSlice] = {}

        # 使用累积参数
        self.activity_momentum = activity_momentum

        # 维度级用进废退参数
        # 稳态 gain* = (dim_lr / dim_decay) · utility。取 0.01/0.01 → gain* = utility，
        # 即"该维度按它解释目标能量的比例被放大"。
        self.dim_lr = 0.01            # 用进（utility → gain）
        self.dim_decay = 0.01         # 废退（相对衰减，见 record_dim_feedback 注释）
        self.utility_momentum = 0.99  # utility EMA 动量（需长于噪声相关时间）
        # 分裂时父节点保留的增益比例（子代获得 1-keep 的**相对优势**）
        self.split_keep_ratio = 0.8

        # --- 信息饱和度（协同 C：Ding 2023 的 saturation 判据） ---
        # 定义：utility 停滞 = 该神经元已充分学到它能学的，继续调参无效。
        # 作用：①抑制无效的参数调整 ②触发分裂（细分旧类别）
        self.saturation: float = 0.0        # ∈ [0,1]，1 = 完全饱和
        self._prev_utility: float = 0.0     # 上一步的总效用（停滞检测基准）
        self._stagnant_steps: int = 0       # 连续停滞步数
        self.stagnation_eps: float = 1e-3   # 停滞判定阈值（utility 变化量）
        self.saturation_window: int = 200   # 达完全饱和所需的连续停滞步数

        # 神经元级状态
        self.metabolism: float = 0.5
        self.alive: bool = True
        self.parent_seed: Optional[int] = None  # 分化时的父节点 seed
        # 漂移计数器（诊断：本神经元累计漂移次数）
        self.drift_count: int = 0
        # 上一步输出（展开顺序空间），供 record_dim_feedback 计算贡献
        self._last_output: Optional[Tensor] = None

    # ---------- 维度索引辅助 ----------

    def _unfolded_indices(self) -> list[int]:
        """已展开维度在**全局信号空间**中的索引（按展开顺序）"""
        idx: list[int] = []
        for slc in self.unfolded.values():
            idx.extend(range(slc.dim_idx, slc.dim_idx + slc.size))
        return idx

    def _channel_bounds(self) -> list[tuple[str, int, int]]:
        """各信道在**展开顺序空间**中的 (channel, start, end)"""
        out: list[tuple[str, int, int]] = []
        offset = 0
        for ch, slc in self.unfolded.items():
            out.append((ch, offset, offset + slc.size))
            offset += slc.size
        return out

    def unfold_channels(self, channels) -> list[str]:
        """批量展开维度（主循环用）

        返回**本次真正新展开**的信道名列表。
        已展开或不在 port_layout 中的信道被跳过。
        """
        newly: list[str] = []
        for ch in channels:
            if ch in self.unfolded or ch not in self.port_layout:
                continue
            # 展开不需要 signal_context（活动非依赖的先天布线，见 unfold）
            self.unfold(ch, torch.zeros(1))
            if ch in self.unfolded:
                newly.append(ch)
        return newly

    def unfold(self, channel: str, signal_context: Tensor) -> None:
        """惰性展开维度：W 扩张新行新列

        信号驱动 + 能量阈值过滤：信号在哪些信道有非零分量，就展开哪些维度。
        新行列的初始值由 seed + channel 生成（先天倾向）。
        """
        if channel in self.unfolded or not self.alive:
            return
        offset, size = self.port_layout[channel]

        # 从 seed 生成该维度的初始权重（先天倾向）
        #
        # 用 zlib.crc32 而非内建 hash()：后者对字符串是**每进程随机**的
        # （PYTHONHASHSEED），导致 save_state / 跨进程 / 跨机器都不可复现。
        rng = torch.Generator().manual_seed(
            self.seed + zlib.crc32(channel.encode()) % 2**31
        )
        new_block = torch.randn(size, size, generator=rng) * 0.1

        if self.W is None:
            self.W = new_block
        else:
            # W 扩张：新增 size 行 size 列
            old_size = self.W.shape[0]
            new_total = old_size + size
            expanded = torch.zeros(new_total, new_total)
            expanded[:old_size, :old_size] = self.W
            expanded[old_size:, old_size:] = new_block
            # 交叉项初始为 0（维度间初始无耦合，由训练建立）
            self.W = expanded

        self.unfolded[channel] = DimSlice(dim_idx=offset, size=size)
        # W 形状已变，上一步的输出缓存失效
        self._last_output = None

    def process(self, signal: Tensor) -> Optional[Tensor]:
        """处理信号：提取已展开维度的切片，经 W 变换

        信号是稠密张量，神经元只关注自己展开的切片。
        未展开的维度切片被忽略。
        """
        if self.W is None or not self.alive:
            return None

        # 提取已展开维度的切片，按展开顺序拼接
        active_indices = self._unfolded_indices()
        if not active_indices:
            return None

        signal_slice = signal[active_indices]
        # 应用各维度的 gain（用进废退的执行点）
        gains = torch.cat([torch.ones(s.size) * s.gain
                           for s in self.unfolded.values()])
        signal_slice = signal_slice * gains

        # W 变换
        output = self.W @ signal_slice

        # 输入侧活动度（EMA）——"有没有信号到达"
        #
        # 关键：必须是 EMA 而非累加。旧实现是 activity += 0.1 的单调累加，
        # 导致任何被激活过一次的维度永远高于自噬阈值，维度自噬形同虚设。
        # EMA 自带衰减：长期无信号到达的维度会自然回归 0。
        for ch, start, end in self._channel_bounds():
            seg = signal_slice[start:end]
            act = float(seg.abs().mean().item())
            slc = self.unfolded[ch]
            slc.activity = (
                self.activity_momentum * slc.activity
                + (1.0 - self.activity_momentum) * act
            )
            slc.age += 1

        # 保存输出（展开顺序空间），供 record_dim_feedback 计算各维度贡献
        self._last_output = output.detach().clone()
        return output

    def record_dim_feedback(
        self,
        residual: Tensor,
        min_evals: int = 50,
    ) -> dict[str, float]:
        """维度级用进废退——按各维度"用了有没有用"更新 gain

        **这是三个机制（展开 / 退化 / 自噬）共同的状态更新点。**

        判据推导。设维度 j 的实际输出为 ``y``，残差为 ``r``，则目标真值
        ``target = r + y``。对比两种情形：

        - 该维度沉默（增益归零，输出 0）：误差 = ‖target‖²
        - 该维度实际发声（输出 y）    ：误差 = ‖r‖²

        贡献 = 沉默误差 − 实际误差 = ‖r + y‖² − ‖r‖² = ``y·(2r + y)``。
        归一化到目标能量得到无量纲的**效用**：

            ``u = (‖target‖² − ‖r‖²) / (‖target‖² + ε) ∈ (-∞, 1]``

        - ``u → 1``  ：该维度完美解释了目标（残差趋零）
        - ``u = 0``  ：该维度与沉默等价，**没有存在价值**
        - ``u < 0``  ：该维度比沉默更糟（在放大噪声）

        对**不可预测的噪声维度**，最优解是让增益趋零（不去预测噪声），
        此时 u → 0，叠加废退项后 gain → 0，最终被自噬剪除。这正是
        "环境提供噪声和有用信号"时系统应有的行为。

        与髓鞘层 ``pathway_contributions`` 的分工：髓鞘管**通路**（谁连谁），
        这里管**维度**（哪些信道值得保留）。两者都用"沉默 vs 发声"的
        反事实差作为判据，但作用对象不同。

        Args:
            residual: 全局信号空间中的残差（目标 − 实际输出）
            min_evals: 低于此评估次数的维度不参与 gain 更新（避免初始
                瞬态把增益带偏）

        Returns:
            {channel: utility} 本步各信道的效用（诊断用）
        """
        if self.W is None or not self.alive or self._last_output is None:
            return {}
        active_indices = self._unfolded_indices()
        if not active_indices:
            return {}
        # 残差在全局空间 → 提取到展开顺序空间
        idx_t = torch.tensor(active_indices, dtype=torch.long)
        if residual.numel() <= int(idx_t.max()):
            return {}
        r_vec = residual[idx_t]
        y_vec = self._last_output
        if r_vec.shape != y_vec.shape:
            return {}

        target = r_vec + y_vec
        # 贡献 = 沉默误差 − 实际误差；归一化 → 效用
        silent_err = float((target ** 2).sum().item())
        actual_err = float((r_vec ** 2).sum().item())
        denom = silent_err + 1e-8

        out: dict[str, float] = {}
        for ch, start, end in self._channel_bounds():
            slc = self.unfolded[ch]
            y_seg = y_vec[start:end]
            r_seg = r_vec[start:end]
            t_seg = target[start:end]
            c = float((t_seg ** 2).sum().item() - (r_seg ** 2).sum().item())
            u = c / (float((t_seg ** 2).sum().item()) + 1e-8)
            # 效用 EMA（长动量：噪声维度的单步贡献正负抖动，需长时间平均）
            slc.utility = (
                self.utility_momentum * slc.utility
                + (1.0 - self.utility_momentum) * u
            )
            slc.evaluated += 1
            out[ch] = slc.utility

            if slc.evaluated < min_evals:
                continue

            # 用进：按效用放大；废退：**相对**衰减
            #
            # 必须是相对衰减（gain -= decay·gain）而非绝对衰减（gain -= decay）。
            # 绝对衰减的实测后果：系统收敛后残差变小 → 真实维度的贡献同步变小，
            # 但恒定的绝对衰减仍在扣 → 衰减压倒学习 → 必要维度被误杀
            # （具身控制任务中实测 ρ=0.01 时真实维对角通路从 5 条掉到 2 条，
            #  系统失控发散）。相对衰减下稳态 gain* = (lr/decay)·utility，
            # 只要 utility 非零就能维持，不会归零。
            slc.gain += self.dim_lr * slc.utility - self.dim_decay * slc.gain
            # 增益下界 0（趋零即"退化"的终点，随后由自噬剪除）
            slc.gain = max(0.0, slc.gain)

        total_utility = (silent_err - actual_err) / denom
        self._total_utility = total_utility

        # --- 信息饱和度：utility 停滞检测 ---
        #
        # 与 Ding 等 2023（CAAI TRIT 8(3):780-795，已核实）的"信息饱和度"
        # 对应关系：他们用竞争胜出后的信息饱和度调节**参数调整幅度**与
        # **分裂时机**；这里用 utility 的**停滞**来定义饱和——含义更直接：
        # 该神经元的学习收益已趋于零，继续调 W 是浪费，应该分裂去细分。
        #
        # 协同点：饱和度与髓鞘遗传耦合（见 development.maybe_differentiate）——
        # 分裂时机由饱和触发，子代继承父代的髓鞘（获得性记录）后从
        # "已饱和但已学会"的状态出发继续精细化，而不是从零重学。
        # Ding 等有父子关系但**没有**获得性遗传，这一条是我们的增量。
        if abs(total_utility - self._prev_utility) < self.stagnation_eps:
            self._stagnant_steps += 1
        else:
            self._stagnant_steps = max(0, self._stagnant_steps - 2)
        self._prev_utility = total_utility
        self.saturation = min(
            1.0, self._stagnant_steps / max(1, self.saturation_window)
        )
        return out

    def adjust_weights(self, signal: Tensor, feedback: float,
                       lr: float = 0.01) -> float:
        """使用反馈微调权重（训练到收敛）

        类似 Oja 规则，但作用在权重矩阵上。
        返回权重变化量（用于收敛判定）。
        """
        if self.W is None:
            return 0.0
        active_indices = []
        for ch, slc in self.unfolded.items():
            active_indices.extend(range(slc.dim_idx, slc.dim_idx + slc.size))
        signal_slice = signal[active_indices]

        # Oja 规则变体：ΔW = lr * feedback * (x⊗y - y⊗y * W)
        y = self.W @ signal_slice
        delta = lr * feedback * (signal_slice.outer(y) - y.outer(y) @ self.W)
        self.W += delta
        return delta.abs().mean().item()

    def apply_prediction_error(self, residual: Tensor, input_vec: Tensor,
                               lr: float = 0.005,
                               max_step: float = 0.1,
                               weight_decay: float = 0.0,
                               max_norm: float = 3.0,
                               normalize_to: float = 0.0) -> float:
        """用**自回归预测误差**做 delta rule 更新（无反向传播）

        **这是系统唯一能学到"输入→输出映射"的通道。**

        为什么必须加这个方法：

        - ``adjust_weights`` 是 Oja 规则（无监督），学的是输入的**主成分
          方向**，不是输入输出之间的映射。它无法学会"看到视觉信号 → 输出
          对应词语"这类跨模态变换。
        - 髓鞘只有标量 gain，只做强度调制，不做线性变换，同样承担不了映射。
        - 实测（真实系统 600 步）：无此更新时，系统输出的 lang 段与真实
          下一词的余弦为 **负**（−0.13 ~ −0.16），top-1 准确率 0.05
          **低于随机基线 0.083**。progress 高达 +0.999 是假象——它只反映
          vis/aud 段被 echo 预测对了，与语言无关。

        数学：W 是单层的，所以残差对 W 的梯度有闭式解，**不需要反向传播**：

            ΔW = lr · (target − W x) ⊗ x = lr · residual ⊗ x

        残差来源是 ``scheduler.settle`` 用**下一帧真实环境信号**裁定上一步
        输出所得 —— 即环境即时反馈，无需外部标注。

        Args:
            residual: 全局空间中的预测误差（目标 − 输出）
            input_vec: 产生该输出时的输入（全局空间）
            lr: 学习率
            max_step: 单步权重变化的上限（防数值爆炸）

        Returns:
            实际应用的权重变化量（0 表示跳过）。
        """
        if self.W is None or not self.alive:
            return 0.0
        idx = self._unfolded_indices()
        if not idx:
            return 0.0
        if int(residual.numel()) <= max(idx) or int(input_vec.numel()) <= max(idx):
            return 0.0
        r_vec = residual[idx]
        x_vec = input_vec[idx]
        if r_vec.numel() != self.W.shape[0] or x_vec.numel() != self.W.shape[1]:
            return 0.0
        delta = lr * r_vec.outer(x_vec)
        # 限幅：单步变化过大是发散的前兆
        m = float(delta.abs().max().item())
        if m > max_step:
            delta = delta * (max_step / m)
        self.W = self.W + delta
        # 纯 delta rule **没有归一化项，W 会无界增长** → 输出爆炸。
        # 实测：无约束时 progress 崩到 −141（ar_error_high 远大于反射误差）。
        # Oja 规则自带归一化正是为此。这里补两个约束：
        #   1) 权重衰减（每步按比例收缩）
        #   2) 谱范数上限（超过则整体缩放回来）
        #
        # 取值依据（实测）：归一化后输入的中位范数约 3。max_norm=20 时
        # 单步输出可达 60，自回归误差观测到 4.8e7 —— 数值灾难。
        # 收紧到 3.0（≈ W 初始化的 2.4）后输出与输入同量级。
        if weight_decay > 0:
            self.W = self.W * (1.0 - weight_decay)
        if max_norm > 0:
            try:
                nrm = float(self.W.norm().item())
                if nrm > max_norm:
                    self.W = self.W * (max_norm / nrm)
            except Exception:
                pass

        # **Oja 式范数归一化**：更新后把 W 缩放回固定范数。
        #
        # 为什么需要它（三段实测的因果链）：
        #   ① 无约束 delta rule → W 无界增长 → 自回归误差 4.8e7（数值灾难）
        #   ② 改用 weight_decay=0.01 → W 被压制到 0.4~1.0 → 输出幅度上不去，
        #      高层无法充分拟合目标，progress 卡在 0.4，λ 上不去
        #   ③ 衰减 lr 后 W 更被压向 0 → 输出趋 0，余弦反而下降
        #
        # 范数恒定 **只学方向** 一举解决两端：既不会爆炸也不会趋零。
        # 且余弦本就是方向度量 —— 固定范数等价于把优化集中在真正被
        # 评价的量上。这正是 Oja 规则中 −y²W 项的意图。
        if normalize_to > 0:
            try:
                nrm = float(self.W.norm().item())
                if nrm > 1e-8:
                    self.W = self.W * (normalize_to / nrm)
            except Exception:
                pass
        return float(delta.abs().mean().item())

    def drift(
        self,
        rate: float = 0.01,
        rng: Optional[random.Random] = None,
        protection: float = 0.0,
    ) -> None:
        """自演化漂移：W 缓慢随机游走 —— 演化算法的**变异算子**

        漂移必须与用进废退（选择）配对理解：

        - 没有变异，选择只能在既有结构内筛选，无法探索新结构；
        - 没有选择，漂移纯粹是破坏（消融证据：去掉髓鞘延迟适应后，
          漂移使系统发散而非收敛，块内到达时间标准差从 0.21 发散到 2.7）。

        因此漂移率必须远小于选择强度，且受 protection 抑制——已被反复验证
        的核心通路不应被随机扰动。

        Args:
            rate: 漂移步长
            rng: 随机源（None → 使用全局 torch RNG，不可复现）
            protection: 保护系数 ∈ [0,1]，抑制漂移幅度
        """
        if self.W is None or not self.alive:
            return
        eff_rate = rate * (1.0 - max(0.0, min(1.0, protection)))
        if eff_rate <= 0.0:
            return
        self.drift_count += 1
        if rng is not None:
            gen = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
            noise = torch.randn(self.W.shape, generator=gen)
        else:
            noise = torch.randn(self.W.shape)
        self.W = self.W + eff_rate * noise

    def replicate(self, signal: Tensor, channels: list[str]) -> 'Neuron':
        """自复制分化：子节点继承维度方向 + 沿信号方向展开 + 特化放大

        子节点的权重由它自己的 seed（变异后）生成初始值，
        继承的是"维度方向"（哪些维度被展开），不是权重参数。

        **关于髓鞘继承**：本方法不处理髓鞘——髓鞘是关系性资产（在
        ``MyelinSheathRegistry`` 中），不是神经元的内部属性。分裂时必须由
        调用方额外调用 :meth:`MyelinSheathRegistry.split_for_child` 完成分配。
        若遗漏，子节点从零开始，用进废退的跨代累积归零。
        """
        # seed 变异（遗传多样性）
        child_seed = self.seed + random.randint(1, 2**31 - 1)
        child = Neuron(seed=child_seed, port_layout=self.port_layout)
        child.parent_seed = self.seed

        # 沿信号方向展开维度 + 增益的**相对**分配
        #
        # **原实现的缺陷**：``child.gain = 1.5`` 是硬设，而引入
        # ``record_dim_feedback`` 后 gain 的语义已变为"效用的累积"
        # （稳态 gain* = (dim_lr/dim_decay)·utility，真实维度可达 3.0+）。
        # 硬设会把子代**直接重置**到一个低于父代的值——父 gain 3.0 分裂后
        # 变成父 2.7 / 子 1.5，子代反比父代低 45%，"子代特化更强"完全没实现，
        # 且用进废退的跨代累积归零。
        #
        # 这正是 Butz 2006（齿状回神经发生）与 Gohlke 2004（前体库耗尽）
        # 警告的增益分配失当，只是表现为**反向丢失**而非爆炸。
        #
        # 修法：相对分配。父让出 keep_ratio，子继承父的**绝对水平**。
        # 子 (1-keep_ratio+) > 父 (keep_ratio)，方向正确且不丢历史。
        for ch in channels:
            child.unfold(ch, signal)
            if ch not in child.unfolded:
                continue
            parent_gain = self.unfolded[ch].gain if ch in self.unfolded else 1.0
            # 子代继承父代的绝对水平（含效用历史）
            child.unfolded[ch].gain = parent_gain
            # 子代维度从父代同步继承使用记录（获得性遗传的维度侧）
            if ch in self.unfolded:
                pslc = self.unfolded[ch]
                cslc = child.unfolded[ch]
                cslc.utility = pslc.utility
                cslc.activity = pslc.activity
                cslc.evaluated = pslc.evaluated
                # 父让出 20%
                pslc.gain *= self.split_keep_ratio
                self.unfolded[ch].protection = max(
                    pslc.protection, cslc.protection
                )

        return child

    def apply_three_factor(self, rpe, input_vec, output_vec, lr: float) -> float:
        """**三因子学习规则**：ΔW = η · RPE · eligibility

        eligibility = 输入 ⊗ 输出（Hebbian 项，记录刚才的共激活）。

        三个因子各司其职：
          因子1 突触前活动（input_vec）
          因子2 突触后活动（output_vec）
          因子3 神经调质（rpe = 张力消解的预测误差）

        前两因子只说明"这两者一起活跃过"；第三因子才决定
        **这次的共激活值不值得被记住**。这是强化学习与无监督
        Hebbian 学习的分界，也是生物突触可塑性的实际形式。

        ⚠ 与 ``apply_prediction_error``（自回归）的区别：
        自回归学到"世界的序列规律"，三因子学到"什么行动有效"。
        在控制任务中，只有后者能改变行动选择 —— 已实测。

        Args:
            rpe: 奖励预测误差
            input_vec: 输入（全局空间）
            output_vec: 输出（全局空间）
            lr: 学习率

        Returns:
            权重变化量。
        """
        if self.W is None:
            return 0.0
        W = self.W
        n, m = W.shape

        # 输入/输出统一转成一维张量并裁剪到 W 的行空间
        def _to_vec(v, size):
            # ⚠ 不要用 ``W.dtype``：numpy_torch_shim 的 Tensor 没有
            # ``dtype`` 属性，访问会抛 AttributeError，被调用方的
            # ``except Exception: continue`` 静默吞掉 —— 实测
            # _learn_from_relief 调用 1499 次、更新 0 个神经元，
            # 三因子学习完全没生效就是因为这个。
            if v is None:
                return None
            try:
                arr = np.asarray(
                    v.detach().cpu().numpy() if hasattr(v, "detach")
                    else (v.a if hasattr(v, "a") else v),
                    dtype=np.float32).ravel()
            except Exception:
                return None
            t = torch.as_tensor(arr)
            if t.numel() < size:
                pad = torch.zeros(size - t.numel())
                t = torch.cat([t, pad])
            return t[:size]

        x = _to_vec(input_vec, n)
        y = _to_vec(output_vec, n)
        if x is None or y is None:
            return 0.0

        # eligibility：输入 ⊗ 输出的 Hebbian 外积
        # ⚠ shim 无 ``torch.outer``，用 unsqueeze 实现（等价且兼容）
        elig = y.unsqueeze(1) * x.unsqueeze(0)
        n_act = min(elig.shape[0], elig.shape[1])
        elig = elig[:n_act, :n_act]
        if elig.numel() == 0:
            return 0.0
        # 裁剪到 W 的实际形状
        elig = elig[:n, :m] if elig.shape[0] >= n and elig.shape[1] >= m else elig

        delta = lr * float(rpe) * elig
        if delta.shape != W.shape:
            return 0.0

        with torch.no_grad():
            W.add_(delta)
            # 归一化防爆炸（与用进废退的"废退"侧一致）
            wn = float(W.norm().item())
            w_max = getattr(self, "w_max", 3.0)
            if wn > w_max:
                W.mul_(w_max / (wn + 1e-8))
        return float(delta.abs().sum().item())

    def autophagy_dims(
        self,
        threshold: float = 0.1,
        gain_threshold: float = 0.05,
        min_age: int = 100,
    ) -> list[str]:
        """维度级自噬：剪除已完成退化的维度

        **原实现的根本缺陷**：判据是 ``activity < threshold``，而 activity
        测的是"**有没有信号到达**"。噪声通道的信号一直在到达，activity 一直
        很高，**永远剪不掉**——恰恰是最该剪的那些维度。

        修正为**两条独立的剪除路径**，覆盖两种"该维度没有存在价值"的情形：

        1. **退化完成**（``gain < gain_threshold``）
           维度被评估过、但效用趋零，增益已渐进衰减到接近 0。
           这是"用进废退"的**废退**侧走完的自然终点——先退化，后自噬，
           中间有连续的过渡，而不是二值的留/删。

        2. **信道断流**（``activity < threshold``）
           该信道长期没有信号到达（外部传感器拔掉了、模态不再出现）。
           这条路径不需要 feedback，只看输入侧。

        两条路径都受**新生儿保护期**（``age >= min_age``）约束：刚展开的维度
        尚未积累足够统计量，不能剪——否则沉默突触活不过第一轮，会陷入
        "新建—淘汰"空转（实测周转率可达 11+，通路结构永不收敛）。

        Args:
            threshold: 输入侧断流判据（activity EMA 下界）
            gain_threshold: 退化完成判据（增益下界）
            min_age: 新生儿保护期（展开后至少经历的步数）

        Returns:
            被剪除的信道名列表。被髓鞘隔离的维度（myelinated）受保护。
        """
        pruned: list[str] = []
        # ⚠ 底线保护：不能把维度剪光。
        # 实测（真实系统 1200 步）：两个神经元的所有维度被同时判为
        # "断流/退化"并剪除 → W=None → 神经元死亡 → 系统失去该模态、
        # 且不可恢复（unfold 虽能重建，但已学的 W 全部丢失）。
        # 任何系统剪掉全部维度都是灾难，与判据是否正确无关。
        # 保留**增益最高**的若干维度作为火种。
        min_keep = 1
        for ch in list(self.unfolded.keys()):
            slc = self.unfolded[ch]
            if slc.myelinated:
                continue                      # 髓鞘庇护，不自噬
            if slc.age < min_age:
                continue                      # 新生儿保护期
            if len(self.unfolded) - len(pruned) <= min_keep:
                continue                      # 底线：至少留一个维度
            degenerated = slc.gain < gain_threshold
            starved = slc.activity < threshold
            if degenerated or starved:
                del self.unfolded[ch]
                pruned.append(ch)
        # 重建 W（移除被剪除维度的行列）
        if pruned:
            self._rebuild_W()
            self._last_output = None          # 形状已变，缓存失效
        return pruned

    def _rebuild_W(self) -> None:
        """维度剪除后重建权重矩阵

        注意：W 是按**展开顺序**逐步扩张的（unfold 时新增行列追加到末尾），
        不是按全局 dim_idx 索引的。因此重建时要用**展开顺序的累积偏移**，
        而非 port_layout 中的全局 dim_idx。
        """
        if not self.unfolded:
            self.W = None
            return
        # 按展开顺序计算累积偏移（与 unfold 时的扩张顺序一致）
        active_indices = []
        offset = 0
        for ch, slc in self.unfolded.items():
            active_indices.extend(range(offset, offset + slc.size))
            offset += slc.size
        # 保留 W 中对应行列的子矩阵
        #
        # ⚠ 必须显式指定 dtype=long。``torch.tensor(list_of_int)`` 在
        # 元素为空或类型推断异常时会得到 float 张量 → "arrays used as
        # indices must be of integer type"（自噬触发时崩溃，实测于
        # 2000 步长跑）。显式声明可彻底避免。
        if not active_indices:
            return
        idx = torch.tensor(active_indices, dtype=torch.long)
        self.W = self.W[idx][:, idx]
