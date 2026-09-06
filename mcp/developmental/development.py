"""发育引擎 — 分化 / 自噬 / 髓鞘强化 / 收敛判据

所有规则基于局部统计量，无反向传播。
新奇性基于覆盖度（find_coverage），不是余弦相似度。
髓鞘是包裹层（thicken_sheath/decay_sheath），不再是 Q 矩阵。
"""
from __future__ import annotations
import numpy as np
import torch

from mcp.developmental.neurode import NeuronEcosystem
from mcp.developmental.myelin import MyelinSheathRegistry, ActivationRecord
from mcp.developmental.neuron import ConvergenceState


class DevelopmentEngine:
    """发育引擎：编排分化、自噬、髓鞘强化、收敛判据"""

    def __init__(
        self,
        ecosystem: NeuronEcosystem,
        sheath_registry: MyelinSheathRegistry,
        novelty_threshold: float = 0.7,
        novelty_decay: float = 0.99,
        stability_threshold: float = 0.6,
        stability_min_gain: float = 1.0,
        inherit_ratio: float = 0.8,
        saturation_split: bool = True,
        saturation_threshold: float = 0.9,
    ):
        self.ecosystem = ecosystem
        self.sheath_registry = sheath_registry
        self.novelty_threshold = novelty_threshold
        self.novelty_decay = novelty_decay
        # 分化触发的"稳定反应"判据
        self.stability_threshold = stability_threshold
        self.stability_min_gain = stability_min_gain
        # 分裂时父节点保留的髓鞘比例（子节点获得 1 - inherit_ratio）
        self.inherit_ratio = inherit_ratio
        self._novelty_accumulator = 0.0
        self._differentiation_count = 0
        self._total_signal_count = 0
        # 惊奇（预测误差）通道：0=关闭，只走覆盖度
        self.surprise_scale = 1.0
        # 自回归误差基线（EMA），用于归一化惊奇
        self._ar_error_baseline = 1.0
        self._last_surprise = 0.0
        # 生长闸门状态
        self._diff_cooldown = 0
        self.diff_refractory = 200      # 分裂后等待步数

        # --- 表征冲突：需求驱动的分化判据 ---
        #
        # ⚠ 惊奇（NMSE）作为分化触发判据**已证实失效**：
        # 任务的不可约误差（一阶 Oracle 仅 0.341）使 NMSE 恒接近 1，
        # 于是"只要时间够就分裂"。实测（diag_gate.py）：
        #   不应期 50 → 神经元 16    不应期 200 → 12
        #   不应期 400 → 7           不应期 800 → 4
        # 神经元数 ≈ 步数 / 不应期 —— **分化由闸门节拍驱动，不是需求**。
        # 闸门从"抑制失控"异化成了"节拍器"。
        #
        # 正确的触发条件应该是：**相似的上下文要求互斥的不同输出**。
        # 单个神经元只能表示一个输出方向，遇到这种冲突就必须分裂。
        # 这与误差的绝对大小无关，只取决于任务结构。
        # one-hot 裸编码下：不同词的余弦**恒为 0**，故"互斥"自动成立；
        # 关键是"相同处境"的判定 —— 只有同一个上文词才算相同处境，
        # 余弦接近 1（vis/aud 也相同）。设 0.9 保证不把不同词混为一谈。
        self.conflict_sim_threshold = 0.9   # 上下文"相同"的余弦下界
        self.conflict_div_threshold = 0.3   # 输出"互斥"的余弦上界（one-hot 下恒满足）
        self.conflict_trigger = 20          # 累积多少次冲突才分裂
        self.max_ctx_protos = 64            # 上下文原型上限
        self._ctx_protos: list[dict] = []
        self._conflict_count = 0
        self._obs_calls = 0   # 诊断：observe_context 调用次数
        self._last_ctx_sim = 0.0   # 诊断：最近一次上下文匹配余弦
        self._last_out_sim = 0.0   # 诊断：最近一次输出余弦
        self._obs_conflicts = 0    # 诊断：累计判定为冲突的次数
        self._error_ema = None
        self._error_prev_ema = None
        self.plateau_eps = 0.02         # 相对改善低于此值视为平台

        # --------------------------------------------------------------
        # 消融结论（真实系统，3 seeds × 2000 步，见 final_eval.py）
        # --------------------------------------------------------------
        #   A 基线（覆盖度新奇）      top1 0.107  神经元 1.3   分化 0.3
        #   B +NMSE 惊奇             top1 0.167  神经元 11.3  分化 10.3
        #   C +生长闸门              top1 0.164  周转 1017
        #   D +滞回豁免（当前）       top1 0.169±0.018  周转 801
        #
        # 结论 1：**惊奇通道是决定性的**（+56%）。覆盖度判据在"一个神经元
        #   处理所有信号"的设计下恒为 1.0，结构上不可能触发分化。
        # 结论 2：生长闸门在 2000 步内**不提升** top1（0.167→0.164，噪声内）。
        #   它的价值在更长的运行：3000 步时无闸门会失控（神经元 48、
        #   周转 9035、progress −8.74）。它是长程稳定性保险，非短期收益。
        # 结论 3：滞回主要降低**方差与周转**（±0.049→±0.018，973→801），
        #   并让 top3/余弦达最高。
        #
        # **瓶颈已定位（勿再调学习机制）**：top1 卡在 ~0.19 不是"学不会"，
        # 而是"分不开"。实测（diag_bottleneck.py，2500 步）：
        #   同类词向量余弦   0.803   （mama/baba/baobao 几乎重合）
        #   cos(输出, 真值)  0.553   ← 学到了语义
        #   cos(输出, 竞争者) 0.745  ← **比真值还高**
        #   真值余弦 > 竞争者余弦的比例仅 19.2%
        # 线性模型输出的是"合理反应的平均"，平均后谁都不像 →
        # 最近邻必然落到同类的另一个词。这是**表示/解码**的天花板。
        #
        # 按"合理反应"评价（diag_reaction.py）则完全不同：
        #   末 500 步  top1=0.190  top3=0.524  **类别正确=0.544**
        #   （随机基线：top1=0.083 / top3=0.250 / 类别=0.250）
        # 即：系统能给出**语义合理的反应**，只是分不出同类个体。
        # 这与设计诉求"重点是合理的反应而不是预测"一致。
        #
        # 参考线：随机基线 top1=0.083，一阶 Oracle=0.341。
        # 当前达 Oracle 的 50%，为随机基线的 2 倍。
        # --------------------------------------------------------------
        # 当前目标能量 |target|²，用于 NMSE 归一化
        self._target_energy = 1.0
        # 信息饱和度触发分裂（协同 C）
        self.saturation_split = saturation_split
        self.saturation_threshold = saturation_threshold
        self._last_trigger: str = ""

    def accumulate_novelty(
        self,
        signal: torch.Tensor,
        active_channels: set[str],
        ar_error: Optional[float] = None,
        target_energy: Optional[float] = None,
    ) -> None:
        """累积新奇性：覆盖度不足 **或** 预测不了（惊奇）

        两条互为补充的通道：

        **1. 覆盖度不足**（结构性）
           ``novelty = 1 − coverage_score``，对应"遇到了没见过的模态组合"。

        **2. 惊奇**（内容性）—— 本次新增
           归一化自回归预测误差，对应"这个输入我处理不了"。

        ⚠ **为什么必须有通道 2**（实测，真实系统 1500 步）：

        在"一个神经元处理所有信号"的设计下，神经元展开了全部信道，
        ``coverage_score`` **恒为 1.0** → 新奇性恒为 0 → 分化永不触发。
        实测：``nov_acc`` 全程 0.000，1500 步只靠饱和路径分裂了 1 次，
        之后停滞在 2 个神经元。

        这不是调参问题——覆盖度判据**在结构上**看不见"内容层面的新奇"，
        它只数信道名。而"没见过的新东西"恰恰常表现为**同样的信道、
        不同的内容**（新词、新句式）。

        惊奇的定义有坚实的理论支撑：预测编码 / 自由能原理中，
        惊奇（surprise）就是驱动模型更新与结构生长的量。

        Args:
            ar_error: 本步的自回归预测误差。None 时只走覆盖度通道。
        """
        _, coverage_score = self.ecosystem.find_coverage(signal, active_channels)
        novelty = 1.0 - coverage_score

        surprise = 0.0
        if ar_error is not None and self.surprise_scale > 0:
            # 归一化：**NMSE**（误差 / 目标能量），绝对标度，无需自适应基线。
            #
            # ⚠ 原实现用历史误差 EMA 做分母，实测（真实系统 1500 步）失效：
            # 训练初期误差达 2.8e7，EMA 衰减（0.99）需要约 100 步适应，
            # 而误差在 1500 步内下降 5 个数量级 → 基线永远滞后在真实误差
            # 之上 → 惊奇恒 ≈0 → 分化仍不触发（实测 surprise 仅 0.226）。
            #
            # NMSE 的含义是明确且绝对的：
            #   0   = 完美预测
            #   1   = 和"预测恒为 0"一样差
            #   >1  = 还不如不预测
            # 它不依赖历史，训练初期接近 1（该长结构），学成后自然降到
            # 任务的不可约误差（该停止生长）—— 正是发育应有的时间剖面。
            denom = max(self._target_energy, 1e-8)
            surprise = min(1.0, float(ar_error) / denom)

        if target_energy is not None and target_energy > 0:
            self._target_energy = float(target_energy)
        # 生长闸门状态推进
        if self._diff_cooldown > 0:
            self._diff_cooldown -= 1
        if ar_error is not None:
            e = float(ar_error)
            self._error_prev_ema = self._error_ema
            self._error_ema = (
                e if self._error_ema is None
                else 0.95 * self._error_ema + 0.05 * e
            )

        combined = max(novelty, surprise)
        self._novelty_accumulator = (
            self.novelty_decay * self._novelty_accumulator
            + (1 - self.novelty_decay) * combined
        )
        self._last_surprise = surprise
        self._total_signal_count += 1

    def _error_plateaued(self) -> bool:
        """误差是否已进入平台（不再显著下降）

        目的：区分「学不会」和「还没学会」。

        - 误差仍在下降 → 现有结构还在学，此时分裂是**打断学习**；
        - 误差不再下降且仍很高 → 现有结构不够用，此时分裂才是**扩容**。

        判据：近期相对改善 ``(prev − now) / prev < plateau_eps``。

        需要足够的历史才有意义：误差 EMA 尚未建立时返回 False
        （宁可不分裂，也不要在信息不足时盲目生长）。
        """
        if self._error_ema is None or self._error_prev_ema is None:
            return False
        prev = self._error_prev_ema
        if prev <= 1e-12:
            return True
        rel = (prev - self._error_ema) / prev
        return rel < self.plateau_eps

    def has_stable_output_reaction(self) -> bool:
        """是否存在"稳定反应"的输出通路——分化触发的第二个条件

        判据：某条髓鞘的 ``stability`` 超阈值，且 ``gain`` 维持在高位。

        ``stability`` 由 :meth:`MyelinSheath.apply_selection` 累积，记录的是
        "该通路**已经被**反复、稳定地使用"这一**历史事实**。

        这是**反应**而非预测：它不看"未来可能需要什么"，只看"什么已经被
        稳定地用起来了"。对应生物学——稳定使用的环路先被髓鞘化固化，
        然后才在此基础上进一步分化。

        Returns:
            bool: 存在稳定反应通路则为 True。
        """
        for s in self.sheath_registry._sheaths.values():
            if (s.stability >= self.stability_threshold
                    and s.gain >= self.stability_min_gain):
                return True
        return False

    def observe_context(self, input_vec, target_vec) -> None:
        """记录「上下文 → 应有的输出」，累积**表征冲突**

        **表征冲突** = 相似的输入要求**互斥**的不同输出。

        这是真正的"需要更多容量"的信号，且与误差的绝对大小无关：

        - 一个神经元只能表示一个输出方向；
        - 当相同处境反复要求相距很远的输出时，它只能输出平均值，
          **谁都不像** —— 这正是实测观察到的现象
          （cos(输出,竞争者) 0.745 > cos(输出,真值) 0.553）；
        - 此时分裂才有意义：让两个子代各自承接一种反应。

        ⚠ 用**目标输出**（世界要求的反应）而非系统自己的输出来判断
        分歧 —— 要检测的是"世界要求什么"，不是"系统有多困惑"。

        Args:
            input_vec: 当前输入（全局信号空间）
            target_vec: 真实的下一帧（"应有的反应"）
        """
        self._obs_calls += 1
        x = self._to_np(input_vec)
        y = self._to_np(target_vec)
        if x is None or y is None:
            return
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        if nx < 1e-8 or ny < 1e-8:
            return
        x = x / nx
        y = y / ny

        best_sim, best_i = -1.0, -1
        for i, p in enumerate(self._ctx_protos):
            s = float(np.dot(x, p["ctx"]))
            if s > best_sim:
                best_sim, best_i = s, i

        if best_sim < self.conflict_sim_threshold:
            # 新上下文：记下来，不构成冲突
            if len(self._ctx_protos) < self.max_ctx_protos:
                self._ctx_protos.append(
                    {"ctx": x.copy(), "out": y.copy(), "n": 1})
            else:
                # 满了：替换使用次数最少的原型
                j = min(range(len(self._ctx_protos)),
                        key=lambda k: self._ctx_protos[k]["n"])
                self._ctx_protos[j] = {"ctx": x.copy(), "out": y.copy(), "n": 1}
            return

        p = self._ctx_protos[best_i]
        lr = 1.0 / (p["n"] + 1)
        p["ctx"] = (1 - lr) * p["ctx"] + lr * x
        p["ctx"] /= (float(np.linalg.norm(p["ctx"])) + 1e-9)
        p["n"] += 1

        out_sim = float(np.dot(y, p["out"]))
        self._last_ctx_sim = best_sim
        self._last_out_sim = out_sim
        if out_sim < self.conflict_div_threshold:
            self._obs_conflicts += 1
            # 同样的处境，要求了完全另一种反应 → 冲突
            self._conflict_count += 1
        else:
            p["out"] = (1 - lr) * p["out"] + lr * y
            p["out"] /= (float(np.linalg.norm(p["out"])) + 1e-9)
            self._conflict_count = max(0, self._conflict_count - 1)

    def _to_np(self, v):
        """把 torch/numpy 张量转成一维 float numpy；失败返回 None"""
        if v is None:
            return None
        try:
            a = v.detach().cpu().numpy() if hasattr(v, "detach") else (
                v.a if hasattr(v, "a") else v)
            a = np.asarray(a, dtype=np.float64).ravel()
            return a if a.size > 0 and np.all(np.isfinite(a)) else None
        except Exception:
            return None

    def maybe_differentiate(
        self,
        signal: torch.Tensor,
        active_channels: set[str],
    ) -> int:
        """分化触发：**新奇性** AND **对输出部分的髓鞘化稳定反应**

        两个条件必须同时满足：

        1. ``novelty_accumulator ≥ novelty_threshold``（覆盖度判据）
        2. :meth:`has_stable_output_reaction` 为真（存在已稳定使用的输出通路）

        条件 2 的意义：稳定反应是**反应**而非预测。它确保分化发生在"已有
        环路被充分验证、固化"的基础上，而不是在噪声中盲目扩张。这正是
        "先有稳定的输出反应，再在此基础上分化"的发育顺序。

        只有新奇性一个条件时，系统会在尚未建立任何可靠通路时就分化，
        分裂出的是无根基的随机结构。

        分裂后调用 :meth:`MyelinSheathRegistry.split_for_child` 完成**获得性
        遗传**：W 由 seed 重抽（先天、不可遗传），髓鞘按 usage 分配
        （获得性、可遗传）。遗漏此步则用进废退的跨代累积归零。

        返回新神经元 idx（未触发返回 -1）。
        """
        if self.ecosystem.count() >= self.ecosystem.max_nodes:
            return -1

        # ⚠ 生长闸门必须**先于**路径选择，对两条路径同时生效。
        #
        # 旧实现把不应期检查放在新奇路径内部（``if novel: if cooldown...
        # novel=False``），于是新奇路径被拦住后，控制流落到
        # ``elif saturated_idx >= 0`` —— **饱和路径完全绕过了闸门**。
        #
        # 实测后果：神经元持续增长（2500 步达 20~35 个），因为分裂只是
        # 换了一条路径继续发生。不应期形同虚设。
        if self._diff_cooldown > 0:
            return -1

        # --- 触发判据：新奇性 → 或 → 信息饱和度（协同 C）---
        #
        # 原判据只有新奇性（覆盖度不足），对应"遇到了没见过的新东西"。
        # 但发育生物学与 Ding 等 2023（CAAI TRIT 8(3):780-795，已核实）
        # 表明还有**第二类**分裂时机：神经元已充分学到它能学的
        # （信息饱和），需要分裂去**细分**已有类别，而不是等待新类别出现。
        #
        # 这里用 utility 停滞定义饱和（见 Neuron.record_dim_feedback）。
        # 两条路径共用同一套分裂与遗传逻辑，只是触发条件不同：
        #   新奇性路径 → 覆盖未知（往外扩）
        #   饱和路径   → 细分已知（往深处走）
        novel = self._novelty_accumulator >= self.novelty_threshold

        # ⚠ 生长闸门（实测必需，真实系统 3000 步）
        #
        # 只有"惊奇高"一个条件时，系统在**还没来得及学**的阶段就被判定
        # "处理不了" → 立即分裂 → 新结构也来不及学 → 又判定处理不了
        # → 又分裂 …… 实测失控：
        #     步 500→3000：神经元 1 → 48，分化 47 次，周转 9035，
        #     progress 从 +0.66 崩到 −8.74。
        # 这正是"癌细胞前体"的运行形态：以自身增殖为动力、无负反馈。
        #
        # 补两个闸门，二者缺一不可：
        #
        # (a) **不应期**：分裂后强制等待，给新结构学习的时间窗口。
        #     没有它，生长速率由惊奇决定，而惊奇在学习发生前恒为高。
        #
        # (b) **误差平台**：只有当误差**不再下降**时才判定"结构不够用"。
        #     误差仍在快速下降说明现有结构还在学，此时分裂是打断学习。
        #     这条把"学不会"和"还没学会"区分开——此前的判据混淆了二者。
        if novel and not self._error_plateaued():
            # 误差仍在下降 → 现有结构还在学，此时分裂是打断学习
            novel = False

        # **分化改用表征冲突判据**（替代惊奇，见 :meth:`observe_context`）
        #
        # 惊奇（NMSE）恒接近 1 是任务不可约误差所致，与"需不需要更多
        # 容量"无关，用它会得到节拍式生长。冲突判据只在"相同处境要求
        # 互斥反应"时触发，才是需求驱动。
        conflicted = self._conflict_count >= self.conflict_trigger
        novel = False          # 惊奇仅保留作诊断，不再触发分化
        saturated_idx = self._find_saturated_neuron() if (
            self.saturation_split and not conflicted
        ) else -1

        if conflicted:
            if not self.has_stable_output_reaction():
                return -1
            trigger = "conflict"
            # 分裂**最冲突**的那个神经元：即当前上下文的最佳覆盖者
            best_idx, _ = self.ecosystem.find_coverage(signal, active_channels)
            if best_idx < 0:
                return -1
        elif novel:
            if not self.has_stable_output_reaction():
                return -1
            trigger = "novelty"
            best_idx, _ = self.ecosystem.find_coverage(signal, active_channels)
            if best_idx < 0:
                return -1
        elif saturated_idx >= 0:
            trigger = "saturation"
            best_idx = saturated_idx
        else:
            return -1

        parent = self.ecosystem.get_neuron(best_idx)
        if parent is None:
            return -1

        # 调用 replicate 产生子神经元（特化放大 + 父节点削弱）
        child = parent.replicate(signal, list(active_channels))
        idx = self.ecosystem.add_neuron(child)
        if idx < 0:
            return -1

        # 获得性遗传：髓鞘按 usage 在父子间分配（W 不继承，由 seed 重抽）
        self.sheath_registry.split_for_child(
            parent=best_idx, child=idx, keep_ratio=self.inherit_ratio
        )

        # 重置触发状态：饱和触发的分裂会重置父节点的停滞计数，
        # 给它一个重新积累的窗口（否则会连续分裂直到撞上限）。
        self._novelty_accumulator = 0.0
        if trigger == "conflict":
            self._conflict_count = 0
        self._diff_cooldown = self.diff_refractory
        if trigger == "saturation":
            parent._stagnant_steps = 0
            parent.saturation = 0.0
        self._differentiation_count += 1
        self._last_trigger: str = trigger
        return idx

    def _find_saturated_neuron(self) -> int:
        """找已达信息饱和的神经元（饱和路径的分裂对象）

        与 :meth:`NeuronEcosystem.find_coverage` 的选择逻辑**相反**：
        覆盖度选"最能处理当前信号的"（往外扩），饱和技术选"已经学
        不动了的"（往深处细分）。

        饱和判据：``saturation >= saturation_threshold`` 且该神经元
        已积累足够评估次数（避免选中刚展开的、尚未被充分评估的新维度）。

        Returns:
            神经元 idx，无满足条件的返回 -1。
        """
        best_idx = -1
        best_sat = self.saturation_threshold
        for idx, neuron in self.ecosystem.neurons.items():
            if not neuron.alive or not neuron.unfolded:
                continue
            if neuron.saturation < best_sat:
                continue
            # 需已充分评估（否则是新展开的维度，还没机会饱和）
            evaluated = sum(1 for s in neuron.unfolded.values() if s.evaluated > 50)
            if evaluated == 0:
                continue
            best_sat = neuron.saturation
            best_idx = idx
        return best_idx

    def maybe_differentiate_joint(
        self,
        signal: torch.Tensor,
        parent_a: int,
        parent_b: int,
        active_channels: set[str],
    ) -> int:
        """联合分化——从两个父节点分化，产生多信号神经元

        子节点继承两个父节点的维度方向。
        """
        if self.ecosystem.count() >= self.ecosystem.max_nodes:
            return -1

        neuron_a = self.ecosystem.get_neuron(parent_a)
        neuron_b = self.ecosystem.get_neuron(parent_b)
        if neuron_a is None or neuron_b is None:
            return -1

        # 合并两个父节点的维度方向 + 信号活跃信道
        combined_channels = (
            set(neuron_a.unfolded.keys())
            | set(neuron_b.unfolded.keys())
            | active_channels
        )

        # 调用 parent_a.replicate 创建子节点（削弱 parent_a 的 gain）
        child = neuron_a.replicate(signal, list(combined_channels))

        # 手动削弱 parent_b 的 gain（replicate 只削弱 parent_a）
        for ch in combined_channels:
            if ch in neuron_b.unfolded:
                neuron_b.unfolded[ch].gain *= 0.9

        idx = self.ecosystem.add_neuron(child)
        if idx < 0:
            return -1

        self._differentiation_count += 1
        return idx

    def autophagy(
        self,
        starvation_threshold: float = 0.01,
        dim_activity_threshold: float = 0.1,
        gain_threshold: float = 0.05,
        min_age: int = 100,
    ) -> dict:
        """自噬——维度级 + 饥饿级 + 髓鞘清理

        三个层级分工：

        1. **维度级**：剪除已完成退化的维度（先退化后自噬，中间连续过渡）
        2. **饥饿级**：清除代谢耗尽的神经元（前提是 :meth:`tick_metabolism`
           每步被调用，否则 metabolism 恒为 0.5，此分支永不触发）
        3. **髓鞘清理**：回收死亡神经元的关系性资产

        Returns:
            {"dims_pruned": {idx: [channels]}, "starved": [indices]}
        """
        # 1. 维度级自噬
        dims_pruned = self.ecosystem.autophagy_dims(
            threshold=dim_activity_threshold,
            gain_threshold=gain_threshold,
            min_age=min_age,
        )

        # 2. 饥饿级自噬
        starved = self.ecosystem.autophagy_starvation(
            starvation_threshold=starvation_threshold,
        )

        # 3. 清理死亡神经元的髓鞘
        for idx in starved:
            self.sheath_registry.remove_sheaths_for(idx)

        return {"dims_pruned": dims_pruned, "starved": starved}

    def dimension_report(self) -> dict:
        """维度健康诊断：展开 / 退化 / 自噬的状态快照

        用于观察三个机制是否真在工作：

        - ``unfolded``      : 已展开维度数（活动非依赖的先天布线）
        - ``degenerating``  : 正在退化的维度数（0 < gain < 阈值·4）
        - ``prunable``      : 已满足自噬条件的维度数（下次 autophagy 会被剪）
        - ``protected``     : 受髓鞘庇护 / 处于新生儿保护期的维度数
        - ``mean_utility``  : 平均效用（>0 表示整体上"用了有用"）

        健康的发育过程应看到：早期 unfolded 上升，中期出现 degenerating，
        后期 prunable 被清除且 gain 分布两极分化（真的留下少数高效维度）。
        若 degenerating 恒为 0，说明维度退化没有发生；
        若 prunable 恒为 0 且 unfolded 单调增长，说明自噬没在工作。
        """
        unfolded = degenerating = prunable = protected = 0
        utils, gains = [], []
        for neuron in self.ecosystem.neurons.values():
            if not neuron.alive:
                continue
            for slc in neuron.unfolded.values():
                unfolded += 1
                utils.append(slc.utility)
                gains.append(slc.gain)
                if slc.myelinated or slc.age < 100:
                    protected += 1
                    continue
                if slc.gain < 0.05:
                    prunable += 1
                elif slc.gain < 0.2:
                    degenerating += 1
        n = max(1, len(utils))
        return {
            "unfolded": unfolded,
            "degenerating": degenerating,
            "prunable": prunable,
            "protected": protected,
            "mean_utility": sum(utils) / n,
            "mean_gain": sum(gains) / n,
            "evaluated_dims": sum(
                1 for neuron in self.ecosystem.neurons.values()
                if neuron.alive
                for s in neuron.unfolded.values() if s.evaluated > 0
            ),
        }

    def thicken_myelin_from_coincidence(
        self,
        pairs: list[tuple[ActivationRecord, ActivationRecord]],
    ) -> int:
        """从同时激活对建立/强化髓鞘包裹层

        对每对 (A, ch_a) + (B, ch_b)：
        - 同信道（ch_a == ch_b）→ 增厚或建立髓鞘
        - 不同信道 → 跳过
        返回强化的连接数。
        """
        count = 0
        for A, B in pairs:
            if A.channel != B.channel:
                continue
            # 同信道 → 尝试增厚（强化）
            ok = self.sheath_registry.thicken_sheath(
                A.neuron, A.channel, B.neuron, B.channel)
            if not ok:
                # 髓鞘不存在 → 建立
                self.sheath_registry.add_sheath(
                    A.neuron, A.channel, B.neuron, B.channel)
            count += 1
        return count

    def check_convergence(
        self,
        weight_delta: float,
        myelin_stable: bool,
    ) -> ConvergenceState:
        """返回收敛状态

        differentiation_rate = 本轮分化次数 / 总信号数。
        """
        if self._total_signal_count > 0:
            diff_rate = self._differentiation_count / self._total_signal_count
        else:
            diff_rate = float('inf')
        return ConvergenceState(
            weight_delta=weight_delta,
            myelin_stable=myelin_stable,
            differentiation_rate=diff_rate,
        )
