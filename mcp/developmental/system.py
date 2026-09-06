"""DevelopmentalSystem — 发育智能系统主循环

双模主循环（以 port_layout 构造）：
  觉醒态（AWAKE）：读取入端口 → 高层脑 process 介入 → 反射弧 → 高层脑 intervene
                  → 事件驱动 dispatch_signal 强化路径 → 写出端口 → 记录 4 元组
                  → 累积新奇性 → 可能分化
  睡眠态（SLEEP）：四相位做梦学习（sheath_registry 操作，无梯度）
                  + 睡眠感觉敏感性（外界输入注入 + lambda_sleep 压低）

核心变更（相比旧实现）：
- myelin.* → sheath_registry.*（homeostatic_scale / thicken_sheath / decay_sheath）
- propagate_signal / simulate_internally → dispatch_signal（事件驱动并行分发）
- myelin._edges → sheath_registry._sheaths（统计髓鞘数）
- global_dim 构造 → port_layout 构造
- higher_brain.learn → engine.accumulate_novelty + maybe_differentiate + 事件驱动 sheath 强化
- 新增 check_convergence 收敛判据集成
"""
from __future__ import annotations
import json
import os
import random
import torch
from enum import Enum
from typing import Any, Optional

from mcp.developmental.signal import Signal
from mcp.developmental.port import PortRegistry, InputPort, OutputPort
from mcp.developmental.reptilian import ReptilianKernel
from mcp.developmental.route import Route, RouteExecutor
from mcp.developmental.reflex import ReflexArc
from mcp.developmental.higher_brain import DefaultHigherBrain
from mcp.developmental.neuron import ConvergenceState
from mcp.developmental.selection import pathway_contributions, DriveSatisfaction


class SystemMode(Enum):
    """系统模式：觉醒 / 睡眠"""
    AWAKE = "awake"
    SLEEP = "sleep"


class DevelopmentalSystem:
    """发育智能系统主循环（觉醒/睡眠双模 + 四相位做梦学习 + 事件驱动）"""

    def __init__(
        self,
        port_layout: dict[str, tuple[int, int]],
        max_nodes: int = 10000,
        decay_rate: float = 0.01,
        use_scale: float = 0.5,
        drift_rate: float = 0.005,
        drift_interval: int = 50,
        wiring_block: int = 50,
        target_influx: float = 8.0,
        protection_cost: float = 1.0,
        saturation_split: bool = True,
        adaptive_budget: bool = False,
        use_attention: bool = True,
        use_spikes: bool = True,
        attn_depth: float = 1.0,
        attn_lr: float = 0.05,
        spike_threshold: float = 0.2,
        spike_decay: float = 0.3,
        spike_refractory: int = 1,
        eprop: bool = False,
    ):
        """
        Args:
            port_layout: 信道布局 {channel_name: (offset, size)}
            max_nodes: 最大神经元数
            decay_rate: 髓鞘基础衰减率 ρ。**这是本系统最重要的参数**——
                它同时是选择阈值：存活条件为
                ``contribution · use_scale > ρ · (1 - protection)``。
                纯衰减无硬约束下，总容量上界 ≈ 总使用增量流 / ρ，
                随环境活跃度伸缩。若 :meth:`capacity_report` 显示边数持续
                增长而 mean_usage 不升，说明环境活跃度超出承载 → 提高 ρ。
            use_scale: 贡献到髓鞘增量的缩放
            drift_rate: 神经元自演化漂移步长（变异算子）
            drift_interval: 每隔多少步执行一次漂移
            wiring_block: 结构更新（成边/脱落）的 block 长度。**必须与权重
                修饰拉开一个量级**——成边每步都跑会与选择动力学处于同一
                时间尺度，导致"新建—淘汰"空转（实测 turnover 11.5）。
                见 :meth:`MyelinSheathRegistry.should_run_wiring`。
            target_influx: 每个目标神经元的设计总输入（承载力/发放率截断）。
                代谢约束，非外部硬上限。见 :meth:`...registry.influx_of`。
            protection_cost: protection 的预算权重 λ。**保护的通路占用
                更多预算**，使"护住旧技能"有真实代价，防止 protection
                无界累积。λ=0 则保护免费（退化为无预算约束）。
            saturation_split: 是否启用信息饱和度触发分裂（协同 C）。
                启用后，分裂时机由"utility 停滞"决定而非仅靠新奇性，
                子代继承父代髓鞘后从"已饱和但已学会"的状态继续精细化。
        """
        self.port_layout = port_layout
        self.global_dim = sum(s for _, s in port_layout.values())

        # 核心组件装配
        self.ports = PortRegistry()
        self.reptile = ReptilianKernel()
        self.routes: list[Route] = []
        self.reflex = ReflexArc(kernel=self.reptile)
        self.higher_brain = DefaultHigherBrain(
            port_layout=port_layout, max_nodes=max_nodes,
            saturation_split=saturation_split,
            use_attention=use_attention, use_spikes=use_spikes,
            attn_depth=attn_depth, attn_lr=attn_lr,
            spike_threshold=spike_threshold, spike_decay=spike_decay,
            spike_refractory=spike_refractory,
        )
        self.executor = RouteExecutor(self.reptile)

        # 超模态统一编码（逐信道自适应归一化）
        self.normalizer = self.higher_brain.normalizer

        # 选择动力学参数（用进废退）
        self.decay_rate = decay_rate
        # 自回归 delta rule 的学习率。0 = 关闭（会退化成纯 Oja，
        # 学不到输入输出映射，实测 lang 段余弦为负）。
        self.autoreg_lr = 0.005
        # 学习率衰减时间常数（步）。越大衰减越慢。
        self.autoreg_lr_tau = 500.0

        # --------------------------------------------------------------
        # 实测记录（婴幼儿语言任务，真实系统 1200 步，勿凭直觉改动）
        # --------------------------------------------------------------
        # 1. **反射基线必须提供量级合理的起点**。
        #    babbling（零附近噪声）虽更符合生物学，但基线太弱：
        #    λ 已达 0.94 接管，ar_high ≈ |target|² 说明高层输出趋零。
        #    实测 vis 映射 top1=0.193 / 余弦 0.395
        #         babbling top1=0.123 / 余弦 0.227
        #
        # 2. **normalize_to 是一个 trade-off，不是越调越好**：
        #      =0.0 → 余弦 0.471（学到语义类别）但 top1 仅 0.137
        #      =1.1 → top1 0.190 但余弦 0.327
        #    成因：同类词（mama/baba）的 lang 向量余弦本就达 0.67，
        #    系统能学到**语义类别**，却分不清**同类个体**。
        #    余弦高不等于 top1 高，评价时须明确测的是哪个。
        #
        # 3. 当前结构生长不足：神经元恒为 1、髓鞘恒为 3（多 seed 一致）。
        #    这是性能的主要瓶颈，优先于任何调参。
        # --------------------------------------------------------------
        # W 的目标范数（Oja 式归一化）。0=关闭。
        # 取值依据：归一化后输入范数≈2.6，lang 段目标范数≈2.86，
        # 故 |W|≈2.86/2.6≈1.1 时输出幅度与目标匹配。
        self.autoreg_w_norm = 1.1
        # 上一步的归一化输入（delta rule 需要它，见 step_awake）
        self._prev_norm_input: Optional[object] = None

        # --- 影子模式 + 驱动满足度 ---
        #
        # ⚠ 这两者解决的是同一个实测陷阱：λ 由**自回归预测误差**裁定时，
        # 高层把世界搞成一片死寂，观测变全零，恒零预测误差极低 →
        # progress 上升 → λ 升高 → 更彻底接管。
        # 实测（baby_loop.py）：满足率 反射 0.941 → λ 自适应 0.003。
        #
        # 修正：裁定权从高层脑（预测误差）交还**爬虫脑**（满足度）。
        #
        #   shadow_mode=True  → 反射全权，高层只观察
        #                       （以 ε 概率探索性接管，累积证据）
        #   competence 显著>0 → 退出影子模式，λ 由 competence 裁定
        self.shadow_mode: bool = True
        self.drive_sat = DriveSatisfaction(
            eps=getattr(self, "explore_eps", 0.1),
            momentum=0.95,
            min_samples=getattr(self, "min_trials", 20),
        )
        self._last_controller: str = "reflex"

        # --- 三因子学习：张力消解（第三因子）调制突触可塑性 ---
        #
        # ⚠ 缺失这一环，高层**永远学不会控制**（实测）。
        #
        # 现有学习通路只有自回归残差（预测下一帧）。它能学到序列模式，
        # 但学不到"说什么能消解张力" —— 因为张力消解从未进入权重更新。
        #
        # 生物学：三因子学习规则（three-factor learning rule）
        #   因子1 突触前活动 × 因子2 突触后活动 = Hebbian 项（eligibility）
        #   因子3 神经调质（多巴胺 = 奖励预测误差 RPE）决定**是否巩固**
        #
        # 前两因子只说明"这两者一起活跃过"，第三因子才说明
        # "这次的共激活**值不值得被记住**"。
        self._relief_baseline: float = 0.0
        self._pending_relief: Optional[float] = None
        self.relief_gain: float = 1.0        # 第三因子对学习的调制强度
        self.relief_momentum: float = 0.9
        # 接管/收回的滞回阈值（避免 competence 在 0 附近抖动导致频繁切换）
        self.takeover_threshold: float = 0.05
        # 裁定权归爬虫脑：higher_brain 读的是 system 持有的满足度追踪器
        self.higher_brain.drive_sat = self.drive_sat
        # 上一步输入（供冲突观察用，避免被当前帧覆盖）
        self._prev_input_for_observe: Optional[object] = None
        # 首帧状态：正常时序下 _last_high_output 在 step_awake 末尾赋值、
        # _prev_output_for_relief 在下一帧的学习块之前就位，但只要外部
        # 在首个 step_awake 之前调用过 report_drive_satisfaction，
        # _learn_from_relief 就会因属性不存在而 AttributeError。
        self._last_high_output: Optional[object] = None
        self._prev_output_for_relief: Optional[object] = None
        # 本步最终输出的来源标签集合（来自信号 metadata 的 "source"），
        # 供 report_drive_satisfaction 记入按来源分账的咨询台账。
        self._last_output_sources: frozenset = frozenset()

        # 表征冲突观察所针对的**输出通道**。
        #
        # ⚠ 必须为 None 以外的具体通道，否则冲突检测失效（已取证）：
        # target 是**完整下一帧**（所有通道拼接），而高层通常只负责
        # 其中一部分。拿整帧比较时，高层无关通道（如被反射 echo 的
        # vis/aud）的相似度会**稀释**真正需要区分的输出差异。
        # 实测：整帧比较时 out_sim 恒 > 0.3 → 冲突计数恒 0 → 从不分裂。
        #
        # 指定为高层实际负责输出的通道名（按 port_layout 切片）。
        self.observe_output_channel: Optional[str] = None
        # 世界显示信道：环境"当前帧"里携带待预测内容的信道（如 vis）。
        # 通路级预测结算用它读出世界揭示的真实 token——信用来自环境
        # 即时揭示，不是标注。
        self.world_display_channel: Optional[str] = None

        # --- 癫痫刹车（防发散：发育障碍与癫痫是生物学前车之鉴）---
        #
        # 发育期有多重正反馈：髓鞘 gain↑ → 传输↑ → 共发射↑ → 髓鞘化↑；
        # attention 调制、三因子 lr 调制同向放大。生物的对应灾难是
        # 同步放电失控（癫痫）。刹车机制：监控输出能量 EMA，单步能量
        # 突破基线 ratio 倍 → 整体 damp（抑制性制动），并计数以便诊断。
        self.seizure_guard: bool = True
        self.seizure_ratio: float = 6.0
        self.seizure_damp: float = 0.5
        self._energy_ema: Optional[float] = None
        self.seizure_events: int = 0
        self.use_scale = use_scale
        self.higher_brain.sheath_registry.decay_rate = decay_rate

        # 演化参数（变异）
        self.drift_rate = drift_rate
        self.drift_interval = drift_interval
        self._rng = random.Random(0)

        # 结构更新的时间尺度分离（协同 D）
        self.wiring_block = wiring_block
        # 承载力：代谢预算（协同 B）
        self.higher_brain.sheath_registry.target_influx = target_influx
        self.higher_brain.sheath_registry.protection_cost = protection_cost
        # 信息饱和度触发分裂（协同 C）
        self.saturation_split = saturation_split
        # T3: e-prop 资格迹（髓鞘链深度信用）——RPE 按 elig 分账
        self.eprop_enabled = bool(eprop)
        # 承载力预算随自回归误差伸缩（协同 E）
        self.adaptive_budget = adaptive_budget
        self.higher_brain.sheath_registry.adaptive_budget = adaptive_budget
        self._last_maintenance: dict = {"probed": 0, "pruned": 0}

        # 紧急度 ∈ [0,1]：由外部按情境设定，决定反射 vs 高层的接管比例
        self.urgency: float = 0.0

        # 发育状态
        self.phase = "embryonic"  # embryonic | exploratory | mature
        self.mode = SystemMode.AWAKE
        self.step_count = 0

        # 觉醒轨迹日志（供睡眠态重放）
        # 每条记录：4 元组 (input_signals, reflex_outputs, final_outputs, match_result)
        # reflex_outputs 与 final_outputs 的差异 = 反事实素材
        self._trajectory_log: list[tuple[dict, dict, dict, dict]] = []

        # 做梦学习参数
        self.sleep_beta = 0.1              # 稳态缩放强度
        self.sleep_preplay_consistency = 0.5  # 前向预演一致性阈值

    def add_route(self, route: Route) -> None:
        """添加路由（兼容 RouteExecutor 接口）"""
        self.routes.append(route)
        self.executor.add_route(route)

    def _read_input_ports(self) -> dict[str, Signal]:
        """读取所有入端口，用 port name 作为键（无 name 时用 port_id）"""
        input_signals: dict[str, Signal] = {}
        # 反向映射：port_id → name
        id_to_name = {pid: name for name, pid in self.ports._names.items()}
        for port_id, port in self.ports._ports.items():
            if not isinstance(port, InputPort):
                continue
            try:
                sig = port.get_signal()
            except Exception:
                continue
            if sig is not None:
                key = id_to_name.get(port_id, port_id)
                input_signals[key] = sig
        return input_signals

    def _write_output_ports(self, outputs: dict[str, Signal]) -> None:
        """写入所有出端口"""
        for port in self.ports.get_output_ports():
            for key, signal in outputs.items():
                if signal.mime_type in port.get_accepted_types():
                    try:
                        port.put_signal(signal)
                    except (TypeError, RuntimeError):
                        pass

    def _combine_signals(self, input_signals: dict[str, Signal]) -> Signal:
        """合并多信道信号为全局张量（按 port_layout 偏移拼接）"""
        global_vec = torch.zeros(self.global_dim)
        for ch_name, sig in input_signals.items():
            if ch_name in self.port_layout:
                offset, size = self.port_layout[ch_name]
                data = sig.data.flatten()[:size]
                global_vec[offset:offset + data.numel()] = data
        return Signal(
            data=global_vec,
            mime_type="multi/channel",
            metadata={"step": self.step_count},
        )

    def _learn_from_autoreg(self, residual, input_vec, lr=None) -> int:
        """用自回归残差更新神经元权重（delta rule，无反向传播）

        这是**环境即时反馈**落到权重上的唯一通道。没有它，系统只能靠
        髓鞘 gain 做强度调制，学不到输入到输出的映射 —— 实测表现为
        输出与目标的余弦为负。

        只更新**存活且已展开**的神经元。学习率由 ``autoreg_lr`` 控制。

        Args:
            residual: 全局空间中的预测误差（目标 − 输出）
            input_vec: 产生该输出时的输入（全局空间，已归一化）

        Returns:
            被更新的神经元数。
        """
        if residual is None or input_vec is None:
            return 0
        updated = 0
        for neuron in self.higher_brain.ecosystem.neurons.values():
            if neuron is None or not getattr(neuron, "alive", False):
                continue
            if getattr(neuron, "W", None) is None:
                continue
            if not getattr(neuron, "unfolded", None):
                continue
            try:
                d = neuron.apply_prediction_error(
                    residual, input_vec,
                    lr=(self.autoreg_lr if lr is None else lr),
                    normalize_to=self.autoreg_w_norm)
            except Exception:
                continue
            if d:
                updated += 1
        return updated

    def _reinforce_path_via_sheaths(
        self, path: list[int], amount: float = 0.02
    ) -> int:
        """沿路径增厚髓鞘（用 sheath_registry.thicken_sheath）

        对路径中每对相邻神经元，找到连接它们的髓鞘并增厚。
        返回增厚的髓鞘数。
        """
        sheath_reg = self.higher_brain.sheath_registry
        count = 0
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for key in list(sheath_reg._sheaths.keys()):
                if key[0] == src and key[2] == dst:
                    if sheath_reg.thicken_sheath(*key, amount=amount):
                        count += 1
        return count

    def _decay_path_via_sheaths(
        self, path: list[int], amount: float = 0.02
    ) -> int:
        """沿路径衰减髓鞘（用 sheath_registry.decay_sheath）

        对路径中每对相邻神经元，找到连接它们的髓鞘并衰减。
        返回衰减的髓鞘数。
        """
        sheath_reg = self.higher_brain.sheath_registry
        count = 0
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for key in list(sheath_reg._sheaths.keys()):
                if key[0] == src and key[2] == dst:
                    if sheath_reg.decay_sheath(*key, amount=amount):
                        count += 1
        return count

    def _slice_output_channel(self, target):
        """按 :attr:`observe_output_channel` 从完整帧中切出高层负责的部分

        未配置时返回原向量（等价于旧行为，但那已被证明会稀释冲突信号）。
        """
        ch = getattr(self, "observe_output_channel", None)
        if not ch or not getattr(self, "port_layout", None):
            return target
        if ch not in self.port_layout:
            return target
        off, size = self.port_layout[ch]
        try:
            return target[off:off + size]
        except Exception:
            return target

    def _channel_argmax(self, frame, channel: str) -> Optional[int]:
        """从（归一化）帧的某信道段读出 argmax token；段为空返回 None"""
        if channel is None or channel not in self.port_layout:
            return None
        off, size = self.port_layout[channel]
        try:
            seg = frame[off:off + size]
            mx = seg.max()
            mx = float(mx.item()) if hasattr(mx, "item") else float(mx)
            if mx <= 1e-6:
                return None
            am = seg.argmax()
            return int(am.item()) if hasattr(am, "item") else int(am)
        except Exception:
            return None

    def report_drive_satisfaction(self, satisfaction: float) -> None:
        # fmt: off
        """**身体/世界**回报内在驱动的满足度 —— 裁定 λ 的反馈通路

        ⚠ 这是**爬虫脑的内在信号回流**，不是感知通道。

        调用时机：世界的响应产生之后（即"刚才那个行动有效吗"已知时）。
        由外部（身体/环境适配层）调用，系统本身不产生这个量。
        需求是身体设定的，满足与否只有身体知道。

        Args:
            satisfaction: 内在驱动被满足的程度 ∈ [0, 1]
        """
        # fmt: on
        if satisfaction is None:
            return
        self._pending_relief = float(satisfaction)
        self.drive_sat.observe(float(satisfaction), self._last_controller)
        # 咨询台账：按**输出来源**（器官级 provenance）分账记录满足度。
        # 与 reflex/higher 二分臂正交——同一来源可能出现在两臂里
        # （如反射工具的输出既可能在影子期也可能在接管期出现）。
        # 融合框架下 LLM 教师器官的输出也走这里：咨询频率随髓鞘化
        # 下降（"咨询费只付一次"）就是从这份台账读出来的。
        for _tag in self._last_output_sources:
            self.drive_sat.observe_source(_tag, float(satisfaction))
        # 影子模式的**进出都是可逆的**
        #
        # ⚠ 初版只写了"退出"（competence>0 → shadow=False），
        # 一旦退出就**永不返回**。实测灾难：早期样本噪声使 competence
        # 短暂为正 → 永久接管 → 2000 步中 1805 步由高层控制，
        # 而高层成功率仅 7.5%（反射 26.2%）。
        #
        # 控制权必须**可收回**：高层退步时退回影子模式，
        # 等它重新证明自己。这正是"接管可逆"的含义。
        if self.drive_sat.has_sufficient_evidence:
            comp = self.drive_sat.competence
            if self.shadow_mode and comp > self.takeover_threshold:
                self.shadow_mode = False
            elif (not self.shadow_mode
                  and comp < -self.takeover_threshold):
                self.shadow_mode = True      # 收回控制权

    def _learn_from_relief(self, rpe, input_vec, output_vec, lr=None) -> int:
        """**第三因子**驱动的学习：ΔW = η · RPE · eligibility

        这是三因子学习规则的核心。与自回归学习的区别：

        ============  ==========================  ========================
                      自回归学习                    三因子学习
        ============  ==========================  ========================
        信号来源       预测误差（目标 − 输出）        张力消解的预测误差 RPE
        学到什么       世界的序列规律                **什么行动有效**
        能否学控制     不能（实测）                   能
        ============  ==========================  ========================

        **为什么必须有这条通路**（实测教训）：

        lang 通道 = 系统自己上一帧的发声 → 自回归残差趋近于零 →
        高层"完美预测"却什么也没学到。competence 恒为 −0.985。

        只有 RPE 才能回答"刚才那个行动好不好"，而这才是控制要学的。

        Args:
            rpe: 奖励预测误差（实际消解 − 基线预期）
            input_vec: 产生该输出时的输入
            output_vec: 该输出本身（eligibility 的突触后因子）
            lr: 学习率

        Returns:
            被更新的神经元数。
        """
        if rpe is None or input_vec is None or output_vec is None:
            return 0
        if abs(rpe) < 1e-9:
            return 0
        lr = self.autoreg_lr if lr is None else lr
        updated = 0
        for neuron in self.higher_brain.ecosystem.neurons.values():
            if neuron is None or not getattr(neuron, "alive", False):
                continue
            if getattr(neuron, "W", None) is None:
                continue
            if not getattr(neuron, "unfolded", None):
                continue
            try:
                neuron.apply_three_factor(
                    rpe, input_vec, output_vec, lr=lr
                )
                updated += 1
            except Exception:
                continue
        return updated

    def step_awake(
        self,
        external_inputs: Optional[dict[str, Signal]] = None,
    ) -> dict[str, Any]:
        """觉醒态单步运行（事件驱动主循环）

        流程：
        1. 读取入端口（或用 external_inputs）
        2. 高层脑 process() 介入输入信号化（始终活跃）
        3. 反射弧 execute() 产生原始输出
        4. 高层脑 intervene() 连续 λ 残差修正（失败时反射弧保底）
        5. 事件驱动：dispatch_signal 分发组合信号 → resolve_triggers → 沿触发路径 thicken_sheath
        6. 写出端口
        7. 记录 4 元组到 _trajectory_log（反事实素材）
        8. engine.accumulate_novelty 累积新奇性
        9. engine.maybe_differentiate 可能分化

        返回 {outputs, match_result, lambda, differentiated, ...}
        """
        self.step_count += 1
        self.mode = SystemMode.AWAKE
        # T2: 推进分发器逻辑时钟（跨步事件队列的时间基准）
        self.higher_brain.dispatcher.tick()

        # 1. 读取入端口（或用 external_inputs）
        #    注意：原始信号保持物理单位，归一化只在进入发育网络时进行
        input_signals = (
            external_inputs if external_inputs is not None
            else self._read_input_ports()
        )
        if not input_signals:
            return {
                "outputs": {},
                "match_result": {},
                "lambda": 0.0,
                "differentiated": -1,
            }

        # 胚胎期：向输入注入小噪声（探索性驱动）
        if self.phase == "embryonic":
            noisy_signals = {}
            for key, sig in input_signals.items():
                noisy_data = sig.data + torch.randn_like(sig.data) * 0.01
                noisy_signals[key] = Signal(
                    data=noisy_data,
                    mime_type=sig.mime_type,
                    metadata={**sig.metadata, "noisy": True},
                )
            input_signals = noisy_signals

        # 1.5 自回归结算：用**当前**输入裁定**上一步**的预测
        #
        # 时序说明：自回归目标 x_{t+1} 在 step t 尚不存在，必须延迟一步结算。
        # 因此在 step t 开始时，用刚到达的 input_t 去结算 step t−1 的输出。
        # 这是**环境即时反馈**的落点，也是唯一能裁定"反射本身是否正确"
        # 的机制——纯蒸馏最多追平反射，只有自回归能超越基线。
        combined_input = self._combine_signals(input_signals)
        # 超模态统一编码：归一化后再送入发育网络与自回归结算
        self.normalizer.update(combined_input.data)
        # ⚠ 捕获 settle 的返回值：它带**自回归残差**（下一帧真实 − 上一步
        # 输出），是唯一能让系统学到"输入→输出映射"的训练信号。
        # 原实现丢弃了这个返回值。
        _ar_fb = self.higher_brain.scheduler.settle(
            self.normalizer.normalize(combined_input.data)
        )
        # --- 通路级预测结算：世界已揭示当前帧 → 上一帧备案的各通路
        #     预测逐个对答案。信用只来自环境即时揭示（无标注、无反传），
        #     "被世界证实的通路才有发言权"。
        if getattr(self, "world_display_channel", None):
            _actual = self._channel_argmax(
                self.normalizer.normalize(combined_input.data),
                self.world_display_channel,
            )
            if _actual is not None:
                self.higher_brain.settle_source_predictions(_actual)
        # delta rule：用自回归残差更新参与输出的神经元权重（无反向传播）
        #
        # ⚠ 时序：残差是"上一步输出 vs 这一步真实"，而产生那个输出的
        # 输入是**上一步**的输入，不是当前步。用当前步输入做外积是错的
        # （学的是 residual(t) ⊗ input(t+1)，两者无因果关系）。
        # 修复：缓存上一步的归一化输入专门用于权重更新。
        if getattr(_ar_fb, "settled", False) and self._prev_norm_input is not None:
            # 学习率随步数衰减：lr = lr0 / (1 + step/tau)
            #
            # **为什么必须衰减**（实测，2000 步长跑）：
            # W 收敛后，残差主要是任务**固有不可预测**的部分（本任务
            # 一阶 Oracle 仅 30.4%）。持续用这个残差更新等于拟合噪声，
            # W 的方向被逐步带偏 —— 实测余弦在 750~1000 步达峰 0.76，
            # 之后回落到 0.37，而 W 范数反而涨到上限。
            # 固定 lr 无法区分"还有东西可学"和"已经在拟合噪声"，
            # 衰减让后期更新步长趋零，把已达成的方向锁住。
            lr_t = self.autoreg_lr / (1.0 + self.step_count / max(1.0, self.autoreg_lr_tau))

            # 三因子：用**张力消解的预测误差**调制本次学习
            #
            # RPE = 实际消解 − 基线预期。优于预期则巩固本次的
            # 输入→输出映射，劣于预期则削弱。这让"有效的行动"
            # 被选择性地记住 —— 纯粹的预测误差做不到这件事。
            if self._pending_relief is not None:
                rpe = self._pending_relief - self._relief_baseline
                self._relief_baseline = (
                    self.relief_momentum * self._relief_baseline
                    + (1.0 - self.relief_momentum) * self._pending_relief
                )
                self._pending_relief = None
                # T3: e-prop——RPE 到来，按资格迹给髓鞘分账（深度信用）
                if self.eprop_enabled:
                    self.higher_brain.sheath_registry.apply_eprop(rpe)
                # ① 调制自回归学习的强度
                lr_t = lr_t * (1.0 + self.relief_gain * rpe)
                # ② **RPE 直接驱动学习**（真正的第三因子）
                #
                # 只调制 lr 是不够的：本任务中自回归残差趋近于零
                # （lang 通道 = 自己上一帧的发声，完美可预测），
                # 调制一个接近零的残差什么也学不到。
                #
                # 第三因子必须**独立**驱动更新：
                #   ΔW = η · RPE · eligibility
                #   eligibility = 输入 ⊗ 输出（刚才的共激活，Hebbian 项）
                #
                # 前两因子说明"这两者一起活跃过"，
                # 第三因子说明"这次的共激活**值不值得被记住**"。
                self._learn_from_relief(
                    rpe, self._prev_input_for_observe,
                    self._prev_output_for_relief,
                    lr=self.autoreg_lr * self.relief_gain,
                )
            self._learn_from_autoreg(_ar_fb.residual, self._prev_norm_input, lr=lr_t)
        # ⚠ 必须在覆盖前保存**上一步**的输入。
        #
        # 自回归的正确配对是 (input_{t-1}, target_t)：
        # 上一步的输入 → 当前帧（作为那一步预测的真实结果）。
        #
        # 但此处覆盖后，``_prev_norm_input`` 立刻变成**当前帧**。
        # 下游若再拿它当"上下文"，就会配成 (input_t, target_t)
        # —— **恒等映射**。实测后果：
        #   observe_context 的 out_sim 恒为 1.000 → 冲突计数恒 0
        #   → 分化从不触发 → 神经元锁死在 1 个。
        # 这个 bug 让整套结构生长机制静默失效了数轮。
        self._prev_input_for_observe = self._prev_norm_input
        self._prev_output_for_relief = getattr(self, "_last_high_output", None)
        self._prev_norm_input = self.normalizer.normalize(combined_input.data)

        # --- 以下是原有逻辑（保留）---

        # 1.8 维度展开：保证生态层面覆盖本步出现的信道
        #
        # **原实现中 unfold 从未在主循环被调用**，新出现的信道（新传感器/
        # 新模态）不会展开维度，系统对它完全是瞎的。
        # 只保证**生态层面**的覆盖，不消除**个体层面**的差异——否则
        # find_coverage 的覆盖度恒为 1.0，新奇性恒为 0，分化永不触发。
        active_channels_now = set(input_signals.keys())
        newly_unfolded = self.higher_brain.ecosystem.ensure_channel_coverage(
            active_channels_now
        )

        # 2. 高层脑始终介入输入信号化
        match_result = self.higher_brain.process(input_signals)
        # 注入紧急度：决定 λ 门控中反射 vs 高层的接管比例。
        # 紧急时等不到下一帧环境信号（无法做自回归裁定），反射直接接管。
        match_result["urgency"] = self.urgency

        # --- 影子模式：决定本步谁真正控制 ---
        #
        # 影子期反射全权，高层只观察（学习照常，但不影响输出）。
        # 以 ε 概率让高层**真正接管**做探索性试验 —— 否则永远
        # 无从知道高层的实际能力，影子期无法结束（这是交替试验）。
        # 每步把 explore_eps 同步到追踪器 —— 否则运行期修改
        # ``brain.explore_eps`` 完全无效（初值在 __init__ 时已被读走）。
        # 实测后果：设了 0.02 却按 0.1 探索，且随后触发了下面的
        # 不可逆接管，导致 2000 步里 1805 步由高层控制。
        if hasattr(self, "drive_sat") and self.drive_sat is not None:
            if hasattr(self, "explore_eps"):
                self.drive_sat.eps = float(self.explore_eps)

        self._exploring = False
        if self.shadow_mode:
            self._exploring = self.drive_sat.should_explore(
                rng=getattr(self, "_rng", None))
            self._last_controller = "higher" if self._exploring else "reflex"
        else:
            self._last_controller = "higher"
        # ⚠ 探索步必须关掉影子标志，否则高层依然无法真正接管
        match_result["shadow_mode"] = bool(self.shadow_mode and not self._exploring)
        match_result["exploring"] = bool(self._exploring)

        # 3. 反射弧产生原始输出
        context = {
            "cortex_confidence": self.higher_brain.get_confidence(),
            "phase": self.phase,
        }
        raw_outputs = self.reflex.execute(input_signals, context)

        # 4. 高层脑介入输出处理（失败时反射弧保底）
        try:
            # ⚠ 必须传全局组合信号。intervene 内部要用它 dispatch，
            # 而神经元展开的是全局维度 —— 传单通道信号会索引越界，
            # 被下面 except 静默吞掉 → 高层脑从不介入（已取证）。
            final_outputs = self.higher_brain.intervene(
                raw_outputs, match_result,
                global_signal=self.normalizer.normalize(combined_input.data),
            )
            if not final_outputs and raw_outputs:
                final_outputs = raw_outputs
            self._last_high_output = self._combine_signals(final_outputs)
        except Exception as _e:
            # 不再静默吞异常：第一次发生时打印，便于定位。
            if getattr(self, "_intervene_warned", False) is False:
                import logging
                logging.getLogger(__name__).warning(
                    "higher_brain.intervene 失败，已回退纯反射: "
                    "%s: %s", type(_e).__name__, _e)
                self._intervene_warned = True
            final_outputs = raw_outputs

        # 输出来源标签：记录本步输出携带的 provenance（如反射工具的
        # metadata["source"]）。供咨询台账按来源分账——"哪个器官的行为
        # 满足了驱动"，而不是只按 reflex/higher 二分。
        _srcs = set()
        for _sig in (final_outputs or {}).values():
            _tag = (getattr(_sig, "metadata", None) or {}).get("source")
            if _tag:
                _srcs.add(str(_tag))
        self._last_output_sources = frozenset(_srcs)

        # --- 癫痫刹车：单步输出能量突破基线 ratio 倍 → 整体制动 ---
        # （发育期正反馈失控的生物学对应是癫痫；刹车即抑制性制动）
        if self.seizure_guard and final_outputs:
            _energy = 0.0
            for _s in final_outputs.values():
                _d = _s.data
                _energy += float((_d * _d).sum().item())
            self._energy_ema = (
                _energy if self._energy_ema is None
                else 0.99 * self._energy_ema + 0.01 * _energy
            )
            if (self._energy_ema > 1e-9
                    and _energy > self.seizure_ratio * self._energy_ema):
                final_outputs = {
                    _k: Signal(data=_s.data * self.seizure_damp,
                               mime_type=_s.mime_type,
                               metadata=_s.metadata)
                    for _k, _s in final_outputs.items()
                }
                self.seizure_events += 1

        # 4.5 双目标调度：给出本步应逼近的目标与残差
        #
        # 目标是两条时间尺度的混合：
        #   紧急（urgency→1）→ target = 反射（等不到下一帧）
        #   从容（urgency→0）→ target 按毕业度掺入高层自身输出
        # 残差用于计算各髓鞘的贡献，是"用进废退"的判据来源。
        feedback = None
        if raw_outputs and final_outputs:
            y_reflex = self._combine_signals(raw_outputs)
            y_high = self._combine_signals(final_outputs)
            # 归一化到统一编码空间后再比较（反射增益是原始量纲下的，
            # 此处仅用于计算残差/贡献，不改变实际输出）
            feedback = self.higher_brain.scheduler.begin_step(
                self.normalizer.normalize(y_reflex.data),
                self.normalizer.normalize(y_high.data),
                urgency=self.urgency,
            )

        # 5. 事件驱动主循环：用 dispatcher 分发组合信号，强化激活路径
        combined_signal = self._combine_signals(input_signals)
        events = self.higher_brain.dispatch_signal(
            self.normalizer.normalize(combined_signal.data),
            source_tag=f"awake_{self.step_count}",
        )
        # --- 通路级预测备案：本步指向输出信道的跨模态事件 payload ---
        # （本步输入 = token_t，payload = 对 token_{t+1} 的预测；
        #   下一帧结算时与世界揭示的 token_{t+1} 对答案）
        if getattr(self, "observe_output_channel", None):
            _preds = []
            for _e in events:
                _sk = _e.sheath_key
                if (_e.channel == self.observe_output_channel
                        and _sk is not None
                        and _sk[1] != self.observe_output_channel):
                    _p = self._channel_argmax(_e.data,
                                              self.observe_output_channel)
                    if _p is not None:
                        _preds.append(((_sk[0], _sk[1]), _p))
            self.higher_brain.note_source_predictions(_preds)
        # 重合窗口由 delay 分布导出，而非硬编码常量——
        # 否则"同时发射"的判据与传导延迟脱节，delay 就失去了对成边的影响力。
        window = self.higher_brain.sheath_registry.suggest_window()
        triggered = self.higher_brain.resolve_triggers(
            events, threshold=0.01, coincidence_window=window
        )
        if len(triggered) >= 2:
            path = [e.target_neuron for e in triggered]
            self._reinforce_path_via_sheaths(path, amount=0.02)

        # 5.5 选择动力学（用进废退）+ 时间轴反馈环
        #
        # (a) 按贡献结算全局衰减——"物竞天择，谁用的最多谁就留存"的执行点。
        #     贡献基于**残差**而非目标，因此系统自限：输出过头时同一通路的
        #     贡献自动转负，无需硬约束即可防止无界增长。
        # (b) delay 向成功绑定的群体到达时间同步，闭合
        #     "髓鞘调 delay → 改变到达时序 → 改变成边"的反馈环。
        died = 0
        if feedback is not None:
            contributions = pathway_contributions(events, feedback.residual)
            # T3: 资格迹推进——瞬时贡献沉淀为持久 elig（延迟 RPE 可分账）
            if self.eprop_enabled and contributions:
                self.higher_brain.sheath_registry.update_eligibility(
                    contributions)
            # 注意力键的局部学习：c_j · q（query 由 attention.modulate 缓存）。
            # 解耦：只动 attention 自己的键，W / 髓鞘 / 衰减一概不碰。
            if (self.higher_brain.attention is not None
                    and contributions):
                self.higher_brain.attention.learn_from_contributions(
                    contributions)
            died = self.higher_brain.sheath_registry.apply_global_decay(
                contributions,
                use_scale=self.use_scale,
                decay_rate=self.decay_rate,
            )
        self.higher_brain.dispatcher.adapt_delays()

        # 5.55 三个此前定义了但**从未被调用**的机制，在此接线
        #
        # (a) age 推进：每条髓鞘记录"多久没成功共发射"。EMA 是衰减的记忆，
        #     渐近趋零但永不到零；离散 age 是硬计数，用于杀死噪声通路。
        self.higher_brain.sheath_registry.tick_ages()

        # (b) 代谢账本：被路由选中的神经元获得供给，长期不用的衰减。
        #     原实现中 tick_metabolism 从未被调用 → metabolism 恒为 0.5
        #     → 饥饿自噬的判据 (metabolism < 0.01) 永不成立 → 该分支是死代码。
        #     接上后，神经元死亡会**涌现**（所有维度都被剪、所有边都断掉的
        #     神经元自然耗尽代谢），不需要额外的特设规则。
        winner_id = match_result.get("winner_id", -1)
        self.higher_brain.ecosystem.tick_metabolism(
            [winner_id] if winner_id >= 0 else []
        )

        # (c) 维度级用进废退：用残差裁定各维度"用了有没有用"。
        #     这是维度展开 / 退化 / 自噬三者共同的状态更新点。
        if feedback is not None:
            self.higher_brain.ecosystem.record_dim_feedback(
                feedback.residual,
            )

        # 5.6 自演化漂移（变异算子）—— 按 age 分配探索预算（协同 F）
        #
        # 漂移必须远小于选择强度，且受 protection 抑制。单独存在时漂移纯粹
        # 是破坏——它必须与用进废退配对才有探索价值。
        #
        # 协同 F：漂移不再是全局均匀的。长期未被共发射的通路（age 大）
        # 获得**更大**漂移幅度去探索新的绑定；刚被验证的通路（age 小）
        # 受保护少漂移。于是同样的漂移预算从"均匀噪声"变成
        # "定向探索"——对无用者是机会，对有用者是保护。
        if self.drift_interval > 0 and self.step_count % self.drift_interval == 0:
            for neuron in self.higher_brain.ecosystem.neurons.values():
                neuron.drift(rate=self.drift_rate, rng=self._rng)

        # 5.7 结构更新的时间尺度分离（协同 D）
        #
        # 成边/脱落必须比权重修饰慢一个量级，否则成边永远赢过选择，
        # 结构无法稳定（实测周转率 11.5，上万次"新建—淘汰"空转）。
        if self.higher_brain.sheath_registry.should_run_wiring(
            self.wiring_block
        ):
            # ⚠ 必须传 align_fn，否则 probe_delay 退化为随机扰动，
            # 而随机扰动已被实验证明与不探索无差异
            # （verify_control.py：随机 vs 冻结 t=-1.24 / -1.69，不显著）。
            # 对齐度来自 dispatcher 的源信号历史缓冲。
            registry = self.higher_brain.sheath_registry
            dispatcher = self.higher_brain.dispatcher
            window = registry.suggest_window()
            self._last_maintenance = registry.maintain_ages(
                align_fn=dispatcher.make_align_fn(window),
                rng=self._rng,
            )

        # 协同 E：承载力预算随自回归误差伸缩
        # 预测不准 → 需要更多容量去解释；预测准 → 容量需求下降。
        if self.adaptive_budget and feedback is not None:
            self.higher_brain.sheath_registry.update_budget_signal(
                float(feedback.residual.norm().item())
            )

        # 6. 写出端口
        self._write_output_ports(final_outputs)

        # 7. 记录 4 元组到 _trajectory_log（反事实素材）
        # 同时调用 higher_brain.record_for_dream 供高层脑自身做梦
        self.higher_brain.record_for_dream(
            input_signals, raw_outputs, final_outputs, match_result
        )
        self._trajectory_log.append(
            (dict(input_signals), dict(raw_outputs),
             dict(final_outputs), dict(match_result))
        )
        if len(self._trajectory_log) > 1000:
            self._trajectory_log.pop(0)

        # 8. 累积新奇性：覆盖度不足 **或** 预测不了（惊奇）
        #
        # ⚠ 必须传入自回归误差。只传覆盖度时，在"一个神经元处理所有
        # 信号"的设计下 coverage 恒为 1.0 → 新奇性恒 0 → 分化永不触发。
        active_channels = set(input_signals.keys())
        _ar_err = None
        _target_energy = None
        if getattr(_ar_fb, "settled", False):
            _ar_err = getattr(_ar_fb, "ar_error_high", None)
            if _ar_err is not None:
                if _ar_err != _ar_err:                      # NaN
                    _ar_err = None
                else:
                    # 目标能量 |target|²：NMSE 的分母
                    try:
                        _tn = _ar_fb.target
                        _target_energy = float((_tn * _tn).sum().item())
                    except Exception:
                        _target_energy = None
        self.higher_brain.engine.accumulate_novelty(
            combined_signal.data, active_channels, ar_error=_ar_err,
            target_energy=_target_energy,
        )

        # 表征冲突观察：记录「上下文 → 应有的输出」
        # 这是**需求驱动**的分化判据（替代恒为 1 的 NMSE 惊奇）。
        if getattr(_ar_fb, "settled", False) and _ar_fb.target is not None:
            _ot = self._slice_output_channel(_ar_fb.target)
            self._last_ar_target = _ot      # 诊断
            self.higher_brain.engine.observe_context(
                self._prev_input_for_observe, _ot,
            )

        # 9. 可能分化（threshold 内部门控）
        diff_idx = self.higher_brain.engine.maybe_differentiate(
            combined_signal.data, active_channels
        )
        if self.step_count % 1000 == 0:
            self.higher_brain.engine.autophagy()

        # 10. 阶段切换
        if self.step_count == 10000:
            self.phase = "exploratory"
        if self.step_count == 100000:
            self.phase = "mature"
            self.reflex.disable_for_mime("text/*")

        return {
            "outputs": final_outputs,
            "match_result": match_result,
            "lambda": self.higher_brain._last_lambda,
            "differentiated": diff_idx,
            "step": self.step_count,
            # --- 选择动力学诊断 ---
            "sheaths_died": died,                       # 本步被淘汰的髓鞘数
            "progress": self.higher_brain.scheduler.progress,  # 毕业进度（>0 已超越反射）
            "distill_error": self.higher_brain.scheduler.last_distill_error,
            "capacity": self.higher_brain.sheath_registry.capacity_report(),
            "attention": (self.higher_brain.attention.report()
                          if self.higher_brain.attention is not None else None),
            "spikes": (self.higher_brain.spike_coder.report()
                       if self.higher_brain.spike_coder is not None else None),
            "seizure_brakes": self.seizure_events,
            "src_cred": self.higher_brain.src_cred_report(),
            "maintenance": self._last_maintenance,
            "split_trigger": getattr(self.higher_brain.engine,
                                     "_last_trigger", ""),
        }

    def step_sleep(
        self,
        external_inputs: Optional[dict[str, Signal]] = None,
    ) -> dict[str, Any]:
        """睡眠态单步运行：四相位做梦学习（增强学习机制，对外界敏感）

        睡眠态不是"切断外界纯重放"，而是以做梦学习为主、外界输入为辅的
        增强学习模式。外界输入在两个相位被利用：
        - NREM S2 重放：外界输入作为"梦境扰动"叠加到重放信号（创造性融合）
        - REM 前向预演：用**当前外界输入**（而非历史 input）触发反射弧预测

        每个 sleep cycle 依次执行四个相位（全部 sheath_registry 操作，无梯度）：
        1. NREM S3 稳态缩放：sheath_registry.homeostatic_scale(beta)
        2. NREM S2 生成重放：重放轨迹 + 外界输入扰动，dispatch_signal 分发，thicken_sheath 增厚
        3. REM 反事实：对比"高层干预 vs 反射本会输出"，learn_from_intervention_delta
        4. REM 前向预演（Preplay）：当前外界输入触发预测，路径一致性 thicken/decay_sheath

        睡眠态仍产生弱输出：反射弧对外界输入响应 + 高层脑低 λ 轻度介入
        （非完全切断，对应"梦游/梦话"的弱响应）

        返回 {sleep_outputs, phases: {homeostasis, replay, counterfactual, preplay}}
        """
        self.mode = SystemMode.SLEEP

        # 读取外界输入（默认从入端口）
        if external_inputs is None:
            external_inputs = self._read_input_ports()
        has_external = bool(external_inputs)
        has_trajectory = bool(self._trajectory_log)

        # 无历史轨迹且无外界输入：无事可做
        if not has_trajectory and not has_external:
            return {}

        results: dict[str, Any] = {}

        # --- 睡眠归因账本 ---
        #
        # 每个相位前后对全部髓鞘做 (gain, stability) 快照差分，把增量
        # 归因到"相位 × 髓鞘"。没有这份账本，"重放到底巩固了什么"
        # 只能靠假设——对比"跳过某相位的对照 episode"即可归因各相位
        # 的实际巩固贡献。账本本体在 MyelinSheathRegistry（sleep_ledger
        # + 每条髓鞘的 sleep_gain_delta / sleep_phase_gain）。
        sheath_reg = self.higher_brain.sheath_registry

        def _run_phase(name: str, fn, *args, **kwargs):
            before = sheath_reg.snapshot_sheaths()
            out = fn(*args, **kwargs)
            out = dict(out) if out else {}
            out["sleep_attr"] = sheath_reg.attribute_sleep_phase(name, before)
            return out

        # 相位 1：NREM S3 稳态缩放（纯 sheath_registry 操作，与外界无关）
        if has_trajectory:
            results["homeostasis"] = _run_phase(
                "homeostasis", self._sleep_phase_homeostasis)

        # 相位 2：NREM S2 生成重放（外界输入作为梦境扰动注入）
        if has_trajectory:
            results["replay"] = _run_phase(
                "replay", self._sleep_phase_replay, external_inputs)

        # 相位 3：REM 反事实（对比历史干预差异）
        if has_trajectory:
            results["counterfactual"] = _run_phase(
                "counterfactual", self._sleep_phase_counterfactual)

        # 相位 4：REM 前向预演（当前外界输入触发预测，对比历史实际路径）
        if has_trajectory and has_external:
            results["preplay"] = _run_phase(
                "preplay", self._sleep_phase_preplay, external_inputs)

        results["sleep_report"] = sheath_reg.sleep_report()

        # 睡眠态输出：反射弧对外界输入响应 + 高层脑低 λ 轻度介入
        sleep_outputs: dict[str, Signal] = {}
        if has_external:
            context = {
                "cortex_confidence": self.higher_brain.get_confidence(),
                "phase": "sleep",
            }
            reflex_outputs = self.reflex.execute(external_inputs, context)
            if reflex_outputs:
                # 高层脑在睡眠态仍介入，但 λ 整体偏低（运动弛缓 + 弱响应）
                match = self.higher_brain.process(external_inputs)
                match["sleep_mode"] = True  # 触发低 λ 路径
                try:
                    sleep_outputs = self.higher_brain.intervene(
                        reflex_outputs, match
                    )
                    if not sleep_outputs:
                        sleep_outputs = reflex_outputs
                except Exception:
                    sleep_outputs = reflex_outputs
                self._write_output_ports(sleep_outputs)
        results["sleep_outputs"] = sleep_outputs
        return results

    def _sleep_phase_homeostasis(self) -> dict[str, Any]:
        """相位 1：NREM S3 稳态缩放

        突触稳态假说：全局髓鞘按保护系数加权衰减 gain。
        核心层（高保护）几乎不变，活跃层（低保护）优先衰减。
        相对强弱保持，绝对强度下降，清理噪声 + 重置学习容量。
        用 sheath_registry.homeostatic_scale(beta)。
        """
        sheath_reg = self.higher_brain.sheath_registry
        sheaths_before = len(sheath_reg._sheaths)
        sheath_reg.homeostatic_scale(beta=self.sleep_beta)
        sheaths_after = len(sheath_reg._sheaths)
        return {
            "phase": "homeostasis",
            "sheaths_before": sheaths_before,
            "sheaths_after": sheaths_after,
            "scaled": sheaths_before,
        }

    def _sleep_phase_replay(
        self,
        external_inputs: Optional[dict[str, Signal]] = None,
    ) -> dict[str, Any]:
        """相位 2：NREM S2 生成重放（外界输入作为梦境扰动注入）

        从轨迹日志取最近一段，用 higher_brain.dispatch_signal 分发重放信号，
        走通的路径用 sheath_registry.thicken_sheath 增厚（用进废退）。
        若存在外界输入，将其作为"梦境扰动"叠加到重放信号上——对应生物学中
        外界刺激被整合进梦境的现象（如水声→梦中瀑布），让重放不再是机械复刻，
        而是与当前感知的创造性融合。
        """
        input_signals, _, _, _ = self._trajectory_log[-1]

        # 外界输入作为梦境扰动叠加到重放信号
        replay_inputs = dict(input_signals)
        if external_inputs:
            for key, ext_sig in external_inputs.items():
                if key in replay_inputs:
                    orig = replay_inputs[key].data
                    ext = ext_sig.data
                    if orig.shape == ext.shape:
                        blended = orig + ext * 0.1
                        replay_inputs[key] = Signal(
                            data=blended,
                            mime_type=replay_inputs[key].mime_type,
                            metadata={**replay_inputs[key].metadata,
                                      "dream_disturbed": True},
                        )
                else:
                    # 新端口的外界输入直接加入梦境
                    replay_inputs[key] = ext_sig

        # 高层脑处理重放信号（激活路径 + 记录激活）
        replay_match = self.higher_brain.process(replay_inputs)

        # 事件驱动分发重放信号，强化走通的路径
        combined = self._combine_signals(replay_inputs)
        events = self.higher_brain.dispatch_signal(
            combined.data, source_tag="replay"
        )
        triggered = self.higher_brain.resolve_triggers(
            events, threshold=0.01, coincidence_window=0.1
        )
        if len(triggered) >= 2:
            path = [e.target_neuron for e in triggered]
            self._reinforce_path_via_sheaths(path, amount=0.02)

        return {
            "phase": "replay",
            "winner_id": replay_match.get("winner_id", -1),
            "disturbed_by_external": bool(external_inputs),
            "triggered_count": len(triggered),
        }

    def _sleep_phase_counterfactual(self) -> dict[str, Any]:
        """相位 3：REM 反事实

        对比"高层干预 vs 反射本会输出"的差异，调用
        higher_brain.learn_from_intervention_delta 微调权重。
        差异越大，说明高层干预越显著，权重调整幅度越大。
        """
        _, reflex_outputs, final_outputs, _ = self._trajectory_log[-1]
        # 从干预差异学习——微调被干预神经元的权重
        self.higher_brain.learn_from_intervention_delta(
            reflex_outputs, final_outputs
        )
        # 计算差异幅度
        total_delta = 0.0
        count = 0
        for key in final_outputs:
            if key in reflex_outputs:
                diff = (
                    final_outputs[key].data - reflex_outputs[key].data
                ).abs().mean().item()
                total_delta += diff
                count += 1
        avg_delta = total_delta / count if count > 0 else 0.0
        return {
            "phase": "counterfactual",
            "intervention_delta": avg_delta,
        }

    def _sleep_phase_preplay(
        self,
        external_inputs: dict[str, Signal],
    ) -> dict[str, Any]:
        """相位 4：REM 前向预演（Preplay）— 真正的 WAM-TTT 测试时训练

        用**当前外界输入**（而非历史 input_signals）触发反射弧生成预测，
        用 higher_brain.dispatch_signal 分发预测信号，按路径激活一致性
        sheath_registry.thicken_sheath 强化 / sheath_registry.decay_sheath 衰减。

        对应 WAM-TTT 的自监督预测，但用髓鞘语言实现（无梯度）：
        - 对**新输入**做预测（不是重放历史）→ 这才是"测试时训练"
        - 预测路径与历史实际路径一致 → thicken_sheath 强化（模型可迁移）
        - 预测路径与历史实际路径不一致 → decay_sheath 衰减（模型需修正）
        """
        _, _, _, match_result = self._trajectory_log[-1]
        actual_trajectory = match_result.get("trajectory", [])

        # 事件驱动分发当前外界输入，看激活哪条路径（预测路径）
        predicted_match = self.higher_brain.process(external_inputs)
        combined = self._combine_signals(external_inputs)
        events = self.higher_brain.dispatch_signal(
            combined.data, source_tag="preplay"
        )
        triggered = self.higher_brain.resolve_triggers(
            events, threshold=0.01, coincidence_window=0.1
        )
        predicted_path = [e.target_neuron for e in triggered]

        if predicted_path and actual_trajectory:
            pred_set = set(predicted_path)
            actual_set = set(actual_trajectory)
            overlap = (
                len(pred_set & actual_set) / len(actual_set)
                if actual_set else 0.0
            )

            if overlap >= self.sleep_preplay_consistency:
                # 预测与历史经验一致 → 强化历史实际路径（thicken_sheath）
                self._reinforce_path_via_sheaths(
                    actual_trajectory, amount=0.03
                )
            else:
                # 不一致 → 衰减预测路径（decay_sheath）
                self._decay_path_via_sheaths(
                    predicted_path, amount=0.02
                )

            return {
                "phase": "preplay",
                "overlap": overlap,
                "consistent": overlap >= self.sleep_preplay_consistency,
                "predicted_on": "external_input",
            }
        return {
            "phase": "preplay",
            "overlap": 0.0,
            "consistent": False,
            "predicted_on": "external_input",
        }

    def check_convergence(
        self,
        weight_delta: float,
        myelin_stable: bool,
    ) -> ConvergenceState:
        """收敛判据集成——委托 engine.check_convergence

        Args:
            weight_delta: 权重变化量（< ε 时收敛）
            myelin_stable: 髓鞘拓扑是否稳定
        """
        return self.higher_brain.engine.check_convergence(
            weight_delta=weight_delta,
            myelin_stable=myelin_stable,
        )

    # ============================================================
    # 状态持久化（只保存脑状态：权重 W + 髓鞘 + 发育元数据）
    # ============================================================

    def save_state(self, path: str) -> None:
        """保存脑状态到目录

        保存内容：
        - neurons: seed / W (Tensor) / unfolded (DimSlice) / parent_seed / alive
        - sheaths: 所有髓鞘的 delay/gain/protection + 连接信息
        - engine: novelty_accumulator / novelty_threshold / differentiation_count
        - system: phase / step_count / port_layout

        不保存：_trajectory_log（梦境素材，临时性）、ports（外部连接，运行时重建）

        格式：{path}/brain.json（元数据）+ {path}/tensors.pt（所有 Tensor）
        """
        os.makedirs(path, exist_ok=True)

        # 1. 收集神经元元数据 + Tensor
        eco = self.higher_brain.ecosystem
        neurons_meta = []
        tensors: dict[str, torch.Tensor] = {}
        for idx, neuron in eco.neurons.items():
            unfolded_meta = {
                ch: {
                    "dim_idx": slc.dim_idx,
                    "size": slc.size,
                    "gain": slc.gain,
                    "activity": slc.activity,
                    "protection": slc.protection,
                    "myelinated": slc.myelinated,
                }
                for ch, slc in neuron.unfolded.items()
            }
            neurons_meta.append({
                "idx": idx,
                "seed": neuron.seed,
                "parent_seed": neuron.parent_seed,
                "alive": neuron.alive,
                "metabolism": neuron.metabolism,
                "unfolded": unfolded_meta,
                "has_W": neuron.W is not None,
            })
            if neuron.W is not None:
                tensors[f"neuron_{idx}_W"] = neuron.W

        # 2. 收集髓鞘
        sheaths_meta = []
        reg = self.higher_brain.sheath_registry
        for key, sheath in reg._sheaths.items():
            sheaths_meta.append({
                "src_neuron": sheath.src_neuron,
                "src_channel": sheath.src_channel,
                "dst_neuron": sheath.dst_neuron,
                "dst_channel": sheath.dst_channel,
                "delay": sheath.delay,
                "gain": sheath.gain,
                "protection": sheath.protection,
            })

        # 3. 发育引擎状态
        engine = self.higher_brain.engine
        engine_meta = {
            "novelty_accumulator": engine._novelty_accumulator,
            "novelty_threshold": engine.novelty_threshold,
            "novelty_decay": engine.novelty_decay,
            "differentiation_count": engine._differentiation_count,
            "total_signal_count": engine._total_signal_count,
        }

        # 4. 系统元数据
        system_meta = {
            "phase": self.phase,
            "step_count": self.step_count,
            "port_layout": self.port_layout,
        }

        brain_json = {
            "system": system_meta,
            "neurons": neurons_meta,
            "sheaths": sheaths_meta,
            "engine": engine_meta,
        }

        with open(os.path.join(path, "brain.json"), "w", encoding="utf-8") as f:
            json.dump(brain_json, f, ensure_ascii=False, indent=2)

        if tensors:
            torch.save(tensors, os.path.join(path, "tensors.pt"))

    def load_state(self, path: str) -> None:
        """从目录加载脑状态

        恢复 save_state 保存的所有内容。
        注意：ports（外部连接）需在 load_state 后由外部重新注册。
        """
        from mcp.developmental.neuron import Neuron, DimSlice

        with open(os.path.join(path, "brain.json"), "r", encoding="utf-8") as f:
            brain_json = json.load(f)

        tensors: dict[str, torch.Tensor] = {}
        tensors_path = os.path.join(path, "tensors.pt")
        if os.path.exists(tensors_path):
            tensors = torch.load(tensors_path, weights_only=False)

        # 1. 恢复系统元数据
        sys_meta = brain_json["system"]
        # port_layout 不一致时报错（JSON 把 tuple 序列化成 list，需归一化比较）
        saved_layout = {k: tuple(v) for k, v in sys_meta["port_layout"].items()}
        if saved_layout != self.port_layout:
            raise ValueError(
                f"port_layout mismatch: saved={saved_layout} "
                f"current={self.port_layout}"
            )
        self.phase = sys_meta["phase"]
        self.step_count = sys_meta["step_count"]

        # 2. 恢复神经元生态
        eco = self.higher_brain.ecosystem
        eco.neurons.clear()
        eco._next_idx = 0

        for n_meta in brain_json["neurons"]:
            neuron = Neuron(
                seed=n_meta["seed"],
                port_layout=self.port_layout,
            )
            neuron.parent_seed = n_meta.get("parent_seed")
            neuron.alive = n_meta["alive"]
            neuron.metabolism = n_meta.get("metabolism", 0.5)

            # 恢复 unfolded + W
            for ch, slc_meta in n_meta["unfolded"].items():
                slc = DimSlice(
                    dim_idx=slc_meta["dim_idx"],
                    size=slc_meta["size"],
                    gain=slc_meta["gain"],
                    activity=slc_meta["activity"],
                    protection=slc_meta["protection"],
                    myelinated=slc_meta["myelinated"],
                )
                neuron.unfolded[ch] = slc

            if n_meta["has_W"]:
                key = f"neuron_{n_meta['idx']}_W"
                if key in tensors:
                    neuron.W = tensors[key]

            eco.neurons[n_meta["idx"]] = neuron
            if n_meta["idx"] >= eco._next_idx:
                eco._next_idx = n_meta["idx"] + 1

        # 3. 恢复髓鞘
        reg = self.higher_brain.sheath_registry
        reg._sheaths.clear()
        for s_meta in brain_json["sheaths"]:
            reg.add_sheath(
                src_neuron=s_meta["src_neuron"],
                src_channel=s_meta["src_channel"],
                dst_neuron=s_meta["dst_neuron"],
                dst_channel=s_meta["dst_channel"],
                delay=s_meta["delay"],
                gain=s_meta["gain"],
                protection=s_meta["protection"],
            )

        # 4. 恢复发育引擎
        engine = self.higher_brain.engine
        eng_meta = brain_json["engine"]
        engine._novelty_accumulator = eng_meta["novelty_accumulator"]
        engine.novelty_threshold = eng_meta["novelty_threshold"]
        engine.novelty_decay = eng_meta["novelty_decay"]
        engine._differentiation_count = eng_meta["differentiation_count"]
        engine._total_signal_count = eng_meta["total_signal_count"]
