"""Layer 5: 高层脑 — 始终活跃的可塑发育层

核心职责：
1. process(): 介入输入信号化（始终活跃）——用 find_coverage 做覆盖度匹配，
   记录激活到 CoincidenceDetector，检测同时激活对并建立髓鞘。
2. compute_lambda(): 连续干预强度 λ ∈ [0,1]（高/低/中区间 + 非语言模态 λ=1.0 + sleep λ=lambda_sleep）。
3. intervene(): 残差修正 y = (1-λ)·y_reflex + λ·y_higher；高层输出 y_higher 来源：
   用 SignalDispatcher 并行分发信号，收集触发事件作为高层输出。
4. dispatch_signal() / resolve_triggers(): 委托 SignalDispatcher 做事件驱动并行分发
   （时序竞争 + 重合窗口跨模态绑定）。
5. learn_from_intervention_delta(): 从干预差异学习——对被干预神经元 adjust_weights 微调权重到收敛。
6. record_for_dream(): 记录 4 元组供做梦学习。

核心变更（相比旧实现）：
- propagate_signal（softmax 随机游走）→ dispatch_signal（事件驱动并行分发）
- match（余弦相似度）→ find_coverage（覆盖度匹配）
- myelin.update（Q 矩阵更新）→ sheath_registry.thicken_sheath + CoincidenceDetector 追溯源
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import math
import torch

from mcp.developmental.signal import Signal
from mcp.developmental.neurode import NeuronEcosystem
from mcp.developmental.myelin import (
    MyelinSheath,
    MyelinSheathRegistry,
    CoincidenceDetector,
    SignalDispatcher,
    SignalEvent,
)
from mcp.developmental.development import DevelopmentEngine
from mcp.developmental.selection import DualTargetScheduler
from mcp.developmental.normalizer import ChannelNormalizer
from mcp.developmental.attention import AttentionModulator
from mcp.developmental.spikes import SpikeCoder


class HigherBrain(ABC):
    """高层脑抽象基类 — 用户可替换发育策略"""

    @abstractmethod
    def process(self, input_signals: dict[str, Signal]) -> dict[str, Any]:
        """介入输入信号化（始终活跃）：find_coverage 匹配 + 记录激活 + 建立髓鞘"""
        pass

    @abstractmethod
    def compute_lambda(self, match_result: dict) -> float:
        """计算连续干预强度 λ ∈ [0,1]"""
        pass

    @abstractmethod
    def intervene(
        self,
        raw_outputs: dict[str, Signal],
        match_result: dict[str, Any],
    ) -> dict[str, Signal]:
        """介入输出处理（连续 λ 残差修正）：y = (1-λ)·y_reflex + λ·y_higher"""
        pass

    @abstractmethod
    def dispatch_signal(
        self, signal: torch.Tensor, source_tag: str
    ) -> list[SignalEvent]:
        """事件驱动并行分发：委托 SignalDispatcher.dispatch"""
        pass

    @abstractmethod
    def resolve_triggers(
        self,
        events: list[SignalEvent],
        threshold: float,
        coincidence_window: float,
    ) -> list[SignalEvent]:
        """解析触发：时序竞争 + 重合窗口跨模态绑定"""
        pass

    @abstractmethod
    def learn_from_intervention_delta(
        self,
        reflex_outputs: dict[str, Signal],
        final_outputs: dict[str, Signal],
    ) -> None:
        """从干预差异学习——微调被干预神经元的权重"""
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """获取最近一次匹配的置信度"""
        pass

    @abstractmethod
    def record_for_dream(
        self,
        input_signals: dict[str, Signal],
        reflex_outputs: dict[str, Signal],
        final_outputs: dict[str, Signal],
        match_result: dict[str, Any],
    ) -> None:
        """记录 4 元组到轨迹日志供做梦学习"""
        pass


class DefaultHigherBrain(HigherBrain):
    """默认实现：始终介入 + 连续 λ 残差干预 + 事件驱动并行分发 + 维度级处理 + 权重微调"""

    def __init__(
        self,
        port_layout: dict[str, tuple[int, int]],
        max_nodes: int = 10000,
        llm_scorer=None,
        llm_topk: int = 20,
        theta_high: float = 0.8,
        theta_low: float = 0.2,
        lambda_high: float = 0.9,
        lambda_low: float = 0.7,
        lambda_mid: float = 0.2,
        lambda_sleep: float = 0.1,
        lambda_beta: float = 8.0,
        lambda_threshold: float = 0.0,
        # --- 解耦扩展层（均可开关，默认开；消融时可关）---
        use_attention: bool = True,
        attn_depth: float = 1.0,
        attn_lr: float = 0.05,
        use_spikes: bool = True,
        spike_threshold: float = 0.2,
        spike_decay: float = 0.3,
        spike_refractory: int = 1,
        coincidence_retention: float = 1.0,
        gate_beta: float = 8.0,
        self_proof: float = 0.3,
        saturation_split: bool = True,
    ):
        self.port_layout = port_layout
        self.global_dim = sum(s for _, s in port_layout.values())
        # 超模态统一编码：逐信道自适应归一化
        #
        # 必须在发育机制之前。异构信道尺度差一个数量级以上时，通路存亡由
        # **尺度**而非**可预测性**决定——实测未归一化时噪声通路比真实通路
        # 更耐衰减（ρ=0.2 下真实维剩 3.0 条、噪声维剩 4.6 条）。
        self.normalizer = ChannelNormalizer(port_layout)
        # 双目标调度器：自回归裁定蒸馏，产出 λ 与贡献判据
        self.scheduler = DualTargetScheduler(beta=gate_beta)
        # 核心组件
        self.ecosystem = NeuronEcosystem(
            port_layout=port_layout, max_nodes=max_nodes
        )
        self.sheath_registry = MyelinSheathRegistry()
        # 成边门控阈值（先自证再连接）
        #
        # ⚠ 必须在 sheath_registry **创建之后**设置。原代码把这一行放在
        # registry 构造之前（第 133 行），导致 AttributeError——
        # DevelopmentalSystem 一构造就崩。此 bug 与 torch 无关，
        # 此前从未被发现是因为代码从未真正运行过（沙盒无 torch）。
        self.sheath_registry.self_proof = self_proof
        self.engine = DevelopmentEngine(
            self.ecosystem, self.sheath_registry,
            saturation_split=saturation_split,
        )
        self.coincidence = CoincidenceDetector(retention=coincidence_retention)
        # SignalDispatcher 持有 ecosystem.neurons 与 sheath_registry._sheaths 的同一引用，
        # 新增神经元/髓鞘时自动可见（dict 按引用传递）
        #
        # history_capacity 需覆盖 ``DELAY_MAX + 2×window``，否则 delay 对齐度
        # 计算会因超出缓冲范围而失败（计入 align_stats["out_of_range"]）。
        max_delay = MyelinSheath.DELAY_MAX
        self.dispatcher = SignalDispatcher(
            self.ecosystem.neurons, self.sheath_registry._sheaths,
            history_capacity=int(max_delay * 3 + 16),
            port_layout=port_layout,
            connections=self.sheath_registry._connections,
        )
        # 可选 LLM 预筛（新匹配用 find_coverage，LLM 预筛保留接口供未来扩展）
        self.llm_scorer = llm_scorer
        self.llm_topk = llm_topk
        # 接管判定阈值
        self.theta_high = theta_high
        self.theta_low = theta_low
        # 连续 λ 参数
        self.lambda_high = lambda_high
        self.lambda_low = lambda_low
        self.lambda_mid = lambda_mid
        self.lambda_sleep = lambda_sleep
        # 驱动满足度门控的 σ 斜率与阈值（compute_lambda 主逻辑使用）。
        # ⚠ 此前 __init__ 未定义这两个属性，而 compute_lambda 在
        # drive_sat 就位后引用它们 → 接管瞬间 AttributeError。
        # 影子模式通常不可达该分支，所以从未在测试中炸过。
        # 默认值与 DriveSatisfaction.lambda_gate 的签名一致（β=8, θ=0）。
        self.lambda_beta = lambda_beta
        self.lambda_threshold = lambda_threshold
        # --- 解耦扩展层：注意力调制 + 脉冲编码 ---
        #
        # 两者都挂在 dispatch 的咽喉点，互不侵入既有算子：
        #   attention : 内容寻址的乘性再分配（键向量是第三组独立参数）
        #   spike     : 通路级漏积分过阈放行（胞体保持速率编码）
        # 顺序：先 attention（改变增益 → 影响膜电流），后 spike。
        self.attention = (
            AttentionModulator(self.global_dim, depth=attn_depth, lr=attn_lr)
            if use_attention else None
        )
        self.spike_coder = (
            SpikeCoder(threshold=spike_threshold, decay=spike_decay,
                       refractory=spike_refractory)
            if use_spikes else None
        )
        # --- 源可信度：**世界裁定**的发言读出门控（反应学习的落点）---
        #
        # 实测（grammar 任务）：y_higher_lang 是无差别求和，分化子代
        # （W 重新播种、预测垃圾）的髓鞘化边 gain=2.0 与种子神经元的
        # 好预测等权 → 垃圾淹没信号 → 探索全错（sat_high=0/58）→
        # RPE 恒负 → 高层被锁死。
        #
        # 进一步实测（逐源解剖）：即使按源门控，读出 argmax 仍由
        # **payload 幅值**决定而非正确性——t=568 源 0:aud 预测了正确
        # 答案却被 cred 压到 gate=0.05，错误的 0:vis 靠幅值获胜。
        #
        # 最终方案：**确认投票制**。每条跨模态通路对输出信道的 argmax
        # token 投一票，票权 = 世界对该 (源, token) 转移的逐条确认度
        # （EMA 命中率，初始 0.5 中立）。被世界反复确认的转移统治读出，
        # 幅值无发言权——这是"共发射确认→髓鞘化"在读出层的重演，
        # 也是"逻辑先于统计"的机制版。反馈只有环境即时揭示。
        self.vote_cred: dict[tuple[tuple[int, str], int], float] = {}
        self.vote_seen: dict[tuple[tuple[int, str], int], int] = {}
        self.vote_lr: float = 0.15
        self._pending_preds: list[tuple[tuple[int, str], int]] = []
        # 符号信道集合：这些信道的读出走确认投票；其余信道保持幅值求和
        # （连续模态的 argmax 无意义）。由任务侧设置，如 {"lang"}。
        self.symbolic_channels: set = set()
        # 状态
        self._last_confidence = 0.0
        self._last_lambda = 0.0
        self._trajectory: list[int] = []
        self._trajectory_log: list[tuple] = []  # 做梦学习 4 元组日志

    def process(self, input_signals: dict[str, Signal]) -> dict[str, Any]:
        """介入输入信号化（始终活跃）

        (a) 调用 ecosystem.find_coverage 找最佳覆盖神经元（覆盖度匹配，替代余弦相似度）
        (b) 对每个信道信号记录激活到 CoincidenceDetector（追溯激活源）
        (c) 检测同时激活对，调用 engine.thicken_myelin_from_coincidence 建立髓鞘
        """
        active_channels = set(input_signals.keys())
        # (a) 覆盖度匹配——find_coverage 实际只用 active_channels 计算重合度
        global_signal = torch.zeros(self.global_dim)
        winner_id, coverage_score = self.ecosystem.find_coverage(
            global_signal, active_channels
        )

        # (b) 推进逻辑时钟，并记录每个信道的激活
        #
        # 旧实现传入恒定的 t=0.0，导致 CoincidenceDetector 的 retention
        # 清理条件 (t - r.timestamp < retention) 恒为真，记录无限累积，
        # find_coincident_pairs 的全组合复杂度随之爆炸。
        # 现在时钟由检测器自己维护，每个系统步推进一次。
        self.coincidence.tick()
        for ch_name, sig in input_signals.items():
            source_tag = sig.metadata.get("source_tag", "default")
            for idx in self.ecosystem.get_active_indices():
                neuron = self.ecosystem.get_neuron(idx)
                if neuron is not None and ch_name in neuron.unfolded:
                    # 不传 t → 使用检测器内部时钟
                    self.coincidence.record(idx, ch_name, source_tag)

        # (c) 共发射成边（**不是**追溯激活源）
        #
        # 旧实现按 source_tag 相等配対——那是因果追溯，不是共发射。
        # 新实现按到达时刻接近配対，且允许跨信道 → 超模态混合。
        window = self.sheath_registry.suggest_window()
        pairs = self.coincidence.find_coincident_pairs(
            window=window, allow_cross_channel=True
        )
        if pairs:
            self.sheath_registry.co_fire_wire(pairs)

        # 记录轨迹
        if winner_id >= 0:
            self._trajectory.append(winner_id)
            if len(self._trajectory) > 100:
                self._trajectory.pop(0)

        self._last_confidence = coverage_score

        # 检测非语言模态（供 compute_lambda 判定 λ=1.0）
        has_non_language = any(
            not s.mime_type.startswith("text/") for s in input_signals.values()
        )
        return {
            "winner_id": winner_id,
            "confidence": coverage_score,
            "coverage_score": coverage_score,
            "active_channels": list(active_channels),
            "trajectory": list(self._trajectory),
            "non_language": has_non_language,
        }

    def compute_lambda(self, match_result: dict) -> float:
        """计算连续干预强度 λ ∈ [0,1]

        间歇替代模型：高层脑始终在干扰，强度连续可变。
        y = (1-λ)·y_reflex + λ·y_higher

        **主逻辑已改为由双目标调度器裁定**（见 :mod:`selection`）：

            λ = (1 − urgency) · σ(β · (progress − threshold))

        其中 progress 来自自回归裁定——高层只有在环境反馈下真的比反射
        预测得更准时才逐步接管。这替换了旧的置信度阈值查表（五个魔数），
        因为置信度只说明"高层见过类似输入"，不能说明"高层比反射更对"。

        前置短路（优先于调度器，它们是安全约束而非学习目标）：
        - 胚胎期（winner 的 W=None）：高层无法处理 → λ=0，反射保底
        - 非语言模态：λ=1.0（高层完全接管）
        - 睡眠态：压低至 lambda_sleep（弱响应，非完全切断）
        - 紧急（urgency→1）：λ→0，反射接管（等不到下一帧环境信号）
        """
        # 睡眠态优先：压低 λ（运动弛缓对所有模态生效，对应睡眠时的肌张力丧失）
        is_sleep = match_result.get("sleep_mode", False)
        if is_sleep:
            lam = self.lambda_sleep
            self._last_lambda = lam
            return lam
        # 胚胎期保底：winner 神经元 W=None（未展开维度）→ 高层无法处理 → λ=0
        # 对应"反射弧自完备"——高层脑未发育时反射弧保底输出
        winner_id = match_result.get("winner_id", -1)
        if winner_id >= 0:
            neuron = self.ecosystem.get_neuron(winner_id)
            if neuron is None or neuron.W is None:
                self._last_lambda = 0.0
                return 0.0
        # ---- 影子模式：反射全权，高层只观察 ----
        #
        # 这不是"高层不学习"，而是"高层不影响输出"。学习（自回归残差
        # 更新、髓鞘选择、共发射成边）照常进行，只是输出走反射通路。
        #
        # 生物学：这对应**观察学习** —— 婴儿在能自主完成动作之前，
        # 先在观察与被动运动中建立感觉运动映射。
        if match_result.get("shadow_mode", False):
            self._last_lambda = 0.0
            return 0.0

        # ---- 探索步：高层**真正**接管 ----
        #
        # ⚠ 缺失这一段，"探索"名存实亡（实测根因之一）。
        #
        # 探索标志只让 shadow_mode=False，但 λ 仍由 competence 决定。
        # 而 competence 恒为负（因为高层从未被允许真正输出过，
        # 无从证明自己）→ λ=σ(β·(−0.95))≈0.0005 → 输出几乎全是反射的。
        #
        # 后果：eligibility 记的是"输入 → **反射**输出"，三因子强化的
        # 也是这个映射。高层自己的输出从未进入评估，永远学不会 ——
        # 失败的资格被算到了反射头上，形成死锁。
        #
        # 探索步必须强制 λ=1，让高层的输出真正作用于世界，
        # 才能被 RPE 公正地评判。
        if match_result.get("exploring", False):
            self._last_lambda = 1.0
            return 1.0

        # 非语言模态判定
        #
        # ⚠ 必须排在影子模式**之后**：影子期反射全权是安全约束，
        # 优先级高于模态判定。旧顺序下 non_language 先返回 λ=1.0，
        # 影子期形同虚设（实测：shadow=True 但 λ=1.000）。
        is_non_language = match_result.get("non_language", False)
        if is_non_language:
            self._last_lambda = 1.0
            return 1.0

        # ---- 主逻辑：由**驱动满足度**裁定 ----
        #
        # ⚠ 两代判据的更替（均有实测支撑，勿改回）
        #
        # ① 置信度阈值查表（五个魔数）—— 已弃
        #    缺陷：置信度只说明"高层见过类似输入"，
        #          不能说明"高层比反射更对"。
        #
        # ② 自回归预测误差 —— 已弃，**实测有陷阱**
        #    λ = (1−urgency)·σ(β·(progress−θ))
        #
        #    在反应式控制 loop 里，下一帧观测取决于系统自己的输出：
        #        高层输出错误 → 世界无响应 → 观测变全零
        #        → 恒零预测误差极低 → progress 上升
        #        → λ 升高 → 更彻底接管 → 世界彻底死寂
        #    实测满足率：反射 0.941 / 高层 0.001 / λ 自适应 0.003。
        #    **"预测得准" ≠ "控制得好"。**
        #
        # ③ 驱动满足度（本实现）—— 裁定权归爬虫脑
        #    λ = (1−urgency)·σ(β·(competence−θ))
        #
        #    competence 由**交替试验**测得：影子期以 ε 概率让高层真正
        #    接管，比较两种控制下的内在驱动满足度。只有高层让世界更
        #    接近想要的状态，才允许接管。
        #
        #    满足度是**爬虫脑的内在信号**（多巴胺能系统的角色），
        #    不属于感知通道 —— 高层看不到它，也无法通过"把世界搞死寂"
        #    来伪造它。
        urgency = match_result.get("urgency", 0.0)
        ds = getattr(self, "drive_sat", None)
        if ds is not None:
            # 感知建模护栏：competence 按感知情境条件化（T4 修复）
            ctx = (match_result.get("winner_id", -1)
                   if getattr(self, "contextual_competence", False) else None)
            lam = ds.lambda_gate(urgency, beta=self.lambda_beta,
                                 threshold=self.lambda_threshold,
                                 context=ctx)
        else:
            lam = self.scheduler.lambda_gate(urgency)   # 退化路径
        self._last_lambda = lam
        return lam

    def intervene(
        self,
        raw_outputs: dict[str, Signal],
        match_result: dict[str, Any],
        global_signal: Optional[object] = None,
    ) -> dict[str, Signal]:
        """介入输出处理（连续 λ 残差修正）

        y = (1-λ)·y_reflex + λ·y_higher

        高层输出 y_higher 来源：用 dispatcher 并行分发信号，
        收集触发事件作为高层输出（事件驱动并行分发，各维度同时经各自算子传输）。

        Args:
            global_signal: **全局组合信号**（归一化后，长度 = global_dim）。
                必须提供，否则会退化用单通道信号去 dispatch，而神经元的
                ``active_indices`` 是全局索引 → 索引越界 → 被上层
                ``except Exception`` 静默吞掉 → 高层脑从不介入。
        """
        lam = self.compute_lambda(match_result)
        if lam <= 0.0:
            return dict(raw_outputs)
        if lam >= 1.0 and not raw_outputs:
            return {}

        final: dict[str, Signal] = {}
        for key, reflex_sig in raw_outputs.items():
            higher_data = self._generate_higher_output(
                reflex_sig, match_result,
                global_signal=global_signal, channel=key)
            blended = (1.0 - lam) * reflex_sig.data + lam * higher_data
            final[key] = Signal(
                data=blended,
                mime_type=reflex_sig.mime_type,
                metadata={
                    **reflex_sig.metadata,
                    "lambda": lam,
                    "intervened": True,
                },
            )
        return final

    # ---------- 确认投票（世界裁定的读出） ----------

    def note_source_predictions(
        self, preds: list[tuple[tuple[int, str], int]]
    ) -> None:
        """备案本步各通路的预测 payload（由 system 在 dispatch 后调用）

        Args:
            preds: ``[((src_n, src_ch), voted_token)]``——指向输出信道的
                跨模态事件，其 payload 在输出信道段的 argmax。
        """
        self._pending_preds = list(preds)

    def settle_source_predictions(self, actual_token: int) -> int:
        """世界已揭示真实 token → 逐条 (源, token) 转移结算确认度

        Args:
            actual_token: 环境当前帧展示的真实 token（从世界显示信道读出）

        Returns:
            结算的票数。
        """
        n = 0
        for src, v in self._pending_preds:
            key = (src, v)
            e = self.vote_cred.get(key, 0.5)     # 未知转移 → 中立半票
            hit = 1.0 if v == actual_token else 0.0
            self.vote_cred[key] = e + self.vote_lr * (hit - e)
            self.vote_seen[key] = self.vote_seen.get(key, 0) + 1
            n += 1
        self._pending_preds = []
        return n

    def vote_gate(self, src: tuple[int, str], token: int) -> float:
        """(源, token) 转移的世界确认度 ∈ (0, 1)；未知 → 0.5 中立"""
        return self.vote_cred.get((src, token), 0.5)

    def src_cred_report(self) -> dict:
        top = sorted(self.vote_cred.items(),
                     key=lambda kv: -kv[1])[:12]
        out = {}
        for (sn_sc, tok), e in top:
            out[f"{sn_sc[0]}:{sn_sc[1]}->tok{tok}"] = {
                "confirm": round(e, 3),
                "seen": self.vote_seen.get((sn_sc, tok), 0),
            }
        return out

    def _generate_higher_output(
        self,
        reflex_sig: Signal,
        match_result: dict[str, Any],
        global_signal: Optional[object] = None,
        channel: Optional[str] = None,
    ) -> torch.Tensor:
        """生成高层输出 y_higher——用 dispatcher 并行分发信号，收集触发事件

        各维度切片同时经各自算子（W）传输，髓鞘包裹层（delay/gain）调制传输特性。

        Args:
            global_signal: 全局组合信号。dispatch 必须用它——神经元展开的
                是全局维度，用单通道信号会索引越界。
            channel: 输出通道名，用于从全局高层输出中切出本通道的段。
        """
        # 事件驱动并行分发
        src = global_signal if global_signal is not None else reflex_sig.data
        events = self.dispatch_signal(src, source_tag="higher")
        if not events:
            # 无事件（无髓鞘连接）：胚胎期注入小噪声（探索性驱动）
            return reflex_sig.data + torch.randn_like(reflex_sig.data) * 0.05

        # 解析触发：时序竞争 + 重合窗口跨模态叠加
        triggered = self.resolve_triggers(
            events, threshold=0.01, coincidence_window=0.1
        )

        # ⚠ 按**目标通道**过滤事件（实测修复，真实系统 1200 步）
        #
        # 事件 data 经 ``_scatter_to_global`` 散射后，每条边只在其 dst_ch
        # 对应的全局位置有值。原实现取"全局最强那条边"，若这条边不指向
        # 当前输出通道，切出来的段**恒为 0**。
        #
        # 实测后果：lang 通道的高层输出趋近 0 → ar_high ≈ |target|² = 24
        # （与实测 24.24 吻合）→ progress 卡在 0.35，λ 虽已达 0.94~0.99
        # 但高层接管后输出的是零向量，等于没接。
        #
        # 修正：只用 dst_ch == channel 的事件。若该通道没有任何入边，
        # 退回反射值 + 探索噪声（而不是返回 0）。
        if channel is not None:
            ev_ch = [e for e in triggered if e.channel == channel]
            if not ev_ch:
                ev_ch = [e for e in events if e.channel == channel]
            if not ev_ch:
                return reflex_sig.data + torch.randn_like(reflex_sig.data) * 0.05
            # ⚠ 发言读出**跨模态优先**（实测定位的学习断点，勿删）：
            #
            # lang 信道的输入 = 系统自己上一帧的话（自指目标）→ W 的
            # lang 行学到的只是平凡回声映射。若让 lang→lang 传输参与
            # 发言求和，y_higher_lang 的 argmax 恒为"上一个 token"
            # （自回声），把 vis/aud 段真正学到的 next-token 预测淹没。
            # 实测（grammar 任务 2000 步）：W 的 vis 段探测命中 8/16、
            # 跨模态边 0:vis→0:lang 已髓鞘化 gain=2.0，但 sat_high=0/57
            # ——读出被回声淹没，competence 恒负，高层锁死在影子模式。
            #
            # 修正：优先取源信道 ≠ 输出信道的事件（答案从预测段读出，
            # 这正是超模态的本义）；无跨模态事件才回退同信道。
            cross = [e for e in ev_ch if e.sheath_key is not None
                     and e.sheath_key[1] != channel]
            if cross:
                ev_ch = cross
        else:
            ev_ch = triggered if triggered else events

        if ev_ch and channel in self.symbolic_channels:
            # --- 确认投票读出（符号信道）---
            #
            # 每条通路对 payload argmax token 投一票，票权 = 世界对该
            # (源, token) 转移的确认度。幅值不参与——谁说得对由世界
            # 说了算（逻辑先于统计）。同源多边投同一 token 去重。
            off_s, size_s = self.port_layout[channel]
            higher = torch.zeros(self.global_dim)
            votes: dict[tuple[tuple[int, str], int], float] = {}
            for e in ev_ch:
                if e.sheath_key is None:
                    continue
                try:
                    seg = e.data[off_s:off_s + size_s]
                    am = seg.argmax()
                    am = int(am.item()) if hasattr(am, "item") else int(am)
                    mx = seg.max()
                    mx = float(mx.item()) if hasattr(mx, "item") else float(mx)
                except Exception:
                    continue
                if mx <= 1e-9:
                    continue
                key = (e.sheath_key[:2], am)
                w = self.vote_gate(*key)
                if w > votes.get(key, -1.0):
                    votes[key] = w
            for (src, tok), w in votes.items():
                higher[off_s + tok] += w
        elif ev_ch:
            # 非符号信道：幅值求和（连续模态无 argmax 语义）
            higher = sum(e.data for e in ev_ch)
        else:
            higher = max(events, key=lambda e: e.data.norm().item()).data

        # 对齐到 reflex_sig.data 的形状
        #
        # ⚠ higher 现在是**全局**向量（dispatch 事件已 scatter 回全局空间）。
        # 原实现取 ``higher[:n]``（前 n 个），对第一个信道碰巧正确，对其余
        # 信道是**错的**——aud 通道会拿到 vis 通道的值。
        # 修复：按 port_layout 切出本通道对应的一段。
        seg = None
        if channel is not None and channel in self.port_layout:
            off, size = self.port_layout[channel]
            if int(higher.numel()) >= off + size:
                seg = higher[off:off + size]
        if seg is None:
            # 回退：无布局信息或形状不符时按旧行为
            if higher.numel() >= reflex_sig.data.numel():
                seg = higher[: reflex_sig.data.numel()]
            else:
                padded = torch.zeros_like(reflex_sig.data)
                padded[: higher.numel()] = higher
                return padded
        return seg.clone()

    def dispatch_signal(
        self, signal: torch.Tensor, source_tag: str
    ) -> list[SignalEvent]:
        """事件驱动并行分发——委托给 SignalDispatcher.dispatch

        信号各维度分量同时经各自算子传输（并行性从维度对偶性自然涌现）。
        每次分发前清空事件队列，避免历史事件累积。
        """
        self.dispatcher.event_queue = []  # 清空队列
        # T1: 走边批向量化路径（内部自适应回退：小边数走 loop，语义逐位等价）
        events = self.dispatcher.batched_dispatch(signal, source_tag)
        # --- 解耦扩展层（单一咽喉点：三条 dispatch 调用路径都经过这里）---
        if self.attention is not None:
            events = self.attention.modulate(events, query=signal)
        if self.spike_coder is not None:
            events = self.spike_coder.encode(events)
        return events

    def resolve_triggers(
        self,
        events: list[SignalEvent],
        threshold: float,
        coincidence_window: float,
    ) -> list[SignalEvent]:
        """解析触发——委托给 SignalDispatcher.resolve_triggers

        - 第一个到达阈值的赢（时序竞争）
        - 时间差 < 重合窗口的多信号叠加触发（跨模态绑定）
        """
        # 按到达时间排序后注入 dispatcher 队列
        self.dispatcher.event_queue = sorted(
            events, key=lambda e: e.arrival_time
        )
        return self.dispatcher.resolve_triggers(threshold, coincidence_window)

    def learn_from_intervention_delta(
        self,
        reflex_outputs: dict[str, Signal],
        final_outputs: dict[str, Signal],
    ) -> None:
        """从干预差异学习——对被干预的神经元调用 adjust_weights 微调权重到收敛

        差异越大，反馈越强，权重调整幅度越大。迭代到权重变化量 < ε 收敛。
        """
        if not self._trajectory:
            return
        winner_id = self._trajectory[-1]
        neuron = self.ecosystem.get_neuron(winner_id)
        if neuron is None or neuron.W is None:
            return

        for key in final_outputs:
            if key not in reflex_outputs:
                continue
            reflex_sig = reflex_outputs[key]
            final_sig = final_outputs[key]
            # 干预差异作为反馈信号
            diff = final_sig.data - reflex_sig.data
            feedback = diff.abs().mean().item()
            if feedback < 1e-6:
                continue
            # ⚠ neuron.adjust_weights 需要**全局**信号——active_indices
            # 是全局索引，而轨迹日志里的反射输出是**分通道**的。
            # 原实现直接传分通道数据 → 索引越界。凡高层接管过
            # （final ≠ reflex）的系统，睡眠反事实相位必然崩溃。
            # 修复：按 port_layout 散射回全局空间后再传入。
            g = torch.zeros(self.global_dim)
            off, size = self.port_layout.get(
                key, (0, int(reflex_sig.data.numel())))
            d = reflex_sig.data.reshape(-1)[:size]
            g[off:off + int(d.numel())] = d
            # 微调权重到收敛
            for _ in range(100):
                delta = neuron.adjust_weights(g, feedback)
                if delta < 1e-4:
                    break

    def get_confidence(self) -> float:
        """返回最近一次 match 的 confidence"""
        return self._last_confidence

    def record_for_dream(
        self,
        input_signals: dict[str, Signal],
        reflex_outputs: dict[str, Signal],
        final_outputs: dict[str, Signal],
        match_result: dict[str, Any],
    ) -> None:
        """记录 4 元组到 _trajectory_log 供做梦学习"""
        self._trajectory_log.append(
            (input_signals, reflex_outputs, final_outputs, match_result)
        )
