"""髓鞘包裹层 + 共发射成边 + 选择动力学（用进废退）

核心修订（相比旧实现）：

1. **髓鞘是第二算子，双轴可塑**：
   - ``delay``（时间轴算子）：信号**何时**到达 → 决定谁能"同时发射"。
   - ``gain``（强度轴算子）：信号**多强**到达 → 由用进废退直接调制。
   两者的更新规则必须**分离**：gain 由贡献驱动，delay 由到达时序驱动。

2. **成边规则改为共发射时序**：
   旧实现是"追溯激活源（同源同信道）"——那是因果追溯，不是共发射。
   新实现按"窗口内同时发射"成边，且允许**跨信道**成边（超模态混合）。

3. **选择动力学 = 持续衰减 + 按贡献增厚，无硬约束**：
   存活条件 ``contribution · use_scale > decay_rate · (1 - protection)``。
   这构成**自适应预算**（总容量随环境活跃度伸缩）而非固定预算，
   因此必须监控容量是否发散（见 :meth:`MyelinSheathRegistry.capacity_report`）。

生物学对应：活动依赖性髓鞘形成（adaptive myelination）——髓鞘改变传导延迟
→ 改变到达时序 → 改变谁能同时发射 → 改变成边 → 改变谁能被髓鞘化。
**选择作用于时间，而不只是作用于强度。**
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from torch import Tensor


# ---------------------------------------------------------------------------
# 髓鞘：第二算子
# ---------------------------------------------------------------------------

@dataclass
class MyelinSheath:
    """髓鞘：算子外的第二算子（时间轴 delay + 强度轴 gain）

    髓鞘不改变神经元内部的权重算子 W，而是作为包裹在外的**第二算子**，
    对传输过程施加两个可塑变换：

    - ``delay``（时间轴）：信号**何时**到达。增厚 → 延迟↓、时序更精确。
    - ``gain``（强度轴）：信号**多强**到达。由用进废退直接调制。

    更新规则分离（这是本设计的关键，两者混淆会互相干扰）：

    ==========  ==============================  ==========================
    轴           驱动信号                        方法
    ==========  ==============================  ==========================
    gain        贡献（对降低环境误差的贡献）      :meth:`apply_selection`
    delay       到达时序（共发射同步）            :meth:`adapt_delay`
    ==========  ==============================  ==========================

    ``protection`` 是保护系数，抑制衰减（对应生物学中核心通路的稳定性），
    在 :meth:`apply_selection` 中按 ``(1 - protection)`` 缩放衰减率。
    """
    src_neuron: int
    src_channel: str
    dst_neuron: int
    dst_channel: str
    delay: float = 1.0            # 时间轴算子：传导延迟（可塑）
    gain: float = 1.0             # 强度轴算子：传输增益（可塑）
    protection: float = 0.0       # 保护系数：抑制衰减
    # --- 选择与演化状态 ---
    usage: float = 0.0            # 使用累积（EMA）：用进废退中"用"的度量
    stability: float = 0.0        # 髓鞘化稳定度：分化触发的"稳定反应"判据
    birth_step: int = 0           # 建立时的步数（诊断用）

    # --- 停滞计数：「废退」的正确度量 ---
    #
    # 「废退」= **停止增厚**，不是衰减到消失。
    # 生物学：髓鞘蛋白半衰期 55 天 ~ 6 个月；纵向成像显示多数髓鞘
    # 一旦形成即保持稳定。故未使用的后果是"不再增长"，而非"被清除"。
    starving: int = 0             # 连续未被（正向）使用的步数

    # --- 离散未使用计数（协同 A）---
    #
    # 与 GNG（Fritzke 1995）的 edge age 对应关系，但**用途被扩展了**：
    # GNG 中 age 只是删除判据；这里 age 首先触发 **delay 探索**，
    # 探索失败后才进入删除。
    #
    # 为什么需要离散计数：EMA（usage/gain）是"衰减的记忆"——渐近趋零
    # 但**永不到零**。实测噪声通路的 gain 是零均值随机游走，衰减只能
    # 拉回 0 附近却杀不死（波动幅度 ∝ σ/√ρ，实测可达 0.2）。离散 age
    # 是"多久没用了"的硬计数，超阈值即剪，没有渐近尾巴。
    #
    # 两者互补：EMA 判"值不值得留"，age 判"多久没用了"。
    since_co_fire: int = 0        # 距上次成功共发射的步数
    delay_probes: int = 0         # 已尝试的 delay 探索次数（防无限重试）
    # 建立后经历的总步数（**不受**共发射重置）——新生儿豁免期判据
    age_steps: int = 0

    # --- 睡眠归因账本（sleep attribution）---
    #
    # 每个 sleep cycle 的四个相位（稳态缩放/重放/反事实/预演）各自对
    # 本髓鞘造成的 gain/stability 变化，分相位累积在此。
    #
    # **为什么必须有**：没有这份账本，"重放到底巩固了什么"只是一个
    # 假设——睡眠机制存在 ≠ 睡眠机制有效。账本让"教育是把外源沉积
    # 重新编码成能驱动内部重放的形式"这类命题第一次变成可测量对象：
    # 对比"跳过某相位的对照 episode"，即可归因各相位的巩固贡献。
    sleep_gain_delta: float = 0.0       # 睡眠相位累计造成的 gain 变化
    sleep_stability_delta: float = 0.0  # 睡眠相位累计造成的 stability 变化
    # {相位名: 累计 Δgain}，相位名见 DevelopmentalSystem.step_sleep
    sleep_phase_gain: dict = field(default_factory=dict)

    # --- e-prop 资格迹（T3：髓鞘链深度信用的地基）---
    #
    # E_j ← decay·E_j + contribution_j。瞬时贡献只在传输当下存在，
    # 而 RPE（世界裁定）延迟到来——elig 让"当时传输过"的通路在
    # RPE 到达时仍能按残迹分账。链 A→B→C 的深度信用由此可能：
    # 上游通路因下游有用而在延迟裁定中保留资格。
    elig: float = 0.0

    # --- 阈值张力张量（T8：标量资格迹的升维，绑定在髓鞘回路上）---
    #
    # T_j 是**逐维度负荷向量**（dst 信道各维上 |transmitted × residual|
    # 的 EMA）：标量 elig 只说"这条通路有没有用"，张量 T 说"它压在
    # 哪些维度上、压多重"。回路张力 = 神经元各汇入/汇出髓鞘的张力
    # 范数和——高张力 = 承重墙（漂移受阈值治理、变异被回滚探针
    # 甄别），低张力 = 隔墙（探索预算自由）。
    tension: Optional[np.ndarray] = None

    # --- 新生儿豁免期（必需，非可选优化）---
    #
    # **为什么必须有**：新通路初始增益较低（见
    # :meth:`MyelinSheathRegistry.co_fire_wire`），需要若干步的使用才能
    # 增厚到稳固水平。豁免期保护这段"尚未被充分验证"的窗口。
    #
    # 注：历史上这里曾配合 ``initial_gain=0`` 与每步衰减淘汰，导致周转率
    # 11.5 的"新建—淘汰"空转。那两个机制均已移除（髓鞘改为独立且稳定），
    # 豁免期现在的角色是**保护新通路的验证窗口**，而非续命。
    #
    # 豁免期内不参与死亡判据，但 gain 仍为 0（**不污染输出**），只累积增厚。
    # 到期后按正常判据决定存亡。
    #
    # 豁免期与成边 block 的协同（实测，5 seeds，40 候选 / 8 真通路）：
    #
    # ======  =======  =======  ======  ========  ========  ======
    # 豁免     block    真通路    噪声    周转率    精确率    召回
    # ======  =======  =======  ======  ========  ========  ======
    # 0        1        3.2/8     9.6     2910      0.242     0.40
    # 0        50       0/8       0.0      727      0.000     0.00
    # 10       1        8/8      25.6      232      0.239     1.00
    # 10       50       8/8       0.0       74      1.000     1.00
    # ======  =======  =======  ======  ========  ========  ======
    #
    # 单独任何一个都不行：豁免期单独用 → 噪声也活下来（精确率 0.24）；
    # block 单独用 → 全部死光（召回 0.00）。两者结合才是解。
    GRACE_PERIOD: int = 10

    # 上下界（类级常量）
    GAIN_MAX: float = 2.0
    GAIN_DEATH: float = 0.1       # gain 跌破此值 → 该髓鞘被淘汰
    DELAY_MIN: float = 0.01
    DELAY_MAX: float = 10.0
    # age 相关阈值
    AGE_PROBE: int = 50           # 多久没共发射 → 触发 delay 探索
    AGE_DEATH: int = 2000         # 多次探索仍失败 → 硬剪
    #    ⚠ 原值 200，对髓鞘而言**过短**。生物学：髓鞘一旦形成极稳定
    #    （蛋白半衰期 55 天 ~ 6 个月）。200 步就剪除等于让髓鞘朝生暮死，
    #    实测周转率 2832、髓鞘数剧烈振荡。放宽到 2000，使其回归
    #    "清理完全静默通路"的定位，而非常规维护手段。
    MAX_DELAY_PROBES: int = 3     # 最多探索几次，之后放弃

    def thicken(self, amount: float = 0.05) -> None:
        """髓鞘增厚：delay↓, gain↑, protection↑

        保留旧语义供路径级强化调用。注意本方法**不更新** usage，
        usage 由 :meth:`apply_selection` 按贡献累积。
        """
        self.delay = max(self.DELAY_MIN, self.delay - amount * 0.5)
        self.gain = min(self.GAIN_MAX, self.gain + amount)
        self.protection = min(1.0, self.protection + amount * 0.3)

    def decay(self, amount: float = 0.05) -> None:
        """髓鞘衰减：delay↑, gain↓

        protection 不随使用衰减，只在稳态缩放时调整。
        """
        self.delay = min(self.DELAY_MAX, self.delay + amount * 0.5)
        self.gain = max(self.GAIN_DEATH, self.gain - amount)

    def apply_independent_plasticity(
        self,
        contribution: float = 0.0,
        use_scale: float = 1.0,
        usage_momentum: float = 0.9,
        half_life: float = 0.0,
    ) -> bool:
        """**独立**可塑性：用则增厚，不用则停滞（不是衰减到死）

        ⚠ **本方法替换了原 ``apply_selection``**，因为原实现把三件
        违背生物学、也违背设计意图的东西引入了髓鞘：

        1. **全局竞争**：所有髓鞘共享一个衰减率，谁贡献大谁留存。
           但设计意图是"**所有髓鞘都独立**"——不存在两条髓鞘争夺
           同一份资源。
        2. **每步衰减**：``gain -= ρ·gain`` 每步执行，使髓鞘朝生暮死。
           实测周转率 800+，结构反复生灭。
        3. **负贡献加速淘汰**：一次错误预测就双倍衰减。

        **生物学事实**（已核实，直接否定上述三条）：

        - 髓鞘蛋白半衰期 **55 天 ~ 6 个月**：CNP 55 天、Cldn11 133 天、
          PLP 约 6 个月（in vivo 同位素标记）。髓鞘是极长寿命结构。
        - 纵向成像：**"the majority remain stable after initial
          formation"** —— 多数髓鞘一旦形成即保持稳定，只有一部分亚群
          表现长度可塑性。
        - 成人新生的少突胶质细胞**不是替换**发育期形成的髓鞘，而是
          **增加**总量（"increase the total number of oligodendrocytes
          and add additional myelin to the existing white matter"）。
        - 活动减少的后果是**新生减少**，不是已有髓鞘被快速清除。

        **因此"用进废退"在髓鞘层的正确形式是**：

        - **用进**：使用 → 增厚，渐近饱和（有上限，符合"增加更多髓鞘
          层"的形态学）。
        - **废退**：不用 → **停止增厚**。已有的髓鞘仍然稳定存在，
          不会因"这一步没用上"而消失。
        - 真正的移除只发生在两种情形：神经元死亡（连带），或**极长期**
          完全不用（以半衰期计，与"天到月"对应）。

        Args:
            contribution: 本步贡献 c ∈ [-1, 1]
            use_scale: 贡献到增量的缩放
            usage_momentum: usage EMA 动量
            half_life: 髓鞘半衰期（步）。0 = 不衰减（理想稳定）。
                若要模拟"极长期不用才缓慢退化"，设为很大的值。

        Returns:
            bool: 是否存活。恒定 True，除非 half_life 生效且 gain
                已降到死亡阈值以下。
        """
        # 1. 使用累积（EMA）——记录"用"的强度与方向
        self.usage = (
            usage_momentum * self.usage
            + (1.0 - usage_momentum) * contribution
        )

        # 2. 增厚：只有**正**贡献才增厚，且渐近饱和
        #
        # 为什么渐近饱和：髓鞘层数有物理上限（轴突直径决定），
        # 不是无限增厚。形态学上新增膜层沉积在轴突-髓鞘界面。
        if contribution > 0.0:
            self.gain += (use_scale * contribution
                          * (self.GAIN_MAX - self.gain))
            self.starving = 0           # 有使用 → 重置停滞计数
        else:
            # **废退 = 停止增厚，不是衰减**
            # 负贡献或零贡献都不惩罚：髓鞘不会因为一次未参与就消失。
            self.starving += 1

        # 3. 极缓慢的自然退化（可选，默认关闭）
        #
        # 只在 half_life > 0 时生效，且必须是很长的时间尺度。
        # 这是"不维护就缓慢退化"，与"每步竞争淘汰"有本质区别：
        # 前者是时间常数极大的一阶过程，后者是每步的零和博弈。
        if half_life > 0:
            self.gain *= (1.0 - 0.693147 / half_life)

        self.gain = max(0.0, min(self.GAIN_MAX, self.gain))

        # 4. 稳定度
        self.stability = (
            usage_momentum * self.stability
            + (1.0 - usage_momentum) * (1.0 if self.gain > 1.0 else 0.0)
        )

        return self.gain >= self.GAIN_DEATH

    def adapt_delay(
        self,
        group_arrival: float,
        base_time: float,
        lr: float = 0.1,
    ) -> None:
        """delay 向"成功绑定的群体到达时间"同步（时间轴算子的更新）

        成功参与跨模态绑定的事件，其 delay 向该绑定组的平均到达时间靠拢：

            ``delay ← delay + lr · (group_arrival - (base_time + delay))``

        效果：经常一起发射的通路，到达时间越来越同步 → 更容易再次共发射
        → 形成功能模块。这是"**同时发射的神经元相连接**"的正反馈实现，
        也是选择作用于时间的落点。

        未参与绑定的事件其 delay 不受调整，保持随机漂移 → 逐渐被淘汰。

        **「髓鞘 delay 是镇定器」——已验证（2026-09）**

        我们架构里最紧的正反馈环是：
        髓鞘调 delay → 改变到达时序 → 改变谁共发射 → 改变成边 →
        改变谁被髓鞘化。理论上这会发散。

        机制推演：耦合的有效强度是 ``gain · cos(φ)``，φ 由 delay 决定。
        delay 可塑先把 φ 调到 0（建设性），冻结则可能锁死在 φ→π
        （反相放大）。这与 Lefebvre 等 2025（*Communications Physics*
        8:145，已核实：髓鞘实施增益控制、把动力学稳定在远离振荡区；
        髓鞘缺失反而促进振荡）在机制上一致。

        实验验证（临界工作点 gain=0.18，14 seeds，纯内部延迟循环动力学）：

        ==============  ==========  ==========  ==========
        配置             delay位移    半衰期      频谱熵
        ==============  ==========  ==========  ==========
        适应性爬山        6.09        81.4        6.015
        随机扰动          6.20        36.8        6.757
        完全冻结          0.00        41.4        6.691
        ==============  ==========  ==========  ==========

        - 爬山 vs 冻结：半衰期 t=+3.49、频谱熵 t=+10.34（显著）
        - 爬山 vs 随机：t=+3.84 / +12.54（显著）
        - 随机 vs 冻结：t=-1.24 / -1.69（**不显著**）

        **效应来自方向性（适应性），不是"动起来"** —— 位移量相同
        （6.09 vs 6.20 步）但只有向对齐度更高方向的移动有效。

        由此确立两条工程约束：
        1. :meth:`probe_delay` **必须**走爬山，随机扰动无效
        2. 本方法只在**成功绑定**时调用，因此是 winner-take-all；
           失败通路的修复由 :meth:`probe_delay` 承担，两者互补

        **工作点校准是前提**：tanh + 正反馈循环天然双稳态（死亡 or
        饱和），可用带极窄。扫描显示 target_influx 从 0.6 到 1.0，
        幅度从 0.115 跳到 0.974 —— 中间无平滑过渡。任何验证必须先
        扫描工作点，否则会测到"两边都死"或"两边都饱和"的假阴性。
        """
        target_delay = group_arrival - base_time
        self.delay += lr * (target_delay - self.delay)
        self.delay = max(self.DELAY_MIN, min(self.DELAY_MAX, self.delay))
        # 成功绑定 → age 归零（"刚被用到"）
        self.since_co_fire = 0

    # ---------- age 推进与 delay 探索（协同 A）----------

    def tick_age(self) -> None:
        """每步推进未使用计数与总年龄"""
        self.since_co_fire += 1
        self.age_steps += 1

    def in_grace_period(self) -> bool:
        """是否处于新生儿豁免期（期间不参与死亡判据）"""
        return self.age_steps < self.GRACE_PERIOD

    def needs_delay_probe(self) -> bool:
        """是否该尝试一次 delay 探索

        长期未成功共发射，但还没到该剪的地步 → 先给它一次改时序的机会。
        """
        return (
            self.since_co_fire >= self.AGE_PROBE
            and self.delay_probes < self.MAX_DELAY_PROBES
        )

    def should_die_by_age(self) -> bool:
        """多次 delay 探索仍无法成功绑定 → 硬剪

        **能力边界（实验否定后的修正，务必不要夸大本方法的作用）**：

        本方法**只能**清理**完全静默**的通路——信道断流、传感器拔掉、
        模态不再出现、或多次 delay 探索后仍无法建立绑定的死连接。

        它**不能**杀死有随机贡献的噪声通路。实测（5 seeds，3000 步）：

        ================  ==========  ==============
        配置               杀死噪声     噪声平均存活
        ================  ==========  ==============
        仅 EMA 衰减        5/5          1547 步
        age（单步贡献为正） 0/5          3000 步
        age（显著性判据）   0/5          3000 步
        ================  ==========  ==============

        原因：噪声通路贡献 ~ N(0, σ)，即使要求"超过 1 倍标准差"，
        仍有约 8% 概率触发共发射 → 平均每 12 步重置一次 age →
        永远累积不到死亡阈值。**age 在这类通路上甚至比纯 EMA 更差**
        （EMA 至少能靠随机游走偶然跌破死亡线）。

        我最初的主张"EMA 渐近趋零但永不到零，离散 age 能硬杀噪声通路"
        经实验检验：**前半句成立，后半句不成立**。

        噪声通路的正确工具是 EMA + 相对衰减（让 gain 趋近 0），
        不是 age。两者分工必须写清楚，否则会误用。
        """
        return (
            self.since_co_fire >= self.AGE_DEATH
            or self.delay_probes >= self.MAX_DELAY_PROBES
        ) and self.protection < 0.5

    def probe_delay(
        self,
        window: float,
        align: Optional[tuple[float, float, float]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """delay 探索：在重合窗口尺度上重新采样延迟（协同 A 的核心）

        **现有 adapt_delay 的盲区**（这是本方法存在的理由）：
        它只把 delay 同步到**成功绑定**的群体到达时间。这是一个
        winner-take-all——成功的通路越来越同步，失败的通路 delay
        **永远得不到修复**，只能在原位等死。它形成了功能模块，
        却也把偶然的初始 delay 固化为永久判决。

        生物学的对应：髓鞘是可塑的，轴突传导延迟会随活动模式被**双向**
        调节（Asokan, Chhabria & Chakravarthy 2015，*BMC Neuroscience*
        16(Suppl 1):P29，用 STDP 核训练传导延迟）。

        **⚠ 引用勘误（2026-09-03）**：此处原写作「Lakatos 等 2016……已核实」，
        **作者、年份、文献类型三者皆错**，且错标为"已核实"。真实出处是
        Asokan 等 2015。该条是髓鞘 delay 机制的**唯一直达前身**。
        又：它系 CNS*2015 会议**海报摘要**，非期刊论文 —— 证据强度低于
        期刊文献，不应作为该机制正确性的强支撑（报告 §来源核验记录 C-1）。
        失败的轴突不是不能改，是没人去改它。

        **⚠ 必须使用爬山法，随机扰动无效（实验否定，务必不要退回）**

        本方法初版用随机扰动 ``delay += U(-window, window)``。三配置对照
        实验（14 seeds，临界工作点）否定了这个设计：

        ==============  ==========  ==========  ==========
        配置             delay位移    半衰期      频谱熵
        ==============  ==========  ==========  ==========
        适应性爬山        6.09        81.4        6.015
        **随机扰动**      6.20        36.8        6.757
        完全冻结          0.00        41.4        6.691
        ==============  ==========  ==========  ==========

        随机扰动 vs 冻结：半衰期 t=-1.24、频谱熵 t=-1.69 —— **不显著**
        （且方向为负，随机略差于不动）。而爬山 vs 随机：t=+3.84 / +12.54，
        显著。

        即：**位移量相同（6.09 vs 6.20 步）但只有带方向的移动有效。**
        效应来自"向对齐度更高的方向移动"这一适应性，不是来自"动起来"。

        **⚠ 也不要"改进"成抛物线插值或加平滑（后续实验同样否定）**

        爬山确立后又测了两种看似更高级的方案，均**不优于**当前实现
        （无预算，10 seeds，临界工作点 gain=0.18）：

        ==================  ==========  ==============  ================
        配置                 频谱熵      vs 爬山+瞬时     判定
        ==================  ==========  ==============  ================
        **爬山 + 瞬时**      **5.831**    —               **当前实现**
        抛物线 + 瞬时        5.971      t=-1.36          不显著，点估计更差
        爬山 + EMA 0.9       6.414      t=-6.04          显著更差
        抛物线 + EMA 0.9     6.451      t=-6.85          显著更差
        ==================  ==========  ==============  ================

        两点教训：

        1. **抛物线插值无效**：用三点拟合二次曲线直接跳到估计极值，
           对噪声过于敏感，不如只比较大小关系的爬山鲁棒。且 delay
           最终被 round 成整数索引，小数微调会被量化掉。
        2. **平滑对齐度反而有害**（最反直觉的一条）：对齐度虽有噪声，
           但它是对**当前** delay 配置的实时反馈；而 delay 自身在变，
           环境非平稳。EMA 混合了旧配置下的相关性信息 → 决策滞后。
           **在非平稳环境下，追新比降噪更重要。**

        → 维持「离散爬山 + 瞬时对齐度」。不要平滑，不要插值。

        Args:
            window: 重合窗口（= delay 分布的特征尺度，见 suggest_window）
            align: ``(align_minus, align_now, align_plus)`` —— 延迟分别为
                ``delay-window`` / ``delay`` / ``delay+window`` 时的对齐度。
                提供则执行**爬山**（向对齐度最高的方向移动）；
                为 None 时退化为随机扰动（**仅作兜底，已知无效**）。
            rng: 随机源（仅 align 为 None 时使用）
        """
        if align is not None:
            a_minus, a_now, a_plus = align
            if a_plus > a_now and a_plus >= a_minus:
                self.delay += window
            elif a_minus > a_now and a_minus > a_plus:
                self.delay -= window
            # 三者相等或 now 最优 → 已在局部最优，不动
        elif rng is not None:
            self.delay += rng.uniform(-window, window)
        else:
            self.delay += random.uniform(-window, window)
        self.delay = max(
            self.DELAY_MIN, min(self.DELAY_MAX, self.delay)
        )
        self.delay_probes += 1
        # 归零：给它一个完整的观察期来判断这次探索是否成功
        self.since_co_fire = 0


# ---------------------------------------------------------------------------
# 共发射检测：成边规则
# ---------------------------------------------------------------------------

@dataclass
class ActivationRecord:
    """激活记录：神经元某维度被激活时记录

    用于**共发射检测**——在重合窗口内同时发射的神经元对 → 候选连接。
    与旧实现的差别：旧实现按 ``source_tag`` 相等（因果追溯）配対，
    新实现按**到达时间接近**（共发射）配対，``source_tag`` 退化为诊断信息。
    """
    neuron: int
    channel: str       # 被激活的信道
    source_tag: str    # 激活源标识（诊断用，不再作为成边判据）
    timestamp: float   # 发射时刻（逻辑时钟，单调递增）
    arrival_time: float = 0.0  # 若经髓鞘传输，为到达时刻；外部注入时为发射时刻


class CoincidenceDetector:
    """共发射检测器：按时间窗判定"同时发射"，建立算子连接

    成边规则（对应"同时发射的神经元相连接"）：
    - 两个神经元的激活时刻差 < 重合窗口 → 候选连接
    - **不限信道**：不同信道之间也可成边 → 超模态混合（跨模态绑定）
    - 反复共发射 → 髓鞘化加固（由 registry 增益）

    时钟：内部维护单调逻辑时钟，避免旧实现中调用方传 ``t=0.0`` 恒定值
    导致 ``retention`` 清理失效、记录无限累积的问题。
    """

    def __init__(self, retention: float = 1.0):
        self.recent: list[ActivationRecord] = []
        self.retention = retention
        self._clock: float = 0.0

    @property
    def clock(self) -> float:
        """当前逻辑时钟（单调）"""
        return self._clock

    def tick(self, dt: float = 1.0) -> None:
        """推进逻辑时钟一步——应在每个系统步调用一次

        时间基准由检测器自己维护，不再依赖调用方传入真实时间戳。
        """
        self._clock += dt

    def record(
        self,
        neuron: int,
        channel: str,
        source_tag: str,
        t: Optional[float] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """记录一次激活，并清理超出保留窗口的旧记录

        Args:
            neuron: 神经元 idx
            channel: 被激活的信道
            source_tag: 激活源标识（诊断用）
            t: 发射时刻。None → 使用内部逻辑时钟
            arrival_time: 经髓鞘传输后的到达时刻。None → 等于发射时刻
        """
        ts = self._clock if t is None else t
        arr = ts if arrival_time is None else arrival_time
        self.recent.append(
            ActivationRecord(neuron, channel, source_tag, ts, arr)
        )
        # 按到达时刻清理（保留窗口以到达时刻为准）
        self.recent = [
            r for r in self.recent if arr - r.arrival_time < self.retention
        ]

    def find_coincident_pairs(
        self,
        window: Optional[float] = None,
        allow_cross_channel: bool = True,
    ) -> list[tuple[ActivationRecord, ActivationRecord]]:
        """找出共发射的激活对 → 候选算子连接

        判据是**到达时刻接近**（共发射），不是 source_tag 相同。

        Args:
            window: 重合窗口。None → 使用 ``retention``
            allow_cross_channel: 是否允许跨信道成边（超模态混合）。
                False 时退化为旧行为（仅同信道）。

        Returns:
            共发射的激活对列表（不同神经元）。
        """
        w = self.retention if window is None else window
        pairs: list[tuple[ActivationRecord, ActivationRecord]] = []
        # 按到达时刻排序，便于滑动窗口
        ordered = sorted(self.recent, key=lambda r: r.arrival_time)
        n = len(ordered)
        for i in range(n):
            ai = ordered[i]
            for j in range(i + 1, n):
                aj = ordered[j]
                # 已排序：超出窗口即可中断内层
                if aj.arrival_time - ai.arrival_time > w:
                    break
                # ⚠ 原判据 ``ai.neuron == aj.neuron → continue`` 造成
                # **冷启动死锁**，实测（真实系统，300 步）确认：
                #   只有 1 个神经元 → 全部记录同属它 → 永远配不上对
                #   → 永不成边 → 无髓鞘 → 无稳定输出反应 → 不分化
                #   → 仍然只有 1 个神经元 …… 死循环。
                # 实测表现：sheaths=0, neurons=1, 系统纯靠反射，什么都没学。
                #
                # 修正为：只排除**同一神经元的同一信道**（自环无意义），
                # 允许同一神经元的不同信道配对 —— 这正是"一个神经元处理
                # 所有信号、超模态混合"的起始形态：跨模态绑定首先发生在
                # 单神经元内部的信道之间，之后分化出多神经元才扩展到
                # 神经元之间。
                if ai.neuron == aj.neuron and ai.channel == aj.channel:
                    continue
                if not allow_cross_channel and ai.channel != aj.channel:
                    continue
                pairs.append((ai, aj))
        return pairs


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

# ============================================================================
# 分层澄清（2026-09，设计更正）
# ----------------------------------------------------------------------------
# **共发射相连 = 神经元之间；髓鞘是另一回事。**
#
# 此前把两层合并了：``co_fire_wire`` 直接 ``add_sheath``，等于"建立连接"
# 和"给连接裹绝缘皮"是同一件事。后果是髓鞘被迫承担连接的生灭，变得
# 朝生暮死（实测周转 10194、髓鞘数 3→15→45→122→14 剧烈振荡）。
#
# 正确分层：
#
#   ============  ===================================================
#   层次           内容
#   ============  ===================================================
#   连接           轴突 A→B，共发射成边。**可以生灭**（长期不用才淘汰）
#   髓鞘           包裹在已存在的连接上。**后加、可选、稳定**的增强层
#   ============  ===================================================
#
# 关键推论：未髓鞘的轴突**可以工作**，只是慢（delay 6.0）且弱（gain 0.3）。
# 髓鞘化是性能升级，不是连接存在的前提。
# ============================================================================


@dataclass
class Connection:
    """神经元之间的**轴突连接** —— 共发射成边的产物

    这一层与髓鞘是两回事，此前被错误合并了。

    生物学分层：

    ============  ====================================================
    层次           内容
    ============  ====================================================
    **连接**       轴突从 A 长到 B —— 拓扑问题（"连到哪儿"）。
                  发育期轴突寻路 + 活动依赖的共发射成边。**可以生灭**。
    **髓鞘**       包裹在**已存在**的轴突上 —— 性能问题
                  （"裹不裹绝缘皮"）。后加的、可选的、稳定的增强层。
    ============  ====================================================

    一个少突胶质细胞可包裹**多段**轴突（多个 internode），因此髓鞘
    不是"一条连接一个"的东西。

    未髓鞘的轴突**可以存在且工作**，只是传导慢、信号弱。髓鞘化是后续
    的性能升级，不是连接存在的前提。
    """
    src_neuron: int
    src_channel: str
    dst_neuron: int
    dst_channel: str
    strength: float = 0.2      # 连接强度（用进废退作用于此）
    usage: float = 0.0         # 使用累积 EMA
    co_fire_count: int = 0     # 累计共发射次数（采样确认的计数）
    starving: int = 0          # 连续未被使用的步数
    myelinated: bool = False   # 是否已被髓鞘包裹
    birth_step: int = 0

    # 未髓鞘轴突的传导特性：慢且弱
    UNMYELINATED_DELAY = 6.0
    UNMYELINATED_GAIN = 0.3

    def effective_delay(self, sheath=None) -> float:
        """有效传导延迟：有髓鞘 → 跳跃式传导（快）；无 → 慢"""
        return sheath.delay if sheath is not None else self.UNMYELINATED_DELAY

    def effective_gain(self, sheath=None) -> float:
        """有效增益：有髓鞘 → 强；无 → 弱"""
        return sheath.gain if sheath is not None else self.UNMYELINATED_GAIN


class MyelinSheathRegistry:
    """髓鞘包裹层注册表

    键为 ``(src_neuron, src_channel, dst_neuron, dst_channel)``，值为
    :class:`MyelinSheath`。除增删查外，承载三类群体操作：

    - :meth:`apply_global_decay` — 选择动力学（用进废退的"废"侧）
    - :meth:`co_fire_wire` — 共发射成边（超模态混合）
    - :meth:`capacity_report` — 容量诊断（监控自适应预算是否发散）
    """

    def __init__(self, decay_rate: float = 0.01, self_proof: float = 0.0):
        self._sheaths: dict[tuple[int, str, int, str], MyelinSheath] = {}
        self._connections = {}
        self.decay_rate = decay_rate
        # 成边门控：源通道需先自证可信，才允许向其他通道出边
        self.self_proof = self_proof
        # 冷启动豁免：髓鞘数少于此值时跳过 self_proof 门控。
        # 0 = 永不豁免（会死锁，除非 system 自己先建立同信道边）。
        # 生物学：发育早期过量生成 → 后期修剪。
        # 双水位滞回：低于 low 开启豁免，高于 high 关闭豁免
        self.exuberant_low = 2
        self.exuberant_high = 10
        self._exuberant_on = True
        # 通道级可信度：{(neuron, channel): EMA(|贡献|)}
        self._channel_cred: dict[tuple[int, str], float] = {}
        self._step: int = 0
        # 诊断：累计建立 / 淘汰计数
        self._born: int = 0
        self._died: int = 0
        # --- 睡眠归因账本（群体级）---
        # {相位名: {"cycles", "touched", "gain_delta", "stab_delta",
        #           "born", "died"}}，由 attribute_sleep_phase 逐相位累积。
        self.sleep_ledger: dict[str, dict[str, float]] = {}

        # --- 承载力：代谢预算（协同 B）---
        #
        # 出处：Baxter & Levy 2019（PMID 31675628，摘要已逐条核实）——
        # 自适应突触发生网络的承载力是**发放率截断**：突触后神经元的总
        # 输入超过设计发放率时，对新突触的**接收性降为 0**。这不是外部
        # 硬约束（``max_nodes``），而是系统自身的代谢信号。
        #
        # 与 protection 的耦合（本设计的增量，1+1>2）：
        # 保护的通路**占用更多预算**（权重 1+λ·protection）。于是
        # "护住旧技能"有了明确代价——系统必须在维持旧技能与学习新东西
        # 之间分配同一个预算。
        #
        # 这同时解决了两个已知问题：
        # ①protection 无界累积（现在受预算约束）
        # ②"癌细胞前体"式无成本扩张（增长现在真实消耗预算）
        self.target_influx: float = 8.0   # 每个目标神经元的设计总输入（发放率）
        self.protection_cost: float = 1.0 # λ：protection 的预算权重
        # 预算自适应：目标容量随环境预测难度伸缩（协同 E）
        # 髓鞘半衰期（步）。0 = 不衰减（理想稳定）。
        # 生物学：髓鞘蛋白半衰期 55 天 ~ 6 个月，是极长的时间尺度，
        # 与"系统步"完全不是同一量级。故默认 0。
        self.myelin_half_life: float = 0.0
        # 连接被判定为"长期无共发射"的步数上限（超过则删除连接）
        self.conn_starve_limit: int = 3000
        # ------------------------------------------------------------------
        # **两级确认**（此前的两个阈值功能重叠，现合并为清晰的级联）
        #
        #   共发射 × wire_threshold        → 建立**连接**（拓扑）
        #   连接再共发射至 myelination_threshold → 被**髓鞘包裹**（性能）
        #
        # 一级（wire）管"该不该连"，二级（myelination）管"值不值得加速"。
        # 两者都需要重复验证，但作用对象不同、不可互换。
        #
        # 生物学：轴突先长到靶点并建立突触（可生灭、可修剪），之后
        # 少突胶质细胞选择性髓鞘化其中一部分。髓鞘化发生在连接之后。
        # ------------------------------------------------------------------
        self.wire_threshold: int = 1            # 成边所需共发射次数
        self.myelination_threshold: int = 3     # 髓鞘化所需累计共发射次数
        #
        # 实测注记（2500 步）：把生长闸门前置到路径选择之前后
        # （``development.maybe_differentiate``），结构被闸门钉死，
        # 两级阈值的差异**被吸收**—— 各配置收敛到同一结构
        # （连接 93 / 髓鞘 48 / 周转 405 / 神经元 12）。
        # 即：闸门是主导因素，两级阈值退化为次要。
        # 这本身是好消息：结构终于稳定了（此前周转 10194 → 405）。
        self._candidates: dict[tuple[int, str, int, str], int] = {}
        self.adaptive_budget: bool = False
        self._ar_error_ema: float = 0.0

    # ---------- 承载力（协同 B / E）----------

    def influx_of(self, dst_neuron: int, dst_channel: str) -> float:
        """某目标 (神经元, 信道) 当前的总输入（按保护加权）"""
        total = 0.0
        for (sn, sc, dn, dc), s in self._sheaths.items():
            if dn == dst_neuron and dc == dst_channel:
                total += s.gain * (1.0 + self.protection_cost * s.protection)
        return total

    def budget_of(self) -> float:
        """当前预算上限

        协同 E：若启用自适应，预算随自回归预测误差伸缩——
        预测不准 → 需要更多容量去解释；预测准 → 容量需求下降。
        这让承载力从魔数变成从环境反馈导出的量。
        """
        if not self.adaptive_budget:
            return self.target_influx
        # 误差越大 → 预算越大（上界 3 倍，防爆炸）
        return self.target_influx * min(3.0, 1.0 + 2.0 * self._ar_error_ema)

    def update_budget_signal(self, ar_error: float, momentum: float = 0.99) -> None:
        """用自回归误差更新预算信号（协同 E）"""
        self._ar_error_ema = (
            momentum * self._ar_error_ema + (1.0 - momentum) * ar_error
        )

    def receptive(self, dst_neuron: int, dst_channel: str) -> bool:
        """目标是否还接收新连接

        ⚠ 本方法**曾是竞争机制的载体**，现已移除竞争。

        旧实现 ``influx < budget``：目标神经元的总输入超过预算就拒绝
        新连接 —— 这意味着两条髓鞘在争夺同一份"发放率配额"，即竞争。
        设计意图明确是"**所有髓鞘都独立**"，不存在这种争夺；生物学上
        新髓鞘的加入是**增加**总量，不挤占既有髓鞘。

        现仅做**结构性**检查（目标是否存在），不做任何预算判断。
        承载力相关的诊断保留在 :meth:`capacity_report`，供观察而非限制。
        """
        return True

    # ---------- 基础增删查 ----------

    def add_sheath(
        self,
        src_neuron: int,
        src_channel: str,
        dst_neuron: int,
        dst_channel: str,
        delay: float = 1.0,
        gain: float = 1.0,
        protection: float = 0.0,
    ) -> MyelinSheath:
        """新增一条髓鞘包裹层（若已存在则覆盖）

        已存在时**保留**既有的 usage/stability——使用记录是获得性资产，
        不应被一次 add 重置（旧实现在此处丢失了全部使用历史）。
        """
        key = (src_neuron, src_channel, dst_neuron, dst_channel)
        is_new = key not in self._sheaths
        prior = self._sheaths.get(key)
        sheath = MyelinSheath(
            src_neuron=src_neuron,
            src_channel=src_channel,
            dst_neuron=dst_neuron,
            dst_channel=dst_channel,
            delay=delay,
            gain=gain,
            protection=protection,
            usage=prior.usage if prior else 0.0,
            stability=prior.stability if prior else 0.0,
            birth_step=self._step,
        )
        self._sheaths[key] = sheath
        if is_new:
            self._born += 1
        return sheath

    def get_sheath(
        self,
        src_neuron: int,
        src_channel: str,
        dst_neuron: int,
        dst_channel: str,
    ) -> Optional[MyelinSheath]:
        """读取单条髓鞘，不存在返回 None"""
        return self._sheaths.get((src_neuron, src_channel, dst_neuron, dst_channel))

    def get_sheaths_from(self, neuron: int, channel: str) -> list[MyelinSheath]:
        """查询从某神经元某信道出发的所有髓鞘（出边）"""
        return [
            s for s in self._sheaths.values()
            if s.src_neuron == neuron and s.src_channel == channel
        ]

    def remove_sheaths_for(self, neuron: int) -> int:
        """删除涉及某神经元（作为源或目标）的所有髓鞘。返回删除条数。"""
        keys_to_remove = [
            k for k, s in self._sheaths.items()
            if s.src_neuron == neuron or s.dst_neuron == neuron
        ]
        for k in keys_to_remove:
            del self._sheaths[k]
        self._died += len(keys_to_remove)
        return len(keys_to_remove)

    # ---------- 稳态缩放 ----------

    def homeostatic_scale(self, beta: float = 0.1) -> None:
        """稳态缩放（突触稳态假说，sleep NREM S3 相位）

        按保护系数加权衰减 gain：``gain' = gain · (1 - beta · (1 - protection))``

        与 :meth:`apply_global_decay` 的区别：本方法是**乘性**缩放（保持相对
        强弱、重置学习容量），选择动力学是**加性**衰减 + 贡献增厚（改变相对
        强弱、执行淘汰）。两者不可互相替代。
        """
        for sheath in self._sheaths.values():
            decay_rate = beta * (1.0 - sheath.protection)
            sheath.gain = sheath.gain * (1.0 - decay_rate)

    # ---------- 路径级操作 ----------

    def thicken_sheath(
        self,
        src_neuron: int,
        src_channel: str,
        dst_neuron: int,
        dst_channel: str,
        amount: float = 0.05,
    ) -> bool:
        """路径级操作：增厚指定髓鞘。返回是否成功命中。"""
        sheath = self.get_sheath(src_neuron, src_channel, dst_neuron, dst_channel)
        if sheath is None:
            return False
        sheath.thicken(amount)
        return True

    def decay_sheath(
        self,
        src_neuron: int,
        src_channel: str,
        dst_neuron: int,
        dst_channel: str,
        amount: float = 0.05,
    ) -> bool:
        """路径级操作：衰减指定髓鞘。返回是否成功命中。"""
        sheath = self.get_sheath(src_neuron, src_channel, dst_neuron, dst_channel)
        if sheath is None:
            return False
        sheath.decay(amount)
        return True

    # ---------- 选择动力学（用进废退） ----------

    def apply_global_decay(
        self,
        contributions: Optional[dict[tuple[int, str, int, str], float]] = None,
        use_scale: float = 1.0,
        decay_rate: Optional[float] = None,
    ) -> int:
        """**各髓鞘独立**的可塑性结算 —— 不是竞争

        ⚠ 方法名保留（调用点众多），但**语义已改变**，请勿按旧名理解。

        ------------------------------------------------------------------
        旧语义（已废弃）：全局选择动力学 —— 每步对所有髓鞘施加衰减，
        按贡献增厚，gain 跌破阈值即淘汰。这是"物竞天择，谁用的最多谁
        就留存"。**其中"竞争"是误解，不是设计意图。**
        ------------------------------------------------------------------

        **为什么废弃**（设计澄清 + 生物学事实双重否定）：

        1. 设计意图是"**所有髓鞘都独立**"——不存在两条髓鞘争夺同一份
           资源。竞争是我自行引入的。
        2. 生物学上髓鞘**不竞争**：成人新生的少突胶质细胞是"增加"总
           髓鞘量，而非替换发育期形成的旧髓鞘。
        3. 实测后果：周转率 800+，结构反复生灭，学习成果被反复清空。

        **新语义**：遍历所有髓鞘，各自独立地
        "用则增厚、不用则停滞"。彼此不共享预算、不竞争淘汰。

        Args:
            contributions: ``{sheath_key: contribution}``，contribution ∈ [-1, 1]。
                **未提供 ≠ 受惩罚**，只是"本步没增厚"。
            use_scale: 贡献到增量的缩放
            decay_rate: **已废弃**，保留仅为兼容旧调用点，不再生效。
                真正的退化尺度由 ``myelin_half_life`` 控制（默认 0 = 不衰减）。

        Returns:
            int: 本步被移除的髓鞘条数（正常为 0；仅在启用半衰期且
                长期饥饿时才 > 0）。
        """
        rho = self.decay_rate if decay_rate is None else decay_rate
        contrib = contributions or {}
        self._step += 1

        dead: list[tuple[int, str, int, str]] = []
        for key, sheath in self._sheaths.items():
            c = contrib.get(key, 0.0)
            # 更新源通道可信度（EMA of |贡献|）——供 self_proof 成边门控使用
            ck = (sheath.src_neuron, sheath.src_channel)
            # ⚠ 原写法 ``if prev else abs(c)`` 把 prev==0.0 当成"未初始化"，
            # 导致贡献为 0 的通道每步都被重置为 abs(c)，EMA 永不累积。
            # 改为显式判断键是否存在。
            if ck in self._channel_cred:
                self._channel_cred[ck] = (
                    0.9 * self._channel_cred[ck] + 0.1 * abs(c))
            else:
                self._channel_cred[ck] = abs(c)
            alive = sheath.apply_independent_plasticity(
                contribution=c,
                use_scale=use_scale,
                half_life=self.myelin_half_life,
            )
            # 新生儿豁免期：期间只累积增厚，不执行死亡判据。
            # 无此豁免时 gain=0 的新边当步就死 → 下一步重建 → 空转。
            if not alive and not sheath.in_grace_period():
                dead.append(key)

        for key in dead:
            del self._sheaths[key]
        self._died += len(dead)
        return len(dead)

    def channel_credibility(self, neuron: int, channel: str) -> float:
        """查询通道级可信度（该通道出边的历史贡献 EMA）"""
        return self._channel_cred.get((neuron, channel), 0.0)

    def co_fire_wire(
        self,
        pairs: list[tuple[ActivationRecord, ActivationRecord]],
        initial_gain: float = 0.2,
    ) -> int:
        """共发射成边：窗口内同时发射的神经元对 → 建立/加固髓鞘

        **超模态混合**：允许跨信道成边（不同信道之间也建立连接）。
        这是"对外界统一编码"的结构基础——不同模态的信号在同一神经元层面
        被绑定到一起。

        **新通路的初始增益（``initial_gain=0.2``）**：

        ⚠ 原实现用 ``initial_gain=0.0``（"沉默突触"），理由是防止新边立即
        污染输出。但那是**治标**：真正的问题是当时的每步衰减淘汰机制
        （已移除）。用"让新结构失效"去规避"新结构被误杀"，代价是

        - 新边必须先在死亡线上挣扎若干步才有机会被验证；
        - 配合当时的衰减机制，形成上万次"新建—淘汰"空转。

        生物学上，新形成的髓鞘**是有效的**：一旦有髓鞘包裹，轴突即获得
        跳跃式传导能力，只是层数少、增益低。它不是沉默的。

        故改为小的正值（0.2）：新通路**立即可用但较弱**，其去留由后续
        使用决定 —— 用则增厚，不用则停滞。

        已存在的连接走 :meth:`thicken_sheath` 加固。

        **先自证，再连接**（``self_proof > 0`` 时启用）：
        向其他通道出边前，来源通道必须已积累足够的可信度。

        无差别成边的实测问题：噪声通道的通路增益是零均值随机游走，衰减只能
        拉回 0 附近却杀不死（波动幅度 ∝ σ/√ρ，实测可达 0.2）。结果 41 条
        有效通路里 29 条涉及噪声维，跨模态成边从"有益"退化为"有害"。

        启用门控后（跨模态必要的环境下）：

        ================  ==========  ==========
        配置               代价        改进
        ================  ==========  ==========
        仅对角             0.1413      63.4%
        跨模态, 无门控      0.1469      63.9%（被噪声淹没）
        跨模态, 门控 0.5    0.0862      77.6%
        ================  ==========  ==========

        Returns:
            int: 本步新建的连接数（不含加固的既有连接）。
        """
        created = 0
        for A, B in pairs:
            key = (A.neuron, A.channel, B.neuron, B.channel)

            # --- ① 连接层：共发射 → 建立/加强神经元之间的连接 ---
            #
            # ⚠ 这一步此前被错误地写成"直接建立髓鞘"，把两层混为一谈。
            # 共发射相连说的是**神经元之间**，髓鞘是包裹在连接上的另一回事。
            conn = self._connections.get(key)
            if conn is not None:
                conn.co_fire_count += 1
                conn.usage = 0.9 * conn.usage + 0.1 * 1.0
                conn.starving = 0
                conn.strength = min(1.0, conn.strength + 0.02)
                sh = self._sheaths.get(key)
                if sh is not None:
                    sh.thicken(0.02)
                    sh.since_co_fire = 0
                elif conn.co_fire_count >= self.myelination_threshold:
                    # ② 第二级：连接被反复验证 → 才被髓鞘包裹
                    self._myelinate(conn, key)
                continue

            # ① 第一级：共发射满 wire_threshold 次才建立连接
            if self.wire_threshold > 1:
                n = self._candidates.get(key, 0) + 1
                if n < self.wire_threshold:
                    self._candidates[key] = n
                    continue
                self._candidates.pop(key, None)

            # 跨通道出边需先自证（带滞回的豁免，见 _exuberant_on）
            if self.self_proof > 0.0 and A.channel != B.channel:
                n = len(self._sheaths)
                if n >= self.exuberant_high:
                    self._exuberant_on = False
                elif n <= self.exuberant_low:
                    self._exuberant_on = True
                if not self._exuberant_on:
                    if self.channel_credibility(
                            A.neuron, A.channel) < self.self_proof:
                        continue

            self._connections[key] = Connection(
                src_neuron=A.neuron, src_channel=A.channel,
                dst_neuron=B.neuron, dst_channel=B.channel,
                strength=0.2,
                co_fire_count=max(1, self.wire_threshold),
                birth_step=self._step,
            )
            if self.myelination_threshold <= max(1, self.wire_threshold):
                self._myelinate(self._connections[key], key)
            self._born += 1
            created += 1

        return created


    # ---------- age 驱动的维护（协同 A / D）----------

    def _prune_starved_connections(self) -> int:
        """清理**长期无共发射的连接**（连带其髓鞘）

        这是"用进废退"在**连接层**的落点，与髓鞘层无关。

        - 连接：长期不被共发射 → 淘汰（时间尺度长，见 conn_starve_limit）
        - 髓鞘：一旦形成即稳定，**不因"最近没用"而消失**

        生物学：突触修剪确有其事（青春期大规模发生），但对象是未被
        稳定使用的**突触/连接**；已形成的髓鞘不是这个尺度上的东西。

        Returns:
            被删除的连接数。
        """
        dead = [k for k, c in self._connections.items()
                if c.starving >= self.conn_starve_limit]
        for k in dead:
            del self._connections[k]
            self._sheaths.pop(k, None)
            self._died += 1
        return len(dead)

    def tick_ages(self) -> None:
        """每步推进未使用计数（髓鞘 + 连接）"""
        for s in self._sheaths.values():
            s.tick_age()
        for c in self._connections.values():
            c.starving += 1

    def maintain_ages(
        self,
        align_fn: Optional[object] = None,
        rng: Optional[random.Random] = None,
    ) -> dict[str, int]:
        """age 驱动的结构维护：先修复（delay 探索），后删除

        **这是"先修复后删除"的双向时间轴搜索**（协同 A 的执行点）。
        应成 block 调用（见 :meth:`should_run_wiring`），与清醒期的
        快速修饰处于不同时间尺度。

        Args:
            align_fn: 可选的可调用对象 ``align_fn(sheath, key) -> (m, now, p)``，
                返回该髓鞘在 delay 分别为 ``d-w`` / ``d`` / ``d+w`` 时的
                对齐度（由 :meth:`SignalDispatcher.make_align_fn` 生成）。
                **必须提供** —— 无对齐度时 delay 探索退化为随机扰动，
                而随机扰动已被实验证明与不探索无差异
                （见 :meth:`MyelinSheath.probe_delay` 的数据表）。
                返回 None 时同样退回随机，但会计入
                :attr:`SignalDispatcher.align_stats` 供诊断。
            rng: 随机源（仅 align_fn 缺失或返回 None 时使用）

        Returns:
            {"probed": 触发 delay 探索的条数, "pruned": 被硬剪的条数}
        """
        window = self.suggest_window()
        probed = pruned = 0
        for key in list(self._sheaths.keys()):
            s = self._sheaths[key]
            # 豁免期内的新通路既不探索也不剪——给它完整机会被增厚
            if s.in_grace_period():
                continue
            if s.needs_delay_probe():
                align = align_fn(s, key) if callable(align_fn) else None
                s.probe_delay(window, align=align, rng=rng)
                probed += 1
            # ⚠ 原实现此处有 ``elif s.should_die_by_age(): del self._sheaths[key]``
            # —— 直接删除**髓鞘**。按分层这是错误的：髓鞘一旦形成极稳定
            # （蛋白半衰期 55 天 ~ 6 个月），不该被 age 计数剪除。
            # 实测后果：髓鞘数剧烈振荡 3→15→45→122→14，周转 10194。
            #
            # 该淘汰的是**连接**（长期无共发射），不是裹在它上面的髓鞘。
            # 连接删除时其髓鞘自然一并消失。
        pruned = self._prune_starved_connections()
        return {"probed": probed, "pruned": pruned}

    def should_run_wiring(self, block: int = 50) -> bool:
        """结构更新（成边 / 脱落）是否该在本步运行 —— 时间尺度分离

        **协同 D：这是"有没有必要做梦"这个问题的技术答案。**

        出处：Baxter & Levy 2019（PMID 31675628）明确区分三个时间尺度，
        由短到长：①输入呈现（每步）②突触修饰（每次输入后）③突触发生
        与脱落（每 block，比前两者慢一个量级）。

        我们之前的周转率空转（实测 turnover 11.5，上万次"新建—淘汰"
        循环）根因就在这里：成边每步都跑，与选择动力学**处在同一时间
        尺度**——成边必然赢过选择，结构永远无法稳定。

        这不是调参问题，是架构缺了尺度分离。结构更新必须比权重修饰慢
        一个量级，否则系统永远在追自己的尾巴。

        与做梦的耦合（1+1>2）：把结构更新主要放到**睡眠期**，做梦就从
        "可选的性能增强"变成"结构重整的必要场所"——清醒期在线学习
        无法做结构更新（会破坏正在使用的表征），只有离线期能做。
        这为四相位做梦提供了独立于"记忆巩固"的第二重必要性论证。
        """
        return block > 0 and self._step % block == 0

    def suggest_window(self, k: float = 1.0) -> float:
        """由 delay 分布导出重合窗口，替代硬编码常量

        窗口必须随 delay 的分布伸缩，否则"同时发射"的判据与传导延迟脱节，
        delay 作为时间轴算子就失去了对成边的影响力。
        """
        delays = [s.delay for s in self._sheaths.values()]
        if len(delays) < 2:
            return 1.0
        mean = sum(delays) / len(delays)
        var = sum((d - mean) ** 2 for d in delays) / len(delays)
        return max(1e-3, k * math.sqrt(var))

    # ---------- 诊断 ----------

    def _myelinate(self, conn: "Connection", key) -> bool:
        """给**已充分验证的连接**裹上髓鞘

        髓鞘是后来的加固，不是连接本身。判据：累计共发射次数达到
        ``myelination_threshold``。

        生物学：少突胶质细胞前体先采样多个轴突，再选择性髓鞘化其中
        一部分。髓鞘形成在连接**之后**，一旦形成极稳定（蛋白半衰期
        55 天 ~ 6 个月）。
        """
        if conn.myelinated or key in self._sheaths:
            return False
        self._sheaths[key] = MyelinSheath(
            src_neuron=conn.src_neuron, src_channel=conn.src_channel,
            dst_neuron=conn.dst_neuron, dst_channel=conn.dst_channel,
            delay=MyelinSheath.DELAY_MIN + 1.0,
            gain=0.5, birth_step=self._step,
        )
        conn.myelinated = True
        return True

    # ---------- 睡眠归因账本 ----------

    def snapshot_sheaths(self) -> dict[tuple[int, str, int, str], tuple[float, float]]:
        """记录当前所有髓鞘的 (gain, stability) 快照（睡眠相位前调用）"""
        return {
            key: (s.gain, s.stability)
            for key, s in self._sheaths.items()
        }

    def attribute_sleep_phase(
        self,
        phase: str,
        before: dict[tuple[int, str, int, str], tuple[float, float]],
    ) -> dict[str, float]:
        """把一个睡眠相位造成的增量归因到各髓鞘与群体账本

        相位结束后调用。对相位期间存活的髓鞘：逐条累计
        ``sleep_gain_delta`` / ``sleep_stability_delta``（及分相位
        ``sleep_phase_gain``）；相位期间新建/淘汰的髓鞘单独计数。
        群体级累积进 :attr:`sleep_ledger`。

        Returns:
            dict: 本相位的归因摘要（touched / gain_delta / stab_delta /
                  born / died）。
        """
        touched = 0
        gain_sum = 0.0
        stab_sum = 0.0
        born = 0
        for key, s in self._sheaths.items():
            if key not in before:
                # 相位期间新建的髓鞘：其全部增益归因于本相位
                born += 1
                s.sleep_gain_delta += s.gain
                s.sleep_phase_gain[phase] = (
                    s.sleep_phase_gain.get(phase, 0.0) + s.gain
                )
                continue
            g0, st0 = before[key]
            dg = s.gain - g0
            dst = s.stability - st0
            if dg != 0.0:
                touched += 1
            s.sleep_gain_delta += dg
            s.sleep_stability_delta += dst
            if dg != 0.0:
                s.sleep_phase_gain[phase] = (
                    s.sleep_phase_gain.get(phase, 0.0) + dg
                )
            gain_sum += dg
            stab_sum += dst
        # 相位期间消失的髓鞘数：N_now = N_before − died + born
        died = max(0, len(before) + born - len(self._sheaths))
        entry = self.sleep_ledger.setdefault(phase, {
            "cycles": 0.0, "touched": 0.0, "gain_delta": 0.0,
            "stab_delta": 0.0, "born": 0.0, "died": 0.0,
        })
        entry["cycles"] += 1
        entry["touched"] += touched
        entry["gain_delta"] += gain_sum
        entry["stab_delta"] += stab_sum
        entry["born"] += born
        entry["died"] += died
        return {
            "touched": touched,
            "gain_delta": gain_sum,
            "stab_delta": stab_sum,
            "born": born,
            "died": died,
        }

    def sleep_report(self) -> dict:
        """睡眠归因账本的诊断快照（群体级 + 髓鞘级汇总）"""
        sheaths = list(self._sheaths.values())
        n = len(sheaths)
        affected = sum(
            1 for s in sheaths
            if abs(s.sleep_gain_delta) > 1e-9
            or abs(s.sleep_stability_delta) > 1e-9
        )
        return {
            "phases": {
                p: dict(v) for p, v in self.sleep_ledger.items()
            },
            "n_sheaths": n,
            "n_sleep_affected": affected,
            "sleep_gain_mean": (
                sum(s.sleep_gain_delta for s in sheaths) / n if n else 0.0
            ),
            "sleep_stab_mean": (
                sum(s.sleep_stability_delta for s in sheaths) / n if n else 0.0
            ),
        }

    # ---------- e-prop 资格迹（T3） ----------

    def update_eligibility(self, contributions: dict, decay: float = 0.9,
                           clamp: float = 5.0) -> int:
        """推进资格迹：E_j ← decay·E_j + contribution_j

        Args:
            contributions: 本步 {sheath_key: 贡献}（pathway_contributions 输出）
            decay: 迹衰减（0.9 ≈ 覆盖约 10 步的传输-RPE 延迟）
            clamp: 迹幅值界（防正反馈失控——防发散教义）

        Returns:
            被更新的髓鞘数。
        """
        for s in self._sheaths.values():
            s.elig = max(-clamp, min(clamp, s.elig * decay))
        for key, c in contributions.items():
            s = self._sheaths.get(key)
            if s is None:
                continue
            s.elig = max(-clamp, min(clamp, s.elig + float(c)))
        return len(contributions)

    def apply_eprop(self, rpe: float, lr: float = 0.05) -> int:
        """RPE 到来：按资格迹分账 gain（三因子的第三因子 × 迹）

        Δgain_j = lr · RPE · E_j，钳位 [0, GAIN_MAX]。

        Returns:
            被更新的髓鞘数。
        """
        n = 0
        for s in self._sheaths.values():
            if abs(s.elig) < 1e-9 or abs(rpe) < 1e-9:
                continue
            s.gain = max(0.0, min(s.GAIN_MAX,
                                  s.gain + lr * float(rpe) * s.elig))
            n += 1
        return n

    # ---------- 阈值张力（T8） ----------

    def update_tension(self, events, residual, layout, decay: float = 0.95) -> int:
        """推进张力张量：T_j ← decay·T_j + |transmitted × residual|（逐维）

        张力是**负荷**不是方向——取模长。无事件的髓鞘纯衰减。

        Returns:
            被更新的髓鞘数。
        """
        if residual is None:
            return 0
        rn = residual.detach().cpu().numpy() if hasattr(residual, "detach") \
            else (residual.a if hasattr(residual, "a") else residual)
        rn = np.asarray(rn, dtype=float).ravel()
        # 全表先衰减（无事件的髓鞘纯衰减）
        for s in self._sheaths.values():
            if s.tension is not None:
                s.tension = s.tension * decay
        n = 0
        for e in events:
            key = e.sheath_key
            if key is None or key not in self._sheaths:
                continue
            dc = key[3]
            if layout is None or dc not in layout:
                continue
            off, size = layout[dc]
            d = e.data.detach().cpu().numpy() if hasattr(e.data, "detach") \
                else (e.data.a if hasattr(e.data, "a") else e.data)
            d = np.asarray(d, dtype=float).ravel()
            seg_d = d[off:off + size]
            seg_r = rn[off:off + size]
            if seg_d.size == 0:
                continue
            load = np.abs(seg_d * seg_r[:seg_d.size])
            s = self._sheaths[key]
            if s.tension is None or np.shape(s.tension) != load.shape:
                s.tension = np.zeros(size)
            s.tension = s.tension + load      # 全局衰减已做，此处只加负荷
            n += 1
        return n

    def tension_load(self, nid: int) -> float:
        """神经元的回路张力 = 各汇入/汇出髓鞘张量均值的和

        高值 = 承重墙（漂移受抑制、变异被回滚甄别）；
        低值 = 隔墙（探索预算自由）。
        """
        total, cnt = 0.0, 0
        for (sn, sc, dn, dc), s in self._sheaths.items():
            if sn == nid or dn == nid:
                if s.tension is not None:
                    t = s.tension
                    if hasattr(t, "detach"):
                        t = t.detach().cpu().numpy()
                    elif hasattr(t, "a"):
                        t = t.a
                    total += float(np.abs(np.asarray(t, dtype=float)).mean())
                    cnt += 1
        return total / max(1, cnt)

    def capacity_report(self) -> dict:
        """容量诊断——监控"只有衰减、无硬约束"下的自适应预算是否发散

        纯衰减时总容量上界随环境活跃度伸缩，因此**不存在**静态上界。
        ρ 是唯一的选择压力来源，且**存在最优值**（实测消融，8 维 / 64 通路上界）：

        ==========  ==========  ==========  ================================
        ρ            终态通路    有效通路     说明
        ==========  ==========  ==========  ================================
        0            42.0        27.2        接近全连接，无用通路被养活
        0.005        29.6        17.2        容量仍偏大
        0.02         18.4        7.6         **最优**：MSE 最低
        0.05         9.8         3.0         误杀弱信号通路，性能回落
        ==========  ==========  ==========  ================================

        两端都失败：ρ 太小 → 容量膨胀、低质连接堆积；ρ 太大 → 衰减压垮学习
        （衰减造成稳态偏差 ``gain* = g_opt − ρ·E‖x‖²/(lr·E[x_j²])``），
        弱但真实的通路被误杀。

        调优判据：
        - ``unused_ratio`` 持续偏高 → ρ 太小，提高
        - ``n_sheaths`` 持续下降且性能不升 → ρ 太大，降低

        Returns:
            dict: 含 n_sheaths / total_gain / mean_gain / mean_usage /
                  unused_ratio / turnover / born / died。
        """
        sheaths = list(self._sheaths.values())
        n = len(sheaths)
        if n == 0:
            return {
                "n_sheaths": 0, "total_gain": 0.0, "mean_gain": 0.0,
                "mean_usage": 0.0, "unused_ratio": 0.0,
                "born": self._born, "died": self._died, "step": self._step,
            }
        total_gain = sum(s.gain for s in sheaths)
        # turnover：累计新建/淘汰数相对当前规模的比值。
        # 显著大于 1 说明存在"新建—淘汰"空转。
        #
        # 空转的两个已知根因（均已修复，保留诊断以防回归）：
        # ①新通路初始增益过高（应为 0，见 co_fire_wire）
        # ②**缺少新生儿豁免期** —— gain=0 的新边在下一步即因 0 < 0.1 被删，
        #   下一步再被重建。这才是实测周转率 11.5 的主因。
        turnover = (self._born + self._died) / max(1, n)
        # 豁免期内的通路数：持续偏高说明成边速率远超选择能稳定的量
        in_grace = sum(1 for s in sheaths if s.in_grace_period())
        # 已触发过 delay 探索的通路数
        probed_paths = sum(1 for s in sheaths if s.delay_probes > 0)
        return {
            "n_sheaths": n,
            "total_gain": total_gain,
            "mean_gain": total_gain / n,
            "mean_usage": sum(s.usage for s in sheaths) / n,
            "unused_ratio": sum(1 for s in sheaths if s.usage < 1e-3) / n,
            "turnover": turnover,
            "born": self._born,
            "died": self._died,
            "step": self._step,
            # --- 结构健康诊断（新增）---
            "in_grace": in_grace,          # 豁免期内通路数（持续偏高 → block 太小）
            "grace_ratio": in_grace / n,   # 占比，>0.5 说明几乎全是新边
            "probed_paths": probed_paths,  # 触发过 delay 探索的通路数
            "mean_since_co_fire": (
                sum(s.since_co_fire for s in sheaths) / n
            ),
            "budget_used": max(self.influx_of(s.dst_neuron, s.dst_channel)
                               for s in sheaths) if sheaths else 0.0,
            "budget_limit": self.budget_of(),
        }

    def split_for_child(
        self,
        parent: int,
        child: int,
        keep_ratio: float = 0.8,
    ) -> int:
        """分裂时的髓鞘分配：父保留 keep_ratio，子获得剩余部分

        这是**获得性遗传**的落点。W 由 seed 生成 → 先天、个体、不可遗传
        （genotype）；髓鞘由使用累积 → 获得性、关系性、可遗传（表观）。
        子节点继承"哪些连接被证明有用"这一结构知识，重新实例化算子本身。

        若连髓鞘也不继承，则分裂出的子节点从零开始，用进废退的跨代累积
        归零——拉马克式规则配上魏斯曼式遗传，累积效应消失。

        Returns:
            int: 为子节点建立的髓鞘条数。
        """
        created = 0
        for key, s in list(self._sheaths.items()):
            src_n, src_ch, dst_n, dst_ch = key
            # 父节点作为源 → 子节点接替为源
            if src_n == parent:
                self.add_sheath(
                    child, src_ch, dst_n, dst_ch,
                    delay=s.delay,
                    gain=s.gain * (1.0 - keep_ratio),
                    protection=s.protection,
                )
                s.gain *= keep_ratio
                created += 1
            # 父节点作为目标 → 子节点接替为目标
            elif dst_n == parent:
                self.add_sheath(
                    src_n, src_ch, child, dst_ch,
                    delay=s.delay,
                    gain=s.gain * (1.0 - keep_ratio),
                    protection=s.protection,
                )
                s.gain *= keep_ratio
                created += 1
        return created


# ---------------------------------------------------------------------------
# 事件驱动分发
# ---------------------------------------------------------------------------

@dataclass
class SignalEvent:
    """信号事件：信号在某时刻到达某神经元的某信道

    事件驱动并行分发的最小单位——各信道分量同时经各自算子传输，
    到达目标神经元时生成一个事件。

    ``sheath_key`` 记录本事件经由的髓鞘，用于 :meth:`SignalDispatcher.adapt_delays`
    在绑定成功后回调调整 delay（时间轴算子的更新路径）。
    """
    arrival_time: float
    target_neuron: int
    channel: str
    data: Tensor               # 该信道的信号分量
    source_tag: str            # 激活源标识
    origin_neuron: int         # 起源神经元（-1=外部信号）
    sheath_key: Optional[tuple[int, str, int, str]] = None


class SignalDispatcher:
    """信号分发器：事件驱动并行传输

    信号各维度分量同时经各自算子传输——并行性从维度对偶性自然涌现。
    不是"顺序游走选一条路"，是"各维度同时走各自的算子"。

    时序竞争 + 重合窗口：
    - 第一个到达阈值的赢（时序竞争）
    - 时间差 < 重合窗口的多信号会叠加触发（跨模态绑定）

    绑定成功的事件组记录在 :attr:`last_binding_groups`，供
    :meth:`adapt_delays` 调整 delay，闭合"选择作用于时间"的反馈环。
    """

    def __init__(self, neurons: dict[int, "Neuron"],
                 sheaths: dict[tuple[int, str, int, str], MyelinSheath],
                 history_capacity: int = 64,
                 port_layout: Optional[dict[str, tuple[int, int]]] = None,
                 connections: Optional[dict] = None):
        self.neurons = neurons
        self.sheaths = sheaths
        # 神经元之间的**连接层**。与 sheaths（包裹层）分离：
        # 未髓鞘的连接同样参与传导，只是慢且弱。
        self.connections = connections if connections is not None else {}
        # 全局信号空间布局。有了它，事件 data 才能 scatter 回全局维度，
        # 否则不同神经元展开维度数不同 → 后续 sum(e.data) 形状冲突。
        self.port_layout = port_layout
        self.global_dim = (
            sum(s for _, s in port_layout.values()) if port_layout else None)
        self.event_queue: list[SignalEvent] = []
        # 最近一步的绑定组：[(目标到达时间, [sheath_key, ...]), ...]
        self.last_binding_groups: list[tuple[float, list[tuple]]] = []
        self._base_time: float = 0.0

        # --- 源信号历史缓冲（供 delay 对齐度计算）---
        #
        # 为什么需要它：``probe_delay`` 的爬山需要比较 delay 在
        # d-w / d / d+w 三处的对齐度 ⟨源信号(delay=d), 目标活动⟩。
        # 目标活动是当前值，源信号必须回溯 delay 步。
        #
        # 无此缓冲时 ``align_fn`` 为 None → probe_delay 退化为随机扰动，
        # 而随机扰动已被实验证明与不探索无差异
        # （见 MyelinSheath.probe_delay 的数据表）。
        #
        # 成本：每步最多 N×C 个张量。history_enabled=False 可完全关闭。
        self.history_enabled: bool = True
        self.history_capacity: int = history_capacity
        self._src_hist: list[dict[tuple[int, str], Tensor]] = []
        # 目标神经元最近一次在该信道上收到的合成信号
        self._tgt_last: dict[tuple[int, str], Tensor] = {}
        # 诊断计数：对齐度计算失败的原因分布（全是 miss 说明缓冲没喂上）
        self.align_stats: dict[str, int] = {
            "ok": 0, "no_sheath": 0, "no_target": 0,
            "out_of_range": 0, "no_source": 0, "shape": 0,
        }
        # --- T2: 跨步事件队列（delay 一等公民 / 内生同步地基）---
        #
        # cross_step=False（默认）：事件当步结算——现有语义逐位保留。
        # cross_step=True：delay>0.5 的传输事件跨帧存活，挂在 self.pending，
        # 在 arrival_tick = 发出tick + delay 时进入彼时的结算。
        # 逻辑时钟 self._tick 由 :meth:`tick` 驱动（每系统步一次，由
        # system.step_awake 调用）。
        self.cross_step: bool = False
        self.pending: list[SignalEvent] = []
        self._tick: int = 0

    def tick(self) -> None:
        """推进分发器逻辑时钟（每系统步一次）"""
        self._tick += 1

    def _apply_cross_step(self, events: list[SignalEvent],
                          t: float) -> list[SignalEvent]:
        """cross_step 模式：分离当期到期事件与挂起事件

        到期事件并入当期事件流。cross_step=False 时原样直通（现有语义）。
        """
        if not self.cross_step:
            return events
        horizon = t + 0.5
        due = [e for e in events if e.arrival_time <= horizon]
        defer = [e for e in events if e.arrival_time > horizon]
        still = [p for p in self.pending if p.arrival_time > horizon]
        due += [p for p in self.pending if p.arrival_time <= horizon]
        # 新事件的未到期者必须入队挂起（否则慢通路永远到不了——实测教训）
        self.pending = still + defer
        return due

    # ---------- 历史缓冲 ----------

    def push_history(self) -> None:
        """推进历史缓冲（每个 dispatch 步调用一次）"""
        if not self.history_enabled:
            return
        self._src_hist.append({})
        while len(self._src_hist) > self.history_capacity:
            self._src_hist.pop(0)

    def record_source(self, nid: int, ch: str, data: Tensor) -> None:
        """记录本步某信道的源信号（传输前、增益前的原始输出）"""
        if not self.history_enabled or not self._src_hist:
            return
        self._src_hist[-1][(nid, ch)] = data

    def record_target(self, dst_n: int, dst_ch: str, data: Tensor) -> None:
        """记录目标神经元在该信道上最近一次收到的合成信号"""
        if not self.history_enabled:
            return
        self._tgt_last[(dst_n, dst_ch)] = data

    # ---------- delay 对齐度 ----------

    def compute_delay_alignment(
        self,
        sheath_key: tuple[int, str, int, str],
        window: float,
    ) -> Optional[tuple[float, float, float]]:
        """计算某髓鞘在 delay ∈ {d-w, d, d+w} 三处的对齐度

        对齐度 = ⟨源信号(经 delay 传输), 目标神经元当前活动⟩。
        与原型实验一致：该值越高，说明源与目标在该延迟下越同步。

        Args:
            sheath_key: ``(src_n, src_ch, dst_n, dst_ch)``
            window: 重合窗口（delay 探索的步长尺度）

        Returns:
            ``(align_minus, align_now, align_plus)``；信息不足时返回 None
            （调用方应退回随机，并计入 :attr:`align_stats`）。
        """
        def bump(k: str) -> None:
            self.align_stats[k] = self.align_stats.get(k, 0) + 1

        src_n, src_ch, dst_n, dst_ch = sheath_key
        if not self.history_enabled:
            return None
        sheath = self.sheaths.get(sheath_key)
        if sheath is None:
            bump("no_sheath")
            return None
        tgt = self._tgt_last.get((dst_n, dst_ch))
        if tgt is None:
            bump("no_target")
            return None

        d = float(sheath.delay)
        out: list[float] = []
        for delta in (-window, 0.0, window):
            dd = max(0.0, d + delta)
            idx = int(round(dd))
            if idx < 0 or idx >= len(self._src_hist):
                bump("out_of_range")
                return None
            entry = self._src_hist[-(1 + idx)]
            src = entry.get((src_n, src_ch))
            if src is None:
                bump("no_source")
                return None
            try:
                if src.shape != tgt.shape:
                    bump("shape")
                    return None
                out.append(float((src * tgt).sum().item()))
            except Exception:      # pragma: no cover - 张量后端差异
                bump("shape")
                return None
        bump("ok")
        return (out[0], out[1], out[2])

    def make_align_fn(self, window: float):
        """生成供 :meth:`MyelinSheathRegistry.maintain_ages` 使用的 align_fn

        Args:
            window: delay 探索步长，通常用 ``registry.suggest_window()``

        Returns:
            ``f(sheath, key) -> (m, now, p) | None``
        """
        def align_fn(sheath, key):
            return self.compute_delay_alignment(key, window)
        return align_fn

    def _scatter_to_global(self, output: Tensor, src_bounds, dst_ch: str,
                           gain: float) -> Tensor:
        """把神经元某信道的输出切片，散射到目标信道的全局位置

        Args:
            output: 神经元 process 的输出（展开顺序空间）
            src_bounds: ``(start, end)`` 源信道在展开空间中的切片范围
            dst_ch: 目标信道名
            gain: 髓鞘增益

        Returns:
            全局信号空间中的向量（长度恒为 ``global_dim``），
            仅在 ``dst_ch`` 对应位置有值。
        """
        import torch
        if self.global_dim is None or self.port_layout is None:
            return output * gain            # 旧行为（无布局信息）
        full = torch.zeros(self.global_dim)
        if src_bounds is None or dst_ch not in self.port_layout:
            return full
        s, e = src_bounds
        vec = output[s:e]
        off, size = self.port_layout[dst_ch]
        n = min(int(vec.numel()), int(size))
        if n > 0:
            full[off:off + n] = vec[:n] * gain
        return full

    def dispatch(self, signal: Tensor, source_tag: str,
                 t: float = 0.0) -> list[SignalEvent]:
        """并行分发信号：各信道同时经各自算子传输

        从所有激活的神经元分发，沿每个已展开信道查找髓鞘连接，
        传输（应用算子 + 髓鞘 delay/gain），按到达时间排序返回。
        """
        self._base_time = t
        self.event_queue = []
        # 推进源信号历史缓冲（delay 对齐度需要回溯）
        self.push_history()
        t_eff = self._tick if self.cross_step else t

        # 从所有激活的神经元分发
        for nid, neuron in self.neurons.items():
            if not neuron.alive or neuron.W is None:
                continue
            # 神经元处理信号（各维度切片经 W 变换）
            output = neuron.process(signal)
            if output is None:
                continue

            # 记录本步各信道的源信号（传输前、增益前）
            for ch in neuron.unfolded:
                self.record_source(nid, ch, output)

            # 沿每个已展开信道分发（并行）
            #
            # ⚠ 遍历的是**连接层**，不是髓鞘层。
            #
            # 旧实现遍历 ``self.sheaths``，等于"没有髓鞘就没有连接"——
            # 这正是把两层混为一谈的后果：未髓鞘的轴突无法传导，而实际
            # 上未髓鞘轴突**可以工作**，只是慢（delay 6.0）且弱（gain 0.3）。
            # 髓鞘化是后续的性能升级，不是连接存在的前提。
            #
            # ⚠ 事件 data 必须是**全局信号空间**中的向量，长度恒为
            # global_dim（原实现用压缩输出会形状冲突）。
            bounds = dict((c, (s0, e0)) for c, s0, e0 in neuron._channel_bounds())
            for ch in neuron.unfolded:
                for key, conn in self.connections.items():
                    src_n, src_ch, dst_n, dst_ch = key
                    if src_n != nid or src_ch != ch:
                        continue
                    sh = self.sheaths.get(key)
                    arrival = t_eff + conn.effective_delay(sh)
                    transmitted = self._scatter_to_global(
                        output, bounds.get(src_ch), dst_ch, conn.effective_gain(sh))
                    event = SignalEvent(
                        arrival_time=arrival,
                        target_neuron=dst_n,
                        channel=dst_ch,
                        data=transmitted,
                        source_tag=source_tag,
                        origin_neuron=nid,
                        sheath_key=key,
                    )
                    self.event_queue.append(event)

        # 跨步事件队列：延迟事件挂起，到期者并入当期（cross_step 模式）
        events = self._apply_cross_step(self.event_queue, t_eff)
        # 按到达时间排序（时序竞争基础）
        events.sort(key=lambda e: e.arrival_time)
        self.event_queue = events
        return self.event_queue

    def resolve_triggers(self, threshold: float,
                         coincidence_window: float) -> list[SignalEvent]:
        """解析触发：时序竞争 + 重合窗口

        - 第一个到达阈值的赢（时序竞争）：强度低于阈值的事件被过滤
        - 时间差 < 重合窗口的多信号叠加触发（跨模态绑定）：
          同一神经元在窗口内收到多信道信号时叠加

        成功绑定的事件组记录到 :attr:`last_binding_groups`。
        """
        triggered = []
        used = set()
        self.last_binding_groups = []

        for i, event in enumerate(self.event_queue):
            if i in used:
                continue
            strength = event.data.norm().item()
            if strength < threshold:
                continue

            # 检查重合窗口内的其他事件（跨模态叠加）
            combined = event.data.clone()
            group_keys: list[tuple] = []
            group_arrivals: list[float] = []
            if event.sheath_key is not None:
                group_keys.append(event.sheath_key)
                group_arrivals.append(event.arrival_time)

            for j in range(i + 1, len(self.event_queue)):
                if j in used:
                    continue
                other = self.event_queue[j]
                if other.arrival_time - event.arrival_time > coincidence_window:
                    break  # 已按到达时间排序，后续都超窗口
                if other.target_neuron == event.target_neuron:
                    # 同一神经元的另一信道信号叠加（跨模态绑定）
                    combined += other.data
                    used.add(j)
                    if other.sheath_key is not None:
                        group_keys.append(other.sheath_key)
                        group_arrivals.append(other.arrival_time)

            # 只有真正发生叠加（≥2 条通路）才算"绑定成功"，才触发 delay 同步
            if len(group_keys) >= 2:
                group_arrival = sum(group_arrivals) / len(group_arrivals)
                self.last_binding_groups.append((group_arrival, group_keys))

            # 记录目标神经元在该信道上的合成活动（对齐度的另一端）
            self.record_target(event.target_neuron, event.channel, combined)

            triggered.append(SignalEvent(
                arrival_time=event.arrival_time,
                target_neuron=event.target_neuron,
                channel=event.channel,
                data=combined,
                source_tag=event.source_tag,
                origin_neuron=event.origin_neuron,
                sheath_key=event.sheath_key,
            ))
            used.add(i)

        return triggered

    def adapt_delays(self, lr: float = 0.1) -> int:
        """闭合反馈环：绑定成功的通路，其 delay 向群体到达时间同步

        应在 :meth:`resolve_triggers` 之后调用。这完成了：

            髓鞘调 delay → 改变到达时序 → 改变谁能同时发射
            → 改变成边 → 改变谁能被髓鞘化

        Returns:
            int: 被调整的髓鞘条数。
        """
        adjusted = 0
        for group_arrival, keys in self.last_binding_groups:
            for key in keys:
                sheath = self.sheaths.get(key)
                if sheath is None:
                    continue
                sheath.adapt_delay(group_arrival, self._base_time, lr=lr)
                adjusted += 1
        return adjusted

    # ------------------------------------------------------------------
    # T1 边批向量化（macdev plan 20260906 任务 1）
    # ------------------------------------------------------------------

    def batched_dispatch(self, signal: Tensor, source_tag: str,
                         t: float = 0.0) -> list[SignalEvent]:
        """边批向量化分发：语义与 :meth:`dispatch` 逐位等价

        把逐边 Python 循环（每条边创建一个 global_dim 张量 + 拷贝）改为
        堆叠张量散射：边结构（局部切片、gain、目标偏移）一次性收进
        索引矩阵，gather → 乘 gain → scatter 三步张量操作完成全部传输。
        等价性与性能验证见
        ``docs/devo-project/verify/test_batched_dispatch.py``。

        神经元前向暂保持逐神经元（跨神经元 batch 属 T1 后续/CUDA 迁移）。
        """
        def _to_np_local(x):
            if hasattr(x, "detach"):
                return np.asarray(x.detach().cpu().numpy(), dtype=float)
            if hasattr(x, "a"):
                return np.asarray(x.a, dtype=float)
            return np.asarray(x, dtype=float)

        self._base_time = t
        self.event_queue = []
        self.push_history()
        t_eff = self._tick if self.cross_step else t

        if self.port_layout is None or self.global_dim is None:
            return self.dispatch(signal, source_tag, t=t)   # 退回旧路径

        # 1) 神经元前向（与 dispatch 相同的激活语义）
        outputs = {}
        for nid, neuron in self.neurons.items():
            if not neuron.alive or neuron.W is None:
                continue
            out = neuron.process(signal)
            if out is not None:
                outputs[nid] = out
                for ch in neuron.unfolded:
                    self.record_source(nid, ch, out)
        if not outputs:
            return []

        # 2) 神经元输出堆叠（行 = 神经元，列 = 局部展开序，零填充）
        nids = sorted(outputs.keys())
        row_of = {nid: i for i, nid in enumerate(nids)}
        lmax = max(int(o.numel()) for o in outputs.values())
        stack_np = np.zeros((len(nids), lmax))
        for nid, o in outputs.items():
            a = _to_np_local(o)
            stack_np[row_of[nid], :min(a.size, lmax)] = a.ravel()[:lmax]

        # 3) 收集活跃边参数（O(E) 轻量标量循环，重活在下面的张量步）
        edges = []
        for (sn, sc, dn, dc), conn in self.connections.items():
            n = self.neurons.get(sn)
            if (n is None or sn not in outputs or dc not in self.port_layout):
                continue
            bounds = {c: (s0, e0) for c, s0, e0 in n._channel_bounds()}
            if sc not in bounds:
                continue
            s0, e0 = bounds[sc]
            sh = self.sheaths.get((sn, sc, dn, dc))
            edges.append((
                row_of[sn], s0, e0, float(conn.effective_gain(sh)),
                float(conn.effective_delay(sh)),
                self.port_layout[dc], (sn, sc, dn, dc), dn, dc))
        if not edges:
            return []
        # 自适应回退：边数少时堆叠开销 > 循环开销（实测 E=3 时 0.4x），
        # 阈值以下直接走 loop 路径——batched 的收益在大 E 处兑现。
        if len(edges) < getattr(self, "batched_min_edges", 16):
            return self.dispatch(signal, source_tag, t=t)

        E = len(edges)
        seg_max = max(e[2] - e[1] for e in edges)

        # 4) gather：(E, seg_max) ← stack 平铺索引（-1 填充屏蔽）。
        #    向量化数学用 numpy（shim 无 torch.full/arange，见 COVERED），
        #    结果在末尾包回环境张量类型。
        ix = np.full((E, seg_max), -1, dtype=np.int64)
        dst_off = np.zeros(E, dtype=np.int64)
        dst_size = np.zeros(E, dtype=np.int64)
        gains = np.zeros(E)
        for i, (row, s0, e0, g, delay, (doff, dsize), key, dn, dc) in \
                enumerate(edges):
            L = e0 - s0
            ix[i, :L] = row * lmax + np.arange(s0, e0)
            dst_off[i] = doff
            dst_size[i] = dsize
            gains[i] = g

        stack_np = np.zeros((len(nids), lmax))
        for nid, o in outputs.items():
            a = _to_np_local(o)
            stack_np[row_of[nid], :a.size] = a.ravel()[:lmax]
        flat = stack_np.reshape(-1)
        valid = ix >= 0
        vals = np.where(valid, np.take(flat, np.clip(ix, 0, None)),
                        0.0) * gains[:, None]

        # 5) scatter：每条边写自己的 (global_dim,) 行（行内位置互不重复，
        #    高级索引赋值安全）；越出目标信道宽度的尾部截断（与
        #    _scatter_to_global 的 min(seg, size) 语义一致）
        pos = np.tile(np.arange(seg_max), (E, 1))
        write = valid & (pos < dst_size[:, None])
        Z_np = np.zeros((E, self.global_dim))
        rows_idx = np.repeat(np.arange(E), seg_max).reshape(E, seg_max)
        didx = dst_off[:, None] + np.clip(pos, 0, np.maximum(dst_size[:, None] - 1, 0))
        Z_np[rows_idx[write], didx[write]] = vals[write]

        # 5.5) 包回环境张量类型（shim Tensor / torch Tensor 均支持
        #      new_zeros 与 numpy 值的高级索引赋值）
        first_out = next(iter(outputs.values()))
        Z = first_out.new_zeros((E, self.global_dim))
        Z[rows_idx[write], didx[write]] = vals[write]

        # 6) 事件对象（轻量，逐边构造不可避免）
        events = []
        for i, (row, s0, e0, g, delay, (doff, dsize), key, dn, dc) in \
                enumerate(edges):
            events.append(SignalEvent(
                arrival_time=t_eff + delay,
                target_neuron=dn,
                channel=dc,
                data=Z[i],
                source_tag=source_tag,
                origin_neuron=key[0],
                sheath_key=key,
            ))
        events = self._apply_cross_step(events, t_eff)
        events.sort(key=lambda e: e.arrival_time)
        self.event_queue = events
        return events
