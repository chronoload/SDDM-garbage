# 髓鞘发育式学习系统（Myelination-style Developmental Learning System）

# [SHIT](./SHIT_zh.md)

评述：我们对机器学习与人工智能的看法（[English](./SHIT.md)）。My Views on Machine Learning and Artificial Intelligence

[English](./README_EN.md) | 中文

这是一个面向 AGI 的发育式学习架构。它起于一个判断：网络结构不该由人来设定，而应当从学习中长出来。种子神经元在环境信号驱动下展开维度、共发射成边、髓鞘化、分化分裂；所有学习信号来自环境即时反馈——没有反向传播，没有标注，也没有预训练嵌入。

> 一句话：把发育神经生物学的机制（髓鞘化、用进废退、睡眠重放、观察学习）当作机器学习的工程约束，构造一个"轻内部、重耦合"的持续学习系统。

## 设计教义

五条教义贯穿全部实现，改动任何一条都意味着改动系统本身。

1. **反应先于预测。** 系统学的是控制内外系统运动反馈的行动，不是序列统计。自回归预测只是蒸馏的手段；裁定权在世界对行动的回应那里，不在预测误差手里。
2. **逻辑先于统计。** 符号信道的行为读出不按幅值加权求和，而用确认投票：每条通路对候选 token 投票，票权是世界对该"源→token"转移的逐条确认度。幅值不参与决策。
3. **裸编码。** 各模态以 one-hot 或原始物理量直入，实验者不预设距离结构；跨模态绑定要靠系统自己发育出来。预先塞入训练好的相似度结构（嵌套预编码）是被明令禁止的捷径。
4. **防发散。** 发育期有多重正反馈——髓鞘增厚、注意力调制、三因子学习率互相放大，其生物学上的失灵形态就是癫痫。能量刹车与张力治理因此是存在性保证，不是性能组件。
5. **棘轮。** 发育状态跨 run 持久化（`persistence.py`），在已有成果上叠加且不倒退。而任何"学习成功了"的说法，都要先过平凡基线（persistence / 多数类）这一关。

## 架构总览

```
环境(沉积物/沙箱/沙盘) ⇄ 感知信道 vis/aud/... ⇄ 端口(Port, 裸编码+normalizer)
        ⇅
┌─────────────────────────────────────────────────────┐
│ 爬虫脑(ReptilianFunction): 冻结反射 = 肢体行动面       │  ← 可插拔: babble/echo/LLM/沙箱桥
│   ReflexArc 预编排路由（λ=0 时的基线行为）              │
├─────────────────────────────────────────────────────┤
│ 发育基底: 种子神经元 → unfold → 共发射成边 → 髓鞘化     │
│   髓鞘 = 第二算子 (delay 时间轴 × gain 强度轴)          │
│   选择 = 用进废退 (contribution > ρ·(1-protection))    │
│   事件队列(跨步到达) / 资格迹 / 张力张量                 │
├─────────────────────────────────────────────────────┤
│ 高层脑: 自回归残差(delta rule) + 三因子RPE(驱动满足度)   │
│   影子模式 → 交替试验 → 可逆接管 λ (情境条件化)          │
│   确认投票读出 / 注意力调制 / 脉冲编码 (解耦扩展层)       │
├─────────────────────────────────────────────────────┤
│ 睡眠: NREM稳态缩放 → 生成重放 → REM反事实 → 前向预演    │
│   归因账本: 每相位对每条髓鞘的 gain 增量记账             │
└─────────────────────────────────────────────────────┘
```

核心实现集中在 [`mcp/developmental/myelin.py`](mcp/developmental/myelin.py)（髓鞘/分发/共发射）、[`system.py`](mcp/developmental/system.py)（主循环）、[`neuron.py`](mcp/developmental/neuron.py)、[`selection.py`](mcp/developmental/selection.py)（选择信号）、[`attention.py`](mcp/developmental/attention.py)、[`spikes.py`](mcp/developmental/spikes.py)、[`geometry.py`](mcp/developmental/geometry.py)（跨模态几何分析）、[`persistence.py`](mcp/developmental/persistence.py)（脑序列化）。

## 数学核心

记号：$x_t \in \mathbb{R}^D$ 全局信号；髓鞘 $j$ 增益 $g_j$、延迟 $\tau_j$；传输 $p_j = g_j \cdot \mathrm{scatter}(y_{src}, ch_{dst})$；残差 $r_t = x_{t+1} - y_t$。完整推导见 [paper/preprint.Rmd](paper/preprint.Rmd) 的"方法"章。

**学习（三条局部通道，无反向传播）**

```
自回归 delta rule:  ΔW = η_t · r_t ⊗ x_t,    η_t = η₀/(1 + t/τ₀),  ‖W‖ → W_target
通路贡献 (NLMS):    c_j = ⟨p_j, r_t⟩ / ‖x_t‖²
三因子:             ΔW = η · RPE · (y_t ⊗ x_t),   RPE = relief_t − EMA(relief)
e-prop 资格迹:      E_j ← γ_j·E_j + c_j,          g_j += η_ep · RPE · E_j
                    γ_j = γ + (1−γ)·g_j/GAIN_MAX   （RWKV-7 式承重自适应衰减）
```

**选择与结构**

```
存活条件:           c_j · use_scale > ρ · (1 − protection)     （自适应预算：g* ∝ 增量流/ρ）
共发射成边:         |t_arr_i − t_arr_j| < w = std({τ});  1 次成连接, 3 次成髓鞘
承载力:             influx(d) = Σ g_j(1+λ_p·p_j) < B = B₀·min(3, 1+2·EMA(err))
```

**仲裁（裁定权在反应侧，不在预测误差——实测死锁：λ 由误差裁定 → 世界死寂）**

```
competence = (S_high − S_reflex) / max(S_reflex, ε)      （情境条件化：按刺激簇分账）
λ = (1−urgency)·σ(β·(competence − θ)),   β=8, θ=0.05,  接管可逆
确认投票:  cred(src,tok) ← 0.85·cred + 0.15·1[hit];  读出按 cred 加权，幅值不参与
```

**防发散与治理**

```
张力张量:   T_j ← γ_T·T_j + |p_j ⊙ r_t|;   load(n) = Σ mean|T_j|
漂移治理:   ΔW = ρ_drift·ξ/(1+load(n));     err_post > 1.1·err_pre ⇒ W 回滚
能量刹车:   E_t > 6·EMA(E) ⇒ y ← 0.5·y     （癫痫的机制对应）
脉冲门控:   m_j ← γ·m_j + ‖p_j‖;  m_j ≥ θ ⇒ 放行+不应期;  阈下静默
稳态缩放:   g_j ← g_j·(1 − β·(1−protection))  （睡眠 NREM S3）
```

## 参数指南

最重要的参数是 $\rho$（decay_rate）——它同时是衰减率与选择阈值：存活条件 `contribution·use_scale > ρ·(1−protection)`，总容量上界 ≈ 增量流/ρ。两端都失败：太小 → 噪声通路堆积；太大 → 误杀弱信号通路。实测最优 0.01–0.025。

| 参数 | 默认 | 说明 |
|---|---|---|
| decay_rate (ρ) | 0.01 | 选择阈值；实测最优 0.01–0.025 |
| use_scale | 0.5 | 贡献→增量缩放 |
| autoreg_lr / tau | 0.05 / 5000 | delta rule 步长与衰减常数 |
| autoreg_w_norm | 3.0 | W 范数目标（1.1→3.0 提升 top1） |
| drift_rate / interval | 0.005 / 50 | 变异步长与周期（必须与选择分离） |
| wiring_block | 50 | 成边 block（每步成边 → 周转空转 11.5） |
| wire / myelination_threshold | 1 / 3 | 成边 / 髓鞘化确认次数 |
| GRACE_PERIOD | 10 | 新生儿豁免期 |
| conn_starve_limit | 3000 | 连接饿死上限 |
| AGE_PROBE / AGE_DEATH | 50 / 2000 | delay 探索 / 删除阈值 |
| self_proof | 0.3 | 跨信道成边门控 |
| exuberant_low / high | 2 / 10 | 冷启动豁免双水位 |
| target_influx / protection_cost | 8.0 / 1.0 | 承载力预算 |
| spike θ / γ / refractory | 0.2 / 0.3 / 1 | 脉冲门控 |
| attn depth / lr | 1.0 / 0.05 | 注意力调制 |
| eprop lr / γ | 0.05 / 0.9 | 资格迹分账 / 衰减（γ 承重自适应） |
| relief_gain | 1.0 | 三因子调制强度 |
| λ 门控 β / θ | 8 / 0.05 | competence 斜率 / 阈值 |
| takeover_threshold | 0.05 | 接管/收回滞回 |
| explore_eps / background | 0.1 / 0.02 | 两档探索率（单档锁死影子） |
| sleep_beta | 0.1 | 稳态缩放强度 |
| seizure_ratio / damp | 6.0 / 0.5 | 能量刹车 |

⚠ 以上默认值在特定实验环境扫出，换环境需重扫。按领域通行范式：手工设计形式，参数做敏感性分析或自动搜索。

## 快速开始

```bash
git clone <repo>
cd devo_project
pip install numpy            # 唯一硬依赖
pip install torch            # 可选：真实 PyTorch（CPU/GPU）；缺省时 numpy_torch_shim 兜底

# 语法习得实验（FSM 符号流的自回归补全，16 token，3 seeds）
python baby_grammar.py

# 统一词表持续课程：语法 → 图片 → 电影 → 语法回访（棘轮保持率）
python unified_curriculum.py

# 沉积物阅读：任意文件（字节级打字机，256 键裸编码）
python deposit_reader.py baby_grammar.py

# 多肢体婴儿：双行动面 + 协同需求世界
python baby_multilimb.py
```

## 测试

每个机制都有 TDD 红→绿的验证脚本，收在 `docs/devo-project/verify/`：

```bash
python docs/devo-project/verify/test_batched_dispatch.py   # 边批向量化逐位等价
python docs/devo-project/verify/test_event_queue.py        # 跨步事件队列到达时序
python docs/devo-project/verify/test_eprop.py              # e-prop 资格迹
python docs/devo-project/verify/test_context_competence.py # 情境条件化 competence
python docs/devo-project/verify/test_tension.py            # 阈值张力治理
python docs/devo-project/verify/test_sandbox_bridge.py     # 世界桥 HTTP 端点
python docs/devo-project/verify/test_persistence.py        # 脑序列化/棘轮
python docs/devo-project/verify/quant_audit.py             # 量化审计（平凡基线）
```

## 主要实验结果

### 语法习得（FSM 符号流，16 token，10000 步，无反向传播）

| 配置 | top1（下一 token 命中率） |
|---|---|
| 均匀随机 | 0.062 |
| echo 基线 | 0.071 |
| bigram 统计学习器（参照） | 0.556–0.611 |
| 本系统（w_norm=1.1） | 0.243 ± 0.055 |
| 本系统（w_norm=3.0） | 0.383 ± 0.017 |

真值探针：系统恢复了 9/16 条 FSM 转移，确定性转移全部命中；学完图片和电影沉积物之后再回访语法，棘轮保持率 1.02——技能没有打滑。

### 消融（6000 步，3 seeds）

| 配置 | top1 | 说明 |
|---|---|---|
| baseline | 0.291±0.156 | 判据对齐后闭环成立（λ≈0.93，高层接管 93%）|
| +attention | 0.294±0.134 | 中性：键已学出，但读出已被确认投票接管 |
| +spike | 0.108–0.118 | 负结果：不应期砍掉了票数。脉冲是安全机制，不是性能组件 |

### 诚实的限制

- 量化审计给过一次教训：自制媒体任务被平凡基线击穿（电影 persistence=0.924，系统只有 0.44）——任何"学习成功"的说法都必须先过 persistence/多数类检验。预测编码版的图片沉积物已让任务信号成立（一阶条件最优 0.423），但系统提取还在爬坡。
- 信用分配只有一层深度：单层 W 加局部规则，层级组合问题没有解决（e-prop 资格迹是在铺路）。
- 多肢体实验里，已学会的技能得而复失。四种干预都没能阻止，嫌疑收敛到漂移缺少对抗选择、normalizer 非平稳、投票与 W 的交互三处（阈值张力是第一回应）。
- 这是研究原型，不是产品。

## 仓库结构

```
mcp/developmental/     架构本体（myelin/system/neuron/selection/higher_brain/
                       attention/spikes/geometry/persistence/reflex/...）
baby_grammar.py        语法习得实验（含 off-by-one 判据教训的修复史）
baby_loop.py           婴儿发声控制回路（反应 vs 预测对照）
baby_multilimb.py      多肢体婴儿（双行动面 + 协同需求）
deposit_reader.py      沉积物阅读器（字节级打字机）
deposit_media.py       图片/电影沉积物适配（预测编码编码）
unified_curriculum.py  统一词表持续课程
devo_control.py        具身控制任务（不稳定系统 + 用进废退）
mcp/developmental/builtin/sandbox_bridge.py  世界桥（HTTP JSON，Three.js 兼容）
docs/research/         文献对照与定位调研（66 条来源核验）
docs/devo-project/     macdev 产物：spec/plan/log（双轨）
docs/devo-project/verify/  全部验证与诊断脚本（8 个 TDD 契约测试 +
                       quant_audit 量化审计 + 历史实验脚本归档）
paper/preprint.Rmd     arXiv 预印本草稿
```

## 路线图

已经落地：边批向量化、跨步事件队列、e-prop 资格迹、多肢体、世界桥骨架、量化审计、阈值张力、脑序列化、RWKV-7 式自适应衰减。

正在做：Moving MNIST 专业基准、Crafter 对接、情境条件化 competence 在媒体任务上的验证、技能衰减三嫌疑的逐一排查（漂移回滚已落地）、Transformer 教师器官（make-or-buy 融合）。

## 参考文献

**直接前身与理论支撑**

- Åström & Wittenmark (1973). On Self-Tuning Regulators. *Automatica* 9(2)——"信号驱动 + 在线即时学习"的最近前身
- Brooks (1986). Subsumption Architecture——基线层分离的直接前身；本系统补上了它自陈缺失的学习与记忆
- Conant & Ashby (1970). Every good regulator of a system must be a model of that system——"只需建模本质变量的低维投影"是惰性维度展开的依据
- Oja (1982)——权重层用进废退的数学形式
- Tononi & Cirelli (2019). Sleep and synaptic down-selection——睡眠稳态缩放的来源
- Asokan, Chhabria & Chakravarthy (2015). A model of learning temporal delays, representative of adaptive myelination（CNS*2015 海报）——髓鞘作为可训练量的唯一直接前身
- Lefebvre et al. (2025). Myelin-induced gain control in nonlinear neural networks. *Communications Physics*——髓鞘 gain 维度的理论支撑

**学习机制对照**

- Fritzke (1995). Growing Neural Gas——edge age 机制的对应
- Rao & Ballard (1999). Predictive coding in the visual cortex——自回归残差通道的学院派正式版
- Barlow (1961). Efficient coding hypothesis——冗余削减与用进废退去重的共同源头
- Hosoya et al. (2005). Dynamic predictive coding by the retina——normalizer 自适应统计的正典
- Hinton (2022). Forward-Forward——无反向传播局部学习的现代参照
- Ding et al. 信息饱和的结构发育神经网络（CAAI TIT）——分裂时机与父子关系的对照系
- Baxter & Levy (2019). 自适应突触发生网络——自噬/清除的对照系

**持续学习与序列建模范式**

- Peng et al. (2025). RWKV-7 "Goose"（arXiv:2503.14456）——逐通道 in-context 学习率，已落地为自适应衰减
- Sun et al. Learning to (Learn at Test Time)——状态即测试时可训记忆
- Mamba-CL (arXiv:2411.15469)——选择性门控作为遗忘控制器

**基准**

- Crafter、MineDojo/MineStudio（具身代理）
- Moving MNIST（视频预测）、CIFAR-10（图像，标签裁定协议）

更完整的 66 条来源核验（含逐条真伪与降级标注）见 [`docs/research/发育智能系统_相似研究调研.md`](docs/research/发育智能系统_相似研究调研.md)。

## License

MIT
