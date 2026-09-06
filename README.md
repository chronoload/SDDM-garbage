# 髓鞘发育式学习系统（Myelination-style Developmental Learning System）

[English](./README_EN.md) | 中文

一个面向 AGI 的**发育式学习架构**：结构不是前提，而是学习产物。种子神经元在环境信号驱动下展开维度、共发射成边、髓鞘化、分化分裂；所有学习信号来自环境即时反馈——无反向传播、无标注、无预训练嵌入。

> **一句话**：把发育神经生物学的机制（髓鞘化、用进废退、睡眠重放、观察学习）当作机器学习的工程约束，构造一个"轻内部、重耦合"的持续学习系统。

---

## 设计教义

1. **反应先于预测**——系统学习的是控制内外系统运动反馈的行动，不是序列统计。预测（自回归）只是蒸馏的手段；裁定权永远属于世界对行动的回应。
2. **逻辑先于统计**——符号信道的行为读出不采用幅值加权求和，而采用确认投票：每条通路对候选 token 投票，票权是世界对该"源→token"转移的逐条确认度。幅值不参与决策。
3. **裸编码**——所有模态 one-hot/原始物理量直入，实验者不预设距离结构；跨模态绑定由系统自己发育。禁止"嵌套预编码"（预先塞入训练好的相似度结构）。
4. **防发散**——发育期存在多重正反馈（髓鞘增厚、注意力调制、三因子学习率），其生物学灾难对应是癫痫。能量刹车与张力治理是存在性保证，非性能组件。
5. **棘轮**——发育状态跨 run 持久化（`persistence.py`），在已有成果上叠加且不倒退。学习证据必须通过平凡基线（persistence/多数类）检验。

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
│   T2 事件队列(跨步到达) / T3 资格迹 / T8 张力张量        │
├─────────────────────────────────────────────────────┤
│ 高层脑: 自回归残差(delta rule) + 三因子RPE(驱动满足度)   │
│   影子模式 → 交替试验 → 可逆接管 λ (情境条件化)          │
│   确认投票读出 / 注意力调制 / 脉冲编码 (解耦扩展层)       │
├─────────────────────────────────────────────────────┤
│ 睡眠: NREM稳态缩放 → 生成重放 → REM反事实 → 前向预演    │
│   归因账本: 每相位对每条髓鞘的 gain 增量记账             │
└─────────────────────────────────────────────────────┘
```

核心文件：[`mcp/developmental/myelin.py`](mcp/developmental/myelin.py)（髓鞘/分发/共发射）、[`system.py`](mcp/developmental/system.py)（主循环）、[`neuron.py`](mcp/developmental/neuron.py)、[`selection.py`](mcp/developmental/selection.py)（选择信号）、[`attention.py`](mcp/developmental/attention.py)、[`spikes.py`](mcp/developmental/spikes.py)、[`geometry.py`](mcp/developmental/geometry.py)（跨模态几何分析）、[`persistence.py`](mcp/developmental/persistence.py)（T9 序列化）。

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

每个机制都有 TDD 红→绿的验证脚本（`docs/devo-project/verify/`）：

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

### 语法习得（FSM 符号流，16 token，10000 步，无反传）

| 配置 | top1（下一 token 命中率） |
|---|---|
| 均匀随机 | 0.062 |
| echo 基线 | 0.071 |
| bigram 统计学习器（参照） | 0.556–0.611 |
| **本系统（w_norm=1.1）** | **0.243 ± 0.055** |
| **本系统（w_norm=3.0）** | **0.383 ± 0.017** |

真值探针：系统恢复了 9/16 条 FSM 转移（确定性转移全部命中）；棘轮保持率 1.02（学图片/电影后语法技能保持）。

### 消融（6000 步，3 seeds）

| 配置 | top1 | 说明 |
|---|---|---|
| baseline | 0.291±0.156 | 判据对齐后闭环成立（λ≈0.93，高层接管 93%）|
| +attention | 0.294±0.134 | 中性：键已学出，读出已被确认投票统治 |
| +spike | 0.108–0.118 | 负：不应期砍票数，安全机制非性能组件 |

### 诚实的限制

- **量化审计教训**：自制媒体任务被平凡基线击穿（电影 persistence=0.924 vs 系统 0.44）——"学习成功"的宣称必须先过 persistence/多数类检验。已引入预测编码图片沉积物（任务信号存在：一阶条件最优 0.423），系统提取仍在爬坡。
- **信用分配深度**：单层 W + 局部规则，层级组合未解（e-prop 资格迹已铺路）。
- **技能衰减**：多肢体实验中已学会技能得而复失，四种干预排除后嫌疑收敛于漂移无对抗选择/normalizer 非平稳/投票-W 交互（阈值张力为第一回应）。
- 本仓库是研究原型，不是产品。

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
docs/devo-project/     macdev 产物：spec/plan/log（双轨）
docs/devo-project/verify/  全部验证与诊断脚本（8 个 TDD 契约测试 +
                       quant_audit 量化审计 + 历史实验脚本归档）
paper/preprint.Rmd     arXiv 预印本（中文）
```

## 路线图

已完成：边批向量化、跨步事件队列、e-prop 资格迹、多肢体、世界桥骨架、量化审计、阈值张力、脑序列化、RWKV-7 式自适应衰减。

进行中：Moving MNIST 专业基准、Crafter 对接、情境条件化 competence 在媒体任务验证、技能衰减三嫌疑排查（漂移回滚已落地）、Transformer 教师器官（make-or-buy 融合）。

## 参考

- Rao & Ballard 1999, *Predictive coding in the visual cortex*
- Barlow 1961, *Possible principles underlying the transformations of sensory messages*
- Fritzke 1995, *Growing Neural Gas*（edge age 对应）
- Peng et al. 2025, *RWKV-7 "Goose"*（逐通道 in-context 学习率）
- Hosoya et al. 2005, *Dynamic predictive coding by the retina*
- Sun et al., *Learning to (Learn at Test Time)*

## License

MIT
