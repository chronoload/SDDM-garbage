# 需求清单（结构化，由 `macdev requirement` 维护）

> 数据源: `requirements.db` · 共 8 条 · 本文件自动同步，勿手改


## open (8)

| id | 类型 | 名称 | 状态 | 来源 | 说明 |
|----|------|------|------|------|------|
| 1 | spec | `deposit-replay` | open | cli | 沉积物重放通道：静态信息转动态反馈回路。已落地字节级deposit_reader(256键裸编码打字机)；待扩展：图片(扫视重放)、电影(帧差event信道)、PDF(文本层+渲染页双信道)、重放批处 |
| 2 | spec | `gpu-vectorize` | open | cli | 边批向量化+CUDA：dispatcher逐边Python循环改堆叠张量散射，delay按桶离散化，共发射O(E^2)改排序滑窗；neurons/edges/experience三维并行，时间轴因果串 |
| 3 | spec | `event-queue` | open | cli | 跨步事件队列：传输事件跨帧存活，t步发出t+delay步到达参与彼时共发射；delay一等公民(内生同步)，同步由delay算子学出而非程序编排 |
| 4 | spec | `eprop-depth` | open | cli | 髓鞘深度信用：每条髓鞘维护eligibility trace，RPE到来时按E分账；信用沿A->B->C链回流，链式髓鞘化=深度；geometry.cycle_coherence作生成回路建立判据 |
| 5 | spec | `multi-limb` | open | cli | 多肢体爬虫脑：声带/手指/肢体多个ReptilianFunction原生并行，世界tick统一更新全部感知信道，无程序编排的信道配对；协同需求由世界设计给出 |
| 6 | spec | `world-bridge` | open | cli | 世界模型桥：Three.js/Minecraft风格沙箱经HTTP/WS桥(复用builtin/web_agent.py代理模式)暴露reset/step/observe/action，爬虫脑挂载为肢 |
| 7 | spec | `benchmark` | open | cli | 跨模态benchmark对接：调研选型(MineDojo/Crafter/多模态代理基准族)后适配本架构肢体接口，成绩对照已发表LLM/RL基线 |
| 8 | spec | `stability` | open | cli | 稳定性护栏回归：癫痫刹车与接管可逆性纳入每次消融必测项，防发散是存在性保证非性能组件 |