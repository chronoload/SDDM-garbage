"""脉冲编码（解耦版）：通路级的漏积分发尖峰

## 在架构中的位置

挂在 :meth:`DefaultHigherBrain.dispatch_signal`——dispatch 之后、
resolve_triggers 之前（attention 调制之后）。把连续的分级传输流
转成**事件驱动的尖峰流**：

    膜电位 m_j ← decay · m_j + ‖传输_j‖
    m_j ≥ θ → 发尖峰（事件放行），m_j 清零，进入不应期
    m_j < θ → 阈下静默（事件门控掉，能量留在膜里）

## 为什么 delay 由此成为一等公民

速率编码下，髓鞘的 delay 只把信号稀释、不影响任何判定——
"时间轴算子"有名无实。脉冲化后，**到达时刻决定膜积分能否在
重合窗口内叠加**：增厚（delay↓）→ 尖峰到得早 → 与别的通路
同步到达 → 共发射成边。这闭合了 myelin.py 开篇承诺的回路——
"选择作用于时间，而不只是作用于强度"——的最后一环。

## 解耦契约

- 神经元**胞体保持速率编码**：W、utility、metabolism、分化、自噬
  一概不动。这是"通路发尖峰，胞体算速率"的混合设计。
- 髓鞘算子不读不写：delay/gain 只被**消费**（gain 缩放进入膜的电
  流，delay 决定到达时刻），不被本模块更新。
- 选择动力学照旧：尖峰事件**携带分级载荷**（gate 前的原始传输量），
  ``pathway_contributions`` 的局部信用分配不需要任何改动。

## 工程取舍（诚实声明）

真·LIF 的尖峰载荷是全或无的，信息全在时序里；那要求重写全部
学习通道。本实现是**gateway 脉冲化**：离散决策（发/不发）是脉冲
的，载荷保留分级信息——用进废退的判据精度不受损。代价是信息
仍在载荷里而不纯在时序里；这是通往全脉冲化的中间形态，也是
e-prop eligibility trace 的现成宿主。

生物学对应：activity-dependent myelination 中，髓鞘改变传导延迟
→ 改变尖峰到达时序 → 改变 STDP 式共发射（本架构的
CoincidenceDetector 按到达时刻配对，即 STDP 的同发窗口）。
"""
from __future__ import annotations


class SpikeCoder:
    """通路级漏积分发尖峰（integrate-and-fire gateway）

    Args:
        threshold: 发尖峰阈值 θ（对事件强度 ‖data‖ 的膜积分）
        decay: 膜漏电系数（每步保留比例）。0 = 纯瞬时阈；0.3 ≈ 积分 ~3 步
        refractory: 不应期（步）。期间该通路的传输被硬门控
    """

    def __init__(
        self,
        threshold: float = 0.2,
        decay: float = 0.3,
        refractory: int = 1,
    ):
        self.threshold = threshold
        self.decay = decay
        self.refractory = refractory
        # {pathway_key: 膜电位}；key 优先用 sheath_key，无髓鞘事件退化为
        # (origin, target, channel)
        self.membrane: dict = {}
        self._refrac: dict = {}
        self.stats: dict[str, float] = {
            "steps": 0, "events_in": 0, "spikes": 0,
            "gated": 0, "refractory": 0,
        }

    # ---------- 编码 ----------

    def encode(self, events: list) -> list:
        """连续事件流 → 尖峰流（放行或门控）

        放行的事件**保留分级载荷**（供局部信用分配）；被门控的事件
        被丢弃，其能量留在膜电位里（不是丢失，是积分）。
        """
        self.stats["steps"] += 1
        if not events:
            return []
        out = []
        for e in events:
            self.stats["events_in"] += 1
            key = (e.sheath_key if e.sheath_key is not None
                   else (e.origin_neuron, e.target_neuron, e.channel))
            # 不应期：硬关断（离散计数，无渐近尾巴）
            r = self._refrac.get(key, 0)
            if r > 0:
                self._refrac[key] = r - 1
                self.stats["refractory"] += 1
                continue
            # 漏积分
            m = self.membrane.get(key, 0.0) * self.decay
            m += float(e.data.norm().item())
            if m >= self.threshold:
                # 发尖峰：全或无的门控决策，载荷保留分级信息
                self.membrane[key] = 0.0
                self._refrac[key] = self.refractory
                self.stats["spikes"] += 1
                out.append(e)
            else:
                # 阈下静默：能量留在膜里（漏积分记忆）
                self.membrane[key] = m
                self.stats["gated"] += 1
        return out

    # ---------- 诊断 ----------

    def report(self) -> dict:
        total = max(1, self.stats["events_in"])
        return {
            "n_membranes": len(self.membrane),
            "spike_rate": self.stats["spikes"] / total,
            **{k: float(v) for k, v in self.stats.items()},
        }
