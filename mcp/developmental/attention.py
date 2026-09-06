"""自注意力（解耦版）：内容寻址的通路调制

## 在架构中的位置

挂在 :meth:`DefaultHigherBrain.dispatch_signal`——dispatch 之后、
resolve_triggers 之前的单一咽喉点。对事件流做**乘性再分配**：

    f_j = 1 + depth · (n·w_j − 1)，   w = softmax(q·k_j/√d)

- ``q`` = 本步分发输入（归一化全局信号）＝ query
- ``k_j`` = 每条通路的**键向量**（本模块私有参数）
- 中心化保证调制是零和的：注意力只**重新分配**放大倍数，
  不凭空注入能量（均值恒为 1）。键全零时 w 均匀 → f≡1（恒等）。

## 解耦契约（本模块**永不**触碰的东西）

- 神经元权重 ``W``：不读不写
- 髓鞘算子（delay/gain/protection）：不读不写
- 选择动力学（贡献/衰减/周转）：不干预

键是第三组参数，生命周期独立。注意力权重是**逐瞬态**的——
本步算出、本步用完，不沉淀成结构。结构仍只由共发射成边与
用进废退产生。

## 学习规则（局部、三因子兼容）

    Δk_j = lr · c_j · q

``c_j`` 是通路 j 的环境残差贡献（``pathway_contributions`` 的输出，
即"用进废退"的判据）。语义：**这个上下文里这条通路有用 →
把"这类上下文"绑到这条通路上**。这正是注意力教育
（Gibson: education of attention）的机制版——可读性是逐通路
逐上下文学出来的，不是预设的。有界性由单位球投影保证。

与变换器的关系与差别：q·k 相似度 + softmax 与 Transformer 同构；
但这里没有值投影（value = 事件本身的传输量）、没有多层堆叠、
没有反向传播——键由局部残差信用直接塑造。
"""
from __future__ import annotations

import math

import torch


class AttentionModulator:
    """通路级自注意力：query=本步输入，key=通路上下文绑定，value=传输本身

    Args:
        global_dim: 全局信号维度（键向量长度）
        depth: 调制深度 α。0 = 关闭（f≡1）；1 = 全幅再分配
        lr: 键学习率
        key_scale: 键范数上界（单位球投影，保证有界）
    """

    def __init__(
        self,
        global_dim: int,
        depth: float = 1.0,
        lr: float = 0.05,
        key_scale: float = 1.0,
    ):
        self.global_dim = global_dim
        self.depth = depth
        self.lr = lr
        self.key_scale = key_scale
        # {sheath_key: Tensor(global_dim)}——通路 → 上下文绑定
        self.keys: dict[tuple, torch.Tensor] = {}
        self._last_query: torch.Tensor | None = None
        # 诊断
        self.stats: dict[str, float] = {
            "steps": 0, "events": 0, "keyed_events": 0,
            "max_gain": 1.0, "min_gain": 1.0, "learned": 0,
        }

    # ---------- 前向：逐事件乘性调制 ----------

    def modulate(self, events: list, query: torch.Tensor) -> list:
        """按内容相似度再分配各事件的传输增益

        Args:
            events: ``dispatch`` 产出的事件列表（已被 attention 消费前）
            query: 本步分发输入（归一化全局信号，长度=global_dim）

        Returns:
            新的事件列表（data 已乘 f_j；事件对象不复用，避免污染调用方）
        """
        self.stats["steps"] += 1
        self.stats["events"] += len(events)
        if not events or self.depth <= 0.0:
            self._last_query = query
            return list(events)
        q = query.reshape(-1)
        if int(q.numel()) != self.global_dim:
            self._last_query = query
            return list(events)
        self._last_query = q.detach().clone()

        n = len(events)
        # logits：有键的事件算 q·k/√d，无键记 0（= 无偏好）
        logits = []
        for e in events:
            k = self.keys.get(e.sheath_key)
            if k is None:
                logits.append(0.0)
            else:
                self.stats["keyed_events"] += 1
                logits.append(float((q * k).sum().item())
                              / math.sqrt(self.global_dim))
        m = max(logits)
        exps = [math.exp(min(60.0, z - m)) for z in logits]
        total = sum(exps)
        w = [x / total for x in exps]
        # 中心化：均值因子 = 1（零和再分配）
        gains = [1.0 + self.depth * (n * wi - 1.0) for wi in w]
        self.stats["max_gain"] = max(self.stats["max_gain"], max(gains))
        self.stats["min_gain"] = min(self.stats["min_gain"], min(gains))

        out = []
        for e, f in zip(events, gains):
            if abs(f - 1.0) < 1e-9:
                out.append(e)
            else:
                out.append(type(e)(
                    arrival_time=e.arrival_time,
                    target_neuron=e.target_neuron,
                    channel=e.channel,
                    data=e.data * f,
                    source_tag=e.source_tag,
                    origin_neuron=e.origin_neuron,
                    sheath_key=e.sheath_key,
                ))
        return out

    # ---------- 学习：贡献驱动的键更新 ----------

    def learn_from_contributions(
        self, contributions: dict[tuple, float]
    ) -> int:
        """用通路的残差贡献更新键（局部规则，无反向传播）

        Args:
            contributions: ``{sheath_key: c_j}``（pathway_contributions 输出）

        Returns:
            被更新的键数。
        """
        q = self._last_query
        if q is None or not contributions:
            return 0
        n = 0
        for key, c in contributions.items():
            k = self.keys.get(key)
            if k is None:
                k = q * self.lr * c
            else:
                k = k + self.lr * c * q
            # ⚠ 键存**方向**（单位向量），幅度交给 depth。
            # 旧方案（累积 + 范数封顶）实测失效：|k| 停留在 1e-3 量级，
            # logits≈0 → softmax 均匀 → 调制恒为恒等（±0.5%），机制空转。
            # 方向归一化后 logits 量级 ~|q|/√d ≈ 1，softmax 才有真实
            # 再分配；有界性由单位球天然保证。
            nn = float(k.norm().item())
            if nn > 1e-12:
                k = k * (1.0 / nn)
            self.keys[key] = k
            n += 1
        self.stats["learned"] += n
        return n

    # ---------- 诊断 ----------

    def report(self) -> dict:
        return {
            "n_keys": len(self.keys),
            **{k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in self.stats.items()},
        }
