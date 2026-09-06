"""超模态统一编码：逐信道自适应归一化

**为什么必须有这一层**

异构信道的原始尺度差异可以是数量级的（视觉读数 std 1.5、本体 1.0、
离散事件 0.5，而携带真实状态的通道可能只有 0.05）。若直接把原始信号
喂给发育机制，通路增益的稳态将由信道的**尺度**决定，而非由信道的
**可预测性**决定：

- 大幅度噪声通道的通路靠随机游走抵抗衰减，跌跌撞撞存活；
- 弱信号但真实可预测的通路，稳态增益 ``gain* ∝ increment/ρ`` 偏小，
  反而跌破死亡阈值被误杀。

实测（具身控制任务，ρ=0.2）：未归一化时真实维度通路剩 3.0 条、噪声
维度剩 4.6 条——**噪声通路比真实通路更耐衰减**。归一化后通路存亡才由
"是否携带可预测信息"决定。

生物学对应：感受野增益控制（gain control / 亮度适应）。视网膜在
10 个数量级的光强范围内工作，靠的就是逐通道自适应归一化，而不是
把原始光强直接送进皮层。

用法
----

归一化在信号进入网络前完成，反归一化在交给反射弧（肢体）前完成，
这样**控制律本身不需要改动**——改进的只是输入质量::

    norm = ChannelNormalizer(port_layout)
    on = norm.encode(raw_signal)        # 进入发育网络
    ...                                  # 网络内部在归一化空间运算
    out = norm.decode(ohat)             # 回到原始物理单位，交给反射

注意：归一化是**有状态**的（运行 RMS），需随脑状态一起持久化。
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class ChannelNormalizer:
    """逐信道自适应归一化（超模态统一编码）

    对每个信道维护运行 RMS，把异构尺度的信号统一到单位方差空间：

        ``rms_c ← sqrt(momentum · rms_c² + (1 − momentum) · x_c²)``
        ``x̂_c = x_c / (rms_c + ε)``

    这样不同模态、不同量纲、不同量级的信号在同一个数值空间中被处理，
    后续的成边、选择、衰减都作用在**可比的**量上。

    Args:
        port_layout: 信道布局 ``{channel_name: (offset, size)}``
        momentum: RMS 的 EMA 动量。越大越平滑，但对非平稳信号响应越慢
        eps: 数值稳定项，同时作为未激活通道的下限
        warmup: 预热步数。预热期内只估计 RMS 不参与学习，避免初始
            瞬态把通路增益带偏。
    """

    def __init__(
        self,
        port_layout: dict[str, tuple[int, int]],
        momentum: float = 0.99,
        eps: float = 1e-6,
        warmup: int = 200,
    ):
        self.port_layout = port_layout
        self.momentum = momentum
        self.eps = eps
        self.warmup = warmup
        self._rms: Optional[Tensor] = None
        self._step: int = 0

    # ---------- 状态 ----------

    @property
    def ready(self) -> bool:
        """RMS 是否已充分收敛（预热完成）"""
        return self._step >= self.warmup

    @property
    def step(self) -> int:
        return self._step

    def state_dict(self) -> dict:
        """导出状态（随脑状态一起持久化）"""
        return {
            "rms": (None if self._rms is None
                    else self._rms.detach().clone()),
            "step": self._step,
        }

    def load_state_dict(self, state: dict) -> None:
        """恢复状态"""
        self._rms = state.get("rms")
        self._step = state.get("step", 0)

    # ---------- 核心 ----------

    def update(self, x: Tensor) -> Tensor:
        """用当前观测更新运行 RMS，返回归一化后的信号

        先更新后归一化（避免首帧用未初始化的 RMS）。
        """
        x_flat = x.reshape(-1)
        if self._rms is None:
            # 首帧：用观测本身初始化，避免第一步缩放失真
            self._rms = (x_flat ** 2).detach().clone()
        else:
            n = min(self._rms.numel(), x_flat.numel())
            if n:
                m = self.momentum
                self._rms[:n] = (
                    m * self._rms[:n] + (1.0 - m) * (x_flat[:n] ** 2).detach()
                )
            if x_flat.numel() > self._rms.numel():
                # 维度扩张（新信道展开）：补上新分量的初始估计
                extra = (x_flat[self._rms.numel():] ** 2).detach()
                self._rms = torch.cat([self._rms, extra])
        self._step += 1
        return self.normalize(x)

    def normalize(self, x: Tensor) -> Tensor:
        """归一化：原始物理单位 → 统一编码空间"""
        if self._rms is None:
            return x
        x_flat = x.reshape(-1)
        n = min(x_flat.numel(), self._rms.numel())
        scale = torch.sqrt(self._rms[:n]) + self.eps
        out = x_flat.clone()
        out[:n] = x_flat[:n] / scale
        return out.reshape(x.shape)

    def denormalize(self, x: Tensor) -> Tensor:
        """反归一化：统一编码空间 → 原始物理单位

        交给反射弧/肢体前必须调用，因为反射增益是在原始物理量纲下
        定义的（例如"位置误差 1 个单位 → 电机转 0.6 个单位"）。
        """
        if self._rms is None:
            return x
        x_flat = x.reshape(-1)
        n = min(x_flat.numel(), self._rms.numel())
        scale = torch.sqrt(self._rms[:n]) + self.eps
        out = x_flat.clone()
        out[:n] = x_flat[:n] * scale
        return out.reshape(x.shape)

    # ---------- 便捷接口 ----------

    def encode(self, x: Tensor) -> Tensor:
        """进入发育网络前调用：更新 RMS 并归一化"""
        return self.update(x)

    def decode(self, x: Tensor) -> Tensor:
        """离开发育网络、交给反射弧前调用：反归一化"""
        return self.denormalize(x)

    def report(self) -> dict[str, float]:
        """各信道的当前 RMS（诊断：尺度差异是否被抹平）"""
        if self._rms is None:
            return {}
        out = {}
        for ch, (offset, size) in self.port_layout.items():
            seg = self._rms[offset:offset + size]
            if seg.numel():
                out[ch] = float(torch.sqrt(seg.mean()).item())
        out["_step"] = float(self._step)
        return out
