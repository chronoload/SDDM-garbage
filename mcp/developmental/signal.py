"""Layer 1: 信号协议 — 统一数据载体，非裸张量"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import torch


@dataclass(frozen=True)
class Signal:
    """统一信号载体：数据 + 类型 + 元数据

    frozen=True 保证信号不可变，避免在路由传递中被意外修改
    """
    data: torch.Tensor
    mime_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.data, torch.Tensor):
            object.__setattr__(self, 'data', torch.as_tensor(self.data))

    def to(self, target_type: str, registry: ConverterRegistry) -> "Signal":
        """通过注册表转换类型"""
        if self.mime_type == target_type:
            return self
        conv = registry.find(self.mime_type, target_type)
        if conv is None:
            raise ValueError(f"No converter for {self.mime_type} → {target_type}")
        return conv.convert(self)


class SignalConverter(ABC):
    """信号类型转换器 — 用户可扩展"""

    @abstractmethod
    def can_convert(self, from_type: str, to_type: str) -> bool:
        """声明是否能处理该转换"""
        pass

    @abstractmethod
    def convert(self, signal: Signal) -> Signal:
        """执行转换"""
        pass


class ConverterRegistry:
    """转换器注册表"""

    def __init__(self):
        self._converters: list[SignalConverter] = []

    def register(self, converter: SignalConverter) -> None:
        self._converters.append(converter)

    def find(self, from_type: str, to_type: str) -> Optional[SignalConverter]:
        for conv in self._converters:
            if conv.can_convert(from_type, to_type):
                return conv
        return None


@dataclass(frozen=True)
class MultiChannelSignal:
    """多信道信号：把多个模态的信号拼接成统一向量

    符号信道与视觉/听觉信道同级，无特殊待遇。
    高层脑处理的是拼接后的统一向量，不区分信道来源。
    """
    channels: Dict[str, Signal]
    mime_type: str = "multi/channel"

    def __post_init__(self):
        # 拼接所有信道的数据
        parts = [s.data.flatten() for s in self.channels.values()]
        combined = torch.cat(parts) if parts else torch.zeros(0)
        object.__setattr__(self, '_data', combined)
        # 记录每信道的偏移
        offsets = {}
        offset = 0
        for name, sig in self.channels.items():
            offsets[name] = offset
            offset += sig.data.flatten().numel()
        object.__setattr__(self, 'channel_offsets', offsets)

    @property
    def data(self) -> torch.Tensor:
        return self._data

    def to_global_dim(self, global_dim: int) -> torch.Tensor:
        """对齐到 global_dim（不足补零，超出截断）"""
        flat = self._data
        n = flat.numel()
        if n < global_dim:
            pad = torch.zeros(global_dim - n)
            return torch.cat([flat, pad])
        elif n > global_dim:
            return flat[:global_dim]
        return flat
