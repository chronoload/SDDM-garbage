"""ServicePort：信号化服务（v15 协议——cordis Service → ServicePort）

ServicePort = Port 契约 + Signal 数据 + SessionEvent 持久化。
- 服务调用 = 信号对往返（RequestSignal → handle → ResponseSignal）
- 注册表替代 cordis ctx.get（ServicePortRegistry.get）
- 服务全部可接入事件流持久化（会话级活性上下文）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mcp.developmental.signal import Signal


@dataclass(frozen=True)
class RequestSignal(Signal):
    """请求信号：data + mime_type + 负载字典"""
    data: Any = None
    mime_type: str = "application/x.request"
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 服务信号用 payload 而非 tensor data，跳过 Signal 的张量转换
        pass


@dataclass(frozen=True)
class ResponseSignal(Signal):
    """响应信号：data + mime_type + 负载字典"""
    data: Any = None
    mime_type: str = "application/x.response"
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pass


class ServicePort:
    """服务端口基类：声明服务名 + 处理请求信号返回响应信号"""

    name: str = "unnamed"

    def handle(self, req: RequestSignal) -> ResponseSignal:
        """处理请求（子类实现）"""
        raise NotImplementedError

    def call(self, **payload) -> ResponseSignal:
        """便捷调用：构造 RequestSignal → handle → ResponseSignal"""
        return self.handle(RequestSignal(payload=payload))


class ServicePortRegistry:
    """服务端口注册表（替代 cordis ctx 服务发现）"""

    def __init__(self):
        self._services: Dict[str, ServicePort] = {}

    def register(self, service: ServicePort) -> None:
        self._services[service.name] = service

    def get(self, name: str) -> Optional[ServicePort]:
        return self._services.get(name)

    def call(self, name: str, **payload) -> ResponseSignal:
        svc = self.get(name)
        if svc is None:
            raise KeyError(f"service not registered: {name}")
        return svc.call(**payload)

    def names(self) -> list:
        return list(self._services.keys())
