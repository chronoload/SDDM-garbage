"""Layer 2: 端口抽象 — 系统边界的信号端点

端口不关心信号内容含义，只负责读写和类型契约。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import uuid

from mcp.developmental.signal import Signal


class DataSource(ABC):
    """任何能产生信号的外部实体（用户可扩展）"""

    @abstractmethod
    def read(self) -> Signal:
        pass


class DataSink(ABC):
    """任何能消费信号的外部实体（用户可扩展）"""

    @abstractmethod
    def write(self, signal: Signal) -> None:
        pass


class Port(ABC):
    """端口基类：系统边界上的信号端点"""

    @abstractmethod
    def get_signal(self) -> Signal:
        """读取信号"""
        pass

    @abstractmethod
    def put_signal(self, signal: Signal) -> None:
        """写入信号"""
        pass

    @abstractmethod
    def get_accepted_types(self) -> List[str]:
        """声明接受的 MIME 类型"""
        pass


class InputPort(Port):
    """入端口 — 连接外部数据源，只读"""

    def __init__(self, source: DataSource, accepted_types: List[str]):
        self.source = source
        self._accepted_types = accepted_types

    def get_signal(self) -> Signal:
        return self.source.read()

    def put_signal(self, signal: Signal) -> None:
        raise RuntimeError("Input ports cannot be written to")

    def get_accepted_types(self) -> List[str]:
        return self._accepted_types


class OutputPort(Port):
    """出端口 — 连接外部数据宿，只写"""

    def __init__(self, sink: DataSink, accepted_types: List[str]):
        self.sink = sink
        self._accepted_types = accepted_types

    def get_signal(self) -> Signal:
        raise RuntimeError("Output ports cannot be read from")

    def put_signal(self, signal: Signal) -> None:
        if signal.mime_type not in self._accepted_types:
            raise TypeError(
                f"Port accepts {self._accepted_types}, got {signal.mime_type}"
            )
        self.sink.write(signal)

    def get_accepted_types(self) -> List[str]:
        return self._accepted_types


class PortRegistry:
    """端口注册表 — 管理所有端口的 ID 映射"""

    def __init__(self):
        self._ports: dict[str, Port] = {}
        self._names: dict[str, str] = {}  # name → port_id

    def register(self, port: Port, name: str | None = None) -> str:
        """注册端口，返回 port_id"""
        port_id = f"port_{uuid.uuid4().hex[:8]}"
        self._ports[port_id] = port
        if name:
            self._names[name] = port_id
        return port_id

    def get(self, port_id: str) -> Port | None:
        return self._ports.get(port_id)

    def get_by_name(self, name: str) -> Port | None:
        port_id = self._names.get(name)
        return self._ports.get(port_id) if port_id else None

    def get_input_ports(self) -> list[InputPort]:
        return [p for p in self._ports.values() if isinstance(p, InputPort)]

    def get_output_ports(self) -> list[OutputPort]:
        return [p for p in self._ports.values() if isinstance(p, OutputPort)]

    def all_ports(self) -> list[Port]:
        return list(self._ports.values())

    def iter_entries(self) -> list[tuple[str, Port]]:
        """返回 (port_id, port) 对列表 — 主循环按 port_id 索引信号池时使用"""
        return list(self._ports.items())
