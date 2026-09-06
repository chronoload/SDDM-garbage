"""感知层：SensoryOrgan → Signal(感知信道) → PerceptionEvent（事件流持久化）

感知-记忆闭环（反射弧映射）：
  SensoryOrgan（感受器）→ Signal（神经冲动）→ append_perception（神经纤维持久化）
  → replay_perceptions（回放恢复）→ Agent 决策（中枢）
sensory 注册为 ServicePort（信号化服务，可经 Registry 调用）。
"""
from __future__ import annotations
from typing import List, Optional

from mcp.developmental.events import EventEnvelope, Session
from mcp.developmental.signal import Signal
from mcp.developmental.service_port import RequestSignal, ResponseSignal, ServicePort

_PERCEPTION_KIND = "perception"


class SensoryOrgan(ServicePort):
    """感知器官：读外部数据源 → Signal（mime_type=感知信道契约）"""

    name = "sensory"

    def __init__(self, source):
        self.source = source  # DataSource（read() → Signal）

    def sense(self) -> Signal:
        """读取一次感知信号（感受器触发）"""
        return self.source.read()

    def handle(self, req: RequestSignal) -> ResponseSignal:
        action = req.payload.get("action")
        if action == "sense":
            sig = self.sense()
            return ResponseSignal(payload={
                "mime_type": sig.mime_type,
                "frame": sig.metadata.get("frame"),
                "data": sig.data.tolist() if hasattr(sig.data, "tolist") else sig.data,
            })
        return ResponseSignal(payload={"error": f"unknown action: {action}"})


def append_perception(session: Session, signal: Signal,
                      channel: Optional[str] = None) -> EventEnvelope:
    """感知事件写入事件流（感知-记忆闭环的写入端，mime_type 保留信道契约）"""
    return session.append({
        "kind": _PERCEPTION_KIND,
        "payload": {"signal": signal, "channel": channel or signal.mime_type},
        "mime_type": signal.mime_type,
    })


def replay_perceptions(session: Session) -> List[Signal]:
    """感知-记忆回放：从事件流恢复感知信号序列（闭环读取端）"""
    out = []
    for e in session.events:
        if e.kind != _PERCEPTION_KIND:
            continue
        sig = e.payload.get("signal")
        if isinstance(sig, Signal):
            out.append(sig)
    return out
