"""事件流核心：SessionEvent 事件信封 + 状态机 + Session/SessionStore（复现 dsh session 语义）

v15 协议：事件信封(seq/version/ignorable + mime_type 负载) = 活性上下文时间轴 × 信道矩阵。
- ignorable 守卫：未知事件类型安全放行（协议可演进）
- open-step 状态机：closed 会话禁止 agent/tool 追加（dsh invariant.applyTransition）
- SessionStore 分块持久化 + load 重放 + fork 分叉（活性上下文可恢复/可分叉）
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

SESSION_FORMAT_VERSION = 1

# 已知事件类型词汇表（dsh 44 词收敛为组织场景核心集，可扩展）
KNOWN_KINDS = {
    "session.seed", "user", "agent", "tool", "tool.result",
    "perception", "signal", "organize.meeting", "organize.vote",
    "governance.critique", "supervise.patrol", "mutation",
}

# 结构化事件默认 mime_type（信道契约，超模态负载走 image/* audio/* 经 ConverterRegistry）
_DEFAULT_MIME = "application/x.event"


class EventError(Exception):
    """事件协议错误（未知类型 / 状态机违规）"""


@dataclass(frozen=True)
class EventEnvelope:
    """事件信封：seq 递增 + version + ignorable 守卫 + mime_type 负载契约"""
    seq: int
    kind: str
    payload: Any = None
    version: int = SESSION_FORMAT_VERSION
    ignorable: bool = False
    mime_type: str = _DEFAULT_MIME


class SessionState(Enum):
    OPEN = "open"
    CLOSED = "closed"


def validate_event(event: dict) -> None:
    """未知事件类型必须拒绝；ignorable 事件安全放行（协议可演进）"""
    kind = event.get("kind", "")
    if kind in KNOWN_KINDS:
        return
    if event.get("ignorable"):
        return
    raise EventError(f"unknown event kind: {kind}")


def apply_transition(state: SessionState, kind: str) -> SessionState:
    """open-step 守卫：closed 状态不允许 agent/tool 事件追加（dsh invariant 语义）"""
    if state == SessionState.CLOSED and kind in ("agent", "tool", "tool.result"):
        raise EventError(f"cannot append {kind} to closed session")
    return state


class Session:
    """事件流会话：seed 建流 → append 追加（seq 连续）→ flush 持久化"""

    def __init__(self, sid: str):
        self.sid = sid
        self.seq = 0
        self.state = SessionState.OPEN
        self.events: List[EventEnvelope] = []

    def seed(self, seed_event: dict) -> EventEnvelope:
        """初始化事件流（session.seed 事件，seq=1）"""
        return self._append(seed_event)

    def append(self, event: dict) -> EventEnvelope:
        """追加事件（校验 + 状态机 + 信封封装）"""
        return self._append(event)

    def _append(self, event: dict) -> EventEnvelope:
        validate_event(event)
        kind = event.get("kind", "")
        self.state = apply_transition(self.state, kind)
        self.seq += 1
        env = EventEnvelope(
            seq=self.seq,
            kind=kind,
            payload=event.get("payload"),
            ignorable=bool(event.get("ignorable", False)),
            mime_type=event.get("mime_type", _DEFAULT_MIME),
        )
        self.events.append(env)
        return env

    def flush(self, store: "SessionStore") -> None:
        store.save(self)

    def snapshot(self) -> dict:
        return {"sid": self.sid, "seq": self.seq,
                "state": self.state.value,
                "events": [self._row(e) for e in self.events]}

    @staticmethod
    def _row(e: EventEnvelope) -> dict:
        return {"seq": e.seq, "kind": e.kind, "payload": e.payload,
                "version": e.version, "ignorable": e.ignorable,
                "mime_type": e.mime_type}

    @classmethod
    def restore(cls, sid: str, rows: List[dict]) -> "Session":
        """从持久化行重放恢复（活性上下文复活）"""
        s = cls(sid)
        for r in rows:
            s.seq = r["seq"]
            s.events.append(EventEnvelope(
                seq=r["seq"], kind=r["kind"], payload=r.get("payload"),
                ignorable=bool(r.get("ignorable", False)),
                mime_type=r.get("mime_type", _DEFAULT_MIME),
            ))
        return s


class SessionStore:
    """会话存储：分块持久化（每事件一行，chunk_size 预留合并）+ load 重放 + fork 分叉"""

    def __init__(self, chunk_size: int = 32):
        self._data: Dict[str, List[dict]] = {}
        self.chunk_size = chunk_size

    def save(self, session: Session) -> None:
        self._data[session.sid] = [session._row(e) for e in session.events]

    def _rows(self, sid: str) -> List[dict]:
        return self._data.get(sid, [])

    def load(self, sid: str) -> Session:
        if sid not in self._data:
            raise KeyError(f"session not found: {sid}")
        return Session.restore(sid, self._data[sid])

    def fork(self, sid: str, new_sid: str) -> Session:
        """分叉会话（dsh SessionForkError 语义：fork 即复制事件流起点）"""
        rows = self._data.get(sid, [])
        s = Session.restore(new_sid, rows)
        self._data[new_sid] = rows
        return s
