"""工具事件写入：append_tool_call / append_tool_result（dsh agent-loop/tool-calls.ts 语义）

事件流工具轨迹：tool_call_id 关联调用与结果，写入 Session 事件流（持久化可重放）。
"""
from __future__ import annotations
from typing import Any

from mcp.developmental.events import EventEnvelope, Session


def append_tool_call(session: Session, tool_name: str, args: dict,
                     tool_call_id: str) -> EventEnvelope:
    """追加 tool 调用事件（dsh appendToolCall 语义）"""
    return session.append({
        "kind": "tool",
        "payload": {"tool_call_id": tool_call_id, "name": tool_name, "args": args},
    })


def append_tool_result(session: Session, tool_call_id: str, result: Any,
                       is_error: bool = False) -> EventEnvelope:
    """追加 tool 结果事件（tool_call_id 关联回调用，dsh appendToolResult 语义）"""
    return session.append({
        "kind": "tool.result",
        "payload": {"tool_call_id": tool_call_id, "result": result, "is_error": is_error},
    })
