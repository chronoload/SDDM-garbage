"""执行层：BattleUnit 任务单元 + 茧化（spec v16 P3）

PartyMember 执行真实任务：领任务 → 执行（事件流记录）→ 汇报 → 完成/茧化。
- BattleUnit：任务单元抽象（成员 + 任务列表 + 茧化终态）
- Task：任务（状态机 TODO → RUNNING → DONE）
- 茧化：终态（完成使命退出，活性上下文允许收束）
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional

from mcp.developmental.agent_registry import AgentPort
from mcp.developmental.organization import PartyMember
from mcp.developmental.tool_events import append_tool_call


class TaskStatus(Enum):
    TODO = "TODO"
    RUNNING = "RUNNING"
    DONE = "DONE"


class Task:
    """任务：状态机 TODO → RUNNING → DONE"""

    def __init__(self, task_id: str, title: str):
        self.task_id = task_id
        self.title = title
        self.status = TaskStatus.TODO
        self.result: Optional[str] = None

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING

    def finish(self, result: str) -> None:
        self.status = TaskStatus.DONE
        self.result = result


class BattleUnit:
    """战斗单元：成员执行真实任务的抽象"""

    def __init__(self, unit_id: str, member: PartyMember):
        self.unit_id = unit_id
        self.member = member  # PartyMember（agent 承载信箱）
        self.tasks: List[Task] = []
        self._task_seq = 0
        self.is_cocooned = False
        self.cocoon_reason: Optional[str] = None

    @property
    def agent(self) -> AgentPort:
        return self.member.agent

    def assign(self, title: str) -> Task:
        """领任务（茧化后拒绝新任务）"""
        if self.is_cocooned:
            raise RuntimeError(f"unit {self.unit_id} cocooned, cannot assign")
        self._task_seq += 1
        t = Task(f"{self.unit_id}-t{self._task_seq}", title)
        self.tasks.append(t)
        return t

    def execute(self, task_id: str, result: str) -> None:
        """执行：状态推进 + 事件流记录（工具事件带 task_id）"""
        t = self._find(task_id)
        t.mark_running()
        append_tool_call(self.agent.session, tool_name="task.execute",
                         args={"task_id": task_id, "title": t.title},
                         tool_call_id=task_id)
        t.finish(result)
        # 执行结果写入事件流（持久化可回放）
        self.agent.session.append({
            "kind": "tool.result",
            "payload": {"task_id": task_id, "result": result},
        })
        self.agent.session.flush(self.agent.store)

    def report(self, task_id: str) -> dict:
        """汇报：任务结果（写入事件流，可追溯）"""
        t = self._find(task_id)
        self.agent.session.append({
            "kind": "agent",
            "payload": {"action": "report", "task_id": task_id,
                        "result": t.result, "status": t.status.value},
        })
        self.agent.session.flush(self.agent.store)
        return {"task_id": task_id, "result": t.result, "status": t.status.value}

    def cocoon(self, reason: str) -> None:
        """茧化：终态（完成使命退出）"""
        self.is_cocooned = True
        self.cocoon_reason = reason
        self.agent.session.append({
            "kind": "mutation",
            "payload": {"action": "cocoon", "unit_id": self.unit_id, "reason": reason},
        })
        self.agent.session.flush(self.agent.store)

    def _find(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"task not found: {task_id}")
