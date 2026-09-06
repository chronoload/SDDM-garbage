"""党组织架构：PartyMember + PartyGroup + 上级组织 + 会议制度（spec v16 P1）

- PartyMember：Agent 的组织封装（身份/岗位/隶属/批评）——治理主体
- PartyGroup：聚合根（成员集 + 上级引用 + 会议）——组织单元
- 上级组织：组合模式（支部→上级→中央，parent/children）
- Meeting：会议制度（召集→议事→表决→民主集中多数决）
- 治理闭环：批评/会议事件写入成员事件流（事件溯源可追溯）
"""
from __future__ import annotations
from typing import Dict, List, Optional

from mcp.developmental.agent_registry import AgentPort


class Meeting:
    """会议：召集→议事→表决→民主集中（多数决）"""

    def __init__(self, group: "PartyGroup", agenda: str):
        self.group = group
        self.agenda = agenda
        self.items: List[dict] = []
        self.votes: Dict[str, str] = {}  # member_id -> 选项

    def add_item(self, topic: str) -> dict:
        item = {"topic": topic, "status": "讨论中"}
        self.items.append(item)
        return item

    def vote(self, member_id: str, option: str) -> None:
        if member_id not in self.group.member_ids():
            raise KeyError(f"member not in group: {member_id}")
        self.votes[member_id] = option

    def result(self) -> dict:
        """民主集中：多数决（赞成 > 反对 且 参与过半）"""
        total = len(self.group.members)
        voted = len(self.votes)
        counts = {}
        for v in self.votes.values():
            counts[v] = counts.get(v, 0) + 1
        yes = counts.get("赞成", 0)
        no = counts.get("反对", 0)
        adopted = voted >= (total + 1) // 2 and yes > no
        return {"adopted": adopted, "votes": dict(self.votes),
                "counts": counts, "voted": voted, "total": total}


class PartyMember:
    """党员：Agent 的组织封装（治理主体）"""

    def __init__(self, member_id: str, agent: AgentPort, role: str = "普通党员"):
        self.member_id = member_id
        self.agent = agent  # AgentPort（信箱=事件流）
        self.role = role
        self.group: Optional[PartyGroup] = None

    def join(self, group: "PartyGroup") -> None:
        group.add_member(self)

    def submit_critique(self, target_id: str, content: str) -> None:
        """自下而上批评：写入成员事件流（治理事件持久化可追溯）"""
        self.agent.session.append({
            "kind": "governance.critique",
            "payload": {"target_id": target_id, "content": content},
        })
        self.agent.session.flush(self.agent.store)


class PartyGroup:
    """组织单元（聚合根）：成员集 + 上级引用 + 会议"""

    def __init__(self, group_id: str, name: str,
                 parent: Optional["PartyGroup"] = None):
        self.group_id = group_id
        self.name = name
        self.parent = parent
        self.members: List[PartyMember] = []
        _register(self)  # 登记全局注册表（children 组合遍历用）

    def add_member(self, member: PartyMember) -> None:
        if member not in self.members:
            self.members.append(member)
            member.group = self

    def member_ids(self) -> List[str]:
        return [m.member_id for m in self.members]

    def children(self) -> List["PartyGroup"]:
        """下级组织（组合模式：上级可达下级）"""
        return [g for g in _ALL_GROUPS if g.parent is self]

    def convene_meeting(self, agenda: str) -> Meeting:
        """召集会议（会议制度入口）"""
        return Meeting(self, agenda)


# 全局组织注册表（组合模式遍历：children 依赖）
_ALL_GROUPS: List[PartyGroup] = []


def _register(group: PartyGroup) -> PartyGroup:
    if group not in _ALL_GROUPS:
        _ALL_GROUPS.append(group)
    return group
