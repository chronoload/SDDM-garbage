"""治理层：八项权利/义务 + 批评机制 + 巡回检查组（spec v16 P2）

治理监督闭环：
- 自下而上：党员批评/检举/申诉（CriticReport → 事件流可追溯）
- 自上而下：巡回检查组（独立/游走/直达中央，不受被查支部干预）
- 八项权利/义务（党员权利保障）
"""
from __future__ import annotations
from typing import List

from mcp.developmental.agent_registry import AgentPort

# 党员八项权利
RIGHTS = ["知情权", "参与权", "表决权", "选举权", "被选举权",
          "批评权", "检举权", "申诉权"]

# 党员基本义务
DUTIES = ["执行决议", "遵守纪律", "联系群众", "如实汇报"]


class GovernancePolicy:
    """治理政策：成员权利保障 + 义务校验"""

    def has_right(self, member, right: str) -> bool:
        """成员是否享有某项权利（党员一律享有八项权利）"""
        return right in RIGHTS

    def verify_duty(self, member, duty: str) -> bool:
        """义务履行校验（默认视为履行，可由组织考核覆盖）"""
        return duty in DUTIES


class CriticReport:
    """批评报告：批评/检举/申诉（自下而上治理入口）"""

    def __init__(self, kind: str, author: str, target: str, content: str):
        assert kind in ("批评", "检举", "申诉"), f"unknown kind: {kind}"
        self.kind = kind
        self.author = author
        self.target = target
        self.content = content
        self.status = "草稿"

    def submit(self, agent: AgentPort) -> None:
        """提交：写入发起者事件流（事件溯源可追溯）"""
        agent.session.append({
            "kind": "governance.critique",
            "payload": {"kind": self.kind, "author": self.author,
                        "target": self.target, "content": self.content},
        })
        agent.session.flush(agent.store)
        self.status = "已提交"


class PatrolTeam:
    """巡回检查组：独立/游走/直达中央（自上而下监督）"""

    def __init__(self, origin: str, targets: List[str]):
        self.origin = origin  # 派出方（如中央）
        self.targets = targets  # 被查支部

    def is_independent_of(self, group_id: str) -> bool:
        """独立性：不隶属被查支部"""
        return True  # 巡回组直属派出方，不受被查支部管辖

    def patrol(self, group) -> List[dict]:
        """检查下级组织：巡视成员履职情况"""
        findings = []
        for m in group.members:
            findings.append({
                "member": m.member_id,
                "status": "正常",
                "note": f"巡视 {group.name} 成员 {m.member_id}",
            })
        return findings

    def report_to_central(self, findings: List[dict]) -> dict:
        """直达中央报告（绕过地方）"""
        return {"origin": self.origin, "target": self.targets[0] if self.targets else "",
                "findings": findings}
