"""Agent Actor 化：AgentPort + AgentRegistry + AgentFactory（dsh AgentRegistry/Factory 借鉴）

Agent = Actor（对象 + 信箱 + 消息循环）：
- AgentPort：独立会话事件流=信箱（agent:<id>），receive 消息 → 事件流 append → 符号信道
  决策（decision_fn 可注入，后续接 TS2 model_selector ServicePort）→ tool 事件 → 响应。
- AgentRegistry：单例注册表（按 agent_id，替代 cordis ctx.get）
- AgentFactory：工厂创建（dsh Factory 语义）
"""
from __future__ import annotations
from typing import Callable, Dict, Optional

from mcp.developmental.events import Session, SessionStore
from mcp.developmental.service_port import RequestSignal, ResponseSignal, ServicePort
from mcp.developmental.tool_events import append_tool_call


class AgentPort(ServicePort):
    """Agent 端口：信箱（会话事件流）+ 消息循环（receive → 决策 → 行动）"""

    def __init__(self, agent_id: str, store: SessionStore,
                 decision_fn: Optional[Callable[[dict], str]] = None):
        self.agent_id = agent_id
        self.store = store
        self.session = Session(f"agent:{agent_id}")
        self.session.seed(seed_event={"kind": "session.seed",
                                      "payload": {"agent_id": agent_id}})
        self.decision_fn = decision_fn or (lambda payload: "act:ack")

    @property
    def name(self) -> str:
        return self.agent_id

    def receive(self, **payload) -> dict:
        """接收消息：入信箱 → 决策 → 行动（感知-决策-行动闭环）"""
        # 1. 消息入信箱（user 事件）
        self.session.append({"kind": "user", "payload": payload})
        # 2. 符号信道决策（decision_fn 可注入；后续接 LLM via model_selector ServicePort）
        decision = self.decision_fn(payload)
        self.session.append({"kind": "agent", "payload": {"decision": decision}})
        # 3. 行动：tool 事件（决策含 act: 前缀 → 工具调用）
        if decision.startswith("act:"):
            tool_name = decision.split(":", 1)[1]
            append_tool_call(self.session, tool_name=tool_name,
                             args={"from": payload}, tool_call_id=f"tc-{self.session.seq}")
        # 4. 持久化
        self.session.flush(self.store)
        return {"decision": decision}

    def handle(self, req: RequestSignal) -> ResponseSignal:
        action = req.payload.get("action")
        if action == "receive":
            r = self.receive(**{k: v for k, v in req.payload.items() if k != "action"})
            return ResponseSignal(payload=r)
        if action == "load":
            return ResponseSignal(payload={"seq": self.session.seq})
        return ResponseSignal(payload={"error": f"unknown action: {action}"})


class AgentRegistry:
    """Agent 单例注册表（替代 cordis ctx 服务发现）"""

    def __init__(self):
        self._agents: Dict[str, AgentPort] = {}

    def register(self, agent: AgentPort) -> None:
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> Optional[AgentPort]:
        return self._agents.get(agent_id)

    def ids(self) -> list:
        return list(self._agents.keys())


class AgentFactory:
    """Agent 工厂：创建带独立信箱的 AgentPort（dsh Factory 语义）"""

    def __init__(self, store: SessionStore,
                 decision_fn: Optional[Callable[[dict], str]] = None):
        self.store = store
        self.decision_fn = decision_fn

    def create(self, agent_id: str) -> AgentPort:
        return AgentPort(agent_id=agent_id, store=self.store,
                         decision_fn=self.decision_fn)
