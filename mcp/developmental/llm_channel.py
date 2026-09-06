"""符号信道：Agent 决策接 model_selector ServicePort + LLM 调用（复用 TS2 llm.py 接口形态）

Agent 脑 = MultiChannelSignal{符号信道(LLM) + 感知信道(SensoryOrgan) + 记忆信道(事件流)}。
- model_selector.resolve(session_id) → 模型名（对齐 TS2 mcp/model_selector.py 门面）
- llm.chat(model, messages) → 决策文本（对齐 TS2 mcp/llm.py chat 接口）
- 两者均为 ServicePort（信号化服务，可注入/可替换/可持久化审计）
"""
from __future__ import annotations
from typing import Callable

from mcp.developmental.service_port import ServicePortRegistry


def build_llm_decision_fn(registry: ServicePortRegistry,
                          llm_service: str = "llm") -> Callable[[dict], str]:
    """构造决策函数：resolve 模型 → llm.chat → 决策文本（符号信道）"""

    def decision_fn(payload: dict) -> str:
        # 1. 模型路由（model_selector ServicePort → 复用 TS2 model_selector.py 语义）
        sel = registry.call("model_selector", action="resolve",
                            session_id=payload.get("session_id", ""))
        model = sel.payload.get("model")
        # 2. LLM 调用（llm ServicePort → 复用 TS2 llm.py chat 形态；缺服务明确报错）
        llm = registry.get(llm_service)
        if llm is None:
            raise RuntimeError(f"llm service not registered: {llm_service}")
        messages = [{"role": "user",
                     "content": str(payload.get("text", payload))}]
        r = llm.call(model=model, messages=messages)
        return r.payload.get("content", "")

    return decision_fn
