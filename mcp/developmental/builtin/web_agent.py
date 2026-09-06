"""原子工具：WebAgent 符号处理信道

把 app-webagent（mcp.server.app._get_web_agent 管理的单例 Agent）暴露给发育智能系统。

两种集成模式：
  - HTTP 代理（默认，跨进程）：通过 POST /api/agent/chat 调用正在运行的 MCP server
    里的 webagent 实例。复用其完整状态（对话历史、工具、上下文注入器、WS2 系统）。
    需要 server 在跑。CLI/GUI 是独立进程时用这个。
  - local（同进程自建）：调用 mcp.agent.Agent 自己构造一个新实例。配置来源一致
    但实例独立，对话历史不与 server 共享。server 没跑时用这个。

输入：text/plain Signal（文本存于 metadata["raw"]，data 为 token ids）
输出：text/plain Signal（响应文本经分词器编码为 token ids）

## ⚠ 接口变更（从测试桩升级为完整实现，2026-09-03）

===================  ==============  ==============
项目                 测试桩（旧）      完整实现（现）
===================  ==============  ==============
输入键               ``query``        ``text``
输出键               ``result``       ``response``
构造参数             3 个              +session_id, +model_id
===================  ==============  ==============

**兼容性说明**：全包内无任何代码读取旧输入/输出键（已 grep 确认，
``execution.py`` 里的 "result" 是任务结果，与本工具无关）。升级安全。

注意 ``DEFAULT_SERVER_URL = "http://localhost:6906"`` 是**正确**的端口。
``gui.py`` 曾出现 ``690``（少一位）的拼写错误，见该文件修复记录。
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import torch

from mcp.developmental.reptilian import ReptilianFunction
from mcp.developmental.signal import Signal
from mcp.developmental.text_encoder import TextEncoder

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://localhost:6906"


class HTTPWebAgentProxy:
    """通过 HTTP 调用 MCP server 的 webagent 实例（跨进程复用）

    server 进程里的 _get_web_agent() 维护着一个单例 Agent（含对话历史、工具、
    上下文注入器、WS2 系统）。本类不持有那个 Agent，只通过 HTTP 转发 chat 请求。
    相当于一个"中间人"——developmental 系统发出的文本信号经此转发到 server，
    server 内的 webagent 处理后返回回复文本。

    session_id 用于在 server 端隔离不同 developmental 会话的对话历史（可选）。
    """

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        session_id: str = "",
        timeout: float = 300.0,
    ):
        self._server_url = server_url.rstrip("/")
        self._session_id = session_id
        self._timeout = timeout
        self._available: Optional[bool] = None  # 缓存 ping 结果

    def chat(self, message: str) -> str:
        """同步调用 server /api/agent/chat，返回回复文本"""
        import requests
        try:
            resp = requests.post(
                f"{self._server_url}/api/agent/chat",
                json={
                    "message": message,
                    "session_id": self._session_id,
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("code") != 0:
                return f"[WebAgent HTTP 错误] server 返回 code={payload.get('code')}"
            data = payload.get("data") or {}
            reply = data.get("reply", "")
            source = data.get("source", "")
            if source == "error":
                return f"[WebAgent server 端错误] {reply}"
            self._available = True
            return reply
        except requests.exceptions.ConnectionError:
            self._available = False
            return (f"[WebAgent 连接失败] server 未运行或地址错误: "
                    f"{self._server_url}/api/agent/chat")
        except Exception as e:
            self._available = False
            return f"[WebAgent HTTP 异常] {e}"

    def ping(self) -> bool:
        """探测 server 是否可达（GET /api/health 或根路径）"""
        import requests
        try:
            # 试几个可能的健康检查端点
            for path in ["/api/health", "/health", "/"]:
                try:
                    r = requests.get(
                        f"{self._server_url}{path}",
                        timeout=5.0,
                    )
                    if r.status_code < 500:
                        self._available = True
                        return True
                except Exception:
                    continue
            self._available = False
            return False
        except Exception:
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self.ping()
        return bool(self._available)

    @property
    def server_url(self) -> str:
        return self._server_url

    # 兼容 Agent 接口的桩（避免外部代码访问 .tools 等属性时崩溃）
    @property
    def tools(self):
        return []

    def register_context_injector(self, *args, **kwargs):
        """HTTP 模式下上下文注入器在 server 端，本地忽略"""
        pass


def _make_adapter(mgr, model_id: str):
    """构造统一 chat 接口的 LLM 适配器（消除原实现中的重复类定义）"""

    class _AdapterLLM:
        total_prompt_tokens = 0
        total_completion_tokens = 0

        def __init__(self, m):
            self._mgr = m

        def chat(self, messages, tools=None, on_token=None):
            resp = self._mgr.chat_with_fallback(messages, tools, on_token)
            self.total_prompt_tokens += resp.prompt_tokens
            self.total_completion_tokens += resp.completion_tokens
            return resp

        def is_available(self):
            return self._mgr.get_provider() is not None

    return _AdapterLLM(mgr)


def _build_local_llm():
    """构造本地 LLM 实例（local 模式用）

    返回 (llm, model_id) 元组。失败时返回 (None, "")。
    """
    # 1. 尝试从配置管理器加载
    try:
        from mcp.config import get_config_manager
        from mcp.llm import MultiProviderManager, SimulatorLLM

        config_mgr = get_config_manager()
        provider_configs = config_mgr.get_provider_configs_for_manager()
        enabled = [
            cfg for cfg in provider_configs
            if cfg.enabled and cfg.provider.value != 'simulator'
        ]
        if enabled:
            raw_llm = MultiProviderManager(enabled)
            model_id = enabled[0].model or ""
            return _make_adapter(raw_llm, model_id), model_id
    except Exception as e:
        logger.debug(f"配置管理器加载失败，尝试环境变量: {e}")

    # 2. 尝试从环境变量加载
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
    model_id = os.environ.get("TS2_MODEL_ID", "gpt-4o-mini")

    if api_key:
        try:
            from mcp.llm import MultiProviderManager, ProviderConfig, ProviderType

            env_config = ProviderConfig(
                provider=ProviderType.OPENAI_COMPATIBLE,
                name="env-openai",
                api_key=api_key,
                base_url=base_url or None,
                model_id=model_id,
                enabled=True,
            )
            raw_llm = MultiProviderManager([env_config])
            return _make_adapter(raw_llm, model_id), model_id
        except Exception as e:
            logger.warning(f"环境变量 LLM 初始化失败: {e}")

    # 3. 兜底：模拟器
    try:
        from mcp.llm import SimulatorLLM
        logger.warning("WebAgentFunction(local): 未找到 LLM 配置，使用模拟模式")
        return SimulatorLLM(), ""
    except Exception:
        logger.error("SimulatorLLM 也不可用")
        return None, ""


def _build_local_agent(workspace_dir: str, model_id: str = ""):
    """构造本地 Agent 实例（local 模式用）"""
    from pathlib import Path
    from mcp.agent import Agent, AgentConfig

    llm, llm_model_id = _build_local_llm()
    if llm is None:
        raise RuntimeError("无法构造 LLM（配置和环境变量均不可用）")

    ws2_system = None
    try:
        import sys as _sys
        _project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)
        from course_tracker import CourseSystem
        ws2_system = CourseSystem()
    except Exception as e:
        logger.debug(f"WS2 系统初始化跳过: {e}")

    agent_config = AgentConfig(
        name="Developmental WebAgent (local)",
        base_dir=Path(workspace_dir),
        workspace_root=workspace_dir,
        mode="act",
        model_id=model_id or llm_model_id,
        ws2_system=ws2_system,
    )
    return Agent(llm=llm, config=agent_config)


class WebAgentFunction(ReptilianFunction):
    """WebAgent 符号处理工具 — 暴露 app-webagent 给发育智能系统

    模式：
      mode="http"（默认）：HTTP 代理，调用运行中的 MCP server 的 webagent 实例
      mode="local"：自己构造一个新的 Agent 实例

    输入：text/plain Signal（文本存于 metadata["raw"]，data 为 token ids）
    输出：text/plain Signal（响应文本经分词器编码为 token ids）
    """

    def __init__(
        self,
        mode: str = "http",
        server_url: str = DEFAULT_SERVER_URL,
        session_id: str = "",
        workspace_dir: Optional[str] = None,
        model_id: str = "",
    ):
        if mode not in ("http", "local"):
            raise ValueError(f"mode 必须是 'http' 或 'local'，得到: {mode}")
        self._mode = mode
        self._server_url = server_url
        self._session_id = session_id
        self._workspace_dir = workspace_dir or os.getcwd()
        self._model_id = model_id
        self._agent = None  # HTTPWebAgentProxy 或 Agent 实例
        self._init_error: Optional[str] = None
        self._encoder = TextEncoder.instance()

    def _ensure_agent(self) -> None:
        """懒加载 agent（首次调用时初始化）"""
        if self._agent is not None:
            return
        if self._init_error:
            raise RuntimeError(f"WebAgent 之前初始化失败：{self._init_error}")

        try:
            if self._mode == "http":
                self._agent = HTTPWebAgentProxy(
                    server_url=self._server_url,
                    session_id=self._session_id,
                )
                # 探测 server 可达性（不阻塞，失败也允许后续 chat 时再报错）
                available = self._agent.ping()
                if available:
                    logger.info(
                        f"WebAgentFunction(http) 已连接 server: {self._server_url} "
                        f"(session_id={self._session_id or '默认'})"
                    )
                else:
                    logger.warning(
                        f"WebAgentFunction(http) server 不可达: {self._server_url} "
                        f"—— 后续 chat 会返回连接错误，请启动 MCP server 或切换 mode=local"
                    )
            else:  # local
                self._agent = _build_local_agent(self._workspace_dir, self._model_id)
                logger.info(
                    f"WebAgentFunction(local) 已初始化 "
                    f"(工具数={len(self._agent.tools)})"
                )
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"WebAgentFunction 初始化失败: {e}")
            raise

    def get_input_spec(self):
        return {"text": "text/plain"}

    def get_output_spec(self):
        return {"response": "text/plain"}

    def execute(self, inputs):
        text_sig = inputs["text"]
        # 文本内容优先从 metadata["raw"] 取，fallback 用 encoder 解码 data
        raw_text = text_sig.metadata.get("raw", "")
        if not raw_text:
            raw_text = self._encoder.decode(text_sig.data)

        try:
            self._ensure_agent()
            # HTTPWebAgentProxy.chat 和 Agent.chat 接口不同：
            # - HTTPWebAgentProxy.chat(message: str) -> str
            # - Agent.chat(message: str) -> str（同步）
            response_text = self._agent.chat(raw_text)
        except Exception as e:
            response_text = f"[WebAgent 错误] {e}"
            logger.error(f"WebAgent chat 失败: {e}")

        # 用分词器编码响应文本 → token id 张量（非占位符）
        response_tokens = self._encoder.encode(response_text)
        return {
            "response": Signal(
                data=response_tokens,
                mime_type="text/plain",
                metadata={
                    "raw": response_text,
                    "source": f"webagent:{self._mode}",
                    "input": raw_text[:200],
                    "n_tokens": len(response_tokens),
                },
            )
        }

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def server_url(self) -> str:
        return self._server_url
