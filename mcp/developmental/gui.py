"""发育智能系统 GUI 控制面板 — Textual 终端 GUI

启动：
    python -m mcp.developmental.gui

布局：
    ┌─────────────────┬─────────────────────────┐
    │  反射弧编排      │      脑状态观察          │
    │  (工具/路由)     │  (神经元/髓鞘/tick/λ)   │
    ├─────────────────┼─────────────────────────┤
    │  运行控制        │      日志/输出          │
    │  (start/step/   │  (tick 输出/事件流)     │
    │   inject/save)  │                         │
    └─────────────────┴─────────────────────────┘
"""
from __future__ import annotations

import torch
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import (
    Header, Footer, Button, Static, Input, RichLog
)
from textual.reactive import reactive
from textual.binding import Binding

from mcp.developmental.signal import Signal
from mcp.developmental.system import DevelopmentalSystem, SystemMode
from mcp.developmental.reptilian import EchoFunction, SaturateFunction
from mcp.developmental.builtin.web_agent import WebAgentFunction
from mcp.developmental.text_encoder import TextEncoder
from mcp.developmental.organize import (
    Organizer,
    QueueSensoryOrgan,
    ReflexAssembly,
)


BUILTIN_TOOLS = {
    "echo": lambda: EchoFunction(),
    "saturate": lambda: SaturateFunction(-1.0, 1.0),
}


class BrainStatusPanel(Static):
    """脑状态观察面板"""

    def update_status(self, brain: DevelopmentalSystem, organizer: Organizer) -> None:
        eco = brain.higher_brain.ecosystem
        reg = brain.higher_brain.sheath_registry
        engine = brain.higher_brain.engine
        mode = "觉醒" if organizer.is_awake() else "睡眠"
        last_lambda = organizer.last_result.get("lambda", 0) if organizer.last_result else 0

        self.update(
            f"[bold cyan]=== 脑状态 ===[/]\n"
            f"  阶段: [yellow]{brain.phase}[/]\n"
            f"  模式: [{'green' if mode == '觉醒' else 'magenta'}]{mode}[/]\n"
            f"  step_count: {brain.step_count}\n"
            f"  tick: {organizer.tick_count}\n"
            f"  神经元: [cyan]{eco.count()}[/]\n"
            f"  髓鞘: [cyan]{len(reg._sheaths)}[/]\n"
            f"  新奇性: {engine._novelty_accumulator:.4f} / {engine.novelty_threshold}\n"
            f"  分化次数: {engine._differentiation_count}\n"
            f"  轨迹日志: {len(brain._trajectory_log)}\n"
            f"  反射弧: {len(organizer.brain.reflex._routes)} 条\n"
            f"  工具: {len(brain.reptile.list_functions())} 个\n"
            f"  上次 λ: [yellow]{last_lambda:.4f}[/]"
        )


class DevelopmentalGUI(App):
    """发育智能系统 Textual 终端 GUI"""

    TITLE = "发育智能系统控制面板"
    CSS = """
    Screen {
        layout: vertical;
    }
    #main-row {
        height: 1fr;
        layout: horizontal;
    }
    #left-col, #right-col {
        width: 1fr;
        layout: vertical;
    }
    #assembly-panel { border: round $accent; height: 1fr; }
    #status-panel { border: round $success; height: 1fr; }
    #control-panel { border: round $warning; height: 1fr; }
    #log-panel { border: round $primary; height: 1fr; }
    .btn-row { layout: horizontal; height: 3; }
    Button { margin: 0 1; }
    RichLog { height: 1fr; }
    .panel-title { background: $accent 20%; }
    """

    BINDINGS = [
        Binding("q", "quit", "退出"),
        Binding("s", "step", "单步"),
        Binding("r", "run_toggle", "运行/暂停"),
        Binding("a", "awake", "觉醒"),
        Binding("z", "sleep", "睡眠"),
    ]

    running = reactive(False)

    def __init__(self):
        super().__init__()
        # 构造默认系统
        self.brain = DevelopmentalSystem(port_layout={"input": (0, 4), "y": (4, 4)})
        self.brain.phase = "exploratory"
        self.brain.higher_brain.ecosystem.get_neuron(0).unfold("input", torch.zeros(4))

        self.assembly = ReflexAssembly(self.brain)
        self.sensory = QueueSensoryOrgan(channel="input", mime_type="generic/tensor")
        self.text_sensory = QueueSensoryOrgan(channel="text", mime_type="text/plain")
        self.organizer = Organizer(self.brain)
        self.organizer.add_sensory(self.sensory)
        self.organizer.add_sensory(self.text_sensory)
        self._encoder = TextEncoder.instance()
        self._webagent_mode = "http"  # http | local
        self._webagent_server_url = "http://localhost:6906"  # 修复: 此前少一位(690)，与 web_agent.DEFAULT_SERVER_URL 不一致

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-row"):
            with Vertical(id="left-col"):
                with Vertical(id="assembly-panel"):
                    yield Static("[bold]反射弧编排[/]", classes="panel-title")
                    yield Static("工具：echo, saturate, webagent\n路由：source → tool", id="assembly-info")
                    with Horizontal(classes="btn-row"):
                        yield Button("add echo", id="btn-add-echo")
                        yield Button("add saturate", id="btn-add-saturate")
                        yield Button("add webagent", id="btn-add-webagent")
                    with Horizontal(classes="btn-row"):
                        yield Button("route input→echo", id="btn-route-input-echo")
                        yield Button("route y→saturate", id="btn-route-y-sat")
                    with Horizontal(classes="btn-row"):
                        yield Button("route text→webagent", id="btn-route-text-webagent")
                    yield Input(placeholder="inject: 0.5 0.1 0.0 0.0", id="inject-input")
                    yield Input(placeholder="text: 输入文本（经分词器编码）", id="text-input")
                    yield Input(placeholder="webagent server URL（http 模式）", id="server-url-input", value="http://localhost:6906")
                    with Horizontal(classes="btn-row"):
                        yield Button("切 webagent→http", id="btn-webagent-http")
                        yield Button("切 webagent→local", id="btn-webagent-local")
                with Vertical(id="control-panel"):
                    yield Static("[bold]运行控制[/]", classes="panel-title")
                    with Horizontal(classes="btn-row"):
                        yield Button("单步 (s)", id="btn-step")
                        yield Button("运行 (r)", id="btn-run")
                    with Horizontal(classes="btn-row"):
                        yield Button("觉醒 (a)", id="btn-awake")
                        yield Button("睡眠 (z)", id="btn-sleep")
                    with Horizontal(classes="btn-row"):
                        yield Button("保存", id="btn-save")
                        yield Button("加载", id="btn-load")
                    yield Input(placeholder="save/load 路径", id="path-input")
            with Vertical(id="right-col"):
                with Vertical(id="status-panel"):
                    yield BrainStatusPanel(id="status-display")
                with Vertical(id="log-panel"):
                    yield Static("[bold]日志/输出[/]", classes="panel-title")
                    yield RichLog(id="event-log")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_status()
        self._log("系统已启动，按 s 单步，r 运行/暂停，q 退出")

    def _log(self, msg: str) -> None:
        self.query_one("#event-log", RichLog).write(msg)

    def _refresh_status(self) -> None:
        self.query_one("#status-display", BrainStatusPanel).update_status(self.brain, self.organizer)

    def _refresh_assembly(self) -> None:
        tools = self.brain.reptile.list_functions()
        routes = self.assembly.list_routes()
        tools_str = ", ".join(tools) if tools else "(无)"
        routes_str = "\n".join(
            f"  {r['source_channel']} → {r['tool_name']} ({r['priority']})"
            for r in routes
        ) if routes else "  (无)"
        self.query_one("#assembly-info", Static).update(
            f"[bold]工具:[/] {tools_str}\n[bold]路由:[/]\n{routes_str}"
        )

    # === 按钮事件 ===

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-add-echo":
            self.assembly.register_tool("echo", EchoFunction())
            self._log("✓ 工具已注册：echo")
            self._refresh_assembly()
        elif btn_id == "btn-add-saturate":
            self.assembly.register_tool("saturate", SaturateFunction(-1.0, 1.0))
            self._log("✓ 工具已注册：saturate")
            self._refresh_assembly()
        elif btn_id == "btn-add-webagent":
            # 用当前模式 + URL 输入框的值构造 webagent
            try:
                server_url_input = self.query_one("#server-url-input", Input)
                url = server_url_input.value.strip() or "http://localhost:6906"
            except Exception:
                url = self._webagent_server_url
            self._webagent_server_url = url
            func = WebAgentFunction(
                mode=self._webagent_mode,
                server_url=url,
            )
            self.assembly.register_tool("webagent", func)
            mode_desc = (f"HTTP 代理 → {url}" if self._webagent_mode == "http"
                         else "local（自建 Agent）")
            self._log(f"✓ 工具已注册：webagent（模式: {mode_desc}，分词器: {self._encoder.backend}）")
            self._refresh_assembly()
        elif btn_id == "btn-webagent-http":
            self._webagent_mode = "http"
            self._log(f"✓ webagent 模式切换为 http（复用运行中的 MCP server 实例）")
        elif btn_id == "btn-webagent-local":
            self._webagent_mode = "local"
            self._log(f"✓ webagent 模式切换为 local（自建新 Agent，不依赖 server）")
        elif btn_id == "btn-route-input-echo":
            if "echo" in self.brain.reptile.list_functions():
                self.assembly.add_route("input", "echo", 1.0)
                self._log("✓ 路由：input → echo")
                self._refresh_assembly()
            else:
                self._log("✗ 先添加 echo 工具")
        elif btn_id == "btn-route-y-sat":
            if "saturate" in self.brain.reptile.list_functions():
                self.assembly.add_route("y", "saturate", 0.5)
                self._log("✓ 路由：y → saturate")
                self._refresh_assembly()
            else:
                self._log("✗ 先添加 saturate 工具")
        elif btn_id == "btn-route-text-webagent":
            if "webagent" in self.brain.reptile.list_functions():
                self.assembly.add_route("text", "webagent", 1.0)
                self._log("✓ 路由：text → webagent")
                self._refresh_assembly()
            else:
                self._log("✗ 先添加 webagent 工具")
        elif btn_id == "btn-step":
            self._do_step()
        elif btn_id == "btn-run":
            self.running = not self.running
            if self.running:
                self._log("▶ 运行中（r 暂停）")
                self._do_run()
            else:
                self._log("⏸ 已暂停")
        elif btn_id == "btn-awake":
            self.organizer.force_awake()
            self._log("✓ 强制觉醒")
            self._refresh_status()
        elif btn_id == "btn-sleep":
            self.organizer.force_sleep()
            self._log("✓ 强制睡眠")
            self._refresh_status()
        elif btn_id == "btn-save":
            path = self.query_one("#path-input", Input).value or "./brain_state"
            try:
                self.brain.save_state(path)
                self._log(f"✓ 脑状态已保存到 {path}")
            except Exception as e:
                self._log(f"✗ 保存失败：{e}")
        elif btn_id == "btn-load":
            path = self.query_one("#path-input", Input).value or "./brain_state"
            try:
                self.brain.load_state(path)
                self._log(f"✓ 脑状态已从 {path} 加载")
                self._refresh_status()
            except Exception as e:
                self._log(f"✗ 加载失败：{e}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "inject-input":
            try:
                vals = [float(v) for v in event.value.split()]
                self.sensory.push(Signal(torch.tensor(vals), "generic/tensor"))
                self._log(f"✓ 张量信号推入 input：{vals}")
                event.input.value = ""
            except ValueError:
                self._log("✗ 参数必须是数字")
        elif event.input.id == "text-input":
            text = event.value.strip()
            if text:
                token_ids = self._encoder.encode(text)
                self.text_sensory.push(Signal(
                    data=token_ids,
                    mime_type="text/plain",
                    metadata={"raw": text, "source": "gui", "n_tokens": len(token_ids)},
                ))
                self._log(f"✓ 文本信号推入 text（{len(token_ids)} tokens, {self._encoder.backend}）：{text[:60]}")
                event.input.value = ""
        elif event.input.id == "server-url-input":
            url = event.value.strip()
            if url:
                self._webagent_server_url = url
                self._log(f"✓ webagent server URL 已更新：{url}（下次 add webagent 生效）")

    # === 运行逻辑 ===

    def _do_step(self) -> None:
        result = self.organizer.tick()
        outputs = result.get("outputs") or result.get("sleep_outputs", {})
        mode = "觉醒" if self.organizer.is_awake() else "睡眠"
        if outputs:
            parts = []
            for k, v in outputs.items():
                if v.mime_type.startswith("text/"):
                    raw = v.metadata.get("raw", "")
                    if raw:
                        preview = raw[:80].replace("\n", " ")
                        parts.append(f'{k}="{preview}{"..." if len(raw) > 80 else ""}"')
                    else:
                        parts.append(f"{k}={v.data.tolist()}")
                else:
                    parts.append(f"{k}={v.data.tolist()}")
            out_str = ", ".join(parts)
            self._log(f"  [{mode}] λ={result.get('lambda', 0):.2f} | {out_str}")
        else:
            phase = ""
            if "homeostasis" in result: phase = "稳态缩放"
            elif "replay" in result: phase = "重放"
            elif "counterfactual" in result: phase = "反事实"
            self._log(f"  [{mode}] λ={result.get('lambda', 0):.2f} | {phase or '无输出'}")
        self._refresh_status()

    def _do_run(self) -> None:
        """连续单步（用定时器驱动，不阻塞 UI）"""
        if self.running:
            self._do_step()
            self.set_timer(0.1, self._do_run)

    # === 键盘快捷键 ===

    def action_step(self) -> None:
        self._do_step()

    def action_run_toggle(self) -> None:
        self.running = not self.running
        if self.running:
            self._log("▶ 运行中")
            self._do_run()
        else:
            self._log("⏸ 已暂停")

    def action_awake(self) -> None:
        self.organizer.force_awake()
        self._log("✓ 强制觉醒")
        self._refresh_status()

    def action_sleep(self) -> None:
        self.organizer.force_sleep()
        self._log("✓ 强制睡眠")
        self._refresh_status()


def main():
    app = DevelopmentalGUI()
    app.run()


if __name__ == "__main__":
    main()
