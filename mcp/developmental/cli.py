"""发育智能系统 CLI 控制面板 — 交互式 REPL

核心用途：编排爬虫脑（动态增删工具/路由）+ 运行控制 + 状态观察 + 持久化

启动：
    python -m mcp.developmental.cli
    python -m mcp.developmental.cli --load path/to/brain_state

命令分组：
    [编排] tool add|list|remove | route add|list|remove
    [运行] start | stop | step [N] | inject <vals...> | awake | sleep
    [状态] status | routes | tools | neurons | sheaths
    [持久] save <path> | load <path>

示例：
    > tool add echo
    > tool add saturate -1.0 1.0
    > route add input echo 1.0
    > route add y saturate 0.5
    > inject 0.5 0.1 0.0 0.0
    > step 10
    > status
    > save ./brain_state
"""
from __future__ import annotations

import argparse
import cmd
import shlex
import sys
import torch

from mcp.developmental.signal import Signal
from mcp.developmental.system import DevelopmentalSystem, SystemMode
from mcp.developmental.reptilian import EchoFunction, SaturateFunction, LambdaFunction
from mcp.developmental.builtin.web_agent import WebAgentFunction
from mcp.developmental.text_encoder import TextEncoder  # P0: 第 87 行用到，此前未导入
from mcp.developmental.organize import (
    Organizer,
    QueueSensoryOrgan,
    ReflexAssembly,
)


# === 内置工具注册表（CLI 可按名添加）===

BUILTIN_TOOLS = {
    "echo": lambda args: EchoFunction(),
    "saturate": lambda args: SaturateFunction(
        float(args[0]) if len(args) > 0 else -1.0,
        float(args[1]) if len(args) > 1 else 1.0,
    ),
    "symbol": lambda args: __import__(
        "mcp.developmental.builtin.symbol_channel", fromlist=["SymbolChannelFunction"]
    ).SymbolChannelFunction(
        int(args[0]) if len(args) > 0 else 512
    ),
    # webagent [http|local] [url_or_workspace]
    # 默认 http 模式，连 http://localhost:6906
    "webagent": lambda args: WebAgentFunction(
        mode=args[0] if args and args[0] in ("http", "local") else "http",
        server_url=args[1] if args and args[0] == "http" and len(args) > 1 else "http://localhost:6906",
        workspace_dir=args[1] if args and args[0] == "local" and len(args) > 1 else None,
    ),
}


class DevelopmentalCLI(cmd.Cmd):
    """发育智能系统交互式 REPL"""

    intro = (
        "=== 发育智能系统 CLI 控制面板 ===\n"
        "输入 help 查看命令，exit 退出。\n"
        "核心：编排爬虫脑（tool/route）+ 运行控制 + 状态观察 + 持久化\n"
    )
    prompt = "brain> "

    def __init__(self, brain: DevelopmentalSystem, organizer: Organizer,
                 assembly: ReflexAssembly, sensory: QueueSensoryOrgan,
                 text_sensory: QueueSensoryOrgan):
        super().__init__()
        self.brain = brain
        self.organizer = organizer
        self.assembly = assembly
        self.sensory = sensory
        self.text_sensory = text_sensory
        self._running = False
        self._encoder = TextEncoder.instance()

    # ============================================================
    # [编排] tool 命令
    # ============================================================

    def do_tool(self, arg: str) -> None:
        """tool add <name> [args...] | tool list | tool remove <name>
        示例：tool add echo | tool add saturate -1.0 1.0 | tool list
              tool add webagent                  # HTTP 代理，默认连 localhost:6906
              tool add webagent http http://localhost:6906
              tool add webagent local /path/to/workspace
        """
        parts = shlex.split(arg)
        if not parts:
            print("用法：tool add <name> [args...] | tool list | tool remove <name>")
            return

        sub = parts[0]
        if sub == "add":
            if len(parts) < 2:
                print("用法：tool add <name> [args...]")
                return
            name = parts[1]
            tool_args = parts[2:]
            if name in BUILTIN_TOOLS:
                func = BUILTIN_TOOLS[name](tool_args)
                self.assembly.register_tool(name, func)
                print(f"✓ 工具已注册：{name}")
            else:
                print(f"✗ 未知内置工具：{name}（可用：{', '.join(BUILTIN_TOOLS.keys())}）")
        elif sub == "list":
            tools = self.brain.reptile.list_functions()
            if tools:
                print("已注册工具：")
                for t in tools:
                    print(f"  - {t}")
            else:
                print("（无工具）")
        elif sub == "remove":
            if len(parts) < 2:
                print("用法：tool remove <name>")
                return
            name = parts[1]
            if name in self.brain.reptile._functions:
                del self.brain.reptile._functions[name]
                # 同时移除相关路由
                before = len(self.assembly._routes)
                self.assembly._routes = [
                    r for r in self.assembly._routes if r["tool_name"] != name
                ]
                removed = before - len(self.assembly._routes)
                print(f"✓ 工具已移除：{name}（同时移除 {removed} 条相关路由）")
            else:
                print(f"✗ 工具不存在：{name}")
        else:
            print(f"未知子命令：{sub}（可用：add, list, remove）")

    # ============================================================
    # [编排] route 命令
    # ============================================================

    def do_route(self, arg: str) -> None:
        """route add <source_channel> <tool_name> [priority] | route list | route remove <source> <tool>
        示例：route add input echo 1.0 | route add y saturate 0.5 | route list
        """
        parts = shlex.split(arg)
        if not parts:
            print("用法：route add <src> <tool> [priority] | route list | route remove <src> <tool>")
            return

        sub = parts[0]
        if sub == "add":
            if len(parts) < 3:
                print("用法：route add <source_channel> <tool_name> [priority]")
                return
            src, tool = parts[1], parts[2]
            priority = float(parts[3]) if len(parts) > 3 else 1.0
            if tool not in self.brain.reptile.list_functions():
                print(f"✗ 工具未注册：{tool}（先 tool add {tool}）")
                return
            self.assembly.add_route(src, tool, priority)
            print(f"✓ 路由已添加：{src} → {tool} (priority={priority})")
        elif sub == "list":
            routes = self.assembly.list_routes()
            if routes:
                print("反射弧编排表：")
                for r in routes:
                    print(f"  {r['source_channel']} → {r['tool_name']} (priority={r['priority']})")
            else:
                print("（无路由）")
        elif sub == "remove":
            if len(parts) < 3:
                print("用法：route remove <source_channel> <tool_name>")
                return
            src, tool = parts[1], parts[2]
            self.assembly.remove_route(src, tool)
            print(f"✓ 路由已移除：{src} → {tool}")
        else:
            print(f"未知子命令：{sub}（可用：add, list, remove）")

    # ============================================================
    # [运行] start / stop / step / inject / awake / sleep
    # ============================================================

    def do_start(self, arg: str) -> None:
        """start — 启动持续运行（按 Ctrl+C 停止）"""
        print("启动持续运行（Ctrl+C 停止）...")
        self._running = True
        try:
            n = 0
            while self._running:
                result = self.organizer.tick()
                outputs = result.get("outputs") or result.get("sleep_outputs", {})
                if outputs:
                    out_str = _format_outputs(outputs)
                    print(f"  tick {n}: λ={result.get('lambda', 0):.2f} | {out_str}")
                n += 1
                if not outputs:
                    import time
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print(f"\n停止运行（共 {n} tick）")
        self._running = False

    def do_stop(self, arg: str) -> None:
        """stop — 停止持续运行"""
        self._running = False
        print("已请求停止")

    def do_step(self, arg: str) -> None:
        """step [N] — 单步运行 N 个 tick（默认 1）
        示例：step | step 10
        """
        parts = shlex.split(arg)
        n = int(parts[0]) if parts else 1
        for i in range(n):
            result = self.organizer.tick()
            outputs = result.get("outputs") or result.get("sleep_outputs", {})
            mode = "觉醒" if self.organizer.is_awake() else "睡眠"
            if outputs:
                out_str = _format_outputs(outputs)
                print(f"  tick {i+1}/{n} [{mode}] λ={result.get('lambda', 0):.2f} | {out_str}")
            else:
                phase_info = ""
                if "homeostasis" in result:
                    phase_info = "稳态缩放"
                elif "replay" in result:
                    phase_info = "重放"
                elif "counterfactual" in result:
                    phase_info = "反事实"
                print(f"  tick {i+1}/{n} [{mode}] λ={result.get('lambda', 0):.2f} | {phase_info or '无输出'}")

    def do_inject(self, arg: str) -> None:
        """inject <v1> <v2> ... — 推入张量感知信号到 input 信道
        示例：inject 0.5 0.1 0.0 0.0
        """
        parts = shlex.split(arg)
        if not parts:
            print("用法：inject <v1> <v2> ...")
            return
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            print("✗ 参数必须是数字")
            return
        self.sensory.push(Signal(torch.tensor(vals), "generic/tensor"))
        print(f"✓ 张量信号已推入 input：{vals}")

    def do_text(self, arg: str) -> None:
        """text <message...> — 推入文本感知信号到 text 信道（经分词器编码）
        示例：text 你好世界
              text "tell me a joke"
        反射弧需编排 route add text webagent 1.0 才会处理。
        """
        text = arg.strip()
        if not text:
            print("用法：text <message...>")
            return
        token_ids = self._encoder.encode(text)
        self.text_sensory.push(Signal(
            data=token_ids,
            mime_type="text/plain",
            metadata={"raw": text, "source": "cli", "n_tokens": len(token_ids)},
        ))
        print(f"✓ 文本信号已推入 text（{len(token_ids)} tokens, {self._encoder.backend}）：{text[:60]}{'...' if len(text) > 60 else ''}")

    def do_awake(self, arg: str) -> None:
        """awake — 强制觉醒"""
        self.organizer.force_awake()
        print("✓ 已强制觉醒")

    def do_sleep(self, arg: str) -> None:
        """sleep — 强制睡眠"""
        self.organizer.force_sleep()
        print("✓ 已强制睡眠")

    # ============================================================
    # [状态] status / routes / tools / neurons / sheaths
    # ============================================================

    def do_status(self, arg: str) -> None:
        """status — 显示脑状态总览"""
        brain = self.brain
        eco = brain.higher_brain.ecosystem
        reg = brain.higher_brain.sheath_registry
        engine = brain.higher_brain.engine
        mode = "觉醒" if self.organizer.is_awake() else "睡眠"

        print("=== 脑状态 ===")
        print(f"  阶段: {brain.phase}")
        print(f"  模式: {mode}")
        print(f"  step_count: {brain.step_count}")
        print(f"  organizer tick: {self.organizer.tick_count}")
        print(f"  神经元数: {eco.count()}")
        print(f"  髓鞘数: {len(reg._sheaths)}")
        print(f"  新奇性累积: {engine._novelty_accumulator:.4f}")
        print(f"  新奇性阈值: {engine.novelty_threshold}")
        print(f"  分化次数: {engine._differentiation_count}")
        print(f"  轨迹日志: {len(brain._trajectory_log)} 条")
        print(f"  反射弧: {len(self.assembly.list_routes())} 条路由")
        print(f"  工具: {len(brain.reptile.list_functions())} 个")
        last = self.organizer.last_result
        if last:
            print(f"  上次 λ: {last.get('lambda', 0):.4f}")

    def do_neurons(self, arg: str) -> None:
        """neurons — 列出所有神经元详情"""
        eco = self.brain.higher_brain.ecosystem
        for idx, n in eco.neurons.items():
            status = "活" if n.alive else "死"
            w_shape = str(tuple(n.W.shape)) if n.W is not None else "None"
            unfolded = ", ".join(n.unfolded.keys()) if n.unfolded else "(无)"
            parent = f" parent={n.parent_seed}" if n.parent_seed else ""
            print(f"  [{idx}] seed={n.seed}{parent} [{status}] W={w_shape} unfolded=[{unfolded}]")

    def do_sheaths(self, arg: str) -> None:
        """sheaths — 列出所有髓鞘"""
        reg = self.brain.higher_brain.sheath_registry
        if not reg._sheaths:
            print("（无髓鞘）")
            return
        for key, s in reg._sheaths.items():
            print(f"  {s.src_neuron}.{s.src_channel} → {s.dst_neuron}.{s.dst_channel} "
                  f"| delay={s.delay:.3f} gain={s.gain:.3f} protection={s.protection:.3f}")

    def do_tools(self, arg: str) -> None:
        """tools — 列出已注册工具"""
        tools = self.brain.reptile.list_functions()
        if tools:
            print("已注册工具：")
            for t in tools:
                print(f"  - {t}")
        else:
            print("（无工具）")

    def do_routes(self, arg: str) -> None:
        """routes — 列出反射弧编排表"""
        routes = self.assembly.list_routes()
        if routes:
            print("反射弧编排表：")
            for r in routes:
                print(f"  {r['source_channel']} → {r['tool_name']} (priority={r['priority']})")
        else:
            print("（无路由）")

    # ============================================================
    # [持久] save / load
    # ============================================================

    def do_save(self, arg: str) -> None:
        """save <path> — 保存脑状态到目录
        示例：save ./brain_state
        """
        parts = shlex.split(arg)
        if not parts:
            print("用法：save <path>")
            return
        path = parts[0]
        try:
            self.brain.save_state(path)
            print(f"✓ 脑状态已保存到 {path}")
        except Exception as e:
            print(f"✗ 保存失败：{e}")

    def do_load(self, arg: str) -> None:
        """load <path> — 从目录加载脑状态
        示例：load ./brain_state
        """
        parts = shlex.split(arg)
        if not parts:
            print("用法：load <path>")
            return
        path = parts[0]
        try:
            self.brain.load_state(path)
            print(f"✓ 脑状态已从 {path} 加载")
            print(f"  神经元: {self.brain.higher_brain.ecosystem.count()}")
            print(f"  髓鞘: {len(self.brain.higher_brain.sheath_registry._sheaths)}")
            print(f"  phase: {self.brain.phase}, step_count: {self.brain.step_count}")
        except Exception as e:
            print(f"✗ 加载失败：{e}")

    # ============================================================
    # 退出
    # ============================================================

    def do_exit(self, arg: str) -> bool:
        """exit — 退出 CLI"""
        print("再见")
        return True

    def do_quit(self, arg: str) -> bool:
        """quit — 退出 CLI"""
        return self.do_exit(arg)

    def do_EOF(self, arg: str) -> bool:
        """Ctrl+D 退出"""
        print()
        return self.do_exit(arg)


def _format_outputs(outputs: dict) -> str:
    """格式化输出信号：text/* 显示文本，其他显示张量"""
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
    return ", ".join(parts)


def _build_default_system():
    """构造默认脑 + 装配 + 组织器（含张量 + 文本双感知器官）"""
    brain = DevelopmentalSystem(port_layout={"input": (0, 4), "y": (4, 4)})
    brain.phase = "embryonic"
    brain.higher_brain.ecosystem.get_neuron(0).unfold("input", torch.zeros(4))
    brain.phase = "exploratory"

    assembly = ReflexAssembly(brain)
    sensory = QueueSensoryOrgan(channel="input", mime_type="generic/tensor")
    text_sensory = QueueSensoryOrgan(channel="text", mime_type="text/plain")
    organizer = Organizer(brain)
    organizer.add_sensory(sensory)
    organizer.add_sensory(text_sensory)

    return brain, organizer, assembly, sensory, text_sensory


def main():
    parser = argparse.ArgumentParser(description="发育智能系统 CLI 控制面板")
    parser.add_argument("--load", type=str, help="启动时加载脑状态目录")
    parser.add_argument("--port-layout", type=str, default="input:4,y:4",
                        help="端口布局（默认 input:4,y:4）")
    args = parser.parse_args()

    # 解析 port_layout
    layout_parts = args.port_layout.split(",")
    port_layout = {}
    offset = 0
    for part in layout_parts:
        name, size = part.split(":")
        size = int(size)
        port_layout[name] = (offset, size)
        offset += size

    brain = DevelopmentalSystem(port_layout=port_layout)
    brain.phase = "exploratory"
    brain.higher_brain.ecosystem.get_neuron(0).unfold(
        list(port_layout.keys())[0], torch.zeros(list(port_layout.values())[0][1])
    )

    assembly = ReflexAssembly(brain)
    sensory = QueueSensoryOrgan(
        channel=list(port_layout.keys())[0],
        mime_type="generic/tensor",
    )
    text_sensory = QueueSensoryOrgan(
        channel="text",
        mime_type="text/plain",
    )
    organizer = Organizer(brain)
    organizer.add_sensory(sensory)
    organizer.add_sensory(text_sensory)

    if args.load:
        try:
            brain.load_state(args.load)
            print(f"✓ 已加载脑状态：{args.load}")
        except Exception as e:
            print(f"✗ 加载失败：{e}")

    cli = DevelopmentalCLI(brain, organizer, assembly, sensory, text_sensory)
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n再见")


if __name__ == "__main__":
    main()
