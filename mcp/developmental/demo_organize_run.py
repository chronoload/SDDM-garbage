"""发育智能系统真实接轨示例

演示器官组织层如何让发育脑"活起来"：
- 感知器官（视觉/符号）持续推入信号
- 反射弧装配层编排工具链
- 组织器主循环驱动觉醒/睡眠节律
- 观察窗输出每 tick 的脑状态

运行：python -m mcp.developmental.demo_organize_run
"""
from __future__ import annotations

import torch

from mcp.developmental.signal import Signal
from mcp.developmental.system import DevelopmentalSystem
from mcp.developmental.reptilian import EchoFunction, SaturateFunction
from mcp.developmental.organize import (
    Organizer,
    QueueSensoryOrgan,
    ReflexAssembly,
)


def main():
    # 1. 构造发育脑：input(4) + y(4) 信道
    brain = DevelopmentalSystem(port_layout={"input": (0, 4), "y": (4, 4)})
    brain.phase = "exploratory"
    # 展开种子神经元的 input 信道（脱离胚胎期 λ=0 基线，进入探索期）
    brain.higher_brain.ecosystem.get_neuron(0).unfold("input", torch.zeros(4))

    # 2. 装配反射弧（脑不感知装配层）
    assembly = ReflexAssembly(brain)
    assembly.register_tool("echo", EchoFunction())              # 直通工具
    assembly.register_tool("saturate", SaturateFunction(-1.0, 1.0))  # 饱和工具
    assembly.add_route(source_channel="input", tool_name="echo", priority=1.0)
    assembly.add_route(source_channel="y", tool_name="saturate", priority=0.5)

    print("=== 反射弧编排表 ===")
    for r in assembly.list_routes():
        print(f"  {r['source_channel']} → {r['tool_name']} (priority={r['priority']})")

    # 3. 构造感知器官（视觉信道，向 input 推信号）
    vision = QueueSensoryOrgan(channel="input", mime_type="generic/tensor")

    # 4. 组织器：接通脑 + 感知器官
    organizer = Organizer(brain)
    organizer.add_sensory(vision)
    organizer.awake_steps_per_cycle = 20   # 20 tick 觉醒
    organizer.sleep_steps_per_cycle = 5    # 5 tick 睡眠

    print("\n=== 主循环启动（25 tick：20 觉醒 + 5 睡眠）===")

    # 5. 觉醒期：推入交替信号
    print("\n--- 觉醒期 ---")
    for i in range(20):
        val = float(i % 3) * 0.3
        vision.push(Signal(torch.tensor([val, 0.1, 0.0, 0.0]),
                           "generic/tensor",
                           {"tick": i, "modality": "demo"}))
        result = organizer.tick()
        out = result.get("outputs", {})
        lam = result.get("lambda", 0.0)
        y_val = out.get("y")
        y_str = f"{y_val.data.tolist()}" if y_val else "(空)"
        print(f"  tick {i:2d} | λ={lam:.2f} | y={y_str}")

    # 6. 睡眠期：四相位做梦学习
    print("\n--- 睡眠期（四相位做梦学习）---")
    for i in range(5):
        result = organizer.tick()
        if "homeostasis" in result:
            h = result["homeostasis"]
            print(f"  tick {20+i:2d} | 稳态缩放: sheaths {h['sheaths_before']}→{h['sheaths_after']}")
        if "replay" in result:
            r = result["replay"]
            print(f"  tick {20+i:2d} | 重放: disturbed={r.get('disturbed_by_external', False)}")
        if "counterfactual" in result:
            c = result["counterfactual"]
            print(f"  tick {20+i:2d} | 反事实: delta={c.get('intervention_delta', 0.0):.4f}")
        if "preplay" in result:
            print(f"  tick {20+i:2d} | 前向预演: executed")
        if "sleep_outputs" in result:
            print(f"  tick {20+i:2d} | 睡眠输出: {list(result['sleep_outputs'].keys())}")

    # 7. 总结
    print("\n=== 运行总结 ===")
    print(f"  总 tick 数: {organizer.tick_count}")
    print(f"  轨迹日志: {len(brain._trajectory_log)} 条（供做梦重放）")
    print(f"  神经元数: {brain.higher_brain.ecosystem.count()}")
    print(f"  髓鞘数: {len(brain.higher_brain.sheath_registry._sheaths)}")
    print(f"  当前模式: {'觉醒' if organizer.is_awake() else '睡眠'}")

    print("\n=== 系统已持续运行，可继续推入信号或保存状态 ===")


if __name__ == "__main__":
    main()
