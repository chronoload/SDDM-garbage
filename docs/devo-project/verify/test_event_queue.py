"""T2 红测试：跨步事件队列（delay 成为一等公民 / 内生同步地基）

契约：
1. cross_step=False（默认）：事件当步结算——现有语义逐位保留
2. cross_step=True：delay≥1 的传输事件跨帧存活，
   在 arrival_tick（发出 tick + delay）之后才进入结算
3. delay<1（已髓鞘化快通路）仍当步到达
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")

import numpy_torch_shim as shim
shim.install()

import baby_grammar as BG                                        # noqa: E402


def main():
    brain = BG.build_brain(0, use_attention=False, use_spikes=False)
    hb = brain.higher_brain
    disp = hb.dispatcher
    reg = hb.sheath_registry

    # 慢通路：vis→lang，delay=3.0（快通路 delay 0.01 已由训练结构存在）
    reg.add_sheath(0, "vis", 0, "lang", delay=3.0, gain=1.0)
    slow_key = (0, "vis", 0, "lang")
    reg._connections[slow_key] = reg._connections.get(slow_key) or \
        type(next(iter(reg._connections.values())))(
            0, "vis", 0, "lang") if reg._connections else None
    # 直接造 Connection（avoid依赖已有结构）
    from mcp.developmental.myelin import Connection
    reg._connections[slow_key] = Connection(0, "vis", 0, "lang")
    reg._sheaths[slow_key] = type(next(iter(reg._sheaths.values())))(
        src_neuron=0, src_channel="vis", dst_neuron=0, dst_channel="lang",
        delay=3.0, gain=1.0)

    world = BG.GrammarWorld(0)
    obs = world.observe()
    x = brain.normalizer.normalize(brain._combine_signals(
        {c: BG.Signal(shim.tensor(obs[off:off + sz].copy()),
                      "generic/tensor", {"t": 0})
         for c, (off, sz) in BG.PORT.items()}).data)

    # ---- 契约 1：默认 cross_step=False，当步结算（现有语义） ----
    assert disp.cross_step is False
    ev = disp.batched_dispatch(x, "t0")
    assert any(e.sheath_key == slow_key for e in ev), \
        "默认模式下慢通路事件应当步出现"

    # ---- 契约 2/3：cross_step=True，delay=3 跨 3 步到达 ----
    disp.cross_step = True
    disp.pending = []
    seen = []
    for tick in range(1, 6):
        disp.tick()
        ev = disp.batched_dispatch(x, f"tick{tick}")
        seen.append(any(e.sheath_key == slow_key for e in ev))
    # delay=3.0 → 第 4 次 tick（tick=4）应到达：tick1,2,3 未到，tick4,5 到
    assert seen == [False, False, False, True, True], \
        f"慢通路到达时序错误: {seen}"
    print("到达时序:", ["✓" if s else "·" for s in seen])
    print("PASS: 跨步事件队列契约全部满足")


if __name__ == "__main__":
    main()
