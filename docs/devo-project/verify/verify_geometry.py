sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")
"""几何对照：未训练 vs 训练后——超模态几何应当从无到有"""
import numpy_torch_shim as shim
shim.install()
import numpy as np
import baby_grammar as BG
from mcp.developmental.geometry import modal_geometry_report


def show(tag, hb):
    rep = modal_geometry_report(hb)
    print(f"\n── {tag} ──")
    for p, a in rep["pairs"].items():
        print(f"  传递 {p}: 能量={a['energy']:.3f} "
              f"首奇异值={a['top_sv']:.3f} 参与率={a['participation']:.2f}")
    for c, a in rep["cycles"].items():
        print(f"  循环 {c}: 一致性比={a['ratio']:.3f} "
              f"谱={['%.3f' % x for x in a['spectrum']]}")


# 未训练（几何尚未生长）
brain0 = BG.build_brain(0, use_attention=False, use_spikes=False)
show("未训练（种子态）", brain0.higher_brain)

# 训练后
world = BG.GrammarWorld(0)
brain = BG.build_brain(0, use_attention=False, use_spikes=False)
prev = None
STEPS = 4000
for t in range(STEPS):
    r = None
    if t > 0:
        world._advance()
        r = 1.0 if prev == world.cur else 0.0
        brain.report_drive_satisfaction(r)
    obs = world.observe()
    obs[BG.LANG_SLICE] = world.observe_lang()
    inputs = {c: BG.Signal(shim.tensor(obs[off:off + sz].copy()),
                           "generic/tensor", {"t": t})
              for c, (off, sz) in BG.PORT.items()}
    res = brain.step_awake(external_inputs=inputs)
    prev = BG.decode_lang((res.get("outputs") or {}).get("lang"))
    world.last_guess = prev
show(f"训练 {STEPS} 步后（top1 曲线见 run）", brain.higher_brain)
