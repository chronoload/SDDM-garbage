"""T4 行动解码诊断"""
import sys
sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")
import numpy as np
import numpy_torch_shim as shim
shim.install()
import baby_multilimb as M
from mcp.developmental.signal import Signal

np.random.seed(0)
shim.manual_seed(0)
world = M.MultilimbWorld(0)
brain = M.build_brain(0)
pending = (None, None)
for t in range(20):
    r = None
    if t > 0:
        r = world.judge(pending[0], pending[1])
        brain.report_drive_satisfaction(r)
    obs = world.observe()
    inputs = {c: Signal(shim.tensor(obs[off:off + sz].copy()),
                        "generic/tensor", {"t": t})
              for c, (off, sz) in M.PORT.items()}
    res = brain.step_awake(external_inputs=inputs)
    outs = res.get("outputs") or {}
    ld = outs.get("lang")
    hd = outs.get("hand")
    la = ld.detach().cpu().numpy() if ld is not None and hasattr(ld, "detach") else None
    ha = hd.detach().cpu().numpy() if hd is not None and hasattr(hd, "detach") else None
    v = M._argmax_channel(la, 0, M.D_LANG) if la is not None else "NO-LANG"
    h = M._argmax_channel(ha, 0, M.D_HAND) if ha is not None else "NO-HAND"
    lam = res.get("lambda", 0.0)
    if t < 8 or t % 5 == 0:
        print(f"t={t} outs_keys={list(outs.keys())} vocal={v} hand={h} "
              f"want={world.want} lambda={lam:.2f}")
    world.last_vocal = v if isinstance(v, int) else None
    world.last_hand = h if isinstance(h, int) else None
    pending = (v if isinstance(v, int) else None,
               h if isinstance(h, int) else None)
