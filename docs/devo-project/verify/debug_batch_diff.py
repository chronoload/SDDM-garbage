"""逐事件对照 dump：定位 batched 与 loop 的数据差异"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")
import numpy_torch_shim as shim
shim.install()
import baby_grammar as BG

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project\docs\devo-project\verify")
from test_batched_dispatch import build_trained

brain = build_trained()
hb = brain.higher_brain
disp = hb.dispatcher
world = BG.GrammarWorld(1)
obs = world.observe()
combined = brain._combine_signals(
    {c: BG.Signal(shim.tensor(obs[off:off + sz].copy()),
                  "generic/tensor", {"t": 0})
     for c, (off, sz) in BG.PORT.items()})
x = brain.normalizer.normalize(combined.data)

ev_loop = sorted(disp.dispatch(x, "l"), key=lambda e: (e.sheath_key))
ev_batch = sorted(disp.batched_dispatch(x, "b"), key=lambda e: (e.sheath_key))
print(f"loop={len(ev_loop)} batched={len(ev_batch)}")
lm = {e.sheath_key: e for e in ev_loop}
bm = {e.sheath_key: e for e in ev_batch}
for key in sorted(set(lm) | set(bm)):
    a, b = lm.get(key), bm.get(key)
    if a is None or b is None:
        print(f"{key}: 缺失 loop={a is not None} batched={b is not None}")
        continue
    da = np.asarray(a.data.detach().cpu().numpy() if hasattr(a.data, "detach")
                    else a.data, dtype=float)
    db = np.asarray(b.data.detach().cpu().numpy() if hasattr(b.data, "detach")
                    else b.data, dtype=float)
    diff = np.abs(da - db).max()
    flag = "OK " if diff < 1e-9 else "DIFF"
    print(f"{key}: arrival {a.arrival_time:.2f}/{b.arrival_time:.2f} "
          f"|loop|={np.linalg.norm(da):.3f} |batch|={np.linalg.norm(db):.3f} "
          f"maxdiff={diff:.3e} {flag}")
    if diff >= 1e-9:
        na, nb = np.nonzero(da)[0], np.nonzero(db)[0]
        print(f"    loop非零位={na[:8]} 值={da[na][:8]}")
        print(f"    batch非零位={nb[:8]} 值={db[nb][:8]}")
