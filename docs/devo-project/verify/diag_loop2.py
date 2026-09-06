import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
import baby_loop as L
from mcp.developmental.signal import Signal

np.random.seed(0)
world = L.ReactiveWorld(0)
brain = L.build_brain(0)
ref = brain.reflex._kernel.get("babble")
hb = brain.higher_brain
n0 = hb.ecosystem.get_neuron(0)
for c in L.PORT: n0.unfold(c, shim.zeros(L.D_ALL))

print(f"总维度 D_ALL={L.D_ALL}  PORT={L.PORT}")
print(f"LANG_SLICE={L.LANG_SLICE}")
print()
print(f"{'步':>4} | {'want':>8} | {'反射输出':>8} | {'系统lang范数':>12} | {'系统解码':>8} | {'lambda':>6} | {'满足':>5}")
print("-"*72)
for t in range(500):
    ref.set_drive(world.want)
    obs = world.observe()
    inputs = {c: Signal(shim.tensor(obs[o:o+s].copy()), 'generic/tensor', {'t':t})
              for c,(o,s) in L.PORT.items()}
    res = brain.step_awake(external_inputs=inputs)
    out = (res.get('outputs') or {}).get('lang')
    a = L._as_np(out)
    w = L.decode_lang(out)
    seg = a if a is not None else None
    r,_ = world.act(w)
    if (t+1) % 100 == 0:
        refout = ref.execute({'lang': Signal(shim.tensor(np.zeros(L.D_LANG)),'generic/tensor',{})})
        ra = L._as_np(refout['lang'])
        print(f"{t+1:4d} | {world.want:>8} | {L.WORDS[int(np.argmax(ra))]:>8} | "
              f"{float(np.linalg.norm(seg)) if seg is not None else -1:12.4f} | {str(w):>8} | "
              f"{res.get('lambda',0):.3f} | {r:5.1f}")
print("-"*72)
print(f"总满足: {world.satisfied}/{world.trials}")
print()
print("反射基线检查：反射直接照抄 drive，应达到 100%（若无噪声）")
print(f"  反射带噪声 0.3，实际满足率需实测")
