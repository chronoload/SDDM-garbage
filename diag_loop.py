import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
import baby_loop as L
from mcp.developmental.signal import Signal

np.random.seed(0)
world = L.ReactiveWorld(0)
brain = L.build_brain(0)
hb = brain.higher_brain
n0 = hb.ecosystem.get_neuron(0)
for c in L.PORT: n0.unfold(c, shim.zeros(L.D_ALL))

print("观测帧示例：")
obs = world.observe()
print(f"  vis段  = {np.round(obs[0:8],2)}")
print(f"  want段 = {np.round(obs[L.WANT_SLICE],2)}  -> {L.WORDS[int(np.argmax(obs[L.WANT_SLICE]))]}")
print()
print(f"{'步':>4} | {'输出类型':>10} | {'lang段范数':>10} | {'解码':>8} | {'want':>8} | {'lambda':>7} | {'reward':>6}")
print("-"*70)
for t in range(400):
    obs = world.observe()
    inputs = {c: Signal(shim.tensor(obs[o:o+s].copy()), 'generic/tensor', {'t':t})
              for c,(o,s) in L.PORT.items()}
    res = brain.step_awake(external_inputs=inputs)
    out = (res.get('outputs') or {}).get('lang')
    a = L._as_np(out)
    norm = float(np.linalg.norm(a)) if a is not None else -1
    if a is None:
        print("  !! out 转 numpy 失败, out =", out); break
    if a.size < 4:
        print(f"  !! 输出尺寸异常: {a.size}, 前几步就退出"); break
    if t < 3:
        print(f"  [t={t}] 输出 size={a.size}  内容={np.round(a,3)}")
    seg = a[L.LANG_SLICE] if (a is not None and a.size >= L._OFF['lang']+L.D_LANG) else None
    w = L.WORDS[int(np.argmax(seg))] if seg is not None else None
    r,_ = world.act(w)
    if (t+1) % 80 == 0:
        print(f"{t+1:4d} | {type(out).__name__:>10} | {norm:10.3f} | {str(w):>8} | "
              f"{world.want:>8} | {res.get('lambda',0):.3f} | {r:6.1f}")
print("-"*70)
print(f"总满足: {world.satisfied}/{world.trials}")
