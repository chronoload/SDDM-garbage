"""诊断：progress 为何崩塌？跟踪范数与 NaN"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
from mcp.developmental.signal import Signal
import baby_real as B

vis, aud, lang = B.build_world(0)
sents = B.make_sentences(seed=0)
stream=[]
for s in sents: stream.extend(s)

brain = B.build_brain(0)
hb = brain.higher_brain
n0 = hb.ecosystem.get_neuron(0)
for c in B.CH: n0.unfold(c, shim.zeros(B.D))
D=B.D

print('步 | out_norm | W_norm | resid_norm | sheaths | born | died | progress | ar_h | ar_r')
print('-'*84)
for t in range(600):
    w = stream[t%len(stream)]
    inputs={'vis':Signal(shim.tensor(vis[w].copy()),'generic/tensor',{'t':t}),
            'aud':Signal(shim.tensor(aud[w].copy()),'generic/tensor',{'t':t}),
            'lang':Signal(shim.tensor(lang[w].copy()),'generic/tensor',{'t':t})}
    res = brain.step_awake(external_inputs=inputs)
    outs = res.get('outputs') or {}
    lg = outs.get('lang')
    onorm = np.nan
    if lg is not None:
        v=np.asarray(lg.data.a if hasattr(lg.data,'a') else lg.data,dtype=float).ravel()
        onorm = float(np.linalg.norm(v))
    nns=[n for n in hb.ecosystem.neurons.values() if n is not None and getattr(n,'W',None) is not None]
    wnorm = max([float(n.W.norm().item()) for n in nns]) if nns else 0.0
    reg=hb.sheath_registry
    sc=hb.scheduler
    if (t+1)%100==0:
        print(f'{t+1:4d} | {onorm:8.3f} | {wnorm:6.3f} | '
              f'{sc._ar_high_ema if sc._ar_high_ema else 0:9.4f} | {len(reg._sheaths):7d} | '
              f'{reg._born:4d} | {reg._died:4d} | {sc.progress:+8.2f} | '
              f'{(sc._ar_high_ema or 0):8.3f} | {(sc._ar_reflex_ema or 0):8.3f}')
