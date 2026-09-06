"""诊断：性能为何在 750-1000 步达峰后回落"""
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

print('步  | top1  | lang余弦 | 输出范数 | W范数 | 维度数 | 髓鞘 | gain和 | 自噬剪 | progress')
print('-'*88)
hits, coses = [], []
for t in range(2000):
    w = stream[t%len(stream)]; wn = stream[(t+1)%len(stream)]
    inputs={'vis':Signal(shim.tensor(vis[w].copy()),'generic/tensor',{'t':t}),
            'aud':Signal(shim.tensor(aud[w].copy()),'generic/tensor',{'t':t}),
            'lang':Signal(shim.tensor(lang[w].copy()),'generic/tensor',{'t':t})}
    res = brain.step_awake(external_inputs=inputs)
    outs = res.get('outputs') or {}
    lg = outs.get('lang')
    if lg is None: continue
    v=np.asarray(lg.data.a if hasattr(lg.data,'a') else lg.data,dtype=float).ravel()
    hits.append(B.decode_lang(v)==wn)
    coses.append(float(np.dot(v,lang[wn])/(np.linalg.norm(v)*np.linalg.norm(lang[wn])+1e-9)))

    if (t+1)%250==0:
        nns=[n for n in hb.ecosystem.neurons.values() if n is not None and getattr(n,'W',None) is not None]
        wnorm = max([float(n.W.norm().item()) for n in nns]) if nns else 0.0
        onorm = float(np.linalg.norm(v))
        ndim = sum(len(n.unfolded) for n in nns)
        reg=hb.sheath_registry
        gsum = sum(s.gain for s in reg._sheaths.values())
        r1=np.mean(hits[-250:]); cc=np.mean(coses[-250:])
        print(f'{t+1:4d} | {r1:.3f} | {cc:8.3f} | {onorm:8.3f} | {wnorm:5.2f} | {ndim:6d} | '
              f'{len(reg._sheaths):4d} | {gsum:6.2f} | {getattr(hb.engine,"_pruned_total",0):6d} | {hb.scheduler.progress:+.3f}')
