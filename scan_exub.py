"""扫描 exuberant_until：冷启动豁免上限对结构生长的影响"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
from mcp.developmental.signal import Signal
import baby_real as B

vis,aud,lang=B.build_world(0)
sents=B.make_sentences(seed=0); stream=[]
for s in sents: stream.extend(s)
D=B.D

def run(exub, steps=800, seed=0):
    np.random.seed(seed)
    brain=B.build_brain(0)
    hb=brain.higher_brain
    hb.sheath_registry.exuberant_until = exub
    n0=hb.ecosystem.get_neuron(0)
    for c in B.CH: n0.unfold(c, shim.zeros(D))
    hits,coses=[],[]
    for t in range(steps):
        w=stream[t%len(stream)]; wn=stream[(t+1)%len(stream)]
        inputs={'vis':Signal(shim.tensor(vis[w].copy()),'generic/tensor',{'t':t}),
                'aud':Signal(shim.tensor(aud[w].copy()),'generic/tensor',{'t':t}),
                'lang':Signal(shim.tensor(lang[w].copy()),'generic/tensor',{'t':t})}
        res=brain.step_awake(external_inputs=inputs)
        lg=(res.get('outputs') or {}).get('lang')
        if lg is None: continue
        v=np.asarray(lg.data.a if hasattr(lg.data,'a') else lg.data,dtype=float).ravel()
        hits.append(B.decode_lang(v)==wn)
        coses.append(float(np.dot(v,lang[wn])/(np.linalg.norm(v)*np.linalg.norm(lang[wn])+1e-9)))
    reg=hb.sheath_registry
    dst={}
    for (sn,sc,dn,dc),sh in reg._sheaths.items():
        dst[dc]=dst.get(dc,0)+1
    return (np.mean(hits[-200:]), np.mean(coses[-200:]),
            len(reg._sheaths), dst, hb.scheduler.progress)

print('exub | top1  | 余弦  | 髓鞘 | 指向各通道的边      | progress')
print('-'*68)
for exub in [0,2,4,8,12,24]:
    a,c,n,dst,pr = run(exub)
    print(f'{exub:4d} | {a:.3f} | {c:.3f} | {n:4d} | {str(dst):20s} | {pr:+.3f}')
print()
print(f'随机基线 top1={1/12:.3f}   一阶 Oracle=0.304')
