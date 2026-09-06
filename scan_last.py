"""定位：Oja 归一化 / 通道过滤 各自的影响（2x2）"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
from mcp.developmental.signal import Signal
from mcp.developmental.reptilian import ReptilianFunction
import baby_real as B

D=B.D

class ReflexVis(ReptilianFunction):
    def get_input_spec(self): return {"vis":"generic/tensor"}
    def get_output_spec(self): return {"lang":"generic/tensor"}
    def execute(self, inputs):
        v=inputs["vis"]
        return {"lang":Signal(data=v.data, mime_type="generic/tensor",
                              metadata={"source":"vis_map"})}

def run(norm=1.1, seed=0, steps=1200):
    vis,aud,lang=B.build_world(seed)
    sents=B.make_sentences(seed=seed); stream=[]
    for s in sents: stream.extend(s)
    np.random.seed(seed)
    brain=B.build_brain(seed)
    brain.reflex._kernel.register("vocal", ReflexVis())
    brain.autoreg_w_norm = norm
    hb=brain.higher_brain
    n0=hb.ecosystem.get_neuron(0)
    for c in B.CH: n0.unfold(c, shim.zeros(D))
    hits,h3,coses=[],[],[]
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
        h3.append(wn in [x for _,x in sims[:3]])
        coses.append(float(np.dot(v,lang[wn])/(np.linalg.norm(v)*np.linalg.norm(lang[wn])+1e-9)))
    nns=[n for n in hb.ecosystem.neurons.values() if getattr(n,'W',None) is not None]
    wn_ = max([float(n.W.norm().item()) for n in nns]) if nns else 0
    return (np.mean(hits[-300:]), np.mean(h3[-300:]), np.mean(coses[-300:]), wn_)

print("="*70)
print("Oja 归一化影响（真实系统，1200 步，vis映射反射）")
print("="*70)
print(f"{'normalize_to':>13} | {'top1':>6} | {'top3':>6} | {'余弦':>6} | {'W范数':>6}")
print("-"*70)
for nv in [0.0, 0.5, 1.1, 3.0]:
    a,b,c,w = run(nv)
    print(f"{nv:13.1f} | {a:.3f} | {b:.3f} | {c:.3f} | {w:6.2f}")
print("-"*70)
print(f"随机基线 top1={1/12:.3f}  top3={3/12:.3f}")
print("一阶 Oracle top1=0.341")
print("早先峰值（未加归一化/通道过滤时）: top1=0.268  余弦=0.751")
