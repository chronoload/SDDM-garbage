"""反射类型消融：反射基线量级对高层学习的影响

假设：反射应提供**量级合理**的起点。babbling（零附近噪声）虽更符合
生物学，但基线太弱导致高层输出幅度上不去（实测 λ 已达 0.94 接管，
但 ar_high≈|target|² 说明输出趋零）。
"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
from mcp.developmental.signal import Signal
from mcp.developmental.reptilian import ReptilianFunction
import baby_real as B

D=B.D

class ReflexVis(ReptilianFunction):
    """映射型：把 vis 当 lang 输出（量级合理但语义错误）"""
    def get_input_spec(self): return {"vis":"generic/tensor"}
    def get_output_spec(self): return {"lang":"generic/tensor"}
    def execute(self, inputs):
        v=inputs["vis"]
        return {"lang":Signal(data=v.data, mime_type="generic/tensor",
                              metadata={"source":"vis_map"})}

class ReflexBabble(ReptilianFunction):
    """babbling：零附近随机发声"""
    def __init__(self, amp=0.1, seed=0):
        self._amp=amp; self._rng=np.random.default_rng(seed)
    def get_input_spec(self): return {"vis":"generic/tensor"}
    def get_output_spec(self): return {"lang":"generic/tensor"}
    def execute(self, inputs):
        v=inputs["vis"]
        n=int(np.asarray(v.data.a if hasattr(v.data,"a") else v.data).size)
        return {"lang":Signal(data=shim.tensor(self._rng.normal(0,self._amp,n).astype(np.float32)),
                              mime_type="generic/tensor",metadata={"source":"babble"})}

def run(reflex_cls, seed=0, steps=1200):
    vis,aud,lang=B.build_world(seed)
    sents=B.make_sentences(seed=seed); stream=[]
    for s in sents: stream.extend(s)
    np.random.seed(seed)
    brain=B.build_brain(seed)
    k=brain.reflex._kernel
    k.register("vocal", reflex_cls())
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
    return (np.mean(hits[-300:]), np.mean(h3[-300:]), np.mean(coses[-300:]),
            hb.scheduler.progress, len(hb.sheath_registry._sheaths))

print("="*74)
print("反射类型消融（真实系统，1200 步）")
print("="*74)
print(f"{'反射类型':>12} | {'top1':>6} | {'top3':>6} | {'余弦':>6} | {'progress':>8} | {'髓鞘':>4}")
print("-"*74)
for name, cls in [("vis映射", ReflexVis), ("babbling", ReflexBabble)]:
    a,b,c,p,n = run(cls)
    print(f"{name:>12} | {a:.3f} | {b:.3f} | {c:.3f} | {p:+8.3f} | {n:4d}")
print("-"*74)
print(f"随机基线     top1={1/12:.3f}  top3={3/12:.3f}")
print("一阶 Oracle  top1=0.341")
