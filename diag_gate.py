"""诊断：分化是需求驱动还是闸门节拍驱动？

假设：若神经元数 ≈ 总步数 / 不应期，则分化由**闸门节奏**决定，
而非由"需要更多容量"决定 —— 闸门从"抑制失控"异化成了"节拍器"。
"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
from mcp.developmental.signal import Signal
from mcp.developmental.reptilian import ReptilianFunction
import baby_real as B

class ReflexVis(ReptilianFunction):
    def get_input_spec(self): return {'vis':'generic/tensor'}
    def get_output_spec(self): return {'lang':'generic/tensor'}
    def execute(self, inputs):
        v=inputs['vis']
        return {'lang':Signal(data=v.data, mime_type='generic/tensor',
                              metadata={'source':'vis'})}

D=B.D
def run(refrac, steps=2500, seed=0):
    vis,aud,lang=B.build_world(seed)
    sents=B.make_sentences(seed=seed); stream=[]
    for s in sents: stream.extend(s)
    np.random.seed(seed)
    brain=B.build_brain(seed)
    brain.reflex._kernel.register('vocal', ReflexVis())
    hb=brain.higher_brain
    hb.engine.diff_refractory = refrac
    n0=hb.ecosystem.get_neuron(0)
    for c in B.CH: n0.unfold(c, shim.zeros(D))
    hits=[]
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
    return (len(hb.ecosystem.neurons), hb.engine._differentiation_count,
            np.mean(hits[-300:]) if hits else 0)

print("="*72)
print("不应期 × 神经元数（2500 步）")
print("="*72)
print(f"{'不应期':>6} | {'预测 neurons=2500/不应期':>22} | {'实测':>6} | {'分化次数':>8} | {'top1':>6}")
print("-"*72)
for rf in [50, 100, 200, 400, 800]:
    n, d, a = run(rf)
    pred = 2500 // rf
    print(f"{rf:6d} | {pred:22d} | {n:6d} | {d:8d} | {a:.3f}")
print("-"*72)
print("若「实测」紧跟「预测」→ 分化由闸门节拍驱动，非需求驱动（机制不平衡）")
print(f"随机基线 top1={1/12:.3f}")
