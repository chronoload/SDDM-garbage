"""扫描两级确认阈值：wire（成边）× myelination（髓鞘化）"""
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
def run(wire, myel, seed=0, steps=2500):
    vis,aud,lang=B.build_world(seed)
    sents=B.make_sentences(seed=seed); stream=[]
    for s in sents: stream.extend(s)
    np.random.seed(seed)
    brain=B.build_brain(seed)
    brain.reflex._kernel.register('vocal', ReflexVis())
    hb=brain.higher_brain
    hb.sheath_registry.wire_threshold = wire
    hb.sheath_registry.myelination_threshold = myel
    n0=hb.ecosystem.get_neuron(0)
    for c in B.CH: n0.unfold(c, shim.zeros(D))
    reg=hb.sheath_registry; hits=[]
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
    k=max(1,len(hits)//3)
    return dict(top1=np.mean(hits[-k:]), conns=len(reg._connections),
                sheaths=len(reg._sheaths), turnover=reg._born+reg._died,
                neurons=len(hb.ecosystem.neurons))

print("="*78)
print("两级确认阈值扫描（真实系统，2500 步）")
print("="*78)
print(f"{'wire':>5} {'myel':>5} | {'top1':>6} | {'连接':>5} | {'髓鞘':>5} | {'周转':>6} | {'神经元':>6}")
print("-"*78)
for wire in [1, 2, 3]:
    for myel in [3, 6, 10]:
        if myel <= wire: continue
        r = run(wire, myel)
        print(f"{wire:5d} {myel:5d} | {r['top1']:.3f} | {r['conns']:5d} | "
              f"{r['sheaths']:5d} | {r['turnover']:6d} | {r['neurons']:6d}")
print("-"*78)
print(f"随机基线 top1={1/12:.3f}   一阶 Oracle=0.341")
print("对照：分层前（混淆两层）周转 10194 / 分层后 1012")
