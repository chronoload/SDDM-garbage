"""诊断：NMSE 惊奇能否激活分化与结构生长"""
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
        return {'lang':Signal(data=v.data, mime_type='generic/tensor', metadata={'source':'vis'})}

D=B.D
def run(seed=0, steps=1500, surprise=True):
    vis,aud,lang=B.build_world(seed)
    sents=B.make_sentences(seed=seed); stream=[]
    for s in sents: stream.extend(s)
    np.random.seed(seed)
    brain=B.build_brain(seed)
    brain.reflex._kernel.register('vocal', ReflexVis())
    hb=brain.higher_brain
    hb.engine.surprise_scale = 1.0 if surprise else 0.0
    n0=hb.ecosystem.get_neuron(0)
    for c in B.CH: n0.unfold(c, shim.zeros(D))
    eng=hb.engine; hits=[]
    for t in range(steps):
        w=stream[t%len(stream)]; wn=stream[(t+1)%len(stream)]
        inputs={'vis':Signal(shim.tensor(vis[w].copy()),'generic/tensor',{'t':t}),
                'aud':Signal(shim.tensor(aud[w].copy()),'generic/tensor',{'t':t}),
                'lang':Signal(shim.tensor(lang[w].copy()),'generic/tensor',{'t':t})}
        res=brain.step_awake(external_inputs=inputs)
        lg=(res.get('outputs') or {}).get('lang')
        if lg is not None:
            v=np.asarray(lg.data.a if hasattr(lg.data,'a') else lg.data,dtype=float).ravel()
            hits.append(B.decode_lang(v)==wn)
    return dict(top1=np.mean(hits[-300:]) if hits else 0,
                neurons=len(hb.ecosystem.neurons),
                sheaths=len(hb.sheath_registry._sheaths),
                diffs=eng._differentiation_count,
                trigger=getattr(eng,'_last_trigger','-'),
                nov=eng._novelty_accumulator,
                surprise=eng._last_surprise,
                energy=eng._target_energy)

print("="*78)
print("NMSE 惊奇对分化的影响（真实系统，1500 步）")
print("="*78)
print(f"{'惊奇':>6} | {'top1':>6} | {'神经元':>6} | {'髓鞘':>5} | {'分化':>5} | {'触发':>10} | {'nov':>6} | {'surp':>6} | {'能量':>7}")
print("-"*78)
for sp in [False, True]:
    r = run(0, 1500, sp)
    print(f"{str(sp):>6} | {r['top1']:.3f} | {r['neurons']:6d} | {r['sheaths']:5d} | "
          f"{r['diffs']:5d} | {r['trigger']:>10s} | {r['nov']:.3f} | {r['surprise']:.3f} | {r['energy']:7.2f}")
print("-"*78)
print(f"随机基线 top1={1/12:.3f}   一阶 Oracle=0.341")
