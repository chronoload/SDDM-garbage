"""结构解锁后：性能是否随之提升？长跑 + 周转监控"""
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
vis,aud,lang=B.build_world(0)
sents=B.make_sentences(seed=0); stream=[]
for s in sents: stream.extend(s)
np.random.seed(0)
brain=B.build_brain(0)
brain.reflex._kernel.register('vocal', ReflexVis())
hb=brain.higher_brain
n0=hb.ecosystem.get_neuron(0)
for c in B.CH: n0.unfold(c, shim.zeros(D))
eng=hb.engine; reg=hb.sheath_registry
hits=[]
print('步   | top1  | 神经元 | 髓鞘 | 分化 | 周转  | nov   | surp  | progress')
print('-'*74)
for t in range(3000):
    w=stream[t%len(stream)]; wn=stream[(t+1)%len(stream)]
    inputs={'vis':Signal(shim.tensor(vis[w].copy()),'generic/tensor',{'t':t}),
            'aud':Signal(shim.tensor(aud[w].copy()),'generic/tensor',{'t':t}),
            'lang':Signal(shim.tensor(lang[w].copy()),'generic/tensor',{'t':t})}
    res=brain.step_awake(external_inputs=inputs)
    lg=(res.get('outputs') or {}).get('lang')
    if lg is not None:
        v=np.asarray(lg.data.a if hasattr(lg.data,'a') else lg.data,dtype=float).ravel()
        hits.append(B.decode_lang(v)==wn)
    if (t+1)%500==0:
        r1=np.mean(hits[-300:]) if hits else 0
        print(f'{t+1:4d} | {r1:.3f} | {len(hb.ecosystem.neurons):6d} | {len(reg._sheaths):4d} | '
              f'{eng._differentiation_count:4d} | {reg._born+reg._died:5d} | {eng._novelty_accumulator:.3f} | '
              f'{eng._last_surprise:.3f} | {hb.scheduler.progress:+.3f}')
print('-'*74)
print(f'随机基线 top1={1/12:.3f}   一阶 Oracle=0.341')
