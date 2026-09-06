"""诊断：冲突计数与 stable_output 的实际状态"""
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

vis,aud,lang=B.build_world(0)
sents=B.make_sentences(seed=0); stream=[]
for s in sents: stream.extend(s)
np.random.seed(0)
brain=B.build_brain(0)
brain.reflex._kernel.register('vocal', ReflexVis())
hb=brain.higher_brain
n0=hb.ecosystem.get_neuron(0)
for c in B.CH: n0.unfold(c, shim.zeros(B.D))
eng=hb.engine
eng.diff_refractory = 10

print(f"{'步':>5} | {'冲突计数':>8} | {'原型数':>6} | {'stable':>6} | {'神经元':>6} | {'分化':>5} | {'触发':>10}")
print("-"*72)
for t in range(1200):
    w=stream[t%len(stream)]
    inputs={'vis':Signal(shim.tensor(vis[w].copy()),'generic/tensor',{'t':t}),
            'aud':Signal(shim.tensor(aud[w].copy()),'generic/tensor',{'t':t}),
            'lang':Signal(shim.tensor(lang[w].copy()),'generic/tensor',{'t':t})}
    brain.step_awake(external_inputs=inputs)
    if (t+1)%200==0:
        print(f"{t+1:5d} | {eng._conflict_count:8d} | {len(eng._ctx_protos):6d} | "
              f"{str(eng.has_stable_output_reaction()):>6s} | {len(hb.ecosystem.neurons):6d} | "
              f"{eng._differentiation_count:5d} | {getattr(eng,'_last_trigger','-'):>10s}")
print("-"*72)
print(f"conflict_trigger = {eng.conflict_trigger}")
print(f"conflict_sim_threshold = {eng.conflict_sim_threshold}")
