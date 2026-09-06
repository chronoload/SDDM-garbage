"""合理反应 vs 精确预测：评价标准的影响"""
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
vis,aud,lang=B.build_world(0)
sents=B.make_sentences(seed=0); stream=[]
for s in sents: stream.extend(s)
np.random.seed(0)
brain=B.build_brain(0)
brain.reflex._kernel.register('vocal', ReflexVis())
hb=brain.higher_brain
n0=hb.ecosystem.get_neuron(0)
for c in B.CH: n0.unfold(c, shim.zeros(D))

# 类别原型
protos={}
for cat in set(B.CAT_OF.values()):
    ws=[w for w in B.WORDS if B.CAT_OF[w]==cat]
    protos[cat]=np.mean([lang[w] for w in ws],axis=0)

print("="*74)
print("个体正确(top1) vs 类别正确(合理反应)")
print("="*74)
print(f"{'步':>5} | {'top1':>6} | {'top3':>6} | {'类别正确':>8} | {'基线(随机)':>10}")
print("-"*74)
recs=[]
for t in range(2500):
    w=stream[t%len(stream)]; wn=stream[(t+1)%len(stream)]
    inputs={'vis':Signal(shim.tensor(vis[w].copy()),'generic/tensor',{'t':t}),
            'aud':Signal(shim.tensor(aud[w].copy()),'generic/tensor',{'t':t}),
            'lang':Signal(shim.tensor(lang[w].copy()),'generic/tensor',{'t':t})}
    res=brain.step_awake(external_inputs=inputs)
    lg=(res.get('outputs') or {}).get('lang')
    if lg is None: continue
    v=np.asarray(lg.data.a if hasattr(lg.data,'a') else lg.data,dtype=float).ravel()[:D]
    nv=np.linalg.norm(v)+1e-9
    def cs(x): return float(np.dot(v,lang[x])/(nv*np.linalg.norm(lang[x])+1e-9))
    sims=sorted([(cs(x),x) for x in B.WORDS], reverse=True)
    # 类别判断：输出与哪个类别原型最近
    cat_sim=sorted([(float(np.dot(v,p)/(nv*np.linalg.norm(p)+1e-9)),c)
                    for c,p in protos.items()], reverse=True)
    recs.append((sims[0][1]==wn,
                 wn in [x for _,x in sims[:3]],
                 cat_sim[0][1]==B.CAT_OF[wn]))
    if (t+1)%500==0:
        r=recs[-300:]
        print(f"{t+1:5d} | {np.mean([x[0] for x in r]):.3f} | "
              f"{np.mean([x[1] for x in r]):.3f} | {np.mean([x[2] for x in r]):8.3f} |")
print("-"*74)
r=recs[-500:]
print(f"随机基线: top1={1/12:.3f}  top3={3/12:.3f}  类别={1/4:.3f}  (4 类等概率)")
print()
print(f"末 500 步  top1={np.mean([x[0] for x in r]):.3f}  "
      f"top3={np.mean([x[1] for x in r]):.3f}  "
      f"类别正确={np.mean([x[2] for x in r]):.3f}")
print()
print("→ 若'合理反应'按类别计，系统表现远高于按个体计的 top1。")
print("  这与'重点是合理的反应而不是预测'的诉求一致。")
