"""最终评估：真实系统上的婴幼儿语言习得（3 seeds）

对比四种配置，量化各阶段修复的贡献：
  A 基线       ：原始代码（惊奇关闭、无生长闸门、单阈值豁免）
  B +惊奇      ：NMSE 惊奇通道（解锁结构生长）
  C +生长闸门  ：不应期 + 误差平台（防失控增殖）
  D +滞回豁免  ：双水位（防死锁 + 防振荡）  ← 当前实现
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
SEEDS=[0,1,2]
STEPS=2000

def run(cfg, seed):
    vis,aud,lang=B.build_world(seed)
    sents=B.make_sentences(seed=seed); stream=[]
    for s in sents: stream.extend(s)
    np.random.seed(seed)
    brain=B.build_brain(seed)
    brain.reflex._kernel.register('vocal', ReflexVis())
    hb=brain.higher_brain
    reg=hb.sheath_registry; eng=hb.engine
    if cfg=='A':
        eng.surprise_scale=0.0
        eng.diff_refractory=0; eng.plateau_eps=1e9   # 闸门恒通过（禁用）
        reg.exuberant_low=0; reg.exuberant_high=10**9  # 恒豁免（单阈值行为）
    elif cfg=='B':
        eng.surprise_scale=1.0
        eng.diff_refractory=0; eng.plateau_eps=1e9
        reg.exuberant_low=0; reg.exuberant_high=10**9
    elif cfg=='C':
        eng.surprise_scale=1.0
        eng.diff_refractory=200; eng.plateau_eps=0.02
        reg.exuberant_low=0; reg.exuberant_high=10**9
    else:  # D
        eng.surprise_scale=1.0
        eng.diff_refractory=200; eng.plateau_eps=0.02
        reg.exuberant_low=2; reg.exuberant_high=10
    n0=hb.ecosystem.get_neuron(0)
    for c in B.CH: n0.unfold(c, shim.zeros(D))
    hits,h3,coses=[],[],[]
    for t in range(STEPS):
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
    k=max(1,len(hits)//3)
    return dict(top1=np.mean(hits[-k:]), top3=np.mean(h3[-k:]),
                cos=np.mean(coses[-k:]),
                neurons=len(hb.ecosystem.neurons),
                sheaths=len(reg._sheaths),
                turnover=reg._born+reg._died,
                diffs=eng._differentiation_count,
                prog=hb.scheduler.progress)

print("="*84)
print(f"最终评估（真实 DevelopmentalSystem，{len(SEEDS)} seeds × {STEPS} 步）")
print("="*84)
print(f"{'配置':>22} | {'top1':>13} | {'top3':>6} | {'余弦':>6} | {'神经元':>5} | {'髓鞘':>4} | {'周转':>5} | {'分化':>4}")
print("-"*84)
names={'A':'A 基线（原始）',
       'B':'B +NMSE 惊奇',
       'C':'C +生长闸门',
       'D':'D +滞回豁免（当前）'}
for cfg in ['A','B','C','D']:
    rs=[run(cfg,sd) for sd in SEEDS]
    m=lambda k: np.mean([r[k] for r in rs])
    s=lambda k: np.std([r[k] for r in rs])
    print(f"{names[cfg]:>22} | {m('top1'):.3f}±{s('top1'):.3f} | {m('top3'):.3f} | {m('cos'):.3f} | "
          f"{m('neurons'):5.1f} | {m('sheaths'):4.1f} | {m('turnover'):5.0f} | {m('diffs'):4.1f}")
print("-"*84)
print(f"随机基线 top1={1/12:.3f}  top3={3/12:.3f}")
print("一阶 Oracle top1=0.341")
