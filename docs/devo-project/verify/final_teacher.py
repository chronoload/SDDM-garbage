"""最终验证：教师强制下系统的真实上限（多种子）

区分"理解/预测"与"自持生成"：
- 教师强制：每步给真实上下文，测系统能否预测下一个词 → 理解能力
- 自反馈产出：用自己上一步输出当输入 → 生成能力
"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
from mcp.developmental.signal import Signal
import baby_real as B

D=B.D
SEEDS=[0,1,2]
STEPS=1200

print("="*72)
print("教师强制 vs 自持生成（真实系统，3 seeds）")
print("="*72)
print(f"{'seed':>4} | {'top1':>6} | {'top3':>6} | {'余弦':>6} | {'髓鞘':>5} | {'神经元':>6}")
print("-"*72)

t1,t3,cs=[],[],[]
for sd in SEEDS:
    vis,aud,lang=B.build_world(sd)
    sents=B.make_sentences(seed=sd); stream=[]
    for s in sents: stream.extend(s)
    brain=B.build_brain(sd); hb=brain.higher_brain
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
    a=np.mean(hits[-300:]); b=np.mean(h3[-300:]); c=np.mean(coses[-300:])
    t1.append(a); t3.append(b); cs.append(c)
    print(f"{sd:4d} | {a:.3f} | {b:.3f} | {c:.3f} | {len(hb.sheath_registry._sheaths):5d} | {len(hb.ecosystem.neurons):6d}")

print("-"*72)
print(f"均值   | {np.mean(t1):.3f} | {np.mean(t3):.3f} | {np.mean(cs):.3f} |")
print()
print(f"随机基线     top1={1/12:.3f}  top3={3/12:.3f}")
# Oracle
from collections import Counter, defaultdict
vis,aud,lang=B.build_world(0)
sents=B.make_sentences(seed=0)
trans=defaultdict(Counter)
for s in sents:
    for i in range(len(s)-1): trans[s[i]][s[i+1]]+=1
orc=np.mean([trans[s[i]][s[i+1]]/sum(trans[s[i]].values()) for s in sents for i in range(len(s)-1)])
print(f"一阶 Oracle  top1={orc:.3f}")
print()
print(f"教师强制 top1 / Oracle = {np.mean(t1)/orc:.2f}")
print("（自持生成见 final_ablation.py：top1≈0.11，远低于此）")
