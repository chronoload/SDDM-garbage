"""诊断：系统到底学到了什么？分离 lang 段与全通道"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
from mcp.developmental.signal import Signal
import baby_real as B

vis, aud, lang = B.build_world(0)
sents = B.make_sentences(seed=0)
stream = []
for s in sents:
    stream.extend(s)

brain = B.build_brain(0)
hb = brain.higher_brain
n0 = hb.ecosystem.get_neuron(0)
for c in B.CH: n0.unfold(c, shim.zeros(B.D))

D = B.D
# 教师强制：给真实上下文，看系统输出的 lang 段能否预测下一词
def probe(t, w, w_next):
    inputs = {'vis': Signal(shim.tensor(vis[w].copy()),'generic/tensor',{'t':t}),
              'aud': Signal(shim.tensor(aud[w].copy()),'generic/tensor',{'t':t}),
              'lang': Signal(shim.tensor(lang[w].copy()),'generic/tensor',{'t':t})}
    res = brain.step_awake(external_inputs=inputs)
    outs = res.get('outputs') or {}
    lg = outs.get('lang')
    if lg is None: return None, None
    v = np.asarray(lg.data.a if hasattr(lg.data,'a') else lg.data, dtype=float).ravel()[:D]
    return v, res

print('步数 | 下一词top1 | top3 | lang段余弦 | 髓鞘 | 周转 | progress')
print('-'*62)
hits1, hits3, coses = [], [], []
for t in range(2000):
    w, w_next = stream[t%len(stream)], stream[(t+1)%len(stream)]
    v, res = probe(t, w, w_next)
    if v is None: continue
    # 余弦相似度排序
    sims = [(float(np.dot(v, lang[x])/(np.linalg.norm(v)*np.linalg.norm(lang[x])+1e-9)), x)
            for x in B.WORDS]
    sims.sort(reverse=True)
    top3 = [x for _,x in sims[:3]]
    hits1.append(sims[0][1]==w_next)
    hits3.append(w_next in top3)
    coses.append(float(np.dot(v, lang[w_next])/(np.linalg.norm(v)*np.linalg.norm(lang[w_next])+1e-9)))
    if (t+1)%250==0:
        r1=np.mean(hits1[max(0,len(hits1)-100):]); r3=np.mean(hits3[max(0,len(hits3)-100):])
        cc=np.mean(coses[max(0,len(coses)-100):])
        reg=hb.sheath_registry
        print(f'{t+1:5d} | {r1:9.3f} | {r3:.3f} | {cc:9.4f} | {len(reg._sheaths):4d} | '
              f'{reg._born+reg._died:4d} | {hb.scheduler.progress:+.3f}')
print('-'*62)
print(f'随机基线 top1={1/12:.3f} top3={3/12:.3f}')
print(f'一阶转移 Oracle:', end=' ')
from collections import Counter, defaultdict
trans=defaultdict(Counter)
for i in range(len(stream)-1): trans[stream[i]][stream[i+1]]+=1
orc=np.mean([trans[s[i]][s[i+1]]/sum(trans[s[i]].values()) for s in sents for i in range(len(s)-1)])
print(f'{orc:.3f}')
