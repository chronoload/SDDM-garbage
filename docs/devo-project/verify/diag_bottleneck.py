"""瓶颈诊断：top1 卡住是「学不会」还是「分不开」？

两种可能：
  A 学不会：输出向量本身远离目标 → 需要更多容量/更好的学习
  B 分不开：输出已接近目标，但**同类词向量本身太近** → 最近邻解码
            必然误判，与学习无关，是表示方式的天花板

判据：比较 cos(输出, 真值) 与 cos(输出, 最强竞争者)
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
vis,aud,lang=B.build_world(0)
sents=B.make_sentences(seed=0); stream=[]
for s in sents: stream.extend(s)
np.random.seed(0)
brain=B.build_brain(0)
brain.reflex._kernel.register('vocal', ReflexVis())
hb=brain.higher_brain
n0=hb.ecosystem.get_neuron(0)
for c in B.CH: n0.unfold(c, shim.zeros(D))

# 先测：词向量本身的可分性（与学习无关，是表示的天花板）
print("="*72)
print("① 词向量可分性（与学习无关的天花板）")
print("="*72)
same, diff = [], []
for i,a in enumerate(B.WORDS):
    for b in B.WORDS[i+1:]:
        c = float(np.dot(lang[a],lang[b])/(np.linalg.norm(lang[a])*np.linalg.norm(lang[b])))
        (same if B.CAT_OF[a]==B.CAT_OF[b] else diff).append(c)
print(f"  同类词余弦: {np.mean(same):.3f}   异类词余弦: {np.mean(diff):.3f}")
print(f"  同类-异类差距: {np.mean(same)-np.mean(diff):+.3f}")
print(f"  → 同类词向量天然相近，最近邻解码难以区分个体")

# 跑训练，测输出与真值/竞争者的关系
print()
print("="*72)
print("② 训练中：输出 vs 真值 / 输出 vs 最强竞争者")
print("="*72)
print(f"{'步':>5} | {'cos(出,真)':>10} | {'cos(出,竞)':>10} | {'真>竞比例':>9} | {'top1':>6}")
print("-"*72)
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
    ct = cs(wn)
    # 最强竞争者 = 非真值中余弦最高者
    _comp = [s for s,x in sims if x!=wn]
    comp = max(_comp) if _comp else 0.0   # 标量，不是元组
    recs.append((ct, comp, sims[0][1]==wn))
    if (t+1)%500==0:
        r=recs[-300:]
        print(f"{t+1:5d} | {np.mean([x[0] for x in r]):10.3f} | "
              f"{np.mean([x[1] for x in r]):10.3f} | "
              f"{np.mean([x[0]>x[1] for x in r]):9.3f} | {np.mean([x[2] for x in r]):6.3f}")
print("-"*72)
r=recs[-500:]
print()
print(f"末 500 步：cos(出,真)={np.mean([x[0] for x in r]):.3f}  "
      f"cos(出,竞)={np.mean([x[1] for x in r]):.3f}")
print(f"          真值余弦 > 竞争者余弦 的比例 = {np.mean([x[0]>x[1] for x in r]):.3f}")
print()
if np.mean([x[0] for x in r]) > 0.4 and np.mean([x[0]>x[1] for x in r]) < 0.5:
    print("结论：**分不开**（瓶颈 B）")
    print("  输出已具备语义（cos 与真值不低），但同类词向量太近，")
    print("  最近邻解码无法区分个体 → 调学习机制无效，需换表示/解码。")
else:
    print("结论：**学不会**（瓶颈 A）—— 输出与真值的余弦本身就低。")

# ============================================================
# ③ 若评价标准从「个体正确」改为「类别正确」（合理反应）？
#
# 设计诉求："重点是合理的反应而不是预测"。
# 一个上下文可以有多个合理反应（"chi" 后接 pingguo 或 xigua 都合理），
# 线性模型输出它们的**平均**，最近邻解码必然落空。
# 那么按"反应是否合理"评价，结果如何？
# ============================================================
print()
print("="*72)
print("③ 评价标准：个体正确 vs 类别正确（合理反应）")
print("="*72)
cat_hits = [B.CAT_OF[sims_top]==B.CAT_OF[tw] for sims_top, tw in
            zip([s[0][1] for s in []], [])]  # 占位，下面重算
