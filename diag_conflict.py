"""诊断：冲突检测中各量的实际分布，用于校准阈值"""
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

# 直接分析数据本身的冲突结构（不跑系统）
print("="*70)
print("任务本身的冲突结构（与学习无关）")
print("="*70)
from collections import defaultdict
nexts=defaultdict(list)
for s in sents:
    for i in range(len(s)-1): nexts[s[i]].append(s[i+1])

print(f"{'上文':>8} | {'可能的下文数':>12} | {'下文两两余弦(最小~最大)':>24}")
print("-"*70)
for w,ns in sorted(nexts.items()):
    uniq=sorted(set(ns))
    if len(uniq)<2: continue
    sims=[]
    for i,a in enumerate(uniq):
        for b in uniq[i+1:]:
            sims.append(float(np.dot(lang[a],lang[b])/(np.linalg.norm(lang[a])*np.linalg.norm(lang[b]))))
    print(f"{w:>8} | {len(uniq):12d} | {min(sims):10.3f} ~ {max(sims):.3f}")
print("-"*70)
allsims=[]
for w,ns in nexts.items():
    uniq=sorted(set(ns))
    for i,a in enumerate(uniq):
        for b in uniq[i+1:]:
            allsims.append(float(np.dot(lang[a],lang[b])/(np.linalg.norm(lang[a])*np.linalg.norm(lang[b]))))
print(f"所有歧义下文的两两余弦: 均值 {np.mean(allsims):.3f}  "
      f"范围 {min(allsims):.3f}~{max(allsims):.3f}")
print()
print("→ conflict_div_threshold 必须**高于**这些值才能判定为互斥。")
print("  当前设 0.3 远低于 0.6 量级，故冲突永远检测不到。")
