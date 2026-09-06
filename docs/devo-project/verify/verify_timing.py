"""验证时序修复：冲突检测是否恢复、分化是否由需求驱动"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0, '/data/workspace')
import numpy as np
from mcp.developmental.signal import Signal
import baby_real as B

vis, aud, lang = B.build_world(0)
sents = B.make_sentences(seed=0)
stream = []
for s in sents:
    stream.extend(s)

np.random.seed(0)
brain = B.build_brain(0)
hb = brain.higher_brain
n0 = hb.ecosystem.get_neuron(0)
for c in B.CH:
    n0.unfold(c, shim.zeros(B.D))
eng = hb.engine
eng.diff_refractory = 10          # 放开闸门，让需求主导
hits = []

print(f"{'步':>5} | {'冲突':>5} | {'累计':>6} | {'神经元':>6} | {'分化':>5} | {'触发':>10} | {'top1':>6}")
print("-" * 62)
for t in range(1500):
    w = stream[t % len(stream)]
    wn = stream[(t + 1) % len(stream)]
    inputs = {
        'vis': Signal(shim.tensor(vis[w].copy()), 'generic/tensor', {'t': t}),
        'aud': Signal(shim.tensor(aud[w].copy()), 'generic/tensor', {'t': t}),
        'lang': Signal(shim.tensor(lang[w].copy()), 'generic/tensor', {'t': t}),
    }
    res = brain.step_awake(external_inputs=inputs)
    lg = (res.get('outputs') or {}).get('lang')
    if lg is not None:
        d = lg.data.a if hasattr(lg.data, 'a') else lg.data
        hits.append(B.decode_lang(d) == wn)
    if (t + 1) % 300 == 0:
        r = np.mean(hits[-200:]) if hits else 0
        print(f"{t+1:5d} | {eng._conflict_count:5d} | {eng._obs_conflicts:6d} | "
              f"{len(hb.ecosystem.neurons):6d} | {eng._differentiation_count:5d} | "
              f"{getattr(eng,'_last_trigger','-'):>10s} | {r:.3f}")
print("-" * 62)
print(f"随机基线={1/12:.3f}   一阶 Oracle=0.341")
