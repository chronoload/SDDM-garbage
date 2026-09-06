"""长程学习曲线：三因子学习能否让高层持续改善？"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
import baby_tension as T
from mcp.developmental.signal import Signal


def run(eps, seed=0, steps=8000):
    np.random.seed(seed)
    w = T.TensionWorld(seed)
    brain = T.build_brain(seed)
    hb = brain.higher_brain
    n0 = hb.ecosystem.get_neuron(0)
    for c in T.PORT: n0.unfold(c, shim.zeros(T.D_ALL))
    succ, ctl = [], []
    for t in range(steps):
        brain.drive_sat.eps = eps
        brain.drive_sat.eps_background = eps
        obs = w.observe()
        inputs = {c: Signal(shim.tensor(obs[o:o+s].copy()),'generic/tensor',{'t':t})
                  for c,(o,s) in T.PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        out = (res.get('outputs') or {}).get('lang')
        r = w.act(T.decode_lang(out))
        brain.report_drive_satisfaction(max(0.0, r))
        succ.append(max(0.0, r)); ctl.append(brain._last_controller)
    return np.array(succ), np.array(ctl)


print("="*70)
print("高层成功率随时间的演化（eps=0.3，4 seeds × 8000 步）")
print("="*70)
allc = []
for s in range(4):
    succ, ctl = run(0.3, seed=s)
    h = ctl == 'higher'
    n = len(succ); seg = n // 8
    curve = []
    for i in range(8):
        m = np.zeros(n, bool); m[i*seg:(i+1)*seg] = True
        sel = m & h
        curve.append(np.mean(succ[sel] > 0) if sel.sum() >= 5 else np.nan)
    allc.append(curve)
    print(f"  seed{s}: " + "  ".join(
        f"{v:.2f}" if not np.isnan(v) else " -- " for v in curve))
print("-"*70)
mean = np.nanmean(np.array(allc), axis=0)
print("  均值  : " + "  ".join(f"{v:.3f}" for v in mean))
print()
print(f"  首段 {mean[0]:.3f}  →  末段 {mean[-1]:.3f}   变化 {mean[-1]-mean[0]:+.3f}")
print(f"  反射参考 ≈ 0.23")
print()
if mean[-1] > mean[0] + 0.02:
    print("  → 三因子学习**有效**：高层在持续积累能力")
else:
    print("  → 学习曲线平坦：三因子未产生持续改善")
