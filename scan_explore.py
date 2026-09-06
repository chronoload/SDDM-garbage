"""探索率扫描：检验三因子学习能否让高层真正学会控制

这是**探索—利用权衡**的实测：

- 探索少 → 性能高（反射主导），但高层没机会学，永远接管不了
- 探索多 → 性能暂时下降，但高层有机会积累经验

关键判据：高层的成功率是否**随时间上升**。
若上升 → 三因子学习有效，只是需要探索预算
若持平 → 学习通路仍然不通
"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
import baby_tension as T
from mcp.developmental.signal import Signal


def run(eps, seed=0, steps=3000):
    np.random.seed(seed)
    w = T.TensionWorld(seed)
    brain = T.build_brain(seed)
    hb = brain.higher_brain
    n0 = hb.ecosystem.get_neuron(0)
    for c in T.PORT: n0.unfold(c, shim.zeros(T.D_ALL))

    succ, ctl = [], []
    for t in range(steps):
        # ⚠ 必须同时设置 eps_background：证据充分后 should_explore
        # 会切到 eps_background（默认 0.02），只设 eps 的话前 20 次
        # 探索后就降到 2% —— 实测 300 步里只有 7 步在探索。
        brain.explore_eps = eps
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


print("="*76)
print("探索率扫描：高层成功率是否随时间上升？（3 seeds × 3000 步）")
print("="*76)
print(f"{'探索率':>6} | {'高层成功率(前1/3→后1/3)':>24} | {'整体消解':>9} | {'胜任度':>7}")
print("-"*76)
for eps in [0.05, 0.15, 0.30, 0.50]:
    gains, allrelief, comps = [], [], []
    for s in range(3):
        succ, ctl = run(eps, seed=s)
        h = ctl == 'higher'
        n = len(succ); third = n // 3
        # 高层在**后半程** vs **前半程**的成功率
        def rate(a, b):
            m = np.zeros(n, bool); m[a:b] = True
            sel = m & h
            return np.mean(succ[sel] > 0) if sel.sum() >= 5 else np.nan
        early, late = rate(0, third), rate(2*third, n)
        gains.append((early, late))
        allrelief.append(np.mean(succ[-third:]))
        comps.append(0.0)
    e = np.nanmean([g[0] for g in gains]); l = np.nanmean([g[1] for g in gains])
    print(f"{eps:6.2f} | {e:.3f}  →  {l:.3f}        | {np.mean(allrelief):9.4f} |")
print("-"*76)
print(f"反射参考：成功率 ≈ 0.23，整体消解 ≈ 0.030")
print()
print("判读：")
print("  · 后段 > 前段  → 三因子学习有效，高层在积累能力")
print("  · 持平         → 学习通路仍不通（RPE 未真正驱动权重）")
