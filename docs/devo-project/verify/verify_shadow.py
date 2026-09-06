"""验证：影子模式 + 驱动满足度裁定，能否修好 λ 陷阱

对照三组：
  A 旧行为（λ 由预测误差裁定，无影子模式）
  B 纯反射（上界参考）
  C 影子模式 + 满足度裁定
"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
import baby_loop as L
from mcp.developmental.signal import Signal


def run(mode, seed=0, steps=1500, explore_eps=0.1):
    np.random.seed(seed)
    world = L.ReactiveWorld(seed)
    brain = L.build_brain(seed)
    brain.explore_eps = explore_eps
    ref = brain.reflex._kernel.get("babble")
    hb = brain.higher_brain
    n0 = hb.ecosystem.get_neuron(0)
    for c in L.PORT: n0.unfold(c, shim.zeros(L.D_ALL))

    if mode == "old":
        # 旧行为：关掉影子模式，λ 走预测误差裁定
        brain.shadow_mode = False
        # 移除 drive_sat 引用 → compute_lambda 退化到 scheduler 路径
        hb.drive_sat = None

    rewards, lambdas = [], []
    for t in range(steps):
        ref.set_drive(world.want)
        obs = world.observe()
        inputs = {c: Signal(shim.tensor(obs[o:o+s].copy()), 'generic/tensor', {'t':t})
                  for c,(o,s) in L.PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        out = (res.get('outputs') or {}).get('lang')
        w = L.decode_lang(out)
        r, _ = world.act(w)
        if mode != "old":
            brain.report_drive_satisfaction(r)
        rewards.append(r)
        lambdas.append(getattr(hb, '_last_lambda', 0.0))

    k = max(1, len(rewards)//3)
    return dict(sat=float(np.mean(rewards[-k:])),
                lam=float(np.mean(lambdas[-k:])),
                shadow=bool(brain.shadow_mode),
                comp=float(brain.drive_sat.competence),
                nr=int(brain.drive_sat.report()['n_reflex']),
                nh=int(brain.drive_sat.report()['n_high']))


print("="*76)
print("影子模式 + 驱动满足度    vs    旧的 λ 预测误差裁定")
print("3 seeds × 1500 步")
print("="*76)
print(f"{'模式':>16} | {'满足率':>14} | {'末段λ':>7} | {'影子中':>6} | {'胜任度':>7} | {'样本(反/高)':>12}")
print("-"*76)
res = {}
for mode, name in (("old","旧：预测误差裁定"),
                   ("reflex_only","参考：强制反射"),
                   ("new","新：影子+满足度")):
    if mode == "reflex_only":
        # 强制反射：影子模式常开且不探索
        rs = []
        for s in range(3):
            np.random.seed(s)
            world = L.ReactiveWorld(s); brain = L.build_brain(s)
            brain.explore_eps = 0.0
            ref = brain.reflex._kernel.get("babble")
            hb = brain.higher_brain
            n0 = hb.ecosystem.get_neuron(0)
            for c in L.PORT: n0.unfold(c, shim.zeros(L.D_ALL))
            rr=[]
            for t in range(1500):
                ref.set_drive(world.want)
                obs = world.observe()
                inputs={c: Signal(shim.tensor(obs[o:o+sz].copy()),'generic/tensor',{'t':t})
                        for c,(o,sz) in L.PORT.items()}
                brain.step_awake(external_inputs=inputs)
                rr.append(1.0 if L.decode_lang(
                    ref.execute({'lang': inputs['lang']})['lang']) == world.want else 0.0)
                world.act(L.decode_lang(ref.execute({'lang': inputs['lang']})['lang']))
            rs.append(dict(sat=float(np.mean(rr[-500:])), lam=0.0, shadow=True,
                           comp=0.0, nr=0, nh=0))
    else:
        rs = [run(mode, seed=s) for s in range(3)]
    res[mode] = rs
    m = lambda k: np.mean([r[k] for r in rs])
    print(f"{name:>16} | {m('sat'):.3f}±{np.std([r['sat'] for r in rs]):.3f} | "
          f"{m('lam'):7.3f} | {str(bool(m('shadow'))):>6} | {m('comp'):7.3f} | "
          f"{int(m('nr')):5d}/{int(m('nh')):<6d}")
print("-"*76)
print(f"随机基线 = 1/12 = {1/12:.3f}")
print()
old = np.mean([r['sat'] for r in res['old']])
new = np.mean([r['sat'] for r in res['new']])
base = np.mean([r['sat'] for r in res['reflex_only']])
print(f"旧方案 {old:.3f}  →  新方案 {new:.3f}   (反射上界 {base:.3f})")
print(f"修复幅度: {new/old:.1f}×    占反射上界的 {new/base*100:.0f}%")
