"""谁在控制？纯反射 vs 高层接管

关键问题：高层脑**看不到**内在驱动（want 在爬虫脑里）。
那么在这个控制 loop 里，它接管是帮忙还是帮倒忙？
"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
import baby_loop as L
from mcp.developmental.signal import Signal

def run(mode, seed=0, steps=1500):
    """mode: 'reflex'=强制纯反射  'higher'=强制高层  'auto'=λ自适应"""
    np.random.seed(seed)
    world = L.ReactiveWorld(seed)
    brain = L.build_brain(seed)
    ref = brain.reflex._kernel.get("babble")
    hb = brain.higher_brain
    n0 = hb.ecosystem.get_neuron(0)
    for c in L.PORT: n0.unfold(c, shim.zeros(L.D_ALL))

    rewards = []
    for t in range(steps):
        ref.set_drive(world.want)
        obs = world.observe()
        inputs = {c: Signal(shim.tensor(obs[o:o+s].copy()), 'generic/tensor', {'t':t})
                  for c,(o,s) in L.PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        out = (res.get('outputs') or {}).get('lang')

        if mode == 'reflex':
            # 强制用反射输出（绕过高层混合）
            w = L.decode_lang(ref.execute({'lang': inputs['lang']})['lang'])
        elif mode == 'higher':
            w = L.decode_lang(out)
        else:
            w = L.decode_lang(out)      # auto 就是系统自己的混合结果
        r, _ = world.act(w)
        rewards.append(r)
    k = max(1, len(rewards)//3)
    return np.mean(rewards[-k:])

print("="*68)
print("谁在满足需求？（3 seeds × 1500 步，末 1/3 满足率）")
print("="*68)
print(f"{'控制者':>16} | {'满足率':>16} | 说明")
print("-"*68)
for mode, desc in (('reflex','反射（知道 want）'),
                   ('higher','高层（看不到 want）'),
                   ('auto','λ 自适应混合')):
    rs = [run(mode, seed=s) for s in range(3)]
    print(f"{desc:>16} | {np.mean(rs):.3f}±{np.std(rs):.3f}   |")
print("-"*68)
print(f"随机基线 = 1/12 = {1/12:.3f}")
print()
print("判读：")
print("  · 反射 >> 高层 → 高层看不到 want 却在接管，λ 自适应失效")
print("  · 高层 ≈ 反射 → 高层成功蒸馏了反射的能力")
print("  · 高层 > 反射  → 高层超越了反射（真正的学会控制）")
