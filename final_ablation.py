"""最终验证：感知校准期的价值（多种子）

核心假设："有时候输入倒是不必有输出，比如说看到爸爸妈妈，可能什么都不会干，
但是会对知觉有校准"

设计：前 X% 步为**感知校准期**（lang 来自环境，系统只做自回归预测、不产出）；
之后为**产出期**（打字机：lang 输入 = 系统自己上一步的输出）。
若校准期有价值，则 X>0 应显著优于 X=0。
"""
import numpy_torch_shim as shim; shim.install()
import sys; sys.path.insert(0,'/data/workspace')
import numpy as np
import baby_real as B

SEEDS = [0,1,2]
STEPS = 1000
CALIBS = [0.0, 0.2, 0.4, 0.6]

print("="*76)
print("感知校准期消融（真实 DevelopmentalSystem，3 seeds）")
print("="*76)
print(f"{'校准占比':>8} | {'下一词top1':>10} | {'top3':>6} | {'髓鞘':>5} | {'神经元':>6}")
print("-"*76)

results = {}
for cf in CALIBS:
    accs, t3s, shs, nns = [], [], [], []
    for sd in SEEDS:
        r = B.run(calib_frac=cf, steps=STEPS, seed=sd)
        if r["error"]:
            print(f"  seed {sd} 崩溃: {r['error']}")
            continue
        gw, tw = r["gen_words"], [str(x) for x in r["true_words"]]
        if not gw:
            continue
        acc = np.mean([g==t for g,t in zip(gw,tw)])
        # top3
        vis,aud,lang = B.build_world(sd)
        hit3=[]
        for g,t in zip(gw,tw):
            sims=[(float(np.dot(lang[g],lang[x])),x) for x in B.WORDS]
            sims.sort(reverse=True)
            hit3.append(t in [x for _,x in sims[:3]])
        accs.append(acc); t3s.append(np.mean(hit3))
        shs.append(r["sheaths"]); nns.append(r["neurons"])
    if not accs: continue
    results[cf]=(np.mean(accs),np.std(accs),np.mean(t3s),np.mean(shs),np.mean(nns))
    print(f"{cf:8.1f} | {np.mean(accs):.3f}±{np.std(accs):.3f} | {np.mean(t3s):.3f} | "
          f"{np.mean(shs):5.1f} | {np.mean(nns):6.1f}")

print("-"*76)
print(f"随机基线 top1={1/12:.3f}  top3={3/12:.3f}")
print()
if 0.0 in results and 0.2 in results:
    d = results[0.2][0]-results[0.0][0]
    print(f"校准期 0%→20% 的 top1 变化: {d:+.4f}")
    print("（此前 numpy 原型上该效应为 0.12→0.19；真实代码需独立验证）")
