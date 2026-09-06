"""预测编码图片沉积物：信息结构审计 + 系统运行"""
import sys

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")
import numpy as np
import numpy_torch_shim as shim
shim.install()
import deposit_media as DM
from collections import Counter

# ---- 信息结构审计 ----
w = DM.ImageDepositPC(seed=0)
toks = []
for _ in range(20000):
    toks.append(w.cur_token())
    w._advance()
hist = np.bincount(toks, minlength=16) / len(toks)
ent = float(-(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-12)).sum())
maj = float(hist.max())
p = np.mean([1.0 if toks[i] == toks[i - 1] else 0.0
             for i in range(1, len(toks))])
# 一阶条件最优（bigram 上界）：给定上一 token 的最优预测命中率
trans = {}
for i in range(1, len(toks)):
    trans.setdefault(toks[i - 1], Counter())[toks[i]] += 1
hit = sum(max(trans[toks[i - 1]], key=trans[toks[i - 1]].get) == toks[i]
          for i in range(1, len(toks)))
print(f"创新量token: 熵={ent:.2f}bits(上限4)  majority={maj:.3f}  "
      f"persistence={p:.3f}  一阶条件最优={hit / (len(toks) - 1):.3f}")
print("对照 v1原值等级: majority=0.625 persistence=0.563 平凡支配")

# ---- 系统运行 ----
r = DM.run_media(DM.ImageDepositPC(seed=0), seed=0, steps=10000)
print(f"系统(预测编码编码, 1e4步): top1(后半)={r['top1']:.3f}  曲线="
      + " ".join(f"{v:.2f}" for v in r["curve"]))
