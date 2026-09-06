"""检验超模态融合的价值：在"跨模态信息必要"的环境下

前一版环境中位置是**直接可观测**的，速度只提供冗余信息。线性系统下
s_next 就是 s_t 的线性函数，对角预测已经饱和，跨模态只增加方差不增加
信息——因此"跨模态成边有害"是环境的性质，不是机制的缺陷。

本环境改为**跨模态必要**：
- ch0/ch1（vision）：位置的观测，噪声大（std 0.3）
- ch10（event，异构信道）：同一位置的观测，噪声小（std 0.05）
- 预测 ch0 的下一帧时，用 ch0 自身会把 0.3 的观测噪声经增益放大后带入
  预测；用 ch10 则可规避——残差少掉一项 0.33·n。
- 因此跨通路 10→0 有**正的信息价值**，且必须跨信道才能获得。

这模拟现实中常见的多模态冗余观测（显微镜视觉读数噪声大但位置传感器
精确），是"超模态混合"真正应当发挥作用的场景。
"""
import numpy as np
import devo_control as dc
from devo_control import agg, D, T, SEEDS

# 跨模态必要的观测：位置在两个异构信道上有不同精度的观测
dc.REAL_DIMS = [0, 1, 6, 7, 10]
dc.NOISE_DIMS = [2, 3, 4, 5, 8, 9, 11]


def observe_fusion(s, s_prev, rng):
    o = np.zeros(D)
    vel = s - s_prev
    o[0] = s[0] + 0.30 * rng.normal()          # vision: 位置，噪声大
    o[1] = s[1] + 0.30 * rng.normal()
    o[2:6] = 1.5 * rng.normal(size=4)          # vision 纯噪声通道
    o[6:8] = vel + 0.05 * rng.normal(size=2)   # proprio: 速度，精确
    o[8:10] = 1.0 * rng.normal(size=2)         # proprio 纯噪声
    o[10] = s[0] + 0.05 * rng.normal()         # event: 位置，精确 ← 融合源
    o[11] = 0.8 * rng.normal()                 # event 纯噪声
    return o


dc.observe = observe_fusion   # monkeypatch：run() 内部引用模块级 observe

RHO = 0.025
print("=" * 96)
print("超模态融合检验：跨模态必要环境（ρ=0.025，seeds=8）")
print("=" * 96)
print(f"{'配置':<30}{'代价':>11}{'改进%':>9}{'通路':>8}{'有效':>7}{'跨模态':>8}")
print("-" * 96)
res = {}
for name, kw in [
    ("跨模态 OFF（仅对角）", dict(rho=RHO, allow_cross=False, wire=True, self_proof=0.0)),
    ("跨模态 ON, self_proof=0", dict(rho=RHO, allow_cross=True, wire=True, self_proof=0.0)),
    ("跨模态 ON, self_proof=0.15", dict(rho=RHO, allow_cross=True, wire=True, self_proof=0.15)),
    ("跨模态 ON, self_proof=0.30", dict(rho=RHO, allow_cross=True, wire=True, self_proof=0.30)),
    ("跨模态 ON, self_proof=0.50", dict(rho=RHO, allow_cross=True, wire=True, self_proof=0.50)),
]:
    r = agg(**kw)
    res[name] = r
    print(f"{name:<30}{r['cost_high'][0]:>11.4f}{r['improve'][0]*100:>8.1f}%"
          f"{r['n_paths'][0]:>8.1f}{r['n_eff'][0]:>7.1f}{r['cross'][0]:>8.1f}")

ref = res["跨模态 OFF（仅对角）"]["cost_high"][0]
best = min(res.items(), key=lambda x: x[1]["cost_high"][0])
print()
print(f"仅对角基线代价 = {ref:.4f}")
print(f"最优配置 = {best[0]}  代价 {best[1]['cost_high'][0]:.4f}  "
      f"相对仅对角 {'改善' if best[1]['cost_high'][0] < ref else '恶化'} "
      f"{abs(ref - best[1]['cost_high'][0])/ref*100:.1f}%")

print()
print("=" * 96)
print("对照：同一配置在「跨模态不必要」的原环境下的表现（来自 diag_selfproof）")
print("=" * 96)
print("  仅对角 94.3%  >  跨模态 ON + self_proof=0.3 的 91.1%   → 环境性质决定")
