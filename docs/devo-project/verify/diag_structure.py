"""诊断：ρ 最优值 + 通路结构（噪声通路是否真被淘汰）"""
import numpy as np
from devo_control import (D, REAL_DIMS, NOISE_DIMS, run, agg, T, SEEDS,
                          observe, make_brain, A_MAT, B_MAT, PROC_NOISE, LAM_A)

print("=" * 92)
print("A. ρ 细粒度扫描（跨模态成边 ON）")
print("=" * 92)
print(f"{'ρ':>8}{'代价(高层)':>12}{'代价(反射)':>12}{'改进%':>9}"
      f"{'通路':>8}{'有效':>7}{'对角真':>8}{'对角噪':>8}{'对角噪有效':>11}")
print("-" * 92)
best = None
for rho in (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07):
    r = agg(rho=rho)
    # 单独统计噪声维对角通路的"有效"（增益>0.05）占比
    ne = r["n_eff"][0]
    print(f"{rho:>8.3f}{r['cost_high'][0]:>12.4f}{r['cost_reflex'][0]:>12.4f}"
          f"{r['improve'][0]*100:>8.1f}%{r['n_paths'][0]:>8.1f}{ne:>7.1f}"
          f"{r['diag_real'][0]:>8.1f}{r['diag_noise'][0]:>8.1f}"
          f"{'':>11}")
    if best is None or r["cost_high"][0] < best[1]:
        best = (rho, r["cost_high"][0], r)
print(f"\n最优 ρ = {best[0]}  代价 {best[1]:.4f}  改进 {best[2]['improve'][0]*100:.1f}%")

print()
print("=" * 92)
print(f"B. 跨模态成边的价值（在最优 ρ={best[0]}）")
print("=" * 92)
print(f"{'配置':<26}{'代价':>11}{'改进%':>9}{'通路':>8}{'有效':>7}{'跨模态':>8}")
print("-" * 92)
for name, kw in [
    ("跨模态 ON", dict(rho=best[0], allow_cross=True, wire=True)),
    ("跨模态 OFF（仅对角）", dict(rho=best[0], allow_cross=False, wire=True)),
    ("完全不成边", dict(rho=best[0], wire=False)),
]:
    r = agg(**kw)
    print(f"{name:<26}{r['cost_high'][0]:>11.4f}{r['improve'][0]*100:>8.1f}%"
          f"{r['n_paths'][0]:>8.1f}{r['n_eff'][0]:>7.1f}{r['cross'][0]:>8.1f}")

# ---- C. 通路结构可视化（单 seed）----
print()
print("=" * 92)
print("C. 学到的通路结构（seed=0，最优 ρ）")
print("=" * 92)
import devo_control as dc
rng = np.random.default_rng(0)
K_refl = make_brain(0)
G = np.eye(D) * 1.0
M = np.eye(D).astype(bool); co_fire = np.zeros((D, D), int)
rms = np.ones(D)
s_h = np.array([1.0, -0.5]); s_prev_h = s_h.copy()
ohat_next = None; o_prev = None
for t in range(T):
    o_h = observe(s_h, s_prev_h, rng)
    rms = np.sqrt(0.99 * rms ** 2 + 0.01 * o_h ** 2)
    on_h = o_h / (rms + 1e-6)
    a_h = -K_refl @ (ohat_next * (rms + 1e-6) if ohat_next is not None else o_h)
    s_next_h = A_MAT @ s_h + B_MAT @ a_h + PROC_NOISE * rng.normal(size=2)
    s_next_h = np.clip(s_next_h, -1e3, 1e3)
    if ohat_next is not None and o_prev is not None:
        r_vec = on_h - ohat_next
        denom = float(np.dot(o_prev, o_prev)) + 1e-8
        if t >= 200:
            G += M * (dc_run_lr := 0.3) * np.outer(o_prev, r_vec) / denom
            G -= M * (best[0] * G)
            dead = M & (G < 0.02); G[dead] = 0.0; M[dead] = False
        active = np.where(np.abs(o_prev) > 0.5)[0]
        for i in active:
            for j in active:
                co_fire[i, j] += 1
                if (not M[i, j]) and co_fire[i, j] >= 10:
                    M[i, j] = True; G[i, j] = 0.0
    o_prev = on_h; ohat_next = G.T @ on_h
    s_prev_h = s_h; s_h = s_next_h

print("列 = 被预测的维度 j；行 = 来源维度 i。数值 = 通路增益（— 表示不存在）")
print("真实维(应保留): " + str(REAL_DIMS) + "   噪声维(应淘汰): " + str(NOISE_DIMS))
print()
hdr = "      " + "".join(f"{j:>7}" for j in range(D))
print(hdr)
for i in range(D):
    tag = "R" if i in REAL_DIMS else "n"
    row = f"{tag}{i:>2} | "
    for j in range(D):
        row += f"{G[i,j]:>7.2f}" if M[i, j] else f"{'—':>7}"
    print(row)
rr = sum(1 for i in REAL_DIMS for j in REAL_DIMS if M[i, j] and abs(G[i, j]) > 0.05)
rn = sum(1 for i in range(D) for j in range(D)
         if M[i, j] and abs(G[i, j]) > 0.05 and (i in NOISE_DIMS or j in NOISE_DIMS))
print()
print(f"有效通路总数 {int((M & (np.abs(G) > 0.05)).sum())}  "
      f"其中真实维内部 {rr}  涉及噪声维 {rn}")
