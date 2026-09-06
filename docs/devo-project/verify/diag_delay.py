"""诊断：delay 到底动了多少？半衰期差异是否显著？"""
import numpy as np
from verify_myelin_stabilizer8 import run, autocorr_halflife, spectral_entropy, TAU_MAX

seeds = list(range(8))
g = 0.18

print("=" * 72)
print(f"诊断 gain={g}：delay 实际移动量 + 逐 seed 方差")
print("=" * 72)

for mode, ad in [("可塑", True), ("冻结", False)]:
    moves, finals, hls, ses = [], [], [], []
    for s in seeds:
        # 单独跑一次拿初始 delay
        rng0 = np.random.default_rng(s)
        # 复现 run 内部的 delay 初始化（需同顺序）
        r = run(seed=s, adapt=ad, g=g)
        finals.append(r["delay"])
        hls.append(autocorr_halflife(r["act"][3000:]))
        ses.append(spectral_entropy(r["act"][3000:]))
    d = np.concatenate(finals)
    # 初始 delay 是 rng.integers(1,21) -> 均匀分布
    print(f"\n{mode}:")
    print(f"  delay 均值 {d.mean():.2f}  标准差 {d.std():.2f}  "
          f"(均匀分布理论 std = {np.std(np.arange(1,TAU_MAX+1)):.2f})")
    print(f"  半衰期 逐seed: {[round(h,1) for h in hls]}")
    print(f"    均值 {np.mean(hls):.1f}  标准差 {np.std(hls):.1f}")
    print(f"  频谱熵 逐seed: {[round(x,3) for x in ses]}")
    print(f"    均值 {np.mean(ses):.3f}  标准差 {np.std(ses):.3f}")

# 直接测 delay 位移：跑两次，一次记录初始
print("\n" + "=" * 72)
print("delay 位移测量（对比初始随机值）")
print("=" * 72)
for s in range(4):
    rng = np.random.default_rng(s)
    # 重建与 run 内部相同顺序的随机数消耗
    _ = [rng.choice(32, size=6, replace=False) for _ in range(32)]
    init_delay = rng.integers(1, TAU_MAX + 1, size=32 * 6).astype(float)
    r = run(seed=s, adapt=True, g=g)
    disp = np.abs(r["delay"] - init_delay)
    print(f"  seed {s}: 平均位移 {disp.mean():.2f} 步  "
          f"最大 {disp.max():.0f}  "
          f"位移>2 的比例 {np.mean(disp > 2):.2%}")
