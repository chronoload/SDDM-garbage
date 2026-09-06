"""判决性实验：区分两种机制

机制A（位置）：delay 收敛到最优相位 → 有序。
   预测：冻结在可塑终态 ≈ 可塑（仍有序）
机制B（游走）：delay 持续变化破坏共振 → 有序。
   预测：冻结在可塑终态 ≈ 冻结（回到无序）

组3设计：先跑可塑拿到终态 delay，再冻住重跑后半段。
"""
import numpy as np
from verify_myelin_stabilizer8 import autocorr_halflife, spectral_entropy, TAU_MAX

N, IN_DEG, TAU_MAX = 32, 6, 20

def run2(seed=0, steps=6000, warmup=1500, adapt=True, g=0.18,
         lr_delay=1.0, delay_block=20, noise=0.10,
         init_delay=None, freeze_at=None):
    """init_delay: 指定初始 delay；freeze_at: 从此步起强制冻结"""
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for i in range(N):
        for j in rng.choice(N, size=IN_DEG, replace=False):
            src.append(j); dst.append(i)
    src, dst = np.array(src), np.array(dst)

    gain = np.full(len(src), g)
    if init_delay is None:
        delay = rng.integers(1, TAU_MAX+1, size=len(src)).astype(float)
    else:
        delay = init_delay.copy()

    hist = np.zeros((TAU_MAX+1, N))
    x = rng.normal(0, 0.1, size=N)
    act = np.zeros(steps)

    for t in range(steps):
        d_int = np.clip(np.round(delay).astype(int), 1, TAU_MAX)
        agg = np.zeros(N)
        np.add.at(agg, dst, gain * hist[d_int, src])
        x = np.tanh(agg + rng.normal(0, noise, size=N))
        act[t] = x.mean()
        hist[t % (TAU_MAX+1)] = x

        do_adapt = adapt and t >= warmup and t % delay_block == 0 and t > TAU_MAX+1
        if freeze_at is not None and t >= freeze_at:
            do_adapt = False
        if do_adapt:
            dm = np.clip(d_int-1, 1, TAU_MAX); dp = np.clip(d_int+1, 1, TAU_MAX)
            cm1 = hist[(t-dm) % (TAU_MAX+1), src] * x[dst]
            c0  = hist[(t-d_int) % (TAU_MAX+1), src] * x[dst]
            cp1 = hist[(t-dp) % (TAU_MAX+1), src] * x[dst]
            bp = (cp1 > c0) & (cp1 >= cm1); bm = (cm1 > c0) & (cm1 > cp1)
            delay = np.clip(delay + lr_delay*(bp.astype(float)-bm.astype(float)),
                            1.0, float(TAU_MAX))
    return act, delay

seeds = list(range(8))
half = 3000
res = {"可塑": [], "冻结": [], "冻结在终态": []}

for s in seeds:
    # 组1 可塑
    a1, d1 = run2(seed=s, adapt=True)
    res["可塑"].append(spectral_entropy(a1[half:]))
    # 组2 冻结（初始随机 delay 不动）
    a2, _ = run2(seed=s, adapt=False)
    res["冻结"].append(spectral_entropy(a2[half:]))
    # 组3 冻结在可塑终态：用 d1 作为初始 delay，且全程冻结
    a3, _ = run2(seed=s, adapt=False, init_delay=d1)
    res["冻结在终态"].append(spectral_entropy(a3[half:]))

print("=" * 66)
print("判决：机制A(位置) vs 机制B(游走)   gain=0.18, 8 seeds")
print("=" * 66)
for k, v in res.items():
    v = np.array(v)
    print(f"  {k:12s} 频谱熵 {v.mean():.3f} ± {v.std():.3f}")
print("-" * 66)
a = np.array(res["可塑"]); f = np.array(res["冻结"]); t3 = np.array(res["冻结在终态"])
print(f"  可塑 vs 冻结         Δ = {f.mean()-a.mean():+.3f}")
print(f"  冻结在终态 vs 冻结   Δ = {f.mean()-t3.mean():+.3f}")
print(f"  冻结在终态 vs 可塑   Δ = {a.mean()-t3.mean():+.3f}")
print("-" * 66)
if abs(t3.mean()-a.mean()) < abs(t3.mean()-f.mean()):
    print("  → 更接近【可塑】= 机制A(位置) 占优：delay 收敛到好位置")
else:
    print("  → 更接近【冻结】= 机制B(游走) 占优：变化本身破坏共振")
print("=" * 66)
