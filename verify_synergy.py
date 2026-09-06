"""验证两个协同机制（纯 numpy）——第二版

第一版实验否定了我的初始假设，暴露出两个真问题：

问题 A：age 归零条件若为"单步贡献为正"，噪声通路平均每 3 步重置一次
        age，永远累积不到死亡阈值。必须用**显著性**判据区分真假成功。
问题 D：真实空转的根源是新边 initial_gain=0，而存活判据是 gain >= 0.1
        → 新边**当步就死**，下一步又被重建。第一版模拟没复现这一点。

本版针对两者重做。
"""
import numpy as np

GAIN_DEATH = 0.1
GAIN_MAX = 2.0


# ---------------------------------------------------------------------------
# 协同 A：age 的归零判据
# ---------------------------------------------------------------------------

def test_age_criterion(seed=0, steps=3000, rho=0.01, sigma=0.5):
    """对比 age 归零判据：单步贡献为正 vs 贡献显著超过噪声水平

    真通路：贡献稳定为正（0.05）
    噪声通路：贡献零均值随机 N(0, sigma*0.05)

    显著性判据的设计：单步贡献必须超过该通路自身贡献波动的 k 倍。
    真通路贡献 0.05 >> 0.025，稳定通过；噪声通路显著为正的尾部事件
    概率大幅下降 → age 能累积到死亡阈值。
    """
    rng = np.random.default_rng(seed)
    out = {}

    for mode in ["ema_only", "age_naive", "age_significant"]:
        g_real, g_noise = 0.5, 0.5
        since = 0
        usage = 0.0
        noise_died_at = None
        noise_alive = 0
        # 在线估计噪声通路贡献的波动（用于显著性判据）
        c_sq = 0.0

        for t in range(steps):
            c_real = 0.05
            c_noise = float(rng.normal(0, sigma * 0.05))

            g_real = max(0.0, min(GAIN_MAX, g_real + c_real - rho * g_real))

            g_noise = max(0.0, min(GAIN_MAX, g_noise + c_noise - rho * g_noise))
            usage = 0.9 * usage + 0.1 * c_noise
            c_sq = 0.9 * c_sq + 0.1 * c_noise ** 2
            c_std = max(1e-6, np.sqrt(c_sq))

            if mode == "ema_only":
                if g_noise < GAIN_DEATH:
                    if noise_died_at is None:
                        noise_died_at = t
                    continue
                noise_alive += 1
                continue

            if mode == "age_naive":
                co_fired = c_noise > 0.01
            else:  # age_significant
                # 显著性：超过自身波动的 1 倍标准差，且 EMA 方向为正
                co_fired = (c_noise > c_std) and (usage > 0)

            if co_fired:
                since = 0
            else:
                since += 1

            if since >= 200:
                if noise_died_at is None:
                    noise_died_at = t
                continue
            noise_alive += 1

        out[mode] = {
            "died_at": noise_died_at,
            "noise_alive": noise_alive,
            "real_gain": g_real,
            # 真通路在同样的 age 判据下是否能存活（不能误杀真通路）
            "real_survived": g_real >= GAIN_DEATH,
        }
    return out


# ---------------------------------------------------------------------------
# 协同 D：时间尺度分离 —— 复现 gain=0 新边当步死亡的空转
# ---------------------------------------------------------------------------



def test_wiring_block2(seed=0, steps=1500, n_cand=40, n_real=8, thicken=0.02):
    rng = np.random.default_rng(seed)
    truth = np.zeros(n_cand)
    truth[:n_real] = 1.0
    out = {}

    for block in [1, 10, 50]:
        r2 = np.random.default_rng(seed)
        born = died = 0
        gains, since = {}, {}
        for t in range(steps):
            for i in list(gains.keys()):
                p = 0.6 if truth[i] == 1.0 else 0.05
                if r2.random() < p:
                    gains[i] = min(GAIN_MAX, gains[i] + thicken)
                    since[i] = 0
                else:
                    since[i] = since.get(i, 0) + 1
            for i in list(gains.keys()):
                gains[i] = max(0.0, gains[i] - 0.01 * gains[i])
                if gains[i] < GAIN_DEATH or since.get(i, 0) >= 200:
                    del gains[i]
                    since.pop(i, None)
                    died += 1
            if block > 0 and t % block == 0:
                for i in range(n_cand):
                    if i in gains:
                        continue
                    if r2.random() < 0.3:
                        gains[i] = 0.0   # 沉默突触
                        since[i] = 0
                        born += 1
        n = len(gains)
        real_kept = sum(1 for i in gains if truth[i] == 1.0)
        out[f"block={block}"] = {
            "edges": n,
            "real_kept": real_kept,
            "noise_kept": n - real_kept,
            "born": born,
            "died": died,
            "turnover": (born + died) / max(1, n),
            "precision": real_kept / max(1, n),
        }
    return out


if __name__ == "__main__":
    print("=" * 74)
    print("协同 A：age 归零判据 —— 能否杀死噪声通路且不误杀真通路")
    print("=" * 74)
    rows = [test_age_criterion(seed=s) for s in range(5)]
    for mode in ["ema_only", "age_naive", "age_significant"]:
        killed = sum(1 for r in rows if r[mode]["died_at"] is not None)
        alive = np.mean([r[mode]["noise_alive"] for r in rows])
        real = np.mean([r[mode]["real_gain"] for r in rows])
        safe = all(r[mode]["real_survived"] for r in rows)
        print(f"{mode:18s} 杀死噪声 {killed}/5  噪声存活 {alive:6.0f} 步  "
              f"真通路 gain {real:.3f}  真通路存活 {'是' if safe else '否'}")

    print()
    print("=" * 74)
    print("协同 D：成边频率 vs 周转率（复现 gain=0 新边当步死亡）")
    print("=" * 74)
    agg = {}
    for s in range(5):
        for k, v in test_wiring_block2(seed=s).items():
            agg.setdefault(k, []).append(v)
    for k, vals in agg.items():
        print(f"{k:11s} 边数 {np.mean([v['edges'] for v in vals]):5.1f}  "
              f"真通路 {np.mean([v['real_kept'] for v in vals]):4.1f}/8  "
              f"噪声 {np.mean([v['noise_kept'] for v in vals]):5.1f}  "
              f"新建 {np.mean([v['born'] for v in vals]):6.0f}  "
              f"周转率 {np.mean([v['turnover'] for v in vals]):6.1f}  "
              f"精确率 {np.mean([v['precision'] for v in vals]):.3f}")
