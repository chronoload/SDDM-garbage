"""验证修正方案：新生儿豁免期（纯 numpy）

前两版实验暴露的根因：
  A. age 杀不死有随机贡献的噪声通路，只能清理完全静默的通路
     → age 的定位必须收窄
  D. initial_gain=0 与 GAIN_DEATH=0.1 冲突：新边必须在连续数步内
     被增厚才能存活；成边稀疏化后新边更少有机会 → 加剧空转

修正假设：给新边一个**豁免期**，期间不参与死亡判据（gain 仍为 0，
不污染输出），只累积增厚。到期后按正常判据决定存亡。
"""
import numpy as np

GAIN_DEATH = 0.1
GAIN_MAX = 2.0


def run(seed=0, steps=1500, n_cand=40, n_real=8, thicken=0.02,
        block=1, grace=0, p_real=0.6, p_noise=0.05):
    """返回终态结构统计

    Args:
        block: 成边 block 长度（时间尺度分离）
        grace: 新生儿豁免期步数（本实验的核心变量）
        p_real / p_noise: 真通路 / 噪声通路的共发射概率
    """
    rng = np.random.default_rng(seed)
    truth = np.zeros(n_cand)
    truth[:n_real] = 1.0

    born = died = 0
    gains, since, age = {}, {}, {}

    for t in range(steps):
        # 1. 共发射 → 增厚
        for i in list(gains.keys()):
            p = p_real if truth[i] == 1.0 else p_noise
            if rng.random() < p:
                gains[i] = min(GAIN_MAX, gains[i] + thicken)
                since[i] = 0
            else:
                since[i] = since.get(i, 0) + 1
            age[i] = age.get(i, 0) + 1

        # 2. 选择动力学：相对衰减 + 死亡判据（豁免期内跳过）
        for i in list(gains.keys()):
            gains[i] = max(0.0, gains[i] - 0.01 * gains[i])
            if age.get(i, 0) < grace:
                continue                      # 豁免期内不死
            if gains[i] < GAIN_DEATH or since.get(i, 0) >= 200:
                del gains[i]
                since.pop(i, None)
                age.pop(i, None)
                died += 1

        # 3. 成边
        if block > 0 and t % block == 0:
            for i in range(n_cand):
                if i in gains:
                    continue
                if rng.random() < 0.3:
                    gains[i] = 0.0            # 沉默突触，不污染输出
                    since[i] = 0
                    age[i] = 0
                    born += 1

    n = len(gains)
    real_kept = sum(1 for i in gains if truth[i] == 1.0)
    return {
        "edges": n,
        "real_kept": real_kept,
        "noise_kept": n - real_kept,
        "born": born,
        "died": died,
        "turnover": (born + died) / max(1, n),
        "precision": real_kept / max(1, n),
        "recall": real_kept / n_real,
    }


def sweep(grace_list, block_list, seeds=5):
    agg = {}
    for g in grace_list:
        for b in block_list:
            vals = [run(seed=s, grace=g, block=b) for s in range(seeds)]
            agg[(g, b)] = {
                "edges": np.mean([v["edges"] for v in vals]),
                "real": np.mean([v["real_kept"] for v in vals]),
                "noise": np.mean([v["noise_kept"] for v in vals]),
                "born": np.mean([v["born"] for v in vals]),
                "turnover": np.mean([v["turnover"] for v in vals]),
                "prec": np.mean([v["precision"] for v in vals]),
                "recall": np.mean([v["recall"] for v in vals]),
            }
    return agg


if __name__ == "__main__":
    print("=" * 78)
    print("新生儿豁免期 × 成边 block：能否让真通路存活且压住空转")
    print("=" * 78)
    print(f"{'豁免':>4} {'block':>6} {'边数':>6} {'真通路':>7} {'噪声':>6} "
          f"{'新建':>7} {'周转率':>8} {'精确率':>7} {'召回':>6}")
    print("-" * 78)
    agg = sweep(grace_list=[0, 10, 30, 60], block_list=[1, 10, 50])
    for (g, b), v in agg.items():
        print(f"{g:>4} {b:>6} {v['edges']:>6.1f} {v['real']:>7.1f} "
              f"{v['noise']:>6.1f} {v['born']:>7.0f} {v['turnover']:>8.1f} "
              f"{v['prec']:>7.3f} {v['recall']:>6.2f}")

    print()
    print("最优判据：精确率 × 召回（F1），同时周转率尽可能低")
    print("-" * 78)
    best = sorted(
        agg.items(),
        key=lambda kv: -(2 * kv[1]["prec"] * kv[1]["recall"]
                         / max(1e-9, kv[1]["prec"] + kv[1]["recall"]))
    )
    for (g, b), v in best[:4]:
        f1 = 2 * v["prec"] * v["recall"] / max(1e-9, v["prec"] + v["recall"])
        print(f"  豁免={g:>3} block={b:>3}  F1={f1:.3f}  精确率={v['prec']:.3f}  "
              f"召回={v['recall']:.2f}  周转率={v['turnover']:.1f}")
