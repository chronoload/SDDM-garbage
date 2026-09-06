"""量化审计：平凡基线能拿多少分？——"学习"必须先打败这些

对电影（弹球象限）与图片（patch 亮度等级）两个沉积物流，
计算不需要任何学习的预测器成绩：
  - persistence：预测"下一帧与当前相同"
  - 恒速外推（电影 oracle）：用真实物理参数外推（上界参照）
  - 多数类（图片）：永远预测最常见亮度等级
系统成绩只有超过这些线，"学到了"才成立。
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")
import numpy_torch_shim as shim
shim.install()
from deposit_media import MovieDeposit, ImageDeposit            # noqa: E402


def movie_baselines(steps=20000, seed=0):
    w = MovieDeposit(seed)
    quads, persistence, oracle = [], 0, 0
    for t in range(steps):
        q = w.cur_token()
        # 恒速外推 oracle：当前位置 + 真实速度，反弹裁剪
        nxt = w.pos + w.vel
        for i in range(2):
            if nxt[i] < 0 or nxt[i] > w.grid - 1:
                nxt[i] = np.clip(nxt[i], 0, w.grid - 1)
        quad_next = int((nxt[0] >= w.grid / 2)) + 2 * int((nxt[1] >= w.grid / 2))
        oracle += (quad_next == w.cur_token())  # 此刻的 cur 即下帧（advance 前）
        persistence += (q == w.cur_token())
        quads.append(q)
        w._advance()
    n = len(quads)
    # persistence 的正确口径：下一象限 == 当前象限
    p_acc = sum(1 for i in range(1, n) if quads[i] == quads[i - 1]) / (n - 1)
    return {"uniform": 0.25, "persistence": p_acc,
            "oracle_const_vel": oracle / n}


def image_baselines(steps=20000, seed=0):
    w = ImageDeposit(seed)
    levels = []
    for _ in range(steps):
        levels.append(w.cur_token())
        w._advance()          # ⚠ 必须推进扫描——否则全读同一 patch（审计脚本自bug）
    hist = np.bincount(levels, minlength=w.LEVELS) / len(levels)
    majority = float(hist.max())
    p_acc = np.mean([1.0 if levels[i] == levels[i - 1] else 0.0
                     for i in range(1, len(levels))])
    return {"uniform": 1 / w.LEVELS, "majority": majority,
            "persistence": float(p_acc),
            "level_entropy": float(-(hist[hist > 0] *
                                     np.log2(hist[hist > 0] + 1e-12)).sum())}


if __name__ == "__main__":
    print("电影（弹球象限，4 键）平凡基线：")
    for k, v in movie_baselines().items():
        print(f"  {k:>16}: {v:.3f}")
    print("图片（patch 亮度等级，16 键）平凡基线：")
    for k, v in image_baselines().items():
        print(f"  {k:>16}: {v:.3f}")
    print("\n判据: 系统 top1 必须超过 persistence/majority 才算学到东西；"
          "oracle 是本任务的可学上界参照")
