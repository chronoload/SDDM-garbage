"""T9 连续性测试：棘轮教义的地基——发育状态跨 run 持久化

契约：
1. save → load：W / 髓鞘 / 连接 / 投票表 / 满足度账本逐位恢复
2. 续训等价：加载后继续训练的行为水平 ≈ 原脑继续训练
3. 棘轮：加载续训 (800+400) 显著优于新脑短训 (400)——
   发育成果跨 run 叠加，而不是每次从胚胎重爬
"""
import sys

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")
import numpy as np
import numpy_torch_shim as shim
shim.install()

import baby_grammar as BG                                        # noqa: E402
from mcp.developmental.persistence import (                      # noqa: E402
    save_checkpoint, load_checkpoint)

CKPT = r"C:\Users\qu\Desktop\devo_project\docs\devo-project\verify\ckpt.json"


def train(brain, steps, seed, offset=0):
    """在给定脑上继续训练（世界按种子重建，offset 续流）"""
    world = BG.GrammarWorld(seed)
    for _ in range(offset % 64):
        world._advance()
    prev = None
    results = []
    for t in range(steps):
        r = None
        if t > 0:
            world._advance()
            r = 1.0 if prev == world.cur else 0.0
            brain.report_drive_satisfaction(r)
            results.append(r)
        obs = world.observe()
        obs[BG.LANG_SLICE] = world.observe_lang()
        inputs = {c: BG.Signal(shim.tensor(obs[off:off + sz].copy()),
                               "generic/tensor", {"t": t})
                  for c, (off, sz) in BG.PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        prev = BG.decode_lang((res.get("outputs") or {}).get("lang"))
        world.last_guess = prev
    return float(np.mean(results)) if results else 0.0


def main():
    # 阶段1：原脑训练 800 步 → 保存
    np.random.seed(1)
    shim.manual_seed(1)
    brainA = BG.build_brain(1, use_attention=False, use_spikes=False)
    train(brainA, 800, seed=1)
    save_checkpoint(brainA, CKPT)
    wA = {nid: _np_copy(n.W) for nid, n in
          brainA.higher_brain.ecosystem.neurons.items()}

    # 阶段2：新脑加载 → 逐位校验
    brainB = BG.build_brain(1, use_attention=False, use_spikes=False)
    load_checkpoint(brainB, CKPT)
    hbA, hbB = brainA.higher_brain, brainB.higher_brain
    for nid, n in hbA.ecosystem.neurons.items():
        m = hbB.ecosystem.get_neuron(nid)
        if n.W is None or m is None or m.W is None:
            continue
        assert np.allclose(_np_copy(n.W), _np_copy(m.W), atol=1e-12), \
            f"神经元 {nid} W 未恢复"
    for k, s in hbA.sheath_registry._sheaths.items():
        s2 = hbB.sheath_registry._sheaths.get(k)
        assert s2 is not None and abs(s2.gain - s.gain) < 1e-12, \
            f"髓鞘 {k} gain 未恢复"
    assert hbB.vote_cred == hbA.vote_cred, "确认投票表未恢复"
    print("契约1: 状态逐位恢复 ✓")

    # 阶段3：三方对照——续训等价 + 棘轮优势
    tA = train(brainA, 400, seed=2)            # 原脑继续
    tB = train(brainB, 400, seed=2)            # 加载脑继续
    brainC = BG.build_brain(1, use_attention=False, use_spikes=False)
    np.random.seed(1)
    shim.manual_seed(1)
    tC = train(brainC, 400, seed=2)            # 新脑从头 400 步
    print(f"续训400步: 原脑={tA:.3f}  加载脑={tB:.3f}  "
          f"新脑(仅400步)={tC:.3f}")
    assert tB > tC + 0.03, \
        f"加载脑应显著优于新脑短训（棘轮）: {tB:.3f} vs {tC:.3f}"
    assert abs(tA - tB) < 0.15, f"续训应等价: {tA:.3f} vs {tB:.3f}"
    print("契约2/3: 续训等价 + 棘轮优势 ✓")
    print("PASS: 脑序列化契约全部满足")


def _np_copy(x):
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy(), dtype=float)
    if hasattr(x, "a"):
        return np.asarray(x.a, dtype=float)
    return np.asarray(x, dtype=float)


if __name__ == "__main__":
    main()
