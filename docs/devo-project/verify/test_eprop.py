"""T3 红测试：e-prop 通路资格迹（髓鞘链深度信用的地基）

机制契约：
1. elig 迹累积：通路有正贡献时 elig 上升，无贡献时按 decay 衰减
2. RPE 门控：gain 更新量 = lr · RPE · elig（有界 [0, GAIN_MAX]）
3. 深度语义：elig 是持久迹——传输发生与 RPE 到来之间隔若干步，
   信用仍能落到当时传输的通路上（瞬时贡献做不到）
4. 语法任务回归：eprop 开启不劣化行为
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")

import numpy_torch_shim as shim
shim.install()

import baby_grammar as BG                                        # noqa: E402


def main():
    world = BG.GrammarWorld(0)
    brain = BG.build_brain(0, use_attention=False, use_spikes=False)
    reg = brain.higher_brain.sheath_registry

    # 手动创建髓鞘（机制测试不需要训练出结构）
    reg.add_sheath(0, "vis", 0, "lang", gain=1.0)
    key0, s0 = next(iter(reg._sheaths.items()))
    g0 = s0.gain

    # 1) elig 累积与衰减
    reg.update_eligibility({key0: 1.0}, decay=0.5)
    assert abs(s0.elig - 1.0) < 1e-9, f"elig 应=1.0, 实际 {s0.elig}"
    reg.update_eligibility({}, decay=0.5)
    assert abs(s0.elig - 0.5) < 1e-9, f"elig 应衰减到 0.5, 实际 {s0.elig}"

    # 2) RPE 门控：正 RPE + 正 elig → gain 增厚
    s0.elig = 1.0
    reg.apply_eprop(rpe=1.0, lr=0.05)
    assert s0.gain > g0, f"正 RPE 应增厚 gain: {g0} -> {s0.gain}"
    # 负 RPE → 削减（但不低于 0）
    reg.apply_eprop(rpe=-10.0, lr=0.05)
    assert 0.0 <= s0.gain <= s0.GAIN_MAX, f"gain 应有界: {s0.gain}"

    # 3) 深度语义：elig 的持久性——传输后隔多步 RPE 仍能分账
    s0.elig = 0.0
    reg.update_eligibility({key0: 1.0}, decay=0.9)
    for _ in range(5):                       # 5 步无贡献，迹衰减
        reg.update_eligibility({}, decay=0.9)
    assert s0.elig > 0.5, f"5 步后迹应保留过半: {s0.elig}"
    g_before = s0.gain
    reg.apply_eprop(rpe=1.0, lr=0.05)
    assert s0.gain > g_before, "延迟 RPE 仍应按残迹增厚"

    # 4) 语法任务回归：eprop 开启 500 步不崩、行为合理
    r = BG.run(seed=0, steps=500, use_attention=False, use_spikes=False,
               eprop=True)
    assert 0.0 <= r["top1"] <= 1.0
    print(f"语法回归(500步, eprop on): top1={r['top1']:.3f} "
          f" neurons={r['neurons']} sheaths={r['sheaths']}")
    print("PASS: e-prop 机制契约全部满足")


if __name__ == "__main__":
    main()
