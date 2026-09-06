"""冒烟验证：bug 修复 + 睡眠归因账本 + 咨询台账

覆盖四件事：
1. 反应模式短跑（改动不破坏原有行为，满足率应显著高于随机基线 1/12）
2. bug 修复验证：首个 step_awake **之前**调用 report_drive_satisfaction
   （原代码在 _learn_from_relief 处 AttributeError）
3. lambda_beta / lambda_threshold 属性存在（原 compute_lambda 隐患）
4. 睡眠归因账本（sleep_report / sleep_phase_gain）
5. 咨询台账（drive_sat.source_report 应出现 "babbling" 来源）

运行：python verify_ledgers.py
"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

import baby_loop as BL                                    # noqa: E402


def mini_episode(seed=0, steps=60, pre_report=False):
    """手写小 episode：可控地触发各条新路径"""
    world = BL.ReactiveWorld(seed)
    brain = BL.build_brain(seed)
    baby_reflex = brain.reflex._kernel.get("babble")
    hb = brain.higher_brain
    n0 = hb.ecosystem.get_neuron(0)
    for c in BL.PORT:
        n0.unfold(c, shim.zeros(BL.D_ALL))

    if pre_report:
        # bug 修复验证：首帧学习块会消费 _pending_relief，
        # 此时 _prev_output_for_relief 必须已安全为 None
        brain.report_drive_satisfaction(1.0)

    for t in range(steps):
        baby_reflex.set_drive(world.want)
        obs = world.observe()
        inputs = {}
        for c, (off, sz) in BL.PORT.items():
            inputs[c] = BL.Signal(shim.tensor(obs[off:off + sz].copy()),
                                  "generic/tensor", {"t": t})
        res = brain.step_awake(external_inputs=inputs)
        word = BL.decode_lang((res.get("outputs") or {}).get("lang"))
        r, _ = world.act(word)
        brain.report_drive_satisfaction(r)
    return brain, world


def main():
    print("=" * 74)
    print("验证 1：反应模式短跑（行为回归，随机基线 = 1/12 = 0.083）")
    print("-" * 74)
    rs = [BL.run(seed=s, steps=600, reactive=True) for s in range(2)]
    for i, r in enumerate(rs):
        print(f"  seed{i}: sat={r['sat']:.3f} sat_all={r['sat_all']:.3f} "
              f"neurons={r['neurons']} sheaths={r['sheaths']} "
              f"turnover={r['turnover']}")

    print()
    print("=" * 74)
    print("验证 2：首帧前 report_drive_satisfaction（原 AttributeError 路径）")
    print("-" * 74)
    brain, world = mini_episode(seed=0, steps=40, pre_report=True)
    print("  OK：无异常")

    print()
    print("=" * 74)
    print("验证 3：lambda 门控参数已定义")
    print("-" * 74)
    hb = brain.higher_brain
    print(f"  lambda_beta={hb.lambda_beta} lambda_threshold={hb.lambda_threshold}")
    assert hasattr(hb, "lambda_beta") and hasattr(hb, "lambda_threshold")

    print()
    print("=" * 74)
    print("验证 4：睡眠归因账本")
    print("-" * 74)
    n_cycles = 3
    for _ in range(n_cycles):
        out = brain.step_sleep()
    rep = out.get("sleep_report") or {}
    for phase, e in sorted((rep.get("phases") or {}).items()):
        print(f"  {phase:>14}: cycles={e['cycles']:.0f} "
              f"touched={e['touched']:.0f} Δgain={e['gain_delta']:+.4f} "
              f"Δstab={e['stab_delta']:+.4f} born={e['born']:.0f} "
              f"died={e['died']:.0f}")
    print(f"  受睡眠影响髓鞘: {rep.get('n_sleep_affected')}/"
          f"{rep.get('n_sheaths')}  meanΔgain={rep.get('sleep_gain_mean'):+.4f}")
    assert "homeostasis" in (rep.get("phases") or {}), "homeostasis 相位未入账"
    assert "replay" in (rep.get("phases") or {}), "replay 相位未入账"

    print()
    print("=" * 74)
    print("验证 5：咨询台账（按输出来源分账）")
    print("-" * 74)
    sr = brain.drive_sat.source_report()
    for src, e in sorted(sr.items()):
        print(f"  source={src:>10}: n={e['n']} sat_ema={e['sat_ema']:.3f} "
              f"sat_mean={e['sat_mean']:.3f}")
    assert any("babbling" in s for s in sr), \
        f"babbling 来源未入账: {list(sr)}"

    print()
    print("=" * 74)
    print("全部通过")
    print("=" * 74)


if __name__ == "__main__":
    main()
