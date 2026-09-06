"""消融实验：自注意力 + 脉冲（解耦层）对语法习得的贡献

四组配置（baby_grammar 任务，裸编码 one-hot 三模态）：
  baseline : attention=off, spikes=off
  +attn    : attention=on,  spikes=off
  +spike   : attention=off, spikes=on
  +both    : attention=on,  spikes=on

判据：各配置的 top1(末1/4) 对照 uniform / echo / bigram；
解耦层的诊断（spike_rate / attention 键数与增益范围）验证机制
真的在工作而不是空转。

运行：python verify_attention_spikes.py
"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

import baby_grammar as BG                                     # noqa: E402

CONFIGS = [
    ("baseline", dict(use_attention=False, use_spikes=False)),
    ("+attn",    dict(use_attention=True,  use_spikes=False)),
    ("+spike",   dict(use_attention=False, use_spikes=True,
                      spike_threshold=0.1)),
    ("+both",    dict(use_attention=True,  use_spikes=True,
                      spike_threshold=0.1)),
]
SEEDS = (0, 1, 2)
STEPS = 6000


def main():
    print("=" * 96)
    print(f"消融：自注意力 × 脉冲（grammar 任务，{BG.N_TOKEN} tokens，"
          f"steps={STEPS}，seeds={SEEDS}，autoreg_lr=0.05）")
    print("=" * 96)
    results = {}
    for name, kw in CONFIGS:
        rows = []
        for s in SEEDS:
            r = BG.run(seed=s, steps=STEPS, autoreg_lr=0.05, **kw)
            rows.append(r)
        results[name] = rows
        m = lambda k: np.mean([r[k] for r in rows])
        sd = lambda k: np.std([r[k] for r in rows])
        print(f"\n{name:>9}: top1={m('top1'):.3f}±{sd('top1'):.3f}  "
              f"(all {m('top1_all'):.3f})  uniform {m('uniform'):.3f}  "
              f"echo {m('echo_base'):.3f}  bigram {m('bigram_base'):.3f}")
        print(f"{'':>9}  神经元 {m('neurons'):.1f}  髓鞘 {m('sheaths'):.1f}  "
              f"周转 {m('turnover'):.0f}  λ均值 {m('mean_lambda'):.3f}  "
              f"高层占比 {m('pct_high'):.2f}")
        # 机制在场的证据（用第一个 seed 的诊断）
        r0 = rows[0]
        if "attn" in name or "both" in name:
            print(f"{'':>9}  attention: {r0.get('attention')}")
        if "spike" in name or "both" in name:
            print(f"{'':>9}  spikes: {r0.get('spikes')}")

    print()
    print("=" * 96)
    print("汇总（top1 末1/4，3 seeds 均值±sd）")
    print("-" * 96)
    for name, _ in CONFIGS:
        rows = results[name]
        m = np.mean([r["top1"] for r in rows])
        sd = np.std([r["top1"] for r in rows])
        print(f"  {name:>9}: {m:.3f} ± {sd:.3f}")


if __name__ == "__main__":
    main()
