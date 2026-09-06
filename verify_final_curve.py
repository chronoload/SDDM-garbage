"""最终冲刺：10000 步长训 + W 范数预算对照（冲 bigram 方向）"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

import baby_grammar as BG                                     # noqa: E402

CONFIGS = [
    ("baseline(w=1.1)", dict(use_attention=False, use_spikes=False)),
    ("w_norm=3.0",      dict(use_attention=False, use_spikes=False,
                             autoreg_w_norm=3.0)),
]
SEEDS = (0, 1, 2)
STEPS = 10000


def main():
    print("=" * 96)
    print(f"长训：steps={STEPS}，seeds={SEEDS}，autoreg_lr=0.05")
    print("=" * 96)
    for name, kw in CONFIGS:
        rows = []
        for s in SEEDS:
            r = BG.run(seed=s, steps=STEPS, autoreg_lr=0.05, **kw)
            rows.append(r)
            print(f"\n{name} seed{s}: top1={r['top1']:.3f} "
                  f"(all {r['top1_all']:.3f}) bigram={r['bigram_base']:.3f} "
                  f"曲线={' '.join(f'{v:.2f}' for v in r['curve'])} "
                  f"神经元={r['neurons']} 髓鞘={r['sheaths']} "
                  f"λ={r['mean_lambda']:.2f} 刹车={r.get('seizure_brakes', 0)}")
        m = np.mean([r['top1'] for r in rows])
        sd = np.std([r['top1'] for r in rows])
        print(f"  ==> {name}: top1 = {m:.3f} ± {sd:.3f}")


if __name__ == "__main__":
    main()
