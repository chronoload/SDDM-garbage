"""T4 多肢体婴儿：声带 + 手两个行动面原生并行，协同需求由世界给出

## 教义验证目标

1. **多肢体并行**：爬虫脑原生携带两个行动面（声带发声、手比符号），
   全部同时在线，系统必须自己发现哪个行动面改变世界的哪个侧面。
2. **协同需求**：一半物体"说出名字即出现"（单肢体足够），
   另一半"必须手上比着正确形状的同时说出名字"（双肢体联合）。
   跨信道协同是任务要求的，不是程序编排的。
3. **可测量**：按物体类型（simple/joint）分账满足率——
   预期 simple 先学会，joint 后学会（需要跨信道绑定）。

## 协议（每帧）

    vis : 当前需求物体的视觉特征
    aud : 当前需求物体的声音特征
    lang: 自己上一帧的发声（8 维 one-hot）
    hand: 自己上一帧的手型（8 维 one-hot，世界可见——手举着）
    行动: lang + hand 两信道同时输出
    裁定: simple 物体 → vocal==name 即满足；
         joint 物体 → vocal==name 且 hand==shape 才满足
"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

from mcp.developmental.signal import Signal                      # noqa: E402
from mcp.developmental.system import DevelopmentalSystem         # noqa: E402
from mcp.developmental.reptilian import ReptilianFunction        # noqa: E402

N = 8                      # 词表/手型各 8
D_VIS, D_AUD, D_LANG, D_HAND = N, N, N, N
_OFF, _o = {}, 0
for _c, _d in (("vis", D_VIS), ("aud", D_AUD),
               ("lang", D_LANG), ("hand", D_HAND)):
    _OFF[_c] = _o
    _o += _d
D_ALL = _o
PORT = {c: (_OFF[c], d) for c, d in
        (("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG), ("hand", D_HAND))}
LANG_SLICE = slice(_OFF["lang"], _OFF["lang"] + D_LANG)
HAND_SLICE = slice(_OFF["hand"], _OFF["hand"] + D_HAND)


class MultilimbWorld:
    """双行动面世界：simple 物体单声道可求，joint 物体需手口协同"""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # 8 个需求：前 4 个 simple（仅需发声），后 4 个 joint（手口协同）
        self.wants = [(i, i) for i in range(N)]   # (vocal, hand)
        self.simple = {i: i < N // 2 for i in range(N)}
        self.er = np.random.default_rng(seed + 11)
        self.vis_mat = self.er.normal(size=(N, D_VIS))
        self.aud_mat = self.er.normal(size=(N, D_AUD))
        self.rng = np.random.default_rng(seed + 99)
        self.want = 0
        self.last_vocal, self.last_hand = None, None
        self.stats = {"simple_hit": [0, 0], "joint_hit": [0, 0]}

    def observe(self):
        v = np.zeros(D_ALL)
        v[_OFF["vis"]:_OFF["vis"] + D_VIS] = (
            self.vis_mat[self.want] + 0.05 * self.er.normal(size=D_VIS))
        v[_OFF["aud"]:_OFF["aud"] + D_AUD] = (
            self.aud_mat[self.want] + 0.05 * self.er.normal(size=D_AUD))
        if self.last_vocal is not None:
            v[_OFF["lang"] + self.last_vocal] = 1.0
        if self.last_hand is not None:
            v[_OFF["hand"] + self.last_hand] = 1.0
        return v

    def judge(self, vocal, hand):
        """裁定上一帧行动（世界翻到新需求后调用）"""
        is_simple = self.simple[self.want]
        hit = (vocal == self.want) and (is_simple or hand == self.want)
        bucket = self.stats["simple_hit" if is_simple else "joint_hit"]
        bucket[1] += 1
        bucket[0] += int(hit)
        if hit:
            self.want = int(self.rng.integers(N))
        return 1.0 if hit else 0.0


class BabbleVocal(ReptilianFunction):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def get_input_spec(self):
        return {"lang": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        out = np.zeros(D_LANG)
        out[int(self.rng.integers(N))] = 1.0
        return {"lang": Signal(data=shim.tensor(out),
                               mime_type="generic/tensor",
                               metadata={"source": "babble-vocal"})}


class BabbleHand(ReptilianFunction):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed + 5)

    def get_input_spec(self):
        return {"hand": "generic/tensor"}

    def get_output_spec(self):
        return {"hand": "generic/tensor"}

    def execute(self, inputs):
        out = np.zeros(D_HAND)
        out[int(self.rng.integers(N))] = 1.0
        return {"hand": Signal(data=shim.tensor(out),
                               mime_type="generic/tensor",
                               metadata={"source": "babble-hand"})}


def build_brain(seed=0, autoreg_lr=0.05, **kw):
    brain = DevelopmentalSystem(port_layout=PORT, decay_rate=0.01, **kw)
    brain.phase = "exploratory"
    brain.autoreg_lr = autoreg_lr
    brain.autoreg_lr_tau = 5000.0
    brain.autoreg_w_norm = 3.0
    k = brain.reflex._kernel
    k.register("babble-vocal", BabbleVocal(seed))
    k.register("babble-hand", BabbleHand(seed))
    brain.reflex.add_preorchestrated_route(source="lang", target="babble-vocal",
                                           priority=1.0)
    brain.reflex.add_preorchestrated_route(source="hand", target="babble-hand",
                                           priority=1.0)
    n0 = brain.higher_brain.ecosystem.get_neuron(0)
    for c in PORT:
        n0.unfold(c, shim.zeros(D_ALL))
    brain.observe_output_channel = "lang"
    brain.world_display_channel = "vis"
    brain.higher_brain.symbolic_channels = {"lang"}
    return brain


def _argmax_channel(vec, off, size):
    seg = np.asarray(vec, dtype=float).ravel()[off:off + size]
    return int(np.argmax(seg)) if seg.max() > 0.05 else None


def run(seed=0, steps=4000, sleep_every=400, sleep_len=60, **kw):
    np.random.seed(seed)
    shim.manual_seed(seed)
    world = MultilimbWorld(seed)
    brain = build_brain(seed, **kw)
    hb = brain.higher_brain
    pending = None           # (vocal, hand)
    curves = []
    for t in range(steps):
        r = None
        if t > 0:
            world._advance_hand = None
            r = world.judge(pending[0], pending[1])
            brain.report_drive_satisfaction(r)
        obs = world.observe()
        inputs = {c: Signal(shim.tensor(obs[off:off + sz].copy()),
                            "generic/tensor", {"t": t})
                  for c, (off, sz) in PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        outs = res.get("outputs") or {}
        lang_sig = outs.get("lang")
        hand_sig = outs.get("hand")
        # ⚠ outputs 里是 Signal 对象，数据在 .data 里
        ld = lang_sig.data if lang_sig is not None else None
        hd = hand_sig.data if hand_sig is not None else None
        la = ld.detach().cpu().numpy() if ld is not None and hasattr(ld, "detach") else (ld.a if ld is not None and hasattr(ld, "a") else None)
        ha = hd.detach().cpu().numpy() if hd is not None and hasattr(hd, "detach") else (hd.a if hd is not None and hasattr(hd, "a") else None)
        vocal = _argmax_channel(la, 0, D_LANG) if la is not None else None
        hand = _argmax_channel(ha, 0, D_HAND) if ha is not None else None
        world.last_vocal, world.last_hand = vocal, hand
        pending = (vocal, hand)
        if sleep_every and (t + 1) % sleep_every == 0:
            for _ in range(sleep_len):
                brain.step_sleep()
        if (t + 1) % (steps // 8) == 0:
            sh, nh = world.stats["simple_hit"]
            jh, nj = world.stats["joint_hit"]
            curves.append((sh / max(1, nh), jh / max(1, nj)))
    sh, nh = world.stats["simple_hit"]
    jh, nj = world.stats["joint_hit"]
    return dict(
        simple=float(sh / max(1, nh)), joint=float(jh / max(1, nj)),
        n_simple=nh, n_joint=nj,
        curves=curves,
        neurons=len(hb.ecosystem.neurons),
        sheaths=hb.sheath_registry.capacity_report()["n_sheaths"],
        mean_lambda=float(np.mean([0.0])) if False else None,
        uniform=1.0 / N,
        joint_uniform=(1.0 / N) ** 2,
    )


def main():
    print("=" * 88)
    print("T4 多肢体婴儿：simple 物体(单声道) vs joint 物体(手口协同)")
    print("=" * 88)
    for seed in (0, 1):
        r = run(seed=seed)
        print(f"\nseed {seed}: simple={r['simple']:.3f} (n={r['n_simple']})  "
              f"joint={r['joint']:.3f} (n={r['n_joint']})  "
              f"uniform simple={r['uniform']:.3f} "
              f"joint={r['joint_uniform']:.4f}")
        print("  曲线(simple,joint): " +
              " ".join(f"({a:.2f},{b:.2f})" for a, b in r["curves"]))
        print(f"  结构: 神经元 {r['neurons']}  髓鞘 {r['sheaths']}")
    print("\n判据: simple 满足率先升(单声道可学); joint 随后上升"
          "(需要跨信道协同/绑定); joint >> (1/8)^2 说明系统学到了协同而非随机")


if __name__ == "__main__":
    main()
