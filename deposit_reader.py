"""沉积物阅读器：把静态文本转成动态反馈回路（超越物理时间的压缩学习）

## 思想

文本沉积物是**冻结的交互痕迹**——作者年复一年的智能活动被压缩成
字节流。把它转成动态环境（每 tick 揭示下一个字节，系统的猜测由
沉积物自身裁定），系统就在"重放"沉积物：用分钟重放作者用掉的
岁月。这不是"硬塞静态数据"，是架构已有的重放 substrate（睡眠
生成重放 + 轨迹日志）向沉积层的延伸——阅读即重放引导的内源化。

## 裸编码的极致：字节级打字机

词表 = 256 字节。没有任何分词器、没有嵌入、没有预设距离结构——
打字机有 256 个键，这是"给爬虫脑捆打字机"的零预编码版本。

## 协议（与 baby_grammar 完全同构）

    vis : 沉积物当前字节（one-hot / 256）
    lang: 系统自己上一帧的发声
    输出 : 对下一字节的猜测（行动）
    下一帧世界揭示 → 命中给满足度 1.0

判据对齐（baby_grammar 的 off-by-one 教训）：先 advance 后裁定。

运行：python deposit_reader.py [corpus_path]
"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

from mcp.developmental.signal import Signal                      # noqa: E402
from mcp.developmental.system import DevelopmentalSystem         # noqa: E402
from mcp.developmental.reptilian import ReptilianFunction        # noqa: E402

V = 256
D_VIS, D_LANG = V, V
_OFF, _o = {}, 0
for _c, _d in (("vis", D_VIS), ("lang", D_LANG)):
    _OFF[_c] = _o
    _o += _d
D_ALL = _o
PORT = {c: (_OFF[c], d) for c, d in
        (("vis", D_VIS), ("lang", D_LANG))}
LANG_SLICE = slice(_OFF["lang"], _OFF["lang"] + D_LANG)
VIS_SLICE = slice(_OFF["vis"], _OFF["vis"] + D_VIS)


class DepositWorld:
    """沉积物世界：字节流自主演进，猜测由下一字节裁定"""

    def __init__(self, deposit: bytes, seed=0):
        self.deposit = deposit
        self.pos = 0
        self.last_guess = None

    def _advance(self):
        self.pos = (self.pos + 1) % len(self.deposit)

    def observe(self):
        v = np.zeros(D_ALL)
        v[_OFF["vis"] + self.deposit[self.pos]] = 1.0
        if self.last_guess is not None:
            v[_OFF["lang"] + self.last_guess] = 1.0
        return v

    def cur_byte(self):
        return self.deposit[self.pos]


class BabbleByte(ReptilianFunction):
    """哭闹反射：随机按一个键（先天、不含因果知识）"""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def get_input_spec(self):
        return {"lang": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        out = np.zeros(D_LANG)
        out[int(self.rng.integers(V))] = 1.0
        return {"lang": Signal(data=shim.tensor(out),
                               mime_type="generic/tensor",
                               metadata={"source": "babble"})}


class EchoByte(ReptilianFunction):
    """模仿反射：复读上一帧发声（注册但不主导输出面——回声链毒化教训）"""

    def get_input_spec(self):
        return {"lang": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        d = inputs["lang"].data
        a = np.asarray(d.detach().cpu().numpy() if hasattr(d, "detach")
                       else d, dtype=float).ravel()
        out = np.zeros(D_LANG)
        if a.size == D_LANG and a.max() > 0.5:
            out[int(np.argmax(a))] = 1.0
        return {"lang": Signal(data=shim.tensor(out),
                               mime_type="generic/tensor",
                               metadata={"source": "echo"})}


def build_brain(seed=0, decay_rate=0.01, autoreg_lr=0.05,
                autoreg_w_norm=3.0, **kw):
    brain = DevelopmentalSystem(port_layout=PORT, decay_rate=decay_rate, **kw)
    brain.phase = "exploratory"
    brain.autoreg_lr = autoreg_lr
    brain.autoreg_lr_tau = 5000.0
    if autoreg_w_norm is not None:
        brain.autoreg_w_norm = autoreg_w_norm
    k = brain.reflex._kernel
    k.register("babble", BabbleByte(seed))
    k.register("echo", EchoByte())
    brain.reflex.add_preorchestrated_route(source="lang", target="echo",
                                           priority=1.0)
    brain.reflex.add_preorchestrated_route(source="lang", target="babble",
                                           priority=0.5)
    n0 = brain.higher_brain.ecosystem.get_neuron(0)
    n0.unfold("vis", shim.zeros(D_ALL))
    n0.unfold("lang", shim.zeros(D_ALL))
    brain.observe_output_channel = "lang"
    brain.world_display_channel = "vis"
    brain.higher_brain.symbolic_channels = {"lang"}
    return brain


def decode_lang(vec):
    if vec is None:
        return None
    d = vec.data if isinstance(vec, Signal) else vec
    if hasattr(d, "detach"):
        d = d.detach().cpu().numpy()
    elif hasattr(d, "a"):
        d = d.a
    a = np.asarray(d, dtype=float).ravel()
    if a.size == D_LANG:
        seg = a
    elif a.size >= _OFF["lang"] + D_LANG:
        seg = a[LANG_SLICE]
    else:
        return None
    return int(np.argmax(seg)) if seg.max() > 0.05 else None


def run(deposit: bytes, seed=0, steps=20000, sleep_every=500, sleep_len=80,
        **kw):
    np.random.seed(seed)
    shim.manual_seed(seed)
    world = DepositWorld(deposit, seed)
    brain = build_brain(seed, **kw)
    hb = brain.higher_brain

    results, lambdas = [], []
    prev_guess = None
    for t in range(steps):
        r = None
        if t > 0:
            world._advance()
            r = 1.0 if prev_guess == world.cur_byte() else 0.0
            brain.report_drive_satisfaction(r)
            results.append(r)

        obs = world.observe()
        inputs = {c: Signal(shim.tensor(obs[off:off + sz].copy()),
                            "generic/tensor", {"t": t})
                  for c, (off, sz) in PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        out = (res.get("outputs") or {}).get("lang")
        prev_guess = decode_lang(out)
        world.last_guess = prev_guess
        lambdas.append(res.get("lambda", 0.0))

        if sleep_every and (t + 1) % sleep_every == 0:
            for _ in range(sleep_len):
                brain.step_sleep()

    arr = np.array(results, dtype=float)
    n = len(arr)
    stream = world.deposit
    bigram = _bigram_top1(stream)
    return dict(
        top1=float(arr[int(0.75 * n):].mean()),
        top1_all=float(arr.mean()),
        curve=[float(arr[i:i + max(1, n // 20):].mean())
               for i in range(0, n, max(1, n // 20))],
        uniform=1.0 / V,
        bigram_base=bigram,
        mean_lambda=float(np.mean(lambdas)),
        neurons=len(hb.ecosystem.neurons),
        sheaths=hb.sheath_registry.capacity_report()["n_sheaths"],
        seizure=brain.seizure_events,
    )


def _bigram_top1(stream):
    counts = {}
    for a, b in zip(stream[:-1], stream[1:]):
        counts.setdefault(a, {})
        counts[a][b] = counts[a].get(b, 0) + 1
    hit, tot = 0, 0
    for i in range(1, len(stream)):
        a = stream[i - 1]
        if a in counts:
            pred = max(counts[a], key=counts[a].get)
            hit += (pred == stream[i])
        tot += 1
    return hit / max(1, tot)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "baby_grammar.py"
    with open(path, "rb") as f:
        deposit = f.read()
    steps = min(len(deposit) * 3, 30000)
    print("=" * 88)
    print(f"沉积物阅读：{path}（{len(deposit)} 字节，词表 256，steps={steps}）")
    print("=" * 88)
    r = run(deposit, seed=0, steps=steps)
    print(f"系统 top1 = {r['top1']:.3f}  (all {r['top1_all']:.3f})   "
          f"uniform {r['uniform']:.3f}   bigram {r['bigram_base']:.3f}")
    print("学习曲线: " + " ".join(f"{v:.2f}" for v in r["curve"]))
    print(f"结构: 神经元 {r['neurons']}  髓鞘 {r['sheaths']}  "
          f"λ均值 {r['mean_lambda']:.2f}  癫痫刹车 {r['seizure']} 次")


if __name__ == "__main__":
    main()
