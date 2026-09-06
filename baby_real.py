"""婴幼儿多模态语言习得 —— 在**真实代码**上运行

## 与 baby_lang.py 的根本差别

此前所有婴幼儿实验都在 numpy 原型上跑，验证的不是这套代码。
本脚本用 ``mcp.developmental`` 的真实 ``DevelopmentalSystem``，
通过 numpy_torch_shim 运行。

## 任务设计

**世界**：12 个词的微型词汇表，句子由模板生成，逐词呈现。
每个词有三个通道的特征：

=========  ============================================
通道        内容
=========  ============================================
vis         视觉：具体名词（人/物）有强视觉，动作/修饰弱
aud         听觉：拼音的声学特征（声母 + 韵母）
lang        语言：**有语义结构**的嵌入（同类词相近）
=========  ============================================

**关键**：lang 用手工语义嵌入而非哈希嵌入。哈希嵌入已实测无语义结构
（词对余弦标准差 0.0368 ≈ 随机向量理论值 0.0442），系统学不到东西。

## 婴幼儿教育分期

- **Phase 0 感知校准**：lang 输入 = 真实词。系统只做自回归预测，
  建立 vis↔aud↔lang 的跨模态绑定。**不要求输出**（对应"看到爸爸妈妈
  什么都不会干，但知觉在校准"）。
- **Phase 1 产出（打字机）**：lang 输入 = 系统自己上一步的输出。
  系统必须真的会生成，不能依赖教师强制。

## 自回归裁定

系统输出三个通道（vis/aud/lang），settle 用下一帧全局输入裁定上一步
输出。反射让 vis/aud 直接 echo 输入，所以自回归误差**主要落在 lang 段**
—— 即"预测下一个词"，这正是语言建模的任务。
"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

sys.path.insert(0, "/data/workspace")

from mcp.developmental.signal import Signal                      # noqa: E402
from mcp.developmental.system import DevelopmentalSystem         # noqa: E402
from mcp.developmental.reptilian import ReptilianFunction        # noqa: E402

# ---------------------------------------------------------------------------
# 词汇与世界
# ---------------------------------------------------------------------------

PEOPLE = ["mama", "baba", "baobao"]
OBJECTS = ["pingguo", "xigua", "qiu"]
ACTIONS = ["bao", "chi", "kan"]
MODIFIERS = ["da", "xiao", "hong"]
WORDS = PEOPLE + OBJECTS + ACTIONS + MODIFIERS
NW = len(WORDS)

# 语义类别（用于构造有结构的 lang 嵌入）
CAT_OF = {}
for w in PEOPLE:
    CAT_OF[w] = 0
for w in OBJECTS:
    CAT_OF[w] = 1
for w in ACTIONS:
    CAT_OF[w] = 2
for w in MODIFIERS:
    CAT_OF[w] = 3

# 拼音声学：声母发音部位 + 韵母（用于构造 aud 特征）
PINYIN = {
    "mama": ("m", "a"), "baba": ("b", "a"), "baobao": ("b", "ao"),
    "pingguo": ("p", "ing"), "xigua": ("x", "ua"), "qiu": ("q", "iu"),
    "bao": ("b", "ao"), "chi": ("ch", "i"), "kan": ("k", "an"),
    "da": ("d", "a"), "xiao": ("x", "iao"), "hong": ("h", "ong"),
}
SHENGMU = {"m": 0, "b": 1, "p": 1, "d": 2, "k": 3, "h": 4, "x": 5,
           "q": 5, "ch": 6}
YUNMU = {"a": 0, "ao": 1, "ing": 2, "ua": 3, "iu": 4, "i": 5,
         "an": 6, "iao": 7, "ong": 8}

# ---------------------------------------------------------------------------
# 通道维度
#
# ⚠ lang 改为 **one-hot 裸编码**（维度 = 词表大小 NW）。
#
# 旧编码是 "类别 one-hot(4) + 个体随机(4)" 的混合分布式表示，它让同类词
# 余弦高达 0.62~0.80。但**这种相近是构造出来的伪影，不是语言的性质**：
# 词作为符号，本来就是离散且互斥的。"妈妈"和"爸爸"都是人，但这是
# **语义**，应当从共现中学到，而不是预置进输入向量。
#
# 旧编码的三个后果（均已实测）：
#   1. 最近邻解码必然在同类词间混淆（cos(输出,竞争者) 0.745 > cos(输出,真值) 0.553）
#   2. 冲突判据失效（歧义下文的余弦 0.42~0.74，远高于 div 阈值 0.3）
#   3. 语义类别被硬编码，系统无事可学
#
# one-hot 下：不同词余弦恒为 0，解码 = argmax，语义结构交给系统去学。
# ---------------------------------------------------------------------------
D_VIS = 8      # 视觉通道
D_AUD = 8      # 听觉通道
D_LANG = NW    # 语言通道 = 词表大小（one-hot）
D = D_VIS + D_AUD + D_LANG      # 全局信号总维度（28）

NSH, NYU = 7, 9


def build_world(seed=0):
    """构造三通道特征：vis / aud / lang"""
    rng = np.random.default_rng(seed)
    vis, aud, lang = {}, {}, {}

    # lang：**one-hot 裸编码**
    #
    # 每个词是**一个独立维度**，词与词之间正交（余弦恒 0）。
    # 这是符号的正确建模：符号之间没有内在距离，只有"是/不是"。
    #
    # 语义类别（"妈妈和爸爸都是人"）**不再预置**，改由系统从共现中
    # 自己学 —— 那才是它该做的事。
    for i, w in enumerate(WORDS):
        v = np.zeros(D_LANG)
        v[i] = 1.0
        lang[w] = v

    # vis：人/物有强视觉（类内共享成分），动作/修饰弱
    proto = {0: rng.normal(0, 1, 4), 1: rng.normal(0, 1, 4)}
    for i, w in enumerate(WORDS):
        v = rng.normal(0, 0.05, D_VIS)
        c = CAT_OF[w]
        if c in proto:
            v[:4] = proto[c] + rng.normal(0, 0.25, 4)   # 类原型 + 个体偏移
        vis[w] = v

    # aud：声母部位 one-hot + 韵母 one-hot
    for w in WORDS:
        sm, ym = PINYIN[w]
        v = np.zeros(D_AUD)
        v[SHENGMU[sm] % 4] = 1.0
        v[4 + (YUNMU[ym] % 4)] = 1.0
        aud[w] = v

    return vis, aud, lang


def make_sentences(n=400, seed=0):
    """按模板生成句子：主谓宾 / 主谓 / 修饰+宾语"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.5:
            s = [rng.choice(PEOPLE), rng.choice(ACTIONS), rng.choice(OBJECTS)]
        elif r < 0.75:
            s = [rng.choice(PEOPLE), rng.choice(ACTIONS)]
        else:
            s = [rng.choice(MODIFIERS), rng.choice(OBJECTS)]
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# 反射：发声器官（定义"肢体"的行动面）
# ---------------------------------------------------------------------------

class VocalReflex(ReptilianFunction):
    """发声反射：婴儿的 babbling —— 能发声，但内容与语义无关

    ⚠ **为什么输出随机小噪声而不是视觉信号**（实测决定）：

    初版把 vis 直接当 lang 输出（"能映射"的直觉）。但这是个**错误映射**，
    会带来两个致命后果：

    1. λ 混合 ``y = (1-λ)·y_reflex + λ·y_higher`` 把这个错误值掺进输出，
       高层学得再好也被稀释；
    2. 自回归 progress = (err_reflex − err_high)/err_reflex。反射在 lang 段
       输出 vis 值，误差不够大 → progress 上不去 → λ 无法趋向 1
       → 高层永远接管不了。实测 progress 卡在 0.4~0.6。

    改为**随机 babbling**（零附近的小噪声）后：
    - 反射在 lang 段的误差 ≈ 目标自身的能量（很大）
    - 只要高层输出比噪声更接近目标，progress 就显著为正 → λ 快速趋向 1
    - 高层真正接管，不再被错误基线稀释

    这也更符合生物学：婴儿的 babbling 是随机音节练习，不是视觉的复述。
    """

    def __init__(self, amp: float = 0.1, seed: int = 0, out_dim: int = None):
        self._amp = amp
        self._rng = np.random.default_rng(seed)
        #
        # ⚠ 输出维度必须**显式指定**为 lang 端口的维度。
        #
        # 旧实现按**输入** vis 的长度生成噪声，输出给 lang 端口。
        # 当两通道维度相同时（都是 8）碰巧能跑；一旦 lang 改为
        # one-hot（维度 = 词表大小 12）而 vis 仍是 8，就会
        # ``shapes (8,) (12,) 不匹配`` → intervene 抛异常 → 被
        # except 吞掉 → **回退纯反射，高层从未介入**。
        #
        # 反射的输出维度由它驱动的**器官**决定，与输入无关——
        # 发声器官能发出的音素数不取决于看到的图像分辨率。
        self._out_dim = out_dim

    def get_input_spec(self):
        return {"vis": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        n = self._out_dim
        if n is None:
            v = inputs["vis"]
            n = int(np.asarray(v.data.a if hasattr(v.data, "a") else v.data).size)
        babble = shim.tensor((self._rng.normal(0, self._amp, n)).astype(np.float32))
        return {"lang": Signal(data=babble, mime_type="generic/tensor",
                               metadata={"source": "vocal_babbling"})}


class EchoCh(ReptilianFunction):
    """把某通道原样输出（自回归中 vis/aud 段的基线预测器）"""

    def __init__(self, name="echo"):
        self._name = name

    def get_input_spec(self):
        return {"x": "generic/tensor"}

    def get_output_spec(self):
        return {self._name: "generic/tensor"}

    def execute(self, inputs):
        x = inputs["x"]
        return {self._name: Signal(data=x.data,
                                   mime_type="generic/tensor",
                                   metadata={"source": "echo"})}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

CH = ("vis", "aud", "lang")
# PORT：每个通道自己的维度（lang = NW，one-hot）
_OFF = {}
_o = 0
for _c, _d in (("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG)):
    _OFF[_c] = _o
    _o += _d
PORT = {c: (_OFF[c], d) for c, d in
        (("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG))}

LANG_SLICE = slice(_OFF["lang"], _OFF["lang"] + D_LANG)


def decode_lang(vec) -> str:
    """**裸编码解码：argmax**，不用余弦

    one-hot 下每个词是一个独立维度，取最大分量即该词。
    不需要相似度、不需要归一化 —— 符号识别是**离散判定**，
    不是"找最接近的向量"。

    注：对 one-hot 目标，argmax 与余弦最近邻在数学上等价
    （|v| 对所有词相同），但 argmax 更直接，且保留幅度语义
    （输出很弱时仍可选出，只是可信度低）。
    """
    a = np.asarray(vec, dtype=float).ravel()
    # 两种情况：完整全局帧（取 lang 段）或单通道 lang 信号（直接用）
    if a.size == D:
        a = a[LANG_SLICE]
    elif a.size != D_LANG:
        return None
    i = int(np.argmax(a))
    return WORDS[i] if i < NW else None


def build_brain(seed=0, decay_rate=0.01, wiring_block=20):
    np.random.seed(seed)
    brain = DevelopmentalSystem(port_layout=PORT, decay_rate=decay_rate)
    brain.phase = "exploratory"
    brain.wiring_block = wiring_block
    # 冲突观察只针对 lang —— 高层真正要学的输出
    # （vis/aud 由反射 echo，不参与冲突判定，否则稀释差异）
    brain.observe_output_channel = "lang"

    k = brain.reflex._kernel
    k.register("vocal", VocalReflex(out_dim=D_LANG))
    k.register("echo_vis", EchoCh("vis"))
    k.register("echo_aud", EchoCh("aud"))
    # 路由：vis/aud 原样输出（基线），vis → vocal 产生 lang（要学的）
    brain.reflex.add_preorchestrated_route(source="vis", target="echo_vis",
                                           priority=1.0)
    brain.reflex.add_preorchestrated_route(source="aud", target="echo_aud",
                                           priority=1.0)
    brain.reflex.add_preorchestrated_route(source="vis", target="vocal",
                                           priority=0.5)
    return brain


def decode(vec, lang_tab):
    """语言向量 → 最近邻词"""
    best, bi = -1e9, -1
    for i, w in enumerate(WORDS):
        s = float(np.dot(vec, lang_tab[w]) /
                  (np.linalg.norm(vec) * np.linalg.norm(lang_tab[w]) + 1e-9))
        if s > best:
            best, bi = s, i
    return WORDS[bi], best


def run(calib_frac=0.4, steps=3000, seed=0, decay_rate=0.01,
        wiring_block=20, verbose=False):
    vis, aud, lang = build_world(seed)
    sents = make_sentences(seed=seed)
    # 句子流 → 词流
    stream = []
    for s in sents:
        stream.extend(s)
        if len(stream) >= steps + 100:
            break

    brain = build_brain(seed, decay_rate, wiring_block)
    n0 = brain.higher_brain.ecosystem.get_neuron(0)
    for c in CH:
        n0.unfold(c, shim.zeros(D))

    calib_steps = int(steps * calib_frac)
    prev_lang_out = np.zeros(D)

    ar_errs, gen_words, true_words = [], [], []
    for t in range(steps):
        w = stream[t % len(stream)]
        w_next = stream[(t + 1) % len(stream)]

        production = t >= calib_steps
        # 感知校准期：lang 来自环境（听到）
        # 产出期：lang 来自系统自己上一步的输出（打字机自反馈）
        lang_in = prev_lang_out if production else lang[w]

        inputs = {
            "vis": Signal(shim.tensor(vis[w].copy()), "generic/tensor", {"t": t}),
            "aud": Signal(shim.tensor(aud[w].copy()), "generic/tensor", {"t": t}),
            "lang": Signal(shim.tensor(lang_in.copy()), "generic/tensor",
                           {"t": t}),
        }
        try:
            res = brain.step_awake(external_inputs=inputs)
        except Exception as e:
            if verbose:
                print(f"  step {t} 崩溃: {type(e).__name__}: {e}")
            return {"error": f"{type(e).__name__}: {e}"}

        outs = res.get("outputs") or {}
        lg = outs.get("lang")
        if lg is not None:
            prev_lang_out = np.asarray(
                lg.data.a if hasattr(lg.data, "a") else lg.data,
                dtype=float).ravel()[:D]

        if production:
            gw, _ = decode(prev_lang_out, lang)
            gen_words.append(gw)
            true_words.append(w_next)

    reg = brain.higher_brain.sheath_registry
    eco = brain.higher_brain.ecosystem
    acc = (np.mean([g == r for g, r in zip(gen_words, true_words)])
           if gen_words else float("nan"))
    return {
        "error": None,
        "next_word_acc": float(acc),
        "neurons": len(eco.neurons),
        "sheaths": len(reg._sheaths),
        "turnover": reg._born + reg._died,
        "progress": float(getattr(brain.higher_brain.scheduler, "progress", 0.0)),
        "gen_words": gen_words,
        "true_words": true_words,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--calib", type=float, default=0.4)
    args = ap.parse_args()

    print("=" * 72)
    print("婴幼儿多模态语言习得（真实 DevelopmentalSystem）")
    print("=" * 72)

    print("\n【冒烟】单种子")
    r = run(calib_frac=args.calib, steps=args.steps, seed=0, verbose=True)
    if r["error"]:
        print(f"  崩溃: {r['error']}")
        sys.exit(1)
    print(f"  { {k: v for k, v in r.items() if k not in ('gen_words', 'true_words')} }")
    print(f"  生成样例: {r['gen_words'][:12]}")
    print(f"  真值样例: {r['true_words'][:12]}")
