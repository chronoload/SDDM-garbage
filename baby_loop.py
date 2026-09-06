"""**反应**而非预测：系统输出参与真实回路，学会控制

## 与预测任务的根本区别

======  ==================================================  ==================
        预测（此前）                                        反应（本脚本）
======  ==================================================  ==================
因果     环境自己演进，系统旁观并猜下一个                    系统**输出**改变环境状态
角色     旁观者                                            参与者
目标     拟合序列的统计规律                                  让世界进入想要的状态
======  ==================================================  ==================

在预测任务里，"下一个输入"由环境独立决定，系统说什么都无所谓。
在反应任务里，**下一个输入取决于系统刚才说了什么** —— 系统必须
发现并掌握这个因果结构，否则得不到它想要的东西。

这正是"有意识地学会控制一段 loop"。

## 任务

婴儿通过**发声**操纵世界：

- **内在驱动 ``want`` 属于爬虫脑**，不是外部输入通道。
  它是身体的内部状态（类比下丘脑的饥饿/渴/体温驱动），
  由身体设定，通过**反射的输出**影响世界。
- **输入通道只有 vis / aud / lang** —— 全部是外部感知。
  其中 lang 包含**自己上一帧发出的声音**（婴儿听得到自己说话）。
- 系统输出一个词（发声）→ 环境响应：**说对了 → 该物体出现在
  下一帧的 vis 通道**；说错了 → 什么都没有。

控制 loop 的因果链：

    内在驱动(爬虫脑) → 发声(行动) → 世界响应 → 感知变化 → 学习

高层脑**看不到** want。它只能通过"感知到物体出现了"这个结果，
反过来学会哪个发声有效。这正是"有意识地学会控制一段 loop"。

## 对照组

- **反射基线**：随机哭闹（babbling），满足率 = 1/12
- **预测对照**：输出不接入环境（教师强制），学不到因果

## 编码

lang / want 均为 **one-hot 裸编码**（符号是离散互斥的，无内在距离）。
解码用 argmax，不用余弦。
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
import baby_real as B                                            # noqa: E402

WORDS = B.WORDS
NW = B.NW
OBJECTS = B.OBJECTS          # 可"被要到"的东西
D_VIS, D_AUD, D_LANG = B.D_VIS, B.D_AUD, B.D_LANG
D_WANT = NW                  # 需求通道：也是 one-hot

# 通道偏移 —— **只有外部感知通道**，没有 want
#
# ⚠ want 是爬虫脑的内在驱动，不是信号通道（见模块文档）。
# 把内在状态做成输入通道，等于假设身体能"感知"自己的需求，
# 但需求不是感知——它是驱动，作用在行动侧而非感觉侧。
_OFF, _o = {}, 0
for _c, _d in (("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG)):
    _OFF[_c] = _o
    _o += _d
D_ALL = _o
PORT = {c: (_OFF[c], d) for c, d in
        (("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG))}

LANG_SLICE = slice(_OFF["lang"], _OFF["lang"] + D_LANG)


def _as_np(v):
    """Signal / shim 张量 → 一维 float numpy"""
    if v is None:
        return None
    d = v.data if isinstance(v, Signal) else v
    if hasattr(d, "detach"):
        d = d.detach().cpu().numpy()
    elif hasattr(d, "a"):
        d = d.a
    return np.asarray(d, dtype=float).ravel()


def decode_lang(vec):
    """裸编码解码：argmax（符号识别是离散判定，不是找最近邻）

    ⚠ 兼容两种尺寸：
      - ``size == D_LANG``：输出信号是**分通道**的，lang 通道本身就是 12 维
      - ``size == D_ALL``：全局信号，需按 LANG_SLICE 取出 lang 段
    """
    a = _as_np(vec)
    if a is None:
        return None
    if a.size == D_LANG:
        seg = a
    elif a.size >= _OFF["lang"] + D_LANG:
        seg = a[LANG_SLICE]
    else:
        return None
    i = int(np.argmax(seg))
    return WORDS[i] if i < NW else None


class BabyReflex(ReptilianFunction):
    """爬虫脑：**持有内在驱动**并据此发声

    ⚠ 关键澄清：``want`` 是**内在驱动**，不是输入信号。

    生物学对应：下丘脑的饥饿/渴/体温驱动是身体的内部状态，
    它通过**反射**作用于行动（哭闹、寻找），而不是作为"被感知
    到的输入"。需求驱动行动，不是驱动感知。

    反射定义"发声器官"这个**行动面**：婴儿天生会发声，但不知道
    发声能做什么 —— 它提供量级合理的起点，不含因果知识。

    因果知识只能由高层脑从"发声 → 世界响应"的观察中学到。
    """

    def get_input_spec(self):
        return {"lang": "generic/tensor"}     # 听到自己上一帧的声音

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.drive: str | None = None         # 内在驱动（由身体设定）

    def set_drive(self, want: str) -> None:
        """身体设定内在驱动 —— **不是**输入通道"""
        self.drive = want

    def execute(self, inputs):
        out = np.zeros(D_LANG)
        if self.drive is not None:
            # 反射只能"照抄驱动"，它**不知道**这么做会怎样
            out[WORDS.index(self.drive)] = 1.0
        out = out + self.rng.normal(0, 0.3, D_LANG)   # 哭闹噪声
        return {"lang": Signal(data=shim.tensor(out), mime_type="generic/tensor",
                               metadata={"source": "babbling"})}


class EchoVis(ReptilianFunction):
    def get_input_spec(self):
        return {"vis": "generic/tensor"}

    def get_output_spec(self):
        return {"vis": "generic/tensor"}

    def execute(self, inputs):
        return {"vis": inputs["vis"]}


class EchoAud(ReptilianFunction):
    def get_input_spec(self):
        return {"aud": "generic/tensor"}

    def get_output_spec(self):
        return {"aud": "generic/tensor"}

    def execute(self, inputs):
        return {"aud": inputs["aud"]}


def build_brain(seed=0, decay_rate=0.01, wiring_block=20):
    np.random.seed(seed)
    brain = DevelopmentalSystem(port_layout=PORT, decay_rate=decay_rate)
    brain.phase = "exploratory"
    brain.wiring_block = wiring_block
    k = brain.reflex._kernel
    k.register("babble", BabyReflex(seed))
    k.register("echo_vis", EchoVis())
    k.register("echo_aud", EchoAud())
    brain.reflex.add_preorchestrated_route(source="vis", target="echo_vis", priority=1.0)
    brain.reflex.add_preorchestrated_route(source="aud", target="echo_aud", priority=1.0)
    brain.reflex.add_preorchestrated_route(source="lang", target="babble", priority=0.5)
    return brain


class ReactiveWorld:
    """反应式世界：系统的输出**改变**环境状态

    因果链：系统发声 → 世界响应 → 下一帧观测

    这是"控制 loop"，不是观测序列。
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.vis, self.aud, self.lang = B.build_world(seed)
        self.obj = None            # 当前在场物体（None = 什么都没有）
        self.want = OBJECTS[0]     # 当前需求（**内在驱动**，不进入观测）
        self.last_utterance = None # 婴儿上一帧发出的声音
        self.satisfied = 0
        self.trials = 0

    def observe(self):
        """构造当前观测帧（**只有外部感知**）

        - vis/aud：在场物体的视觉与声音（没有物体则为零）
        - lang：婴儿**自己上一帧发出的声音**（听得到自己说话）

        注意：**没有 want 段** —— 需求是内在驱动，不被感知。
        """
        v = np.zeros(D_ALL)
        if self.obj is not None:
            v[_OFF["vis"]:_OFF["vis"] + D_VIS] = self.vis[self.obj]
            v[_OFF["aud"]:_OFF["aud"] + D_AUD] = self.aud[self.obj]
        if self.last_utterance is not None:
            v[LANG_SLICE] = self.lang[self.last_utterance]
        return v

    def act(self, word):
        """系统发声 → 世界响应

        这是控制 loop 的核心：**世界的下一个状态由系统的输出决定**。

        Returns:
            (reward, 需求是否被满足)
        """
        self.trials += 1
        self.last_utterance = word     # 婴儿听到自己说的话
        if word == self.want:
            self.obj = self.want                       # 说对了 → 东西出现
            self.satisfied += 1
            self.want = OBJECTS[self.rng.integers(len(OBJECTS))]
            return 1.0, True
        # 说错了 → 什么都没有（世界不响应）
        self.obj = None
        return 0.0, False


def run(seed=0, steps=2000, reactive=True, reflect=True):
    """跑一个 episode

    Args:
        reactive: True = 输出接入世界（反应）；False = 教师强制（预测）
        reflect: 反射是否参与混合
    """
    np.random.seed(seed)
    world = ReactiveWorld(seed)
    brain = build_brain(seed)
    baby_reflex = brain.reflex._kernel.get("babble")
    hb = brain.higher_brain
    n0 = hb.ecosystem.get_neuron(0)
    for c in PORT:
        n0.unfold(c, shim.zeros(D_ALL))

    rewards, last_word = [], None
    for t in range(steps):
        # 内在驱动由**身体**设定给爬虫脑 —— 不是输入通道
        baby_reflex.set_drive(world.want)

        obs = world.observe()
        if not reactive:
            # 教师强制：lang 通道直接给"正确答案"，输出不接入世界
            obs[LANG_SLICE] = world.lang[world.want]

        inputs = {}
        for c, (off, sz) in PORT.items():
            inputs[c] = Signal(shim.tensor(obs[off:off + sz].copy()),
                               "generic/tensor", {"t": t})

        res = brain.step_awake(external_inputs=inputs)
        out = (res.get("outputs") or {}).get("lang")

        if reactive:
            word = decode_lang(out) if out is not None else None
            r, _ = world.act(word)
            # **身体**回报内在驱动的满足度 → 裁定 λ
            # 这是爬虫脑的内在信号回流，不是感知输入
            brain.report_drive_satisfaction(r)
            rewards.append(r)
        else:
            # 预测模式：系统输出什么无所谓，世界按自己的规律走
            rewards.append(1.0 if decode_lang(out) == world.want else 0.0)
            world.act(world.want)   # 自动满足

    k = max(1, len(rewards) // 4)
    return dict(sat=np.mean(rewards[-k:]),
                sat_all=np.mean(rewards),
                neurons=len(hb.ecosystem.neurons),
                sheaths=len(hb.sheath_registry._sheaths),
                turnover=hb.sheath_registry._born + hb.sheath_registry._died,
                shadow=brain.shadow_mode,
                competence=brain.drive_sat.competence,
                **{f"ds_{k2}": v for k2, v in brain.drive_sat.report().items()})


if __name__ == "__main__":
    print("=" * 74)
    print("反应 vs 预测：系统输出是否参与真实回路")
    print("=" * 74)
    print(f"{'模式':>22} | {'满足率(末1/4)':>13} | {'神经元':>6} | {'髓鞘':>5} | {'周转':>6}")
    print("-" * 74)
    for name, reactive in (("预测（教师强制）", False), ("反应（输出接入世界）", True)):
        rs = [run(seed=s, steps=2000, reactive=reactive) for s in range(3)]
        m = lambda k: np.mean([r[k] for r in rs])
        sd = np.std([r["sat"] for r in rs])
        print(f"{name:>22} | {m('sat'):.3f}±{sd:.3f}     | {m('neurons'):6.1f} | "
              f"{m('sheaths'):5.1f} | {m('turnover'):6.0f}")
    print("-" * 74)
    print(f"随机哭闹基线 = 1/12 = {1/12:.3f}")
    print()
    print("判据：")
    print("  · 预测模式若也能达标 → 说明任务没真正构成控制 loop")
    print("  · 反应模式显著更高   → 系统确实学会了用输出改变世界")
