"""张力驱动的反应式控制 —— **感受不是值，是张力**

## 核心纠正

上一版把内在驱动建模为**标量满足度**（reward ∈ [0,1]）。
这是错的。感受不是一个可以被读取和优化的**值**。

**张力（tension）** 与 **奖励（reward）** 的根本区别：

============  ==============================  ==========================
              奖励 reward                      张力 tension
============  ==============================  ==========================
本体           外部注入的标量                   系统**内在的状态**
关系           系统 ↔ 外部评价者                 系统当前状态 ↔ 稳态设定点
方向性         无（只有大小）                    **有**（在身体状态空间中指向稳态）
作用方式       被优化（最大化）                  被**消解**（relaxation）
通达方式       输入通道（点対点）                 **弥散调制**（volume transmission）
============  ==============================  ==========================

## 张力的三个性质（必须在代码中体现）

1. **矢量性**：tension = setpoint − state，在身体状态空间中有**大小和方向**。
   不是标量，不能被 argmax 优化。

2. **调制性**：张力**弥散地改变整个系统的运作模式** —— 探索率、学习强度、
   门控阈值。这不是输入通道，对应神经调质的 volume transmission
   （多巴胺/血清素/去甲肾上腺素是弥散释放，改变整个神经系统的增益，
   而非传递点对点的数值）。

3. **消解性**：行动的目标是**消解张力**，不是最大化数值。
   反馈 = 张力矢量的变化，而非外部打分。

## 内感受 vs 驱动

- **驱动（drive）** 在爬虫脑：下丘脑产生，作用在**行动侧**，不可感知
- **内感受（interoception）** 在感觉侧：岛叶皮层产生，
  是"身体当前状态在感知层的投影"，**可感知但不直接给出答案**

内感受给出张力的**方向**（缺什么），但不告诉你**哪个物体**能补充 ——
那个映射必须由学习获得。

## 任务设计（让高层有可能超越反射）

- 身体有 3 种需求（饥/渴/玩），持续漂移 → 张力累积
- 每步**随机有物体缺席** —— 视野只反映在场物体
- **反射**：只读内感受 → 说出对应物体名，**不看视野** → 可能说到缺席的东西
- **高层**：同时看到内感受 + 视野 → 学会"在场的东西中，哪个能消解张力"

这给了高层真正超越反射的空间（反射会说到不在场的东西）。
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
D_LANG = B.D_LANG
D_VIS, D_AUD = B.D_VIS, B.D_AUD

# 身体状态空间：3 种需求
NEEDS = ["hungry", "thirsty", "play"]
N_NEED = len(NEEDS)
# 每个（可满足需求的）物体补充哪个维度
# 注意：只有 3 个物体有需求效应，其余词是"无效发声"
OBJ_EFFECT = {
    "pingguo": np.array([1.0, 0.0, 0.0]),   # 苹果 → 充饥
    "xigua":   np.array([0.0, 1.0, 0.0]),   # 西瓜 → 解渴
    "qiu":     np.array([0.0, 0.0, 1.0]),   # 球 → 玩
}
TARGETS = list(OBJ_EFFECT.keys())


class Body:
    """身体：维持稳态，**产生张力**

    ⚠ 张力不是奖励。它是系统内在的、有方向的、需要被消解的状态。
    """

    def __init__(self, seed=0, drift_rate=0.03):
        self.rng = np.random.default_rng(seed)
        self.state = np.ones(N_NEED)       # 从满足开始
        self.setpoint = np.ones(N_NEED)
        self.drift_rate = drift_rate

    @property
    def tension(self) -> np.ndarray:
        """**张力矢量**：指向恢复稳态的方向

        这才是"感受"的正确形式 —— 不是一个值，
        而是**被拉扯的方向与强度**。
        """
        return self.setpoint - self.state

    @property
    def magnitude(self) -> float:
        """张力强度（仅用于调制强度与诊断，不是学习目标）"""
        return float(np.linalg.norm(self.tension))

    def drift(self) -> None:
        """身体自然漂移：需求随时间增长 → 张力累积"""
        self.state = np.maximum(0.0, self.state - self.drift_rate)

    def intake(self, word) -> np.ndarray:
        """使用某物 → 身体状态改变 → 张力**消解**

        Returns:
            实际产生的效应向量（受稳态上限截断）
        """
        eff = OBJ_EFFECT.get(word)
        if eff is None:
            return np.zeros(N_NEED)
        before = self.state.copy()
        self.state = np.minimum(self.setpoint, self.state + eff)
        return self.state - before

    def interoception(self, noise: float = 0.15) -> np.ndarray:
        """**内感受**：张力在感知层的投影

        给出张力的**方向**（缺什么），但有噪声、且**不指向任何具体物体**。

        生物学：岛叶皮层的内感受。与下丘脑的驱动同源但不同层 ——
        驱动在行动侧，内感受在感觉侧。
        """
        T = self.tension
        n = np.linalg.norm(T)
        if n < 1e-8:
            return np.zeros(N_NEED)
        direction = T / n
        direction = direction + self.rng.normal(0, noise, N_NEED)
        return np.maximum(0.0, direction)


class TensionGate:
    """张力如何**弥散地调制**整个系统

    对应神经调质的 volume transmission：张力不是输入通道，
    而是改变系统运作模式的背景场。
    """

    def __init__(self, body: Body, base_explore: float = 0.05,
                 modulation: float = 0.35):
        self.body = body
        self.base_explore = base_explore
        self.modulation = modulation

    @property
    def explore_rate(self) -> float:
        """张力越大 → 越"不安" → 探索越多"""
        return self.base_explore + self.modulation * min(1.0, self.body.magnitude)

    @property
    def arousal(self) -> float:
        """唤醒度：调制学习强度"""
        return min(1.0, self.body.magnitude)


# ---------------------------------------------------------------------------
# 端口：内感受是一个**独立通道**，但它调制而非被优化
# ---------------------------------------------------------------------------
_OFF, _o = {}, 0
for _c, _d in (("intero", N_NEED), ("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG)):
    _OFF[_c] = _o
    _o += _d
D_ALL = _o
PORT = {c: (_OFF[c], d) for c, d in
        (("intero", N_NEED), ("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG))}
LANG_SLICE = slice(_OFF["lang"], _OFF["lang"] + D_LANG)


def _as_np(v):
    if v is None:
        return None
    d = v.data if isinstance(v, Signal) else v
    if hasattr(d, "detach"):
        d = d.detach().cpu().numpy()
    elif hasattr(d, "a"):
        d = d.a
    return np.asarray(d, dtype=float).ravel()


def decode_lang(vec):
    a = _as_np(vec)
    if a is None:
        return None
    seg = a if a.size == D_LANG else (a[LANG_SLICE] if a.size >= _OFF["lang"] + D_LANG else None)
    if seg is None:
        return None
    i = int(np.argmax(seg))
    return WORDS[i] if i < NW else None


class DriveReflex(ReptilianFunction):
    """爬虫脑反射：**只读内感受，不看视野**

    这定义了它的能力边界：知道"缺什么"，不知道"有什么"。
    当对应物体缺席时，它会说出不在场的东西 → 失败。

    高层若学会看视野，就能超越它。
    """

    def get_input_spec(self):
        return {"intero": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def __init__(self, mapping=None, seed=0):
        # 内感受方向 → 需求维度 → 物体（先天映射，不需学习）
        self.mapping = mapping or TARGETS
        self.rng = np.random.default_rng(seed)

    def execute(self, inputs):
        x = inputs["intero"]
        a = _as_np(x)
        out = np.zeros(D_LANG)
        if a is not None and a.size >= N_NEED:
            k = int(np.argmax(a[:N_NEED]))
            if k < len(self.mapping):
                out[WORDS.index(self.mapping[k])] = 1.0
        return {"lang": Signal(data=shim.tensor(out + self.rng.normal(0, 0.2, D_LANG)),
                               mime_type="generic/tensor",
                               metadata={"source": "drive_reflex"})}


class EchoVis(ReptilianFunction):
    def get_input_spec(self): return {"vis": "generic/tensor"}
    def get_output_spec(self): return {"vis": "generic/tensor"}
    def execute(self, inputs): return {"vis": inputs["vis"]}


class EchoAud(ReptilianFunction):
    def get_input_spec(self): return {"aud": "generic/tensor"}
    def get_output_spec(self): return {"aud": "generic/tensor"}
    def execute(self, inputs): return {"aud": inputs["aud"]}


class TensionWorld:
    """世界：物体的**在场与否**决定行动能否生效"""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.vis, self.aud, self.lang = B.build_world(seed)
        self.body = Body(seed)
        self.present: list[str] = list(TARGETS)
        self.relief_total = 0.0
        self.trials = 0
        self.last_utterance = None

    def roll_presence(self):
        """随机让 1 个物体缺席 —— 制造反射会失败的情形"""
        absent = self.rng.choice(TARGETS)
        self.present = [t for t in TARGETS if t != absent]

    def observe(self) -> np.ndarray:
        v = np.zeros(D_ALL)
        # 内感受（感觉侧投影）
        v[_OFF["intero"]:_OFF["intero"] + N_NEED] = self.body.interoception()
        # 视野：只反映**在场**物体
        if self.present:
            vv = np.mean([self.vis[t] for t in self.present], axis=0)
            aa = np.mean([self.aud[t] for t in self.present], axis=0)
            v[_OFF["vis"]:_OFF["vis"] + D_VIS] = vv
            v[_OFF["aud"]:_OFF["aud"] + D_AUD] = aa
        if self.last_utterance is not None:
            v[LANG_SLICE] = self.lang[self.last_utterance]
        return v

    def act(self, word) -> float:
        """行动 → 身体状态改变 → **张力消解量**

        Returns:
            张力范数的减小量（正 = 张力被消解，负 = 张力累积）
        """
        self.trials += 1
        self.last_utterance = word
        before = self.body.magnitude
        if word in self.present:
            self.body.intake(word)      # 只有在场才生效
            self.roll_presence()        # 用掉后重新布置
        self.body.drift()               # 需求持续累积
        relief = before - self.body.magnitude
        self.relief_total += max(0.0, relief)
        return relief


def build_brain(seed=0, decay_rate=0.01, wiring_block=20):
    np.random.seed(seed)
    brain = DevelopmentalSystem(port_layout=PORT, decay_rate=decay_rate)
    brain.phase = "exploratory"
    brain.wiring_block = wiring_block
    k = brain.reflex._kernel
    k.register("drive", DriveReflex(seed=seed))
    k.register("echo_vis", EchoVis())
    k.register("echo_aud", EchoAud())
    brain.reflex.add_preorchestrated_route(source="vis", target="echo_vis", priority=1.0)
    brain.reflex.add_preorchestrated_route(source="aud", target="echo_aud", priority=1.0)
    brain.reflex.add_preorchestrated_route(source="intero", target="drive", priority=0.5)
    return brain


def run(seed=0, steps=2000, mode="tension", world=None):
    """mode: 'tension'=影子+张力裁定  'reflex'=强制纯反射
             'old'=λ 由预测误差裁定（对照组）"""
    np.random.seed(seed)
    world = world or TensionWorld(seed)
    brain = build_brain(seed)
    hb = brain.higher_brain
    n0 = hb.ecosystem.get_neuron(0)
    for c in PORT:
        n0.unfold(c, shim.zeros(D_ALL))

    gate = TensionGate(world.body)
    if mode == "old":
        brain.shadow_mode = False
        hb.drive_sat = None

    reliefs, lambdas = [], []
    for t in range(steps):
        # 张力的**弥散调制**
        #
        # ⚠ 方向曾写反：初版是"张力越大 → 探索越多"，实测反而拉低
        # 性能（0.0225 vs 反射 0.0301）—— 高张力时恰恰最需要可靠行动，
        # 此时大量探索是在最坏的时机冒最大的险。
        #
        # 正确的调制：高张力 → **收缩**探索（依赖已验证的可靠通路）；
        # 低张力（已接近稳态）→ 才放开探索（有余裕去试新东西）。
        # 这也符合"探索应在安全余量内进行的直觉"。
        if mode == "tension":
            brain.explore_eps = max(0.02, gate.explore_rate * (1.0 - gate.arousal))

        obs = world.observe()
        inputs = {c: Signal(shim.tensor(obs[o:o + s].copy()),
                            "generic/tensor", {"t": t})
                  for c, (o, s) in PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        out = (res.get("outputs") or {}).get("lang")
        w = decode_lang(out)
        relief = world.act(w)

        if mode == "tension":
            # 反馈是**张力的消解**，不是外部奖励值
            brain.report_drive_satisfaction(max(0.0, relief))
        reliefs.append(max(0.0, relief))
        lambdas.append(getattr(hb, "_last_lambda", 0.0))

    k = max(1, len(reliefs) // 3)
    return dict(relief=float(np.mean(reliefs[-k:])),
                relief_all=float(np.mean(reliefs)),
                lam=float(np.mean(lambdas[-k:])),
                shadow=bool(brain.shadow_mode),
                comp=float(brain.drive_sat.competence),
                neurons=len(hb.ecosystem.neurons))


if __name__ == "__main__":
    print("=" * 78)
    print("张力驱动的反应式控制（3 seeds × 2000 步）")
    print("=" * 78)
    print(f"{'模式':>16} | {'张力消解(末1/3)':>15} | {'λ':>6} | {'影子中':>6} | {'胜任度':>7} | {'神经元':>6}")
    print("-" * 78)
    for mode, name in (("old", "旧：预测误差裁定"),
                       ("reflex", "强制反射"),
                       ("tension", "新：影子+张力")):
        rs = []
        for s in range(3):
            if mode == "reflex":
                # 强制纯反射
                w = TensionWorld(s)
                br = build_brain(s)
                br.explore_eps = 0.0
                hb2 = br.higher_brain
                nn = hb2.ecosystem.get_neuron(0)
                for c in PORT:
                    nn.unfold(c, shim.zeros(D_ALL))
                dr = br.reflex._kernel.get("drive")
                rr = []
                for t in range(2000):
                    obs = w.observe()
                    inputs = {c: Signal(shim.tensor(obs[o:o + sz].copy()),
                                        "generic/tensor", {"t": t})
                              for c, (o, sz) in PORT.items()}
                    br.step_awake(external_inputs=inputs)
                    wo = decode_lang(dr.execute({"intero": inputs["intero"]})["lang"])
                    rr.append(max(0.0, w.act(wo)))
                rs.append(dict(relief=float(np.mean(rr[-666:])), lam=0.0,
                               shadow=True, comp=0.0, neurons=1))
            else:
                rs.append(run(seed=s, mode=mode))
        m = lambda k: np.mean([r[k] for r in rs])
        sd = np.std([r["relief"] for r in rs])
        print(f"{name:>16} | {m('relief'):.4f}±{sd:.4f}   | {m('lam'):6.3f} | "
              f"{str(bool(m('shadow'))):>6} | {m('comp'):7.3f} | {m('neurons'):6.1f}")
    print("-" * 78)
    print("张力消解 = 每步张力范数的减小量（正=消解，0=无效行动）")
    print()
    print("判读：")
    print("  · 高层 competence > 0  → 高层超越了反射（学会了看视野）")
    print("  · 新方案 ≥ 强制反射    → 影子模式正确地把控制权留在了强者手里")
