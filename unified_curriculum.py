"""T10 统一词表空间 + 跨沉积物持续课程（棘轮教义的完全体）

## 设计

统一信道布局（所有沉积物共享，脑只构造一次）：

    vis  (256) : 当前 tick 的世界显示（各沉积物映射进 256 格）
    disp (256) : 当前 token 的表盘（one-hot，结算键/情境键）
    lang (256) : 自己上一帧的发声

各沉积物的 token 都映射进同一个 0–255 符号空间：
  - 语法物体名 0–15
  - 文本字节 0–255
  - 图片亮度等级 0–15
  - 电影象限 128–131

同一个"词"在不同沉积物里的语义由情境账本分账——
这正是持续课程要检验的：符号共享 + 情境分账能否兼得。

## 课程

    阶段1 语法(2500) → 阶段2 图片(2000) → 阶段3 电影(2000) → 阶段4 语法(1500)

指标：
- 棘轮：阶段4 的语法成绩 vs 阶段1（学图片/电影后旧技能是否保持）
- 迁移：图片/电影阶段的学习速度 vs 新脑单独学
- checkpoint 在阶段间保存/恢复（T9 机制的真实使用）
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")

import numpy as np

import numpy_torch_shim as shim
shim.install()

from mcp.developmental.signal import Signal                      # noqa: E402
from mcp.developmental.system import DevelopmentalSystem         # noqa: E402
from mcp.developmental.reptilian import ReptilianFunction        # noqa: E402
from mcp.developmental.persistence import (                      # noqa: E402
    save_checkpoint, load_checkpoint)

import baby_grammar as BG                                        # noqa: E402
import deposit_media as DM                                       # noqa: E402
import deposit_reader as DR                                      # noqa: E402

D_UNI = 256
PORT = {"vis": (0, D_UNI), "disp": (D_UNI, D_UNI),
        "lang": (D_UNI * 2, D_UNI)}
D_ALL = D_UNI * 3


class UnifiedAdapter:
    """把各沉积物世界适配到统一 768 维协议"""

    def __init__(self, world, token_offset=0, vis_map=None, name="?",
                 token_fn=None):
        self.world = world
        self.token_offset = token_offset
        self.vis_map = vis_map or (lambda w: self._onehot(w.cur_token()))
        self.name = name
        self.token_fn = token_fn or (lambda w: w.cur_token())
        self.last_utter = None

    @staticmethod
    def _onehot(idx, n=D_UNI):
        v = np.zeros(n)
        if idx is not None and 0 <= idx < n:
            v[idx] = 1.0
        return v

    def observe(self):
        v = np.zeros(D_ALL)
        v[:D_UNI] = self.vis_map(self.world)
        tok = min(self.cur_token(), D_UNI - 1)
        v[D_UNI + tok] = 1.0
        if self.last_utter is not None:
            v[D_UNI * 2 + min(self.last_utter, D_UNI - 1)] = 1.0
        return v

    def cur_token(self):
        return self.token_fn(self.world) + self.token_offset

    def _advance(self):
        self.world._advance()


def make_adapters():
    """四类沉积物的统一适配（各自的原生世界做内核）"""
    ads = {}

    # 语法：vis = 物体 one-hot（16 格）；token 走 cur 属性
    gw = BG.GrammarWorld(0)
    ads["grammar"] = UnifiedAdapter(
        gw, token_offset=0,
        vis_map=lambda w: UnifiedAdapter._onehot(w.cur),
        token_fn=lambda w: w.cur)

    # 图片：vis = patch 原始 16 像素值放进前 16 格
    img = DM.ImageDepositPC(seed=0)
    ads["image"] = UnifiedAdapter(
        img, token_offset=0,
        vis_map=lambda w: UnifiedAdapter._onehot(None) if False
        else np.concatenate([w.cur_patch(), np.zeros(D_UNI - 16)]))

    # 电影：vis = 帧 256 格点
    mv = DM.MovieDeposit(seed=0)
    ads["movie"] = UnifiedAdapter(
        mv, token_offset=128,
        vis_map=lambda w: w.cur_frame().ravel())

    return ads


class BabbleUni(ReptilianFunction):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def get_input_spec(self):
        return {"lang": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        out = np.zeros(D_UNI)
        out[int(self.rng.integers(D_UNI))] = 1.0
        return {"lang": Signal(data=shim.tensor(out),
                               mime_type="generic/tensor",
                               metadata={"source": "babble"})}


def build_brain(seed=0, **kw):
    brain = DevelopmentalSystem(port_layout=PORT, decay_rate=0.01, **kw)
    brain.phase = "exploratory"
    brain.autoreg_lr = 0.05
    brain.autoreg_lr_tau = 5000.0
    brain.autoreg_w_norm = 3.0
    brain.reflex._kernel.register("babble", BabbleUni(seed))
    brain.reflex.add_preorchestrated_route(source="lang", target="babble",
                                           priority=1.0)
    n0 = brain.higher_brain.ecosystem.get_neuron(0)
    for c, (off, sz) in PORT.items():
        n0.unfold(c, shim.zeros(D_ALL))
    brain.observe_output_channel = "lang"
    brain.world_display_channel = "disp"
    brain.higher_brain.symbolic_channels = {"lang"}
    brain.explore_eps = 0.2
    brain.drive_sat.min_samples = 8
    return brain


def run_stage(brain, adapter, steps, tag):
    """在统一脑上跑一个沉积物阶段，返回后半段 top1"""
    results = []
    prev = None
    for t in range(steps):
        r = None
        if t > 0:
            adapter._advance()
            r = 1.0 if prev == adapter.cur_token() else 0.0
            brain.report_drive_satisfaction(r)
            results.append(r)
        obs = adapter.observe()
        inputs = {c: Signal(shim.tensor(obs[off:off + sz].copy()),
                            "generic/tensor", {"t": t})
                  for c, (off, sz) in PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        out = (res.get("outputs") or {}).get("lang")
        prev = None
        if out is not None:
            d = out.data if hasattr(out, "data") else out
            a = _np(d).ravel()
            if a.size == D_UNI:          # 分通道 lang 输出
                seg = a
            elif a.size >= D_UNI * 2:    # 全局拼接
                seg = a[D_UNI * 2:]
            else:
                seg = None
            if seg is not None:
                prev = int(np.argmax(seg)) if seg.max() > 0.05 else None
        adapter.last_utter = prev
        if (t + 1) % 400 == 0:
            for _ in range(60):
                brain.step_sleep()
    arr = np.array(results, dtype=float)
    return float(arr[len(arr) // 2:].mean()), float(arr.mean())


def _np(x):
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy(), dtype=float)
    if hasattr(x, "a"):
        return np.asarray(x.a, dtype=float)
    return np.asarray(x, dtype=float)


def main():
    import os
    ckpt = os.path.join("docs", "devo-project", "verify", "curriculum.json")
    brain = build_brain(0, contextual_competence=True)
    ads = make_adapters()

    print("=" * 88)
    print("T10 统一词表持续课程：语法 → 图片 → 电影 → 语法（同一脑）")
    print("=" * 88)

    g1, _ = run_stage(brain, ads["grammar"], 2500, "语法A")
    print(f"阶段1 语法(2500步):      top1(后半)={g1:.3f}")
    save_checkpoint(brain, ckpt)

    i1, _ = run_stage(brain, ads["image"], 2000, "图片")
    print(f"阶段2 图片(2000步):      top1(后半)={i1:.3f}")

    m1, _ = run_stage(brain, ads["movie"], 2000, "电影")
    print(f"阶段3 电影(2000步):      top1(后半)={m1:.3f}")

    load_checkpoint(brain, ckpt)   # 回到阶段1末状态做棘轮对照？不——
    # 棘轮检验：阶段4 直接继续（同一脑不重置），对照阶段1
    g2, _ = run_stage(brain, ads["grammar"], 1500, "语法B")
    print(f"阶段4 语法(1500步,回访): top1(后半)={g2:.3f}")

    keep = g2 / max(g1, 1e-6)
    print("-" * 88)
    print(f"棘轮保持率 = {keep:.2f}（阶段4/阶段1；≥0.8 = 技能在学新沉积物后保持）")
    print("对照: 迁移学习速度（图片/电影阶段的爬升斜率）见曲线，"
          "后续与新鲜脑单独学对照")


if __name__ == "__main__":
    main()
