"""验证 align_fn 接线：probe_delay 是否真的拿到了对齐度

## 为什么需要这个测试

``system.py`` 里 ``maintain_ages(align_fn=..., rng=...)`` 已接线，
但**接线正确 ≠ 数据流通**。可能仍然空转的情形：

- 历史缓冲没被喂（push_history 没调用 / 容量不足）
- 源信号的 key 与目标信号的 key 对不上
- 形状不匹配 → 每次都 bump("shape")
- delay 超出缓冲范围 → bump("out_of_range")

判据是 :attr:`SignalDispatcher.align_stats`：
**ok > 0 才说明爬山真的拿到了方向**。

## 方法

沙盒无 torch，注入 numpy 后端的假 Tensor，然后用最小的假神经元构造
一个真实可跑的 dispatcher，走完整链路：
dispatch → resolve_triggers → maintain_ages
"""
import sys
import types
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 注入假 torch
# ---------------------------------------------------------------------------

class T:
    """numpy 后端的极简 Tensor"""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    @property
    def shape(self):
        return self.a.shape

    def __mul__(self, o):
        return T(self.a * (o.a if isinstance(o, T) else o))

    def __rmul__(self, o):
        return T(self.a * o)

    def __add__(self, o):
        return T(self.a + (o.a if isinstance(o, T) else o))

    def __iadd__(self, o):
        return T(self.a + (o.a if isinstance(o, T) else o))

    def norm(self):
        return T(np.array(float(np.linalg.norm(self.a))))

    def sum(self):
        return T(np.array(float(self.a.sum())))

    def item(self):
        return float(self.a)

    def clone(self):
        return T(self.a.copy())

    def detach(self):
        return T(self.a.copy())


fake = types.ModuleType("torch")
fake.Tensor = T
fake.zeros = lambda *a, **k: T(np.zeros(a[0] if a else k.get("shape", 1)))
fake.randn = lambda *a, **k: T(np.random.default_rng(0).normal(size=a[0]))
fake.Generator = lambda *a, **k: None
fake.no_grad = lambda: None
sys.modules["torch"] = fake

# 直接按文件路径加载 myelin，绕过包的 __init__（它依赖 torch 子包）
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "myelin_mod", "/data/workspace/mcp/developmental/myelin.py")
_myelin = importlib.util.module_from_spec(_spec)
sys.modules["myelin_mod"] = _myelin
_spec.loader.exec_module(_myelin)

MyelinSheath = _myelin.MyelinSheath
MyelinSheathRegistry = _myelin.MyelinSheathRegistry
SignalDispatcher = _myelin.SignalDispatcher

# ---------------------------------------------------------------------------
# 假神经元
# ---------------------------------------------------------------------------

@dataclass
class FakeNeuron:
    """最小可用神经元：process 返回一个随步数缓慢旋转的向量"""
    nid: int
    chans: tuple = ("v", "a")
    dim: int = 4
    alive: bool = True
    W: object = None
    unfolded: tuple = field(default_factory=tuple)
    _k: int = 0

    def __post_init__(self):
        if self.unfolded == ():
            self.unfolded = self.chans
        # 真实 Neuron 的 W 是张量；dispatch 用 `W is None` 跳过未初始化者
        self.W = object()
        rng = np.random.default_rng(self.nid)
        self._phase = rng.uniform(0, 2 * np.pi)

    def process(self, signal):
        self._k += 1
        # 缓慢旋转的正弦向量：保证历史值随步数变化，对齐度有区分度
        ph = self._phase + 0.05 * self._k
        return T(np.sin(ph + np.arange(self.dim) * 0.7))


def build(n_neurons=6, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    neurons = {i: FakeNeuron(nid=i, dim=dim) for i in range(n_neurons)}

    sheaths = {}
    # 两类通路：
    #  - 绑定成功者（gain 高）：adapt_delay 每步重置 since_co_fire
    #    → 永远不触发探测。这是**正确行为**，只有失败者才需要修复。
    #  - 绑定失败者（gain 低于 threshold）：since_co_fire 累积
    #    → 触发 needs_delay_probe → 走 probe_delay
    for src in range(n_neurons):
        for k in range(2):
            dst = (src + k + 1) % n_neurons
            key = (src, "v", dst, "a")
            sheaths[key] = MyelinSheath(
                src, "v", dst, "a",
                delay=float(rng.integers(1, 8)),
                gain=0.3 if k == 0 else 0.001,   # k=1 → 绑定失败者
                protection=0.0,
            )
    return neurons, sheaths


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_once(use_align=True, steps=600, block=20, seed=0, capacity=64):
    neurons, sheaths = build(seed=seed)
    registry = MyelinSheathRegistry()
    registry._sheaths = sheaths
    disp = SignalDispatcher(neurons, sheaths, history_capacity=capacity)

    probed_total = 0
    for t in range(steps):
        sig = T(np.ones(4))
        events = disp.dispatch(sig, source_tag="s", t=float(t))
        disp.resolve_triggers(threshold=0.1, coincidence_window=2.0)
        disp.adapt_delays(lr=0.1)
        registry.tick_ages()          # 推进 since_co_fire / age_steps

        if t % block == 0:
            window = registry.suggest_window()
            fn = disp.make_align_fn(window) if use_align else None
            r = registry.maintain_ages(align_fn=fn, rng=None)
            probed_total += r["probed"]

    return disp, registry, probed_total


def run_instrumented(seed=0, steps=600, block=20):
    """记录每次探测的 (对齐度三元组, 实际位移)，验证爬山方向正确"""
    _orig = MyelinSheath.probe_delay
    log = []

    def patched(self, window, align=None, rng=None):
        before = self.delay
        _orig(self, window, align=align, rng=rng)
        log.append((align, self.delay - before, window, before))

    MyelinSheath.probe_delay = patched
    try:
        run_once(use_align=True, steps=steps, block=block, seed=seed)
    finally:
        MyelinSheath.probe_delay = _orig
    return log


def check_direction(log):
    """爬山方向是否与对齐度 argmax 一致"""
    ok = bad = tie = clipped = 0
    for align, delta, w, before in log:
        if align is None:
            continue
        m, now, p = align
        best = max(m, now, p)
        winners = [i for i, v in enumerate([m, now, p]) if v == best]
        if len(winners) > 1:                    # 平局：不动是合法的
            tie += 1
            continue
        expect = {0: -w, 1: 0.0, 2: w}[winners[0]]
        if abs(delta - expect) < 1e-6:
            ok += 1
        elif before + expect < MyelinSheath.DELAY_MIN - 1e-9 or \
                before + expect > MyelinSheath.DELAY_MAX + 1e-9:
            # 期望位移会越界，被 probe_delay 的 clamp 截断 —— 正常行为
            clipped += 1
        else:
            bad += 1
    return ok, bad, tie, clipped


if __name__ == "__main__":
    print("=" * 78)
    print("align_fn 接线验证：probe_delay 是否真的拿到对齐度")
    print("=" * 78)

    disp, reg, probed = run_once(use_align=True)
    st = disp.align_stats
    total = sum(st.values())

    print(f"\n探测触发次数 {probed}（若无探测发生，下面的比例无意义）\n")
    print(f"{'结果':>14} {'次数':>8} {'占比':>8}")
    print("-" * 34)
    for k in ["ok", "no_sheath", "no_target", "out_of_range",
              "no_source", "shape"]:
        c = st.get(k, 0)
        print(f"{k:>14} {c:>8} {c / max(1, total):>8.1%}")

    print()
    print("=" * 78)
    if total == 0:
        print("⚠ 对齐度从未被请求 —— 没有通路进入 needs_delay_probe。")
        print("  无法判断接线是否成功，需要跑更久或调低 AGE_PROBE。")
    elif st.get("ok", 0) > 0:
        frac = st["ok"] / total
        print(f"✓ 接线成功：{st['ok']}/{total} 次对齐度计算成功 "
              f"({frac:.0%})，probe_delay 走爬山。")
        if frac < 0.5:
            print(f"  ⚠ 但成功率偏低（{frac:.0%}），失败主因：")
            worst = max((k for k in st if k != "ok"), key=lambda k: st[k])
            print(f"    {worst} = {st[worst]} 次")
            print("  建议：查 align_stats 里计数最高的失败原因。")
    else:
        print("✗ 接线失败：对齐度计算 0 次成功 —— probe_delay 仍在随机扰动。")
        worst = max((k for k in st if k != "ok"), key=lambda k: st[k])
        print(f"  失败主因：{worst} = {st[worst]} 次")
        print("  修复方向：")
        hints = {
            "no_target": "record_target 没被调用 / 目标 key 不匹配",
            "no_source": "record_source 的 key 与 sheath_key 的 (src,ch) 不一致",
            "out_of_range": "history_capacity 太小，或 delay 超过了已积累的步数",
            "shape": "源信号与目标信号的形状不同（信道切片问题）",
            "no_sheath": "sheath_key 在 registry 中不存在（dict 引用不同步）",
        }
        print(f"    → {hints.get(worst, '未知')}")
    print("=" * 78)

    # ---------- 方向正确性 ----------
    print()
    print("=" * 78)
    print("爬山方向正确性：位移是否等于对齐度 argmax 对应的方向")
    print("=" * 78)
    tot_ok = tot_bad = tot_tie = tot_clip = 0
    for seed in range(3):
        log = run_instrumented(seed=seed)
        ok, bad, tie, clip = check_direction(log)
        tot_ok += ok; tot_bad += bad; tot_tie += tie; tot_clip += clip
        print(f"  seed {seed}: 一致 {ok:>3}  边界截断 {clip:>3}  "
              f"不一致 {bad:>3}  平局(不动) {tie:>3}")
    print("-" * 78)
    print(f"  合计: 一致 {tot_ok}  边界截断 {tot_clip}  "
          f"不一致 {tot_bad}  平局 {tot_tie}")
    if tot_bad == 0 and tot_ok > 0:
        print("  ✓ 爬山方向始终与对齐度一致 —— 端到端正确")
    elif tot_bad > 0:
        print(f"  ✗ 有 {tot_bad} 次方向不一致，probe_delay 的比较逻辑有问题")
    else:
        print("  ⚠ 无有效样本")
    print("=" * 78)

    # 对照：不传 align_fn 时应全部走随机
    disp2, reg2, p2 = run_once(use_align=False)
    print(f"\n对照（不传 align_fn）：align_stats 全零 = "
          f"{sum(disp2.align_stats.values()) == 0}")
    print("  （确认 align_stats 只在有 align_fn 时才有计数）")
