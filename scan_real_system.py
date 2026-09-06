"""在**真实系统**上做参数敏感性扫描（分化率 × 自噬率）

## 与之前所有实验的关键差别

此前所有验证都在 numpy 原型上做。这份脚本跑的是
``mcp.developmental`` 的**真实代码**——通过 numpy 后端的 torch
兼容层（numpy_torch_shim）实现。

## 补的缺口

报告 §6 缺口 4：分化率与自噬率的**配对上下界**。

- Butz 等 2006（PMID 17014989，已核实）：中等增殖率最优；
  **过高 CP 致网络失稳**，高 CP + 扰动输入 → 细胞存活率下降
- Gohlke 等 2004：神经发生期死亡率过高会**耗尽前体库**

两端各有失效模式 → 需要二维扫描找安全区间。

## 扫描轴

- **分化率**：``novelty_threshold``（低 = 易分化）+ ``saturation_threshold``
- **自噬率**：``decay_rate``（ρ，髓鞘相对衰减）+ ``dim_activity_threshold``

## 判据

1. ``mse``：自回归预测误差（低 = 好）
2. ``neurons`` / ``sheaths``：结构规模（两端失效都体现为异常值）
3. ``alive_frac``：神经元存活率（Butz 说的"细胞存活率下降"）
4. ``turnover``：周转率（myelin 的 born/died，高 = 空转）
"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

sys.path.insert(0, "/data/workspace")

from mcp.developmental.signal import Signal                    # noqa: E402
from mcp.developmental.system import DevelopmentalSystem       # noqa: E402
from mcp.developmental.reptilian import EchoFunction           # noqa: E402

CH = ("vis", "aud", "lang")
DIM = 3
PORT = {c: (i * DIM, DIM) for i, c in enumerate(CH)}
TOTAL = DIM * len(CH)


# ---------------------------------------------------------------------------
# 任务：隐 regime 切换的线性映射 + 噪声信道
# ---------------------------------------------------------------------------

def make_task(n_regimes=3, seed=0, noise_frac=0.34):
    """vis/aud 携带信息，lang 是纯噪声（应被自噬清除）"""
    rng = np.random.default_rng(seed)
    As = [rng.normal(0, 0.8, (DIM, DIM)) for _ in range(n_regimes)]
    for A in As:
        A /= max(1e-9, np.linalg.norm(A, 2)) / 0.9
    return {"A": As, "n": n_regimes, "noise": noise_frac}


def sample(task, t, rng):
    """生成一步的 (x, y)：y = A_k x，lang 为噪声"""
    k = (t // 400) % task["n"]
    A = task["A"][k]
    x_info = rng.normal(0, 1, DIM)
    y = A @ x_info
    x = np.concatenate([x_info + rng.normal(0, 0.05, DIM),
                        x_info + rng.normal(0, 0.05, DIM),
                        rng.normal(0, task["noise"], DIM)])
    return x, y


# ---------------------------------------------------------------------------
# 单次运行
# ---------------------------------------------------------------------------

def run(novelty_threshold=0.7, saturation_threshold=0.9,
        decay_rate=0.01, dim_activity_threshold=0.1,
        steps=1500, seed=0, wiring_block=20):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    brain = DevelopmentalSystem(port_layout=PORT)
    brain.phase = "exploratory"
    brain.wiring_block = wiring_block

    hb = brain.higher_brain
    eng = hb.engine
    eng.novelty_threshold = novelty_threshold
    eng.saturation_threshold = saturation_threshold
    # ⚠ 真实旋钮是 DevelopmentalSystem.decay_rate（system.py 用它调用
    # apply_global_decay），不是 sheath_registry.decay_rate。
    brain.decay_rate = decay_rate
    # autophagy() 的参数不是 engine 属性 → 用 partial 注入
    if dim_activity_threshold is not None:
        _orig_autophagy = eng.autophagy

        def _patched(*a, **k):
            k.setdefault("dim_activity_threshold", dim_activity_threshold)
            return _orig_autophagy(*a, **k)

        eng.autophagy = _patched

    # ⚠ 必须装配反射弧。否则 reflex.execute 返回空 → feedback 为 None
    # → 整条选择链（贡献/用进废退/分化）不启动，progress 恒为 0，
    # 参数扫描会得到"所有格子一样"的假结果。
    brain.reflex._kernel._functions.setdefault(
        "_echo", EchoFunction()) if hasattr(
        brain.reflex._kernel, "_functions") else None
    for c in CH:
        brain.reflex.add_preorchestrated_route(source=c, target="_echo",
                                               priority=1.0)

    # ⚠ 只展开单信道。多信道展开会让 intervene 抛 IndexError
    # （neuron.process 用 signal[active_indices] 索引，信号只有 DIM 长），
    # 而 step_awake 用 except Exception 静默吞掉 → 高层脑从不介入。
    # 已单独取证，见 P0 报告。
    n0 = hb.ecosystem.get_neuron(0)
    n0.unfold("vis", shim.zeros(DIM))

    task = make_task(seed=seed)
    born0 = hb.sheath_registry._born
    died0 = hb.sheath_registry._died

    errs, progs, derrs = [], [], []
    for t in range(steps):
        x, y = sample(task, t, rng)
        inputs = {c: Signal(shim.tensor(x[i * DIM:(i + 1) * DIM].copy()),
                            "generic/tensor", {"t": t})
                  for i, c in enumerate(CH)}
        try:
            res = brain.step_awake(external_inputs=inputs)
        except Exception as e:                      # 崩溃也是一种失效模式
            return {"error": f"{type(e).__name__}: {e}", "mse": np.nan,
                    "neurons": np.nan, "sheaths": np.nan,
                    "alive_frac": np.nan, "turnover": np.nan}

        # 输出端口名为 "y" 不在 PORT 里 → 取任一输出，与 y 比前 DIM 维
        outs = res.get("outputs") or {}
        for _k, sig in outs.items():
            a = np.asarray(sig.data.a if hasattr(sig.data, "a") else sig.data,
                           dtype=float).ravel()
            if a.size >= DIM:
                errs.append(float(np.mean((a[:DIM] - y) ** 2)))
                break
        # 系统自身指标（它的目标就是追平并超越反射，不是我这个合成 y）
        if isinstance(res, dict):
            if "progress" in res:
                progs.append(float(res["progress"]))
            if "distill_error" in res:
                derrs.append(float(res["distill_error"]))

    reg = hb.sheath_registry
    eco = hb.ecosystem
    n_alive = sum(1 for n in eco.neurons.values() if n.alive)
    n_total = max(1, len(eco.neurons))
    tail = errs[len(errs) // 2:] if errs else []
    pt = progs[len(progs) // 2:] if progs else []
    dt = derrs[len(derrs) // 2:] if derrs else []
    return {
        "error": None,
        "mse": float(np.mean(tail)) if tail else np.nan,
        "progress": float(np.mean(pt)) if pt else np.nan,
        "distill": float(np.mean(dt)) if dt else np.nan,
        "neurons": n_total,
        "alive_frac": n_alive / n_total,
        "sheaths": len(reg._sheaths),
        "turnover": (reg._born - born0) + (reg._died - died0),
    }


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------

def scan(axis_a, vals_a, axis_b, vals_b, steps=1500, seeds=(0, 1)):
    """扫描并输出结构健康表格

    每格显示 ``髓鞘数/存活率%``。失效模式：
    - 髓鞘 → 0      = 前体库耗尽（Gohlke 2004）
    - 存活率明显下降 = 高增殖率失稳（Butz 2006）
    - 髓鞘爆炸      = 选择性不足，结构膨胀
    """
    print("=" * 78)
    print(f"二维扫描：{axis_a} × {axis_b}   steps={steps}, seeds={list(seeds)}")
    print("  每格 = 髓鞘数 / 存活率%")
    print("=" * 78)
    print(f"{axis_a:>10} " + " ".join(f"{v:>13.3g}" for v in vals_b))
    print("-" * 78)
    grid = {}
    for va in vals_a:
        cells = []
        for vb in vals_b:
            kw = {axis_a: va, axis_b: vb}
            sh, af, tn, er = [], [], [], None
            for s in seeds:
                r = run(steps=steps, seed=s, **kw)
                if r["error"]:
                    er = r["error"]
                    break
                sh.append(r["sheaths"])
                af.append(r["alive_frac"])
                tn.append(r["turnover"])
            if er:
                cells.append(f"{'崩溃':>13}")
                grid[(va, vb)] = None
            else:
                c = f"{np.mean(sh):.0f}/{np.mean(af)*100:.0f}%"
                cells.append(f"{c:>13}")
                grid[(va, vb)] = (np.mean(sh), np.mean(af), np.mean(tn))
        print(f"{va:>10.3g} " + " ".join(cells))
    print("-" * 78)
    return grid


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--axis", default="both")
    args = ap.parse_args()

    # 冒烟：先看真实系统能否长出结构
    print("【冒烟测试】默认参数下结构是否生长")
    r = run(steps=args.steps, seed=0)
    print(f"  {r}")
    if r["error"]:
        sys.exit(1)
    print()

    if args.axis in ("both", "a"):
        scan("novelty_threshold", [0.3, 0.5, 0.7, 0.9],
             "decay_rate", [0.0, 0.005, 0.02, 0.08],
             steps=args.steps)
        print()
    if args.axis in ("both", "b"):
        scan("saturation_threshold", [0.5, 0.7, 0.9],
             "dim_activity_threshold", [0.02, 0.1, 0.3],
             steps=args.steps)
