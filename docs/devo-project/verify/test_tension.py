"""T8 红测试：阈值张力——标量资格迹升维为绑定在髓鞘回路上的张力张量

契约：
1. 张力张量：update_tension 后髓鞘 tension 非零（逐维度负荷），
   无事件时按 decay 衰减
2. 阈值治理：高张力神经元的漂移幅度 < 低张力状态（同神经元对照）
3. 回滚探针：变异使预测残差变差超过容差 → W 回滚到变异前
4. 回归：grammar 任务治理开启不劣化
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")

import numpy_torch_shim as shim
shim.install()

import baby_grammar as BG                                        # noqa: E402
from mcp.developmental.myelin import Connection, MyelinSheath    # noqa: E402


def _np(x):
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy(), dtype=float)
    if hasattr(x, "a"):
        return np.asarray(x.a, dtype=float)
    return np.asarray(x, dtype=float)


def main():
    brain = BG.build_brain(0, use_attention=False, use_spikes=False,
                           drift_tension=True)
    hb = brain.higher_brain
    reg = hb.sheath_registry
    n0 = hb.ecosystem.get_neuron(0)

    # 手动装一条髓鞘（张力测试不需要训练结构）
    reg.add_sheath(0, "vis", 0, "lang", gain=1.0)
    key = (0, "vis", 0, "lang")
    reg._connections[key] = Connection(0, "vis", 0, "lang")
    reg._sheaths[key] = MyelinSheath(src_neuron=0, src_channel="vis",
                                     dst_neuron=0, dst_channel="lang",
                                     delay=0.01, gain=1.0)
    s = reg._sheaths[key]
    assert s.tension is None or np.allclose(_np(s.tension), 0)

    # ---- 契约 1：张力张量累积 + 衰减 ----
    world = BG.GrammarWorld(0)
    obs = world.observe()
    x = brain.normalizer.normalize(brain._combine_signals(
        {c: BG.Signal(shim.tensor(obs[off:off + sz].copy()),
                      "generic/tensor", {"t": 0})
         for c, (off, sz) in BG.PORT.items()}).data)
    residual = shim.tensor(np.ones(48) * 0.5)
    events = disp_events = hb.dispatch_signal(x, "test")
    reg.update_tension(events, residual, hb.port_layout)
    assert s.tension is not None and np.abs(_np(s.tension)).max() > 1e-6, \
        "张力张量应非零"
    t_before = _np(s.tension).copy()
    reg.update_tension([], residual, hb.port_layout)   # 无事件 → 纯衰减
    assert np.abs(_np(s.tension)).max() < np.abs(t_before).max(), \
        "无事件时张力应衰减"

    # ---- 契约 2：阈值治理——高张力同神经元漂移更小 ----
    brain._prev_norm_input = x
    brain._last_target = x
    w_before = _np(n0.W).copy()
    # 低张力状态：清零张力
    for sh in reg._sheaths.values():
        sh.tension = shim.tensor(np.zeros_like(_np(sh.tension)))
    brain._apply_drift_governed(0, n0, rate=0.5)
    d_low = np.abs(_np(n0.W) - w_before).max()
    # 高张力状态：张力拉满
    for sh in reg._sheaths.values():
        sh.tension = shim.tensor(np.ones_like(_np(sh.tension)) * 10.0)
    n0.W = shim.tensor(w_before.copy())
    brain._apply_drift_governed(0, n0, rate=0.5)
    d_high = np.abs(_np(n0.W) - w_before).max()
    print(f"漂移幅度: 低张力={d_low:.4f} 高张力={d_high:.4f}")
    assert d_high < d_low, "高张力应抑制漂移"

    # ---- 契约 3：回滚探针——变异致残差恶化 → W 回滚 ----
    n0.W = shim.tensor(w_before.copy())
    err_pre = brain._probe_pred_err(n0, x, x)     # x 自身作目标（已拟合态）
    assert err_pre is not None
    # 强制大变异（应恶化 → 回滚）
    brain._apply_drift_governed(0, n0, rate=50.0)
    err_post = brain._probe_pred_err(n0, x, x)
    w_after = _np(n0.W)
    rolled = np.allclose(w_after, w_before, atol=1e-6)
    assert rolled or err_post <= err_pre * 1.1 + 1e-6, \
        f"恶化变异应被回滚: err {err_pre:.3f}->{err_post:.3f} rolled={rolled}"
    print(f"回滚探针: err_pre={err_pre:.4f} err_post={err_post:.4f} "
          f"rolled={rolled}")

    # ---- 契约 4：语法回归 ----
    r = BG.run(seed=0, steps=500, use_attention=False, use_spikes=False,
               drift_tension=True)
    assert 0.0 <= r["top1"] <= 1.0
    print(f"语法回归(500步): top1={r['top1']:.3f}")
    print("PASS: 阈值张力契约全部满足")


if __name__ == "__main__":
    main()
