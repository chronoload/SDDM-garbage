"""T1 红测试：batched_dispatch 与 loop 路径事件逐位等价 + 性能对比

按 macdev TDD 纪律：先红（batched_dispatch 不存在 → FAIL），
实现 SignalDispatcher.batched_dispatch 后转绿。
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")

import numpy_torch_shim as shim
shim.install()

import baby_grammar as BG                                        # noqa: E402


def build_trained(steps=2500, seed=1):
    """训练出真实边结构的脑（固定全局随机流；步数足够让边数超阈值）"""
    np.random.seed(seed)
    shim.manual_seed(seed)
    world = BG.GrammarWorld(seed)
    brain = BG.build_brain(seed, use_attention=False, use_spikes=False)
    prev = None
    for t in range(steps):
        r = None
        if t > 0:
            world._advance()
            r = 1.0 if prev == world.cur else 0.0
            brain.report_drive_satisfaction(r)
        obs = world.observe()
        obs[BG.LANG_SLICE] = world.observe_lang()
        inputs = {c: BG.Signal(shim.tensor(obs[off:off + sz].copy()),
                               "generic/tensor", {"t": t})
                  for c, (off, sz) in BG.PORT.items()}
        res = brain.step_awake(external_inputs=inputs)
        prev = BG.decode_lang((res.get("outputs") or {}).get("lang"))
        world.last_guess = prev
    return brain


def main():
    brain = build_trained()
    hb = brain.higher_brain
    disp = hb.dispatcher
    disp.batched_min_edges = 0   # 强制走 batched 路径（等价性必须测真路径）
    world = BG.GrammarWorld(0)
    obs = world.observe()
    combined = brain._combine_signals(
        {c: BG.Signal(shim.tensor(obs[off:off + sz].copy()),
                      "generic/tensor", {"t": 0})
         for c, (off, sz) in BG.PORT.items()})
    x = brain.normalizer.normalize(combined.data)
    n_edges = len(disp.connections)
    print(f"边数 = {n_edges}  神经元数 = {len(hb.ecosystem.neurons)}")
    assert n_edges > 0, "需要已长出边的脑"

    # loop 路径
    ev_loop = disp.dispatch(x, "loop_test")
    # batched 路径
    ev_batch = disp.batched_dispatch(x, "batch_test")

    assert len(ev_batch) == len(ev_loop), \
        f"事件数不一致: batched={len(ev_batch)} loop={len(ev_loop)}"

    # 按 sheath_key（每条边唯一）配对——(arrival,target,channel) 有大量
    # 平局（同一目标神经元同信道的多条边），排序配对会错配
    key = lambda e: e.sheath_key
    ev_loop = sorted(ev_loop, key=key)
    ev_batch = sorted(ev_batch, key=key)
    for a, b in zip(ev_loop, ev_batch):
        assert abs(a.arrival_time - b.arrival_time) < 1e-9, \
            f"到达时间不一致: {a.arrival_time} vs {b.arrival_time}"
        assert a.channel == b.channel and a.target_neuron == b.target_neuron
        da = np.asarray(a.data.detach().cpu().numpy()
                        if hasattr(a.data, "detach") else a.data, dtype=float)
        db = np.asarray(b.data.detach().cpu().numpy()
                        if hasattr(b.data, "detach") else b.data, dtype=float)
        assert np.allclose(da, db, rtol=1e-10, atol=1e-12), \
            f"事件数据不一致: max|diff|={np.abs(da - db).max():.3e}"

    # 性能对比（同输入重复分发）
    t0 = time.perf_counter()
    for _ in range(200):
        disp.dispatch(x, "perf")
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(200):
        disp.batched_dispatch(x, "perf")
    t_batch = time.perf_counter() - t0
    print(f"loop 路径: {t_loop * 1000:.1f} ms / 200 次")
    print(f"batched  : {t_batch * 1000:.1f} ms / 200 次  "
          f"(加速 {t_loop / max(t_batch, 1e-9):.1f}x)")
    print("PASS: batched 与 loop 逐位等价")
    # 性能断言仅在大边数下生效（小 E 时 numpy 固定开销占优，自适应回退兜底）
    if n_edges >= 64:
        assert t_batch <= t_loop * 1.1, "大边数下 batched 不应显著慢于 loop"


if __name__ == "__main__":
    main()
