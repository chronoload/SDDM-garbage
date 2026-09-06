"""T9 脑序列化：发育状态的跨 run 持久化（棘轮教义的地基）

发育 = 在已有成果上叠加且不倒退。此前每个实验 build_brain(seed)
从胚胎重爬、跑完即弃——确认投票表、张力张量、情境账本这些最贵的
积累全部归零，棘轮效应无从谈起。

本模块提供 save_checkpoint / load_checkpoint：把脑的全部发育状态
（W / 展开维度 / 连接 / 髓鞘含张力与资格迹 / 确认投票表 / 归一化
统计 / 满足度账本 / 影子情境账本）写入单一 JSON，跨进程复活。

序列化范围（发育状态 = 全部学出来的东西）：
- neurons: W、unfolded（dim/gain/activity/utility/protection/metabolism）
- sheaths: delay/gain/protection/usage/stability/elig/tension/计数
- connections: strength/usage/co_fire_count/myelinated
- registry 计数与账本: born/died/tick/vote_cred/vote_seen/sleep_ledger
- drive_sat: 全局与情境满足度账本
- normalizer: 逐信道 RMS 统计
- system: shadow_ctx / explore_eps / 学习率参数

不序列化：轨迹日志（ episodic，可再生）、RNG 状态（恢复后用显式种子）。
"""
from __future__ import annotations

import json

import numpy as np


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy(), dtype=float)
    if hasattr(x, "a"):
        return np.asarray(x.a, dtype=float)
    return np.asarray(x, dtype=float)


def _tensorize(x):
    """numpy 数组 → 环境张量（shim / 真 torch 均兼容）"""
    if x is None:
        return None
    import torch
    return torch.as_tensor(x)


def save_checkpoint(brain, path: str) -> None:
    """把发育中的脑完整状态写入 JSON"""
    hb = brain.higher_brain
    sys_state = {
        "shadow_mode": brain.shadow_mode,
        "shadow_ctx": brain._shadow_ctx,
        "explore_eps": getattr(brain, "explore_eps", None),
        "autoreg_lr": brain.autoreg_lr,
        "autoreg_w_norm": brain.autoreg_w_norm,
        "step_count": brain.step_count,
        "last_winner": getattr(brain, "_last_winner", None),
    }
    neurons = {}
    for nid, n in hb.ecosystem.neurons.items():
        if n.W is None:
            continue
        neurons[str(nid)] = {
            "W": _np(n.W).tolist(),
            "alive": bool(n.alive),
            "metabolism": float(getattr(n, "metabolism", 0.5)),
            "drift_count": int(getattr(n, "drift_count", 0)),
            "unfolded": {
                ch: {
                    "dim_idx": int(slc.dim_idx), "size": int(slc.size),
                    "gain": float(slc.gain),
                    "activity": float(getattr(slc, "activity", 0.0)),
                    "utility": float(getattr(slc, "utility", 0.0)),
                    "protection": float(getattr(slc, "protection", 0.0)),
                    "age": int(getattr(slc, "age", 0)),
                    "evaluated": int(getattr(slc, "evaluated", 0)),
                } for ch, slc in n.unfolded.items()
            },
        }
    sheaths = {
        "|".join(map(str, k)): {
            "delay": float(s.delay), "gain": float(s.gain),
            "protection": float(s.protection),
            "usage": float(s.usage), "stability": float(s.stability),
            "elig": float(s.elig),
            "tension": (None if s.tension is None
                        else _np(s.tension).tolist()),
            "starving": int(s.starving),
            "since_co_fire": int(s.since_co_fire),
            "age_steps": int(s.age_steps),
            "sleep_gain_delta": float(s.sleep_gain_delta),
        } for k, s in hb.sheath_registry._sheaths.items()
    }
    connections = {
        "|".join(map(str, k)): {
            "strength": float(c.strength), "usage": float(c.usage),
            "co_fire_count": int(c.co_fire_count),
            "starving": int(c.starving),
            "myelinated": bool(c.myelinated),
        } for k, c in hb.sheath_registry._connections.items()
    }
    ds = brain.drive_sat
    payload = {
        "port_layout": {k: list(v) for k, v in brain.port_layout.items()},
        "system": sys_state,
        "neurons": neurons,
        "sheaths": sheaths,
        "connections": connections,
        "registry": {
            "born": hb.sheath_registry._born,
            "died": hb.sheath_registry._died,
            "tick": getattr(hb.dispatcher, "_tick", 0),
            "vote_cred": {f"{k[0][0]}|{k[0][1]}|{k[1]}": v
                          for k, v in hb.vote_cred.items()},
            "vote_seen": {f"{k[0][0]}|{k[0][1]}|{k[1]}": v
                          for k, v in hb.vote_seen.items()},
            "sleep_ledger": hb.sheath_registry.sleep_ledger,
            "elig_decay_default": 0.9,
        },
        "drive_sat": {
            "sat_reflex": ds._sat_reflex_ema,
            "sat_high": ds._sat_high_ema,
            "n_reflex": ds._n_reflex,
            "n_high": ds._n_high,
            "ctx": {str(k): v for k, v in ds._ctx.items()},
        },
        "normalizer": _norm_state(hb.normalizer),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _norm_state(n) -> dict:
    """normalizer 状态导出（张量 → list，JSON 兼容）"""
    if not hasattr(n, "state_dict"):
        return {}
    out = {}
    for k, v in n.state_dict().items():
        if v is None:
            out[k] = None
        elif hasattr(v, "detach") or hasattr(v, "a"):
            out[k] = _np(v).tolist()
        else:
            out[k] = v
    return out


def load_checkpoint(brain, path: str) -> None:
    """把 JSON 状态恢复进一个**新构造**的同构脑（原地覆盖）"""
    hb = brain.higher_brain
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)

    import numpy_torch_shim  # noqa: F401  确保 shim 已安装时 torch 可用

    # --- 神经元：W 与展开维度（缺失的神经元新建）---
    from mcp.developmental.neuron import Neuron, DimSlice

    eco = hb.ecosystem
    for sid, nd in p["neurons"].items():
        nid = int(sid)
        n = eco.get_neuron(nid)
        if n is None:
            n = Neuron(seed=nid, port_layout=brain.port_layout)
            eco.neurons[nid] = n
            eco._next_idx = max(getattr(eco, "_next_idx", 0), nid + 1)
        n.W = _tensorize(np.asarray(nd["W"], dtype=float))
        n.alive = nd["alive"]
        n.metabolism = nd["metabolism"]
        n.drift_count = nd["drift_count"]
        for ch, ud in nd["unfolded"].items():
            slc = n.unfolded.get(ch)
            if slc is None:
                n.unfolded[ch] = DimSlice(dim_idx=ud["dim_idx"],
                                          size=ud["size"])
                slc = n.unfolded[ch]
            slc.dim_idx = ud["dim_idx"]
            slc.size = ud["size"]
            slc.gain = ud["gain"]
            slc.activity = ud["activity"]
            slc.utility = ud["utility"]
            slc.protection = ud["protection"]
            slc.age = ud["age"]
            slc.evaluated = ud["evaluated"]

    # --- 髓鞘与连接（缺失的新建）---
    from mcp.developmental.myelin import Connection, MyelinSheath

    def parse_key(s):
        sn, sc, dn, dc = s.split("|")
        return int(sn), sc, int(dn), dc

    for k, sd in p["sheaths"].items():
        pk = parse_key(k)
        s = hb.sheath_registry._sheaths.get(pk)
        if s is None:
            s = MyelinSheath(src_neuron=pk[0], src_channel=pk[1],
                             dst_neuron=pk[2], dst_channel=pk[3])
            hb.sheath_registry._sheaths[pk] = s
        s.delay, s.gain = sd["delay"], sd["gain"]
        s.protection, s.usage = sd["protection"], sd["usage"]
        s.stability, s.elig = sd["stability"], sd["elig"]
        s.starving, s.since_co_fire = sd["starving"], sd["since_co_fire"]
        s.age_steps = sd["age_steps"]
        s.sleep_gain_delta = sd["sleep_gain_delta"]
        s.tension = (None if sd["tension"] is None
                     else np.asarray(sd["tension"], dtype=float))
    for k, cd in p["connections"].items():
        pk = parse_key(k)
        c = hb.sheath_registry._connections.get(pk)
        if c is None:
            c = Connection(*pk)
            hb.sheath_registry._connections[pk] = c
        c.strength, c.usage = cd["strength"], cd["usage"]
        c.co_fire_count, c.starving = cd["co_fire_count"], cd["starving"]
        c.myelinated = cd["myelinated"]

    # --- registry 账本 ---
    r = p["registry"]
    hb.sheath_registry._born = r["born"]
    hb.sheath_registry._died = r["died"]
    hb.dispatcher._tick = r["tick"]
    hb.vote_cred.clear()
    for k, v in r["vote_cred"].items():
        sn, sc, tok = k.split("|")
        hb.vote_cred[((int(sn), sc), int(tok))] = v
    hb.vote_seen.clear()
    for k, v in r["vote_seen"].items():
        sn, sc, tok = k.split("|")
        hb.vote_seen[((int(sn), sc), int(tok))] = v
    hb.sheath_registry.sleep_ledger = r.get("sleep_ledger", {})

    # --- 满足度账本 ---
    ds = brain.drive_sat
    d = p["drive_sat"]
    ds._sat_reflex_ema = d["sat_reflex"]
    ds._sat_high_ema = d["sat_high"]
    ds._n_reflex, ds._n_high = d["n_reflex"], d["n_high"]
    ds._ctx = {int(k): v for k, v in d.get("ctx", {}).items()}

    # --- 影子情境账本与系统参数 ---
    brain._shadow_ctx = {int(k): v for k, v
                         in p["system"].get("shadow_ctx", {}).items()}
    brain.shadow_mode = p["system"]["shadow_mode"]
    brain.step_count = p["system"].get("step_count", 0)
    if p["system"].get("explore_eps") is not None:
        brain.explore_eps = p["system"]["explore_eps"]

    # --- 归一化统计（native state_dict；rms 列表 → 张量）---
    nd = p.get("normalizer") or {}
    if nd and hasattr(hb.normalizer, "load_state_dict"):
        try:
            if nd.get("rms") is not None:
                nd = dict(nd)
                nd["rms"] = _tensorize(np.asarray(nd["rms"], dtype=float))
            hb.normalizer.load_state_dict(nd)
        except Exception:
            pass
