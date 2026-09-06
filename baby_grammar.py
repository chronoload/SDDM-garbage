"""超模态语法习得：FSM 符号流的自回归补全（LLM 式 next-token 任务）

## 从 baby 到 grammar：换肢体不换架构

baby_loop 证明了系统能学会"发声 → 世界响应"的控制 loop。
本实验把任务升级为**自回归序列预测**——LLM 的训练目标，用来回答：

    同一个发育架构，面对更复杂的超模态信号（带语法的符号流），
    还能不能学？学到的结构是不是语法本身？

## 面向原生 AGI 的接口主张

爬虫脑 = 任何可自定义捕捉的反馈回路，它**天生定义肢体的行动面**。
本任务的行动面 = "发声器官"（每帧产出 1/16 个 token）；
换掉肢体与信号编码（工厂调度 = 输出排程动作；显微镜 = 输出旋钮增量），
架构其余部分不动。只要任务能被一组简单反射 + 肢体复现，就该能学会。

## 任务协议（每帧）

    ─────────────────────────────────────────────────────
    帧 t：
      vis/aud : 世界正在"说"的 token（两个模态、不同随机嵌入）
      lang    : 系统自己上一帧发出的声音（听得到自己说话）
      输出     : 系统对**下一帧** token 的猜测（行动）
    帧 t+1 开始时结算：
      猜中 → 驱动满足度 1.0（三因子 RPE 回流，配对 t-1 的 eligibility）
      猜错 → 0.0
    ─────────────────────────────────────────────────────

与 baby_loop 的反应任务不同：这里的世界不因系统的输出而改变
（语法流自主演进）——语言理解本来就是预测一个你不控制的世界。
但**满足度仍然绑定在系统自己的行动上**：猜是行动，对错由身体裁定。
因果链：猜测(行动) → 下一帧揭示 → RPE → 三因子巩固。

## 语法

带种子的随机有限状态机：n_states 个状态，每状态 1~3 条出边
（出边 = (发出的 token, 下一状态)）。token 在多个状态间共享
（歧义性），序列有真实的一阶结构，bigram 可大部分捕获——
因此 bigram 是诚实的"统计学习器"参照（LLM-lite）。

## 超模态编码（裸编码教义）

- lang：one-hot（符号离散互斥，无内在距离；argmax 解码，不用余弦）
- vis/aud：同为 one-hot，**实验者不预设任何距离结构**
  → 三个模态之间没有馈赠的对应关系：跨模态绑定（vis_i ↔ aud_i ↔ lang_i）
  是系统要用共发射成边自己长出来的东西。
  ⚠ 勿把 vis/aud 换成随机嵌入——那是"嵌套预编码"陷阱：环境代劳了
  系统该发育的编码，且污染实验归因（此前踩过）。

## 对照组

- uniform  : 1/16（盲目猜测）
- echo     : 复读上一帧自己的话（先天模仿反射的裸输出）
- bigram   : 全流统计的 top-1（"LLM-lite"参照，不可学习、只做标尺）

## 判据

系统 top-1 显著 > uniform/echo，且接近 bigram → 架构对复杂
符号流有效。学习曲线按段打印，结构指标（神经元/髓鞘/周转）并行。
"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

from mcp.developmental.signal import Signal                      # noqa: E402
from mcp.developmental.system import DevelopmentalSystem         # noqa: E402
from mcp.developmental.reptilian import ReptilianFunction        # noqa: E402

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
N_TOKEN = 16
N_STATES = 6
MAX_OUT = 3            # 每状态最多出边数（1~3，含必有一条环边）
# ⚠ 裸编码教义（勿惯性采用嵌入）：所有模态都是 one-hot 裸编码，
# 实验者**不预设任何距离结构**。跨模态绑定是系统要自己长出来的
# 东西（共发射成边），不是环境馈赠的。此前的 6 维随机嵌入是
# "嵌套预编码"陷阱：把人工相似度结构嵌进信号，既代劳了系统该
# 发育的编码，又污染实验归因。
D_VIS, D_AUD, D_LANG = N_TOKEN, N_TOKEN, N_TOKEN
EMB_NOISE = 0.05       # 呈现噪声（传感器噪声，非编码）
STEPS = 4000
SLEEP_EVERY = 300      # 每 300 觉醒步
SLEEP_LEN = 60         # 睡 60 步
SEEDS = (0, 1, 2)

# 通道偏移
_OFF, _o = {}, 0
for _c, _d in (("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG)):
    _OFF[_c] = _o
    _o += _d
D_ALL = _o
PORT = {c: (_OFF[c], d) for c, d in
        (("vis", D_VIS), ("aud", D_AUD), ("lang", D_LANG))}
LANG_SLICE = slice(_OFF["lang"], _OFF["lang"] + D_LANG)


# ---------------------------------------------------------------------------
# 语法世界
# ---------------------------------------------------------------------------

def build_grammar(seed, n_states=N_STATES, n_tokens=N_TOKEN, max_out=MAX_OUT):
    """随机 FSM：trans[s] = [(token, next_state), ...]

    先建一条环边保证连通，再随机加 0~(max_out-1) 条出边制造歧义。
    """
    rng = np.random.default_rng(seed)
    trans = {}
    for s in range(n_states):
        nxt = (s + 1) % n_states
        trans[s] = [(int(rng.integers(n_tokens)), nxt)]
        for _ in range(int(rng.integers(0, max_out))):
            trans[s].append((int(rng.integers(n_tokens)),
                             int(rng.integers(n_states))))
    return trans


class GrammarWorld:
    """语法流世界：世界自主演进，系统的猜测由下一帧的揭示来裁定"""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed + 999)
        self.grammar = build_grammar(seed)
        # 裸编码：各模态 = 单位矩阵（one-hot），无预设距离结构。
        # 跨模态对应关系（vis_i ≈ aud_i ≈ lang_i）留给系统自己发现。
        eye = np.eye(N_TOKEN)
        self.vis_mat = eye
        self.aud_mat = eye
        self.state = 0
        self.cur = None            # 当前帧世界正在展示的 token
        self.stream = []           # 实际 token 流（对照基线用）
        self._advance()

    def _advance(self):
        edges = self.grammar[self.state]
        tok, nxt = edges[int(self.rng.integers(len(edges)))]
        self.cur = tok
        self.state = nxt
        self.stream.append(tok)

    def observe(self):
        """当前观测帧：vis/aud = 当前 token 的嵌入；lang = 自己上一帧的话"""
        v = np.zeros(D_ALL)
        v[_OFF["vis"]:_OFF["vis"] + D_VIS] = (
            self.vis_mat[self.cur] + EMB_NOISE * self.rng.normal(size=D_VIS))
        v[_OFF["aud"]:_OFF["aud"] + D_AUD] = (
            self.aud_mat[self.cur] + EMB_NOISE * self.rng.normal(size=D_AUD))
        return v

    def show_utterance(self, guess_idx):
        """系统听得到自己上一帧说的话（写入观测帧的 lang 段）"""
        self.last_guess = guess_idx

    last_guess = None

    def observe_lang(self):
        v = np.zeros(D_LANG)
        if self.last_guess is not None:
            v[self.last_guess] = 1.0
        return v


# ---------------------------------------------------------------------------
# 爬虫脑：先天反射 = 肢体的行动面（可替换，架构其余不动）
# ---------------------------------------------------------------------------

class EchoReflex(ReptilianFunction):
    """模仿反射：复读自己上一帧的话（先天、无学习）

    裸 echo 的命中率 = P(next == prev)，是"先天肢体"的天然基线。
    """

    def get_input_spec(self):
        return {"lang": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        out = np.zeros(D_LANG)
        d = inputs["lang"].data
        a = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
        a = np.asarray(a, dtype=float).ravel()
        if a.size == D_LANG and a.max() > 0.5:
            out[int(np.argmax(a))] = 1.0
        return {"lang": Signal(data=shim.tensor(out),
                               mime_type="generic/tensor",
                               metadata={"source": "echo"})}


class BabbleReflex(ReptilianFunction):
    """哭闹反射：随机发声（先天、无学习、不含因果知识）"""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def get_input_spec(self):
        return {"lang": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        out = np.zeros(D_LANG)
        out[int(self.rng.integers(N_TOKEN))] = 1.0
        return {"lang": Signal(data=shim.tensor(out),
                               mime_type="generic/tensor",
                               metadata={"source": "babble"})}


def build_brain(seed=0, decay_rate=0.01, autoreg_lr=0.03, autoreg_w_norm=None,
                **kw):
    """**kw 穿透到 DevelopmentalSystem（消融用：use_attention / use_spikes）"""
    brain = DevelopmentalSystem(port_layout=PORT, decay_rate=decay_rate, **kw)
    brain.phase = "exploratory"
    brain.autoreg_lr = autoreg_lr
    if autoreg_w_norm is not None:
        brain.autoreg_w_norm = autoreg_w_norm
    # 学习率衰减时间常数放慢：逻辑表（确定性转移）需要足够步数长进 W，
    # tau=500 时后期 lr 趋零，罕见 token 的行来不及学会（实测 9/16）。
    brain.autoreg_lr_tau = 5000.0
    k = brain.reflex._kernel
    k.register("babble", BabbleReflex(seed))
    k.register("echo", EchoReflex())
    # 路由按优先级降序执行，后执行的覆盖先执行的。
    # ⚠ 输出面 = babble（随机发声）。echo 做输出面是实验设计的退化
    # 吸引子（实测）：echo 回读自己的话 → 陈旧常量驻留 lang 通道 →
    # W 在线输入分布与训练分布错位 → 在线预测被污染，永远到不了
    # 探针测出的质量。baby_loop 的教义：反射提供"量级合理的起点、
    # 不含因果知识"——随机 babble 才是这个角色；echo 保留注册
    # （模仿肢体仍在反射库里），但不主导输出面。
    brain.reflex.add_preorchestrated_route(source="lang", target="echo",
                                           priority=1.0)
    brain.reflex.add_preorchestrated_route(source="lang", target="babble",
                                           priority=0.5)
    # 种子神经元展开全部信道（全息起点）
    n0 = brain.higher_brain.ecosystem.get_neuron(0)
    n0.unfold("vis", shim.zeros(D_ALL))
    n0.unfold("aud", shim.zeros(D_ALL))
    n0.unfold("lang", shim.zeros(D_ALL))
    # 源可信度门控的两个通道锚点：
    #   observe_output_channel = "lang"  → 发言信道（预测 payload 备案处）
    #   world_display_channel  = "vis"   → 世界显示信道（结算时读真实 token）
    brain.observe_output_channel = "lang"
    brain.world_display_channel = "vis"
    # lang 是符号信道：读出走确认投票（argmax 有离散语义）
    brain.higher_brain.symbolic_channels = {"lang"}
    return brain


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def decode_lang(vec):
    """裸编码解码：argmax（符号识别是离散判定）"""
    if vec is None:
        return None
    d = vec.data if isinstance(vec, Signal) else vec
    if hasattr(d, "detach"):
        d = d.detach().cpu().numpy()
    elif hasattr(d, "a"):
        d = d.a
    a = np.asarray(d, dtype=float).ravel()
    # 兼容两种尺寸：输出是分通道的（D_LANG），或全局拼接（D_ALL）
    if a.size == D_LANG:
        seg = a
    elif a.size >= _OFF["lang"] + D_LANG:
        seg = a[LANG_SLICE]
    else:
        return None
    return int(np.argmax(seg)) if seg.max() > 0.05 else None


def run(seed=0, steps=STEPS, sleep_every=SLEEP_EVERY, sleep_len=SLEEP_LEN,
        verbose=False, **kw):
    np.random.seed(seed)
    shim.manual_seed(seed)
    world = GrammarWorld(seed)
    brain = build_brain(seed, **kw)
    hb = brain.higher_brain

    results = []           # (step, correct) 逐帧猜测
    lambdas, controllers = [], []
    sleep_reports = []

    prev_guess = None
    for t in range(steps):
        # 结算上一帧的猜测。
        # ⚠ 顺序必须 先 advance 后裁定：guess 在看到 token_t 后输出，
        # 预测的是 token_{t+1}；先把世界翻到 token_{t+1}，再比较。
        # 旧顺序（先裁定后 advance）把猜测拿去和**已经看过的**
        # token_t 比——判据退化成"复述当前帧"，探索永远 miss
        # （实测 0/74，而 payload 结算命中率 40%+，两套裁定矛盾
        #  才暴露出这个 off-by-one）。
        r = None
        if t > 0:
            world._advance()          # 世界翻到被预测的那一帧
            r = 1.0 if prev_guess == world.cur else 0.0
            brain.report_drive_satisfaction(r)
            results.append(r)

        obs = world.observe()
        obs[LANG_SLICE] = world.observe_lang()   # 听到自己上一帧的话
        inputs = {}
        for c, (off, sz) in PORT.items():
            inputs[c] = Signal(shim.tensor(obs[off:off + sz].copy()),
                               "generic/tensor", {"t": t})

        res = brain.step_awake(external_inputs=inputs)
        out = (res.get("outputs") or {}).get("lang")
        prev_guess = decode_lang(out)
        world.last_guess = prev_guess
        lambdas.append(res.get("lambda", 0.0))
        controllers.append(brain._last_controller)

        # 觉醒/睡眠节律
        if sleep_every and (t + 1) % sleep_every == 0:
            for _ in range(sleep_len):
                sleep_reports.append(brain.step_sleep())

    arr = np.array(results, dtype=float)
    n = len(arr)
    tail = arr[int(0.75 * n):]
    curve = [float(arr[i:i + 500].mean())
             for i in range(0, n, 500)]
    comp = brain.drive_sat.report()
    cap = hb.sheath_registry.capacity_report()
    sleep_agg = {}
    for sr in sleep_reports:
        for p, e in (sr.get("sleep_report", {}).get("phases") or {}).items():
            d = sleep_agg.setdefault(p, [0.0, 0])
            d[0] += e["gain_delta"]; d[1] += 1
    return dict(
        top1=float(tail.mean()),
        top1_all=float(arr.mean()),
        curve=curve,
        uniform=1.0 / N_TOKEN,
        echo_base=float(np.mean([1.0 if world.stream[i] == world.stream[i - 1]
                                 else 0.0
                                 for i in range(1, len(world.stream))])),
        bigram_base=_bigram_top1(world.stream),
        neurons=len(hb.ecosystem.neurons),
        sheaths=cap["n_sheaths"],
        turnover=cap["turnover"],
        mean_lambda=float(np.mean(lambdas)),
        pct_high=float(np.mean([c == "higher" for c in controllers])),
        competence=comp["competence"],
        shadow=brain.shadow_mode,
        sleep_gain={p: (v[0] / v[1]) for p, v in sleep_agg.items()},
        attention=res.get("attention"),
        spikes=res.get("spikes"),
        seizure_brakes=res.get("seizure_brakes", 0),
        src_cred=res.get("src_cred", {}),
    )


def _bigram_top1(stream):
    """bigram 参照（"LLM-lite"）：全流统计，argmax 预测下一个"""
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
    print("=" * 88)
    print("超模态语法习得：FSM 符号流的自回归补全")
    print(f"  {N_TOKEN} tokens / {N_STATES} states / vis+aud 嵌入异构 / "
          f"steps={STEPS} / sleep {SLEEP_LEN}x每{SLEEP_EVERY}步")
    print("=" * 88)
    rows = []
    for s in SEEDS:
        r = run(seed=s)
        rows.append((s, r))
        print(f"\n── seed {s} ─────────────────────────────────────")
        print(f"  top1(末1/4) = {r['top1']:.3f}   (all {r['top1_all']:.3f})   "
              f"uniform {r['uniform']:.3f}   echo {r['echo_base']:.3f}   "
              f"bigram {r['bigram_base']:.3f}")
        print(f"  学习曲线(每500步): "
              + " ".join(f"{v:.2f}" for v in r["curve"]))
        print(f"  结构: 神经元 {r['neurons']}  髓鞘 {r['sheaths']}  "
              f"周转 {r['turnover']:.0f}   λ均值 {r['mean_lambda']:.3f}   "
              f"高层占比 {r['pct_high']:.2f}   shadow={r['shadow']}   "
              f"competence {r['competence']:+.2f}")
        if r["sleep_gain"]:
            print("  睡眠Δgain/周期: "
                  + "  ".join(f"{p} {v:+.3f}" for p, v in r["sleep_gain"].items()))

    print()
    print("=" * 88)
    m = lambda k: np.mean([r[k] for _, r in rows])
    sd = lambda k: np.std([r[k] for _, r in rows])
    print(f"汇总(3 seeds): 系统 top1 = {m('top1'):.3f}±{sd('top1'):.3f}   "
          f"uniform {m('uniform'):.3f}   echo {m('echo_base'):.3f}   "
          f"bigram {m('bigram_base'):.3f}")
    print("判据：系统 top1 显著高于 uniform/echo，接近 bigram → "
          "架构对复杂符号流有效")


if __name__ == "__main__":
    main()
