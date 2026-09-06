"""婴幼儿式多模态语言习得 —— 发育髓鞘网络 vs 回声反射

任务：通过分词器打字机学会造基础句子，贯穿视觉/听觉/语言多模态，
并允许"有输入无输出"的纯感知校准期。

=========================== 架构对应关系 ===========================

- 爬虫脑反射 = **延迟回声 + babbling 噪声**（婴儿的 echolalia，真实存在）
  它定义了发声器官的"行动面"：能重复 d 帧前听到的词，但无法造新句子。
- 高层脑 = 发育式髓鞘通路网络，从多模态上下文预测下一帧全部通道。
- 自回归目标 = 下一帧真实观测（环境即时反馈，无外部标注）。
- 蒸馏目标 = 回声反射的输出（仅作为起步示范）。
- 感知校准 = output_gate=0 时**内部预测照常进行**，只是不发声。
  这正是"看到爸爸妈妈什么都不会干，但知觉被校准"的实现。

=========================== 多模态结构 ===========================

- vis (8维)  : 物体的视觉外观。同类共享原型成分（人形/物形）+ 个体差异。
- aud (8维)  : 拼音的声学特征。声母按**发音部位/方式**编码，韵母按
               舌位/开口度编码 → 语音相近的词声学特征自然相近
               （如 mama/baba 同为双唇+a，baobao/bao 同为 b+ao）。
- lang (12维) : 词的分布式表征（随机近似正交），是打字机要产出的目标。
- noise (4维) : 纯传感器噪声通道（视网膜噪点/环境声），**无信息**。

真实可学结构：
  1. 跨模态对应：同一词的 vis/aud/lang 同时出现 → 超模态绑定
  2. 序列结构：句子按模板生成 → 相邻词可预测
  3. 噪声通道必须被识别并淘汰

=========================== 婴幼儿教育阶段 ===========================

- Stage 0 感知校准 (0-40%)  : output_gate=0，只听只看，不发声。
- Stage 1 产出练习 (40-70%) : output_gate=1，反射主导发声，高层蒸馏+自回归。
- Stage 2 自由产出 (70-100%): 高层按毕业度逐步接管。

核心待检验假设：**Stage 0 的纯感知校准能加速并提升后续的语言产出。**
"""
import numpy as np

# ===================================================================
# 1. 世界：词汇与多模态特征
# ===================================================================

# 拼音 → (声母, 韵母)
PINYIN = {
    "mama":    ("m",  "a"),
    "baba":    ("b",  "a"),
    "baobao":  ("b",  "ao"),
    "pingguo": ("p",  "ing"),
    "xigua":   ("x",  "i"),
    "qiu":     ("q",  "iu"),
    "bao":     ("b",  "ao"),
    "chi":     ("ch", "i"),
    "kan":     ("k",  "an"),
    "da":      ("d",  "a"),
    "xiao":    ("x",  "iao"),
    "hong":    ("h",  "ong"),
}

# 声母：发音部位 (5维 one-hot) + 发音方式 (4维 one-hot)
_INIT_PLACE = {
    "m": 0, "b": 0, "p": 0,          # 双唇
    "d": 1, "t": 1,                   # 舌尖中
    "g": 2, "k": 2, "h": 2,           # 舌根
    "x": 3, "q": 3, "j": 3,           # 舌面
    "ch": 4, "zh": 4, "sh": 4,        # 卷舌
}
_INIT_MANNER = {
    "m": 0,                            # 鼻音
    "b": 1, "p": 1, "d": 1, "t": 1, "g": 1, "k": 1,   # 塞音
    "x": 2, "q": 2, "j": 2, "h": 2, "ch": 3, "zh": 3, "sh": 3,  # 擦/塞擦
}

# 韵母：开口度 (2维) + 舌位前后 (2维) + 鼻音韵尾 (1维)
_FINAL_OPEN = {"a": 1.0, "ao": 1.0, "an": 1.0, "ang": 1.0, "ong": 0.4,
               "i": 0.0, "ing": 0.0, "iu": 0.1, "iao": 0.3, "o": 0.5, "e": 0.5}
_FINAL_BACK = {"a": 0.2, "ao": 0.8, "an": 0.3, "ang": 0.9, "ong": 1.0,
               "i": 0.0, "ing": 0.1, "iu": 0.2, "iao": 0.3, "o": 1.0, "e": 0.7}
_FINAL_NASAL = {"an": 1.0, "ang": 1.0, "ing": 1.0, "ong": 1.0}

PEOPLE = ["mama", "baba", "baobao"]
OBJECTS = ["pingguo", "xigua", "qiu"]
ACTIONS = ["bao", "chi", "kan"]
MODIFIERS = ["da", "xiao", "hong"]
VOCAB = PEOPLE + OBJECTS + ACTIONS + MODIFIERS
NW = len(VOCAB)

# 句子模板（序号用于合法性判定）
TEMPLATES = [
    ["P", "A", "O"],   # mama chi pingguo
    ["P", "A", "P"],   # mama bao baobao
    ["M", "O"],        # da pingguo
    ["P", "A"],        # baobao kan
    ["P", "A", "M", "O"],  # mama chi da pingguo
]

D_VIS, D_AUD, D_LANG, D_NOISE = 8, 8, 12, 4
D = D_VIS + D_AUD + D_LANG + D_NOISE
S_VIS = slice(0, D_VIS)
S_AUD = slice(D_VIS, D_VIS + D_AUD)
S_LANG = slice(D_VIS + D_AUD, D_VIS + D_AUD + D_LANG)
S_NOISE = slice(D_VIS + D_AUD + D_LANG, D)


def _aud_features(word, rng):
    """拼音 → 8维声学特征（结构化的，非随机）"""
    init, final = PINYIN[word]
    v = np.zeros(D_AUD)
    # 声母：部位 one-hot (5维 → 前5维)
    v[_INIT_PLACE.get(init, 0)] = 1.0
    # 声母：方式（第5维，用连续值区分鼻/塞/擦）
    v[5] = _INIT_MANNER.get(init, 2) / 3.0
    # 韵母：开口度 / 舌位 / 鼻韵尾
    v[6] = _FINAL_OPEN.get(final, 0.5)
    v[7] = _FINAL_BACK.get(final, 0.5)
    v[5] += 0.3 * _FINAL_NASAL.get(final, 0.0)
    return v


def _vis_features(word, proto, rng):
    """8维视觉特征：类别原型 + 个体偏移 + 属性调制"""
    if word in PEOPLE:
        base = proto["person"]
    elif word in OBJECTS:
        base = proto["object"]
    elif word in ACTIONS:
        base = proto["action"]
    else:
        base = proto["modifier"]
    v = base + 0.35 * rng.normal(size=D_VIS)
    # 修饰词携带属性调制（"红"影响颜色维度，"大"影响尺度维度）
    if word == "hong":
        v[0] += 0.8
    elif word == "da":
        v[1] += 0.8
    elif word == "xiao":
        v[1] -= 0.8
    return v


def _sem_class(w):
    """词的语义类别（决定语言表征的原型成分）"""
    if w in PEOPLE:
        return "P"
    if w in OBJECTS:
        return "O"
    if w in ACTIONS:
        return "A"
    return "M"


def build_world(seed):
    """构造世界：每个词的 vis / aud / lang 特征

    **lang 特征必须有类别结构，不能是纯随机正交向量。**

    初版用纯随机向量，导致任务不可学：句子模板本质是**槽位（类别）序列**
    [P A O]，序列转移发生在类别层面；而 vis/aud 到 lang 的映射也只能在
    类别层面对齐。纯随机的 lang 与 vis/aud 之间无任何系统关联，线性通路
    既学不到类别转移，也学不到跨模态对应（实测准确率 6.9%，低于随机猜
    的 8.3%）。

    修正：lang = 类别原型 + 个体偏移。
    - 同类词的 lang 相近（余弦 ≈ 0.58）→ 槽位转移可学、跨模态可对齐
    - 异类词的 lang 近正交（余弦 ≈ 0）→ 最近邻解码仍可区分个体
    """
    rng = np.random.default_rng(seed)
    proto = {k: 0.9 * rng.normal(size=D_VIS)
             for k in ("person", "object", "action", "modifier")}

    # 语言：类别原型 + 个体偏移
    cls_proto = {c: rng.normal(size=D_LANG) for c in ("P", "O", "A", "M")}
    for c in cls_proto:
        cls_proto[c] /= np.linalg.norm(cls_proto[c])
    L = np.zeros((NW, D_LANG))
    for i, w in enumerate(VOCAB):
        ind = rng.normal(size=D_LANG)
        ind /= np.linalg.norm(ind)
        v = 0.85 * cls_proto[_sem_class(w)] + 0.75 * ind
        L[i] = v / np.linalg.norm(v) * 1.5

    VIS = np.zeros((NW, D_VIS))
    AUD = np.zeros((NW, D_AUD))
    for i, w in enumerate(VOCAB):
        VIS[i] = _vis_features(w, proto, rng)
        AUD[i] = _aud_features(w, rng)
    return {"vis": VIS, "aud": AUD, "lang": L,
            "class": [_sem_class(w) for w in VOCAB],
            "index": {w: i for i, w in enumerate(VOCAB)}}


# 语义选择限制（selectional restrictions）——真实语言的核心结构
# 动词对宾语有约束：吃→可食物，抱→可抱的，看→任意。
# 这让"下一词预测"有真实可学的结构，而非纯类别转移的等概率猜测。
VOBJ = {
    "chi": ["pingguo", "xigua"],
    "bao": ["baobao", "qiu"],
    "kan": ["mama", "baba", "baobao", "pingguo", "xigua", "qiu"],
}
MOBJ = {
    "hong": ["pingguo", "xigua"],
    "da":   OBJECTS,
    "xiao": OBJECTS,
}


def gen_sentence(rng):
    """从模板 + 语义选择限制采样一个句子"""
    t = TEMPLATES[rng.integers(len(TEMPLATES))]
    out = []
    pending_verb = None
    pending_mod = None
    for slot in t:
        if slot == "P":
            out.append(str(rng.choice(PEOPLE)))
        elif slot == "A":
            v = str(rng.choice(ACTIONS))
            out.append(v)
            pending_verb = v
        elif slot == "O":
            if pending_verb is not None:
                out.append(str(rng.choice(VOBJ[pending_verb])))
                pending_verb = None
            elif pending_mod is not None:
                out.append(str(rng.choice(MOBJ[pending_mod])))
                pending_mod = None
            else:
                out.append(str(rng.choice(OBJECTS)))
        else:
            m = str(rng.choice(MODIFIERS))
            out.append(m)
            pending_mod = m
    return out


# ===================================================================
# 2. 观测流
# ===================================================================

def make_stream(world, n_sent, seed, obs_noise=0.05, noise_amp=1.0):
    """生成多模态帧序列

    每帧 = [vis(w_t), aud(w_t), lang(w_t), noise]
    自回归目标 = 下一帧。预测 lang 部分即"预测下一个词"（打字机）。

    句子之间插入静默帧（全零），模拟对话间隙。
    """
    rng = np.random.default_rng(seed)
    idx = world["index"]
    frames, words, sent_id = [], [], []
    for s in range(n_sent):
        sent = gen_sentence(rng)
        for w in sent:
            o = np.zeros(D)
            o[S_VIS] = world["vis"][idx[w]]
            o[S_AUD] = world["aud"][idx[w]]
            o[S_LANG] = world["lang"][idx[w]]
            o[S_NOISE] = noise_amp * rng.normal(size=D_NOISE)
            o += obs_noise * rng.normal(size=D)
            frames.append(o)
            words.append(w)
            sent_id.append(s)
        # 句间静默
        gap = np.zeros(D)
        gap[S_NOISE] = noise_amp * rng.normal(size=D_NOISE)
        frames.append(gap); words.append("<gap>"); sent_id.append(-1)
    return np.array(frames), words, np.array(sent_id)


# ===================================================================
# 3. 爬虫脑反射：延迟回声 + babbling
# ===================================================================

class EchoReflex:
    """延迟回声反射（婴儿 echolalia）

    y_reflex[t] = lang 通道在 t-d 帧的值 + babbling 噪声。
    能重复刚听到的词，但**无法**造新句子——它不包含任何序列结构知识。
    """

    def __init__(self, delay=2, babble=0.3, seed=0):
        self.delay = delay
        self.babble = babble
        self.rng = np.random.default_rng(seed)
        self.buf = []

    def __call__(self, o):
        """输出 D 维向量：语言通道 = 延迟回声，其余通道 = 0（反射不预测感知）"""
        self.buf.append(o[S_LANG].copy())
        if len(self.buf) > self.delay:
            echo = self.buf[-1 - self.delay]
        else:
            echo = np.zeros(D_LANG)
        y = np.zeros(D)
        y[S_LANG] = echo + self.babble * self.rng.normal(size=D_LANG)
        return y


# ===================================================================
# 4. 发育网络
# ===================================================================

class DevoNet:
    """发育式髓鞘通路网络

    复用了前序实验验证过的机制：
    - 逐通道自适应归一化（否则通路存亡由尺度而非可预测性决定）
    - 相对衰减 ρ（绝对衰减会在收敛后误杀必要通路）
    - 源维度可信度门控 self_proof（防止噪声维度乱连）
    - 局部梯度 + NLMS 归一化步长（无反向传播）
    """

    def __init__(self, D, rho=0.02, lr=0.5, death=0.0,
                 allow_cross=True, wire=True, self_proof=0.0,
                 wire_confirm=8, momentum=0.99, min_age=200,
                 active_thresh=0.3):
        self.D = D
        self.rho = rho
        self.lr = lr
        # death 必须 ≤ 新通路初始增益（沉默突触 = 0）。
        # 若 death > 0，新边一建立就低于阈值被删，下一轮又被重建
        # → "新建—淘汰"空转（实测 born/died 各 5 万次，准确率反而低于
        #    随机猜）。用 death=0 表示"增益转负才淘汰"。
        self.death = death
        self.allow_cross = allow_cross
        self.wire = wire
        self.self_proof = self_proof
        self.wire_confirm = wire_confirm
        self.momentum = momentum
        self.min_age = min_age          # 新通路豁免期（步）
        self.active_thresh = active_thresh

        self.G = np.eye(D) * 1.0
        self.M = np.eye(D).astype(bool)
        self.age = np.zeros((D, D), dtype=int)
        self.age[np.diag_indices(D)] = min_age   # 初始通路视为已成熟
        self.co_fire = np.zeros((D, D), dtype=int)
        self.cred = np.zeros(D)          # 带符号 EMA 的贡献
        self.cred_abs = np.zeros(D)      # 信噪比式可信度（门控用）
        self.energy = np.ones(D)         # 各维度输入能量（EMA）
        self.rms = np.ones(D)
        self._step = 0
        self.born, self.died = D, 0

    # ---- 归一化 ----
    def norm(self, o):
        return o / (self.rms + 1e-6)

    def denorm(self, o):
        return o * (self.rms + 1e-6)

    def update_rms(self, o):
        self.rms = np.sqrt(self.momentum * self.rms ** 2
                           + (1 - self.momentum) * o ** 2)

    # ---- 前向 ----
    def forward(self, o):
        """预测下一帧（在归一化空间）"""
        return self.G.T @ self.norm(o)

    # ---- 学习 ----
    def step(self, o, o_next, learn=True):
        """一步：更新通路 + 成边 + 选择

        Args:
            o: 当前观测（原始空间）
            o_next: 下一帧真实观测（原始空间）—— 环境即时反馈
            learn: 是否更新（False = 纯推理）
        """
        self.update_rms(o)
        on = self.norm(o)
        on_next = self.norm(o_next)
        pred = self.G.T @ on
        resid = on_next - pred

        if not learn:
            return resid, pred

        # 局部梯度：∂‖resid‖²/∂G[i,j] = -2·resid[j]·on[i]，NLMS 归一化
        denom = float(np.dot(on, on)) + 1e-8
        self.G += self.M * (self.lr * np.outer(on, resid) / denom)

        # 用进废退：相对衰减
        self.G -= self.M * (self.rho * self.G)

        # 源维度可信度 → 供 self_proof 成边门控
        #
        # **必须用带符号 EMA，不能用 |贡献|**。
        # 用 |贡献| 时噪声维度得分最高（实测 noise 1.94 vs lang 0.44）——
        # 噪声幅度大 → |贡献| 大，但那是**无信息的大**，正负随机。
        # 带符号 EMA 下：噪声贡献正负抵消 → 0；真实维度有系统性方向 → 非零。
        # 用更长的时间常数（0.99）抑制噪声残余波动，并除以输入能量
        # 得到**信噪比式**度量：单位输入能量的系统性贡献。
        # 只做带符号 EMA 时噪声维仍最高（实测 noise 0.238 vs lang 0.018）
        # ——噪声单步幅度大，即使 EMA 衰减后仍压过真实信号。
        signed = (np.outer(on, resid) / denom).sum(axis=1)
        self.cred = 0.99 * self.cred + 0.01 * signed
        self.energy = 0.99 * self.energy + 0.01 * on ** 2
        self.cred_abs = np.abs(self.cred) / (self.energy + 1e-6)

        # 淘汰（新通路在豁免期内不判死，否则沉默突触活不过一轮）
        mature = self.age >= self.min_age
        dead = self.M & mature & (self.G < self.death)
        self.G[dead] = 0.0
        self.M[dead] = False
        self.co_fire[dead] = 0
        self.age[dead] = 0
        self.died += int(dead.sum())
        self.age[self.M] += 1

        # 共发射成边
        if self.wire:
            active = np.where(np.abs(on) > self.active_thresh)[0]
            for i in active:
                for j in active:
                    if not self.allow_cross and i != j:
                        continue
                    self.co_fire[i, j] += 1
                    if (not self.M[i, j]) and self.co_fire[i, j] >= self.wire_confirm:
                        if self.self_proof > 0.0 and i != j:
                            if self.cred_abs[i] < self.self_proof:
                                continue
                        self.M[i, j] = True
                        self.G[i, j] = 0.0
                        self.age[i, j] = 0
                        self.born += 1

        self._step += 1
        return resid, pred

    def stats(self):
        return {"n_paths": int(self.M.sum()),
                "n_eff": int((self.M & (np.abs(self.G) > 0.1)).sum()),
                "born": self.born, "died": self.died}


# ===================================================================
# 5. 训练（婴幼儿教育三阶段）
# ===================================================================

def decode_word(lang_vec, world):
    """最近邻解码：语言向量 → 词"""
    L = world["lang"]
    sims = L @ lang_vec / (np.linalg.norm(L, axis=1) * np.linalg.norm(lang_vec) + 1e-9)
    return int(np.argmax(sims)), float(sims.max())


def train(world, seed, n_sent=400, rho=0.02, lr=0.5, self_proof=0.0,
          allow_cross=True, wire=True, norm=True,
          f_percept=0.40, f_practice=0.70, log_every=None,
          active_thresh=0.3, min_age=200, wire_confirm=8):
    """婴幼儿式训练

    Args:
        f_percept:  Stage 0（纯感知校准）占比
        f_practice: Stage 0+1 占比（余下为 Stage 2 自由产出）
    """
    frames, words, sent_id = make_stream(world, n_sent, seed + 1000)
    T = len(frames)
    t0 = int(T * f_percept)
    t1 = int(T * f_practice)

    net = DevoNet(D, rho=rho, lr=lr, self_proof=self_proof,
                  allow_cross=allow_cross, wire=wire,
                  active_thresh=active_thresh, min_age=min_age,
                  wire_confirm=wire_confirm)
    reflex = EchoReflex(delay=2, babble=0.3, seed=seed)

    if not norm:
        net.rms = np.ones(D) * 1e6   # 关闭归一化（除以极大值≈不缩放）

    progress = 0.0
    hist = {"stage": [], "pred_mse": [], "next_acc1": [], "lambda": []}
    errs = []
    # 分窗口统计准确率：全程平均会被学习初期的随机猜测稀释，
    # 无法反映收敛后的真实能力。改为末段窗口。
    win_correct, win_total = 0, 0
    win_cos = 0.0
    tail_start = int(T * 0.66)

    last_pred_lang = np.zeros(D_LANG)

    for t in range(T - 1):
        o, o_next = frames[t], frames[t + 1]
        stage = 0 if t < t0 else (1 if t < t1 else 2)

        # ---- 感知 vs 产出的真实区别 ----
        #
        # 感知期（output_gate=0）：vis + aud + lang **全部来自环境**——
        #   看到实物、同时听到词，是完整的跨模态配对。系统只预测，不发声。
        #   这正是"看到爸爸妈妈什么都不会干，但知觉被校准"：大量的
        #   vis/aud→lang 配对数据在此期建立，且不消耗输出资源。
        #
        # 产出期（output_gate=1）：vis + aud 来自环境，但 **lang 由系统自己
        #   产生**——语言通道的输入是上一帧自己的预测（自反馈闭环）。
        #   只有此时才需要真正"说出一个词"。
        #
        # 关键：感知期提供完整配对（听到"mama"同时看到妈妈），产出期只有
        # 视觉/听觉线索、语言要自己补全。感知期建立的绑定是产出期的基础。
        output_gate = 0.0 if stage == 0 else 1.0
        if output_gate > 0.0:
            o_input = o.copy()
            o_input[S_LANG] = last_pred_lang   # 自反馈：说上一个词
        else:
            o_input = o                        # 环境提供：听到完整的词

        # ---- 内部预测（**始终进行**，与是否发声无关）----
        resid, pred = net.step(o_input, o_next, learn=True)
        errs.append(float(np.mean(resid ** 2)))

        # ---- 反射（回声）----
        y_refl = reflex(o)

        # ---- 高层输出的语言部分 ----
        y_high = net.denorm(pred)
        if output_gate > 0.0:
            last_pred_lang = y_high[S_LANG].copy()

        # ---- 毕业度：高层 vs 反射，谁的语言预测更准 ----
        # 只在有真实语言内容的帧上评估（跳过静默帧）
        if np.linalg.norm(o_next[S_LANG]) > 1e-6:
            e_refl = float(np.mean((o_next[S_LANG] - y_refl[S_LANG]) ** 2))
            e_high = float(np.mean((o_next[S_LANG] - y_high[S_LANG]) ** 2))
            raw = (e_refl - e_high) / max(e_refl, 1e-8)
            progress = 0.98 * progress + 0.02 * raw
            # 下一词预测准确率（高层）——只统计末段窗口
            wi, _ = decode_word(y_high[S_LANG], world)
            true_w = words[t + 1]
            if true_w != "<gap>" and t >= tail_start:
                win_total += 1
                win_correct += int(VOCAB[wi] == true_w)
                # 余弦相似度：线性模型在"下一词有多个等概率候选"时只能预测
                # 均值，top-1 必然失败；余弦能反映是否学到了正确的语义区域。
                tv = world["lang"][world["index"][true_w]]
                pv = y_high[S_LANG]
                cs = float(np.dot(tv, pv) / (np.linalg.norm(tv) * np.linalg.norm(pv) + 1e-9))
                win_cos += cs

        lam = 1.0 / (1.0 + np.exp(-max(-60, min(60, 8.0 * progress))))

        if log_every and (t % log_every == 0 or t in (t0, t1)):
            hist["stage"].append(stage)
            hist["pred_mse"].append(float(np.mean(errs[-200:])) if errs else 0.0)
            hist["next_acc1"].append(correct / max(total, 1))
            hist["lambda"].append(lam)

    return {
        "net": net,
        "progress": progress,
        "lambda": lam,
        "next_acc1": win_correct / max(win_total, 1),   # 末段（收敛后）准确率
        "next_cos": win_cos / max(win_total, 1),        # 语义余弦（抗歧义）
        "final_mse": float(np.mean(errs[-500:])) if errs else float("nan"),
        "hist": hist,
        "stats": net.stats(),
    }


# ===================================================================
# 6. 评估
# ===================================================================

def eval_cross_modal(net, world, seed=77):
    """跨模态检索：给视觉，预测听觉（测超模态绑定是否建立）

    遮住听觉通道（置零），让网络预测下一帧，看听觉部分的预测
    与真实听觉的相关度。高相关 = 建立了 vis→aud 绑定。
    """
    frames, words, _ = make_stream(world, 60, seed)
    cors = []
    for t in range(len(frames) - 1):
        o = frames[t].copy()
        o[S_AUD] = 0.0                      # 遮住听觉
        pred = net.denorm(net.G.T @ net.norm(o))
        true_aud = frames[t + 1][S_AUD]
        p = pred[S_AUD]
        if np.linalg.norm(true_aud) > 1e-6 and np.linalg.norm(p) > 1e-9:
            cors.append(float(np.corrcoef(true_aud, p)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


def _gen_slots(gen):
    """生成序列 → 槽位（词性类别）序列"""
    out = []
    for w in gen:
        if w in PEOPLE:
            out.append("P")
        elif w in OBJECTS:
            out.append("O")
        elif w in ACTIONS:
            out.append("A")
        else:
            out.append("M")
    return out


def _sem_legal(gen):
    """语义合法性：动词/修饰词对其后词的选择限制是否满足

    这是比模板匹配更细的判据——模板只检查词性序列（P A O），
    选择限制检查具体内容（吃了没有？吃的是不是能吃的？）。
    """
    for i, w in enumerate(gen):
        if w in VOBJ and i + 1 < len(gen):
            if gen[i + 1] not in VOBJ[w]:
                return False
        if w in MOBJ and i + 1 < len(gen):
            if gen[i + 1] not in MOBJ[w]:
                return False
    return True


def eval_generation(net, world, n_sent=200, seed=99, max_len=5):
    """自由生成：从首词的多模态上下文出发，自回归造句子

    注意：这里用**纯自回归**生成（每一步把自己的预测作为下一步输入），
    与训练时的 teacher forcing 不同，误差会累积——这是真实"造句"的设定。

    判据分三层（由粗到细）：
    1. tpl_valid  : 词性序列匹配任一模板（如 P A O）
    2. sem_valid  : 同时满足选择限制（如 chi 后必须是可食物）
    3. exact      : 与某个真实句子完全一致
    """
    rng = np.random.default_rng(seed)
    idx = world["index"]
    tpl_hit = sem_hit = exact_hit = 0
    lens = []
    uniq = set()
    for _ in range(n_sent):
        target = gen_sentence(rng)
        w0 = target[0]
        o = np.zeros(D)
        o[S_VIS] = world["vis"][idx[w0]]
        o[S_AUD] = world["aud"][idx[w0]]
        o[S_LANG] = world["lang"][idx[w0]]
        pred = net.denorm(net.G.T @ net.norm(o))
        gen = [w0]
        for _ in range(max_len - 1):
            nxt = np.zeros(D)
            nxt[S_LANG] = pred[S_LANG]
            nxt[S_VIS] = pred[S_VIS]
            nxt[S_AUD] = pred[S_AUD]
            pred = net.denorm(net.G.T @ net.norm(nxt))
            wi, _ = decode_word(pred[S_LANG], world)
            gen.append(VOCAB[wi])
        lens.append(len(gen))
        uniq.add(tuple(gen))
        slots = _gen_slots(gen)
        if tuple(slots) in [tuple(t) for t in TEMPLATES]:
            tpl_hit += 1
            if _sem_legal(gen):
                sem_hit += 1
        if gen[:len(target)] == target:
            exact_hit += 1
    return {"tpl_valid": tpl_hit / n_sent,
            "sem_valid": sem_hit / n_sent,
            "exact": exact_hit / n_sent,
            "diversity": len(uniq) / n_sent,
            "mean_len": float(np.mean(lens))}


def eval_reflex_generation(world, n_sent=200, seed=99, max_len=5):
    """回声反射的自由生成基线：只能重复，衡量其"造句"能力上限"""
    rng = np.random.default_rng(seed)
    idx = world["index"]
    reflex = EchoReflex(delay=2, babble=0.3, seed=0)
    tpl_hit = sem_hit = 0
    for _ in range(n_sent):
        target = gen_sentence(rng)
        # 反射先"听"一遍目标句（这是它的输入来源）
        for w in target:
            o = np.zeros(D)
            o[S_VIS] = world["vis"][idx[w]]
            o[S_AUD] = world["aud"][idx[w]]
            o[S_LANG] = world["lang"][idx[w]]
            reflex(o)
        gen = []
        for _ in range(max_len):
            y = reflex(np.zeros(D))
            wi, _ = decode_word(y[S_LANG], world)
            gen.append(VOCAB[wi])
        slots = _gen_slots(gen)
        if tuple(slots) in [tuple(t) for t in TEMPLATES]:
            tpl_hit += 1
            if _sem_legal(gen):
                sem_hit += 1
    return {"tpl_valid": tpl_hit / n_sent, "sem_valid": sem_hit / n_sent}


def eval_reflex_baseline(world, n_sent=200, seed=99):
    """回声反射基线：只能重复，无法造句"""
    rng = np.random.default_rng(seed)
    idx = world["index"]
    reflex = EchoReflex(delay=2, babble=0.3, seed=0)
    frames, words, _ = make_stream(world, n_sent, seed)
    correct = 0
    total = 0
    for t in range(len(frames) - 1):
        y = reflex(frames[t])
        if np.linalg.norm(frames[t + 1][S_LANG]) > 1e-6:
            wi, _ = decode_word(y[S_LANG], world)
            true_w = words[t + 1]
            if true_w != "<gap>":
                total += 1
                correct += int(VOCAB[wi] == true_w)
    return {"next_acc1": correct / max(total, 1)}


# ===================================================================
# 7. 主实验
# ===================================================================

SEEDS = (0, 1, 2, 3, 4)


def agg(fn, **kw):
    out = {}
    for s in SEEDS:
        r = fn(world=build_world(s), seed=s, **kw)
        for k, v in r.items():
            if isinstance(v, (int, float)):
                out.setdefault(k, []).append(float(v))
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))))
            for k, v in out.items()}


def run_config(kw, seeds=SEEDS, n_sent=400, n_gen=150):
    """跑一个配置：每个 seed 只训练一次，复用网络做全部评估"""
    rows = []
    for sd in seeds:
        w = build_world(sd)
        r = train(w, sd, n_sent=n_sent, **kw)
        net = r["net"]
        cm = eval_cross_modal(net, w)
        g = eval_generation(net, w, n_sent=n_gen, seed=sd + 500)
        st = net.stats()
        # 噪声通道相关通路占比（越低说明越会"不去预测噪声"）
        noise_sl = list(range(D_NOISE))
        nz = D - D_NOISE
        noise_paths = sum(1 for i in range(D) for j in range(D)
                          if net.M[i, j] and (i >= nz or j >= nz))
        rows.append({
            "next_acc1": r["next_acc1"],
            "next_cos": r["next_cos"],
            "cross_modal": cm,
            "tpl_valid": g["tpl_valid"],
            "sem_valid": g["sem_valid"],
            "exact": g["exact"],
            "diversity": g["diversity"],
            "progress": r["progress"],
            "n_paths": st["n_paths"],
            "n_eff": st["n_eff"],
            "noise_ratio": noise_paths / max(1, st["n_paths"]),
            "turnover": (st["born"] + st["died"]) / max(1, st["n_paths"]),
        })
    keys = rows[0].keys()
    return {k: (float(np.mean([r[k] for r in rows])),
                float(np.std([r[k] for r in rows]) / np.sqrt(len(rows))))
            for k in keys}


def main():
    print("=" * 94)
    print("婴幼儿式多模态语言习得：发育髓鞘网络 vs 回声反射")
    print(f"词表 {NW} 词 | 维度 vis{D_VIS}+aud{D_AUD}+lang{D_LANG}+noise{D_NOISE}={D}"
          f" | 句子 {len(TEMPLATES)} 模板 + 语义选择限制")
    print(f"seeds={len(SEEDS)}  n_sent=400")
    print("=" * 94)

    # 参照基线
    orc, rnd, rfl = [], [], []
    for s in SEEDS:
        w = build_world(s)
        rfl.append(eval_reflex_baseline(w)["next_acc1"])
    print(f"\n参照：随机猜 {100/NW:.1f}%   回声反射 {np.mean(rfl)*100:.1f}%"
          f"   Oracle(一阶转移) ~28.5%")

    configs = [
        ("完整（推荐）", dict(rho=0.02, self_proof=0.05, allow_cross=True,
                          wire=True, norm=True, f_percept=0.40)),
        ("去掉感知校准期", dict(rho=0.02, self_proof=0.05, allow_cross=True,
                          wire=True, norm=True, f_percept=0.0)),
        ("去掉用进废退(ρ=0)", dict(rho=0.0, self_proof=0.05, allow_cross=True,
                              wire=True, norm=True, f_percept=0.40)),
        ("去掉跨模态成边", dict(rho=0.02, self_proof=0.05, allow_cross=False,
                          wire=True, norm=True, f_percept=0.40)),
        ("去掉self_proof门控", dict(rho=0.02, self_proof=0.0, allow_cross=True,
                               wire=True, norm=True, f_percept=0.40)),
        ("去掉归一化", dict(rho=0.02, self_proof=0.05, allow_cross=True,
                       wire=True, norm=False, f_percept=0.40)),
    ]

    print()
    print("=" * 94)
    print("A. 核心对照（5 seeds 均值 ± 标准误）")
    print("=" * 94)
    print(f"{'配置':<22}{'下一词%':>9}{'语义余弦':>9}{'跨模态':>8}"
          f"{'模板合法':>9}{'语义合法':>9}{'完全匹配':>9}{'多样性':>8}")
    print("-" * 94)
    res = {}
    for name, kw in configs:
        r = run_config(kw)
        res[name] = r
        print(f"{name:<22}{r['next_acc1'][0]*100:>8.1f}%"
              f"{r['next_cos'][0]:>9.3f}"
              f"{r['cross_modal'][0]:>8.3f}"
              f"{r['tpl_valid'][0]*100:>8.1f}%"
              f"{r['sem_valid'][0]*100:>8.1f}%"
              f"{r['exact'][0]*100:>8.1f}%"
              f"{r['diversity'][0]:>8.2f}")

    print()
    print(f"{'配置':<22}{'通路':>8}{'有效':>8}{'噪声占比':>10}{'周转率':>9}"
          f"{'毕业度':>9}")
    print("-" * 94)
    for name, _ in configs:
        r = res[name]
        print(f"{name:<22}{r['n_paths'][0]:>8.0f}{r['n_eff'][0]:>8.0f}"
              f"{r['noise_ratio'][0]:>10.2f}{r['turnover'][0]:>9.1f}"
              f"{r['progress'][0]:>9.2f}")

    # 反射的生成基线
    rg = []
    for s in SEEDS:
        rg.append(eval_reflex_generation(build_world(s)))
    print(f"\n回声反射生成：模板合法 {np.mean([x['tpl_valid'] for x in rg])*100:.1f}%"
          f"   语义合法 {np.mean([x['sem_valid'] for x in rg])*100:.1f}%")

    # ---- 感知校准期长度 ----
    print()
    print("=" * 94)
    print('B. 感知校准期长度敏感性（“有输入无输出”的价值）')
    print("=" * 94)
    print(f"{'校准期占比':>10}{'下一词%':>9}{'语义余弦':>10}{'模板合法':>10}"
          f"{'语义合法':>10}{'跨模态':>9}{'有效通路':>9}")
    print("-" * 94)
    for fp in (0.0, 0.15, 0.30, 0.40, 0.55, 0.70):
        r = run_config(dict(rho=0.02, self_proof=0.05, allow_cross=True,
                            wire=True, norm=True, f_percept=fp))
        print(f"{fp:>10.2f}{r['next_acc1'][0]*100:>8.1f}%"
              f"{r['next_cos'][0]:>10.3f}"
              f"{r['tpl_valid'][0]*100:>9.1f}%{r['sem_valid'][0]*100:>9.1f}%"
              f"{r['cross_modal'][0]:>9.3f}{r['n_eff'][0]:>9.0f}")


if __name__ == "__main__":
    main()
