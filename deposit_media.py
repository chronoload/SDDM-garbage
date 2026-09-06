"""T7 重放资料扩展：图片（扫视重放）/ 电影（帧差 event 信道）/ PDF

## 教义：同一架构不改，只换沉积物适配器

三类沉积物适配成**同一个反应协议**：

    vis  : 当前 tick 的感觉内容（原始物理量，裸编码）
    lang : 自己上一帧的发声（离散 token）
    行动 : 对下一 tick 的预测（发声）
    裁定 : 世界下一 tick 揭示真相 → 满足度回流

- **图片**：扫视重放——光栅扫描 patch 序列，发声 = 量化亮度等级
  （16 键），预测下一个注视点的亮度 = 图像局部统计的反应化。
- **电影**：帧差 event 信道——合成物理世界（弹球），vis = 当前帧
  （球体格点），aud = 帧差（变化的格点，对应视网膜瞬变细胞），
  发声 = 下一帧球所在象限（4 键）。恒速弹跳是**逻辑动力学**，
  可从帧差推断——世界模型的反应化。
- **PDF**：文本层走字节阅读器（deposit_reader 已覆盖任意文件）；
  结构化抽取（pypdf 可用时）后续接入。

架构侧（DevelopmentalSystem）零改动。
"""
from __future__ import annotations

import sys

import numpy as np

import numpy_torch_shim as shim
shim.install()

from mcp.developmental.signal import Signal                      # noqa: E402
from mcp.developmental.system import DevelopmentalSystem         # noqa: E402
from mcp.developmental.reptilian import ReptilianFunction        # noqa: E402


def _np(x):
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy(), dtype=float)
    if hasattr(x, "a"):
        return np.asarray(x.a, dtype=float)
    return np.asarray(x, dtype=float)


class BabbleReflex(ReptilianFunction):
    """通用哭闹反射：随机发声（V 键）"""

    def __init__(self, vocab, seed=0):
        self.V = vocab
        self.rng = np.random.default_rng(seed)

    def get_input_spec(self):
        return {"lang": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        out = np.zeros(self.V)
        out[int(self.rng.integers(self.V))] = 1.0
        return {"lang": Signal(data=shim.tensor(out),
                               mime_type="generic/tensor",
                               metadata={"source": "babble"})}


# ---------------------------------------------------------------------------
# 图片：扫视重放（光栅扫描 patch，量化亮度为 16 键）
# ---------------------------------------------------------------------------

class ImageDeposit:
    """图片 → 扫视 patch 流。发声 = 当前 patch 的量化亮度等级。"""

    LEVELS = 16
    PATCH = 4          # 4×4 patch = 16 维原始像素（裸编码物理量）

    def __init__(self, source=None, seed=0, grid=32):
        rng = np.random.default_rng(seed)
        if source is not None:
            try:
                from PIL import Image
                img = Image.open(source).convert("L")
                img = img.resize((grid, grid))
                self.img = np.asarray(img, dtype=float) / 255.0
            except Exception:
                self.img = self._synthetic(grid, rng)
        else:
            self.img = self._synthetic(grid, rng)
        self.n_patch = grid // self.PATCH
        self.last_utter = None
        self.patch_idx = 0
        self.rng = np.random.default_rng(seed + 7)

    @staticmethod
    def _synthetic(grid, rng):
        """合成图：平滑噪声 + 亮块（有可学结构）"""
        base = rng.normal(0.4, 0.08, (grid, grid))
        base[4:12, 4:12] = 0.9          # 亮块
        base[20:28, 18:30] = 0.1        # 暗块
        return np.clip(base, 0, 1)

    def _patch_level(self, idx):
        r, c = divmod(idx, self.n_patch)
        patch = self.img[r * self.PATCH:(r + 1) * self.PATCH,
                         c * self.PATCH:(c + 1) * self.PATCH]
        return int(np.clip(patch.mean() * self.LEVELS, 0, self.LEVELS - 1))

    def cur_token(self):
        return self._patch_level(self.patch_idx)

    def cur_patch(self):
        r, c = divmod(self.patch_idx, self.n_patch)
        return self.img[r * self.PATCH:(r + 1) * self.PATCH,
                        c * self.PATCH:(c + 1) * self.PATCH].ravel()

    def _advance(self):
        self.patch_idx = (self.patch_idx + 1) % (self.n_patch ** 2)

    def observe(self):
        V = self.LEVELS
        P = self.PATCH * self.PATCH
        D = P + V + V          # vis 原始像素 + disp 当前等级表盘 + lang
        v = np.zeros(D)
        v[:P] = self.cur_patch()                     # 原始像素（裸）
        v[P + self.cur_token()] = 1.0                # 亮度表盘（当前态冗余编码）
        if self.last_utter is not None:
            v[P + V + self.last_utter] = 1.0
        return v

    def port(self):
        V = self.LEVELS
        P = self.PATCH * self.PATCH
        return {"vis": (0, P), "disp": (P, V), "lang": (P + V, V)}, P + V + V

    def output_channels(self):
        return {"lang"}


# ---------------------------------------------------------------------------
# 电影：合成物理世界（弹球）+ 帧差 event 信道，预测下一帧象限
# ---------------------------------------------------------------------------

class MovieDeposit:
    """弹球世界 → 帧流。vis = 当前帧（球体格点），aud = 帧差格点
    （event 信道），发声 = 球下一帧所在象限（4 键）。"""

    QUADS = 4

    def __init__(self, seed=0, grid=16):
        self.grid = grid
        self.rng = np.random.default_rng(seed + 3)
        self.pos = np.array([4.0, 4.0])
        self.vel = np.array([0.7, 0.5])
        self.prev_cells = self._cells()
        self.last_utter = None

    def _cells(self):
        c = np.zeros((self.grid, self.grid))
        x, y = int(self.pos[0]), int(self.pos[1])
        c[y, x] = 1.0
        return c

    def _quad(self):
        x, y = self.pos
        return int((x >= self.grid / 2)) + 2 * int((y >= self.grid / 2))

    def cur_token(self):
        return self._quad()

    def cur_frame(self):
        return self._cells()

    def cur_diff(self):
        return np.abs(self._cells() - self.prev_cells)

    def _advance(self):
        self.prev_cells = self._cells()
        self.pos = self.pos + self.vel
        for i in range(2):
            if self.pos[i] < 0 or self.pos[i] > self.grid - 1:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 0, self.grid - 1)

    def observe(self):
        g2 = self.grid * self.grid
        D = g2 + g2 + self.QUADS + self.QUADS   # vis + 帧差 + disp + lang
        v = np.zeros(D)
        v[:g2] = self.cur_frame().ravel()
        v[g2:g2 + g2] = self.cur_diff().ravel()
        v[g2 + g2 + self.cur_token()] = 1.0     # 象限表盘（当前态冗余编码）
        if self.last_utter is not None:
            v[g2 + g2 + self.QUADS + self.last_utter] = 1.0
        return v

    def port(self):
        g2 = self.grid * self.grid
        q = self.QUADS
        return {"vis": (0, g2), "aud": (g2, g2),
                "disp": (g2 + g2, q), "lang": (g2 + g2 + q, q)}, \
            g2 * 2 + q * 2

    def output_channels(self):
        return {"lang"}


# ---------------------------------------------------------------------------
# 通用运行器（与 baby_grammar/deposit_reader 同一反应协议）
# ---------------------------------------------------------------------------

def build_brain(port, global_dim, vocab, seed=0, **kw):
    brain = DevelopmentalSystem(port_layout=port, decay_rate=0.01, **kw)
    brain.phase = "exploratory"
    brain.autoreg_lr = 0.05
    brain.autoreg_lr_tau = 5000.0
    brain.autoreg_w_norm = 3.0
    brain.reflex._kernel.register("babble", BabbleReflex(vocab, seed))
    brain.reflex.add_preorchestrated_route(source="lang", target="babble",
                                           priority=1.0)
    n0 = brain.higher_brain.ecosystem.get_neuron(0)
    for c, (off, sz) in port.items():
        n0.unfold(c, shim.zeros(global_dim))
    brain.observe_output_channel = "lang"
    brain.world_display_channel = "disp"    # 表盘信道 = 感知情境/结算键
    brain.higher_brain.symbolic_channels = {"lang"}
    # 小词表媒体：接管证据门槛放低 + 探索率调高（否则 2% 背景探索
    # 喂不饱 competence，高层永远锁在影子里——多肢体实验的教训）
    brain.explore_eps = 0.2
    brain.drive_sat.min_samples = 8
    return brain


def decode(vec, vocab, lang_off):
    if vec is None:
        return None
    a = _np(vec).ravel()
    # 兼容两种尺寸：分通道输出（vocab 维）与全局拼接（lang_off+vocab）
    if a.size == vocab:
        seg = a
    elif a.size >= lang_off + vocab:
        seg = a[lang_off:lang_off + vocab]
    else:
        return None
    return int(np.argmax(seg)) if seg.max() > 0.05 else None


def run_media(world, seed=0, steps=6000, sleep_every=400, sleep_len=60,
              **kw):
    np.random.seed(seed)
    shim.manual_seed(seed)
    port, gdim = world.port()
    vocab = getattr(world, "LEVELS", 0) or getattr(world, "QUADS", 0)
    brain = build_brain(port, gdim, vocab, seed, **kw)
    lang_off = port["lang"][0]
    results = []
    prev = None
    for t in range(steps):
        r = None
        if t > 0:
            world._advance()
            r = 1.0 if prev == world.cur_token() else 0.0
            brain.report_drive_satisfaction(r)
            results.append(r)
        obs = world.observe()
        inputs = {c: Signal(shim.tensor(obs[off:off + sz].copy()),
                            "generic/tensor", {"t": t})
                  for c, (off, sz) in port.items()}
        res = brain.step_awake(external_inputs=inputs)
        out = (res.get("outputs") or {}).get("lang")
        prev = decode(out.data if out is not None else None, vocab, lang_off) \
            if out is not None else None
        world.last_utter = prev
        if sleep_every and (t + 1) % sleep_every == 0:
            for _ in range(sleep_len):
                brain.step_sleep()
    arr = np.array(results, dtype=float)
    tail = arr[len(arr) // 2:]
    return dict(top1=float(tail.mean()), top1_all=float(arr.mean()),
                curve=[float(arr[i:i + len(arr) // 10].mean())
                       for i in range(0, len(arr), len(arr) // 10)],
                neurons=len(brain.higher_brain.ecosystem.neurons))


def main():
    print("=" * 88)
    print("T7 重放资料扩展：同一架构，三类沉积物")
    print("=" * 88)

    print("\n── 图片（扫视重放，16 键亮度）──")
    r = run_media(ImageDeposit(seed=0), seed=0, steps=6000)
    print(f"  top1(后半)={r['top1']:.3f} (all {r['top1_all']:.3f})  "
          f"uniform={1/16:.3f}  神经元={r['neurons']}")
    print("  曲线: " + " ".join(f"{v:.2f}" for v in r["curve"]))

    print("\n── 电影（弹球帧差 event 信道，4 键象限）──")
    r = run_media(MovieDeposit(seed=0), seed=0, steps=6000)
    print(f"  top1(后半)={r['top1']:.3f} (all {r['top1_all']:.3f})  "
          f"uniform={1/4:.3f}  神经元={r['neurons']}")
    print("  曲线: " + " ".join(f"{v:.2f}" for v in r["curve"]))

    print("\n── PDF/任意文件（字节阅读器已覆盖，见 deposit_reader.py）──")
    print("判据: 两类新媒体 top1 显著高于均匀随机 = 同一架构零改动"
          "读新沉积物")


if __name__ == "__main__":
    main()
