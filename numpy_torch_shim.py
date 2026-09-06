"""numpy 后端的 torch 兼容层（沙盒无 torch，用于运行真实代码）

**这不是 torch 的完整替代**，只覆盖本项目实际用到的 API 面
（见下方 COVERED）。目的是让真实代码能在无 torch 环境下跑起来，
从而把"静态检查"升级为"实际运行"。

语义对齐 torch 的关键点：
- ``Tensor`` 可变，``+=`` / ``index_put_`` 等原地操作影响自身
- ``@`` 是矩阵乘，``*`` 是逐元素
- ``.norm()`` 返回 0 维 Tensor（需再 ``.item()``）
- ``.split(sizes, dim)`` 按尺寸切分
"""


class Size(tuple):
    """模拟 torch.Size（tuple 子类，支持 .size(0) 这种调用）"""

    def __call__(self, i):
        return self[i]


class Tensor:
    __slots__ = ("a",)

    def __init__(self, a):
        import numpy as np
        self.a = np.asarray(a)

    # ---------- 形状 ----------
    @property
    def shape(self):
        return Size(self.a.shape)

    def size(self, i=None):
        return Size(self.a.shape) if i is None else self.a.shape[i]

    def numel(self):
        return int(self.a.size)

    def dim(self):
        return int(self.a.ndim)

    def reshape(self, *args):
        return Tensor(self.a.reshape(*args))

    def view(self, *args):
        return Tensor(self.a.reshape(*args))

    def flatten(self, start_dim=0, end_dim=-1):
        import numpy as np
        return Tensor(self.a.reshape(-1))

    def ravel(self):
        return Tensor(self.a.reshape(-1))

    def squeeze(self, i=None):
        return Tensor(self.a.squeeze() if i is None else self.a.squeeze(i))

    def unsqueeze(self, i):
        return Tensor(np.expand_dims(self.a, i))

    def transpose(self, i, j):
        return Tensor(np.swapaxes(self.a, i, j))

    def t(self):
        return Tensor(self.a.T)

    def outer(self, o):
        import numpy as np
        return Tensor(np.outer(self.a.ravel(), (o.a if isinstance(o, Tensor) else o).ravel()))

    # ---------- 归约 ----------
    def item(self):
        return float(self.a) if self.a.ndim == 0 else float(self.a.flat[0])

    def sum(self, dim=None, keepdim=False):
        import numpy as np
        if dim is None:
            return Tensor(np.array(self.a.sum()))
        return Tensor(self.a.sum(axis=dim, keepdims=keepdim))

    def mean(self, dim=None, keepdim=False):
        import numpy as np
        if dim is None:
            return Tensor(np.array(self.a.mean()))
        return Tensor(self.a.mean(axis=dim, keepdims=keepdim))

    def max(self, dim=None, keepdim=False):
        import numpy as np
        if dim is None:
            return Tensor(np.array(self.a.max()))
        return Tensor(self.a.max(axis=dim, keepdims=keepdim))

    def min(self, dim=None, keepdim=False):
        import numpy as np
        if dim is None:
            return Tensor(np.array(self.a.min()))
        return Tensor(self.a.min(axis=dim, keepdims=keepdim))

    def std(self, dim=None):
        import numpy as np
        return Tensor(np.array(self.a.std())) if dim is None \
            else Tensor(self.a.std(axis=dim))

    def norm(self, p=2, dim=None):
        import numpy as np
        if dim is None:
            return Tensor(np.array(float(np.linalg.norm(self.a.ravel(), ord=p))))
        return Tensor(np.linalg.norm(self.a, ord=p, axis=dim))

    def abs(self):
        import numpy as np
        return Tensor(np.abs(self.a))

    def sqrt(self):
        import numpy as np
        return Tensor(np.sqrt(np.maximum(self.a, 0)))

    def clamp(self, lo=None, hi=None):
        import numpy as np
        return Tensor(np.clip(self.a, lo, hi))

    def sign(self):
        import numpy as np
        return Tensor(np.sign(self.a))

    # ---------- 原地 ----------
    def zero_(self):
        self.a = 0 * self.a
        return self

    def fill_(self, v):
        import numpy as np
        self.a = np.full_like(self.a, v)
        return self

    def copy_(self, other):
        import numpy as np
        self.a = np.array(other.a if isinstance(other, Tensor) else other)
        return self

    def add_(self, other, alpha=1.0):
        self.a = self.a + alpha * (other.a if isinstance(other, Tensor) else other)
        return self

    def mul_(self, other):
        self.a = self.a * (other.a if isinstance(other, Tensor) else other)
        return self

    def sub_(self, other):
        self.a = self.a - (other.a if isinstance(other, Tensor) else other)
        return self

    def normal_(self, mean=0.0, std=1.0):
        import numpy as np
        self.a = np.random.normal(mean, std, size=self.a.shape)
        return self

    def uniform_(self, lo=0.0, hi=1.0):
        import numpy as np
        self.a = np.random.uniform(lo, hi, size=self.a.shape)
        return self

    # ---------- 拷贝 / 类型 ----------
    def clone(self):
        import numpy as np
        return Tensor(self.a.copy())

    def detach(self):
        return Tensor(self.a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def tolist(self):
        return self.a.tolist()

    def float(self):
        import numpy as np
        return Tensor(self.a.astype(np.float64))

    def long(self):
        import numpy as np
        return Tensor(self.a.astype(np.int64))

    def int(self):
        import numpy as np
        return Tensor(self.a.astype(np.int64))

    def to(self, *a, **k):
        return self

    def new_zeros(self, *args):
        import numpy as np
        return Tensor(np.zeros(args if len(args) > 1 else args[0]))

    def new_ones(self, *args):
        import numpy as np
        return Tensor(np.ones(args if len(args) > 1 else args[0]))

    # ---------- 切分 / 索引 ----------
    def split(self, split_size_or_sections, dim=0):
        import numpy as np
        if isinstance(split_size_or_sections, int):
            n = self.a.shape[dim]
            sizes = [split_size_or_sections] * (n // split_size_or_sections)
            if n % split_size_or_sections:
                sizes.append(n % split_size_or_sections)
        else:
            sizes = list(split_size_or_sections)
        out, cur = [], 0
        for s in sizes:
            idx = [slice(None)] * self.a.ndim
            idx[dim] = slice(cur, cur + s)
            out.append(Tensor(self.a[tuple(idx)]))
            cur += s
        return out

    def chunk(self, n, dim=0):
        import numpy as np
        return [Tensor(x) for x in np.array_split(self.a, n, axis=dim)]

    def index_select(self, dim, index):
        import numpy as np
        idx = index.a if isinstance(index, Tensor) else np.asarray(index)
        return Tensor(np.take(self.a, idx, axis=dim))

    def sort(self, dim=-1, descending=False):
        import numpy as np
        s = np.sort(self.a, axis=dim)
        if descending:
            s = np.flip(s, axis=dim)
        idx = np.argsort(self.a, axis=dim)
        return Tensor(s), Tensor(idx.astype(np.int64))

    def topk(self, k, dim=-1):
        idx = np.argsort(self.a, axis=dim)[::-1][:k]
        return Tensor(np.take(self.a, idx, axis=dim)), Tensor(idx.astype(np.int64))

    def argmax(self, dim=None):
        import numpy as np
        return Tensor(np.array(np.argmax(self.a))) if dim is None \
            else Tensor(np.argmax(self.a, axis=dim).astype(np.int64))

    def argmin(self, dim=None):
        import numpy as np
        return Tensor(np.array(np.argmin(self.a))) if dim is None \
            else Tensor(np.argmin(self.a, axis=dim).astype(np.int64))

    def __getitem__(self, k):
        import numpy as np
        if isinstance(k, Tensor):
            k = k.a
        if isinstance(k, tuple):
            k = tuple(x.a if isinstance(x, Tensor) else x for x in k)
        return Tensor(self.a[k])

    def __setitem__(self, k, v):
        import numpy as np
        v = v.a if isinstance(v, Tensor) else v
        if isinstance(k, Tensor):
            k = k.a
        if isinstance(k, tuple):
            k = tuple(x.a if isinstance(x, Tensor) else x for x in k)
        self.a[k] = v

    # ---------- 运算 ----------
    def __matmul__(self, o):
        return Tensor(self.a @ (o.a if isinstance(o, Tensor) else o))

    def __rmatmul__(self, o):
        return Tensor((o.a if isinstance(o, Tensor) else o) @ self.a)

    def __mul__(self, o):
        return Tensor(self.a * (o.a if isinstance(o, Tensor) else o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return Tensor(self.a / (o.a if isinstance(o, Tensor) else o))

    def __add__(self, o):
        return Tensor(self.a + (o.a if isinstance(o, Tensor) else o))

    __radd__ = __add__

    def __sub__(self, o):
        return Tensor(self.a - (o.a if isinstance(o, Tensor) else o))

    def __rsub__(self, o):
        return Tensor((o.a if isinstance(o, Tensor) else o) - self.a)

    def __neg__(self):
        return Tensor(-self.a)

    def __pow__(self, o):
        return Tensor(self.a ** o)

    def __eq__(self, o):
        return Tensor(self.a == (o.a if isinstance(o, Tensor) else o))

    def __lt__(self, o):
        return Tensor(self.a < (o.a if isinstance(o, Tensor) else o))

    def __gt__(self, o):
        return Tensor(self.a > (o.a if isinstance(o, Tensor) else o))

    def __le__(self, o):
        return Tensor(self.a <= (o.a if isinstance(o, Tensor) else o))

    def __ge__(self, o):
        return Tensor(self.a >= (o.a if isinstance(o, Tensor) else o))

    def __len__(self):
        return len(self.a)

    def __iter__(self):
        return iter([Tensor(x) for x in self.a])

    def __repr__(self):
        return f"Tensor({self.a!r})"

    def __float__(self):
        return float(self.a)

    def __int__(self):
        return int(self.a)

    def __index__(self):
        return int(self.a)

    def __bool__(self):
        return bool(self.a)


# ---------------------------------------------------------------------------
# 模块级函数
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

float32 = np.float64
float64 = np.float64
long = np.int64
int64 = np.int64


def tensor(x, **k):
    dt = k.get("dtype")
    if dt is not None:
        dt = dt if not isinstance(dt, str) else getattr(np, dt)
        return Tensor(np.asarray(x, dtype=dt))
    return Tensor(np.asarray(x, dtype=np.float64))


def _shape(args, k):
    if "shape" in k:
        return k["shape"]
    if len(args) == 1:
        return args[0] if not isinstance(args[0], int) else (args[0],)
    return args


def zeros(*args, **k):
    return Tensor(np.zeros(_shape(args, k)))


def ones(*args, **k):
    return Tensor(np.ones(_shape(args, k)))


def zeros_like(x, **k):
    return Tensor(np.zeros_like(x.a))


def ones_like(x, **k):
    return Tensor(np.ones_like(x.a))


def randn(*args, generator=None, **k):
    if generator is not None:
        rng = generator
    else:
        rng = np.random.default_rng()
    return Tensor(rng.normal(size=args[0] if len(args) == 1 else args))


def randn_like(x, **k):
    return Tensor(np.random.normal(size=x.a.shape))


def rand(*args, **k):
    return Tensor(np.random.random(args[0] if len(args) == 1 else args))


def randint(lo, hi, size=None, **k):
    return Tensor(np.random.randint(lo, hi, size=size))


def sqrt(x):
    return Tensor(np.sqrt(x.a if isinstance(x, Tensor) else x))


def abs(x):
    return Tensor(np.abs(x.a if isinstance(x, Tensor) else x))


def clamp(x, lo=None, hi=None):
    return Tensor(np.clip(x.a if isinstance(x, Tensor) else x, lo, hi))


def cat(seq, dim=0):
    return Tensor(np.concatenate([t.a for t in seq], axis=dim))


def stack(seq, dim=0):
    return Tensor(np.stack([t.a for t in seq], axis=dim))


def as_tensor(x, **k):
    return Tensor(np.asarray(x, dtype=np.float64))


class Generator:
    """模拟 torch.Generator（用 numpy 的 Generator 提供种子控制）"""

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def manual_seed(self, seed):
        self._rng = np.random.default_rng(seed)
        return self

    def normal(self, size=None, mean=0.0, std=1.0):
        return self._rng.normal(mean, std, size=size)

    def normal_(self, tensor, mean=0.0, std=1.0):
        tensor.a = self._rng.normal(mean, std, size=tensor.a.shape)
        return tensor

    def seed(self):
        return 0


def manual_seed(seed):
    np.random.seed(seed % (2 ** 32))


def no_grad():
    class _C:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return _C()


def save(obj, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path, **k):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def is_tensor(x):
    return isinstance(x, Tensor)


COVERED = """已覆盖：Tensor（含 @ * + - / 索引 split cat norm sum mean 等）、
zeros/ones/zeros_like/randn/randn_like/cat/stack/sqrt/abs/clamp/as_tensor/tensor、
Generator/manual_seed/no_grad/save/load。
未覆盖（本项目未用到）：autograd、CUDA、nn 模块、分布式、大部分 dtype 语义。"""


def install():
    """把本模块注册为 sys.modules['torch']"""
    import sys
    import types
    mod = types.ModuleType("torch")
    for name in dir(_self := __import__(__name__)):
        if name.startswith("_"):
            continue
        setattr(mod, name, getattr(_self, name))
    sys.modules["torch"] = mod
    return mod
