"""文本编码器 — 用分词器把文本转成 token id 张量

Signal 协议要求 data 是 torch.Tensor。文本信号不能只塞占位符 [0.0]，
必须经过分词器编码，这样：
  - 反射弧工具能从 data 解码还原文本（或直接读 metadata["raw"]）
  - 高层脑能通过 embed() 拿到定长 float 向量处理
  - token id 序列本身携带语义粒度（比 hash 占位符有意义的得多）

编码器选择：
  1. tiktoken（cl100k_base，gpt-4o 系列分词器）— 项目已有依赖
  2. fallback: UTF-8 字节级编码（不依赖 tiktoken 也能跑）

单例模式：tiktoken 首次加载有开销（~100ms），全局复用。
"""
from __future__ import annotations

import torch
from typing import Optional


class TextEncoder:
    """文本 ↔ token id 张量 编码器

    用法：
        enc = TextEncoder()
        token_ids = enc.encode("你好")      # tensor([151770, 57668], dtype=torch.long)
        text = enc.decode(token_ids)         # "你好"
        vec = enc.embed("你好", dim=64)      # 定长 float 向量（供高层脑）
    """

    _instance: Optional["TextEncoder"] = None

    def __init__(self, model: str = "gpt-4o"):
        self._model = model
        self._enc = None
        self._backend = "none"
        try:
            import tiktoken
            try:
                self._enc = tiktoken.encoding_for_model(model)
                self._backend = f"tiktoken:{self._enc.name}"
            except KeyError:
                self._enc = tiktoken.get_encoding("cl100k_base")
                self._backend = "tiktoken:cl100k_base"
        except ImportError:
            self._backend = "utf8-bytes"  # fallback

    @classmethod
    def instance(cls) -> "TextEncoder":
        """全局单例（避免重复加载分词器）"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, text: str) -> torch.Tensor:
        """编码文本 → token id 张量（torch.long，变长）

        Args:
            text: 输入文本
        Returns:
            1D long 张量，每个元素是一个 token id
        """
        if self._enc is not None:
            ids = self._enc.encode(text)
        else:
            # fallback: UTF-8 字节级（每字节一个 id，通用且无损）
            ids = list(text.encode("utf-8"))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """解码 token id 张量 → 文本

        Args:
            token_ids: 1D long 张量
        Returns:
            还原的文本
        """
        ids = token_ids.flatten().tolist()
        if self._enc is not None:
            return self._enc.decode(ids)
        else:
            return bytes(ids).decode("utf-8", errors="replace")

    def embed(self, text: str, dim: int = 64) -> torch.Tensor:
        """bag-of-tokens 嵌入 → 定长 float 向量

        把变长 token id 序列压缩成定长向量，供高层脑（需要固定维度）处理。
        方法：每个 token id 取模映射到 dim 维的一个槽位，累加后 L2 归一化。
        相似文本 token 重合多 → 向量接近（余弦相似度有意义）。

        这不是预训练嵌入，但比 hash 占位符有语义意义，且零依赖、确定性。
        若需真实嵌入，可自行接入外部 embedding 服务（sentence-transformers 已因 numpy 版本冲突移除）。

        Args:
            text: 输入文本
            dim: 输出维度（需与 port_layout 中 text 信道的 size 一致）
        Returns:
            定长 float32 向量，L2 归一化
        """
        ids = self.encode(text)
        vec = torch.zeros(dim, dtype=torch.float32)
        for tid in ids.tolist():
            vec[tid % dim] += 1.0
        norm = vec.norm()
        if norm > 0:
            vec /= norm
        return vec

    def embed_tokens(self, token_ids: torch.Tensor, dim: int = 64) -> torch.Tensor:
        """从 token id 张量生成定长嵌入（与 embed 等价，但跳过编码步骤）"""
        vec = torch.zeros(dim, dtype=torch.float32)
        for tid in token_ids.flatten().tolist():
            vec[tid % dim] += 1.0
        norm = vec.norm()
        if norm > 0:
            vec /= norm
        return vec
