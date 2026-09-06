"""LLM TopK 预筛器 — 反射弧的符号抽象组件

定位：高层脑完全不干预时的"父母呓语"，给原型匹配提供基线候选。
属性：
- 只接收 text/* 信号（非语言模态直接返回空）
- 输出 topk 原型 ID 列表，不产生文本输出
- 可被高层脑髓鞘化后替代
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List
import hashlib

from mcp.developmental.signal import Signal


class LLMTopKScorer(ABC):
    """LLM 预筛器抽象基类（反射弧组件）"""

    @abstractmethod
    def score(self, signal: Signal, proto_descriptions: List[str],
              k: int = 20) -> List[int]:
        """返回 topk 原型索引"""
        pass


class MockLLMScorer(LLMTopKScorer):
    """Mock scorer — 用确定性哈希模拟 LLM 检索

    用于测试和无 LLM 环境下的回退。
    """

    def score(self, signal: Signal, proto_descriptions: List[str],
              k: int = 20) -> List[int]:
        # LLM 只接收语言模态，非文本直接返回空
        if not signal.mime_type.startswith("text/"):
            return []
        if not proto_descriptions:
            return []
        # 用信号 raw 文本 + 描述列表的哈希确定性地选 topk
        raw = signal.metadata.get("raw", "")
        combined = raw + "|" + "|".join(proto_descriptions)
        h = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        # 确定性选取 k 个不同的索引
        indices: List[int] = []
        seen: set[int] = set()
        seed = h
        while (len(indices) < min(k, len(proto_descriptions))
               and len(seen) < len(proto_descriptions)):
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            idx = seed % len(proto_descriptions)
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return indices


class FunctionLLMScorer(LLMTopKScorer):
    """将任意 Callable 包装为 scorer

    用于接入真实 LLM API（OpenAI、本地模型等）。
    """

    def __init__(self,
                 scorer_fn: Callable[[Signal, List[str], int], List[int]]):
        self._fn = scorer_fn

    def score(self, signal: Signal, proto_descriptions: List[str],
              k: int = 20) -> List[int]:
        if not signal.mime_type.startswith("text/"):
            return []
        return self._fn(signal, proto_descriptions, k)
