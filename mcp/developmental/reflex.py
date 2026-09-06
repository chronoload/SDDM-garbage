"""反射弧 — LLM + 预编排路由封装

定义：高层脑完全不干预时的基线行为。
属性：
- 预编排路由：按优先级执行的工具链（input → echo → saturate 之类）
- 可整体禁用（高层脑全权接管）
- 可按模态部分禁用（高层脑只接管部分模态）
- 对高层脑透明：高层脑只看到最终输出，不知道是反射弧还是自己产生的
"""
from __future__ import annotations
import fnmatch
from typing import Dict, Any

from mcp.developmental.signal import Signal
from mcp.developmental.reptilian import ReptilianKernel


class _PreorchestratedRoute:
    """预编排路由项：source_key → target_func_name，含优先级"""

    def __init__(self, source: str, target: str, priority: float = 1.0):
        self.source = source
        self.target = target
        self.priority = priority


class ReflexArc:
    """反射弧：高层脑零干预时的基线行为

    由预编排路由 + 可选 LLM 符号信道工具组成。
    可被高层脑整体禁用或按模态部分禁用。
    """

    def __init__(self, kernel: ReptilianKernel):
        self._kernel = kernel
        self._routes: list[_PreorchestratedRoute] = []
        self._disabled_globally = False
        self._disabled_mime_patterns: list[str] = []

    def add_preorchestrated_route(self, source: str, target: str,
                                  priority: float = 1.0) -> None:
        """添加预编排路由"""
        self._routes.append(_PreorchestratedRoute(source, target, priority))

    def clear_routes(self) -> None:
        """清空全部预编排路由（供装配层整体重建编排）

        装配层（organize.py）重建路由时必须走这个公有接口，
        而不是直接操作 ``_routes`` 私有字段——否则反射弧一旦换成
        可替换实现，越界访问会静默失效。
        """
        self._routes.clear()

    def disable(self) -> None:
        """整体禁用（高层脑全权接管）"""
        self._disabled_globally = True

    def enable(self) -> None:
        """整体启用"""
        self._disabled_globally = False
        self._disabled_mime_patterns.clear()

    def disable_for_mime(self, mime_pattern: str) -> None:
        """按模态禁用（高层脑只接管该模态）"""
        self._disabled_mime_patterns.append(mime_pattern)

    def _is_disabled_for(self, mime_type: str) -> bool:
        if self._disabled_globally:
            return True
        return any(
            fnmatch.fnmatch(mime_type, p)
            for p in self._disabled_mime_patterns
        )

    def execute(
        self,
        input_signals: Dict[str, Signal],
        context: Dict[str, Any],
    ) -> Dict[str, Signal]:
        """执行反射弧：按优先级跑预编排路由

        返回输出信号池。被禁用的模态不参与。
        """
        # 整体禁用 → 直接返回空（高层脑全权接管）
        if self._disabled_globally:
            return {}

        # 初始信号池：过滤掉被禁用模态的输入（高层脑已接管的模态不参与）
        signal_pool: Dict[str, Signal] = {
            k: v for k, v in input_signals.items()
            if not self._is_disabled_for(v.mime_type)
        }
        sorted_routes = sorted(
            self._routes, key=lambda r: r.priority, reverse=True
        )

        for route in sorted_routes:
            src = route.source
            if src not in signal_pool:
                continue
            # 检查该信号模态是否被禁用
            if self._is_disabled_for(signal_pool[src].mime_type):
                continue
            func = self._kernel.get(route.target)
            if func is None:
                continue
            # 构造输入：按键名匹配池 + source 回填
            input_spec = func.get_input_spec()
            inputs: Dict[str, Signal] = {}
            for key in input_spec:
                if key in signal_pool:
                    inputs[key] = signal_pool[key]
            for key in input_spec:
                if key not in inputs:
                    inputs[key] = signal_pool[src]
                    break  # source 信号只回填第一个缺失键
            outputs = func.execute(inputs)
            signal_pool.update(outputs)

        return signal_pool
