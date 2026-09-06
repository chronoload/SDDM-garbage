"""Layer 4: 路由抽象 — 决定信号从哪个端口流向哪个函数

路由本身不执行函数，只做决策。执行由 ReptilianKernel 或系统主循环负责。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class Route(ABC):
    """路由基类：从源到目标的映射，含优先级和条件门控"""

    @abstractmethod
    def get_source(self) -> str:
        """源端口 ID"""
        pass

    @abstractmethod
    def get_target(self) -> str:
        """目标函数名"""
        pass

    @abstractmethod
    def should_activate(self, context: Dict[str, Any]) -> bool:
        """条件门控：是否应该激活此路由"""
        pass

    @abstractmethod
    def get_priority(self) -> float:
        """优先级（数值越大越优先）"""
        pass


class StaticRoute(Route):
    """静态路由：总是激活，固定优先级"""

    def __init__(self, source_port: str, target_function: str,
                 priority: float = 0.5):
        self._source = source_port
        self._target = target_function
        self._priority = priority

    def get_source(self) -> str:
        return self._source

    def get_target(self) -> str:
        return self._target

    def should_activate(self, context: Dict[str, Any]) -> bool:
        return True

    def get_priority(self) -> float:
        return self._priority


class ConfidenceGatedRoute(Route):
    """置信度门控路由：仅在高层脑置信度低于阈值时激活

    用于成熟期：高层脑置信度高时绕过爬虫脑，低时回退到爬虫脑。
    """

    def __init__(self, source_port: str, target_function: str,
                 threshold: float = 0.5, priority: float = 0.8):
        self._source = source_port
        self._target = target_function
        self._threshold = threshold
        self._priority = priority

    def get_source(self) -> str:
        return self._source

    def get_target(self) -> str:
        return self._target

    def should_activate(self, context: Dict[str, Any]) -> bool:
        confidence = context.get("cortex_confidence", 0.0)
        return confidence < self._threshold

    def get_priority(self) -> float:
        return self._priority


from mcp.developmental.reptilian import ReptilianKernel
from mcp.developmental.signal import Signal


class RouteExecutor:
    """路由执行器 — 按优先级执行激活的路由

    路由的输出信号会加入信号池，供下游路由使用（链式调用）。
    """

    def __init__(self, kernel: ReptilianKernel):
        self._kernel = kernel
        self._routes: list[Route] = []

    def add_route(self, route: Route) -> None:
        self._routes.append(route)

    def execute(
        self,
        input_signals: dict[str, Signal],
        context: dict[str, Any]
    ) -> dict[str, Signal]:
        """执行所有激活路由，返回输出信号池"""
        # 按优先级降序排序
        sorted_routes = sorted(self._routes, key=lambda r: r.get_priority(),
                               reverse=True)
        signal_pool = dict(input_signals)  # 复制，避免污染输入

        for route in sorted_routes:
            if not route.should_activate(context):
                continue
            src = route.get_source()
            if src not in signal_pool:
                continue
            func_name = route.get_target()
            func = self._kernel.get(func_name)
            if func is None:
                continue
            # 构造输入：优先按 input_spec 键名从信号池匹配（支持链式调用，
            # 如上游输出键 "y" 与下游输入键 "y" 同名），匹配不到的键用
            # source 信号回填，仍缺失的键由函数自身校验抛 KeyError
            input_spec = func.get_input_spec()
            inputs = {}
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
