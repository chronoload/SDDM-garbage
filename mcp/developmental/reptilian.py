"""Layer 3: 爬虫脑原子工具层

每个 ReptilianFunction 是一个独立的、有明确 I/O 契约的原子操作。
工具间无预设关系，组合方式由高层脑髓鞘决定。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict
import fnmatch

import torch

from mcp.developmental.signal import Signal


class ReptilianFunction(ABC):
    """爬虫脑原子操作 — 完全冻结，任意复杂

    用户扩展点：继承此类，实现 execute() 和规格声明。
    """

    @abstractmethod
    def get_input_spec(self) -> Dict[str, str]:
        """声明输入: {参数名: MIME类型}"""
        pass

    @abstractmethod
    def get_output_spec(self) -> Dict[str, str]:
        """声明输出: {返回值名: MIME类型}"""
        pass

    @abstractmethod
    def execute(self, inputs: Dict[str, Signal]) -> Dict[str, Signal]:
        """核心执行（内部可以是 API、本地代码、任何东西）"""
        pass


class LambdaFunction(ReptilianFunction):
    """将普通 Python 函数包装成 ReptilianFunction"""

    def __init__(
        self,
        func: Callable[[Dict[str, Signal]], Dict[str, Signal]],
        input_spec: Dict[str, str],
        output_spec: Dict[str, str],
    ):
        self._func = func
        self._input_spec = input_spec
        self._output_spec = output_spec

    def get_input_spec(self) -> Dict[str, str]:
        return self._input_spec

    def get_output_spec(self) -> Dict[str, str]:
        return self._output_spec

    def execute(self, inputs: Dict[str, Signal]) -> Dict[str, Signal]:
        # 校验输入键
        for key in self._input_spec:
            if key not in inputs:
                raise KeyError(f"Missing required input: {key}")
        return self._func(inputs)


class ReptilianKernel:
    """爬虫脑内核 — 原子工具注册表

    只负责注册和查找，不负责路由决策（那是 Layer 4 的职责）。
    工具间无预设关系，组合方式由高层脑髓鞘决定。
    """

    def __init__(self):
        self._functions: dict[str, ReptilianFunction] = {}

    def register(self, name: str, func: ReptilianFunction) -> None:
        """注册原子工具"""
        self._functions[name] = func

    def get(self, name: str) -> ReptilianFunction | None:
        """按名查找"""
        return self._functions.get(name)

    def list_functions(self) -> list[str]:
        """列出所有工具名"""
        return list(self._functions.keys())

    def execute(self, name: str, inputs: Dict[str, Signal]) -> Dict[str, Signal]:
        """执行指定工具"""
        func = self._functions.get(name)
        if func is None:
            raise KeyError(f"Unknown function: {name}")
        return func.execute(inputs)


class ModalityRouter:
    """模态分流路由器 — 爬虫脑预编排的核心

    按 MIME 类型把信号分流到对应工具组。
    图像→图像工具，语言→语言工具，张量→张量工具。
    这是白箱编排，不是学习。
    """

    def __init__(self):
        # 绑定列表：[(pattern, target_func_name), ...]
        # 后绑定的优先级更高（允许覆盖通配符）
        self._bindings: list[tuple[str, str]] = []

    def bind(self, mime_pattern: str, target_func_name: str) -> None:
        """绑定 MIME 模式到目标工具名

        支持 fnmatch 通配符：image/* 匹配所有图像类型
        后绑定的优先级更高（允许具体类型覆盖通配符）
        """
        self._bindings.append((mime_pattern, target_func_name))

    def route(self, mime_type: str) -> str | None:
        """按 MIME 类型查找目标工具名

        从后往前匹配，第一个命中的胜出（后绑定优先）
        """
        for pattern, target in reversed(self._bindings):
            if fnmatch.fnmatch(mime_type, pattern):
                return target
        return None


class EchoFunction(ReptilianFunction):
    """输入直通输出，不做任何变换

    用于最小预设路由的基准范例。
    """

    def get_input_spec(self):
        return {"x": "generic/tensor"}

    def get_output_spec(self):
        return {"y": "generic/tensor"}

    def execute(self, inputs):
        x = inputs["x"]
        return {"y": Signal(data=x.data.clone(), mime_type=x.mime_type,
                            metadata=x.metadata)}


class SaturateFunction(ReptilianFunction):
    """限幅器：y = clamp(x, min, max)

    将输出钳制在物理可行范围 [min, max] 内，保证系统安全。
    """

    def __init__(self, y_min: float = -1.0, y_max: float = 1.0):
        self.y_min = y_min
        self.y_max = y_max

    def get_input_spec(self):
        return {"x": "generic/tensor"}

    def get_output_spec(self):
        return {"y": "generic/tensor"}

    def execute(self, inputs):
        x = inputs["x"]
        saturated = torch.clamp(x.data, self.y_min, self.y_max)
        return {"y": Signal(data=saturated, mime_type=x.mime_type,
                            metadata={**x.metadata, "saturated": True})}
