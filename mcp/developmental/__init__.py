"""发育智能系统 — 信号驱动的连续具身发育智能

架构分层（每层只依赖下层）：
  Layer 1: Signal        — 统一信号载体（数据 + 类型 + 元数据）
  Layer 2: Port          — 系统边界端点（入端口/出端口 + 注册表）
  Layer 3: Reptilian     — 爬行脑原子工具集（自完备的预编排系统）
  Layer 4: Route/Reflex  — 路由 + 反射弧（LLM 预筛 + 预编排路由保底）
  Layer 5: Neuron        — 动态张量神经元（自复制分化 + 维度展开/自噬）
  Layer 6: Ecosystem/Myelin — 神经元生态 + 髓鞘（事件驱动并行分发 + 重合检测）
  Layer 7: HigherBrain   — 高层脑（始终活跃 + 连续 λ 残差干预 + 做梦学习）
  LLM Scorer             — LLM top-k 语义预筛（可替换的符号抽象示范）
  System                 — 发育智能系统主循环（觉醒/睡眠双模 + 四相位做梦学习）

核心范式：
- 神经元是动态张量生命体，权重矩阵 W 维数随分化/生长/自噬/髓鞘隔离动态变化
- 算子 = 权重 = 线性变换矩阵 W（seed 给初始值，反馈微调到收敛）
- 髓鞘是 W 的包裹层（delay/gain/protection），不是额外算子
- 信号驱动维度展开（惰性扩张）+ 维度级自噬（低活跃剪除）
- 事件驱动并行分发：各维度切片同时经各自算子传输，时序竞争 + 重合窗口跨模态绑定
- 间歇替代模型：高层脑始终在干扰，强度 λ(t) ∈ [0,1] 连续可变
- 四相位做梦学习：稳态缩放 + 生成重放 + 反事实 + 前向预演（全部髓鞘操作，无梯度）
"""
from mcp.developmental.signal import (
    Signal,
    MultiChannelSignal,
    SignalConverter,
    ConverterRegistry,
)
from mcp.developmental.port import (
    Port,
    InputPort,
    OutputPort,
    DataSource,
    DataSink,
    PortRegistry,
)
from mcp.developmental.reptilian import (
    ReptilianFunction,
    LambdaFunction,
    ReptilianKernel,
    ModalityRouter,
    EchoFunction,
    SaturateFunction,
)
from mcp.developmental.builtin.symbol_channel import SymbolChannelFunction
from mcp.developmental.builtin.web_agent import WebAgentFunction
from mcp.developmental.text_encoder import TextEncoder
from mcp.developmental.route import (
    Route,
    StaticRoute,
    ConfidenceGatedRoute,
    RouteExecutor,
)
from mcp.developmental.reflex import ReflexArc
from mcp.developmental.neuron import (
    Neuron,
    DimSlice,
    ConvergenceState,
)
from mcp.developmental.neurode import NeuronEcosystem
from mcp.developmental.myelin import (
    MyelinSheath,
    MyelinSheathRegistry,
    CoincidenceDetector,
    ActivationRecord,
    SignalDispatcher,
    SignalEvent,
)
from mcp.developmental.development import DevelopmentEngine
from mcp.developmental.higher_brain import (
    HigherBrain,
    DefaultHigherBrain,
)
from mcp.developmental.llm_scorer import (
    LLMTopKScorer,
    MockLLMScorer,
    FunctionLLMScorer,
)
from mcp.developmental.system import (
    DevelopmentalSystem,
    SystemMode,
)
from mcp.developmental.organize import (
    SensoryOrgan,
    QueueSensoryOrgan,
    ReflexAssembly,
    Organizer,
)

__all__ = [
    # Layer 1: Signal
    "Signal", "MultiChannelSignal", "SignalConverter", "ConverterRegistry",
    # Layer 2: Port
    "Port", "InputPort", "OutputPort", "DataSource", "DataSink", "PortRegistry",
    # Layer 3: Reptilian
    "ReptilianFunction", "LambdaFunction", "ReptilianKernel", "ModalityRouter",
    "EchoFunction", "SaturateFunction", "SymbolChannelFunction",
    "WebAgentFunction", "TextEncoder",
    # Layer 4: Route + Reflex
    "Route", "StaticRoute", "ConfidenceGatedRoute", "RouteExecutor",
    "ReflexArc",
    # Layer 5: Neuron
    "Neuron", "DimSlice", "ConvergenceState",
    # Layer 6: NeuronEcosystem + Myelin
    "NeuronEcosystem", "MyelinSheath", "MyelinSheathRegistry",
    "CoincidenceDetector", "ActivationRecord", "SignalDispatcher", "SignalEvent",
    "DevelopmentEngine",
    # Layer 7: HigherBrain
    "HigherBrain", "DefaultHigherBrain",
    # LLM scorer
    "LLMTopKScorer", "MockLLMScorer", "FunctionLLMScorer",
    # System
    "DevelopmentalSystem", "SystemMode",
    # Layer 8: Organize（器官组织）
    "SensoryOrgan", "QueueSensoryOrgan", "ReflexAssembly", "Organizer",
]
