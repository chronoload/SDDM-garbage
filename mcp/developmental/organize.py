"""Layer 8: 器官组织 — 感知器官 + 反射弧装配层 + 主循环组织器

将发育脑（DevelopmentalSystem）与外部世界接通：
- SensoryOrgan：感知器官 ABC，将外界刺激转化为 Signal
- ReflexAssembly：反射弧装配层，编排工具+路由注册到脑内 ReflexArc（脑不感知）
- Organizer：组织器，主循环 driver — 感知器官 poll + 节律切换 + 持续驱动 + 观察窗

器官不改变脑的内部逻辑，只负责"接通"——
脑不知道器官和装配层存在，只看到 Signal 流。

设计原则（spec 2026-08-05-organize.md）：
- 原则 20：器官是身体的延伸，不改变脑内部状态
- 原则 21：反射弧不向脑暴露编排细节
- 原则 22：感知器官声明信道 + MIME
- 原则 23：装配层知道感知器官 + 工具，编排路由注册到脑内 ReflexArc
- 原则 24：动作 = 爬虫脑工具执行，不需要动作器官 ABC
- 原则 25：反馈 = 工具返回值，自动回流信号池
- 原则 26：组织器控制节律，脑不感知时间
- 原则 27：观察窗保留（tick() 返回 outputs）

模态扩展：器官通过 Signal.mime_type（如 vision/rgb、audio/wav、text/zh）+ metadata
扩展模态，不修改 Signal 类本身——符号信道与视觉/听觉信道同级，靠 mime_type 区分。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Optional

from mcp.developmental.signal import Signal
from mcp.developmental.reptilian import ReptilianFunction
from mcp.developmental.system import DevelopmentalSystem, SystemMode


# ============================================================
# 感知器官：外界刺激 → Signal
# ============================================================

class SensoryOrgan(ABC):
    """感知器官 ABC — 将外界刺激转化为 Signal

    每个感知器官对应一种模态/一个信道：
    - 视觉器官 → image/* Signal
    - 听觉器官 → audio/* Signal
    - 符号器官 → text/* Signal

    Organizer 主循环每次 tick 调用 poll()，收集所有非 None 的 Signal 推入脑。

    模态扩展通过 mime_type + metadata 实现，不修改 Signal 类。
    """

    @abstractmethod
    def get_channel(self) -> str:
        """对应 port_layout 中的信道名（如 'vision', 'audio', 'symbol'）"""
        pass

    @abstractmethod
    def get_mime_type(self) -> str:
        """输出 Signal 的 MIME 类型（如 'vision/rgb', 'audio/wav', 'text/zh'）"""
        pass

    @abstractmethod
    def poll(self) -> Optional[Signal]:
        """获取一次感知信号（无刺激返回 None）"""
        pass


class QueueSensoryOrgan(SensoryOrgan):
    """内存队列感知器官 — 用于基线测试和单进程仿真

    外部代码调用 push(signal) 推入刺激，Organizer 主循环 poll() 取出。
    """

    def __init__(self, channel: str, mime_type: str, maxlen: int = 100):
        self._channel = channel
        self._mime = mime_type
        self._queue: deque[Signal] = deque(maxlen=maxlen)

    def get_channel(self) -> str:
        return self._channel

    def get_mime_type(self) -> str:
        return self._mime

    def push(self, signal: Signal) -> None:
        """外部推入刺激信号"""
        self._queue.append(signal)

    def poll(self) -> Optional[Signal]:
        if not self._queue:
            return None
        return self._queue.popleft()


# ============================================================
# 反射弧装配层：编排工具+路由，注册到脑内 ReflexArc
# ============================================================

class ReflexAssembly:
    """反射弧装配层 — 编排工具+路由，注册到脑内 ReflexArc

    职责（spec 原则 23）：
    - 登记感知器官（知道有哪些输入信道）
    - 注册爬虫脑工具（ReptilianFunction，可包装任何函数）到 brain.reptile
    - 编排"感知信道 → 工具"的路由表，注册到 brain.reflex
    - 动态更新编排（新增器官/工具时）

    边界：
    - 不执行反射弧（执行在脑内 ReflexArc.execute）
    - 不调 brain.step_awake（驱动在 Organizer）
    - 脑不感知装配层存在（脑只看到 ReflexArc 的路由）

    动作 = 爬虫脑工具执行（原则 24），不需要 MotorOrgan ABC。
    反馈 = 工具返回值，自动回流信号池（原则 25）。
    """

    def __init__(self, brain: DevelopmentalSystem):
        self.brain = brain
        self.sensory_organs: list[SensoryOrgan] = []
        self._routes: list[dict] = []  # 编排表

    def register_sensory(self, organ: SensoryOrgan) -> None:
        """登记感知器官（知道有哪些输入信道）"""
        self.sensory_organs.append(organ)

    def register_tool(self, name: str, func: ReptilianFunction) -> None:
        """注册爬虫脑工具到 brain.reptile"""
        self.brain.reptile.register(name, func)

    def add_route(
        self,
        source_channel: str,
        tool_name: str,
        priority: float = 1.0,
    ) -> None:
        """编排路由：感知信道 → 工具 → 注册到 brain.reflex

        脑不感知"这条路由是装配层加的"——它只看到 ReflexArc 的路由。
        """
        self.brain.reflex.add_preorchestrated_route(
            source=source_channel,
            target=tool_name,
            priority=priority,
        )
        self._routes.append({
            "source_channel": source_channel,
            "tool_name": tool_name,
            "priority": priority,
        })

    def remove_route(
        self,
        source_channel: str,
        tool_name: str,
    ) -> None:
        """移除路由（动态更新编排）

        通过清空脑内 ReflexArc 的 _routes + 重建保留的路由实现。
        """
        # 从编排表中移除
        self._routes = [
            r for r in self._routes
            if not (r["source_channel"] == source_channel and r["tool_name"] == tool_name)
        ]
        # 重建脑内 ReflexArc 的预编排路由（走公有接口，不碰私有字段）
        self.brain.reflex.clear_routes()
        for r in self._routes:
            self.brain.reflex.add_preorchestrated_route(
                source=r["source_channel"],
                target=r["tool_name"],
                priority=r["priority"],
            )

    def list_routes(self) -> list[dict]:
        """查看当前编排表"""
        return list(self._routes)


# ============================================================
# 组织器：主循环 driver
# ============================================================

class Organizer:
    """组织器 — 持续主循环 driver

    职责：
    1. 维护感知器官列表
    2. 主循环 tick()：感知器官 poll → 喂脑 step_awake/step_sleep → 返回 outputs（观察窗）
    3. 觉醒/睡眠节律控制（外部时钟，脑不感知时间）
    4. 单步 tick() 可手动驱动（基线测试用），也可 run() 持续运行

    不做的事（原则 20/21/24）：
    - 不修改脑的内部状态（只调 step_awake / step_sleep）
    - 不管动作分发（动作在脑内反射弧执行）
    - 不管反射弧编排（编排是 ReflexAssembly 的职责）
    - 不做反馈通道（反馈 = 工具返回值，脑内自动处理）
    """

    def __init__(self, brain: DevelopmentalSystem):
        """
        Args:
            brain: 被组织的发育脑
        """
        self.brain = brain
        self.sensory_organs: list[SensoryOrgan] = []

        # 节律控制（原则 26）
        self.awake_steps_per_cycle: int = 100
        self.sleep_steps_per_cycle: int = 20
        self._step_in_phase: int = 0
        self.mode: SystemMode = SystemMode.AWAKE

        # 运行统计
        self.tick_count: int = 0
        self.last_result: Optional[dict[str, Any]] = None

    def add_sensory(self, organ: SensoryOrgan) -> None:
        """注册感知器官"""
        self.sensory_organs.append(organ)

    def tick(self) -> dict[str, Any]:
        """单步推进 — 感知 → 脑处理 → 返回 outputs（观察窗）

        返回脑的 step 结果（含 outputs/lambda/...），供外部观察。
        在觉醒态执行 step_awake，在睡眠态执行 step_sleep。
        """
        self.tick_count += 1
        self._step_in_phase += 1

        # 节律切换（原则 26）
        if self.mode == SystemMode.AWAKE and self._step_in_phase >= self.awake_steps_per_cycle:
            self.mode = SystemMode.SLEEP
            self._step_in_phase = 0
        elif self.mode == SystemMode.SLEEP and self._step_in_phase >= self.sleep_steps_per_cycle:
            self.mode = SystemMode.AWAKE
            self._step_in_phase = 0

        # 1. 感知：所有感知器官 poll
        input_signals: dict[str, Signal] = {}
        for organ in self.sensory_organs:
            sig = organ.poll()
            if sig is not None:
                input_signals[organ.get_channel()] = sig

        # 2. 脑处理（动作在脑内反射弧执行，反馈=工具返回值自动回流）
        if self.mode == SystemMode.AWAKE:
            if input_signals:
                result = self.brain.step_awake(external_inputs=input_signals)
            else:
                result = {
                    "outputs": {}, "match_result": {},
                    "lambda": 0.0, "differentiated": -1,
                    "step": self.brain.step_count,
                }
        else:
            if input_signals:
                result = self.brain.step_sleep(external_inputs=input_signals)
            else:
                result = self.brain.step_sleep()

        # 3. 返回 outputs（观察窗，原则 27）
        self.last_result = result
        return result

    def run(
        self,
        max_ticks: Optional[int] = None,
        idle_sleep_sec: float = 0.01,
        on_tick: Optional[Callable[[int, dict[str, Any]], None]] = None,
    ) -> None:
        """持续运行主循环（阻塞）

        Args:
            max_ticks: 最大 tick 数（None 表示无限）
            idle_sleep_sec: 无输入时短暂休眠，避免空转 CPU
            on_tick: 每个 tick 后的回调（用于外部观察/记录）
        """
        n = 0
        while max_ticks is None or n < max_ticks:
            result = self.tick()
            if on_tick is not None:
                on_tick(n, result)
            n += 1
            if not result.get("outputs") and not result.get("sleep_outputs"):
                time.sleep(idle_sleep_sec)

    def force_awake(self) -> None:
        """强制觉醒"""
        self.mode = SystemMode.AWAKE
        self._step_in_phase = 0

    def force_sleep(self) -> None:
        """强制睡眠"""
        self.mode = SystemMode.SLEEP
        self._step_in_phase = 0

    def is_awake(self) -> bool:
        return self.mode == SystemMode.AWAKE

    def is_sleep(self) -> bool:
        return self.mode == SystemMode.SLEEP
