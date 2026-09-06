"""神经元生态 — Neuron 对象容器 + 代谢 + 覆盖度匹配 + 自噬

神经元生态管理 Neuron 对象（dict[int, Neuron]）。
分化 = 新增神经元；自噬 = 维度剪除 / 饥饿清除。
匹配 = 覆盖度（已展开维度与信号非零信道的重合度），替代余弦相似度。
"""
from __future__ import annotations
from typing import Optional
import torch
from mcp.developmental.neuron import Neuron


class NeuronEcosystem:
    """神经元生态：管理 Neuron 对象的容器"""

    def __init__(self, port_layout: dict[str, tuple[int, int]],
                 max_nodes: int = 10000):
        """
        Args:
            port_layout: 信道布局 {channel_name: (offset, size)}
            max_nodes: 最大神经元数
        """
        self.port_layout = port_layout
        self.max_nodes = max_nodes

        # Neuron 对象容器
        self.neurons: dict[int, Neuron] = {}
        self._next_idx = 0

        # 种子神经元（seed=0）
        seed_neuron = Neuron(seed=0, port_layout=port_layout)
        self.neurons[self._next_idx] = seed_neuron
        self._next_idx += 1

        # 代谢参数
        self._metabolism_increment = 0.1
        self._metabolism_decay = 0.95

    def count(self) -> int:
        """存活神经元数"""
        return sum(1 for n in self.neurons.values() if n.alive)

    def add_neuron(self, neuron: Neuron) -> int:
        """添加神经元对象，返回索引"""
        if self.count() >= self.max_nodes:
            return -1
        idx = self._next_idx
        self.neurons[idx] = neuron
        self._next_idx += 1
        return idx

    def get_neuron(self, idx: int) -> Optional[Neuron]:
        """获取神经元"""
        return self.neurons.get(idx)

    def get_active_indices(self) -> list[int]:
        """获取存活神经元索引"""
        return [i for i, n in self.neurons.items() if n.alive]

    def kill(self, idx: int) -> None:
        """标记神经元死亡"""
        if idx in self.neurons:
            self.neurons[idx].alive = False

    def tick_metabolism(self, active_indices: list[int]) -> None:
        """代谢更新：活跃神经元升高，不活跃的衰减

        **必须每步调用**。原实现中此方法定义了却从未被调用，导致
        ``metabolism`` 恒为初始值 0.5，而饥饿自噬的判据是
        ``metabolism < 0.01``——该分支永远进不去，饥饿自噬形同虚设。

        代谢是**神经元的存活账本**：被路由选中、真正参与了处理的神经元
        获得代谢供给；长期不被选中的神经元代谢衰减，最终被饥饿自噬清除。
        它不参与"谁更好"的价值判断（那是髓鞘层的事），只回答"有没有被用到"。
        """
        active_set = set(active_indices)
        for i, n in self.neurons.items():
            if not n.alive:
                continue
            if i in active_set:
                n.metabolism = min(1.0, n.metabolism + self._metabolism_increment)
            else:
                n.metabolism *= self._metabolism_decay
            # 钳制下界
            n.metabolism = max(0.0, n.metabolism)

    def ensure_channel_coverage(self, channels: set[str]) -> dict[int, list[str]]:
        """保证生态层面每个活跃信道至少被一个神经元展开（维度展开的主入口）

        **原实现中 ``unfold`` 从未被主循环调用**——只有 ``replicate`` 内部
        和 cli/gui 的手动初始化会调它。后果：一个新信道（新的传感器、新的
        模态）出现时，系统不会为它展开维度，对它完全是瞎的，只能等某次
        分化时子节点顺带继承。

        **为什么不给所有神经元都展开所有信道**：那会让 ``find_coverage``
        的覆盖度恒为 1.0 → 新奇性恒为 0 → **分化永不触发**。分化依赖的就是
        "单个神经元覆盖不全"这一事实。

        因此这里只保证**生态层面**的覆盖（至少有一个神经元能处理该信道），
        不消除**个体层面**的覆盖差异。分化机制因此完好。

        展开是活动非依赖的（初始权重由 seed 决定，对应轴突寻路的先天布线），
        所以选哪个神经元展开不会引入信号依赖的偏置。这里选已展开维度最多
        的（最"通用"的）那个——对应"初期由一个神经元处理所有信号，
        后来才分化特化"的发育顺序。

        Args:
            channels: 本步出现的活跃信道名集合

        Returns:
            {neuron_idx: [新展开的信道]}（只含真正新增的）
        """
        result: dict[int, list[str]] = {}
        alive = [i for i, n in self.neurons.items() if n.alive]
        if not alive:
            return result
        for ch in channels:
            if ch not in self.port_layout:
                continue
            # 已有神经元覆盖该信道 → 不动（保留个体覆盖差异）
            if any(ch in n.unfolded for n in self.neurons.values() if n.alive):
                continue
            # 无覆盖 → 交给最通用的神经元（已展开维度最多）
            best = max(alive, key=lambda i: len(self.neurons[i].unfolded))
            newly = self.neurons[best].unfold_channels([ch])
            if newly:
                result.setdefault(best, []).extend(newly)
        return result

    def record_dim_feedback(
        self,
        residual: torch.Tensor,
        active_indices: Optional[list[int]] = None,
    ) -> dict[int, dict[str, float]]:
        """维度级用进废退——对本步参与的神经元转发反馈

        Args:
            residual: 全局信号空间中的残差
            active_indices: 本步参与处理的神经元索引（None = 全部存活）

        Returns:
            {neuron_idx: {channel: utility}}
        """
        targets = (self.get_active_indices() if active_indices is None
                   else [i for i in active_indices
                         if i in self.neurons and self.neurons[i].alive])
        out: dict[int, dict[str, float]] = {}
        for idx in targets:
            u = self.neurons[idx].record_dim_feedback(residual)
            if u:
                out[idx] = u
        return out

    def find_coverage(self, signal: torch.Tensor,
                      active_channels: set[str]) -> tuple[int, float]:
        """覆盖度匹配——找已展开维度与信号非零信道重合度最高的神经元

        coverage_score = 重合信道数 / 信号非零信道数
        完全覆盖时 coverage=1.0。
        """
        if not active_channels:
            return -1, 0.0

        best_idx = -1
        best_score = -1.0
        for idx in self.get_active_indices():
            neuron = self.neurons[idx]
            unfolded_channels = set(neuron.unfolded.keys())
            overlap = len(unfolded_channels & active_channels)
            score = overlap / len(active_channels)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            return -1, 0.0
        return best_idx, best_score

    def autophagy_dims(
        self,
        threshold: float = 0.1,
        gain_threshold: float = 0.05,
        min_age: int = 100,
    ) -> dict[int, list[str]]:
        """维度级自噬——对所有存活神经元调用 neuron.autophagy_dims

        Args:
            threshold: 输入侧断流判据（activity 下界）
            gain_threshold: 退化完成判据（gain 下界），见 :meth:`Neuron.autophagy_dims`
            min_age: 新生儿保护期

        Returns:
            {neuron_idx: [pruned_channels]}
        """
        result: dict[int, list[str]] = {}
        for idx, neuron in self.neurons.items():
            if not neuron.alive:
                continue
            pruned = neuron.autophagy_dims(
                threshold=threshold,
                gain_threshold=gain_threshold,
                min_age=min_age,
            )
            if pruned:
                result[idx] = pruned
        return result

    def autophagy_starvation(self, starvation_threshold: float = 0.01,
                             protection_threshold: float = 0.1) -> list[int]:
        """饥饿自噬——代谢低于阈值且保护系数低的神经元被 kill

        保护系数取神经元已展开维度的最大 protection（髓鞘庇护）。
        返回被杀的索引列表。
        """
        killed: list[int] = []
        for idx, neuron in self.neurons.items():
            if not neuron.alive:
                continue
            if neuron.metabolism < starvation_threshold:
                # 神经元保护系数 = 已展开维度的最大 protection
                if neuron.unfolded:
                    max_protection = max(
                        s.protection for s in neuron.unfolded.values())
                else:
                    max_protection = 0.0
                if max_protection < protection_threshold:
                    neuron.alive = False
                    killed.append(idx)
        return killed
