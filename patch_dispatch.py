"""dispatch 改为遍历连接层：未髓鞘轴突也能传导，只是慢且弱"""
P = "/data/workspace/mcp/developmental/myelin.py"
s = open(P).read()

OLD = """            # 沿每个已展开信道分发（并行）
            #
            # ⚠ 事件 data 必须是**全局信号空间**中的向量，长度恒为
            # global_dim。原实现用 ``output * gain``（神经元压缩输出，
            # 长度 = 该神经元已展开维度数），导致三个后果：
            #   1. 不同神经元展开维度数不同 → 后续 sum(e.data) 形状冲突
            #   2. data 是"全部信道的混合"，与 event.channel=dst_ch 不符
            #   3. 取的是整个输出而非 src_ch 对应切片 —— 跨模态传输语义错乱
            # 修复：按 src_ch 切片神经元输出 → scatter 到 dst_ch 的全局位置。
            bounds = dict((c, (s, e)) for c, s, e in neuron._channel_bounds())
            for ch in neuron.unfolded:
                # 查找该信道上的所有髓鞘连接
                for (src_n, src_ch, dst_n, dst_ch), sheath in self.sheaths.items():
                    if src_n == nid and src_ch == ch:
                        arrival = t + sheath.delay
                        transmitted = self._scatter_to_global(
                            output, bounds.get(src_ch), dst_ch, sheath.gain)
                        event = SignalEvent(
                            arrival_time=arrival,
                            target_neuron=dst_n,
                            channel=dst_ch,
                            data=transmitted,
                            source_tag=source_tag,
                            origin_neuron=nid,
                            sheath_key=(src_n, src_ch, dst_n, dst_ch),
                        )
                        self.event_queue.append(event)"""

NEW = """            # 沿每个已展开信道分发（并行）
            #
            # ⚠ 遍历的是**连接层**，不是髓鞘层。
            #
            # 旧实现遍历 ``self.sheaths``，等于"没有髓鞘就没有连接"——
            # 这正是把两层混为一谈的后果：未髓鞘的轴突无法传导，而实际
            # 上未髓鞘轴突**可以工作**，只是慢（delay 6.0）且弱（gain 0.3）。
            # 髓鞘化是后续的性能升级，不是连接存在的前提。
            #
            # ⚠ 事件 data 必须是**全局信号空间**中的向量，长度恒为
            # global_dim（原实现用压缩输出会形状冲突）。
            bounds = dict((c, (s0, e0)) for c, s0, e0 in neuron._channel_bounds())
            for ch in neuron.unfolded:
                for key, conn in self._connections.items():
                    src_n, src_ch, dst_n, dst_ch = key
                    if src_n != nid or src_ch != ch:
                        continue
                    sh = self.sheaths.get(key)
                    arrival = t + conn.effective_delay(sh)
                    transmitted = self._scatter_to_global(
                        output, bounds.get(src_ch), dst_ch, conn.effective_gain(sh))
                    event = SignalEvent(
                        arrival_time=arrival,
                        target_neuron=dst_n,
                        channel=dst_ch,
                        data=transmitted,
                        source_tag=source_tag,
                        origin_neuron=nid,
                        sheath_key=key,
                    )
                    self.event_queue.append(event)"""

assert OLD in s, "dispatch anchor not found"
s = s.replace(OLD, NEW, 1)
open(P, "w").write(s)
print("dispatch 已改为遍历连接层")
