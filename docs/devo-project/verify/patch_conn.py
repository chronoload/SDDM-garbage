"""建立 Connection 层：把「神经元之间的连接」与「包裹它的髓鞘」分开"""
P = "/data/workspace/mcp/developmental/myelin.py"
s = open(P).read()

CONN = '''
@dataclass
class Connection:
    """神经元之间的**轴突连接** —— 共发射成边的产物

    这一层与髓鞘是两回事，此前被错误合并了。

    生物学分层：

    ============  ====================================================
    层次           内容
    ============  ====================================================
    **连接**       轴突从 A 长到 B —— 拓扑问题（"连到哪儿"）。
                  发育期轴突寻路 + 活动依赖的共发射成边。**可以生灭**。
    **髓鞘**       包裹在**已存在**的轴突上 —— 性能问题
                  （"裹不裹绝缘皮"）。后加的、可选的、稳定的增强层。
    ============  ====================================================

    一个少突胶质细胞可包裹**多段**轴突（多个 internode），因此髓鞘
    不是"一条连接一个"的东西。

    未髓鞘的轴突**可以存在且工作**，只是传导慢、信号弱。髓鞘化是后续
    的性能升级，不是连接存在的前提。
    """
    src_neuron: int
    src_channel: str
    dst_neuron: int
    dst_channel: str
    strength: float = 0.2      # 连接强度（用进废退作用于此）
    usage: float = 0.0         # 使用累积 EMA
    co_fire_count: int = 0     # 累计共发射次数（采样确认的计数）
    starving: int = 0          # 连续未被使用的步数
    myelinated: bool = False   # 是否已被髓鞘包裹
    birth_step: int = 0

    # 未髓鞘轴突的传导特性：慢且弱
    UNMYELINATED_DELAY = 6.0
    UNMYELINATED_GAIN = 0.3

    def effective_delay(self, sheath=None) -> float:
        """有效传导延迟：有髓鞘 → 跳跃式传导（快）；无 → 慢"""
        return sheath.delay if sheath is not None else self.UNMYELINATED_DELAY

    def effective_gain(self, sheath=None) -> float:
        """有效增益：有髓鞘 → 强；无 → 弱"""
        return sheath.gain if sheath is not None else self.UNMYELINATED_GAIN


'''

ANCHOR = "class MyelinSheathRegistry:"
assert ANCHOR in s, "registry anchor not found"
s = s.replace(ANCHOR, CONN.lstrip("\n") + ANCHOR, 1)

# registry 中加 _connections
OLD = "        self._sheaths: dict[tuple[int, str, int, str], MyelinSheath] = {}"
if OLD not in s:
    OLD2 = "        self._channel_cred: dict[tuple[int, str], float] = {}"
    assert OLD2 in s
    s = s.replace(OLD2,
        "        # 神经元之间的连接层（与髓鞘层分离）\n"
        "        self._connections: dict[tuple[int, str, int, str], Connection] = {}\n"
        + OLD2, 1)
    print("OK: _connections 已加")
else:
    s = s.replace(OLD, OLD + "\n        self._connections = {}", 1)
    print("OK: _connections 已加（另一锚点）")

open(P, "w").write(s)
print("Connection 类已插入")
