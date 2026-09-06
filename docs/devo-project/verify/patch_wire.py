"""co_fire_wire 改为建立 Connection（而非直接建髓鞘），并加髓鞘化判据"""
P = "/data/workspace/mcp/developmental/myelin.py"
s = open(P).read()

OLD = """        created = 0
        for A, B in pairs:
            key = (A.neuron, A.channel, B.neuron, B.channel)
            if key in self._sheaths:
                self._sheaths[key].thicken(0.02)
                # 成功共发射 → age 归零（"刚被用到"）
                self._sheaths[key].since_co_fire = 0
                continue"""

NEW = """        created = 0
        for A, B in pairs:
            key = (A.neuron, A.channel, B.neuron, B.channel)

            # --- ① 连接层：共发射 → 建立/加强神经元之间的连接 ---
            #
            # ⚠ 这一步此前被错误地写成"直接建立髓鞘"，把两层混为一谈。
            # 共发射相连说的是**神经元之间**，髓鞘是包裹在连接上的另一回事。
            conn = self._connections.get(key)
            if conn is not None:
                conn.co_fire_count += 1
                conn.usage = 0.9 * conn.usage + 0.1 * 1.0
                conn.starving = 0
                conn.strength = min(1.0, conn.strength + 0.02)
                # 已髓鞘的同步增厚
                sh = self._sheaths.get(key)
                if sh is not None:
                    sh.thicken(0.02)
                    sh.since_co_fire = 0
                continue

            # 跨通道出边需先自证（带滞回的豁免，见 _exuberant_on）
            if self.self_proof > 0.0 and A.channel != B.channel:
                n = len(self._sheaths)
                if n >= self.exuberant_high:
                    self._exuberant_on = False
                elif n <= self.exuberant_low:
                    self._exuberant_on = True
                if not self._exuberant_on:
                    if self.channel_credibility(
                            A.neuron, A.channel) < self.self_proof:
                        continue

            self._connections[key] = Connection(
                src_neuron=A.neuron, src_channel=A.channel,
                dst_neuron=B.neuron, dst_channel=B.channel,
                strength=0.2, co_fire_count=1, birth_step=self._step,
            )
            self._born += 1
            created += 1

        # --- ② 髓鞘层：连接被反复验证 → 才被髓鞘包裹 ---
        self._myelinate_ready_connections()
        return created"""

assert OLD in s, "co_fire_wire anchor not found"
s = s.replace(OLD, NEW, 1)

# 新增髓鞘化方法
METHOD = '''
    def _myelinate_ready_connections(self) -> int:
        """把**已充分验证**的连接髓鞘化 —— 髓鞘是后来的加固，不是连接本身

        判据：共发射次数达到 ``myelination_threshold``。

        生物学：少突胶质细胞前体先采样多个轴突（数小时），再选择性
        髓鞘化其中一部分。髓鞘形成在连接**之后**，且一旦形成极稳定
        （蛋白半衰期 55 天 ~ 6 个月）。

        Returns:
            本次新增的髓鞘条数。
        """
        formed = 0
        for key, conn in self._connections.items():
            if conn.myelinated:
                continue
            if conn.co_fire_count < self.myelination_threshold:
                continue
            sh = MyelinSheath(
                src_neuron=conn.src_neuron, src_channel=conn.src_channel,
                dst_neuron=conn.dst_neuron, dst_channel=conn.dst_channel,
                delay=MyelinSheath.DELAY_MIN + 1.0,
                gain=0.5, birth_step=self._step,
            )
            self._sheaths[key] = sh
            conn.myelinated = True
            formed += 1
        return formed

    def capacity_report(self) -> dict:'''

OLD_REP = "\n    def capacity_report(self) -> dict:"
assert OLD_REP in s, "capacity_report anchor not found"
s = s.replace(OLD_REP, METHOD, 1)

# 加髓鞘化阈值参数
OLD_P = "        self.myelin_half_life: float = 0.0\n"
NEW_P = (OLD_P +
         "        # 连接被髓鞘化所需的累计共发射次数\n"
         "        self.myelination_threshold: int = 3\n")
assert OLD_P in s
s = s.replace(OLD_P, NEW_P, 1)

open(P, "w").write(s)
print("co_fire_wire 已改为建立 Connection，髓鞘化独立判据已加")
