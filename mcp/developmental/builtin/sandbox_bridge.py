"""T5 沙箱世界桥：Three.js/Minecraft 风格环境经 HTTP 暴露给爬虫脑

## 设计

- ``GridWorld``：自包含的网格世界（8×8，agent/goal/item），动作离散
  5 键，观测 JSON——Three.js 前端只是同一 JSON API 的**渲染客户端**，
  世界状态权威在服务器侧（后续可换 MineStudio/MineDojo 适配器）。
- ``SandboxServer``：stdlib http.server，GET /observe、POST /step。
- ``SandboxLimb``：把爬虫脑的 lang 信道输出（5 键 one-hot）转成桥的
  动作调用——世界成为爬虫脑的一个可挂载肢体（web_agent 同款模式）。

## 教义对应

世界模型不是新模块：自回归核心 + 睡眠重放已经是世界模型的机制本体，
世界桥只是给这条回路接上**更厚的外部承载层**。闭环生成回路的建立
以 geometry.cycle_coherence > 0 为判据。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from mcp.developmental.reptilian import ReptilianFunction

ACTIONS = ["up", "down", "left", "right", "grab"]


class GridWorld:
    """8×8 网格：agent 拿到 item 后回到 goal 得奖励（两段任务）"""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.agent = [int(self.rng.integers(8)), int(self.rng.integers(8))]
        self.goal = [int(self.rng.integers(8)), int(self.rng.integers(8))]
        self.item = [int(self.rng.integers(8)), int(self.rng.integers(8))]
        self.has_item = False
        return self.observe()

    def step(self, action: str):
        if action in ("up", "down"):
            self.agent[1] = max(0, min(7, self.agent[1] + (1 if action == "down" else -1)))
        elif action in ("left", "right"):
            self.agent[0] = max(0, min(7, self.agent[0] + (1 if action == "right" else -1)))
        elif action == "grab":
            if self.agent == self.item:
                self.has_item = True
        reward = 0.0
        done = False
        if self.has_item and self.agent == self.goal:
            reward, done = 1.0, True
        return reward, done, self.observe()

    def observe(self):
        return {
            "agent": list(self.agent), "goal": list(self.goal),
            "item": list(self.item), "has_item": self.has_item,
            "actions": ACTIONS,
        }


class SandboxServer:
    """JSON/HTTP 桥：GET /observe、POST /step {"action": "..."}"""

    def __init__(self, world: GridWorld = None, port: int = 8765):
        self.world = world or GridWorld()
        self.port = port
        self._srv = None

    def _handler(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):      # 静默
                pass

            def do_GET(self):
                body = json.dumps(outer.world.observe()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                ln = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(ln) or b"{}")
                action = req.get("action", "")
                if action not in ACTIONS:
                    body = json.dumps({"error": f"unknown action {action!r}"},
                                      ).encode()
                else:
                    reward, done, obs = outer.world.step(action)
                    if done:
                        outer.world.reset()
                    body = json.dumps({"reward": reward, "done": done,
                                       "obs": obs}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return H

    def start(self):
        self._srv = ThreadingHTTPServer(("127.0.0.1", self.port),
                                        self._handler())
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        if self._srv is not None:
            self._srv.shutdown()
            self._srv = None


class SandboxLimb(ReptilianFunction):
    """爬虫脑肢体：lang 信道的 5 键 one-hot → 桥动作调用（同步 HTTP）"""

    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/step"
        self.last_obs = None
        self.last_reward = 0.0

    def get_input_spec(self):
        return {"lang": "generic/tensor"}

    def get_output_spec(self):
        return {"lang": "generic/tensor"}

    def execute(self, inputs):
        import urllib.request
        sig = inputs["lang"]
        # 兼容 Signal 与裸张量两种输入（信号池里是 Signal）
        d = sig.data if hasattr(sig, "data") else sig
        a = np.asarray(d.detach().cpu().numpy() if hasattr(d, "detach")
                       else d, dtype=float).ravel()
        idx = int(np.argmax(a)) if a.size >= len(ACTIONS) and a.max() > 0.05 \
            else self.rng.integers(len(ACTIONS)) if hasattr(self, "rng") \
            else int(np.random.integers(len(ACTIONS)))
        req = urllib.request.Request(
            self.url, data=json.dumps({"action": ACTIONS[idx]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            resp = json.loads(r.read())
        self.last_obs = resp.get("obs")
        self.last_reward = float(resp.get("reward", 0.0))
        # 回给脑的是环境回执（作为 lang 信道回执——简化：回显动作）
        out = np.zeros(len(ACTIONS))
        out[idx] = 1.0
        return {"lang": ReptilianSignal(out)}


def ReptilianSignal(vec):
    from mcp.developmental.signal import Signal
    import numpy_torch_shim as _shim
    return Signal(data=_shim.tensor(vec), mime_type="generic/tensor",
                  metadata={"source": "sandbox"})
