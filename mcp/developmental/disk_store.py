"""SessionStore 磁盘后端：活性上下文跨进程持久化（每会话一 json 文件）

活性上下文无 done 终止：事件流落盘，重启后 load 恢复（跨进程复活）。
内存层复用 SessionStore（_data 索引），磁盘层兜底（首次 load 从磁盘读入）。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

from mcp.developmental.events import Session, SessionStore


class FileSessionStore(SessionStore):
    """文件持久化会话存储：save 落盘 + 新实例 load 恢复"""

    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.base_dir / f"{sid}.json"

    def save(self, session: Session) -> None:
        super().save(session)  # 内存索引
        self._write(session.sid)

    def _write(self, sid: str) -> None:
        payload = {"sid": sid, "rows": self._data.get(sid, [])}
        self._path(sid).write_text(json.dumps(payload, ensure_ascii=False),
                                   encoding="utf-8")

    def _read(self, sid: str) -> None:
        p = self._path(sid)
        if not p.exists():
            raise KeyError(f"session not found: {sid}")
        data = json.loads(p.read_text(encoding="utf-8"))
        self._data[sid] = data.get("rows", [])

    def load(self, sid: str) -> Session:
        if sid not in self._data:
            self._read(sid)  # 磁盘兜底（跨进程恢复）
        return super().load(sid)

    def _rows(self, sid: str) -> List[dict]:
        return self._data.get(sid, [])

    def fork(self, sid: str, new_sid: str) -> Session:
        """分叉会话并落盘（活性上下文可分支演化）"""
        s = super().fork(sid, new_sid)
        self._write(new_sid)
        return s
