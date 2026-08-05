# -*- coding: utf-8 -*-
"""运行流水日志：同时写机器可读的 events.jsonl 与人读的 RUNLOG.md。

设计目标是**断点可恢复**：即使执行者（我）中途丢失上下文，
从 RUNLOG.md + 输出文件即可无歧义地判断"跑到哪了、哪些失败了、该续什么"。

用法：
    from runlog import log
    log("F-1", "START", note="9 job × 990 子集", gpu="0,1", eta="~25min")
    log("F-1", "DONE",  elapsed_s=1544, n_done=9, n_total=9, note="无问题")

事件约定（event 字段）：
    START / DONE / FAIL / RETRY / PROGRESS / NOTE / DECISION
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "outputs" / "events.jsonl"
RUNLOG = ROOT / "RUNLOG.md"

_ICON = {"START": "▶", "DONE": "✔", "FAIL": "✘", "RETRY": "↻",
         "PROGRESS": "·", "NOTE": "※", "DECISION": "◆"}


def _fmt_elapsed(s) -> str:
    if s is None:
        return ""
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def log(phase: str, event: str, *, note: str = "", gpu: str = "",
        elapsed_s: float | None = None, n_done: int | None = None,
        n_total: int | None = None, **extra) -> None:
    """追加一条事件。进程安全（flock），可被调度器子进程并发调用。"""
    ts = datetime.now()
    rec = dict(ts=ts.isoformat(timespec="seconds"), phase=phase, event=event,
               note=note, gpu=gpu, elapsed_s=elapsed_s,
               n_done=n_done, n_total=n_total, pid=os.getpid(), **extra)

    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    bits = []
    if elapsed_s is not None:
        bits.append(f"实耗 {_fmt_elapsed(elapsed_s)}")
    if n_total is not None:
        bits.append(f"进度 {n_done}/{n_total}")
    if gpu:
        bits.append(f"GPU {gpu}")
    for k, v in extra.items():
        bits.append(f"{k}={v}")
    tail = "　".join(bits)

    line = (f"- `{ts.strftime('%H:%M:%S')}` {_ICON.get(event, '·')} **[{phase}] {event}**"
            + (f"　{note}" if note else "")
            + (f"　　{tail}" if tail else "") + "\n")

    with open(RUNLOG, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            _header(f, ts)
        f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)


def _header(f, ts) -> None:
    f.write(f"# {ROOT.name} 运行流水\n\n"
            f"> 本文件由 `src/runlog.py` 自动追加，与 `CHANGELOG.md`（结论）并列。\n"
            f"> 机器可读版本：`outputs/events.jsonl`\n\n"
            f"## {ts.strftime('%Y-%m-%d')}\n")


def section(title: str) -> None:
    """在 RUNLOG.md 里插一个小节标题。"""
    ts = datetime.now()
    with open(RUNLOG, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            _header(f, ts)
        f.write(f"\n### {title}\n\n")
        fcntl.flock(f, fcntl.LOCK_UN)


class Timer:
    """with Timer() as t: ...   然后 t.elapsed"""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0

    @property
    def now(self) -> float:
        return time.time() - self.t0
