# -*- coding: utf-8 -*-
"""DNN78 运行流水：同时记录 JSONL 与 Markdown。

78 号是纯 CPU 的口径修正项目。日志必须能回答：
1. 读了哪些既有结果；
2. 改了哪个定义或统计单位；
3. 修正前后的数字是什么；
4. 哪些问题只能靠新 GPU 实验解决。
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

_ICON = {
    "START": "▶",
    "DONE": "✔",
    "FAIL": "✘",
    "NOTE": "※",
    "DECISION": "◆",
}


def _elapsed(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _header(handle, now: datetime) -> None:
    handle.write(
        f"# {ROOT.name} 运行流水\n\n"
        "> 本文件由 `src/runlog.py` 自动生成；机器可读记录在 "
        "`outputs/events.jsonl`。\n\n"
        f"## {now:%Y-%m-%d}\n"
    )


def section(title: str) -> None:
    now = datetime.now()
    with open(RUNLOG, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if handle.tell() == 0:
            _header(handle, now)
        handle.write(f"\n### {title}\n\n")
        fcntl.flock(handle, fcntl.LOCK_UN)


def log(
    phase: str,
    event: str,
    *,
    note: str = "",
    elapsed_s: float | None = None,
    **extra,
) -> None:
    now = datetime.now()
    record = {
        "ts": now.isoformat(timespec="seconds"),
        "phase": phase,
        "event": event,
        "note": note,
        "elapsed_s": elapsed_s,
        "pid": os.getpid(),
        **extra,
    }
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        fcntl.flock(handle, fcntl.LOCK_UN)

    tail = []
    if elapsed_s is not None:
        tail.append(f"实耗 {_elapsed(elapsed_s)}")
    tail.extend(f"{key}={value}" for key, value in extra.items())
    line = (
        f"- `{now:%H:%M:%S}` {_ICON.get(event, '·')} "
        f"**[{phase}] {event}**"
        + (f"　{note}" if note else "")
        + (f"　　{'　'.join(tail)}" if tail else "")
        + "\n"
    )
    with open(RUNLOG, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if handle.tell() == 0:
            _header(handle, now)
        handle.write(line)
        fcntl.flock(handle, fcntl.LOCK_UN)


class Timer:
    def __enter__(self):
        self.started = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self.started

