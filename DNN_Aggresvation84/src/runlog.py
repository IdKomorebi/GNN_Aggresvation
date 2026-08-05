# -*- coding: utf-8 -*-
"""DNN82 并发安全运行日志。"""
from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "outputs" / "events.jsonl"
RUNLOG = ROOT / "RUNLOG.md"
ICONS = {"START": "▶", "DONE": "✔", "FAIL": "✘", "NOTE": "※", "DECISION": "◆"}


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


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
        tail.append(f"实耗 {_duration(elapsed_s)}")
    tail.extend(f"{key}={value}" for key, value in extra.items())
    with open(RUNLOG, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if handle.tell() == 0:
            handle.write(
                f"# {ROOT.name} 运行流水\n\n"
                "> 自动日志；机器可读事件见 `outputs/events.jsonl`。\n\n"
                f"## {now:%Y-%m-%d}\n\n"
            )
        handle.write(
            f"- `{now:%H:%M:%S}` {ICONS.get(event, '·')} "
            f"**[{phase}] {event}**"
            + (f"　{note}" if note else "")
            + (f"　　{'　'.join(tail)}" if tail else "")
            + "\n"
        )
        fcntl.flock(handle, fcntl.LOCK_UN)

