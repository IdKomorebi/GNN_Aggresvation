# -*- coding: utf-8 -*-
"""81号运行日志：同时记录人读日志和机器可读事件。"""
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


def log(phase: str, event: str, *, note: str = "", elapsed_s=None, **extra) -> None:
    ts = datetime.now()
    rec = {
        "ts": ts.isoformat(timespec="seconds"),
        "phase": phase,
        "event": event,
        "note": note,
        "elapsed_s": elapsed_s,
        "pid": os.getpid(),
        **extra,
    }
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fcntl.flock(handle, fcntl.LOCK_UN)

    details = []
    if elapsed_s is not None:
        details.append(f"实耗 {elapsed_s:.1f}s")
    details.extend(f"{key}={value}" for key, value in extra.items())
    suffix = "　　" + "　".join(details) if details else ""
    with RUNLOG.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if handle.tell() == 0:
            handle.write(
                f"# {ROOT.name} 运行流水\n\n"
                "> 与 `CHANGELOG.md` 并列；机器可读事件见 `outputs/events.jsonl`。\n\n"
                f"## {ts:%Y-%m-%d}\n"
            )
        handle.write(
            f"- `{ts:%H:%M:%S}` {_ICON.get(event, '·')} "
            f"**[{phase}] {event}**"
            f"{'　' + note if note else ''}{suffix}\n"
        )
        fcntl.flock(handle, fcntl.LOCK_UN)


class Timer:
    def __enter__(self):
        self.started = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self.started

