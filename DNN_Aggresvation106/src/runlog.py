"""DNN106 运行日志。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(stage: str, status: str, note: str, **fields: object) -> None:
    now = datetime.now()
    record = {
        "time": now.isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
        "note": note,
        **fields,
    }
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    runlog = ROOT / "RUNLOG.md"
    if not runlog.exists():
        runlog.write_text(
            "# DNN_Aggresvation106 运行流水\n\n"
            "> 自动记录；实验结论见 `CHANGELOG.md`。\n\n",
            encoding="utf-8",
        )
    mark = {"START": "▶", "DONE": "✔", "FAIL": "✘", "DECISION": "◆"}.get(
        status, "·"
    )
    with runlog.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- `{now:%Y-%m-%d %H:%M:%S}` {mark} "
            f"**[{stage}] {status}**　{note}\n"
        )
