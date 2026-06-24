#!/usr/bin/env python3
"""DNN40 串行调度器：只用 GPU0，跳过已经完成的 run。"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation40_InterimSummary")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPU = 0
LOG_DIR = ROOT / "logs_serial"
LOG_DIR.mkdir(exist_ok=True)


def _single_target(cfg: Path) -> str | None:
    text = cfg.read_text(encoding="utf-8")
    m = re.search(r"^\s*single_target:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    return m.group(1) if m else None


def _is_complete(cfg: Path) -> bool:
    if cfg.name == "multi.yaml":
        base = ROOT / "outputs" / "graph_multi_window1_fixed_info"
        return any(base.glob("*/results/summary.json"))
    target = _single_target(cfg)
    if not target:
        return False
    base = ROOT / "outputs" / "graph_single_window1_fixed_info" / target
    return any(base.glob("*/results/summary.json"))


def main() -> int:
    configs = sorted((ROOT / "configs").glob("single_*.yaml"))
    configs.append(ROOT / "configs" / "multi.yaml")

    pending = [cfg for cfg in configs if not _is_complete(cfg)]
    skipped = [cfg for cfg in configs if _is_complete(cfg)]

    print(f"DNN40 serial: total={len(configs)}, skipped={len(skipped)}, pending={len(pending)}")
    if skipped:
        print("Skipped:", [cfg.name for cfg in skipped])
    print("Pending:", [cfg.name for cfg in pending])

    for cfg in pending:
        name = cfg.stem
        log_path = LOG_DIR / f"{name}.log"
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(GPU),
            NUMBA_CACHE_DIR="/tmp/numba_cache",
        )
        cmd = [PY, "-u", str(ROOT / "scripts" / "run_pipeline.py"), "--config", str(cfg)]
        print(f"\n[serial] start {name} on GPU {GPU}; log={log_path}")
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        print(f"[serial] done {name}; rc={proc.returncode}")
        if proc.returncode != 0:
            print(f"[serial] failed: {name}, see {log_path}", file=sys.stderr)
            return proc.returncode
    print("\nDNN40 serial complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
