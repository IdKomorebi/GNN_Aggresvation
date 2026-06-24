#!/usr/bin/env python3
"""DNN40 双卡调度器：只用 GPU0/1，跳过已经完成的 run。"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation40_InterimSummary")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1]
LOG_DIR = ROOT / "logs_dual"
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


def _launch(cfg: Path, gpu: int) -> subprocess.Popen:
    name = cfg.stem
    log_path = LOG_DIR / f"{name}.log"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        NUMBA_CACHE_DIR="/tmp/numba_cache",
    )
    cmd = [PY, "-u", str(ROOT / "scripts" / "run_pipeline.py"), "--config", str(cfg)]
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    proc._dnn40_log_f = log_f  # type: ignore[attr-defined]
    print(f"[dual] start {name} on GPU {gpu}; pid={proc.pid}; log={log_path}")
    return proc


def main() -> int:
    configs = sorted((ROOT / "configs").glob("single_*.yaml"))
    configs.append(ROOT / "configs" / "multi.yaml")
    pending = [cfg for cfg in configs if not _is_complete(cfg)]
    skipped = [cfg for cfg in configs if _is_complete(cfg)]

    print(f"DNN40 dual: total={len(configs)}, skipped={len(skipped)}, pending={len(pending)}")
    print("Skipped:", [cfg.name for cfg in skipped])
    print("Pending:", [cfg.name for cfg in pending])

    queue = deque(pending)
    free = list(GPUS)
    running: dict[int, tuple[subprocess.Popen, Path]] = {}

    while queue or running:
        while queue and free:
            gpu = free.pop(0)
            cfg = queue.popleft()
            running[gpu] = (_launch(cfg, gpu), cfg)

        time.sleep(10)
        for gpu, (proc, cfg) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log_f = getattr(proc, "_dnn40_log_f", None)
            if log_f is not None:
                log_f.close()
            print(f"[dual] done {cfg.stem} on GPU {gpu}; rc={rc}")
            del running[gpu]
            free.append(gpu)
            if rc != 0:
                print(f"[dual] failed: {cfg.name}", file=sys.stderr)
                return int(rc)

    print("DNN40 dual complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
