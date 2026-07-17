#!/usr/bin/env python3
"""Four-GPU resumable scheduler for missing D67/D68 GCN truth."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
DEFAULT_GPUS = [1, 2, 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 means all missing jobs")
    ap.add_argument("--source", choices=["all", "d67_single_pair", "d68_triple"], default="all")
    ap.add_argument("--gpus", default="1,2,3", help="comma-separated physical GPU ids")
    ap.add_argument("--full", action="store_true", help="run every missing D67/D68 subset instead of the audit sample")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]

    subsets = json.loads((ROOT / "outputs/subsets.json").read_text())
    audit_path = ROOT / "outputs/gcn_audit_subsets.json"
    audit_ids = set(json.loads(audit_path.read_text())) if audit_path.exists() else set()
    jobs = []
    for sid, meta in sorted(subsets.items()):
        if meta["source"] == "d60_random":
            continue
        if not args.full and sid not in audit_ids:
            continue
        if args.source != "all" and meta["source"] != args.source:
            continue
        out = ROOT / f"outputs/retrain_gcn/{sid}_gcn_seed0.json"
        if not out.exists():
            jobs.append(sid)
    if args.limit > 0:
        jobs = jobs[:args.limit]
    queue = deque(jobs)
    print(f"missing GCN retrains: {len(queue)}", flush=True)

    log_dir = ROOT / "logs/retrain_gcn"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[subprocess.Popen, str, object]] = {}

    def launch(sid: str, gpu: int):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
        handle = (log_dir / f"{sid}.log").open("w")
        proc = subprocess.Popen(
            [PYTHON, "-u", str(ROOT / "scripts/retrain_gcn_worker.py"),
             "--subset-id", sid, "--seed", "0"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return proc, sid, handle

    for gpu in gpus:
        if queue:
            running[gpu] = launch(queue.popleft(), gpu)
    done = failed = 0
    started = time.time()
    while running:
        time.sleep(2)
        for gpu, (proc, sid, handle) in list(running.items()):
            if proc.poll() is None:
                continue
            handle.close()
            done += 1
            if proc.returncode != 0:
                failed += 1
                print(f"FAIL {sid} rc={proc.returncode}", flush=True)
            if done % 25 == 0 or proc.returncode != 0:
                print(f"done={done}/{len(jobs)} failed={failed} elapsed={time.time()-started:.0f}s", flush=True)
            del running[gpu]
            if queue:
                running[gpu] = launch(queue.popleft(), gpu)
    print(f"complete done={done} failed={failed} elapsed={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
