#!/usr/bin/env python3
"""Dynamic four-GPU scheduler for MLP/GNN equal-step fine-tuning shards."""
from __future__ import annotations

import argparse
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
    ap.add_argument("--nshard", type=int, default=12)
    ap.add_argument("--arch", choices=["all", "mlp", "gnn"], default="all")
    ap.add_argument("--limit-per-shard", type=int, default=0)
    ap.add_argument("--gpus", default="1,2,3", help="comma-separated physical GPU ids")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]

    arches = ["mlp", "gnn"] if args.arch == "all" else [args.arch]
    # Interleave architectures so neither gets all early GPU slots.
    jobs = deque((arch, shard) for shard in range(args.nshard) for arch in arches)
    log_dir = ROOT / "logs/estimate"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[subprocess.Popen, str, int, object]] = {}

    def launch(arch: str, shard: int, gpu: int):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
        handle = (log_dir / f"{arch}_shard{shard:02d}.log").open("w")
        cmd = [PYTHON, "-u", str(ROOT / "scripts/estimate_oracle_worker.py"),
               "--arch", arch, "--shard", str(shard), "--nshard", str(args.nshard)]
        if args.limit_per_shard > 0:
            cmd += ["--limit", str(args.limit_per_shard)]
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
        return proc, arch, shard, handle

    for gpu in gpus:
        if jobs:
            arch, shard = jobs.popleft()
            running[gpu] = launch(arch, shard, gpu)
    completed = failed = 0
    started = time.time()
    total = len(running) + len(jobs)
    while running:
        time.sleep(2)
        for gpu, (proc, arch, shard, handle) in list(running.items()):
            if proc.poll() is None:
                continue
            handle.close()
            completed += 1
            if proc.returncode != 0:
                failed += 1
                print(f"FAIL {arch}/shard{shard} rc={proc.returncode}", flush=True)
            print(f"shards={completed}/{total} failed={failed} elapsed={time.time()-started:.0f}s", flush=True)
            del running[gpu]
            if jobs:
                next_arch, next_shard = jobs.popleft()
                running[gpu] = launch(next_arch, next_shard, gpu)
    print(f"complete shards={completed} failed={failed} elapsed={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
