# -*- coding: utf-8 -*-
"""把低阶闭包和全部三元组 K 网格扫描分配到多张 GPU。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from runlog import log  # noqa: E402

PYTHON = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
LOG_DIR = ROOT / "logs"
N_TRIPLE_SHARDS = 4


def output_path(job: tuple[str, int, int]) -> Path:
    kind, shard, nshard = job
    return (
        ROOT
        / "outputs"
        / f"kgrid_{kind}_uniform_seed0_shard{shard}of{nshard}.csv.gz"
    )


def launch(job: tuple[str, int, int], gpu: int) -> subprocess.Popen:
    kind, shard, nshard = job
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        NUMBA_CACHE_DIR=f"/tmp/numba_cache_79_{gpu}",
    )
    log_file = LOG_DIR / f"kgrid_{kind}_shard{shard}of{nshard}.log"
    return subprocess.Popen(
        [
            PYTHON,
            "-u",
            str(ROOT / "scripts" / "scan_kgrid.py"),
            "--kind",
            kind,
            "--shard",
            str(shard),
            "--nshard",
            str(nshard),
        ],
        stdout=open(log_file, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_jobs = [("low", 0, 1)] + [
        ("triple", shard, N_TRIPLE_SHARDS) for shard in range(N_TRIPLE_SHARDS)
    ]
    queue = deque(job for job in all_jobs if not output_path(job).exists())
    log(
        "SCHED",
        "START",
        note=f"待运行 {len(queue)} 个 K 网格任务，GPU={gpus}",
        n_jobs=len(queue),
        gpus=args.gpus,
    )
    if not queue:
        log("SCHED", "DONE", note="所有输出已存在，无需重跑")
        return

    started = time.time()
    free = list(gpus)
    running: dict[int, tuple[subprocess.Popen, tuple[str, int, int]]] = {}
    while queue and free:
        gpu = free.pop(0)
        job = queue.popleft()
        running[gpu] = (launch(job, gpu), job)

    failed = []
    while running:
        time.sleep(5)
        for gpu, (process, job) in list(running.items()):
            if process.poll() is None:
                continue
            del running[gpu]
            if process.returncode != 0 or not output_path(job).exists():
                failed.append((job, process.returncode))
                log(
                    "SCHED",
                    "FAIL",
                    note=f"{job} 退出码 {process.returncode}；见 logs/",
                    gpu=gpu,
                )
            if queue:
                next_job = queue.popleft()
                running[gpu] = (launch(next_job, gpu), next_job)
            else:
                free.append(gpu)

    elapsed = time.time() - started
    if failed:
        log("SCHED", "FAIL", note=f"{len(failed)} 个任务失败：{failed}", elapsed_s=elapsed)
        raise SystemExit(1)
    log("SCHED", "DONE", note="全部 K 网格任务成功", elapsed_s=elapsed)


if __name__ == "__main__":
    main()
