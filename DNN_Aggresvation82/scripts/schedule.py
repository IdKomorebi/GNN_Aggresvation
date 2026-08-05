# -*- coding: utf-8 -*-
"""在用户指定GPU上调度82号训练、K0评估和胜者K网格。"""
from __future__ import annotations

import argparse
import json
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
NEW_SCHEMES = (
    "triple_only",
    "triple90_pair10",
    "triple70_pair20_uniform10",
    "triple50_pair40_uniform10",
    "triple40_pair30_uniform30",
    "local234_20_60_20",
    "local234_uniform10",
    "warm_triple70",
    "warm_local234",
    "triple70_group8",
    "triple70_group32",
    "local234_group8",
    "local234_group32",
)


def jobs_for(phase: str) -> list[tuple]:
    if phase == "train":
        return [(scheme, 0) for scheme in NEW_SCHEMES]
    if phase == "k0":
        jobs = [(scheme, 0) for scheme in NEW_SCHEMES + ("uniform",)]
        jobs.extend(
            (scheme, 0)
            for scheme in ("logunif", "small80", "workload_hard")
        )
        return jobs
    if phase in {"confirm", "confirm_k0"}:
        selection = json.loads(
            (ROOT / "outputs" / "k0_selection.json").read_text(encoding="utf-8")
        )
        schemes = sorted(
            {
                selection["best_parent_scheme"],
                selection["best_s1_scheme"],
            }
        )
        if phase == "confirm":
            return [(scheme, 1) for scheme in schemes]
        return [(scheme, 1) for scheme in schemes] + [("uniform", 1)]
    if phase == "kgrid":
        selection_path = ROOT / "outputs" / "k0_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        schemes = sorted(
            {
                selection["best_parent_scheme"],
                selection["best_s1_scheme"],
            }
        )
        jobs = []
        for scheme in schemes:
            jobs.append((scheme, "low", 0, 1))
            jobs.extend((scheme, "triple", shard, 2) for shard in range(2))
        return jobs
    raise ValueError(phase)


def output_path(phase: str, job: tuple) -> Path:
    if phase in {"train", "confirm"}:
        scheme, seed = job
        return ROOT / "outputs" / f"oracle_{scheme}_seed{seed}.pt"
    if phase in {"k0", "confirm_k0"}:
        scheme, seed = job
        return ROOT / "outputs" / f"k0_{scheme}_seed{seed}.parquet"
    scheme, kind, shard, nshard = job
    return (
        ROOT
        / "outputs"
        / f"kgrid_{kind}_{scheme}_seed0_shard{shard}of{nshard}.csv.gz"
    )


def command_for(phase: str, job: tuple) -> list[str]:
    if phase in {"train", "confirm"}:
        scheme, seed = job
        return [
            PYTHON,
            "-u",
            str(ROOT / "scripts" / "train_oracle.py"),
            "--scheme",
            scheme,
            "--seed",
            str(seed),
        ]
    if phase in {"k0", "confirm_k0"}:
        scheme, seed = job
        return [
            PYTHON,
            "-u",
            str(ROOT / "scripts" / "eval_k0.py"),
            "--scheme",
            scheme,
            "--seed",
            str(seed),
        ]
    scheme, kind, shard, nshard = job
    return [
        PYTHON,
        "-u",
        str(ROOT / "scripts" / "scan_kgrid.py"),
        "--scheme",
        scheme,
        "--kind",
        kind,
        "--shard",
        str(shard),
        "--nshard",
        str(nshard),
    ]


def launch(phase: str, job: tuple, gpu: int) -> subprocess.Popen:
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        NUMBA_CACHE_DIR=f"/tmp/numba_cache_82_{gpu}",
    )
    label = "_".join(map(str, job))
    log_path = LOG_DIR / f"{phase}_{label}.log"
    return subprocess.Popen(
        command_for(phase, job),
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("train", "k0", "confirm", "confirm_k0", "kgrid"),
        required=True,
    )
    parser.add_argument("--gpus", default="0,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue = deque(
        job
        for job in jobs_for(args.phase)
        if not output_path(args.phase, job).exists()
    )
    log(
        "SCHED",
        "START",
        note=f"{args.phase}待运行{len(queue)}个任务；GPU={gpus}",
        phase_name=args.phase,
        jobs=len(queue),
        gpus=args.gpus,
    )
    if not queue:
        log("SCHED", "DONE", note=f"{args.phase}输出均已存在")
        return

    started = time.time()
    running: dict[int, tuple[subprocess.Popen, tuple]] = {}
    free = list(gpus)
    while queue and free:
        gpu = free.pop(0)
        job = queue.popleft()
        running[gpu] = (launch(args.phase, job, gpu), job)

    failed = []
    while running:
        time.sleep(3)
        for gpu, (process, job) in list(running.items()):
            if process.poll() is None:
                continue
            del running[gpu]
            if process.returncode != 0 or not output_path(args.phase, job).exists():
                failed.append((job, process.returncode))
                log(
                    "SCHED",
                    "FAIL",
                    note=f"{args.phase}/{job}失败；见logs",
                    gpu=gpu,
                )
            if queue:
                next_job = queue.popleft()
                running[gpu] = (launch(args.phase, next_job, gpu), next_job)
            else:
                free.append(gpu)

    elapsed = time.time() - started
    if failed:
        log(
            "SCHED",
            "FAIL",
            note=f"{args.phase}有{len(failed)}个任务失败：{failed}",
            elapsed_s=elapsed,
        )
        raise SystemExit(1)
    log(
        "SCHED",
        "DONE",
        note=f"{args.phase}全部任务成功",
        elapsed_s=elapsed,
    )


if __name__ == "__main__":
    main()
