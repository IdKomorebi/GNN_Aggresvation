#!/usr/bin/env python3
"""Run clean retrain certificates over a GPU queue."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LOG = OUT / "clean_retrain_logs"
LOG.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    candidates = list(json.loads((OUT / "candidates.json").read_text())["candidates"])
    jobs = [(candidate, seed) for candidate in candidates for seed in args.seeds]

    def run_lane(lane_index, gpu):
        lane_results = []
        for job_index in range(lane_index, len(jobs), len(args.gpus)):
            candidate, seed = jobs[job_index]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [sys.executable, str(ROOT / "scripts/05_clean_retrain_worker.py"),
                   "--candidate", candidate, "--seed", str(seed),
                   "--epochs", str(args.epochs)]
            if args.force:
                cmd.append("--force")
            proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
            (LOG / f"{candidate}_seed{seed}.log").write_text(proc.stdout + "\nSTDERR\n" + proc.stderr)
            print(f"gpu{gpu} {candidate}/seed{seed} exit={proc.returncode}", flush=True)
            lane_results.append((candidate, seed, gpu, proc.returncode))
        return lane_results

    failures = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_lane, i, gpu) for i, gpu in enumerate(args.gpus)]
        for fut in as_completed(futures):
            for candidate, seed, gpu, code in fut.result():
                if code:
                    failures.append((candidate, seed, gpu, code))
    summary = {"jobs": len(jobs), "completed": len(jobs) - len(failures), "failures": failures}
    (OUT / "clean_retrain_schedule.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise SystemExit(f"failures: {failures}")


if __name__ == "__main__":
    main()
