# -*- coding: utf-8 -*-
"""F-1 调度：9 job（3 方案 × 3 seed）下发到指定 GPU，跳过已完成 + 失败重试 + 事件日志。

用法：python scripts/sched.py --gpus 0,1
"""
import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

R76 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R76 / "src"))
from runlog import log  # noqa: E402

PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
LOG = R76 / "logs"
SCHEMES = ["uniform", "none", "bern50"]
SEEDS = [0, 1, 2]
MAX_RETRY = 2


def done(sc, sd):
    return (R76 / "outputs" / f"fine_{sc}_seed{sd}.csv").exists()


def launch(job, gpu):
    sc, sd = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               NUMBA_CACHE_DIR="/tmp/numba_cache")
    return subprocess.Popen(
        [PY, "-u", str(R76 / "scripts/fine_ksweep.py"), "--scheme", sc, "--seed", str(sd)],
        stdout=open(LOG / f"{sc}_seed{sd}.log", "w"), stderr=subprocess.STDOUT, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    args = ap.parse_args()
    LOG.mkdir(exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]

    jobs = deque([(sc, sd) for sc in SCHEMES for sd in SEEDS if not done(sc, sd)])
    total = len(jobs)
    log("F-1", "START", note=f"相变扫描 {total} job（3 方案 × 3 seed）× 990 子集 × 17 个 K",
        gpu=args.gpus, n_done=0, n_total=total)
    if not total:
        log("F-1", "DONE", note="无待跑任务", gpu=args.gpus)
        return

    t0 = time.time()
    retry, pool, run = {}, list(gpus), {}
    while jobs and pool:
        g = pool.pop(0)
        j = jobs.popleft()
        run[g] = (launch(j, g), j)

    ok = bad = 0
    while run:
        time.sleep(5)
        for g, (p, j) in list(run.items()):
            if p.poll() is None:
                continue
            del run[g]
            if p.returncode == 0 and done(*j):
                ok += 1
            else:
                n = retry.get(j, 0)
                if n < MAX_RETRY:
                    retry[j] = n + 1
                    jobs.append(j)
                    log("F-1", "RETRY", note=f"{j[0]} seed{j[1]} 退出码 {p.returncode}，"
                                             f"第 {n+1} 次重试", gpu=str(g))
                else:
                    bad += 1
                    log("F-1", "FAIL", note=f"{j[0]} seed{j[1]} 连续失败，跳过；"
                                            f"见 logs/{j[0]}_seed{j[1]}.log", gpu=str(g))
            if jobs:
                j2 = jobs.popleft()
                run[g] = (launch(j2, g), j2)
            else:
                pool.append(g)

    log("F-1", "DONE", note=f"相变扫描成功 {ok}，失败 {bad}",
        gpu=args.gpus, n_done=ok, n_total=total, elapsed_s=time.time() - t0)


if __name__ == "__main__":
    main()
