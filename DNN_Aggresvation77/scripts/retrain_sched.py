# -*- coding: utf-8 -*-
"""H-2 重训调度：分片 + 跳过已完成 + 失败自动重试 + 事件日志。

为什么要分片
------------
H-2 是本次夜跑的长杆（约 12.5 GPU·h）。它与 F-1/F-2/H-1 完全独立，所以先用 GPU 2,3
起两个分片跑起来，等 F-1/H-1 腾出 GPU 0,1 再补起另两个分片。
**分片互不重叠**，因此不同时间点启动也不会重复劳动、无竞态（不像"共享待跑队列"那样
会在启动瞬间双方都看到同一批未完成任务）。

用法：
    python scripts/retrain_sched.py --shard 2 --nshard 4 --gpus 2
    python scripts/retrain_sched.py --shard 0,1 --nshard 4 --gpus 0,1
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

R77 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R77 / "src"))
from runlog import log  # noqa: E402

PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
LOG = R77 / "logs"
RETRAIN = R77 / "outputs" / "retrain"
MAX_RETRY = 2


def done(sid: str) -> bool:
    return (RETRAIN / f"{sid}_dnn_seed0.json").exists()


def launch(sid: str, gpu: int):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               NUMBA_CACHE_DIR="/tmp/numba_cache")
    return subprocess.Popen(
        [PY, "-u", str(R77 / "scripts/retrain_worker.py"),
         "--subset-id", sid, "--struct", "dnn", "--seed", "0"],
        stdout=open(LOG / f"{sid}.log", "w"), stderr=subprocess.STDOUT, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="逗号分隔的分片号，如 2 或 0,1")
    ap.add_argument("--nshard", type=int, default=4)
    ap.add_argument("--gpus", required=True)
    args = ap.parse_args()

    LOG.mkdir(exist_ok=True)
    RETRAIN.mkdir(parents=True, exist_ok=True)
    shards = [int(x) for x in args.shard.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]

    subs = json.load(open(R77 / "outputs/subsets.json"))
    mine = [sid for i, sid in enumerate(sorted(subs)) if i % args.nshard in shards]
    jobs = deque([s for s in mine if not done(s)])
    total = len(jobs)
    tag = f"shard{args.shard}/{args.nshard}"
    log("H-2", "START", note=f"重训 {tag}：{total} 个待跑（本分片共 {len(mine)}，已跳过完成的）",
        gpu=args.gpus, n_done=0, n_total=total)
    if not total:
        log("H-2", "DONE", note=f"{tag} 无待跑任务", gpu=args.gpus)
        return

    t0 = time.time()
    retry = {}
    pool, run = list(gpus), {}
    while jobs and pool:
        g = pool.pop(0)
        sid = jobs.popleft()
        run[g] = (launch(sid, g), sid)

    ok = bad = 0
    last_report = 0
    while run:
        time.sleep(3)
        for g, (p, sid) in list(run.items()):
            if p.poll() is None:
                continue
            del run[g]
            if p.returncode == 0 and done(sid):
                ok += 1
            else:
                n = retry.get(sid, 0)
                if n < MAX_RETRY:
                    retry[sid] = n + 1
                    jobs.append(sid)          # 放回队尾，换卡重试
                    log("H-2", "RETRY", note=f"{sid} 退出码 {p.returncode}，第 {n+1} 次重试",
                        gpu=str(g))
                else:
                    bad += 1
                    log("H-2", "FAIL", note=f"{sid} 连续失败 {MAX_RETRY+1} 次，跳过；"
                                            f"见 logs/{sid}.log", gpu=str(g))
            if jobs:
                s2 = jobs.popleft()
                run[g] = (launch(s2, g), s2)
            else:
                pool.append(g)
            if ok - last_report >= 100:
                last_report = ok
                rate = ok / max(time.time() - t0, 1) * 3600
                eta = (total - ok - bad) / max(rate, 1e-9)
                log("H-2", "PROGRESS", note=f"{tag} 速率 {rate:.0f} 个/小时，"
                                            f"预计还需 {eta:.1f} h",
                    gpu=args.gpus, n_done=ok, n_total=total,
                    elapsed_s=time.time() - t0)

    log("H-2", "DONE", note=f"{tag} 成功 {ok}，失败 {bad}",
        gpu=args.gpus, n_done=ok, n_total=total, elapsed_s=time.time() - t0)


if __name__ == "__main__":
    main()
