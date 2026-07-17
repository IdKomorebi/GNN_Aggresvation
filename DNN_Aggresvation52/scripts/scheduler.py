#!/usr/bin/env python3
"""DNN52 调度器：3 编码 × 6 聚合 = 18 个训练，4 GPU 并行。"""
import os, subprocess, time
from collections import deque
from pathlib import Path

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation52")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG = D / "logs"; LOG.mkdir(exist_ok=True)

ENCODERS = ["linear_noact", "linear", "mlp"]
MODES = ["gcn_noprior", "gcn_static", "gcn_dynamic",
         "gat_noprior", "gat_static", "gat_dynamic"]
jobs = deque((e, m) for e in ENCODERS for m in MODES)
pool = list(GPUS); run = {}


def launch(job, gpu):
    enc, mode = job
    tag = f"{enc}__{mode}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    p = subprocess.Popen(
        [PY, "-u", str(D / "scripts/train_variant.py"),
         "--config", str(D / "configs/base.yaml"), "--encoder", enc, "--mode", mode],
        stdout=open(LOG / f"{tag}.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {tag} on GPU {gpu}")
    return p


print(f"DNN52 调度器：{len(jobs)} 个任务，{len(GPUS)} 卡\n")
while jobs and pool:
    j = jobs.popleft(); g = pool.pop(0); run[g] = (launch(j, g), f"{j[0]}__{j[1]}")
done = 0
while run:
    time.sleep(10); fin = []
    for g, (p, n) in list(run.items()):
        if p.poll() is not None:
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {n} [{'OK' if p.returncode == 0 else 'FAIL'}]")
            done += 1; fin.append(g)
    for g in fin:
        if jobs:
            j = jobs.popleft(); run[g] = (launch(j, g), f"{j[0]}__{j[1]}")
        else:
            del run[g]
print(f"\n全部完成：{done}")
