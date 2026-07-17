#!/usr/bin/env python3
"""DNN59 调度器：5 排名 × 2 结构 × 11 个 k = 110 次重训，4 GPU 并行，跳过已完成。"""
import os, subprocess, time
from collections import deque
from pathlib import Path

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation59")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG = D / "logs"; LOG.mkdir(exist_ok=True)
RANKINGS = ["dnn", "noprior", "static", "zeromask", "masked"]
STRUCTS = ["dnn", "gcn"]
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 44]

jobs = deque()
for r in RANKINGS:
    for s in STRUCTS:
        for k in KS:
            if not (D / "outputs/retrain" / f"{r}_{s}_k{k}.json").exists():
                jobs.append((r, s, k))
print(f"待跑 {len(jobs)} 次重训（已跳过完成的）")
pool = list(GPUS); run = {}


def launch(job, gpu):
    r, s, k = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    p = subprocess.Popen([PY, "-u", str(D / "scripts/retrain_worker.py"),
                          "--ranking", r, "--struct", s, "--k", str(k)],
                         stdout=open(LOG / f"{r}_{s}_k{k}.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {r}_{s}_k{k} on GPU {gpu}")
    return p


while jobs and pool:
    j = jobs.popleft(); gpu = pool.pop(0); run[gpu] = (launch(j, gpu), f"{j[0]}_{j[1]}_k{j[2]}")
done = 0
while run:
    time.sleep(10); fin = []
    for gpu, (p, n) in list(run.items()):
        if p.poll() is not None:
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {n} [{'OK' if p.returncode == 0 else 'FAIL'}]")
            done += 1; fin.append(gpu)
    for gpu in fin:
        if jobs:
            j = jobs.popleft(); run[gpu] = (launch(j, gpu), f"{j[0]}_{j[1]}_k{j[2]}")
        else:
            del run[gpu]
print(f"\n全部完成：{done}")
