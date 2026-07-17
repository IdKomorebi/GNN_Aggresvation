#!/usr/bin/env python3
"""DNN58 retrain 调度器：k×struct 网格，4 GPU 并行重训。跳过已完成。"""
import os, subprocess, time
from collections import deque
from pathlib import Path

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation58")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG = D / "logs_retrain"; LOG.mkdir(exist_ok=True)
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 44]
STRUCTS = ["dnn", "gcn"]

jobs = deque()
for s in STRUCTS:
    for k in KS:
        if not (D / "outputs/retrain" / f"{s}_k{k}.json").exists():
            jobs.append((s, k))
print(f"待跑 {len(jobs)} 个重训（已跳过完成的）")
pool = list(GPUS); run = {}


def launch(job, gpu):
    s, k = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    p = subprocess.Popen([PY, "-u", str(D / "scripts/retrain_worker.py"), "--struct", s, "--k", str(k)],
                         stdout=open(LOG / f"{s}_k{k}.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {s}_k{k} on GPU {gpu}")
    return p


while jobs and pool:
    j = jobs.popleft(); gpu = pool.pop(0); run[gpu] = (launch(j, gpu), f"{j[0]}_k{j[1]}")
done = 0
while run:
    time.sleep(10); fin = []
    for gpu, (p, n) in list(run.items()):
        if p.poll() is not None:
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {n} [{'OK' if p.returncode == 0 else 'FAIL'}]")
            done += 1; fin.append(gpu)
    for gpu in fin:
        if jobs:
            j = jobs.popleft(); run[gpu] = (launch(j, gpu), f"{j[0]}_k{j[1]}")
        else:
            del run[gpu]
print(f"\n全部完成：{done}")
