#!/usr/bin/env python3
"""DNN60 调度器：噪声底(10 子集×5 种子×2 结构) + 随机评测(50 子集×2 结构) = 200 次重训。
4 GPU 并行，跳过已完成。GCN 任务先排（耗时长），DNN 任务后补。
"""
import json, os, subprocess, time
from collections import deque
from pathlib import Path

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation60")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG = D / "logs"; LOG.mkdir(exist_ok=True)

subsets = json.load(open(D / "outputs/subsets.json"))
jobs = []
for sid, info in subsets.items():
    seeds = range(5) if info["group"] == "noise_floor" else [0]
    for struct in ["gcn", "dnn"]:
        for seed in seeds:
            if not (D / "outputs/retrain" / f"{sid}_{struct}_seed{seed}.json").exists():
                jobs.append((sid, struct, seed))
# GCN 任务先跑（单个 ~6-7 分钟），DNN（<1 分钟）后补
jobs = deque(sorted(jobs, key=lambda j: (j[1] != "gcn", j[0], j[2])))
print(f"待跑 {len(jobs)} 次重训（已跳过完成的）")
pool = list(GPUS); run = {}


def launch(job, gpu):
    sid, struct, seed = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    name = f"{sid}_{struct}_seed{seed}"
    p = subprocess.Popen([PY, "-u", str(D / "scripts/retrain_worker.py"),
                          "--subset-id", sid, "--struct", struct, "--seed", str(seed)],
                         stdout=open(LOG / f"{name}.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {name} on GPU {gpu}", flush=True)
    return p, name


while jobs and pool:
    j = jobs.popleft(); gpu = pool.pop(0); run[gpu] = launch(j, gpu)
done = 0
while run:
    time.sleep(10); fin = []
    for gpu, (p, n) in list(run.items()):
        if p.poll() is not None:
            done += 1
            ok = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {n} [{ok}]", flush=True)
            fin.append(gpu)
    for gpu in fin:
        del run[gpu]
        if jobs:
            run[gpu] = launch(jobs.popleft(), gpu)
print(f"\n全部完成：{done}")
