#!/usr/bin/env python3
"""DNN51 聚合消融调度器：4 GPU 并行跑 configs/ 下 5 个聚合变体。"""
import os, subprocess, time
from pathlib import Path
from collections import deque

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation51")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1]  # GPU 2/3 被其他任务占用，避让；模型小且快，双卡轮转即可
LOG = D / "logs"; LOG.mkdir(exist_ok=True)
# 顺序：先基准，再 4 个消融
order = ["split_baseline", "gcn_noprior", "gcn_prior", "gat_noprior", "gat_prior"]
queue = deque(D / "configs" / f"{n}.yaml" for n in order)
pool = list(GPUS); run = {}


def launch(cfg, g):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), NUMBA_CACHE_DIR="/tmp/numba_cache")
    p = subprocess.Popen(
        [PY, "-u", str(D / "scripts/train_variant.py"), "--config", str(cfg)],
        stdout=open(LOG / f"{cfg.stem}.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {cfg.stem} on GPU {g}")
    return p


print(f"DNN51 聚合消融调度器：{len(queue)} 个任务，{len(GPUS)} 卡\n")
while queue and pool:
    c = queue.popleft(); g = pool.pop(0); run[g] = (launch(c, g), c.stem)
done = 0
while run:
    time.sleep(10); fin = []
    for g, (p, n) in list(run.items()):
        if p.poll() is not None:
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {n} [{'OK' if p.returncode == 0 else 'FAIL'}]")
            done += 1; fin.append(g)
    for g in fin:
        if queue:
            c = queue.popleft(); run[g] = (launch(c, g), c.stem)
        else:
            del run[g]
print(f"\n全部完成：{done}")
