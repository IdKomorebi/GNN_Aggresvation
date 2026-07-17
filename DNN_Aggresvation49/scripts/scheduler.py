#!/usr/bin/env python3
"""DNN49 训练调度器：4 GPU 并行跑 configs/ 下各低数据比例的 multi graph-combo。"""
import os, subprocess, time
from pathlib import Path
from collections import deque
D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation49")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0,1,2,3]; LOG = D/"logs"; LOG.mkdir(exist_ok=True)
queue = deque(sorted((D/"configs").glob("*.yaml"))); pool=list(GPUS); run={}
def launch(cfg,g):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),NUMBA_CACHE_DIR="/tmp/numba_cache")
    p=subprocess.Popen([PY,"-u",str(D/"scripts/run_pipeline.py"),"--config",str(cfg)],
        stdout=open(LOG/f"{cfg.stem}.log","w"),stderr=subprocess.STDOUT,env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {cfg.stem} on GPU {g}"); return p
print(f"DNN49 训练调度器：{len(queue)} 个任务，{len(GPUS)} 卡\n")
while queue and pool: c=queue.popleft(); g=pool.pop(0); run[g]=(launch(c,g),c.stem)
done=0
while run:
    time.sleep(10); fin=[]
    for g,(p,n) in list(run.items()):
        if p.poll() is not None:
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {n} [{'OK' if p.returncode==0 else 'FAIL'}]"); done+=1; fin.append(g)
    for g in fin:
        if queue: c=queue.popleft(); run[g]=(launch(c,g),c.stem)
        else: del run[g]
print(f"\n全部完成：{done}")
