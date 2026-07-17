#!/usr/bin/env python3
"""DNN49 评估调度器：4 GPU 并行,对每个 frac 模型跑 evaluate_lowdata。"""
import os, subprocess, time, glob
from pathlib import Path
from collections import deque
D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation49")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0,1,2,3]; LOG = D/"logs"
tasks = []
for cfg in sorted((D/"configs").glob("*.yaml")):
    mp = sorted(glob.glob(str(D/f"outputs/tuning/{cfg.stem}/*/training/model.pt")))
    if mp: tasks.append((cfg, mp[-1]))
queue = deque(tasks); pool=list(GPUS); run={}
def launch(t,g):
    cfg,mp=t; env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),NUMBA_CACHE_DIR="/tmp/numba_cache")
    p=subprocess.Popen([PY,"-u",str(D/"scripts/evaluate_lowdata.py"),"--config",str(cfg),"--model",mp],
        stdout=open(LOG/f"eval_{cfg.stem}.log","w"),stderr=subprocess.STDOUT,env=env)
    print(f"  启动 eval {cfg.stem} on GPU {g}"); return p
print(f"DNN49 评估调度器：{len(queue)} 个 frac，{len(GPUS)} 卡\n")
while queue and pool: t=queue.popleft(); g=pool.pop(0); run[g]=(launch(t,g),t[0].stem)
done=0
while run:
    time.sleep(8); fin=[]
    for g,(p,n) in list(run.items()):
        if p.poll() is not None:
            print(f"  完成 eval {n} [{'OK' if p.returncode==0 else 'FAIL'}]"); done+=1; fin.append(g)
    for g in fin:
        if queue: t=queue.popleft(); run[g]=(launch(t,g),t[0].stem)
        else: del run[g]
print(f"\n全部完成：{done}")
