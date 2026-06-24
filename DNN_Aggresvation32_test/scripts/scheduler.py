#!/usr/bin/env python3
"""DNN32_test 调度器：4 GPU 并行跑 13 个 run（1 multi + 12 single）。

策略：维护待运行队列和 GPU 空闲池，每有一个 run 完成就立刻在空出来的
GPU 上启动下一个。用 subprocess + 轮询，稳定可靠。
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import deque

D32 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation32_test")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG_DIR = D32 / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 构建任务队列：(config_name, gpu)
configs = sorted((D32 / "configs").glob("*.yaml"))
configs = [c for c in configs if c.name != "multi.yaml"]  # single 先跑
configs.append(D32 / "configs" / "multi.yaml")  # multi 最后

queue = deque()
gpu_pool = list(GPUS)
running = {}  # gpu -> (proc, config_name)

for cfg in configs:
    queue.append(cfg)

print(f"DNN32_test 调度器：{len(configs)} 个任务，{len(GPUS)} 张 GPU")
print(f"任务顺序: {[c.name for c in configs]}\n")

completed = 0
total = len(configs)

def launch(cfg: Path, gpu: int):
    name = cfg.stem
    # 覆盖 config 里的 device
    log = LOG_DIR / f"{name}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    cmd = [PY, "-u", str(D32 / "scripts" / "run_pipeline.py"), "--config", str(cfg)]
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {name} on GPU {gpu} (PID {proc.pid})")
    return proc

# 初始填满 4 GPU
while queue and gpu_pool:
    cfg = queue.popleft()
    gpu = gpu_pool.pop(0)
    proc = launch(cfg, gpu)
    running[gpu] = (proc, cfg.stem)

# 轮询等待完成，每完成一个立刻补一个
while running:
    time.sleep(10)
    finished_gpus = []
    for gpu, (proc, name) in list(running.items()):
        if proc.poll() is not None:
            rc = proc.returncode
            tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {name} on GPU {gpu} [{tag}]")
            completed += 1
            finished_gpus.append(gpu)
            del running[gpu]
    for gpu in finished_gpus:
        if queue:
            cfg = queue.popleft()
            proc = launch(cfg, gpu)
            running[gpu] = (proc, cfg.stem)
        else:
            gpu_pool.append(gpu)

print(f"\n全部完成：{completed}/{total}")
