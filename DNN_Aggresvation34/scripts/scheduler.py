#!/usr/bin/env python3
"""DNN34 调度器：4 GPU 并行跑 configs/ 下的全部实验。

DNN34 共 5 组：baseline_1head（单头基线）+ static_{4,8}head（GLM 静态 gate）
+ dynamic_{4,8}head（用户的动态 head 注意力）。1 head 时两种聚合等价。

策略：维护待运行队列和 GPU 空闲池，每完成一个 run 就立刻在空出的 GPU 上
启动下一个。subprocess + 轮询，稳定可靠。
"""
import os
import subprocess
import time
from pathlib import Path
from collections import deque

D34 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation34")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG_DIR = D34 / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 任务顺序：1head 基线先跑（最快），随后 static / dynamic 多头交替
order = [
    "baseline_1head",
    "dynamic_4head",
    "static_4head",
    "dynamic_8head",
    "static_8head",
]
configs = [D34 / "configs" / f"{n}.yaml" for n in order]
configs = [c for c in configs if c.exists()]

queue = deque(configs)
gpu_pool = list(GPUS)
running = {}  # gpu -> (proc, name)

print(f"DNN34 调度器：{len(configs)} 个任务，{len(GPUS)} 张 GPU")
print(f"任务顺序: {[c.stem for c in configs]}\n")

completed = 0
total = len(configs)


def launch(cfg: Path, gpu: int):
    name = cfg.stem
    log = LOG_DIR / f"{name}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [PY, "-u", str(D34 / "scripts" / "run_pipeline.py"), "--config", str(cfg)]
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {name} on GPU {gpu} (PID {proc.pid})")
    return proc


# 初始填满 GPU
while queue and gpu_pool:
    cfg = queue.popleft()
    gpu = gpu_pool.pop(0)
    running[gpu] = (launch(cfg, gpu), cfg.stem)

# 轮询等待，每完成一个立刻补一个
while running:
    time.sleep(10)
    finished = []
    for gpu, (proc, name) in list(running.items()):
        if proc.poll() is not None:
            rc = proc.returncode
            tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {name} on GPU {gpu} [{tag}]")
            completed += 1
            finished.append(gpu)
            del running[gpu]
    for gpu in finished:
        if queue:
            cfg = queue.popleft()
            running[gpu] = (launch(cfg, gpu), cfg.stem)
        else:
            gpu_pool.append(gpu)

print(f"\n全部完成：{completed}/{total}")
