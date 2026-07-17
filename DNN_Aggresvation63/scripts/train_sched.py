"""DNN63 训练调度：满数据 3 种子 × 2 arch + 低数据 {0.1,0.2,0.5} × 2 种子 × 2 arch。
4 GPU 并行，跳过已完成。"""
import os, subprocess, time
from collections import deque
from pathlib import Path

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation63")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG = D / "logs"; LOG.mkdir(exist_ok=True)

jobs = []
for arch in ["mlp", "gnn"]:
    for seed in [0, 1, 2]:                       # 满数据 3 种子
        jobs.append((arch, seed, 1.0))
    for frac in [0.1, 0.2, 0.5]:                 # 低数据 2 种子
        for seed in [0, 1]:
            jobs.append((arch, seed, frac))


def done(arch, seed, frac):
    tag = f"_f{frac}" if frac < 1.0 else ""
    return (D / "outputs" / f"oracle_{arch}_seed{seed}{tag}.pt").exists()


jobs = deque([j for j in jobs if not done(*j)])
print(f"待跑 {len(jobs)} 个训练任务（已跳过完成的）", flush=True)
pool = list(GPUS); run = {}


def launch(job, gpu):
    arch, seed, frac = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    name = f"{arch}_s{seed}_f{frac}"
    p = subprocess.Popen([PY, "-u", str(D / "scripts/train_oracle.py"),
                          "--arch", arch, "--seed", str(seed), "--data-frac", str(frac)],
                         stdout=open(LOG / f"{name}.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {name} on GPU {gpu}", flush=True)
    return p, name


while jobs and pool:
    g = pool.pop(0); run[g] = launch(jobs.popleft(), g)
n_done = 0
while run:
    time.sleep(5)
    for gpu, (p, n) in list(run.items()):
        if p.poll() is not None:
            n_done += 1
            ok = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
            print(f"  [{time.strftime('%H:%M:%S')}] 完成 {n} [{ok}]", flush=True)
            del run[gpu]
            if jobs:
                run[gpu] = launch(jobs.popleft(), gpu)
print(f"\n全部完成：{n_done}", flush=True)
