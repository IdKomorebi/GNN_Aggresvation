"""DNN64 调度：50 个无偏随机子集 × 2 种子的微调收敛探针，4 GPU 并行，跳过已完成。"""
import json, os, subprocess, time
from collections import deque
from pathlib import Path

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation64")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG = D / "logs"; LOG.mkdir(exist_ok=True)
(D / "outputs/ft").mkdir(parents=True, exist_ok=True)

subs = json.load(open(D.parent / "DNN_Aggresvation60/outputs/subsets.json"))
rs = [k for k, v in subs.items() if v["group"] == "random_eval"]
jobs = deque([(s, seed) for s in rs for seed in [0, 1]
              if not (D / "outputs/ft" / f"{s}_seed{seed}.json").exists()])
print(f"待跑 {len(jobs)} 个微调探针（已跳过完成的）", flush=True)

pool = list(GPUS); run = {}


def launch(job, gpu):
    s, seed = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    p = subprocess.Popen([PY, "-u", str(D / "scripts/finetune_probe.py"),
                          "--subset-id", s, "--seed", str(seed)],
                         stdout=open(LOG / f"{s}_seed{seed}.log", "w"), stderr=subprocess.STDOUT, env=env)
    return p, f"{s}_seed{seed}"


while jobs and pool:
    g = pool.pop(0); run[g] = launch(jobs.popleft(), g)
n_done = 0
while run:
    time.sleep(5)
    for gpu, (p, n) in list(run.items()):
        if p.poll() is not None:
            n_done += 1
            ok = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
            print(f"  [{time.strftime('%H:%M:%S')}] {n} [{ok}] ({n_done})", flush=True)
            del run[gpu]
            if jobs:
                run[gpu] = launch(jobs.popleft(), gpu)
print(f"\n全部完成：{n_done}", flush=True)
