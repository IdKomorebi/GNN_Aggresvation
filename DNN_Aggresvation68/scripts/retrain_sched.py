"""DNN68 调度：中段密集真值重训（DNN 攻击者，逐 conf），4 GPU，跳过已完成。"""
import json, os, subprocess, time
from collections import deque
from pathlib import Path

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation68")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG = D / "logs"; LOG.mkdir(exist_ok=True)
(D / "outputs/retrain").mkdir(parents=True, exist_ok=True)

subs = json.load(open(D / "outputs/subsets.json"))
jobs = deque([s for s in subs if not (D / "outputs/retrain" / f"{s}_dnn_seed0.json").exists()])
print(f"待跑 {len(jobs)} 个重训", flush=True)
pool = list(GPUS); run = {}


def launch(sid, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    p = subprocess.Popen([PY, "-u", str(D / "scripts/retrain_worker.py"),
                          "--subset-id", sid, "--struct", "dnn", "--seed", "0"],
                         stdout=open(LOG / f"{sid}.log", "w"), stderr=subprocess.STDOUT, env=env)
    return p, sid


while jobs and pool:
    g = pool.pop(0); run[g] = launch(jobs.popleft(), g)
done = 0; t0 = time.time()
while run:
    time.sleep(3)
    for gpu, (p, n) in list(run.items()):
        if p.poll() is not None:
            done += 1
            if p.returncode != 0:
                print(f"  FAIL {n} rc={p.returncode}", flush=True)
            del run[gpu]
            if jobs:
                run[gpu] = launch(jobs.popleft(), gpu)
print(f"\n全部完成：{done}，用时 {time.time()-t0:.0f}s", flush=True)
