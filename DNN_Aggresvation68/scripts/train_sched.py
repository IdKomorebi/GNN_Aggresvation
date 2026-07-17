"""DNN68 专家训练调度：3 容量配置 × 5 难度带 = 15 个专家，4 GPU 并行。
- equal   : 每专家 h=256（5× uniform 参数，慷慨上界）
- matched : 每专家 h=104（5×≈162k ≈ uniform 157k，同预算按尺寸均分）
- tilt    : 中段大两端小（≈同预算，按难度分配）
uniform 对照用 63 号现成 oracle_mlp_seed0.pt。
"""
import os, subprocess, time
from collections import deque
from pathlib import Path

D = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation68")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
GPUS = [0, 1, 2, 3]
LOG = D / "logs_train"; LOG.mkdir(exist_ok=True)

BANDS = [("E1", 1, 5), ("E2", 6, 9), ("E3", 10, 13), ("E4", 14, 17), ("E5", 18, 44)]
CONFIGS = {
    "equal":   {"E1": 256, "E2": 256, "E3": 256, "E4": 256, "E5": 256},
    "matched": {"E1": 104, "E2": 104, "E3": 104, "E4": 104, "E5": 104},
    "tilt":    {"E1": 64,  "E2": 128, "E3": 128, "E4": 96,  "E5": 64},
}

jobs = deque()
for cfg, hmap in CONFIGS.items():
    for band, lo, hi in BANDS:
        h = hmap[band]
        tag = f"{cfg}_{band}_h{h}"
        if not (D / "outputs" / f"expert_{cfg}_{band}_h{h}_seed0.pt").exists():
            jobs.append((cfg, band, lo, hi, h))
print(f"待训 {len(jobs)} 个专家", flush=True)
pool = list(GPUS); run = {}


def launch(job, gpu):
    cfg, band, lo, hi, h = job
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    name = f"{cfg}_{band}_h{h}"
    p = subprocess.Popen(
        [PY, "-u", str(D / "scripts/train_banded.py"), "--band", f"{cfg}_{band}",
         "--size-lo", str(lo), "--size-hi", str(hi), "--hidden", str(h), "--seed", "0"],
        stdout=open(LOG / f"{name}.log", "w"), stderr=subprocess.STDOUT, env=env)
    return p, name


while jobs and pool:
    g = pool.pop(0); run[g] = launch(jobs.popleft(), g)
done = 0
while run:
    time.sleep(3)
    for gpu, (p, n) in list(run.items()):
        if p.poll() is not None:
            done += 1
            rc = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
            print(f"  [{time.strftime('%H:%M:%S')}] {n} [{rc}] ({done})", flush=True)
            del run[gpu]
            if jobs:
                run[gpu] = launch(jobs.popleft(), gpu)
print(f"\n全部完成：{done}", flush=True)
