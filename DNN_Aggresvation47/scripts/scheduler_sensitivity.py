#!/usr/bin/env python3
"""DNN47 敏感度调度器：4 GPU 并行跑四种方法 (mask/single/attn/ig)。"""
import os
import subprocess
import time
from pathlib import Path

D47 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation47")
PY = "/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python"
METHODS = ["mask", "single", "attn", "ig"]
LOG_DIR = D47 / "logs"
LOG_DIR.mkdir(exist_ok=True)

procs = {}
for gpu, mth in enumerate(METHODS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NUMBA_CACHE_DIR="/tmp/numba_cache")
    log = open(LOG_DIR / f"sens_{mth}.log", "w")
    cmd = [PY, "-u", str(D47 / "scripts" / "compute_sensitivity.py"), "--method", mth]
    procs[mth] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"  [{time.strftime('%H:%M:%S')}] 启动 {mth} on GPU {gpu} (PID {procs[mth].pid})")

failed = []
for mth, p in procs.items():
    rc = p.wait()
    tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
    print(f"  [{time.strftime('%H:%M:%S')}] 完成 {mth} [{tag}]")
    if rc != 0:
        failed.append(mth)
print(f"\n全部完成。失败: {failed if failed else '无'}")
