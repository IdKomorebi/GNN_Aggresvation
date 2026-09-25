# -*- coding: utf-8 -*-
"""127 号 步骤 3：TabPFN v2（tabpfn==2.2.1，默认设置）在全部规模 ≤3 集合上：每个（集合，目标）以训练行的 S 列为上下文，直接预测测试行。
分片运行：--shard i --nshard k 处理集合下标 ≡ i (mod k)。输出 outputs/tabpfn/<组>_shard{i}.npz。
须用隔离环境 /data1/duhaocun/envs/tabpfn_venv 且设 SCIPY_ARRAY_API=1。"""
import os, sys, json, time, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); a = ap.parse_args()
T = a.tag.replace("/", "_"); rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[a.tag]; d = registry.load_ds(rel); D, keys = d["D"], d["keys"]
k3 = [k for k in keys if len(k) <= 3]; C = D["Ytr"].shape[1]
out = os.path.join(ROOT, "outputs", "tabpfn"); os.makedirs(out, exist_ok=True); f = os.path.join(out, f"{T}_shard{a.shard}.npz")
idx = [i for i in range(len(k3)) if i % a.nshard == a.shard]; est = np.zeros((len(idx), C), np.float32); t0 = time.time()
reg = TabPFNRegressor(device=f"cuda:{a.gpu}", random_state=0)
for j, i in enumerate(idx):
    S = list(k3[i])
    for c in range(C):
        reg.fit(D["Xtr"][:, S], D["Ytr"][:, c]); est[j, c] = max(0.0, r2_score(D["Yte"][:, c], reg.predict(D["Xte"][:, S])))
    if j % 200 == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {a.tag} 分片{a.shard} {j}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
np.savez(f, est=est, idx=np.array(idx), sec=(time.time() - t0) / len(idx))
print("完成", a.tag, a.shard, time.time() - t0)
