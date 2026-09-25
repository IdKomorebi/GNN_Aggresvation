# -*- coding: utf-8 -*-
"""122 号 步骤 5：为"重建主干 + 三种子预测平均"补训 recon 的种子 1、2（与主口径 E 的三种子对等比较）。
用法：train_seeds.py --tags ... --dev cuda:1|cpu"""
import os, sys, argparse, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src")); import m122  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--dev", default="cpu"); ap.add_argument("--kind", default="recon"); a = ap.parse_args()
torch.set_num_threads(2)
for tag in a.tags.split(","):
    rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]; D = registry.load_ds(rel)["D"]; T = tag.replace("/", "_")
    for s in (1, 2):
        f = os.path.join(ROOT, "outputs", "backbones", T, f"{a.kind}_seed{s}.pt")
        if os.path.exists(f):
            continue
        mdl, inf = m122.train(D, a.kind, s, a.dev); torch.save({k: v.cpu() for k, v in mdl.state_dict().items()}, f)
        print(f"[{time.strftime('%H:%M:%S')}] {tag} {a.kind} 种子 {s}：{inf}", flush=True)
