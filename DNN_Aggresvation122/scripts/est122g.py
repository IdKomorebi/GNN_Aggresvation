# -*- coding: utf-8 -*-
"""122 号 步骤 7：GPU 读出，加第三道数值防护——预测截断到训练目标取值范围（clip_y）。
动机：PJM-gen/ic 上重建主干含"尖峰探测"单元（训练行几乎常数、少数行激活，标准化后 |z| 达 25–68），
单字段集合 {da_as_total_mw_*} 的少数测试行预测爆掉、V̂=0，Δ̂ 以其为背景的字段全部被高估。
截断只用训练标签的最小/最大值，不看测试标签。输出 outputs/est/<组>/<变体>_cy.npz。
用法：est122g.py --gpu 0 --tags CAISO-load --variants random,uniform,recon,recon+random,uniformE,reconE"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src")); import m122  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts")); from est122 import VARIANTS  # noqa: E402
m122.FIX.update(clip_y=True)

ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--tags", required=True)
ap.add_argument("--variants", default="random,uniform,recon,recon+random,uniformE,reconE"); a = ap.parse_args()
dev = f"cuda:{a.gpu}"
for tag in a.tags.split(","):
    T = tag.replace("/", "_"); rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]; D = registry.load_ds(rel)["D"]
    p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]; masks = np.load(os.path.join(ROOT, "outputs", "est", T, "eval_masks.npy"))

    def load(k):
        kind, sd = (k.split("@") + ["0"])[:2]; sd = int(sd)
        mdl = m122.orc.MLPOracle(p, C + p if kind.endswith("recon") else C)
        f = os.path.join(REPO, rel, f"oracle_seed{sd}.pt") if (kind == "uniform" and sd > 0) else \
            os.path.join(ROOT, "outputs", "backbones", T, f"{kind}_seed{sd}.pt")
        mdl.load_state_dict(torch.load(f, map_location="cpu", weights_only=True)); return m122.fr.FrozenPhi(mdl.eval().to(dev), "last")
    for v in a.variants.split(","):
        out = os.path.join(ROOT, "outputs", "est", T, f"{v}_cy.npz")
        if os.path.exists(out):
            continue
        t0 = time.time(); sv = VARIANTS[v]
        if isinstance(sv[0], list):
            est, sec = m122.readout_avg(D, [[load(k) for k in g] for g in sv], masks, dev)
        else:
            est, sec = m122.readout(D, [load(k) for k in sv], masks, dev, B=4)
        np.savez(out, est=est, sec=sec); print(f"[{time.strftime('%H:%M:%S')}] 完成 {tag} {v}  {time.time() - t0:.0f}s", flush=True)
print("全部完成")
