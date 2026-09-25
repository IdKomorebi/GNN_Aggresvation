# -*- coding: utf-8 -*-
"""126 号：一次性预训练耗时（同一张空闲 GPU，逐组训练一个只预测目标主干与一个重建主干，种子 0，设置与 122 号相同）。
权重不保存（只计时）。输出 outputs/pretrain_time.csv。用法：pretrain126.py --gpu 1"""
import os, sys, argparse
import pandas as pd, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation122", "src")); import m122  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=1); a = ap.parse_args(); dev = f"cuda:{a.gpu}"; rows = []
for tag, rel, _, _ in registry.DATASETS:
    D = registry.load_ds(rel)["D"]
    for kind in ("uniform", "recon"):
        torch.cuda.synchronize(dev); _, info = m122.train(D, kind, 0, dev)
        rows.append(dict(数据=tag, 主干=kind, 秒=info["sec"], 轮数=info["epochs"], 训练行=len(D["fit_idx"])))
        print(tag, kind, f"{info['sec']:.1f}s", info["epochs"], "轮", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "outputs", "pretrain_time.csv"), index=False)
