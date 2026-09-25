# -*- coding: utf-8 -*-
"""125 号 步骤 4：单集合耗时（与 118 号 timeE 同口径：同一张空闲 GPU、200 个随机规模 ≤3 集合、批大小 8）。
同时重测旧口径 E（只预测目标 ×3）与新估计器，比值在同一张卡、同一批集合上得到。输出 outputs/time/<组>.npz。
用法：time125.py --gpu 0"""
import os, sys, time, argparse
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts")); import est125 as E  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args(); dev = f"cuda:{a.gpu}"
os.makedirs(os.path.join(ROOT, "outputs", "time"), exist_ok=True)
for tag, rel, _, _ in E.registry.DATASETS:
    T = tag.replace("/", "_"); D = E.registry.load_ds(rel)["D"]; keys = E.registry.load_ds(rel)["keys"]
    p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]; n3 = sum(len(k) <= 3 for k in keys)
    masks = np.load(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "est", T, "eval_masks.npy"))[:n3]
    pick = np.random.RandomState(2).choice(n3, min(200, n3), replace=False); M = masks[pick]
    old = []
    for sd in range(3):   # 旧口径 E 的主干：各号 oracle_seed{sd}.pt
        mdl = E.m122.orc.MLPOracle(p, C); mdl.load_state_dict(torch.load(os.path.join(REPO, rel, f"oracle_seed{sd}.pt"), map_location="cpu", weights_only=True))
        old.append([E.m122.fr.FrozenPhi(mdl.eval().to(dev), "last")])
    new = [[E.load_phi(T, "recon", sd, p, C, dev), E.load_phi(T, "random", sd, p, C, dev)] for sd in range(3)]
    E.estimate(D, old, M[:8], dev, B=8)   # 预热
    _, t_old = E.estimate(D, old, M, dev, B=8); _, t_new = E.estimate(D, new, M, dev, B=8)
    np.savez(os.path.join(ROOT, "outputs", "time", f"{T}.npz"), old=t_old, new=t_new, n3=n3, p=p)
    print(f"{tag}: 旧 E {t_old * 1000:.0f} ms，新估计器 {t_new * 1000:.0f} ms（每集合）", flush=True)
