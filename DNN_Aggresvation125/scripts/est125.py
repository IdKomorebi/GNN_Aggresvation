# -*- coding: utf-8 -*-
"""125 号 步骤 1：候选 C3（重建⊕随机拼接 × 3 种子预测平均，三道防护）的读出。
复用 122 号的主干（outputs/backbones/<组>/{recon,random}_seed{0,1,2}.pt）与评测集合（eval_masks.npy）。
批量版三种子平均：每批 B 个集合，每个种子一组特征 [x⊙m, (x⊙m)², φ_recon_s, φ_random_s] 单独闭式岭读出，预测平均后算 R²。
输出 outputs/est/<组>/RRE_cy.npz（est: (N,C)，sec: 单集合耗时）。
用法：est125.py --gpu 0 --tags CAISO-load"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
B122 = os.path.join(REPO, "DNN_Aggresvation122")
sys.path.insert(0, os.path.join(B122, "src")); import m122  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
FIX = dict(m122.FIX, clip_y=True)
SEEDS = (0, 1, 2)


def load_phi(T, kind, sd, p, C, dev):
    mdl = m122.orc.MLPOracle(p, C + p if kind == "recon" else C)
    mdl.load_state_dict(torch.load(os.path.join(B122, "outputs", "backbones", T, f"{kind}_seed{sd}.pt"), map_location="cpu", weights_only=True))
    return m122.fr.FrozenPhi(mdl.eval().to(dev), "last")


@torch.no_grad()
def estimate(D, groups, masks, dev, B=8):
    """groups：每个种子一组 FrozenPhi 列表。返回 (N,C) 测试 R² 与单集合耗时。"""
    Xtr = torch.as_tensor(D["Xtr"], device=dev); Xte = torch.as_tensor(D["Xte"], device=dev)
    Ytr = torch.as_tensor(D["Ytr"], device=dev, dtype=torch.float64); Yte = torch.as_tensor(D["Yte"], device=dev, dtype=torch.float64)
    n = len(Xtr); fi, vi = [t.cpu().numpy() for t in m122.fr.fit_val_idx(n, dev)]
    folds = np.array_split(np.random.RandomState(0).permutation(n), 5)
    Mt = torch.as_tensor(np.asarray(masks, np.float32), device=dev); out = []
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)
    t0 = time.time()
    for s in range(0, len(Mt), B):
        m = Mt[s:s + B]; preds = []
        for phis in groups:
            def feats(X):
                xm = X.unsqueeze(0) * m.unsqueeze(1)
                return torch.cat([xm, xm ** 2] + [ph(X, m) for ph in phis], 2).double()
            preds.append(m122.ro.ridge_predict(feats(Xtr), Ytr, feats(Xte), fi, vi, cv="kfold", folds=folds, **FIX))
        out.append(m122.ro.r2_from_pred(sum(preds) / len(preds), Yte))
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)
    return torch.cat(out).cpu().numpy().astype(np.float32), (time.time() - t0) / max(len(Mt), 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--tags", required=True)
    ap.add_argument("--B", type=int, default=4); a = ap.parse_args(); dev = f"cuda:{a.gpu}"
    for tag in a.tags.split(","):
        T = tag.replace("/", "_"); rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]; D = registry.load_ds(rel)["D"]
        p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]; masks = np.load(os.path.join(B122, "outputs", "est", T, "eval_masks.npy"))
        out = os.path.join(ROOT, "outputs", "est", T); os.makedirs(out, exist_ok=True)
        if os.path.exists(os.path.join(out, "RRE_cy.npz")):
            continue
        groups = [[load_phi(T, "recon", sd, p, C, dev), load_phi(T, "random", sd, p, C, dev)] for sd in SEEDS]
        t0 = time.time(); est, sec = estimate(D, groups, masks, dev, B=a.B)
        np.savez(os.path.join(out, "RRE_cy.npz"), est=est, sec=sec)
        print(f"[{time.strftime('%H:%M:%S')}] 完成 {tag}  {len(masks)} 个集合  {time.time() - t0:.0f}s", flush=True)
    print("全部完成")
