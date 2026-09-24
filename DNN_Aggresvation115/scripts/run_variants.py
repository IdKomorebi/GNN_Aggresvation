# -*- coding: utf-8 -*-
"""115 号 步骤 2：同一批冻结主干上，比较读出方式对"字段风险 M"的保真度。只估计规模 ≤3 的集合（K≤2 所需）。
  A 原口径三种子拼接、B 原口径单种子          —— 直接取各实验已有的 est.npz（run_stage 的输出）
  C 三种子拼接 + 数值修正（sd≥1e-3、测试特征截断到训练范围），holdout 选 λ
  D 单种子 + 数值修正 + 5 折选 λ
  E 三个单种子读出（均为 D 的设置）的预测取平均
  F 交叉拟合：训练集随机分两半，每半用 E 的设置各拟一次读出 → 两份独立估计 V̂_a、V̂_b；
    计算 M 时一半挑背景、另一半评估（见 analyze115.py）
全部只用训练集信息；测试集只用于计算 R²。
用法：run_variants.py --exp <实验目录或组目录> --gpu 0"""
import os, sys, json, time, argparse, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); from readout import ridge_predict, r2_from_pred  # noqa: E402


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


orc = _load("orc69", os.path.join(REPO, "DNN_Aggresvation69/src/oracle.py"))
fr = _load("fr91", os.path.join(REPO, "DNN_Aggresvation91/src/featridge.py"))
ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}")
O = os.path.join(os.path.abspath(a.exp), "outputs")
z = np.load(os.path.join(O, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
sel = np.array([len(k) <= 3 for k in keys]); masks = torch.as_tensor(z["masks"][sel], device=dev)
p, C = z["Xtr"].shape[1], z["Ytr"].shape[1]
models = []
for s in range(3):
    m = orc.MLPOracle(p, C).to(dev)
    m.load_state_dict(torch.load(os.path.join(O, f"oracle_seed{s}.pt"), map_location=dev, weights_only=True)); models.append(m.eval())
phis = [fr.FrozenPhi(m, "last") for m in models]
Xtr = torch.as_tensor(z["Xtr"], device=dev); Ytr = torch.as_tensor(z["Ytr"], device=dev, dtype=torch.float64)
Xte = torch.as_tensor(z["Xte"], device=dev); Yte = torch.as_tensor(z["Yte"], device=dev, dtype=torch.float64)
n = len(Xtr); fi, vi = [t.cpu().numpy() for t in fr.fit_val_idx(n, dev)]
folds = np.array_split(np.random.RandomState(0).permutation(n), 5)
half = np.random.RandomState(7).permutation(n); halves = [half[: n // 2], half[n // 2:]]
FIX = dict(sd_floor=1e-3, clip_te=True)


def feats(X, m, seeds):
    xm = X.unsqueeze(0) * m.unsqueeze(1)
    return torch.cat([xm, xm ** 2] + [phis[s](X, m) for s in seeds], 2).double()


def holdout(idx):
    r = np.random.RandomState(11).permutation(len(idx)); nv = round(0.15 * len(idx))
    return r[nv:], r[:nv]


out = {k: [] for k in ["C", "D", "E", "Fa", "Fb"]}
t0 = time.time(); B = 8 if p <= 20 else 4
for s in range(0, len(masks), B):
    m = masks[s:s + B]
    F3tr, F3te = feats(Xtr, m, [0, 1, 2]), feats(Xte, m, [0, 1, 2])
    out["C"].append(r2_from_pred(ridge_predict(F3tr, Ytr, F3te, fi, vi, **FIX), Yte))
    preds, pa, pb = [], [], []
    for sd in range(3):
        Ftr, Fte = feats(Xtr, m, [sd]), feats(Xte, m, [sd])
        preds.append(ridge_predict(Ftr, Ytr, Fte, fi, vi, cv="kfold", folds=folds, **FIX))
        for h, acc in zip(halves, (pa, pb)):
            f_, v_ = holdout(h)
            acc.append(ridge_predict(Ftr[:, h], Ytr[h], Fte, f_, v_, **FIX))
    out["D"].append(r2_from_pred(preds[0], Yte))
    out["E"].append(r2_from_pred(sum(preds) / 3, Yte))
    out["Fa"].append(r2_from_pred(sum(pa) / 3, Yte)); out["Fb"].append(r2_from_pred(sum(pb) / 3, Yte))
    if s % (B * 50) == 0:
        print(f"{s + len(m)}/{len(masks)} {time.time() - t0:.0f}s", flush=True)
res = {k: torch.cat(v).cpu().numpy().astype(np.float32) for k, v in out.items()}
np.savez(os.path.join(O, "est_variants.npz"), sel=sel, sec_per_set=(time.time() - t0) / len(masks), **res)
print(f"完成 {len(masks)} 个集合，{(time.time() - t0) / len(masks) * 1000:.0f} ms/集合")
