# -*- coding: utf-8 -*-
"""115 号 步骤 1：验证"估计崩为 0"是否来自标准化的数值放大。
只取 NEM（112 号）三种子估计器下 est≈0 而真值 >0.15 的集合，用同一批冻结主干重新估计：
  原口径（sd 下限 1e-8）/ sd 下限 1e-3 / 剔除训练集近常数特征（sd<1e-4 置零）。
同时报告这些集合里"训练集上近常数、测试集上非常数"的特征个数。"""
import os, sys, json, importlib.util
import numpy as np, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); from readout import ridge_predict, r2_from_pred  # noqa: E402


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


orc = _load("orc69", os.path.join(REPO, "DNN_Aggresvation69/src/oracle.py"))
fr = _load("fr91", os.path.join(REPO, "DNN_Aggresvation91/src/featridge.py"))
dev = torch.device(f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 2}")
O = os.path.join(REPO, "DNN_Aggresvation112", "outputs")
z = np.load(os.path.join(O, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
V = np.load(os.path.join(O, "V_official.npy")); E = np.load(os.path.join(O, "est.npz"))["est"]
p, C = z["Xtr"].shape[1], z["Ytr"].shape[1]
bad = np.where(((E <= 1e-6) & (V > 0.15)).any(1))[0]
print(f"崩溃集合数 {len(bad)}（任一目标）")
models = []
for s in range(3):
    m = orc.MLPOracle(p, C).to(dev); m.load_state_dict(torch.load(os.path.join(O, f"oracle_seed{s}.pt"), map_location=dev)); models.append(m.eval())
phis = [fr.FrozenPhi(m, "last") for m in models]
Xtr = torch.as_tensor(z["Xtr"], device=dev); Ytr = torch.as_tensor(z["Ytr"], device=dev, dtype=torch.float64)
Xte = torch.as_tensor(z["Xte"], device=dev); Yte = torch.as_tensor(z["Yte"], device=dev, dtype=torch.float64)
fi, vi = fr.fit_val_idx(len(Xtr), dev)


def feats(X, m):
    xm = X.unsqueeze(0) * m.unsqueeze(1)
    return torch.cat([xm, xm ** 2] + [ph(X, m) for ph in phis], 2).double()


masks = torch.as_tensor(z["masks"][bad], device=dev)
res = {}
n_dead = []
for s in range(0, len(bad), 8):
    m = masks[s:s + 8]; Ftr, Fte = feats(Xtr, m), feats(Xte, m)
    sd_tr, sd_te = Ftr.std(1), Fte.std(1)
    n_dead += ((sd_tr < 1e-4) & (sd_te > 1e-3)).sum(1).tolist()
    folds = np.array_split(np.random.RandomState(0).permutation(len(Xtr)), 5)
    for tag, kw in [("原口径 sd≥1e-8", dict()), ("sd≥1e-3", dict(sd_floor=1e-3)), ("剔除近常数", dict(drop_const=1e-4, sd_floor=1e-4)),
                    ("sd≥1e-3+截断", dict(sd_floor=1e-3, clip_te=True)),
                    ("sd≥1e-3+截断+5折", dict(sd_floor=1e-3, clip_te=True, cv="kfold", folds=folds))]:
        r2 = r2_from_pred(ridge_predict(Ftr, Ytr, Fte, fi.cpu().numpy(), vi.cpu().numpy(), **kw), Yte).cpu().numpy()
        res.setdefault(tag, []).append(r2)
Vb = V[bad]
print(f"这些集合中'训练集近常数、测试集非常数'的特征数：均值 {np.mean(n_dead):.1f}，最大 {np.max(n_dead)}")
for tag, rs in res.items():
    R = np.concatenate(rs); col = (R <= 1e-6) & (Vb > 0.15)
    print(f"{tag:12s} 仍崩溃 {int(col.sum()):4d}；对真值的平均误差 {np.abs(R - Vb).mean():.3f}")
