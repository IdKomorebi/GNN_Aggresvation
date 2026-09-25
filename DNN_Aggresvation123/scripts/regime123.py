# -*- coding: utf-8 -*-
"""123 号 步骤 4：运行状态依赖——按线路 C6 是否阻塞把测试时段分两组，分别计算出力字段 P_g 在不同背景下的边际增益
Δ_P(T) = V(T∪{P_g}) − V(T)（梯度提升树攻击者，两种配置 val 选择；模型在全部训练行上拟合，只在测试行上分组评价 R²）。"""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TAG = os.environ.get("TAG123", "wide"); OUTD = os.path.join(ROOT, "outputs", TAG)
z = np.load(os.path.join(OUTD, "D.npz")); spec = json.load(open(os.path.join(OUTD, "fields.json"), encoding="utf-8"))
cand = spec["cand"]; ix = {c: i for i, c in enumerate(cand)}
Xtr, Ytr, Xte, Yte, fi, vi = z["Xtr"], z["Ytr"][:, 0], z["Xte"], z["Yte"][:, 0], z["fit_idx"], z["val_idx"]
cong = Xte[:, ix["Congestion C6"]] > np.median(Xte[:, ix["Congestion C6"]])     # 标准化后的 0/1 标志
P = ix["P_g (unit 313 output)"]
BGS = {"∅": [], "LMP_g": ["LMP_g (bus 313)"], "LMP_near": ["LMP_near (bus 306)"], "LMP_far": ["LMP_far (bus 303, behind C6)"],
       "LMP_area2": ["LMP_area2 (bus 223)"], "System load": ["System load"], "Congestion C6": ["Congestion C6"],
       "LMP_far + C6": ["LMP_far (bus 303, behind C6)", "Congestion C6"]}


def fit(cols):
    if not cols:
        return np.full(len(Yte), Ytr[fi].mean())
    best = (-np.inf, None)
    for lr, leaf in [(0.05, 15), (0.1, 31)]:
        m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaf, max_iter=300, early_stopping=True,
                                          validation_fraction=0.15, random_state=0).fit(Xtr[fi][:, cols], Ytr[fi])
        v = r2_score(Ytr[vi], m.predict(Xtr[vi][:, cols]))
        if v > best[0]:
            best = (v, m)
    return best[1].predict(Xte[:, cols])


def job(item):
    name, bg = item; cols = [ix[c] for c in bg]
    a, b = fit(cols), fit(cols + [P]); out = dict(背景=name)
    for lab, msk in [("全部", np.ones(len(Yte), bool)), ("C6 不阻塞", ~cong), ("C6 阻塞", cong)]:
        r = lambda pr: max(0.0, r2_score(Yte[msk], pr[msk]))
        out[f"{lab}_V(T)"], out[f"{lab}_V(T+P)"] = r(a), r(b); out[f"{lab}_Δ"] = r(b) - r(a)
    return out


if __name__ == "__main__":
    with Pool(8) as pool:
        R = pd.DataFrame(pool.map(job, list(BGS.items())))
    os.makedirs(os.path.join(OUTD, "analysis"), exist_ok=True); R.to_csv(os.path.join(OUTD, "analysis", "regime.csv"), index=False)
    print(f"测试行：C6 阻塞 {cong.mean():.3f}"); print(R.round(3).to_string(index=False))
