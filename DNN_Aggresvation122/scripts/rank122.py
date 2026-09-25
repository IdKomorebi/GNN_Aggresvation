# -*- coding: utf-8 -*-
"""122 号 步骤 6：秩截断的因果检验——预训练特征有用的是"几个方向"？
对每个集合，把主干特征 φ(x⊙m) 在训练行上标准化后做 SVD，只保留前 r 个主成分，再与 [x⊙m, (x⊙m)²] 拼接做同口径读出。
  random 主干：r = 0（无 φ）、2、4、8、32、全部
  uniform（只预测目标）：r = 2、8、全部  —— 若表征已塌缩，r=2 就应与全部相当
  recon（目标 + 重建）：r = 2、8、全部   —— 若重建的好处来自额外方向，截断到 r=2 应失去优势
每组抽 200 个规模 ≤3 的集合（固定种子），与闭包后的重训真值比较 V 的绝对误差。CPU 多进程、单线程。"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
TAGS = ["RTS-GMLC", "NEM", "PJM-gen/ic", "CAISO-load"]; NS = 200; CH = 25
CONF = [("random", 0), ("random", 2), ("random", 4), ("random", 8), ("random", 32), ("random", -1),
        ("uniform", 2), ("uniform", 8), ("uniform", -1), ("recon", 2), ("recon", 8), ("recon", -1)]
_C = {}


def pick(tag):
    T = tag.replace("/", "_"); n3 = json.load(open(os.path.join(ROOT, "outputs", "est", T, "eval_sets.json")))["n3"]
    return np.sort(np.random.RandomState(122).choice(n3, min(NS, n3), replace=False))


def work(job):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(ROOT, "src")); import m122  # noqa
    sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa
    tag, kind, r, lo, hi = job; T = tag.replace("/", "_")
    if tag not in _C:
        rel = dict((t, q) for t, q, _, _ in registry.DATASETS)[tag]; _C[tag] = registry.load_ds(rel)["D"]
    D = _C[tag]; p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]
    masks = np.load(os.path.join(ROOT, "outputs", "est", T, "eval_masks.npy"))[pick(tag)[lo:hi]]
    mdl = m122.orc.MLPOracle(p, C + p if kind == "recon" else C)
    mdl.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "backbones", T, f"{kind}_seed0.pt"), map_location="cpu", weights_only=True))
    ph = m122.fr.FrozenPhi(mdl.eval(), "last")
    Xtr = torch.as_tensor(D["Xtr"]); Xte = torch.as_tensor(D["Xte"])
    Ytr = torch.as_tensor(D["Ytr"], dtype=torch.float64); Yte = torch.as_tensor(D["Yte"], dtype=torch.float64)
    n = len(Xtr); fi, vi = [t.cpu().numpy() for t in m122.fr.fit_val_idx(n, torch.device("cpu"))]
    folds = np.array_split(np.random.RandomState(0).permutation(n), 5); out = []
    with torch.no_grad():
        for m in torch.as_tensor(masks, dtype=torch.float32):
            m = m[None]; xtr = (Xtr * m)[None].double(); xte = (Xte * m)[None].double(); F = [xtr, xtr ** 2]; G = [xte, xte ** 2]
            if r != 0:
                Ftr = ph(Xtr, m).double(); Fte = ph(Xte, m).double()
                mu = Ftr.mean(1, keepdim=True); sd = Ftr.std(1, keepdim=True).clamp_min(1e-3)
                Ztr = (Ftr - mu) / sd; Zte = (Fte - mu) / sd
                if r > 0:
                    V = torch.linalg.svd(Ztr[0], full_matrices=False)[2][:r].T
                    Ztr, Zte = Ztr @ V, Zte @ V
                F.append(Ztr); G.append(Zte)
            pred = m122.ro.ridge_predict(torch.cat(F, 2), Ytr, torch.cat(G, 2), fi, vi, cv="kfold", folds=folds, **m122.FIX)
            out.append(m122.ro.r2_from_pred(pred, Yte))
    return job, torch.cat(out).numpy()


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa
    sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa
    import pandas as pd
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = [(t, k, r, lo, lo + CH) for t in TAGS for k, r in CONF for lo in range(0, NS, CH)]
    print(f"{len(jobs)} 块", flush=True); t0 = time.time(); res = {}
    with Pool(nj) as pool:
        for i, (job, est) in enumerate(pool.imap_unordered(work, jobs)):
            res.setdefault(job[:3], []).append((job[3], est))
            if i % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    rows = []
    for tag in TAGS:
        d = registry.load_ds(dict((t, q) for t, q, _, _ in registry.DATASETS)[tag]); keys = d["keys"]
        V3 = d["V"][np.array([len(k) <= 3 for k in keys])][pick(tag)]
        for k, r in CONF:
            est = np.concatenate([e for _, e in sorted(res[(tag, k, r)], key=lambda x: x[0])])
            for c, y in enumerate(d["spec"]["targ"]):
                rows.append(dict(数据=tag, 目标=y.replace("Y_", ""), 主干=k, 秩=("全部" if r < 0 else r),
                                 V误差=float(np.abs(np.clip(est[:, c], 0, 1) - V3[:, c]).mean()), 偏差=float((np.clip(est[:, c], 0, 1) - V3[:, c]).mean())))
    R = pd.DataFrame(rows); os.makedirs(os.path.join(ROOT, "outputs", "analysis"), exist_ok=True)
    R.to_csv(os.path.join(ROOT, "outputs", "analysis", "rank_truncation.csv"), index=False)
    print(R.pivot_table(index=["主干", "秩"], columns="数据", values="V误差", sort=False).round(4).to_string())
    print("全部完成", time.time() - t0)
