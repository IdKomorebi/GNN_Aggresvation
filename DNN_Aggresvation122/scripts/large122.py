# -*- coding: utf-8 -*-
"""122 号 步骤 3：大集合上的对照——预训练的作用是否随集合规模增大？
每组随机抽规模 6、10、16（不超过 p−1）的集合各 60 个；真值只用梯度提升树（两种配置 val 选择，测试 R²，不做闭包），
这里比较的是同一真值下不同主干的相对误差，树已足够（GPU 被他人占满，CPU 上 DNN 真值太贵）。
然后用 random / uniform / recon / small_recon 四种主干做同口径读出。全部 CPU 多进程。"""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
TAGS = ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen/ic", "CAISO-load"]; SIZES = (6, 10, 16); NPER = 60
VARS = ["random", "uniform", "recon", "small_recon"]
_C = {}


def data(tag):
    if tag not in _C:
        sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa
        rel = dict((t, r) for t, r, _, _ in registry.DATASETS)[tag]; _C[tag] = registry.load_ds(rel)["D"]
    return _C[tag]


def tree_job(args):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    tag, r, s = args; D = data(tag); s = list(s); fi, vi = D["fit_idx"], D["val_idx"]; out = []
    for c in range(D["Ytr"].shape[1]):
        best = (-np.inf, None)
        for lr, leaf in [(0.05, 15), (0.1, 31)]:
            m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaf, max_iter=300, early_stopping=True,
                                              validation_fraction=0.15, random_state=0).fit(D["Xtr"][fi][:, s], D["Ytr"][fi, c])
            v = r2_score(D["Ytr"][vi, c], m.predict(D["Xtr"][vi][:, s]))
            if v > best[0]:
                best = (v, m)
        out.append(max(0.0, r2_score(D["Yte"][:, c], best[1].predict(D["Xte"][:, s]))))
    return tag, r, out


def est_job(args):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(ROOT, "src")); import m122  # noqa
    tag, v, masks = args; D = data(tag); T = tag.replace("/", "_"); p, C = D["Xtr"].shape[1], D["Ytr"].shape[1]
    mdl = m122.orc.MLPOracle(p, C + p if v.endswith("recon") else C)
    mdl.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "backbones", T, f"{v}_seed0.pt"), map_location="cpu", weights_only=True))
    est, _ = m122.readout(D, [m122.fr.FrozenPhi(mdl.eval(), "last")], masks, "cpu", B=1)
    return tag, v, est


if __name__ == "__main__":
    O = os.path.join(ROOT, "outputs", "truth_large"); os.makedirs(O, exist_ok=True); sets = {}
    for tag in TAGS:
        p = data(tag)["Xtr"].shape[1]; rs = np.random.RandomState(122)
        sets[tag] = [tuple(sorted(rs.choice(p, k, replace=False))) for k in SIZES if k < p for _ in range(NPER)]
    json.dump({t: ["|".join(map(str, s)) for s in v] for t, v in sets.items()}, open(os.path.join(O, "sets.json"), "w"))
    t0 = time.time(); truth = {t: [None] * len(v) for t, v in sets.items()}
    with Pool(24) as pool:
        for tag, r, out in pool.imap_unordered(tree_job, [(t, r, s) for t, v in sets.items() for r, s in enumerate(v)], chunksize=2):
            truth[tag][r] = out
    print("树真值完成", time.time() - t0, flush=True)
    jobs = []
    for tag, v in sets.items():
        p = data(tag)["Xtr"].shape[1]; M = np.zeros((len(v), p), np.float32)
        for r, s in enumerate(v):
            M[r, list(s)] = 1
        for va in VARS:
            for lo in range(0, len(M), 30):
                jobs.append((tag, va, M[lo:lo + 30]))
    est = {}
    with Pool(24) as pool:
        for k, (tag, va, e) in enumerate(pool.imap(est_job, jobs)):
            est.setdefault((tag, va), []).append(e)
    for tag in TAGS:
        T = tag.replace("/", "_")
        np.savez(os.path.join(O, f"{T}.npz"), truth=np.array(truth[tag], np.float32), sizes=np.array([len(s) for s in sets[tag]]),
                 **{va: np.concatenate(est[(tag, va)]) for va in VARS})
    print("全部完成", time.time() - t0)
