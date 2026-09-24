# -*- coding: utf-8 -*-
"""117 号 步骤 1：各种字段级打分方法（全部只用训练集；测试集不参与任何打分）。

  pearson  —— 单字段线性相关 r²（现行"相关性"口径）
  spearman —— 单字段秩相关 ρ²
  mi       —— 单字段互信息（kNN 估计，sklearn）
  dcor     —— 单字段距离相关
  loco     —— 留一字段：全字段模型 val R² − 去掉该字段后重训的 val R²（梯度提升树）
  perm     —— 置换重要性：全字段模型上打乱该字段后 val R² 的下降（5 次平均）
  sage     —— SAGE（Covert et al. 2020）：全字段模型 + 边际填补，排列抽样估计 Shapley 形式的全局重要性
  graph    —— 仿 IGNN 的成对推断图传播：边权 w_ij = 由字段 i 单独推断字段 j 的 val R²（梯度提升树），
              目标节点风险为 1，沿图传播 3 跳（Katz 形式，衰减 0.5）；s_i = Σ_h 0.5^{h-1} (W^h)_{i,y}
M^(0)、M^(2)（精确 / 通用模型估计 / 扫描—认证）在 analyze117.py 里由真值表与估计表直接算。
用法：scores.py --jobs 40"""
import os, sys, argparse, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"   # 共享服务器：每个进程单线程（dcor 会拉起 numba 线程池）
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); import registry  # noqa: E402

G = {}


def _init(D):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1); G.update(D)


def hgb():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=31, max_iter=300, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def job_fit(args):
    """在 fit 行上用字段集合 cols 预测列 target（X 的列或 Y 的列），返回 val R²。"""
    cols, tgt = args
    X, fi, vi = G["X"], G["fi"], G["vi"]; y = G["Y"][:, tgt[1]] if tgt[0] == "Y" else X[:, tgt[1]]
    m = hgb().fit(X[fi][:, cols], y[fi]); return r2(y[vi], m.predict(X[vi][:, cols]))


def job_full(c):
    """全字段模型上的 perm 与 SAGE（目标 c）。"""
    X, Y, fi, vi = G["X"], G["Y"], G["fi"], G["vi"]; y = Y[:, c]; p = X.shape[1]
    m = hgb().fit(X[fi], y[fi]); Xv, yv = X[vi], y[vi]; base = r2(yv, m.predict(Xv)); rng = np.random.RandomState(0)
    perm = np.zeros(p)
    for i in range(p):
        d = []
        for _ in range(5):
            Xp = Xv.copy(); Xp[:, i] = Xp[rng.permutation(len(Xp)), i]; d.append(base - r2(yv, m.predict(Xp)))
        perm[i] = np.mean(d)
    # SAGE：边际填补。v(S) = R²(y, E_bg[f(x_S, X_bg,~S)])，排列抽样
    ne, nb, npm = min(512, len(Xv)), 32, 24
    ev = rng.choice(len(Xv), ne, replace=False); bg = X[fi][rng.choice(len(fi), nb, replace=False)]
    Xe, ye = Xv[ev], yv[ev]

    def val(S):
        Z = np.repeat(bg[None], ne, 0).copy()          # (ne,nb,p)
        if S:
            Z[:, :, S] = Xe[:, None, S]
        pr = m.predict(Z.reshape(-1, p)).reshape(ne, nb).mean(1)
        return r2(ye, pr)
    sage = np.zeros(p); v0 = val([])
    for _ in range(npm):
        order = rng.permutation(p); S = []; prev = v0
        for i in order:
            S.append(i); cur = val(S); sage[i] += cur - prev; prev = cur
    return c, perm, sage / npm


def score_all(X, Y, fi, vi, cand, targ, jobs=40):
    """返回 (逐 目标×字段 的打分表, 候选间成对推断权重 W)。只用训练行（fit 拟合、val 评估）。"""
    import dcor
    from scipy.stats import spearmanr
    from sklearn.feature_selection import mutual_info_regression
    p, C = X.shape[1], Y.shape[1]
    rows = []; rng = np.random.RandomState(0); sub = rng.choice(len(X), min(2000, len(X)), replace=False)
    for c in range(C):
        y = Y[:, c]
        mi = mutual_info_regression(X, y, random_state=0)
        for i in range(p):
            rows.append(dict(目标=targ[c], 字段=cand[i], pearson=np.corrcoef(X[:, i], y)[0, 1] ** 2,
                             spearman=spearmanr(X[:, i], y)[0] ** 2, mi=mi[i],
                             dcor=dcor.distance_correlation(X[sub, i].astype(np.float64), y[sub].astype(np.float64))))
    S = pd.DataFrame(rows)
    with Pool(jobs, initializer=_init, initargs=(dict(X=X, Y=Y, fi=fi, vi=vi),)) as pool:
        allc = list(range(p))   # LOCO
        jobs_ = [(allc, ("Y", c)) for c in range(C)] + [([j for j in allc if j != i], ("Y", c)) for c in range(C) for i in allc]
        res = pool.map(job_fit, jobs_, chunksize=1)
        full = np.array(res[:C]); drop = np.array(res[C:]).reshape(C, p)
        S["loco"] = np.concatenate([full[c] - drop[c] for c in range(C)])
        pair = [([i], ("X", j)) for i in allc for j in allc if i != j]   # 成对推断图
        wx = pool.map(job_fit, pair, chunksize=4)
        W = np.zeros((p, p)); k = 0
        for i in allc:
            for j in allc:
                if i != j:
                    W[i, j] = max(wx[k], 0); k += 1
        wy = np.array(pool.map(job_fit, [([i], ("Y", c)) for c in range(C) for i in allc], chunksize=2)).reshape(C, p).clip(0)
        fullres = dict((c, (pm, sg)) for c, pm, sg in pool.map(job_full, range(C), chunksize=1))
    gs, pm, sg = [], [], []
    for c in range(C):
        w = wy[c]; s_ = w.copy(); h = w.copy()
        for hop in (2, 3):
            h = W @ h; s_ = s_ + 0.5 ** (hop - 1) * h
        gs.append(s_); pm.append(fullres[c][0]); sg.append(fullres[c][1])
    S["graph"] = np.concatenate(gs); S["perm"] = np.concatenate(pm); S["sage"] = np.concatenate(sg)
    S["pair_tree_r2"] = np.concatenate(list(wy))
    return S, W


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=40); ap.add_argument("--only", default="")
    a = ap.parse_args()
    for tag, rel, _, _ in registry.DATASETS:
        if a.only and tag not in a.only.split(","):
            continue
        out = os.path.join(ROOT, "outputs", f"scores_{tag.replace('/', '_')}.csv")
        if os.path.exists(out):
            print("已存在，跳过", tag); continue
        t0 = time.time(); d = registry.load_ds(rel); D, spec = d["D"], d["spec"]
        S, W = score_all(D["Xtr"], D["Ytr"], D["fit_idx"], D["val_idx"], spec["cand"], spec["targ"], a.jobs)
        S.to_csv(out, index=False); np.save(out.replace(".csv", "_W.npy"), W)
        print(f"{tag}: {len(spec['cand'])} 字段 × {len(spec['targ'])} 目标，用时 {time.time() - t0:.0f}s", flush=True)
