# -*- coding: utf-8 -*-
"""RQ3：近似 full-MCI（背景规模不限）的候选背景搜索（通用模型 L0）→ 专用重训认证 → 已认证的 MCI 下界。

两类候选（Catav 同样用采样给 MCI 下界）：
  贪心：对每个 (目标 c, 字段 i)，从空背景出发，每步加入使 Δ̃_i(T∪j) 最大的 j，扩展到 |T|=12；
  随机排列前驱：每个字段 600 个随机背景，规模均匀取 4..40。
每个 (i,c) 取贪心路径上 |T|≥4 的 Δ̃ 最大 2 个 + 随机前驱 Δ̃ 最大 2 个 → 写成集合文件 {ds}_mci.npy（T 与 T∪i 各一行），
交给 run_truth.py --set mci 重训认证。大背景真值不做单调闭包（子集太多），报告为 raw 值。
用法：mci_search.py --ds pjm --confs 0,1,2,3 --tag s0   （三卡按目标分片后，merge 合并）
"""
import sys, time, argparse, pickle, glob
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(REPO / "DNN_Aggresvation91/src"))
from common100 import load
from featridge import FrozenPhi, ridge_r2, fit_val_idx, load_oracle

ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--confs", default="")
ap.add_argument("--tag", default="s0"); ap.add_argument("--merge", action="store_true"); ap.add_argument("--steps", type=int, default=12); ap.add_argument("--nrand", type=int, default=600)
a = ap.parse_args()

if a.merge:
    parts = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(ROOT / f"outputs/analysis/{a.ds}_mci_cand_*.pkl")))]
    cands = [c for p in parts for c in p["cands"]]
    nG = parts[0]["nG"]; rows, meta = [], []
    for (c, i, T, d) in cands:
        for with_i in (0, 1):
            m = np.zeros(nG, np.uint8); m[list(T)] = 1
            if with_i: m[i] = 1
            rows.append(m)
        meta.append(dict(conf=c, i=i, T=T, dhat=d))
    np.save(ROOT / f"outputs/sets/{a.ds}_mci.npy", np.stack(rows)); pickle.dump(meta, open(ROOT / f"outputs/sets/{a.ds}_mci_meta.pkl", "wb"))
    print("merged", len(meta), "候选背景 →", len(rows), "个集合"); sys.exit(0)

dev = torch.device("cuda"); D = load(a.ds); nG, nC = len(D["general"]), len(D["conf"]); act = D["active"]
Xtr = torch.as_tensor(D["Xtr"], device=dev); Ytr = torch.as_tensor(D["Ytr"], device=dev).double()
Xte = torch.as_tensor(D["Xte"], device=dev); Yte = torch.as_tensor(D["Yte"], device=dev).double()
fit_idx, val_idx = fit_val_idx(len(Xtr), dev)
ck = REPO / ("DNN_Aggresvation75/outputs/oracle_uniform_seed0.pt" if a.ds == "pjm" else "DNN_Aggresvation95_caiso/outputs/oracle_uniform_seed0.pt")
phi = FrozenPhi(load_oracle(ck, nG, nC, dev), "last")
cache = {}


def query(sets):
    """sets: list of frozenset → (len, C) numpy，带缓存。"""
    todo = [s for s in dict.fromkeys(sets) if s not in cache]
    for b in range(0, len(todo), 64):
        chunk = todo[b:b + 64]; m = torch.zeros(len(chunk), nG, device=dev)
        for r, s in enumerate(chunk): m[r, list(s)] = 1
        with torch.no_grad():
            v = ridge_r2(phi(Xtr, m).double(), Ytr, phi(Xte, m).double(), Yte, fit_idx, val_idx).clamp(0, 1).cpu().numpy()
        for r, s in enumerate(chunk): cache[s] = v[r]
    return np.stack([cache[s] if s else np.zeros(nC) for s in sets])


confs = [int(x) for x in a.confs.split(",")] if a.confs else list(range(nC))
rng = np.random.RandomState(100 + confs[0]); t0 = time.time(); cands = []
# 随机排列前驱（各目标共享查询）
rand_bg = {i: [frozenset(rng.choice([x for x in act if x != i], rng.randint(4, len(act)), replace=False)) for _ in range(a.nrand)] for i in act}
for i in act:
    sets = [T for T in rand_bg[i]] + [T | {i} for T in rand_bg[i]]
    v = query(sets); rand_bg[i] = (rand_bg[i], v[a.nrand:] - v[:a.nrand])
print(f"[mci {a.ds} {a.tag}] 随机前驱完成 {time.time()-t0:.0f}s，缓存 {len(cache)}", flush=True)
for c in confs:
    paths = {i: [frozenset()] for i in act}; best_path = {i: [] for i in act}
    for t in range(a.steps):
        sets = []; plan = []
        for i in act:
            T = paths[i][-1]
            for j in act:
                if j != i and j not in T:
                    sets += [T | {j}, T | {j, i}]; plan.append((i, j))
        v = query(sets)[:, c]; d = v[1::2] - v[0::2]
        best = {}
        for (i, j), dd in zip(plan, d):
            if i not in best or dd > best[i][1]: best[i] = (j, dd)
        for i in act:
            j, dd = best[i]; Tn = paths[i][-1] | {j}; paths[i].append(Tn); best_path[i].append((Tn, dd))
    for i in act:
        g = sorted([x for x in best_path[i] if len(x[0]) >= 4], key=lambda x: -x[1])[:2]
        bgs, dr = rand_bg[i]; top = np.argsort(-dr[:, c])[:2]
        for T, dd in g: cands.append((c, i, tuple(sorted(T)), float(dd)))
        for k in top: cands.append((c, i, tuple(sorted(bgs[k])), float(dr[k, c])))
    print(f"[mci {a.ds} {a.tag}] 目标 {c} 完成 {time.time()-t0:.0f}s，缓存 {len(cache)}", flush=True)
pickle.dump(dict(cands=cands, nG=nG), open(ROOT / f"outputs/analysis/{a.ds}_mci_cand_{a.tag}.pkl", "wb"))
print("done", len(cands), f"{time.time()-t0:.0f}s")
