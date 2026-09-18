# -*- coding: utf-8 -*-
"""103 号分析（P0-3 + RQ-A1/A2/A3/A5）：
真值口径 = FINAL_PROTOCOL §7 正式攻击器族（单目标 DNN ∪ 多目标 DNN ∪ 梯度提升树，val 选择）+ 单调闭包。
A1 V 保真（按 |S| 分层）；A2 基线与消融；A3 掩码分布消融；A4 M 保真；A5 效率。
"""
import sys, glob, json, pickle
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent; R100 = REPO / "DNN_Aggresvation100"; R102 = REPO / "DNN_Aggresvation102"
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure, index_of, attack_max
from analyze_est import marg_tables
A = ROOT / "outputs/analysis"; A.mkdir(parents=True, exist_ok=True); GR = [0.05, 0.2, 0.5]
rep = ["# 103 号分析：正式真值口径下的通用模型基线与消融\n"]
md = lambda df: df.to_markdown(index=False, floatfmt=".4f")


def shards(pat, n, key, ncol=12):
    fs = sorted(glob.glob(pat))
    if not fs: return None
    need = int(fs[0].split("of")[-1].split(".")[0])
    if len(fs) != need: return None
    out = np.zeros((n, ncol), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z[key]
    return out


def official_truth(ds, keys):
    """三类攻击器取 max（val 选择）→ 单调闭包。返回 V̄ 与逐攻击器份额。"""
    n = len(keys)
    meta4 = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb")); keys4 = meta4["keys"]
    c4 = np.zeros((len(keys4), 12), np.float32); v4 = np.zeros((len(keys4), 12), np.float32)
    for f in sorted(glob.glob(str(R100 / f"outputs/truth/{ds}_k4_seed0_s*of3.npz"))):
        z = np.load(f); c4[z["idx"]] = z["clean"]; v4[z["idx"]] = z["val_r2"]
    sel = np.array([len(k) <= 3 for k in keys4]); mt_c, mt_v = np.clip(c4[sel], -1, 1), v4[sel]
    g = np.load(R100 / f"outputs/gbr/{ds}_k3.npz"); gb_c, gb_v = np.clip(g["clean"], -1, 1), g["val_r2"]
    st_c = np.zeros((n, 12), np.float32); st_v = np.zeros((n, 12), np.float32)
    for t in range(12):
        z = np.load(ROOT / f"outputs/truth/{ds}_k3_single_t{t}_s0of1.npz")
        st_c[z["idx"], t] = np.clip(z["clean"][:, 0], -1, 1); st_v[z["idx"], t] = z["val_r2"][:, 0]
    att, att_v, pick = attack_max([st_c, mt_c, gb_c], [st_v, mt_v, gb_v])
    return closure(att, keys, Vval=att_v), dict(single=st_c, multi=mt_c, tree=gb_c), pick


rows_share, rows_a1, rows_a2, rows_a5 = [], [], [], []
ests_saved = {}
for ds in ["pjm", "caiso"]:
    meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb")); keys, act = meta["keys_k3"], meta["active"]; n = len(keys)
    Vt, per_att, pick = official_truth(ds, keys)
    sz = np.array([len(k) for k in keys])
    for s in [1, 2, 3]:
        k = sz == s
        rows_share.append(dict(数据集=ds, 规模=s, 单目标DNN被选中=float((pick[k] == 0).mean()), 多目标DNN被选中=float((pick[k] == 1).mean()),
                               树模型被选中=float((pick[k] == 2).mean()), 正式真值均值=float(Vt[k].mean()),
                               仅多目标DNN均值=float(np.clip(per_att["multi"][k], 0, 1).mean()),
                               正式减仅多目标=float(Vt[k].mean() - np.clip(per_att["multi"][k], 0, 1).mean())))
    Dt, bk = marg_tables(Vt, keys, act, kmax=2); bsz = np.array([len(T) for T in bk[0]])
    cand = {"raw ridge": f"{ds}_k3_raw", "raw+x² ridge": f"{ds}_k3_raw2", "随机特征 ridge": f"{ds}_k3_rff",
            "共享输出头（direct）": f"{ds}_k3_direct", "φ ridge": f"{ds}_k3_L0",
            "φ+x,x²（单种子）": None, "φ+x,x²（三种子集成）": f"{ds}_k3_L0ensx", "元训练 φ+x,x²": f"{ds}_k3_L1x",
            "掩码消融：无掩码": f"{ds}_k3_mask-none", "掩码消融：Bernoulli": f"{ds}_k3_mask-bern50",
            "single-target φ+x,x²（102 号）": None}
    for nm, stem in cand.items():
        if nm == "φ+x,x²（单种子）":
            E = shards(str(R102 / f"outputs/est/{ds}_k3_multix_s*of3.npz"), n, "v")
        elif nm.startswith("single-target"):
            E = np.zeros((n, 12), np.float32); ok = True
            for t in range(12):
                v = shards(str(R102 / f"outputs/est/{ds}_k3_single_t{t}_s*of3.npz"), n, "v", 1)
                if v is None: ok = False; break
                E[:, t] = v[:, 0]
            E = E if ok else None
        else:
            E = shards(str(ROOT / f"outputs/est/{stem}_s*of3.npz"), n, "v")
        if E is None: continue
        Eb = closure(np.clip(E, 0, 1), keys); De, _ = marg_tables(Eb, keys, act, kmax=2)
        r = dict(数据集=ds, 估计器=nm, V_MAE=np.abs(Eb - Vt).mean(), V偏差=(Eb - Vt).mean(),
                 Spearman=np.nanmean([spearmanr(Eb[:, c], Vt[:, c])[0] for c in range(12)]),
                 危险漏判=float(((Eb <= 0.7) & (Vt > 0.7)).sum() / max((Vt > 0.7).sum(), 1)),
                 边际MAE=np.abs(De - Dt).mean(), 边际q95=np.quantile(np.abs(De - Dt), 0.95))
        for K in [1, 2]:
            selK = bsz <= K; Me = De[:, selK].max(1); Mt = Dt[:, selK].max(1)
            L = np.take_along_axis(Dt[:, selK], De[:, selK].argmax(1)[:, None, :], 1)[:, 0]
            r[f"M_MAE_K{K}"] = np.abs(Me - Mt).mean(); r[f"Kendall_K{K}"] = np.nanmean([kendalltau(Me[:, c], Mt[:, c])[0] for c in range(12)])
            r[f"档位一致_K{K}"] = (np.digitize(Me, GR) == np.digitize(Mt, GR)).mean(); r[f"见证召回_K{K}"] = (L >= Mt - 0.02).mean()
        rows_a2.append(r)
        if nm in ("φ+x,x²（三种子集成）", "φ+x,x²（单种子）", "raw+x² ridge", "共享输出头（direct）"):
            for s in [1, 2, 3]:
                k = sz == s
                rows_a1.append(dict(数据集=ds, 估计器=nm, 规模=s, V_MAE=np.abs(Eb[k] - Vt[k]).mean(), 偏差=(Eb[k] - Vt[k]).mean()))
        ests_saved[(ds, nm)] = Eb
    np.savez(A / f"{ds}_truth_official.npz", V=Vt, pick=pick)
# A5 效率（本号与 100/102 号事件）
for src, tag in [(ROOT, "103"), (R102, "102"), (R100, "100")]:
    p = src / "outputs/events.jsonl"
    if not p.exists(): continue
    for e in [json.loads(l) for l in open(p, encoding="utf-8")]:
        if e["stage"] in ("EST", "EST1", "ESTM", "ESTBASE", "ESTMASK") and "ms_per_set" in e and "_k3" in e["note"]:
            rows_a5.append(dict(来源=tag, note=e["note"].split(" sh")[0], ms=e["ms_per_set"]))
E5 = pd.DataFrame(rows_a5).groupby(["来源", "note"]).ms.mean().reset_index() if rows_a5 else pd.DataFrame()
for t, df in [("A. 攻击器族构成（正式真值中各攻击器被 val 选中的占比）", pd.DataFrame(rows_share)),
              ("B. 估计器对照（k3 全部集合，正式真值）", pd.DataFrame(rows_a2)),
              ("C. 按集合规模分层的 V 误差", pd.DataFrame(rows_a1)),
              ("D. 每集合查询耗时（ms，k3）", E5)]:
    if len(df):
        df.to_csv(A / f"103_{t[0]}.csv", index=False); rep.append(f"## {t}\n\n" + md(df))
(A / "report103.md").write_text("\n\n".join(rep), encoding="utf-8"); print("\n\n".join(rep))
