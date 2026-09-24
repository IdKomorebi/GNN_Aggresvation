# -*- coding: utf-8 -*-
"""119 号 步骤 2：稳健性与 K=3 证据。
  (A) 真值种子 / 随机划分：同一套协议换攻击器随机种子（RTS-GMLC、PJM 实际负荷）或换一次随机 70/30 划分（RTS-GMLC），
      重算全部结论性数字，与主结果对照：危险组合字段数、单字段即危险数、危险小组合数、最少扣留数、M^(0/2/3) 均值、M^(2)/M^(3)；
      以及字段 M^(2) 排序的 Spearman、τ-关键字段集合的 Jaccard。
  (B) NEM 的 K=3：112 号（规模 ≤3）真值 + 本号规模 4 的真值合并后重做攻击器选择与单调闭包 → M^(3)、M^(2)/M^(3)。
  (C) τ 敏感性：五组主结果在 τ∈[0.3, 0.9] 上：单字段即危险的字段数、单看安全但组合危险的字段数、单字段定级后仍暴露的危险组合比例。
"""
import os, sys, json, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
AN = os.path.join(ROOT, "outputs", "analysis")


def truth_of(O, keys):
    parts = []
    for nm in ("single", "multi", "tree"):
        f = os.path.join(O, f"truth_{nm}.npz")
        if os.path.exists(f):
            t = np.load(f); parts.append((t["clean"], t["val"]))
    return pipe.official_truth(parts, keys)


def hit(mus, p):
    if not mus:
        return 0
    A = np.zeros((len(mus), p))
    for r, m in enumerate(mus):
        A[r, list(m)] = 1
    return int(round(milp(c=np.ones(p), constraints=LinearConstraint(A, lb=1), integrality=np.ones(p), bounds=Bounds(0, 1)).fun))


def summarize(V, keys, p, C, targ, kb):
    """返回 逐目标 的结论性数字 与 M^(2)、τ-关键集合（用于跨种子一致性）。"""
    M, _, _, _, _ = pipe.m_table(V, keys, list(range(p)), kb); idx = {k: r for r, k in enumerate(keys)}
    out, M2, crit = [], [], []
    for c in range(C):
        rec = dict(目标=targ[c].replace("Y_", ""))
        for K in range(kb + 1):
            rec[f"M{K}均值"] = float(M[:, K, c].mean())
        if kb >= 3:
            rec["M2/M3"] = float(M[:, 2, c].sum() / max(M[:, 3, c].sum(), 1e-9))
        rec["M1/M2"] = float(M[:, 1, c].sum() / max(M[:, 2, c].sum(), 1e-9))
        cr = {}
        for tau in (0.5, 0.7):
            mus = pipe.mus_list(V, keys, list(range(p)), c, tau, 3)
            c2 = np.array([any(i in m for m in mus) for i in range(p)]); c0 = np.array([V[idx[(i,)], c] > tau for i in range(p)])
            single = {m[0] for m in mus if len(m) == 1}
            rec.update({f"τ{tau}_单字段即危险": int(c0.sum()), f"τ{tau}_组合危险字段": int((c2 & ~c0).sum()),
                        f"τ{tau}_危险小组合": len(mus), f"τ{tau}_单字段定级后仍暴露": sum(1 for m in mus if not set(m) & single),
                        f"τ{tau}_最少扣留": hit(mus, p)})
            cr[tau] = set(np.where(c2)[0])
        out.append(rec); M2.append(M[:, 2, c]); crit.append(cr)
    return out, M2, crit


def jacc(a, b):
    return len(a & b) / max(len(a | b), 1)


def part_A():
    rows, cons = [], []
    for base_rel, variants in [("DNN_Aggresvation111/outputs", [("rts_seed1", "种子 1"), ("rts_seed2", "种子 2"),
                                                               ("rts_split43", "划分 43"), ("rts_split44", "划分 44")]),
                               ("DNN_Aggresvation116/groups/pjm_load/outputs", [("pjm_load_seed1", "种子 1"), ("pjm_load_seed2", "种子 2")])]:
        B = os.path.join(REPO, base_rel); spec = json.load(open(os.path.join(B, "fields.json"), encoding="utf-8"))
        z = np.load(os.path.join(B, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
        p, C, kb = len(spec["cand"]), len(spec["targ"]), spec["kmax"] - 1
        ds = "RTS-GMLC" if "111" in base_rel else "PJM 实际负荷"
        Vb = np.load(os.path.join(B, "V_official.npy")); sb, M2b, cb = summarize(Vb, keys, p, C, spec["targ"], kb)
        for r in sb:
            rows.append(dict(数据=ds, 版本="主结果（种子 0、划分 42）", **r))
        for g, lab in variants:
            O = os.path.join(ROOT, "groups", g, "outputs")
            if not all(os.path.exists(os.path.join(O, f"truth_{s}.npz")) for s in (["single", "tree"] + (["multi"] if C > 1 or "rts" in g else []))):
                print("未就绪", g); continue
            V, _ = truth_of(O, keys); s, M2, cr = summarize(V, keys, p, C, spec["targ"], kb)
            for c, r in enumerate(s):
                rows.append(dict(数据=ds, 版本=lab, **r))
                cons.append(dict(数据=ds, 版本=lab, 目标=r["目标"],
                                 M2排序Spearman=float(pd.Series(M2[c]).corr(pd.Series(M2b[c]), method="spearman")),
                                 M2最大绝对差=float(np.abs(M2[c] - M2b[c]).max()),
                                 关键集合Jaccard_τ05=jacc(cr[c][0.5], cb[c][0.5]), 关键集合Jaccard_τ07=jacc(cr[c][0.7], cb[c][0.7]),
                                 V绝对差均值=float(np.abs(V[:, c] - Vb[:, c]).mean())))
    pd.DataFrame(rows).to_csv(os.path.join(AN, "robust_numbers.csv"), index=False)
    pd.DataFrame(cons).to_csv(os.path.join(AN, "robust_consistency.csv"), index=False)
    print(pd.DataFrame(cons).round(3).to_string(index=False))


def part_B():
    B = os.path.join(REPO, "DNN_Aggresvation112/outputs"); O = os.path.join(ROOT, "groups", "nem_k4", "outputs")
    if not all(os.path.exists(os.path.join(O, f"truth_{s}.npz")) for s in ("single", "multi", "tree")):
        print("NEM K=3 未就绪"); return
    spec = json.load(open(os.path.join(B, "fields.json"), encoding="utf-8")); p, C = len(spec["cand"]), len(spec["targ"])
    k3 = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(B, "D.npz"))["keys"]]
    k4 = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(O, "D.npz"))["keys"]]
    parts = []
    for nm in ("single", "multi", "tree"):
        a, b = np.load(os.path.join(B, f"truth_{nm}.npz")), np.load(os.path.join(O, f"truth_{nm}.npz"))
        parts.append((np.concatenate([a["clean"], b["clean"]]), np.concatenate([a["val"], b["val"]])))
    keys = k3 + k4; V, share = pipe.official_truth(parts, keys); np.save(os.path.join(O, "V_official_k4.npy"), V)
    s, _, _ = summarize(V, keys, p, C, spec["targ"], 3)
    V3old = np.load(os.path.join(B, "V_official.npy")); s3, _, _ = summarize(V3old, k3, p, C, spec["targ"], 2)
    rows = [dict(口径="规模 ≤4（本号）", **r) for r in s] + [dict(口径="规模 ≤3（112 号）", **r) for r in s3]
    pd.DataFrame(rows).to_csv(os.path.join(AN, "nem_k3.csv"), index=False)
    pd.DataFrame(dict(攻击器=["单目标 DNN", "多目标 DNN", "梯度提升树"], 被val选中比例=share)).to_csv(os.path.join(AN, "nem_k3_attackers.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def part_C():
    rows = []
    for tag, rel, _, _ in registry.DATASETS:
        try:
            d = registry.load_ds(rel)
        except FileNotFoundError:
            print("未就绪", tag); continue
        spec, keys, V = d["spec"], d["keys"], d["V"]; p = len(spec["cand"])
        s3 = [k for k in keys if len(k) <= 3]; sel = np.array([len(k) <= 3 for k in keys]); V3 = V[sel]; idx = {k: r for r, k in enumerate(s3)}
        for c, y in enumerate(spec["targ"]):
            for tau in np.round(np.arange(0.30, 0.91, 0.05), 2):
                mus = pipe.mus_list(V3, s3, list(range(p)), c, tau, 3)
                c2 = np.array([any(i in m for m in mus) for i in range(p)]); c0 = np.array([V3[idx[(i,)], c] > tau for i in range(p)])
                single = {m[0] for m in mus if len(m) == 1}
                rows.append(dict(数据=tag, 目标=y.replace("Y_", ""), τ=tau, 字段数=p, 单字段即危险=int(c0.sum()),
                                 组合危险字段=int((c2 & ~c0).sum()), 危险小组合=len(mus),
                                 仍暴露比例=(sum(1 for m in mus if not set(m) & single) / len(mus)) if mus else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(AN, "tau_sweep.csv"), index=False); print("τ 扫描完成", len(rows))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ABC"
    for w, f in [("A", part_A), ("B", part_B), ("C", part_C)]:
        if w in which:
            f()
