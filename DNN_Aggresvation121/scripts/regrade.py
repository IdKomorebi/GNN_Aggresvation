# -*- coding: utf-8 -*-
"""121 号（只汇报、不进论文）步骤 2：公开基底变化后的再定级。
对每个"已公开字段" b（B={b}，且 V({b}) ≤ τ，否则基底本身已泄露、跳过），在其余字段上重算
  M^(2)|B(i) = max_{|T|≤2} V(b∪T∪i) − V(b∪T)，以及 τ-关键性（存在 |T|≤2 使 V(b∪T) ≤ τ < V(b∪T∪i)）。
真值：各号三攻击器 + 单调闭包，规模 ≤4 的集合全部可得，因此再定级可以精确验证。
估计：新主口径 E（规模 ≤3 取自各号，规模 4 取自本号 est_k4.py），闭包取子集最大值。
统计：与 B=∅ 相比关键性翻转（新增 / 解除）的字段数；估计器对 B 下关键字段与"新增关键"的召回/精确；
      理论检查 M^(2)|{b}(i) ≤ M^(3)|∅(i)（相对基底的风险不超过多一个背景的无基底风险）。"""
import os, sys, json, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
CFG = {"pjm_load": (os.path.join(REPO, "DNN_Aggresvation116/groups/pjm_load/outputs"), "est_group.npz"),
       "rts": (os.path.join(REPO, "DNN_Aggresvation111/outputs"), "est_variants.npz")}


def tables(V, keys, p, c, tau, b):
    """B={b} 下每个字段的 M^(2)、关键性；b=None 表示 B=∅。"""
    idx = {k: r for r, k in enumerate(keys)}; base = () if b is None else (b,)
    v = lambda S: V[idx[tuple(sorted(S))], c] if S else 0.0
    H = [f for f in range(p) if f != b]; M = {}; crit = {}
    for i in H:
        oth = [j for j in H if j != i]; best, cr = -1, False
        for k in range(3):
            for T in itertools.combinations(oth, k):
                vt, vti = v(base + T), v(base + T + (i,))
                best = max(best, vti - vt); cr = cr or (vt <= tau < vti)
        M[i], crit[i] = best, cr
    return M, crit


def main():
    rows, summ = [], []
    for g, (O, ef) in CFG.items():
        f4 = os.path.join(ROOT, "outputs", f"est4_{g}.npz")
        if not os.path.exists(f4):
            print("缺估计", g); continue
        spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8")); z = np.load(os.path.join(O, "D.npz"))
        keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; V = np.load(os.path.join(O, "V_official.npy"))
        e3 = np.load(os.path.join(O, ef)); e4 = np.load(f4); p = len(spec["cand"])
        raw = np.full(V.shape, np.nan, np.float32); raw[e3["sel"]] = e3["E"]; raw[e4["sel4"]] = e4["E"]
        keep = ~np.isnan(raw[:, 0]); k4 = [k for k, t in zip(keys, keep) if t]; Ve = pipe.closure_max(raw[keep], k4); Vt = V[keep]
        M3 = pipe.m_table(Vt, k4, list(range(p)), 3)[0]
        targets = range(len(spec["targ"])) if g == "pjm_load" else [spec["targ"].index("Y_线路潮流_C35")]
        for c in targets:
            y = spec["targ"][c]
            for tau in (0.5, 0.7):
                M0t, C0t = tables(Vt, k4, p, c, tau, None)
                for b in range(p):
                    if Vt[k4.index((b,)), c] > tau:
                        continue
                    Mt, Ct = tables(Vt, k4, p, c, tau, b); Me, Ce = tables(Ve, k4, p, c, tau, b); _, Cd = tables(Ve, k4, p, c, tau - 0.05, b)
                    H = list(Mt)
                    new = [i for i in H if Ct[i] and not C0t[i]]; gone = [i for i in H if C0t[i] and not Ct[i]]
                    new_e = [i for i in H if Ce[i] and not C0t[i]]; new_d = [i for i in H if Cd[i] and not C0t[i]]
                    rows.append(dict(组=g, 目标=y, τ=tau, 已公开字段=spec["cand"][b], 字段数=len(H),
                                     无基底关键=sum(C0t[i] for i in H), 基底下关键=sum(Ct[i] for i in H), 新增关键=len(new), 解除关键=len(gone),
                                     M2平均变化=float(np.mean([Mt[i] - M0t[i] for i in H])),
                                     M2上升字段=sum(Mt[i] > M0t[i] + 1e-9 for i in H), M2下降字段=sum(Mt[i] < M0t[i] - 1e-9 for i in H),
                                     违反M3上界=sum(Mt[i] > M3[i, 3, c] + 1e-6 for i in H),
                                     估计关键召回=(sum(Ct[i] and Ce[i] for i in H) / max(sum(Ct[i] for i in H), 1)),
                                     估计关键精确=(sum(Ct[i] and Ce[i] for i in H) / max(sum(Ce[i] for i in H), 1)),
                                     新增关键召回=(len(set(new) & set(new_e)) / len(new)) if new else np.nan,
                                     余量005关键召回=(sum(Ct[i] and Cd[i] for i in H) / max(sum(Ct[i] for i in H), 1)),
                                     余量005关键精确=(sum(Ct[i] and Cd[i] for i in H) / max(sum(Cd[i] for i in H), 1)),
                                     余量005新增召回=(len(set(new) & set(new_d)) / len(new)) if new else np.nan,
                                     M2估计误差=float(np.mean([abs(Me[i] - Mt[i]) for i in H]))))
            print(g, y, "完成", flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "outputs", "analysis", "regrade.csv"), index=False)
    S = R.groupby(["组", "目标", "τ"]).agg(基底选择数=("已公开字段", "size"), 平均新增关键=("新增关键", "mean"), 最大新增关键=("新增关键", "max"),
                                          平均解除关键=("解除关键", "mean"), M2平均变化=("M2平均变化", "mean"), 违反M3上界=("违反M3上界", "sum"),
                                          估计关键召回=("估计关键召回", "mean"), 估计关键精确=("估计关键精确", "mean"),
                                          新增关键召回=("新增关键召回", "mean"), 余量005关键召回=("余量005关键召回", "mean"), 余量005关键精确=("余量005关键精确", "mean"),
                                          余量005新增召回=("余量005新增召回", "mean"), M2估计误差=("M2估计误差", "mean")).reset_index()
    S.to_csv(os.path.join(ROOT, "outputs", "analysis", "regrade_summary.csv"), index=False); print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
