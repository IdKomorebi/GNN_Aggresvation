# -*- coding: utf-8 -*-
"""117 号 步骤 3：定义对比——各种字段级打分方法能否发现组合推断风险、按其扣留字段的效率如何。

参与比较的打分（全部为"分数越高越该保护"）：
  相关类   pearson / spearman / mi / dcor
  单字段   M0：单字段专用重训推断能力（= 现行单字段定级）
  模型归因 loco / perm / sage（全字段模型上）
  图传播   graph：仿 IGNN 的成对推断图 + 风险传播
  本文     M2_est：通用模型（新主口径 E）估计的 M^(2)，不用任何重训真值；
           M2_cert：扫描—认证，只对 E 挑出的每字段前 3 个背景重训认证得到的下界；
           M2：精确 M^(2)（参考上界，需全部规模 ≤3 的重训真值）
评测（真值 = 三攻击器专用重训 + 单调闭包，规模 ≤3 的集合；各数据集统一口径）：
  (1) 关键字段检测：标签 = 字段属于某个规模 ≤3 的 τ-最小不安全集合（K=2 关键）；
      另看"隐藏风险"字段 = 关键但单字段不越阈的字段，在单字段安全的字段里检测。指标：平均精确率 AP。
  (2) 扣留效率：按分数从高到低依次扣留字段，直到剩余字段里任何规模 ≤3 的组合都不越过 τ，
      记所需扣留数 k_τ；与最优扣留数（最小命中集，整数规划）比较。另画剩余风险曲线。
"""
import os, sys, json, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); import registry  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "..", "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
AN = os.path.join(ROOT, "outputs", "analysis")
METHODS = ["pearson", "spearman", "mi", "dcor", "M0", "loco", "perm", "sage", "graph", "M2_est", "M2_cert", "M2"]
TAUS = (0.5, 0.7)


def min_hitting(mus, p):
    if not mus:
        return 0
    A = np.zeros((len(mus), p))
    for r, m in enumerate(mus):
        A[r, list(m)] = 1
    return int(round(milp(c=np.ones(p), constraints=LinearConstraint(A, lb=1), integrality=np.ones(p), bounds=Bounds(0, 1)).fun))


def main():
    det, prot, curves, fields_tab, rules, adapt = [], [], [], [], [], []
    for tag, rel, ef, ek in registry.DATASETS:
        fn = os.path.join(ROOT, "outputs", f"scores_{tag.replace('/', '_')}.csv")
        try:
            d = registry.load_ds(rel, ef, ek)
        except FileNotFoundError:
            print("跳过（真值未就绪）", tag); continue
        if not os.path.exists(fn) or "Vhat3" not in d:
            print("跳过（未就绪）", tag); continue
        Sc = pd.read_csv(fn); spec, keys = d["spec"], d["keys"]; p = len(spec["cand"]); fields = list(range(p))
        s3 = np.array([len(k) <= 3 for k in keys]); keys3 = [k for k, t in zip(keys, s3) if t]
        V3 = d["V"][s3]; assert np.array_equal(d["sel3"], s3)
        Ve = pipe.closure_max(d["Vhat3"], keys3)
        M, _, Dt, BK, bsz = pipe.m_table(V3, keys3, fields, 2); Me, _, De, _, _ = pipe.m_table(Ve, keys3, fields, 2)
        idx3 = {k: r for r, k in enumerate(keys3)}
        for c, y in enumerate(spec["targ"]):
            sc = Sc[Sc.目标 == y].set_index("字段").reindex(spec["cand"])
            _, _, _, Lk = pipe.certify(Dt[:, :, c], De[:, :, c], bsz, 2, top=3)
            score = {m: sc[m].values.astype(float) for m in ["pearson", "spearman", "mi", "dcor", "loco", "perm", "sage", "graph"]}
            score.update(M0=M[:, 0, c], M2_est=Me[:, 2, c], M2_cert=Lk, M2=M[:, 2, c])
            for i in fields:
                fields_tab.append(dict(数据=tag, 目标=y, 字段=spec["cand"][i], **{m: score[m][i] for m in METHODS}))
            for tau in TAUS:
                mus = pipe.mus_list(V3, keys3, fields, c, tau, 3)
                crit = np.array([any(i in m for m in mus) for i in fields]); single = M[:, 0, c] > tau
                hidden = crit & ~single; kopt = min_hitting(mus, p)
                for m in METHODS:
                    s = score[m]
                    rec = dict(数据=tag, 目标=y, τ=tau, 方法=m, 字段数=p, 关键数=int(crit.sum()), 隐藏数=int(hidden.sum()),
                               AP_关键=average_precision_score(crit, s) if 0 < crit.sum() < p else np.nan,
                               AP_隐藏=average_precision_score(hidden[~single], s[~single]) if 0 < hidden.sum() < (~single).sum() else np.nan,
                               R精确率=float(crit[np.argsort(-s, kind="stable")[:crit.sum()]].mean()) if 0 < crit.sum() else np.nan)
                    det.append(rec)
                    order = np.argsort(-s, kind="stable"); held = set(); k_tau = p
                    res_curve = []
                    for k in range(p + 1):
                        if k:
                            held.add(order[k - 1])
                        left = [r for kk, r in idx3.items() if not set(kk) & held]
                        rv = float(V3[left, c].max()) if left else 0.0
                        nm = sum(1 for mm in mus if not set(mm) & held)
                        res_curve.append((k, rv, nm))
                        if nm == 0 and k_tau == p:
                            k_tau = k
                    prot.append(dict(数据=tag, 目标=y, τ=tau, 方法=m, 字段数=p, 最优扣留数=kopt, 扣留数=k_tau,
                                     超出最优=k_tau - kopt, 危险组合数=len(mus)))
                    if tau == 0.5:
                        curves += [dict(数据=tag, 目标=y, 方法=m, k=k, 剩余最大V=rv, 剩余危险组合=nm) for k, rv, nm in res_curve]
                # ---- 决策规则（选择只用估计值，真值只用于评价）：
                #   单字段规则：V({i})>τ（现行做法，需重训单字段）；
                #   扫描标记：在估计值上、阈值 τ−δ 处的关键字段（δ 为保守余量，抵消估计器的共模低估）；
                #   扫描—认证：对扫描标记的字段，取估计值上越过 τ−δ 的背景中 V̂(T∪i) 最大的前 3 个，重训确认 V(T)≤τ<V(T∪i)。
                for dlt in (0.0, 0.05):
                    thr = tau - dlt
                    mus_e = pipe.mus_list(Ve, keys3, fields, c, thr, 3)
                    crit_e = np.array([any(i in mm for mm in mus_e) for i in fields]); cert = np.zeros(p, bool)
                    for i in np.where(crit_e)[0]:
                        cands = []
                        for T in BK[i]:
                            if len(T) > 2:
                                continue
                            vT = Ve[idx3[T], c] if T else 0.0; vTi = Ve[idx3[tuple(sorted(T + (i,)))], c]
                            if vT <= thr < vTi:
                                cands.append((vTi, T))
                        for _, T in sorted(cands, reverse=True)[:3]:
                            vT = V3[idx3[T], c] if T else 0.0
                            if vT <= tau < V3[idx3[tuple(sorted(T + (i,)))], c]:
                                cert[i] = True
                    flags = [("扫描标记（估计）", crit_e), ("扫描—认证标记", cert)] + ([("单字段规则 V({i})>τ", single)] if dlt == 0 else [])
                    for rule, flag in flags:
                        rules.append(dict(数据=tag, 目标=y, τ=tau, δ=dlt if rule != "单字段规则 V({i})>τ" else np.nan, 规则=rule,
                                          关键数=int(crit.sum()), 标记数=int(flag.sum()),
                                          召回=float((flag & crit).sum() / max(crit.sum(), 1)) if crit.sum() else np.nan,
                                          精确=float((flag & crit).sum() / flag.sum()) if flag.sum() else np.nan))
                    # ---- 扣留策略（不用真值选择）：估计 MUS 的最小命中集；自适应 M̂^(2) 贪心直到估计上无关键字段
                    if mus_e:
                        A = np.zeros((len(mus_e), p))
                        for r_, mm in enumerate(mus_e):
                            A[r_, list(mm)] = 1
                        W = set(np.where(milp(c=np.ones(p), constraints=LinearConstraint(A, lb=1), integrality=np.ones(p), bounds=Bounds(0, 1)).x > 0.5)[0])
                    else:
                        W = set()
                    held = []
                    while len(held) < p:
                        left = [f for f in fields if f not in held]
                        if not pipe.mus_list(Ve, keys3, left, c, thr, 3):
                            break
                        Mx = pipe.m_table(Ve, keys3, left, 2)[0][:, 2, c]; held.append(left[int(np.argmax(Mx))])
                    for lab, Wset in [("估计MUS最小命中集", W), ("自适应M̂2贪心", set(held))]:
                        adapt.append(dict(数据=tag, 目标=y, τ=tau, δ=dlt, 策略=lab, 扣留数=len(Wset), 最优扣留数=kopt,
                                          残余真危险组合=sum(1 for mm in mus if not set(mm) & Wset), 危险组合数=len(mus)))
                adapt.append(dict(数据=tag, 目标=y, τ=tau, δ=np.nan, 策略="单字段定级（扣留单字段越阈者）", 扣留数=int(single.sum()),
                                  最优扣留数=kopt, 残余真危险组合=sum(1 for mm in mus if not set(mm) & set(np.where(single)[0])), 危险组合数=len(mus)))
        print("完成", tag, flush=True)
    det, prot, curves = pd.DataFrame(det), pd.DataFrame(prot), pd.DataFrame(curves)
    det.to_csv(os.path.join(AN, "detection.csv"), index=False); prot.to_csv(os.path.join(AN, "protection.csv"), index=False)
    curves.to_csv(os.path.join(AN, "protection_curves.csv"), index=False); pd.DataFrame(fields_tab).to_csv(os.path.join(AN, "field_scores.csv"), index=False)
    pd.DataFrame(rules).to_csv(os.path.join(AN, "decision_rules.csv"), index=False); pd.DataFrame(adapt).to_csv(os.path.join(AN, "withholding_strategies.csv"), index=False)
    print(pd.DataFrame(rules).fillna({"δ": -1}).groupby(["τ", "δ", "规则"])[["召回", "精确", "标记数", "关键数"]].mean().round(3).to_string())
    print(pd.DataFrame(adapt).fillna({"δ": -1}).groupby(["τ", "δ", "策略"])[["扣留数", "最优扣留数", "残余真危险组合", "危险组合数"]].mean().round(2).to_string())
    agg = det.groupby(["τ", "方法"])[["AP_关键", "AP_隐藏", "R精确率"]].mean().unstack(0)
    agg2 = prot.groupby(["τ", "方法"]).agg(平均扣留数=("扣留数", "mean"), 平均超出最优=("超出最优", "mean"),
                                         达到最优的目标比例=("超出最优", lambda x: float((x == 0).mean()))).unstack(0)
    agg.reindex(METHODS).to_csv(os.path.join(AN, "detection_mean.csv")); agg2.reindex(METHODS).to_csv(os.path.join(AN, "protection_mean.csv"))
    print(agg.reindex(METHODS).round(3).to_string()); print(agg2.reindex(METHODS).round(2).to_string())


if __name__ == "__main__":
    main()
