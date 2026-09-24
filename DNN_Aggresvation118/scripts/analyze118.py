# -*- coding: utf-8 -*-
"""118 号 步骤 2：各估计变体对照精确真值（三攻击器专用重训，规模 ≤3）。
指标（逐目标，再对 11 个目标平均）：
  V 误差 / V 偏差、边际 Δ(T,i)（|T|≤2）误差、M^(2) 误差 / 偏差 / 排序 Spearman、
  扫描—认证：只重训估计器挑出的前 1 / 前 3 个背景得到的下界 ÷ 精确 M^(2)、
  τ=0.5 的关键字段（属于某个规模 ≤3 的最小不安全集）召回率与精确率。
热启动微调只在 150 个随机集合上做，单独与同一批集合上的单种子读出比较 V 误差。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
AN = os.path.join(ROOT, "outputs", "analysis")
LAB = {"lin": "线性读出（无主干）", "poly": "x, x² 读出（无主干）", "polyS": "集合内二阶多项式（无主干）",
       "head3": "共享输出头（三种子）", "rand1": "随机初始化主干 + 读出", "phionly1": "只用 φ（不拼 x, x²）",
       "nofix1": "不做数值修正", "masknone1": "预训练不掩码", "maskbern1": "预训练伯努利掩码",
       "D1": "本文·单种子", "C3": "三种子特征拼接", "E": "本文·三种子预测平均（新主口径）"}


def crit_fields(V, keys3, p, c, tau=0.5):
    mus = pipe.mus_list(V, keys3, list(range(p)), c, tau, 3)
    return np.array([any(i in m for m in mus) for i in range(p)])


def main():
    rows, ft = [], []
    for tag, rel, ef, ek in registry.DATASETS:
        try:
            d = registry.load_ds(rel, ef, ek)
        except FileNotFoundError:
            print("跳过（真值未就绪）", tag); continue
        E_dir = os.path.join(ROOT, "outputs", "est", tag.replace("/", "_"))
        if "Vhat3" not in d or not os.path.isdir(E_dir):
            print("跳过", tag); continue
        spec, keys = d["spec"], d["keys"]; s3 = d["sel3"]; keys3 = [k for k, t in zip(keys, s3) if t]
        V = d["V"][s3]; p = len(spec["cand"]); fields = list(range(p))
        _, _, Dt, _, bsz = pipe.m_table(V, keys3, fields, 2); s2 = bsz <= 2
        raw = {"E": d["Vhat3"]}
        if ef == "est_variants.npz":
            ev = np.load(os.path.join(d["O"], ef)); raw["D1"] = ev["D"]; raw["C3"] = ev["C"]
        else:
            raw["D1"] = np.load(os.path.join(d["O"], ef))["E1"]
        for v in LAB:
            f = os.path.join(E_dir, f"{v}.npz")
            if v not in raw and os.path.exists(f):
                raw[v] = np.load(f)["est"]
        for v, r in raw.items():
            Ve = pipe.closure_max(r, keys3); De = pipe.m_table(Ve, keys3, fields, 2)[2]
            for c, y in enumerate(spec["targ"]):
                Mt, Me, L1, L3 = pipe.certify(Dt[:, :, c], De[:, :, c], bsz, 2)
                tc, ec = crit_fields(V, keys3, p, c), crit_fields(Ve, keys3, p, c)
                rows.append(dict(数据=tag, 目标=y.replace("Y_", ""), 变体=v, 名称=LAB[v],
                                 V误差=float(np.abs(Ve[:, c] - V[:, c]).mean()), V偏差=float((Ve[:, c] - V[:, c]).mean()),
                                 边际误差=float(np.abs(De[:, s2, c] - Dt[:, s2, c]).mean()),
                                 M2误差=float(np.abs(Me - Mt).mean()), M2偏差=float((Me - Mt).mean()),
                                 M2排序Spearman=float(pd.Series(Me).corr(pd.Series(Mt), method="spearman")),
                                 认证前1=float(L1.sum() / max(Mt.sum(), 1e-9)), 认证前3=float(L3.sum() / max(Mt.sum(), 1e-9)),
                                 关键召回=float((tc & ec).sum() / max(tc.sum(), 1)), 关键精确=float((tc & ec).sum() / max(ec.sum(), 1)),
                                 耗时ms=float(np.load(os.path.join(E_dir, f"{v}.npz"))["sec"]) * 1000 if os.path.exists(os.path.join(E_dir, f"{v}.npz")) else np.nan))
        f = os.path.join(E_dir, "ft.npz")
        if os.path.exists(f):
            z = np.load(f); pick = z["pick"]
            for c, y in enumerate(spec["targ"]):
                ft.append(dict(数据=tag, 目标=y.replace("Y_", ""), 集合数=len(pick),
                               微调V误差=float(np.abs(z["est"][:, c] - V[pick, c]).mean()),
                               单种子读出V误差=float(np.abs(np.clip(raw["D1"][pick, c], 0, 1) - V[pick, c]).mean()),
                               新主口径V误差=float(np.abs(np.clip(raw["E"][pick, c], 0, 1) - V[pick, c]).mean()),
                               微调耗时s=float(z["sec"])))
        print("完成", tag, flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(AN, "variants_by_target.csv"), index=False)
    cols = ["V误差", "V偏差", "边际误差", "M2误差", "M2偏差", "M2排序Spearman", "认证前1", "认证前3", "关键召回", "关键精确", "耗时ms"]
    m = R.groupby("变体")[cols].mean().reindex([v for v in LAB if v in set(R.变体)]); m.insert(0, "名称", [LAB[v] for v in m.index])
    m["目标数"] = R.groupby("变体").size().reindex(m.index)
    m.to_csv(os.path.join(AN, "variants_mean.csv")); print(m.round(3).to_string())
    if ft:
        F = pd.DataFrame(ft); F.to_csv(os.path.join(AN, "finetune.csv"), index=False); print(F.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
