# -*- coding: utf-8 -*-
"""116 号 步骤 4：三种主干（本组 / 全列 / 留目标）的对比，读出方式完全相同（新主口径 E）。
  (1) 表征：φ 的死神经元比例（训练行上标准差 <1e-6 的单元）、有效维度（特征协方差特征值的参与比 (Σλ)²/Σλ²），
      在 300 个随机集合上平均（三个种子各算一遍再平均）；
  (2) 误差分解：V 的带符号偏差、V 绝对误差、系统偏差占比 = |平均带符号误差| / 平均绝对误差；
      边际 Δ(T,i)（|T|≤2）的绝对误差——共同偏差在差分中抵消；
  (3) 字段风险：M^(2) 误差、排序 Spearman、认证前 1 / 前 3 个背景的下界比。
用法：backbone_compare.py [GPU]"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src")); import bb116  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
dev = torch.device(f"cuda:{sys.argv[1] if len(sys.argv) > 1 else 2}")
BB = os.path.join(ROOT, "outputs", "backbones"); DS_OF = {"pjm_load": "pjm", "pjm_gen_ic": "pjm", "caiso_load": "caiso"}
NAME = {"group": "本组主干", "full": "全列主干", "lto": "留目标主干"}
rows = []
for g, ds in DS_OF.items():
    O = os.path.join(ROOT, "groups", g, "outputs")
    if not os.path.exists(os.path.join(O, "V_official.npy")):
        print("真值未就绪，跳过", g); continue
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8")); z = np.load(os.path.join(O, "D.npz"))
    keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; sel = np.array([len(k) <= 3 for k in keys])
    keys3 = [k for k, s in zip(keys, sel) if s]; p = len(spec["cand"]); C = len(spec["targ"])
    V = np.load(os.path.join(O, "V_official.npy"))[sel]
    _, _, Dt, _, bsz = pipe.m_table(V, keys3, list(range(p)), 2); s2 = bsz <= 2
    rs = np.random.RandomState(0); pick = rs.choice(len(keys3), 300, replace=False)
    for bb in ("group", "full", "lto"):
        f = os.path.join(O, f"est_{bb}.npz")
        if not os.path.exists(f):
            print("缺", g, bb); continue
        # ---- (1) 表征
        if bb == "group":
            models = bb116.load_models([os.path.join(O, f"oracle_seed{s}.pt") for s in range(3)], p, C, dev)
            X = torch.as_tensor(z["Xtr"], device=dev); width = p; cmap = list(range(p))
        else:
            tag = f"{ds}_full" if bb == "full" else f"{g}_lto"
            drop = [t.replace("Y_", "") for t in spec["targ"]] if bb == "lto" else ()
            Xtr, _, names, _, _ = bb116.full_matrix(ds, drop)
            models = bb116.load_models([os.path.join(BB, f"{tag}_seed{s}.pt") for s in range(3)], len(names), len(names), dev)
            X = torch.as_tensor(Xtr, device=dev); width = len(names); cmap = [names.index(c) for c in spec["cand"]]
        dead, pr = [], []
        for m in models:
            ph = bb116.fr.FrozenPhi(m, "last")
            for r in pick:
                mk = torch.zeros(1, width, device=dev); mk[0, [cmap[i] for i in keys3[r]]] = 1
                F = ph(X, mk)[0].double(); dead.append(float((F.std(0) < 1e-6).float().mean()))
                ev = torch.linalg.eigvalsh(torch.cov(F.T)).clamp_min(0); pr.append(float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)))
        # ---- (2)(3) 误差
        Ve = pipe.closure_max(np.load(f)["E"], keys3); De = pipe.m_table(Ve, keys3, list(range(p)), 2)[2]
        for c, y in enumerate(spec["targ"]):
            e = Ve[:, c] - V[:, c]; Mt, Me, L1, L3 = pipe.certify(Dt[:, :, c], De[:, :, c], bsz, 2)
            rows.append(dict(组=g, 目标=y.replace("Y_", ""), 主干=NAME[bb], 死神经元比例=np.mean(dead), 有效维度=np.mean(pr),
                             V带符号偏差=float(e.mean()), V绝对误差=float(np.abs(e).mean()),
                             系统偏差占比=float(abs(e.mean()) / max(np.abs(e).mean(), 1e-12)),
                             边际Δ绝对误差=float(np.abs(De[:, s2, c] - Dt[:, s2, c]).mean()),
                             M2误差=float(np.abs(Me - Mt).mean()), M2偏差=float((Me - Mt).mean()),
                             M2排序Spearman=float(pd.Series(Me).corr(pd.Series(Mt), method="spearman")),
                             认证前1=float(L1.sum() / max(Mt.sum(), 1e-9)), 认证前3=float(L3.sum() / max(Mt.sum(), 1e-9)),
                             估计耗时ms=float(np.load(f)["sec_per_set"]) * 1000))
        print(g, bb, "完成", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "outputs", "analysis", "backbone_compare.csv"), index=False)
print(R.round(4).to_string(index=False))
m = R.groupby("主干")[[c for c in R.columns if c not in ("组", "目标", "主干")]].mean().reindex(list(NAME.values()))
m.to_csv(os.path.join(ROOT, "outputs", "analysis", "backbone_compare_mean.csv")); print(m.round(4).T.to_string())
