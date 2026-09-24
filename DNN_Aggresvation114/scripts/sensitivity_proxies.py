# -*- coding: utf-8 -*-
"""114 号 步骤 4（敏感性）：灰色地带的"近似代理字段"剔除与否，结论是否变化。不重训——所需集合都在各组真值中。
  PJM 发电总出力：再剔除两个系统负荷预测（发电≈负荷，数值接近但不是同一物理量）
  CAISO 实际负荷：再剔除全部分区负荷预测（PGE/SCE/SDGE）与日前分区负荷计划（北/东/南）——它们是总负荷的组成部分"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
CASES = [("pjm_gen_ic", "Y_total_gen", "PJM 发电总出力", lambda f: f.startswith("forecast_load")),
         ("caiso_load", "Y_actual_load__mw__ca_iso_tac", "CAISO 实际负荷",
          lambda f: f.startswith("dam_load_forecast") or f.startswith("dam_schedule__mw__load"))]
rows = []
for g, t, name, drop in CASES:
    O = os.path.join(ROOT, "groups", g, "outputs"); spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8"))
    z = np.load(os.path.join(O, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
    V = np.load(os.path.join(O, "V_official.npy")); c = spec["targ"].index(t); idx = {k: r for r, k in enumerate(keys)}
    for lab, pool in [("现实口径（114 主结果）", list(range(len(spec["cand"])))),
                      ("再剔除近似代理", [i for i, f in enumerate(spec["cand"]) if not drop(f)])]:
        M = pipe.m_table(V, keys, pool, 2)[0]
        r = dict(目标=name, 口径=lab, 候选数=len(pool), 剔除=", ".join(f for f in spec["cand"] if drop(f)) if "剔除" in lab else "—",
                 最强单字段=float(max(V[idx[(i,)], c] for i in pool)), M0均值=float(M[:, 0, c].mean()), M2均值=float(M[:, 2, c].mean()))
        for tau in (0.5, 0.7):
            mus = pipe.mus_list(V, keys, pool, c, tau, 3)
            c0 = {i for i in pool if V[idx[(i,)], c] > tau}; c2 = {i for m in mus for i in m}
            r[f"τ{tau}_单看安全组合危险"] = f"{len(c2 - c0)}/{len(pool)}"; r[f"τ{tau}_危险小组合"] = len(mus)
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "outputs", "114_sensitivity_proxies.csv"), index=False)
print(R.round(3).to_string(index=False))
