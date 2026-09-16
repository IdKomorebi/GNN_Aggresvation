# -*- coding: utf-8 -*-
"""F：交互指标能否识别已知结构（gameA）——风电对协同(+)、负荷预测近重复(−)、ss_mw 精确重复(−)。
对真值与各估计器，列出目标 total_gen / metered_load_mw / total_lmp_da 的二阶 Faith-Shap 与 SII 最大正/负项，
并报告三个已知对的数值与排名；另报告全部 91 对上"强交互"(|真值|>0.02) 的符号一致率。"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import games as G
from analyze import load_truth, load_est, meta, CONF, OUT
P = meta["gameA"]; pairs = list(combinations(range(len(P)), 2))
short = [f.replace("gen_fuel_", "").replace("forecast_load_mw_", "fl_").replace("da_as_", "").replace("_reserve", "") for f in P]
known = {"wind(mw,pct)": (0, 1), "fl(latest,DA)": (2, 3), "ss_mw(pri,sync)": (4, 5)}
Tr = load_truth("gameA"); T = (Tr[(0, "clean")] + Tr[(1, "clean")]) / 2
srcs = {"truth": T}
for m in ["direct", "L0x", "L1x", "L1ens", "L0ensx", "lin", "poly2"]:
    E = load_est("gameA", m)
    if E is not None: srcs[m] = E
rows, sign = [], []
for name, V in srcs.items():
    f1, f2 = G.faith2(V); s2 = G.sii(V)
    F2 = np.stack([f2[pr] for pr in pairs]); S2 = np.stack([s2[pr] for pr in pairs])
    if name == "truth": F2T, S2T = F2, S2
    strong = np.abs(F2T) > 0.02
    sign.append(dict(src=name, n_strong=int(strong.sum()), faith2_sign_agree=float((np.sign(F2[strong]) == np.sign(F2T[strong])).mean()),
                     sii_sign_agree=float((np.sign(S2[np.abs(S2T) > 0.02]) == np.sign(S2T[np.abs(S2T) > 0.02])).mean())))
    for c in ["total_gen", "metered_load_mw", "total_lmp_da"]:
        ci = CONF.index(c); order = np.argsort(-F2[:, ci])
        r = dict(src=name, conf=c)
        for kn, pr in known.items():
            k = pairs.index(pr); r[kn] = f"{F2[k, ci]:+.3f}(#{int(np.where(order == k)[0][0]) + 1})"
        r["top+"] = ", ".join(f"{short[pairs[k][0]]}×{short[pairs[k][1]]}{F2[k, ci]:+.2f}" for k in order[:2])
        r["top−"] = ", ".join(f"{short[pairs[k][0]]}×{short[pairs[k][1]]}{F2[k, ci]:+.2f}" for k in order[-2:])
        rows.append(r)
df = pd.DataFrame(rows); sg = pd.DataFrame(sign)
df.to_csv(OUT / "F_interaction_cases.csv", index=False); sg.to_csv(OUT / "F_interaction_sign.csv", index=False)
print(df.to_markdown(index=False)); print(); print(sg.to_markdown(index=False, floatfmt=".3f"))
