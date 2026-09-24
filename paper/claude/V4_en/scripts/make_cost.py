# -*- coding: utf-8 -*-
"""成本段落（cost_text.tex）：从 118 号在空闲 GPU 上的计时结果生成。只与逐个串行重训比较（不比较并行重训）。"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
E = os.path.join(REPO, "DNN_Aggresvation118", "outputs", "est")
rows = []
for tag, lab in [("RTS-GMLC", "RTS-GMLC"), ("NEM", "NEM"), ("PJM-load", "PJM load"), ("PJM-gen_ic", "PJM gen./interch."), ("CAISO-load", "CAISO")]:
    s, t = os.path.join(E, tag, "seq.npz"), os.path.join(E, tag, "timeE.npz")
    if not (os.path.exists(s) and os.path.exists(t)):
        continue
    z = np.load(s)["t"]; e = np.load(t)
    rows.append(dict(lab=lab, seq=float(z[:, 0].mean() + z[:, 1].mean()), E=float(e["E"]), n3=int(e["n3"]), p=int(e["p"])))
D = pd.DataFrame(rows); D["speed"] = D.seq / D.E; D["h_seq"] = D.seq * D.n3 / 3600; D["min_E"] = D.E * D.n3 / 60
D.to_csv(os.path.join(HERE, "..", "tables", "cost.csv"), index=False)
c = D.set_index("lab")
txt = (f"Measured on one otherwise idle GPU, retraining the attacker family sequentially for one field set takes "
       f"{D.seq.min():.0f}--{D.seq.max():.0f}\\,s (the perceptrons dominate; one network per target), whereas the amortized "
       f"estimate takes {1000 * D.E.min():.0f}--{1000 * D.E.max():.0f}\\,ms, a factor of {D.speed.min():.0f}--{D.speed.max():.0f}. "
       f"A complete $K=2$ table (all sets of size $\\le3$) therefore takes {D.min_E.min():.0f}--{D.min_E.max():.0f}\\,min to scan, "
       f"against {D.h_seq.min():.0f}--{D.h_seq.max():.0f}\\,h of sequential retraining; certification adds three retrained "
       f"backgrounds per field and target, i.e.\\ a few hundred sets. Pre-training a backbone takes about half a minute and is "
       f"done once per data set. The savings compound over the recurrent workloads of an audit: re-grading after every release, "
       f"redefined targets, and the sets of size four needed for $K=3$ disposal.\n")
open(os.path.join(HERE, "..", "cost_text.tex"), "w").write(txt); print(D.round(3).to_string()); print(txt)
