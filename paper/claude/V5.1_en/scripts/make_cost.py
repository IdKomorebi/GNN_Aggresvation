# -*- coding: utf-8 -*-
"""成本段落（cost_text.tex）：串行重训耗时取 118 号 seq.npz（空闲 GPU）；估计器与对照方法耗时取 126 号统一计时
（同一张空闲 GPU、每组同一批 200 个集合、批大小 8、等价快速读出实现 time_fast_<组>.csv）；预训练耗时取 126 号 pretrain_time.csv。
只与逐个串行重训比较（不比较并行重训）。"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
E = os.path.join(REPO, "DNN_Aggresvation118", "outputs", "est"); B = os.path.join(REPO, "DNN_Aggresvation126", "outputs")
rows = []
for tag in ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen_ic", "CAISO-load"]:
    z = np.load(os.path.join(E, tag, "seq.npz"))["t"]; t = pd.read_csv(os.path.join(B, f"time_fast_{tag}.csv")).set_index("方法").每集合ms / 1000
    n3 = int(np.load(os.path.join(E, tag, "timeE.npz"))["n3"])
    rows.append(dict(tag=tag, seq=float(z[:, 0].mean() + z[:, 1].mean()), ours=t["C3"], one=t["C2"], head=t["head3"], ft=t["ft"], n3=n3))
D = pd.DataFrame(rows); D["speed"] = D.seq / D.ours; D["h_seq"] = D.seq * D.n3 / 3600; D["min_ours"] = D.ours * D.n3 / 60
P = pd.read_csv(os.path.join(B, "pretrain_time.csv")); pr = P[P.主干 == "recon"].秒
D.to_csv(os.path.join(HERE, "..", "tables", "cost.csv"), index=False)
ms = lambda s: f"{1000 * s.min():.0f}--{1000 * s.max():.0f}"
txt = (f"All times below were measured on one otherwise idle GPU, on the same 200 random field sets per data set and with the "
       f"same batch size. Retraining the attacker family sequentially for one field set takes {D.seq.min():.0f}--{D.seq.max():.0f}\\,s "
       f"(the perceptrons dominate; one network per target). Our estimator takes {ms(D.ours)}\\,ms per set, a factor of "
       f"{D.speed.min():.0f}--{D.speed.max():.0f}; a complete $K=2$ table (all sets of size $\\le3$) therefore takes "
       f"{D.min_ours.min():.0f}--{D.min_ours.max():.0f}\\,min to scan, against {D.h_seq.min():.0f}--{D.h_seq.max():.0f}\\,h of sequential "
       f"retraining, and certification adds three retrained backgrounds per field and target, i.e.\\ a few hundred sets. "
       f"The time of a readout grows with the feature dimension and our estimator solves three of them, one per seed, so a one-seed "
       f"estimator costs a third ({ms(D.one)}\\,ms); Table~\\ref{{tab:ablation}} lists the time of every variant. Reading the "
       f"prediction off the network's output head takes well under a millisecond ({1000 * D['head'].max():.1f}\\,ms) but its error of "
       f"$M^{{(2)}}$ is almost four times larger, and warm-start fine-tuning costs {ms(D.ft)}\\,ms per set for one seed. Pre-training one "
       f"reconstruction backbone takes {pr.min():.0f}--{pr.max():.0f}\\,s; three are trained once per data set, and the untrained networks "
       f"need no training. The savings compound over the recurrent workloads of an audit: re-grading after every release, redefined "
       f"targets, and the sets of size four needed for $K=3$.\n")
open(os.path.join(HERE, "..", "cost_text.tex"), "w").write(txt); print(D.round(4).to_string()); print(txt)
