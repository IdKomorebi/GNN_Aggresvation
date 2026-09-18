# -*- coding: utf-8 -*-
"""108-A3：K 递进（Fig.5）的口径处置。

正文主图只画 K=0,1,2（103 号正式三攻击器真值）；K=3 仅在附录，且必须标注口径差异：
  正式口径（K≤2）：单目标 DNN ∪ 多目标 DNN ∪ 梯度提升树
  K=3 可得口径      ：仅多目标 DNN（100 号 k4 真值，M_dnn.npz）
本脚本量化这两个口径本身差多少，从而说明「附录 K=3 的数值不可与正文直接并列」，
并给出 K=2 相对 K=3（同一旧口径内部）的饱和度。
"""
import sys, pickle, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
R100, R103 = REPO / "DNN_Aggresvation100", REPO / "DNN_Aggresvation103"
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure, index_of
from analyze_est import marg_tables
from runlog import log
A = ROOT / "outputs/analysis"

rows = []
for ds in ["pjm", "caiso"]:
    meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb"))
    keys, act = meta["keys_k3"], meta["active"]
    Vt = np.load(R103 / f"outputs/analysis/{ds}_truth_official.npz")["V"]
    Dt, bk = marg_tables(Vt, keys, act, kmax=2); bsz = np.array([len(T) for T in bk[0]])
    Mo = {K: Dt[:, bsz <= K].max(1) for K in [0, 1, 2]}          # 正式三攻击器
    Ma = np.load(R100 / f"outputs/analysis/{ds}_M_attack.npz")["M"]   # 多目标DNN+树, K=0..2
    Md = np.load(R100 / f"outputs/analysis/{ds}_M_dnn.npz")["M"]      # 仅多目标DNN, K=0..3
    for K in [0, 1, 2, 3]:
        r = dict(数据集=ds, K=K)
        r["仅多目标DNN"] = float(Md[:, K].mean())
        r["多目标DNN+树"] = float(Ma[:, K].mean()) if K <= 2 else np.nan
        r["正式三攻击器"] = float(Mo[K].mean()) if K <= 2 else np.nan
        r["正式减仅多目标DNN"] = r["正式三攻击器"] - r["仅多目标DNN"] if K <= 2 else np.nan
        rows.append(r)
E = pd.DataFrame(rows); E.to_csv(A / "108_A3_escalation.csv", index=False)

sat = []
for ds in ["pjm", "caiso"]:
    d = E[E.数据集 == ds].set_index("K")
    sat.append(dict(数据集=ds,
                    旧口径M2除以M3=d.loc[2, "仅多目标DNN"] / d.loc[3, "仅多目标DNN"],
                    旧口径M1除以M3=d.loc[1, "仅多目标DNN"] / d.loc[3, "仅多目标DNN"],
                    正式口径M1除以M2=d.loc[1, "正式三攻击器"] / d.loc[2, "正式三攻击器"],
                    K2处口径差=d.loc[2, "正式三攻击器"] - d.loc[2, "仅多目标DNN"],
                    K2到K3增量_旧口径=d.loc[3, "仅多目标DNN"] - d.loc[2, "仅多目标DNN"]))
S = pd.DataFrame(sat); S.to_csv(A / "108_A3_saturation.csv", index=False)
log("ESCAL", "DONE", "K 递进口径对照完成")
print(E.to_markdown(index=False, floatfmt=".4f")); print(); print(S.to_markdown(index=False, floatfmt=".4f"))
