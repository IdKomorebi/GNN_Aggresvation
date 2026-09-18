# -*- coding: utf-8 -*-
"""108-A2：元训练 φ（L1x，仅 PJM）的定位论证——支持「降级到附录」而非补跑 CAISO。

论证三步：
 1. 元训练权重来自 97 号的 PJM 专用管线（69 号 prepare_data），与 FINAL_PROTOCOL 冻结的
    41 字段口径不是同一条数据路径；补跑 CAISO 需移植该管线，有引入协议不一致的风险。
 2. 元训练在 V/M 精度上并不优于主口径（φ+x,x² 三种子集成）。
 3. 元训练唯一占优处是「见证召回」；本脚本验证：主口径改用 top-3 认证后，
    其认证下界已超过元训练的 top-1 认证 ⟹ 该优势被 A1 的区间口径吸收，不构成保留理由。
"""
import sys, pickle, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
R100, R103 = REPO / "DNN_Aggresvation100", REPO / "DNN_Aggresvation103"
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure
from analyze_est import marg_tables
from runlog import log
A = ROOT / "outputs/analysis"


def shards(pat, n, key, ncol=12):
    fs = sorted(glob.glob(pat))
    if not fs or len(fs) != int(fs[0].split("of")[-1].split(".")[0]): return None
    out = np.zeros((n, ncol), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z[key]
    return out


ds = "pjm"
meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb"))
keys, act = meta["keys_k3"], meta["active"]; n = len(keys)
Vt = np.load(R103 / f"outputs/analysis/{ds}_truth_official.npz")["V"]
Dt, bk = marg_tables(Vt, keys, act, kmax=2); bsz = np.array([len(T) for T in bk[0]])

EST = {"主口径 φ+x,x²（三种子集成）": str(R103 / f"outputs/est/{ds}_k3_L0ensx_s*of3.npz"),
       "元训练 φ+x,x²（L1x）": str(R103 / f"outputs/est/{ds}_k3_L1x_s*of3.npz")}
rows = []
for nm, pat in EST.items():
    E = shards(pat, n, "v")
    if E is None: continue
    Eb = closure(np.clip(E, 0, 1), keys)
    De = marg_tables(Eb, keys, act, kmax=2)[0]
    for K in [0, 1, 2]:
        sel = bsz <= K; Des, Dts = De[:, sel], Dt[:, sel]
        Mt = Dts.max(1); je = Des.argmax(1)
        Me = np.take_along_axis(Des, je[:, None], 1)[:, 0]
        L1 = np.take_along_axis(Dts, je[:, None], 1)[:, 0]
        top3 = np.argsort(-Des, 1)[:, :3, :]
        L3 = np.take_along_axis(Dts, top3, 1).max(1)
        rows.append(dict(估计器=nm, K=K, M_MAE=float(np.abs(Me - Mt).mean()),
                         认证下界比_top1=float(L1.sum() / Mt.sum()), 认证下界比_top3=float(L3.sum() / Mt.sum()),
                         见证召回_top1=float((L1 >= Mt - 0.02).mean()), 见证召回_top3=float((L3 >= Mt - 0.02).mean())))
T = pd.DataFrame(rows); T.to_csv(A / "108_A2_metatrain.csv", index=False)
log("META", "DONE", "元训练定位对照完成")
print(T.to_markdown(index=False, floatfmt=".4f"))
print("\n判定：主口径 top-3 认证 vs 元训练 top-1 认证")
for K in [1, 2]:
    a = T[(T.估计器.str.startswith("主口径")) & (T.K == K)].iloc[0]
    b = T[(T.估计器.str.startswith("元训练")) & (T.K == K)].iloc[0]
    print(f"  K={K}: 主口径top3 下界比 {a.认证下界比_top3:.4f} / 见证召回 {a.见证召回_top3:.4f}"
          f"  vs  元训练top1 {b.认证下界比_top1:.4f} / {b.见证召回_top1:.4f}"
          f"  → {'主口径胜' if a.认证下界比_top3 > b.认证下界比_top1 else '元训练仍占优'}")
