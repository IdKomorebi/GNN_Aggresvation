# -*- coding: utf-8 -*-
"""109 号补充诊断：为什么 V 单调上升，M^(1)/M^(2) 却在小样本下反超 full？

假设：M^(K)=max_T Δ(T) 是**最大值统计量**，小样本下边际噪声大，取 max 会系统性挑中正噪声
（winner's curse）；候选背景数随 K 增长（K=0 只有空背景，K=1 有 40 个，K=2 有 820 个），
所以正偏应随 K 单调增强。

分解：M_frac − M_full = [M_frac − Δ_frac(T_full)]  −  [M_full − Δ_frac(T_full)]
                          ↑ 选择增益（winner's curse）      ↑ 样本量导致的真实低估
若假设成立：选择增益随 K 增大，且在小 n_aux 时足以盖过真实低估。
"""
import sys, glob, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent
R100 = REPO / "DNN_Aggresvation100"
sys.path.insert(0, str(R100 / "src")); sys.path.insert(0, str(R100 / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from mkfull import closure
from analyze_est import marg_tables
from runlog import log
A = ROOT / "outputs/analysis"; FRACS = [10, 25, 50, 100]


def load_truth(ds, f):
    fs = sorted(glob.glob(str(ROOT / f"outputs/truth/{ds}_f{f:03d}_seed0_s*of3.npz")))
    zs = [np.load(x) for x in fs]; n = max(int(z["idx"].max()) for z in zs) + 1
    out = {}
    for k in ["clean", "val_r2"]:
        a = np.zeros((n, 12), np.float32)
        for z in zs: a[z["idx"]] = z[k]
        out[k] = a
    return out, int(zs[0]["n_aux"])


rows = []
for ds in ["pjm", "caiso"]:
    meta = pickle.load(open(R100 / f"outputs/sets/{ds}_meta.pkl", "rb"))
    keys, act = meta["keys_k3"], meta["active"]
    Dm, naux = {}, {}
    for f in FRACS:
        T, na = load_truth(ds, f); naux[f] = na
        Vb = closure(np.clip(T["clean"], 0, 1), keys, Vval=T["val_r2"])
        D, bk = marg_tables(Vb, keys, act, kmax=2); Dm[f] = D
        bsz = np.array([len(t) for t in bk[0]])
    for f in FRACS:
        for K in [0, 1, 2]:
            sel = bsz <= K; nT = int(sel.sum())
            Df, Du = Dm[f][:, sel], Dm[100][:, sel]
            jf, ju = Df.argmax(1), Du.argmax(1)
            Mf = np.take_along_axis(Df, jf[:, None], 1)[:, 0]        # frac 自选见证
            Mu = np.take_along_axis(Du, ju[:, None], 1)[:, 0]        # full 自选见证
            Df_at_u = np.take_along_axis(Df, ju[:, None], 1)[:, 0]   # frac 在 full 见证上
            Du_at_f = np.take_along_axis(Du, jf[:, None], 1)[:, 0]   # full 在 frac 见证上（认证下界）
            rows.append(dict(数据集=ds, 比例=f"{f}%", n_aux=naux[f], K=K, 候选背景数=nT,
                             M_frac=float(Mf.mean()), M_full=float(Mu.mean()),
                             总偏差=float((Mf - Mu).mean()),
                             选择增益=float((Mf - Df_at_u).mean()),
                             真实低估=float((Mu - Df_at_u).mean()),
                             见证一致率=float((jf == ju).mean()),
                             frac见证在full上的边际比=float(Du_at_f.sum() / Mu.sum())))
W = pd.DataFrame(rows); W.to_csv(A / "109_E_winners_curse.csv", index=False)
rp = A / "report109.md"
if rp.exists():
    t = rp.read_text(encoding="utf-8")
    mark = "## E. winner's curse 分解"
    body = mark + "（M_frac − M_full = 选择增益 − 真实低估）\n\n" + W.to_markdown(index=False, floatfmt=".4f") + "\n"
    t = t.split(mark)[0].rstrip() + "\n\n" + body
    rp.write_text(t, encoding="utf-8")
log("CURSE", "DONE", "winner's curse 分解完成")
print(W.to_markdown(index=False, floatfmt=".4f"))
