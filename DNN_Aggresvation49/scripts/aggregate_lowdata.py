#!/usr/bin/env python3
"""DNN49 低数据汇总：对比不同训练比例下 6 方法的 top-k 曲线 + 各自上界。"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
LD = ROOT/"outputs/lowdata"
ORDER = ["frac_05","frac_10","frac_20","frac_50","frac_100"]
LABEL = {"frac_05":"5%","frac_10":"10%","frac_20":"20%","frac_50":"50%","frac_100":"100%"}
METHODS = ["mask","single","attn","ig","shapley","lrp"]
fracs = [f for f in ORDER if (LD/f/"topk.csv").exists()]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(fracs), figsize=(4.2*len(fracs), 4.6), sharey=True)
if len(fracs)==1: axes=[axes]
colors={"mask":"#4c72b0","single":"#dd8452","attn":"#55a868","ig":"#c44e52","shapley":"#8172b3","lrp":"#da8bc3"}
summary=[]
for ax,f in zip(axes,fracs):
    t=pd.read_csv(LD/f/"topk.csv"); up=float((LD/f/"upper.txt").read_text())
    ks=t["k"].values
    for m in METHODS:
        if m in t: ax.plot(ks,t[m],lw=1.4,label=m,color=colors[m])
    ax.axhline(up,ls="--",color="gray",lw=1.2)
    ax.set_title(f"train={LABEL[f]}  (upper={up:.3f})",fontsize=10)
    ax.set_xlabel("top-k general"); ax.grid(alpha=0.3)
    # top-10 处方法间分化(max-min)
    sp = t[METHODS].iloc[9].max()-t[METHODS].iloc[9].min() if len(t)>=10 else np.nan
    summary.append({"frac":LABEL[f],"upper":round(up,4),"top10_spread":round(sp,4),
                    "best@top10":t[METHODS].iloc[9].idxmax() if len(t)>=10 else ""})
axes[0].set_ylabel("mean R² over 12 confidential"); axes[-1].legend(fontsize=8,loc="lower right")
fig.suptitle("DNN49 low-data: top-k ranking quality of 6 methods across train sizes")
fig.tight_layout()
(LD/"_summary").mkdir(exist_ok=True)
fig.savefig(LD/"_summary/lowdata_topk.png",dpi=140)
sdf=pd.DataFrame(summary); sdf.to_csv(LD/"_summary/lowdata_summary.csv",index=False)
print(sdf.to_string(index=False))
print(f"\n图: {LD/'_summary/lowdata_topk.png'}")
