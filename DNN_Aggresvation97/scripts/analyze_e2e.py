# -*- coding: utf-8 -*-
"""97 号端到端判读：修好的 φ 在**真实机密字段**六阶上到底行不行。

与 96 号构成配对对照——同一候选池（sample_seed 960727、n_sample 150000×2），
唯一变量是 φ。96 号老 φ 的基线写死在 BASE 里，取自其 WORKLOG 的 k=6 认证节。

预注册判据（开跑前已写下，见对话与 WORKLOG）：
  top 认证显著高于对照（Mann-Whitney p<0.05 且 AUC>0.7）⟹ k*=6 成立；
  仍无判别力 ⟹ 合成注入上的增益不迁移到真实场景，k*=5 保持，且这本身要如实报告。
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE = {"top_est": 0.102, "top_cert": 0.031, "ctrl_est": 0.028, "ctrl_cert": 0.032,
        "auc": 0.42, "rho": -0.16, "prec010": 0}


def auc(a, b):
    """P(随机取的 top > 随机取的 ctrl)，含并列折半。"""
    a, b = np.asarray(a), np.asarray(b)
    return float((np.subtract.outer(a, b) > 0).mean()
                 + 0.5 * (np.subtract.outer(a, b) == 0).mean())


fs = sorted(glob.glob("../DNN_Aggresvation96/outputs/certify_o6_e2e_s*of*.csv"))
assert fs, "还没有认证结果"
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
print(f"读入 {len(fs)} 个分片，共 {len(d)} 条\n")

gcol = "group" if "group" in d.columns else ("kind" if "kind" in d.columns else None)
ecol = next(c for c in d.columns if "est" in c or "scan" in c)
ccol = next(c for c in d.columns if "cert" in c or c.endswith("_true"))
print(f"[列名] 分组={gcol} 估计={ecol} 认证={ccol}\n")

top = d[d[gcol].astype(str).str.contains("top|strong", case=False)]
ctl = d[~d.index.isin(top.index)]
if len(top) == 0 or len(ctl) == 0:
    print(d.head(20).to_string())
    raise SystemExit("分组识别失败，请人工看列名")

p = mannwhitneyu(top[ccol], ctl[ccol], alternative="greater").pvalue
A = auc(top[ccol], ctl[ccol])
rho = pd.Series(top[ecol]).corr(pd.Series(top[ccol]), method="spearman")
prec = int((top[ccol] > 0.10).sum())

print("=========== 97 号（修好的 φ） vs 96 号（老 φ）配对对照 ===========")
print(f"{'指标':<22}{'96 号老 φ':>14}{'97 号新 φ':>14}")
rows = [("top 估计均值", BASE["top_est"], top[ecol].mean()),
        ("top 认证均值", BASE["top_cert"], top[ccol].mean()),
        ("对照 认证均值", BASE["ctrl_cert"], ctl[ccol].mean()),
        ("AUC(top vs 对照)", BASE["auc"], A),
        ("ρ(估计, 认证)", BASE["rho"], rho),
        ("精确率@0.10", f"{BASE['prec010']}/10", f"{prec}/{len(top)}")]
for name, b, v in rows:
    bs = b if isinstance(b, str) else f"{b:.3f}"
    vs = v if isinstance(v, str) else f"{v:.3f}"
    print(f"{name:<22}{bs:>14}{vs:>14}")
print(f"\nMann-Whitney（top > 对照，单侧）p = {p:.4f}")
print(f"top 最强认证值 = {top[ccol].max():.4f}")

ok = (p < 0.05) and (A > 0.7)
print("\n★判定：", "k*=6 成立——真实机密字段上六阶可认证" if ok
      else "未达预注册判据 ⟹ k*=5 保持；注入定标高估了真实场景能力（如实报告）")
