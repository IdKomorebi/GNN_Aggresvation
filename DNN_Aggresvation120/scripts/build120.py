# -*- coding: utf-8 -*-
"""120 号（只汇报、不进论文）：大字段空间。RTS-GMLC 同一份逐时出清（111 号），候选扩到约 100 个：
全部 73 个节点电价（附表 6.52：节点电价市场须披露所有节点电价）、全年阻塞时段 ≥1% 的线路的阻塞标志（6.54）、
10 个预测与需求字段（与 111 号相同）。目标：内部线路 C35 逐时潮流（6.66 只披露重要线路日平均潮流）。
去别名规则同冻结协议：标准化后逐行相等（精确或确定性仿射）的字段只保留第一个；近重复保留。"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
S111 = os.path.join(REPO, "DNN_Aggresvation111", "outputs")
RTS = os.path.join(REPO, "data", "external", "RTS-GMLC", "RTS_Data", "SourceData")
br = pd.read_csv(os.path.join(RTS, "branch.csv")); bus = pd.read_csv(os.path.join(RTS, "bus.csv"))
z = np.load(os.path.join(S111, "rts_dispatch_raw.npz")); d111 = pd.read_csv(os.path.join(S111, "dataset.csv"))
base = [c for c in d111.columns if not c.startswith(("Y_", "节点电价", "系统电能价格", "阻塞"))]
df = d111[base].copy()
for k, b in enumerate(bus["Bus ID"]):
    df[f"节点电价_{b}"] = z["LMP"][:, k]
freq = z["BIND"].mean(0)
for k in np.where(freq >= 0.01)[0]:
    df[f"阻塞_{br.UID[k]}"] = z["BIND"][:, k].astype(float)
X = (df - df.mean()) / df.std().replace(0, 1); keep, drop = [], []
for c in df.columns:
    if df[c].std() == 0:
        drop.append(c); continue
    dup = False
    for k in keep:
        r = np.corrcoef(X[c], X[k])[0, 1]
        if abs(r) > 1 - 1e-9:
            dup = True; drop.append(c); break
    if not dup:
        keep.append(c)
df = df[keep]; df["Y_线路潮流_C35"] = d111["Y_线路潮流_C35"].values
spec = dict(dataset="RTS-GMLC 大字段空间（120 号）", cand=keep, targ=["Y_线路潮流_C35"], kmax=3, taus=[0.5, 0.7],
            dropped_alias=drop, rule={})
df.to_csv(os.path.join(ROOT, "outputs", "dataset.csv"), index=False)
json.dump(spec, open(os.path.join(ROOT, "outputs", "fields.json"), "w"), ensure_ascii=False, indent=1)
D = pipe.prep(df, keep, ["Y_线路潮流_C35"]); masks, keys = pipe.enumerate_sets(len(keep), 3)
np.savez(os.path.join(ROOT, "outputs", "D.npz"), **{k: v for k, v in D.items() if isinstance(v, np.ndarray)},
         masks=masks.astype(np.float32), keys=np.array(["|".join(map(str, k)) for k in keys]))
print(f"候选 {len(keep)}（去别名 {len(drop)}：{drop[:8]}…），阻塞线路 {int((freq >= 0.01).sum())}，集合 {len(keys)}，train {len(D['Xtr'])}")
