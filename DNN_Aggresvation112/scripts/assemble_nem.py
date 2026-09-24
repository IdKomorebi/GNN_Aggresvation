# -*- coding: utf-8 -*-
"""112 号 步骤 1：AEMO NEM 2024 全年真实数据 → 小时级数据集与字段角色表。

来源：AEMO NEMWEB MMSDM 月度归档（data/external/NEM_2024/raw），2024-01..12：
  DISPATCHREGIONSUM（区域需求、可用发电、半调度风光无约束出力预测 UIGF）、DISPATCHPRICE（区域出清价）、
  DISPATCHINTERCONNECTORRES（区域间联络线潮流）、DISPATCH_UNIT_SCADA（机组 5 分钟实际出力）。
只取 INTERVENTION=0 的正常出清结果；5 分钟数据取小时均值。

字段角色：候选 = 在我国《电力市场信息披露基本规则》中属公开信息类别的量（出清价 6.52、实际负荷 6.61、
系统备用 6.62、新能源出力 6.59、联络线输电 6.65）；敏感目标 = 单台机组实际出力（第十七条(三)，特定信息）。
这些量在 NEM 均已作为历史数据公开——本号即"类别敏感、但以历史数据形式公开"的真实数据案例。
"""
import os, io, zipfile, glob, json
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "..", "data", "external", "NEM_2024", "raw")
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
REG = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
RN = dict(NSW1="新南威尔士", QLD1="昆士兰", VIC1="维多利亚", SA1="南澳", TAS1="塔斯马尼亚")


def read_table(tag):
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, f"PUBLIC_DVD_{tag}_2024*.zip"))):
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode("utf-8", "replace").splitlines()
        head = next(l for l in raw if l.startswith("I,")).split(",")[4:]
        body = "\n".join(l.split(",", 4)[4] for l in raw if l.startswith("D,"))
        frames.append(pd.read_csv(io.StringIO(body), header=None, names=head, low_memory=False))
    d = pd.concat(frames, ignore_index=True)
    d["SETTLEMENTDATE"] = pd.to_datetime(d["SETTLEMENTDATE"])
    if "INTERVENTION" in d.columns:
        d = d[d["INTERVENTION"] == 0]
    # 区间结束时刻 → 所属小时（00:05..01:00 归入 00 时）
    d["HOUR"] = (d["SETTLEMENTDATE"] - pd.Timedelta(minutes=5)).dt.floor("h")
    return d


rs = read_table("DISPATCHREGIONSUM"); pr = read_table("DISPATCHPRICE")
ic = read_table("DISPATCHINTERCONNECTORRES"); sc = read_table("DISPATCH_UNIT_SCADA")
print("行数", len(rs), len(pr), len(ic), len(sc))

cols = {}
g = rs.groupby(["HOUR", "REGIONID"])
for key, name in [("TOTALDEMAND", "实际需求"), ("AVAILABLEGENERATION", "可用发电容量"),
                  ("SS_WIND_UIGF", "风电出力预测"), ("SS_SOLAR_UIGF", "光伏出力预测")]:
    if key in rs.columns:
        t = g[key].mean().unstack()
        for r in REG:
            if r in t.columns and t[r].std() > 1e-6:
                cols[f"{name}_{RN[r]}"] = t[r]
t = pr.groupby(["HOUR", "REGIONID"])["RRP"].mean().unstack()
for r in REG:
    cols[f"出清价_{RN[r]}"] = t[r]
t = ic.groupby(["HOUR", "INTERCONNECTORID"])["METEREDMWFLOW"].mean().unstack()
for c in t.columns:
    if t[c].std() > 1e-6:
        cols[f"联络线潮流_{c}"] = t[c]
X = pd.DataFrame(cols)
cand = list(X.columns)

# 敏感目标：挑选可调度、出力波动大的大机组（每类一个：燃煤/水电/燃气，按事先规则挑选，不看与候选的关系）
sch = sc.groupby(["HOUR", "DUID"])["SCADAVALUE"].mean().unstack()
PICK = ["BW01", "TUMUT3", "PPCCGT"]
stats = sch[[c for c in PICK if c in sch.columns]].describe().T
print("目标机组统计:\n", stats[["mean", "std", "min", "max"]].round(1))
for u in PICK:
    if u in sch.columns:
        X[f"Y_机组出力_{u}"] = sch[u]
print("缺失小时数（按列）:", X.isna().sum()[X.isna().sum() > 0].to_dict())
X = X.dropna()
targ = [c for c in X.columns if c.startswith("Y_")]
X.reset_index(drop=True).to_csv(os.path.join(OUT, "dataset.csv"), index=False)

rule = {}
for c in cand:
    if c.startswith("出清价"): rule[c] = "附表 6.52 出清电价"
    elif c.startswith("实际需求"): rule[c] = "附表 6.61 实际负荷"
    elif c.startswith("可用发电"): rule[c] = "附表 6.62 系统备用信息"
    elif "出力预测" in c: rule[c] = "附表 6.42 新能源（分电源类型）总出力预测（AEMO 无约束间歇性发电预测 UIGF）"
    elif c.startswith("联络线"): rule[c] = "附表 6.65 省间联络线输电情况"
for u in targ:
    rule[u] = "第十七条（三）机组实际出力——特定信息（NEM 中已作为历史数据公开）"
unit_desc = {"BW01": "Bayswater 1 号机（新南威尔士，燃煤）", "TUMUT3": "Tumut 3（新南威尔士，抽水蓄能/水电）",
             "PPCCGT": "Pelican Point（南澳，燃气联合循环）"}
spec = dict(dataset="AEMO NEM 2024 全年小时级", cand=cand, targ=targ, kmax=3, taus=[0.5, 0.7, 0.9],
            rule=rule, units=unit_desc)
json.dump(spec, open(os.path.join(OUT, "fields.json"), "w"), ensure_ascii=False, indent=1)
print(f"样本 {len(X)}，候选 {len(cand)}，目标 {len(targ)}")
print(X.describe().T[["mean", "std", "min", "max"]].round(1).to_string())
