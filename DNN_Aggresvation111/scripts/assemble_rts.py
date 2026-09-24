# -*- coding: utf-8 -*-
"""111 号 步骤 2：由逐时出清结果组装数据集与字段角色表（outputs/dataset.csv、outputs/fields.json）。"""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RTS = os.path.join(ROOT, "..", "data", "external", "RTS-GMLC", "RTS_Data")
SRC, TS = os.path.join(RTS, "SourceData"), os.path.join(RTS, "timeseries_data_files")
OUT = os.path.join(ROOT, "outputs")
br = pd.read_csv(os.path.join(SRC, "branch.csv")); gen = pd.read_csv(os.path.join(SRC, "gen.csv"))
bus = pd.read_csv(os.path.join(SRC, "bus.csv")); bix = {b: i for i, b in enumerate(bus["Bus ID"])}
th = gen[gen["Unit Type"].isin(["CC", "CT", "STEAM", "NUCLEAR"])].reset_index(drop=True)
z = np.load(os.path.join(OUT, "rts_dispatch_raw.npz")); hrs = z["hours"]


def da(sub, name):
    d = pd.read_csv(os.path.join(TS, sub, f"DAY_AHEAD_{name}.csv"))
    if "Period" in d.columns:
        return d.drop(columns=["Year", "Month", "Day", "Period"]).reset_index(drop=True)
    return pd.DataFrame(d.drop(columns=["Year", "Month", "Day"]).values.reshape(-1, 1))


load_da = da("Load", "regional_Load").values
typ = dict(zip(gen["GEN UID"], gen["Unit Type"]))


def da_type(sub, name, types):
    d = da(sub, name); cols = [c for c in d.columns if typ.get(c) in types]
    return d[cols].sum(1).values


df = pd.DataFrame({
    # ---------------- 候选：及时披露的公开信息
    "负荷预测_区1": load_da[:, 0], "负荷预测_区2": load_da[:, 1], "负荷预测_区3": load_da[:, 2],
    "风电出力预测": da_type("WIND", "wind", {"WIND"}), "光伏出力预测": da_type("PV", "pv", {"PV"}),
    "屋顶光伏预测": da_type("RTPV", "rtpv", {"RTPV"}), "水电出力预测": da_type("Hydro", "hydro", {"HYDRO", "ROR"}),
    "调节上调需求": da("Reserves", "regional_Reg_Up").values[:, 0],
    "调节下调需求": da("Reserves", "regional_Reg_Down").values[:, 0],
    "旋转备用需求": sum(da("Reserves", f"regional_Spin_Up_R{k}").values[:, 0] for k in (1, 2, 3)),
}).iloc[hrs].reset_index(drop=True)
L = z["LMP"]
df["系统电能价格"] = L[:, bix[101]]
for b in [116, 121, 215, 223, 303, 309, 317, 318, 325]:
    df[f"节点电价_{b}"] = L[:, bix[b]]
for u in ["C6", "C29", "CA-1"]:
    df[f"阻塞_{u}"] = z["BIND"][:, br.index[br.UID == u][0]].astype(float)
cand = list(df.columns)
# ---------------- 敏感目标
PU, F = z["PU"], z["FLOW"]
for u in ["223_STEAM_3", "221_CC_1"]:
    df[f"Y_机组出力_{u}"] = PU[:, th.index[th["GEN UID"] == u][0]]
df["Y_线路潮流_C35"] = F[:, br.index[br.UID == "C35"][0]]
targ = [c for c in df.columns if c.startswith("Y_")]
df.to_csv(os.path.join(OUT, "dataset.csv"), index=False)

rule = {}
for c in cand:
    if c.startswith("负荷预测"): rule[c] = "第三十二条（六）系统负荷预测；附表 6.41"
    elif c.endswith("预测"): rule[c] = "第三十二条（六）新能源（分电源类型）/水电出力预测；附表 6.42"
    elif c.endswith("需求"): rule[c] = "第三十二条（七）辅助服务需求信息；附表 6.46"
    elif "价格" in c or "电价" in c: rule[c] = "附表 6.52 节点边际电价及电能、阻塞分量，出清后及时披露"
    elif c.startswith("阻塞"): rule[c] = "附表 6.54 各时段出清的断面约束及阻塞情况"
rule.update({"Y_机组出力_223_STEAM_3": "第十七条（三）机组实际出力——特定信息",
             "Y_机组出力_221_CC_1": "第十七条（三）机组实际出力——特定信息",
             "Y_线路潮流_C35": "附表 6.66 仅披露重要线路与变压器平均潮流（日），逐时潮流不在清单"})
spec = dict(dataset="RTS-GMLC 全年逐时 DC-OPF 出清", cand=cand, targ=targ, kmax=4, taus=[0.5, 0.7, 0.9], rule=rule)
json.dump(spec, open(os.path.join(OUT, "fields.json"), "w"), ensure_ascii=False, indent=1)
print(f"样本 {len(df)}，候选 {len(cand)}，目标 {len(targ)}")
print(df.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
