# -*- coding: utf-8 -*-
"""111 号 步骤 1：RTS-GMLC 全年逐时市场出清（DC-OPF，线性规划，HiGHS）。

数据来源：Barrows 等，The IEEE Reliability Test System: A Proposed 2019 Update（GitHub GridMod/RTS-GMLC），
本地克隆于 data/external/RTS-GMLC。全年 8,784 小时。

建模（简化，均写入 CHANGELOG）：
  - 73 母线、120 条交流支路按连续额定容量（Cont Rating）限流；1 条直流线路、储能、同步调相机忽略。
  - 火电 73 台：出力比例点 Output_pct_k × PMax 分段；首段 [0, pct_0] 按平均热耗率，其后各段按增量热耗率，
    边际成本 = 热耗率 × 燃料价格 / 1000 + VOM。不做机组组合，最小出力放宽为 0（按报价顺序经济调度）。
  - 风、光伏、屋顶光伏、水电、径流、光热：实时 5 分钟序列取小时均值为可用出力，零成本，可弃。
  - 负荷：三个区域的实时负荷按母线基准负荷（MW Load）在区内的比例分配；每母线可切负荷，价格 1000 $/MWh。
  - 节点电价 = 节点平衡约束对偶；阻塞 = 线路约束对偶非零。

字段角色依据《电力市场信息披露基本规则》（国能发监管〔2024〕9号）——见 CHANGELOG 表格。
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy import sparse

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RTS = os.path.join(ROOT, "..", "data", "external", "RTS-GMLC", "RTS_Data")
SRC, TS = os.path.join(RTS, "SourceData"), os.path.join(RTS, "timeseries_data_files")
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
VOLL, BASE = 1000.0, 100.0

bus = pd.read_csv(os.path.join(SRC, "bus.csv")); br = pd.read_csv(os.path.join(SRC, "branch.csv"))
gen = pd.read_csv(os.path.join(SRC, "gen.csv"))
B = list(bus["Bus ID"]); bix = {b: i for i, b in enumerate(B)}; nb = len(B)
THERMAL = ["CC", "CT", "STEAM", "NUCLEAR"]; RENEW = ["WIND", "PV", "RTPV", "HYDRO", "ROR", "CSP"]
th = gen[gen["Unit Type"].isin(THERMAL)].reset_index(drop=True)
rn = gen[gen["Unit Type"].isin(RENEW)].reset_index(drop=True)


def ts_hourly(sub, name, rt=True):
    """读取 RT（5 分钟→小时均值）或 DA（小时）序列，返回 (8784, cols) DataFrame。"""
    f = os.path.join(TS, sub, f"{'REAL_TIME' if rt else 'DAY_AHEAD'}_{name}.csv")
    d = pd.read_csv(f)
    if "Period" in d.columns:
        d = d.drop(columns=["Year", "Month", "Day"]).reset_index(drop=True)
        per = 12 if rt else 1
        vals = d.drop(columns=["Period"])
        return vals.groupby(np.arange(len(vals)) // per).mean().reset_index(drop=True)
    # 行=日、列=1..24 的格式（部分备用文件）
    return pd.DataFrame(d.drop(columns=["Year", "Month", "Day"]).values.reshape(-1, 1))


# ---------------- 可用出力（RT 小时均值）与预测（DA）
RT_parts, DA_parts = [], []
for sub, name in [("WIND", "wind"), ("PV", "pv"), ("RTPV", "rtpv"), ("Hydro", "hydro"), ("CSP", "Natural_Inflow")]:
    RT_parts.append(ts_hourly(sub, name, True)); DA_parts.append(ts_hourly(sub, name, False))
RT = pd.concat(RT_parts, axis=1); DA = pd.concat(DA_parts, axis=1)
H = len(RT)
avail = np.zeros((H, len(rn)), np.float32); miss = []
for k, u in enumerate(rn["GEN UID"]):
    if u in RT.columns:
        avail[:, k] = np.minimum(RT[u].values[:H], rn.loc[k, "PMax MW"])
    else:
        miss.append(u)
load_rt = ts_hourly("Load", "regional_Load", True).values[:H]     # (H,3) 区域 1..3
load_da = ts_hourly("Load", "regional_Load", False).values[:H]
share = np.zeros((nb, 3))
for a in (1, 2, 3):
    m = bus["Area"].values == a
    share[m, a - 1] = bus.loc[m, "MW Load"].values / bus.loc[m, "MW Load"].sum()

# ---------------- 火电分段报价
seg_bus, seg_cap, seg_cost, seg_unit = [], [], [], []
for u, r in th.iterrows():
    pmax, fp, vom = r["PMax MW"], r["Fuel Price $/MMBTU"], (0 if pd.isna(r["VOM"]) else r["VOM"])
    pts = [r[f"Output_pct_{k}"] for k in range(5) if not pd.isna(r.get(f"Output_pct_{k}", np.nan))]
    hrs = [r["HR_avg_0"]] + [r[f"HR_incr_{k}"] for k in range(1, 5) if not pd.isna(r.get(f"HR_incr_{k}", np.nan))]
    lo = 0.0
    for k, pt in enumerate(pts):
        cap = (pt - lo) * pmax
        if cap > 1e-6:
            seg_bus.append(bix[r["Bus ID"]]); seg_cap.append(cap); seg_unit.append(u)
            seg_cost.append(hrs[min(k, len(hrs) - 1)] * fp / 1000.0 + vom)
        lo = pt
seg_bus, seg_cap, seg_cost, seg_unit = map(np.array, (seg_bus, seg_cap, seg_cost, seg_unit))
ren_bus = np.array([bix[b] for b in rn["Bus ID"]])

# ---------------- 网络
fb = np.array([bix[b] for b in br["From Bus"]]); tb = np.array([bix[b] for b in br["To Bus"]])
bl = BASE / br["X"].values; rate = br["Cont Rating"].values.astype(float); nl = len(br)
Af = sparse.csr_matrix((np.r_[bl, -bl], (np.r_[np.arange(nl), np.arange(nl)], np.r_[fb, tb])), shape=(nl, nb))
Bbus = (sparse.csr_matrix((np.ones(nl), (np.arange(nl), fb)), shape=(nl, nb))
        - sparse.csr_matrix((np.ones(nl), (np.arange(nl), tb)), shape=(nl, nb))).T @ Af   # (nb,nb)
ns, nr = len(seg_cap), len(ren_bus)
# 变量：[θ (nb) | 火电段 (ns) | 新能源 (nr) | 切负荷 (nb)]
Gs = sparse.csr_matrix((np.ones(ns), (seg_bus, np.arange(ns))), shape=(nb, ns))
Gr = sparse.csr_matrix((np.ones(nr), (ren_bus, np.arange(nr))), shape=(nb, nr))
A_eq = sparse.hstack([-Bbus, Gs, Gr, sparse.eye(nb)]).tocsc()
Z = sparse.csr_matrix((nl, ns + nr + nb))
A_ub = sparse.vstack([sparse.hstack([Af, Z]), sparse.hstack([-Af, Z])]).tocsc()
b_ub = np.r_[rate, rate]
c = np.r_[np.zeros(nb), seg_cost, np.zeros(nr), np.full(nb, VOLL)]
REF = 0


def solve(h):
    loadb = share @ load_rt[h]
    bounds = ([(0, 0) if i == REF else (-np.pi, np.pi) for i in range(nb)]
              + [(0, x) for x in seg_cap] + [(0, float(x)) for x in avail[h]] + [(0, float(x)) for x in loadb])
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=loadb, bounds=bounds, method="highs")
    if r.status != 0:
        return None
    x = r.x; theta = x[:nb]; flow = Af @ theta
    pseg = x[nb:nb + ns]; pren = x[nb + ns:nb + ns + nr]; shed = x[nb + ns + nr:]
    punit = np.bincount(seg_unit, weights=pseg, minlength=len(th))
    lmp = r.eqlin.marginals; mu = r.ineqlin.marginals
    return dict(h=h, lmp=lmp, flow=flow, bind=(np.abs(mu[:nl]) + np.abs(mu[nl:]) > 1e-6),
                punit=punit, shed=shed.sum(), curt=float((avail[h] - pren).sum()))


if __name__ == "__main__":
    print(f"H={H} 小时；火电 {len(th)} 台 / {ns} 段；新能源 {nr} 台（缺序列 {len(miss)}：{miss[:5]}）；支路 {nl}")
    t0 = time.time()
    with Pool(40) as pool:
        res = pool.map(solve, range(H), chunksize=16)
    ok = [r for r in res if r is not None]
    print(f"求解 {len(ok)}/{H}，{time.time()-t0:.0f}s")
    LMP = np.array([r["lmp"] for r in ok]); FLOW = np.array([r["flow"] for r in ok])
    BIND = np.array([r["bind"] for r in ok]); PU = np.array([r["punit"] for r in ok])
    np.savez_compressed(os.path.join(OUT, "rts_dispatch_raw.npz"), LMP=LMP, FLOW=FLOW, BIND=BIND, PU=PU,
                        hours=np.array([r["h"] for r in ok]), shed=np.array([r["shed"] for r in ok]),
                        curt=np.array([r["curt"] for r in ok]))
    print("切负荷小时占比", np.mean([r["shed"] > 1e-3 for r in ok]).round(4),
          "；有阻塞小时占比", BIND.any(1).mean().round(3),
          "；各线路阻塞频率前 8：", sorted(((BIND[:, l].mean(), br["UID"][l]) for l in range(nl)), reverse=True)[:8])
    print("LMP 分位数（1/50/99%）", np.percentile(LMP, [1, 50, 99]).round(2), "；不同节点 LMP 序列相关最小值",
          np.corrcoef(LMP.T).min().round(3))
