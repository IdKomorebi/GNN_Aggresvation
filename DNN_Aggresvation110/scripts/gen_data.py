# -*- coding: utf-8 -*-
"""可行性试验：IEEE 30 节点 DC-OPF 市场出清，生成"公开信息 / 特定信息"数据。
字段角色依据《电力市场信息披露基本规则》（国能发监管〔2024〕9号）：
  公开信息（第三十二条）：系统实际负荷、系统负荷预测、新能源分类型总出力预测、出清电价（节点）、
                        输电断面约束及阻塞情况、系统备用
  特定信息（第十七条(三)）：机组实际出力 —— 作为敏感推断目标
"""
import warnings, time, sys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pandapower as pp, pandapower.networks as pn
from multiprocessing import Pool

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = np.random.default_rng(20260923)

base = pn.case30()
BASE_CP1 = base.poly_cost.cp1_eur_per_mw.values.copy()
# 两个新能源场站：风电 bus 26（容量 40MW），光伏 bus 21（容量 25MW）
WIND_BUS, SOLAR_BUS = 26, 21
# 收紧两条关键线路，使阻塞时有时无
TIGHT = {34: 88.0}   # line idx -> max_loading_percent（L24-26）

hours = np.arange(N) % 24
day = np.arange(N) // 24
g = 1.0 + 0.18 * np.sin(2 * np.pi * (hours - 7) / 24) + 0.10 * np.sin(2 * np.pi * day / 90) + rng.normal(0, 0.04, N)
g = np.clip(g, 0.70, 1.18)
bus_noise = rng.normal(1, 0.08, (N, len(base.load)))
solar_cf = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) * rng.beta(4, 2, N)
wind_cf = np.clip(0.45 + 0.35 * np.sin(2 * np.pi * day / 11 + 1.3) + rng.normal(0, 0.18, N), 0, 1)
bid_mult = rng.uniform(0.85, 1.15, (N, len(BASE_CP1)))


def solve(t):
    net = pn.case30()
    net.load["p_mw"] = base.load.p_mw.values * g[t] * bus_noise[t]
    net.load["q_mvar"] = base.load.q_mvar.values * g[t]
    pp.create_sgen(net, WIND_BUS, p_mw=40 * wind_cf[t], controllable=False)
    pp.create_sgen(net, SOLAR_BUS, p_mw=25 * solar_cf[t], controllable=False)
    net.poly_cost["cp1_eur_per_mw"] = BASE_CP1 * bid_mult[t]
    for li, pct in TIGHT.items():
        net.line.at[li, "max_loading_percent"] = pct
    try:
        pp.rundcopp(net)
    except Exception:
        return None
    if not net.OPF_converged:
        return None
    lam = net.res_bus.lam_p.values
    ld = net.res_line.loading_percent.values
    return dict(t=t, load=net.load.p_mw.sum(), wind=40 * wind_cf[t], solar=25 * solar_cf[t],
                lmp=lam.copy(), cong34=float(ld[34] >= TIGHT[34] - 0.5), cong9=0.0,
                pg=net.res_gen.p_mw.values.copy(), pext=float(net.res_ext_grid.p_mw.values[0]),
                flow34=float(net.res_line.p_from_mw.values[34]),
                cap_conv=float(net.gen.max_p_mw.sum()))


if __name__ == "__main__":
    import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
    t0 = time.time()
    with Pool(96) as p:
        out = [r for r in p.map(solve, range(N)) if r is not None]
    print(f"solved {len(out)}/{N} in {time.time()-t0:.0f}s")
    r = np.random.default_rng(7)
    rows = []
    for o in out:
        rows.append({
            # ---- 公开信息（第三十二条）----
            "系统实际负荷": o["load"],
            "系统负荷预测": o["load"] * (1 + r.normal(0, 0.03)),
            "风电总出力预测": o["wind"] * (1 + r.normal(0, 0.08)),
            "光伏总出力预测": o["solar"] * (1 + r.normal(0, 0.06)),
            "节点电价_bus1": o["lmp"][1], "节点电价_bus7": o["lmp"][7],
            "节点电价_bus14": o["lmp"][14], "节点电价_bus24": o["lmp"][24],
            "节点电价_bus26": o["lmp"][26],
            "断面阻塞_L24-26": o["cong34"],
            "系统备用": o["cap_conv"] + 80 - (o["pg"].sum() + o["pext"]),   # 常规机组可用容量 - 常规出力
            # ---- 特定信息（第十七条(三)）：敏感目标 ----
            "Y_机组出力_bus26": o["pg"][2],
            "Y_机组出力_bus21": o["pg"][1],
        })
    df = pd.DataFrame(rows); df.to_csv("case30_market.csv", index=False)
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))
    print("阻塞发生率 L24-26:", df["断面阻塞_L24-26"].mean().round(3))
