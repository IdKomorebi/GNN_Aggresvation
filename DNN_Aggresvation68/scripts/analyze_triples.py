"""DNN68 三阶分析：用 67 号 pair/single 真值 + 68 三元组估计，算三阶增量协同，
选 top 候选 + 随机样本写待认证子集（供 retrain 认证）。

三阶增量协同：syn3_c(i,j,k) = v_c(ijk) − max( v_c(ij), v_c(ik), v_c(jk) )
（>0 且显著 = 超出最好二元子集的真三阶效应）。三元组 v 用估计器 K200，二元/一元用重训真值。
输出：outputs/triples_top.csv（按 syn3 排序）+ outputs/certify_subsets.json（top200+random200）。
"""
from __future__ import annotations

import json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRUTH67 = ROOT.parent / "DNN_Aggresvation67/outputs/retrain"
TRI = ROOT / "outputs/triples"
N_TOP = 200
N_RAND = 200


def main():
    import yaml
    sys.path.insert(0, str(ROOT))
    from src.data_processing import prepare_data
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    gen = prepare_data(cfg)["general"]

    singles = {}; pairs = {}; conf_names = None
    for f in TRUTH67.glob("*_dnn_seed0.json"):
        j = json.loads(f.read_text()); sid = j["subset_id"]
        if conf_names is None:
            conf_names = sorted(j["per_conf_r2"].keys())
        if sid.startswith("s"):
            singles[int(sid[1:])] = j["per_conf_r2"]
        elif sid.startswith("p"):
            a, b = sid[1:].split("_"); pairs[(int(a), int(b))] = j["per_conf_r2"]

    def vpair(a, b, c):
        return pairs[(min(a, b), max(a, b))][c]

    files = list(TRI.glob("*.json"))
    print(f"三元组估计文件: {len(files)}")
    rows = []
    for f in files:
        d = json.loads(f.read_text()); i, j, k = d["ijk"]; v = d["v"]
        for c in conf_names:
            best_pair = max(vpair(i, j, c), vpair(i, k, c), vpair(j, k, c))
            rows.append({"i": i, "j": j, "k": k, "conf": c,
                         "v_ijk_est": v[c], "best_pair": best_pair,
                         "syn3_est": v[c] - best_pair})
    df = pd.DataFrame(rows)
    df["fi"] = df.i.map(lambda x: gen[x]); df["fj"] = df.j.map(lambda x: gen[x])
    df["fk"] = df.k.map(lambda x: gen[x])
    df.sort_values("syn3_est", ascending=False, inplace=True)
    df.head(300)[["fi", "fj", "fk", "conf", "v_ijk_est", "best_pair", "syn3_est"]].to_csv(
        ROOT / "outputs/triples_top.csv", index=False)

    print("=== 三阶增量协同 top15（估计器 K200；正=超出最好二元子集）===")
    print(df.head(15)[["fi", "fj", "fk", "conf", "v_ijk_est", "best_pair", "syn3_est"]].to_string(index=False))
    print(f"\nsyn3_est>0.05 的条目: {(df.syn3_est>0.05).sum()} / {len(df)}")
    print(f"syn3_est>0.10 的条目: {(df.syn3_est>0.10).sum()}")

    # 选认证子集：top 三元组（去重 ijk）+ 随机三元组
    top_ijk = df.head(N_TOP * 3)[["i", "j", "k"]].drop_duplicates().head(N_TOP)
    rng = np.random.RandomState(0)
    all_tri = list(combinations(range(len(gen)), 3))
    rand_idx = rng.choice(len(all_tri), size=N_RAND, replace=False)
    cert = {}
    for _, r in top_ijk.iterrows():
        i, j, k = int(r.i), int(r.j), int(r.k)
        cert[f"t{i:02d}_{j:02d}_{k:02d}"] = {"group": "triple_top", "size": 3,
                                             "fields": [gen[i], gen[j], gen[k]]}
    for ridx in rand_idx:
        i, j, k = all_tri[ridx]
        cert[f"t{i:02d}_{j:02d}_{k:02d}"] = {"group": "triple_rand", "size": 3,
                                             "fields": [gen[i], gen[j], gen[k]]}
    (ROOT / "outputs/certify_subsets.json").write_text(json.dumps(cert, indent=1, ensure_ascii=False))
    print(f"\n写待认证三元组 {len(cert)} 个 -> certify_subsets.json（top≤{N_TOP} + rand{N_RAND}）")


if __name__ == "__main__":
    main()
