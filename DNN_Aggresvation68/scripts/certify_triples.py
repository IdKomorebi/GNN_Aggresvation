"""DNN68 三阶认证：用重训真值重算 top+random 三元组的 syn3，验证估计器筛选保真。

syn3_true_c(i,j,k) = v_c^retrain(ijk) − max(v_c(ij), v_c(ik), v_c(jk))  （二元用 67 真值）
对比：估计 syn3 vs 认证 syn3 的排序保真（top 组）；是否存在显著三阶协同（认证 syn3>阈值）。
输出：outputs/triples_certified.csv + 摘要打印。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TRUTH67 = ROOT.parent / "DNN_Aggresvation67/outputs/retrain"
CERT = ROOT / "outputs/retrain"   # 认证重训（t..）与中段真值(bt..)同目录


def main():
    import yaml
    sys.path.insert(0, str(ROOT))
    from src.data_processing import prepare_data
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    gen = prepare_data(cfg)["general"]; name2i = {n: i for i, n in enumerate(gen)}

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

    cert_meta = json.load(open(ROOT / "outputs/certify_subsets.json"))
    # 估计 syn3（analyze 已算）用于对齐
    est = pd.read_csv(ROOT / "outputs/triples_top.csv")
    est_map = {(r.fi, r.fj, r.fk, r.conf): r.syn3_est for r in est.itertuples()}

    rows = []
    for tid, meta in cert_meta.items():
        f = CERT / f"{tid}_dnn_seed0.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text()); fields = d["fields"]
        i, j, k = [name2i[x] for x in fields]
        for c in conf_names:
            vijk = d["per_conf_r2"][c]
            bp = max(vpair(i, j, c), vpair(i, k, c), vpair(j, k, c))
            rows.append({"group": meta["group"], "fi": gen[i], "fj": gen[j], "fk": gen[k],
                         "conf": c, "v_ijk_true": vijk, "best_pair": bp,
                         "syn3_true": vijk - bp,
                         "syn3_est": est_map.get((gen[i], gen[j], gen[k], c), np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/triples_certified.csv", index=False)

    top = df[df.group == "triple_top"]; rand = df[df.group == "triple_rand"]
    print(f"认证条目: top {len(top)}，random {len(rand)}")
    print("\n=== 认证后三阶协同 top15（重训真值）===")
    t = df.sort_values("syn3_true", ascending=False).head(15)
    print(t[["fi", "fj", "fk", "conf", "v_ijk_true", "best_pair", "syn3_true", "syn3_est"]].to_string(index=False))

    # 估计 vs 认证 排序保真（top 组有 est 值的）
    tv = top.dropna(subset=["syn3_est"])
    if len(tv) > 5:
        rho = spearmanr(tv.syn3_est, tv.syn3_true).statistic
        print(f"\n估计 vs 认证 syn3 Spearman（top 组 {len(tv)} 条）= {rho:.3f}")
    print(f"\n认证 syn3_true>0.10 的条目: top 组 {(top.syn3_true>0.10).sum()}，"
          f"random 组 {(rand.syn3_true>0.10).sum()}")
    print(f"认证 syn3_true>0.20 的条目: top 组 {(top.syn3_true>0.20).sum()}")
    print(f"random 组 syn3 中位/最大: {rand.syn3_true.median():.3f} / {rand.syn3_true.max():.3f}")
    print(f"top 组 syn3 中位/最大: {top.syn3_true.median():.3f} / {top.syn3_true.max():.3f}")


if __name__ == "__main__":
    main()
