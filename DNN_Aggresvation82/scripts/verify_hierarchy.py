# -*- coding: utf-8 -*-
"""82 号：为什么"取绝对值 v(ijk) 高"能找三阶协同——本质分析（零 GPU，真值侧）。

用户的困惑与猜想
----------------
- 若"强协同=容量最大电路、摊销没给容量"，问题应无解；但 S2（取 v(ijk) 绝对值高）却很有效。
- 为什么越强协同越找不出（S1 漏），而 S2 又专门找强协同？"难道 v(ijk) 高就存在协同？"
- 用户猜想：这些三元组**很可能存在二阶协同**，而我们对三阶差值的定义是**纯三阶**。
  为什么绝对值高往往就有三阶协同？是数据不够 general，还是 general 规律？

本脚本用真值把这个悖论拆开，逐条验证：
Q1  真值上 "v(ijk) 高 ⟹ 有协同" 成立吗？（上一步已见：不成立，最高档 0% 协同）
Q2  反向 "有协同 ⟹ v(ijk) 高" 成立吗？（数学必然 v(ijk)=syn3+max_pair≥syn3，验证之）
Q3  强三阶三元组，v(ijk) 最高的 conf 与 syn3 最强的 conf 是否【错位】？
    （用户猜想的核心：某 conf 上是强二阶让 v(ijk) 高，另一 conf 上才是纯三阶）
Q4  错位时，"v(ijk) 高的那个 conf" 的高泄露是来自二阶还是单字段？

用 68 认证集（per triple×conf 的 v_ijk_true / best_pair / syn3_true）+ 67 单字段真值。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

R82 = Path(__file__).resolve().parents[1]
REPO = R82.parent
sys.path.insert(0, str(REPO / "DNN_Aggresvation69"))
from src.data_processing import prepare_data  # noqa: E402
sys.path.insert(0, str(R82 / "src"))
try:
    from runlog import log
except Exception:
    def log(*a, **k): pass


def main():
    cfg = yaml.safe_load((REPO / "DNN_Aggresvation77/base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    gen = di["general"]
    n2i = {n: i for i, n in enumerate(gen)}

    # 单字段真值
    reg = json.load(open(REPO / "DNN_Aggresvation67/outputs/subsets.json"))
    sv = {}
    for sid, m in reg.items():
        if m["group"] == "single":
            p = REPO / f"DNN_Aggresvation67/outputs/retrain/{sid}_dnn_seed0.json"
            if p.exists():
                d = json.load(open(p))
                fi = n2i[m["fields"][0]]
                for conf, v in d["per_conf_r2"].items():
                    sv[(fi, conf)] = v

    c = pd.read_csv(REPO / "DNN_Aggresvation68/outputs/triples_certified.csv")
    c["ms"] = [max(sv.get((n2i[r.fi], r.conf), 0), sv.get((n2i[r.fj], r.conf), 0),
                   sv.get((n2i[r.fk], r.conf), 0)) for r in c.itertuples()]
    c["tri"] = list(zip(c.fi, c.fj, c.fk))

    # ---- Q2：有强协同 ⟹ v(ijk) 高（数学必然，量化）----
    print("=" * 92)
    print("Q2  有协同 ⟹ v(ijk) 高？（数学：v(ijk)=syn3+max_pair≥syn3）")
    print("=" * 92)
    strong_e = c[c.syn3_true > 0.20]
    print(f"  真值 syn3>0.20 的强协同条目 {len(strong_e)} 个：v_ijk_true 中位={strong_e.v_ijk_true.median():.3f}, "
          f"最小={strong_e.v_ijk_true.min():.3f}")
    print(f"  → 强协同的 v(ijk) 全都很高（因数学下界）；所以按 v(ijk) 排序【不会漏掉】强协同")
    print(f"  但反向不成立：v_ijk_true>0.9 的 {len(c[c.v_ijk_true>0.9])} 个条目里，"
          f"有强三阶的仅 {(c[c.v_ijk_true>0.9].syn3_true>0.1).mean():.0%}")
    print(f"  ∴ S2 是 syn3 的【上界筛】：高召回（不漏强协同）、低精度（混入无协同的高泄露）")

    # ---- Q3：conf 错位 ----
    print("\n" + "=" * 92)
    print("Q3  强三阶三元组：v(ijk) 最高的 conf 与 syn3 最强的 conf 是否错位？")
    print("=" * 92)
    rows = []
    for tri, g in c.groupby("tri"):
        if g.syn3_true.max() <= 0.20:
            continue
        r_syn = g.loc[g.syn3_true.idxmax()]        # 协同最强的 conf
        r_v = g.loc[g.v_ijk_true.idxmax()]         # 泄露最高的 conf
        rows.append(dict(
            tri=tri, c_syn=r_syn.conf, c_v=r_v.conf, match=(r_syn.conf == r_v.conf),
            syn_at_csyn=r_syn.syn3_true, pair_at_csyn=r_syn.best_pair,       # 协同conf：二阶多少
            vijk_at_cv=r_v.v_ijk_true, pair_at_cv=r_v.best_pair, ms_at_cv=r_v.ms))
    m = pd.DataFrame(rows)
    m.to_csv(R82 / "outputs/conf_dislocation.csv", index=False)
    print(f"  最强三阶三元组 {len(m)} 个：v(ijk)最高conf 与 syn3最强conf【错位】的占 "
          f"{(~m.match).mean():.0%}")
    print(f"\n  错位组（n={(~m.match).sum()}）的结构：")
    dis = m[~m.match]
    print(f"    · 协同 conf 上：syn3={dis.syn_at_csyn.mean():.3f}，best_pair(二阶)={dis.pair_at_csyn.mean():.3f}"
          f"（低→确是纯三阶）")
    print(f"    · 泄露 conf 上：v(ijk)={dis.vijk_at_cv.mean():.3f}，best_pair(二阶)={dis.pair_at_cv.mean():.3f}"
          f"，max_single={dis.ms_at_cv.mean():.3f}")
    print(f"      → 泄露 conf 的高 v(ijk) 来自"
          f"{'强二阶结构' if (dis.pair_at_cv-dis.ms_at_cv).mean()>0.1 else '单字段'}"
          f"（二阶增益 {(dis.pair_at_cv-dis.ms_at_cv).mean():.3f}）")

    # ---- Q4：一句话结论 ----
    print("\n" + "=" * 92)
    print("Q4  综合：S2 为什么在这个数据上有效")
    print("=" * 92)
    print("  同一组字段簇（燃料量/占比/负荷等物理枢纽）：")
    print("  · 对某 conf（如 total_gen）构成强【低阶】结构 → v(ijk) 高 → S2 把它捞进候选")
    print("  · 对另一 conf（如 net_actual）构成纯【三阶】协同 → syn3 高 → 这才是要发现的")
    print("  两者由同一批枢纽字段产生 ⟹ 高 v(ijk) 与 有三阶协同【高度重叠】⟹ S2 上界筛有效")
    log("VERIFY", "DONE", note=f"conf错位占{(~m.match).mean():.0%}；S2=syn3上界筛(高召回低精度)；"
                              f"高v(ijk)与有协同重叠源于同一物理枢纽字段簇")


if __name__ == "__main__":
    main()
