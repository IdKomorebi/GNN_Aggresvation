# -*- coding: utf-8 -*-
"""86 号认证分析：结构化 syn4 vs DNN 重训 syn4_true。

syn4_true = max_c [ v_quad(c) − max_{4个三元子集} v_triple(c) ]，v 为 DNN 重训 per-conf R²。
回答：结构化查询说强的四阶，在真正 DNN 攻击者下是否真强？结构化是否忠实下界？
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RET = ROOT / "outputs/retrain"


def load_r2(sid: str) -> dict:
    p = RET / f"{sid}_dnn_seed0.json"
    return json.load(open(p))["per_conf_r2"] if p.exists() else None


def main() -> None:
    qm = json.load(open(ROOT / "outputs/quad_map.json"))
    confs = None
    rows = []
    for qid, info in qm.items():
        vq = load_r2(qid)
        vts = [load_r2(t) for t in info["triples"]]
        if vq is None or any(v is None for v in vts):
            continue
        if confs is None:
            confs = list(vq)
        vq_a = np.array([vq[c] for c in confs])
        vts_a = np.stack([[vt[c] for c in confs] for vt in vts])   # (4, nC)
        inc = vq_a - vts_a.max(axis=0)                             # (nC,)
        c = int(np.argmax(inc))
        rows.append(dict(qid=qid, tag=info["tag"], syn4_struct=info["syn4_struct"],
                         syn4_true=float(inc.max()), conf=confs[c],
                         vquad_true=float(vq_a.max())))
    df = pd.DataFrame(rows).sort_values("syn4_struct", ascending=False)
    df.to_csv(ROOT / "outputs/cert_syn4.csv", index=False)

    top = df[df.tag == "top"]
    rnd = df[df.tag == "rand"]
    print(f"认证完成 {len(df)}/40 四元组")
    print(f"\n结构化-真值 相关: Spearman={spearmanr(df.syn4_struct, df.syn4_true).correlation:.3f} "
          f"Pearson={df.syn4_struct.corr(df.syn4_true):.3f}")
    print(f"\ntop20(结构化最强): syn4_struct 均值={top.syn4_struct.mean():.3f} "
          f"→ syn4_true 均值={top.syn4_true.mean():.3f} 中位={top.syn4_true.median():.3f} "
          f"max={top.syn4_true.max():.3f}")
    print(f"  真值 syn4_true>0.10 的: {(top.syn4_true>0.10).sum()}/{len(top)}；"
          f">0.05: {(top.syn4_true>0.05).sum()}/{len(top)}")
    print(f"rand20(参照): syn4_true 均值={rnd.syn4_true.mean():.3f} 中位={rnd.syn4_true.median():.3f}")
    print(f"\n结构化是否忠实下界(syn4_true≥syn4_struct? 因结构化只是一种攻击族):")
    print(f"  syn4_true ≥ syn4_struct 的比例: {(df.syn4_true >= df.syn4_struct - 0.02).mean():.0%}")
    print("\n最强5个:")
    print(top.head(5)[["qid", "conf", "syn4_struct", "syn4_true", "vquad_true"]].to_string(index=False))


if __name__ == "__main__":
    main()
