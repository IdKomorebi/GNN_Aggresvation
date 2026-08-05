# -*- coding: utf-8 -*-
"""90 号 ★判定：用置换零分布把"真信号"与"字典带来的噪声"分开。

full 字典参数更多 ⟹ syn 分布整体上移（实测中位数也涨 1.3–1.5×），
所以不能直接说"高阶协同变多了"。置换检验给出干净判据：

  打乱 Y 的行（破坏 X–Y 关系、保留边际分布与 X 的相关结构）重跑同一流程，
  得到零分布 syn_null。则：
    · 经验 FDR ≈ #{null > τ} / #{real > τ}   —— 直接估计"发现里有多少是假的"
    · 用 null 的高分位（如 p99.9 / max）作阈值，可给出**近似无假阳**的强协同数
"""
from __future__ import annotations

import glob
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAUS = (0.05, 0.10, 0.15, 0.20, 0.30)


def load(order: int, kind: str, perm: bool):
    tag = "_perm7" if perm else ""
    fs = sorted(glob.glob(str(ROOT / f"outputs/rescan_o{order}_{kind}_audit{tag}_s*.parquet")))
    if not fs:
        return None
    return np.concatenate([pd.read_parquet(f)["syn"].to_numpy() for f in fs])


def main() -> None:
    rows = []
    for order in (3, 4, 5):
        for kind in ("poly2", "full"):
            real, null = load(order, kind, False), load(order, kind, True)
            if real is None or null is None:
                continue
            r = dict(order=order, kind=kind, n=len(real),
                     real_p50=round(float(np.percentile(real, 50)), 4),
                     null_p50=round(float(np.percentile(null, 50)), 4),
                     real_max=round(float(real.max()), 4),
                     null_max=round(float(null.max()), 4),
                     null_p999=round(float(np.percentile(null, 99.9)), 4))
            for t in TAUS:
                nr, nn = int((real > t).sum()), int((null > t).sum())
                r[f"real>{t}"] = nr
                r[f"null>{t}"] = nn
                r[f"fdr>{t}"] = round(nn / nr, 4) if nr else np.nan
            # 用 null 的 max 作保守阈值 ⟹ 近似 0 假阳的强协同数
            thr = float(null.max())
            r["thr_nullmax"] = round(thr, 4)
            r["n_above_nullmax"] = int((real > thr).sum())
            rows.append(r)
    df = pd.DataFrame(rows).sort_values(["order", "kind"])
    df.to_csv(ROOT / "outputs/perm_fdr.csv", index=False)
    pd.set_option("display.width", 320, "display.max_columns", 50)

    print("=== 真实 vs 置换零分布 ===")
    print(df[["order", "kind", "real_p50", "null_p50", "real_max", "null_max", "null_p999"]]
          .to_string(index=False))

    print("\n=== 各阈值下的经验 FDR（null计数/real计数，越小越可信）===")
    cols = ["order", "kind"] + [f"real>{t}" for t in TAUS] + [f"fdr>{t}" for t in TAUS]
    print(df[cols].to_string(index=False))

    print("\n=== ★ 保守判定：超过 null 最大值的强协同数（近似 0 假阳）===")
    for _, r in df.iterrows():
        print(f"  order-{int(r.order)} {r['kind']:6s}: null_max={r.thr_nullmax:.4f}  "
              f"⟹ 真实中超过它的有 {int(r.n_above_nullmax)} 个 "
              f"(占空间 {r.n_above_nullmax/comb(44,int(r.order))*100:.4f}%)")

    print("\n=== ★ 结论：87 号衰减律在扣除噪声后是否还成立 ===")
    for kind in ("poly2", "full"):
        d = df[df.kind == kind].sort_values("order")
        if len(d) == 0:
            continue
        seq = " → ".join(f"o{int(r.order)}:{int(r.n_above_nullmax)}" for _, r in d.iterrows())
        dens = " → ".join(f"{r.n_above_nullmax/comb(44,int(r.order))*100:.4f}%" for _, r in d.iterrows())
        print(f"  {kind:6s} 扣噪声后强协同数 {seq}")
        print(f"  {kind:6s} 扣噪声后密度     {dens}")


if __name__ == "__main__":
    main()
