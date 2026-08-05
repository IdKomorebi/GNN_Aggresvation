# -*- coding: utf-8 -*-
"""90 号 汇总：full vs poly2 字典下的 syn 分布，并排除"只是噪声变大"的混淆。

关键鉴别：full 字典参数更多，若 syn 的增大只是估计噪声，则**整个分布**都会上移；
若是真信号，则**尾部**显著上移而中位数/中低分位基本不变。
故同时报中位数、75%、95%、99%、max 与 >τ 计数。
"""
from __future__ import annotations

import glob
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QS = [50, 75, 90, 95, 99, 99.9]
TAUS = (0.05, 0.10, 0.15, 0.20)


def load(order: int, kind: str, ev: str = "audit") -> np.ndarray:
    fs = sorted(glob.glob(str(ROOT / f"outputs/rescan_o{order}_{kind}_{ev}_s*.parquet")))
    if not fs:
        return None
    return np.concatenate([pd.read_parquet(f)["syn"].to_numpy() for f in fs])


def main() -> None:
    rows = []
    for order in (3, 4, 5):
        for kind in ("poly2", "full"):
            s = load(order, kind)
            if s is None:
                continue
            r = dict(order=order, kind=kind, n=len(s), full_space=comb(44, order),
                     max_syn=round(float(s.max()), 4))
            for q in QS:
                r[f"p{q}"] = round(float(np.percentile(s, q)), 4)
            for t in TAUS:
                r[f"n>{t}"] = int((s > t).sum())
                r[f"dens>{t}"] = round(float((s > t).mean()), 6)
            rows.append(r)
    df = pd.DataFrame(rows).sort_values(["order", "kind"])
    df.to_csv(ROOT / "outputs/rescan_summary.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)

    print("=== 分布分位数（鉴别：噪声会整体上移，真信号只抬尾部）===")
    print(df[["order", "kind", "n", "p50", "p75", "p90", "p95", "p99", "p99.9", "max_syn"]]
          .to_string(index=False))

    print("\n=== 强协同计数与密度 ===")
    print(df[["order", "kind", "n>0.05", "n>0.1", "n>0.15", "n>0.2",
              "dens>0.05", "dens>0.1"]].to_string(index=False))

    print("\n=== ★ 字典效应（full / poly2）===")
    for order in (3, 4, 5):
        a = df[(df.order == order) & (df.kind == "poly2")]
        b = df[(df.order == order) & (df.kind == "full")]
        if len(a) == 0 or len(b) == 0:
            continue
        a, b = a.iloc[0], b.iloc[0]
        print(f"  order-{order}: 中位 {a.p50:.4f}→{b.p50:.4f} (×{b.p50/max(a.p50,1e-9):.2f})   "
              f"p99 {a['p99']:.4f}→{b['p99']:.4f} (×{b['p99']/max(a['p99'],1e-9):.2f})   "
              f"max {a.max_syn:.4f}→{b.max_syn:.4f}   "
              f"n>0.10 {int(a['n>0.1'])}→{int(b['n>0.1'])} (×{b['n>0.1']/max(a['n>0.1'],1):.1f})")

    print("\n=== ★ 87 号'衰减律'在两种字典下的对照（密度 syn>0.10）===")
    for kind in ("poly2", "full"):
        d = df[df.kind == kind].sort_values("order")
        if len(d) == 0:
            continue
        seq = " → ".join(f"o{int(r.order)}:{r['dens>0.1']*100:.3f}%" for _, r in d.iterrows())
        mx = " → ".join(f"{r.max_syn:.3f}" for _, r in d.iterrows())
        drop = d.iloc[0]['dens>0.1'] / max(d.iloc[-1]['dens>0.1'], 1e-12)
        print(f"  {kind:6s} 密度 {seq}   (3阶/5阶 = {drop:.0f}×)")
        print(f"  {kind:6s} max_syn {mx}")


if __name__ == "__main__":
    main()
