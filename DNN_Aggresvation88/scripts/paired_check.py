# -*- coding: utf-8 -*-
"""88 号 决定性检查（配对版，修正上一版的设计缺陷）。

上一版对比"S+随机(K−4)" vs "随机K"，两组背景分布不同（反层级集合本身由低泄露
字段构成），导致 gap 为负、不可解释。本版做**配对**比较：

  固定同一背景 B（随机 K−4 个字段，不含 S 的元素）：
      Δ_S  = v_c(B ∪ S) − v_c(B ∪ R)     （R = 另取的随机 4 个字段）
      marg = v_c(B ∪ S) − v_c(B)          （S 在背景 B 上的边际贡献）

判定：
  · 若 Δ_S 显著 > 0 ⟹ 反层级集合在大集合中仍可分辨 ⟹ group-testing/残差化可行；
  · 若 Δ_S ≈ 0     ⟹ 背景已吸收该信息（单调饱和）⟹ 基于 v 的确定性定位不可能，
                      只能随机抽样 + 统计遗漏上界。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from query import Poly2Query  # noqa: E402
from runlog import log  # noqa: E402

KS = (8, 12, 16, 22)
N_REP = 50


def main() -> None:
    q = Poly2Query()
    n = q.n_general
    blind = pd.read_csv(ROOT / "outputs/blind_signal.csv")
    blind = blind[blind.group == "blind"].sort_values("syn4", ascending=False)

    rng = np.random.RandomState(1)
    rows = []
    for _, r in blind.head(8).iterrows():
        S = tuple(int(x) for x in eval(r["indices"]))
        _, c = q.syn(S)
        rest = np.array([i for i in range(n) if i not in S])
        for K in KS:
            d_list, m_list = [], []
            for _ in range(N_REP):
                B = tuple(sorted(rng.choice(rest, K - len(S), replace=False)))
                pool = [i for i in rest if i not in B]
                R = tuple(sorted(rng.choice(pool, len(S), replace=False)))
                vBS = q.vhat(tuple(sorted(B + S)))[c]
                vBR = q.vhat(tuple(sorted(B + R)))[c]
                vB = q.vhat(B)[c]
                d_list.append(vBS - vBR)
                m_list.append(vBS - vB)
            d, mg = np.array(d_list), np.array(m_list)
            rows.append(dict(S=str(S), conf=q.conf_names[c], syn4=round(r["syn4"], 4), K=K,
                             delta=round(float(d.mean()), 4), delta_sd=round(float(d.std()), 4),
                             cohen_d=round(float(d.mean() / (d.std() + 1e-9)), 3),
                             marginal=round(float(mg.mean()), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/paired_signal.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 30)

    print("=== 汇总：按大集合尺寸 K（配对）===")
    g = df.groupby("K").agg(delta=("delta", "mean"), cohen_d=("cohen_d", "mean"),
                            marginal=("marginal", "mean"))
    print(g.round(4).to_string())
    print(f"\n（参考：这些 S 单独的 syn4 均值 = {df.groupby('S').syn4.first().mean():.4f}）")
    print("\n=== 逐 S（K=12）===")
    print(df[df.K == 12].to_string(index=False))

    best = g["cohen_d"].abs().max()
    print("\n=== 判定 ===")
    if best > 0.8:
        print(f"  大集合中仍可分辨（最大 |d|={best:.2f}）⟹ group-testing / 残差化值得做")
    elif best > 0.3:
        print(f"  信号微弱（最大 |d|={best:.2f}）⟹ 需大量重复，代价可能超过直接枚举")
    else:
        print(f"  信号被吸收（最大 |d|={best:.2f}）⟹ **不可能性**：基于 v 查询的确定性定位不可行，"
              f"实践上只能随机抽样 + 统计遗漏上界")


if __name__ == "__main__":
    main()
