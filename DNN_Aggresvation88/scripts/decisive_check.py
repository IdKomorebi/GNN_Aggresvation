# -*- coding: utf-8 -*-
"""88 号 决定性检查：反层级集合在【超集】上还留不留信号？

三次搜索尝试均失败（beam 层级 / peel 极小性 / clique 必要条件剪枝），
失败模式一致。在换第四个方案前，必须先回答一个更根本的问题：

  v 是单调的，所以 S ⊆ T ⟹ v(T) ≥ v(S)。
  即反层级集合 S 虽然【子集】无信号，但它的【超集】原则上带着它的信号。
  ⟹ 若"含 S 的大集合"的 v 显著高于"不含 S 的同尺寸集合"，
     则 group-testing / 残差化探测有希望（信号存在，只是被混淆）；
  ⟹ 若两者无差别（v 在大集合上已饱和到 1），则信号被单调性抹平，
     **任何基于 v 查询的算法都无法定位 S** —— 这是不可能性论证，
     实践上只能转为随机抽样 + 统计遗漏上界。

对每个已知反层级强四阶 S 与其协同 conf c，测：
  含 S 的随机大集合（size K）的 v_c   vs   不含 S 的随机大集合的 v_c
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R86 = REPO / "DNN_Aggresvation86"
sys.path.insert(0, str(ROOT / "src"))
from query import Poly2Query  # noqa: E402
from runlog import log  # noqa: E402

KS = (8, 12, 16, 22)
N_REP = 60


def main() -> None:
    q = Poly2Query()
    n = q.n_general
    blind = pd.read_csv(ROOT / "outputs/blind_signal.csv")
    blind = blind[blind.group == "blind"].sort_values("syn4", ascending=False)
    log("CHECK", "START", note=f"{len(blind)} 个盲区强四阶")

    rng = np.random.RandomState(0)
    rows = []
    for _, r in blind.head(8).iterrows():
        S = tuple(int(x) for x in eval(r["indices"]))
        _, c = q.syn(S)
        rest = [i for i in range(n) if i not in S]
        for K in KS:
            if K < len(S):
                continue
            with_v, without_v = [], []
            for _ in range(N_REP):
                # 含 S：S + 随机补齐到 K
                extra = rng.choice(rest, K - len(S), replace=False)
                with_v.append(q.vhat(tuple(sorted(S + tuple(extra))))[c])
                # 不含 S：从 rest 里抽 K 个（保证不含 S 的任何元素组合完整）
                pick = rng.choice(rest, K, replace=False)
                without_v.append(q.vhat(tuple(sorted(pick)))[c])
            wv, ov = np.array(with_v), np.array(without_v)
            rows.append(dict(S=str(S), conf=q.conf_names[c], syn4=round(r["syn4"], 4), K=K,
                             v_with=round(float(wv.mean()), 4),
                             v_without=round(float(ov.mean()), 4),
                             gap=round(float(wv.mean() - ov.mean()), 4),
                             # 效应量：gap 相对不含组的标准差
                             cohen_d=round(float((wv.mean() - ov.mean()) / (ov.std() + 1e-9)), 3),
                             v_with_sat=round(float((wv > 0.98).mean()), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/superset_signal.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 30)
    print(df.to_string(index=False))

    print("\n=== 汇总：按大集合尺寸 K ===")
    g = df.groupby("K").agg(v_with=("v_with", "mean"), v_without=("v_without", "mean"),
                            gap=("gap", "mean"), cohen_d=("cohen_d", "mean"),
                            saturated=("v_with_sat", "mean"))
    print(g.round(4).to_string())
    print("\n=== 判定 ===")
    best = g["cohen_d"].abs().max()
    if best > 0.8:
        print(f"  超集仍带明显信号（最大 |d|={best:.2f}）⟹ 残差化/group-testing 有希望")
    elif best > 0.3:
        print(f"  超集信号微弱（最大 |d|={best:.2f}）⟹ 需大量重复才能分辨，代价可能超过枚举")
    else:
        print(f"  超集信号被单调性抹平（最大 |d|={best:.2f}）⟹ **不可能性**："
              f"任何基于 v 查询的确定性搜索都无法定位反层级集合；只能随机抽样+统计遗漏上界")


if __name__ == "__main__":
    main()
