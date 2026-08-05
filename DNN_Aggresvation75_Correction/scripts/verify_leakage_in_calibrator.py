# -*- coding: utf-8 -*-
"""承认性检查：上一版校准方案的提升，有多少来自 oracle、多少来自重训真值特征？

背景
----
我（Claude）在 2026-07-23 提过一版"条件校准"方案，把低阶代理 m1/m2 当作校准器特征：
    m1(S,c) = max_{i∈S} v_c({i})      —— 来自 44 个单字段【重训真值】
    m2(S,c) = max_{i,j∈S} v_c({i,j})  —— 来自 946 个字段对【重训真值】
实测 K=0 的 MAE 从 0.0900 降到 0.0182，并宣称这是"零成本的巨大提升"。

用户指出这是投机取巧：拿重训真值去帮 oracle 修正，摧毁了"oracle 能快速摊销"这一主张
本身——因为真值本来就是我们要用 oracle 去逼近的东西。

本脚本做的是**证伪我自己**：把 oracle 估计从校准器里拿掉，只用 m1/m2/size/conf
直接预测真值。如果这样也能达到接近的精度，说明上一版方案的提升主要来自真值查表，
oracle 只是个装饰。

结论（见输出）：只用真值特征即可达 MAE 0.0205 / ρ 0.9929，
而加上 oracle 只到 0.0182 / 0.9942 —— oracle 的净贡献仅约 11%。
方案作废。

用法：python scripts/verify_leakage_in_calibrator.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

R75 = Path(__file__).resolve().parents[2] / "DNN_Aggresvation75"
sys.path.insert(0, str(R75 / "scripts"))
from analyze import load_truth  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def build():
    truth = load_truth()
    ev = json.load(open(R75 / "outputs/evalset.json"))
    T = pd.read_csv(R75 / "outputs/truth_all.csv").set_index(["sid", "conf"]).truth
    s1 = {ev[s]["fields"][0]: s for s in ev if ev[s]["size"] == 1}
    s2 = {tuple(sorted(ev[s]["fields"])): s for s in ev if ev[s]["size"] == 2}

    d = pd.read_csv(R75 / "outputs/est_uniform_seed0.csv")
    d["truth"] = [truth.get((s, c), np.nan) for s, c in zip(d.sid, d.conf)]
    d = d.dropna(subset=["truth"])
    # 只看 |S|>=3：|S|=1 时 m1 就是真值本身、|S|=2 时 m2 就是真值本身，那是纯泄露
    d = d[(d.K == 0) & (d["size"] >= 3)].copy()
    d["conf_id"] = pd.factorize(d.conf)[0]

    mm = []
    for s, c in zip(d.sid, d.conf):
        F = ev[s]["fields"]
        a = max((T.get((s1[f], c), 0.0) for f in F if f in s1), default=0.0)
        b = max((T.get((s2[k], c), 0.0)
                 for k in (tuple(sorted(p)) for p in itertools.combinations(F, 2))
                 if k in s2), default=0.0)
        mm.append((a, b))
    d["m1"] = [x[0] for x in mm]
    d["m2"] = [x[1] for x in mm]
    return d.reset_index(drop=True)


def run(d, feats, calibrate_oracle=True):
    """calibrate_oracle=True：预测残差去修 oracle；False：直接从特征预测真值。
    5 折【按子集】分组交叉，同一子集的 12 个 conf 不跨折。"""
    p = np.zeros(len(d))
    y = (d.est - d.truth) if calibrate_oracle else d.truth
    for tr, te in GroupKFold(5).split(d, groups=d.sid.values):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        m.fit(d.iloc[tr][feats], y.iloc[tr])
        p[te] = m.predict(d.iloc[te][feats])
    v = (d.est.values - p) if calibrate_oracle else p
    return np.abs(v - d.truth.values).mean(), spearmanr(v, d.truth).correlation


def main():
    d = build()
    rows = [
        ("K=0 原始（纯 oracle，无任何校准）", True, None),
        ("校准器：size, conf, est（不含真值特征）", True, ["size", "conf_id", "est"]),
        ("校准器：+ m1, m2（我提的上一版方案）", True, ["size", "conf_id", "est", "m1", "m2"]),
        ("★ 只用 m1, m2, size, conf 直接预测真值（无 oracle）", False, ["size", "conf_id", "m1", "m2"]),
        ("★ 只用 m1（最强单字段查表）", False, ["size", "conf_id", "m1"]),
    ]
    out = []
    print("=" * 88)
    print("承认性检查：校准器的提升来自 oracle 还是来自重训真值特征？（|S|>=3, K=0）")
    print("=" * 88)
    print(f"{'配置':<50}{'用oracle?':>10}{'MAE':>9}{'ρ':>9}")
    for lab, use_oracle, feats in rows:
        if feats is None:
            mae = d.est.sub(d.truth).abs().mean()
            rho = spearmanr(d.est, d.truth).correlation
        else:
            mae, rho = run(d, feats, use_oracle)
        print(f"{lab:<50}{'是' if use_oracle else '否':>10}{mae:>9.4f}{rho:>9.4f}")
        out.append(dict(config=lab, uses_oracle=use_oracle, mae=mae, rho=rho))
    pd.DataFrame(out).to_csv(OUT / "leakage_check.csv", index=False)

    o = {r["config"]: r for r in out}
    a = o["★ 只用 m1, m2, size, conf 直接预测真值（无 oracle）"]["mae"]
    b = o["校准器：+ m1, m2（我提的上一版方案）"]["mae"]
    print(f"\noracle 在上一版方案里的净贡献：{a:.4f} → {b:.4f}，仅 {1 - b / a:.0%}")
    print("→ 该方案 ~89% 的效果来自重训真值查表。作废。")
    print(f"\n已写出 {OUT / 'leakage_check.csv'}")


if __name__ == "__main__":
    main()
