# -*- coding: utf-8 -*-
"""DNN75 Phase 0：把项目里**全部**可用的重训真值汇总成单一评测集。

为什么要重建评测集
------------------
73/74 用的 394 子集里 89% 是 |S|<=3，中大带每带只有 10 个子集。本子项目完全是关于
尺寸分配的，那个评测集分辨不出"小集合变好、大集合变差"的权衡。

来源（全部 dnn 重训口径，数据切分 seed 42）：
  69 truth_long.csv       1437 = 44 单 + 946 对 + 397 三元组 + 50 wide_random
  68 retrain/bt*.json       85 中段真值（尺寸 5-25）—— 73/74 **从未使用**
  75 retrain/lg*.json       60 大集合真值（尺寸 17-43）—— 本子项目新增

按"字段集合"去重（69 号已知有 6 个 wide_random 与穷举集合重合，用 canonical 保留一份）。

输出：
  outputs/evalset.json   {sid: {group, size, fields}}
  outputs/truth_all.csv  sid, group, size, band, conf, truth
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R75 = Path(__file__).resolve().parents[1]
R69 = R75.parent / "DNN_Aggresvation69"
R68 = R75.parent / "DNN_Aggresvation68"
BANDS = [(1, 2, "1-2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 16, "9-16"),
         (17, 32, "17-32"), (33, 44, "33-44")]


def band(n):
    for lo, hi, name in BANDS:
        if lo <= n <= hi:
            return name
    return "?"


def main():
    reg69 = json.load(open(R69 / "outputs/subsets.json"))
    tl = pd.read_csv(R69 / "outputs/truth_long.csv")
    # 69 号自带 canonical_sid 去重标记；只取非重复条目的 dnn 真值
    tl = tl[~tl.is_duplicate_entry] if "is_duplicate_entry" in tl.columns else tl

    evalset, rows = {}, []
    seen_fields = {}          # frozenset(fields) -> sid，跨来源去重

    def add(sid, group, fields, per_conf):
        key = frozenset(fields)
        if key in seen_fields:
            return False
        seen_fields[key] = sid
        evalset[sid] = {"group": group, "size": len(fields), "fields": sorted(fields)}
        for c, v in per_conf.items():
            rows.append(dict(sid=sid, group=group, size=len(fields),
                             band=band(len(fields)), conf=c, truth=float(v)))
        return True

    # ---- 来源 1：69 号 truth_long（1437） ----
    n1 = 0
    for sid, g in tl.groupby("sid"):
        if sid not in reg69:
            continue
        per_conf = dict(zip(g.conf, g.dnn))
        n1 += add(sid, reg69[sid]["group"], reg69[sid]["fields"], per_conf)

    # ---- 来源 2：68 号 85 个中段真值（fields 在 json 内，registry 已被三元组覆盖） ----
    n2 = 0
    for p in sorted((R68 / "outputs/retrain").glob("bt*_dnn_seed0.json")):
        d = json.load(open(p))
        n2 += add(d["subset_id"], "band_truth", d["fields"], d["per_conf_r2"])

    # ---- 来源 3：75 号新增大集合真值 ----
    n3 = 0
    for p in sorted((R75 / "outputs/retrain").glob("lg*_dnn_seed0.json")):
        d = json.load(open(p))
        n3 += add(d["subset_id"], "large_random", d["fields"], d["per_conf_r2"])

    (R75 / "outputs/evalset.json").write_text(json.dumps(evalset, indent=1, ensure_ascii=False))
    df = pd.DataFrame(rows)
    df.to_csv(R75 / "outputs/truth_all.csv", index=False)

    print(f"=== 评测集汇总（去重后 {len(evalset)} 个子集，{len(df)} 条 (子集,conf) 真值）===")
    print(f"  69号 truth_long : {n1:>4}")
    print(f"  68号 中段 bt*   : {n2:>4}   ← 73/74 从未使用")
    print(f"  75号 大集合 lg* : {n3:>4}   ← 本子项目新增")
    sz = pd.Series({s: v["size"] for s, v in evalset.items()})
    print(f"\n{'规模带':>8}{'子集数':>8}   （73/74 旧评测集对照）")
    old = {"1-2": 250, "3-4": 104, "5-8": 10, "9-16": 10, "17-32": 10, "33-44": 10}
    for lo, hi, name in BANDS:
        n = int(((sz >= lo) & (sz <= hi)).sum())
        print(f"{name:>8}{n:>8}   （旧 {old[name]}）")

    # 与 69 号旧评测集的一致性：旧 394 必须是新集合的子集，且真值逐值相等
    sys.path.insert(0, str(R75.parent / "DNN_Aggresvation74/scripts"))
    rng = np.random.RandomState(0)
    byg = {}
    for sid, m in reg69.items():
        byg.setdefault(m["group"], []).append(sid)
    sel = sorted(byg["single"]) + sorted(byg["wide_random"])
    sel += list(rng.choice(sorted(byg["pair"]), 200, replace=False))
    tri = sorted(byg.get("triple_top", [])) + sorted(byg.get("triple_rand", []))
    sel += list(rng.choice(tri, 100, replace=False))
    # 69 号有 6 个 wide_random 与穷举集合是同一字段集合（rs00→s22 等），
    # 按字段去重时只保留了 canonical 一份；这里把旧 sid 映射过去，
    # 保证复现闸门比的是同一批【字段集合】。
    tl_all = pd.read_csv(R69 / "outputs/truth_long.csv")
    canon = dict(zip(tl_all.sid, tl_all.canonical_sid)) if "canonical_sid" in tl_all.columns else {}
    old394 = sorted({canon.get(s, s) for s in set(sel)})
    missing = [s for s in old394 if s not in evalset]
    print(f"\n[{'OK ' if not missing else 'FAIL'}] 旧 394 子集全部在新评测集内"
          f"（去重后 {len(old394)} 个字段集合，缺 {len(missing)} 个）")
    (R75 / "outputs/old394.json").write_text(json.dumps(old394))
    print(f"  已写出 outputs/old394.json（V1 复现闸门用）")


if __name__ == "__main__":
    main()
