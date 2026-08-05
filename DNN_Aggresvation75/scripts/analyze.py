# -*- coding: utf-8 -*-
"""DNN75：采样测度方案的汇总分析与判据。

主指标：K=0 对扩充评测集（~1576 子集）重训真值的 MAE，分三个区间：
    扫描区间 |S|<=3   （协同全扫，14234 次查询）
    中段     4..12    （几乎不查，但"任意集合"主张的一部分）
    防护区间 |S|>=13  （防护集贪心，~900 次查询）
副指标：二阶协同检出（44 单 + 946 对 -> syn = v̂(ij) − max(v̂(i), v̂(j))，
        对 68 号 synergy2_perconf.csv 的重训真值）。

子命令：
    --check-protocol   口径自检
    --check-samplers   采样器尺寸分布自检（V2）
    --check-repro      V1 复现闸门（uniform vs 74 号 MLP，在旧 388 字段集合上）
    --round main       主结果表 + 判据
    --round routing    路由双模型后处理（零 GPU）
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

R75 = Path(__file__).resolve().parents[1]
R69 = R75.parent / "DNN_Aggresvation69"
R68 = R75.parent / "DNN_Aggresvation68"
OUT = R75 / "outputs"

BANDS = [(1, 2, "1-2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 16, "9-16"),
         (17, 32, "17-32"), (33, 44, "33-44")]
REGIONS = [(1, 3, "扫描|S|<=3"), (4, 12, "中段4-12"), (13, 44, "防护|S|>=13")]

# 74 号 MLP（= uniform 方案）在旧 394/388 子集上的实测，用作 V1 闸门
REF74 = dict(K0_mae=0.0595, K0_rho=0.9693, r2_full=0.8336)
REPRO_TOL = dict(K0_mae=0.008, K0_rho=0.010, r2_full=0.010)

# 预注册判据
GATE = dict(scan_rel_drop=0.10, protect_abs_degrade=0.010,
            syn_rho_gain=0.03, syn_top20_gain=0.10)


def band(n):
    for lo, hi, nm in BANDS:
        if lo <= n <= hi:
            return nm
    return "?"


def region(n):
    for lo, hi, nm in REGIONS:
        if lo <= n <= hi:
            return nm
    return "?"


def load_truth():
    t = pd.read_csv(OUT / "truth_all.csv")
    return t.set_index(["sid", "conf"]).truth.to_dict()


def load_est(scheme, seed, truth):
    p = OUT / f"est_{scheme}_seed{seed}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["truth"] = [truth.get((s, c), np.nan) for s, c in zip(d.sid, d.conf)]
    d = d.dropna(subset=["truth"])
    d["err"] = d.est - d.truth
    d["band"] = d["size"].map(band)
    d["region"] = d["size"].map(region)
    return d


def schemes_present():
    return sorted({p.stem.replace("est_", "").rsplit("_seed", 1)[0]
                   for p in OUT.glob("est_*_seed*.csv")})


def seeds_of(scheme):
    return sorted(int(p.stem.split("seed")[-1]) for p in OUT.glob(f"est_{scheme}_seed*.csv"))


# ------------------------------------------------------------------ #
def synergy_metrics(d0: pd.DataFrame) -> dict:
    """用 K=0 的估计算二阶协同，对 68 号重训真值 syn2 评检出质量。"""
    syn2 = pd.read_csv(R68 / "outputs/synergy2_perconf.csv")   # fi,fj,conf,synergy(真值)
    evalset = json.load(open(OUT / "evalset.json"))
    # sid -> fields（单字段与字段对）
    f1, f2 = {}, {}
    for sid, m in evalset.items():
        if m["size"] == 1:
            f1[m["fields"][0]] = sid
        elif m["size"] == 2:
            f2[tuple(sorted(m["fields"]))] = sid
    est = d0.set_index(["sid", "conf"]).est.to_dict()

    rows = []
    for r in syn2.itertuples():
        key = tuple(sorted((r.fi, r.fj)))
        if key not in f2 or r.fi not in f1 or r.fj not in f1:
            continue
        vij = est.get((f2[key], r.conf))
        vi = est.get((f1[r.fi], r.conf))
        vj = est.get((f1[r.fj], r.conf))
        if vij is None or vi is None or vj is None:
            continue
        rows.append((r.synergy, vij - max(vi, vj)))
    if not rows:
        return {}
    t = np.array([x[0] for x in rows])
    e = np.array([x[1] for x in rows])
    order_t = np.argsort(-t)
    rank_e = pd.Series(-e).rank().values          # 1 = 估计最强
    out = dict(syn_n=len(t), syn_rho=spearmanr(t, e).correlation)
    for N in (20, 50):
        top_t = set(order_t[:N])
        top_e = set(np.argsort(-e)[:N])
        out[f"syn_top{N}"] = len(top_t & top_e) / N
    out["syn_top10_medrank"] = float(np.median(rank_e[order_t[:10]]))
    out["syn_fp_rate"] = float((e[t <= 0.02] > 0.1).mean())    # 真无协同却报 >0.1
    return out


def collect(schemes) -> pd.DataFrame:
    assert_evalset_consistent()
    truth = load_truth()
    rows = []
    for sc in schemes:
        for sd in seeds_of(sc):
            d = load_est(sc, sd, truth)
            if d is None or d.empty:
                continue
            ck = OUT / f"oracle_{sc}_seed{sd}.pt"
            meta = torch.load(ck, map_location="cpu", weights_only=False) if ck.exists() else {}
            r = dict(scheme=sc, seed=sd, r2_full=meta.get("r2_full", np.nan),
                     epochs_run=meta.get("epochs_run", np.nan))
            d0 = d[d.K == 0]
            for K in (0, 10, 50, 200):
                dk = d[d.K == K]
                if len(dk):
                    r[f"K{K}_mae"] = dk.err.abs().mean()
                    r[f"K{K}_rho"] = spearmanr(dk.est, dk.truth).correlation
            for _, _, nm in REGIONS:
                sub = d0[d0.region == nm]
                if len(sub):
                    r[f"mae_{nm}"] = sub.err.abs().mean()
            for _, _, nm in BANDS:
                sub = d0[d0.band == nm]
                if len(sub):
                    r[f"band_{nm}"] = sub.err.abs().mean()
            r.update(synergy_metrics(d0))
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
def cmd_check_protocol():
    ok = True
    print("=" * 78 + "\nV3 口径自检\n" + "=" * 78)
    r = subprocess.run(["diff", str(R69 / "base.yaml"), str(R75 / "base.yaml")],
                       capture_output=True, text=True)
    print(f"[{'OK ' if r.returncode == 0 else 'FAIL'}] base.yaml 与 69 号一致")
    ok &= r.returncode == 0

    ev = json.load(open(OUT / "evalset.json"))
    old = json.load(open(OUT / "old394.json"))
    miss = [s for s in old if s not in ev]
    print(f"[{'OK ' if not miss else 'FAIL'}] 旧 {len(old)} 个字段集合都在新评测集内")
    ok &= not miss

    # 旧子集的真值必须与 69 号 truth_long 逐值相等
    t75 = pd.read_csv(OUT / "truth_all.csv").set_index(["sid", "conf"]).truth
    t69 = pd.read_csv(R69 / "outputs/truth_long.csv")
    t69 = t69[~t69.is_duplicate_entry].set_index(["sid", "conf"]).dnn
    common = t75.index.intersection(t69.index)
    dmax = float((t75.loc[common] - t69.loc[common]).abs().max())
    print(f"[{'OK ' if dmax < 1e-9 else 'FAIL'}] {len(common)} 条共有真值与 69 号逐值相等"
          f"（最大差 {dmax:.2e}）")
    ok &= dmax < 1e-9

    sz = pd.Series({s: m["size"] for s, m in ev.items()})
    print(f"\n评测集 {len(ev)} 子集，分带：", {nm: int(((sz >= lo) & (sz <= hi)).sum())
                                              for lo, hi, nm in BANDS})
    print("\n结论：" + ("口径一致，可以开跑。" if ok else "口径不一致，先修好！"))
    return ok


def cmd_check_samplers():
    sys.path.insert(0, str(R75 / "src"))
    from samplers import SAMPLERS, ALL, size_histogram
    sys.path.insert(0, str(R69))
    from src.oracle import sample_mask
    a = SAMPLERS["uniform"](512, 44, np.random.RandomState(1234))
    b = sample_mask(512, 44, np.random.RandomState(1234))
    same = np.array_equal(a, b)
    print(f"[{'OK ' if same else 'FAIL'}] uniform 与 69 号 sample_mask 逐位相同（控制组无偏移）")
    print(f"\n{'采样器':<16}{'|S|<=3':>9}{'4-12':>8}{'13-44':>8}{'中位k':>7}{'全零':>6}")
    for nm in ALL:
        h = size_histogram(nm, 44, 200_000, seed=0)
        print(f"{nm:<16}{h[1:4].sum():>9.1%}{h[4:13].sum():>8.1%}{h[13:].sum():>8.1%}"
              f"{int(np.searchsorted(np.cumsum(h), 0.5)):>7}{h[0]:>6.0%}")
    return same


def cmd_check_repro():
    truth = load_truth()
    old = set(json.load(open(OUT / "old394.json")))
    rows = []
    for sd in seeds_of("uniform"):
        d = load_est("uniform", sd, truth)
        d = d[d.sid.isin(old) & (d.K == 0)]
        ck = torch.load(OUT / f"oracle_uniform_seed{sd}.pt", map_location="cpu", weights_only=False)
        rows.append(dict(K0_mae=d.err.abs().mean(),
                         K0_rho=spearmanr(d.est, d.truth).correlation,
                         r2_full=ck["r2_full"]))
    if not rows:
        raise SystemExit("还没有 uniform 的评测结果")
    m = pd.DataFrame(rows).mean()
    print("=" * 78)
    print(f"V1 复现闸门：uniform（{len(rows)} seed）在旧 388 个字段集合上 vs 74 号 MLP")
    print("=" * 78)
    print(f"{'指标':<14}{'74号':>10}{'本轮':>10}{'差':>10}{'容差':>9}  判定")
    allok = True
    for k, lab in [("r2_full", "满输入 R²"), ("K0_mae", "K=0 MAE"), ("K0_rho", "K=0 Spearman")]:
        good = abs(m[k] - REF74[k]) <= REPRO_TOL[k]
        allok &= bool(good)
        print(f"{lab:<14}{REF74[k]:>10.4f}{m[k]:>10.4f}{m[k]-REF74[k]:>+10.4f}"
              f"{REPRO_TOL[k]:>9.3f}  {'OK' if good else 'FAIL'}")
    print("\n结论：" + ("复现通过，继续 Phase 1。" if allok else "**复现失败——停下查 bug。**"))
    return allok


def cmd_main():
    df = collect(schemes_present())
    if df.empty:
        raise SystemExit("没有可用结果")
    df.to_csv(OUT / "raw_by_seed.csv", index=False)
    g = df.groupby("scheme")
    s, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    s["n_seed"] = g.size()
    base = s.loc["uniform"] if "uniform" in s.index else None

    order = ["none", "bern50", "uniform", "logunif", "small50", "small80",
             "workload", "workload_hard", "large50"]
    s = s.reindex([x for x in order if x in s.index])

    print("=" * 104)
    print("DNN75 主结果：K=0 MAE 分区间（基准 = uniform；扩充评测集 ~1576 子集）")
    print("=" * 104)
    print(f"{'方案':<15}{'满输入R²':>9}{'扫描<=3':>10}{'±σ':>7}{'中段4-12':>10}"
          f"{'防护>=13':>10}{'全局':>8}{'协同ρ':>8}{'top20':>7}{'top10排名':>10}")
    print("-" * 104)
    for sc, r in s.iterrows():
        print(f"{sc:<15}{r.r2_full:>9.4f}{r['mae_扫描|S|<=3']:>10.4f}"
              f"{sd.loc[sc,'mae_扫描|S|<=3'] if sc in sd.index and pd.notna(sd.loc[sc,'mae_扫描|S|<=3']) else 0:>7.4f}"
              f"{r['mae_中段4-12']:>10.4f}{r['mae_防护|S|>=13']:>10.4f}"
              f"{r.K0_mae:>8.4f}{r.syn_rho:>8.4f}{r.syn_top20:>7.2f}"
              f"{r.syn_top10_medrank:>10.0f}")
    print("-" * 104)

    print(f"\n=== K=0 分规模带 MAE ===")
    bcols = [f"band_{nm}" for _, _, nm in BANDS if f"band_{nm}" in s.columns]
    print(s[bcols].round(4).to_string())

    if base is not None:
        print(f"\n=== 相对 uniform 的变化（预注册判据）===")
        print(f"{'方案':<15}{'扫描降幅':>10}{'防护退化':>10}{'协同ρ增':>10}"
              f"{'top20增':>9}  判定")
        verdicts = {}
        for sc, r in s.iterrows():
            if sc == "uniform":
                continue
            drop = 1 - r["mae_扫描|S|<=3"] / base["mae_扫描|S|<=3"]
            deg = r["mae_防护|S|>=13"] - base["mae_防护|S|>=13"]
            drho = r.syn_rho - base.syn_rho
            dtop = r.syn_top20 - base.syn_top20
            main_ok = drop >= GATE["scan_rel_drop"]
            cons_ok = deg <= GATE["protect_abs_degrade"]
            side_ok = (drho >= GATE["syn_rho_gain"]) or (dtop >= GATE["syn_top20_gain"])
            v = ("采纳" if (main_ok and cons_ok and side_ok) else
                 "存在权衡" if (main_ok and not cons_ok) else
                 "主项达标/副项不足" if (main_ok and cons_ok) else "不达标")
            verdicts[sc] = v
            print(f"{sc:<15}{drop:>10.1%}{deg:>+10.4f}{drho:>+10.4f}{dtop:>+9.2f}  {v}")
        s["verdict"] = pd.Series(verdicts)
        print(f"\n判据：扫描区间降 ≥{GATE['scan_rel_drop']:.0%} | "
              f"防护退化 ≤{GATE['protect_abs_degrade']} | "
              f"协同 ρ 增 ≥{GATE['syn_rho_gain']} 或 top20 增 ≥{GATE['syn_top20_gain']:.0%}")
    s.to_csv(OUT / "scheme_comparison.csv")
    print(f"\n已写出 outputs/scheme_comparison.csv")
    return s


def cmd_routing():
    """路由：按 |S| 阈值选模型。纯后处理，零 GPU。|S| 是查询输入不是秘密，路由不算作弊。

    两路：|S|<=t 用 small 方案，否则用 large 方案。
    三路：|S|<=t1 用 small，t1<|S|<=t2 用 uniform（中段本来就是 uniform 最好），
          |S|>t2 用 large。中段不该被牺牲，所以三路才是诚实的形态。
    """
    assert_evalset_consistent()
    truth = load_truth()
    have = schemes_present()
    smalls = [x for x in ("logunif", "small50", "small80", "workload", "workload_hard") if x in have]
    larges = [x for x in ("large50", "uniform") if x in have]
    if not smalls or not larges or "uniform" not in have:
        raise SystemExit("路由需要小集合方案、大集合方案和 uniform")

    cache = {sc: {sd: load_est(sc, sd, truth) for sd in seeds_of(sc)} for sc in have}
    RN = [nm for _, _, nm in REGIONS]

    def score(parts_by_seed):
        """parts_by_seed: {seed: [df 片段...]} -> 各区间与全局 MAE 的 seed 均值。"""
        per = []
        for sd, parts in parts_by_seed.items():
            d = pd.concat(parts)
            per.append({f"mae_{nm}": d[d.region == nm].err.abs().mean() for nm in RN}
                       | {"K0_mae": d.err.abs().mean()})
        m = pd.DataFrame(per).mean()
        return m, len(per)

    def k0(sc, sd, lo, hi):
        d = cache[sc][sd]
        return d[(d.K == 0) & (d["size"] >= lo) & (d["size"] <= hi)]

    rows = []
    # ---- 两路 ----
    for a, b, t in itertools.product(smalls, larges, (3, 4, 6, 8, 12, 16)):
        if a == b:
            continue
        sds = sorted(set(cache[a]) & set(cache[b]))
        parts = {sd: [k0(a, sd, 1, t), k0(b, sd, t + 1, 44)] for sd in sds}
        m, n = score(parts)
        rows.append(dict(kind="两路", small=a, mid="-", large=b, t1=t, t2=t, n_seed=n, **m.to_dict()))
    # ---- 三路（中段交回 uniform）----
    for a, b, t1, t2 in itertools.product(smalls, larges, (3, 4), (8, 12, 16)):
        if a == b or b == "uniform":
            continue
        sds = sorted(set(cache[a]) & set(cache[b]) & set(cache["uniform"]))
        parts = {sd: [k0(a, sd, 1, t1), k0("uniform", sd, t1 + 1, t2), k0(b, sd, t2 + 1, 44)]
                 for sd in sds}
        m, n = score(parts)
        rows.append(dict(kind="三路", small=a, mid="uniform", large=b, t1=t1, t2=t2,
                         n_seed=n, **m.to_dict()))

    base, _ = score({sd: [cache["uniform"][sd][cache["uniform"][sd].K == 0]]
                     for sd in seeds_of("uniform")})
    r = pd.DataFrame(rows)
    r["三区间全不劣"] = [all(x[f"mae_{nm}"] <= base[f"mae_{nm}"] + 1e-9 for nm in RN)
                        for _, x in r.iterrows()]
    r = r.sort_values("K0_mae")
    r.to_csv(OUT / "routing_comparison.csv", index=False)

    print("=" * 104)
    print("Phase 2 路由（K=0；基准 = uniform 单模型）")
    print("=" * 104)
    print(f"{'形态':<5}{'小集合':<15}{'中段':<9}{'大集合':<10}{'t1':>3}{'t2':>4}"
          f"{'扫描<=3':>10}{'中段4-12':>10}{'防护>=13':>10}{'全局':>9}  三区间全不劣")
    print("-" * 104)
    for _, x in r.head(14).iterrows():
        print(f"{x['kind']:<5}{x['small']:<15}{x['mid']:<9}{x['large']:<10}"
              f"{x['t1']:>3}{x['t2']:>4}"
              f"{x['mae_扫描|S|<=3']:>10.4f}{x['mae_中段4-12']:>10.4f}"
              f"{x['mae_防护|S|>=13']:>10.4f}{x['K0_mae']:>9.4f}"
              f"{'      是' if x['三区间全不劣'] else '      否'}")
    print("-" * 104)
    print(f"{'[基准] uniform 单模型':<46}{base['mae_扫描|S|<=3']:>10.4f}"
          f"{base['mae_中段4-12']:>10.4f}{base['mae_防护|S|>=13']:>10.4f}{base['K0_mae']:>9.4f}")
    win = r[r["三区间全不劣"]]
    print(f"\n三区间全部不劣于 uniform 的组合：{len(win)} 个"
          + (f"，最优 = {win.iloc[0].kind} {win.iloc[0].small}|{win.iloc[0].mid}|{win.iloc[0].large}"
             f" (t1={win.iloc[0].t1}, t2={win.iloc[0].t2})，全局 MAE {win.iloc[0].K0_mae:.4f}"
             f"（uniform {base['K0_mae']:.4f}）" if len(win) else "")) 
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-protocol", action="store_true")
    ap.add_argument("--check-samplers", action="store_true")
    ap.add_argument("--check-repro", action="store_true")
    ap.add_argument("--round", choices=["main", "routing"])
    a = ap.parse_args()
    if a.check_protocol:
        sys.exit(0 if cmd_check_protocol() else 1)
    if a.check_samplers:
        sys.exit(0 if cmd_check_samplers() else 1)
    if a.check_repro:
        sys.exit(0 if cmd_check_repro() else 1)
    if a.round == "main":
        cmd_main()
    elif a.round == "routing":
        cmd_routing()
    else:
        ap.print_help()




def assert_evalset_consistent():
    """所有 est CSV 必须覆盖同一个评测集（防止评测集变更后混入旧结果）。
    教训：74→75 过渡时曾因 pkill 漏杀 eval_variant.py，混入过 1538 子集的脏数据。"""
    n_ref = len(json.load(open(OUT / "evalset.json")))
    bad = []
    for p in sorted(OUT.glob("est_*_seed*.csv")):
        n = pd.read_csv(p, usecols=["sid"]).sid.nunique()
        if n != n_ref:
            bad.append((p.name, n))
    if bad:
        raise SystemExit(f"评测集不一致（应为 {n_ref} 子集）：{bad}\n请删除这些文件后重跑。")
    return True

if __name__ == "__main__":
    main()
