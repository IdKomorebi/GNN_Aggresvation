# -*- coding: utf-8 -*-
"""96 号 阶段 A：注入实验——把"尺子坏"与"搜索瞎"分开。

背景
----
k=6/7 的认证显示 top 与随机对照完全分不开（AUC 0.42/0.21），但同时观察到
**单调性违反率 20%→38%→60%→90%**（子集认证值高于全集，理论上不可能）。
这说明 k>=6 时**重训真值本身**已被噪声主导。于是"精确率 0/10"混合了三种可能：
方法找不到 / 数据里本来没有 / 尺子测不准。本脚本用已知强度的注入信号拆开它们。

注入信号构造——★本号修正了 90/91/93 号沿用的构造
------------------------------------------------
旧构造 y = α·normalize(∏ z_i) 在真实电网字段上**不是纯高阶交互**。自检实测：
k=2 注入 0.35，某元组只量回 0.028——因为 v_set=0.386 而 v_maxsub=0.358，
**单个字段已几乎完全预测"交互项"**。根因是重尾：该元组的乘积峰度达 481.6，
乘积被少数极端值主导，而极端值由单个因子驱动 ⟹ 乘积退化成单字段的函数。
k=6 时 z 构造的乘积峰度中位数高达 1127，信号集中在极少数样本上，根本学不出来。

本号改为  y = α·normalize(∏_{i∈S} tanh(z_i)) + √(1−α²)·ε ,  真实 R² = α²
tanh 有界单调，把 k=6 的乘积峰度中位数从 1127 压到 18。并加三重元组筛选：
  (a) 字段两两 |corr| < MAX_CORR   (b) 乘积峰度 < MAX_KURT
  (c) 单字段柔性解释力（20 分位分箱）< MAX_PURITY
筛选后各阶均可得到合格元组（k=8 通过率 3.8%，仍够用）。
**纯度不假设、而是逐 trial 实测并报告**（v_maxsub 列）；v_maxsub 偏高的 trial
标记为不纯并排除出主统计。

两个独立问题
------------
A1 真值端：对注入集合跑**重训认证**，syn_true 能否量回 α²？
           量不回 ⟹ 尺子坏（噪声底 > 信号），必须先修真值协议。
A2 搜索端：L1 扫描给注入集合的 syn，在同阶随机集合中排第几百分位？
           低百分位 ⟹ 搜索瞎。

效率优化
--------
同一元组 S 上叠加 len(R2S) 个不同强度的列，**共用同一次重训**（输出维=强度数），
省 len(R2S) 倍算力；φ 与 y 无关，搜索端的 Gram 也跨强度复用。

★红线：注入信号与重训真值只用于评估，绝不进入扫描/路由/校准路径。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))
from src.data_processing import prepare_data  # noqa: E402
from featridge import (FrozenPhi, SPLIT_SEED, build_features, fit_val_idx,  # noqa: E402
                       load_oracle, ridge_r2)
from runlog import log  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE = 400, 120          # 与 88/93 号认证协议一致
INJ_SEED = 960730
CKPT = REPO / "DNN_Aggresvation93/outputs/oracle_l1_aug8_seed0.pt"
MAX_CORR, MAX_KURT, MAX_PURITY = 0.5, 30.0, 0.05   # 元组筛选阈（见文件头）


class DNN(nn.Module):
    """与 certify_highorder.py 逐字一致（hidden=128, 2 隐层, dropout 0.15）。"""

    def __init__(self, n_in, n_out, hidden=128, depth=2, p=0.15):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p)]
            d = hidden
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_hgb(Xtr, Ytr, Xte, seed: int):
    """梯度提升树：天然表达交互，用于检验"攻击者模型类是否太弱"。

    v_c(S) 定义为对模型族取上确界 ⟹ 若某个模型学得更好，就该用它。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    preds = []
    for j in range(Ytr.shape[1]):
        m = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=seed)
        m.fit(Xtr, Ytr[:, j])
        preds.append(m.predict(Xte))
    return np.column_stack(preds)


def train_generic(model, Xtr, Ytr, seed: int):
    """88 号修正版：早停用训练集内部 15% val，测试集完全不参与。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    g = torch.Generator()
    g.manual_seed(20260725)
    perm = torch.randperm(len(Xtr), generator=g).to(DEV)
    n_val = max(int(len(Xtr) * 0.15), 1)
    vi, ti = perm[:n_val], perm[n_val:]
    Xv, Yv, Xt, Yt = Xtr[vi], Ytr[vi], Xtr[ti], Ytr[ti]
    best, bs, pat, n = 1e9, None, 0, len(Xt)
    for _ in range(EPOCHS):
        model.train()
        pm = torch.randperm(n, device=DEV)
        for k in range(0, n, 128):
            ix = pm[k:k + 128]
            opt.zero_grad()
            ((model(Xt[ix]) - Yt[ix]) ** 2).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(Xv) - Yv) ** 2).mean())
        if v < best:
            best, bs, pat = v, deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    model.load_state_dict(bs)
    model.eval()
    return model


def per_col_r2(pred, true):
    ss = ((true - pred) ** 2).sum(0)
    st = ((true - true.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="6,7,8", help="逗号分隔；2 用于自检闸门")
    ap.add_argument("--r2s", default="0.05,0.10,0.20,0.35", help="注入的真实 R²")
    ap.add_argument("--n_trial", type=int, default=3, help="每阶随机元组数")
    ap.add_argument("--n_rand", type=int, default=400, help="搜索端随机对照集合数")
    ap.add_argument("--seeds", default="0,1", help="重训 seed（多 seed 给噪声底）")
    ap.add_argument("--mode", choices=["truth", "search", "both"], default="both")
    ap.add_argument("--attacker", default="dnn128",
                    choices=["dnn128", "dnn512", "hgb", "best"],
                    help="攻击者模型族；best = 对三者取上确界（v 的定义）")
    ap.add_argument("--n_train", type=int, default=0,
                    help=">0 则只用前 N 行训练（样本量缩放实验，检验统计极限假说）")
    ap.add_argument("--dataset", default="pjm", choices=["pjm", "caiso"],
                    help="caiso 用 95_caiso 的 base.yaml / 清洗数据 / L1 checkpoint")
    ap.add_argument("--est_kind", default="last", choices=["last", "poly2", "full"],
                    help="搜索端估计器。★对照用：poly2/full 是与目标无关的手工字典，"
                         "可把'高阶难'与'φ 对合成信号分布外'区分开")
    ap.add_argument("--ckpt", default="",
                    help="覆盖搜索端使用的 L1 主干（97 号改造 φ 的元训练分布后用它判决）")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    orders = [int(x) for x in args.orders.split(",")]
    r2s = [float(x) for x in args.r2s.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    alphas = [np.sqrt(r) for r in r2s]

    if args.dataset == "caiso":
        R95 = REPO / "DNN_Aggresvation95_caiso"
        cfg = yaml.safe_load((R95 / "base.yaml").read_text(encoding="utf-8"))
        cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/caiso_2025_hourly_cleaned.csv")
        ckpt_l1 = R95 / "outputs/oracle_l1_aug8_seed0.pt"
    else:
        cfg = yaml.safe_load((ROOT / "base.yaml").read_text(encoding="utf-8"))
        cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
        ckpt_l1 = CKPT
    if args.ckpt:
        ckpt_l1 = Path(args.ckpt)
    torch.manual_seed(42)
    np.random.seed(42)
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    tr_all, te_all = data["train_data"], data["test_data"]
    if args.n_train:                       # 样本量缩放：截断训练集，测试集不动
        tr_all = tr_all[:args.n_train]
        print(f"[缩放] 训练集截断为 {len(tr_all)} 行", flush=True)
    Xtr_np = tr_all[:, gi].astype(np.float64)
    Xte_np = te_all[:, gi].astype(np.float64)
    n_gen = Xtr_np.shape[1]
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(Xte_np))
    a_idx = perm[len(perm) // 2:]

    mu, sd = Xtr_np.mean(0), Xtr_np.std(0)
    sd[sd < 1e-9] = 1.0
    # ★ tanh 压缩：有界单调，压掉重尾（否则乘积退化成单字段函数，见文件头）
    Ztr_np = (Xtr_np - mu) / sd
    Zte_np = (Xte_np - mu) / sd
    Ttr_np = np.tanh(Ztr_np)
    Tte_np = np.tanh(Zte_np)
    CORR = np.corrcoef(Ztr_np.T)

    def make_signal(S, rng):
        """返回 (Ytr, Yte)：同一元组上 len(alphas) 个强度的列，共用一次重训。"""
        ptr = np.prod(Ttr_np[:, list(S)], axis=1)
        pte = np.prod(Tte_np[:, list(S)], axis=1)
        m_, s_ = ptr.mean(), ptr.std() + 1e-12
        ptr, pte = (ptr - m_) / s_, (pte - m_) / s_
        ytr, yte = [], []
        for a in alphas:
            k = np.sqrt(max(1 - a ** 2, 0))
            ytr.append(a * ptr + k * rng.randn(len(ptr)))
            yte.append(a * pte + k * rng.randn(len(pte)))
        return np.column_stack(ytr), np.column_stack(yte)

    def flex_r2(P, x, nb=20):
        """单字段对乘积的柔性解释力（20 分位分箱均值）——纯度的反面。"""
        b = np.clip(np.digitize(x, np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])),
                    0, nb - 1)
        pred = np.array([P[b == j].mean() if (b == j).sum() > 5 else P.mean()
                         for j in range(nb)])[b]
        return 1 - ((P - pred) ** 2).sum() / (((P - P.mean()) ** 2).sum() + 1e-12)

    def screen(S):
        """三重筛选：低互相关 / 低峰度 / 低单字段解释力。返回 (是否合格, 诊断)。"""
        sl = list(S)
        cmax = float(np.abs(CORR[np.ix_(sl, sl)][np.triu_indices(len(sl), 1)]).max()) \
            if len(sl) > 1 else 0.0
        P = np.prod(Ttr_np[:, sl], axis=1)
        P = (P - P.mean()) / (P.std() + 1e-12)
        ku = float(pd.Series(P).kurt())
        pur = max(flex_r2(P, Ttr_np[:, c]) for c in sl)
        ok = cmax <= MAX_CORR and ku <= MAX_KURT and pur <= MAX_PURITY
        return ok, dict(corr_max=round(cmax, 3), kurtosis=round(ku, 1),
                        single_field_r2=round(float(pur), 4))

    def greedy_tuple(m, rng):
        """贪心构造低互相关元组：随机起点，每步加入"与已选最大相关最小"的字段。

        纯拒绝采样在强相关数据集上不可行（CAISO k=7 通过率仅 0.6%，k>=9 无望），
        贪心把复杂度降到 O(k·n)，使高阶实验可行。
        """
        cur = [int(rng.randint(n_gen))]
        while len(cur) < m:
            worst = np.abs(CORR[:, cur]).max(1)
            worst[cur] = np.inf
            cand = np.argsort(worst)[:max(3, n_gen // 10)]   # 在最优的一批里随机取
            cur.append(int(cand[rng.randint(len(cand))]))
        return tuple(sorted(cur))

    # ---- 各阶的注入元组（固定种子 + 筛选；truth/search 两端共用同一批） ----
    plan = []
    for m in orders:
        rng = np.random.RandomState(INJ_SEED + m)
        found, tried = 0, 0
        while found < args.n_trial and tried < 20000:
            tried += 1
            # 前 2000 次用纯随机（无偏）；之后改贪心（高阶/强相关数据集的必要手段）
            S = (tuple(sorted(rng.choice(n_gen, m, replace=False)))
                 if tried <= 2000 else greedy_tuple(m, rng))
            ok, diag = screen(S)
            if not ok or any(p["S"] == S for p in plan):
                continue
            plan.append(dict(order=m, trial=found, S=S,
                             sig_seed=INJ_SEED + m * 100 + found, **diag))
            found += 1
        assert found == args.n_trial, f"k={m} 只筛到 {found}/{args.n_trial} 个合格元组"
        print(f"[筛选] k={m}: {found} 个合格元组（尝试 {tried} 次，通过率 "
              f"{found/tried*100:.1f}%）", flush=True)
    mine = plan[args.shard::args.nshard]
    log("INJ", "START", note=f"orders={orders} r2s={r2s} trials={args.n_trial} "
                             f"seeds={seeds} 本分片 {len(mine)}/{len(plan)} 个元组 "
                             f"mode={args.mode}")

    # ================= A1 真值端：重训认证能否量回 =================
    if args.mode in ("truth", "both"):
        rows = []
        ntag = f"_n{args.n_train}" if args.n_train else ""
        out_csv = ROOT / (f"outputs/inject_truth{args.tag}_{args.dataset}_{args.attacker}{ntag}"
                          f"_s{args.shard}of{args.nshard}.csv")
        t0 = time.perf_counter()
        for job in mine:
            m, S = job["order"], job["S"]
            rng = np.random.RandomState(job["sig_seed"])
            Ytr_np, Yte_np = make_signal(S, rng)
            Ytr_t = torch.as_tensor(Ytr_np, dtype=torch.float32, device=DEV)
            Yte_audit = Yte_np[a_idx]

            def v_of(sel, seed, attacker=None):
                """按攻击者模型算 v；'best' 对模型族取上确界（= v 的定义）。"""
                atk = attacker or args.attacker
                cols = gi[list(sel)]
                if atk == "best":
                    return np.maximum.reduce([v_of(sel, seed, a)
                                              for a in ("dnn128", "dnn512", "hgb")])
                if atk == "hgb":
                    p = fit_hgb(tr_all[:, cols], Ytr_np, te_all[:, cols], seed)
                    return per_col_r2(p[a_idx], Yte_audit)
                hid, dep = (128, 2) if atk == "dnn128" else (512, 3)
                Xa = torch.as_tensor(tr_all[:, cols], dtype=torch.float32, device=DEV)
                Xb = torch.as_tensor(te_all[:, cols], dtype=torch.float32, device=DEV)
                mdl = train_generic(
                    DNN(len(sel), len(alphas), hidden=hid, depth=dep).to(DEV),
                    Xa, Ytr_t, seed)
                with torch.no_grad():
                    p = mdl(Xb).cpu().numpy()
                return per_col_r2(p[a_idx], Yte_audit)

            subs = list(combinations(S, m - 1))
            for seed in seeds:
                vS = v_of(S, seed)
                vT = np.stack([v_of(T, seed) for T in subs])
                syn = vS - vT.max(0)
                for j, r2 in enumerate(r2s):
                    # 纯度实测：注入信号有多少被真子集捞走（应远小于 r2 才算纯）
                    impure = float(vT.max(0)[j]) / r2
                    rows.append(dict(order=m, trial=job["trial"], S=str(S), seed=seed,
                                     r2_inject=r2, v_set=float(vS[j]),
                                     v_maxsub=float(vT.max(0)[j]),
                                     syn_true=float(syn[j]),
                                     recovered=float(syn[j]) / r2,
                                     impurity=round(impure, 3),
                                     pure=bool(impure < 0.30),
                                     mono_violation=bool(vT.max(0)[j] > vS[j]),
                                     corr_max=job["corr_max"], kurtosis=job["kurtosis"],
                                     single_field_r2=job["single_field_r2"]))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                el = time.perf_counter() - t0
                print(f"  [k={m} trial={job['trial']} seed={seed}] "
                      f"syn_true={np.round(syn, 4).tolist()} (注入 {r2s}) [{el:.0f}s]",
                      flush=True)
        log("INJ", "TRUTH", note=f"分片{args.shard} 真值端完成 "
                                 f"{time.perf_counter()-t0:.0f}s → {out_csv.name}")

    # ================= A2 搜索端：注入集合在随机集合中的百分位 =================
    if args.mode in ("search", "both"):
        dev = DEV
        Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=dev)
        Xev = torch.tensor(Xte_np[a_idx], dtype=torch.float32, device=dev)
        Ztr = torch.tensor(Ztr_np, dtype=torch.float64, device=dev)
        Zev = torch.tensor(Zte_np[a_idx], dtype=torch.float64, device=dev)
        fit_i, val_i = fit_val_idx(len(Xtr_np), dev)
        phi = FrozenPhi(load_oracle(ckpt_l1, n_gen, 12, dev), "last")
        rows = []
        t0 = time.perf_counter()
        for job in mine:
            m, S = job["order"], job["S"]
            rng = np.random.RandomState(job["sig_seed"])
            Ytr_np, Yte_np = make_signal(S, rng)
            Ytr = torch.tensor(Ytr_np, dtype=torch.float64, device=dev)
            Yev = torch.tensor(Yte_np[a_idx], dtype=torch.float64, device=dev)

            r2rng = np.random.RandomState(job["sig_seed"] + 7)
            rand_sets, seen = [], {S}
            while len(rand_sets) < args.n_rand:
                T = tuple(sorted(r2rng.choice(n_gen, m, replace=False)))
                if T not in seen:
                    seen.add(T)
                    rand_sets.append(T)
            cand = [S] + rand_sets
            subs = sorted({tuple(sorted(t)) for C in cand for t in combinations(C, m - 1)})
            spos = {s: i for i, s in enumerate(subs)}

            def ev(sl):
                out = []
                for s in range(0, len(sl), args.chunk):
                    sb = torch.as_tensor(np.array(sl[s:s + args.chunk]),
                                         dtype=torch.long, device=dev)
                    F_tr = build_features(phi, Xtr, Ztr, sb, n_gen, args.est_kind)
                    F_ev = build_features(phi, Xev, Zev, sb, n_gen, args.est_kind)
                    out.append(ridge_r2(F_tr, Ytr, F_ev, Yev, fit_i, val_i).cpu().numpy())
                return np.concatenate(out, 0)

            v_sub, v_cand = ev(subs), ev(cand)
            syn = np.stack([v_cand[i] - v_sub[[spos[t] for t in combinations(C, m - 1)]].max(0)
                            for i, C in enumerate(cand)])
            for j, r2 in enumerate(r2s):
                s_inj, s_rand = syn[0, j], syn[1:, j]
                rows.append(dict(order=m, trial=job["trial"], S=str(S), r2_inject=r2,
                                 syn_inject=float(s_inj),
                                 rand_median=float(np.median(s_rand)),
                                 rand_p99=float(np.percentile(s_rand, 99)),
                                 percentile=float((s_rand < s_inj).mean() * 100),
                                 rank=int((s_rand >= s_inj).sum()) + 1, n_rand=len(s_rand)))
            el = time.perf_counter() - t0
            print(f"  [k={m} trial={job['trial']}] 百分位="
                  f"{[round(r['percentile'], 1) for r in rows[-len(r2s):]]} "
                  f"(注入 {r2s}) [{el:.0f}s]", flush=True)
            pd.DataFrame(rows).to_csv(
                ROOT / f"outputs/inject_search{args.tag}_{args.dataset}_{args.est_kind}_s{args.shard}of{args.nshard}.csv",
                index=False)
        log("INJ", "SEARCH", note=f"分片{args.shard} 搜索端完成 "
                                  f"{time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
