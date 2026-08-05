# -*- coding: utf-8 -*-
"""DNN78：对 76/77 号做不增加 GPU 成本的口径修正。

本脚本只读取既有 CSV/JSON/PT 实验产物，不重新训练模型。修正四件事：

1. F-2 的停机单位从 `(字段对, conf)` 改成真正可执行的“每个字段对一个 K”；
2. 三阶 K=25 协同改成同保真度：
      v_K25(ijk) - max[v_K25(ij), v_K25(ik), v_K25(jk)]；
3. 把高阶统计拆成 `(集合, conf)` 条目级与集合级，避免混用；
4. 对“397 个全量已知 + 剩余空间随机抽 1800 个”的设计做分层加权，
   不再把排除 top197 后的 2000 池称为简单随机无偏池。

重训 worker 使用测试集早停的问题只做静态审计记录；修复它需要新 GPU 重训，
不在本脚本伪造“已解决”。
"""
from __future__ import annotations

import ast
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R68 = REPO / "DNN_Aggresvation68"
R69 = REPO / "DNN_Aggresvation69"
R75 = REPO / "DNN_Aggresvation75"
R76 = REPO / "DNN_Aggresvation76"
R77 = REPO / "DNN_Aggresvation77"
OUT = ROOT / "outputs"

sys.path.insert(0, str(ROOT / "src"))
from runlog import Timer, log, section  # noqa: E402

STRONG2 = 0.20
STRONG3 = 0.10
NULL2 = 0.02
DETECT2 = 0.10
N_ALL_TRIPLES = 13244
N_EXACT = 397
N_REMAINDER = N_ALL_TRIPLES - N_EXACT
N_REMAINDER_SAMPLE = 1800
REMAINDER_WEIGHT = N_REMAINDER / N_REMAINDER_SAMPLE
RETENTION = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def json_safe(value):
    """把 numpy 标量与 NaN 转成严格 JSON 可表示的对象。"""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def field_context():
    """只为取得固定的 44 general 字段顺序；数据不参与重新训练。"""
    import yaml

    sys.path.insert(0, str(R69))
    from src.data_processing import prepare_data

    cfg = yaml.safe_load((R77 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    return data, {name: idx for idx, name in enumerate(data["general"])}


def build_second_order_index():
    evalset = json.load(open(R75 / "outputs/evalset.json"))
    singles = {meta["fields"][0]: sid for sid, meta in evalset.items() if meta["size"] == 1}
    pairs = {
        tuple(sorted(meta["fields"])): sid
        for sid, meta in evalset.items()
        if meta["size"] == 2
    }
    truth = pd.read_csv(R68 / "outputs/synergy2_perconf.csv")
    truth["sid_ij"] = [
        pairs.get(tuple(sorted((left, right)))) for left, right in zip(truth.fi, truth.fj)
    ]
    truth["sid_i"] = truth.fi.map(singles)
    truth["sid_j"] = truth.fj.map(singles)
    return truth.dropna(subset=["sid_ij", "sid_i", "sid_j"]).reset_index(drop=True), evalset


def load_second_order_curves(truth: pd.DataFrame):
    matrices = []
    ks = None
    for seed in (0, 1, 2):
        data = pd.read_csv(R76 / "outputs" / f"fine_uniform_seed{seed}.csv")
        ks = sorted(int(k) for k in data.K.unique())
        curve = np.full((len(ks), len(truth)), np.nan)
        for pos, k in enumerate(ks):
            estimates = data[data.K == k].set_index(["sid", "conf"]).est.to_dict()
            pair = np.array(
                [estimates.get((sid, conf), np.nan) for sid, conf in zip(truth.sid_ij, truth.conf)]
            )
            left = np.array(
                [estimates.get((sid, conf), np.nan) for sid, conf in zip(truth.sid_i, truth.conf)]
            )
            right = np.array(
                [estimates.get((sid, conf), np.nan) for sid, conf in zip(truth.sid_j, truth.conf)]
            )
            curve[pos] = pair - np.maximum(left, right)
        matrices.append(curve)
    return matrices, ks


def pair_groups(truth: pd.DataFrame):
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(truth.itertuples()):
        key = tuple(sorted((row.fi, row.fj)))
        groups.setdefault(key, []).append(idx)
    if len(groups) != 946 or set(map(len, groups.values())) != {12}:
        raise RuntimeError("二阶真值不是预期的 946 字段对 × 12 conf")
    keys = list(groups)
    return keys, [np.asarray(groups[key], dtype=int) for key in keys]


def simulate_query_stop(
    curve: np.ndarray,
    ks: list[int],
    groups: list[np.ndarray],
    *,
    eps: float,
    delta: float,
    patience: int,
    mode: str,
):
    """为一个字段对只给一个停机 K。

    mode=maxconf：按 max_c syn_c 的轨迹判断，符合“任一 conf 危险就记录”的扫描目标。
    mode=allconf：要求 12 个 conf 同时稳定；这是更严格、也更昂贵的诊断上界。
    """
    n_groups = len(groups)
    stopped = np.full(n_groups, ks[-1], dtype=int)
    values = curve[-1].copy()
    stable = np.zeros(n_groups, dtype=int)
    active = np.ones(n_groups, dtype=bool)

    for pos in range(1, len(ks)):
        for group_id, entry_idx in enumerate(groups):
            if not active[group_id]:
                continue
            current = curve[pos, entry_idx]
            previous = curve[pos - 1, entry_idx]
            if mode == "maxconf":
                score_now = float(np.max(current))
                score_prev = float(np.max(previous))
                change = abs(score_now - score_prev)
                rise = score_now - score_prev
                low = score_now < delta
            elif mode == "allconf":
                change = float(np.max(np.abs(current - previous)))
                rise = float(np.max(current - previous))
                low = float(np.max(current)) < delta
            else:
                raise ValueError(mode)

            stable[group_id] = stable[group_id] + 1 if change < eps else 0
            convergence = stable[group_id] >= patience
            early_negative = low and rise <= 0
            if convergence or early_negative:
                stopped[group_id] = ks[pos]
                values[entry_idx] = current
                active[group_id] = False
    return stopped, values


def evaluate_stop(
    truth_values: np.ndarray,
    matrices: list[np.ndarray],
    ks: list[int],
    groups: list[np.ndarray],
    pair_ids: np.ndarray,
    *,
    eps: float,
    delta: float,
    patience: int,
    mode: str,
):
    selected_groups = [groups[idx] for idx in pair_ids]
    selected_entries = np.concatenate(selected_groups)
    strong = truth_values[selected_entries] > STRONG2
    null = truth_values[selected_entries] <= NULL2
    recalls, false_positives, steps = [], [], []
    for curve in matrices:
        stopped, estimates = simulate_query_stop(
            curve,
            ks,
            groups,
            eps=eps,
            delta=delta,
            patience=patience,
            mode=mode,
        )
        chosen = estimates[selected_entries]
        recalls.append(float((chosen[strong] > DETECT2).mean()))
        false_positives.append(float((chosen[null] > DETECT2).mean()))
        steps.append(float(stopped[pair_ids].mean()))
    return {
        "mean_steps": float(np.mean(steps)),
        "recall": float(np.mean(recalls)),
        "fp": float(np.mean(false_positives)),
        "n_pairs": int(len(pair_ids)),
        "n_strong_entries": int(strong.sum()),
    }


def evaluate_fixed(
    truth_values: np.ndarray,
    matrices: list[np.ndarray],
    ks: list[int],
    groups: list[np.ndarray],
    pair_ids: np.ndarray,
    k: int,
):
    pos = ks.index(k)
    selected_entries = np.concatenate([groups[idx] for idx in pair_ids])
    strong = truth_values[selected_entries] > STRONG2
    null = truth_values[selected_entries] <= NULL2
    return {
        "mean_steps": float(k),
        "recall": float(
            np.mean([(curve[pos, selected_entries][strong] > DETECT2).mean() for curve in matrices])
        ),
        "fp": float(
            np.mean([(curve[pos, selected_entries][null] > DETECT2).mean() for curve in matrices])
        ),
        "n_pairs": int(len(pair_ids)),
        "n_strong_entries": int(strong.sum()),
    }


def correct_f2(truth: pd.DataFrame):
    section("阶段 1：F-2 停机单位修正")
    log(
        "F2-UNIT",
        "START",
        note="把 11352 个独立 entry 停机改为 946 个字段对各自一个停机 K",
    )
    matrices, ks = load_second_order_curves(truth)
    _, groups = pair_groups(truth)
    truth_values = truth.synergy.to_numpy()

    pair_has_strong = np.array(
        [bool(np.any(truth_values[entry_idx] > STRONG2)) for entry_idx in groups]
    )
    rng = np.random.RandomState(7801)
    tune, test = [], []
    for label in (False, True):
        ids = np.flatnonzero(pair_has_strong == label)
        rng.shuffle(ids)
        cut = int(round(len(ids) * 0.60))
        tune.extend(ids[:cut])
        test.extend(ids[cut:])
    tune = np.asarray(sorted(tune), dtype=int)
    test = np.asarray(sorted(test), dtype=int)
    all_pairs = np.arange(len(groups), dtype=int)

    base_tune = evaluate_fixed(truth_values, matrices, ks, groups, tune, 50)
    sweep_rows = []
    for eps, delta, patience in itertools.product(
        (0.001, 0.002, 0.005, 0.010),
        (0.01, 0.02, 0.05, 0.08),
        (1, 2, 3),
    ):
        result = evaluate_stop(
            truth_values,
            matrices,
            ks,
            groups,
            tune,
            eps=eps,
            delta=delta,
            patience=patience,
            mode="maxconf",
        )
        sweep_rows.append(
            {
                "eps": eps,
                "delta": delta,
                "patience": patience,
                **result,
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    eligible = sweep[sweep.recall >= base_tune["recall"]]
    if eligible.empty:
        eligible = sweep.sort_values(["recall", "mean_steps"], ascending=[False, True]).head(1)
    chosen = eligible.sort_values(["mean_steps", "recall"], ascending=[True, False]).iloc[0]

    result_rows = []
    for scope, ids in (("tune", tune), ("test", test), ("all", all_pairs)):
        fixed = evaluate_fixed(truth_values, matrices, ks, groups, ids, 50)
        result_rows.append({"scope": scope, "rule": "fixed_k50", **fixed})
        selected = evaluate_stop(
            truth_values,
            matrices,
            ks,
            groups,
            ids,
            eps=float(chosen.eps),
            delta=float(chosen.delta),
            patience=int(chosen.patience),
            mode="maxconf",
        )
        result_rows.append(
            {
                "scope": scope,
                "rule": "query_maxconf_tuned",
                "eps": float(chosen.eps),
                "delta": float(chosen.delta),
                "patience": int(chosen.patience),
                **selected,
            }
        )

    # 复算 76 号写入 CHANGELOG 的固定参数，展示“逐 entry / 逐 query”的差别。
    current = {"eps": 0.002, "delta": 0.02, "patience": 2}
    for mode in ("maxconf", "allconf"):
        result = evaluate_stop(
            truth_values,
            matrices,
            ks,
            groups,
            all_pairs,
            mode=mode,
            **current,
        )
        result_rows.append(
            {
                "scope": "all",
                "rule": f"query_{mode}_published_params",
                **current,
                **result,
            }
        )

    result_df = pd.DataFrame(result_rows)
    sweep.to_csv(OUT / "f2_query_rule_sweep.csv", index=False)
    result_df.to_csv(OUT / "f2_query_level.csv", index=False)
    summary = {
        "split_seed": 7801,
        "tune_pairs": int(len(tune)),
        "test_pairs": int(len(test)),
        "chosen_rule": {
            "eps": float(chosen.eps),
            "delta": float(chosen.delta),
            "patience": int(chosen.patience),
        },
        "test": result_df[result_df.scope == "test"].to_dict(orient="records"),
    }
    (OUT / "f2_query_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    test_rows = result_df[result_df.scope == "test"].set_index("rule")
    adaptive = test_rows.loc["query_maxconf_tuned"]
    baseline = test_rows.loc["fixed_k50"]
    log(
        "F2-UNIT",
        "DECISION",
        note=(
            f"独立 test：逐查询早停平均 {adaptive.mean_steps:.1f} 步、召回 "
            f"{adaptive.recall:.1%}；固定 K50 召回 {baseline.recall:.1%}。"
        ),
        eps=float(chosen.eps),
        delta=float(chosen.delta),
        patience=int(chosen.patience),
    )
    return result_df, summary


def pair_estimates_at_k25(evalset: dict, name_to_idx: dict[str, int]):
    sid_to_indices = {
        sid: tuple(sorted(name_to_idx[name] for name in meta["fields"]))
        for sid, meta in evalset.items()
    }
    data = pd.read_csv(R76 / "outputs/fine_uniform_seed0.csv")
    data = data[(data.K == 25) & (data["size"] == 2)].copy()
    data["indices"] = data.sid.map(sid_to_indices)
    return {
        (row.indices[0], row.indices[1], row.conf): row.est for row in data.itertuples()
    }


def metrics(frame: pd.DataFrame, estimate_col: str):
    data = frame.dropna(subset=["syn3_true", estimate_col])
    strong = data.syn3_true > STRONG3
    return {
        "n": int(len(data)),
        "rho": float(spearmanr(data.syn3_true, data[estimate_col]).correlation),
        "mae": float((data[estimate_col] - data.syn3_true).abs().mean()),
        "bias": float((data[estimate_col] - data.syn3_true).mean()),
        "n_strong_entries": int(strong.sum()),
        "strong_true_mean": float(data.loc[strong, "syn3_true"].mean()),
        "strong_est_mean": float(data.loc[strong, estimate_col].mean()),
        "strong_threshold_detection": float((data.loc[strong, estimate_col] > STRONG3).mean()),
    }


def correct_syn3(evalset: dict, name_to_idx: dict[str, int]):
    section("阶段 2：三阶同保真度修正")
    log(
        "SYN3-K",
        "START",
        note="将 v_K25(ijk)-max v_K0(pair) 改为 K25-K25；候选集合保持不变",
    )
    pair25 = pair_estimates_at_k25(evalset, name_to_idx)
    low0 = pd.read_csv(R77 / "outputs/lowfour_k0.csv")
    pair0 = {}
    for row in low0[low0["size"] == 2].itertuples():
        indices = tuple(int(value) for value in row.key.split("_"))
        pair0[(indices, row.conf)] = row.est

    triple25 = pd.concat(
        [pd.read_csv(path) for path in sorted((R77 / "outputs").glob("triples_kstar_shard*.csv"))],
        ignore_index=True,
    )
    corrected_rows = []
    for row in triple25.itertuples():
        child_pairs = [
            tuple(sorted(pair))
            for pair in ((row.i, row.j), (row.i, row.k), (row.j, row.k))
        ]
        children25 = [pair25.get((*pair, row.conf)) for pair in child_pairs]
        children0 = [pair0.get((pair, row.conf)) for pair in child_pairs]
        if any(value is None for value in children25 + children0):
            continue
        corrected_rows.append(
            {
                "i": row.i,
                "j": row.j,
                "k": row.k,
                "conf": row.conf,
                "v_triple_k25": row.est,
                "syn3_k25_mixed": row.est - max(children0),
                "syn3_k25_consistent": row.est - max(children25),
            }
        )
    corrected = pd.DataFrame(corrected_rows)
    corrected.to_csv(OUT / "syn3_k25_consistent_entries.csv", index=False)

    k0_data = pd.read_csv(R77 / "outputs/triples_syn3_k0.csv")
    k0_map = {
        (row.i, row.j, row.k, row.conf): row.syn3_est_k0 for row in k0_data.itertuples()
    }
    mixed_map = {
        (row.i, row.j, row.k, row.conf): row.syn3_k25_mixed
        for row in corrected.itertuples()
    }
    consistent_map = {
        (row.i, row.j, row.k, row.conf): row.syn3_k25_consistent
        for row in corrected.itertuples()
    }

    pool = pd.read_csv(R77 / "outputs/h2_unbiased_pool.csv")
    pool["indices"] = pool.ix.map(ast.literal_eval)
    pool["syn3_k0"] = [
        k0_map.get((*indices, conf)) for indices, conf in zip(pool.indices, pool.conf)
    ]
    pool["syn3_k25_mixed"] = [
        mixed_map.get((*indices, conf)) for indices, conf in zip(pool.indices, pool.conf)
    ]
    pool["syn3_k25_consistent"] = [
        consistent_map.get((*indices, conf)) for indices, conf in zip(pool.indices, pool.conf)
    ]

    metric_rows = []
    for scope, frame, columns in (
        ("reported_pool_all", pool, ("syn3_k0",)),
        (
            "stage2_covered_common",
            pool.dropna(subset=["syn3_k25_consistent"]),
            ("syn3_k0", "syn3_k25_mixed", "syn3_k25_consistent"),
        ),
    ):
        for column in columns:
            metric_rows.append({"scope": scope, "estimator": column, **metrics(frame, column)})
    # 当前候选排序的实际单位是“三元组”：12 个 conf 取最大值。
    # 单独报告这个口径，避免把 24000 个 entry 的 rho 与候选召回混为一谈。
    triple_max = (
        pool.groupby("indices", as_index=False)
        .agg(syn3_true=("syn3_true", "max"), syn3_k0=("syn3_k0", "max"))
    )
    metric_rows.append(
        {
            "scope": "reported_pool_triple_max",
            "estimator": "syn3_k0",
            **metrics(triple_max, "syn3_k0"),
        }
    )
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT / "syn3_value_metrics.csv", index=False)

    truth_strong_triples = sorted(pool[pool.syn3_true > STRONG3].indices.unique())
    rank0 = (
        k0_data.groupby(["i", "j", "k"]).syn3_est_k0.max().sort_values(ascending=False)
    )
    rank_mixed = (
        corrected.groupby(["i", "j", "k"]).syn3_k25_mixed.max().sort_values(ascending=False)
    )
    rank_consistent = (
        corrected.groupby(["i", "j", "k"])
        .syn3_k25_consistent.max()
        .sort_values(ascending=False)
    )

    recall_rows = []
    for fraction in RETENTION:
        n_keep = int(round(N_ALL_TRIPLES * fraction))
        for label, ranking in (
            ("k0", rank0),
            ("k25_mixed", rank_mixed),
            ("k25_consistent", rank_consistent),
        ):
            if n_keep > len(ranking):
                continue
            keep = set(ranking.index[:n_keep])
            recall = np.mean([indices in keep for indices in truth_strong_triples])
            recall_rows.append(
                {
                    "population": "reported_pool_2000",
                    "estimator": label,
                    "retention": fraction,
                    "n_keep": n_keep,
                    "recall": float(recall),
                    "n_strong_triples": int(len(truth_strong_triples)),
                }
            )
    recall_df = pd.DataFrame(recall_rows)
    recall_df.to_csv(OUT / "syn3_recall_current_pool.csv", index=False)

    common = metric_df[metric_df.scope == "stage2_covered_common"].set_index("estimator")
    before = common.loc["syn3_k0"]
    after = common.loc["syn3_k25_consistent"]
    log(
        "SYN3-K",
        "DECISION",
        note=(
            f"同保真度后：stage2 覆盖条目的 rho {before.rho:.3f}→{after.rho:.3f}，"
            f"MAE {before.mae:.4f}→{after.mae:.4f}；不再接受 mixed-K 的“均值命中真值”。"
        ),
    )
    return pool, corrected, rank0, rank_mixed, rank_consistent, metric_df, recall_df


def old_certified_entries(name_to_idx: dict[str, int]):
    old = pd.read_csv(R68 / "outputs/triples_certified.csv")
    old["indices"] = [
        tuple(sorted((name_to_idx[left], name_to_idx[middle], name_to_idx[right])))
        for left, middle, right in zip(old.fi, old.fj, old.fk)
    ]
    truth2 = pd.read_csv(R68 / "outputs/synergy2_perconf.csv")
    pair_truth = {
        (tuple(sorted((row.fi, row.fj))), row.conf): row.synergy for row in truth2.itertuples()
    }
    max_children = []
    for row in old.itertuples():
        children = itertools.combinations((row.fi, row.fj, row.fk), 2)
        values = [pair_truth.get((tuple(sorted(pair)), row.conf), 0.0) for pair in children]
        max_children.append(max(values))
    old["syn2_true_max"] = max_children
    return old[["indices", "conf", "syn3_true", "syn2_true_max", "group"]]


def weighted_ratio(frame: pd.DataFrame, numerator: str, denominator: str):
    top = float((frame.weight * frame[numerator].astype(float)).sum())
    bottom = float((frame.weight * frame[denominator].astype(float)).sum())
    return top / bottom if bottom else np.nan


def correct_sampling(
    pool: pd.DataFrame,
    corrected: pd.DataFrame,
    rank0: pd.Series,
    rank_mixed: pd.Series,
    rank_consistent: pd.Series,
    name_to_idx: dict[str, int],
):
    section("阶段 3：抽样与统计单位修正")
    log(
        "SAMPLE-UNIT",
        "START",
        note="397 个旧认证视为已知层；1800 个视为剩余 12847 个空间的简单随机样本",
    )
    old = old_certified_entries(name_to_idx)
    extension = pool[pool.group == "triple_rand_ext"][
        ["indices", "conf", "syn3_true", "syn2_true_max", "group"]
    ].copy()
    old["weight"] = 1.0
    extension["weight"] = REMAINDER_WEIGHT
    design_entries = pd.concat([old, extension], ignore_index=True)
    design_entries["strong"] = design_entries.syn3_true > STRONG3
    design_entries["pure"] = design_entries.strong & (
        design_entries.syn2_true_max <= STRONG3
    )

    triple_design = (
        design_entries.groupby(["indices", "group"], as_index=False)
        .agg(
            syn3_true=("syn3_true", "max"),
            syn2_true_max=("syn2_true_max", "max"),
            weight=("weight", "first"),
        )
    )
    triple_design["strong"] = triple_design.syn3_true > STRONG3
    triple_design["pure"] = triple_design.strong & (
        triple_design.syn2_true_max <= STRONG3
    )

    current_entries = pool.copy()
    current_entries["strong"] = current_entries.syn3_true > STRONG3
    current_entries["pure"] = current_entries.strong & (
        current_entries.syn2_true_max <= STRONG3
    )
    current_triples = (
        current_entries.groupby("indices", as_index=False)
        .agg(syn3_true=("syn3_true", "max"), syn2_true_max=("syn2_true_max", "max"))
    )
    current_triples["strong"] = current_triples.syn3_true > STRONG3
    current_triples["pure"] = current_triples.strong & (
        current_triples.syn2_true_max <= STRONG3
    )

    summary = {
        "design": {
            "all_triples": N_ALL_TRIPLES,
            "exact_known_triples": N_EXACT,
            "remainder_triples": N_REMAINDER,
            "remainder_sample": N_REMAINDER_SAMPLE,
            "remainder_weight": REMAINDER_WEIGHT,
        },
        "reported_pool_unweighted": {
            "strong_entry_rate": float(current_entries.strong.mean()),
            "strong_triple_rate": float(current_triples.strong.mean()),
            "pure_share_among_strong_entries": float(
                current_entries.loc[current_entries.strong, "pure"].mean()
            ),
            "pure_share_among_strong_triples": float(
                current_triples.loc[current_triples.strong, "pure"].mean()
            ),
        },
        "stratified_population_estimate": {
            "strong_entry_rate": float(
                (design_entries.weight * design_entries.strong).sum()
                / (N_ALL_TRIPLES * 12)
            ),
            "strong_triple_rate": float(
                (triple_design.weight * triple_design.strong).sum() / N_ALL_TRIPLES
            ),
            "pure_share_among_strong_entries": float(
                weighted_ratio(design_entries, "pure", "strong")
            ),
            "pure_share_among_strong_triples": float(
                weighted_ratio(triple_design, "pure", "strong")
            ),
        },
    }

    recall_rows = []
    strong_design = triple_design[triple_design.strong].copy()
    for fraction in tuple(np.arange(0.10, 1.00, 0.10)):
        n_keep = int(round(N_ALL_TRIPLES * fraction))
        for label, ranking in (
            ("k0", rank0),
            ("k25_mixed", rank_mixed),
            ("k25_consistent", rank_consistent),
        ):
            if n_keep > len(ranking):
                continue
            keep = set(ranking.index[:n_keep])
            hit = strong_design.indices.map(lambda indices: indices in keep)
            recall = float(
                (strong_design.weight * hit.astype(float)).sum()
                / strong_design.weight.sum()
            )
            recall_rows.append(
                {
                    "population": "stratified_population_estimate",
                    "estimator": label,
                    "retention": fraction,
                    "n_keep": n_keep,
                    "recall": recall,
                    "estimated_strong_triples": float(strong_design.weight.sum()),
                }
            )
    weighted_recall = pd.DataFrame(recall_rows)
    weighted_recall.to_csv(OUT / "syn3_recall_stratified.csv", index=False)
    (OUT / "sampling_unit_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    unweighted = summary["reported_pool_unweighted"]
    weighted = summary["stratified_population_estimate"]
    log(
        "SAMPLE-UNIT",
        "DECISION",
        note=(
            f"强三元组率：未加权池 {unweighted['strong_triple_rate']:.2%}，"
            f"分层总体估计 {weighted['strong_triple_rate']:.2%}；"
            f"纯高阶份额必须区分 entry {weighted['pure_share_among_strong_entries']:.2%} "
            f"与 triple {weighted['pure_share_among_strong_triples']:.2%}。"
        ),
    )
    return summary, weighted_recall


def audit_retrain_protocol():
    section("阶段 4：重训真值协议静态审计")
    worker = (R77 / "scripts/retrain_worker.py").read_text(encoding="utf-8")
    uses_test_for_selection = "v = float(((model(Xte)" in worker
    report = {
        "worker": str(R77 / "scripts/retrain_worker.py"),
        "test_used_for_early_stopping": uses_test_for_selection,
        "final_test_reused_after_selection": uses_test_for_selection,
        "seed_count_for_h2_extension": 1,
        "status": "needs_new_gpu_retraining" if uses_test_for_selection else "not_detected",
        "required_fix": (
            "训练集更新参数；独立验证集早停；测试集只在模型冻结后评估一次；"
            "对阳性与阈值附近样本增加多 seed。"
        ),
    }
    (OUT / "retrain_protocol_audit.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    log(
        "TRUTH-AUDIT",
        "NOTE",
        note=(
            "检测到 retrain_worker 使用 Xte/Yte 选择最佳 epoch 后又在同一测试集报 R²；"
            "78 号只记录问题，不把旧真值伪装成已修复。"
        ),
    )
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(
        "ALL",
        "START",
        note="纯 CPU 复算：F2 停机单位、三阶同 K、抽样权重、entry/triple 口径",
    )
    with Timer() as timer:
        _, name_to_idx = field_context()
        truth2, evalset = build_second_order_index()
        f2_results, f2_summary = correct_f2(truth2)
        (
            pool,
            corrected,
            rank0,
            rank_mixed,
            rank_consistent,
            syn3_metrics,
            current_recall,
        ) = correct_syn3(evalset, name_to_idx)
        sampling, weighted_recall = correct_sampling(
            pool, corrected, rank0, rank_mixed, rank_consistent, name_to_idx
        )
        audit = audit_retrain_protocol()

        combined = {
            "f2": f2_summary,
            "sampling": sampling,
            "retrain_audit": audit,
            "syn3_metrics": syn3_metrics.to_dict(orient="records"),
        }
        (OUT / "correction_summary.json").write_text(
            json.dumps(json_safe(combined), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    log(
        "ALL",
        "DONE",
        note="所有修正输出完成；未使用 GPU，未改写 76/77 原始结果",
        elapsed_s=timer.elapsed,
    )


if __name__ == "__main__":
    main()
