# -*- coding: utf-8 -*-
"""分析 K 网格，并从独立调参/测试划分中选择分层预算协议。"""
from __future__ import annotations

import ast
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R68 = REPO / "DNN_Aggresvation68"
R69 = REPO / "DNN_Aggresvation69"
R75 = REPO / "DNN_Aggresvation75"
R77 = REPO / "DNN_Aggresvation77"
OUT = ROOT / "outputs"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))

from src.data_processing import prepare_data  # noqa: E402
from runlog import log  # noqa: E402

KGRID = (0, 1, 5, 10, 25, 50)
STRONG = 0.10
N_ALL = 13244
N_EXACT = 397
N_REMAINDER = N_ALL - N_EXACT
N_REMAINDER_SAMPLE = 1800
REMAINDER_WEIGHT = N_REMAINDER / N_REMAINDER_SAMPLE
RETENTIONS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90)
SPLIT_SEED = 7901


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_corr(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    left_mean = weighted_mean(left, weights)
    right_mean = weighted_mean(right, weights)
    left_centered = left - left_mean
    right_centered = right - right_mean
    covariance = np.sum(weights * left_centered * right_centered)
    denominator = np.sqrt(
        np.sum(weights * left_centered**2) * np.sum(weights * right_centered**2)
    )
    return float(covariance / denominator) if denominator else np.nan


def weighted_spearman(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    return weighted_corr(rankdata(left), rankdata(right), weights)


def field_context():
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    return data, {name: idx for idx, name in enumerate(data["general"])}


def load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    low_paths = sorted(OUT.glob("kgrid_low_uniform_seed0_shard*.csv.gz"))
    triple_paths = sorted(OUT.glob("kgrid_triple_uniform_seed0_shard*.csv.gz"))
    if len(low_paths) != 1 or len(triple_paths) != 4:
        raise RuntimeError(
            f"K 网格输出不完整：low={len(low_paths)}，triple={len(triple_paths)}"
        )
    low = pd.concat([pd.read_csv(path) for path in low_paths], ignore_index=True)
    triples = pd.concat([pd.read_csv(path) for path in triple_paths], ignore_index=True)
    expected_low = 990 * len(KGRID) * 12
    expected_triples = N_ALL * len(KGRID) * 12
    assert len(low) == expected_low, (len(low), expected_low)
    assert len(triples) == expected_triples, (len(triples), expected_triples)
    assert set(low.K.unique()) == set(KGRID)
    assert set(triples.K.unique()) == set(KGRID)
    return low, triples


def make_syn3(low: pd.DataFrame, triples: pd.DataFrame) -> pd.DataFrame:
    pairs = low[low["size"] == 2][["i", "j", "K", "conf", "est"]].copy()
    pair_map = {
        (int(row.i), int(row.j), int(row.K), row.conf): float(row.est)
        for row in pairs.itertuples()
    }
    child_max = []
    for row in triples.itertuples():
        children = (
            pair_map[(min(row.i, row.j), max(row.i, row.j), row.K, row.conf)],
            pair_map[(min(row.i, row.k), max(row.i, row.k), row.K, row.conf)],
            pair_map[(min(row.j, row.k), max(row.j, row.k), row.K, row.conf)],
        )
        child_max.append(max(children))
    result = triples[["i", "j", "k", "K", "conf", "est"]].copy()
    result["best_pair_est"] = child_max
    result["syn3_est"] = result.est - result.best_pair_est
    result.to_csv(OUT / "syn3_entries_kgrid.csv.gz", index=False, compression="gzip")
    return result


def load_truth(name_to_idx: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(R68 / "outputs/triples_certified.csv")
    old["indices"] = [
        tuple(sorted((name_to_idx[left], name_to_idx[middle], name_to_idx[right])))
        for left, middle, right in zip(old.fi, old.fj, old.fk)
    ]
    old = old[["indices", "conf", "syn3_true"]].copy()
    old["source"] = "exact397"
    old["weight"] = 1.0

    extension = pd.read_csv(R77 / "outputs/h2_unbiased_pool.csv")
    extension = extension[extension.group == "triple_rand_ext"].copy()
    extension["indices"] = extension.ix.map(ast.literal_eval)
    extension = extension[["indices", "conf", "syn3_true"]]
    extension["source"] = "remainder1800"
    extension["weight"] = REMAINDER_WEIGHT

    entries = pd.concat([old, extension], ignore_index=True)
    assert entries.groupby("source")["indices"].nunique().to_dict() == {
        "exact397": 397,
        "remainder1800": 1800,
    }
    triples = (
        entries.groupby(["indices", "source"], as_index=False)
        .agg(syn3_true=("syn3_true", "max"), weight=("weight", "first"))
    )
    triples["strong"] = triples.syn3_true > STRONG
    return entries, triples


def split_truth(truth: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(SPLIT_SEED)
    result = truth.copy()
    result["split"] = ""
    # 同时按采样层与阳性标签分层，避免某个隔离集碰巧几乎没有强协同。
    for (_, _), indices in result.groupby(["source", "strong"]).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        rng.shuffle(positions)
        cut = int(round(0.60 * len(positions)))
        result.loc[positions[:cut], "split"] = "tune"
        result.loc[positions[cut:], "split"] = "test"
    assert not (result.split == "").any()
    return result


def triple_and_entry_metrics(
    syn3: pd.DataFrame, truth_entries: pd.DataFrame, truth_triples: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    syn3 = syn3.copy()
    syn3["indices"] = list(zip(syn3.i, syn3.j, syn3.k))
    score = (
        syn3.groupby(["indices", "K"], as_index=False)
        .agg(score=("syn3_est", "max"))
        .pivot(index="indices", columns="K", values="score")
        .sort_index()
    )
    assert score.shape == (N_ALL, len(KGRID))

    triple_design = truth_triples.merge(
        score.reset_index(), on="indices", how="left", validate="one_to_one"
    )
    metric_rows = []
    rank_rows = []
    base_rate = float(
        (triple_design.weight * triple_design.strong.astype(float)).sum()
        / triple_design.weight.sum()
    )
    for k_value in KGRID:
        y_true = triple_design.strong.astype(int).to_numpy()
        y_score = triple_design[k_value].to_numpy()
        weight = triple_design.weight.to_numpy()
        metric_rows.append(
            {
                "unit": "triple_max",
                "K": k_value,
                "roc_auc": roc_auc_score(y_true, y_score, sample_weight=weight),
                "average_precision": average_precision_score(
                    y_true, y_score, sample_weight=weight
                ),
                "weighted_spearman": weighted_spearman(
                    triple_design.syn3_true.to_numpy(), y_score, weight
                ),
                "mae": weighted_mean(
                    np.abs(y_score - triple_design.syn3_true.to_numpy()), weight
                ),
                "bias": weighted_mean(
                    y_score - triple_design.syn3_true.to_numpy(), weight
                ),
                "strong_true_mean": weighted_mean(
                    triple_design.loc[triple_design.strong, "syn3_true"].to_numpy(),
                    triple_design.loc[triple_design.strong, "weight"].to_numpy(),
                ),
                "strong_est_mean": weighted_mean(
                    triple_design.loc[triple_design.strong, k_value].to_numpy(),
                    triple_design.loc[triple_design.strong, "weight"].to_numpy(),
                ),
                "base_rate": base_rate,
            }
        )

        ranking = score[k_value].sort_values(ascending=False)
        for retention in RETENTIONS:
            n_keep = int(round(N_ALL * retention))
            keep = set(ranking.index[:n_keep])
            hits = triple_design["indices"].map(lambda item: item in keep).to_numpy()
            positives = triple_design.strong.to_numpy()
            true_positive = float(np.sum(weight * hits * positives))
            estimated_kept_positive = true_positive
            estimated_all_positive = float(np.sum(weight * positives))
            recall = true_positive / estimated_all_positive
            precision = estimated_kept_positive / n_keep
            rank_rows.append(
                {
                    "K": k_value,
                    "retention": retention,
                    "n_keep": n_keep,
                    "recall": recall,
                    "precision": precision,
                    "f1": 2 * recall * precision / (recall + precision),
                    "enrichment": precision / base_rate,
                }
            )

        predicted = y_score > STRONG
        tp = float(np.sum(weight * predicted * y_true))
        fp = float(np.sum(weight * predicted * (1 - y_true)))
        fn = float(np.sum(weight * (1 - predicted) * y_true))
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        metric_rows[-1].update(
            {
                "threshold_precision": precision,
                "threshold_recall": recall,
                "threshold_f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else np.nan
                ),
            }
        )

    # entry 级误差：用于判断数值是否真的贴近逐 conf 真值。
    entry_score = syn3[["indices", "K", "conf", "syn3_est"]]
    entry_design = truth_entries.merge(
        entry_score, on=["indices", "conf"], how="left", validate="one_to_many"
    )
    entry_rows = []
    for k_value, group in entry_design.groupby("K"):
        weight = group.weight.to_numpy()
        estimate = group.syn3_est.to_numpy()
        actual = group.syn3_true.to_numpy()
        strong = actual > STRONG
        entry_rows.append(
            {
                "unit": "entry",
                "K": int(k_value),
                "weighted_spearman": weighted_spearman(actual, estimate, weight),
                "mae": weighted_mean(np.abs(estimate - actual), weight),
                "bias": weighted_mean(estimate - actual, weight),
                "n_strong_sample": int(strong.sum()),
                "strong_est_mean": weighted_mean(estimate[strong], weight[strong]),
                "strong_true_mean": weighted_mean(actual[strong], weight[strong]),
                "strong_threshold_detection": weighted_mean(
                    (estimate[strong] > STRONG).astype(float), weight[strong]
                ),
            }
        )

    metric_df = pd.DataFrame(metric_rows)
    rank_df = pd.DataFrame(rank_rows)
    entry_df = pd.DataFrame(entry_rows)
    metric_df.to_csv(OUT / "kgrid_triple_metrics.csv", index=False)
    rank_df.to_csv(OUT / "kgrid_recall_precision.csv", index=False)
    entry_df.to_csv(OUT / "kgrid_entry_metrics.csv", index=False)
    score.to_csv(OUT / "triple_score_kgrid.csv")
    return score, metric_df, rank_df


def truth_strength_recall(score: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.10, 0.12, 0.15, 0.20, np.inf]
    labels = ["非强", "0.10–0.12", "0.12–0.15", "0.15–0.20", ">0.20"]
    data = truth.copy()
    data["strength_bin"] = pd.cut(data.syn3_true, bins=bins, labels=labels, right=True)
    rows = []
    for k_value in KGRID:
        keep = set(score[k_value].nlargest(int(round(N_ALL * 0.30))).index)
        data["hit"] = data["indices"].map(lambda item: item in keep)
        for label, group in data.groupby("strength_bin", observed=True):
            rows.append(
                {
                    "K": k_value,
                    "strength_bin": str(label),
                    "n_sample": len(group),
                    "estimated_count": float(group.weight.sum()),
                    "recall_top30": weighted_mean(
                        group.hit.astype(float).to_numpy(), group.weight.to_numpy()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "recall_by_truth_strength.csv", index=False)
    return result


def schedule_keep(score_array: np.ndarray, k_to_col: dict[int, int], stages):
    active = np.arange(len(score_array), dtype=int)
    for k_value, retention in stages:
        n_keep = int(round(N_ALL * retention))
        if n_keep >= len(active):
            continue
        values = score_array[active, k_to_col[k_value]]
        chosen = np.argpartition(values, -n_keep)[-n_keep:]
        active = active[chosen]
    return active


def schedule_cost(stages) -> tuple[float, float]:
    """返回每个初始候选的平均更新数：重启实现 / 保存状态连续实现。"""
    restart = 0.0
    continuation = 0.0
    active_fraction = 1.0
    previous_k = 0
    for k_value, retention in stages:
        if k_value > 0:
            restart += active_fraction * k_value
            continuation += active_fraction * (k_value - previous_k)
        active_fraction = retention
        previous_k = k_value
    return restart, continuation


def enumerate_schedules(final_fraction: float):
    retain_grid = sorted(set((0.20, 0.30, 0.40, 0.50, 0.60, 0.80)))
    schedules = set()
    for final_k in KGRID:
        schedules.add(((final_k, final_fraction),))
    # 两层：第一层可为 K0（复现旧方案）或非零 K。
    for k1, k2 in itertools.combinations(KGRID, 2):
        for r1 in retain_grid:
            if r1 > final_fraction:
                schedules.add(((k1, r1), (k2, final_fraction)))
    # 三层只保留有实际意义的 K 序列，避免把参数搜索变成无边界试探。
    for k1, k2, k3 in itertools.combinations(KGRID, 3):
        for r1 in retain_grid:
            for r2 in retain_grid:
                if r1 > r2 > final_fraction:
                    schedules.add(((k1, r1), (k2, r2), (k3, final_fraction)))
    return sorted(schedules, key=str)


def bootstrap_recall_ci(frame: pd.DataFrame, hit_column: str, seed: int = 7902):
    rng = np.random.RandomState(seed)
    values = []
    strong = frame[frame.strong].copy()
    for _ in range(1000):
        sampled_parts = []
        for _, group in strong.groupby("source"):
            positions = rng.randint(0, len(group), size=len(group))
            sampled_parts.append(group.iloc[positions])
        sample = pd.concat(sampled_parts)
        values.append(
            weighted_mean(
                sample[hit_column].astype(float).to_numpy(), sample.weight.to_numpy()
            )
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_schedules(score: pd.DataFrame, truth: pd.DataFrame):
    keys = list(score.index)
    key_to_pos = {key: pos for pos, key in enumerate(keys)}
    score_array = score.loc[:, list(KGRID)].to_numpy()
    k_to_col = {k_value: pos for pos, k_value in enumerate(KGRID)}
    truth_pos = np.asarray([key_to_pos[key] for key in truth["indices"]], dtype=int)
    weight = truth.weight.to_numpy()
    positive = truth.strong.to_numpy()
    split = truth.split.to_numpy()
    rows = []

    for final_fraction in (0.05, 0.10, 0.20, 0.30):
        for stages in enumerate_schedules(final_fraction):
            keep = schedule_keep(score_array, k_to_col, stages)
            selected = np.zeros(N_ALL, dtype=bool)
            selected[keep] = True
            hit = selected[truth_pos]
            restart, continuation = schedule_cost(stages)
            record = {
                "schedule": " -> ".join(f"K{k}@{r:.0%}" for k, r in stages),
                "stages_json": json.dumps(stages),
                "n_stages": len(stages),
                "final_fraction": final_fraction,
                "final_k": stages[-1][0],
                "avg_updates_restart": restart,
                "avg_updates_continuation": continuation,
            }
            for scope in ("tune", "test", "all"):
                use = np.ones(len(truth), dtype=bool) if scope == "all" else split == scope
                strong_use = use & positive
                record[f"{scope}_recall"] = float(
                    np.sum(weight[strong_use] * hit[strong_use])
                    / np.sum(weight[strong_use])
                )
            rows.append(record)

    result = pd.DataFrame(rows).drop_duplicates("schedule")
    result.to_csv(OUT / "schedule_sweep.csv", index=False)

    final10 = result[result.final_fraction == 0.10].copy()
    final10 = final10.sort_values(["avg_updates_restart", "tune_recall"], ascending=[True, False])
    frontier = []
    best_recall = -np.inf
    for row in final10.itertuples():
        if row.tune_recall > best_recall + 1e-12:
            frontier.append(row._asdict())
            best_recall = row.tune_recall
    frontier_df = pd.DataFrame(frontier)
    frontier_df.to_csv(OUT / "schedule_pareto_top10.csv", index=False)

    old_label = "K0@30% -> K25@10%"
    old = final10[final10.schedule == old_label].iloc[0]
    tiers = [
        ("same_cost", float(old.avg_updates_restart)),
        ("balanced", 12.5),
        ("high_recall", 20.0),
    ]
    chosen_rows = []
    for tier, cap in tiers:
        eligible = final10[final10.avg_updates_restart <= cap + 1e-12]
        chosen = eligible.sort_values(
            ["tune_recall", "avg_updates_restart"], ascending=[False, True]
        ).iloc[0]
        chosen_rows.append({"tier": tier, **chosen.to_dict()})
    chosen_rows.append({"tier": "old_protocol", **old.to_dict()})
    chosen = pd.DataFrame(chosen_rows).drop_duplicates(["tier"])

    # 参数选择只看 tune；以下 test 是第一次用于最终汇报。
    truth_report = truth.copy()
    for row in chosen.itertuples():
        stages = json.loads(row.stages_json)
        keep = schedule_keep(score_array, k_to_col, stages)
        selected = np.zeros(N_ALL, dtype=bool)
        selected[keep] = True
        column = f"hit_{row.tier}"
        truth_report[column] = selected[truth_pos]
        test_frame = truth_report[truth_report.split == "test"]
        low, high = bootstrap_recall_ci(test_frame, column)
        chosen.loc[chosen.tier == row.tier, "test_recall_ci_low"] = low
        chosen.loc[chosen.tier == row.tier, "test_recall_ci_high"] = high
    # 这是单次 tune/test 划分下的 Top10 探索，不作为最终推荐。
    # 最终协议由 synthesize.py 结合 Top30 发现目标与 staged_rescue.py 收束。
    chosen.to_csv(OUT / "top10_tune_exploration.csv", index=False)
    return result, frontier_df, chosen


def main() -> None:
    log(
        "ANALYSIS",
        "START",
        note="合并同保真度 K 网格，按分层总体口径评价并在 tune/test 隔离下选择调度",
    )
    _, name_to_idx = field_context()
    low, triples = load_scores()
    syn3 = make_syn3(low, triples)
    truth_entries, truth_triples = load_truth(name_to_idx)
    truth_triples = split_truth(truth_triples)
    truth_triples.to_csv(OUT / "truth_design_split.csv", index=False)
    score, metrics, rank_metrics = triple_and_entry_metrics(
        syn3, truth_entries, truth_triples
    )
    strength = truth_strength_recall(score, truth_triples)
    schedules, frontier, chosen = evaluate_schedules(score, truth_triples)

    summary = {
        "kgrid": list(KGRID),
        "truth_design": {
            "exact_triples": N_EXACT,
            "remainder_sample": N_REMAINDER_SAMPLE,
            "remainder_weight": REMAINDER_WEIGHT,
            "split_seed": SPLIT_SEED,
            "split_counts": truth_triples.groupby(["split", "source", "strong"])
            .size()
            .to_dict(),
        },
        "k_metrics": metrics.to_dict(orient="records"),
        "top10": rank_metrics[rank_metrics.retention == 0.10].to_dict(orient="records"),
        "top10_tune_exploration": chosen.to_dict(orient="records"),
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    k0 = rank_metrics[(rank_metrics.K == 0) & (rank_metrics.retention == 0.30)].iloc[0]
    log(
        "ANALYSIS",
        "NOTE",
        note=(
            f"K0 top30 recall={k0.recall:.1%}；Top10单划分调度只作探索，"
            "不作为最终推荐，最终方案见 SYNTHESIS/STAGED-RESCUE。"
        ),
    )
    print(chosen[["tier", "schedule", "avg_updates_restart", "tune_recall", "test_recall"]])


if __name__ == "__main__":
    main()
