#!/usr/bin/env python3
"""Merge LoRA with subspace estimates, recompute metrics, and make final figures."""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from common import load_data, load_design  # noqa: E402


def safe_spearman(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


def task_id_from_indices(indices):
    return "q_" + "_".join(f"{x:02d}" for x in sorted(indices))


def main() -> None:
    design = load_design()
    split = design["splits"]
    tasks = {task["task_id"]: task for task in design["tasks"]}
    data, _ = load_data(__import__("torch").device("cpu"))
    conf_names = list(data["confidential"])

    truth_long = pd.read_csv(REPO / "DNN_Aggresvation69/outputs/truth_long.csv")
    truth_lookup = {
        (row.sid, row.conf): float(row.dnn)
        for row in truth_long.itertuples(index=False)
    }
    truth_by_task = {
        task_id: np.asarray(
            [truth_lookup[(task["sid69"], conf)] for conf in conf_names],
            dtype=np.float32,
        )
        for task_id, task in tasks.items()
    }

    estimates = pd.read_csv(ROOT / "outputs/estimates.csv.gz")
    lora_frames = []
    for path in sorted((ROOT / "outputs").glob("lora_lr*_shard*of*.csv.gz")):
        frame = pd.read_csv(path)
        if "lr" not in frame:
            tag = path.name.split("lora_lr", 1)[1].split("_shard", 1)[0]
            frame["lr"] = float(tag.replace("p", "."))
        lora_frames.append(frame)
    lora = pd.concat(lora_frames, ignore_index=True)
    if lora["task_id"].nunique() != len(
        set(split["validation_all"]) | set(split["audit_all"])
    ):
        raise RuntimeError("LoRA output does not cover every evaluation task")

    lora_rows = []
    audit_all = set(split["audit_all"])
    audit_targets = set(split["audit_targets"])
    val_targets = set(split["validation_targets"])
    for row in lora.itertuples(index=False):
        task = tasks[row.task_id]
        split_name = "audit" if row.task_id in audit_all else "validation"
        role = (
            "target"
            if row.task_id in audit_targets or row.task_id in val_targets
            else "parent"
        )
        lora_rows.append({
            "task_id": row.task_id,
            "sid69": task["sid69"],
            "split": split_name,
            "role": role,
            "size": task["size"],
            "group": task["group"],
            "method": f"lora_lr{float(row.lr):g}_K{int(row.K)}",
            "rank": int(row.rank),
            "conf": row.conf,
            "est": float(row.est),
            "truth": truth_lookup[(task["sid69"], row.conf)],
        })
    estimates = pd.concat([estimates, pd.DataFrame(lora_rows)], ignore_index=True)
    estimates.to_csv(
        ROOT / "outputs/all_estimates.csv.gz", index=False, compression="gzip"
    )

    direct_rows = []
    target_frame = estimates[estimates["role"] == "target"].copy()
    for keys, frame in target_frame.groupby(["split", "method", "rank"]):
        split_name, method, rank = keys
        error = frame["est"].to_numpy() - frame["truth"].to_numpy()
        direct_rows.append({
            "split": split_name,
            "method": method,
            "rank": int(rank),
            "group": "all",
            "n_values": len(frame),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "spearman": safe_spearman(frame["est"], frame["truth"]),
        })
    for keys, frame in target_frame.groupby(["split", "method", "rank", "group"]):
        split_name, method, rank, group = keys
        error = frame["est"].to_numpy() - frame["truth"].to_numpy()
        direct_rows.append({
            "split": split_name,
            "method": method,
            "rank": int(rank),
            "group": group,
            "n_values": len(frame),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "spearman": safe_spearman(frame["est"], frame["truth"]),
        })
    direct = pd.DataFrame(direct_rows)
    direct.to_csv(ROOT / "outputs/all_direct_metrics.csv", index=False)

    lookup = {
        (row.task_id, row.method, int(row.rank), row.conf): float(row.est)
        for row in estimates.itertuples(index=False)
    }
    method_keys = sorted(set(zip(estimates["method"], estimates["rank"])))
    detail_rows = []
    for split_name in ("validation", "audit"):
        target_ids = (
            split["validation_targets"] if split_name == "validation"
            else split["audit_targets"]
        )
        for task_id in target_ids:
            task = tasks[task_id]
            if task["size"] not in (2, 3):
                continue
            if task["size"] == 2:
                parent_ids = [task_id_from_indices([x]) for x in task["indices"]]
            else:
                parent_ids = [
                    task_id_from_indices(x)
                    for x in combinations(task["indices"], task["size"] - 1)
                ]
            truth_syn = truth_by_task[task_id] - np.stack(
                [truth_by_task[parent] for parent in parent_ids]
            ).max(0)
            for method, rank in method_keys:
                try:
                    child = np.asarray([
                        lookup[(task_id, method, int(rank), conf)]
                        for conf in conf_names
                    ])
                    parents = np.stack([
                        [
                            lookup[(parent, method, int(rank), conf)]
                            for conf in conf_names
                        ]
                        for parent in parent_ids
                    ])
                except KeyError:
                    continue
                est_syn = child - parents.max(0)
                for conf_pos, conf in enumerate(conf_names):
                    detail_rows.append({
                        "split": split_name,
                        "task_id": task_id,
                        "order": task["size"],
                        "group": task["group"],
                        "method": method,
                        "rank": int(rank),
                        "conf": conf,
                        "est_syn": float(est_syn[conf_pos]),
                        "truth_syn": float(truth_syn[conf_pos]),
                    })
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(
        ROOT / "outputs/all_synergy_detail.csv.gz", index=False, compression="gzip"
    )

    metrics_rows = []
    for keys, frame in detail.groupby(["split", "order", "method", "rank"]):
        split_name, order, method, rank = keys
        error = frame["est_syn"].to_numpy() - frame["truth_syn"].to_numpy()
        task_score = frame.groupby("task_id").agg(
            est_score=("est_syn", "max"), truth_score=("truth_syn", "max")
        ).reset_index()
        pool_n = max(1, int(math.ceil(0.30 * len(task_score))))
        pool = set(task_score.nlargest(pool_n, "est_score")["task_id"])
        for threshold in (0.05, 0.10, 0.20):
            strong = set(
                task_score.loc[task_score["truth_score"] > threshold, "task_id"]
            )
            metrics_rows.append({
                "split": split_name,
                "order": int(order),
                "method": method,
                "rank": int(rank),
                "n_tasks": len(task_score),
                "per_conf_mae": float(np.mean(np.abs(error))),
                "per_conf_bias": float(np.mean(error)),
                "per_conf_spearman": safe_spearman(
                    frame["est_syn"], frame["truth_syn"]
                ),
                "maxconf_spearman": safe_spearman(
                    task_score["est_score"], task_score["truth_score"]
                ),
                "pool_fraction": 0.30,
                "truth_threshold": threshold,
                "n_strong": len(strong),
                "strong_recall": (
                    len(strong & pool) / len(strong) if strong else np.nan
                ),
            })
    synergy = pd.DataFrame(metrics_rows)
    synergy.to_csv(ROOT / "outputs/all_synergy_metrics.csv", index=False)

    # Choose deployable variants using validation only. Two objectives are kept
    # separate because direct v and a small parent-child difference are not identical.
    deployable_prefix = ("gradmap_", "gradmask_", "scalar_grad")
    direct_val = direct[
        (direct["split"] == "validation")
        & (direct["group"] == "all")
        & direct["method"].str.startswith(deployable_prefix)
    ].sort_values(["mae", "spearman"], ascending=[True, False])
    syn_val = synergy[
        (synergy["split"] == "validation")
        & (synergy["order"] == 3)
        & (synergy["truth_threshold"] == 0.10)
        & synergy["method"].str.startswith(deployable_prefix)
    ].sort_values(["per_conf_mae", "per_conf_spearman"], ascending=[True, False])
    selected_v = direct_val.iloc[0]
    selected_syn = syn_val.iloc[0]
    lora_direct_val = direct[
        (direct["split"] == "validation")
        & (direct["group"] == "all")
        & direct["method"].str.startswith("lora_")
    ].sort_values(["mae", "spearman"], ascending=[True, False])
    lora_syn_val = synergy[
        (synergy["split"] == "validation")
        & (synergy["order"] == 3)
        & (synergy["truth_threshold"] == 0.10)
        & synergy["method"].str.startswith("lora_")
    ].sort_values(["per_conf_mae", "per_conf_spearman"], ascending=[True, False])
    lora_k1_direct_val = lora_direct_val[
        lora_direct_val["method"].str.endswith("_K1")
    ]
    lora_k1_syn_val = lora_syn_val[lora_syn_val["method"].str.endswith("_K1")]
    selected_lora_v = lora_direct_val.iloc[0]
    selected_lora_syn = lora_syn_val.iloc[0]
    selected_lora_k1_v = lora_k1_direct_val.iloc[0]
    selected_lora_k1_syn = lora_k1_syn_val.iloc[0]

    selections = {
        "selected_for_direct_v": {
            "method": selected_v.method,
            "rank": int(selected_v["rank"]),
            "validation_mae": float(selected_v.mae),
        },
        "selected_for_syn3": {
            "method": selected_syn.method,
            "rank": int(selected_syn["rank"]),
            "validation_per_conf_mae": float(selected_syn.per_conf_mae),
        },
        "selected_lora_for_direct_v": {
            "method": selected_lora_v.method,
            "rank": int(selected_lora_v["rank"]),
            "validation_mae": float(selected_lora_v.mae),
        },
        "selected_lora_for_syn3": {
            "method": selected_lora_syn.method,
            "rank": int(selected_lora_syn["rank"]),
            "validation_per_conf_mae": float(selected_lora_syn.per_conf_mae),
        },
        "selected_lora_K1_for_direct_v": {
            "method": selected_lora_k1_v.method,
            "rank": int(selected_lora_k1_v["rank"]),
            "validation_mae": float(selected_lora_k1_v.mae),
        },
        "selected_lora_K1_for_syn3": {
            "method": selected_lora_k1_syn.method,
            "rank": int(selected_lora_k1_syn["rank"]),
            "validation_per_conf_mae": float(selected_lora_k1_syn.per_conf_mae),
        },
    }
    (ROOT / "outputs/validation_selections.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    wanted = [
        ("full_K0", 0, "Oracle K0"),
        ("full_K1", 0, "Full K1"),
        ("full_K5", 0, "Full K5"),
        ("full_K25", 0, "Full K25"),
        ("projected_teacher", 32, "Projection ceiling r32"),
        ("projected_teacher", 128, "Projection ceiling r128"),
        ("projected_teacher", 320, "Projection ceiling r320"),
        (selected_v.method, int(selected_v["rank"]), "Selected one-grad (v)"),
        (selected_syn.method, int(selected_syn["rank"]), "Selected one-grad (syn3)"),
        (
            selected_lora_k1_v.method,
            int(selected_lora_k1_v["rank"]),
            "Selected LoRA K1 (v)",
        ),
        (
            selected_lora_k1_syn.method,
            int(selected_lora_k1_syn["rank"]),
            "Selected LoRA K1 (syn3)",
        ),
        (
            selected_lora_v.method,
            int(selected_lora_v["rank"]),
            "Selected LoRA all-K (v)",
        ),
        (
            selected_lora_syn.method,
            int(selected_lora_syn["rank"]),
            "Selected LoRA all-K (syn3)",
        ),
    ]
    comparison_rows = []
    for method, rank, label in wanted:
        drow = direct[
            (direct["split"] == "audit")
            & (direct["group"] == "all")
            & (direct["method"] == method)
            & (direct["rank"] == rank)
        ]
        if drow.empty:
            continue
        output = {
            "label": label, "method": method, "rank": rank,
            "direct_mae": float(drow.iloc[0].mae),
            "direct_bias": float(drow.iloc[0].bias),
            "direct_spearman": float(drow.iloc[0].spearman),
        }
        for order in (2, 3):
            srow = synergy[
                (synergy["split"] == "audit")
                & (synergy["order"] == order)
                & (synergy["truth_threshold"] == 0.10)
                & (synergy["method"] == method)
                & (synergy["rank"] == rank)
            ]
            if not srow.empty:
                output.update({
                    f"syn{order}_mae": float(srow.iloc[0].per_conf_mae),
                    f"syn{order}_spearman": float(srow.iloc[0].per_conf_spearman),
                    f"syn{order}_maxconf_spearman": float(
                        srow.iloc[0].maxconf_spearman
                    ),
                    f"syn{order}_recall30": float(srow.iloc[0].strong_recall),
                    f"syn{order}_nstrong": int(srow.iloc[0].n_strong),
                })
        comparison_rows.append(output)
    comparison = pd.DataFrame(comparison_rows).drop_duplicates(
        subset=["method", "rank"], keep="first"
    )
    comparison.to_csv(ROOT / "outputs/main_comparison.csv", index=False)

    # Geometric compression, direct fidelity, and the actual parent-child target.
    diagnostics = pd.read_csv(ROOT / "outputs/subspace_diagnostics.csv")
    spectrum = diagnostics.pivot(index="rank", columns="split", values="mean").reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
    axes[0].plot(spectrum["rank"], spectrum["train"], marker="o", label="train tasks")
    axes[0].plot(
        spectrum["rank"], spectrum["validation_audit"], marker="o",
        label="unseen tasks"
    )
    axes[0].set_xlabel("subspace rank")
    axes[0].set_ylabel("fraction of K25 update energy")
    axes[0].set_ylim(0.68, 1.01)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    plot = comparison[
        comparison["label"].isin([
            "Oracle K0", "Full K5", "Full K25", "Projection ceiling r32",
            "Projection ceiling r128", "Projection ceiling r320",
            "Selected one-grad (v)", "Selected LoRA K1 (v)",
            "Selected LoRA all-K (v)",
        ])
    ]
    axes[1].barh(plot["label"], plot["direct_mae"], color="#4C78A8")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("audit direct v MAE (lower is better)")
    axes[1].grid(axis="x", alpha=0.25)
    axes[2].barh(plot["label"], plot["syn3_mae"], color="#F58518")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("audit syn3 MAE (lower is better)")
    axes[2].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/subspace_summary.png", dpi=180)
    plt.close(fig)

    print(json.dumps(selections, indent=2, ensure_ascii=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
