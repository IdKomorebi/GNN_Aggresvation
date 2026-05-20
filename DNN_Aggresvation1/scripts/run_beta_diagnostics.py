import os
import sys
import copy
import itertools
import numpy as np
import pandas as pd
import yaml

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import prepare_pipeline_data
from src.train import train_gnn_model


METRICS = ["Pearson", "Spearman", "Kendall", "NMI", "dCor"]


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def make_anchor_mask(base_data, anchor_count, strategy):
    s_init = base_data["s_init"]
    pseudo_y = base_data["pseudo_y"]
    gen_indices = np.where(s_init == 0.0)[0]
    conf_indices = np.where(s_init == 1.0)[0]

    mask = np.zeros_like(s_init, dtype=np.float32)
    mask[conf_indices] = 1.0

    label_source = np.array(["unlabeled_general"] * len(s_init), dtype=object)
    label_source[conf_indices] = "confidential_seed"

    count = min(int(anchor_count), len(gen_indices))
    if count <= 0:
        return mask, label_source

    sorted_gen = gen_indices[np.argsort(pseudo_y[gen_indices])]
    if strategy == "extreme":
        high_count = int(np.ceil(count * 0.7))
        low_count = count - high_count
        selected_low = sorted_gen[:low_count] if low_count > 0 else np.array([], dtype=int)
        selected_high = sorted_gen[-high_count:] if high_count > 0 else np.array([], dtype=int)
        selected = np.unique(np.concatenate([selected_low, selected_high]))
        label_source[selected_low] = "general_pseudo_low"
        label_source[selected_high] = "general_pseudo_high"
    elif strategy == "quantile":
        positions = np.linspace(0, len(sorted_gen) - 1, count)
        selected = np.unique(sorted_gen[np.round(positions).astype(int)])
        if len(selected) < count:
            fill = [idx for idx in sorted_gen if idx not in set(selected.tolist())]
            selected = np.array(list(selected) + fill[:count - len(selected)], dtype=int)
        label_source[selected] = "general_pseudo_quantile"
    else:
        raise ValueError(f"Unknown anchor strategy: {strategy}")

    mask[selected] = 1.0
    return mask, label_source


def with_anchors(base_data, anchor_count, strategy):
    data = copy.copy(base_data)
    mask, label_source = make_anchor_mask(base_data, anchor_count, strategy)
    data["supervision_mask"] = mask
    data["label_source"] = label_source
    data["y"] = base_data["pseudo_y"].copy()
    return data


def summarize_runs(rows):
    alphas = np.vstack([r["alpha"] for r in rows])
    spreads = alphas.max(axis=1) - alphas.min(axis=1)
    top_sets = [r["top5"] for r in rows]
    jaccards = []
    for i, j in itertools.combinations(range(len(top_sets)), 2):
        union = top_sets[i] | top_sets[j]
        jaccards.append(len(top_sets[i] & top_sets[j]) / len(union) if union else 0.0)
    return {
        "alpha_mean": alphas.mean(axis=0),
        "alpha_std": alphas.std(axis=0),
        "spread_mean": float(np.mean(spreads)),
        "spread_std": float(np.std(spreads)),
        "unsup_corr_mean": float(np.nanmean([r["unsup_corr"] for r in rows])),
        "unsup_mae_mean": float(np.mean([r["unsup_mae"] for r in rows])),
        "all_gen_mae_mean": float(np.mean([r["all_gen_mae"] for r in rows])),
        "top5_jaccard_mean": float(np.mean(jaccards)) if jaccards else 0.0,
        "loss_mean": float(np.mean([r["loss"] for r in rows])),
    }


def evaluate_config(data, config, seeds, hidden_dim, beta_lr, alpha_temperature):
    rows = []
    columns = np.array(data["columns"])
    s_init = data["s_init"]
    gen_mask = s_init == 0.0
    unsup_mask = (s_init == 0.0) & (data["supervision_mask"] == 0.0)

    for seed in seeds:
        result = train_gnn_model(
            data,
            hidden_dim=hidden_dim,
            num_layers=config["model"]["num_layers"],
            lr=config["training"]["lr"],
            epochs=config["training"]["epochs"],
            w_c=config["training"]["w_c"],
            lmbda=config["training"]["lmbda"],
            seed=seed,
            beta_lr=beta_lr,
            alpha_temperature=alpha_temperature,
            verbose=False,
        )
        y_pred = result["y_pred"]
        if unsup_mask.sum() > 2 and np.std(y_pred[unsup_mask]) > 1e-8 and np.std(data["pseudo_y"][unsup_mask]) > 1e-8:
            unsup_corr = float(np.corrcoef(y_pred[unsup_mask], data["pseudo_y"][unsup_mask])[0, 1])
        else:
            unsup_corr = np.nan
        top_indices = np.where(gen_mask)[0][np.argsort(y_pred[gen_mask])[-5:][::-1]]
        rows.append({
            "alpha": result["alpha"],
            "loss": result["history"]["loss"][-1],
            "unsup_corr": unsup_corr,
            "unsup_mae": float(np.mean(np.abs(y_pred[unsup_mask] - data["pseudo_y"][unsup_mask]))) if unsup_mask.any() else 0.0,
            "all_gen_mae": float(np.mean(np.abs(y_pred[gen_mask] - data["pseudo_y"][gen_mask]))),
            "top5": set(columns[top_indices]),
        })
    return summarize_runs(rows)


def main():
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pseudo_cfg = config.get("pseudo_label", {})
    raw_csv_path = resolve_path(config["data"].get("raw_csv_path", config["data"].get("csv_path")))
    processed_csv_path = config["data"].get("processed_csv_path")
    processed_csv_path = resolve_path(processed_csv_path) if processed_csv_path else None
    dnn_q_cache_dir = pseudo_cfg.get("dnn_q_cache_dir", os.path.join(config["outputs"]["dir"], "dnn_q_cache"))
    dnn_q_cache_dir = resolve_path(dnn_q_cache_dir)
    print("Preparing DNN-q data once...")
    base_data = prepare_pipeline_data(
        raw_csv_path=raw_csv_path,
        processed_csv_path=processed_csv_path,
        drop_columns=config["data"].get("drop_columns", []),
        confidential_columns=config["data"].get("confidential_columns", []),
        use_processed_cache=config["data"].get("use_processed_cache", True),
        K_neighbors=config["graph"]["k_neighbors"],
        theta=config["graph"]["theta"],
        downsample_size=config["data"]["downsample_size"],
        general_label_count=49,
        high_risk_ratio=pseudo_cfg.get("high_risk_ratio", 0.7),
        inference_weight=pseudo_cfg.get("inference_weight", 0.65),
        predictive_test_ratio=pseudo_cfg.get("predictive_test_ratio", 0.3),
        dnn_epochs=pseudo_cfg.get("dnn_epochs", 80),
        dnn_hidden_dim=pseudo_cfg.get("dnn_hidden_dim", 16),
        dnn_lr=pseudo_cfg.get("dnn_lr", 0.01),
        dnn_weight_decay=pseudo_cfg.get("dnn_weight_decay", 1e-4),
        dnn_seed=pseudo_cfg.get("dnn_seed", 2026),
        dnn_q_cache_dir=dnn_q_cache_dir,
    )

    seeds = [0, 1, 2]
    experiments = []
    for anchor_count in [20, 30, 40, 49]:
        for strategy in ["extreme", "quantile"]:
            experiments.append({
                "name": f"anchors={anchor_count},strategy={strategy},baseline",
                "anchor_count": anchor_count,
                "strategy": strategy,
                "hidden_dim": 64,
                "beta_lr": None,
                "alpha_temperature": 1.0,
            })
    for beta_lr in [0.01, 0.02, 0.05]:
        experiments.append({
            "name": f"anchors=20,quantile,beta_lr={beta_lr}",
            "anchor_count": 20,
            "strategy": "quantile",
            "hidden_dim": 64,
            "beta_lr": beta_lr,
            "alpha_temperature": 1.0,
        })
    for temperature in [0.7, 0.5, 0.3]:
        experiments.append({
            "name": f"anchors=20,quantile,temp={temperature}",
            "anchor_count": 20,
            "strategy": "quantile",
            "hidden_dim": 64,
            "beta_lr": 0.02,
            "alpha_temperature": temperature,
        })
    for hidden_dim in [16, 32]:
        experiments.append({
            "name": f"anchors=20,quantile,hidden={hidden_dim},beta_lr=0.02,temp=0.5",
            "anchor_count": 20,
            "strategy": "quantile",
            "hidden_dim": hidden_dim,
            "beta_lr": 0.02,
            "alpha_temperature": 0.5,
        })

    output_rows = []
    for idx, exp in enumerate(experiments, 1):
        print(f"[{idx}/{len(experiments)}] {exp['name']}")
        data = with_anchors(base_data, exp["anchor_count"], exp["strategy"])
        summary = evaluate_config(
            data,
            config,
            seeds=seeds,
            hidden_dim=exp["hidden_dim"],
            beta_lr=exp["beta_lr"],
            alpha_temperature=exp["alpha_temperature"],
        )
        row = {
            **exp,
            "loss_mean": summary["loss_mean"],
            "alpha_spread_mean": summary["spread_mean"],
            "alpha_spread_std": summary["spread_std"],
            "unsup_corr_mean": summary["unsup_corr_mean"],
            "unsup_mae_mean": summary["unsup_mae_mean"],
            "all_gen_mae_mean": summary["all_gen_mae_mean"],
            "top5_jaccard_mean": summary["top5_jaccard_mean"],
        }
        for metric, value in zip(METRICS, summary["alpha_mean"]):
            row[f"alpha_{metric}_mean"] = float(value)
        output_rows.append(row)
        alpha_text = ", ".join([f"{m}:{row[f'alpha_{m}_mean']:.3f}" for m in METRICS])
        print(
            f"  spread={row['alpha_spread_mean']:.4f}, "
            f"unsup_corr={row['unsup_corr_mean']:.3f}, "
            f"top5={row['top5_jaccard_mean']:.3f}, "
            f"alpha={alpha_text}"
        )

    out_dir = resolve_path(config["outputs"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "beta_diagnostics.csv")
    df = pd.DataFrame(output_rows)
    df.to_csv(out_path, index=False)
    print(f"Saved diagnostics to {out_path}")


if __name__ == "__main__":
    main()
