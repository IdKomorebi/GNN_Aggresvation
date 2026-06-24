"""
DNN_Aggresvation29 评估设置诊断脚本。

对比三种 train/test 切分方式下，General→Confidential 的可推断性上界：
1. 时序 70/30（当前 GNN 用的方式）：前 70% 训练，后 30% 测试
2. 随机 70/30 shuffle split（打破时序，同分布参照）
3. 滚动时序 CV（多窗口，定位漂移发生在哪个月）

对每种切分，用 Ridge 线性回归（alpha=1.0）报告每个 Confidential 字段的
测试 R²，作为"GNN 能达到的线性上界参照"。

输出：outputs/eval_diagnosis/<timestamp>/{results.csv, summary.json}
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, ShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import prepare_data


def _ridge_r2(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    alpha: float = 1.0,
) -> float:
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    return float(r2_score(y_te, model.predict(X_te)))


def run_temporal_split(di: dict, gen_idx: list[int], conf_idx: list[int]) -> list[dict]:
    """方式1: 时序 70/30（当前 GNN 用的方式）。"""
    tr, te = di["train_data"], di["test_data"]
    rows = []
    for ci, name in zip(conf_idx, di["confidential"]):
        r2 = _ridge_r2(tr[:, gen_idx], tr[:, ci], te[:, gen_idx], te[:, ci])
        rows.append({"field": name, "split": "temporal_70_30", "r2": r2})
    return rows


def run_shuffle_split(di: dict, gen_idx: list[int], conf_idx: list[int], seed: int = 42) -> list[dict]:
    """方式2: 随机 70/30 shuffle split，单次。"""
    all_data = np.vstack([di["train_data"], di["test_data"]])
    n = all_data.shape[0]
    ss = ShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr_idx, te_idx = next(ss.split(np.arange(n)))
    rows = []
    for ci, name in zip(conf_idx, di["confidential"]):
        r2 = _ridge_r2(
            all_data[tr_idx][:, gen_idx], all_data[tr_idx][:, ci],
            all_data[te_idx][:, gen_idx], all_data[te_idx][:, ci],
        )
        rows.append({"field": name, "split": "shuffle_70_30", "r2": r2})
    return rows


def run_shuffle_cv(di: dict, gen_idx: list[int], conf_idx: list[int], seed: int = 42) -> list[dict]:
    """方式2b: 5-fold shuffle CV，报告均值。"""
    all_data = np.vstack([di["train_data"], di["test_data"]])
    n = all_data.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for ci, name in zip(conf_idx, di["confidential"]):
        r2s = []
        for tr_idx, te_idx in kf.split(np.arange(n)):
            r2 = _ridge_r2(
                all_data[tr_idx][:, gen_idx], all_data[tr_idx][:, ci],
                all_data[te_idx][:, gen_idx], all_data[te_idx][:, ci],
            )
            r2s.append(r2)
        rows.append({
            "field": name,
            "split": "shuffle_5fold_mean",
            "r2": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "r2_min": float(np.min(r2s)),
            "r2_max": float(np.max(r2s)),
        })
    return rows


def run_rolling_temporal_cv(
    di: dict, gen_idx: list[int], conf_idx: list[int], n_splits: int = 4
) -> list[dict]:
    """方式3: 滚动时序 CV，按时间顺序切多个窗口。

    把全量数据按时间均分成 n_splits+1 段，每次用前 k 段训练、第 k+1 段测试。
    """
    all_data = np.vstack([di["train_data"], di["test_data"]])
    n = all_data.shape[0]
    seg = n // (n_splits + 1)
    rows = []
    for k in range(1, n_splits + 1):
        tr_start, tr_end = 0, k * seg
        te_start, te_end = k * seg, min((k + 1) * seg, n)
        split_name = f"rolling_fold{k}_train[0:{tr_end}]_test[{te_start}:{te_end}]"
        for ci, name in zip(conf_idx, di["confidential"]):
            r2 = _ridge_r2(
                all_data[:tr_end][:, gen_idx], all_data[:tr_end][:, ci],
                all_data[te_start:te_end][:, gen_idx], all_data[te_start:te_end][:, ci],
            )
            rows.append({"field": name, "split": split_name, "r2": r2})
    return rows


def main() -> None:
    import yaml

    config_path = PROJECT_ROOT / "configs" / "bipartite_allloss.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["csv_path"] = str(
        (PROJECT_ROOT / cfg["dataset"]["csv_path"]).resolve()
    )
    cfg["dataset"]["output_dir"] = "outputs"

    print("=" * 70)
    print("  DNN_Aggresvation29 评估设置诊断")
    print("=" * 70)

    di = prepare_data(cfg)
    gen_idx = list(range(di["n_general"]))
    conf_idx = di["confidential_indices"]
    n_total = len(di["train_data"]) + len(di["test_data"])
    print(f"  总样本: {n_total}, General: {di['n_general']}, Confidential: {di['n_confidential']}")
    print(f"  时序 split: train={len(di['train_data'])}, test={len(di['test_data'])}")

    all_rows: list[dict] = []

    print("\n  [1/3] 时序 70/30 split...")
    all_rows.extend(run_temporal_split(di, gen_idx, conf_idx))

    print("  [2/3] 随机 70/30 shuffle split...")
    all_rows.extend(run_shuffle_split(di, gen_idx, conf_idx))

    print("  [2b/3] 5-fold shuffle CV...")
    all_rows.extend(run_shuffle_cv(di, gen_idx, conf_idx))

    print("  [3/3] 滚动时序 CV (4 folds)...")
    all_rows.extend(run_rolling_temporal_cv(di, gen_idx, conf_idx, n_splits=4))

    df = pd.DataFrame(all_rows)

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "eval_diagnosis" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "results.csv", index=False)

    # 汇总表：每个字段 × 三种 split 的 R²
    pivot = df[df["split"].isin(["temporal_70_30", "shuffle_70_30", "shuffle_5fold_mean"])].pivot_table(
        index="field", columns="split", values="r2"
    )
    pivot = pivot.reindex(columns=["temporal_70_30", "shuffle_70_30", "shuffle_5fold_mean"])
    pivot["shuffle_vs_temporal_gap"] = pivot["shuffle_5fold_mean"] - pivot["temporal_70_30"]
    pivot = pivot.sort_values("shuffle_5fold_mean", ascending=False)
    pivot.to_csv(out_dir / "pivot_summary.csv")

    # 滚动时序 CV 汇总
    rolling = df[df["split"].str.startswith("rolling_")].pivot_table(
        index="field", columns="split", values="r2"
    )
    rolling.to_csv(out_dir / "rolling_cv.csv")

    # summary.json
    summary = {
        "n_total": n_total,
        "n_general": di["n_general"],
        "n_confidential": di["n_confidential"],
        "temporal_train_size": len(di["train_data"]),
        "temporal_test_size": len(di["test_data"]),
        "pivot_summary": pivot.to_dict(orient="index"),
        "description": {
            "temporal_70_30": "当前GNN用的方式：前70%时序训练，后30%测试。训练集1-9月，测试集10-12月",
            "shuffle_70_30": "随机70/30切分，打破时序。训练集和测试集同分布（都有1-12月数据）",
            "shuffle_5fold_mean": "5-fold随机CV均值，更稳健的同分布参照",
            "rolling_fold1-4": "滚动时序CV，每次用前k段训练第k+1段测试，定位漂移发生在哪个月",
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # 打印主结果表
    print("\n" + "=" * 90)
    print("  主结果：时序 split vs 随机 split vs 5-fold CV (Ridge alpha=1.0)")
    print("=" * 90)
    print(f"  {'field':<45s} {'temporal':>9s} {'shuffle':>9s} {'5fold':>9s} {'gap':>9s}")
    print("  " + "-" * 88)
    for field, row in pivot.iterrows():
        print(
            f"  {field:<45s} {row['temporal_70_30']:>9.4f} {row['shuffle_70_30']:>9.4f} "
            f"{row['shuffle_5fold_mean']:>9.4f} {row['shuffle_vs_temporal_gap']:>+9.4f}"
        )
    print()

    # 滚动时序 CV
    print("=" * 90)
    print("  滚动时序 CV（定位漂移）")
    print("=" * 90)
    rolling_cols = sorted(rolling.columns)
    header = f"  {'field':<40s}" + "".join(f" {c.split('_')[1]:>18s}" for c in rolling_cols)
    print(header)
    print("  " + "-" * (40 + 19 * len(rolling_cols)))
    for field, row in rolling.iterrows():
        vals = "".join(f" {row[c]:>18.4f}" for c in rolling_cols)
        print(f"  {field:<40s}{vals}")

    print(f"\n  结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()
