"""61 号流水线总入口：建边 -> 链敏感度 -> 链式攻击验证 -> 报告。

用法（在 DNN_Aggresvation61 目录下）:
    python scripts/run_all.py [--config base.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chain import chain_sensitivity, format_chain, walk_sensitivity
from src.data_processing import load_and_preprocess, shuffle_split
from src.edges import build_edge_matrix
from src.validate import run_validation, score_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "base.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    out_dir = ROOT / cfg["dataset"]["output_dir"]
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()

    print("[1/4] 加载数据")
    csv_path = (ROOT / cfg["dataset"]["csv_path"]).resolve()
    df, general, confidential = load_and_preprocess(
        str(csv_path),
        cfg["fields"]["drop_columns"],
        cfg["fields"]["confidential"],
        cfg["fields"]["drop_constant_columns"],
    )
    train_df, test_df = shuffle_split(
        df, cfg["dataset"]["train_ratio"], cfg["dataset"]["split_seed"]
    )
    print(f"  字段: general={len(general)}, confidential={len(confidential)}; "
          f"train={len(train_df)}, test={len(test_df)}")

    print("[2/4] 成对可推断度建边")
    W, models = build_edge_matrix(train_df, test_df, general, confidential,
                                  cfg["edges"])
    W.to_csv(out_dir / "edge_matrix.csv")

    print("[3/4] 推断链敏感度")
    s_cfg = cfg["sensitivity"]
    scores, chains = chain_sensitivity(
        W, general, confidential, s_cfg["alpha"], s_cfg["edge_threshold"]
    )
    s_walk, walk_scale = walk_sensitivity(
        W, general, confidential, s_cfg["alpha"], s_cfg["edge_threshold"]
    )
    scores["s_walk"] = s_walk
    if walk_scale != 1.0:
        print(f"  注意: 矩阵游走谱半径>=1, 已缩放 {walk_scale:.4f}")

    print("[4/4] 链式攻击验证")
    val = run_validation(W, models, df, test_df, general, confidential,
                         cfg["validation"])
    merged = scores.join(val)
    merged.sort_values("s_max", ascending=False).to_csv(
        out_dir / "sensitivity.csv"
    )

    score_cols = ["s_max", "s_or", "s_walk", "s_1hop", "pearson_base"]
    alignment = score_alignment(merged, score_cols)

    # 泄露报告（中文，按 s_max 降序，每字段 top-k 推断链）
    lines = ["# 61 号：推断链敏感度泄露报告\n"]
    lines.append("按主敏感度 s_max（最强推断链强度）降序。\n")
    for field in merged.sort_values("s_max", ascending=False).index:
        row = merged.loc[field]
        lines.append(
            f"## {field}\n"
            f"- s_max={row['s_max']:.3f}, s_or={row['s_or']:.3f}, "
            f"s_walk={row['s_walk']:.3f}, 可达机密字段 "
            f"{int(row['n_reachable_conf'])}/{len(confidential)}\n"
            f"- 真实攻击: 直接={row['realized_direct']:.3f}, "
            f"两跳={row['realized_2hop']:.3f}\n"
        )
        field_chains = sorted(
            ((s, p) for (src, _), (s, p) in chains.items() if src == field),
            reverse=True,
        )[: s_cfg["top_chains_per_field"]]
        if field_chains:
            lines.append("主要泄露链:\n")
            for strength, path in field_chains:
                lines.append(f"- [{strength:.3f}] {format_chain(path, W)}\n")
        lines.append("\n")
    (out_dir / "chains_report.md").write_text("".join(lines))

    summary = {
        "config": cfg,
        "n_general": len(general),
        "n_confidential": len(confidential),
        "walk_scale": walk_scale,
        "spearman_vs_realized_chained": alignment,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\n==== 各打分与真实链式攻击成功率的 Spearman ====")
    for k, v in alignment.items():
        print(f"  {k:14s} rho={v['spearman']:+.4f} (p={v['p']:.2e})")
    print(f"\n完成, 耗时 {summary['runtime_sec']}s, 结果在 {out_dir}/")


if __name__ == "__main__":
    main()
