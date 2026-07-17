"""62 号第一步：建边 + 预计算两跳攻击张量，全部缓存到 outputs/。

产出（后续孤岛实验纯内存分析，不再训练/推断）：
  outputs/edge_matrix.csv     可推断度矩阵 W（general 行 × 全字段列）
  outputs/tensors.npz         R2_direct[nG,nC], R2_2hop[nG,nG,nC], pearson_direct[nG,nC],
                              以及 general/confidential 字段名
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import load_and_preprocess, shuffle_split
from src.edges import build_edge_matrix
from src.twohop import compute_twohop_tensor


def main() -> None:
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    out_dir = ROOT / cfg["dataset"]["output_dir"]
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()

    print("[1/3] 加载数据")
    csv_path = (ROOT / cfg["dataset"]["csv_path"]).resolve()
    df, general, confidential = load_and_preprocess(
        str(csv_path), cfg["fields"]["drop_columns"],
        cfg["fields"]["confidential"], cfg["fields"]["drop_constant_columns"],
    )
    train_df, test_df = shuffle_split(
        df, cfg["dataset"]["train_ratio"], cfg["dataset"]["split_seed"])
    print(f"  general={len(general)}, confidential={len(confidential)}; "
          f"train={len(train_df)}, test={len(test_df)}")

    print("[2/3] 成对可推断度建边（含 general->general 桥接边）")
    W, models = build_edge_matrix(train_df, test_df, general, confidential, cfg["edges"])
    W.to_csv(out_dir / "edge_matrix.csv")

    print("[3/3] 预计算两跳攻击张量 R2_2hop[i,j,c]")
    R2_direct, R2_2hop = compute_twohop_tensor(
        W, models, test_df, general, confidential, n_jobs=cfg["edges"]["n_jobs"])

    corr = df[general + confidential].corr().abs()
    pearson_direct = corr.loc[general, confidential].to_numpy().astype(np.float64)

    np.savez(
        out_dir / "tensors.npz",
        R2_direct=R2_direct, R2_2hop=R2_2hop, pearson_direct=pearson_direct,
        general=np.array(general), confidential=np.array(confidential),
    )
    dt = time.time() - t0
    print(f"\n完成，耗时 {dt:.1f}s")
    print(f"  R2_direct: {R2_direct.shape}, R2_2hop: {R2_2hop.shape}")
    print(f"  直接攻击 R²>0.5 的 (i,c) 对: {(R2_direct > 0.5).sum()}")
    # 两跳能否在某处超过直接（孤岛外的合理性检查，应几乎没有）
    best_2hop = np.nanmax(R2_2hop, axis=1)  # [nG, nC]
    gain = best_2hop - R2_direct
    print(f"  两跳最优 - 直接 的最大增益: {np.nanmax(gain):.4f} "
          f"(>0.05 的对数: {(gain > 0.05).sum()})  # 全数据下应接近 0（DPI）")


if __name__ == "__main__":
    main()
