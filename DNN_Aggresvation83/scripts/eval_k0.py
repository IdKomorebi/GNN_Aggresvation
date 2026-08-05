# -*- coding: utf-8 -*-
"""批量评估 D83 oracle 在全部 1/2/3 元集合上的 K=0 R²。"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from src.oracle import MLPOracle  # noqa: E402
from runlog import log  # noqa: E402
from train_tail_oracle import VARIANTS, load_data  # noqa: E402

MASK_BATCH = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    log("K0", "START", f"{args.variant}/seed{args.seed} 全低阶空间前向")

    data = load_data()
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    conf_names = data["confidential"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test = torch.as_tensor(
        data["test_data"][:, general_idx], dtype=torch.float32, device=device
    )
    y_test = data["test_data"][:, conf_idx]

    checkpoint = torch.load(
        ROOT / "outputs" / f"oracle_{args.variant}_seed{args.seed}.pt",
        map_location=device,
        weights_only=False,
    )
    model = MLPOracle(n_general, n_conf)
    model.load_state_dict(checkpoint["state"])
    model = model.to(device).eval()

    keys = (
        [(index,) for index in range(n_general)]
        + list(combinations(range(n_general), 2))
        + list(combinations(range(n_general), 3))
    )
    rows: list[dict[str, object]] = []
    n_test = len(x_test)
    target_total = ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    with torch.no_grad():
        for begin in range(0, len(keys), MASK_BATCH):
            chunk = keys[begin : begin + MASK_BATCH]
            batch = len(chunk)
            masks_np = np.zeros((batch, n_general), dtype=np.float32)
            for row, key in enumerate(chunk):
                masks_np[row, list(key)] = 1.0
            masks = torch.as_tensor(masks_np, device=device)
            expanded_x = (
                x_test.unsqueeze(0)
                .expand(batch, n_test, n_general)
                .reshape(batch * n_test, n_general)
            )
            expanded_masks = (
                masks.unsqueeze(1)
                .expand(batch, n_test, n_general)
                .reshape(batch * n_test, n_general)
            )
            prediction = (
                model(expanded_x, expanded_masks)
                .reshape(batch, n_test, n_conf)
                .cpu()
                .numpy()
            )
            residual = ((prediction - y_test[None, :, :]) ** 2).sum(axis=1)
            r2 = np.clip(1.0 - residual / target_total[None, :], 0.0, None)
            for row, key in enumerate(chunk):
                padded = list(key) + [-1] * (3 - len(key))
                for conf_pos, conf_name in enumerate(conf_names):
                    rows.append(
                        {
                            "i": padded[0],
                            "j": padded[1],
                            "k": padded[2],
                            "size": len(key),
                            "conf": conf_name,
                            "est": float(r2[row, conf_pos]),
                        }
                    )

    output = ROOT / "outputs" / f"k0_{args.variant}_seed{args.seed}.parquet"
    pd.DataFrame(rows).to_parquet(output, index=False)
    elapsed = time.time() - started
    log("K0", "DONE", f"{args.variant}/seed{args.seed} 完成", elapsed_s=elapsed)
    print(f"完成 {output.name}：{len(rows)} 行 [{elapsed:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
