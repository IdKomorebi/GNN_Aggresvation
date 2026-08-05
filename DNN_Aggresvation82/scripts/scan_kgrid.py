# -*- coding: utf-8 -*-
"""对82号K0胜者做全部低阶/三元组同轨迹K网格微调。"""
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
from train_oracle import load_data, per_conf_r2  # noqa: E402

KGRID = (0, 1, 5, 10, 25, 50)
BATCH = 256
LR = 1e-3
WD = 5e-4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--kind", choices=("low", "triple"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    args = parser.parse_args()
    if args.kind == "low":
        assert args.shard == 0 and args.nshard == 1
    started = time.time()
    log(
        "KGRID",
        "START",
        note=f"{args.scheme}/{args.kind} shard{args.shard}/{args.nshard}",
        scheme=args.scheme,
        kind=args.kind,
        shard=args.shard,
        nshard=args.nshard,
    )

    data = load_data()
    general_idx = np.asarray(data["general_indices"])
    conf_idx = np.asarray(data["confidential_indices"])
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    conf_names = data["confidential"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.as_tensor(
        data["train_data"][:, general_idx], dtype=torch.float32, device=device
    )
    y_train = torch.as_tensor(
        data["train_data"][:, conf_idx], dtype=torch.float32, device=device
    )
    x_test = torch.as_tensor(
        data["test_data"][:, general_idx], dtype=torch.float32, device=device
    )
    y_test = data["test_data"][:, conf_idx]
    checkpoint = torch.load(
        ROOT / "outputs" / f"oracle_{args.scheme}_seed0.pt",
        map_location=device,
        weights_only=False,
    )

    if args.kind == "low":
        all_keys = [(index,) for index in range(n_general)] + list(
            combinations(range(n_general), 2)
        )
    else:
        all_keys = list(combinations(range(n_general), 3))
    keys = [
        key
        for position, key in enumerate(all_keys)
        if position % args.nshard == args.shard
    ]
    rows: list[dict[str, object]] = []
    for done, key in enumerate(keys, 1):
        mask = torch.zeros(1, n_general, device=device)
        mask[0, list(key)] = 1.0
        train_mask = mask.expand(len(x_train), -1)
        test_mask = mask.expand(len(x_test), -1)
        torch.manual_seed(0)
        np.random.seed(0)
        model = MLPOracle(n_general, n_conf)
        model.load_state_dict(checkpoint["state"])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        def snapshot(k_value: int) -> None:
            model.eval()
            with torch.no_grad():
                r2 = per_conf_r2(
                    model(x_test, test_mask).cpu().numpy(),
                    y_test,
                )
            padded = list(key) + [-1] * (3 - len(key))
            for conf_pos, conf_name in enumerate(conf_names):
                rows.append(
                    {
                        "i": padded[0],
                        "j": padded[1],
                        "k": padded[2],
                        "size": len(key),
                        "K": k_value,
                        "conf": conf_name,
                        "est": float(r2[conf_pos]),
                    }
                )

        snapshot(0)
        rng = np.random.RandomState(0)
        step = 0
        while step < max(KGRID):
            order = rng.permutation(len(x_train))
            for begin in range(0, len(order), BATCH):
                batch_idx = torch.as_tensor(
                    order[begin : begin + BATCH],
                    device=device,
                )
                model.train()
                optimizer.zero_grad()
                loss = (
                    (
                        model(x_train[batch_idx], train_mask[batch_idx])
                        - y_train[batch_idx]
                    )
                    ** 2
                ).mean()
                loss.backward()
                optimizer.step()
                step += 1
                if step in KGRID:
                    snapshot(step)
                if step >= max(KGRID):
                    break
        if done % 250 == 0:
            print(
                f"{args.scheme}/{args.kind} shard{args.shard}: "
                f"{done}/{len(keys)} [{time.time() - started:.0f}s]",
                flush=True,
            )

    output = (
        ROOT
        / "outputs"
        / (
            f"kgrid_{args.kind}_{args.scheme}_seed0_"
            f"shard{args.shard}of{args.nshard}.csv.gz"
        )
    )
    pd.DataFrame(rows).to_csv(output, index=False, compression="gzip")
    elapsed = time.time() - started
    log(
        "KGRID",
        "DONE",
        note=f"{args.scheme}/{args.kind} shard{args.shard}完成",
        elapsed_s=elapsed,
        scheme=args.scheme,
        kind=args.kind,
        shard=args.shard,
        n_rows=len(rows),
    )
    print(f"完成 {output.name} [{elapsed:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
