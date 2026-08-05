# -*- coding: utf-8 -*-
"""对全部低阶集合或三元组做同轨迹 K 网格微调。

每个集合都从同一个 uniform/seed0 通用 oracle 出发，固定该集合的掩码，
连续更新到 K=50，并在 K={0,1,5,10,25,50} 保存测试 R²。

关键约束：
1. K 是优化器 mini-batch 更新次数，不是 epoch；
2. 每个集合独立复制通用权重，不把上一个集合的权重传给下一个；
3. 三阶协同在分析时严格使用 v_K(ijk)-max v_K(pair)，父子集合 K 相同。
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R75 = REPO / "DNN_Aggresvation75"
R77 = REPO / "DNN_Aggresvation77"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(ROOT / "src"))

from src.data_processing import prepare_data  # noqa: E402
from src.oracle import MLPOracle  # noqa: E402
from runlog import log  # noqa: E402

KGRID = (0, 1, 5, 10, 25, 50)
BATCH = 256
LR = 1e-3
WD = 5e-4


def per_conf_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = ((target - prediction) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return np.clip(1.0 - residual / total, 0.0, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("low", "triple"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--scheme", default="uniform")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()

    if args.kind == "low":
        assert args.shard == 0 and args.nshard == 1

    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    data = prepare_data(cfg)
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
        R75 / "outputs" / f"oracle_{args.scheme}_seed{args.seed}.pt",
        map_location=device,
        weights_only=False,
    )

    if args.kind == "low":
        all_keys = [(i,) for i in range(n_general)] + list(combinations(range(n_general), 2))
    else:
        all_keys = list(combinations(range(n_general), 3))
    keys = [key for pos, key in enumerate(all_keys) if pos % args.nshard == args.shard]

    out = (
        ROOT
        / "outputs"
        / f"kgrid_{args.kind}_uniform_seed{args.seed}_shard{args.shard}of{args.nshard}.csv.gz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    log(
        "KGRID",
        "START",
        note=f"{args.kind} 同轨迹扫描：{len(keys)}/{len(all_keys)} 个集合，K={list(KGRID)}",
        kind=args.kind,
        shard=args.shard,
        nshard=args.nshard,
        gpu=str(device),
    )

    rows: list[dict] = []
    for done, key in enumerate(keys, 1):
        mask = torch.zeros(1, n_general, device=device)
        mask[0, list(key)] = 1.0
        mask_train = mask.expand(len(x_train), -1)
        mask_test = mask.expand(len(x_test), -1)

        # 与 76/77 的协议一致：每个集合使用同一初始化与同一 mini-batch 顺序，
        # 使 K 的差异只来自更新步数，不来自额外随机噪声。
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = MLPOracle(n_general, n_conf)
        model.load_state_dict(checkpoint["state"])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        def snapshot(k_value: int) -> None:
            model.eval()
            with torch.no_grad():
                pred = model(x_test, mask_test).cpu().numpy()
            r2 = per_conf_r2(pred, y_test)
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
        rng = np.random.RandomState(args.seed)
        step = 0
        while step < max(KGRID):
            order = rng.permutation(len(x_train))
            for begin in range(0, len(order), BATCH):
                indices = torch.as_tensor(order[begin : begin + BATCH], device=device)
                model.train()
                optimizer.zero_grad()
                loss = (
                    (model(x_train[indices], mask_train[indices]) - y_train[indices]) ** 2
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
                f"{args.kind} shard {args.shard}/{args.nshard}: "
                f"{done}/{len(keys)} [{time.time() - started:.0f}s]",
                flush=True,
            )

    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
    elapsed = time.time() - started
    log(
        "KGRID",
        "DONE",
        note=f"{args.kind} shard {args.shard}/{args.nshard} 完成 {len(keys)} 个集合",
        elapsed_s=elapsed,
        kind=args.kind,
        shard=args.shard,
        n_queries=len(keys),
        n_rows=len(rows),
    )
    print(f"完成 -> {out} [{elapsed:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
