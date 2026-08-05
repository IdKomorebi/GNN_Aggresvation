#!/usr/bin/env python3
"""Proper LoRA PEFT baseline on the exact validation/audit tasks."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import (  # noqa: E402
    load_base_model,
    load_data,
    load_design,
    per_conf_r2,
    task_mask,
)

ALL_KGRID = (1, 5, 25)
BATCH = 256
WD = 5e-4


class LoRALinear(nn.Module):
    def __init__(self, source: nn.Linear, rank: int):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.a = nn.Parameter(torch.empty(rank, source.in_features))
        self.b = nn.Parameter(torch.zeros(source.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 1.0

    def forward(self, x):
        update = (self.b @ self.a) * self.scale
        return F.linear(x, self.weight + update, self.bias)


def inject_lora(module: nn.Module, rank: int):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank))
        else:
            inject_lora(child, rank)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshard", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-k", type=int, choices=ALL_KGRID, default=25)
    args = parser.parse_args()
    kgrid = tuple(k for k in ALL_KGRID if k <= args.max_k)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    data, tensors = load_data(device)
    design = load_design()
    task_by_id = {task["task_id"]: task for task in design["tasks"]}
    eval_ids = sorted(
        set(design["splits"]["validation_all"]) | set(design["splits"]["audit_all"])
    )
    eval_ids = [
        task_id for pos, task_id in enumerate(eval_ids)
        if pos % args.nshard == args.shard
    ]
    n_general = int(data["n_general"])
    n_conf = int(data["n_confidential"])
    conf_names = list(data["confidential"])
    base = load_base_model(device, n_general, n_conf)
    rows = []
    started = time.time()

    for done, task_id in enumerate(eval_ids, 1):
        task = task_by_id[task_id]
        mask = task_mask(task, n_general, device)
        mask_train = mask.expand(len(tensors["x_train"]), -1)
        mask_test = mask.expand(len(tensors["x_test"]), -1)
        for rank in (4, 16):
            torch.manual_seed(92000 + rank)
            np.random.seed(92000 + rank)
            model = deepcopy(base)
            inject_lora(model, rank)
            model = model.to(device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=WD)
            rng = np.random.RandomState(0)
            step = 0
            tune_started = time.perf_counter()
            while step < max(kgrid):
                order = rng.permutation(len(tensors["x_train"]))
                for begin in range(0, len(order), BATCH):
                    idx = torch.as_tensor(
                        order[begin : begin + BATCH], dtype=torch.long, device=device
                    )
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss = (
                        (
                            model(tensors["x_train"][idx], mask_train[idx])
                            - tensors["y_train"][idx]
                        )
                        ** 2
                    ).mean()
                    loss.backward()
                    optimizer.step()
                    step += 1
                    if step in kgrid:
                        model.eval()
                        with torch.no_grad():
                            prediction = model(
                                tensors["x_test"], mask_test
                            ).cpu().numpy()
                        r2 = per_conf_r2(prediction, tensors["y_test_np"])
                        elapsed = time.perf_counter() - tune_started
                        for conf_pos, conf in enumerate(conf_names):
                            rows.append({
                                "task_id": task_id,
                                "rank": rank,
                                "K": step,
                                "lr": args.lr,
                                "conf": conf,
                                "est": float(r2[conf_pos]),
                                "elapsed": elapsed,
                                "trainable_parameters": int(
                                    sum(p.numel() for p in trainable)
                                ),
                            })
                    if step >= max(kgrid):
                        break
        if done % 25 == 0:
            print(
                f"LoRA shard {args.shard}/{args.nshard}: {done}/{len(eval_ids)} "
                f"{time.time()-started:.1f}s",
                flush=True,
            )

    lr_tag = f"{args.lr:g}".replace(".", "p")
    out = (
        ROOT / "outputs"
        / f"lora_lr{lr_tag}_shard{args.shard}of{args.nshard}.csv.gz"
    )
    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
    meta = {
        "shard": args.shard,
        "nshard": args.nshard,
        "n_tasks": len(eval_ids),
        "ranks": [4, 16],
        "K": list(kgrid),
        "lr": args.lr,
        "seconds": time.time() - started,
    }
    out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
