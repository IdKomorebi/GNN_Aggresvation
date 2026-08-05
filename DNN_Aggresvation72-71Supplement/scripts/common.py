#!/usr/bin/env python3
"""Shared deterministic protocol for the 71/72 supplement."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
R72 = REPO / "DNN_Aggresvation72"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(R69))
from src.data_processing import prepare_data  # noqa: E402
from src.model import build_edge_mask  # noqa: E402
from src.oracle import GNNOracle, MLPOracle, build_priors  # noqa: E402


TAU = 0.5
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 5e-4


def per_conf_r2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def load_cfg_data():
    cfg = yaml.safe_load((R69 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42)
    np.random.seed(42)
    return cfg, prepare_data(cfg)


class OracleEvaluator:
    """Fixed-mask K-step evaluator matching D69, parameterized by architecture/seed."""

    def __init__(self, arch: str, seed: int, device: str | None = None):
        if arch not in {"mlp", "gnn"}:
            raise ValueError(arch)
        self.arch = arch
        self.seed = int(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg, self.data = load_cfg_data()
        gi = np.asarray(self.data["general_indices"])
        ci = np.asarray(self.data["confidential_indices"])
        self.nG, self.nC = self.data["n_general"], self.data["n_confidential"]
        self.fields = list(self.data["general"])
        self.conf_names = list(self.data["confidential"])
        self.name2local = {n: i for i, n in enumerate(self.fields)}
        self.xtr = torch.as_tensor(
            self.data["train_data"][:, gi], dtype=torch.float32, device=self.device
        )
        self.ytr = torch.as_tensor(
            self.data["train_data"][:, ci], dtype=torch.float32, device=self.device
        )
        self.xte = torch.as_tensor(
            self.data["test_data"][:, gi], dtype=torch.float32, device=self.device
        )
        self.yte = self.data["test_data"][:, ci]
        self.ckpt = torch.load(
            R69 / f"outputs/oracle_{arch}_seed{seed}.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.a_gg = self.prior_cg = None
        if arch == "gnn":
            metric = np.load(R69 / "outputs/relationship_cache/metric_tensor.npy")
            gc = self.cfg["graph"]
            edge_mask = build_edge_mask(
                metric,
                top_k=gc["top_k"],
                threshold=gc["threshold"],
                symmetrize=True,
                n_general=self.nG,
                bipartite=True,
            )
            self.a_gg, self.prior_cg = build_priors(metric, edge_mask, self.nG)

    def build_model(self):
        if self.arch == "mlp":
            model = MLPOracle(self.nG, self.nC)
        else:
            model = GNNOracle(
                self.a_gg,
                self.prior_cg,
                self.nC,
                hidden=self.ckpt.get("gnn_hidden", 128),
                n_layers=self.ckpt.get("gnn_layers", 3),
            )
        model.load_state_dict(self.ckpt["state"])
        return model.to(self.device)

    def eval_fields(self, fields: list[str], k_steps: int = 200) -> np.ndarray:
        if not fields:
            return np.zeros(self.nC, dtype=float)
        sel = [self.name2local[f] for f in fields]
        mask = torch.zeros(1, self.nG, device=self.device)
        mask[0, sel] = 1.0
        mtr = mask.expand(len(self.xtr), -1)
        mte = mask.expand(len(self.xte), -1)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        model = self.build_model()
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        rng = np.random.RandomState(self.seed)
        step = 0
        while step < k_steps:
            order = rng.permutation(len(self.xtr))
            for b in range(0, len(order), BATCH):
                ix = torch.as_tensor(order[b:b + BATCH], device=self.device)
                model.train()
                opt.zero_grad()
                loss = ((model(self.xtr[ix], mtr[ix]) - self.ytr[ix]) ** 2).mean()
                loss.backward()
                opt.step()
                step += 1
                if step >= k_steps:
                    break
        model.eval()
        with torch.no_grad():
            pred = model(self.xte, mte).cpu().numpy()
        return per_conf_r2(pred, self.yte)


def low_order_problem(tau: float = TAU) -> dict:
    """Build deterministic low-order sequences from D69 dedicated-DNN truth."""
    registry = json.loads((R69 / "outputs/subsets.json").read_text())
    truth_long = pd.read_csv(R69 / "outputs/truth_long.csv")
    truth = {(r.sid, r.conf): r.dnn for r in truth_long.itertuples()}
    confs = sorted(truth_long.conf.unique())
    singles: dict[str, float] = {}
    pairs: dict[tuple[str, str], float] = {}
    fields: list[str] = []
    for sid, meta in registry.items():
        if meta["group"] == "single":
            field = meta["fields"][0]
            fields.append(field)
            singles[field] = max(truth.get((sid, c), 0.0) for c in confs)
        elif meta["group"] == "pair":
            pair = tuple(sorted(meta["fields"]))
            pairs[pair] = max(truth.get((sid, c), 0.0) for c in confs)
    fields = sorted(set(fields))
    dangerous_singles = {f: v for f, v in singles.items() if v > tau}
    dangerous_pairs = {p: v for p, v in pairs.items() if v > tau}

    def coverage(protected):
        protected = set(protected)
        return (
            sum(v for f, v in dangerous_singles.items() if f in protected)
            + sum(v for (a, b), v in dangerous_pairs.items() if a in protected or b in protected)
        )

    total = sum(dangerous_singles.values()) + sum(dangerous_pairs.values())
    protected: list[str] = []
    remaining = set(fields)
    current = 0.0
    for _ in range(len(fields)):
        best_field, best_gain = None, -1.0
        for f in sorted(remaining):
            gain = coverage(protected + [f]) - current
            if gain > best_gain:
                best_gain, best_field = gain, f
        protected.append(best_field)
        remaining.remove(best_field)
        current = coverage(protected)
    k_cover = next((k for k in range(len(fields) + 1) if coverage(protected[:k]) >= total - 1e-9), len(fields))

    degree = {f: 0.0 for f in fields}
    for f, v in dangerous_singles.items():
        degree[f] += v
    for (a, b), v in dangerous_pairs.items():
        degree[a] += v
        degree[b] += v
    orders = {
        "low_order": protected,
        "degree": sorted(fields, key=lambda f: (-degree[f], f)),
        "single_leak": sorted(fields, key=lambda f: (-singles[f], f)),
        "random": list(np.random.RandomState(0).permutation(fields)),
    }
    return {
        "fields": fields,
        "single_leak": singles,
        "pair_leak": pairs,
        "orders": orders,
        "k_cover": k_cover,
        "n_dangerous_singles": len(dangerous_singles),
        "n_dangerous_pairs": len(dangerous_pairs),
        "tau": tau,
    }


def write_protocol_manifest() -> dict:
    problem = low_order_problem()
    manifest = {
        "tau": problem["tau"],
        "n_fields": len(problem["fields"]),
        "n_dangerous_singles": problem["n_dangerous_singles"],
        "n_dangerous_pairs": problem["n_dangerous_pairs"],
        "low_order_cover_k": problem["k_cover"],
        "orders": problem["orders"],
        "oracle_architectures": ["mlp", "gnn"],
        "oracle_seeds": [0, 1, 2],
        "k_inner": 50,
        "k_cert": 200,
        "outer_split": "shuffle seed42, 70% development / 30% test; test excluded from dedicated-retrain early stopping but reused by historical/oracle candidate selection",
        "inner_validation": "15% of outer development, fixed split seed20260722",
        "retrain_seeds": [0, 1, 2],
        "retrain_max_epochs": 800,
        "retrain_patience": 120,
        "end_to_end_holdout": False,
        "end_to_end_holdout_limitation": "No second year or never-used dataset is present; candidate selection reuses D69's test R2, so certification is clean for model selection but not a prospective end-to-end holdout.",
    }
    (OUT / "protocol_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
