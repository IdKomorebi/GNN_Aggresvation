"""DNN92 shared utilities."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))

from src.data_processing import prepare_data  # noqa: E402
from src.oracle import MLPOracle  # noqa: E402


def load_data(device: torch.device):
    cfg = yaml.safe_load((R69 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(
        REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"
    )
    torch.manual_seed(42)
    np.random.seed(42)
    data = prepare_data(cfg)
    gi = np.asarray(data["general_indices"])
    ci = np.asarray(data["confidential_indices"])
    train = data["train_data"]
    test = data["test_data"]
    return data, {
        "x_train": torch.as_tensor(
            train[:, gi], dtype=torch.float32, device=device
        ),
        "y_train": torch.as_tensor(
            train[:, ci], dtype=torch.float32, device=device
        ),
        "x_test": torch.as_tensor(
            test[:, gi], dtype=torch.float32, device=device
        ),
        "y_test_np": np.asarray(test[:, ci], dtype=np.float32),
    }


def load_base_model(device: torch.device, n_general: int, n_conf: int):
    checkpoint = torch.load(
        R69 / "outputs/oracle_mlp_seed0.pt",
        map_location=device,
        weights_only=False,
    )
    model = MLPOracle(n_general, n_conf).to(device)
    model.load_state_dict(checkpoint["state"])
    return model


def parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters())


def set_parameter_vector(model: torch.nn.Module, vector: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vector, model.parameters())


def flat_gradient(model: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for parameter in model.parameters():
        if parameter.grad is None:
            chunks.append(torch.zeros_like(parameter).reshape(-1))
        else:
            chunks.append(parameter.grad.detach().reshape(-1))
    return torch.cat(chunks)


def per_conf_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = ((target - prediction) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return np.clip(1.0 - residual / total, 0.0, None)


def task_seed(task_id: str, salt: int = 0) -> int:
    digest = hashlib.sha256(f"{task_id}:{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def load_design() -> dict:
    return json.loads((ROOT / "outputs/design.json").read_text(encoding="utf-8"))


def task_mask(task: dict, n_general: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, n_general, dtype=torch.float32, device=device)
    mask[0, task["indices"]] = 1.0
    return mask
