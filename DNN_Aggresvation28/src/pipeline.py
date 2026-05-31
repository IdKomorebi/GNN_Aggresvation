"""Compact end-to-end pipeline for DNN_Aggresvation28.

This module intentionally contains the data, graph, model, training, and
visualization logic behind the project's single command-line entry point.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from scipy.stats import kendalltau as scipy_kendalltau
except ImportError:
    scipy_kendalltau = None

try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    mutual_info_regression = None

try:
    import dcor as dcor_lib
except ImportError:
    dcor_lib = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_FUNCS: dict[str, Any] = {}
EXPENSIVE_METRICS = {"kendall", "nmi", "distance_corr"}
PLOT_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _argument_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return _path(path)


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")
    return text or "run"


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    requested = str(value)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in YAML/CLI, but torch.cuda.is_available() is False.")
    device = torch.device(requested)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"  Device: {device} ({torch.cuda.get_device_name(index)})")
    else:
        print(f"  Device: {device}")
    return device


def prepare_data(cfg: dict[str, Any]) -> dict[str, Any]:
    csv_path = _path(cfg["dataset"]["csv_path"])
    frame = pd.read_csv(csv_path)
    drop_columns = [c for c in cfg["fields"].get("drop_columns", []) if c in frame.columns]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)
    frame = frame.select_dtypes(include=[np.number])
    if cfg["fields"].get("drop_constant_columns", True):
        constants = frame.std()[frame.std() < 1e-10].index.tolist()
        if constants:
            frame = frame.drop(columns=constants)
            print(f"  Dropped constant columns: {constants}")
    before = len(frame)
    frame = frame.dropna().reset_index(drop=True)
    if before != len(frame):
        print(f"  Dropped NaN rows: {before} -> {len(frame)}")

    requested_conf = cfg["fields"]["confidential"]
    confidential = [c for c in requested_conf if c in frame.columns]
    missing = sorted(set(requested_conf) - set(confidential))
    if missing:
        raise ValueError(
            "Configured Confidential column(s) are unavailable after preprocessing: "
            f"{missing}. Check the CSV column names and fields.drop_columns."
        )
    general = [c for c in frame.columns if c not in set(confidential)]
    columns = general + confidential
    frame = frame[columns]
    if not confidential or not general:
        raise ValueError("Data must contain both General and Confidential columns.")

    split = int(len(frame) * float(cfg["training"]["train_ratio"]))
    train_df = frame.iloc[:split].copy()
    test_df = frame.iloc[split:].copy()
    mean = train_df.mean()
    std = train_df.std().replace(0, 1)
    train_data = ((train_df - mean) / std).values.astype(np.float32)
    test_data = ((test_df - mean) / std).values.astype(np.float32)
    n_general = len(general)

    print(f"  Dataset: {csv_path.name}, rows={len(frame)}, nodes={len(columns)}")
    print(f"  General={len(general)}, Confidential={len(confidential)}, split={len(train_df)}/{len(test_df)}")
    return {
        "csv_path": csv_path,
        "raw_df": frame,
        "train_df": train_df,
        "test_df": test_df,
        "train_data": train_data,
        "test_data": test_data,
        "mean": mean,
        "std": std,
        "general": general,
        "confidential": confidential,
        "all_columns": columns,
        "n_general": n_general,
        "n_confidential": len(confidential),
        "n_nodes": len(columns),
        "confidential_indices": list(range(n_general, len(columns))),
    }


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _sample(
    x: np.ndarray, y: np.ndarray, maximum: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if maximum is None or len(x) <= maximum:
        return x, y
    picked = rng.choice(len(x), size=maximum, replace=False)
    return x[picked], y[picked]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return abs(float(np.corrcoef(x, y)[0, 1]))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(
        pd.Series(x).rank(method="average").to_numpy(dtype=float),
        pd.Series(y).rank(method="average").to_numpy(dtype=float),
    )


def kendall(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    if scipy_kendalltau is not None:
        result = scipy_kendalltau(x, y).statistic
        return abs(float(result)) if np.isfinite(result) else 0.0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    tri = np.triu_indices(len(x), k=1)
    products = np.sign(dx[tri] * dy[tri])
    return abs(float(np.mean(products))) if len(products) else 0.0


def nmi(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    def entropy(values: np.ndarray) -> float:
        hist, _ = np.histogram(values, bins=20)
        prob = hist.astype(float) / max(float(hist.sum()), 1.0)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log(prob))) if len(prob) else 0.0

    if mutual_info_regression is not None:
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
        denominator = min(entropy(x), entropy(y))
        return float(np.clip(mi / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
    hist, _, _ = np.histogram2d(x, y, bins=20)
    probability = hist / max(float(hist.sum()), 1.0)
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)
    valid = probability > 0
    denominator = px[:, None] * py[None, :]
    mi = float(np.sum(probability[valid] * np.log(probability[valid] / denominator[valid])))
    h_min = min(entropy(x), entropy(y))
    return float(np.clip(mi / h_min, 0.0, 1.0)) if h_min > 0 else 0.0


def distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    if dcor_lib is not None:
        try:
            return float(dcor_lib.distance_correlation(x, y))
        except Exception:
            pass
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    b = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov = np.sqrt(max(float(np.mean(a * b)), 0.0))
    dvar_x = np.sqrt(max(float(np.mean(a * a)), 0.0))
    dvar_y = np.sqrt(max(float(np.mean(b * b)), 0.0))
    denominator = np.sqrt(dvar_x * dvar_y)
    return float(np.clip(dcov / denominator, 0.0, 1.0)) if denominator > 1e-12 else 0.0


METRIC_FUNCS.update(
    {
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "nmi": nmi,
        "distance_corr": distance_corr,
    }
)


def compute_metric_tensor(
    frame: pd.DataFrame,
    columns: list[str],
    metrics: list[str],
    sample_size: int,
    expensive_sample_size: int,
    seed: int,
) -> np.ndarray:
    unknown = set(metrics) - set(METRIC_FUNCS)
    if unknown:
        raise ValueError(f"Unsupported correlation metric(s): {sorted(unknown)}")
    tensor = np.zeros((len(columns), len(columns), len(metrics)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    total = len(columns) * (len(columns) - 1) // 2
    done = 0
    for i, left in enumerate(columns):
        x = frame[left].to_numpy(dtype=float)
        for j in range(i + 1, len(columns)):
            done += 1
            y = frame[columns[j]].to_numpy(dtype=float)
            x_clean, y_clean = _finite_pair(x, y)
            x_std, y_std = _sample(x_clean, y_clean, sample_size, rng)
            x_exp, y_exp = _sample(x_clean, y_clean, expensive_sample_size, rng)
            for m, metric in enumerate(metrics):
                values = (x_exp, y_exp) if metric in EXPENSIVE_METRICS else (x_std, y_std)
                result = METRIC_FUNCS[metric](*values)
                tensor[i, j, m] = tensor[j, i, m] = float(np.clip(result, 0.0, 1.0))
            if done == 1 or done % 200 == 0 or done == total:
                print(f"  Correlation progress: {done}/{total}")
    return tensor


def load_or_compute_metrics(
    data: dict[str, Any],
    cfg: dict[str, Any],
    artifact_root: Path,
    seed: int,
) -> np.ndarray:
    correlation = cfg["correlation"]
    fit_on = str(correlation.get("fit_on", "all")).lower()
    if fit_on not in {"all", "train"}:
        raise ValueError("correlation.fit_on must be either 'all' or 'train'.")
    source = data["raw_df"] if fit_on == "all" else data["train_df"]
    csv_stat = data["csv_path"].stat()
    signature = {
        "csv": str(data["csv_path"]),
        "size": csv_stat.st_size,
        "mtime_ns": csv_stat.st_mtime_ns,
        "columns": data["all_columns"],
        "metrics": correlation["metrics"],
        "sample_size": correlation.get("sample_size", 3000),
        "expensive_sample_size": correlation.get("expensive_sample_size", 1200),
        "fit_on": fit_on,
        "seed": seed,
    }
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache = artifact_root / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    array_path = cache / f"metric_{digest}.npy"
    meta_path = cache / f"metric_{digest}.json"
    if correlation.get("reuse_cache", True) and array_path.exists():
        print(f"  Reusing metric cache: {array_path.name} (fit_on={fit_on})")
        return np.load(array_path)
    if correlation.get("reuse_cache", True):
        prior_caches = sorted(artifact_root.parents[1].glob(f"*/artifacts/_cache/metric_{digest}.npy"))
        prior_caches = [path for path in prior_caches if path != array_path]
        if prior_caches:
            tensor = np.load(prior_caches[-1])
            np.save(array_path, tensor)
            _dump_json(meta_path, signature)
            print(f"  Reusing previous-run metric cache: {prior_caches[-1].name} (fit_on={fit_on})")
            return tensor
    print(f"  Computing correlation tensor (fit_on={fit_on})")
    tensor = compute_metric_tensor(
        source,
        data["all_columns"],
        correlation["metrics"],
        int(correlation.get("sample_size", 3000)),
        int(correlation.get("expensive_sample_size", 1200)),
        seed,
    )
    np.save(array_path, tensor)
    _dump_json(meta_path, signature)
    return tensor


def build_edge_mask(
    metric_tensor: np.ndarray, top_k: int, threshold: float, symmetrize: bool
) -> np.ndarray:
    average = metric_tensor.mean(axis=2).copy()
    np.fill_diagonal(average, 0.0)
    mask = average >= threshold
    for target in range(len(average)):
        source_order = [x for x in np.argsort(average[target])[::-1] if x != target][:top_k]
        mask[target, source_order] = True
    if symmetrize:
        mask |= mask.T
    np.fill_diagonal(mask, False)
    return mask.astype(np.float32)


def apply_target_top_k(
    mask: np.ndarray,
    metric_tensor: np.ndarray,
    columns: list[str],
    overrides: dict[str, Any],
) -> dict[str, int]:
    average = metric_tensor.mean(axis=2).copy()
    np.fill_diagonal(average, 0.0)
    applied: dict[str, int] = {}
    for field, value in overrides.items():
        if field not in columns:
            print(f"  Warning: target_top_k_overrides ignored missing node: {field}")
            continue
        requested = max(0, min(int(value), len(columns) - 1))
        target = columns.index(field)
        source_order = [x for x in np.argsort(average[target])[::-1] if x != target][:requested]
        mask[target, source_order] = 1.0
        applied[field] = requested
    return applied


class InferenceDrivenGNN(nn.Module):
    """DNN13 graph structure with optional temporal input and multi-head attention."""

    def __init__(
        self,
        metric_tensor: np.ndarray,
        edge_mask: np.ndarray,
        n_nodes: int,
        n_general: int,
        confidential_indices: list[int],
        model_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.register_buffer("metric_tensor", torch.as_tensor(metric_tensor, dtype=torch.float32))
        self.register_buffer("edge_mask", torch.as_tensor(edge_mask, dtype=torch.float32))
        self.register_buffer("conf_indices", torch.as_tensor(confidential_indices, dtype=torch.long))
        self.n_nodes = n_nodes
        self.n_general = n_general
        self.n_confidential = len(confidential_indices)
        self.n_metrics = metric_tensor.shape[2]
        self.hidden_dim = int(model_cfg.get("hidden_dim", 64))
        self.input_dim = int(model_cfg.get("window_size", 1))
        self.attention_dim = int(model_cfg.get("attention_dim", self.hidden_dim))
        self.attention_heads = max(int(model_cfg.get("attention_heads", 1)), 1)
        self.attention_temperature = max(float(model_cfg.get("attention_temperature", 1.0)), 1e-3)
        self.edge_alpha_temperature = max(float(model_cfg.get("edge_alpha_temperature", 1.0)), 1e-3)
        self.prior_log_eps = max(float(model_cfg.get("prior_log_eps", 1e-4)), 1e-12)

        self.beta_general = nn.Parameter(torch.zeros(self.n_metrics))
        self.beta_confidential = nn.Parameter(torch.zeros(self.n_confidential, n_nodes, self.n_metrics))
        alpha_std = float(model_cfg.get("alpha_init_std", 0.0))
        if alpha_std > 0:
            nn.init.normal_(self.beta_general, std=alpha_std)
            nn.init.normal_(self.beta_confidential, std=alpha_std)

        if model_cfg.get("input_encoder", "linear") == "mlp":
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(model_cfg.get("input_encoder_dropout", 0.0))),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
        else:
            self.input_proj = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
        self.conf_embedding = nn.Parameter(torch.randn(self.n_confidential, self.hidden_dim) * 0.01)
        self.target_identity = nn.Parameter(torch.randn(self.n_confidential, self.hidden_dim) * 0.01)

        self.self_layers = nn.ModuleList()
        self.general_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.query_hidden = nn.ModuleList()
        self.query_identity = nn.ModuleList()
        self.keys = nn.ModuleList()
        self.values = nn.ModuleList()
        self.gate_q = nn.ModuleList()
        self.gate_k = nn.ModuleList()
        self.gate_p = nn.ModuleList()
        self.attention_outputs = nn.ModuleList()
        self.prior_scales = nn.ParameterList()

        for _ in range(int(model_cfg.get("num_layers", 3))):
            self.self_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.general_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.conf_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(n_nodes))
            self.query_hidden.append(nn.Linear(self.hidden_dim, self.attention_heads * self.attention_dim, bias=False))
            self.query_identity.append(
                nn.Linear(self.hidden_dim, self.attention_heads * self.attention_dim, bias=False)
            )
            self.keys.append(nn.Linear(self.hidden_dim, self.attention_heads * self.attention_dim, bias=False))
            self.values.append(nn.Linear(self.hidden_dim, self.attention_heads * self.hidden_dim, bias=False))
            self.gate_q.append(nn.Linear(self.attention_dim, 1, bias=False))
            self.gate_k.append(nn.Linear(self.attention_dim, 1, bias=False))
            self.gate_p.append(nn.Linear(1, 1, bias=True))
            nn.init.constant_(self.gate_p[-1].bias, float(model_cfg.get("gate_bias_init", -1.0)))
            self.attention_outputs.append(nn.Linear(self.attention_heads * self.hidden_dim, self.hidden_dim, bias=False))
            self.prior_scales.append(nn.Parameter(torch.tensor(float(model_cfg.get("prior_scale_init", 1.0)))))

        self.dropout = nn.Dropout(float(model_cfg.get("dropout", 0.15)))
        self.attention_dropout = nn.Dropout(float(model_cfg.get("attention_dropout", 0.05)))
        half_hidden = max(self.hidden_dim // 2, 1)
        self.target_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.hidden_dim, half_hidden), nn.ReLU(), nn.Linear(half_hidden, 1))
                for _ in range(self.n_confidential)
            ]
        )
        self.last_attention: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None

    def get_alpha_general(self) -> torch.Tensor:
        return torch.softmax(self.beta_general / self.edge_alpha_temperature, dim=0)

    def get_alpha_confidential(self) -> torch.Tensor:
        return torch.softmax(self.beta_confidential / self.edge_alpha_temperature, dim=-1)

    def _initial_hidden(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        hidden = self.input_proj(values)
        embedding = self.conf_embedding.unsqueeze(0).expand(values.shape[0], -1, -1)
        indices = self.conf_indices.view(1, -1, 1).expand(values.shape[0], -1, self.hidden_dim)
        return hidden.scatter(1, indices, embedding)

    def general_adjacency(self) -> torch.Tensor:
        relation = torch.einsum("ijk,k->ij", self.metric_tensor, self.get_alpha_general())
        relation = relation * self.edge_mask
        allowed = torch.zeros_like(relation)
        allowed[: self.n_general, : self.n_general] = 1.0
        relation = relation * allowed * (1.0 - torch.eye(self.n_nodes, device=relation.device))
        return relation / relation.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def confidential_prior(self) -> torch.Tensor:
        relation = self.metric_tensor.index_select(0, self.conf_indices)
        prior = (relation * self.get_alpha_confidential()).sum(dim=-1)
        prior = prior * self.edge_mask.index_select(0, self.conf_indices)
        for position, node in enumerate(self.conf_indices.tolist()):
            prior[position, node] = 0.0
        return prior

    def _conf_messages(self, hidden: torch.Tensor, layer: int, prior: torch.Tensor) -> torch.Tensor:
        batch = hidden.shape[0]
        conf_hidden = hidden.index_select(1, self.conf_indices)
        identity = self.target_identity.unsqueeze(0).expand(batch, -1, -1)
        query = self.query_hidden[layer](conf_hidden) + self.query_identity[layer](identity)
        key = self.keys[layer](hidden)
        value = self.values[layer](hidden)
        query = query.view(batch, self.n_confidential, self.attention_heads, self.attention_dim)
        key = key.view(batch, self.n_nodes, self.attention_heads, self.attention_dim)
        value = value.view(batch, self.n_nodes, self.attention_heads, self.hidden_dim)
        dynamic = torch.einsum("bchd,bnhd->bchn", query, key) / math.sqrt(self.attention_dim)
        prior_score = torch.log(prior.clamp(min=self.prior_log_eps)).unsqueeze(0).unsqueeze(2)
        prior_score = prior_score.expand(batch, -1, self.attention_heads, -1)
        q_gate = self.gate_q[layer](query).squeeze(-1).unsqueeze(-1)
        k_gate = self.gate_k[layer](key).squeeze(-1).permute(0, 2, 1).unsqueeze(1)
        gate = torch.sigmoid(q_gate + k_gate + self.gate_p[layer](prior_score.unsqueeze(-1)).squeeze(-1))
        scale = F.softplus(self.prior_scales[layer])
        score = (gate * dynamic + (1.0 - gate) * scale * prior_score) / self.attention_temperature
        valid = prior > 0
        score = score.masked_fill(~valid.unsqueeze(0).unsqueeze(2), -1e9)
        attention = torch.softmax(score, dim=-1) * valid.unsqueeze(0).unsqueeze(2).float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        attention = self.attention_dropout(attention)
        self.last_attention = attention.mean(dim=2).detach()
        self.last_gate = gate.mean(dim=2).detach()
        message = torch.einsum("bchn,bnhd->bchd", attention, value)
        message = message.reshape(batch, self.n_confidential, self.attention_heads * self.hidden_dim)
        return self.attention_outputs[layer](message)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self._initial_hidden(values)
        general_adjacency = self.general_adjacency()
        prior = self.confidential_prior()
        for layer, (self_layer, general_layer, conf_layer, batch_norm) in enumerate(
            zip(self.self_layers, self.general_layers, self.conf_layers, self.batch_norms)
        ):
            messages = torch.zeros_like(hidden)
            messages[:, : self.n_general, :] = torch.matmul(
                general_adjacency[: self.n_general, : self.n_general],
                hidden[:, : self.n_general, :],
            )
            messages[:, self.conf_indices, :] = self._conf_messages(hidden, layer, prior)
            transformed = general_layer(messages)
            transformed = transformed.clone()
            transformed[:, self.conf_indices, :] = conf_layer(messages)[:, self.conf_indices, :]
            updated = self.dropout(F.relu(batch_norm(self_layer(hidden) + transformed)))
            hidden = hidden + updated
        confidential_hidden = hidden.index_select(1, self.conf_indices)
        predictions = [head(confidential_hidden[:, position, :]) for position, head in enumerate(self.target_heads)]
        return torch.cat(predictions, dim=1)


def build_samples(
    values: np.ndarray, confidential_indices: list[int], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    samples = len(values) - window_size + 1
    if samples < 1:
        raise ValueError("window_size exceeds available data.")
    features = np.zeros((samples, values.shape[1], window_size), dtype=np.float32)
    targets = np.zeros((samples, len(confidential_indices)), dtype=np.float32)
    for index in range(samples):
        targets[index] = values[index + window_size - 1, confidential_indices].copy()
        window = values[index : index + window_size, :].copy().T
        window[confidential_indices, :] = 0.0
        features[index] = window
    return features, targets


def loss_targets(data: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[int], list[str], list[str]]:
    requested = cfg["training"].get("loss_exclude_confidential", [])
    unknown = sorted(set(requested) - set(data["confidential"]))
    if unknown:
        print(f"  Warning: ignored loss-excluded unknown columns: {unknown}")
    excluded = [name for name in data["confidential"] if name in set(requested)]
    selected = [name for name in data["confidential"] if name not in set(excluded)]
    if not selected:
        raise ValueError("loss_exclude_confidential excludes every Confidential target.")
    return [data["confidential"].index(name) for name in selected], selected, excluded


def _selected_loss(
    predictions: torch.Tensor, targets: torch.Tensor, indices: list[int], loss_name: str
) -> torch.Tensor:
    values = (
        F.smooth_l1_loss(predictions, targets, reduction="none")
        if loss_name in {"smooth_l1", "huber"}
        else F.mse_loss(predictions, targets, reduction="none")
    )
    chosen = torch.as_tensor(indices, dtype=torch.long, device=values.device)
    return values.index_select(1, chosen).mean()


def _mse(predictions: torch.Tensor, targets: torch.Tensor, indices: list[int] | None = None) -> float:
    if indices is not None:
        chosen = torch.as_tensor(indices, dtype=torch.long, device=predictions.device)
        predictions = predictions.index_select(1, chosen)
        targets = targets.index_select(1, chosen)
    return float(F.mse_loss(predictions, targets).item())


class TargetProbeMLP(nn.Module):
    """Single-target DNN baseline fed only the General feature windows."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden_dim
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _train_probe_target(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    probe_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, float | int]:
    _set_seed(seed)
    hidden_dims = [int(value) for value in probe_cfg.get("hidden_dims", [128, 64])]
    dropout = float(probe_cfg.get("dropout", 0.15))
    batch_size = int(probe_cfg["batch_size"])
    epochs = int(probe_cfg["epochs"])
    lr = float(probe_cfg["lr"])
    patience = int(probe_cfg["patience"])
    weight_decay = float(probe_cfg.get("weight_decay", 0.0))
    model = TargetProbeMLP(train_x.shape[1], hidden_dims, dropout).to(device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_x) % batch_size == 1,
    )
    test_features = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    test_targets = torch.as_tensor(test_y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stagnant = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(features), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            test_loss = float(F.mse_loss(model(test_features), test_targets).item())
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(test_features).cpu().numpy()
    denominator = float(np.sum((test_y - test_y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((test_y - predictions) ** 2)) / max(denominator, 1e-12)
    return {"dnn_r2": r2, "dnn_mse": best_loss, "dnn_best_epoch": best_epoch}


def train_dnn_probe(
    data: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    artifact_root: Path,
) -> pd.DataFrame:
    probe_cfg = dict(cfg.get("evaluation", {}).get("dnn_probe", {}))
    if not bool(probe_cfg.get("enabled", True)):
        print("  DNN probe disabled.")
        return pd.DataFrame(columns=["confidential", "dnn_r2", "dnn_mse", "dnn_best_epoch"])
    training_cfg = cfg["training"]
    for key in ("epochs", "batch_size", "lr", "weight_decay", "patience"):
        probe_cfg.setdefault(key, training_cfg.get(key))
    window = int(cfg["model"].get("window_size", 1))
    csv_stat = data["csv_path"].stat()
    signature = {
        "kind": "general_only_target_mlp_v1",
        "csv": str(data["csv_path"]),
        "size": csv_stat.st_size,
        "mtime_ns": csv_stat.st_mtime_ns,
        "general": data["general"],
        "confidential": data["confidential"],
        "train_ratio": float(training_cfg["train_ratio"]),
        "window_size": window,
        "seed": seed,
        "probe": {
            "hidden_dims": [int(value) for value in probe_cfg.get("hidden_dims", [128, 64])],
            "dropout": float(probe_cfg.get("dropout", 0.15)),
            "epochs": int(probe_cfg["epochs"]),
            "batch_size": int(probe_cfg["batch_size"]),
            "lr": float(probe_cfg["lr"]),
            "weight_decay": float(probe_cfg.get("weight_decay", 0.0)),
            "patience": int(probe_cfg["patience"]),
        },
    }
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache = artifact_root / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / f"dnn_probe_{digest}.csv"
    meta_path = cache / f"dnn_probe_{digest}.json"
    expected_targets = data["confidential"]

    def cached_probe(path: Path) -> pd.DataFrame | None:
        frame = pd.read_csv(path)
        required = {"confidential", "dnn_r2", "dnn_mse", "dnn_best_epoch"}
        if required.issubset(frame.columns) and frame["confidential"].tolist() == expected_targets:
            return frame
        return None

    def legacy_config_key(config: dict[str, Any]) -> dict[str, Any]:
        legacy_probe = dict(config.get("evaluation", {}).get("dnn_probe", {}))
        legacy_training = config["training"]
        for key in ("epochs", "batch_size", "lr", "weight_decay", "patience"):
            legacy_probe.setdefault(key, legacy_training.get(key))
        csv_path = _path(config["dataset"]["csv_path"])
        stat = csv_path.stat()
        return {
            "csv": str(csv_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "drop_constant_columns": bool(config["fields"].get("drop_constant_columns", True)),
            "drop_columns": sorted(config["fields"].get("drop_columns", [])),
            "confidential": config["fields"]["confidential"],
            "train_ratio": float(legacy_training["train_ratio"]),
            "window_size": int(config["model"].get("window_size", 1)),
            "seed": int(config.get("runtime", {}).get("seed", 42)),
            "probe": {
                "hidden_dims": [int(value) for value in legacy_probe.get("hidden_dims", [128, 64])],
                "dropout": float(legacy_probe.get("dropout", 0.15)),
                "epochs": int(legacy_probe["epochs"]),
                "batch_size": int(legacy_probe["batch_size"]),
                "lr": float(legacy_probe["lr"]),
                "weight_decay": float(legacy_probe.get("weight_decay", 0.0)),
                "patience": int(legacy_probe["patience"]),
            },
        }

    if bool(probe_cfg.get("reuse_cache", True)) and cache_path.exists():
        reused = cached_probe(cache_path)
        if reused is not None:
            print(f"  Reusing DNN probe cache: {cache_path.name}")
            return reused
    if bool(probe_cfg.get("reuse_cache", True)):
        prior_caches = sorted(artifact_root.parents[1].glob(f"*/artifacts/_cache/dnn_probe_{digest}.csv"))
        prior_caches = [path for path in prior_caches if path != cache_path]
        for prior_path in reversed(prior_caches):
            reused = cached_probe(prior_path)
            if reused is not None:
                reused.to_csv(cache_path, index=False)
                _dump_json(meta_path, signature)
                print(f"  Reusing previous-run DNN probe cache: {prior_path.name}")
                return reused
        current_key = legacy_config_key(cfg)
        prior_results = sorted(artifact_root.parents[1].glob("*/artifacts/dnn_probe.csv"))
        for prior_path in reversed(prior_results):
            if prior_path == artifact_root / "dnn_probe.csv":
                continue
            prior_config = prior_path.parents[1] / "config.yaml"
            if not prior_config.exists():
                continue
            with prior_config.open("r", encoding="utf-8") as handle:
                prior_cfg = yaml.safe_load(handle)
            if legacy_config_key(prior_cfg) != current_key:
                continue
            reused = cached_probe(prior_path)
            if reused is not None:
                reused.to_csv(cache_path, index=False)
                _dump_json(meta_path, signature)
                print(f"  Reusing compatible prior-run DNN probe results: {prior_path}")
                return reused

    train_features, train_targets = build_samples(data["train_data"], data["confidential_indices"], window)
    test_features, test_targets = build_samples(data["test_data"], data["confidential_indices"], window)
    train_general = train_features[:, : data["n_general"], :].reshape(len(train_features), -1)
    test_general = test_features[:, : data["n_general"], :].reshape(len(test_features), -1)
    print(
        f"  DNN probe: General-only input={train_general.shape[1]}, window={window}, "
        f"targets={len(data['confidential'])}"
    )
    rows: list[dict[str, Any]] = []
    for position, name in enumerate(data["confidential"]):
        result = _train_probe_target(
            train_general,
            train_targets[:, position],
            test_general,
            test_targets[:, position],
            probe_cfg,
            device,
            seed + 10_000 + position,
        )
        row = {"confidential": name, **result}
        rows.append(row)
        print(f"    Probe {name}: R2={result['dnn_r2']:.4f}, epoch={result['dnn_best_epoch']}")
    probe = pd.DataFrame(rows)
    probe.to_csv(cache_path, index=False)
    _dump_json(meta_path, signature)
    return probe


def train_model(
    model: InferenceDrivenGNN,
    data: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    window = int(cfg["model"].get("window_size", 1))
    train_x, train_y = build_samples(data["train_data"], data["confidential_indices"], window)
    test_x, test_y = build_samples(data["test_data"], data["confidential_indices"], window)
    selected_indices, selected_names, excluded_names = loss_targets(data, cfg)
    excluded_indices = [data["confidential"].index(name) for name in excluded_names]
    print(f"  Window={window}, train samples={len(train_x)}, test samples={len(test_x)}")
    print(f"  Loss targets={len(selected_names)}/{len(data['confidential'])}, excluded={excluded_names or 'none'}")

    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y)),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
    )
    x_test = torch.as_tensor(test_x, device=device)
    y_test = torch.as_tensor(test_y, device=device)
    base_lr = float(cfg["training"]["lr"])
    multiplier = float(cfg["training"].get("alpha_lr_multiplier", 1.0))
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    alpha_parameters = [model.beta_general, model.beta_confidential]
    alpha_ids = {id(parameter) for parameter in alpha_parameters}
    standard_parameters = [parameter for parameter in model.parameters() if id(parameter) not in alpha_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": standard_parameters, "lr": base_lr, "weight_decay": weight_decay},
            {"params": alpha_parameters, "lr": base_lr * multiplier, "weight_decay": weight_decay},
        ]
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)
    patience = int(cfg["training"].get("patience", epochs))
    loss_name = str(cfg["training"].get("loss_fn", "mse"))
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stagnant = 0

    for epoch in range(1, epochs + 1):
        model.train()
        accumulated = 0.0
        counted = 0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = _selected_loss(model(features), targets, selected_indices, loss_name)
            loss.backward()
            optimizer.step()
            accumulated += loss.item() * len(features)
            counted += len(features)
        train_loss = accumulated / max(counted, 1)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x_test)
            test_selected = _mse(prediction, y_test, selected_indices)
            test_all = _mse(prediction, y_test)
            test_excluded = _mse(prediction, y_test, excluded_indices) if excluded_indices else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_selected,
                "test_mse_all": test_all,
                "test_mse_excluded": test_excluded,
            }
        )
        if test_selected < best_loss:
            best_loss, stagnant, best_state = test_selected, 0, deepcopy(model.state_dict())
        else:
            stagnant += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs or stagnant >= patience:
            print(f"  Epoch {epoch:4d}: train={train_loss:.6f}, test={test_selected:.6f}, all={test_all:.6f}")
        if stagnant >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_prediction = model(x_test).cpu().numpy()
    per_target: list[dict[str, Any]] = []
    for position, name in enumerate(data["confidential"]):
        actual = test_y[:, position]
        prediction = final_prediction[:, position]
        mse = float(np.mean((actual - prediction) ** 2))
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 - float(np.sum((actual - prediction) ** 2)) / max(denominator, 1e-12)
        per_target.append({"confidential": name, "r2": r2, "mse": mse, "in_loss": name in selected_names})
    return {
        "history": pd.DataFrame(history),
        "best_test_loss": best_loss,
        "per_target": pd.DataFrame(per_target).sort_values("r2", ascending=False),
        "test_features": x_test,
        "test_targets": y_test,
        "selected_indices": selected_indices,
        "selected_names": selected_names,
        "excluded_names": excluded_names,
    }


def compute_sensitivity(
    model: InferenceDrivenGNN,
    training: dict[str, Any],
    data: dict[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    features = training["test_features"]
    targets = training["test_targets"]
    model.eval()
    with torch.no_grad():
        baseline = _mse(model(features), targets, training["selected_indices"])
    rows: list[dict[str, Any]] = []
    for position, name in enumerate(data["general"]):
        masked = features.clone()
        masked[:, position, :] = 0.0
        with torch.no_grad():
            masked_loss = _mse(model(masked.to(device)), targets, training["selected_indices"])
        rows.append(
            {
                "general": name,
                "sensitivity": (masked_loss - baseline) / max(baseline, 1e-12),
                "masked_loss": masked_loss,
                "loss_increase": masked_loss - baseline,
            }
        )
    return pd.DataFrame(rows).sort_values("sensitivity", ascending=False), baseline


def _save_figure(fig: plt.Figure, path: Path, dpi: int = 180, tight_box: bool = False) -> None:
    fig.tight_layout()
    kwargs = {"bbox_inches": "tight"} if tight_box else {}
    fig.savefig(path, dpi=dpi, **kwargs)
    plt.close(fig)


def plot_general_alpha(model: InferenceDrivenGNN, metrics: list[str], path: Path) -> pd.DataFrame:
    weights = model.get_alpha_general().detach().cpu().numpy()
    frame = pd.DataFrame({"metric": metrics, "weight": weights})
    draw_general_alpha(frame, path)
    return frame


def draw_general_alpha(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(frame["metric"], frame["weight"], color=PLOT_COLORS[: len(frame)])
    ax.set_ylabel("Weight")
    ax.set_title("Learned Correlation Metric Weights (α)")
    ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, path)


def plot_loss(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"].to_numpy(), label="Train Loss", alpha=0.8)
    ax.plot(history["test_loss"].to_numpy(), label="Test Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Test Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, path)


def confidential_edge_alpha(
    model: InferenceDrivenGNN,
    data: dict[str, Any],
    edge_mask: np.ndarray,
    metrics: list[str],
) -> pd.DataFrame:
    weights = model.get_alpha_confidential().detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for target_position, target_node in enumerate(data["confidential_indices"]):
        target = data["confidential"][target_position]
        for source, source_name in enumerate(data["all_columns"]):
            if source != target_node and edge_mask[target_node, source] > 0:
                source_type = "confidential" if source >= data["n_general"] else "general"
                row: dict[str, Any] = {
                    "edge": f"{target} <- {source_name}",
                    "target": target,
                    "source": source_name,
                    "source_type": source_type,
                }
                row.update({metric: float(weights[target_position, source, index]) for index, metric in enumerate(metrics)})
                rows.append(row)
    return pd.DataFrame(rows)


def plot_confidential_alpha(frame: pd.DataFrame, metrics: list[str], path: Path) -> None:
    if frame.empty:
        return
    view = frame.copy()
    view["_dominant_alpha"] = view[metrics].max(axis=1)
    view = view.sort_values(["target", "_dominant_alpha"], ascending=[True, False])
    if "source_type" in view.columns:
        view["label"] = (
            view["target"] + " <- " + view["source_type"].str[0].str.upper() + "|" + view["source"]
        )
    else:
        view["label"] = view["edge"]
    figure_height = max(10.0, 0.075 * len(view))
    fig, ax = plt.subplots(figsize=(14, figure_height))
    left = np.zeros(len(view))
    y = np.arange(len(view))
    for metric, color in zip(metrics, PLOT_COLORS):
        values = view[metric].to_numpy()
        ax.barh(y, values, left=left, label=metric, color=color)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(view["label"], fontsize=4.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Correlation metric alpha")
    ax.set_title("Directed confidential-target edge alpha weights")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=len(metrics), fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)
    _save_figure(fig, path, dpi=220, tight_box=True)


def plot_sensitivity(frame: pd.DataFrame, path: Path) -> None:
    top = frame.head(20).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [
        "#e15759" if value > 0.1 else "#f28e2b" if value > 0.01 else "#76b7b2"
        for value in top["sensitivity"]
    ]
    positions = range(len(top) - 1, -1, -1)
    ax.barh(positions, top["sensitivity"].values, color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(top["general"].values, fontsize=8)
    ax.set_xlabel("Masking Sensitivity")
    ax.set_title("Top 20 General Fields by Sensitivity")
    ax.grid(True, axis="x", alpha=0.3)
    _save_figure(fig, path)


def comparison_table(
    per_target: pd.DataFrame,
    probe: pd.DataFrame,
    cfg: dict[str, Any],
    path: Path,
) -> pd.DataFrame:
    comparison = per_target.rename(columns={"r2": "gnn_r2"})[["confidential", "gnn_r2", "in_loss"]].copy()
    if not probe.empty:
        comparison = comparison.merge(probe[["confidential", "dnn_r2"]], on="confidential", how="left")
    else:
        comparison["dnn_r2"] = np.nan
    comparison["delta_gnn_minus_dnn"] = comparison["gnn_r2"] - comparison["dnn_r2"]
    comparison = comparison[["confidential", "dnn_r2", "gnn_r2", "delta_gnn_minus_dnn", "in_loss"]]
    configured_order = {name: index for index, name in enumerate(cfg["fields"]["confidential"])}
    comparison["_order"] = comparison["confidential"].map(configured_order).fillna(len(configured_order))
    comparison = comparison.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    draw_comparison_table(comparison, path)
    return comparison


def draw_comparison_table(comparison: pd.DataFrame, path: Path) -> None:
    rows = comparison.to_dict("records")
    fig_h = max(5.2, 0.38 * (len(rows) + 3))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    ax.text(
        0.5,
        0.98,
        "Per-Confidential R2: Single-target DNN Probe vs Current Model",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    headers = ["Confidential field", "DNN probe R2", "Model test R2", "Model - Probe", "In loss"]
    xs = [0.02, 0.54, 0.68, 0.82, 0.94]
    y_top = 0.90
    row_h = 0.82 / (len(rows) + 1)
    ax.hlines(y_top + row_h * 0.45, 0.01, 0.99, colors="black", linewidth=1.4)
    ax.hlines(y_top - row_h * 0.45, 0.01, 0.99, colors="black", linewidth=0.9)
    ax.hlines(y_top - row_h * (len(rows) + 0.55), 0.01, 0.99, colors="black", linewidth=1.4)
    for x, header in zip(xs, headers):
        ax.text(
            x,
            y_top,
            header,
            ha="left" if header == "Confidential field" else "center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
    for index, row in enumerate(rows, start=1):
        y = y_top - row_h * index
        dnn = "" if pd.isna(row["dnn_r2"]) else f"{row['dnn_r2']:.4f}"
        gnn = "" if pd.isna(row["gnn_r2"]) else f"{row['gnn_r2']:.4f}"
        delta = row["delta_gnn_minus_dnn"]
        delta_text = "" if pd.isna(delta) else f"{delta:+.4f}"
        delta_color = "black" if pd.isna(delta) else "#166534" if delta >= 0 else "#991b1b"
        ax.text(xs[0], y, row["confidential"], ha="left", va="center", fontsize=9.2)
        ax.text(xs[1], y, dnn, ha="center", va="center", fontsize=9.2)
        ax.text(xs[2], y, gnn, ha="center", va="center", fontsize=9.2)
        ax.text(xs[3], y, delta_text, ha="center", va="center", fontsize=9.2, color=delta_color)
        ax.text(xs[4], y, "yes" if row["in_loss"] else "no", ha="center", va="center", fontsize=9.2)
    ax.text(
        0.01,
        0.025,
        "DNN probe is trained in this run using only matching General windows; "
        "both models are measured on the same test split.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#4b5563",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def degree_table(
    edge_mask: np.ndarray,
    data: dict[str, Any],
    cfg: dict[str, Any],
    applied_overrides: dict[str, int],
    path: Path,
) -> pd.DataFrame:
    base = int(cfg["graph"]["top_k"])
    rows = []
    for index, name in enumerate(data["all_columns"]):
        override = name in applied_overrides
        rows.append(
            {
                "node": name,
                "type": "Confidential" if index >= data["n_general"] else "General",
                "incoming_edges": int(edge_mask[index].sum()),
                "configured_top_k": applied_overrides.get(name, base),
                "override": "yes" if override else "",
            }
        )
    frame = pd.DataFrame(rows).sort_values(["type", "incoming_edges"], ascending=[True, False])
    draw_degree_table(frame, path)
    return frame


def draw_degree_table(frame: pd.DataFrame, path: Path) -> None:
    rows = frame.to_dict("records")
    fig_h = max(8.0, 0.20 * (len(rows) + 5))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    ax.text(
        0.5,
        0.985,
        "Graph Incoming Edges by Node",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.955,
        "Top-k is the requested minimum; threshold and symmetry may retain additional edges.",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#4b5563",
    )
    headers = ["Node", "Type", "In edges", "Top-k", "Override"]
    xs = [0.02, 0.63, 0.78, 0.88, 0.96]
    y_top = 0.91
    row_h = 0.84 / (len(rows) + 1)
    ax.hlines(y_top + row_h * 0.45, 0.01, 0.99, colors="black", linewidth=1.4)
    ax.hlines(y_top - row_h * 0.45, 0.01, 0.99, colors="black", linewidth=0.9)
    ax.hlines(y_top - row_h * (len(rows) + 0.55), 0.01, 0.99, colors="black", linewidth=1.4)
    for x, header in zip(xs, headers):
        ax.text(
            x,
            y_top,
            header,
            ha="left" if header == "Node" else "center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
        )
    for index, row in enumerate(rows, start=1):
        y = y_top - row_h * index
        override = "" if pd.isna(row["override"]) else str(row["override"])
        is_override = override == "yes"
        text_color = "#b45309" if is_override else "black"
        ax.text(xs[0], y, row["node"], ha="left", va="center", fontsize=6.8, color=text_color)
        ax.text(xs[1], y, row["type"], ha="center", va="center", fontsize=6.8)
        ax.text(xs[2], y, str(row["incoming_edges"]), ha="center", va="center", fontsize=6.8)
        ax.text(xs[3], y, str(row["configured_top_k"]), ha="center", va="center", fontsize=6.8, color=text_color)
        ax.text(xs[4], y, override, ha="center", va="center", fontsize=6.8, color=text_color)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_target_r2(per_target: pd.DataFrame, path: Path) -> None:
    ordered = per_target.sort_values("r2", ascending=False)
    values = ordered["r2"].to_numpy()
    colors = ["#e15759" if value > 0.8 else "#f28e2b" if value > 0.5 else "#76b7b2" for value in values]
    positions = range(len(ordered) - 1, -1, -1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["confidential"].values, fontsize=8)
    ax.set_xlabel("Test R²")
    ax.set_title("Per-Confidential Field Reconstruction R²")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    _save_figure(fig, path)


def redraw_run_plots(run_path: str | Path) -> Path:
    run_root = _argument_path(run_path)
    artifacts = run_root / "artifacts"
    plots = run_root / "plots"
    if not artifacts.is_dir() or not plots.is_dir():
        raise FileNotFoundError(f"Expected plots/ and artifacts/ under run directory: {run_root}")
    config_path = run_root / "config.yaml"
    if not config_path.exists():
        config_path = artifacts / "config_used.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    metrics = cfg["correlation"]["metrics"]
    draw_general_alpha(pd.read_csv(artifacts / "alpha_general.csv"), plots / "general_alpha.png")
    plot_loss(pd.read_csv(artifacts / "training.csv"), plots / "loss.png")
    edge_frame = pd.read_csv(artifacts / "alpha_confidential_edges.csv")
    if "source_type" not in edge_frame.columns:
        confidential = set(cfg["fields"]["confidential"])
        edge_frame["source_type"] = edge_frame["source"].map(
            lambda name: "confidential" if name in confidential else "general"
        )
    plot_confidential_alpha(edge_frame, metrics, plots / "conf_edge_alpha.png")
    plot_sensitivity(pd.read_csv(artifacts / "sensitivity.csv"), plots / "sensitivity.png")
    comparison = pd.read_csv(artifacts / "dnn_comparison.csv")
    configured_order = {name: index for index, name in enumerate(cfg["fields"]["confidential"])}
    comparison["_order"] = comparison["confidential"].map(configured_order).fillna(len(configured_order))
    comparison = comparison.sort_values("_order").drop(columns="_order")
    draw_comparison_table(comparison, plots / "dnn_compare.png")
    draw_degree_table(pd.read_csv(artifacts / "node_edges.csv"), plots / "node_edges.png")
    plot_target_r2(pd.read_csv(artifacts / "per_target.csv"), plots / "target_r2.png")
    print(f"  Redrew 7 plots in original DNN style: {plots}")
    return plots


def run_experiment(config_path: str | Path) -> Path:
    config_path = _argument_path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    supported_graph_keys = {"top_k", "threshold", "symmetrize", "target_top_k_overrides"}
    unexpected_graph_keys = sorted(set(cfg.get("graph", {})) - supported_graph_keys)
    if unexpected_graph_keys:
        raise ValueError(
            "Unknown graph setting(s): "
            f"{unexpected_graph_keys}. Put node-specific top-k values under "
            "graph.target_top_k_overrides."
        )
    seed = int(cfg.get("runtime", {}).get("seed", 42))
    _set_seed(seed)
    device = _device(cfg.get("runtime", {}).get("device", "cpu"))
    output_root = _path(cfg["dataset"].get("output_dir", "outputs"))
    data = prepare_data(cfg)
    configured_exclusions = cfg["training"].get("loss_exclude_confidential", [])
    loss_tag = "lossexclude" if configured_exclusions else "allloss"
    name = _safe_name(cfg.get("experiment", {}).get("name", "dnn28"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{stamp}_{name}_{loss_tag}"
    run_root = output_root / run_id
    plot_root = run_root / "plots"
    artifact_root = run_root / "artifacts"
    plot_root.mkdir(parents=True, exist_ok=False)
    artifact_root.mkdir(parents=True, exist_ok=False)
    run_dir = artifact_root
    prefix = ""
    with (run_root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    print(f"  Run: {run_id}")
    print(f"  Output folders: {plot_root} ; {artifact_root}")

    tensor = load_or_compute_metrics(data, cfg, artifact_root, seed)
    edge_mask = build_edge_mask(
        tensor,
        int(cfg["graph"]["top_k"]),
        float(cfg["graph"]["threshold"]),
        bool(cfg["graph"].get("symmetrize", True)),
    )
    overrides = apply_target_top_k(
        edge_mask,
        tensor,
        data["all_columns"],
        cfg["graph"].get("target_top_k_overrides", {}),
    )
    if overrides:
        print(f"  Applied target top-k overrides: {overrides}")
    model = InferenceDrivenGNN(
        tensor,
        edge_mask,
        data["n_nodes"],
        data["n_general"],
        data["confidential_indices"],
        cfg["model"],
    ).to(device)
    print(
        f"  Model: window={cfg['model'].get('window_size', 1)}, "
        f"attention_heads={cfg['model'].get('attention_heads', 1)}"
    )
    training = train_model(model, data, cfg, device)
    sensitivity, baseline = compute_sensitivity(model, training, data, device)
    probe = train_dnn_probe(data, cfg, device, seed, artifact_root)

    alpha_general = plot_general_alpha(model, cfg["correlation"]["metrics"], plot_root / f"{prefix}general_alpha.png")
    plot_loss(training["history"], plot_root / f"{prefix}loss.png")
    alpha_edges = confidential_edge_alpha(model, data, edge_mask, cfg["correlation"]["metrics"])
    plot_confidential_alpha(alpha_edges, cfg["correlation"]["metrics"], plot_root / f"{prefix}conf_edge_alpha.png")
    plot_sensitivity(sensitivity, plot_root / f"{prefix}sensitivity.png")
    comparison = comparison_table(training["per_target"], probe, cfg, plot_root / f"{prefix}dnn_compare.png")
    degrees = degree_table(edge_mask, data, cfg, overrides, plot_root / f"{prefix}node_edges.png")
    plot_target_r2(training["per_target"], plot_root / f"{prefix}target_r2.png")

    training["history"].to_csv(run_dir / "training.csv", index=False)
    training["per_target"].to_csv(run_dir / "per_target.csv", index=False)
    sensitivity.to_csv(run_dir / "sensitivity.csv", index=False)
    alpha_general.to_csv(run_dir / "alpha_general.csv", index=False)
    alpha_edges.to_csv(run_dir / "alpha_confidential_edges.csv", index=False)
    probe.to_csv(run_dir / "dnn_probe.csv", index=False)
    comparison.to_csv(run_dir / "dnn_comparison.csv", index=False)
    degrees.to_csv(run_dir / "node_edges.csv", index=False)
    np.save(run_dir / "edge_mask.npy", edge_mask)
    torch.save(model.state_dict(), run_dir / "model.pt")
    summary = {
        "run_id": run_id,
        "loss_tag": loss_tag,
        "device": str(device),
        "window_size": int(cfg["model"].get("window_size", 1)),
        "attention_heads": int(cfg["model"].get("attention_heads", 1)),
        "excluded_from_loss": training["excluded_names"],
        "target_top_k_overrides": overrides,
        "best_test_loss": training["best_test_loss"],
        "sensitivity_baseline_loss": baseline,
        "per_target_r2": dict(zip(training["per_target"]["confidential"], training["per_target"]["r2"])),
        "dnn_probe_r2": dict(zip(probe["confidential"], probe["dnn_r2"])),
        "plots": sorted(path.name for path in plot_root.glob("*.png")),
    }
    _dump_json(run_dir / "summary.json", summary)
    print(f"  Best test loss: {training['best_test_loss']:.6f}")
    print(f"  Plots written: {len(summary['plots'])} -> {plot_root}")
    print(f"  Artifacts written: {run_dir}")
    return run_dir
