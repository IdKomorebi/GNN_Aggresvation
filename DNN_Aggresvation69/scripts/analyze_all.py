#!/usr/bin/env python3
"""DNN69 end-to-end analysis and figures.

Full-population truth is the dedicated-DNN retrain result.  Where both dedicated
DNN and GCN retrains are available, a separate audit uses their per-confidential
maximum.  Keeping the two references separate avoids calling incomplete GCN
coverage a universal best-of-structure truth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dnn69")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
KGRID = [0, 10, 50, 200]
BANDS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 44)]


def band_of(size: int) -> str:
    for lo, hi in BANDS:
        if lo <= size <= hi:
            return f"{lo}-{hi}"
    return "unknown"


def safe_rho(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) < 1e-12 or np.ptp(y[keep]) < 1e-12:
        return float("nan")
    return float(spearmanr(x[keep], y[keep]).statistic)


def top_overlap(truth, estimate, k: int) -> float:
    truth = np.asarray(truth)
    estimate = np.asarray(estimate)
    if len(truth) < k:
        return float("nan")
    a = set(np.argsort(-truth)[:k])
    b = set(np.argsort(-estimate)[:k])
    return len(a & b) / k


def binary_stats(truth, estimate, threshold: float = 0.1) -> tuple[float, float]:
    actual = np.asarray(truth) > threshold
    predicted = np.asarray(estimate) > threshold
    tp = int((actual & predicted).sum())
    precision = tp / max(int(predicted.sum()), 1)
    recall = tp / max(int(actual.sum()), 1)
    return precision, recall


def source_truth_path(sid: str, source: str, struct: str) -> Path:
    if source == "d60_random":
        return REPO / f"DNN_Aggresvation60/outputs/retrain/{sid}_{struct}_seed0.json"
    if struct == "gcn":
        return ROOT / f"outputs/retrain_gcn/{sid}_gcn_seed0.json"
    if source == "d67_single_pair":
        return REPO / f"DNN_Aggresvation67/outputs/retrain/{sid}_dnn_seed0.json"
    if source == "d68_triple":
        return REPO / f"DNN_Aggresvation68/outputs/retrain/{sid}_dnn_seed0.json"
    raise ValueError(source)


def load_truth(registry: dict, allow_incomplete: bool) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    audit_path = ROOT / "outputs/gcn_audit_subsets.json"
    audit_ids = set(json.loads(audit_path.read_text())) if audit_path.exists() else set()
    for sid, meta in sorted(registry.items()):
        paths = {s: source_truth_path(sid, meta["source"], s) for s in ["dnn", "gcn"]}
        if not paths["dnn"].exists():
            missing.append(str(paths["dnn"].relative_to(REPO)))
            continue
        gcn_expected = meta["source"] == "d60_random" or sid in audit_ids
        if gcn_expected and not paths["gcn"].exists():
            missing.append(str(paths["gcn"].relative_to(REPO)))
            if not allow_incomplete:
                continue
        values = {"dnn": json.loads(paths["dnn"].read_text())}
        if paths["gcn"].exists():
            values["gcn"] = json.loads(paths["gcn"].read_text())
        for struct, record in values.items():
            if record["fields"] != meta["fields"]:
                raise ValueError(f"field mismatch {sid}/{struct}")
        confs = sorted(values["dnn"]["per_conf_r2"])
        if "gcn" in values and confs != sorted(values["gcn"]["per_conf_r2"]):
            raise ValueError(f"confidential targets mismatch {sid}")
        dnn_mean = float(values["dnn"]["mean_r2"])
        gcn_mean = float(values["gcn"]["mean_r2"]) if "gcn" in values else float("nan")
        mean_winner = ("gcn" if gcn_mean > dnn_mean else "dnn") if "gcn" in values else "unavailable"
        for conf in confs:
            dnn = float(values["dnn"]["per_conf_r2"][conf])
            gcn = float(values["gcn"]["per_conf_r2"][conf]) if "gcn" in values else float("nan")
            rows.append({
                "sid": sid,
                "source": meta["source"],
                "group": meta["group"],
                "size": meta["size"],
                "band": band_of(meta["size"]),
                "canonical_sid": meta.get("canonical_sid", sid),
                "is_duplicate_entry": bool(meta.get("is_duplicate_entry", False)),
                "conf": conf,
                "dnn": dnn,
                "gcn": gcn,
                "truth": dnn,
                "truth_dnn": dnn,
                "truth_best": max(dnn, gcn) if np.isfinite(gcn) else float("nan"),
                "has_gcn": bool(np.isfinite(gcn)),
                "winner": ("gcn" if gcn > dnn else "dnn") if np.isfinite(gcn) else "unavailable",
                "mean_dnn": dnn_mean,
                "mean_gcn": gcn_mean,
                "mean_winner": mean_winner,
            })
    if missing and not allow_incomplete:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} truth files; first entries:\n{preview}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no complete truth records")
    return frame, missing


def cohort_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    primary = ~frame["is_duplicate_entry"] if "is_duplicate_entry" in frame else pd.Series(True, index=frame.index)
    masks = {
        "all": primary,
        "wide_all": frame["source"].eq("d60_random"),
        "single": frame["group"].eq("single"),
        "pair": frame["group"].eq("pair"),
        "triple_all": frame["group"].isin(["triple_top", "triple_rand"]),
        "triple_top": frame["group"].eq("triple_top"),
        "triple_rand": frame["group"].eq("triple_rand"),
    }
    for lo, hi in BANDS:
        label = f"{lo}-{hi}"
        masks[f"wide_{label}"] = frame["source"].eq("d60_random") & frame["band"].eq(label)
    return masks


def attacker_summary(truth: pd.DataFrame) -> pd.DataFrame:
    available = truth[truth.has_gcn].copy()
    subset = available.groupby(["sid", "source", "group", "size", "band",
                            "canonical_sid", "is_duplicate_entry"], as_index=False).agg(
        dnn=("dnn", "mean"),
        gcn=("gcn", "mean"),
        best_per_conf=("truth_best", "mean"),
    )
    rows = []
    for cohort, mask in cohort_masks(subset).items():
        g = subset[mask]
        if g.empty:
            continue
        delta = g.gcn - g.dnn
        rows.append({
            "cohort": cohort,
            "n_subsets": g.sid.nunique(),
            "dnn_mean_r2": g.dnn.mean(),
            "gcn_mean_r2": g.gcn.mean(),
            "gcn_minus_dnn": delta.mean(),
            "gcn_win_fraction": (delta > 0).mean(),
            "gcn_significant_wins": int((delta > 0.02).sum()),
            "dnn_significant_wins": int((delta < -0.02).sum()),
            "best_per_conf_mean_r2": g.best_per_conf.mean(),
            "per_conf_envelope_gain": (g.best_per_conf - np.maximum(g.dnn, g.gcn)).mean(),
        })
    return pd.DataFrame(rows)


def attacker_per_conf_summary(truth: pd.DataFrame) -> pd.DataFrame:
    available = truth[truth.has_gcn].copy()
    rows = []
    for cohort, mask in cohort_masks(available).items():
        part = available[mask]
        if part.empty:
            continue
        for conf, g in part.groupby("conf"):
            rows.append({"cohort": cohort, "conf": conf, "n_subsets": g.sid.nunique(),
                         "dnn_mean_r2": g.dnn.mean(), "gcn_mean_r2": g.gcn.mean(),
                         "gcn_minus_dnn": (g.gcn - g.dnn).mean(),
                         "gcn_win_fraction": (g.gcn > g.dnn).mean()})
    return pd.DataFrame(rows)


def load_estimates(registry: dict, allow_incomplete: bool) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    for arch in ["mlp", "gnn"]:
        for sid, meta in sorted(registry.items()):
            path = ROOT / f"outputs/est/{arch}/{sid}.json"
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            record = json.loads(path.read_text())
            if record["fields"] != meta["fields"]:
                raise ValueError(f"estimate field mismatch {arch}/{sid}")
            available = {int(k) for k in record["by_k"]}
            if available != set(KGRID):
                raise ValueError(f"K grid mismatch {arch}/{sid}: {sorted(available)}")
            for k_text, snap in record["by_k"].items():
                k = int(k_text)
                for conf, value in snap["per_conf_r2"].items():
                    rows.append({
                        "sid": sid,
                        "source": meta["source"],
                        "group": meta["group"],
                        "size": meta["size"],
                        "band": band_of(meta["size"]),
                        "arch": arch,
                        "K": k,
                        "conf": conf,
                        "est": float(value),
                        "time": float(snap["time"]),
                    })
    if missing and not allow_incomplete:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} estimate files; first entries:\n{preview}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no estimates")
    return frame, missing


def fidelity_summary(joined: pd.DataFrame, truth_col: str, reference: str) -> pd.DataFrame:
    rows = []
    for (arch, k), part in joined.groupby(["arch", "K"]):
        for cohort, mask in cohort_masks(part).items():
            g = part[mask]
            if g.empty:
                continue
            by_subset = g.groupby("sid", as_index=False).agg(truth=(truth_col, "mean"), est=("est", "mean"))
            conf_rhos = [safe_rho(x.est, x[truth_col]) for _, x in g.groupby("conf")]
            conf_rhos = [x for x in conf_rhos if np.isfinite(x)]
            subset_times = g.groupby("sid")["time"].first()
            rows.append({
                "arch": arch,
                "K": int(k),
                "cohort": cohort,
                "reference": reference,
                "n_subsets": by_subset.sid.nunique(),
                "n_entries": len(g),
                "mae_per_conf": (g.est - g[truth_col]).abs().mean(),
                "bias_per_conf": (g.est - g[truth_col]).mean(),
                "mean_est_r2": g.est.mean(),
                "mean_truth_r2": g[truth_col].mean(),
                "subset_mean_spearman": safe_rho(by_subset.est, by_subset.truth),
                "median_per_conf_spearman": float(np.median(conf_rhos)) if conf_rhos else float("nan"),
                "top20_subset_overlap": top_overlap(by_subset.truth, by_subset.est, 20),
                "mean_time_per_subset": subset_times.mean(),
            })
    return pd.DataFrame(rows)


def fidelity_per_conf_summary(joined: pd.DataFrame, truth_col: str, reference: str) -> pd.DataFrame:
    rows = []
    keep_cohorts = {"wide_all", "single", "pair", "triple_all", "triple_top", "triple_rand"}
    for (arch, k), part in joined.groupby(["arch", "K"]):
        for cohort, mask in cohort_masks(part).items():
            if cohort not in keep_cohorts:
                continue
            selected = part[mask]
            for conf, g in selected.groupby("conf"):
                rows.append({"reference": reference, "cohort": cohort, "arch": arch,
                             "K": int(k), "conf": conf, "n_subsets": g.sid.nunique(),
                             "mae": (g.est - g[truth_col]).abs().mean(),
                             "bias": (g.est - g[truth_col]).mean(),
                             "spearman": safe_rho(g.est, g[truth_col])})
    return pd.DataFrame(rows)


def estimate_map(est: pd.DataFrame, arch: str, k: int) -> dict[tuple[str, str], float]:
    part = est[(est.arch == arch) & (est.K == k)]
    return {(r.sid, r.conf): float(r.est) for r in part.itertuples()}


def synergy_details(truth: pd.DataFrame, est: pd.DataFrame, registry: dict, order: int) -> pd.DataFrame:
    truth_map = {(r.sid, r.conf): float(r.truth) for r in truth.itertuples()}
    confs = sorted(truth.conf.unique())
    rows = []
    if order == 2:
        target_sids = [sid for sid, meta in registry.items() if meta["group"] == "pair"]
    elif order == 3:
        target_sids = [sid for sid, meta in registry.items()
                       if meta["group"] in ["triple_top", "triple_rand"]]
    else:
        raise ValueError(order)

    for arch in ["mlp", "gnn"]:
        for k in KGRID:
            emap = estimate_map(est, arch, k)
            for sid in target_sids:
                ids = [int(x) for x in sid[1:].split("_")]
                if len(ids) != order:
                    raise ValueError(sid)
                if order == 2:
                    lower = [f"s{idx:02d}" for idx in ids]
                else:
                    lower = [f"p{a:02d}_{b:02d}" for i, a in enumerate(ids)
                             for b in ids[i + 1:]]
                for conf in confs:
                    keys_truth = [(sid, conf)] + [(x, conf) for x in lower]
                    if any(key not in truth_map for key in keys_truth):
                        continue
                    if any(key not in emap for key in keys_truth):
                        continue
                    truth_syn = truth_map[(sid, conf)] - max(truth_map[(x, conf)] for x in lower)
                    est_syn = emap[(sid, conf)] - max(emap[(x, conf)] for x in lower)
                    rows.append({
                        "order": order,
                        "sid": sid,
                        "group": registry[sid]["group"],
                        "arch": arch,
                        "K": k,
                        "conf": conf,
                        "truth_syn": truth_syn,
                        "est_syn": est_syn,
                    })
    return pd.DataFrame(rows)


def synergy_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (order, arch, k), g in detail.groupby(["order", "arch", "K"]):
        conf_rhos = [safe_rho(x.est_syn, x.truth_syn) for _, x in g.groupby("conf")]
        conf_rhos = [x for x in conf_rhos if np.isfinite(x)]
        precision, recall = binary_stats(g.truth_syn, g.est_syn)
        top = g[g.group.eq("triple_top")]
        rand = g[g.group.eq("triple_rand")]
        rows.append({
            "order": int(order),
            "arch": arch,
            "K": int(k),
            "n_entries": len(g),
            "spearman": safe_rho(g.est_syn, g.truth_syn),
            "median_per_conf_spearman": float(np.median(conf_rhos)) if conf_rhos else float("nan"),
            "mae": (g.est_syn - g.truth_syn).abs().mean(),
            "bias": (g.est_syn - g.truth_syn).mean(),
            "top20_overlap": top_overlap(g.truth_syn, g.est_syn, 20),
            "top50_overlap": top_overlap(g.truth_syn, g.est_syn, 50),
            "precision_syn_gt_0.1": precision,
            "recall_syn_gt_0.1": recall,
            "truth_top_minus_random": (top.truth_syn.mean() - rand.truth_syn.mean()) if len(top) and len(rand) else float("nan"),
            "est_top_minus_random": (top.est_syn.mean() - rand.est_syn.mean()) if len(top) and len(rand) else float("nan"),
        })
    return pd.DataFrame(rows)


def synergy_per_conf_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (order, arch, k, conf), g in detail.groupby(["order", "arch", "K", "conf"]):
        precision, recall = binary_stats(g.truth_syn, g.est_syn)
        rows.append({"order": int(order), "arch": arch, "K": int(k), "conf": conf,
                     "n_subsets": g.sid.nunique(), "spearman": safe_rho(g.est_syn, g.truth_syn),
                     "mae": (g.est_syn - g.truth_syn).abs().mean(),
                     "bias": (g.est_syn - g.truth_syn).mean(),
                     "precision_syn_gt_0.1": precision, "recall_syn_gt_0.1": recall})
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, rng: np.random.RandomState, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, float)
    if len(values) < 2:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bootstrap_architecture_comparison(joined: pd.DataFrame, synergy_detail: pd.DataFrame) -> pd.DataFrame:
    """Cluster bootstrap by subset; positive delta means MLP has lower MAE."""
    rng = np.random.RandomState(69)
    rows = []
    for reference, truth_col, source in [
        ("dnn_full", "truth_dnn", joined),
        ("best_struct_audit", "truth_best", joined[joined.has_gcn]),
    ]:
        for k in KGRID:
            part = source[source.K == k]
            for cohort in ["wide_all", "single", "pair", "triple_all"]:
                g = part[cohort_masks(part)[cohort]]
                pivot = g.pivot_table(index=["sid", "conf"], columns="arch", values="est").dropna()
                if pivot.empty:
                    continue
                truth_map = g.drop_duplicates(["sid", "conf"]).set_index(["sid", "conf"])[truth_col]
                pivot["truth"] = truth_map.reindex(pivot.index)
                pivot["delta"] = (pivot.gnn - pivot.truth).abs() - (pivot.mlp - pivot.truth).abs()
                by_sid = pivot.groupby(level="sid").delta.mean().to_numpy()
                lo, hi = bootstrap_mean(by_sid, rng)
                rows.append({"task": "direct_mae", "reference": reference, "cohort": cohort,
                             "order": 0, "K": k, "n_subsets": len(by_sid),
                             "gnn_minus_mlp_mae": by_sid.mean(), "ci95_low": lo, "ci95_high": hi,
                             "fraction_subsets_mlp_lower_mae": (by_sid > 0).mean()})

    for (order, k), g in synergy_detail.groupby(["order", "K"]):
        pivot = g.pivot_table(index=["sid", "conf"], columns="arch", values="est_syn").dropna()
        truth_map = g.drop_duplicates(["sid", "conf"]).set_index(["sid", "conf"])["truth_syn"]
        pivot["truth"] = truth_map.reindex(pivot.index)
        pivot["delta"] = (pivot.gnn - pivot.truth).abs() - (pivot.mlp - pivot.truth).abs()
        by_sid = pivot.groupby(level="sid").delta.mean().to_numpy()
        lo, hi = bootstrap_mean(by_sid, rng)
        rows.append({"task": "synergy_mae", "reference": "dnn_full", "cohort": f"order_{int(order)}",
                     "order": int(order), "K": int(k), "n_subsets": len(by_sid),
                     "gnn_minus_mlp_mae": by_sid.mean(), "ci95_low": lo, "ci95_high": hi,
                     "fraction_subsets_mlp_lower_mae": (by_sid > 0).mean()})
    return pd.DataFrame(rows)


def k0_seed_summary(truth: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "outputs/k0_multiseed.csv"
    if not path.exists():
        return pd.DataFrame()
    seed = pd.read_csv(path).merge(
        truth[["sid", "conf", "truth_dnn", "is_duplicate_entry"]],
        on=["sid", "conf"], how="inner"
    )
    seed["band"] = seed["size"].map(lambda value: band_of(int(value)))
    rows = []
    for (arch, s), part in seed.groupby(["arch", "seed"]):
        for cohort, mask in cohort_masks(part).items():
            g = part[mask]
            if g.empty:
                continue
            by_sid = g.groupby("sid", as_index=False).agg(r2=("r2", "mean"), truth=("truth_dnn", "mean"))
            rows.append({"arch": arch, "seed": int(s), "cohort": cohort,
                         "mae": (g.r2 - g.truth_dnn).abs().mean(),
                         "bias": (g.r2 - g.truth_dnn).mean(),
                         "spearman": safe_rho(by_sid.r2, by_sid.truth)})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    return raw.groupby(["arch", "cohort"], as_index=False).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        bias_mean=("bias", "mean"), spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"), n_seeds=("seed", "nunique"),
    )


def plot_results(attacker, fidelity, synergy, joined) -> None:
    out = ROOT / "outputs"
    fidelity = fidelity[fidelity.reference == "dnn_full"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cohorts = ["wide_1-4", "wide_5-8", "wide_9-16", "wide_17-32", "wide_33-44",
               "single", "pair", "triple_all"]
    a = attacker.set_index("cohort").reindex(cohorts).dropna(how="all")
    axes[0, 0].bar(np.arange(len(a)), a.gcn_minus_dnn, color=np.where(a.gcn_minus_dnn >= 0, "tab:blue", "tab:orange"))
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].axhline(0.02, color="grey", lw=0.8, ls="--")
    axes[0, 0].axhline(-0.02, color="grey", lw=0.8, ls="--")
    axes[0, 0].set_xticks(np.arange(len(a)), a.index, rotation=35, ha="right")
    axes[0, 0].set_ylabel("dedicated GCN - DNN mean R2")
    axes[0, 0].set_title("(a) Dedicated attacker upper-bound comparison")

    colors = {"mlp": "tab:orange", "gnn": "tab:blue"}
    for arch in ["mlp", "gnn"]:
        for cohort, ls in [("wide_all", "-"), ("pair", "--"), ("triple_all", ":")]:
            g = fidelity[(fidelity.arch == arch) & (fidelity.cohort == cohort)].sort_values("K")
            axes[0, 1].plot(g.K, g.mae_per_conf, marker="o", color=colors[arch], ls=ls,
                            label=f"{arch}/{cohort}")
    axes[0, 1].set_xscale("symlog")
    axes[0, 1].set_xlabel("fine-tune steps K")
    axes[0, 1].set_ylabel("per-conf MAE to dedicated DNN retrain")
    axes[0, 1].set_title("(b) Universal-oracle absolute fidelity")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[0, 1].grid(alpha=0.25)

    for arch in ["mlp", "gnn"]:
        for cohort, ls in [("wide_all", "-"), ("pair", "--"), ("triple_all", ":")]:
            g = fidelity[(fidelity.arch == arch) & (fidelity.cohort == cohort)].sort_values("K")
            axes[0, 2].plot(g.K, g.subset_mean_spearman, marker="o", color=colors[arch], ls=ls,
                            label=f"{arch}/{cohort}")
    axes[0, 2].set_xscale("symlog")
    axes[0, 2].set_ylim(0, 1.02)
    axes[0, 2].set_xlabel("fine-tune steps K")
    axes[0, 2].set_ylabel("Spearman of direct v(S)")
    axes[0, 2].set_title("(c) Same-population sensitivity ranking")
    axes[0, 2].grid(alpha=0.25)

    for order, ax in [(2, axes[1, 0]), (3, axes[1, 1])]:
        for arch in ["mlp", "gnn"]:
            g = synergy[(synergy.order == order) & (synergy.arch == arch)].sort_values("K")
            ax.plot(g.K, g.spearman, "o-", color=colors[arch], label=f"{arch} Spearman")
            ax.plot(g.K, g.top20_overlap, "^--", color=colors[arch], alpha=0.7,
                    label=f"{arch} top20")
        ax.set_xscale("symlog")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("fine-tune steps K")
        ax.set_ylabel("agreement with retrain synergy")
        ax.set_title(f"({'d' if order == 2 else 'e'}) Order-{order} synergy fidelity")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    subset_time = joined.groupby(["arch", "K", "sid"], as_index=False).time.first()
    time_summary = subset_time.groupby(["arch", "K"], as_index=False).time.mean()
    for arch in ["mlp", "gnn"]:
        g = time_summary[time_summary.arch == arch]
        axes[1, 2].plot(g.K, g.time, "o-", color=colors[arch], label=arch)
    axes[1, 2].set_xscale("symlog")
    axes[1, 2].set_xlabel("fine-tune steps K")
    axes[1, 2].set_ylabel("seconds / subset")
    axes[1, 2].set_title("(f) Equal-step computational cost")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend()
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=170)
    plt.close(fig)

    # Broad random subsets: make the size-band comparison explicit at K=0 and K=200.
    bands = [f"wide_{lo}-{hi}" for lo, hi in BANDS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(bands))
    width = 0.18
    for panel, k in enumerate([0, 200]):
        ax = axes[panel]
        for ai, arch in enumerate(["mlp", "gnn"]):
            vals = []
            for cohort in bands:
                row = fidelity[(fidelity.arch == arch) & (fidelity.K == k) & (fidelity.cohort == cohort)]
                vals.append(float(row.mae_per_conf.iloc[0]) if len(row) else np.nan)
            ax.bar(x + (ai - 0.5) * width, vals, width, label=arch, color=colors[arch])
        ax.set_xticks(x, [b.replace("wide_", "") for b in bands])
        ax.set_xlabel("subset-size band")
        ax.set_ylabel("per-conf MAE")
        ax.set_title(f"K={k}: fidelity by size")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fidelity_by_size.png", dpi=170)
    plt.close(fig)


def fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str], digits=4) -> str:
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [head, rule]
    for row in frame[columns].itertuples(index=False, name=None):
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append(fmt(float(value), digits))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(registry, missing_truth, missing_est, attacker, fidelity, synergy, seed_summary) -> None:
    attacker_show = attacker[attacker.cohort.isin(
        ["wide_1-4", "wide_5-8", "wide_9-16", "wide_17-32", "wide_33-44",
         "single", "pair", "triple_all"]
    )].copy()
    fidelity_show = fidelity[
        fidelity.cohort.isin(["wide_all", "single", "pair", "triple_all"])
        & fidelity.K.isin([0, 10, 50, 200])
    ].sort_values(["reference", "cohort", "K", "arch"])
    synergy_show = synergy[synergy.K.isin([0, 10, 50, 200])].sort_values(["order", "K", "arch"])
    manifest = json.loads((ROOT / "outputs/source_manifest.json").read_text())

    lines = [
        "# DNN_Aggresvation69 Results",
        "",
        "## Question",
        "",
        "Separate two roles that were previously conflated: (1) dedicated DNN/GCN attacker upper bound, "
        "and (2) MLP/GNN universal masked oracle fidelity before and after equal-step fine-tuning. "
        "Evaluate both direct sensitivity and order-2/order-3 synergy.",
        "",
        "## Plain-language result",
        "",
        "- A dedicated GCN attacker is useful mainly for medium-sized fixed subsets (5-16 fields).",
        "- For one universal random-mask estimator, MLP is closer to retraining than GNN at K=0 and after equal-step fine-tuning.",
        "- GNN does not reveal cleaner order-2/order-3 synergy in this implementation; its global synergy ranking is consistently worse.",
        "- Fine-tuning closes much of the amortization gap for both models, but MLP remains more accurate and substantially faster.",
        "",
        "## Data and truth protocol",
        "",
        f"- Registry: {manifest['n_subsets']} subsets: {manifest['source_counts']}.",
        "- Same shuffled split and normalization as D60/D63/D67/D68; split seed 42.",
        "- Full-population reference: dedicated DNN retrain. Best-of-structure reference is reported separately on all 50 D60 random subsets plus the 18-subset GCN audit sample.",
        "- Fine-tuning comparison: same seed-0 universal checkpoint family, batches, Adam settings, and K grid.",
        f"- Missing truth files at analysis time: {len(missing_truth)}; missing estimate files: {len(missing_est)}.",
        "",
        "## Dedicated attacker upper bound",
        "",
        markdown_table(attacker_show, ["cohort", "n_subsets", "dnn_mean_r2", "gcn_mean_r2",
                                      "gcn_minus_dnn", "gcn_win_fraction",
                                      "gcn_significant_wins", "dnn_significant_wins"]),
        "",
        "## Universal oracle fidelity to retrain references",
        "",
        markdown_table(fidelity_show, ["reference", "cohort", "K", "arch", "mae_per_conf", "bias_per_conf",
                                       "subset_mean_spearman", "median_per_conf_spearman",
                                       "mean_time_per_subset"]),
        "",
        "## Synergy fidelity",
        "",
        markdown_table(synergy_show, ["order", "K", "arch", "spearman", "mae",
                                      "top20_overlap", "precision_syn_gt_0.1",
                                      "recall_syn_gt_0.1"]),
        "",
    ]
    if not seed_summary.empty:
        seed_show = seed_summary[seed_summary.cohort.isin(["wide_all", "single", "pair", "triple_all"])]
        lines.extend([
            "## K=0 three-seed robustness",
            "",
            markdown_table(seed_show, ["cohort", "arch", "mae_mean", "mae_std",
                                       "spearman_mean", "spearman_std", "n_seeds"]),
            "",
        ])
    lines.extend([
        "## Required caveats",
        "",
        "- The full-population reference is one-seed dedicated-DNN retraining, not a mathematical supremum over all models.",
        "- The separate best-of-structure audit is the per-target maximum of the tested DNN and GCN retrains only; it is still an empirical lower bound on an unrestricted attacker's optimum.",
        "- D67/D68 dedicated truth uses one training seed; D60 noise-floor results should be used for significance interpretation.",
        "- GCN is exhaustively available only for D60's 50 broad random subsets and a deterministic 18-subset pair/triple audit sample; full pair/triple synergy fidelity therefore uses DNN retrain truth.",
        "- D68 `triple_top` was selected using an earlier MLP-based scan, so architecture comparison on that stratum has selection bias; "
        "the 200 random triples are the unbiased order-3 check.",
        "- Direct sensitivity ranking and synergy ranking are separate claims because subtraction amplifies estimation error.",
        "",
        "Figures: `outputs/overview.png`, `outputs/fidelity_by_size.png`.",
    ])
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    registry = json.loads((ROOT / "outputs/subsets.json").read_text())
    truth, missing_truth = load_truth(registry, args.allow_incomplete)
    est, missing_est = load_estimates(registry, args.allow_incomplete)
    joined = est.merge(truth[["sid", "conf", "dnn", "gcn", "truth", "truth_dnn",
                              "truth_best", "has_gcn", "canonical_sid", "is_duplicate_entry"]],
                       on=["sid", "conf"], how="inner", validate="many_to_one")
    if joined.empty:
        raise RuntimeError("truth/estimate join is empty")

    attacker = attacker_summary(truth)
    attacker_per_conf = attacker_per_conf_summary(truth)
    fidelity_dnn = fidelity_summary(joined, "truth_dnn", "dnn_full")
    best_joined = joined[joined.has_gcn].copy()
    fidelity_best = fidelity_summary(best_joined, "truth_best", "best_struct_audit")
    fidelity = pd.concat([fidelity_dnn, fidelity_best], ignore_index=True)
    fidelity_per_conf = pd.concat([
        fidelity_per_conf_summary(joined, "truth_dnn", "dnn_full"),
        fidelity_per_conf_summary(best_joined, "truth_best", "best_struct_audit"),
    ], ignore_index=True)
    pair_detail = synergy_details(truth, est, registry, order=2)
    triple_detail = synergy_details(truth, est, registry, order=3)
    synergy_detail = pd.concat([pair_detail, triple_detail], ignore_index=True)
    synergy = synergy_summary(synergy_detail)
    synergy_per_conf = synergy_per_conf_summary(synergy_detail)
    bootstrap = bootstrap_architecture_comparison(joined, synergy_detail)
    seed_summary = k0_seed_summary(truth)

    out = ROOT / "outputs"
    truth.to_csv(out / "truth_long.csv", index=False)
    attacker.to_csv(out / "attacker_upper_bound.csv", index=False)
    attacker_per_conf.to_csv(out / "attacker_upper_bound_per_conf.csv", index=False)
    fidelity.to_csv(out / "oracle_fidelity.csv", index=False)
    fidelity_per_conf.to_csv(out / "oracle_fidelity_per_conf.csv", index=False)
    synergy_detail.to_csv(out / "synergy_detail.csv", index=False)
    synergy.to_csv(out / "synergy_metrics.csv", index=False)
    synergy_per_conf.to_csv(out / "synergy_metrics_per_conf.csv", index=False)
    bootstrap.to_csv(out / "bootstrap_architecture_comparison.csv", index=False)
    if not seed_summary.empty:
        seed_summary.to_csv(out / "k0_seed_robustness.csv", index=False)

    comparison = fidelity.pivot_table(
        index=["reference", "cohort", "K"], columns="arch",
        values=["mae_per_conf", "subset_mean_spearman", "median_per_conf_spearman", "mean_time_per_subset"],
    )
    comparison.columns = [f"{metric}_{arch}" for metric, arch in comparison.columns]
    comparison.reset_index().to_csv(out / "architecture_comparison.csv", index=False)

    per_conf_pivot = fidelity_per_conf.pivot_table(
        index=["reference", "cohort", "K", "conf"], columns="arch", values="mae"
    ).dropna().reset_index()
    per_conf_pivot["gnn_minus_mlp_mae"] = per_conf_pivot["gnn"] - per_conf_pivot["mlp"]
    per_conf_pivot.to_csv(out / "architecture_per_conf_comparison.csv", index=False)

    plot_results(attacker, fidelity, synergy, joined)
    write_report(registry, missing_truth, missing_est, attacker, fidelity, synergy, seed_summary)
    summary = {
        "registry_subsets": len(registry),
        "complete_truth_subsets": int(truth.sid.nunique()),
        "complete_estimate_subsets_by_arch": est.groupby("arch").sid.nunique().to_dict(),
        "missing_truth_files": len(missing_truth),
        "missing_estimate_files": len(missing_est),
        "k_grid": KGRID,
        "truth_definition": {
            "full": "dedicated DNN retrain",
            "structure_audit": "per-conf max(dedicated DNN, dedicated GCN) on D60 50 + stratified 18",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote RESULTS.md and output tables/figures")


if __name__ == "__main__":
    main()
