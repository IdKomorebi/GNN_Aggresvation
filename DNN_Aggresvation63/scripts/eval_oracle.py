"""DNN63 保真度评测：oracle 的 v̂(S) 对照重训真值 v(S)。

真值源：
  60 号 50 个无偏随机子集（random_eval.csv 的 v_best = best-of-struct 重训真值，主基准）
  59 号 55 个 top-k 结构化子集（(ranking,k) → best-of-struct，次基准）
对每个 oracle checkpoint、每个真值子集 S：把测试集按 S 掩码前向，实测 12-conf 平均 R² 得 v̂(S)。
指标：MAE、bias、Spearman，按尺寸带分层；多种子平均。用 60 号噪声底标尺判显著。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data
from src.model import build_edge_mask
from src.oracle import MLPOracle, GNNOracle, build_priors

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D60 = ROOT.parent / "DNN_Aggresvation60"
D59 = ROOT.parent / "DNN_Aggresvation59"
BANDS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 44)]


def per_conf_r2(pred, target):
    ss = ((target - pred) ** 2).sum(0)
    st = ((target - target.mean(0)) ** 2).sum(0) + 1e-12
    return np.clip(1 - ss / st, 0, None)


def load_truth(general):
    """返回 [(subset_id, size, field_indices, v_truth, source)]。"""
    name2idx = {n: i for i, n in enumerate(general)}
    rows = []
    # --- 60 无偏随机子集 ---
    subs = json.load(open(D60 / "outputs/subsets.json"))
    rv = pd.read_csv(D60 / "outputs/random_eval.csv").set_index("subset_id")
    for sid, info in subs.items():
        if info["group"] != "random_eval":
            continue
        idx = [name2idx[f] for f in info["fields"]]
        rows.append((sid, info["size"], idx, float(rv.loc[sid, "v_best"]), "60_random"))
    # --- 59 top-k 结构化子集 ---
    ar = pd.read_csv(D59 / "outputs/retrain/all_retrain.csv")
    best = ar.groupby(["ranking", "k"])["mean_r2"].max()   # best-of-struct
    orders = {r: json.load(open(D59 / f"outputs/retrain/ranking_{r}.json"))["order_fields"]
              for r in ar["ranking"].unique()}
    for (r, k), v in best.items():
        fields = orders[r][:k]
        idx = [name2idx[f] for f in fields]
        rows.append((f"{r}_k{k}", int(k), idx, float(v), "59_topk"))
    return rows


def build_oracle(ckpt, general, nG, nC):
    if ckpt["arch"] == "mlp":
        model = MLPOracle(nG, nC)
    else:
        mt = np.load(ROOT / "outputs/relationship_cache/metric_tensor.npy")
        cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
        em = build_edge_mask(mt, top_k=cfg["graph"]["top_k"], threshold=cfg["graph"]["threshold"],
                             symmetrize=True, n_general=nG, bipartite=True)
        A_gg, prior_cg = build_priors(mt, em, nG)
        model = GNNOracle(A_gg, prior_cg, nC,
                          hidden=ckpt.get("gnn_hidden", 128), n_layers=ckpt.get("gnn_layers", 3))
    model.load_state_dict(ckpt["state"]); model.to(DEV).eval()
    return model


def vhat_all(model, Xte, Yte, truth, nG):
    """对每个真值子集算 v̂(S)。"""
    out = []
    with torch.no_grad():
        for sid, size, idx, vt, src in truth:
            m = torch.zeros(len(Xte), nG, device=DEV)
            m[:, idx] = 1.0
            pred = model(Xte, m).cpu().numpy()
            out.append(per_conf_r2(pred, Yte).mean())
    return np.array(out)


def band_of(size):
    for lo, hi in BANDS:
        if lo <= size <= hi:
            return f"{lo}-{hi}"
    return "?"


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    torch.manual_seed(42); np.random.seed(42)
    di = prepare_data(cfg)
    gi = np.array(di["general_indices"]); ci = np.array(di["confidential_indices"])
    nG, nC = di["n_general"], di["n_confidential"]
    general = di["general"]
    te = di["test_data"]
    Xte = torch.as_tensor(te[:, gi], dtype=torch.float32, device=DEV)
    Yte = te[:, ci]

    truth = load_truth(general)
    sizes = np.array([t[1] for t in truth])
    vt = np.array([t[3] for t in truth])
    src = np.array([t[4] for t in truth])
    bands = np.array([band_of(s) for s in sizes])
    print(f"真值子集: {len(truth)}（60_random={int((src=='60_random').sum())}, "
          f"59_topk={int((src=='59_topk').sum())}）")

    # 每个 arch：多种子 v̂ 平均（只评满数据 f1.0）
    vhat = {}
    for arch in ["mlp", "gnn"]:
        ckpts = sorted(ROOT.glob(f"outputs/oracle_{arch}_seed*.pt"))
        ckpts = [c for c in ckpts if "_f" not in c.name]   # 排除低数据变体
        if not ckpts:
            continue
        preds = []
        for c in ckpts:
            ck = torch.load(c, map_location=DEV, weights_only=False)
            model = build_oracle(ck, general, nG, nC)
            preds.append(vhat_all(model, Xte, Yte, truth, nG))
        vhat[arch] = np.mean(preds, axis=0)
        print(f"  {arch}: {len(ckpts)} 个种子")

    # 汇总指标
    def metrics(v_pred, mask):
        e = v_pred[mask] - vt[mask]
        rho = spearmanr(v_pred[mask], vt[mask]).statistic if mask.sum() > 2 else float("nan")
        return dict(n=int(mask.sum()), mae=float(np.abs(e).mean()),
                    bias=float(e.mean()), spearman=float(rho))

    report = {"overall": {}, "by_source": {}, "by_band": {}}
    for arch in vhat:
        report["overall"][arch] = metrics(vhat[arch], np.ones(len(vt), bool))
        report["by_source"][arch] = {s: metrics(vhat[arch], src == s)
                                     for s in ["60_random", "59_topk"]}
        report["by_band"][arch] = {b: metrics(vhat[arch], bands == b)
                                   for b in [f"{lo}-{hi}" for lo, hi in BANDS]}
    (ROOT / "outputs/fidelity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # 明细表
    det = pd.DataFrame({"subset_id": [t[0] for t in truth], "size": sizes,
                        "band": bands, "source": src, "v_truth": vt})
    for arch in vhat:
        det[f"vhat_{arch}"] = vhat[arch]
    det.sort_values(["source", "size"]).to_csv(ROOT / "outputs/fidelity_detail.csv", index=False)

    # 打印关键对比
    print("\n=== 总体 ===")
    for arch in vhat:
        m = report["overall"][arch]
        print(f"  {arch}: MAE={m['mae']:.4f} bias={m['bias']:+.4f} Spearman={m['spearman']:.4f}")
    print("\n=== 按尺寸带 MAE（越低越好） / Spearman ===")
    print(f"{'band':>8} | {'MLP MAE':>8} {'GNN MAE':>8} {'ΔMAE':>7} | {'MLP rho':>8} {'GNN rho':>8}")
    for lo, hi in BANDS:
        b = f"{lo}-{hi}"
        if "mlp" in vhat and "gnn" in vhat:
            mm, gg = report["by_band"]["mlp"][b], report["by_band"]["gnn"][b]
            dmae = gg["mae"] - mm["mae"]
            print(f"{b:>8} | {mm['mae']:>8.4f} {gg['mae']:>8.4f} {dmae:>+7.4f} | "
                  f"{mm['spearman']:>8.4f} {gg['spearman']:>8.4f}")

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"60_random": "tab:blue", "59_topk": "tab:orange"}
    for ax, arch in zip(axes[:2], ["mlp", "gnn"]):
        if arch not in vhat:
            continue
        for s in ["60_random", "59_topk"]:
            mk = src == s
            ax.scatter(vt[mk], vhat[arch][mk], s=18, alpha=0.7, c=colors[s], label=s)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("v(S) retrain truth"); ax.set_ylabel(f"v̂(S) {arch} oracle")
        ax.set_title(f"{arch}-oracle fidelity"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if "mlp" in vhat and "gnn" in vhat:
        bl = [f"{lo}-{hi}" for lo, hi in BANDS]
        xm = np.arange(len(bl))
        axes[2].bar(xm - 0.2, [report["by_band"]["mlp"][b]["mae"] for b in bl], 0.4, label="MLP")
        axes[2].bar(xm + 0.2, [report["by_band"]["gnn"][b]["mae"] for b in bl], 0.4, label="GNN")
        axes[2].set_xticks(xm); axes[2].set_xticklabels(bl)
        axes[2].set_xlabel("subset size band"); axes[2].set_ylabel("MAE vs truth")
        axes[2].set_title("Fidelity MAE by size band"); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "outputs/fidelity_plots.png", dpi=150)
    print(f"\n完成，结果在 {ROOT}/outputs/")


if __name__ == "__main__":
    main()
