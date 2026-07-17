"""62 号孤岛实验：直接边被遮蔽时，链式敏感度能否恢复单跳方法漏报的泄露。

威胁模型（数据孤岛 / 跨源拼接）：
- 每条 general->confidential 直接边独立以概率 ρ 被遮蔽（攻击者对该 (i,c) 无联合数据）；
- general->general 桥接边始终可观测（攻击者能获得一般字段间的成对关系）；
- 对一个被遮蔽的 (i,c)，攻击者只能经桥接 j 做两跳：i --f_ij--> ĵ --f_jc--> ĉ，
  要求 (i,j) 是真实 g-g 边、且 (j,c) 未被遮蔽（否则 f_jc 也训不出）。

评测（仅在被遮蔽对 H 上，这正是单跳方法判分为 0 的地方）：
- realized_chain(i,c) = max_j R2_2hop[i,j,c]（合法桥接 j 上）——真实两跳攻击成功率；
- score_chain(i,c)    = max_j W[i,j]·α·W[j,c]（本方法链强度，限两跳以对齐 realized）；
- score_pearson(i,c)  = max_j pearson[i,j]·α·pearson[j,c]（IGNN 式相关系数链，基线）；
- score_1hop(i,c)     ≡ 0（直接边被遮蔽，单跳方法系统性漏报——对照的失败面）。

扫 ρ ∈ {0.3,0.5,0.7,0.9} × 多种子；ρ=0 对照（复现 61：链无增益）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RECOVER_TAU = 0.5   # "恢复"阈值：realized_chain 超过它算实质泄露
RHOS = [0.0, 0.3, 0.5, 0.7, 0.9]
N_SEEDS = 20


def compute_valid_bridges(Wgg, thr, O):
    """valid[i,j,c] = (i!=j) & (W[i,j]>=thr) & (j,c 可观测)。O: [nG,nC] 布尔。"""
    nG = Wgg.shape[0]
    gg_ok = (Wgg >= thr) & ~np.eye(nG, dtype=bool)      # [nG,nG]
    return gg_ok[:, :, None] & O[None, :, :]            # [nG,nG,nC]


def best_over_bridges(strength_ijc, valid):
    """在合法桥接上取 max；无合法桥接的 (i,c) 记 0。"""
    masked = np.where(valid, strength_ijc, -np.inf)
    out = masked.max(axis=1)                            # [nG,nC]
    out[~np.isfinite(out)] = 0.0
    return out


def main() -> None:
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    out_dir = ROOT / cfg["dataset"]["output_dir"]
    alpha = cfg["sensitivity"]["alpha"]
    thr = cfg["sensitivity"]["edge_threshold"]

    d = np.load(out_dir / "tensors.npz", allow_pickle=True)
    R2_direct = d["R2_direct"]          # [nG,nC]
    R2_2hop = d["R2_2hop"]              # [nG,nG,nC]
    pearson_direct = d["pearson_direct"]
    general = list(d["general"]); confidential = list(d["confidential"])
    nG, nC = R2_direct.shape

    import pandas as pd
    W = pd.read_csv(out_dir / "edge_matrix.csv", index_col=0)
    Wgg = W.loc[general, general].to_numpy()
    Wgc = W.loc[general, confidential].to_numpy()

    # 本方法两跳链强度张量 W[i,j]·α·W[j,c]
    strength_chain = Wgg[:, :, None] * alpha * Wgc[None, :, :]       # [i,j,c]

    # pearson 基线的两跳链：pearson[i,j]·α·pearson[j,c]（IGNN 式相关系数打分）
    from src.data_processing import load_and_preprocess
    csv_path = (ROOT / cfg["dataset"]["csv_path"]).resolve()
    df, _, _ = load_and_preprocess(
        str(csv_path), cfg["fields"]["drop_columns"],
        cfg["fields"]["confidential"], cfg["fields"]["drop_constant_columns"])
    corr = df[general + confidential].corr().abs().to_numpy()
    pear_gg = corr[:nG, :nG]
    strength_pear = pear_gg[:, :, None] * alpha * pearson_direct[None, :, :]

    # 全数据下"真正泄露"的对（直接 R²>0.5）——恢复率的正确分母，避免被无关对稀释
    leaky_mask = R2_direct > 0.5

    rows = []
    per_rho_detail = {}
    for rho in RHOS:
        pool = {"realized": [], "chain": [], "pear": [], "direct_true": []}
        missed_risk_fields = 0
        missed_risk_examples = []
        n_hidden_total = 0
        n_recovered = 0
        n_hidden_leaky = 0          # 被遮蔽 且 全数据下真正泄露
        n_recovered_leaky = 0       # 其中链式恢复 >0.5
        seeds = [0] if rho == 0.0 else range(N_SEEDS)
        for seed in seeds:
            rng = np.random.RandomState(1000 + seed)
            H = rng.random_sample((nG, nC)) < rho      # 被遮蔽的直接边
            O = ~H                                      # 可观测 g->c
            valid = compute_valid_bridges(Wgg, thr, O)  # [i,j,c]

            realized = best_over_bridges(R2_2hop, valid)      # [nG,nC]
            sc_chain = best_over_bridges(strength_chain, valid)
            sc_pear = best_over_bridges(strength_pear, valid)

            # 仅在被遮蔽对上评测（rho=0 时 H 全 False，取全体做对照）
            eval_mask = H if rho > 0 else np.ones_like(H, dtype=bool)
            ii, cc = np.where(eval_mask)
            pool["realized"].append(realized[ii, cc])
            pool["chain"].append(sc_chain[ii, cc])
            pool["pear"].append(sc_pear[ii, cc])
            pool["direct_true"].append(R2_direct[ii, cc])
            n_hidden_total += len(ii)
            n_recovered += int((realized[ii, cc] > RECOVER_TAU).sum())
            # 条件恢复：仅看被遮蔽且全数据下真正泄露的对
            hl = eval_mask & leaky_mask
            n_hidden_leaky += int(hl.sum())
            n_recovered_leaky += int((realized[hl] > RECOVER_TAU).sum())

            # 字段级：单跳判"安全"（该字段所有直接 c 边被遮蔽）却经链泄露
            for i in range(nG):
                if H[i].all():  # s_1hop(i) ≡ 0
                    best_c = realized[i].argmax()
                    if realized[i, best_c] > RECOVER_TAU:
                        missed_risk_fields += 1
                        if len(missed_risk_examples) < 8:
                            ex = (f"{general[i]} --chain--> {confidential[best_c]} "
                                  f"(R2={realized[i,best_c]:.2f})")
                            if ex not in missed_risk_examples:
                                missed_risk_examples.append(ex)

        realized = np.concatenate(pool["realized"])
        chain = np.concatenate(pool["chain"])
        pear = np.concatenate(pool["pear"])
        direct_true = np.concatenate(pool["direct_true"])

        def safe_spear(a, b):
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return float("nan")
            return float(spearmanr(a, b).statistic)

        n_seeds_run = len(list(seeds))
        row = {
            "rho": rho,
            "n_hidden_pairs": int(n_hidden_total),
            "mean_realized_chain": float(realized.mean()),
            "frac_recovered_all_gt0.5": n_recovered / max(n_hidden_total, 1),
            "frac_recovered_LEAKY_gt0.5": n_recovered_leaky / max(n_hidden_leaky, 1),
            "n_hidden_leaky_per_seed": n_hidden_leaky / max(n_seeds_run, 1),
            "spearman_chain": safe_spear(chain, realized),
            "spearman_pearson": safe_spear(pear, realized),
            "spearman_1hop": float("nan"),   # score_1hop≡0，方差为 0，无法排序（漏报）
            "missed_risk_fields_per_seed": missed_risk_fields / max(n_seeds_run, 1),
        }
        rows.append(row)
        per_rho_detail[str(rho)] = {"missed_risk_examples": missed_risk_examples}
        print(f"ρ={rho}: recovered_LEAKY>{RECOVER_TAU}={row['frac_recovered_LEAKY_gt0.5']:.1%} "
              f"(n_leaky/seed={row['n_hidden_leaky_per_seed']:.0f}), "
              f"spearman(chain)={row['spearman_chain']:.3f} vs "
              f"pearson={row['spearman_pearson']:.3f}, "
              f"missed_fields/seed={row['missed_risk_fields_per_seed']:.1f}")

    import pandas as pd
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / "silo_results.csv", index=False)
    (out_dir / "silo_summary.json").write_text(json.dumps(
        {"config": {"recover_tau": RECOVER_TAU, "rhos": RHOS, "n_seeds": N_SEEDS,
                    "alpha": alpha, "edge_threshold": thr},
         "results": rows, "detail": per_rho_detail}, indent=2, ensure_ascii=False))

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    r = res_df[res_df["rho"] > 0]
    axes[0].plot(r["rho"], r["frac_recovered_LEAKY_gt0.5"], "o-")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("hide fraction ρ"); axes[0].set_ylabel(f"frac of leaky pairs recovered (>{RECOVER_TAU})")
    axes[0].set_title("Chain recovers siloed leakage\n(single-hop reports 0 here)"); axes[0].grid(alpha=0.3)

    axes[1].plot(r["rho"], r["spearman_chain"], "o-", label="chain (ours)")
    axes[1].plot(r["rho"], r["spearman_pearson"], "s-", label="pearson chain (IGNN-style)")
    axes[1].axhline(0, color="gray", lw=1, ls="--", label="1-hop (≡0, cannot rank)")
    axes[1].set_xlabel("hide fraction ρ"); axes[1].set_ylabel("Spearman vs realized chain")
    axes[1].set_title("Score quality on hidden pairs"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(r["rho"], r["missed_risk_fields_per_seed"], "o-", c="tab:red")
    axes[2].set_xlabel("hide fraction ρ"); axes[2].set_ylabel("# fields / seed")
    axes[2].set_title('Fields single-hop calls "safe"\nbut chain finds leaky'); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "silo_plots.png", dpi=150)
    print(f"\n完成，结果在 {out_dir}/")


if __name__ == "__main__":
    main()
