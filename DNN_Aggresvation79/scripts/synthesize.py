# -*- coding: utf-8 -*-
"""把探索性结果收束成不使用 test 选参的最终三档协议。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from runlog import log  # noqa: E402

N_ALL = 13244
ESTIMATED_STRONG = 1038.917777777778


def pick(frame: pd.DataFrame, variant: str, retention: float) -> pd.Series:
    return frame[(frame.variant == variant) & (frame.retention == retention)].iloc[0]


def main() -> None:
    rescue = pd.read_csv(OUT / "rescue_rank_metrics.csv")
    tiers = [
        ("old_k0", "single_K0", 0.30, "旧K0发现基线"),
        ("fast", "single_K5", 0.30, "快速发现：第一次硬筛选放到K5"),
        ("balanced", "single_K10", 0.30, "均衡发现：K10后保留Top30%"),
        (
            "high_assurance",
            "best_rank_0_10_25",
            0.30,
            "高保障发现上限：全体到K25，利用K0/K10/K25最佳排名救援",
        ),
        ("upper_bound", "single_K50", 0.30, "较高成本参考上限"),
        (
            "recall95",
            "best_rank_0_10",
            0.60,
            "约95%召回档：K10后保留Top60%",
        ),
    ]
    rows = []
    for tier, variant, retention, note in tiers:
        row = pick(rescue, variant, retention)
        n_keep = int(round(N_ALL * retention))
        precision = row.all_recall * ESTIMATED_STRONG / n_keep
        rows.append(
            {
                "tier": tier,
                "variant": variant,
                "max_k": int(row.max_k),
                "avg_updates_continuation": float(row.max_k),
                "avg_updates_restart": float(row.max_k),
                "retention": retention,
                "n_keep": n_keep,
                "tune_recall": row.tune_recall,
                "test_recall": row.test_recall,
                "population_recall": row.all_recall,
                "population_precision": precision,
                "enrichment_over_base_rate": precision / (ESTIMATED_STRONG / N_ALL),
                "note": note,
            }
        )
    staged = pd.read_csv(OUT / "staged_rescue_top30.csv")
    hierarchy = staged[
        (staged.first_k == 10)
        & (staged.first_retention == 0.60)
        & (staged.variant == "best_checkpoint_rank")
    ].iloc[0]
    hierarchy_precision = hierarchy.all_recall * ESTIMATED_STRONG / int(round(N_ALL * 0.30))
    rows.append(
        {
            "tier": "recommended_hierarchy",
            "variant": "K10@60% -> K25@30% + best(K0,K10,K25)",
            "max_k": 25,
            "avg_updates_continuation": hierarchy.avg_updates_continuation,
            "avg_updates_restart": hierarchy.avg_updates_restart,
            "retention": 0.30,
            "n_keep": int(round(N_ALL * 0.30)),
            "tune_recall": hierarchy.tune_recall,
            "test_recall": hierarchy.test_recall,
            "population_recall": hierarchy.all_recall,
            "population_precision": hierarchy_precision,
            "enrichment_over_base_rate": hierarchy_precision / (ESTIMATED_STRONG / N_ALL),
            "note": "推荐分层：K10宽进Top60%，保存状态续训K25，再缩到Top30%",
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "final_fidelity_tiers.csv", index=False)

    schedules = pd.read_csv(OUT / "schedule_sweep.csv")
    compare_labels = [
        "K0@30% -> K25@10%",
        "K5@30% -> K25@10%",
        "K50@10%",
    ]
    top10 = schedules[schedules.schedule.isin(compare_labels)][
        [
            "schedule",
            "avg_updates_restart",
            "avg_updates_continuation",
            "tune_recall",
            "test_recall",
            "all_recall",
        ]
    ].copy()
    top10.to_csv(OUT / "top10_priority_shortlist_comparison.csv", index=False)

    conclusion = {
        "accepted": {
            "fast": "K5 全体后保留 Top30%；总体召回 85.8%，隔离 test 88.2%。",
            "balanced": "K10 全体后保留 Top30%；总体召回 87.2%，隔离 test 89.2%。",
            "high_assurance": (
                "全体到 K25，按 K0/K10/K25 最佳百分位排名保留 Top30%；"
                "总体召回 89.7%，隔离 test 91.9%。"
            ),
            "recommended_hierarchy": (
                "全体到K10宽进Top60%，入围者保存状态继续到K25，再按K0/K10/K25"
                "最佳排名收缩到Top30%；平均19步，总体召回88.6%，隔离test 89.9%。"
            ),
            "recall95": "若目标是约95%召回，K10 后需保留约 Top60%，不能只留 Top30%。",
        },
        "rejected_or_limited": {
            "k1": "K1 Top30 召回 80.4%，低于 K0 的82.7%，不可作为第一硬筛选层。",
            "early_hard_pruning": (
                "K0 Top30 后再微调仍受82.7%入围天花板；K5/K10后过早大幅砍候选也会"
                "把后续K25的发现收益吃掉。"
            ),
            "top10": (
                "Top10%本身只能覆盖约68%–71%的强三元组，应称为优先认证清单，"
                "不能称为高召回发现池。"
            ),
            "simple_multi_k_fusion": (
                "K<=10 的平均榜、任一榜靠前和不确定性加成均未稳定超过K10单榜。"
            ),
        },
        "truth_caveat": (
            "仍沿用77号重训真值：测试集参与早停且主要为单训练种子。79号改善的是筛选"
            "协议证据，不等于修复了真值协议。"
        ),
    }
    (OUT / "final_conclusion.json").write_text(
        json.dumps(conclusion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(
        "SYNTHESIS",
        "DECISION",
        note=(
            "最终采用功能/保真度分层：K5快速、K10均衡、K25多榜高保障；"
            "拒绝K1硬筛和把Top10称为高召回池。"
        ),
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
