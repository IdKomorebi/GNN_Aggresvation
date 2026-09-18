# -*- coding: utf-8 -*-
"""108 号自动报告：把四个 CSV 汇成一份 markdown（结论见 CHANGELOG.md）。"""
from pathlib import Path
import pandas as pd
A = Path(__file__).resolve().parents[1] / "outputs/analysis"
secs = [("A1 下界（top-1 vs top-3 见证认证）", "108_A1_bounds.csv"),
        ("A1 上界与区间覆盖率（留一字段分层分位校准）", "108_A1_coverage.csv"),
        ("A1 区间口径下的 τ-critical", "108_A1_critical_interval.csv"),
        ("A2 元训练 φ 定位（仅 PJM）", "108_A2_metatrain.csv"),
        ("A3 K 递进的三种攻击器口径", "108_A3_escalation.csv"),
        ("A3 饱和度与口径差", "108_A3_saturation.csv")]
out = ["# 108 号：A 档三条收尾（自动生成的数值表；结论与解读见 CHANGELOG.md）\n"]
for t, f in secs:
    out.append(f"## {t}\n\n" + pd.read_csv(A / f).to_markdown(index=False, floatfmt=".4f") + "\n")
(A / "report108.md").write_text("\n".join(out), encoding="utf-8")
print("report108.md 已写出")
