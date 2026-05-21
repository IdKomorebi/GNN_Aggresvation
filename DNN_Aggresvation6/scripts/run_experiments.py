"""
顺序运行DNN_Aggresvation6两个强target-conditioned实验，并生成汇总表。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [
    PROJECT_ROOT / "configs" / "target_bilinear_gat.yaml",
    PROJECT_ROOT / "configs" / "target_bilinear_gat_loss_mask.yaml",
]


def main() -> None:
    pipeline = PROJECT_ROOT / "scripts" / "run_pipeline.py"
    for config_path in CONFIGS:
        print("\n" + "=" * 80)
        print(f"Running experiment config: {config_path}")
        print("=" * 80)
        subprocess.run(
            [sys.executable, str(pipeline), "--config", str(config_path)],
            cwd=str(PROJECT_ROOT.parent),
            check=True,
        )

    compare_script = PROJECT_ROOT / "scripts" / "compare_runs.py"
    subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--outputs",
            str(PROJECT_ROOT / "outputs"),
        ],
        cwd=str(PROJECT_ROOT.parent),
        check=True,
    )


if __name__ == "__main__":
    main()
