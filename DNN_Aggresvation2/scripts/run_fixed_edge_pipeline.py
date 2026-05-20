from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-edge DNN/GNN aggregation pipeline.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "data2025_v2_fixed_edge.yaml"),
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()

