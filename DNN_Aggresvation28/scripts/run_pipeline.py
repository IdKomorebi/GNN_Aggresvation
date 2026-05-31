from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import redraw_run_plots, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact DNN_Aggresvation28 experiments.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "config_lossexclude.yaml"),
        help="YAML configuration file; defaults to the loss-exclude experiment.",
    )
    parser.add_argument(
        "--redraw-run",
        default=None,
        help="Redraw plots for an existing run directory without retraining.",
    )
    args = parser.parse_args()
    if args.redraw_run:
        redraw_run_plots(args.redraw_run)
        return
    run_experiment(args.config)


if __name__ == "__main__":
    main()
