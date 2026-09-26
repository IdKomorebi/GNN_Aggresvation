#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gnn_codex_mpl}"
"${PYTHON:-python3}" scripts/make_schematics.py
"${PYTHON:-python3}" scripts/make_figs.py toy combination profile estimator robust amplification mechanism masking scope baselines
