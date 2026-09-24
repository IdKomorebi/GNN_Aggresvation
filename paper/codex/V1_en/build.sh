#!/usr/bin/env bash
# Run from any directory. Pass --figures to regenerate from existing experiment outputs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == "--figures" ]]; then
    export PYTHONDONTWRITEBYTECODE=1
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gnn_codex_mpl}"
    "${PAPER_PYTHON:-python3}" scripts/make_schematics.py
    "${PAPER_PYTHON:-python3}" scripts/make_figs.py
    "${PAPER_PYTHON:-python3}" scripts/make_gallery.py
fi
# Fixed reruns avoid Tectonic 0.15's repeated IEEEtran .bbl change notifications.
tectonic --keep-logs --keep-intermediates --reruns 2 main.tex
