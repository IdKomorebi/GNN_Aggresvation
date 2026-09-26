#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
tectonic --keep-logs --keep-intermediates --reruns 2 main.tex
