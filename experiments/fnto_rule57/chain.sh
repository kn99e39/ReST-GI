#!/usr/bin/env bash
set -euo pipefail
R="$HOME/rest_wl57"
mkdir -p "$R/out/tests" "$R/tmp" "$R/logs"
cd "$R/repo/experiments/fnto_rule57"
exec bash run.sh r57_supervisor.py
