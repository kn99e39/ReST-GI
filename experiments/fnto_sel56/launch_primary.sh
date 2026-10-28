#!/usr/bin/env bash
set -euo pipefail
R=$HOME/rest_wl56
RUN=$R/repo/experiments/fnto_sel56/run.sh
STATUS=$R/launcher/primary.status
trap 'echo failed > "$STATUS"' ERR
echo running > "$STATUS"
# This launcher is invoked only once the independent reference gate and regression checks finish.
"$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python" -c 'import json,os; from pathlib import Path; p=Path.home()/"rest_wl56/out"; assert json.load(open(p/"reference_gate.json"))["proceed"]; assert all(json.load(open(p/"tests"/f))["passed"]==json.load(open(p/"tests"/f)).get("total",json.load(open(p/"tests"/f)).get("n")) for f in ["focused.json","wl53.json","wl54.json","wl55.json"])'
step() { echo "$(date -Is) $*" >> "$R/launcher/primary_steps.log"; "$RUN" "$@"; }
step s56_masks.py
step s56.py audit
step s56.py prepare
step s56.py update rest S 0
step s56.py update rest S 1
step s56.py update dent F 0
step s56.py update dent S 0
step s56.py verify dent 0
step s56.py update dent S 1
step s56.py update dent F 1
step s56.py verify dent 1
for rep in 0 1; do
    step s56.py update dent T "$rep"
    step s56.py replay dent F "$rep" ind
done
step s56.py update rigid F 0
step s56.py update rigid S 0
step s56.py verify rigid 0
step s56.py update rigid S 1
for rep in 0 1; do
    step s56.py replay dent F "$rep"
    step s56.py replay dent S "$rep"
done
step s56_analyze.py
echo done > "$STATUS"
