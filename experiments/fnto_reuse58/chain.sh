#!/usr/bin/env bash
# Server-owned finite gate; no client process is needed after detached launch.
set -Eeuo pipefail
root="${REUSE_ROOT:?set REUSE_ROOT}"
mkdir -p "$root/out" "$root/logs"
exec 9>"$root/chain.lock"
flock -n 9 || exit 73
trap 'printf "FAILED\n" > "$root/STATUS"' ERR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export REUSE_SOURCE_COMMIT="$(cat "$root/source_commit.txt")"
printf 'RUNNING_PREFLIGHT\n' > "$root/STATUS"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
cd "$root/repo"
"$py" -m unittest discover -s experiments/fnto_reuse58 -p 'test_*.py' -v > "$root/logs/tests.log" 2>&1
"$py" -u experiments/fnto_reuse58/angular_preflight.py --out "$root/out" > "$root/logs/angular.log" 2>&1
cp "$root/out/STATUS" "$root/STATUS"
date --iso-8601=seconds > "$root/completed_at.txt"
