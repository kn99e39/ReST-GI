#!/usr/bin/env bash
# WL109B Amendment 4: CPU stages on Main WSL (Ubuntu 22.04, 32 cores). usage: run109b_main.sh COMMIT PHASE
# The frozen interpreter / venv, code and analysis inputs are mirrored from LabServer70 at the same /home/nd paths (pull70.sh),
# and the analyses run with HOME=/home/nd so every path resolves as on the GPU hosts. GPU outputs are pulled from LabServer70
# before each phase; the phase's decision files are pushed back to LabServer70, which holds the authoritative record.
#   tests : tests109b.py
#   bias  : pull paired + xcheck -> analyze109b --phase bias -> push metrics/bias109b.json (+ maps)
#   static: pull grid + timing   -> analyze109b --phase static + decide109b -> push metrics/{metrics,static}109b.json
# Launch through wsl_run_detached.sh (a WSL foreground job dies with its client).
set -eu -o pipefail
commit=$1; phase=$2
export HOME=/home/nd
cd "$HOME/rest_sss101a"
src=experiments/ecro_estimator109b
out="$PWD/results/ecro_estimator109b/evidence"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
L70=nd@100.115.109.101
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
mkdir -p "$out/tmp" "$out/mpl" "$out/metrics"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR="$out/mpl" TMPDIR="$out/tmp" PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
state() { local line; line="$(date -Is) $*"; echo "$line" | tee -a "$out/chain_main.state"; $SSH $L70 "echo 'MAIN $line' >> /home/nd/rest_sss101a/results/ecro_estimator109b/evidence/chain.state" || true; }
stage=start
trap 'rc=$?; if (( rc )); then state "STOP main stage=$stage exit=$rc"; fi' EXIT
pull() { rsync -a --partial -e "$SSH" "$L70:/home/nd/rest_sss101a/results/ecro_estimator109b/evidence/$1" "$out/$(dirname "$1")/"; }
push() { rsync -a -e "$SSH" "$@" "$L70:/home/nd/rest_sss101a/results/ecro_estimator109b/evidence/metrics/"; }
jget() { "$py" -c 'import json,sys; d=json.load(open(sys.argv[1]));
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$1" "$2"; }
state "AMENDMENT 4 $phase START on Main WSL ($(hostname)) commit=$commit"
rsync -a -e "$SSH" "$L70:/home/nd/rest_sss101a/$src/" "$src/"        # code exactly as on the GPU host
pull calib
cd "$src"
case $phase in
  tests)
    stage=tests; "$py" tests109b.py > "$out/tests_main.log" 2>&1
    state "TESTS PASS (Main)";;
  bias)
    stage=pull; pull xcheck; pull cap/paired
    [[ $(jget "$out/xcheck/xcheck109b.json" outcome) == PASS ]] || [[ -d $out/cap/paired_lab69_not_reproduced ]] || { echo "xcheck neither PASS nor recapture" >&2; exit 4; }
    n=$(ls -d "$out"/cap/paired/*_spp* | grep -vc '/\.' || true); [[ $n == 18 ]] || { echo "paired units: $n" >&2; exit 5; }
    stage=bias; "$py" analyze109b.py "$out/cap" "$out/calib/budget109b.json" "$out/metrics" --phase bias > "$out/bias.log" 2>&1
    push "$out/metrics/bias109b.json" "$out/metrics/maps109b.npz"
    state "BIAS verdict=$(jget "$out/metrics/bias109b.json" bias_verdict) (Main)";;
  static)
    stage=pull; pull cap/paired; pull cap/grid; pull cap/timing
    stage=static; "$py" analyze109b.py "$out/cap" "$out/calib/budget109b.json" "$out/metrics" --phase static > "$out/static.log" 2>&1
    "$py" decide109b.py "$out/metrics/metrics109b.json" "$out/metrics/static109b.json" >> "$out/static.log" 2>&1
    push "$out/metrics/metrics109b.json" "$out/metrics/static109b.json"
    state "STATIC verdict=$(jget "$out/metrics/static109b.json" final_static_verdict) (Main)";;
  *) echo "unknown phase $phase" >&2; exit 2;;
esac
stage=done
