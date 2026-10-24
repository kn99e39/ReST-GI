#!/usr/bin/env bash
# Worklog 43 / protocol 8dna_dyn_v1 orchestrator on LabServer63. Runs every stage in order without a session and stops at
# the first failing stage.
#   default: waits for the WL42 FNTO chain (after_train.status and a2_refs.status both 'done ...'; user decision
#            2026-10-24: 8DNA runs after FNTO, on the server), then for 10 minutes with no GPU compute process; then a
#            dry pass of every stage (EDNA_DRYRUN=1: 32x32, <= 16 spp, into $W/out_dry, never evidence) as a code-path
#            gate; only if all dry stages pass, the real stages into $W/out.
#   --dry  : only the dry pass, now, without waiting (needs a GPU with a few GB free; it does not fit beside a 11 GB job).
# Status: $W/launcher/dyn_chain.status ; step log: $W/launcher/dyn_chain_runs.log ; per-stage logs $W/launcher/<stage>[_dry].log
# Launch (from the checkout's experiments/8dna_dynamic_baseline):
#   mkdir -p ~/rest_8dna_baseline/launcher && setsid -f nohup bash server/run_dyn_chain.sh [--dry] \
#       > ~/rest_8dna_baseline/launcher/dyn_chain.stdout.log 2>&1 < /dev/null
set -u
W=$HOME/rest_8dna_baseline
X=$(cd "$(dirname "$0")/.." && pwd)
F=$HOME/rest_fnto_r1/launcher
V=$HOME/NeuralShaderDemo/external/8dna26/.venv
PY=$V/bin/python
export PATH=$V/bin:$PATH EDNA_UPSTREAM=$HOME/NeuralShaderDemo/external/8dna26
Lg=$W/launcher
mkdir -p "$Lg"
exec 9>"$Lg/dyn_chain.lock"
flock -n 9 || { echo "another dyn chain is running"; exit 1; }
st() { echo "$1 $(date -Is)" > "$Lg/dyn_chain.status"; }
log() { echo "$1 $(date -Is)" >> "$Lg/dyn_chain_runs.log"; }
cd "$X"
source server/mi_env.sh || { st "failed: mi_env"; exit 1; }
log "start commit=$(git rev-parse --short HEAD 2>/dev/null) dirty=$(git status --porcelain 2>/dev/null | wc -l) args=${1:-}"

if [ "${1:-}" != "--dry" ]; then
  st "waiting: FNTO chain"
  while true; do
    a=$(cat "$F/after_train.status" 2>/dev/null); r=$(cat "$F/a2_refs.status" 2>/dev/null)
    case "$a" in done*) case "$r" in done*) break;; esac;; esac
    case "$a$r" in *failed*) st "blocked: FNTO chain failed [$a] [$r]; starting is the user's decision"; log "FNTO failed"; exit 2;; esac
    sleep 120
  done
  log "FNTO chain finished: [$a] [$r]"
  st "waiting: GPU idle 10 min"
  idle=0
  while [ $idle -lt 10 ]; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ "$n" = 0 ]; then idle=$((idle + 1)); else idle=0; fi
    sleep 60
  done
  log "GPU idle"
fi

run() {  # run <label> <script> [args]
  log "$1 start"; st "running: $1"
  "$PY" -u "${@:2}" > "$Lg/$1.log" 2>&1; rc=$?
  log "$1 rc=$rc"
  [ $rc = 0 ] || { st "failed: $1 (rc=$rc)"; exit $rc; }
}
STAGES="smoke design static refs neural ladder cost"

# dry pass: code-path gate at 32x32 / <= 16 spp
export EDNA_DRYRUN=1 EDNA_OUT=$W/out_dry
rm -rf "$EDNA_OUT"; mkdir -p "$EDNA_OUT"
for s in $STAGES; do run "${s}_dry" dyn_stages.py "$s"; done
run eval_dry dyn_eval.py
run exports_dry dyn_exports.py
log "dry pass complete"
if [ "${1:-}" = "--dry" ]; then st "done (dry only)"; exit 0; fi

# real stages
unset EDNA_DRYRUN
export EDNA_OUT=$W/out
mkdir -p "$EDNA_OUT"
for s in $STAGES; do run "$s" dyn_stages.py "$s"; done
run eval dyn_eval.py
run exports dyn_exports.py
st "done"
log "done"
