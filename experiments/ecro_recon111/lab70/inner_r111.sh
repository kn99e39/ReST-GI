#!/usr/bin/env bash
# WL111 chain body: runs INSIDE one container (ctr_r111.sh) started by ONE guard (guard109c.py, --gpu-uuid RTX 3090).
# The guard's 120 s idle check happens once, at chain entry; all stages then run back to back in this container, so the
# batch keeps the GPU between stages. A foreign compute process on the RTX 3090 still makes the guard kill this container;
# on restart, stages with a done marker are skipped and the evidence stages resume from their saved outputs.
# usage: inner_r111.sh OUT [--smoke]
set -eu -o pipefail
O=$1; SM=${2:-}
cd "$(dirname "$0")/.."
py=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
mkdir -p "$O/models"
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$O/chain.state"; }
jget() { "$py" -c 'import json,sys; d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$1" "$2"; }
run() { local name=$1; shift
  if [[ -f $O/.done_$name ]]; then return 0; fi
  state "RUNNING $name"
  "$py" "$@" > "$O/$name.log" 2>&1
  touch "$O/.done_$name"; state "DONE $name"; }
stage=start
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc"; fi' EXIT
state "CHAIN BODY START (container; load1=$(cut -d' ' -f1 /proc/loadavg)) $SM"
stage=tests; run tests tests_r111.py "$O/tests.json"
grep -q "ALL_PASS True" "$O/tests.log"
stage=stageA; run stageA profile_r111.py "$O/stageA" $SM
cp "$O/stageA/design.json" "$O/design.json"
if [[ $(jget "$O/design.json" cost_futility_stop) == True ]]; then
  if [[ $SM == --smoke ]]; then state "SMOKE: futility flag ignored to exercise the later code paths";
  else stage=done; state "COMPLETE cost futility stop (NOT_SUPPORTED by cost)"; exit 0; fi
fi
state "DESIGN sub=$(jget "$O/design.json" sub) m=$(jget "$O/design.json" m)"
stage=data; run data data_r111.py "$O/data" "$O/design.json" $SM
for ri in 0 1; do stage=train$ri; run train$ri train_r111.py "$O/data" "$O/design.json" "$O/models" $ri $SM; done
stage=synth; run synth synth_r111.py "$O" $SM
[[ $(jget "$O/synth/synth.json" all_pass) == True ]] || { state "STOP Stage E contract failure"; exit 3; }
# Execution fix 4 (owner request 2026-11-26, GPU efficiency): static and seq run as NSH processes on the same GPU, each taking
# the items i mod NSH (RECON111_SHARD); an unsharded completion pass then finds every item written (and computes any that a
# shard left out). refs and timing stay single-process; timing keeps exclusive use of the GPU.
NSH=3
par() { local name=$1 s=$2 i rc=0; local pids=()
  if [[ -f $O/.done_$name ]]; then return 0; fi
  state "RUNNING $name (shards=$NSH)"
  for ((i = 0; i < NSH; i++)); do
    RECON111_SHARD=$i/$NSH "$py" eval_r111.py "$O" "$s" $SM > "$O/${name}_shard$i.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (( rc == 0 )) || { state "STOP $name: a shard failed"; exit 4; }
  "$py" eval_r111.py "$O" "$s" $SM > "$O/$name.log" 2>&1
  touch "$O/.done_$name"; state "DONE $name"; }
stage=eval_refs; run eval_refs eval_r111.py "$O" refs $SM
stage=eval_static; par eval_static static
stage=eval_seq; par eval_seq seq
stage=eval_timing; run eval_timing eval_r111.py "$O" timing $SM
stage=analyze; run analyze analyze_r111.py "$O" "$O/analysis.json"
stage=figs; run figs figs_r111.py "$O" "$O/figs"
stage=done; state "COMPLETE verdict=$(jget "$O/analysis.json" verdict)"
