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
for s in refs static seq timing; do stage=eval_$s; run eval_$s eval_r111.py "$O" $s $SM; done
stage=analyze; run analyze analyze_r111.py "$O" "$O/analysis.json"
stage=figs; run figs figs_r111.py "$O" "$O/figs"
stage=done; state "COMPLETE verdict=$(jget "$O/analysis.json" verdict)"
