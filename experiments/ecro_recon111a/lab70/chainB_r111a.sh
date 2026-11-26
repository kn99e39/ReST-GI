#!/usr/bin/env bash
# WL111A Stage B chain on LabServer70 (RTX 3090), started by chainB_all_r111a.sh after Stage A:
#   sub-8 guides -> 6 factorial trainings in two waves of 3 parallel processes -> paired dev evaluation -> oracles.
# usage: chainB_r111a.sh            (env RECON111A_OUT; RECON111A_SMOKE=1 / RECON111A_ITERS for the smoke test)
set -eu -o pipefail
R=$HOME/rest_sss101a
O=${RECON111A_OUT:-$R/results/ecro_recon111a}
export RECON111A_OUT=$O
L=$R/experiments/ecro_recon111a/lab70
st() { printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$O/chainB.state"; }
mkdir -p "$O/logs"
run() { local name=$1; shift; "$L/run_r111a.sh" "$name" "$@" > "$O/logs/$name.log" 2>&1; }
done_() { [[ -f $O/.doneB_$1 ]]; }
mark() { touch "$O/.doneB_$1"; st "DONE $1"; }
trap 'rc=$?; (( rc )) && st "STOP exit=$rc"' EXIT
st "START"
done_ guides8 || { st "RUNNING guides8"; run guides8 guides8_r111a.py; mark guides8; }
wave() { local pids=() rc=0 spec
  for spec in "$@"; do
    set -- $(tr ':' ' ' <<< "$spec")       # ARM AGE SUB RI
    if [[ -f $O/models/${1}_r$4.pt ]]; then continue; fi
    run "train_${1}_r$4" train_r111a.py "$1" "$2" "$3" "$4" & pids+=($!)
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || rc=1; done
  return $rc; }
st "RUNNING train wave 1"; wave C4:cold:4:0 C4:cold:4:1 M8:mixed:8:0; st "DONE train wave 1"
st "RUNNING train wave 2"; wave M8:mixed:8:1 C8:cold:8:0 C8:cold:8:1; st "DONE train wave 2"
done_ evalB || { st "RUNNING evalB"; run evalB evalB_r111a.py C4:4 M8:8 C8:8; mark evalB; }
if [[ ${RECON111A_SMOKE:-0} == 1 ]]; then OARGS="--views Q00 --steps 20"; else OARGS=""; fi
done_ oracle || { st "RUNNING oracle"; run oracle oracle_r111a.py $OARGS; mark oracle; }
st "COMPLETE"
