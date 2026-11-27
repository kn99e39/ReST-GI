#!/usr/bin/env bash
# WL115 Stage D/E continuation (execution fix): readouts in two processes after the torch cache fix (cached torch memory had oversubscribed the
# 16 GB card and WDDM paging slowed streams ~100x), then rebuild and analysis; writes STAGE_DE_DONE for chainG.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
export PYTHONUNBUFFERED=1
echo "chainC2 start $(date -Iseconds) commit $(git rev-parse HEAD)" >> $R/logs/chain_progress.log
fail() { echo "FAILED $1 $(date -Iseconds)" >> $R/logs/chain_progress.log; echo "FAILED $1" > $R/chainC.status; exit 1; }
run() { local name=$1; shift; if [ ! -f $R/logs/$name.ok ]; then echo "RUNNING $name $(date -Iseconds)" > $R/chainC.status; "$@" >> $R/logs/$name.log 2>&1 || return 1; touch $R/logs/$name.ok; echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi; }
( run streamsD_0 $PY -B evalD115.py streams $R F4 N1 N2 N3 N4 SEQ_D_independent N5 N6 N7 N8 SEQ_E_fresh ) & P1=$!
( run streamsD_1 $PY -B evalD115.py streams $R F0 F1 F2 F3 SEQ_A_lift SEQ_B_slide SEQ_C_independent ) & P2=$!
wait $P1 || fail streamsD_0; wait $P2 || fail streamsD_1
run rebuild $PY -B evalD115.py rebuild $R || fail rebuild
run analyzeD $PY -B analyzeD115.py $R || fail analyzeD
echo "STAGE_DE_DONE $(date -Iseconds)" > $R/chainC.status
