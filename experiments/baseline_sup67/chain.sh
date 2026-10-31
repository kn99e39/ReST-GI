#!/usr/bin/env bash
# WL67 server chain (protocol baseline_sup67_v1). Launch detached: setsid nohup bash chain.sh > $R/launcher/chain.log 2>&1 < /dev/null &
# Every timed GPU stage first waits for an idle GPU (no compute process for two checks 60 s apart) and records peers before/after.
set -euo pipefail
R=$HOME/rest_bsup67
REPO=$R/repo
EXP=$REPO/experiments/baseline_sup67
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
BL=$HOME/rest_wl55/blender-3.5.1-linux-x64/blender
STATUS=$R/launcher/chain.status
mkdir -p "$R/launcher" "$R/tmp" "$R/out"
trap 'echo "failed $(cat $R/launcher/stage 2>/dev/null)" > "$STATUS"' ERR
echo running > "$STATUS"
export TMPDIR=$R/tmp P53_TMPDIR=$R/tmp PYTHONUNBUFFERED=1 P53_MI_VARIANT=cuda_ad_rgb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so
export B67_ROOT=$R B67_W2IN=$R/w2in WL56=$R/w1oracle

peers() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
wait_idle() {
    local n=0
    while true; do
        if [ -z "$(peers)" ]; then sleep 60; [ -z "$(peers)" ] && break; fi
        n=$((n+1)); echo "$(date -Is) waiting for GPU: $(peers | tr '\n' ';')" >> "$R/launcher/gpu_wait.log"; sleep 120
    done
    echo "$(date -Is) idle after $n waits" >> "$R/launcher/gpu_wait.log"
}
stage() {
    local name=$1; shift
    echo "$name" > "$R/launcher/stage"
    if [ -f "$R/launcher/$name.done" ]; then echo "$(date -Is) skip $name (done)" >> "$R/launcher/steps.log"; return; fi
    wait_idle
    echo "$(date -Is) start $name peers=[$(peers | tr '\n' ';')]" >> "$R/launcher/steps.log"
    "$@" > "$R/launcher/$name.log" 2>&1
    echo "$(date -Is) end $name peers=[$(peers | tr '\n' ';')]" >> "$R/launcher/steps.log"
    touch "$R/launcher/$name.done"
}
s56() { (cd "$REPO/experiments/fnto_sel56" && "$PY" s56.py "$@"); }
w1_oracle() {
    s56 update dent F 0; s56 update dent S 0; s56 verify dent 0; s56 update dent T 0
    s56 update rest S 0; s56 update rigid S 0; s56 resident
}
w1_cycles() {
    cd "$EXP"
    for st in rest dent rigid; do
        PYTHONPATH=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/lib/python3.10/site-packages \
            "$BL" -b --factory-startup --python-use-system-env --python-exit-code 1 --python b67_w1_cycles.py -- "$R/out/w1_ladder" ladder "$st"
    done
    PYTHONPATH=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/lib/python3.10/site-packages \
        "$BL" -b --factory-startup --python-use-system-env --python-exit-code 1 --python b67_w1_cycles.py -- "$R/out/w1_bounce" bounce
}
w2_pt() { (cd "$EXP" && "$PY" b67_w2_pt.py "$R/out/w2_pt" ladder); }
w2_8dna() { (cd "$EXP" && "$PY" b67_w2_pt.py "$R/out/w2_pt" eightdna); }
blockers() { (cd "$EXP" && "$PY" b67_blockers.py "$HOME/NeuralShaderDemo/external/relightable-neural-assets" "$R/out/blockers.json"); }
analyze() {
    (cd "$EXP" && B67_LADDER=$R/out/w1_ladder B67_BOUNCE=$R/out/w1_bounce B67_INHERIT=$HOME/rest_wl56/out/runs B67_OUT=$R/out/analysis "$PY" b67_w1_analyze.py)
    (cd "$EXP" && "$PY" b67_w2_analyze.py "$R/w2in" "$R/out/w2_pt" "$R/out/analysis")
}

stage tests "$PY" "$EXP/tests/test_b67.py" "$R/out/tests.json"
stage w1_oracle w1_oracle
stage w1_cycles w1_cycles
stage w2_pt w2_pt
stage w2_8dna w2_8dna
stage blockers blockers
stage analyze analyze
echo done > "$STATUS"
