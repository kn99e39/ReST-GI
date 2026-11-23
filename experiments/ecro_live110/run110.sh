#!/usr/bin/env bash
# WL110 evidence chain on LabServer69 (run detached: setsid -f bash run110.sh <out_root>).
# Xvfb display with a memory-mapped framebuffer -> live viewer runs (Tk window) with / without the screen recorder, uncapped,
# headless, reconstruction and near-mirror modes -> same-day WL101 run_b101 baseline -> verify110.
# Every timed step waits for the GPU to be free of foreign compute load and logs per-process GPU use (nvidia-smi pmon) beside it.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env110.sh"
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"
PY=$LIVE110_PY
DISP=${LIVE110_DISPLAY:-:110}
IDLE_MAX=${LIVE110_IDLE_MAX:-1200}
mkdir -p "$OUT/logs" "$OUT/runs" "$OUT/rec" "$OUT/xfb"
STATUS="$OUT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
say "START commit=$(git -C "$HERE" rev-parse HEAD) host=$(hostname)"

Xvfb "$DISP" -screen 0 1024x640x24 -fbdir "$OUT/xfb" -nolisten tcp > "$OUT/logs/xvfb.log" 2>&1 &
XPID=$!
trap 'kill $XPID 2>/dev/null' EXIT
sleep 2
export DISPLAY=$DISP
FB="$OUT/xfb/Xvfb_screen0"

idle_gate() {   # foreign compute processes below 3 % SM for 10 consecutive 1-s samples (max IDLE_MAX s)
  local t0=$(date +%s) ok=0
  while [ $ok -lt 10 ]; do
    busy=$(timeout 5 nvidia-smi pmon -s u -c 1 2>/dev/null | awk '!/^#/ && $4 ~ /^[0-9]+$/ && $4 > 2 {n++} END {print n+0}')
    if [ "$busy" = "0" ]; then ok=$((ok+1)); else ok=0; fi
    sleep 1
    if [ $(( $(date +%s) - t0 )) -gt "$IDLE_MAX" ]; then say "idle gate TIMEOUT before $1 (foreign GPU load present; recorded in pmon)"; return; fi
  done
  say "idle gate passed before $1 after $(( $(date +%s) - t0 )) s"
}

pmon_start() { nvidia-smi pmon -s um -d 1 > "$OUT/logs/$1.pmon" 2>&1 & PMON=$!; }
pmon_stop() { kill $PMON 2>/dev/null; wait $PMON 2>/dev/null; }

live() {        # live <name> <recorder-seconds or 0> <args...>
  local name=$1 recs=$2; shift 2
  [ -f "$OUT/runs/$name/summary.json" ] && { say "skip $name (done)"; return; }
  rm -rf "$OUT/runs/$name"
  idle_gate "$name"
  pmon_start "$name"
  if [ "$recs" != "0" ]; then
    "$PY" -u "$HERE/live110.py" --out "$OUT/runs/$name" "$@" > "$OUT/logs/$name.log" 2>&1 &
    local vp=$!
    "$PY" -u "$HERE/record110.py" "$FB" "$OUT/rec/$name" --region 0,0,908,528 --image-region 8,8,512,512 \
        --seconds "$recs" --delay 7 --quality ${LIVE110_WEBP_Q:-60} > "$OUT/logs/$name.rec.log" 2>&1
    wait $vp
  else
    "$PY" -u "$HERE/live110.py" --out "$OUT/runs/$name" "$@" > "$OUT/logs/$name.log" 2>&1
  fi
  pmon_stop
  say "$name: $(grep '^DONE' "$OUT/logs/$name.log" | cut -c1-260)"
}

wl101() {       # same-day WL101 arm B_sel run of the archived V_SSS_ANIM workload (unchanged run_b101.py)
  local name=$1
  [ -f "$OUT/wl101/$name.json" ] && { say "skip $name (done)"; return; }
  mkdir -p "$OUT/wl101"
  idle_gate "$name"
  pmon_start "$name"
  (cd "$HERE/../ecro_dyn101" && "$PY" -u run_b101.py V_SSS_ANIM B_sel "$OUT/wl101/$name") > "$OUT/logs/$name.log" 2>&1
  pmon_stop
  say "$name: $(grep '^DONE' "$OUT/logs/$name.log")"
}

N=${LIVE110_FRAMES:-630}       # 30 warm-up + 600 measured frames (20 s at 30 FPS; 3.3 animation cycles of 6 s)
wl101 wl101_V_SSS_ANIM_B_sel_r1
live sss_raw_cap30_r1   0  --case sss --mode raw --fps-cap 30 --frames $N --save-last
live sss_raw_uncap_r1   0  --case sss --mode raw --fps-cap 0  --frames $N
live sss_raw_headless_r1 0 --case sss --mode raw --fps-cap 0  --frames $N --present none
live sss_raw_cap30_rec  10 --case sss --mode raw --fps-cap 30 --frames $N --save-last
live sss_svgf_cap30_rec 8  --case sss --mode svgf --fps-cap 30 --frames $N --save-last
live near_raw_cap30_rec 8  --case near --mode raw --fps-cap 30 --frames $N --save-last
live sss_raw_cap30_r2   0  --case sss --mode raw --fps-cap 30 --frames $N
live sss_raw_uncap_r2   0  --case sss --mode raw --fps-cap 0  --frames $N
live sss_raw_headless_r2 0 --case sss --mode raw --fps-cap 0  --frames $N --present none
live sss_svgf_uncap_r1  0  --case sss --mode svgf --fps-cap 0 --frames $N
live sss_raw_cap60_r1   0  --case sss --mode raw --fps-cap 60 --frames $N
wl101 wl101_V_SSS_ANIM_B_sel_r2
if [ ! -f "$OUT/verify/verify.json" ]; then
  idle_gate verify
  rm -rf "$OUT/verify"
  "$PY" -u "$HERE/verify110.py" "$OUT/verify" > "$OUT/logs/verify.log" 2>&1
  say "verify: $(grep -c '' "$OUT/logs/verify.log") log lines; $(tail -1 "$OUT/logs/verify.log")"
fi
say "DONE"
