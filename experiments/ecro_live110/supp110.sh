#!/usr/bin/env bash
# WL110 supplementary step (execution fix, disclosed): the frozen chain's SVGF recording started from a fixed 7 s delay and ended
# before the slower-starting SVGF viewer (Triton compilation) showed its window, so it captured an empty screen in both chain
# roots. This step repeats ONLY that run, with the recorder waiting for the viewer to present (record110.py --wait-change).
#   usage: bash supp110.sh <run110 out_root>      (adds runs/sss_svgf_cap30_rec2, rec/sss_svgf_cap30_rec2)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env110.sh"
OUT="$(cd "$1" && pwd)"
PY=$LIVE110_PY
DISP=${LIVE110_DISPLAY:-:111}
NAME=sss_svgf_cap30_rec2
STATUS="$OUT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
[ -f "$OUT/runs/$NAME/summary.json" ] && { say "skip $NAME (done)"; exit 0; }
mkdir -p "$OUT/xfb_supp"
say "SUPP start $NAME commit=$(git -C "$HERE" rev-parse HEAD)"
Xvfb "$DISP" -screen 0 1024x640x24 -fbdir "$OUT/xfb_supp" -nolisten tcp > "$OUT/logs/xvfb_supp.log" 2>&1 &
XPID=$!
trap 'kill $XPID 2>/dev/null' EXIT
sleep 2
export DISPLAY=$DISP
t0=$(date +%s); ok=0
while [ $ok -lt 10 ]; do
  busy=$(timeout 5 nvidia-smi pmon -s u -c 1 2>/dev/null | awk '!/^#/ && $4 ~ /^[0-9]+$/ && $4 > 2 {n++} END {print n+0}')
  if [ "$busy" = "0" ]; then ok=$((ok+1)); else ok=0; fi
  sleep 1
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && { say "idle gate TIMEOUT before $NAME"; break; }
done
say "idle gate passed before $NAME after $(( $(date +%s) - t0 )) s"
nvidia-smi pmon -s um -d 1 > "$OUT/logs/$NAME.pmon" 2>&1 & PMON=$!
"$PY" -u "$HERE/live110.py" --out "$OUT/runs/$NAME" --case sss --mode svgf --fps-cap 30 --frames 630 --save-last > "$OUT/logs/$NAME.log" 2>&1 &
VP=$!
"$PY" -u "$HERE/record110.py" "$OUT/xfb_supp/Xvfb_screen0" "$OUT/rec/$NAME" --region 0,0,908,528 --image-region 8,8,512,512 \
    --seconds 8 --wait-change 180 --delay 2 --quality ${LIVE110_WEBP_Q:-60} > "$OUT/logs/$NAME.rec.log" 2>&1
wait $VP
kill $PMON 2>/dev/null
say "$NAME: $(grep '^DONE' "$OUT/logs/$NAME.log" | cut -c1-260)"
say "$NAME recorder: $(grep '^RECORDED' "$OUT/logs/$NAME.rec.log")"
say "SUPP DONE"
