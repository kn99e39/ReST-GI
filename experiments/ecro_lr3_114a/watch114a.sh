#!/usr/bin/env bash
# WL114A watcher: waits until WL115 has finished including its worklog, then runs chain114a.sh (owner instruction 2026-11-26).
# Condition: docs/worklogs/115_*WORKLOG*.md tracked in HEAD and containing a final-report / architecture-verdict section,
# no python process running experiments/ecro_sr115, and GPU utilisation < 10 % for 10 consecutive minutes (polled every 60 s).
set -u
cd /c/Projects/ReST-GI
W=results/ecro_lr3_114a/watch.log
mkdir -p results/ecro_lr3_114a
echo "watch start $(date -Iseconds)" >> $W
idle=0
while true; do
  f=$(git ls-files 'docs/worklogs/115_*' | grep -i worklog | head -n 1)
  done115=0
  if [ -n "$f" ] && git show HEAD:"$f" 2>/dev/null | grep -qiE "final report|architecture verdict|최종 보고"; then done115=1; fi
  busy=$(powershell -NoProfile -c "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*ecro_sr115*' -or \$_.CommandLine -like '*115.py*' }).Count" 2>/dev/null | tr -d '\r')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' \r')
  if [ "${util:-100}" -lt 10 ] && [ "${busy:-1}" = "0" ]; then idle=$((idle + 1)); else idle=0; fi
  echo "$(date -Iseconds) wl115_worklog=${f:-none} final=$done115 py115=${busy:-?} gpu=${util:-?} idle_min=$idle" >> $W
  if [ "$done115" = "1" ] && [ "$idle" -ge 10 ]; then
    echo "condition met $(date -Iseconds): starting chain114a" >> $W
    cd experiments/ecro_lr3_114a && bash chain114a.sh > ../../results/ecro_lr3_114a/chain_console.log 2>&1
    echo "chain114a exited rc=$? $(date -Iseconds)" >> ../../$W
    exit 0
  fi
  sleep 60
done
