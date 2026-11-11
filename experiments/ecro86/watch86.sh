#!/bin/bash
# watch a WL86 chain: sample GPU utilisation every 2 s while it runs, print per-stage mean utilisation at the end
ST=$1/chain.status
OUT=$1/gpu_util_trace.txt
: > "$OUT"
while ! grep -qE "CHAIN_DONE|CHAIN_FAILED" "$ST" 2>/dev/null; do
  echo "$(date -u +%s) $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" >> "$OUT"
  sleep 2
done
cat "$ST"
python3 - "$ST" "$OUT" <<'PY'
import sys, datetime
st = [l.split() for l in open(sys.argv[1]) if l.strip()]
u = [tuple(map(int, l.split())) for l in open(sys.argv[2]) if l.strip()]
ts = lambda s: datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
starts = {}
for l in st:
    if l[1] == 'START': starts[l[2]] = ts(l[0])
    if l[1] == 'END':
        a, b = starts.get(l[2]), ts(l[0])
        v = [x for t, x in u if a <= t <= b]
        print(f'{l[2]:40s} {b - a:6.0f} s  GPU mean {sum(v) / max(len(v), 1):5.1f} %  ({len(v)} samples)')
PY
