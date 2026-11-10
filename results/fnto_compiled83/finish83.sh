#!/bin/bash
cd ~/rest_compiled83 && . ./env.sh && cd code/experiments/fnto_compiled83
R=$HOME/rest_compiled83/results; ST=$HOME/rest_compiled83/chain.status
while pgrep -f "[p]art=" > /dev/null; do sleep 5; done
$PY - <<PYEOF >> $ST 2>&1
import json, os
out = '$R/render'
fs = [f'render83_part{i}of5.json' for i in (0, 1, 3)] + [f'render83_part{j}of10.json' for j in (2, 4, 7, 9)]
parts = [json.load(open(os.path.join(out, f))) for f in fs]
rows = sorted([r for p in parts for r in p['rows']], key=lambda r: (r['material'], r['light']))
arms = parts[0]['arms']
res = {'rows': rows, 'arms': arms, 'spp_test': parts[0]['spp_test'], 'spp_hq': parts[0]['spp_hq'],
       'RENDER_rule_per_arm': {a: bool(all(r[a]['pass'] for r in rows)) for a in arms},
       'pass_count_per_arm': {a: int(sum(r[a]['pass'] for r in rows)) for a in arms}, 'images': len(rows), 'part_files': fs,
       'concurrency': 'amendment 2: 5 concurrent render processes; parts 2 / 4 of 5 stalled (no output, GPU 0 %) and were killed; their 4 images were re-rendered as 4 single-image processes (parts j/10) with the same seeds; kernel GPU times may include overlap'}
json.dump(res, open(os.path.join(out, 'render83.json'), 'w'), indent=1, default=str)
print('$(date -u +%FT%TZ) END F_merge images', len(rows))
PYEOF
$PY viz83.py $HOME/rest_compiled83 $HOME/rest_compiled81/results/data > $R/logs/V_viz.log 2>&1; echo "$(date -u +%FT%TZ) END V_viz rc=$?" >> $ST
$PY gate83.py verdict $R $R/verdict83.json > $R/logs/X_verdict.log 2>&1; echo "$(date -u +%FT%TZ) END X_verdict rc=$?" >> $ST
echo "$(date -u +%FT%TZ) OUTCOME $($PY -c "import json;print(json.load(open('$R/verdict83.json'))['OUTCOME'])")" >> $ST
echo "$(date -u +%FT%TZ) CHAIN_C_DONE" >> $ST
