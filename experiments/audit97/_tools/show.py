"""WL97 helper: print phase-1 findings (full text) by id or by track.
  python experiments/audit97/_tools/show.py E4-F6 D-09 ...     # finding ids
  python experiments/audit97/_tools/show.py --track E5          # summary + P0 assessments + all findings of a track
  python experiments/audit97/_tools/show.py --index             # one line per finding
"""
import json, sys
R = json.load(open('results/audit97/phase1_results.json', encoding='utf-8'))
a = sys.argv[1:]
out = sys.stdout
out.reconfigure(encoding='utf-8')
def show(t, f):
    print(f"\n### [{f['id']}] ({t}) {f['title']}\ncategory={f['category']} material={f['material']} verdict={f['proposed_verdict']} confidence={f['confidence']} p0={f.get('p0_claims')} worklogs={f.get('worklogs')}")
    for k in ('description', 'evidence', 'independent_check', 'numerical_disagreement', 'decision_impact'):
        if f.get(k):
            print(f"- {k}: {f[k]}")
if a and a[0] == '--index':
    for t, r in R.items():
        for f in r['findings']:
            print(f"{f['id']:9s} {t:4s} {'M' if f['material'] else '-'} {f['proposed_verdict']:22s} {f['title']}")
elif a and a[0] == '--track':
    r = R[a[1]]
    print('SUMMARY:', r['summary'], '\nREPORT:', r['report_path'])
    for p in r['p0_assessments']:
        print('\nP0', p['claim'], '->', p['verdict'], '|', p.get('rationale', ''), '| restrictions:', p.get('restrictions', ''))
    for f in r['findings']:
        show(a[1], f)
    print('\nUNRESOLVED:', r.get('unresolved'))
else:
    for t, r in R.items():
        for f in r['findings']:
            if f['id'] in a:
                show(t, f)
