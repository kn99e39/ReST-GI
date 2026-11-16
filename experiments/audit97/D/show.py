"""WL97 audit track D: compact print of results/audit97/D/ceilings.json (helper)."""
import json, os, sys
o = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'results', 'audit97', 'D', 'ceilings.json')))
r = lambda x: round(x, 4) if isinstance(x, float) else x


def show(d, ind=0, maxd=5):
    for k, v in d.items():
        if isinstance(v, dict) and ind < 2 * maxd:
            print(' ' * ind + str(k) + ':'); show(v, ind + 2, maxd)
        elif isinstance(v, list):
            print(' ' * ind + str(k) + ':', [(tuple(r(a) for a in t) if isinstance(t, (list, tuple)) else r(t)) for t in v][:12])
        else:
            print(' ' * ind + str(k) + ':', r(v))


for k in sys.argv[1:] or list(o):
    print('=====', k); show(o[k]) if isinstance(o[k], dict) else print(o[k])
