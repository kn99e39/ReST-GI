"""WL97 audit track D: geometric validity of the timing-only edits (numpy only; scene86 imported read-only).
WL96A Stage 3: box A translated by +0.15 m in y (and back).  WL96 cost critic (dirty.py): box A shifted by -0.3 m in y.
Checks: scene86.geom_ok (room bounds, sphere and box-box clearance), box A vs every stationary rectangle (OBB corner / segment test)."""
import copy, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'ecro86'))
import scene86 as S  # noqa: E402
ck = json.load(open(os.path.join(HERE, '..', '..', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
g0 = ck['geom']


def shifted(dy):
    g = copy.deepcopy(g0); g['A']['c'][1] += dy
    return g


def rect_box_clear(g):
    """min over stationary rectangles of the distance proxy: does any rect sample point lie inside box A (pad 1 mm)?"""
    c, ax, h = np.array(g['A']['c']), np.stack(S.box_axes(g['A']['yaw'])), np.array(g['A']['h'])
    out = []
    for i, r in enumerate(S.stationary_rects()):
        rc, a1, a2, h1, h2 = r[:5]
        u = np.linspace(-1, 1, 81)
        pts = rc + u[:, None, None] * a1 * h1 + u[None, :, None] * a2 * h2
        loc = (pts.reshape(-1, 3) - c) @ ax.T
        inside = np.all(np.abs(loc) < h - 1e-3, 1)
        out.append({'rect': i, 'mat': int(r[7]), 'points_inside_boxA': int(inside.sum()), 'max_penetration_m': float(np.max(np.min(h - np.abs(loc), 1)))})
    return out


res = {}
for name, dy in (('CK2_T0', 0.0), ('WL96A_plus0.15y', 0.15), ('WL96critic_minus0.3y', -0.3)):
    g = shifted(dy)
    G = S.Geom(g)
    rb = rect_box_clear(g)
    res[name] = {'A_center': g['A']['c'], 'geom_ok': bool(S.geom_ok(g)), 'N': int(G.N),
                 'light_ok_LCK2': bool(S.light_ok(ck['lights'][0]['pos'], g)),
                 'A_corner_min': S._corners(g['A']).min(0).tolist(), 'A_corner_max': S._corners(g['A']).max(0).tolist(),
                 'rects_intersecting_boxA': [r for r in rb if r['points_inside_boxA'] > 0],
                 'closest_rect_penetration_m': max(r['max_penetration_m'] for r in rb)}
out = os.path.join(HERE, '..', '..', '..', 'results', 'audit97', 'D', 'edit_validity.json')
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
