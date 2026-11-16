"""WL97 audit, phase 2, track J5: independent count of the WL96 cost critic's 'dirty pair' set (D-07).
Own segment / oriented-box test (clipping in the box frame), numpy only; scene86 imported read-only for the geometry.
Pair set as in results/wl96_calc/scratch/critic_cost/dirty.py: S1 panel patches x (4 x 4 points), receiver points 4 x 4 on every
rectangular patch except the panel, front-facing on both ends.  Edit: box A shifted by -0.3 m in y (the critic's edit).
Writes results/audit97/J5/j5_dirty.json.   python j5_dirty.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'ecro86'))
import scene86 as S

ck = json.load(open(os.path.join(ROOT, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
G = S.Geom(ck['geom'])
u = (np.arange(4) + 0.5) / 4.0
uu, vv = np.meshgrid(u, u, indexing='ij'); uu = uu.ravel(); vv = vv.ravel()


def pts(p):
    return G.PO[p][None] + uu[:, None] * G.PE1[p][None] + vv[:, None] * G.PE2[p][None]


panel = np.where(G.PMAT == 1)[0]
cells = np.concatenate([pts(p) for p in panel]); pn = G.PN[panel[0]]
rec = [p for p in range(G.SBASE) if G.PMAT[p] != 1]
rp = np.concatenate([pts(p) for p in rec]); rn = np.repeat(G.PN[rec], 16, axis=0); rsid = np.repeat(G.PSID[rec], 16)
onA = rsid == S.SID_BOXA


def crosses(o, e, box):
    """True where the open segment o -> e passes through the interior of the oriented box (Liang-Barsky clipping in the box frame)."""
    c = np.asarray(box['c'], float); h = np.asarray(box['h'], float); ax = np.asarray(S.box_axes(box['yaw']), float)   # rows = box axes
    a = (o - c) @ ax.T; b = (e - c) @ ax.T; d = b - a
    t0 = np.zeros(d.shape[:-1]); t1 = np.ones(d.shape[:-1]); ok = np.ones(d.shape[:-1], bool)
    for k in range(3):
        dk = d[..., k]; ak = a[..., k]
        par = np.abs(dk) < 1e-12
        ok &= ~(par & (np.abs(ak) > h[k]))
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = (-h[k] - ak) / dk; tb = (h[k] - ak) / dk
        lo = np.where(par, -np.inf, np.minimum(ta, tb)); hi = np.where(par, np.inf, np.maximum(ta, tb))
        t0 = np.maximum(t0, lo); t1 = np.minimum(t1, hi)
    # interior crossing of positive length, not merely touching at the receiver end (t = 1) or the cell end (t = 0)
    return ok & (t1 - t0 > 1e-9) & (t1 > 1e-6) & (t0 < 1 - 1e-6)


A0 = ck['geom']['A']
out = {'cells': int(len(cells)), 'receiver_points': int(len(rp)), 'pairs': int(len(cells) * len(rp)), 'shifts': {}}
for name, dy in (('minus0.30y_critic', -0.3), ('plus0.15y_WL96A', 0.15)):
    A1 = dict(A0); A1['c'] = [A0['c'][0], A0['c'][1] + dy, A0['c'][2]]
    n = dict(front=0, front_static=0, front_onA=0, old=0, new=0, union=0, both=0, old_only=0, new_only=0)
    for j in range(0, len(cells), 48):
        o = cells[j:j + 48][:, None, :]; e = rp[None, :, :]
        d = e - o
        ff = ((d @ pn) > 1e-9) & (np.einsum('ijk,jk->ij', -d, rn) > 1e-9)
        st = ff & ~onA[None]
        ob = np.broadcast_to(o, d.shape); eb = np.broadcast_to(e, d.shape)
        h0 = crosses(ob, eb, A0); h1 = crosses(ob, eb, A1)
        n['front'] += int(ff.sum()); n['front_static'] += int(st.sum()); n['front_onA'] += int((ff & onA[None]).sum())
        n['old'] += int((st & h0).sum()); n['new'] += int((st & h1).sum()); n['union'] += int((st & (h0 | h1)).sum()); n['both'] += int((st & h0 & h1).sum())
        n['old_only'] += int((st & h0 & ~h1).sum()); n['new_only'] += int((st & h1 & ~h0).sum())
    f = n['front']
    n['fractions_of_front'] = {'union_static': n['union'] / f, 'union_static_plus_onA': (n['union'] + n['front_onA']) / f, 'both': n['both'] / f,
                               'xor': (n['old_only'] + n['new_only']) / f, 'xor_plus_onA': (n['old_only'] + n['new_only'] + n['front_onA']) / f,
                               'old_only_plus_onA_(visibility rays strictly needed)': (n['old_only'] + n['front_onA']) / f}
    # the critic's conversion: 1.05 M rays per 11.5 ms, linear; ECRO per edit 77 ms
    rate = 11.5 / 1.05e6
    n['ms_at_critic_linear_rate'] = {'union_plus_onA': (n['union'] + n['front_onA']) * rate, 'xor_plus_onA': (n['old_only'] + n['new_only'] + n['front_onA']) * rate,
                                     'old_only_plus_onA': (n['old_only'] + n['front_onA']) * rate}
    n['total_over_ECRO_77ms'] = {k: (77.0 + v) / 77.0 for k, v in n['ms_at_critic_linear_rate'].items()}
    out['shifts'][name] = n
os.makedirs(os.path.join(ROOT, 'results', 'audit97', 'J5'), exist_ok=True)
with open(os.path.join(ROOT, 'results', 'audit97', 'J5', 'j5_dirty.json'), 'w', newline='\n') as fh:
    json.dump(out, fh, indent=1)
for k, v in out['shifts'].items():
    print(k, json.dumps({a: (b if not isinstance(b, dict) else {x: round(y, 4) for x, y in b.items()}) for a, b in v.items()}))
