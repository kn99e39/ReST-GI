"""WL97 audit track D: independent recomputation of the WL96 cost-critic dirty-pair fraction (numpy only; scene86 imported read-only).
The critic's dirty.py counts front-facing S1-cell x receiver-point pairs whose segment crosses box A at the OLD OR the NEW pose (union).
A pair crossing box A at BOTH poses stays blocked (its visibility cannot change, whatever the other occluders are), so the set whose cached
visibility can change is the symmetric difference (XOR).  Same cells / receiver points / OBB test as the critic; shifts: -0.3 m (critic),
+0.15 m (WL96A Stage 3 timing-only edit).   python dirty_xor.py  -> results/audit97/D/dirty_xor.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'ecro86'))
import scene86 as S
ck = json.load(open(os.path.join(ROOT, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
G = S.Geom(ck['geom'])
u = (np.arange(4) + .5) / 4
pid = np.where(G.PMAT == 1)[0]
cells = np.array([G.PO[p] + a * G.PE1[p] + b * G.PE2[p] for p in pid for a in u for b in u]); pn = G.PN[pid[0]]
rp, rn, rpid = [], [], []
for p in range(G.SBASE):
    if G.PMAT[p] == 1:
        continue
    for a in u:
        for b in u:
            rp.append(G.PO[p] + a * G.PE1[p] + b * G.PE2[p]); rn.append(G.PN[p]); rpid.append(p)
rp = np.array(rp); rn = np.array(rn); rpid = np.array(rpid)
onA = np.isin(rpid, np.where(G.PSID == S.SID_BOXA)[0])


def obb_hit(o, d, box):
    c = np.array(box['c']); h = np.array(box['h']); ax = np.array(S.box_axes(box['yaw']))
    lo = np.zeros(d.shape[:-1]); hi = np.full(d.shape[:-1], 1.0)
    for k in range(3):
        oo = (o - c) @ ax[k]; dd = d @ ax[k]
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (-h[k] - oo) / dd; t2 = (h[k] - oo) / dd
        tn = np.minimum(t1, t2); tf = np.maximum(t1, t2)
        par = np.abs(dd) < 1e-12
        tn = np.where(par, np.where(np.abs(oo) <= h[k], -np.inf, np.inf), tn)
        tf = np.where(par, np.where(np.abs(oo) <= h[k], np.inf, -np.inf), tf)
        lo = np.maximum(lo, tn); hi = np.minimum(hi, tf)
    return (lo < hi - 1e-9) & (hi > 1e-6) & (lo < 1 - 1e-6)


A = ck['geom']['A']
out = {'cells': int(len(cells)), 'receiver_points': int(len(rp)), 'pairs': int(len(cells) * len(rp)), 'shifts': {}}
for name, dy in (('minus0.30y_critic', -0.3), ('plus0.15y_WL96A', 0.15)):
    A2 = dict(A); A2['c'] = [A['c'][0], A['c'][1] + dy, A['c'][2]]
    n = dict(front=0, front_static=0, old=0, new=0, union=0, both=0, xor=0, front_onA=0)
    for i in range(0, len(cells), 32):
        c = cells[i:i + 32][:, None, :]; d = rp[None] - c
        ff = ((d @ pn) > 1e-9) & (np.einsum('ijk,jk->ij', -d, rn) > 1e-9)
        st = ff & ~onA[None]
        h0 = obb_hit(c + 0 * d, d, A); h1 = obb_hit(c + 0 * d, d, A2)
        n['front'] += int(ff.sum()); n['front_static'] += int(st.sum()); n['front_onA'] += int((ff & onA[None]).sum())
        n['old'] += int((st & h0).sum()); n['new'] += int((st & h1).sum()); n['union'] += int((st & (h0 | h1)).sum())
        n['both'] += int((st & h0 & h1).sum()); n['xor'] += int((st & (h0 ^ h1)).sum())
    f = n['front']
    n.update(union_frac_of_front=n['union'] / f, xor_frac_of_front=n['xor'] / f, both_frac_of_front=n['both'] / f,
             xor_plus_moved_receivers_frac_of_front=(n['xor'] + n['front_onA']) / f, union_plus_moved_receivers_frac_of_front=(n['union'] + n['front_onA']) / f)
    out['shifts'][name] = n
json.dump(out, open(os.path.join(ROOT, 'results', 'audit97', 'D', 'dirty_xor.json'), 'w'), indent=1)
for k, v in out['shifts'].items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
