"""WL97 audit track E3 - production-side facts (Dr.Jit / Mitsuba, frozen pt86 scene + DrClosure83; nothing here is audited measurement code).

  python val97_dr.py <CODE> <OUT>
    CODE : directory holding experiments/ (private copy of the frozen code)      OUT : output directory (own)
Writes OUT/val_dr.json and OUT/val_pairs.npz:
  T_GEOM   geo97 patch tables == scene86.Geom tables for patches 1088..1183
  T_FRAME  Mitsuba shading frame of the panel = (+y, +z, +x); box-A face normals
  T_COS    production eval on the Lambert state 4: which argument carries the cosine
  T_SYM    production f(a, b) = eval(a, b) / b.z symmetric?
  T_VIS    analytic visibility (front-facing sender and receiver, no other occluder) vs ray tests through the production scene
  pairs    production eval values on fixed direction pairs for the comparison with the independent float64 port (cmp97.py)
"""
import json, math, os, sys
import numpy as np

CODE, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CODE, 'experiments', 'ecro86'))
import geo97  # noqa: E402
import pt86 as PT  # noqa: E402
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
F, U = PT.F, PT.U
fl = PT.fl
res = {}
fx, sc = geo97.load(CODE)
ctx = PT.Ctx(sc['geom'])
G = ctx.G
LP = np.array(sc['lights'][0]['pos'], float)

# ---------------------------------------------------------------- T_GEOM
pan = geo97.panel(fx); faces = geo97.box_faces(sc['geom']['A'])
err = 0.0
for rect in [pan] + faces:
    O, e1, e2, area = geo97.patch_rects(rect)
    ids = rect['base'] + np.arange(len(O))
    err = max(err, np.abs(G.PO[ids] - O).max(), np.abs(G.PE1[ids] - e1).max(), np.abs(G.PE2[ids] - e2).max(), np.abs(G.PN[ids] - rect['n']).max(),
              np.abs(G.PT[ids] - rect['a1']).max(), np.abs(G.PB[ids] - rect['a2']).max(), np.abs(G.PAREA[ids] - area).max())
res['T_GEOM'] = {'max_abs_err': float(err), 'pass': bool(err < 1e-12), 'N': int(G.N), 'panel_pmat': sorted(set(G.PMAT[1088:1130].tolist())),
                 'boxA_pmat': sorted(set(G.PMAT[1130:1184].tolist())), 'panel_psid': sorted(set(G.PSID[1088:1130].tolist())),
                 'boxA_psid': sorted(set(G.PSID[1130:1184].tolist()))}


# ---------------------------------------------------------------- T_FRAME
def v3(a):
    a = np.atleast_2d(np.asarray(a, np.float32))
    return mi.Vector3f(fl(a[:, 0]), fl(a[:, 1]), fl(a[:, 2]))


def p3(a):
    a = np.atleast_2d(np.asarray(a, np.float32))
    return mi.Point3f(fl(a[:, 0]), fl(a[:, 1]), fl(a[:, 2]))


def npv(v):
    return np.stack([np.asarray(v.x, np.float64), np.asarray(v.y, np.float64), np.asarray(v.z, np.float64)], 1)


rng = np.random.default_rng(97301)
n = 64
pp = pan['c'] + (rng.random((n, 1)) * 2 - 1) * pan['h'][0] * pan['a1'] + (rng.random((n, 1)) * 2 - 1) * pan['h'][1] * pan['a2']
si = ctx.scene.ray_intersect(mi.Ray3f(p3(pp + 0.05 * pan['n']), v3(np.tile(-pan['n'], (n, 1)))))
sid = np.asarray(ctx.shape_id(si))
fr = {}
for nm, vec in (('s_local_of_+y', [0, 1, 0]), ('t_local_of_+z', [0, 0, 1]), ('n_local_of_+x', [1, 0, 0])):
    fr[nm] = npv(si.to_local(v3(np.tile(vec, (n, 1))))).mean(0).round(6).tolist()
fr['hit_shape_ids'] = sorted(set(sid.tolist())); fr['hit_dist_err'] = float(np.abs(np.asarray(si.t) - 0.05).max())
fr['pass'] = bool(np.allclose(fr['s_local_of_+y'], [1, 0, 0], atol=1e-5) and np.allclose(fr['t_local_of_+z'], [0, 1, 0], atol=1e-5)
                  and np.allclose(fr['n_local_of_+x'], [0, 0, 1], atol=1e-5) and fr['hit_shape_ids'] == [5])
bx = []
for f in faces:
    q = f['c'] + 0.3 * f['h'][0] * f['a1'] - 0.2 * f['h'][1] * f['a2']
    s2 = ctx.scene.ray_intersect(mi.Ray3f(p3(q + 0.05 * f['n']), v3(-f['n'])))
    nl = npv(s2.to_local(v3(f['n'])))[0]
    bx.append({'base': f['base'], 'shape': int(np.asarray(ctx.shape_id(s2))[0]), 'n_local_z': float(nl[2]), 't': float(np.asarray(s2.t)[0])})
fr['boxA_faces'] = bx
fr['boxA_pass'] = bool(all(b['shape'] == 7 and abs(b['n_local_z'] - 1) < 1e-5 and abs(b['t'] - 0.05) < 1e-5 for b in bx))
res['T_FRAME'] = fr


# ---------------------------------------------------------------- closure pairs
def hemi(nn, zlo=0.0, zhi=1.0):
    z = zlo + (zhi - zlo) * rng.random(nn); ph = rng.random(nn) * 2 * math.pi
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)


def peval(state, wi, wo):
    cl = CD.DrClosure83('ER', state)
    f = cl.eval(v3(wi), v3(wo), mi.Bool(True))
    return np.stack([np.asarray(f.x, np.float64), np.asarray(f.y, np.float64), np.asarray(f.z, np.float64)], 1)


st = ctx.mats.states
NP = 1 << 20
sets = {}
sets['uniform'] = (hemi(NP), hemi(NP))
wl = hemi(NP, 0.02, 0.25)                                         # grazing light, outgoing about the mirror direction (CK2 regime)
mir = wl * np.array([-1, -1, 1.0])
pert = mir + 0.08 * rng.standard_normal((NP, 3)); pert[:, 2] = np.abs(pert[:, 2]) + 1e-4
pert /= np.linalg.norm(pert, axis=1, keepdims=True)
sets['graze_mirror'] = (pert, wl)
sets['graze_any'] = (hemi(NP), hemi(NP, 0.02, 0.25))
save = {}
for nm, (wi, wo) in sets.items():
    wi = wi.astype(np.float32); wo = wo.astype(np.float32)
    save[f'{nm}_wi'] = wi; save[f'{nm}_wo'] = wo
    for m in (0, 1, 2, 4):
        save[f'{nm}_f{m}'] = peval(st[m], wi, wo).astype(np.float64)
np.savez_compressed(os.path.join(OUT, 'val_pairs.npz'), states=st, **save)

# T_COS / T_SYM (production)
wi, wo = save['uniform_wi'], save['uniform_wo']
f4 = save['uniform_f4']
ok = (wi[:, 2] > 1e-3) & (wo[:, 2] > 1e-3)
res['T_COS'] = {'max_rel_vs_rho_over_pi_cos_SECOND_arg': float(np.abs(f4[ok, 0] / (0.5 / math.pi * wo[ok, 2]) - 1).max()),
                'max_rel_vs_rho_over_pi_cos_FIRST_arg': float(np.abs(f4[ok, 0] / (0.5 / math.pi * wi[ok, 2]) - 1).max())}
res['T_COS']['pass_second_argument_carries_cosine'] = bool(res['T_COS']['max_rel_vs_rho_over_pi_cos_SECOND_arg'] < 1e-4)
sym = {}
for nm in ('uniform', 'graze_mirror'):
    wi, wo = save[f'{nm}_wi'], save[f'{nm}_wo']
    ok = (wi[:, 2] > 0.02) & (wo[:, 2] > 0.02)
    a = save[f'{nm}_f1'] / wo[:, 2:3]
    b = peval(st[1], wo, wi) / wi[:, 2:3]
    d = np.abs(a - b)[ok] / np.maximum(np.maximum(a, b)[ok], 1e-12)
    sym[nm] = {'median_rel': float(np.median(d)), 'p999_rel': float(np.quantile(d, 0.999)), 'max_rel': float(d.max())}
res['T_SYM_S1'] = sym


# ---------------------------------------------------------------- T_VIS
def seg_occluded(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = b - a; dist = np.linalg.norm(d, axis=1); w = d / dist[:, None]
    r = mi.Ray3f(p3(a + w * 1e-4), v3(w)); r.maxt = fl(dist - 2e-4)
    return np.asarray(ctx.scene.ray_test(r)).astype(bool)


NV = 1 << 21
ps = pan['c'] + (rng.random((NV, 1)) * 2 - 1) * pan['h'][0] * pan['a1'] + (rng.random((NV, 1)) * 2 - 1) * pan['h'][1] * pan['a2']
occ_l = seg_occluded(ps, np.tile(LP, (NV, 1)))
cl_ = (LP - ps) @ pan['n'] / np.linalg.norm(LP - ps, axis=1)
vis = {'light_to_panel': {'n': NV, 'occluded': int(occ_l.sum()), 'light_cos_min': float(cl_.min()), 'light_cos_max': float(cl_.max()),
                          'light_cos_mean': float(cl_.mean())}}
bc = np.array(sc['geom']['A']['c'], float); bh = np.array(sc['geom']['A']['h'], float)
cy, sy = math.cos(sc['geom']['A']['yaw']), math.sin(sc['geom']['A']['yaw'])
bax = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0, 0, 1.0]])


def inside_len(a, b):
    """Length of the part of segment a -> b that lies inside box A (slab method, float64)."""
    lo = (a - bc) @ bax.T; d = (b - a) @ bax.T
    d = np.where(np.abs(d) < 1e-300, 1e-300, d)
    t0 = (-bh - lo) / d; t1 = (bh - lo) / d
    tmin = np.clip(np.minimum(t0, t1).max(1), 0, 1); tmax = np.clip(np.maximum(t0, t1).min(1), 0, 1)
    return np.maximum(tmax - tmin, 0) * np.linalg.norm(b - a, axis=1)


bad_vis = 0; per = []
for f in faces:
    pr = f['c'] + (rng.random((NV, 1)) * 2 - 1) * f['h'][0] * f['a1'] + (rng.random((NV, 1)) * 2 - 1) * f['h'][1] * f['a2']
    d = pr - ps; dist = np.linalg.norm(d, axis=1); w = d / dist[:, None]
    cs = w @ pan['n']; cr = -(w @ f['n'])
    ana = (cs > 0) & (cr > 0)
    occ = seg_occluded(ps, pr)
    vis_but_occ = ana & occ                                        # would contradict the analytic visibility used by the quadrature
    hid_but_clear = ~ana & ~occ                                    # expected only where the segment's path inside the box is below the ray-test epsilons
    il = inside_len(ps[hid_but_clear], pr[hid_but_clear]) if hid_but_clear.any() else np.zeros(1)
    per.append({'face_base': f['base'], 'analytic_visible_frac': float(ana.mean()), 'visible_but_occluded': int(vis_but_occ.sum()),
                'visible_but_occluded_max_cos_r': float(cr[vis_but_occ].max()) if vis_but_occ.any() else 0.0,
                'visible_but_occluded_cos_r_weight_share': float(cr[vis_but_occ].sum() / cr[ana].sum()) if ana.any() else 0.0,
                'hidden_but_unoccluded': int(hid_but_clear.sum()), 'hidden_but_unoccluded_max_inside_len_m': float(il.max()),
                'cos_s_min': float(cs.min())})
    bad_vis += int((vis_but_occ & (cr > 2e-3)).sum())
vis['panel_to_boxA'] = per; vis['n_per_face'] = NV
vis['note2'] = 'visible_but_occluded pairs all have cos_r below 2e-3 (float32 ray grazing the receiver face); bad = such pairs with cos_r > 2e-3'
vis['note'] = 'ray test: origin a + 1e-4 w, maxt = dist - 2e-4; a hidden pair is reported unoccluded only if its path inside box A is shorter than these epsilons'
vis['pass'] = bool(bad_vis == 0 and vis['light_to_panel']['occluded'] == 0 and all(p['hidden_but_unoccluded_max_inside_len_m'] < 5e-4 for p in per))
res['T_VIS'] = vis
json.dump(res, open(os.path.join(OUT, 'val_dr.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
os._exit(0)
