"""WL115 focused tests (throwaway seeds 7915xxx; never evidence).

T1 readout identity: readout_multi 'pc' K=4 == tracer112.cam_pass 'sub' readout; LOW identical (pairing with WL114 references)
T2 bilinear exactness: a field constant within each rectangle (per bin) and constant on the sphere -> 'bl' == 'pc'
T3 bilinear reproduction: a field linear in the lattice coordinate (floor: world x / y; sphere: lattice row, bin-independent)
   is reproduced exactly at interior queries; sphere periodic wrap and pole clamping give no out-of-range read
T4 nesting: table_from_sums(pool(8 -> 4)) == tracer113.tables4 4 x 4 contract on the same sums; empty cells take the parent
T5 construct_k(K = 4, all patches) == tracer114.construct_deep (same seed / partition): identical per-cell sums
T6 deep_capture == tracer114.deep_pass images (same seed); sum over event-2 cells of the captured HIGH == the image HIGH
T7 address consistency: construction mismatch fraction <= 1e-4 at K = 8 and K = 32
T8 linearity: lookup(a T1 + b T2) == a lookup(T1) + b lookup(T2) for 'pc' and 'bl'; non-negative tables give non-negative values
"""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common115 as C
import tracer115 as T
PT, mi, dr, F, U, S = T.PT, T.mi, T.dr, T.F, T.U, T.S
NB = T.NB
res = {}
g, l = C.fixture('F0'); cam = dict(C.views('F0')['V0'], res=C.RES)
ctx = T.Ctx115(g, 'C'); light = PT.Lights(l); G = ctx.G; N = G.N
z = np.load(C.WL114_ROOT / 'stageB' / 'tables' / 'oracle_F0_r0.npz'); S4 = z['S4_F']

# T1
img, _, _, _ = ctx.cam_pass(light, cam, C.RES, 64, 7915001, 'ecro', tables=[('S4', PT.fl(S4), 'sub')], b=64)
im2 = ctx.readout_multi(light, cam, C.RES, 64, 7915001, {'S4': T.pc_uniform(S4, 4)}, b=64)
r1 = float(np.abs(img['S4'] - im2['S4']).sum() / np.abs(img['S4']).sum())
low = float(max(np.abs(img['e1'] - im2['e1']).max(), np.abs(img['e2'] - im2['e2']).max()) / max(np.abs(img['e1']).max(), 1e-9))
res['T1'] = {'hl_rel_l1': r1, 'low_rel_max': low, 'pass': r1 <= 1e-6 and low <= 1e-5}

# T2
rng = np.random.default_rng(7915002)
surf = ctx.LT['surf']
f_surf = rng.uniform(0.1, 1.0, (S.N_RECT + 1, NB, 3)); f_surf[S.N_RECT] = rng.uniform(0.1, 1.0, (1, 3))   # sphere: constant
Tc = np.repeat(f_surf[surf][:, None], 16, 1).astype(np.float32)                                       # [N,16,NB,3]
im3 = ctx.readout_multi(light, cam, C.RES, 64, 7915003, {'pc': T.pc_uniform(Tc, 4), 'bl': T.bl_rep(Tc, 4)}, b=64)
r2 = float(np.abs(im3['bl'] - im3['pc']).sum() / np.abs(im3['pc']).sum())
res['T2'] = {'bl_vs_pc_rel_l1': r2, 'pass': r2 <= 1e-6}


# T3: synthetic captured queries
def fake_cap(pid, a, b, w):
    pid = np.asarray(pid, np.uint32)
    t, bv, n = G.PT[pid], G.PB[pid], G.PN[pid]
    loc = np.stack([(w * t).sum(1), (w * bv).sum(1), (w * n).sum(1)], 1)
    iz = np.clip(np.floor(loc[:, 2] * S.NZ), 0, S.NZ - 1); ip = np.clip(np.floor((np.arctan2(loc[:, 1], loc[:, 0]) + math.pi) * S.NPHI / (2 * math.pi)), 0, S.NPHI - 1)
    bn = (iz * S.NPHI + ip).astype(np.uint32)
    return {'pid': U(pid), 'bin': U(bn), 'a': F(np.asarray(a, np.float32)), 'b': F(np.asarray(b, np.float32)),
            'w': mi.Vector3f(*[F(np.asarray(w[:, k], np.float32)) for k in range(3)])}, bn


K = 4
nq = 20000
fl_p = np.flatnonzero(G.PSID == 0)
pid = rng.choice(fl_p, nq); a = rng.uniform(0, 1, nq); b = rng.uniform(0, 1, nq)
w = np.tile([0., 0., 1.], (nq, 1)) + rng.normal(0, .3, (nq, 3)); w[:, 2] = np.abs(w[:, 2]) + .2; w /= np.linalg.norm(w, axis=1, keepdims=True)
pos = G.PO[pid] + a[:, None] * G.PE1[pid] + b[:, None] * G.PE2[pid]
# floor field linear in world x, y at cell centres (identical for every bin)
cent = np.zeros((N, 16, 3))
for p in fl_p:
    for s in range(16):
        cent[p, s] = G.PO[p] + ((s % 4) + .5) / 4 * G.PE1[p] + ((s // 4) + .5) / 4 * G.PE2[p]
lin = 0.3 + 0.2 * cent[..., 0] - 0.1 * cent[..., 1]
Tl = np.repeat(np.repeat(lin[:, :, None, None], NB, 2), 3, 3).astype(np.float32)
cap, bn = fake_cap(pid, a, b, w)
v = np.asarray(ctx.lookup(T.bl_rep(Tl, K), cap)[0])
exp = 0.3 + 0.2 * pos[:, 0] - 0.1 * pos[:, 1]
inner = (np.abs(pos[:, 0]) < 2 - 0.5 / 64 * 4) & (np.abs(pos[:, 1]) < 2 - 0.5 / 64 * 4)
e_floor = float(np.abs(v - exp)[inner].max())
# sphere: field linear in the lattice row coordinate (bin independent), constant in phi
sp = np.flatnonzero(G.PSID == S.SID_SPHERE)
pid = rng.choice(sp, nq); a = rng.uniform(0, 1, nq); b = rng.uniform(0, 1, nq)
rowc = np.zeros((N, 16))
for p in sp:
    for s in range(16):
        rowc[p, s] = G.PSPH[p, 1] * K + (s // 4) + .5
lin2 = 1.0 + 0.05 * rowc
Ts = np.repeat(np.repeat(lin2[:, :, None, None], NB, 2), 3, 3).astype(np.float32)
w = rng.normal(0, 1, (nq, 3)); w /= np.linalg.norm(w, axis=1, keepdims=True)
cap, _ = fake_cap(pid, a, b, w)
v = np.asarray(ctx.lookup(T.bl_rep(Ts, K), cap)[0])
y = (G.PSPH[pid, 1] + b) * K
exp = 1.0 + 0.05 * np.clip(y, 0.5, 4 * K - 0.5)
e_sph = float(np.abs(v - exp).max())
res['T3'] = {'floor_linear_max_abs': e_floor, 'sphere_row_linear_max_abs': e_sph, 'finite': bool(np.isfinite(v).all()),
             'pass': e_floor <= 2e-5 and e_sph <= 2e-5 and bool(np.isfinite(v).all())}

# T4
w8 = rng.integers(0, 4, (N, 64, NB)).astype(np.float64); L8 = rng.uniform(0, 1, (N, 64, NB, 3)) * w8[..., None]
w4, L4 = T.pool(w8, 8, 4), T.pool(L8, 8, 4)
acc = {'U_w': w4.reshape(N, 4, 4, NB).reshape(-1)}
for c in range(3):
    acc[f'U_f2{c}'] = L4[..., c].reshape(-1); acc[f'U_fh{c}'] = np.zeros(N * 16 * NB)
_, _, T4ref, _ = T.T113.tables4(acc, N)
T4 = T.table_from_sums(w4, L4, 4)
r4 = float(np.abs(T4 - T4ref).max())
res['T4'] = {'max_abs_vs_tables4': r4, 'empty_cells_4': int((w4 == 0).sum()), 'pass': r4 <= 1e-6}

# T5
accA, _ = ctx.construct_deep(light, 2, 7915005, 23, b=1)
accB, dg = ctx.construct_k(light, 2, 7915005, 23, 4, b=1)
hfA = accA['U_f20'] + accA['U_fh0'] + accA.get('U_c7_120', 0.) + accA.get('U_c13_240', 0.)
hfB = (accB['HF00'] + accB['HF10']).ravel()
r5 = float(np.abs(hfA - hfB).sum() / max(np.abs(hfA).sum(), 1e-30))
rw = float(np.abs(accA['U_w'] - (accB['w0'] + accB['w1']).ravel()).max())
res['T5'] = {'HF_rel_l1': r5, 'w_max_abs': rw, 'pass': r5 <= 1e-12 and rw == 0}

# T6
cam64 = dict(cam, res=128)
imD, _ = ctx.deep_pass(light, cam64, 128, 32, 7915006, 24, b=32)
pids = T.contact_patches(G, g['S']['c'][:2]); loc = -np.ones(N, np.int64); loc[pids] = np.arange(len(pids))
imC, cells = ctx.deep_capture(light, cam64, 128, 32, 7915006, {'Q8': (8, None), 'Q32': (32, loc)}, b=32)
r6 = max(float(np.abs(imD[k] - imC[k]).sum() / max(np.abs(imD[k]).sum(), 1e-30)) for k in ('e1', 'e2', 'h', 'c7_12', 'c13_24'))
hi_img = sum(imC[k].sum((0, 1)) for k in ('h', 'c7_12', 'c13_24')) * 32
hi_cells = np.array([cells['Q8'][f'h{c}'].sum() for c in range(3)])
r6b = float(np.abs(hi_img - hi_cells).max() / max(np.abs(hi_img).max(), 1e-30))
res['T6'] = {'img_rel_l1_max': r6, 'high_conservation_rel': r6b, 'q32_le_q8': bool(cells['Q32']['h1'].sum() <= cells['Q8']['h1'].sum() * (1 + 1e-9)),
             'pass': r6 <= 1e-6 and r6b <= 1e-5}

# T7
_, d8 = ctx.construct_k(light, 2, 7915007, 23, 8, b=2)
_, d32 = ctx.construct_k(light, 2, 7915008, 23, 32, pids=pids, b=2)
m8 = d8['mismatch'] / d8['counted']; m32 = d32['mismatch'] / d32['counted']
res['T7'] = {'mismatch_K8': m8, 'mismatch_K32': m32, 'pass': m8 <= 1e-4 and m32 <= 1e-4}

# T8
pid = rng.choice(N, nq); a = rng.uniform(0, 1, nq); b = rng.uniform(0, 1, nq)
w = rng.normal(0, 1, (nq, 3)); w /= np.linalg.norm(w, axis=1, keepdims=True)
cap, _ = fake_cap(pid, a, b, w)
A_ = rng.uniform(0, 1, (N, 16, NB, 3)).astype(np.float32); B_ = rng.uniform(0, 1, (N, 16, NB, 3)).astype(np.float32)
lin_err = 0.; nonneg = True
for mk in (lambda t_: T.pc_uniform(t_, 4), lambda t_: T.bl_rep(t_, 4)):
    va = np.asarray(mk(A_) and ctx.lookup(mk(A_), cap)[1]); vb = np.asarray(ctx.lookup(mk(B_), cap)[1])
    vab = np.asarray(ctx.lookup(mk((2 * A_ + 3 * B_).astype(np.float32)), cap)[1])
    lin_err = max(lin_err, float(np.abs(vab - (2 * va + 3 * vb)).max() / np.abs(vab).max()))
    nonneg &= bool((va >= 0).all() and (vb >= 0).all())
res['T8'] = {'linearity_rel_max': lin_err, 'nonnegative': nonneg, 'pass': lin_err <= 1e-6 and nonneg}

res['all_pass'] = all(v['pass'] for k, v in res.items() if isinstance(v, dict))
out = Path(sys.argv[1]) / 'tests115.json' if len(sys.argv) > 1 else HERE / 'tests115_dev.json'
out.parent.mkdir(parents=True, exist_ok=True)
C.jd(res, out)
print(json.dumps(res, indent=1), flush=True)
import os; os._exit(0 if res['all_pass'] else 1)
