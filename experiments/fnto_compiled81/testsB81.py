"""WL81 Stage B: focused regression of the WL80D sampler correction on the frozen WL80C parameters (B1-B5, WL80D code exactly as
committed, imported), and the horizon-numerics correction (B6, closure81 / closure81_dr).

  python testsB81.py <wl80c_smoke_dir> <out_dir>
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_closure80d'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import run80d as RD  # noqa: E402  (WL80D: tests(), regen_sets(), adversarial_raw(); CD = closure80d_dr)
import torch  # noqa: E402
mi = RD.mi
import drjit as dr  # noqa: E402

C = RD.C
CO = RD.CO
c80, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
res = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}
dev_t = torch.device('cuda')

# ---------------- B1-B4: WL80D T1-T4 unchanged (WL80D sampler, WL80C numerics, WL80C frozen parameters)
t0 = time.time()
okB14 = RD.tests(c80, out)
res['B1_B4'] = json.load(open(os.path.join(out, 'tests.json')))
res['B1_B4']['PASS'] = bool(okB14)
res['B1_B4']['seconds'] = time.time() - t0

# ---------------- B5: WL80C contract on the frozen WL80C control closures with the WL80D sampler
t0 = time.time()
fitw = np.load(os.path.join(c80, 'fit.npz'))
tabs = {'s8001_test_iid': fitw['states'].astype(np.float32),     # key name required by the WL80C sampling-utility helper;
        'adversarial': C.compiled_state(C.decode(RD.adversarial_raw(RD.P80C['seeds']['adversarial']))).numpy()}   # WL80D's control adversarial set   # these are the WL80C oracle-P control closures
CO.CD = RD.CD
CO.states_for = lambda fit_dir, mats: tabs
os.makedirs(os.path.join(out, 'B5'), exist_ok=True)
CO.run(None, c80, os.path.join(out, 'B5'))
b5 = json.load(open(os.path.join(out, 'B5', 'stage3.json')))
need = ['C1', 'C2', 'C3', 'C5a', 'C6', 'C7']
res['B5'] = {'checks': {k: b5['verdict']['checks'][k] for k in need}, 'C4': b5['verdict']['checks']['C4'], 'C5b': b5['verdict']['checks']['C5b'],
             'sampling_utility_report': b5['sampling_utility'], 'seconds': time.time() - t0}
res['B5']['PASS'] = bool(all(res['B5']['checks'].values()))

# ---------------- B6: horizon numerics
t0 = time.time()
import closure81 as K  # noqa: E402
import closure81_dr as CD81  # noqa: E402
rng = np.random.default_rng(81040)
b6 = {}
# (a) corrected == original away from the horizon (torch eval; Dr.Jit pdf)
z = torch.randn(64, C.N_RAW, generator=torch.Generator().manual_seed(81040)) * 3.0
p = C.decode(z.to(dev_t))
N = 1 << 20
idx = torch.from_numpy(rng.integers(0, 64, N)).to(dev_t)
def updirs(n, cmin):
    v = rng.normal(size=(n, 3)); v[:, 2] = np.abs(v[:, 2]); v /= np.linalg.norm(v, axis=1, keepdims=True)
    bad = v[:, 2] < cmin
    while bad.any():
        w = rng.normal(size=(bad.sum(), 3)); w[:, 2] = np.abs(w[:, 2]); w /= np.linalg.norm(w, axis=1, keepdims=True)
        v[bad] = w; bad = v[:, 2] < cmin
    return v.astype(np.float32)
wi = updirs(N, 0.02); wo = updirs(N, 0.02)
with torch.no_grad():
    pl = C.gather(p, idx)
    e_new = K.eval_closure(pl, torch.from_numpy(wi).to(dev_t), torch.from_numpy(wo).to(dev_t)).double()
    e_old = K.original_eval_closure(pl, torch.from_numpy(wi).to(dev_t), torch.from_numpy(wo).to(dev_t)).double()
m = e_old.abs() > 1e-6 * e_old.abs().max()
r = ((e_new - e_old).abs() / e_old.abs().clamp_min(1e-30))[m]
tab = C.compiled_state(p).cpu().numpy()
cl_old = RD.CD.DrClosure(tab, mi.UInt32(idx.cpu().numpy().astype(np.uint32)))
cl_new = CD81.DrClosure81(tab, mi.UInt32(idx.cpu().numpy().astype(np.uint32)))
_, p_old = cl_old.eval_pdf(RD.V3(wi), RD.V3(wo)); f_new, p_new = cl_new.eval_pdf(RD.V3(wi), RD.V3(wo))
p_old = np.asarray(p_old, np.float64); p_new = np.asarray(p_new, np.float64)
mp = p_old > 1e-6 * p_old.max()
rp = (np.abs(p_new - p_old) / p_old)[mp]
b6['a_away_from_horizon'] = {'eval_max_rel': float(r.max()), 'eval_median_rel': float(r.median()), 'pdf_max_rel': float(rp.max())}
b6['a_away_from_horizon']['PASS'] = b6['a_away_from_horizon']['eval_max_rel'] <= 1e-5 and b6['a_away_from_horizon']['pdf_max_rel'] <= 1e-5
# (b) single-lobe albedo vs mu: original vs corrected (uniform-hemisphere MC, float64 accumulation)
Nq = 1 << 22
u = torch.rand(Nq, 2, device=dev_t, dtype=torch.float64, generator=torch.Generator(device=dev_t).manual_seed(81041))
zc = u[:, 0]; rr = torch.sqrt(1 - zc * zc)
woq = torch.stack([rr * torch.cos(2 * math.pi * u[:, 1]), rr * torch.sin(2 * math.pi * u[:, 1]), zc], 1).float()
rows = []
okb = True
for au, av in [(0.004, 0.004), (0.05, 0.05), (0.3, 0.3), (1.0, 1.0), (0.3, 0.075), (0.05, 0.0125)]:
    pp = {'lam': torch.zeros(1, 3, 8, device=dev_t), 'au': torch.full((1, 7), au, device=dev_t), 'av': torch.full((1, 7), av, device=dev_t),
          'f0': torch.ones(1, 7, device=dev_t)}
    pp['lam'][:, :, 1] = 1.0
    for mu in [1e-1, 1e-2, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 2e-7]:
        for phi in [0.0, math.pi / 2]:
            s = math.sqrt(1 - mu * mu)
            wiq = torch.tensor([[s * math.cos(phi), s * math.sin(phi), mu]], device=dev_t).float().repeat(Nq, 1)
            pl = C.gather(pp, torch.zeros(Nq, dtype=torch.long, device=dev_t))
            with torch.no_grad():
                a_new = K.eval_closure(pl, wiq, woq)[:, 0].double() * 2 * math.pi
                a_old = K.original_eval_closure(pl, wiq, woq)[:, 0].double() * 2 * math.pi
            row = {'au': au, 'av': av, 'mu': mu, 'phi': phi, 'albedo_original': float(a_old.mean()), 'albedo_corrected': float(a_new.mean()),
                   'se_corrected': float(a_new.std() / math.sqrt(Nq))}
            row['corrected_ok'] = row['albedo_corrected'] - 1 <= 4 * row['se_corrected']
            okb &= row['corrected_ok']
            rows.append(row)
b6['b_single_lobe_albedo'] = {'rows': rows, 'PASS': bool(okb),
                              'original_max_albedo': max(r_['albedo_original'] for r_ in rows),
                              'corrected_max_albedo': max(r_['albedo_corrected'] for r_ in rows)}
# (c) corrected Dr.Jit runtime vs corrected torch eval + WL80D T1-T3 on the corrected runtime
K.install()
RD.CD = CD81
outc = os.path.join(out, 'B6c')
os.makedirs(outc, exist_ok=True)
RD.tests(c80, outc)
tc = json.load(open(os.path.join(outc, 'tests.json')))
b6['c_corrected_runtime'] = {'T1': tc['T1'], 'T2': tc['T2'], 'T3': tc['T3'], 'T4_informational': tc['T4'],
                             'PASS': bool(tc['T1']['PASS'] and tc['T2']['PASS'] and tc['T3']['PASS'])}
b6['PASS'] = bool(b6['a_away_from_horizon']['PASS'] and b6['b_single_lobe_albedo']['PASS'] and b6['c_corrected_runtime']['PASS'])
b6['seconds'] = time.time() - t0
res['B6'] = b6
res['SAMPLER_CORRECTNESS'] = 'PASS' if (res['B1_B4']['PASS'] and res['B5']['PASS']) else 'FAIL'
res['HORIZON_FIX'] = 'PASS' if b6['PASS'] else 'FAIL'
res['continue'] = res['SAMPLER_CORRECTNESS'] == 'PASS' and res['HORIZON_FIX'] == 'PASS'
json.dump(res, open(os.path.join(out, 'stageB.json'), 'w'), indent=1, default=float)
print(json.dumps({k: res[k] for k in ['SAMPLER_CORRECTNESS', 'HORIZON_FIX', 'continue']}), flush=True)
os._exit(0)
