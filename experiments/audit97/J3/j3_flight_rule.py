"""J3 adjudication: re-derive WL95A FLIGHT_ATTRIBUTION under the frozen rule and alternatives,
and single-draw vs pooled X1 statistics, from the stored deliv.npz (read-only). numpy only."""
import json, sys, hashlib
import numpy as np
P = sys.argv[1]; OUT = sys.argv[2]
z = np.load(P)
ra, rb = z['ref_a'], z['ref_b']; ref = 0.5 * (ra + rb)
mA = np.zeros(len(ref), bool); mA[1130:1184] = True
rs = ref[mA].sum()
L1 = lambda m: float(np.abs(m[mA] - ref[mA]).sum() / rs)
RAT = lambda m: float(m[mA].sum() / rs)
n0 = float(0.5 * np.abs(ra[mA] - rb[mA]).sum() / rs)
Dr = {k: np.asarray(z['draws|' + k]) for k in ('A4', 'M64_spool', 'q256', 'X1c', 'X1')}
def jk_diff(k1, k2, idx=None):
    X1_, X2_ = Dr[k1], Dr[k2]
    if idx is not None: X1_, X2_ = X1_[idx], X2_[idx]
    n = len(X1_); f = lambda a, b: L1(a) - L1(b)
    loo = np.array([f((X1_.sum(0) - X1_[i]) / (n - 1), (X2_.sum(0) - X2_[i]) / (n - 1)) for i in range(n)])
    return f(X1_.mean(0), X2_.mean(0)), float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))
def jk_one(k, idx=None):
    X = Dr[k] if idx is None else Dr[k][idx]; n = len(X)
    loo = np.array([L1((X.sum(0) - X[i]) / (n - 1)) for i in range(n)])
    return L1(X.mean(0)), float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))
def label(E0, Rs, se_s, Ra, se_a, EX1):
    cnt = lambda r, se: bool(r >= 0.25 * E0 and r > 2 * se)
    s, a = cnt(Rs, se_s), cnt(Ra, se_a)
    if EX1 > 0.5 * E0: return 'INCONCLUSIVE'
    if s and a: return 'MIXED'
    if s and not a and Rs >= 0.5 * E0: return 'SAMPLING_DOMINANT'
    if a and not s: return 'ANGULAR_PROJECTION_EVIDENCED'
    return 'INCONCLUSIVE'
o = {'n0': n0, 'sha256_16': hashlib.sha256(open(P, 'rb').read()).hexdigest()[:16]}
EM, seM = jk_one('M64_spool'); EM -= n0
Ra, seA = jk_diff('M64_spool', 'X1c'); Rp, seP = jk_diff('X1c', 'X1')
EX1 = jk_one('X1')[0] - n0
E0dep = L1(z['dep|A4']) - n0
o['frozen'] = dict(E0=E0dep, Rs=E0dep - EM, Rs_se=seM, Ra=Ra, Ra_se=seA, Rp=Rp, Rp_se=seP, EX1=EX1, E_M64_pool=EM,
                   label=label(E0dep, E0dep - EM, seM, Ra, seA, EX1))
sA4 = np.array([L1(x) for x in Dr['A4']]) - n0
alt = {}
for nm, E0 in (('mean_single_A4', float(sA4.mean())), ('median_single_A4', float(np.median(sA4)))):
    alt[nm] = dict(E0=E0, Rs=E0 - EM, label=label(E0, E0 - EM, seM, Ra, seA, EX1))
# each stored draw as 'deployed', with a leave-one-out pool (draw excluded) -> no self-inclusion
cnts = {}; rsfrac = []
for i in range(64):
    idx = np.array([j for j in range(64) if j != i])
    E0 = L1(Dr['A4'][i]) - n0
    EMi, seMi = jk_one('M64_spool', idx); EMi -= n0
    Rai, seAi = jk_diff('M64_spool', 'X1c', idx)
    EX1i = jk_one('X1', idx)[0] - n0
    lb = label(E0, E0 - EMi, seMi, Rai, seAi, EX1i); cnts[lb] = cnts.get(lb, 0) + 1; rsfrac.append((E0 - EMi) / E0)
alt['each_draw_as_deployed_LOO'] = dict(counts=cnts, Rs_over_E0_median=float(np.median(rsfrac)))
# definition-consistent sampling term: same estimator single-draw mean error minus its pooled error
sM = np.array([L1(x) for x in Dr['M64_spool']]) - n0
sX1 = np.array([L1(x) for x in Dr['X1']]) - n0
alt['sampling_term_same_arm'] = dict(M64_single_mean=float(sM.mean()), M64_pool=EM, Rs_like=float(sM.mean() - EM))
o['alternatives'] = alt
# single-draw paired comparison X1 vs binned (same draw index; relations shared per draw?)
for b in ('A4', 'M64_spool'):
    sb = np.array([L1(x) for x in Dr[b]]) - n0
    d = sX1 - sb
    o[f'single_X1_minus_{b}'] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / 8), frac_X1_better=float((d < 0).mean()),
                                     X1_mean=float(sX1.mean() + n0), X1_median=float(np.median(sX1) + n0), b_mean=float(sb.mean() + n0), b_median=float(np.median(sb) + n0))
o['dep'] = {k: dict(ratio=RAT(z['dep|' + k]), L1=L1(z['dep|' + k])) for k in ('A4', 'M64_spool', 'X1c', 'X1')}
# budget curve: L1 of the mean of D random draws (200 random subsets)
rng = np.random.default_rng(97003); curve = {}
for k in ('M64_spool', 'X1'):
    curve[k] = {}
    for D in (1, 2, 4, 8, 16, 32, 64):
        v = [L1(Dr[k][rng.choice(64, D, replace=False)].mean(0)) for _ in range(200 if D < 64 else 1)]
        curve[k][D] = dict(mean=float(np.mean(v)), p10=float(np.quantile(v, .1)), p90=float(np.quantile(v, .9)))
o['budget_curve_L1'] = curve
# fraction of the A4 pooled bias closed vs reference, at expectation
o['pooled'] = {k: dict(ratio=RAT(Dr[k].mean(0)), L1=L1(Dr[k].mean(0))) for k in Dr}
json.dump(o, open(OUT, 'w'), indent=1)
print(json.dumps(o['frozen'])); print(json.dumps(alt)); 
for k in ('single_X1_minus_A4', 'single_X1_minus_M64_spool'): print(k, json.dumps(o[k]))
print(json.dumps({k: {D: round(v['mean'], 3) for D, v in c.items()} for k, c in curve.items()}))
