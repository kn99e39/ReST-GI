"""WL99 focused tests (Stage 0 instruments).  python tests99.py <out_json>

T1 null body: S2 = S3 = SX = ES = 0 exactly (convex box).
T2 tag partition: at body pixels E = ES = 0; at external pixels B1 = S2 = S3 = SX = X = 0.
T3 oracle_self consistency: with the same seed, oracle_self's B1, X, E, ES equal cam's to float re-association (max |diff| <= 1e-5 x channel max;
   the two modes are different kernels, so bitwise equality is not expected); its S2 = S3 = SX = 0.
T4 certificate: an intruder box inside the groove fails; the frozen occluders pass.
T5 probe vs pixel: exact-direction probes at pixel-centre hits towards the camera agree with the pixel Sd (statistically, 20 px).
T6 rigid co-motion (same seed): Sd of the co-moved scene equals Sd of T0 within 3 sigma of the paired MC difference.
"""
import json, sys
import numpy as np
import pt99, wl99
import scene99 as S

out = {}
WL = {w['id']: w for w in wl99.WL}
R = 128
w = WL['A1_g90s1']
cf = wl99.configs(w)
ctx = pt99.Ctx(cf['T0'])


def rend(cfg, spp, seed, mode='cam', res=R):
    ctx.set_cfg(cfg)
    a, _ = ctx.render(cfg['camera'], res, spp, seed, mode=mode)
    return pt99.pack(a, spp)


# T1
c = wl99.configs(WL['C_box'])['T0']
r = rend(c, 64, 11)
out['T1_null_abs'] = float(sum(np.abs(r[k]).sum() for k in ('S2', 'S3', 'SX', 'ES')))
out['T1_pass'] = out['T1_null_abs'] == 0.0
# T2
r = rend(cf['T0'], 64, 12)
body = r['mb'] > 0.5; ext = r['mb'] < 0.5
out['T2_body_E'] = float(np.abs(r['E'][r['mb'] >= 1.0]).sum() + np.abs(r['ES'][r['mb'] >= 1.0]).sum())
out['T2_ext_body_tags'] = float(sum(np.abs(r[k][r['mb'] <= 0.0]).sum() for k in ('B1', 'S2', 'S3', 'SX', 'X')))
out['T2_pass'] = out['T2_body_E'] == 0.0 and out['T2_ext_body_tags'] == 0.0
# T3
a = rend(cf['T0'], 32, 13, 'cam'); o = rend(cf['T0'], 32, 13, 'oracle_self')
out['T3_maxdiff'] = {k: float(np.abs(a[k] - o[k]).max()) for k in ('B1', 'X', 'E', 'ES')}
out['T3_oracle_self_abs'] = float(sum(np.abs(o[k]).sum() for k in ('S2', 'S3', 'SX')))
out['T3_pass'] = all(out['T3_maxdiff'][k] <= 1e-5 * max(float(np.abs(a[k]).max()), 1e-30) for k in out['T3_maxdiff']) and out['T3_oracle_self_abs'] == 0.0
# T4
out['T4_T0'] = S.certificate(cf['T0']); out['T4_T1'] = S.certificate(cf['T1']); out['T4_T1x'] = S.certificate(cf['T1x'])
out['T4_pass'] = out['T4_T0'] and out['T4_T1'] and not out['T4_T1x']
# T5
ctx.set_cfg(cf['T0'])
rr = rend(cf['T0'], 1024, 14)
sd = rr['Sd'].mean(1)
cand = np.where((rr['mb'] >= 1.0) & (sd > np.quantile(sd[rr['mb'] >= 1.0], 0.5)))[0]
pix = cand[np.linspace(0, len(cand) - 1, 20).astype(int)]
y, wc, isb = ctx.pixel_hits(cf['T0']['camera'], R, pix)
pa = pt99.pack(ctx.probe(y, wc, 8192, 15), 8192)
ratio = pa['Sd'].mean(1) / np.maximum(sd[pix], 1e-30)
out['T5_ratio_median'] = float(np.median(ratio)); out['T5_ratio_iqr'] = [float(np.quantile(ratio, 0.25)), float(np.quantile(ratio, 0.75))]
out['T5_ok_frac'] = float(pa['ok'].mean())
out['T5_pass'] = bool(abs(out['T5_ratio_median'] - 1) < 0.1 and out['T5_ok_frac'] > 0.99)
# T6
a = rend(cf['T0'], 256, 16); b = rend(cf['co'], 256, 16)
m = (a['mb'] >= 1.0) & (b['mb'] >= 1.0) & (a['Sd'].mean(1) > 0)
d = (a['Sd'] - b['Sd'])[m]
a2 = rend(cf['T0'], 256, 17)
dn = (a['Sd'] - a2['Sd'])[m]                                       # MC difference scale between independent seeds
out['T6_rel_same_seed'] = float(np.sqrt((d ** 2).sum() / (a['Sd'][m] ** 2).sum()))
out['T6_rel_indep_seeds'] = float(np.sqrt((dn ** 2).sum() / (a['Sd'][m] ** 2).sum()))
out['T6_pass'] = out['T6_rel_same_seed'] <= out['T6_rel_indep_seeds']
out['all_pass'] = all(v for k, v in out.items() if k.endswith('_pass'))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out, indent=1))
