"""WL102 gate A2: the ECRO physical tracer (pt102, D = 6, the scene's area light, Mitsuba BSDFs) vs the Mitsuba Ref-7 reference.

  python gate_a2.py <scene.xml> <ref7.npz> <out_dir> [spp_per_half=1024]
Criteria (protocol wl102.json gates.A2): GI and FULL mean within 3 % per channel, and image RMSE <= 1.5 x the combined noise
sqrt(s_ours^2 + s_ref^2), each s = half-difference RMSE / 2, at the evaluated spp.
"""
import json, os, sys, time
import numpy as np
import pt102 as P

xml, ref7, out = sys.argv[1], sys.argv[2], sys.argv[3]
spp = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
os.makedirs(out, exist_ok=True)
ctx = P.Ctx(P.load_scene(xml))
L = P.SceneLights(64)
halves = []
for h, seed in enumerate((102001, 102002)):
    acc, T = ctx.render(L, spp, seed, 'cam', b=8)
    W, H = T['res']
    o = P.pack(acc, W, H, spp)
    halves.append({'di': o['e1'], 'gi': o['e2'] + o['h'], 'full': o['e1'] + o['e2'] + o['h'], 'timing': T})
r = np.load(ref7)
res = {'spp_per_half': spp, 'seeds': [102001, 102002], 'N': ctx.N, 'comp': {}}
for k in ('gi', 'full', 'di'):
    a, b = halves[0][k], halves[1][k]
    ours = 0.5 * (a + b)
    so = float(np.sqrt(((a - b) ** 2).mean()) / 2)
    sr = float(np.sqrt(((r[k + '_A'] - r[k + '_B']) ** 2).mean()) / 2)
    rmse = float(np.sqrt(((ours - r[k]) ** 2).mean()))
    comb = float(np.hypot(so, sr))
    mo, mr = ours.mean((0, 1)), r[k].mean((0, 1))
    rel = ((mo - mr) / mr).tolist()
    res['comp'][k] = {'mean_ours': mo.tolist(), 'mean_ref': mr.tolist(), 'mean_rel_diff': rel, 'rmse': rmse, 'noise_ours': so,
                      'noise_ref': sr, 'combined_noise': comb, 'rmse_over_noise': rmse / comb,
                      'pass': bool(max(abs(x) for x in rel) <= 0.03 and rmse <= 1.5 * comb)}
    np.save(os.path.join(out, f'a2_{k}.npy'), ours)
res['A2_pass'] = bool(res['comp']['gi']['pass'] and res['comp']['full']['pass'])
res['timing'] = [h['timing'] for h in halves]
json.dump(res, open(os.path.join(out, 'gate_a2.json'), 'w'), indent=1)
print(json.dumps({k: {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in d.items() if kk in ('mean_rel_diff', 'rmse', 'combined_noise', 'rmse_over_noise', 'pass')}
                  for k, d in res['comp'].items()}), 'A2_pass', res['A2_pass'])
