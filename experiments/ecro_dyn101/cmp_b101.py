"""Arm B equivalence against an arm A dump (same configs, same seeds). FREEZE env selects frozen / eager replay.
  python cmp_b101.py <case> <a_dump.npz> <out.json>
"""
import json, os, sys, time
import numpy as np
import rt101 as R

case, adump, out = sys.argv[1], sys.argv[2], sys.argv[3]
Z = np.load(adump, allow_pickle=False)
meta = json.loads(str(Z['meta']))
idx = [int(v) for v in meta['spec'].split(':')[1].split(',')]
rows = json.loads((R.ROOT100 / 'configs.json').read_text())[case]['test']
model = R.load_model(case)
cam = dict(R.C100.P['datasets'][case]['camera'], res=512)
rt = R.RT(rows[idx[0]]['geom'], case, model, rows[idx[0]]['lights'][0])
rep = {'freeze': R.FREEZE, 'states': []}


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


for n, i in enumerate(idx):
    moved, g_old = rt.set_geom(rows[i]['geom'])
    nd = rt.update_state(g_old, moved)
    rt.set_light(rows[i]['lights'][0])
    s = rt.source()
    B, _ = rt.respond(s)
    Bd = R.t2d(B.view(-1))
    r = {'config': i, 'dirty': nd, 'dirty_A': int(Z[f's{n}_dirty'])}
    hit = rt.hit.cpu().numpy(); dbin = rt.dbin.cpu().numpy(); seg = rt.seg.cpu().numpy()
    r['hit_equal'] = bool(np.array_equal(hit.astype(np.int16), Z[f's{n}_hit']))
    r['dbin_equal'] = bool(np.array_equal(dbin.astype(np.uint8), Z[f's{n}_dbin']))
    r['hit_mismatch'] = int((hit.astype(np.int16) != Z[f's{n}_hit']).sum())
    r['seg_max_abs'] = float(np.abs(seg - Z[f's{n}_seg']).max())
    r['src_rel_max'] = rel(s[0].cpu().numpy(), Z[f's{n}_src'])
    r['src_equal'] = bool(np.array_equal(s[0].cpu().numpy(), Z[f's{n}_src']))
    Bn = B.cpu().numpy()
    r['B_rel_max'] = rel(Bn, Z[f's{n}_B']); r['B_rel_l1'] = float(np.abs(Bn - Z[f's{n}_B']).sum() / Z[f's{n}_B'].sum())
    # full rebuild equality of the resident state
    h0, d0, s0 = rt.hit.clone(), rt.dbin.clone(), rt.seg.clone()
    rt.rebuild_state()
    r['selective_equals_full'] = bool(R.torch.equal(h0, rt.hit) and R.torch.equal(d0, rt.dbin))
    for spp in (1, 4):
        f = rt.frame(Bd, cam, 512, spp, 7000 + n)
        keys = sorted(k for k in Z.files if k.startswith(f's{n}_f{spp}_'))
        names = [k.split('_', 2)[2] for k in keys]
        assert names == sorted(names)
        fr = {}
        for k, v in zip(keys, f):
            a = Z[k]; b = np.asarray(v)
            fr[k.split('_', 2)[2]] = {'max_abs': float(np.abs(a - b).max()), 'mean_abs_over_mean': float(np.abs(a - b).mean() / max(np.abs(a).mean(), 1e-30)),
                                      'frac_pix_diff_gt_1e-4_rel': float((np.abs(a - b) > 1e-4 * np.abs(a).mean()).mean())}
        r[f'frame{spp}'] = fr
    rep['states'].append(r)
    print(json.dumps(r)[:1500], flush=True)
json.dump(rep, open(out, 'w'), indent=1)
os._exit(0)
