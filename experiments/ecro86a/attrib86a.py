"""WL86A Stages C / D: input-information sufficiency and learning attribution on the frozen WL86 ECRO v0 (diagnostic; nothing is trained).

  python attrib86a.py fields <root86> <out> <cfg> [--dev]     controlled incident fields for one configuration (path-traced tables)
  python attrib86a.py beams  <root86> <out> [--dev]           identical-encoding source pairs (collimated patch beams) on the E2 T1 geometry
  python attrib86a.py model  <root86> <out>                   frozen operators: source-estimator effect, responses to the beam encodings, field error by class
  python attrib86a.py render <root86> <out> <cfg> <table> <rep> [--dev]   camera render of a diagnostic table (WL86 ecro estimator, oracle variant)
"""
import glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt86a as A  # noqa: E402
PT, S = A.PT, A.S
mi, dr = PT.mi, PT.dr
FX = PT.FX; NB = PT.NB
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86a.json')))
DEV = '--dev' in sys.argv
SPP = 64 if DEV else P['fields']['spp_per_half']
CFGS = P['configs']
VARIANTS = {'TRUE': ('true', ()), 'ANG_PANEL': ('ang', (1,)), 'ANG_SPHERE': ('ang', (2,)), 'ANG_BROAD': ('ang', (0, 3, 4, 5)),
            'ENC_PANEL': ('enc', (1,)), 'ENC_ALL': ('enc', (0, 1, 2, 3, 4, 5)), 'ENC16_ALL': ('enc16', (0, 1, 2, 3, 4, 5))}


def fields(root, out, cfg):
    geom, ls = PT.config(cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    s16 = ctx.source(lt); shq = A.source_hq(ctx, lt, P['source_hq_points'])
    os.makedirs(os.path.join(out, 'results', 'fields'), exist_ok=True)
    np.savez_compressed(os.path.join(out, 'results', 'fields', f'S_{cfg}.npz'), s16=s16, shq=shq)
    meta = {}
    for i, (name, (mode, mats)) in enumerate(VARIANTS.items()):
        t0 = time.time()
        src = A.Src(lt, 'enc' if mode.startswith('enc') else mode, mats, enc=(shq if mode == 'enc' else s16) if mode.startswith('enc') else None)
        fa = A.field_src(ctx, src, SPP, 86900 + 10 * i); fb = A.field_src(ctx, src, SPP, 86905 + 10 * i)
        np.savez_compressed(os.path.join(out, 'results', 'fields', f'D_{name}_{cfg}.npz'), fa=fa, fb=fb)
        meta[name] = {'seconds': time.time() - t0, 'enc_cover': getattr(src, 'enc_cover', None)}
        print(cfg, name, round(time.time() - t0, 1), 's', meta[name]['enc_cover'], flush=True)
    json.dump(meta, open(os.path.join(out, 'results', 'fields', f'meta_{cfg}.json'), 'w'), indent=1)


def beam_set(ctx, lpos):
    """Probe patches and beam directions (world) chosen by rule from the geometry and the light position only."""
    G = ctx.G
    pick = lambda mask, target: int(np.nonzero(mask)[0][np.argmin(np.linalg.norm(G.PC[mask] - np.array(target), axis=1))])
    patches = {'panel_S1': pick(G.PSID == 5, FX['panel']['center']),
               'sphere_S2': int(np.nonzero(G.PSID == S.SID_SPHERE)[0][np.argmax(G.PN[G.PSID == S.SID_SPHERE] @ (np.array(lpos) - G.SC) / np.linalg.norm(np.array(lpos) - G.SC))]),
               'floor_B': pick(G.PSID == 0, [0.6, -0.9, 0.0]), 'wall_lambert': pick(G.PSID == 2, [0.6, 2.0, 1.6])}
    E0 = P['beams']['irradiance']
    out = {}
    for name, j in patches.items():
        wl = np.array(lpos) - G.PC[j]; wl /= np.linalg.norm(wl)
        loc = np.array([wl @ G.PT[j], wl @ G.PB[j], wl @ G.PN[j]])
        iz = int(np.clip(math.floor(loc[2] * S.NZ), 0, S.NZ - 1)); ip = int(np.clip(math.floor((math.atan2(loc[1], loc[0]) + math.pi) * S.NPHI / (2 * math.pi)), 0, S.NPHI - 1))

        def d(a, b, dz=0):
            z = (iz + dz + a) / S.NZ; phi = (ip + b) / S.NPHI * 2 * math.pi - math.pi
            r = math.sqrt(max(1 - z * z, 0))
            l = np.array([r * math.cos(phi), r * math.sin(phi), z])
            return {'pid': j, 'w': (G.PT[j] * l[0] + G.PB[j] * l[1] + G.PN[j] * l[2]).tolist(), 'E': E0 / z, 'cos': z}
        oz = -1 if iz == S.NZ - 1 else 1
        out[name] = {'pid': j, 'bin': iz * S.NPHI + ip,
                     'polar_a': d(0.2, 0.5), 'polar_b': d(0.8, 0.5), 'azim_a': d(0.5, 0.2), 'azim_b': d(0.5, 0.8), 'centre': d(0.5, 0.5),
                     'other_bin': d(0.5, 0.5, oz),
                     'half_a': {**d(0.5, 0.5), 'half': 0, 'E': 2 * d(0.5, 0.5)['E']}, 'half_b': {**d(0.5, 0.5), 'half': 1, 'E': 2 * d(0.5, 0.5)['E']}}
        if G.PSID[j] == S.SID_SPHERE:                       # half-patch beams are defined on planar patches only
            del out[name]['half_a'], out[name]['half_b']
    return out


def beams(root, out):
    geom, ls = PT.config('E2_T1')
    ctx = PT.Ctx(geom)
    bs = beam_set(ctx, FX['light_sets'][ls][0]['pos'])
    os.makedirs(os.path.join(out, 'results', 'beams'), exist_ok=True)
    res = {}
    k = 0
    for pname, d in bs.items():
        res[pname] = {'pid': d['pid'], 'bin': d['bin'], 'beams': {}}
        for bname, bm in d.items():
            if not isinstance(bm, dict):
                continue
            s, bn, cs = A.beam_encoding(ctx, bm)
            src = A.Src(None, 'true', (), beams=[bm])
            sb = 64 if DEV else P['beams']['spp_per_half']
            fa = A.field_src(ctx, src, sb, 87100 + 10 * k); fb = A.field_src(ctx, src, sb, 87105 + 10 * k); k += 1
            np.savez_compressed(os.path.join(out, 'results', 'beams', f'{pname}_{bname}.npz'), fa=fa, fb=fb, s=s)
            res[pname]['beams'][bname] = {'beam': bm, 'bin': bn, 'cos': cs, 'encoding_sum': float(s.sum()), 'flux': float(((fa + fb) * 0.5).sum())}
            print(pname, bname, bn, round(float(s.sum()), 4), flush=True)
    json.dump(res, open(os.path.join(out, 'results', 'beams', 'beams.json'), 'w'), indent=1)


def model(root, out):
    import torch
    sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
    import ecro86 as EC
    import train86 as TR
    res = {'estimator': {}, 'beams': {}, 'field_by_class': {}}
    cls_names = ['B_floor', 'S1_panel', 'S2_sphere', 'lambert_walls', 'lambert_boxA', 'lambert_boxB']
    for sd in P['seeds']:
        mdl = EC.load_model(root, sd)
        # ---- source estimator: deployed 16 points vs high-quality encoding, same frozen operator and state
        for cfg in CFGS:
            geom, ls = PT.config(cfg)
            ctx = PT.Ctx(geom)
            st = EC.build_state(ctx)
            z = np.load(os.path.join(out, 'results', 'fields', f'S_{cfg}.npz'))
            B16, f = EC.respond(mdl, st, z['s16']); Bhq, _ = EC.respond(mdl, st, z['shq'], factors=f)
            np.savez_compressed(os.path.join(out, 'results', 'fields', f'M_{sd}_{cfg}.npz'), B16=B16, Bhq=Bhq)
            res['estimator'][f'{sd}:{cfg}'] = {'encoding_rel_l1_16_vs_hq': float(np.abs(z['s16'] - z['shq']).sum() / z['shq'].sum()),
                                                'table_rel_l1_16_vs_hq': float(np.abs(B16 - Bhq).sum() / Bhq.sum())}
            if cfg == 'E2_T1':
                bj = json.load(open(os.path.join(out, 'results', 'beams', 'beams.json')))
                for pname, d in bj.items():
                    for bname in d['beams']:
                        zb = np.load(os.path.join(out, 'results', 'beams', f'{pname}_{bname}.npz'))
                        Bm, _ = EC.respond(mdl, st, zb['s'], factors=f)
                        np.save(os.path.join(out, 'results', 'beams', f'M_{sd}_{pname}_{bname}.npy'), Bm)
        # ---- field error of the frozen operator by table-patch material class, training subset (seen geometries) vs validation
        for split in ('trainhq', 'val'):
            items = TR.load(root, split)
            num = np.zeros(6); den = np.zeros(6); noise = np.zeros(6); sq = np.zeros(6); sqd = np.zeros(6)
            with torch.no_grad():
                for it in items:
                    p = mdl(it.gs(), torch.as_tensor(it.s, device=EC.dev)).cpu().numpy()
                    fa = it.f.astype(np.float32); fb = it.fb.astype(np.float32); t = 0.5 * (fa + fb)
                    for c in range(6):
                        m = it.tab['pmat'] == c
                        num[c] += np.abs(p[:, m] - t[:, m]).sum(); den[c] += t[:, m].sum(); noise[c] += 0.5 * np.abs(fa[:, m] - fb[:, m]).sum()
                        sq[c] += ((p[:, m] - fa[:, m]) * (p[:, m] - fb[:, m])).sum(); sqd[c] += (fa[:, m] * fb[:, m]).sum()
            res['field_by_class'][f'{sd}:{split}'] = {cls_names[c]: {'rel_l1': float(num[c] / den[c]), 'target_noise_rel_l1': float(noise[c] / den[c]),
                                                                        'rel_rms_debiased': float(math.sqrt(max(sq[c], 0) / sqd[c]))} for c in range(6)}
        print(sd, json.dumps(res['field_by_class'][f'{sd}:val']), flush=True)
    json.dump(res, open(os.path.join(out, 'results', 'model.json'), 'w'), indent=1)


def render(root, out, cfg, table, rep):
    sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
    import ecro86 as EC
    spp = 64 if DEV else P['render']['spp']
    o, B, T_ = EC.render_cfg(out, cfg, f'oracle:D_{table}_{cfg}', 'oracle', P['render']['seed_base'] + int(rep), spp)
    os.makedirs(os.path.join(out, 'results', 'renders'), exist_ok=True)
    np.savez_compressed(os.path.join(out, 'results', 'renders', f'DR_{table}_{cfg}_r{rep}.npz'), **o)
    print('render', table, cfg, rep, json.dumps(T_), flush=True)


if __name__ == '__main__':
    cmd, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    if cmd == 'fields':
        fields(root, out, sys.argv[4])
    elif cmd == 'beams':
        beams(root, out)
    elif cmd == 'model':
        model(root, out)
    elif cmd == 'render':
        render(root, out, sys.argv[4], sys.argv[5], sys.argv[6])
    os._exit(0)
