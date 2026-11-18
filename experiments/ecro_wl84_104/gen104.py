"""WL104 ECRO dataset for one WL84 material on the fixed WL84 mesh scene (WL102 split sizes and light rule; physical targets with the
learned-ER closure on sphere and cube, Lambert 0.8 floor / wall).

  python gen104.py <tag> <out_root> [split ...]          splits: train (128 x 4 lights, 64 spp), val (16 x 4, 2 x 128 spp),
                                                         test (1 item = the two evaluation lights L1, L2, 2 x 256 spp)
Light rule (frozen, WL102 adapted): white point lights of the ECRO fixture intensity, uniform in the room-side box x, y, z in
[-3, 3] x [-3, 1.5] x [0, 3] (in front of the one-sided back wall), >= 0.15 m from any
surface along 26 directions, not inside a closed mesh; rejected within 0.3 m of the WL84 lights L1 and L2 (held out). The same light
draws are used for every material (seeds independent of the material). Targets are linear in the source, so the training intensity does
not restrict the evaluation intensity (model86.apply is linear in s).
"""
import json
import os
import sys
import time

import numpy as np

import common104 as C
import rt104 as RT
import gen102 as G2           # unchanged WL102 helpers: free_space, sample_lights

PT, S, mi, dr = RT.PT, RT.P.S, RT.mi, RT.dr
SPLITS = {'train': {'items': 128, 'lights': 4, 'spp': 64, 'halves': 1, 'seed': 104100},
          'val': {'items': 16, 'lights': 4, 'spp': 128, 'halves': 2, 'seed': 104200},
          'test': {'items': 1, 'lights': 2, 'spp': 256, 'halves': 2, 'seed': 104300}}
INTENSITY = S.FX['light_intensity']
BOX = (np.array([-3.0, -3.0, 0.0]), np.array([3.0, 1.5, 3.0]))   # room side of the back wall (the scene bbox reaches y = 3 behind it)


def sample_lights(ctx, rng, count, avoid):
    """gen102.sample_lights with the sampling box restricted to the room side of the wall (pre-freeze development finding: lights
    behind the one-sided wall give all-zero field targets and a 0 / 0 loss denominator)."""
    out = []
    while len(out) < count:
        c = rng.uniform(BOX[0], BOX[1], (4096, 3))
        ok = G2.free_space(ctx, c)
        for p in c[ok]:
            if all(np.linalg.norm(p - a) >= G2.EXCL for a in avoid + out):
                out.append(p)
                if len(out) == count:
                    break
    return [np.asarray(p).tolist() for p in out]
if C.SMOKE:
    SPLITS['train']['items'], SPLITS['val']['items'] = 4, 2


def main():
    tag, root = sys.argv[1], sys.argv[2]
    splits = sys.argv[3:] or ['train', 'val', 'test']
    os.makedirs(root, exist_ok=True)
    desc, state, h = RT.register(tag)
    assert h == C.STATE_SHA[tag], 'STOP_STATE_HASH'
    ctx = RT.Ctx(RT.scene(tag, 'er'))
    gfile = os.path.join(root, 'geometry.npz')
    if not os.path.exists(gfile):
        t0 = time.perf_counter()
        hit, dbin, _ = ctx.state_rays()
        t_state = time.perf_counter() - t0
        rows = RT.encoder_rows(ctx, state)
        held = [np.asarray(C.LIGHTS[1][0], float), np.asarray(C.LIGHTS[2][0], float)]
        lights = {}
        for sp, c in SPLITS.items():
            if sp == 'test':
                lights[sp] = [[C.LIGHTS[1][0], C.LIGHTS[2][0]]]
            else:
                L = sample_lights(ctx, np.random.default_rng(c['seed']), c['items'] * c['lights'], held)
                lights[sp] = [L[i * c['lights']:(i + 1) * c['lights']] for i in range(c['items'])]
        g = ctx.G
        np.savez_compressed(gfile, pc=g.PC, pn=g.PN, pt=g.PT, parea=g.PAREA, pmat=g.PMAT, pobj=g.POBJ, pside=g.PSIDE, hit=hit, dbin=dbin,
                            rows=rows, lights=json.dumps(lights), state_seconds=t_state)
        C.write_json(os.path.join(root, 'geometry.json'), {'tag': tag, 'state_sha256': h, 'N': int(g.N), 'state_rays_s': t_state,
                                                           'light_intensity': INTENSITY, 'held_out_lights': [C.LIGHTS[1], C.LIGHTS[2]],
                                                           'layout': g.summary(), 'objects': [m[1] for m in ctx.meshes]})
    z = np.load(gfile)
    lights = json.loads(str(z['lights']))
    for sp in splits:
        c = SPLITS[sp]
        od = os.path.join(root, sp); os.makedirs(od, exist_ok=True)
        t0 = time.perf_counter(); times = []
        for i, ls in enumerate(lights[sp]):
            fn = os.path.join(od, f'g{i:03d}.npz')
            if os.path.exists(fn):
                continue
            ti = time.perf_counter()
            s, f2, fh, f2b, fhb = [], [], [], [], []
            for j, pos in enumerate(ls):
                L = PT.Lights([{'pos': pos, 'rgb': [INTENSITY] * 3}])
                s.append(ctx.source(L))
                seed = c['seed'] * 1000 + i * 10 + j
                a, b, _ = ctx.field_components(L, c['spp'], seed, b=C.BATCH['field'])
                f2.append(a); fh.append(b)
                if c['halves'] == 2:
                    a, b, _ = ctx.field_components(L, c['spp'], seed + 500000, b=C.BATCH['field'])
                    f2b.append(a); fhb.append(b)
            d = {'lights': json.dumps(ls), 's': np.stack(s), 'f2': np.stack(f2), 'fh': np.stack(fh)}
            if f2b:
                d.update(f2b=np.stack(f2b), fhb=np.stack(fhb))
            assert all(np.isfinite(v).all() for v in d.values() if isinstance(v, np.ndarray)), 'STOP_NONFINITE_TARGET'
            assert all(float((d['f2'][j] + d['fh'][j]).mean()) > 0 for j in range(len(ls))), 'STOP_ZERO_TARGET_LIGHT'
            np.savez(fn + '.tmp.npz', **d); os.replace(fn + '.tmp.npz', fn)
            times.append(time.perf_counter() - ti)
            print('GEN', tag, sp, i, round(times[-1], 2), flush=True)
        C.write_json(os.path.join(root, f'gen_{sp}.json'), {'split': sp, **c, 'seconds': time.perf_counter() - t0, 'item_seconds': times})
    os._exit(0)


if __name__ == '__main__':
    main()
