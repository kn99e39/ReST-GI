"""WL86 response supervision: random geometries of the family x random point lights -> geometry state, source encoding, target field.

  python gen86.py <root> <split> <start> <count>
Per geometry (file <root>/data/<split>/g<idx>.npz): geom / lights (json), patch tables, state rays (hit, dbin), and per lighting the
source encoding s [L, N, NB, 3] and the path-traced target field f [L, N, NB, 3] (frozen ER, 2..5 scattering events; val / dev: a second
independent half fb). Deterministic per (split seed, idx): shards can run concurrently.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt86 as PT  # noqa: E402
S = PT.S
TR = S.P['training']


def main():
    root, split, start, count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    seed = TR['splits'][split]['seed']
    spp = TR['splits'][split].get('target_spp', TR['target_spp']); nl = TR['lightings_per_geometry']
    off = TR['splits'][split].get('field_seed_offset', 0)
    out = os.path.join(root, 'data', split); os.makedirs(out, exist_ok=True)
    ctx = None
    t0 = time.time()
    for idx in range(start, start + count):
        fn = os.path.join(out, f'g{idx:05d}.npz')
        if os.path.exists(fn):
            continue
        rng = np.random.default_rng([seed, idx])
        geom = S.sample_geom(rng)
        lights = [S.sample_light(rng, geom) for _ in range(nl)]
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        hit, bh, _ = ctx.state_rays()
        s, f, fb = [], [], []
        for li, l in enumerate(lights):
            lt = PT.Lights(l)
            s.append(ctx.source(lt))
            sd = (seed % 1000) * 100000 + idx * 16 + li + off
            if split == 'train':
                f.append(ctx.field(lt, spp, sd, b=128)[0])
            else:
                f.append(ctx.field(lt, spp // 2, sd, b=128)[0]); fb.append(ctx.field(lt, spp // 2, sd + 8, b=128)[0])
        g = ctx.G
        d = dict(geom=json.dumps(geom), lights=json.dumps(lights), hit=hit, dbin=bh, s=np.stack(s), f=np.stack(f).astype(np.float16),
                 pc=g.PC.astype(np.float32), pn=g.PN.astype(np.float32), pt=g.PT.astype(np.float32), parea=g.PAREA.astype(np.float32),
                 pmat=g.PMAT.astype(np.int16))
        if fb:
            d['fb'] = np.stack(fb).astype(np.float16)
        np.savez_compressed(fn + '.tmp.npz', **d)
        os.replace(fn + '.tmp.npz', fn)
        if (idx - start) % 8 == 0:
            print(split, idx, round(time.time() - t0, 1), 's', flush=True)
    print('DONE', split, start, count, round(time.time() - t0, 1), 's', PT.gpu_mem(), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
