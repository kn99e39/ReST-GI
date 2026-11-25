"""WL114 Stage A (LR3): full-depth reference qualification with the stack that rendered the authenticated Ref-65 / Ref-7
(Mitsuba 3.6.4 stock `path`, independent sampler, box filter; TransGI reproduction `reference.render_passes`, seeds (11, 12),
1,024 spp per launch). The existing scene_transgi.xml is loaded read-only (nothing is written to the TransGI data tree).

  <TransGI venv python> lr3ref114.py OUT_DIR [steps...]     steps: auth ladder masks
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')
import drjit as dr  # noqa: E402
sys.path.insert(0, r'C:\Projects\TransGI_Reproduction')
from transgi import reference as R  # noqa: E402

XML = r'C:\Users\dna10\TransGI-Reproduction-data\scenes\living-room-3\scene_transgi.xml'
REF = Path(r'C:\Users\dna10\TransGI-Reproduction-data\runs\living_room_3\reference')
REF_SHA = {'static_V0.npz': '6262525144298f284558d790a253554a25c93827d1ca53a96961411715fcbf98',
           'static_V0_d7.npz': '904a3d9104c1f73db4823063116c8eba44755da629d46ab43f90549a6c4944ac'}
SCENE_LF_SHA = '62583fb9faa68c87109ab4bcfbe84e67b2ed48cb9dc898c1cfadc1f4a4c0969d'
SEEDS = (11, 12)
DEV = os.environ.get('WL114_DEV') == '1'
SPP_AUTH = 1024 if DEV else 65536
SPP_LADDER = 256 if DEV else 16384
LADDER = (2, 7, 13, 25, 49, 65, 97, 129, 257, -1)
GROUPS = {'rug': ['Rug'], 'cushions': ['Cushions_0001', 'Cushions_0002', 'Cushions_0003'], 'sofa': ['Sofa_0001', 'Sofa_0002', 'Sofa_0003'],
          'floor': ['Floor'], 'walls': ['Walls']}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    lf = hashlib.sha256(Path(XML).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    assert lf == SCENE_LF_SHA, ('scene hash', lf)
    scene = mi.load_file(XML, resx=640, resy=360)
    return scene, scene.sensors()[0]


def halves(scene, sensor, md, spp):
    a, ta = R.render_passes(scene, sensor, md, spp // 2, SEEDS[0])
    b, tb = R.render_passes(scene, sensor, md, spp // 2, SEEDS[1])
    return a, b, ta + tb


def main(out, steps):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    scene, sensor = load()
    info = {'mitsuba': mi.__version__, 'drjit': dr.__version__, 'scene_lf_sha256': SCENE_LF_SHA, 'xml': XML}
    if 'masks' in steps and not (out / 'lr3_masks.npz').exists():
        W, H = 640, 360
        ys, xs = np.mgrid[0:H, 0:W]
        pos = mi.Point2f(mi.Float(((xs + 0.5) / W).ravel().astype(np.float32)), mi.Float(((ys + 0.5) / H).ravel().astype(np.float32)))
        ray, _ = sensor.sample_ray(0.0, 0.5, pos, mi.Point2f(0.5, 0.5))
        si = scene.ray_intersect(mi.Ray3f(ray))
        shapes = list(scene.shapes()); ids = [s.id() for s in shapes]
        ptrs = np.array([int(dr.reinterpret_array(mi.UInt32, mi.ShapePtr(s))[0]) for s in shapes])
        hitp = np.asarray(dr.reinterpret_array(mi.UInt32, si.shape)); valid = np.asarray(si.is_valid())
        idx = np.full(hitp.shape, -1); lut = {p: i for i, p in enumerate(ptrs)}
        for i, p in enumerate(hitp):
            if valid[i]:
                idx[i] = lut.get(int(p), -1)
        idx = idx.reshape(H, W)
        masks = {}
        for g, members in GROUPS.items():
            masks[g] = np.isin(idx, [ids.index(m) for m in members if m in ids])
        masks['paint'] = np.isin(idx, [i for i, n in enumerate(ids) if n.startswith('WhitePaint')])
        np.savez_compressed(out / 'lr3_masks.npz', sid=idx, names=json.dumps(ids), **{f'mask_{k}': v for k, v in masks.items()})
        print('MASKS', {k: int(v.sum()) for k, v in masks.items()}, flush=True)
    if 'auth' in steps and not (out / 'auth_d65.npz').exists():
        ref_sha = {k: sha(REF / k) for k in REF_SHA}
        assert ref_sha == REF_SHA, ref_sha
        t0 = time.time()
        fa, fb, tf = halves(scene, sensor, 65, SPP_AUTH)
        da, db, td = halves(scene, sensor, 2, SPP_AUTH)
        np.savez_compressed(out / 'auth_d65.npz', full_A=fa, full_B=fb, di_A=da, di_B=db)
        json.dump({**info, 'spp': SPP_AUTH, 'max_depth': 65, 'seconds_full': tf, 'seconds_di': td, 'ref_sha256': ref_sha}, open(out / 'auth_d65.json', 'w'), indent=1)
        print('AUTH', round(time.time() - t0, 1), flush=True)
    if 'ladder' in steps:
        for md in LADDER:
            p = out / f'ladder_d{md}.npz'
            if p.exists():
                continue
            a, b, t = halves(scene, sensor, md, SPP_LADDER)
            np.savez_compressed(p, A=a, B=b)
            json.dump({**info, 'spp': SPP_LADDER, 'max_depth': md, 'seconds': t}, open(out / f'ladder_d{md}.json', 'w'), indent=1)
            print('LADDER', md, round(t, 1), np.asarray(0.5 * (a + b)).mean((0, 1)).tolist(), flush=True)
    (out / ('lr3_' + '_'.join(steps) + '.done')).write_text('DONE')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] or ['masks', 'auth', 'ladder'])
    sys.stdout.flush(); os._exit(0)
