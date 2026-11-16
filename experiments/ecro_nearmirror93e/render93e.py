"""Isolated WL93E reference instrumentation; never changes the frozen source files.

python render93e.py pre ROOT
python render93e.py render ROOT MATERIAL VARIANT RES SPP SEED
python render93e.py model ROOT VARIANT RES SPP SEED CHECKER_ROOT
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import textwrap
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro_local93d' / 'addendum_checker'))
os.environ['CK_SCENE'] = str(HERE / 'scene.json')
import checker93d as CK

ALPHA = {'G075': .075, 'G040': .04, 'G010': .01}
RHO = .85


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(o, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(o, indent=2))


def ggx_state(alpha):
    # One outer lobe, constant Fresnel (F0=1), neutral coefficient .85.
    # Other lobes have zero coefficients AND zero sampling probabilities.
    st = CK.lambert_state(0.)
    st[24:38] = alpha
    st[38:45] = 1.
    for c in range(3):
        st[c * 8 + 1] = RHO
    st[45:53] = 0.
    st[46] = 1.
    return st


def setup(root, material, variant, res):
    fx, PT, S, FX, ctx, lights = CK.ctx_of(variant, str(root))
    cam = dict(FX['camera'], res=int(res))
    original = ctx.mats.states.copy()
    if material in ALPHA:
        ctx.mats.states[1] = ggx_state(ALPHA[material])
    elif material not in ('S1', 'MIRROR'):
        raise ValueError(material)
    # Mirror is a real discrete event, not an epsilon-width GGX approximation.
    if material == 'MIRROR':
        src = textwrap.dedent(inspect.getsource(CK._INST['orig_trace']))
        # Use the same texture lookup as checker93d (the patched trace isn't
        # introspectable through inspect because it was compiled at run time).
        needle = '        mid = dr.gather(U, self.SMAT, sid)\n'
        assert src.count(needle) == 1
        src = src.replace(needle, needle +
            '        _pid, _ = self.patch_bin(si, sid, -ray.d)\n'
            '        mid = dr.select(active, dr.gather(U, self.PMATU, _pid, active), mid)\n')
        needle = '                    f = cl.eval(wi, wo_s, vis)'
        assert src.count(needle) == 1
        src = src.replace(needle, needle + '\n                    f = dr.select(dr.eq(mid, 1), 0.0, f)')
        needle = '            wo, pdf, w, valid = cl.sample(wi, u1, u2, active)'
        assert src.count(needle) == 1
        src = src.replace(needle, needle +
            '\n            _mirror = active & dr.eq(mid, 1)'
            '\n            wo = dr.select(_mirror, mi.Vector3f(-wi.x, -wi.y, wi.z), wo)'
            '\n            w = dr.select(_mirror, mi.Color3f(0.85), w)'
            '\n            valid = dr.select(_mirror, wi.z > 1e-6, valid)')
        scope = dict(PT.__dict__)
        exec(compile(src, 'pt86.Ctx.trace[WL93E delta mirror]', 'exec'), scope)
        PT.Ctx.trace = scope['trace']
        patch_hash = hashlib.sha256(src.encode()).hexdigest()
    else:
        patch_hash = CK._INST['patch_sha']
    assert np.array_equal(original[np.arange(len(original)) != 1],
                          ctx.mats.states[np.arange(len(original)) != 1])
    return PT, S, ctx, lights, cam, original, patch_hash


def independent_map(PT, ctx, cam, gb):
    """Independent NumPy mirror-to-OBB face/cell mapping and point-light e2.

    No renderer material/bin lookup is used to compute the expected checker.
    """
    oo, fw, ri, up, scale = PT.cam_basis(cam)
    n = cam['res']
    yy, xx = np.mgrid[:n, :n]
    d = fw + scale * ((2*(xx+.5)/n-1)[..., None]*ri +
                     (1-2*(yy+.5)/n)[..., None]*up)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    d[..., 0] *= -1
    box = CK.SC['geom']['A']
    ax = np.stack(PT.S.box_axes(box['yaw']), axis=1)
    ori = (gb['p'] + np.array([1e-4, 0, 0]) - box['c']) @ ax
    dire = d @ ax
    h = np.asarray(box['h'])
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = (-h-ori)/dire, (h-ori)/dire
        near, far = np.minimum(a, b), np.maximum(a, b)
        t = np.max(near, axis=-1)
    valid = gb['panel'] & (t > 0) & (t <= np.min(far, axis=-1))
    pos = ori + t[..., None]*dire
    face_axis = np.argmax(near, axis=-1)
    sign = np.take_along_axis(pos, face_axis[..., None], axis=-1)[..., 0] > 0
    face = 2*face_axis + sign
    pairs = {(0, 1): (1, 2), (0, 0): (2, 1), (1, 1): (2, 0),
             (1, 0): (0, 2), (2, 1): (0, 1), (2, 0): (1, 0)}
    parity = np.zeros((n,n), bool)
    edge = np.full((n,n), np.inf)
    for (axis, sg), (a1, a2) in pairs.items():
        sel = valid & (face == 2*axis+sg)
        uv = np.stack([(pos[..., a1]/h[a1]+1)*1.5,
                       (pos[..., a2]/h[a2]+1)*1.5], -1)
        ij = np.clip(np.floor(uv).astype(int), 0, 2)
        parity[sel] = ((ij[..., 0]+ij[..., 1]) % 2 == 1)[sel]
        edge[sel] = np.min(np.abs(uv-np.round(uv)), axis=-1)[sel]
    pid = np.clip(gb['mpid'], 0, ctx.N-1)
    expected_mid = np.where(parity, 7, 6)
    roi, direct, _ = CK.target_masks(ctx.G, gb)
    safe = roi & valid & (edge > .01)
    mismatch = int(np.sum(ctx.G.PMAT[pid][safe] != expected_mid[safe]))
    point = gb['p'] + np.array([1e-4,0,0]) + t[..., None]*d
    normal_local = np.zeros_like(point)
    np.put_along_axis(normal_local, face_axis[..., None],
                      np.where(sign, 1., -1.)[..., None], axis=-1)
    normal = normal_local @ ax.T
    light = CK.SC['lights'][0]
    dl = np.asarray(light['pos']) - point
    dist2 = (dl*dl).sum(-1)
    cos = np.maximum((normal*dl).sum(-1)/np.sqrt(dist2), 0)
    expected_e2 = RHO*np.where(parity, .8, .3)*cos/np.pi/dist2*float(light['rgb'][0])
    return {'roi': roi, 'direct': direct, 'parity': parity, 'face': face,
            'safe': safe, 'edge': edge, 'expected_e2': expected_e2,
            'valid': valid, 'mismatches': mismatch}


def pre(root):
    PT, S, ctx, lights, cam, orig, ph = setup(root, 'S1', 'bx38', 512)
    gb = CK.gbuf(PT, ctx, cam)
    im = independent_map(PT, ctx, cam, gb)
    # Original untextured renderer versus inherited checker patch: regression.
    orig_trace, patched_trace = CK._INST['orig_trace'], PT.Ctx.trace
    pm = ctx.G.PMAT.copy()
    CK.set_pmat(PT, ctx, CK.default_pmat(ctx.G))
    PT.Ctx.trace = orig_trace
    a, _ = ctx.render(lights, 'ER', cam, 128, 8, 931480, 'cam', b=2)
    PT.Ctx.trace = patched_trace
    b, _ = ctx.render(lights, 'ER', cam, 128, 8, 931480, 'cam', b=2)
    err = max(float(np.max(np.abs(a[k]-b[k]))/max(np.max(np.abs(a[k])),1e-20)) for k in a)
    CK.set_pmat(PT, ctx, pm)
    # Check the actual eval/sample path reacts to alpha (not merely JSON).
    mi, dr = PT.mi, PT.dr
    theta = np.deg2rad(40.)
    wi = mi.Vector3f(np.sin(theta),0.,np.cos(theta))
    wo = mi.Vector3f(-np.sin(theta),0.,np.cos(theta))
    peaks, widths = {}, {}
    rng = mi.PCG32(size=65536, initstate=mi.UInt64(931481))
    u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
    for name, alpha in ALPHA.items():
        cl = PT.CD.DrClosure83('ER', ggx_state(alpha))
        ev = cl.eval(wi, wo)
        sw, pdf, weight, ok = cl.sample(wi, u1, u2)
        dr.eval(ev, sw, ok)
        peaks[name] = float(np.asarray(ev)[0,0])
        angular = np.arccos(np.clip(np.asarray(sw.x)*(-np.sin(theta)) + np.asarray(sw.z)*np.cos(theta),-1,1))
        widths[name] = float(np.median(angular[np.asarray(ok)]))*180/np.pi
    tests = {'untextured_trace_regression': err < 1e-5,
             'independent_mirror_checker_mapping': im['mismatches'] == 0,
             'roi_visible': int(im['roi'].sum()) >= 10000,
             'eval_peak_increases': peaks['G010'] > peaks['G040'] > peaks['G075'],
             'sample_lobe_narrows': widths['G010'] < widths['G040'] < widths['G075']}
    # Exact full S1 state is essential: there is no single S1 roughness.
    st = orig[1]
    material = {'state_69': st.tolist(), 'lam_rgb_8': st[:24].reshape(3,8).tolist(),
                'alpha_u': st[24:31].tolist(), 'alpha_v': st[31:38].tolist(),
                'F0': st[38:45].tolist(), 'sampling_probabilities': st[45:53].tolist(),
                's_knots_rgb': st[53:68].reshape(3,5).tolist(), 'eta_effective': float(st[68]),
                'physical_layered_descriptor': ctx.mats.desc[1].tolist()}
    write({'tests': tests, 'pass': all(tests.values()), 'regression_rel_max': err,
           'roi_pixels_512': int(im['roi'].sum()), 'mapping_mismatches': im['mismatches'],
           'mapping_safe_pixels': int(im['safe'].sum()), 'eval_peaks': peaks,
           'sample_median_angle_degrees': widths, 'S1': material,
           'trace_patch_sha256': ph}, Path(root)/'results/preflight.json')
    print(json.dumps(tests), flush=True)
    assert all(tests.values()), 'STOP_PREFLIGHT'


def render(root, material, variant, res, spp, seed):
    PT, S, ctx, lights, cam, orig, ph = setup(root, material, variant, int(res))
    out = Path(root)/'results/renders'
    name = f'{material}_{variant}_{res}_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/f'{name}.json').exists():
        print('EXISTS', name, flush=True)
        return
    gb = CK.gbuf(PT, ctx, cam)
    if not (Path(root)/f'results/maps_{res}.npz').exists() and variant == 'bx38':
        im = independent_map(PT, ctx, cam, gb)
        assert im['mismatches'] == 0
        np.savez_compressed(Path(root)/f'results/maps_{res}.npz',
                            **{k:v for k,v in im.items() if isinstance(v,np.ndarray)},
                            sid=gb['sid'], pid=gb['pid'], mpid=gb['mpid'], panel=gb['panel'])
    acc, timing = ctx.render(lights, 'ER', cam, int(res), int(spp), int(seed), 'cam', b=8)
    z = PT.pack(acc, int(res), int(spp))
    # Every path starting on an opaque reflecting panel has only reflection
    # after its first vertex. e1 is direct source lighting at that panel.
    z['reflection'] = (z['e2']+z['h'])*gb['panel'][...,None]
    z['full'] = z['e1']+z['e2']+z['h']
    assert np.all(z['full'] >= 0)
    np.savez_compressed(out/f'{name}.npz', **z)
    write({'material': material, 'variant': variant, 'res': int(res), 'spp': int(spp),
           'seed': int(seed), 'timing': timing, 'scene': CK.SC,
           'scene_sha256': sha(HERE/'scene.json'), 'protocol_sha256': sha(HERE/'protocol.json'),
           'script_sha256': sha(__file__), 'trace_patch_sha256': ph,
           'state_table_sha256': hashlib.sha256(ctx.mats.states.tobytes()).hexdigest(),
           'original_table_sha256': hashlib.sha256(orig.tobytes()).hexdigest(),
           'mitsuba': PT.mi.__version__, 'drjit': PT.dr.__version__,
           'variant_mitsuba': PT.mi.variant(), 'python': sys.version, 'platform': platform.platform(),
           'scope': 'REF-ER elsewhere; specified analytic material at panel'}, out/f'{name}.json')
    print('DONE', name, timing, flush=True)


def model(root, variant, res, spp, seed, checker_root):
    PT, S, ctx, lights, cam, orig, ph = setup(root,'S1',variant,int(res))
    table = Path(checker_root)/'results/tables'/f'CK_{variant}_CK2_T1.npz'
    z = np.load(table)
    for sd in [8601,8602]:
        key = f'B_own_s{sd}_C0'
        acc, timing = ctx.render(lights,'ER',cam,int(res),int(spp),int(seed),'ecro',B=PT.fl(z[key]),b=8)
        arr = PT.pack(acc,int(res),int(spp))
        name = f'ECRO{sd}_{variant}_{res}_{seed}'
        np.savez_compressed(Path(root)/'results/renders'/f'{name}.npz', **arr)
        write({'table':str(table),'table_sha256':sha(table),'key':key,'seed':int(seed),'spp':int(spp),
               'scene_sha256':sha(HERE/'scene.json'),'timing':timing,
               'scope':'S1 unchanged; checker material-distribution OOD diagnostic only; not a near-mirror prediction'},
              Path(root)/'results/renders'/f'{name}.json')
        print('MODEL_DONE',name,timing,flush=True)


if __name__ == '__main__':
    command, root, *args = sys.argv[1:]
    root = Path(root)
    (root/'results').mkdir(parents=True, exist_ok=True)
    {'pre':pre, 'render':render, 'model':model}[command](root,*args)
