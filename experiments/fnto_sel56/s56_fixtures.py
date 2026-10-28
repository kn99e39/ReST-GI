"""WL56 immutable two-domain fixture. No renderer/GPU imports."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for d in ('phys_ref53', 'fnto_c54', 'fnto_k55'):
    sys.path.insert(0, str(HERE.parent / d))
import k55_common as CM
import p53_geom as G
import p53_fixtures as FX
import k55_fixtures as KF

KEYS = ('top', 'bot', 'otop')
CAMERAS = ('top', 'gapdown', 'gapup', 'bot', 'otop')

def scene(state):
    assert state in ('rest', 'dent', 'rigid')
    m, v = FX.domain_geometry('slab')
    u = FX.domain_geometry('dent')[1] if state == 'dent' else v.copy()
    shift = np.array((0.0037, -0.0021, 0.002)) if state == 'rigid' else np.zeros(3)
    # Both rest slab surfaces bound an 8 mm air gap; the upper top dents, its bottom stays fixed.
    comps = [('lower', 'medium', m, v + (0, 0, -0.009)),
             ('upper', 'medium', m, u + (0, 0, 0.009) + shift)]
    # 20 cm square, 2.5 mm triangles. Downward, absorbing on return, unit radiance.
    em = G.open_grid(20, 20, 0.01, (-0.1, -0.1, 0.054), sub=4)
    em = dict(em); em['F'] = em['F'][:, ::-1].copy()
    comps.append(('source', 'emitter', em, em['V'].copy()))
    cams = KF.gap_cams(0.009, 0.009)
    cams['top'] = FX._cam('top', (0, 0, 0.009), 0, 0.11, 220, origin=(0, 0, 0.04))
    cams['otop'] = FX._cam('otop', (0, 0, 0.009), 0.04, 0.11, 220)
    return dict(comps=comps, lights=[], cams=KF.with_px({k: cams[k] for k in CAMERAS}), emit={'source': 'U'}, state=state)

def intrinsic(state, ci):
    return FX.domain_geometry('dent' if state == 'dent' and ci == 1 else 'slab')

def signature(state, ci):
    """Canonical material mesh + current intrinsic vertices; placement is separate by contract."""
    import hashlib, json
    m, v = intrinsic(state, ci)
    h = hashlib.sha256(json.dumps(dict(sigma_t=FX.SIGMA_T, albedo=FX.ALBEDO, g=FX.G_PHASE, level=4, dout=64, din=16)).encode())
    for a in (m['V'], m['F'], m['tri_port'], m['port_tangent'], v):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def dirty(state):
    return [ci for ci in (0, 1) if signature(state, ci) != signature('rest', ci)]
