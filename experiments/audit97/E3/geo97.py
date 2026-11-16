"""WL97 audit track E3: CK2 geometry written independently from the fixture JSON files (numpy only).
Checked against the production scene86.Geom tables by val97_dr.py (patch origins, edges, normals of patches 1088..1183)."""
import json, math, os
import numpy as np

X, Y, Z = np.eye(3)
PANEL0, BOXA0 = 1088, 1130                    # derived: 256 + 256 + 192 + 192 + 192 = 1088 (panel 6 x 7 = 42), then box A 6 x 9
NZ, NPHI = 4, 16
NB = NZ * NPHI
DOM = 2 * math.pi / NB


def load(repo):
    fx = json.load(open(os.path.join(repo, 'experiments', 'ecro86', 'protocol', 'ecro86.json')))['fixture']
    sc = json.load(open(os.path.join(repo, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
    return fx, sc


def panel(fx):
    """Panel rectangle: centre, in-plane axes (a1 = +y, a2 = +z), half sizes, normal +x, 6 x 7 patches (patch = 1088 + iu + 6 iv)."""
    return {'c': np.array(fx['panel']['center'], float), 'a1': Y.copy(), 'a2': Z.copy(), 'h': np.array(fx['panel']['half'], float),
            'n': X.copy(), 'nu': 6, 'nv': 7, 'base': PANEL0}


def box_faces(box):
    """Six faces of an oriented box in the scene86 order f = 2 * axis + positive_side; a1 x a2 = outward normal; 3 x 3 patches each."""
    c = np.array(box['c'], float); h = np.array(box['h'], float)
    cy, sy = math.cos(box['yaw']), math.sin(box['yaw'])
    ax = [np.array([cy, sy, 0.0]), np.array([-sy, cy, 0.0]), Z.copy()]
    pairs = {(0, 1): (1, 2), (0, 0): (2, 1), (1, 1): (2, 0), (1, 0): (0, 2), (2, 1): (0, 1), (2, 0): (1, 0)}
    out = []
    for a in range(3):
        for pos in (0, 1):
            i1, i2 = pairs[(a, pos)]
            n = np.cross(ax[i1], ax[i2])
            assert np.allclose(n, (1 if pos else -1) * ax[a])
            out.append({'c': c + (1 if pos else -1) * h[a] * ax[a], 'a1': ax[i1], 'a2': ax[i2], 'h': np.array([h[i1], h[i2]]), 'n': n,
                        'nu': 3, 'nv': 3})
    for f, r in enumerate(out):
        r['base'] = BOXA0 + 9 * f
    return out


def patch_rects(rect):
    """Per patch of a rectangle: origin o, edges e1, e2 (p = o + a e1 + b e2, a, b in [0, 1]), area; order iu + nu * iv."""
    nu, nv = rect['nu'], rect['nv']
    e1 = rect['a1'] * (2 * rect['h'][0] / nu); e2 = rect['a2'] * (2 * rect['h'][1] / nv)
    o0 = rect['c'] - rect['a1'] * rect['h'][0] - rect['a2'] * rect['h'][1]
    O = np.array([o0 + iu * e1 + iv * e2 for iv in range(nv) for iu in range(nu)])
    return O, e1, e2, float(np.linalg.norm(e1) * np.linalg.norm(e2))


def bin_weights():
    """Exact int_bin cos(theta) dOmega for the 4 rings (uniform in z = cos theta) x 16 sectors layout of scene86 / pt86.Ctx.dir_in_bin:
    ring iz covers z in [iz / 4, (iz + 1) / 4], sector width 2 pi / 16 -> (2 pi / 16) * ((iz + 1)^2 - iz^2) / 32. Sums to pi."""
    iz = np.arange(NB) // NPHI
    return (2 * math.pi / NPHI) * ((iz + 1) ** 2 - iz ** 2) / (2 * NZ * NZ)
