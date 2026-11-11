"""WL86 scene family, patch / direction-bin discretisation and sampling rules (numpy only; no renderer import).

Room x,y in [-2,2], z in [0,3], open towards -y. Stationary: floor (ER B), ceiling + 3 walls (Lambert), panel (ER S1), sphere (ER S2).
Movable: box A and box B (Lambert), each given by centre c, half sizes h and yaw about z.
Surfaces are split into N patches (constant layout for every geometry); directions into NB = NZ x NPHI equal-solid-angle bins of the
patch-local hemisphere (z = cos(theta) rings x azimuth sectors).
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86.json')))
FX = P['fixture']
REP = P['representation']
NZ, NPHI = REP['bins_z'], REP['bins_phi']
NB = NZ * NPHI
KAPPA = REP['state_rays_per_bin']
K = NB * KAPPA
PSRC = REP['source_points_per_patch']
DOMEGA = 2 * math.pi / NB
SPH_NPHI, SPH_NZ = 8, 4
# material ids: 0 B (floor), 1 S1 (panel), 2 S2 (sphere), 3 Lambert walls / ceiling, 4 Lambert box A, 5 Lambert box B
MAT_CLASS = [2, 0, 1, 2, 2, 2]                 # object-level class 0 sharp, 1 moderate, 2 broad
N_RECT = 18                                    # 6 stationary rectangles + 2 x 6 box faces
SID_SPHERE, SID_BOXA, SID_BOXB = 6, 7, 8        # shape ids: 0..5 rectangles, 6 sphere, 7 / 8 boxes
X, Y, Z = np.eye(3)


def stationary_rects():
    """(centre, axis1, axis2, half1, half2, nu, nv, material); normal = axis1 x axis2 (into the room)."""
    pc = np.array(FX['panel']['center']); ph = FX['panel']['half']
    return [
        (np.array([0, 0, 0.0]), X, Y, 2.0, 2.0, 16, 16, 0),            # floor  +z
        (np.array([0, 0, 3.0]), Y, X, 2.0, 2.0, 16, 16, 3),            # ceiling -z
        (np.array([0, 2.0, 1.5]), X, Z, 2.0, 1.5, 16, 12, 3),          # back wall -y
        (np.array([-2.0, 0, 1.5]), Y, Z, 2.0, 1.5, 16, 12, 3),         # left wall +x
        (np.array([2.0, 0, 1.5]), Z, Y, 1.5, 2.0, 12, 16, 3),          # right wall -x
        (pc, Y, Z, ph[0], ph[1], 6, 7, 1),                             # panel +x
    ]


def box_axes(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c, s, 0.0]), np.array([-s, c, 0.0]), Z.copy()


def box_faces(box, mat):
    """Faces f = 2 * axis + (positive side): -x', +x', -y', +y', -z, +z; in-plane axes with a1 x a2 = outward normal."""
    c = np.array(box['c'], np.float64); h = box['h']; ax = box_axes(box['yaw'])
    pairs = {(0, 1): (1, 2), (0, 0): (2, 1), (1, 1): (2, 0), (1, 0): (0, 2), (2, 1): (0, 1), (2, 0): (1, 0)}
    out = []
    for a in range(3):
        for pos in (0, 1):
            i1, i2 = pairs[(a, pos)]
            out.append((c + (1 if pos else -1) * h[a] * ax[a], ax[i1], ax[i2], h[i1], h[i2], 3, 3, mat))
    return out


class Geom:
    """All tables of one geometry. Patch ids: rectangles in order (row-major iu + nu * iv), then the sphere (iphi + 8 * iz)."""

    def __init__(self, g):
        self.g = g
        rects = stationary_rects() + box_faces(g['A'], 4) + box_faces(g['B'], 5)
        assert len(rects) == N_RECT
        self.RC = np.array([r[0] for r in rects]); self.RA1 = np.array([r[1] for r in rects]); self.RA2 = np.array([r[2] for r in rects])
        self.RH = np.array([[r[3], r[4]] for r in rects]); self.RG = np.array([[r[5], r[6]] for r in rects], np.int64)
        self.RMAT = np.array([r[7] for r in rects], np.int64)
        self.RBASE = np.concatenate([[0], np.cumsum(self.RG[:, 0] * self.RG[:, 1])[:-1]])
        self.SBASE = int((self.RG[:, 0] * self.RG[:, 1]).sum())
        self.N = self.SBASE + SPH_NPHI * SPH_NZ
        self.SC = np.array(FX['sphere']['center'], np.float64); self.SR = float(FX['sphere']['radius'])
        N = self.N
        self.PO = np.zeros((N, 3)); self.PE1 = np.zeros((N, 3)); self.PE2 = np.zeros((N, 3))      # rect patch: p = PO + a PE1 + b PE2
        self.PT = np.zeros((N, 3)); self.PB = np.zeros((N, 3)); self.PN = np.zeros((N, 3)); self.PC = np.zeros((N, 3))
        self.PAREA = np.zeros(N); self.PMAT = np.zeros(N, np.int64); self.PSID = np.zeros(N, np.int64)
        self.PSPH = np.zeros((N, 2))                                                               # sphere patch (iphi, iz)
        for r in range(N_RECT):
            nu, nv = self.RG[r]
            n = np.cross(self.RA1[r], self.RA2[r])
            e1 = self.RA1[r] * (2 * self.RH[r, 0] / nu); e2 = self.RA2[r] * (2 * self.RH[r, 1] / nv)
            o0 = self.RC[r] - self.RA1[r] * self.RH[r, 0] - self.RA2[r] * self.RH[r, 1]
            for iv in range(nv):
                for iu in range(nu):
                    p = self.RBASE[r] + iu + nu * iv
                    self.PO[p] = o0 + iu * e1 + iv * e2; self.PE1[p] = e1; self.PE2[p] = e2
                    self.PT[p] = self.RA1[r]; self.PB[p] = self.RA2[r]; self.PN[p] = n
                    self.PC[p] = self.PO[p] + 0.5 * (e1 + e2)
                    self.PAREA[p] = np.linalg.norm(e1) * np.linalg.norm(e2)
                    self.PMAT[p] = self.RMAT[r]
                    self.PSID[p] = r if r < 6 else (SID_BOXA if r < 12 else SID_BOXB)
        for iz in range(SPH_NZ):
            for ip in range(SPH_NPHI):
                p = self.SBASE + ip + SPH_NPHI * iz
                z = (iz + 0.5) / SPH_NZ * 2 - 1; phi = (ip + 0.5) / SPH_NPHI * 2 * math.pi - math.pi
                n = np.array([math.sqrt(1 - z * z) * math.cos(phi), math.sqrt(1 - z * z) * math.sin(phi), z])
                t = np.cross(Z, n); t /= np.linalg.norm(t)
                self.PT[p] = t; self.PB[p] = np.cross(n, t); self.PN[p] = n; self.PC[p] = self.SC + self.SR * n
                self.PAREA[p] = 4 * math.pi * self.SR ** 2 / (SPH_NPHI * SPH_NZ)
                self.PMAT[p] = 2; self.PSID[p] = SID_SPHERE; self.PSPH[p] = (ip, iz)
        self.BX = np.stack([np.concatenate(box_axes(g[b]['yaw'])) for b in 'AB'])                  # [2, 9]

    def rect_table(self):
        """[18, 14]: c 3, a1 3, a2 3, h1, h2, nu, nv, base."""
        return np.concatenate([self.RC, self.RA1, self.RA2, self.RH, self.RG.astype(np.float64), self.RBASE[:, None].astype(np.float64)], 1)

    def patch_table(self):
        """[N, 21]: o 3, e1 3, e2 3, t 3, b 3, n 3, is_sphere, iphi, iz."""
        sph = (self.PSID == SID_SPHERE).astype(np.float64)[:, None]
        return np.concatenate([self.PO, self.PE1, self.PE2, self.PT, self.PB, self.PN, sph, self.PSPH], 1)

    def box_matrix(self, b):
        """4 x 4 to_world of the unit cube [-1, 1]^3."""
        ax = box_axes(self.g[b]['yaw']); h = self.g[b]['h']
        M = np.eye(4)
        for k in range(3):
            M[:3, k] = ax[k] * h[k]
        M[:3, 3] = self.g[b]['c']
        return M

    def obb(self, b, pad=1e-3):
        return np.array(self.g[b]['c'], np.float64), np.stack(box_axes(self.g[b]['yaw'])), np.array(self.g[b]['h'], np.float64) + pad


def rect_matrix(c, a1, a2, h1, h2):
    M = np.eye(4)
    M[:3, 0] = a1 * h1; M[:3, 1] = a2 * h2; M[:3, 2] = np.cross(a1, a2); M[:3, 3] = c
    return M


def bin_dirs():
    """Bin-centre local directions [NB, 3] (bin = iz * NPHI + iphi)."""
    out = np.zeros((NB, 3))
    for iz in range(NZ):
        for ip in range(NPHI):
            z = (iz + 0.5) / NZ; phi = (ip + 0.5) / NPHI * 2 * math.pi - math.pi
            out[iz * NPHI + ip] = [math.sqrt(1 - z * z) * math.cos(phi), math.sqrt(1 - z * z) * math.sin(phi), z]
    return out


def seg_obb(o, e, obb):
    """Segments o -> e ([M,3]) against an oriented box (centre, axes [3,3], half sizes)."""
    c, ax, h = obb
    lo = (o - c) @ ax.T; le = (e - c) @ ax.T
    d = le - lo
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    t0 = (-h - lo) / d; t1 = (h - lo) / d
    tmin = np.minimum(t0, t1).max(1); tmax = np.maximum(t0, t1).min(1)
    return (tmax >= np.maximum(tmin, 0)) & (tmin <= 1)


# ---------------------------------------------------------------- sampling rules (training / validation distribution)
def _corners(box):
    ax = box_axes(box['yaw']); c = np.array(box['c']); h = box['h']
    return np.array([c + sx * h[0] * ax[0] + sy * h[1] * ax[1] + sz * h[2] * ax[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def geom_ok(g):
    sc = np.array(FX['sphere']['center']); sr = FX['sphere']['radius']
    rad = {}
    for b in 'AB':
        cs = _corners(g[b])
        if cs[:, 0].min() < -1.35 or cs[:, 0].max() > 1.9 or np.abs(cs[:, 1]).max() > 1.9 or cs[:, 2].min() < -1e-9 or cs[:, 2].max() > 2.9:
            return False
        rad[b] = float(np.linalg.norm(g[b]['h']))
        if np.linalg.norm(np.array(g[b]['c']) - sc) < rad[b] + sr + 0.05:
            return False
    return np.linalg.norm(np.array(g['A']['c']) - np.array(g['B']['c'])) >= rad['A'] + rad['B'] + 0.05


def sample_geom(rng):
    D = P['training']['geometry_distribution']
    while True:
        hA = rng.uniform(*D['A_half'], 3)
        A = {'c': [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(hA[2] + 0.02, 2.9 - hA[2])], 'h': hA.tolist(), 'yaw': float(rng.uniform(0, math.pi / 2))}
        hB = [rng.uniform(*D['B_half_xy']), rng.uniform(*D['B_half_xy']), rng.uniform(*D['B_half_z'])]
        B = {'c': [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), hB[2]], 'h': [float(v) for v in hB], 'yaw': float(rng.uniform(0, math.pi / 2))}
        g = {'A': A, 'B': B}
        if geom_ok(g) and not near_test_geom(g):
            return g


def test_geoms():
    out = []
    for e in ('E1', 'E2'):
        for t in ('T0', 'T1'):
            out.append(FX[e][t])
    return out


def near_test_geom(g):
    m = P['training']['exclusion_margin_m']
    for tg in test_geoms():
        for b in 'AB':
            if np.linalg.norm(np.array(g[b]['c']) - np.array(tg[b]['c'])) < m['box']:
                return True
    return False


def test_light_positions():
    out = []
    for k in FX['light_sets'].values():
        out += [l['pos'] for l in k]
    return np.array(out)


def light_ok(pos, g):
    pos = np.array(pos)
    if np.linalg.norm(pos - np.array(FX['sphere']['center'])) < FX['sphere']['radius'] + 0.15:
        return False
    for b in 'AB':
        c, ax, h = Geom.obb(type('o', (), {'g': g})(), b, 0.15)
        if np.all(np.abs((pos - c) @ ax.T) <= h):
            return False
    return True


def sample_light(rng, g, exclude_test=True):
    D = P['training']['light_distribution']
    tl = test_light_positions()
    while True:
        pos = [rng.uniform(*D['x']), rng.uniform(*D['y']), rng.uniform(*D['z'])]
        if not light_ok(pos, g):
            continue
        if exclude_test and np.linalg.norm(tl - np.array(pos), axis=1).min() < P['training']['exclusion_margin_m']['light']:
            continue
        return [{'pos': [float(v) for v in pos], 'rgb': [FX['light_intensity']] * 3}]


def light_arrays(lights):
    """-> pos [2,3], rgb [2,3], axis [2,3], cos_cut [2] (point light: cos_cut = -2; unused slot: rgb = 0)."""
    pos = np.zeros((2, 3)); rgb = np.zeros((2, 3)); axis = np.tile(Z, (2, 1)); cc = np.full(2, -2.0)
    for i, l in enumerate(lights):
        pos[i] = l['pos']; rgb[i] = l['rgb']
        if 'aim' in l:
            a = np.array(l['aim'], np.float64) - pos[i]; axis[i] = a / np.linalg.norm(a); cc[i] = math.cos(math.radians(l['cone_deg']))
    return pos, rgb, axis, cc
