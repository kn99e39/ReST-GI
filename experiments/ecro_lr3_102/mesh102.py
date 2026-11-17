"""WL102 mesh patch layout (numpy only): the triangle-mesh analogue of scene86.Geom, built from a loaded Mitsuba scene.

Rule (protocol wl102.json "patch_rule"; Amendment 1, before any LR3 result: exact cell regions instead of centroid binning):
  per object, a patch is the surface region of faces of one 6-way dominant-normal class (axis of the largest |n| component and its
  sign) inside one world-aligned CELL (default 0.25 m) grid cell; faces are clipped at the cell planes;
  frame: n = area-weighted mean normal, t = the world axis least aligned with n projected off n, b = n x t;
  point sampling is area-weighted over the patch's triangles;
  an object whose BSDF is two-sided and whose mesh is not closed (some undirected edge not shared by exactly two faces) also gets a
  back-side patch set (same groups, frame -n, b = n' x t).
Tables mirror scene86.Geom (PC, PN, PT, PB, PAREA, PMAT, N) so model86.GeoState consumes them unchanged; PMAT = patch id (one
encoder feature row per patch, rows supplied separately).
"""
import numpy as np

CELL = 0.25


def mesh_arrays(scene, mi):
    """[(shape, id, V [n,3] float64, F [m,3] int64, two_sided, is_emitter)] in scene.shapes() order."""
    out = []
    for s in scene.shapes():
        sid = s.id()
        p = mi.traverse(s)                     # per shape: unnamed shapes have no scene-level key
        V = np.asarray(p['vertex_positions'], np.float64).reshape(-1, 3)
        F = np.asarray(p['faces'], np.int64).reshape(-1, 3)
        two = bool(int(s.bsdf().flags()) & int(mi.BSDFFlags.BackSide))
        out.append((s, sid, V, F, two, bool(s.is_emitter())))
    return out


def closed(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(cnt == 2))


def frame(n):
    n = n / max(np.linalg.norm(n), 1e-30)
    e = np.eye(3)[int(np.argmin(np.abs(n)))]
    t = e - n * (n @ e)
    t /= np.linalg.norm(t)
    return n, t, np.cross(n, t)


def _clip(poly, axis, val):
    """Sutherland-Hodgman split of a convex polygon [k,3] by the plane x[axis] = val -> (below, above) (either may be None)."""
    lo, hi = [], []
    k = len(poly)
    for i in range(k):
        p, q = poly[i], poly[(i + 1) % k]
        dp, dq = p[axis] - val, q[axis] - val
        (lo if dp <= 0 else hi).append(p)
        if (dp < 0 < dq) or (dq < 0 < dp):
            x = p + (q - p) * (dp / (dp - dq))
            lo.append(x); hi.append(x)
    f = lambda l: np.array(l) if len(l) >= 3 else None
    return f(lo), f(hi)


def cell_pieces(tri, cell):
    """Polygons of one triangle [3,3], each inside a single grid cell."""
    polys = [tri]
    for ax in range(3):
        out = []
        for P in polys:
            c0, c1 = int(np.floor(P[:, ax].min() / cell)), int(np.floor(P[:, ax].max() / cell))
            cur = P
            for c in range(c0 + 1, c1 + 1):
                lo, cur = _clip(cur, ax, c * cell)
                if lo is not None:
                    out.append(lo)
                if cur is None:
                    break
            if cur is not None:
                out.append(cur)
        polys = out
    return polys


class MeshGeom:
    """Patches are exact surface regions {object, dominant-normal class of the face, grid cell of the point} (wl102 Amendment 1):
    faces are clipped at the cell planes, so a large triangle spans many patches and a hit is mapped by its position."""

    def __init__(self, meshes, cell=CELL):
        self.cell = cell
        PC, PN, PT, PB, PA, POBJ, PSIDE, PCOS, PKEY = [], [], [], [], [], [], [], [], []
        T_V0, T_E1, T_E2, T_A, T_P = [], [], [], [], []
        face_obj, face_cls = [], []
        self.objects = []
        for oi, (_, sid, V, F, two, emit) in enumerate(meshes):
            a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            cr = np.cross(b - a, c - a)
            ar = 0.5 * np.linalg.norm(cr, axis=1)
            nrm = cr / np.maximum(2 * ar, 1e-30)[:, None]
            ax = np.argmax(np.abs(nrm), 1)
            cls = (2 * ax + (nrm[np.arange(len(F)), ax] > 0)).astype(np.int64)
            face_obj.append(np.full(len(F), oi, np.int64)); face_cls.append(cls)
            tri = np.stack([a, b, c], 1)
            lo = np.floor(tri.min(1) / cell).astype(np.int64); hi = np.floor(tri.max(1) / cell).astype(np.int64)
            single = np.all(lo == hi, 1) & (ar > 0)
            # pieces: (face, cell[3], v0, e1, e2, area) as fan triangles
            pf, pc, pv0, pe1, pe2, pa = [], [], [], [], [], []
            idx = np.nonzero(single)[0]
            pf.append(idx); pc.append(lo[idx]); pv0.append(a[idx]); pe1.append(b[idx] - a[idx]); pe2.append(c[idx] - a[idx]); pa.append(ar[idx])
            for fi in np.nonzero(~single & (ar > 0))[0]:
                for P in cell_pieces(tri[fi], cell):
                    cc = np.floor(P.mean(0) / cell).astype(np.int64)
                    for j in range(1, len(P) - 1):
                        e1, e2 = P[j] - P[0], P[j + 1] - P[0]
                        A = 0.5 * np.linalg.norm(np.cross(e1, e2))
                        if A > 0:
                            pf.append(np.array([fi])); pc.append(cc[None]); pv0.append(P[0][None]); pe1.append(e1[None]); pe2.append(e2[None])
                            pa.append(np.array([A]))
            pf, pc, pv0, pe1, pe2, pa = (np.concatenate(x) for x in (pf, pc, pv0, pe1, pe2, pa))
            key = np.concatenate([cls[pf][:, None], pc], 1)
            uk, inv = np.unique(key, axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            sides = [1] + ([-1] if (two and not closed(F)) else [])
            order = np.argsort(inv, kind='stable'); bounds = np.searchsorted(inv[order], np.arange(len(uk) + 1))
            pcen = pv0 + (pe1 + pe2) / 3
            for side in sides:
                base = len(PC)
                for g in range(len(uk)):
                    m = order[bounds[g]:bounds[g + 1]]
                    w = pa[m]
                    n, t, bb = frame(side * (nrm[pf[m]] * w[:, None]).sum(0))
                    PC.append((pcen[m] * w[:, None]).sum(0) / w.sum()); PN.append(n); PT.append(t); PB.append(bb); PA.append(w.sum())
                    POBJ.append(oi); PSIDE.append(side); PKEY.append(np.concatenate([[oi], uk[g]]))
                    PCOS.append(float(((side * nrm[pf[m]]) @ n * w).sum() / w.sum()))   # area-weighted mean cos to the patch normal
                T_V0.append(pv0); T_E1.append(pe1); T_E2.append(pe2); T_A.append(pa); T_P.append(base + inv)
            self.objects.append({'id': sid, 'triangles': int(len(F)), 'pieces': int(len(pf)), 'two_sided': two, 'closed': closed(F),
                                 'emitter': emit, 'sides': len(sides), 'patches_per_side': int(len(uk)), 'area': float(ar.sum())})
        self.N = len(PC)
        self.PC, self.PN, self.PT, self.PB = (np.array(x, np.float64) for x in (PC, PN, PT, PB))
        self.PAREA = np.array(PA, np.float64); self.POBJ = np.array(POBJ, np.int64); self.PSIDE = np.array(PSIDE, np.int64)
        self.PCOS = np.array(PCOS, np.float64)
        self.PMAT = np.arange(self.N, dtype=np.int64)
        self.FACE_OBJ = np.concatenate(face_obj); self.FACE_CLS = np.concatenate(face_cls)
        # runtime lookup: integer key of (object, class, cell) -> front / back patch (-1: none)
        K = np.array(PKEY, np.int64)
        self.CMIN = K[:, 2:].min(0); self.CDIM = K[:, 2:].max(0) - self.CMIN + 1
        self.NOBJ = len(meshes)
        code = self.encode(K[:, 0], K[:, 1], K[:, 2:])
        uc = np.unique(code)
        self.KEYS = uc
        self.KFRONT = np.full(len(uc), -1, np.int64); self.KBACK = np.full(len(uc), -1, np.int64)
        pos = np.searchsorted(uc, code)
        self.KFRONT[pos[self.PSIDE > 0]] = np.nonzero(self.PSIDE > 0)[0]
        self.KBACK[pos[self.PSIDE < 0]] = np.nonzero(self.PSIDE < 0)[0]
        # sampling tables: pieces grouped by patch, cdf within each patch
        tp = np.concatenate(T_P); order = np.argsort(tp, kind='stable')
        self.T_V0, self.T_E1, self.T_E2 = (np.concatenate(x)[order] for x in (T_V0, T_E1, T_E2))
        ta = np.concatenate(T_A)[order]; tp = tp[order]
        cnt = np.bincount(tp, minlength=self.N)
        self.P_TOFF = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64); self.P_TCNT = cnt.astype(np.int64)
        cs = np.cumsum(ta); start = np.repeat(cs[self.P_TOFF] - ta[self.P_TOFF], cnt)
        self.T_CDF = (cs - start) / np.repeat(self.PAREA, cnt)
        self.T_CDF[self.P_TOFF + cnt - 1] = 1.0
        assert self.N < 32768, 'STOPPED_ARCHITECTURE_EXPANSION_REQUIRED: int16 patch ids (A3)'

    def encode(self, obj, cls, cellid):
        c = np.asarray(cellid, np.int64) - self.CMIN
        return ((np.asarray(obj, np.int64) * 6 + np.asarray(cls, np.int64)) * self.CDIM[0] + c[:, 0]) * self.CDIM[1] * self.CDIM[2] \
            + c[:, 1] * self.CDIM[2] + c[:, 2]

    def summary(self):
        side = np.sqrt(self.PAREA)
        return {'N': int(self.N), 'objects': self.objects, 'patch_area_m2': {q: float(np.quantile(self.PAREA, p)) for q, p in
                (('min', 0), ('p10', .1), ('median', .5), ('p90', .9), ('max', 1))},
                'patch_side_m_median': float(np.median(side)), 'back_side_patches': int((self.PSIDE < 0).sum()),
                'sampling_triangles': int(len(self.T_CDF)), 'keys': int(len(self.KEYS)),
                'mean_cos_to_patch_normal': {q: float(np.quantile(self.PCOS, p)) for q, p in (('min', 0), ('p01', .01), ('p10', .1), ('median', .5))}}
