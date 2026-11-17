"""WL102 gate A1 (Amendment 1 operationalisation): the mesh code path (pt102 on MeshGeom) vs the original analytic path (pt86) on
the ECRO room, WL86 E1_T0 geometry and light set, every surface Lambert (pt86: ER Lambert rows; pt102: Mitsuba diffuse, same albedo).

Compared on the shell patches whose regions are identical in both partitions (floor, ceiling, three walls: 25 cm cells on a
world-aligned grid), noise = the same quantity from two pt86 runs with different table / field seeds:
  openness and form-factor row sums (state rays), per-patch source totals, field f = f2 + fh (2..5 events) at 256 spp,
and a D = 6 camera render (512^2, 256 spp) on the whole image. Pass: RMSE(ours - pt86_a) <= 2 x RMSE(pt86_b - pt86_a) for each.
  python gate_a1.py <out_dir>
"""
import json, math, os, sys
import numpy as np
import pt102 as P

PT, S, mi, dr = P.PT, P.S, P.mi, P.dr
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
FX = S.FX
geom, ls = PT.config('E1_T0')
LSET = FX['light_sets'][ls]
la = FX['lambert_albedo']
ALB = [0.55, 0.55, 0.55, la['walls_ceiling'], la['boxA'], la['boxB']]          # material ids 0..5 (floor, panel, sphere, walls, A, B)
SPP_FIELD, SPP_CAM, RES = 256, 256, 512


def lambert_row(a):
    st = np.zeros(69, np.float32)
    for c in range(3):
        st[c * 8] = a
    st[24:38] = .3; st[38:45] = .5; st[45] = 1; st[53:] = 1
    return st


def pt86_ctx(table_seed):
    S.P['representation']['table_seed'] = table_seed
    c = PT.Ctx(geom)
    c.mats.states = np.stack([lambert_row(a) for a in ALB])
    return c


def write_ply(path, V, Fc):
    with open(path, 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n'
                f'element face {len(Fc)}\nproperty list uchar int vertex_indices\nend_header\n')
        for v in V: f.write(f'{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n')
        for t in Fc: f.write(f'3 {t[0]} {t[1]} {t[2]}\n')
    return path


def quads(rects):
    """(c, a1, a2, h1, h2, ...) -> vertices / faces of 2 triangles each, normal a1 x a2."""
    V, Fc = [], []
    for r in rects:
        c, a1, a2, h1, h2 = (np.asarray(x, np.float64) for x in r[:5])
        b = len(V)
        V += [c - a1 * h1 - a2 * h2, c + a1 * h1 - a2 * h2, c + a1 * h1 + a2 * h2, c - a1 * h1 + a2 * h2]
        Fc += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return np.array(V), Fc


def mesh_scene():
    """The E1_T0 room as triangle meshes (each analytic rectangle -> 2 triangles, boxes -> 12, sphere -> 128 x 64 UV mesh)."""
    g = S.Geom(geom)
    T = mi.ScalarTransform4f
    diff = lambda a: {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [a, a, a]}}
    rect_mat = [0, 3, 3, 3, 3, 1]
    d = {'type': 'scene'}
    for i, r in enumerate(S.stationary_rects()):
        V, Fc = quads([r])
        d[f'r{i}'] = {'type': 'ply', 'filename': write_ply(os.path.join(out, f'r{i}.ply'), V, Fc), 'face_normals': True,
                      'bsdf': diff(ALB[rect_mat[i]])}
    for b_, mat in (('A', 4), ('B', 5)):
        V, Fc = quads(S.box_faces(geom[b_], mat))
        d[f'box{b_}'] = {'type': 'ply', 'filename': write_ply(os.path.join(out, f'box{b_}.ply'), V, Fc), 'face_normals': True,
                         'bsdf': diff(ALB[mat])}
    th = np.linspace(0, math.pi, 65); ph = np.linspace(-math.pi, math.pi, 129)
    c0 = np.array(FX['sphere']['center']); R = FX['sphere']['radius']
    V = np.array([[R * math.sin(t) * math.cos(p), R * math.sin(t) * math.sin(p), R * math.cos(t)] for t in th for p in ph]) + c0
    Fc = []
    for i in range(64):
        for j in range(128):
            a_, b2, c_, e = i * 129 + j, i * 129 + j + 1, (i + 1) * 129 + j, (i + 1) * 129 + j + 1
            if i > 0: Fc.append([a_, c_, b2])                 # outward normals: (c - a) x (b - a) ~ e_theta x e_phi = r
            if i < 63: Fc.append([b2, c_, e])
    d['sph'] = {'type': 'ply', 'filename': write_ply(os.path.join(out, 'sphere.ply'), V, Fc), 'face_normals': True, 'bsdf': diff(ALB[2])}
    cam = FX['camera']
    d['sensor'] = {'type': 'perspective', 'fov': cam['fov_deg'], 'fov_axis': 'x',
                   'to_world': T.look_at(origin=cam['origin'], target=cam['target'], up=[0, 0, 1]),
                   'film': {'type': 'hdrfilm', 'width': RES, 'height': RES, 'rfilter': {'type': 'box'}}}
    return mi.load_dict(d)


def shell_match(gm, g86):
    """pt86 patch ids 0..1087 (rects r0..r4) -> MeshGeom patch ids with the same region (centroid within 1 mm, same normal)."""
    n86 = int(g86.RBASE[5])
    m = np.full(n86, -1)
    for p in range(n86):
        dd = np.linalg.norm(gm.PC - g86.PC[p], axis=1) + 10 * (1 - gm.PN @ g86.PN[p])
        j = int(np.argmin(dd))
        if dd[j] < 1e-3:
            m[p] = j
    return m


def openness(hit, rw, N):
    ok = hit.reshape(N, -1) >= 0
    return 1.0 - (ok * rw[None]).sum(1) / math.pi


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


import model86 as M  # noqa: E402
rw = M.ray_weights()
lights = PT.Lights(LSET)
A, B = pt86_ctx(86001), pt86_ctx(86002)
mesh = P.Ctx(mesh_scene(), table_seed=86001)
match = shell_match(mesh.G, A.G)
assert (match >= 0).all(), f'shell regions not matched: {(match < 0).sum()}'
res = {'N_pt86': A.N, 'N_mesh': mesh.N, 'shell_patches': int(len(match))}

oa, ob, om = (openness(c.state_rays()[0], rw, c.N) for c in (A, B, mesh))
sa, sb, sm = (c.source(lights).sum(1) for c in (A, B, mesh))
fa = A.field(lights, SPP_FIELD, 1, kmax=5)[0]; fb = B.field(lights, SPP_FIELD, 2, kmax=5)[0]
f2, fh, _ = mesh.field_components(lights, SPP_FIELD, 3); fm = f2 + fh
cam = FX['camera']
ia = PT.pack(A.render(lights, 'ER', cam, RES, SPP_CAM, 11, 'cam')[0], RES, SPP_CAM)
ib = PT.pack(B.render(lights, 'ER', cam, RES, SPP_CAM, 12, 'cam')[0], RES, SPP_CAM)
am, Tm = mesh.render(lights, SPP_CAM, 13, 'cam'); imm = P.pack(am, RES, RES, SPP_CAM)
full = lambda o: o['e1'] + o['e2'] + o['h']
sh = slice(0, len(match))
rows = {
    'openness': (oa[sh], ob[sh], om[match]),
    'source_total': (sa[sh], sb[sh], sm[match]),
    'field_2to5': (fa[sh], fb[sh], fm[match]),
    'render_D6_full': (full(ia), full(ib), full(imm)),
}
for k, (a, b, m) in rows.items():
    noise, d = rmse(b, a), rmse(m, a)
    res[k] = {'rmse_mesh_vs_pt86': d, 'rmse_pt86_seeds': noise, 'ratio': d / max(noise, 1e-30), 'mean_pt86': float(np.mean(a)),
              'mean_mesh': float(np.mean(m)), 'pass': bool(d <= 2 * noise)}
res['A1_pass'] = all(res[k]['pass'] for k in rows)
json.dump(res, open(os.path.join(out, 'gate_a1.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(out, 'gate_a1_renders.npz'), pt86_a=full(ia), pt86_b=full(ib), mesh=full(imm))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in res.items()}))
