"""WL104 common: the WL84 scene (render81.scene_dict pose 0, lights L1 / L2, 256^2 camera) as a mesh scene for the WL102 ECRO adapter,
the three WL84 materials (WL83 test83 descriptors; learned-ER states of compiler seed 8301), paths and helpers.

Only change to the WL84 scene: the analytic sphere is replaced by a fixed UV-sphere PLY (same centre / radius, exact vertex positions and
normals) because the ECRO patch layout is triangle based (WL102 adapter). The same mesh is used by every new arm and reference; its
effect is measured against the WL84 analytic depth-4 reference (gate T).
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
for d in ('ecro_lr3_102', 'ecro86', 'fnto_compiled83', 'fnto_compiled82', 'fnto_compiled81', 'fnto_compiled84'):
    sys.path.insert(0, str(EXP / d))
# Execution amendment 1 (2026-11-18, before any evidence was inspected): samples per launch. Dr.Jit 0.4.6 re-traces Python on every
# launch, so small launches were CPU-bound (GPU ~0 %); only the batching / RNG layout changes, not spp, estimators or seed bases.
BATCH = {'ref': 2048, 'tracer': 512, 'field': 64}
SMOKE = os.environ.get('WL104_SMOKE') == '1'        # pipeline smoke only (tiny counts, never evidence)
PROTOCOL_PATH = HERE / 'protocol' / 'wl104.json'
P = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8')) if PROTOCOL_PATH.exists() else None
ROOT = REPO / 'results' / 'ecro_wl84_104'
IN63 = ROOT / 'inputs63'
W83 = IN63 / 'rest_compiled83' / 'results'
W84REF = IN63 / 'rest_compiled84' / 'results' / 'ref'
W84REP = IN63 / 'rest_compiled84' / 'results' / 'replicates'
W83REN = IN63 / 'rest_compiled83' / 'results' / 'render'
IMAGES = ['test83_iid_1_L2', 'test83_iid_0_L2', 'test83_hard_1_L1']
LIGHTS = {1: ([-1.5, -1.5, 2.5], 25.0), 2: ([2.0, -0.6, 0.45], 12.0)}      # render81.scene_dict, pose 0
RES = 256
CAM = {'origin': (0.0, -3.0, 1.2), 'target': (0.0, 0.0, 0.3), 'fov': 40.0}
SPHERE = {'center': (-0.4, 0.0, 0.35), 'radius': 0.35}
STATE_SHA = {'test83_iid_0': '92ab71db5824db7818108744de04e5695c0df8da8e0f0abac90c0017ef1fa079',
             'test83_iid_1': 'f7d25ce3609514caa6c249ce4549c0375cb7aa9856f84d525f82bcfcfadf06c4',
             'test83_hard_1': 'e04d3dcabeedd3a0a4820a20e44d93a1d5850258c6e5bbd392be0b2b7f028947'}   # WL84 audit84.json ER_L


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def write_json(path, d):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=1, allow_nan=False, default=float) + '\n', encoding='utf-8')


def image_spec(image):
    tag, light = image.rsplit('_L', 1)
    st, k = tag.rsplit('_', 1)
    return tag, int(light), st, int(k)


# ---------------------------------------------------------------------------------------------------------------- sphere mesh
def uv_sphere_ply(path, n_theta=128, n_phi=256):
    """Unit UV sphere (poles + rings), binary little-endian PLY with exact unit normals. Deterministic."""
    th = np.pi * np.arange(1, n_theta) / n_theta
    ph = 2 * np.pi * np.arange(n_phi) / n_phi
    T_, P_ = np.meshgrid(th, ph, indexing='ij')
    ring = np.stack([np.sin(T_) * np.cos(P_), np.sin(T_) * np.sin(P_), np.cos(T_)], -1).reshape(-1, 3)
    V = np.concatenate([[[0, 0, 1.0]], ring, [[0, 0, -1.0]]]).astype(np.float64)
    idx = lambda i, j: 1 + i * n_phi + (j % n_phi)
    F = []
    for j in range(n_phi):
        F.append([0, idx(0, j + 1), idx(0, j)][::-1])
    for i in range(n_theta - 2):
        for j in range(n_phi):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
            F.append([a, c, b]); F.append([b, c, d])
    last = len(V) - 1
    for j in range(n_phi):
        F.append([idx(n_theta - 2, j), last, idx(n_theta - 2, j + 1)])
    F = np.asarray(F, np.int32)
    # outward orientation check
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.einsum('ij,ij->i', np.cross(b - a, c - a), (a + b + c) / 3)
    if (out < 0).mean() > 0.5:
        F = F[:, ::-1]
    assert (np.einsum('ij,ij->i', np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), V[F].mean(1)) > 0).all()
    vert = np.zeros(len(V), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')])
    for k, n in enumerate(('x', 'y', 'z')):
        vert[n] = V[:, k]; vert['n' + n] = V[:, k]
    face = np.zeros(len(F), dtype=[('n', 'u1'), ('i', '<i4', (3,))])
    face['n'] = 3; face['i'] = F
    hdr = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n'
           f'property float nx\nproperty float ny\nproperty float nz\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(hdr.encode('ascii')); f.write(vert.tobytes()); f.write(face.tobytes())
    return {'vertices': int(len(V)), 'faces': int(len(F)), 'n_theta': n_theta, 'n_phi': n_phi}


SPHERE_PLY = ROOT / 'scene' / 'uv_sphere_128x256.ply'
QUAD_PLY = ROOT / 'scene' / 'quad.ply'


def quad_ply(path):
    """Mitsuba 'rectangle' as a mesh: [-1, 1]^2 at z = 0, normal +z (two triangles, counter-clockwise)."""
    V = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32)
    F = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    vert = np.zeros(4, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    vert['x'], vert['y'], vert['z'] = V[:, 0], V[:, 1], V[:, 2]
    face = np.zeros(2, dtype=[('n', 'u1'), ('i', '<i4', (3,))]); face['n'] = 3; face['i'] = F
    hdr = ('ply' + chr(10) + 'format binary_little_endian 1.0' + chr(10) + 'element vertex 4' + chr(10) + 'property float x' + chr(10)
           + 'property float y' + chr(10) + 'property float z' + chr(10) + 'element face 2' + chr(10)
           + 'property list uchar int vertex_indices' + chr(10) + 'end_header' + chr(10))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(hdr.encode('ascii')); f.write(vert.tobytes()); f.write(face.tobytes())


# ---------------------------------------------------------------------------------------------------------------- scene
def scene_dict(mi, T, mat, light=None, sphere='mesh', with_sensor=True):
    """render81.scene_dict(mat, light, pose 0). sphere='mesh': every surface a triangle mesh (floor / wall as two-triangle quads, sphere
    as the fixed UV-sphere PLY) for the ECRO adapter and all new arms; sphere='analytic': the original WL84 scene (analytic rectangle /
    sphere), used only for the tessellation gate."""
    white = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': 0.8}}
    c, r = SPHERE['center'], SPHERE['radius']
    sph = ({'type': 'ply', 'filename': str(SPHERE_PLY), 'face_normals': False} if sphere == 'mesh' else {'type': 'sphere'})
    sph.update({'to_world': T.translate(list(c)) @ T.scale(r), 'bsdf': mat})
    rect = {'type': 'ply', 'filename': str(QUAD_PLY), 'face_normals': True} if sphere == 'mesh' else {'type': 'rectangle'}
    d = {'type': 'scene',
         'floor': dict(rect, to_world=T.scale([3, 3, 1]), bsdf=white),
         'wall': dict(rect, to_world=T.translate([0, 1.5, 1.5]) @ T.rotate([1, 0, 0], 90) @ T.scale([3, 1.5, 1]), bsdf=white),
         'sphere': sph,
         'cube': {'type': 'cube', 'to_world': T.translate([0.45, 0.1, 0.25]) @ T.rotate([0, 0, 1], 35) @ T.scale(0.25), 'bsdf': mat}}
    if light is not None:
        pos, I = LIGHTS[light]
        d['light'] = {'type': 'point', 'position': pos, 'intensity': {'type': 'rgb', 'value': I}}
    if with_sensor:      # only supplies the 256 x 256 film size to the WL102 adapter; rays come from the WL81 camera formula
        d['sensor'] = {'type': 'perspective', 'fov': CAM['fov'], 'fov_axis': 'x',
                       'to_world': T.look_at(origin=list(CAM['origin']), target=list(CAM['target']), up=[0, 0, 1]),
                       'film': {'type': 'hdrfilm', 'width': RES, 'height': RES, 'rfilter': {'type': 'box'}}}
    return d


def cam_basis():
    o = np.array(CAM['origin']); t = np.array(CAM['target'])
    fwd = t - o; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return o, fwd, right, up, math.tan(math.radians(CAM['fov']) / 2)
