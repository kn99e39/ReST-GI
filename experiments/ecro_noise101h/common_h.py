"""WL101H (Denoising Track DN-1) common definitions: frozen inputs, ROI masks and estimator helpers.

Nothing in this module renders. The renderer is imported by run_h.py (original WL101 rt101 runtime).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL_PATH = HERE / 'protocol' / 'noise101h.json'
P = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8')) if PROTOCOL_PATH.exists() else None
ROOT100 = REPO / 'results' / 'ecro_material100'
REND100 = ROOT100 / 'sss' / 'renders'
RES = 512
CHANNELS = ('e1', 'e2', 'hl')
LUMA = np.array([0.2126, 0.7152, 0.0722])
SPHERE_C = np.array([0.9, 0.4, 0.4])
SPHERE_R = 0.4
SUPPORT = np.array([0.9, 0.4, 0.0])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def config0():
    return json.loads((ROOT100 / 'configs.json').read_text(encoding='utf-8'))['sss']['test'][0]


# --------------------------------------------------------------------------------------------- ROI masks (geometry only)
def base_masks(gbuffer_path):
    """WL101A ROIs (exact definitions of experiments/ecro_sss101a/common.py) + WL101C near / flank, from a center-ray gbuffer."""
    with np.load(gbuffer_path) as g:
        valid, sid, pos = g['ok'], g['sid'], g['p']
    cfg = config0()
    sphere = valid & (sid == 6)
    floor = valid & (sid == 0)
    pad = np.pad(sphere, 4, constant_values=False)
    interior = np.logical_and.reduce([pad[y:y + RES, x:x + RES] for y in range(9) for x in range(9)])
    light = np.asarray(cfg['lights'][0][0]['pos'])
    to_light = light - pos
    to_center = SPHERE_C - pos
    t = np.sum(to_center * to_light, -1) / np.sum(to_light ** 2, -1)
    closest = to_center - t[..., None] * to_light
    shadow = (t > 0) & (t < 1) & (np.sum(closest ** 2, -1) <= SPHERE_R ** 2)
    contact = np.sum(to_center[..., :2] ** 2, -1) <= SPHERE_R ** 2
    m = {'full': np.ones_like(valid), 'sphere': sphere, 'sphere_interior': interior, 'sphere_boundary': sphere & ~interior,
         'floor': floor, 'contact_shadow': floor & (shadow | contact)}
    front = floor & (pos[..., 1] <= SUPPORT[1])
    r = np.linalg.norm(pos[..., :2] - SUPPORT[:2], axis=-1)
    m['near'] = front & (r < 0.18)
    m['flank'] = front & (r >= 0.26) & (r < 0.40)
    return m, {'valid': valid, 'sid': sid, 'pos': pos}


def obb_distance(points, box):
    """Euclidean distance from points [...,3] to an oriented box {'c','h','yaw'} (z-up yaw)."""
    c, s = np.cos(box['yaw']), np.sin(box['yaw'])
    ax = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    q = (points - np.asarray(box['c'])) @ ax.T
    d = np.maximum(np.abs(q) - np.asarray(box['h']), 0.0)
    return np.linalg.norm(d, axis=-1)


def all_masks(gbuffer_path, cover):
    """cover: dict with 'sphere_frac' [RES,RES] and 'pure_sid' [RES,RES] (-1 if mixed) from the supersampled id buffer."""
    m, g = base_masks(gbuffer_path)
    frac, pure = cover['sphere_frac'], cover['pure_sid']
    pr = P['rois']
    m['sss_interior'] = m['sphere_interior'] & (frac >= 1.0)
    m['silhouette'] = (frac > 0.0) & (frac < 1.0)
    pos, sid = g['pos'], g['sid']
    cfg = config0()
    dsph = np.abs(np.linalg.norm(pos - SPHERE_C, axis=-1) - SPHERE_R)
    dbox = np.minimum(obb_distance(pos, cfg['geom']['A']), obb_distance(pos, cfg['geom']['B']))
    ctl = pr['control']
    edge = np.minimum.reduce([pos[..., 2], 3.0 - pos[..., 2], pos[..., 0] + 2.0, 2.0 - pos[..., 0]])
    m['control'] = (g['valid'] & (pure >= 0) & (sid == ctl['sid']) & (pure == sid) & (edge >= ctl['room_edge_margin_m'])
                    & (dsph >= ctl['min_sphere_distance_m']) & (dbox >= ctl['min_box_distance_m']) & ~m['contact_shadow'])
    return m


# --------------------------------------------------------------------------------------------- estimators (numpy reference)
def var_ddof1(x, axis=0):
    return np.var(x, axis=axis, ddof=1)


def cov_ddof1(x, y, axis=0):
    xm = x - x.mean(axis, keepdims=True)
    ym = y - y.mean(axis, keepdims=True)
    return (xm * ym).sum(axis) / (x.shape[axis] - 1)
