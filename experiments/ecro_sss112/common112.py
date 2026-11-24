"""WL112 configuration, fixtures, views, sequences, geometric validity and region masks (numpy only; no renderer import)."""
import copy, hashlib, json, math, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROTOCOL = HERE / 'protocol' / 'sss112.json'
P = json.loads(PROTOCOL.read_text()) if PROTOCOL.exists() else {}
R_SPHERE = 0.4
LUMW = np.array([0.2126, 0.7152, 0.0722])
# development mode (throwaway; never evidence): offset seeds and reduced sizes
DEV = os.environ.get('WL112_DEV') == '1'
SO = 5000000 if DEV else 0
def dv(full, dev):
    return dev if DEV else full
RC = dv(P.get('stage_A', {}).get('replicas_C', 8), 2) if P else 8
RN = dv(P.get('stage_A', {}).get('replicas_N', 4), 2) if P else 4
RES = 512
FOV = 32.0
EDGES = [0, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def jd(x, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=1, default=float))
    tmp.replace(path)


def wl100_test(index):
    return json.loads((REPO / 'results/ecro_material100/configs.json').read_text())['sss']['test'][index]


# ------------------------------------------------------------------------------------------------ fixtures
def fixture(name):
    """F0 = WL100 SSS test configuration 0 (WL101 cfg0), light 0. F1 = WL100 test configuration 2 boxes / light 0 with the sphere
    moved to (0.80, 0.75); fixed from geometry before any radiance."""
    if name == 'F0':
        c = wl100_test(0)
        g = copy.deepcopy(c['geom']); g['S'] = {'c': [0.9, 0.4, 0.4]}
        return g, c['lights'][0]
    if name == 'F1':
        c = wl100_test(2)
        g = copy.deepcopy(c['geom']); g['S'] = {'c': [0.80, 0.75, 0.40]}
        return g, c['lights'][0]
    raise KeyError(name)


def with_sphere(geom, x, y, gap):
    g = copy.deepcopy(geom); g['S'] = {'c': [float(x), float(y), R_SPHERE + float(gap)]}
    return g


# ------------------------------------------------------------------------------------------------ cameras
V0_CAM = {'origin': [0.9, -2.4, 1.05], 'target': [0.9, 0.4, 0.55], 'fov_deg': FOV}
WL101G_HELD_OUT = [('H1', -45, 2.2, 18), ('H2', 35, 1.75, 20), ('H3', 165, 1.45, 22), ('H4', -150, 1.6, 22), ('H5', 0, 1.6, 50)]


def rule_cam(target_xy, az, d, el, tz=0.35):
    t = np.array([target_xy[0], target_xy[1], tz])
    a, e = math.radians(az), math.radians(el)
    o = t + np.array([d * math.sin(a), -d * math.cos(a), d * math.tan(e)])
    return {'origin': [float(v) for v in o], 'target': t.tolist(), 'fov_deg': FOV}


def views(fix):
    """F0: the WL101G six-camera setup (V0 + H1..H5). F1: three cameras from the same rule about the new support point."""
    if fix == 'F0':
        out = {'V0': dict(V0_CAM)}
        for vid, az, d, el in WL101G_HELD_OUT:
            out[vid] = rule_cam((0.9, 0.4), az, d, el)
        return out
    if fix == 'F1':
        return {'W0': rule_cam((0.80, 0.75), 0, 2.8, 14), 'W1': rule_cam((0.80, 0.75), -45, 2.2, 18),
                'W2': rule_cam((0.80, 0.75), 35, 1.75, 20)}
    raise KeyError(fix)


# ------------------------------------------------------------------------------------------------ dynamic sequences (Stage E)
def sequences():
    """Rigid sequences; every state is a full geometry (boxes / light from the fixture). Chosen from geometry only."""
    g0, l0 = fixture('F0'); g1, l1 = fixture('F1')
    A = [0.0, 0.01, 0.04, 0.10, 0.04, 0.01, 0.0, 0.04]
    B = [0.90, 0.80, 0.70, 0.80, 0.90]
    C = [0.0, 0.03, 0.0, 0.08, 0.0]
    return {
        'SEQ_A_lift': {'light': l0, 'views': ['V0', 'H2'], 'fixture': 'F0',
                       'states': [with_sphere(g0, 0.9, 0.4, z) for z in A], 'label': [f'gap {100 * z:g} cm' for z in A],
                       'revisits': [[0, 6], [2, 7], [1, 5], [2, 4]]},
        'SEQ_B_slide': {'light': l0, 'views': ['V0', 'H1'], 'fixture': 'F0',
                        'states': [with_sphere(g0, x, 0.4, 0.0) for x in B], 'label': [f'x {x:.2f} m' for x in B],
                        'revisits': [[0, 4], [1, 3]]},
        'SEQ_C_independent': {'light': l1, 'views': ['W0', 'W2'], 'fixture': 'F1',
                              'states': [with_sphere(g1, 0.80, 0.75, z) for z in C], 'label': [f'gap {100 * z:g} cm' for z in C],
                              'revisits': [[0, 2], [0, 4]]},
    }


def seq_view_cams(seq):
    return views(seq['fixture'])


# ------------------------------------------------------------------------------------------------ geometric validity
def box_axes(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def obb_point_distance(box, p):
    ax = box_axes(box['yaw']); d = (np.asarray(p) - np.asarray(box['c'])) @ ax.T
    q = np.clip(d, -np.asarray(box['h']), np.asarray(box['h']))
    return float(np.linalg.norm(d - q))


def box_corners(box):
    ax = box_axes(box['yaw']); c = np.asarray(box['c']); h = box['h']
    return np.array([c + sx * h[0] * ax[0] + sy * h[1] * ax[1] + sz * h[2] * ax[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def obb_overlap(a, b, margin):
    """Separating-axis test for two yawed boxes (z-aligned) with a margin."""
    A, B = box_axes(a['yaw']), box_axes(b['yaw'])
    ca, cb = np.asarray(a['c']), np.asarray(b['c'])
    ha, hb = np.asarray(a['h']) + margin / 2, np.asarray(b['h']) + margin / 2
    axes = [A[0], A[1], B[0], B[1], np.array([0, 0, 1.0])]
    for ax in axes:
        ra = sum(ha[k] * abs(A[k] @ ax) for k in range(3)); rb = sum(hb[k] * abs(B[k] @ ax) for k in range(3))
        if abs((cb - ca) @ ax) > ra + rb:
            return False
    return True


def geometry_valid(g, clearance=0.05):
    sc = np.asarray(g['S']['c'])
    if sc[2] < R_SPHERE - 1e-12 or abs(sc[0]) > 2 - R_SPHERE - 0.05 or abs(sc[1]) > 2 - R_SPHERE - 0.05:
        return False
    for b in 'AB':
        cs = box_corners(g[b])
        if cs[:, 0].min() < -1.9 or cs[:, 0].max() > 1.9 or np.abs(cs[:, 1]).max() > 1.9 or cs[:, 2].min() < -1e-9 or cs[:, 2].max() > 2.9:
            return False
        if obb_point_distance(g[b], sc) < R_SPHERE + clearance:
            return False
    return not obb_overlap(g['A'], g['B'], clearance)


def light_valid(pos, g):
    pos = np.asarray(pos)
    if np.linalg.norm(pos - np.asarray(g['S']['c'])) < R_SPHERE + 0.15:
        return False
    return all(obb_point_distance(g[b], pos) >= 0.15 for b in 'AB')


# ------------------------------------------------------------------------------------------------ training distribution (Stage C)
def train_config(split, index):
    D = P['stage_C']['distribution']
    rng = np.random.default_rng([P['stage_C']['splits'][split]['seed'] + SO, index])
    nl = P['stage_C']['splits'][split]['lights_per_geometry']
    excl = exclusion_states()
    while True:
        A = {'c': [float(rng.uniform(-.35, .05)), float(rng.uniform(.25, .75)), .75], 'h': [.22, .35, .50], 'yaw': float(rng.uniform(0, .4))}
        B = {'c': [float(rng.uniform(1.5, 1.7)), float(rng.uniform(.8, 1.3)), .35], 'h': [.15, .18, .35], 'yaw': float(rng.uniform(0, .3))}
        x = float(rng.uniform(*D['sphere_x'])); y = float(rng.uniform(*D['sphere_y']))
        gap = 0.0 if rng.random() < D['p_contact'] else float(rng.uniform(*D['gap_m']))
        g = {'A': A, 'B': B, 'S': {'c': [x, y, R_SPHERE + gap]}}
        if not geometry_valid(g, D['clearance_m']):
            continue
        if any(np.linalg.norm(np.asarray(g['S']['c']) - np.asarray(e['S']['c'])) < D['exclusion_sphere_m'] and
               all(np.linalg.norm(np.asarray(g[b]['c']) - np.asarray(e[b]['c'])) < D['exclusion_box_m'] for b in 'AB') for e in excl):
            continue
        lights = []
        while len(lights) < nl:
            pos = [float(rng.uniform(.35, 1.5)), float(rng.uniform(.85, 1.5)), float(rng.uniform(1.0, 1.7))]
            if light_valid(pos, g):
                lights.append([{'pos': pos, 'rgb': [float(rng.uniform(20, 32))] * 3}])
        return {'split': split, 'index': index, 'geom': g, 'lights': lights}


def exclusion_states():
    out = [fixture('F0')[0], fixture('F1')[0]]
    for s in sequences().values():
        out += s['states']
    return out


# ------------------------------------------------------------------------------------------------ region masks
def masks(gb, geom, light_pos):
    """WL101A region definitions generalised to the current sphere centre and light (view-independent world definitions)."""
    valid, sid, pos = gb['ok'], gb['sid'], gb['p']
    center = np.asarray(geom['S']['c'])
    sphere = valid & (sid == 6)
    floor = valid & (sid == 0)
    pad = np.pad(sphere, 4, constant_values=False)
    interior = np.logical_and.reduce([pad[y:y + RES, x:x + RES] for y in range(9) for x in range(9)])
    boundary = sphere & ~interior
    light = np.asarray(light_pos)
    to_light = light - pos; to_center = center - pos
    t = np.sum(to_center * to_light, -1) / np.sum(to_light ** 2, -1)
    closest = to_center - t[..., None] * to_light
    shadow = (t > 0) & (t < 1) & (np.sum(closest ** 2, -1) <= R_SPHERE ** 2)
    r = np.linalg.norm(pos[..., :2] - center[:2], axis=-1)
    contact = r <= R_SPHERE
    cs = floor & (shadow | contact)
    reg = {'full': np.ones_like(valid), 'sphere': sphere, 'sphere_interior': interior, 'sphere_boundary': boundary,
           'floor': floor, 'contact_shadow': cs,
           'near': floor & (r < .18), 'flank': floor & (r >= .26) & (r < .40),
           'floor_outside': floor & (r >= .50) & ~cs,
           'far_unchanged': valid & (np.linalg.norm(pos - center, axis=-1) >= 1.0)}
    bands = [floor & (r >= a) & (r < b) for a, b in zip(EDGES, EDGES[1:])]
    return reg, bands


def contact_valid(reg):
    return bool(reg['near'].sum() >= P.get('evaluation', {}).get('contact_valid_near_px', 200) and
                reg['flank'].sum() >= P.get('evaluation', {}).get('contact_valid_flank_px', 1000))
