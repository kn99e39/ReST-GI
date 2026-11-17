"""WL101 workloads (numpy only; imported by both stacks). See protocol/dyn101.json."""
import copy, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
P = json.loads((HERE / 'protocol' / 'dyn101.json').read_text())
CONFIGS = REPO / 'results' / 'ecro_material100' / 'configs.json'
WARM = P['timing']['warmup_events']


def _rows(case):
    return json.loads(CONFIGS.read_text())[case]['test']


def _lerp_geom(g0, g1, t):
    g = copy.deepcopy(g0)
    for b in 'AB':
        g[b]['c'] = [float((1 - t) * a + t * c) for a, c in zip(g0[b]['c'], g1[b]['c'])]
        g[b]['yaw'] = float((1 - t) * g0[b]['yaw'] + t * g1[b]['yaw'])
    return g


def workload(name):
    """-> case, start (geom, light), events [(geom, light, state_id)] with WARM warm-up events first (state ids: config index or
    'a<k>' for interpolated poses)."""
    w = P['workloads'][name]
    case = w['case']
    rows = _rows(case)
    L = rows[0]['lights'][0]
    if name == 'SEL':
        seq = [1, 0] * ((WARM + w['events']) // 2)
        start = (rows[0]['geom'], rows[0]['lights'][0], 0)
        ev = [(rows[i]['geom'], rows[i]['lights'][0], i) for i in seq]
    elif name in ('V_SSS_TOUR', 'V_NEAR_TOUR'):
        tour = [0, 2, 3, 4, 5, 6, 7, 1] if name == 'V_SSS_TOUR' else [0, 1, 2, 3, 4, 5, 6, 7]
        start = (rows[tour[-1]]['geom'], L, tour[-1])
        seq = tour[:WARM] + tour * 3                     # warm-up = first 4 states of a tour; then 3 timed tours
        ev = [(rows[i]['geom'], L, i) for i in seq]
    elif name == 'V_SSS_ANIM':
        g0, g1 = rows[2]['geom'], rows[1]['geom']
        poses = [_lerp_geom(g0, g1, k / 16) for k in range(17)]
        ids = [f'a{k}' for k in range(17)]
        order = list(range(1, 17)) + list(range(15, -1, -1)) + list(range(1, 17))     # 48 frames, ping-pong
        start = (poses[0], L, 'a0')
        order = order[:WARM] + order
        ev = [(poses[k], L, ids[k]) for k in order]
        ev = ev[: WARM + w['events']]
    else:
        raise KeyError(name)
    return case, start, ev


def tour_states(name):
    """First timed tour pass of a validation tour: [(geom, light, state_id, prev_state_id)] (quality states)."""
    case, start, ev = workload(name)
    timed = ev[WARM:]
    first = timed[:8]
    prev = [ev[WARM - 1][2]] + [s[2] for s in first[:-1]]
    return case, [(g, l, s, p) for (g, l, s), p in zip(first, prev)]


def refine(S, s):
    """econ88.refine rule on scene86 (patch refinement by factor s), applied before any context is built."""
    if s == 1.0:
        return
    rects0, faces0 = S.stationary_rects, S.box_faces
    sph0 = (S.SPH_NPHI, S.SPH_NZ)
    S.stationary_rects = lambda: [r[:5] + (max(1, int(round(r[5] * s))), max(1, int(round(r[6] * s))), r[7]) for r in rects0()]

    def faces(box, mat):
        n = max(1, int(round(3 * s)))
        return [f[:5] + (n, n, f[7]) for f in faces0(box, mat)]
    S.box_faces = faces
    S.SPH_NPHI = max(2, int(round(sph0[0] * s))); S.SPH_NZ = max(1, int(round(sph0[1] * s)))


def stats(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return None
    q = np.percentile(v, [25, 50, 75, 90, 99])
    return {'n': int(v.size), 'median': float(q[1]), 'p25': float(q[0]), 'p75': float(q[2]), 'p90': float(q[3]), 'p99': float(q[4]),
            'max': float(v.max()), 'min': float(v.min()), 'mean': float(v.mean())}


def gpu_snapshot():
    import subprocess
    try:
        u = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        a = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,name', '--format=csv,noheader'], capture_output=True, text=True,
                           timeout=20).stdout.strip().splitlines()
        return {'gpu': u, 'compute_apps': [x for x in a if 'python' in x.lower()]}
    except Exception as e:  # noqa: BLE001
        return {'error': str(e)}
