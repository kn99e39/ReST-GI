"""WL99A independent held-out candidates: a seeded random generator (declared before any of its outputs is rendered).

Family = the WL99 workload family (groove / channel / corner rigid bodies of thin ER plates, one point light, one thin Lambert bar between
the light and the body, 'dark' room). Every draw (accepted or rejected) is logged. Writes protocol/workloads99a_indep.json.
Rules: kinds 2 x groove (angle U[70, 110] deg, pitch U[35, 60]), 2 x channel (gap U[0.20, 0.35], height U[0.25, 0.35]), 2 x corner
(side U[0.35, 0.45]) in S1 (type A); 2 x groove in S2 (type B). Camera within 25 deg of the body's opening axis (corner: its diagonal),
distance U[2.4, 3.0] m, fov so that the body spans ~60 % of the frame; light within 40 deg of the opening axis, distance U[1.8, 2.6] m;
bar at a fraction U[0.20, 0.35] of the light distance from the body, horizontal and perpendicular to the light direction, offset
+-U[0.03, 0.10] m across its long axis (so that it shadows part of the body, not the light at the body centre), length U[0.25, 0.35] m,
half-thickness 0.012 m. (Before any render: a first draft offset the bar along its own long axis, so it almost always blocked the light
at the body centre; corrected.) T1: yaw +- U[10, 30] deg and a horizontal translation U[0.05, 0.15] m; camera, light and bar fixed.
Validity: light and camera inside x, y in [-1.9, 1.9] (camera may be at y < -2, the open side), z in [0.3, 2.9]; certificate clean at
T0 and T1; body above the floor; light not occluded from the body's centre by the bar itself (the bar may shadow part of the body).
"""
import json, math, os
import numpy as np
import scene99 as S

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(99_0901)
KINDS = [('groove', 'S1', 'A')] * 2 + [('channel', 'S1', 'A')] * 2 + [('corner', 'S1', 'A')] * 2 + [('groove', 'S2', 'B')] * 2


def opening_axis(kind, pose):
    R = S.pose_matrix(pose)[:3, :3]
    a = {'groove': [0, 0, 1], 'channel': [0, 0, 1], 'corner': [1, 1, 1]}[kind]
    a = R @ np.array(a, float); return a / np.linalg.norm(a)


def cone(axis, deg):
    """Uniform direction within deg of axis."""
    while True:
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        if math.degrees(math.acos(np.clip(v @ axis, -1, 1))) <= deg:
            return v


def body_center(kind, prm, pose):
    pts = S.body_hull_points(S.body_plates(kind, prm))
    M = S.pose_matrix(pose)
    w = pts @ M[:3, :3].T + M[:3, 3]
    return w.mean(0), w


def draw(kind, mat, typ, idx):
    if kind == 'groove':
        prm = {'L': 0.6, 'W': 0.35, 'angle': float(rng.uniform(70, 110))}
        pose = {'c': [float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.2, 0.9)), float(rng.uniform(0.3, 0.6))],
                'yaw': float(rng.uniform(0, 360)), 'pitch': float(rng.uniform(35, 60))}
        size = 0.6
    elif kind == 'channel':
        prm = {'L': 0.6, 'G': float(rng.uniform(0.20, 0.35)), 'H': float(rng.uniform(0.25, 0.35))}
        pose = {'c': [float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.2, 0.9)), 0.05], 'yaw': float(rng.uniform(0, 360)), 'pitch': 0.0}
        size = 0.6
    else:
        prm = {'A': float(rng.uniform(0.35, 0.45))}
        pose = {'c': [float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.2, 0.9)), 0.05], 'yaw': float(rng.uniform(0, 360)), 'pitch': 0.0}
        size = prm['A'] * 1.4
    ctr, hull = body_center(kind, prm, pose)
    ax = opening_axis(kind, pose)
    dc = cone(ax, 25); dist_c = rng.uniform(2.4, 3.0)
    cam_o = ctr + dc * dist_c
    fov = math.degrees(2 * math.atan(0.5 * size / 0.6 / dist_c))
    dl = cone(ax, 40); dist_l = rng.uniform(1.8, 2.6)
    lp = ctr + dl * dist_l
    frac = rng.uniform(0.20, 0.35)
    perp = np.cross(dl, [0, 0, 1.0]); perp /= max(np.linalg.norm(perp), 1e-9)
    up2 = np.cross(dl, perp); up2 /= max(np.linalg.norm(up2), 1e-9)
    bar_c = ctr + dl * dist_l * frac + up2 * (rng.choice([-1, 1]) * rng.uniform(0.03, 0.10))   # offset across the bar's long axis
    yaw_bar = math.degrees(math.atan2(perp[1], perp[0]))
    bar = {'c': [float(v) for v in bar_c], 'h': [float(rng.uniform(0.25, 0.35) / 2), 0.012, 0.012], 'yaw': float(yaw_bar)}
    sgn = 1 if rng.random() < 0.5 else -1
    dyaw = sgn * rng.uniform(10, 30); tdir = rng.uniform(0, 2 * math.pi); tlen = rng.uniform(0.05, 0.15)
    p1 = dict(pose, c=[pose['c'][0] + tlen * math.cos(tdir), pose['c'][1] + tlen * math.sin(tdir), pose['c'][2]], yaw=pose['yaw'] + dyaw)
    w = {'id': f'I{idx}_{kind}_{mat}', 'type': typ, 'role': 'independent held-out candidate (WL99A seeded generator)',
         'body': {'kind': kind, 'prm': prm, 'mat': mat}, 'T0': pose, 'T1': p1,
         'lights': [{'pos': [float(v) for v in lp], 'rgb': [30, 30, 30]}], 'boxes': [bar],
         'camera': {'origin': [float(v) for v in cam_o], 'target': [float(v) for v in ctr], 'fov_deg': float(fov)}}
    return w, hull


def valid(w, hull):
    lp = np.array(w['lights'][0]['pos']); co = np.array(w['camera']['origin'])
    inroom = lambda p, open_side: (-1.9 <= p[0] <= 1.9) and ((p[1] >= -1.9) if not open_side else True) and p[1] <= 1.9 and 0.3 <= p[2] <= 2.9
    if not inroom(lp, False) or not inroom(co, True):
        return 'out_of_room'
    for t in ('T0', 'T1'):
        cfg = {'body': dict(w['body'], pose=w[t]), 'boxes': w['boxes']}
        if not S.certificate(cfg):
            return f'certificate_{t}'
        pts = S.body_hull_points(S.body_plates(w['body']['kind'], w['body']['prm'])) @ S.pose_matrix(w[t])[:3, :3].T + S.pose_matrix(w[t])[:3, 3]
        if pts[:, 2].min() < 0.005:
            return f'below_floor_{t}'
    ctr = hull.mean(0)
    if S.seg_obb if hasattr(S, 'seg_obb') else False:
        pass
    b = w['boxes'][0]; M = S.box_matrix(b); Rb = M[:3, :3] / np.asarray(b['h'])[None, :]
    lo = (ctr - b['c']) @ Rb; le = (lp - b['c']) @ Rb; d = le - lo; d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    t0 = (-np.asarray(b['h']) - lo) / d; t1 = (np.asarray(b['h']) - lo) / d
    if np.maximum(np.minimum(t0, t1).max(), 0) <= min(np.maximum(t0, t1).min(), 1):
        return 'bar_blocks_light_at_centre'
    return 'ok'


log, out = [], []
for i, (kind, mat, typ) in enumerate(KINDS):
    while True:
        w, hull = draw(kind, mat, typ, i + 1)
        v = valid(w, hull)
        log.append({'slot': i + 1, 'kind': kind, 'result': v})
        if v == 'ok':
            out.append(w); break
json.dump({'protocol': 'rbdto99a_v2', 'generator': 'gen_indep99a.py', 'seed': 990901, 'workloads': out, 'draw_log': log},
          open(os.path.join(HERE, 'protocol', 'workloads99a_indep.json'), 'w'), indent=1)
print(len(out), 'accepted;', len(log), 'draws;', {r: sum(1 for x in log if x['result'] == r) for r in set(x['result'] for x in log)})
