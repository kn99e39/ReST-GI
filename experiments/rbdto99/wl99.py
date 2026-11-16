"""WL99 shared helpers: workload -> render configurations, rigid co-motion, masks and image metrics (numpy)."""
import copy, json, math, os
import numpy as np
import scene99 as S

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'protocol', 'rbdto99.json')))
WL = json.load(open(os.path.join(HERE, 'protocol', 'workloads99.json')))['workloads']
RES = P['fixture']['resolution']


def cfg_of(w, pose, lights=None, boxes=None, camera=None):
    return {'room': 'dark', 'body': dict(w['body'], pose=pose), 'lights': copy.deepcopy(lights if lights is not None else w['lights']),
            'boxes': copy.deepcopy(boxes if boxes is not None else w['boxes']), 'camera': copy.deepcopy(camera if camera is not None else w['camera'])}


def motion(w):
    """Rigid world motion D = M1 M0^-1 of the body (T0 -> T1) and its yaw change (poses differ by yaw + translation only)."""
    p0, p1 = w['T0'], w['T1']
    assert abs(p0.get('pitch', 0.0) - p1.get('pitch', 0.0)) < 1e-12 and abs(p0.get('roll', 0.0) - p1.get('roll', 0.0)) < 1e-12
    D = S.pose_matrix(p1) @ np.linalg.inv(S.pose_matrix(p0))
    return D, p1.get('yaw', 0.0) - p0.get('yaw', 0.0)


def comove(w, camera_too):
    """T0 scene with lights, boxes (and optionally the camera) moved by the body motion; body at its T1 pose."""
    D, dyaw = motion(w)
    tp = lambda p: [float(v) for v in D[:3, :3] @ np.asarray(p, float) + D[:3, 3]]
    lights = [dict(l, pos=tp(l['pos'])) for l in w['lights']]
    boxes = [dict(b, c=tp(b['c']), yaw=b.get('yaw', 0.0) + dyaw) for b in w['boxes']]
    cam = dict(w['camera'], origin=tp(w['camera']['origin']), target=tp(w['camera']['target'])) if camera_too else w['camera']
    return cfg_of(w, w['T1'], lights, boxes, cam)


def configs(w):
    out = {'T0': cfg_of(w, w['T0']), 'T1': cfg_of(w, w['T1'])}
    if w['type'] != 'C':
        out['stale'] = comove(w, camera_too=False)
    if w.get('co_check'):
        out['co'] = comove(w, camera_too=True)
    if w.get('intrusion'):
        out['T1x'] = cfg_of(w, w['T1'], boxes=w['boxes'] + [w['intrusion']])
    return out


# ------------------------------------------------------------------------------------------------ images / metrics
def img(r, key):
    return r[key].reshape(RES, RES, 3) if r[key].ndim == 2 else r[key].reshape(RES, RES)


def lum(x):
    return x.mean(-1) if x.ndim == 3 else x


def masks(ref):
    """ref: dict of [RES,RES,3] mean images (+ 'mb' [RES,RES]). Returns ROI_body, ROI_self, ROI_sharp boolean maps."""
    tot, self_, sd = lum(ref['tot']), lum(ref['self']), lum(ref['Sd'])
    body = ref['mb'] >= 0.99
    roi_self = body & (self_ > 0) & (self_ >= 0.25 * tot)
    gy, gx = np.gradient(sd)
    g = np.hypot(gx, gy)
    loc = box_mean(sd, 3)
    rel = g / (loc + 1e-6 * max(sd[roi_self].mean() if roi_self.any() else 1.0, 1e-30))
    sharp = np.zeros_like(roi_self)
    if roi_self.sum() > 10:
        thr = np.quantile(rel[roi_self], 0.90)
        sharp = roi_self & (rel >= thr)
    return body, roi_self, sharp


def box_mean(x, r):
    k = 2 * r + 1
    p = np.pad(x, r, mode='edge')
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def window_stats(x, r=3):
    """p95 - p5 and max |grad| in (2r+1)^2 windows (exact, via stacked shifts)."""
    H, W = x.shape
    p = np.pad(x, r, mode='edge')
    stack = np.stack([p[i:i + H, j:j + W] for i in range(2 * r + 1) for j in range(2 * r + 1)], 0)
    rng = np.quantile(stack, 0.95, axis=0) - np.quantile(stack, 0.05, axis=0)
    gy, gx = np.gradient(x)
    g = np.hypot(gx, gy)
    pg = np.pad(g, r, mode='edge')
    gmax = np.max(np.stack([pg[i:i + H, j:j + W] for i in range(2 * r + 1) for j in range(2 * r + 1)], 0), 0)
    return rng, gmax, stack.mean(0)


def rel_l2(a, b, m):
    """||a - b|| / ||b|| over mask m (all channels)."""
    return float(np.sqrt(((a - b)[m] ** 2).sum() / max((b[m] ** 2).sum(), 1e-300)))


def rel_noise(A, B, m):
    """Per-pixel noise of the mean of two halves relative to the signal: ||A - B|| / (2 ||(A + B) / 2||)."""
    return float(np.sqrt(((A - B)[m] ** 2).sum()) / (2 * max(np.sqrt((((A + B) / 2)[m] ** 2).sum()), 1e-300)))
