"""WL96 d2 scratch: numpy surrogate of CK2 S1 -> box-A panel-face exactly-2-event delivery (no occlusion; ER numpy port).
Not repository evidence; used only to size lobe-state resolution, estimator bias/variance and quadrature cost."""
import json, math, sys, time
import numpy as np
from er_np import ER
R = r'c:/Projects/ReST-GI'
mats = np.load(R + '/experiments/ecro86/data/mats86.npz')['states']
er = ER(mats[1])
import os
I = np.array([30.0, 30.0, 30.0]); LP = np.array([float(v) for v in os.environ.get("LP", "-1.45,-0.6,1.7").split(",")])
# panel: plane x=-1.5, y in [-0.3, 0.9], z in [0.3, 1.7]; 6 x 7 patches (0.2 m); frame t=Y, b=Z, n=X
PY0, PY1, PZ0, PZ1, PX = -0.3, 0.9, 0.3, 1.7, -1.5
NU, NV = 6, 7
# box A panel-facing face (-x'): c - h0 a0, axes a2 (z, half .45), a1 (half .45), normal -a0
yaw = 0.15; a0 = np.array([math.cos(yaw), math.sin(yaw), 0]); a1 = np.array([-math.sin(yaw), math.cos(yaw), 0]); a2 = np.array([0, 0, 1.0])
cA = np.array([-1.08, 0.6, 1.0]); fc = cA - 0.2 * a0; fn = -a0; fe1, fe2 = a2, a1      # face frame t=a2, b=a1, n=-a0
NZ, NPHI, KAP = 4, 16, 8; DOM = 2 * math.pi / 64

def L1(y, vout):
    """once-scattered radiance [.., 3] leaving panel points y [..,3] in world directions vout [..,3]."""
    dl = LP - y; d2 = (dl * dl).sum(-1); wl = dl / np.sqrt(d2)[..., None]
    loc = lambda w: np.stack([w[..., 1], w[..., 2], w[..., 0]], -1)
    return er.eval(loc(vout), loc(wl)) * (I / d2[..., None])

def recv_points(m):
    """m x m stratified points on each of the 9 face patches -> [9, m*m, 3]."""
    pts = []
    for iv in range(3):
        for iu in range(3):
            g = (np.arange(m) + 0.5) / m
            a, b = np.meshgrid(g, g, indexing='ij')
            u = -0.45 + (iu + a.ravel()) * 0.3; v = -0.45 + (iv + b.ravel()) * 0.3
            pts.append(fc + u[:, None] * fe1 + v[:, None] * fe2)
    return np.array(pts)

def hemi_dirs(nz_sub, nphi_sub, jitter=None):
    """stratified directions in the face frame: per bin nz_sub x nphi_sub cells (centre or jittered) -> world dirs [M,3], local z, bin id."""
    out = []; zs = []; bins = []
    for iz in range(NZ):
        for ip in range(NPHI):
            a = (np.arange(nz_sub) + 0.5) / nz_sub; b = (np.arange(nphi_sub) + 0.5) / nphi_sub
            A, B = np.meshgrid(a, b, indexing='ij'); A = A.ravel(); B = B.ravel()
            z = (iz + A) / NZ; ph = (ip + B) * 2 * math.pi / NPHI - math.pi
            r = np.sqrt(1 - z * z)
            w = r[:, None] * np.cos(ph)[:, None] * fe1 + r[:, None] * np.sin(ph)[:, None] * fe2 + z[:, None] * fn
            out.append(w); zs.append(z); bins.append(np.full(len(z), iz * NPHI + ip))
    return np.concatenate(out), np.concatenate(zs), np.concatenate(bins)

def hit_panel(o, v):
    t = (PX - o[..., 0]) / np.where(np.abs(v[..., 0]) > 1e-12, v[..., 0], 1e-12)
    y = o + t[..., None] * v
    ok = (t > 0) & (y[..., 1] >= PY0) & (y[..., 1] <= PY1) & (y[..., 2] >= PZ0) & (y[..., 2] <= PZ1)
    return ok, y

def patch_of(y):
    iu = np.clip(((y[..., 1] - PY0) / 0.2).astype(int), 0, NU - 1); iv = np.clip(((y[..., 2] - PZ0) / 0.2).astype(int), 0, NV - 1)
    return iu + NU * iv, iu, iv

def panel_bin(w):
    x, yv, z = w[..., 1], w[..., 2], w[..., 0]
    iz = np.clip(np.floor(z * NZ), 0, NZ - 1).astype(int); ip = np.clip(np.floor((np.arctan2(yv, x) + math.pi) * NPHI / (2 * math.pi)), 0, NPHI - 1).astype(int)
    return iz * NPHI + ip

def gather(pts, dirs, zloc, value_fn, wscale):
    """irradiance per receiver point: sum over dirs of value(hit) * z * wscale."""
    E = np.zeros(pts.shape[:-1] + (3,))
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            ok, y = hit_panel(pts[i, j][None], dirs)
            if ok.any():
                val = value_fn(y[ok], -dirs[ok])
                E[i, j] = (val * (zloc[ok] * wscale)[:, None]).sum(0)
    return E

if __name__ == '__main__':
    out = {}
    t0 = time.time()
    # ---- geometry facts
    ylc = np.array([PX, 0, 0]); img = LP.copy(); img[0] = 2 * PX - LP[0]
    out['image_source'] = img.tolist()
    # incidence cos on panel patch centres, lobe-axis (mirror direction) variation across each patch
    var = []; cosl = []
    for iv in range(NV):
        for iu in range(NU):
            c = np.array([PX, PY0 + (iu + .5) * .2, PZ0 + (iv + .5) * .2])
            cs = []
            for da in (-.1, .1):
                for db in (-.1, .1):
                    y = c + np.array([0, da, db]); m = y - img; m /= np.linalg.norm(m); cs.append(m)
            m0 = c - img; m0 /= np.linalg.norm(m0)
            var.append(max(math.acos(np.clip(m @ m0, -1, 1)) for m in cs))
            wl = LP - c; cosl.append(wl[0] / np.linalg.norm(wl))
    out['panel_incidence_cos_range'] = [float(min(cosl)), float(max(cosl))]
    out['mirror_axis_halfspread_over_patch_rad'] = {'min': float(min(var)), 'median': float(np.median(var)), 'max': float(max(var))}
    # ---- dense truth on the 9 face patches (receiver 6x6 points, hemisphere 64 bins x 24x24 directions)
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sub = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    pts = recv_points(m)
    dirs, zl, bns = hemi_dirs(sub, sub)
    wsc = DOM / (sub * sub)
    Et = gather(pts, dirs, zl, L1, wsc).mean(1)                       # [9, 3] patch-mean irradiance
    out['truth_patch_irr_sum'] = Et.sum(1).tolist()
    out['t_truth_s'] = time.time() - t0
    # convergence check of the truth at a finer direction grid on the brightest patch
    np.save('Et.npy', Et)
    print(json.dumps(out, indent=1))
