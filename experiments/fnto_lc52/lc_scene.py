"""WL52 dynamic Lambertian scene family (protocol fnto_lc52): parametric grid faces -> triangles, boundary ports (grid
cells) and source patches (groups of ports), deterministic configuration samplers and test trajectories.

Optical domains = objects: room (floor + two walls, static), boxes, articulated arms (base + two hinged links), deforming
sheets (a thin closed slab whose top follows a height field). Every domain is Lambertian with a fixed albedo, so its
intrinsic response K = rho / pi E stays exact under any rigid motion, articulation or deformation (the controlled K).

Port = one grid cell (two triangles); 4 sample points per port ON its two triangles (2 interior barycentric points each, with
that triangle's normal - origins are never below a curved cell's surface), port normal = the cell's diagonal cross product, area = the two triangle areas. Patch = up to PU x PV ports of one face.
"""
import math

import numpy as np

PORT = 0.03125          # target port edge [m] (floor 64 x 64 over 2 m)
PU = 4                  # ports per patch edge
EPS = 1e-4              # ray origin offset along the port normal [m]
SHEET_T = 0.003         # sheet slab thickness [m]
ALB_ROOM = {"floor": (0.55, 0.55, 0.55), "wall_x": (0.75, 0.25, 0.22), "wall_y": (0.25, 0.40, 0.75)}


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    cz, sz, cy, sy, cx, sx = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def res(length):
    return max(1, int(round(length / PORT)))


class Builder:
    def __init__(self):
        self.faces = []                                           # (X [nu+1, nv+1, 3], obj, kind)
        self.objs = []                                            # dict(kind, albedo)

    def obj(self, kind, albedo):
        self.objs.append({"kind": kind, "albedo": tuple(float(a) for a in albedo)}); return len(self.objs) - 1

    def face(self, X, o):
        self.faces.append((np.asarray(X, np.float64), o))

    def plane(self, o, origin, eu, ev, nu, nv):
        a, b = np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1)
        self.face(origin[None, None] + a[:, None, None] * eu[None, None] + b[None, :, None] * ev[None, None], o)

    def box(self, o, c, h, R):
        c, h = np.asarray(c, float), np.asarray(h, float)
        for a in range(3):
            for sg in (1, -1):
                b, d = ((a + 1) % 3, (a + 2) % 3) if sg > 0 else ((a + 2) % 3, (a + 1) % 3)
                e = np.eye(3); origin = c + R @ (sg * h[a] * e[a] - h[b] * e[b] - h[d] * e[d])
                self.plane(o, origin, R @ (2 * h[b] * e[b]), R @ (2 * h[d] * e[d]), res(2 * h[b]), res(2 * h[d]))

    def build(self):
        V, F, t2p, P = [], [], [], {"pos": [], "nrm": [], "area": [], "obj": [], "patch": [], "spos": [], "snrm": []}
        Q = {"cen": [], "nrm": [], "area": [], "obj": []}; nv = 0; npt = 0; npa = 0
        for X, o in self.faces:
            nu_, nv_ = X.shape[0] - 1, X.shape[1] - 1
            idx = np.arange((nu_ + 1) * (nv_ + 1)).reshape(nu_ + 1, nv_ + 1) + nv
            V.append(X.reshape(-1, 3)); nv += (nu_ + 1) * (nv_ + 1)
            a00, a10, a01, a11 = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
            F.append(np.stack([np.stack([a00, a10, a11], 1), np.stack([a00, a11, a01], 1)], 1).reshape(-1, 3))
            ncell = nu_ * nv_; t2p.append(np.repeat(np.arange(ncell) + npt, 2)); npt += ncell
            p00, p10, p01, p11 = X[:-1, :-1].reshape(-1, 3), X[1:, :-1].reshape(-1, 3), X[:-1, 1:].reshape(-1, 3), X[1:, 1:].reshape(-1, 3)
            cr = np.cross(p11 - p00, p01 - p10); nrm = cr / np.linalg.norm(cr, axis=1, keepdims=True)
            area = 0.5 * np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=1) + 0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=1)
            n1 = np.cross(p10 - p00, p11 - p00); n2 = np.cross(p11 - p00, p01 - p00)                 # the cell's two triangles
            n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-20); n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-20)
            b1, b2 = (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)                                   # 2 interior barycentric points per triangle
            sp = [b[0] * p00 + b[1] * p10 + b[2] * p11 for b in (b1, b2)] + [b[0] * p00 + b[1] * p11 + b[2] * p01 for b in (b1, b2)]
            P["pos"].append(0.25 * (p00 + p10 + p01 + p11)); P["nrm"].append(nrm); P["area"].append(area); P["obj"].append(np.full(ncell, o)); P["spos"].append(np.stack(sp, 1))
            P["snrm"].append(np.stack([n1, n1, n2, n2], 1))
            iu, iv = np.meshgrid(np.arange(nu_), np.arange(nv_), indexing="ij"); pu_n = -(-nu_ // PU); pv_n = -(-nv_ // PU)
            pat = (iu // PU) * pv_n + (iv // PU); P["patch"].append(pat.ravel() + npa)
            for k in range(pu_n * pv_n):
                m = pat.ravel() == k; w = area[m]
                Q["cen"].append((P["pos"][-1][m] * w[:, None]).sum(0) / w.sum()); nn = (nrm[m] * w[:, None]).sum(0)
                Q["nrm"].append(nn / max(np.linalg.norm(nn), 1e-12)); Q["area"].append(w.sum()); Q["obj"].append(o)
            npa += pu_n * pv_n
        S = Scene(); S.V = np.concatenate(V).astype(np.float32); S.F = np.concatenate(F).astype(np.int32); S.tri2port = np.concatenate(t2p).astype(np.int32)
        for k in P:
            setattr(S, k, np.concatenate(P[k]).astype(np.float32 if k in ("pos", "nrm", "area", "spos", "snrm") else np.int32))
        S.pcen, S.pnrm, S.parea, S.pobj = (np.asarray(Q[k], np.float32 if k != "obj" else np.int32) for k in ("cen", "nrm", "area", "obj"))
        S.objs = self.objs; S.albedo = np.asarray([self.objs[o]["albedo"] for o in S.obj], np.float32)
        return S


class Scene:
    @property
    def N(self):
        return len(self.pos)

    @property
    def M(self):
        return len(self.pcen)

    def tables(self):
        return {k: getattr(self, k) for k in ("V", "F", "tri2port", "pos", "nrm", "area", "obj", "patch", "spos", "snrm", "pcen", "pnrm", "parea", "pobj", "albedo")}


# ---------------------------------------------------------------- objects

def add_room(B):
    o = B.obj("floor", ALB_ROOM["floor"]); B.plane(o, np.array([-1.0, -1.0, 0.0]), np.array([2.0, 0, 0]), np.array([0, 2.0, 0]), res(2.0), res(2.0))
    o = B.obj("wall_x", ALB_ROOM["wall_x"]); B.plane(o, np.array([-1.0, -1.0, 0.0]), np.array([0, 2.0, 0]), np.array([0, 0, 1.0]), res(2.0), res(1.0))
    o = B.obj("wall_y", ALB_ROOM["wall_y"]); B.plane(o, np.array([-1.0, -1.0, 0.0]), np.array([0, 0, 1.0]), np.array([2.0, 0, 0]), res(1.0), res(2.0))


def add_box(B, p):
    o = B.obj("box", p["albedo"]); B.box(o, p["c"], p["h"], rot(p["yaw"], p.get("pitch", 0.0), p.get("roll", 0.0)))


def arm_parts(p):
    """Base box + two hinged links; hinge axes = local y of the base (yaw p['yaw'])."""
    R0 = rot(p["yaw"]); base_h = np.array([0.07, 0.07, 0.05]); c0 = np.array([p["x"], p["y"], 0.05 + p["gap"]])
    hinge1 = c0 + R0 @ np.array([0, 0, base_h[2] + 0.003]); L1, L2, w = p["L1"], p["L2"], 0.03
    R1 = R0 @ rot(0, p["th1"]); c1 = hinge1 + R1 @ np.array([0, 0, L1 / 2]); hinge2 = hinge1 + R1 @ np.array([0, 0, L1 + 0.004])
    R2 = R1 @ rot(0, p["th2"]); c2 = hinge2 + R2 @ np.array([0, 0, L2 / 2])
    return [(c0, base_h, R0), (c1, np.array([w, w, L1 / 2]), R1), (c2, np.array([w * 0.8, w * 0.8, L2 / 2]), R2)]


def add_arm(B, p):
    o = B.obj("arm", p["albedo"])
    for c, h, R in arm_parts(p):
        B.box(o, c, h, R)


def sheet_grid(p):
    L = p["L"]; n = res(L); a = np.linspace(-L / 2, L / 2, n + 1); X, Y = np.meshgrid(a, a, indexing="ij")
    Z = p["amp"] * np.sin(2 * math.pi * p["f1"] * X / L + p["ph1"]) * np.cos(2 * math.pi * p["f2"] * Y / L + p["ph2"])
    R = rot(p["yaw"], p.get("tilt", 0.0)); top = np.stack([X, Y, Z], -1) @ R.T + np.asarray(p["c"])
    gx, gy = np.gradient(Z, a, a); nl = np.stack([-gx, -gy, np.ones_like(Z)], -1); nl /= np.linalg.norm(nl, axis=-1, keepdims=True)
    return top, top - SHEET_T * (nl @ R.T)


def add_sheet(B, p):
    o = B.obj("sheet", p["albedo"]); top, bot = sheet_grid(p)
    B.face(top, o); B.face(bot[:, ::-1], o)                                           # bottom: reversed winding (outward down)
    for X in (np.stack([top[0], bot[0]], 1)[:, ::-1], np.stack([top[-1], bot[-1]], 1), np.stack([top[:, 0], bot[:, 0]], 1), np.stack([top[:, -1], bot[:, -1]], 1)[:, ::-1]):
        B.face(X, o)


def aabb(kind, p):
    if kind == "box":
        R = rot(p["yaw"], p.get("pitch", 0.0), p.get("roll", 0.0)); ext = np.abs(R) @ np.asarray(p["h"]); return np.asarray(p["c"]) - ext, np.asarray(p["c"]) + ext
    if kind == "arm":
        lo, hi = [], []
        for c, h, R in arm_parts(p):
            ext = np.abs(R) @ h; lo.append(c - ext); hi.append(c + ext)
        return np.min(lo, 0), np.max(hi, 0)
    top, bot = sheet_grid(p); P = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3)]); return P.min(0), P.max(0)


def build(cfg):
    B = Builder(); add_room(B)
    for kind, p in cfg["objects"]:
        {"box": add_box, "arm": add_arm, "sheet": add_sheet}[kind](B, p)
    S = B.build(); S.cfg = cfg; return S


# ---------------------------------------------------------------- samplers

def rand_albedo(rng):
    return tuple(np.round(rng.uniform(0.15, 0.85, 3), 3))


def sample_box(rng):
    h = rng.uniform(0.05, 0.2, 3); yaw = rng.uniform(0, 2 * math.pi); mode = rng.random()
    if mode < 0.55:                                               # resting near the floor (contact gap 0.5 mm - 2 cm)
        z = h[2] + (rng.uniform(0.0005, 0.005) if rng.random() < 0.5 else rng.uniform(0.005, 0.02)); pitch = roll = 0.0
    else:                                                         # elevated / tilted
        z = rng.uniform(0.1, 0.6); pitch, roll = rng.uniform(-0.6, 0.6, 2)
    xy = rng.uniform(-0.85, 0.85, 2)
    if rng.random() < 0.2:                                        # near a wall (gap 1 mm - 1 cm)
        k = rng.integers(2); R = rot(yaw, pitch, roll); ext = (np.abs(R) @ h)[k]; xy[k] = -1.0 + ext + rng.uniform(0.001, 0.01)
    return {"c": [float(xy[0]), float(xy[1]), float(z)], "h": h.tolist(), "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll), "albedo": rand_albedo(rng)}


def sample_arm(rng):
    return {"x": float(rng.uniform(-0.7, 0.7)), "y": float(rng.uniform(-0.7, 0.7)), "yaw": float(rng.uniform(0, 2 * math.pi)), "gap": float(rng.uniform(0.0005, 0.005)),
            "L1": float(rng.uniform(0.25, 0.4)), "L2": float(rng.uniform(0.2, 0.35)), "th1": float(rng.uniform(-1.2, 1.2)), "th2": float(rng.uniform(-2.0, 2.0)), "albedo": rand_albedo(rng)}


def sample_sheet(rng):
    return {"c": [float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.6, 0.6)), float(rng.uniform(0.15, 0.5))], "L": float(rng.uniform(0.4, 0.7)), "amp": float(rng.uniform(0.0, 0.1)),
            "f1": float(rng.uniform(0.5, 1.5)), "f2": float(rng.uniform(0.0, 1.0)), "ph1": float(rng.uniform(0, 2 * math.pi)), "ph2": float(rng.uniform(0, 2 * math.pi)),
            "yaw": float(rng.uniform(0, 2 * math.pi)), "tilt": float(rng.uniform(-0.4, 0.4)), "albedo": rand_albedo(rng)}


def inside_room(lo, hi):
    return lo[0] > -1.0 + 5e-4 and lo[1] > -1.0 + 5e-4 and lo[2] > 2e-4 and hi[0] < 0.98 and hi[1] < 0.98 and hi[2] < 0.95


def place(rng, kinds, tries=400):
    objs, boxes = [], []
    for kind in kinds:
        for _ in range(tries):
            p = {"box": sample_box, "arm": sample_arm, "sheet": sample_sheet}[kind](rng); lo, hi = aabb(kind, p)
            if inside_room(lo, hi) and all(np.any(lo > h2 + 1e-3) or np.any(hi < l2 - 1e-3) for l2, h2 in boxes):
                objs.append((kind, p)); boxes.append((lo, hi)); break
    return objs


def sample_config(seed, n_box=(1, 4), p_arm=0.7, p_sheet=0.7, n_arm=None, n_sheet=None):
    rng = np.random.default_rng(seed)
    nb = int(rng.integers(n_box[0], n_box[1] + 1)); na = n_arm if n_arm is not None else int(rng.random() < p_arm); ns = n_sheet if n_sheet is not None else int(rng.random() < p_sheet)
    kinds = ["arm"] * na + ["sheet"] * ns + ["box"] * nb
    return {"seed": int(seed), "objects": place(rng, kinds), "sun": sample_sun(rng)}


def sample_sun(rng):
    el, az = rng.uniform(math.radians(25), math.radians(70)), rng.uniform(0, 2 * math.pi)
    return [float(math.cos(el) * math.cos(az)), float(math.cos(el) * math.sin(az)), float(math.sin(el))]


# ---------------------------------------------------------------- test trajectories (never sampled for training)

def trajectory(name, seed, n=30):
    rng = np.random.default_rng(seed); base = sample_config(seed, n_box=(1, 2), p_arm=0.0, p_sheet=0.0)
    sun = base["sun"]; others = base["objects"]; frames = []
    for i in range(n):
        s = i / (n - 1)
        if name == "slide_to_wall":                               # box slides toward wall_x until 1 mm contact
            h = [0.12, 0.1, 0.1]; x = -1.0 + h[0] + 0.001 + (1 - s) * 0.5
            objs = [("box", {"c": [x, 0.2, h[2] + 0.001], "h": h, "yaw": 0.0, "albedo": (0.8, 0.7, 0.2)})]
        elif name == "arm_swing":
            objs = [("arm", {"x": 0.2, "y": 0.1, "yaw": 0.6, "gap": 0.001, "L1": 0.35, "L2": 0.3, "th1": -1.0 + 2.0 * s, "th2": 1.8 - 3.0 * s, "albedo": (0.7, 0.7, 0.75)})]
        elif name == "sheet_morph":
            objs = [("sheet", {"c": [0.0, 0.0, 0.3], "L": 0.6, "amp": 0.11 * s, "f1": 1.0, "f2": 0.5, "ph1": 2 * math.pi * s, "ph2": 0.3, "yaw": 0.4, "tilt": 0.0, "albedo": (0.85, 0.85, 0.8)})]
        elif name == "box_under_sheet":
            objs = [("sheet", {"c": [0.0, 0.0, 0.35], "L": 0.6, "amp": 0.03, "f1": 1.0, "f2": 0.0, "ph1": 0.0, "ph2": 0.0, "yaw": 0.0, "tilt": 0.0, "albedo": (0.85, 0.85, 0.8)}),
                    ("box", {"c": [-0.7 + 1.4 * s, 0.0, 0.1005], "h": [0.1, 0.1, 0.1], "yaw": 0.3, "albedo": (0.2, 0.7, 0.3)})]
        elif name == "boxes_meet":
            g = 0.001 + (1 - s) * 0.4
            objs = [("box", {"c": [-0.2 - 0.1 - g / 2, -0.3, 0.1005], "h": [0.1, 0.12, 0.1], "yaw": 0.0, "albedo": (0.8, 0.3, 0.3)}),
                    ("box", {"c": [-0.2 + 0.1 + g / 2, -0.3, 0.1505], "h": [0.1, 0.12, 0.15], "yaw": 0.0, "roll": 0.0, "albedo": (0.3, 0.3, 0.8)})]
        elif name == "lift_off":
            objs = [("box", {"c": [0.3, -0.4, 0.08 + 0.0005 + 0.4 * s], "h": [0.15, 0.15, 0.08], "yaw": 0.5, "albedo": (0.9, 0.9, 0.9)})]
        else:
            raise ValueError(name)
        keep = [o for o in others if all(np.any(aabb(o[0], o[1])[0] > aabb(k, p)[1] + 1e-3) or np.any(aabb(o[0], o[1])[1] < aabb(k, p)[0] - 1e-3) for k, p in objs)]
        frames.append({"seed": int(seed), "traj": name, "frame": i, "objects": objs + keep, "sun": sun})
    return frames


TRAJ = ["slide_to_wall", "arm_swing", "sheet_morph", "box_under_sheet", "boxes_meet", "lift_off"]
