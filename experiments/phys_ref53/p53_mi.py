"""WL53 Mitsuba 3.5.1 helpers (protocol phys_ref53): PLY io, scene / medium / sensor dicts, renders, torch <-> Dr.Jit ray
intersection (Mitsuba is used only as a ray-triangle intersector by the oracle tracer and the C caster), pixel-to-port
classification with the sensor's own ray generator.

Run with the 8dna26 venv; on LabServer63 source mi_env.sh first (OptiX user-space library fix, worklog 43).

Variant (env P53_MI_VARIANT): LabServer63's cuda_ad_rgb film accumulation is wrong (a Lambert plane under a sun, exact
0.12732 with zero variance, renders 0.115 +- 0.016 at 8192 spp; LabServer69 cuda and LabServer63 llvm are exact), while
its cuda ray intersection matches an analytic box to 9e-8 m. So every RENDER uses llvm_ad_rgb; the oracle tracer / C
caster may use cuda_ad_rgb as an intersector only (protocol, environment facts)."""
import os
import tempfile

import numpy as np
import torch

import mitsuba as mi

VARIANT = os.environ.get("P53_MI_VARIANT", "llvm_ad_rgb")
mi.set_variant(VARIANT)
import drjit as dr  # noqa: E402

DEV = "cuda"
_ON_GPU = VARIANT.startswith("cuda")
_TMP = tempfile.mkdtemp(prefix="p53_ply_")


def write_ply(path, V, F):
    V = np.ascontiguousarray(V, "<f4"); F = np.ascontiguousarray(F, "<i4")
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
                 f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(V.tobytes())
        rec = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", (3,))]); rec["n"] = 3; rec["i"] = F; f.write(rec.tobytes())


def tmp_ply(V, F, name="m"):
    p = os.path.join(_TMP, f"{name}_{os.getpid()}_{np.random.randint(1 << 30)}.ply"); write_ply(p, V, F); return p


def rgb(v):
    v = np.broadcast_to(np.asarray(v, float), (3,)); return {"type": "rgb", "value": [float(x) for x in v]}


def medium(albedo, sigma_t, g=0.0):
    d = {"type": "homogeneous", "albedo": rgb(albedo), "sigma_t": float(sigma_t), "scale": 1.0}
    if g:
        d["phase"] = {"type": "hg", "g": float(g)}
    return d


def medium_shape(V, F, med, name="med"):
    return {"type": "ply", "filename": tmp_ply(V, F, name), "face_normals": True, "bsdf": {"type": "null"}, "interior": med}


def lambert_shape(V, F, rho, name="lam"):
    return {"type": "ply", "filename": tmp_ply(V, F, name), "face_normals": True, "bsdf": {"type": "diffuse", "reflectance": rgb(rho)}}


def black_shape(V, F, name="blk"):
    return lambert_shape(V, F, 0.0, name)


def mirror_shape(V, F, name="mir"):
    return {"type": "ply", "filename": tmp_ply(V, F, name), "face_normals": True, "bsdf": {"type": "conductor", "material": "none"}}


def sun(direction, E):
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    return {"type": "directional", "direction": [float(x) for x in d], "irradiance": rgb(E)}


def ortho(origin, target, up, width, height, res_w, res_h, near=1e-5):
    T = mi.ScalarTransform4f.look_at(origin=list(map(float, origin)), target=list(map(float, target)), up=list(map(float, up)))
    T = T @ mi.ScalarTransform4f.scale([width / 2, height / 2, 1.0])
    return {"type": "orthographic", "to_world": T, "near_clip": near, "far_clip": 1e4,
            "film": {"type": "hdrfilm", "width": int(res_w), "height": int(res_h), "rfilter": {"type": "box"}, "pixel_format": "rgb", "component_format": "float32"}}


def quad_mesh(center, nrm, half, up_hint=(0, 0, 1)):
    c = np.asarray(center, float); n = np.asarray(nrm, float); n /= np.linalg.norm(n)
    a = np.cross(up_hint, n)
    if np.linalg.norm(a) < 1e-6:
        a = np.cross((1, 0, 0), n)
    a /= np.linalg.norm(a); b = np.cross(n, a)
    V = np.array([c - half * a - half * b, c + half * a - half * b, c + half * a + half * b, c - half * a + half * b])
    return V, np.array([[0, 1, 2], [0, 2, 3]])                 # CCW about n


def scene(shapes, emitters=(), sensors=None, integrator=None):
    d = {"type": "scene", "integrator": integrator or {"type": "volpath", "max_depth": -1, "rr_depth": 5}}
    for k, v in shapes.items():
        d[k] = v
    for i, e in enumerate(emitters):
        d[f"emitter_{i}"] = e
    for k, v in (sensors or {}).items():
        d[k] = v
    return mi.load_dict(d)


def sensor_by_id(sc, name):
    for s in sc.sensors():
        if s.id() == name:
            return s
    raise KeyError(name)


def render(sc, sensor_name, spp, seed):
    img = mi.render(sc, sensor=sensor_by_id(sc, sensor_name), spp=int(spp), seed=int(seed))
    return np.array(img, dtype=np.float64)


# ------------------------------------------------------------------ torch <-> Dr.Jit


def _f(x):
    return mi.Float(x.contiguous() if _ON_GPU else np.ascontiguousarray(x.detach().cpu().numpy(), np.float32))


def _t(a, long=False):
    out = a.torch() if _ON_GPU else torch.from_numpy(np.array(a))
    return (out.long() if long else out.float()).to(DEV).clone()


def _v3(x, cls):
    x = x.float().contiguous()
    return cls(_f(x[:, 0]), _f(x[:, 1]), _f(x[:, 2]))


class Intersector:
    """Ray-triangle intersection of one merged triangle mesh; prim index == row of F."""

    def __init__(self, V, F):
        self.scene = mi.load_dict({"type": "scene", "m": {"type": "ply", "filename": tmp_ply(V, F, "isect"), "face_normals": True}})
        self.nf = len(F)

    def __call__(self, o, d):
        if len(o) == 0:
            return torch.empty(0, device=DEV), torch.empty(0, dtype=torch.long, device=DEV)
        ray = mi.Ray3f(_v3(o, mi.Point3f), _v3(d, mi.Vector3f))
        pi = self.scene.ray_intersect_preliminary(ray)
        valid = pi.is_valid()
        t = dr.select(valid, pi.t, dr.inf)
        prim = dr.select(valid, mi.Int32(pi.prim_index), -1)
        dr.eval(t, prim)
        return _t(t), _t(prim, long=True)


def sensor_pixel_rays(sensor, offsets=((0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9))):
    """Rays the sensor generates at given sub-pixel offsets; returns list of (o, d) torch (H*W, 3), row-major (y, x)."""
    W, H = sensor.film().size()
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = []
    for ox, oy in offsets:
        px = mi.Float(((xx.ravel() + ox) / W).astype(np.float32)); py = mi.Float(((yy.ravel() + oy) / H).astype(np.float32))
        ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(px, py), mi.Point2f(0.5, 0.5))
        o = torch.stack([_t(ray.o.x), _t(ray.o.y), _t(ray.o.z)], 1)
        d = torch.stack([_t(ray.d.x), _t(ray.d.y), _t(ray.d.z)], 1)
        out.append((o, d))
    return out, (H, W)
