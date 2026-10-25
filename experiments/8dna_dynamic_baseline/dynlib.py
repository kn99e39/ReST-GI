"""Shared helpers for the 8DNA dynamic-scene baseline (worklog 43, protocol 8dna_dyn_v1).

The official 8dna26 tree is imported, never modified. Runs on LabServer63 with the upstream pins
(torch 2.3.1, Mitsuba 3.5.1, DrJit 0.4.6) after `source server/mi_env.sh`. Every entry point ends with
finish(), i.e. os._exit after its outputs are written: DrJit 0.4.6 can crash inside jit_shutdown()'s
variable-leak report at interpreter exit (worklog 43 server diagnosis; also worklog 22 on the PC).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("EDNA_UPSTREAM", str(Path.home() / "NeuralShaderDemo" / "external" / "8dna26")))
OUT = Path(os.environ.get("EDNA_OUT", str(HERE.parents[1] / "results" / "8dna_dynamic_baseline")))
PROTOCOL_PATH = HERE / "protocol" / "dyn_v1.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
OFFICIAL_COMMIT = "4a2157ca24e506c5ac0831f27d656ecc50a64f07"
RED = PROTOCOL["box"]["red"]
BLACK = PROTOCOL["box"]["black"]
# EDNA_DRYRUN=1: code-path check of every stage at 32x32 and <= 16 spp, never evidence
DRY = os.environ.get("EDNA_DRYRUN") == "1"
MAX_LANES = 512 * 512 * 16  # per mi.render launch on the 12 GB RTX 3080 Ti


def R(res: int) -> int:
    return 32 if DRY else res


def S(spp: int, chunk: int) -> tuple[int, int]:
    if not DRY:
        return spp, chunk
    s = min(spp, 16)
    return s, min(chunk, 4, s)

mi = None
dr = None


# --- runtime -------------------------------------------------------------------

def flags() -> dict:
    """Stage-0 decisions (VCallRecord, reference integrators); defaults are the declared first choices."""
    p = OUT / "stage0" / "flags.json"
    base = {"VCallRecord": False, "ref_volume": "volpath", "ref_surface": "path"}
    if p.exists():
        base.update(json.loads(p.read_text()))
    return base


def init(vcall: bool | None = None):
    """Demo-notebook setup: cuda_ad_rgb, LoopRecord off, official integrators registered."""
    global mi, dr
    if mi is not None:
        return mi, dr
    if vcall is None:
        vcall = flags()["VCallRecord"]
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    import mitsuba as _mi

    _mi.set_variant("cuda_ad_rgb")
    import drjit as _dr

    import models.integrator  # noqa: F401  registers neuralpath / neuralvolpath

    _dr.set_flag(_dr.JitFlag.LoopRecord, False)
    _dr.set_flag(_dr.JitFlag.VCallRecord, bool(vcall))
    mi, dr = _mi, _dr
    return mi, dr


def finish(code: int = 0) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def sync() -> None:
    import torch

    dr.sync_thread()
    torch.cuda.synchronize()


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def environment() -> dict:
    import torch

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {
        "host": platform.node(), "os": platform.platform(), "python": sys.version.split()[0],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "mitsuba": mi.__version__, "drjit": dr.__version__,
        "gpu": smi, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "jit_flags": {"LoopRecord": bool(dr.flag(dr.JitFlag.LoopRecord)), "VCallRecord": bool(dr.flag(dr.JitFlag.VCallRecord))},
        "project_commit": git("rev-parse", "HEAD"), "project_dirty": (git("status", "--porcelain") or "").splitlines(),
        "upstream": str(UPSTREAM), "upstream_commit": subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                                                                     capture_output=True, text=True).stdout.strip(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt(name: str) -> Path:
    return UPSTREAM / "checkpoints" / f"{name}.ckpt"


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v))

    path.write_text(json.dumps(payload, indent=1, default=default))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# --- GPU sharing ----------------------------------------------------------------

def gpu_apps() -> list[tuple[int, int]]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            apps.append((int(parts[0]), int(parts[1]) if parts[1].isdigit() else 0))
    return apps


def yield_gpu(limit_mib: int = 500) -> float:
    """Protocol yield rule: wait while another process holds more than limit_mib of GPU memory (not in a dry run,
    which deliberately shares the GPU with whatever is running)."""
    waited = 0.0
    if DRY:
        return waited
    while any(pid != os.getpid() and mem > limit_mib for pid, mem in gpu_apps()):
        if waited == 0.0:
            print("  yielding GPU to another process", flush=True)
        time.sleep(60)
        waited += 60
    return waited


class MemWatch:
    """Max GPU memory of this process from nvidia-smi, sampled every 2 s."""

    def __init__(self):
        self.max_mib = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            for pid, mem in gpu_apps():
                if pid == os.getpid():
                    self.max_mib = max(self.max_mib, mem)
            self._stop.wait(2.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)


# --- learned state ----------------------------------------------------------------

def neural_integrator(assets: dict[str, str]):
    """neuralpath of the demo notebook with one asset_<instance id> key per learned asset."""
    d = {"type": "neuralpath", "max_depth": 10, "rr_depth": 5, "device": 0, "model_type": "8dna"}
    for inst, name in assets.items():
        d[f"asset_{inst}"] = str(ckpt(name))
    return mi.load_dict(d)


def attach(integ, scene) -> None:
    """Point the integrator at a new scene (prepare_scene caches ShapePtrs; a stale cache silently renders plain PT)."""
    integ.is_prepared = False
    integ.prepare_scene(scene)
    if len(integ.asset_shapes) != len(integ.asset_models):
        raise RuntimeError(f"asset ids {integ.asset_names} not all found in the scene")


def model_ids(integ) -> dict[int, str]:
    """id() of the EightDNA module behind each (torch.compile-wrapped) asset model -> checkpoint name."""
    return {id(getattr(m, "_orig_mod", m)): os.path.basename(str(n)) for n, m in zip(integ.asset_names, integ.asset_models)}


def per_model(counters: Counters, integ) -> dict:
    ids = model_ids(integ)
    out = {}
    for k, v in counters.n.items():
        if "@" in k:
            base, mid = k.split("@")
            out.setdefault(ids.get(int(mid), mid), {})[base] = v
    return out


def param_digest(integ) -> dict:
    out = {}
    for name, m in zip(integ.asset_names, integ.asset_models):
        h = hashlib.sha256()
        sd = m.state_dict()
        n = 0
        for k in sorted(sd):
            t = sd[k].detach().cpu().contiguous()
            h.update(k.encode())
            h.update(t.numpy().tobytes())
            n += t.numel()
        out[name] = {"sha256": h.hexdigest(), "values": n}
    return out


# --- query / ray counters ----------------------------------------------------------

class Counters:
    def __init__(self):
        self.n: dict[str, int] = {}

    def add(self, key: str, value) -> None:
        self.n[key] = self.n.get(key, 0) + int(value)


def _lanes(mask) -> int:
    if isinstance(mask, bool):
        return -1
    return int(mask.torch().sum().item())


def instrument(counters: Counters):
    """Count learned-lobe lanes and Python-level scene ray calls. Returns an undo callable.

    Only the Python neural integrator goes through these hooks; the C++ reference integrators are timed only.
    """
    import models.eight_dna as E

    undo = []
    for name, key in (("eval_asset", "learned_nee_lanes"), ("sample_asset", "learned_sample_lanes")):
        orig = getattr(E.EightDNA, name)

        def wrap(self, scene, si, m2w, bbox, active_asset, *a, _o=orig, _k=key, **kw):
            n = _lanes(active_asset)
            counters.add(_k, n)
            counters.add(f"{_k}@{id(self)}", n)  # per model; map with model_ids()
            return _o(self, scene, si, m2w, bbox, active_asset, *a, **kw)

        setattr(E.EightDNA, name, wrap)
        undo.append((E.EightDNA, name, orig))

    def width(x) -> int:
        return int(dr.width(x))

    for name in ("ray_intersect", "ray_intersect_preliminary", "ray_test", "sample_emitter_direction"):
        try:
            orig = getattr(mi.Scene, name)

            def wrap(self, *a, _o=orig, _k=name, **kw):
                first = a[0] if a else None
                try:
                    counters.add(f"{_k}_width", width(first.o if hasattr(first, "o") else first.p))
                except Exception:
                    pass
                return _o(self, *a, **kw)

            setattr(mi.Scene, name, wrap)
            undo.append((mi.Scene, name, orig))
        except (AttributeError, TypeError) as e:  # pybind class not patchable
            counters.add(f"{name}_unpatchable", 1)
            print(f"  counter hook {name} unavailable: {e}", flush=True)

    def restore():
        for cls, name, orig in reversed(undo):
            setattr(cls, name, orig)

    return restore


# --- scenes -------------------------------------------------------------------------

def _T():
    return mi.ScalarTransform4f


def look_at(theta_deg: float, phi_deg: float, r: float, tgt) -> object:
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    src = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)]) * r + np.asarray(tgt, float)
    return _T().look_at(origin=list(map(float, src)), target=list(map(float, tgt)), up=[0, 1, 0])


def official(asset: str, fn: str, res: int) -> dict:
    import importlib

    return getattr(importlib.import_module(f"scenes.{asset}"), fn)(res)


def box_shape(centre, s: float, rgb) -> dict:
    return {"type": "cube", "to_world": _T().translate(list(map(float, centre))).scale([s / 2] * 3),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": list(map(float, rgb))}}}


def bunny_group(res: int, translation) -> dict:
    """The released bunny's shapegroup and BSDF, re-keyed (group1/instance1/marble) with absolute file paths."""
    b = official("bunny", "get_scene", res)
    bdir = UPSTREAM / "scenes" / "bunny"
    shape = dict(b["group0"]["jade"])
    shape["filename"] = str(bdir / shape["filename"])
    interior = dict(shape["interior"])
    albedo = dict(interior["albedo"])
    albedo["filename"] = str(bdir / albedo["filename"])
    # media are evaluated at world-space points and do not follow the instance transform: move the grid with the bunny
    albedo["to_world"] = _T().translate(list(map(float, translation))) @ albedo["to_world"]
    interior["albedo"] = albedo
    shape["interior"] = interior
    shape["bsdf"] = {"type": "ref", "id": "marble_bsdf"}
    return {
        "marble_bsdf": b["marble_bsdf"],
        "group1": {"type": "shapegroup", "marble": shape},
        "instance1": {"type": "instance", "shapegroup": {"type": "ref", "id": "group1"},
                      "to_world": _T().translate(list(map(float, translation)))},
    }


EXPERIMENTS = {
    "E1": {"asset": "seal", "fn": "scene2", "res": 256, "assets": {"instance0": "seal"}, "kind": "volume",
           "support": ["floor"]},
    "E2": {"asset": "teaset", "fn": "scene1", "res": 512, "assets": {"instance0": "teaset"}, "kind": "surface",
           "support": ["table", "wall"]},
    "E3": {"asset": "seal", "fn": "scene2", "res": 256, "assets": {"instance0": "seal", "instance1": "bunny"},
           "kind": "volume", "support": ["floor"]},
}


def build(exp: str, state: str, variant: str, placements: dict, res: int | None = None) -> dict:
    """Scene dict for an experiment state ('absent' variant = no box). E3 has no box: its states move the bunny."""
    e = EXPERIMENTS[exp]
    res = R(res or e["res"])
    d = official(e["asset"], e["fn"], res)
    if exp == "E3":
        p = placements["E3"]
        d.update(bunny_group(res, p["states"][state]["translation"]))
        cam = p["camera"]
        d["camera"]["to_world"] = look_at(cam["theta"], cam["phi"], cam["r"], cam["target"])
        d["camera"]["fov"] = cam["fov"]
        return d
    if variant != "absent":
        p = placements[exp]
        d["box"] = box_shape(p["states"][state]["centre"], p["s"], RED if variant == "red" else BLACK)
    return d


def reference_integrator(kind: str, direct_only: bool = False):
    f = flags()
    if direct_only:
        return mi.load_dict({"type": "path", "max_depth": 2})
    t = f["ref_volume"] if kind == "volume" else f["ref_surface"]
    return mi.load_dict({"type": t, "max_depth": -1, "rr_depth": 5})


# --- rendering ----------------------------------------------------------------------

def render(scene, integ, spp: int, chunk: int, seed0: int) -> tuple[np.ndarray, float]:
    """Mean of spp/chunk renders with seeds seed0, seed0+1, ... (demo-notebook protocol); returns (HxWx3, seconds)."""
    n = spp // chunk
    if n * chunk != spp:
        raise ValueError("spp must be a multiple of chunk")
    w, h = [int(v) for v in scene.sensors()[0].film().size()]
    if w * h * chunk > MAX_LANES:
        # a 16.7 M-lane launch exhausted the 12 GB GPU and DrJit never recovered (worklog 43, 2026-10-25)
        raise ValueError(f"{w}x{h}x{chunk} = {w * h * chunk} lanes per launch exceeds {MAX_LANES}")
    yield_gpu()
    dr.flush_malloc_cache()  # release the previous scene's cached allocations before a new launch size
    sync()
    t = time.perf_counter()
    last = t
    acc = None
    for k in range(n):
        img = mi.render(scene, integrator=integ, spp=chunk, seed=seed0 + k)
        acc = img if acc is None else acc + img
        dr.eval(acc)
        if time.perf_counter() - last > 60:  # liveness for the stage log (a stalled log means a stalled render)
            dr.sync_thread()
            last = time.perf_counter()
            print(f"    chunk {k + 1}/{n} {last - t:.0f}s", flush=True)
    out = np.array(acc, dtype=np.float32)[..., :3] / n
    sync()
    return out, time.perf_counter() - t


def save(path: Path, img: np.ndarray) -> int:
    """Atomic save (a killed process never leaves a truncated file that a re-run would reuse); returns the number of
    non-finite values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, img.astype(np.float32))
    os.replace(tmp, path)
    return int((~np.isfinite(img)).sum())


def sidecar_ok(path: Path, meta: dict) -> bool:
    """A reference may be reused on a re-run only if it was made with the same placements, flags, spp, chunk and seed."""
    side = Path(str(path) + ".json")
    if not (Path(path).exists() and side.exists()):
        return False
    old = json.loads(side.read_text())
    if old != meta:
        raise RuntimeError(f"{path} exists but was made with {old}, not {meta}")
    return True


def write_sidecar(path: Path, meta: dict) -> None:
    Path(str(path) + ".json").write_text(json.dumps(meta, sort_keys=True))


def check_params(before: dict, integ, where: str) -> dict:
    """Protocol parameter check: a learned parameter changed during the process -> stop the chain."""
    after = param_digest(integ)
    if after != before:
        print(f"PARAMETER CHECK FAILED in {where}", flush=True)
        finish(1)
    return after


# --- pixel classes and geometry --------------------------------------------------------

def shapes_by_id(scene) -> dict:
    return {s.id(): s for s in scene.shapes()}


def primary_classes(scene, labels: dict[str, int]) -> np.ndarray:
    """Per pixel: label of the shape hit by the pixel-centre camera ray (instances by si.instance), 0 = background."""
    sensor = scene.sensors()[0]
    w, h = [int(v) for v in sensor.film().size()]
    idx = dr.arange(mi.UInt32, w * h)
    x = (mi.Float(idx % w) + 0.5) / w
    y = (mi.Float(idx // w) + 0.5) / h
    ray, _ = sensor.sample_ray(0.0, 0.0, mi.Point2f(x, y), mi.Point2f(0.5, 0.5))
    si = scene.ray_intersect(ray)
    cls = np.zeros(w * h, np.int16)
    by_id = shapes_by_id(scene)
    for sid, lab in labels.items():
        if sid not in by_id:
            continue
        ptr = mi.ShapePtr(by_id[sid])
        # instanced meshes report the inner mesh in si.shape, so a top-level shape match never hits an instance
        m = si.instance.eq_(ptr) if sid.startswith("instance") else si.shape.eq_(ptr)
        cls[np.array(m, dtype=bool) & np.array(si.is_valid(), dtype=bool)] = lab
    return cls.reshape(h, w)


def obj_vertices(path: Path) -> np.ndarray:
    rows = []
    with open(path, "rb") as f:
        for line in f:
            if line[:2] == b"v ":
                rows.append(line.split()[1:4])
    return np.asarray(rows, dtype=np.float64)


def asset_vertices(exp: str, placements: dict | None = None, state: str | None = None) -> np.ndarray:
    """World-space vertices of the learned assets (OBJ files; instances are identity except the E3 bunny)."""
    sdir = UPSTREAM / "scenes"
    if exp in ("E1", "E3"):
        v = [obj_vertices(sdir / "seal" / "seal.obj")]
        if exp == "E3" and placements is not None:
            v.append(obj_vertices(sdir / "bunny" / "bunny.obj") + np.asarray(placements["E3"]["states"][state]["translation"]))
        return np.concatenate(v)
    return np.concatenate([obj_vertices(sdir / "teaset" / f) for f in ("teaplate.obj", "teapot2.obj", "teapot3.obj", "teapot4.obj")])


def bbox_np(shape) -> tuple[np.ndarray, np.ndarray]:
    b = shape.bbox()
    return np.array([b.min.x, b.min.y, b.min.z], float), np.array([b.max.x, b.max.y, b.max.z], float)


def _rays(o: np.ndarray, d: np.ndarray, maxt: np.ndarray | None = None):
    o = o.astype(np.float32)
    d = d.astype(np.float32)
    ray = mi.Ray3f(mi.Point3f(mi.Float(o[:, 0]), mi.Float(o[:, 1]), mi.Float(o[:, 2])),
                   mi.Vector3f(mi.Float(d[:, 0]), mi.Float(d[:, 1]), mi.Float(d[:, 2])))
    if maxt is not None:
        ray.maxt = mi.Float(maxt.astype(np.float32))
    return ray


def hit_ids(scene, si, candidates: list[str]) -> np.ndarray:
    """Shape id hit by each lane among candidates (instances by si.instance), '' otherwise."""
    by_id = shapes_by_id(scene)
    out = np.full(dr.width(si.t), "", dtype=object)
    for sid in candidates:
        ptr = mi.ShapePtr(by_id[sid])
        m = si.instance.eq_(ptr) if sid.startswith("instance") else si.shape.eq_(ptr)
        out[np.array(m, dtype=bool)] = sid
    return out


def rest_height(scene, cx: float, cz: float, s: float, y_top: float, allowed: list[str]) -> dict:
    """Support under a box footprint: 5x5 downward rays from y_top must all hit an allowed shape; bottom = max hit + 2 mm."""
    g = np.linspace(-s / 2 * 0.9, s / 2 * 0.9, 5)
    xs, zs = np.meshgrid(cx + g, cz + g)
    o = np.stack([xs.ravel(), np.full(25, y_top), zs.ravel()], 1)
    d = np.tile([0.0, -1.0, 0.0], (25, 1))
    si = scene.ray_intersect(_rays(o, d))
    valid = np.array(si.is_valid(), dtype=bool)
    if not valid.all():
        return {"ok": False, "reason": "footprint ray missed"}
    ids = hit_ids(scene, si, [k for k in shapes_by_id(scene) if k in allowed or k.startswith("instance")])
    wrong = sorted(set(ids[~np.isin(ids, allowed)].tolist()))
    y = np.array(si.p.y, dtype=np.float64)
    spread = float(y.max() - y.min())
    ok = (not wrong) and spread < 0.02
    reason = None if ok else (f"footprint hits {wrong}" if wrong else f"uneven support ({spread:.3f})")
    return {"ok": ok, "bottom": float(y.max() + 0.002), "spread": spread, "reason": reason}


def box_segments(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    # 12 edges (differ in one coordinate), 12 face diagonals (two), 4 space diagonals (three): all 28 corner pairs
    return c[[p[0] for p in pairs]], c[[p[1] for p in pairs]]


def box_clear(scene, lo, hi, verts: np.ndarray) -> dict:
    """No hit along all 28 corner-to-corner segments of the box, and no asset vertex inside the box grown by 1 mm
    (less than the 2 mm rest gap, so the supporting surface's own vertices are not counted)."""
    a, b = box_segments(lo, hi)
    d = b - a
    L = np.linalg.norm(d, axis=1)
    hit = np.array(scene.ray_test(_rays(a, d / L[:, None], L)), dtype=bool)
    inside = np.all((verts > lo - 0.001) & (verts < hi + 0.001), axis=1)
    return {"ok": (not hit.any()) and (not inside.any()), "segment_hits": int(hit.sum()), "vertices_inside": int(inside.sum())}
