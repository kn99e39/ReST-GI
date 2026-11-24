"""WL111 GPU engine: Main admission of the frozen WL101 ECRO renderer, camera-input frame / guide programs, edit lifecycle.

Frozen and imported unchanged: the WL101 deployment (inputs/wl101: rt101.py, seq101.py, pt100 / pt86 / model86 / scene86 /
closure83_dr / common100, the WL100 SSS checkpoint and configs) on the frozen LabServer stack (Mitsuba 3.8.0 / Dr.Jit 1.3.1,
WL101 Amendment 1), inside the WL109B Amendment 3 container on LabServer70.

Experiment-local integration (no estimator change):
  * frame_prog: the same statements as rt101.RT._frame_prog, but the camera enters the frozen program as opaque Dr.Jit
    scalars instead of Python constants, so one recording serves every camera (the original records one program per
    camera, which a moving camera would re-record every frame). gen_rays_dyn repeats pt86.gen_rays with the same
    float32 operation order. Bitwise equality with the original program is a test (tests_r111 / profile Stage A).
  * guide_prog: geometry-only guides in one launch (sub x sub fixed sub-pixel rays per pixel: per-shape coverage, mean
    camera-facing normal, mean hit distance, valid fraction), the online form of render105.aux_buffers; centre_prog: the
    WL101B centre-ray guide (position, normal, shape id); vis_prog: previous-eye visibility (WL101B hidden_in_previous).
"""
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import common_r111 as C

os.environ.setdefault("DYN101_MI_DIR", str(Path.home() / "rest_dyn101" / "external" / "mi_3.8.0"))
os.environ.setdefault("DYN101_MI_VERSION", "3.8.0")
os.environ["FREEZE"] = "1"
for d in ("ecro_sss101a", "ecro_sss101d", "ecro_denoise105", "ecro_sss101b"):
    sys.path.insert(0, str(C.EXP / d))
sys.path.insert(0, str(C.WL101 / "experiments" / "ecro_dyn101"))
import rt101 as R  # noqa: E402
import seq101  # noqa: E402

mi, dr, torch, PT = R.mi, R.dr, R.torch, R.PT
F, U, I32, UI64 = mi.Float, mi.UInt32, mi.Int32, mi.UInt64
RES = C.RES
NPIX = RES * RES


def sync():
    dr.sync_thread()
    torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------ host / exclusivity
def gpu_query():
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,clocks.sm,temperature.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout.strip()
    return q


def other_python():
    """Windows: every other python.exe (WDDM lists desktop processes as GPU clients, so CUDA workers are found by name).
    The venv Scripts\\python.exe is a launcher whose child is the interpreter: own and parent PIDs are excluded."""
    if os.name != "nt":
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        return [int(p) for p in out if p.strip().isdigit() and int(p) not in (os.getpid(), os.getppid())]
    out = subprocess.run(["tasklist", "/fi", "imagename eq python.exe", "/fo", "csv", "/nh"], capture_output=True,
                         text=True).stdout
    pids = [int(r.split('","')[1]) for r in out.strip().splitlines() if r.startswith('"python')]
    mine = {os.getpid(), os.getppid()}
    return [p for p in pids if p not in mine]


def load1():
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except OSError:
        return None


def exclusive_ok():
    """Exclusive-GPU check before / after a timing block.
    LabServer70 (RECON111_GUARDED=1): the host-side guard (guard_r111.py) owns exclusivity: it starts the container only
    after the RTX 3090 had no compute process for 120 s and kills it the moment a foreign compute process appears there.
    Inside the container this check records nvidia-smi (utilization, compute apps visible to the container) and the host
    load average, and requires median utilization <= idle limit and load1 <= the protocol's timing load gate.
    Windows / unguarded hosts: no other python process and an idle GPU."""
    t = C.P["timing"]
    util = []
    for _ in range(t["idle_samples"]):
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60).stdout
        util.append(float(q.strip().splitlines()[0]))
        time.sleep(0.5)
    ld = load1()
    if os.environ.get("RECON111_GUARDED") == "1":
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        ok = float(np.median(util)) <= t["idle_util_max_pct"] and (ld is None or ld <= t["max_load1"])
        return ok, {"guarded": True, "compute_apps_visible": apps, "idle_util_samples": util, "load1": ld, "t": time.time()}
    others = other_python()
    ok = (not others) and float(np.median(util)) <= t["idle_util_max_pct"]
    return ok, {"other_python_pids": others, "idle_util_samples": util, "load1": ld, "t": time.time()}


def timed_block(fn, label, log=print):
    """Exclusive-GPU timing block: wait until exclusive, run, check again; repeat a contaminated block (bounded)."""
    t = C.P["timing"]
    rec = []
    for attempt in range(t["max_attempts"]):
        while True:
            ok, pre = exclusive_ok()
            if ok:
                break
            log("timing wait (not exclusive)", label, pre)
            time.sleep(30)
        res = fn()
        ok2, post = exclusive_ok()
        rec.append({"pre": pre, "post": post, "attempt": attempt, "clean": ok2})
        if ok2:
            return res, rec
        log("timing block contaminated; repeating", label, post)
    raise RuntimeError(f"no uncontended timing block for {label}")


def versions():
    import triton
    return {"mitsuba": mi.__version__, "drjit": dr.__version__, "torch": torch.__version__, "triton": triton.__version__,
            "numpy": np.__version__, "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
            "nvidia_smi": gpu_query(), "variant": mi.variant(), "freeze": R.FREEZE, "host": platform.node(),
            "os": platform.platform(), "cudnn": torch.backends.cudnn.version(),
            "allow_tf32": {"cudnn": torch.backends.cudnn.allow_tf32, "matmul": torch.backends.cuda.matmul.allow_tf32}}


# ------------------------------------------------------------------------------------------------ admission
def admit(log=print, replay=True):
    """Frozen WL101 renderer admission on LabServer70. Same checks as WL108 renderer108.frozen_runtime (WL101F inputs), with the
    host pins replaced by the protocol's LabServer70 environment and the two numerical identities graded by protocol G0:
      G0a sources / checkpoint / configs / references / baseline manifests (exact hashes)
      G0b learned HIGH table of the frozen SSS state: exact archived hash, else relative L1 <= tolerance to the archived
          WL101 PRED_B_0 table
      G0c archived 4096-spp V0 predictions (historical seed convention) replayed within the WL105 G1 tolerance.
    -> rt, Bd (Dr.Jit table), B (torch table), audit."""
    from run_oracle import A, array_hash, fingerprint, validate_inputs   # WL101D / WL101A frozen helpers
    G = C.P["gates"]["G0"]
    reference = C.WL101 / A["references"]["path_in_wl101"]
    sources = validate_inputs(C.WL101, reference)
    base = C.INPUTS / "results" / "ecro_sss101e" / "recovery2"
    prior = C.INPUTS / "results" / "ecro_sss101a" / "qualified63"
    assert C.sha(base / "run_manifest.json") == G["wl101e_manifest_sha256"]
    anchor = C.read(base / "run_manifest.json")
    assert C.sha(prior / "run_manifest.json") == anchor["prior_manifest_sha256"]
    assert sources == anchor["renderer_sources_sha256"]
    assert mi.__version__ == G["mitsuba"] and dr.__version__ == G["drjit"], (mi.__version__, dr.__version__)
    case, initial, events = seq101.workload(A["workload"])
    cam = dict(R.C100.P["datasets"][case]["camera"], res=RES)
    assert cam == A["camera"] and events[seq101.WARM] == (A["geometry"], A["light"], 0)
    assert C.sha(R.ROOT100 / "sss" / "model.pt") == G["checkpoint_sha256"]
    rt = R.RT(initial[0], case, R.load_model(case), initial[1])
    for geo, light, _ in events[:seq101.WARM + 1]:
        moved, old = rt.set_geom(geo)
        rt.update_state(old, moved)
        rt.set_light(light)
        B, _ = rt.respond(rt.source())
    assert fingerprint({k: A[k] for k in ("geometry", "camera", "sphere", "light")}) == anchor["scene_sha256"]
    table = B.cpu().numpy().copy()
    audit = {k: anchor[k] for k in ("scene_sha256", "checkpoint_sha256", "cfg_sha256", "learned_table_sha256",
                                    "input_sha256", "renderer_sources_sha256")}
    exact = array_hash(table) == anchor["learned_table_sha256"]
    with np.load(reference / "PRED_B_0.npz") as z:
        arch = z["B"].astype(np.float64).reshape(table.shape)
    rel = float(np.abs(table - arch).sum() / np.abs(arch).sum())
    audit["G0b_table"] = {"exact_hash": bool(exact), "relative_l1_vs_archived_PRED_B_0": rel,
                          "tolerance": G["table_relative_l1"], "table_sha256": array_hash(table)}
    assert exact or rel <= G["table_relative_l1"], audit["G0b_table"]
    Bd = R.t2d(B.view(-1))
    if replay:
        history = C.read(prior / "run_manifest.json")
        errs = []
        for row in history["matched"]:
            raw = prior / row["path"]
            assert C.sha(raw) == row["sha256"]
            comp = historical_components(rt, Bd, cam, 4096, row["seed"])
            with np.load(raw) as z:
                exp = np.concatenate([z[k].reshape(RES, RES, 3).transpose(2, 0, 1) for k in ("e1", "e2", "hl")])
            for i in range(0, 9, 3):
                errs.append(float(np.abs(comp[i:i + 3] - exp[i:i + 3]).sum() / np.abs(exp[i:i + 3]).sum()))
        audit["G0c_replay_relative_l1"] = errs
        assert max(errs) <= G["replay_relative_l1"], errs
    audit["environment"] = versions()
    log("admission", {k: v for k, v in audit.items() if k.startswith("G0")})
    return rt, Bd, B, audit


def historical_components(rt, Bd, cam, spp, seed, b=32):
    """render105.frame_components (same statements): original rt._frame_prog, launch k seeded seed * 1000003 + k (the
    historical convention; used only to replay archived frames), component means [9,H,W] float32 numpy."""
    cam = dict(cam, res=RES)
    b = min(b, spp)
    acc = None
    for k in range(spp // b):
        out = rt._frame_prog(rt.scene, rt.tabs(), rt.light_args(), Bd, dr.opaque(UI64, seed * 1000003 + k), (cam, RES, b))
        dr.eval(out)
        t = torch.stack([o.torch().double() for o in out])
        acc = t if acc is None else acc + t
    return (acc / spp).cpu().numpy().reshape(9, RES, RES).astype(np.float32)


# ------------------------------------------------------------------------------------------------ camera as program input
NCAM = 16          # o (3), fwd (3), right (3), up (3), s (1), previous eye (3)


CAM_MODE = os.environ.get("RECON111_CAM_MODE", "scalars")
_CAM_BUF = {}


def cam_array(cam, prev=None):
    """pt86.cam_basis values (and the previous eye for the visibility query) as frozen-program inputs. float32 rounding
    equals the literal conversion of the original program.
      scalars  16 opaque float32 scalars (the form validated by the multi-camera equivalence test)
      array    one opaque float32 array of 16 values   (diagnostic; replayed stale inputs on Dr.Jit 1.3.1)
      torch    a persistent torch buffer updated in place and shared by DLPack (diagnostic)"""
    o, fwd, right, up, s = PT.cam_basis(cam)
    po = PT.cam_basis(prev)[0] if prev is not None else o
    vals = [float(v) for v in (*o, *fwd, *right, *up, s, *po)]
    if CAM_MODE == "scalars":
        return tuple(dr.opaque(F, v) for v in vals)
    if CAM_MODE == "array":
        a = F(np.array(vals, np.float32))
        dr.make_opaque(a)
        return a
    if "t" not in _CAM_BUF:
        _CAM_BUF["t"] = torch.zeros(NCAM, dtype=torch.float32, device="cuda")
    _CAM_BUF["t"].copy_(torch.tensor(vals, dtype=torch.float32))
    return F(_CAM_BUF["t"].__dlpack__())


def cam_split(ca):
    """Inside a program: the 16 camera values as size-1 arrays (broadcast like the original constants)."""
    if isinstance(ca, tuple):
        return list(ca)
    return [dr.gather(F, ca, dr.opaque(U, k)) for k in range(NCAM)]


def gen_rays_dyn(ct, res, b, rng):
    """pt86.gen_rays with the camera basis read from program inputs (same float32 operation order)."""
    o, fwd, right, up, s = ct[0:3], ct[3:6], ct[6:9], ct[9:12], ct[12]
    n = res * res
    lane = dr.arange(U, n * b)
    pix = lane % n
    ii, jj = pix // res, pix % res
    jx, jy = (rng.next_float32(), rng.next_float32()) if rng is not None else (F(0.5), F(0.5))
    x = (F(jj) + jx) * (2.0 / res) - 1.0
    y = 1.0 - (F(ii) + jy) * (2.0 / res)
    d = dr.normalize(mi.Vector3f(*[fwd[k] + s * x * right[k] + s * y * up[k] for k in range(3)]))
    return mi.Ray3f(mi.Point3f(o[0], o[1], o[2]), d), pix


class Engine:
    """Frozen runtime + camera-input programs. Every frame returns per-pixel component MEANS [9,H,W] float32 torch (keys
    e10 e11 e12 e20 e21 e22 hl0 hl1 hl2: LOW = 0:6 summed per channel, HIGH = 6:9)."""

    def __init__(self, rt, Bd, B):
        self.rt, self.Bd, self.B = rt, Bd, B
        self.revision = 0
        self.state = None
        self._f = {}

    # ---------------------------------------------------------------- lifecycle (frozen WL101 selective update)
    def set_state(self, geom, light, timing=None):
        """WL101 selective edit lifecycle (set_geom -> update_state -> set_light -> source -> respond -> DLPack table).
        Any geometry or light change increments the transport revision (global history invalidation)."""
        rt = self.rt
        changed = self.state is None or self.state != (geom, light)
        moved, old = rt.set_geom(geom)
        rt.update_state(old, moved)
        rt.set_light(light)
        B, _ = rt.respond(rt.source())
        self.B = B
        self.Bd = R.t2d(B.view(-1))
        if changed:
            self.revision += 1
        self.state = (geom, light)
        return moved

    # ---------------------------------------------------------------- frame programs
    def _frame_prog(self, scene, tab, lp, Bd, seed, ca, spp):
        """rt101.RT._frame_prog statements with gen_rays_dyn."""
        rt = self.rt
        rt._bind(tab)
        ct = cam_split(ca)
        n = RES * RES
        N = n * spp
        lt = rt.lt
        lt.pos, lt.rgb, lt.axis, lt.cc = list(lp["pos"]), list(lp["rgb"]), list(lp["axis"]), list(lp["cc"])
        sd = UI64(seed)
        rng = R.pcg_046(N, sd, UI64(0))
        ray, pix = gen_rays_dyn(ct, RES, spp, rng)
        out, _ = rt.trace(ray, N, rng, sd, "ER", 2, lt, "ecro", B=Bd)
        keys = sorted(out)
        if spp == 1:
            return [out[k] for k in keys]
        bufs = []
        for k in keys:
            buf = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, out[k], pix)
            bufs.append(buf)
        return bufs

    def frame_raw(self, cam, spp, seed_value):
        key = ("frame", spp)
        if key not in self._f:
            self._f[key] = R.maybe_freeze(lambda scene, tab, lp, Bd, sd, ca: self._frame_prog(scene, tab, lp, Bd, sd, ca, spp))
        rt = self.rt
        out = self._f[key](rt.scene, rt.tabs(), rt.light_args(), self.Bd, dr.opaque(UI64, seed_value), cam_array(cam))
        dr.eval(out)
        return out

    def frame(self, cam, spp, seed):
        """Evidence frame: component means [9,H,W] (torch float32, GPU). Main-stream seed = launch_seed(seed)."""
        out = self.frame_raw(cam, spp, C.launch_seed(seed))
        t = torch.stack([o.torch() for o in out])
        t = t.clone() if spp == 1 else t
        return (t * (1.0 / spp)).view(9, RES, RES)

    def blocks(self, cam, spp, seed, b=32):
        """High-spp component means [9,H,W] float64 numpy: spp / b launches of b spp, launch k = launch_seed(seed, k)."""
        acc = None
        for k in range(spp // b):
            out = self.frame_raw(cam, b, C.launch_seed(seed, k))
            t = torch.stack([o.torch().double() for o in out])
            acc = t if acc is None else acc + t
        return (acc / spp).view(9, RES, RES).cpu().numpy()

    # ---------------------------------------------------------------- guides (one program per frame)
    def _pack_prog(self, scene, tab, ca, sub, with_centre):
        """sub x sub guides (render105.aux_buffers statements) and, with_centre, the WL101B centre guide plus the
        previous-eye visibility query (WL101B hidden_in_previous) in ONE program."""
        rt = self.rt
        rt._bind(tab)
        ct = cam_split(ca)
        n = NPIX
        M = n * sub * sub
        lane = dr.arange(U, M)
        k = lane // n
        pix = lane % n
        ii, jj = pix // RES, pix % RES
        jx = (F(k % sub) + 0.5) / sub
        jy = (F(k // sub) + 0.5) / sub
        o, fwd, right, up, s = ct[0:3], ct[3:6], ct[6:9], ct[9:12], ct[12]
        x = (F(jj) + jx) * (2.0 / RES) - 1.0
        y = 1.0 - (F(ii) + jy) * (2.0 / RES)
        d = dr.normalize(mi.Vector3f(*[fwd[q] + s * x * right[q] + s * y * up[q] for q in range(3)]))
        si = scene.ray_intersect(mi.Ray3f(mi.Point3f(o[0], o[1], o[2]), d))
        ok = si.is_valid()
        sid = dr.select(ok, I32(rt.shape_id(si)), I32(-1))
        nn = dr.select(dr.dot(si.n, d) > 0, -si.n, si.n)
        one = dr.full(F, 1.0, M)
        cover = dr.zeros(F, 9 * n)
        dr.scatter_reduce(dr.ReduceOp.Add, cover, one, U(dr.maximum(sid, 0)) * n + pix, ok)
        bufs = [cover]
        for q in range(3):
            b_ = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, b_, nn[q], pix, ok)
            bufs.append(b_)
        dep = dr.zeros(F, n)
        dr.scatter_reduce(dr.ReduceOp.Add, dep, si.t, pix, ok)
        cnt = dr.zeros(F, n)
        dr.scatter_reduce(dr.ReduceOp.Add, cnt, one, pix, ok)
        bufs += [dep, cnt]
        if with_centre:
            ray, _ = gen_rays_dyn(ct, RES, 1, None)
            sc = scene.ray_intersect(ray)
            front = sc.is_valid() & (sc.to_local(-ray.d).z > 1e-6)
            cen = [dr.select(front, a, 0) for a in (*sc.p, *sc.n)] + [dr.select(front, F(rt.shape_id(sc)), -1.0)]
            po = mi.Point3f(ct[13], ct[14], ct[15])
            p = mi.Point3f(cen[0], cen[1], cen[2])
            dv = p - po
            length = dr.norm(dv)
            vr = mi.Ray3f(po, dv / dr.maximum(length, 1e-6))
            vr.maxt = dr.maximum(length - 0.0002, 0.0)
            hidden = scene.ray_test(vr)
            bufs += cen + [hidden]
        return bufs

    def pack(self, cam, sub, prev=None, with_centre=True):
        """-> dict of torch GPU tensors: cover (9,H,W), normal (3,H,W), depth (H,W), valid_frac (H,W), pure (H,W bool), and
        with_centre: centre (7, H*W) (WL101B guide) and hidden (H*W bool; occluded from `prev`'s eye, meaningful only
        when prev is given)."""
        key = ("pack", sub, bool(with_centre))
        if key not in self._f:
            self._f[key] = R.maybe_freeze(lambda scene, tab, ca: self._pack_prog(scene, tab, ca, sub, with_centre))
        rt = self.rt
        out = self._f[key](rt.scene, rt.tabs(), cam_array(cam, prev))
        dr.eval(out)
        ss = float(sub * sub)
        cover = out[0].torch().view(9, RES, RES) / ss
        cnt = out[5].torch().view(RES, RES)
        w = cnt.clamp(min=1.0)
        g = {"cover": cover, "normal": torch.stack([out[1 + q].torch().view(RES, RES) for q in range(3)]) / w,
             "depth": out[4].torch().view(RES, RES) / w, "valid_frac": cnt / ss, "pure": cover.amax(0) >= 1.0}
        if with_centre:
            g["centre"] = torch.stack([a.torch() for a in out[6:13]])
            g["hidden"] = out[13].torch().bool()
        return g

    def guides(self, cam, sub):
        """Guides only (R3's 8 x 8 input; online form of render105.aux_buffers, equality tested)."""
        return self.pack(cam, sub, None, with_centre=False)

    def centre(self, cam):
        """-> torch (7, H*W): position (3), normal (3), shape id as float (-1 = miss / back face): the WL101B guide."""
        return self.pack(cam, 1, None, with_centre=True)["centre"]

    def _vis_prog(self, scene, ca, px, py, pz, oracle):
        ct = cam_split(ca)
        o = mi.Point3f(ct[13], ct[14], ct[15])
        p = mi.Point3f(px, py, pz)
        d = p - o
        length = dr.norm(d)
        ray = mi.Ray3f(o, d / dr.maximum(length, 1e-6))
        limit = dr.maximum(length - 0.0002, 0.0)
        if oracle:
            hit = scene.ray_intersect(ray)
            return hit.is_valid() & (hit.t < limit)
        ray.maxt = limit
        return scene.ray_test(ray)

    def hidden_from(self, prev_cam, centre, oracle=False):
        """Bool torch (H*W): the segment from prev_cam's eye to the current centre-ray point is occluded (WL101B
        hidden_in_previous; oracle = independent closest-hit test used only for scoring)."""
        key = ("vis", bool(oracle))
        if key not in self._f:
            self._f[key] = R.maybe_freeze(lambda scene, ca, px, py, pz: self._vis_prog(scene, ca, px, py, pz, oracle))
        pts = [R.t2d(centre[q].contiguous()) for q in range(3)]
        out = self._f[key](self.rt.scene, cam_array(prev_cam, prev_cam), *pts)
        dr.eval(out)
        return out.torch().bool()

    def roi_aux(self, cam):
        """Offline ROI inputs for a camera (render105.aux_buffers + centre_gbuffer semantics): numpy dict."""
        g = self.guides(cam, 8)
        gb = self.rt.gbuffer(dict(cam, res=RES), RES)
        return {"cover": g["cover"].cpu().numpy(), "normal": g["normal"].cpu().numpy(), "depth": g["depth"].cpu().numpy(),
                "valid_frac": g["valid_frac"].cpu().numpy(), "sid": gb["sid"].astype(np.int16), "ok": gb["ok"],
                "p": gb["p"].astype(np.float32)}


def cam_dict(c):
    return {"origin": [float(x) for x in c["origin"]], "target": [float(x) for x in c["target"]], "fov_deg": float(c["fov_deg"]),
            "res": RES}
