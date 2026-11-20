"""WL104H scene-conditioned component pass and attribution.

For every replicate of the archived WL104G scene evidence (same seeds, same 512-spp launch layout replayed in sub-chunks through PCG32
initseq offsets) the first hit on the ER objects exports every response term:
  E1  at the fixed point-light direction (x light rgb / d^2 x visibility),
  E2  at the direction drawn by the native R0 proposal, divided by the same actual R0 pdf, times the fixed R0 terminal response at hit 2.
Non-ER first hits are accumulated natively. A and C (WL104D) are evaluated with the same directions (C - A diagnostic).

Usage: python scenes104h.py render <scene> <out_dir> [--smoke] [--chunk K]
       python scenes104h.py analyze <scene> <out_dir>
"""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.stats as sst

import io104h as IO
import comp104h as K

H = K.H
C, RT, mi, dr = H.C, H.RT, H.mi, H.dr
SC = IO.P["frozen_inputs"]["scenes"]
LAUNCH = 512
EVENTS = ("e1", "e2")
R0_IMG = ("O0", "O1", "O2", "SL", "SI0", "SI1", "SI2", "SI3", "X", "soX", "siX", "SU", "XU", "N")
YS = ("L", "I0", "I1", "I2", "I3")
AC_IMG = tuple([f"{s}.O{k}" for s in "AC" for k in range(3)] + [f"S{x}*{y}.{t}" for x in "AC" for y in "AC" for t in YS])
ROIS = IO.P["scene_phase"]["rois"]
HELD_VIEW = None


def scene_row(scene):
    rows = IO.read(IO.MATERIALS)
    i = next(i for i, r in enumerate(rows) if r["image"] == scene)
    return i, rows[i]


class Pass(RT.Ctx):
    def __init__(self, tag, states):
        key = f"wl104h_{tag}_R0"
        H.RR.REG[key] = ("ER", states["R0"])
        super().__init__(mi.load_dict(C.scene_dict(mi, RT.T, {"type": "cmp83", "key": key}, with_sensor=True)))
        names = [m[1] for m in self.meshes]
        self.sph, self.cub = names.index("sphere"), names.index("cube")
        self.cl = {k: K.DrTerms(v) for k, v in states.items()}

    def uses_er(self, sid):
        return dr.eq(sid, self.sph) | dr.eq(sid, self.cub)

    def rays(self, k, rng):
        """rt104.Ctx.gen_rays formula for a sub-chunk of k spp (lane % n is unchanged by the n-multiple offset)."""
        o, fwd, right, up, s = C.cam_basis()
        W = Hh = C.RES
        n = W * Hh
        pix = dr.arange(mi.UInt32, n * k) % n
        ii, jj = pix // W, pix % W
        jx, jy = rng.next_float32(), rng.next_float32()
        x = (mi.Float(jj) + jx) * (2.0 / W) - 1.0
        y = 1.0 - (mi.Float(ii) + jy) * (2.0 / Hh)
        d = dr.normalize(mi.Vector3f(*[float(fwd[q]) + s * x * float(right[q]) + s * y * float(up[q]) for q in range(3)]))
        return mi.Ray3f(mi.Point3f(*[mi.Float(float(o[q])) for q in range(3)]), d), pix

    def light_term(self, si, light, active):
        dl = light.pos[0] - si.p
        d2 = dr.squared_norm(dl)
        wl = dl * dr.rsqrt(d2)
        wo = si.to_local(wl)
        on = active & (wo.z > 1e-6) & (dr.dot(-wl, light.axis[0]) > light.cc[0])
        vis = on & ~self.scene.ray_test(si.spawn_ray_to(light.pos[0]), on)
        return dr.select(vis, wo, mi.Vector3f(0, 0, 1)), vis, light.rgb[0] * dr.rcp(d2)

    def chunk(self, sd, offset, k, light, with_ac=True):
        n = C.RES * C.RES * k
        rng = mi.PCG32(size=n, initstate=sd, initseq=mi.UInt64(offset))
        ray, pix = self.rays(k, rng)
        si, sid, _, ok = self.surface(ray, mi.Bool(True))
        active = ok & (si.wi.z > 1e-6)
        er = self.uses_er(sid)
        native = si.bsdf()
        lanes, checks = {}, {}

        def put(name, ev, col):
            for c in range(3):
                lanes[f"{name}|{ev}|{c}"] = col[c]

        def export(ev, t_r0, t_ac, scale, mask, nat):
            S = t_r0["S"]
            X = t_r0["L"] + t_r0["I0"] + t_r0["I1"] + t_r0["I2"] + t_r0["I3"]
            XU = t_r0["L"] + t_r0["U0"] + t_r0["U1"] + t_r0["U2"] + t_r0["U3"]
            img = {"O0": t_r0["O0"], "O1": t_r0["O1"], "O2": t_r0["O2"], "SL": S * t_r0["L"],
                   **{f"SI{q}": S * t_r0[f"I{q}"] for q in range(4)}, "X": X, "soX": t_r0["so"] * X, "siX": t_r0["si"] * X,
                   "SU": S * (t_r0["U0"] + t_r0["U1"] + t_r0["U2"] + t_r0["U3"]), "XU": XU}
            for name, v in img.items():
                put(name, ev, dr.select(mask, v * scale, 0.0))
            put("N", ev, nat)
            if t_ac is not None:
                for s in "AC":
                    for q in range(3):
                        put(f"{s}.O{q}", ev, dr.select(mask, t_ac[s][f"O{q}"] * scale, 0.0))
                for x in "AC":
                    for y in "AC":
                        for tname in YS:
                            put(f"S{x}*{y}.{tname}", ev, dr.select(mask, t_ac[x]["S"] * t_ac[y][tname] * scale, 0.0))

        # E1: fixed point-light direction
        wo1, vis, lf = self.light_term(si, light, active)
        m1 = vis & er
        t1 = self.cl["R0"].terms(si.wi, wo1, m1)
        f_nat1 = native.eval(self.bctx, si, wo1, vis)
        t1ac = {s: self.cl[s].terms(si.wi, wo1, m1, unrefracted=False) for s in "AC"} if with_ac else None
        export("e1", t1, t1ac, lf, m1, dr.select(vis & ~er, f_nat1 * lf, 0.0))
        checks["e1_lane_native_minus_terms"] = dr.max(dr.select(m1, dr.max(dr.abs(f_nat1 - K.DrTerms.native_order(t1))), 0.0))
        checks["e1_lane_native_peak"] = dr.max(dr.select(m1, dr.max(f_nat1), 0.0))
        # E2: common frozen native R0 proposal, same actual pdf, fixed R0 terminal at hit 2
        bs, w = native.sample(self.bctx, si, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active)
        used = active & (bs.pdf > 0) & (dr.max(w) > 0)
        pdf = dr.select(used, bs.pdf, 1.0)
        wo2 = dr.select(used, bs.wo, mi.Vector3f(0, 0, 1))
        other, osid, _, ok2 = self.surface(si.spawn_ray(si.to_world(bs.wo)), used)
        act2 = used & ok2 & (other.wi.z > 1e-6)
        wl2, vis2, lf2 = self.light_term(other, light, act2)
        recv = dr.select(vis2, other.bsdf().eval(self.bctx, other, wl2, vis2) * lf2, 0.0)
        m2 = used & er
        t2 = self.cl["R0"].terms(si.wi, wo2, m2)
        f_nat2, p_nat2 = native.eval_pdf(self.bctx, si, wo2, used)
        t2ac = {s: self.cl[s].terms(si.wi, wo2, m2, unrefracted=False) for s in "AC"} if with_ac else None
        scale2 = recv * dr.rcp(pdf)
        export("e2", t2, t2ac, scale2, m2, dr.select(used & ~er, f_nat2 * scale2, 0.0))
        checks["e2_lane_native_minus_terms"] = dr.max(dr.select(m2, dr.max(dr.abs(f_nat2 - K.DrTerms.native_order(t2))), 0.0))
        checks["e2_lane_native_peak"] = dr.max(dr.select(m2, dr.max(f_nat2), 0.0))
        checks["e2_weight_pdf_minus_eval"] = dr.max(dr.select(used, dr.max(dr.abs(w * bs.pdf - f_nat2)), 0.0))
        checks["e2_pdf_sample_minus_eval_rel"] = dr.max(dr.select(used, dr.abs(bs.pdf - p_nat2) * dr.rcp(pdf), 0.0))
        checks["e2_used_lanes"] = dr.count(used)
        checks["e2_er_lanes"] = dr.count(m2)
        acc = self.accum(lanes, pix, C.RES * C.RES)
        dr.eval(*checks.values())
        return acc, {k: float(np.asarray(v).ravel()[0]) for k, v in checks.items()}

    def replicate(self, seed, spp, k, light, with_ac=True, launch=LAUNCH):
        acc, chk = {}, {}
        for j in range(0, spp // launch):
            sd = dr.opaque(mi.UInt64, seed * 1000003 + j)
            for c in range(launch // k):
                a, ch = self.chunk(sd, c * C.RES * C.RES * k, k, light, with_ac)
                for key, v in a.items():
                    acc[key] = acc[key] + v if key in acc else v
                for key, v in ch.items():
                    chk[key] = max(chk.get(key, 0.0), v) if "lanes" not in key else chk.get(key, 0.0) + v
        img = {}
        names = sorted({key.rsplit("|", 2)[0] for key in acc})
        for name in names:
            for ev in EVENTS:
                img[f"{name}|{ev}"] = np.stack([acc[f"{name}|{ev}|{c}"] for c in range(3)], -1).reshape(C.RES, C.RES, 3) / spp
        return img, chk


# ------------------------------------------------------------------ inputs
def archived(scene, idx):
    """Archived reference / R0 events (and A / C native for development scenes), authenticated."""
    role = SC[scene]
    if "held" in role["role"]:
        folder = IO.G_RAW / "scenes/references" / scene
    else:
        folder = IO.G_RAW / "scenes" / scene
    files = [folder / f"r{i:03d}.npz" for i in range(12)]
    masks_path = IO.G_RAW / "scenes" / scene / "region_masks.npz"
    paths = [*files, masks_path]
    e_files = []
    if "held" not in role["role"]:
        e_files = [IO.E_PARENT / scene / f"r{i:03d}.npz" for i in range(12)]
        paths += e_files
    seen = IO.authenticate(paths)
    ev = {k: [] for k in ("reference_e1", "reference_e2", "reference_h", "R0_e1", "R0_e2", "R0_h")}
    for f in files:
        with np.load(f) as z:
            for k in ev:
                ev[k].append(z[k])
    out = {k: np.stack(v).astype(np.float64) for k, v in ev.items()}
    for f in e_files:
        with np.load(f) as z:
            for k in ("A_e1", "A_e2", "C_e1", "C_e2"):
                out.setdefault(k, []).append(z[k])
    for k in ("A_e1", "A_e2", "C_e1", "C_e2"):
        if k in out:
            out[k] = np.stack(out[k]).astype(np.float64)
    with np.load(masks_path) as z:
        masks = {k: z[k].astype(bool) for k in z.files}
    return out, masks, seen


def setup(scene):
    i, row = scene_row(scene)
    _, raw, compiled, _ = IO.inputs()
    if "held" in SC[scene]["role"]:
        C.CAM = IO.read(IO.HERE.parent / "ecro_wl84_104d/protocol.json")["population"]["held_out_view"]
    ctx = Pass(row["tag"] + "_" + scene, {k: compiled[k][i] for k in ("R0", "A", "C")})
    return i, row, ctx, compiled


def l1(x, y):
    return float(np.abs(x - y).sum() / max(np.abs(y).sum(), 1e-30))


def native_sum(img, ev, prefix=""):
    if prefix:
        s = prefix
        return img[f"N|{ev}"] + sum(img[f"{s}.O{q}|{ev}"] for q in range(3)) + sum(img[f"S{s}*{s}.{t}|{ev}"] for t in YS)
    return img[f"N|{ev}"] + sum(img[f"{k}|{ev}"] for k in ("O0", "O1", "O2", "SL", "SI0", "SI1", "SI2", "SI3"))


def render(scene, out, smoke=False, k=64, reps=(0, 12)):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    i, row, ctx, compiled = setup(scene)
    light = RT.point_lights(row["light"])
    seed0 = SC[scene]["seed_base"]
    log = {"scene": scene, "chunk_spp": k, "commit": IO.commit(), "protocol_sha256": IO.sha(IO.HERE / "protocol.json")}
    if smoke:
        # focused: sub-chunked replay vs native render (launch 4, chunk 2), determinism, A/C E1 vs native A/C renders
        spp = 4
        nat = H.HA.render(ctx, scene, spp, seed0, "cam", b=spp)
        img, chk = ctx.replicate(seed0, spp, 2, light, True, launch=spp)
        img2, _ = ctx.replicate(seed0, spp, 2, light, True, launch=spp)
        res = {"e1_vs_native": l1(native_sum(img, "e1"), nat["e1"]), "e2_vs_native": l1(native_sum(img, "e2"), nat["e2"]),
               "deterministic_max_abs": max(float(np.abs(img[q] - img2[q]).max()) for q in img), "lane_checks": chk}
        for s in "AC":
            key = f"wl104h_smoke_{s}"
            H.RR.REG[key] = ("ER", compiled[s][i])
            alt = RT.Ctx(mi.load_dict(C.scene_dict(mi, RT.T, {"type": "cmp83", "key": key}, with_sensor=True)))
            na = H.HA.render(alt, scene, spp, seed0, "cam", b=spp)
            res[f"{s}_e1_vs_native"] = l1(native_sum(img, "e1", s), na["e1"])
        # full-scale probe: one archived 512-spp launch with the production chunk size
        tp = time.monotonic()
        sd = dr.opaque(mi.UInt64, seed0 * 1000003)
        for c in range(LAUNCH // k):
            ctx.chunk(sd, c * C.RES * C.RES * k, k, light, True)
        res["full_launch_seconds"] = time.monotonic() - tp
        # camera / geometry check: centre-ray ROIs equal the archived WL104G region masks
        _, amasks, _ = archived(scene, None)
        cmask = H.HA.masks(ctx, {"sphere_lower_nz_max": 0., "floor_near": {"sphere_axis_m": .60, "cube_footprint_m": .25}})
        res["roi_mismatch_pixels"] = {q: int((cmask[q] != amasks[q]).sum()) for q in ("objects", "sphere", "cube", "sphere_lower", "floor_near")}
        res["allocator"] = [ln.strip() for ln in dr.whos_str().splitlines() if "peak" in ln]
        IO.write(out / "smoke.json", res)
        print(res, flush=True)
        return
    spp = SC[scene]["spp"]
    chk_all = []
    for r in range(*reps):
        tr = time.monotonic()
        img, chk = ctx.replicate(seed0 + r, spp, k, light, True)
        chk["seconds"] = time.monotonic() - tr
        chk_all.append(chk)
        np.savez_compressed(out / f"r{r:03d}.npz", **{q: v.astype(np.float32) for q, v in img.items()})
        print("scene", scene, "replicate", r, round(chk["seconds"], 1), flush=True)
    log["checks"] = chk_all
    log["replicates"] = list(range(*reps))
    log["seconds"] = time.monotonic() - t0
    IO.write(out / f"render_{reps[0]:02d}_{reps[1]:02d}.json", log)


# ------------------------------------------------------------------ analysis
sys_path_done = False


def _stats():
    import sys
    sys.path.insert(0, str(IO.HERE.parent / "ecro_wl84_104b"))
    sys.path.insert(0, str(IO.HERE.parent / "ecro_wl84_104a"))
    sys.path.insert(0, str(IO.HERE.parent / "ecro_wl84_104e"))
    from stats104b import ms_loo
    from stats104a import ratio_jk
    return ms_loo, ratio_jk


T11 = float(sst.t.ppf(0.975, 11))


def jk(pt, loo):
    loo = np.asarray(loo, float)
    R = len(loo)
    se = float(np.sqrt((R - 1) / R * ((loo - loo.mean()) ** 2).sum()))
    return {"point": float(pt), "se": se, "lo": float(pt - T11 * se), "hi": float(pt + T11 * se)}


def delta_images(I, ev):
    """Delta E_X for every intervention (and interactions) from the R0 term images of one event; I[name] = [R, H, W, 3]."""
    g = lambda n: I[f"{n}|{ev}"]
    O = g("O0") + g("O1") + g("O2")
    SI = g("SI0") + g("SI1") + g("SI2") + g("SI3")
    F = O + g("SL") + SI
    fx = {"LAMBERT_OFF": F - g("SL"), "OUTER_OFF": F - O, "INNER_OFF": F - SI, "ATTEN_OFF": O + g("X"),
          "ATTEN_IN_OFF": O + g("soX"), "ATTEN_OUT_OFF": O + g("siX"), "ETA_OFF": O + g("SL") + g("SU"),
          "ETA_ATTEN_OFF": O + g("XU"), **{f"O{q}_OFF": F - g(f"O{q}") for q in range(3)},
          **{f"I{q}_OFF": F - g(f"SI{q}") for q in range(4)}}
    d = {k: v - F for k, v in fx.items()}
    d["ATTEN_INTERACTION"] = d["ATTEN_OFF"] - d["ATTEN_IN_OFF"] - d["ATTEN_OUT_OFF"]
    d["ETA_ATTEN_INTERACTION"] = d["ETA_ATTEN_OFF"] - d["ETA_OFF"] - d["ATTEN_OFF"]
    return F, d


def ca_images(I, ev):
    g = lambda n: I[f"{n}|{ev}"]
    FA = sum(g(f"A.O{q}") for q in range(3)) + sum(g(f"SA*A.{t}") for t in YS)
    FC = sum(g(f"C.O{q}") for q in range(3)) + sum(g(f"SC*C.{t}") for t in YS)
    parts = {f"O{q}": g(f"C.O{q}") - g(f"A.O{q}") for q in range(3)}
    parts["atten"] = 0.5 * sum((g(f"SC*A.{t}") + g(f"SC*C.{t}")) - (g(f"SA*A.{t}") + g(f"SA*C.{t}")) for t in YS)
    for t in YS:
        parts[f"{t}_content"] = 0.5 * ((g(f"SA*C.{t}") + g(f"SC*C.{t}")) - (g(f"SA*A.{t}") + g(f"SC*A.{t}")))
    return FA, FC, parts


def analyze(scene, out):
    ms_loo, ratio_jk = _stats()
    t0 = time.monotonic()
    arch, masks, seen = archived(scene, None)
    logs = [IO.read(p) for p in sorted(out.glob("render_*.json"))]
    render_log = {"checks": [c for lg in logs for c in lg["checks"]], "parts": logs}
    if sorted(r for lg in logs for r in lg["replicates"]) != list(range(12)):
        raise IO.Stop("SCENE_REPLICATES_INCOMPLETE")
    keys = None
    I = {}
    for r in range(12):
        with np.load(out / f"r{r:03d}.npz") as z:
            keys = keys or list(z.files)
            for q in keys:
                I.setdefault(q, []).append(z[q])
    I = {q: np.stack(v).astype(np.float64) for q, v in I.items()}
    # identities against the archived native renders
    ident = {"R0_e1": [l1(native_sum({q: I[q][r] for q in I}, "e1"), arch["R0_e1"][r]) for r in range(12)],
             "R0_e2": [l1(native_sum({q: I[q][r] for q in I}, "e2"), arch["R0_e2"][r]) for r in range(12)]}
    if "A_e1" in arch:
        for s in "AC":
            ident[f"{s}_e1"] = [l1(native_sum({q: I[q][r] for q in I}, "e1", s), arch[f"{s}_e1"][r]) for r in range(12)]
    lim = IO.P["decomposition"]["identity_tolerances"]["scene_image_relative_l1"]
    lane = IO.P["decomposition"]["identity_tolerances"]["scene_lane_relative_to_peak"]
    chk = render_log["checks"]
    lane_ok = all(c["e1_lane_native_minus_terms"] <= lane * max(c["e1_lane_native_peak"], 1e-30) and
                  c["e2_lane_native_minus_terms"] <= lane * max(c["e2_lane_native_peak"], 1e-30) and
                  c["e2_weight_pdf_minus_eval"] <= lane * max(c["e2_lane_native_peak"], 1e-30) and
                  c["e2_pdf_sample_minus_eval_rel"] <= 1e-5 for c in chk)
    ident_ok = all(max(v) <= lim for v in ident.values()) and lane_ok
    ref = arch
    ref_tot = ref["reference_e1"] + ref["reference_e2"] + ref["reference_h"]
    r0_tot = ref["R0_e1"] + ref["R0_e2"] + ref["R0_h"]
    resid = {"e1": ref["reference_e1"] - ref["R0_e1"], "e2": ref["reference_e2"] - ref["R0_e2"], "e12": ref_tot - r0_tot}
    deltas, cas, parts = {}, {}, {}
    for ev in EVENTS:
        _, deltas[ev] = delta_images(I, ev)
        FA, FC, parts[ev] = ca_images(I, ev)
        cas[ev] = FC - FA
    deltas["e12"] = {k: deltas["e1"][k] + deltas["e2"][k] for k in deltas["e1"]}
    cas["e12"] = cas["e1"] + cas["e2"]
    parts["e12"] = {k: parts["e1"][k] + parts["e2"][k] for k in parts["e1"]}
    ca_closure = max(float(np.abs(sum(parts[ev].values()) - cas[ev]).max() / max(np.abs(cas[ev]).max(), 1e-30)) for ev in EVENTS)
    fam = IO.P["interventions"]["families_for_confounding"]
    family = {m: f for f, ms in fam.items() for m in ms}
    result = {"scene": scene, "role": SC[scene]["role"], "identity_relative_l1": ident, "lane_checks_passed": lane_ok,
              "identity_passed": ident_ok, "ca_closure": ca_closure, "rois": {}}
    if not ident_ok:
        IO.write(out / "analysis.json", {**result, "stop": "SCENE_REPLAY_IDENTITY"})
        raise IO.Stop("SCENE_REPLAY_IDENTITY")
    flat = lambda x, m: x[:, m].reshape(x.shape[0], -1)
    for roi in ROIS:
        m = masks[roi]
        if not m.any():
            continue
        den = float((ref_tot[:, m].mean(0) ** 2).mean())
        ref_luma = (ref_tot[:, m] @ IO.K.H.LUMA).sum(1)
        block = {}
        for ev in ("e1", "e2", "e12"):
            R = flat(resid[ev], m)
            rr, rr_l = ms_loo(R, R)
            D = flat(cas[ev], m)
            dd, dd_l = ms_loo(D, D)
            evb = {"residual_rms": jk(np.sqrt(max(rr, 0) / den), np.sqrt(np.maximum(rr_l, 0) / den)),
                   "ca_rms": jk(np.sqrt(max(dd, 0) / den), np.sqrt(np.maximum(dd_l, 0) / den)), "interventions": {}}
            if ev == "e2" and "A_e2" in ref:
                Darch = flat(ref["C_e2"] - ref["A_e2"], m)
                aa, aa_l = ms_loo(Darch, Darch)
                xa, xa_l = ms_loo(D, Darch)
                evb["fixed_R0_proposal_vs_native_CA"] = {"share": jk(xa / aa, xa_l / aa_l),
                                                         "cos": jk(xa / np.sqrt(aa * dd), xa_l / np.sqrt(aa_l * dd_l))}
            evb["ca_shares"] = {}
            for q, v in parts[ev].items():
                x, xl = ms_loo(flat(v, m), D)
                evb["ca_shares"][q] = jk(x / dd, xl / dd_l) if dd > 0 else None
            names = list(deltas[ev])
            vecs = {n: flat(deltas[ev][n], m) for n in names}
            nn = {n: ms_loo(vecs[n], vecs[n]) for n in names}
            for n in names:
                a2, a2l = nn[n]
                cr, crl = ms_loo(vecs[n], R)
                xc, xcl = ms_loo(vecs[n], D)
                rho = lambda c, a, r: c / np.sqrt(a * r) if a > 0 and r > 0 else np.nan
                st = {"rho": jk(rho(cr, a2, rr), [rho(c, a, r) for c, a, r in zip(crl, a2l, rr_l)]),
                      "error_change": jk((2 * cr - a2) / rr, (2 * crl - a2l) / rr_l),
                      "dmse_over_den": jk((a2 - 2 * cr) / den, (a2l - 2 * crl) / den),
                      "change_ms_over_den": jk(a2 / den, a2l / den),
                      "rms_after": jk(np.sqrt(max(rr - 2 * cr + a2, 0) / den), np.sqrt(np.maximum(rr_l - 2 * crl + a2l, 0) / den)),
                      "ca_projection": jk(xc / dd, xcl / dd_l) if dd > 0 else None,
                      "ca_cos": jk(rho(xc, a2, dd), [rho(c, a, r) for c, a, r in zip(xcl, a2l, dd_l)]) if dd > 0 else None,
                      "energy": ratio_jk((deltas[ev][n][:, m] @ IO.K.H.LUMA).sum(1), ref_luma)}
                if n in K.INTERVENTIONS:
                    det = st["change_ms_over_den"]["lo"] > 0 and (st["dmse_over_den"]["lo"] > 0 or st["dmse_over_den"]["hi"] < 0)
                    if rr <= 0 or np.min(rr_l) <= 0:
                        lab = "EVIDENCE_UNQUALIFIED"
                    elif not det:
                        lab = "NO_DETECTABLE_COMPONENT_EFFECT"
                    elif st["rho"]["lo"] > 0.5 or st["rho"]["hi"] < -0.5:
                        lab = "CONDITIONAL_ERROR_LOCALIZATION"
                    else:
                        lab = "DETECTABLE_COMPONENT_EFFECT"
                    st["label"] = lab
                evb["interventions"][n] = st
            cos = np.zeros((len(names), len(names)))
            for a_, x in enumerate(names):
                for b_, y in enumerate(names):
                    if b_ < a_:
                        cos[a_, b_] = cos[b_, a_]
                        continue
                    xy, _ = ms_loo(vecs[x], vecs[y])
                    cos[a_, b_] = xy / np.sqrt(max(nn[x][0] * nn[y][0], 1e-300))
            evb["cos"] = cos.tolist()
            evb["names"] = names
            for n in K.NAMES:
                st = evb["interventions"][n]
                if st["label"] != "CONDITIONAL_ERROR_LOCALIZATION":
                    continue
                i = names.index(n)
                rivals = [y for j, y in enumerate(names) if y in K.NAMES and family.get(y) != family.get(n) and abs(cos[i, j]) >= 0.9
                          and evb["interventions"][y]["label"] in ("CONDITIONAL_ERROR_LOCALIZATION", "CONFOUNDED_OR_NON_IDENTIFIABLE")]
                if rivals:
                    st["label"] = "CONFOUNDED_OR_NON_IDENTIFIABLE"
                    st["confounded_with"] = rivals
                    for y in rivals:
                        evb["interventions"][y]["label"] = "CONFOUNDED_OR_NON_IDENTIFIABLE"
                        evb["interventions"][y].setdefault("confounded_with", []).append(n)
            block[ev] = evb
        result["rois"][roi] = block
    # mean maps for review (objects bounding area handled by the packager)
    maps = {f"resid_{ev}": resid[ev].mean(0).astype(np.float32) for ev in ("e1", "e2")}
    for ev in ("e1", "e2"):
        maps[f"ca_{ev}"] = cas[ev].mean(0).astype(np.float32)
        for n in ("LAMBERT_OFF", "OUTER_OFF", "INNER_OFF", "ATTEN_OFF", "ETA_OFF", *[f"O{q}_OFF" for q in range(3)],
                  *[f"I{q}_OFF" for q in range(4)]):
            maps[f"d_{n}_{ev}"] = deltas[ev][n].mean(0).astype(np.float32)
        for t in ("O0", "O1", "O2", "SL", "SI0", "SI1", "SI2", "SI3"):
            maps[f"term_{t}_{ev}"] = I[f"{t}|{ev}"].mean(0).astype(np.float32)
    maps["ref_total"] = ref_tot.mean(0).astype(np.float32)
    np.savez_compressed(out / "maps.npz", **maps)
    result["inputs_sha256"] = seen
    result["seconds"] = time.monotonic() - t0
    result["commit"] = IO.commit()
    IO.write(out / "analysis.json", result)
    print("analysis", scene, round(time.monotonic() - t0, 1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("render", "analyze"))
    ap.add_argument("scene")
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--reps", default="0:12")
    args = ap.parse_args()
    if args.mode == "render":
        render(args.scene, args.out, args.smoke, args.chunk, tuple(int(x) for x in args.reps.split(":")))
    else:
        analyze(args.scene, args.out)


if __name__ == "__main__":
    main()
