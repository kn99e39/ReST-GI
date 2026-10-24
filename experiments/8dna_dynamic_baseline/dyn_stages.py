"""Render stages of protocol 8dna_dyn_v1 (worklog 43). One process per stage, in this order:

    python dyn_stages.py smoke     # stage 0: flags (VCallRecord, reference integrators), learned-lane, m2w and counter checks
    python dyn_stages.py design    # box placements, geometric validity, pixel classes (geometry only)
    python dyn_stages.py static    # official demonstrations (B) on this machine
    python dyn_stages.py refs      # path-traced references of E1-E3
    python dyn_stages.py neural    # fixed released 8DNA assets in the same scenes
    python dyn_stages.py ladder    # equal-time / equal-error ladders
    python dyn_stages.py cost      # edit-update timing

Outputs go to $EDNA_OUT (default <repo>/results/8dna_dynamic_baseline). Run on LabServer63 after `source server/mi_env.sh`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time

import numpy as np

import dynlib as L

P = L.PROTOCOL
X = P["experiments"]
STATES = {"E1": ["T0", "T1_out", "T1_in"], "E2": ["T0", "T1_out", "T1_in"], "E3": ["T0", "T1"]}
VARIANTS = {"E1": ["red", "black"], "E2": ["red", "black"], "E3": ["red"]}
LABELS = {"instance0": 1, "instance1": 2, "box": 3, "floor": 4, "table": 4, "wall": 5, "background": 6}
SPP_KEY = {"E1": "E1_seal_box", "E2": "E2_teaset_box", "E3": "E3_seal_bunny"}


def scenes_of(exp: str):
    """(state, variant) pairs to render; one 'absent' scene per box experiment."""
    pairs = [(s, v) for s in STATES[exp] for v in VARIANTS[exp]]
    if exp != "E3":
        pairs.append(("none", "absent"))
    return pairs


def tag(state: str, variant: str) -> str:
    return "absent" if variant == "absent" else f"{state}_{variant}"


def e3_states() -> dict:
    a_hi_x, b_lo_y, floor_y = 0.6553, -0.5943, -0.367
    b_hi_x = 0.7087

    def tx(gap):
        return -(a_hi_x + gap + b_hi_x)

    ty = floor_y + 0.003 - b_lo_y
    return {"T0": {"translation": [tx(0.60), ty, 0.0], "gap": 0.60}, "T1": {"translation": [tx(0.08), ty, 0.0], "gap": 0.08}}


def placements() -> dict:
    return L.read_json(L.OUT / "design" / "placements.json")


def placements_sha() -> str:
    return hashlib.sha256((L.OUT / "design" / "placements.json").read_bytes()).hexdigest()


# --- stage 0 --------------------------------------------------------------------

def vcall_probe(vcall: int) -> None:
    """Child process: one launch-size neural render per asset type with the given VCallRecord value."""
    mi, dr = L.init(vcall=bool(vcall))
    rec = {}
    for asset, fn, res in (("seal", "scene2", L.R(256)), ("teaset", "get_scene", L.R(512))):
        scene = mi.load_dict(L.official(asset, fn, res))
        integ = L.neural_integrator({"instance0": asset})
        L.attach(integ, scene)
        img, t = L.render(scene, integ, 4, 4, 0)  # timed after model load and after any GPU yield
        rec[asset] = {"lanes": res * res * 4, "seconds": round(t, 2), "finite": bool(np.isfinite(img).all())}
    print("VCALL_PROBE", rec, flush=True)
    L.finish(0)


def _probe(v: int, out) -> dict:
    """Run one probe in its own session (torch.compile worker grandchildren must not outlive a timeout kill)."""
    L.yield_gpu()
    logf = out / f"vcall_probe_{v}.log"
    t0 = time.time()
    with open(logf, "w") as fh:
        p = subprocess.Popen([sys.executable, "-u", str(L.HERE / "dyn_stages.py"), "vcall_probe", "--vcall", str(v)],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=str(L.HERE), start_new_session=True)
        try:
            rc, timed_out = p.wait(timeout=1200), False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            rc, timed_out = p.wait(), True
    text = logf.read_text(errors="replace")
    secs = None
    if "VCALL_PROBE" in text:
        import ast

        r = ast.literal_eval(text.split("VCALL_PROBE", 1)[1].strip().splitlines()[0])
        secs = max(v_["seconds"] for v_ in r.values())
    # the child's CUDA context must be released before the next process initialises CUDA
    for _ in range(60):
        if not any(pid == p.pid for pid, _ in L.gpu_apps()):
            break
        time.sleep(1)
    time.sleep(3)
    return {"pid": p.pid, "returncode": rc, "timed_out": timed_out, "render_seconds": secs, "wall_seconds": round(time.time() - t0, 1),
            "ok": (not timed_out) and rc == 0 and secs is not None, "tail": text[-800:]}


def stage_smoke() -> None:
    out = L.OUT / "stage0"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"probes": {}}
    p0 = _probe(0, out)
    rec["probes"]["VCallRecord=False"] = p0
    if p0["ok"] and p0["render_seconds"] <= 600:
        vcall = False
    elif p0["timed_out"] or (p0["ok"] and p0["render_seconds"] > 600):
        p1 = _probe(1, out)
        rec["probes"]["VCallRecord=True"] = p1
        if not p1["ok"]:
            L.write_json(out / "smoke.json", rec | {"verdict": "FAIL: neither VCallRecord setting renders"})
            L.finish(1)
        vcall = True
    else:
        L.write_json(out / "smoke.json", rec | {"verdict": "FAIL: VCallRecord=False probe failed (not a timeout)"})
        L.finish(1)

    mi, dr = L.init(vcall=vcall)
    import torch

    rec["environment"] = L.environment()
    checks = {}
    # learned lanes and counters on every experiment scene type (absent variant; E3 with its T0 bunny)
    tiny_pl = {"E3": {"states": e3_states(), "camera": dict(theta=60, phi=240, r=6.75, target=[-0.70, 0.10, 0.0], fov=35)}}
    for exp in ("E1", "E2", "E3"):
        e = L.EXPERIMENTS[exp]
        d = L.build(exp, "T0", "absent" if exp != "E3" else "red", tiny_pl, res=64)
        scene = mi.load_dict(d)
        integ = L.neural_integrator(e["assets"])
        L.attach(integ, scene)
        before = L.param_digest(integ)
        c = L.Counters()
        undo = L.instrument(c)
        img, t = L.render(scene, integ, 4, 4, 0)
        undo()
        after = L.param_digest(integ)
        pm = L.per_model(c, integ)
        checks[exp] = {"finite": bool(np.isfinite(img).all()), "mean": img.reshape(-1, 3).mean(0).tolist(), "seconds": round(t, 2),
                       "counters": c.n, "per_model": pm, "params_unchanged": before == after,
                       "every_asset_active": len(pm) == len(e["assets"]) and all(v.get("learned_sample_lanes", 0) > 0 for v in pm.values())}
    rec["learned_lane_checks"] = checks

    # non-identity instance transform (unexercised upstream; E3 needs it): translating the bunny instance and the camera
    # by the same t under a distant envmap must give the same image as the identity placement, at the same seeds
    t = [0.7, 0.2, -0.3]
    da = L.official("bunny", "get_scene", L.R(64))
    db = L.official("bunny", "get_scene", L.R(64))
    db["instance0"]["to_world"] = mi.ScalarTransform4f.translate(t)
    db["camera"]["to_world"] = mi.ScalarTransform4f.translate(t) @ db["camera"]["to_world"]
    integ = L.neural_integrator({"instance0": "bunny"})
    imgs = {}
    m2w = {}
    for name, d, seed in (("identity", da, 0), ("translated", db, 0), ("identity_seed2", da, 7000)):
        scene = mi.load_dict(d)
        L.attach(integ, scene)
        m2w[name] = [list(map(float, r)) for r in integ.asset_m2ws[0].cpu().numpy()]
        imgs[name], _ = L.render(scene, integ, *L.S(64, 4), seed)
    disp = lambda x: np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)  # noqa: E731
    m_t = float(np.mean((disp(imgs["identity"]) - disp(imgs["translated"])) ** 2))
    m_s = float(np.mean((disp(imgs["identity"]) - disp(imgs["identity_seed2"])) ** 2))
    rec["m2w_translation_check"] = {"t": t, "m2w_seen_by_integrator": m2w, "mse_identity_vs_translated": m_t,
                                    "mse_identity_vs_other_seed": m_s, "pass": m_t <= 0.1 * m_s}

    # reference integrators: C++ volpath/path vs the scenes' own Python prbvolpath/prb (protocol rule)
    ref = {}
    for kind, asset, fn, cpp, py in (("volume", "seal", "scene2", "volpath", "prbvolpath"), ("surface", "teaset", "get_scene", "path", "prb")):
        scene = mi.load_dict(L.official(asset, fn, L.R(128)))
        imgs = {}
        for name, seed in ((cpp, 11000), (py + "_A", 12000), (py + "_B", 13000)):
            integ = mi.load_dict({"type": name.split("_")[0], "max_depth": -1, "rr_depth": 5})
            imgs[name], tsec = L.render(scene, integ, *L.S(2048, 64), seed)
            ref.setdefault(kind, {})[f"{name}_seconds"] = round(tsec, 2)
        m_x = float(np.mean((disp(imgs[cpp]) - disp(imgs[py + "_A"])) ** 2))
        m_ab = float(np.mean((disp(imgs[py + "_A"]) - disp(imgs[py + "_B"])) ** 2))
        ref[kind].update({"mse_cpp_vs_py": m_x, "mse_py_A_vs_B": m_ab, "use_cpp": m_x <= 1.5 * m_ab,
                          "mean_ratio_cpp_over_py": float(imgs[cpp].mean() / max(imgs[py + "_A"].mean(), 1e-12))})
    rec["reference_equivalence"] = ref
    flags = {"VCallRecord": vcall, "ref_volume": "volpath" if ref["volume"]["use_cpp"] else "prbvolpath",
             "ref_surface": "path" if ref["surface"]["use_cpp"] else "prb"}
    old = L.read_json(out / "flags.json") if (out / "flags.json").exists() else None
    rec["flags"], rec["previous_flags"] = flags, old
    rec["torch_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    ok = all(c["finite"] and c["every_asset_active"] and c["params_unchanged"] for c in checks.values()) \
        and rec["m2w_translation_check"]["pass"]
    rec["verdict"] = "PASS" if ok else "FAIL"
    L.write_json(out / "flags.json", flags)
    L.write_json(out / "smoke.json", rec)
    print(rec["verdict"], flags, flush=True)
    L.finish(0 if ok else 1)


# --- design (geometry only) ---------------------------------------------------------

def _box_lohi(cx, bottom, cz, s):
    return np.array([cx - s / 2, bottom, cz - s / 2]), np.array([cx + s / 2, bottom + s, cz + s / 2])


def _place(scene, verts, cx, cz, s, allowed, y_top, inside=None, aabb=None):
    rh = L.rest_height(scene, cx, cz, s, y_top, allowed)
    if not rh["ok"]:
        return None, {"rest": rh}
    bottom = rh["bottom"]
    if inside:
        bottom = max(bottom, aabb[0][1] + 0.002)
    lo, hi = _box_lohi(cx, bottom, cz, s)
    clear = L.box_clear(scene, lo, hi, verts)
    alo, ahi = aabb
    within = bool(np.all(lo >= alo) and np.all(hi <= ahi))
    gap = float(max(np.max(alo - hi), np.max(lo - ahi)))  # > 0: separated along some axis
    env = {"inside_asset_aabb": within, "separation_from_asset_aabb": gap}
    ok = clear["ok"] and (within if inside else gap >= 0.02)
    info = {"rest": rh, "clear": clear, "envelope": env, "lo": lo.tolist(), "hi": hi.tolist()}
    centre = [float(cx), float(bottom + s / 2), float(cz)]
    return (centre if ok else None), info


def _fail(out, report, msg):
    L.write_json(out / "design_report.json", report | {"verdict": f"FAIL: {msg}"})
    print("design FAIL:", msg, flush=True)
    L.finish(1)


def stage_design() -> None:
    mi, dr = L.init()
    out = L.OUT / "design"
    pl, report = {}, {}

    # E1 seal: in-envelope box on a corner of the seal's cuboid pedestal (it fills the AABB footprint up to y = 0.1218)
    scene = mi.load_dict(L.build("E1", "none", "absent", {}))
    aabb = L.bbox_np(L.shapes_by_id(scene)["instance0"])
    verts = L.asset_vertices("E1")
    lo, hi = aabb
    tries, chosen = [], None
    for s in (0.18, 0.12):
        for corner in ("front_right", "front_left", "back_right", "back_left"):
            cx = hi[0] - 0.01 - s / 2 if "right" in corner else lo[0] + 0.01 + s / 2
            cz = lo[2] + 0.01 + s / 2 if "front" in corner else hi[2] - 0.01 - s / 2
            c, info = _place(scene, verts, cx, cz, s, ["instance0"], hi[1] + 0.05, inside=True, aabb=aabb)
            tries.append({"s": s, "corner": corner, "ok": c is not None, **info})
            if c is not None:
                chosen = (s, corner, c)
                break
        if chosen:
            break
    report["E1_in_envelope_tries"] = tries
    if chosen is None:
        _fail(out, report, "E1 no valid in-envelope placement")
    s = chosen[0]
    st = {"T1_in": {"centre": chosen[2], "corner": chosen[1]}}
    for name, (cx, cz) in {"T0": (2.4, 2.4), "T1_out": (0.30, lo[2] - 0.04 - s / 2)}.items():
        c, info = _place(scene, verts, cx, cz, s, ["floor"], lo[1] + s + 0.05, inside=False, aabb=aabb)
        report[f"E1_{name}"] = info
        if c is None:
            _fail(out, report, f"E1 {name} invalid")
        st[name] = {"centre": c}
    pl["E1"] = {"s": s, "states": st, "asset_aabb": [lo.tolist(), hi.tolist()]}

    # E2 teaset: (placement, size) order for T1_in on the tray, then T0 / T1_out on the flat table top
    scene = mi.load_dict(L.build("E2", "none", "absent", {}))
    aabb = L.bbox_np(L.shapes_by_id(scene)["instance0"])
    lo, hi = aabb
    verts = L.asset_vertices("E2")
    tries, chosen = [], None
    for where, s in (("centre", 0.12), ("centre", 0.10), ("centre", 0.08), ("front_left", 0.12)):
        cx, cz = {"centre": (0.01, -0.03), "front_left": (-0.33, 0.33)}[where]
        c, info = _place(scene, verts, cx, cz, s, ["instance0"], lo[1] + s + 0.03, inside=True, aabb=aabb)
        tries.append({"s": s, "placement": where, "ok": c is not None, **info})
        if c is not None:
            chosen = (s, where, c)
            break
    report["E2_in_envelope_tries"] = tries
    if chosen is None:
        _fail(out, report, "E2 no valid in-envelope placement")
    s = chosen[0]
    st = {"T1_in": {"centre": chosen[2], "placement": chosen[1]}}
    t0_tries = []
    for cx, cz in ((0.45, -0.90), (0.85, -0.50), (0.90, -0.20)):
        c, info = _place(scene, verts, cx, cz, s, ["table"], 0.05 + s + 0.05, inside=False, aabb=aabb)
        t0_tries.append({"xz": [cx, cz], "ok": c is not None, **info})
        if c is not None:
            st["T0"] = {"centre": c}
            break
    report["E2_T0_tries"] = t0_tries
    if "T0" not in st:
        _fail(out, report, "E2 T0 invalid")
    c, info = _place(scene, verts, lo[0] - 0.03 - s / 2, -0.33, s, ["table"], 0.05 + s + 0.05, inside=False, aabb=aabb)
    report["E2_T1_out"] = info
    if c is None:
        _fail(out, report, "E2 T1_out invalid")
    st["T1_out"] = {"centre": c}
    pl["E2"] = {"s": s, "states": st, "asset_aabb": [lo.tolist(), hi.tolist()]}

    # E3 seal + bunny: AABBs disjoint and above the floor
    pl["E3"] = {"states": e3_states(), "camera": {"theta": 60, "phi": 240, "r": 6.75, "target": [-0.70, 0.10, 0.0], "fov": 35}}
    for name in STATES["E3"]:
        scene = mi.load_dict(L.build("E3", name, "red", pl))
        sh = L.shapes_by_id(scene)
        a, b = L.bbox_np(sh["instance0"]), L.bbox_np(sh["instance1"])
        gap = float(max(np.max(a[0] - b[1]), np.max(b[0] - a[1])))
        report[f"E3_{name}"] = {"seal_aabb": [x.tolist() for x in a], "bunny_aabb": [x.tolist() for x in b], "aabb_gap": gap,
                                "bunny_above_floor": bool(b[0][1] > -0.367)}
        if gap < 0.02 or b[0][1] <= -0.367:
            _fail(out, report, f"E3 {name} invalid")

    f = out / "placements.json"
    if f.exists():
        old = L.read_json(f)
        L.write_json(out / "placements_new.json", pl)
        if old != L.read_json(out / "placements_new.json"):
            _fail(out, report, "placements differ from the existing placements.json (a re-run must not move boxes)")
    L.write_json(f, pl)

    # pixel classes of every scene that will be rendered
    classes = {}
    for exp in ("E1", "E2", "E3"):
        for state, variant in scenes_of(exp):
            scene = mi.load_dict(L.build(exp, state, variant, pl))
            classes[f"{exp}/{tag(state, variant)}"] = L.primary_classes(scene, LABELS)
    np.savez_compressed(out / "classes.npz", **classes)
    report["class_pixels"] = {k: {int(lab): int((v == lab).sum()) for lab in np.unique(v)} for k, v in classes.items()}
    report["verdict"] = "PASS"
    report["environment"] = L.environment()
    L.write_json(out / "design_report.json", report)
    print("design PASS", {k: v["s"] for k, v in pl.items() if "s" in v}, flush=True)
    L.finish(0)


# --- B: official demonstrations -------------------------------------------------------

def stage_static() -> None:
    mi, dr = L.init()
    import torch

    out = L.OUT / "static"
    rec = {"environment": L.environment()}
    for key, asset, fn, kind in (("seal_scene2", "seal", "scene2", "volume"), ("teaset_get_scene", "teaset", "get_scene", "surface")):
        c = X["B_static"][key]
        scene = mi.load_dict(L.official(asset, fn, L.R(c["res"])))
        integ = L.neural_integrator({"instance0": asset})
        L.attach(integ, scene)
        r = {"checkpoint_sha256": L.sha256(L.ckpt(asset)), "params_before": L.param_digest(integ), "nonfinite": {}}
        L.render(scene, integ, 4, 4, 999)  # warm-up (extension load, kernel compile)
        torch.cuda.reset_peak_memory_stats()
        with L.MemWatch() as mw:
            for i, seed in enumerate(c["neural_seeds"]):
                img, t = L.render(scene, integ, *L.S(c["neural_spp"], 4), seed)
                r["nonfinite"][f"neural_{i}"] = L.save(out / key / f"neural_{i}.npy", img)
                r[f"neural_{i}_seconds"] = t
        r["neural_peak_torch_bytes"] = int(torch.cuda.max_memory_allocated())
        r["neural_peak_process_mib"] = mw.max_mib
        r["params_after"] = L.check_params(r["params_before"], integ, f"static {key}")
        ref_i = L.reference_integrator(kind)
        r["reference_integrator"] = L.flags()["ref_volume" if kind == "volume" else "ref_surface"]
        for i, seed in enumerate(c["ref_seeds"]):
            img, t = L.render(scene, ref_i, *L.S(c["ref_spp"], c["ref_chunk"]), seed)
            r["nonfinite"][f"ref_{i}"] = L.save(out / key / f"ref_{i}.npy", img)
            r[f"ref_{i}_seconds"] = t
        rec[key] = r
        L.write_json(out / "static_runs.json", rec)
        print("static", key, {k: v for k, v in r.items() if k.endswith("seconds")}, flush=True)
    L.finish(0)


# --- references of E1-E3 --------------------------------------------------------------

def stage_refs() -> None:
    mi, dr = L.init()
    pl = placements()
    psha = placements_sha()
    fl = L.flags()
    out = L.OUT / "refs"
    tf = out / "timing.json"
    timing = L.read_json(tf) if tf.exists() else {}
    timing["environment"] = L.environment()
    seeds = {"A": 1000000, "B": 2000000}
    for exp in ("E1", "E2", "E3"):
        e, c = L.EXPERIMENTS[exp], X[SPP_KEY[exp]]["spp"]
        integ = L.reference_integrator(e["kind"])
        direct = L.reference_integrator(e["kind"], direct_only=True) if exp == "E2" else None
        for state, variant in scenes_of(exp):
            scene = mi.load_dict(L.build(exp, state, variant, pl))
            tg = tag(state, variant)
            jobs = [(f"{tg}_{ab}", integ, L.S(c["ref"], c["ref_chunk"]), seed) for ab, seed in seeds.items()]
            if direct is not None and variant in ("black", "absent"):
                jobs += [(f"{tg}_direct_{ab}", direct, L.S(c["ref_direct_only"], 64), seed + 500000) for ab, seed in seeds.items()]
            for name, ig, (spp, chunk), seed in jobs:
                f = out / exp / f"{name}.npy"
                meta = {"placements_sha256": psha, "integrator": "path_md2" if ig is direct else fl["ref_volume" if e["kind"] == "volume" else "ref_surface"],
                        "spp": spp, "chunk": chunk, "seed0": seed, "dry": L.DRY}
                if L.sidecar_ok(f, meta):
                    continue
                img, t = L.render(scene, ig, spp, chunk, seed)
                nf = L.save(f, img)
                L.write_sidecar(f, meta)
                timing[f"{exp}/{name}"] = {"seconds": t, "nonfinite": nf, **meta}
                L.write_json(tf, timing)
                print("ref", exp, name, round(t, 1), "nonfinite", nf, flush=True)
    L.finish(0)


# --- fixed 8DNA in the same scenes ------------------------------------------------------

def stage_neural() -> None:
    mi, dr = L.init()
    import torch

    pl = placements()
    out = L.OUT / "neural"
    rec = {"environment": L.environment(), "placements_sha256": placements_sha()}
    for exp in ("E1", "E2", "E3"):
        e = L.EXPERIMENTS[exp]
        c = X[SPP_KEY[exp]]["spp"]
        integ = L.neural_integrator(e["assets"])
        r = {"checkpoints": {n: L.sha256(L.ckpt(n)) for n in e["assets"].values()}, "params_before": None}
        first = True
        for state, variant in scenes_of(exp):
            scene = mi.load_dict(L.build(exp, state, variant, pl))
            L.attach(integ, scene)
            if first:
                r["params_before"] = L.param_digest(integ)
                L.render(scene, integ, 4, 4, 999)  # warm-up
                first = False
            tg = tag(state, variant)
            cnt = L.Counters()
            undo = L.instrument(cnt)
            L.render(scene, integ, 4, 4, 0)
            undo()
            per_model = L.per_model(cnt, integ)
            if len(per_model) != len(e["assets"]) or not all(v.get("learned_sample_lanes", 0) > 0 for v in per_model.values()):
                print(f"{exp}/{tg}: a learned asset was not active ({per_model})", flush=True)
                L.finish(1)
            s = {"counters_4spp": cnt.n, "per_model_4spp": per_model, "nonfinite": {}}
            torch.cuda.reset_peak_memory_stats()
            with L.MemWatch() as mw:
                for name, seed in (("main", 0), ("repeat", 100000)):
                    img, t = L.render(scene, integ, *L.S(c["neural"], c["neural_chunk"]), seed)
                    s["nonfinite"][name] = L.save(out / exp / f"{tg}_{name}.npy", img)
                    s[f"{name}_seconds"] = t
            s["peak_torch_bytes"] = int(torch.cuda.max_memory_allocated())
            s["peak_process_mib"] = mw.max_mib
            r[tg] = s
            print("neural", exp, tg, round(s["main_seconds"], 1), "nonfinite", s["nonfinite"], flush=True)
            rec[exp] = r
            L.write_json(out / "neural_runs.json", rec)
        r["params_after"] = L.check_params(r["params_before"], integ, f"neural {exp}")
        r["params_unchanged"] = True
        rec[exp] = r
        L.write_json(out / "neural_runs.json", rec)
    L.finish(0)


# --- equal-time / equal-error ladders -----------------------------------------------------

def stage_ladder() -> None:
    mi, dr = L.init()
    pl = placements()
    out = L.OUT / "ladder"
    rec = {"environment": L.environment()}
    for exp in ("E1", "E2"):
        e = L.EXPERIMENTS[exp]
        scene = mi.load_dict(L.build(exp, "T1_out", "red", pl))
        integ = L.neural_integrator(e["assets"])
        L.attach(integ, scene)
        before = L.param_digest(integ)
        ref_i = L.reference_integrator(e["kind"])
        L.render(scene, integ, 4, 4, 999)
        L.render(scene, ref_i, 4, 4, 999)
        r = {}
        for spp in (16, 64, 256, 1024):
            img, t = L.render(scene, integ, *L.S(spp, 4), P["seeds"]["ladder_neural"])
            L.save(out / exp / f"neural_{spp}.npy", img)
            r[f"neural_{spp}"] = t
        for spp in (16, 64, 256, 1024, 4096):
            img, t = L.render(scene, ref_i, *L.S(spp, min(spp, 64 if exp == "E1" else 16)), P["seeds"]["ladder_reference"])
            L.save(out / exp / f"ref_{spp}.npy", img)
            r[f"ref_{spp}"] = t
        L.check_params(before, integ, f"ladder {exp}")
        rec[exp] = r
        L.write_json(out / "ladder_timing.json", rec)
        print("ladder", exp, r, flush=True)
    L.finish(0)


# --- edit update cost -----------------------------------------------------------------------

def stage_cost() -> None:
    mi, dr = L.init()
    pl = placements()
    out = L.OUT / "cost"
    rec = {"environment": L.environment()}
    for exp in ("E1", "E2"):
        e = L.EXPERIMENTS[exp]
        c = X[SPP_KEY[exp]]["spp"]
        scene = mi.load_dict(L.build(exp, "T0", "red", pl))
        integ = L.neural_integrator(e["assets"])
        L.attach(integ, scene)
        before = L.param_digest(integ)
        L.render(scene, integ, 4, 4, 999)
        params = mi.traverse(scene)
        key = "box.vertex_positions"
        v0 = np.array(params[key])
        d = np.asarray(pl[exp]["states"]["T1_out"]["centre"]) - np.asarray(pl[exp]["states"]["T0"]["centre"])
        v1 = (v0.reshape(-1, 3) + d).ravel()
        times = []
        for k in range(10):  # alternate T0 -> T1 (even k) and T1 -> T0 (odd k)
            params[key] = mi.Float((v1 if k % 2 == 0 else v0).astype(np.float32))
            L.sync()
            t = time.perf_counter()
            params.update()
            L.sync()
            times.append(time.perf_counter() - t)
        params[key] = mi.Float(v1.astype(np.float32))
        params.update()
        img, t_render = L.render(scene, integ, *L.S(c["neural"], c["neural_chunk"]), 0)
        L.check_params(before, integ, f"cost {exp}")
        fp = L.OUT / "neural" / exp / "T1_out_red_main.npy"
        fresh = np.load(fp) if fp.exists() else None
        rec[exp] = {"update_seconds": times, "update_median_s": float(np.median(times[0::2])),
                    "update_note": "declared value: median of the 5 T0 -> T1 updates; all 10 alternating calls listed",
                    "render_after_update_s": t_render, "learned_parameters_updated": 0,
                    "learned_bytes": {n: L.ckpt(n).stat().st_size for n in e["assets"].values()}, "transport_state_bytes": 0,
                    "in_place_vs_fresh_scene_max_abs": None if fresh is None else float(np.abs(img - fresh).max()),
                    "in_place_vs_fresh_scene_mean_abs": None if fresh is None else float(np.abs(img - fresh).mean())}
        L.write_json(out / "cost.json", rec)
        print("cost", exp, rec[exp]["update_median_s"], round(t_render, 1), flush=True)
    L.finish(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "vcall_probe", "design", "static", "refs", "neural", "ladder", "cost"])
    ap.add_argument("--vcall", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "vcall_probe":
        vcall_probe(a.vcall)
    {"smoke": stage_smoke, "design": stage_design, "static": stage_static, "refs": stage_refs, "neural": stage_neural,
     "ladder": stage_ladder, "cost": stage_cost}[a.stage]()


if __name__ == "__main__":
    main()
