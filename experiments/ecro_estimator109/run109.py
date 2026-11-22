"""WL109 final evidence capture on the evaluation views (V0, H1-H5).

python run109.py OUT DEV_JSON SYNTH_JSON [--smoke] [--escalate]

Stage 0  admission (frozen_runtime incl. archived 4096-spp replay) + replay of archived WL105 evaluation frames at 4/16/32 spp
Stage 1  paired R0 / C1 frames at the nominal spp (same seed: identical main stream; C1 HIGH from the independent inner stream)
Stage 2  R0 quality-versus-time grid frames (independent seeds)
Stage 3  synchronized frozen timing of every R0 grid level and lean C1 at the nominal spp, end-to-end composition, first-call
         (recording) cost, peak Dr.Jit allocation; foreign GPU processes are waited out
--escalate  the predeclared bias-precision escalation: paired replicates 32..63 for every view and nominal spp
"""
import argparse
from pathlib import Path
import time

import numpy as np

from common109 import P, EVAL_VIEWS, GRID, NOMINAL, apply_smoke, read, seed_grid, seed_paired, seed_timing, sha, write, HERE
import runtime109 as RT


def rel_l1(a, b):
    den = float(np.abs(b).sum())
    return float(np.abs(a - b).sum()) / den if den > 0 else float(np.abs(a - b).sum())


def main(out, dev_json, synth_json, smoke, escalate):
    import torch
    from gpu109 import Engine, timing_schedule
    if smoke:
        apply_smoke()
    dev, syn = read(dev_json), read(synth_json)
    assert dev["complete"] and syn["complete"] and (smoke or (syn["pass"] and not dev["futile"])), "upstream gate"
    assert dev["protocol_sha256"] == syn["protocol_sha256"] == sha(HERE / "protocol109.json")
    assert dev["smoke"] == syn["smoke"] == smoke
    M = int(dev["selected_M"])
    cfg = P["final"]
    tol = P["synthetic"]["structural_rel_l1"]
    if escalate:
        assert (out / "manifest.json").exists()
        man = read(out / "manifest.json")
        r_range = range(cfg["replicates"], cfg["escalated_replicates"])
    else:
        out.mkdir(parents=True, exist_ok=False)
        man = {"protocol_sha256": sha(HERE / "protocol109.json"), "smoke": smoke, "selected_M": M,
               "dev_json_sha256": sha(dev_json), "synth_json_sha256": sha(synth_json),
               "code_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}, "files": {}, "identity": {}, "complete": False}
        r_range = range(cfg["replicates"])
    logf = open(out / "run.log", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(time.strftime("%H:%M:%S"), s, flush=True); logf.write(time.strftime("%H:%M:%S ") + s + "\n"); logf.flush()
    R, rt, Bd, audit, r105 = RT.admit()
    eng = Engine(R, rt, Bd)
    cams = RT.cameras()
    table_sha = audit["table_bytes_sha256"]
    man.setdefault("admission", []).append(audit)
    man.setdefault("versions", []).append(RT.versions(R))
    raw = out / "raw"

    def save(rel, arr):
        p = raw / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr, allow_pickle=False)
        man["files"][rel] = sha(p)

    if not escalate:
        # ---- Stage 0b: archived WL105 evaluation frames (original renderer path)
        rep = []
        for view in EVAL_VIEWS:
            for spp in (4, 16, 32):
                comp, seed = RT.archived_noisy(view, spp, 0)
                b = min(32, spp)
                acc = None
                for k in range(spp // b):
                    o = rt._frame_prog(rt.scene, rt.tabs(), rt.light_args(), Bd, R.dr.opaque(R.mi.UInt64, seed * 1000003 + k),
                                       (dict(cams[view], res=512), 512, b))
                    R.dr.eval(o)
                    t = torch.stack([x.torch().double() for x in o])
                    acc = t if acc is None else acc + t
                mine = (acc / spp).cpu().numpy().reshape(9, 512, 512).astype(np.float32)
                rep.append({"view": view, "spp": spp, "seed": seed, "rel_l1": rel_l1(mine, comp), "bitwise": bool(np.array_equal(mine, comp))})
        man["archived_eval_replay"] = rep
        assert max(x["rel_l1"] for x in rep) <= r105.P["gates"]["replay_relative_l1"], rep
        log("archived WL105 eval replay", max(x["rel_l1"] for x in rep))
        write(out / "manifest.json", man)

    # ---- Stage 1: paired nominal frames
    for vi, view in enumerate(EVAL_VIEWS):
        for spp in NOMINAL:
            for r in r_range:
                seed = seed_paired(vi, spp, r, smoke)
                r0 = eng.r0(cams[view], spp, seed)
                c1 = eng.c1_frame(cams[view], spp, seed, M, orig_hl=True)       # keys e1 e2 hl ho
                ident = {"low_rel_l1": rel_l1(c1[0:6].cpu().numpy(), r0[0:6].cpu().numpy()),
                         "ho_rel_l1": rel_l1(c1[9:12].cpu().numpy(), r0[6:9].cpu().numpy()),
                         "low_bitwise": bool(torch.equal(c1[0:6], r0[0:6])), "ho_bitwise": bool(torch.equal(c1[9:12], r0[6:9]))}
                assert ident["low_rel_l1"] <= tol and ident["ho_rel_l1"] <= tol, (view, spp, r, ident)
                man["identity"][f"{view}/{spp}/{r}"] = ident
                save(f"paired/{view}/spp{spp}/r{r:02d}_R0.npy", r0.cpu().numpy())
                save(f"paired/{view}/spp{spp}/r{r:02d}_C1H.npy", c1[6:9].cpu().numpy())
            log("paired", view, spp, "done")
        write(out / "manifest.json", man)
    if escalate:
        man["escalated"] = True
        man["table_sha_after_escalation"] = RT.table_bytes_sha(Bd)
        assert man["table_sha_after_escalation"] == table_sha
        write(out / "manifest.json", man)
        log("ESCALATION COMPLETE")
        return

    # ---- Stage 2: R0 grid
    for vi, view in enumerate(EVAL_VIEWS):
        for spp in GRID:
            for r in r_range:
                save(f"grid/{view}/spp{spp:03d}/r{r:02d}.npy", eng.r0(cams[view], spp, seed_grid(vi, spp, r, smoke)).cpu().numpy())
        log("grid", view, "done")
        write(out / "manifest.json", man)

    # ---- Stage 3: timing and memory (each view's block on an exclusive GPU, repeated if contaminated)
    timing = {}
    for vi, view in enumerate(EVAL_VIEWS):
        slots = [("R0", s, 0) for s in GRID] + [("C1", s, M) for s in NOMINAL]

        def block():
            rec = {"first_call_ms": {}, "frame_ms": {}, "e2e_ms": {}, "watermark_bytes": {}}
            eng._frozen.clear()                          # first call = recording + compile, measured per block
            for si, (kind, spp, m) in enumerate(slots):
                key = f"{kind}_{spp}"
                eng.sync()
                t0 = time.perf_counter()
                eng.run_frozen(kind, view, cams[view], spp, seed_timing(vi, si, 0, smoke), m)
                eng.sync()
                rec["first_call_ms"][key] = 1e3 * (time.perf_counter() - t0)
                for w in range(cfg["timing_warmup"]):
                    eng.run_frozen(kind, view, cams[view], spp, seed_timing(vi, si, 1 + w, smoke), m)
                rec["watermark_bytes"][key] = eng.watermark(kind, view, cams[view], spp, seed_timing(vi, si, 9, smoke), m)
                rec["frame_ms"][key] = []
                if spp in NOMINAL:
                    rec["e2e_ms"][key] = []
            for rr, (kind, spp, m) in timing_schedule(slots, cfg["timing_rounds"], P["seeds"]["timing_order"] + 10 + vi):
                si = slots.index((kind, spp, m))
                key = f"{kind}_{spp}"
                rec["frame_ms"][key].append(eng.timed(kind, view, cams[view], spp, seed_timing(vi, si, 10 + rr, smoke), m))
                if spp in NOMINAL:
                    rec["e2e_ms"][key].append(eng.timed(kind, view, cams[view], spp, seed_timing(vi, si, 50 + rr, smoke), m, compose=True))
            return rec
        rec, excl = RT.exclusive_block(block, log, require=not smoke)
        rec["exclusivity"] = excl
        timing[view] = rec
        log("timing", view, {k: round(float(np.median(v)), 3) for k, v in rec["frame_ms"].items()})
    man["timing"] = timing
    man["table_sha_after"] = RT.table_bytes_sha(Bd)
    assert man["table_sha_after"] == table_sha
    man["complete"] = True
    write(out / "manifest.json", man)
    log("RUN COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dev_json")
    ap.add_argument("synth_json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--escalate", action="store_true")
    a = ap.parse_args()
    main(Path(a.out).resolve(), Path(a.dev_json).resolve(), Path(a.synth_json).resolve(), a.smoke, a.escalate)
