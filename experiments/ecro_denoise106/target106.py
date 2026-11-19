"""WL106 Part E: additional independent 8192-spp halves of the frozen-ECRO expectation for the evaluation views.

python target106.py WL101_ROOT DATA105 OUT N_NEW [--smoke]

C0: archived 4096-spp V0 replay (render105 G1 rule) before any new half. G4: the new seed family
11600000 + 100 v + h (h < N_NEW <= 64) is disjoint from every WL105 seed and launch stream. Halves are rendered
with render105.frame_components (launch k = seed * 1000003 + k, 32 spp per launch), round-robin over views so
every view has the same count at any time; existing complete files are kept (resumable).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARGS = sys.argv[1:]
sys.path.insert(0, str(HERE.parent / "ecro_denoise105"))
import render105 as R5  # noqa: E402  (imports the frozen WL101 renderer; consumes argv)

BASE = 11600000


def main(root, data, out, n_new, smoke):
    reader, learned, base, wl101e, prior, history, reference, manifest = R5.RG.F101.inputs(root)
    R, rt = reader.R, reader.rt
    Bd = R.PT.fl(learned)
    out.mkdir(parents=True, exist_ok=True)
    dman = json.loads((data / "run_manifest.json").read_text())
    names = [k.split("/")[1] for k in dman["views"] if k.startswith("eval/")]
    run = {"inputs": manifest, "protocol_sha256": R5.sha(HERE / "protocol/denoise106.json"), "n_new": n_new, "smoke": smoke,
           "code_sha256": {f.name: R5.sha(f) for f in sorted(HERE.glob("*.py"))}, "views": {}}
    # G4 seed / launch-stream disjointness against every WL105 seed set.
    old = R5.seed_sets(R5.P["n_train_views"], R5.P["n_val_views"], len(names))
    old_seeds = {s for v in old.values() for s in v} | set(R5.P["seeds"]["replay"]) | {R5.P["seeds"][k] for k in ("determinism", "cameras", "training")}
    new = [BASE + 100 * v + h for v in range(len(names)) for h in range(n_new)]
    assert n_new <= 64 and len(set(new)) == len(new) and not (set(new) & old_seeds), "seed overlap"
    streams = set()
    for key, vals in old.items():
        nl = (R5.P["target_spp_eval"] if key.startswith("eval") else R5.P["target_spp_train"]) // 32 if "target" in key else 1
        for s in vals:
            streams.update(s * 1000003 + k for k in range(nl))
    for s in new:
        assert not ({s * 1000003 + k for k in range(256)} & streams), "launch stream overlap"
    run["G4_seed_disjoint"] = True
    # C0 renderer replay.
    if not smoke:
        g1 = []
        for row in history["matched"]:
            comp = R5.frame_components(R, rt, Bd, R5.CAM.V0, 4096, row["seed"])
            with np.load(prior / row["path"]) as f:
                for k, key in enumerate(("e1", "e2", "hl")):
                    ref = f[key].astype(np.float64).reshape(R5.RES, R5.RES, 3).transpose(2, 0, 1)
                    mine = comp[3 * k:3 * k + 3].astype(np.float64)
                    g1.append({"seed": row["seed"], "component": key, "relative_l1": float(np.abs(mine - ref).sum() / np.abs(ref).sum())})
        run["C0_replay"] = {"rows": g1, "pass": max(r["relative_l1"] for r in g1) <= R5.P["gates"]["replay_relative_l1"]}
        assert run["C0_replay"]["pass"], g1
        print("C0 replay pass", max(r["relative_l1"] for r in g1), flush=True)
    spp = 64 if smoke else 8192
    for h in range(n_new):
        for v, n in enumerate(names):
            cam = dman["views"][f"eval/{n}"]["camera"]
            seed = BASE + 100 * v + h + (5000000 if smoke else 0)
            f = out / n / f"target_new_h{h:02d}.npz"
            f.parent.mkdir(exist_ok=True)
            rec = run["views"].setdefault(n, {"camera": cam, "files": {}, "seconds": []})
            if f.exists():
                try:
                    with np.load(f) as z:
                        assert int(z["seed"]) == seed and z["comp"].shape == (9, R5.RES, R5.RES)
                    rec["files"][f.name] = R5.sha(f)
                    continue
                except Exception:
                    f.unlink()
            t0 = time.perf_counter()
            comp = R5.frame_components(R, rt, Bd, cam, spp, seed)
            np.savez_compressed(f, comp=comp, seed=seed, spp=spp)
            rec["files"][f.name] = R5.sha(f)
            rec["seconds"].append(time.perf_counter() - t0)
            print("half", h, n, round(rec["seconds"][-1], 1), flush=True)
        (out / "target_manifest.json").write_text(json.dumps(run, indent=1))
    print("TARGETS COMPLETE", flush=True)


if __name__ == "__main__":
    main(Path(ARGS[0]).resolve(), Path(ARGS[1]).resolve(), Path(ARGS[2]).resolve(), int(ARGS[3]), "--smoke" in ARGS)
