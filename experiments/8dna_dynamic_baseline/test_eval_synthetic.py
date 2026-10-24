"""Synthetic check of dyn_eval.py / dyn_exports.py (CPU only, no renderer): fabricate every stage output at the dry-run
resolution with known answers, run the evaluation, and assert the declared classification recovers them.

    EDNA_DRYRUN=1 EDNA_OUT=<tmp dir> python -I test_eval_synthetic.py

Construction: per experiment a stationary 'asset' disc (label 1), a box patch (label 3) that moves between states, support
elsewhere (label 4). The reference change T0 -> T1_out is reproduced by the fake 8DNA output (expected TRACKED); T0 -> T1_in
is ignored by it (expected NOT_TRACKED). E3 tracks (TRACKED).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dynlib as L  # noqa: E402
from dyn_stages import STATES, scenes_of, tag  # noqa: E402

assert L.DRY, "run with EDNA_DRYRUN=1"
OUT = L.OUT
rng = np.random.default_rng(0)


def main():
    classes = {}
    neural_runs = {}
    for exp in ("E1", "E2", "E3"):
        res = L.R(L.EXPERIMENTS[exp]["res"])
        yy, xx = np.mgrid[0:res, 0:res]
        disc = (xx - res / 2) ** 2 + (yy - res / 2) ** 2 < (res / 3) ** 2
        base = np.stack([0.4 + 0.2 * xx / res, 0.3 + 0.1 * yy / res, 0.2 * np.ones_like(xx, float)], -1).astype(np.float32)
        boxpos = {"T0": (1, 1), "T1_out": (1, res - 6), "T1_in": (res // 2 - 2, 1), "T1": (2, 2)}
        change = {"T0": 0.0, "T1_out": 0.15, "T1_in": 0.12, "T1": 0.1}
        noise_main = rng.normal(0, 0.003, base.shape).astype(np.float32)  # paired seeds: identical in every state
        runs = {}
        for state, variant in scenes_of(exp):
            t = tag(state, variant)
            cl = np.full((res, res), 4, np.int16)
            cl[disc] = 1
            if variant != "absent" and exp != "E3":
                r0, c0 = boxpos[state]
                cl[r0:r0 + 5, c0:c0 + 5] = 3
            classes[f"{exp}/{t}"] = cl
            # reference: base + a smooth change on the asset that grows with the state; black variant halves it
            c = change.get(state, 0.0) * (0.5 if variant == "black" else 1.0) * (variant != "absent")
            shade = (np.exp(-((xx - res * 0.6) ** 2 + (yy - res * 0.4) ** 2) / (2 * (res / 6) ** 2)) * c)[..., None]
            G = base - shade
            A = G + rng.normal(0, 0.002, G.shape).astype(np.float32)
            B = G + rng.normal(0, 0.002, G.shape).astype(np.float32)
            # fake 8DNA: tracks the change except in T1_in, where it keeps the T0 (box-free) response
            tracked = 0.0 if state == "T1_in" else 1.0
            N = base - shade * tracked + noise_main + 0.01
            Nr = base - shade * tracked + rng.normal(0, 0.003, base.shape).astype(np.float32) + 0.01
            L.save(OUT / "refs" / exp / f"{t}_A.npy", A)
            L.save(OUT / "refs" / exp / f"{t}_B.npy", B)
            if exp == "E2" and variant in ("black", "absent"):
                L.save(OUT / "refs" / exp / f"{t}_direct_A.npy", 0.5 * A)
                L.save(OUT / "refs" / exp / f"{t}_direct_B.npy", 0.5 * B)
            L.save(OUT / "neural" / exp / f"{t}_main.npy", N)
            L.save(OUT / "neural" / exp / f"{t}_repeat.npy", Nr)
            runs[t] = {"counters_4spp": {"learned_nee_lanes": 100, "learned_sample_lanes": 120, "ray_intersect_width": 4096},
                       "main_seconds": 1.0, "peak_torch_bytes": 1, "peak_process_mib": 1}
        runs["params_unchanged"] = True
        neural_runs[exp] = runs
        if exp in ("E1", "E2"):
            g = 0.5 * (np.load(OUT / "refs" / exp / "T1_out_red_A.npy") + np.load(OUT / "refs" / exp / "T1_out_red_B.npy"))
            lad = {}
            for spp, s in ((16, 0.05), (64, 0.025), (256, 0.0125), (1024, 0.00625)):
                L.save(OUT / "ladder" / exp / f"neural_{spp}.npy", g + 0.01 + rng.normal(0, s, g.shape).astype(np.float32))
                lad[f"neural_{spp}"] = spp / 64
            for spp, s in ((16, 0.2), (64, 0.1), (256, 0.05), (1024, 0.025), (4096, 0.0125)):
                L.save(OUT / "ladder" / exp / f"ref_{spp}.npy", g + rng.normal(0, s, g.shape).astype(np.float32))
                lad[f"ref_{spp}"] = spp / 32
            neural_runs.setdefault("_ladder", {})[exp] = lad
    (OUT / "design").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "design" / "classes.npz", **classes)
    lad = neural_runs.pop("_ladder")
    L.write_json(OUT / "neural" / "neural_runs.json", neural_runs)
    L.write_json(OUT / "ladder" / "ladder_timing.json", lad)
    L.write_json(OUT / "cost" / "cost.json", {"E1": {"update_median_s": 0.01}, "E2": {"update_median_s": 0.02}})
    for key in ("seal_scene2", "teaset_get_scene"):
        g = rng.uniform(0.1, 0.5, (32, 32, 3)).astype(np.float32)
        for i in range(2):
            L.save(OUT / "static" / key / f"ref_{i}.npy", g + rng.normal(0, 0.002, g.shape).astype(np.float32))
            L.save(OUT / "static" / key / f"neural_{i}.npy", g + rng.normal(0, 0.01, g.shape).astype(np.float32))

    env = dict(os.environ)
    for script in ("dyn_eval.py", "dyn_exports.py"):
        p = subprocess.run([sys.executable, str(HERE / script)], env=env, cwd=str(HERE), capture_output=True, text=True)
        print(script, p.returncode, p.stdout[-1500:], p.stderr[-3000:])
        assert p.returncode == 0, script
    ev = json.loads((OUT / "eval" / "eval.json").read_text())
    got = {(e, k): ev[e][k]["asset_receivers"]["class"] for e in ("E1", "E2") for k in ev[e] if k.startswith("T0->")}
    got[("E3", "T0->T1")] = ev["E3"]["T0->T1"]["seal_receivers"]["class"]
    want = {("E1", "T0->T1_out"): "TRACKED", ("E1", "T0->T1_in"): "NOT_TRACKED", ("E2", "T0->T1_out"): "TRACKED",
            ("E2", "T0->T1_in"): "NOT_TRACKED", ("E3", "T0->T1"): "TRACKED"}
    print("classes", got)
    assert got == want, (got, want)
    r = ev["E1"]["T0->T1_out"]["recovery"]["R"]
    assert r is not None and r > 0.8, r
    lad = ev["ladder"]["E1"]["summary"]["mse_full"]
    print("ladder", lad)
    assert lad["pt_seconds_to_same_error"] is not None
    print("SYNTHETIC EVAL PASS")


if __name__ == "__main__":
    main()
