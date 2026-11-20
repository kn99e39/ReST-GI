"""WL104H frozen inputs: authentication against the WL104G manifest, materials, raw / compiled states, angular reference."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104d"))
import comp104h as K  # noqa: E402
from common104d import historical_states  # noqa: E402

P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
MANIFEST = REPO / "evaluation/104G/manifest.json"
ANGULAR = REPO / "results/ecro_wl84_104d/amendment1/amended_reference.npz"
COMPILED = REPO / "evaluation/104D/amendment1/compiled_states.npz"
FIT_STATES = REPO / "evaluation/104D/fit_states.npz"
MATERIALS = REPO / "evaluation/104D/amendment1/materials.json"
WL83_FITS = REPO / "results/fnto_compiled83/results"
G_RAW = REPO / "results/ecro_wl84_104g"
E_PARENT = REPO / "results/ecro_wl84_104e/parent/scenes"
OUT = REPO / "results/ecro_wl84_104h"
ANGLES = np.array([5, 20, 35, 50, 57.5, 60, 62.5, 70, 80, 86], float)
AZIMUTHS = np.array([22.5, 112.5])
POLAR = np.repeat(ANGLES, 2)
BLOCKS = 16


class Stop(RuntimeError):
    """A frozen stop condition was reached."""


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip()


def manifest_hashes():
    m = read(MANIFEST)
    table = {}
    for key in ("frozen_WL104F_input_sha256", "frozen_WL104G_fit_input_sha256"):
        table.update({k.replace("\\", "/"): v for k, v in m[key].items()})
    table.update({("results/ecro_wl84_104g/" + k).replace("\\", "/"): v for k, v in m["raw_sha256"].items()})
    return table


def authenticate(paths):
    """Check every path (absolute) against the manifest; returns {relative: sha}."""
    table = manifest_hashes()
    seen = {}
    for p in paths:
        rel = Path(p).resolve().relative_to(REPO).as_posix()
        if rel not in table:
            raise Stop(f"NO_FROZEN_HASH: {rel}")
        digest = sha(p)
        if digest != table[rel]:
            raise Stop(f"FROZEN_HASH_MISMATCH: {rel}")
        seen[rel] = digest
    return seen


def compile_raw(raw):
    with torch.no_grad():
        z = torch.as_tensor(np.asarray(raw), device=K.H.dev, dtype=torch.float32)
        return K.CD.compiled_state("ER", K.CL.decode("ER", "cert", z))


def decode64(raw_row):
    z = torch.as_tensor(np.asarray(raw_row, np.float64)[None], device=K.H.dev, dtype=torch.float64)
    return K.CL.decode("ER", "cert", z)


def inputs():
    """Rows, raw states {R0, A, C} [9, 64], compiled states {R0, A, C} [9, 69] and the authenticated hash table."""
    rows = read(MATERIALS)
    fits = sorted({WL83_FITS / "fit" / f"fit_{r['set']}_ER_cert_metric_12000_generic.npz" for r in rows})
    seen = authenticate([ANGULAR, COMPILED, FIT_STATES, MATERIALS, *fits])
    raw = {"R0": historical_states(WL83_FITS, rows)["ER_DF"]}
    with np.load(FIT_STATES) as z:
        raw["A"], raw["C"] = z["A"].astype(np.float64), z["C"].astype(np.float64)
    with np.load(COMPILED) as z:
        compiled = {"R0": z["ER_DF"], "A": z["A"], "C": z["C"]}
    for k in raw:
        if raw[k].shape != (len(rows), 64) or compiled[k].shape != (len(rows), 69):
            raise Stop(f"STATE_LAYOUT: {k}")
        if not np.allclose(compile_raw(raw[k]), compiled[k], atol=2e-6, rtol=1e-5):
            raise Stop(f"COMPILED_STATE_MISMATCH: {k}")
    return rows, raw, compiled, seen


def reference():
    with np.load(ANGULAR) as z:
        return {k: z[k] for k in z.files}


def reference_precision(ref, nm):
    """WL104D Amendment 1 rule recomputed on the archived halves (must pass)."""
    sys.path.insert(0, str(HERE.parent / "ecro_wl84_104d"))
    from response104d import metric_arrays, terms
    w = (ref["use"] / ref["q"])[..., None]
    noise = metric_arrays(terms(ref["a"], ref["b"], w), nm)
    band = (POLAR >= 55) & (POLAR <= 65)
    graze = POLAR >= 80
    rows = [{"band_shape": float(sh[band].mean() * .5), "grazing_shape": float(sh[graze].mean() * .5),
             "albedo": float(a.mean() * .5)} for sh, a in zip(noise["eshape"], noise["ealb"])]
    ok = all(r["band_shape"] <= .015 and r["grazing_shape"] <= .025 and r["albedo"] <= .008 for r in rows)
    return rows, ok
