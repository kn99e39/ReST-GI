"""WL104K-C focused contracts (pre-fit and leak). Fitted-state physical contracts (reciprocity, envelope, MC albedo / passivity,
178 parameters, gates in [0, 1]) use the unchanged WL104K-B tests104kb.py --fitted through run_kb.py.

  python tests104kc.py prefit <out.json> <root>     response identity vs fit104kb.Data, region partition, augmented-Lagrangian
                                                    value / gradient (float64 finite differences), zero-step fallback = warm start,
                                                    synthetic target reproduction vs WL104K-B S_O0, warm-r4 layout
  python tests104kc.py leak <out.json> <root>       eval104kc test refuses without / with a tampered fit decision or changed states
Exit status 1 after writing when any test fails.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import common104kc as C

KB, CM = C.KB, C.CM
import fit104kb as FK  # noqa: E402
import fit104kc as FC  # noqa: E402

DEV = C.DEV
N = C.N_MAT


def entry(ok, **v):
    return {"passed": bool(ok), "values": v}


def prefit(root):
    t = {}
    ds = KB.training_set("original", N)
    x = torch.as_tensor(C.flat_x(C.load_states(C.KB_ROOT / "fits/O0/states.npz", 8)), device=DEV, dtype=torch.float32)
    T = FC.Terms(ds, N, 8)
    D = FK.Data(ds, N, 8, "original")
    with torch.no_grad():
        pt, dt = T.probe_terms(x), D.terms(x)
    a, b = FC.original_obj(pt).cpu().numpy(), dt["objective"].cpu().numpy()
    t["response_identity_vs_fit104kb"] = entry(np.array_equal(a, b), max_abs=float(np.abs(a - b).max()))
    D2 = FK.Data(ds, N, 8, "e2")
    with torch.no_grad():
        e2a, e2b = FC.e2_obj(pt).cpu().numpy(), D2.terms(x)["objective"].cpu().numpy()
    t["e2_identity_vs_fit104kb"] = entry(np.allclose(e2a, e2b, rtol=1e-6, atol=0), max_rel=float(np.max(np.abs(e2a - e2b) / e2b)))
    part = (T.core + T.shoulder + T.tail).cpu().numpy()
    t["region_partition"] = entry(np.array_equal(part, np.ones_like(part)), min=float(part.min()), max=float(part.max()))
    # augmented Lagrangian: float64 finite differences on one material, one start
    sub = {k: v[2:3] for k, v in ds.items()}
    T64 = FC.Terms(sub, 1, 1)
    for k in ("wi", "wo", "ref", "w", "core", "shoulder", "tail"):
        setattr(T64, k, getattr(T64, k).double())
    x64 = x[2:3].double()
    c = FC.cfg()
    with torch.no_grad():
        t0 = T64.probe_terms(x64)
    al = FC.AL(t0, c)
    rng = np.random.default_rng(1043360)
    al.lam = torch.as_tensor(rng.uniform(0, 1, al.lam.shape), device=DEV, dtype=torch.float64)
    al.rho = torch.full_like(al.rho, 37.0)
    xp = (x64 + torch.as_tensor(rng.normal(0, 1e-3, x64.shape), device=DEV, dtype=torch.float64)).clamp_min(0)
    xp[:, 58:] = xp[:, 58:].clamp(0.05, 0.95)
    xr = xp.clone().requires_grad_(True)
    L = al.lagrangian(T64.probe_terms(xr)).sum()
    (g,) = torch.autograd.grad(L, xr)
    errs = []
    for _ in range(6):
        v = torch.as_tensor(rng.normal(0, 1, x64.shape), device=DEV, dtype=torch.float64)
        h = 1e-6
        with torch.no_grad():
            fd = (al.lagrangian(T64.probe_terms(xp + h * v)).sum() - al.lagrangian(T64.probe_terms(xp - h * v)).sum()) / (2 * h)
        errs.append(float(abs(fd - (g * v).sum()) / max(abs(float(fd)), 1e-12)))
    t["al_gradient_fd"] = entry(np.median(errs) <= 1e-4, rel_errors=errs)
    with torch.no_grad():
        F0 = float(al.F(t0)[0]); cv = al.cviol(t0)
    t["al_warm_start_feasible"] = entry(abs(F0 - 1) < 1e-12 and float(cv.max()) <= 0, F=F0, max_c=float(cv.max()))
    # zero learning rate: every arm returns the warm start exactly (fallback path)
    cz = {**c, "lr0": 0.0, "lr_end": 0.0, "steps": 4, "inner": 2}
    for arm in ("CR", "O0C", "E2W"):
        xs = x[:N]
        T9 = FC.Terms(ds, N, 1)
        xb, ck, _, _, _ = FC.run(T9, xs.clone(), arm, cz, lambda s: None)
        t[f"zero_lr_returns_warm_start_{arm}"] = entry(torch.equal(xb, xs) and all(k == 0 for k in ck), checkpoints=ck)
    # synthetic targets: identical to WL104K-B S_O0 initial terms
    rows, raw, _, _, _, _ = CM.inputs()
    truth, target = FC.synthetic_targets(ds, raw)
    T4 = FC.Terms(ds, N, 4, target)
    x0 = torch.as_tensor(np.concatenate([KB.starts_y(N, KB.START_KEYS[:4]), np.ones((4 * N, 120))], 1), device=DEV, dtype=torch.float32)
    with torch.no_grad():
        o = FC.original_obj(T4.probe_terms(x0)).cpu().numpy()
    ref = np.asarray(C.read(C.KB_ROOT / "fits/S_O0/fit.json")["initial_terms"]["objective"], float)
    t["synthetic_target_reproduction"] = entry(np.array_equal(o.astype(np.float64), ref), max_abs=float(np.abs(o - ref).max()))
    # warm-r4 layout
    st8 = C.load_states(C.KB_ROOT / "fits/O0/states.npz", 8)
    st4 = C.load_states(C.KB_ROOT / "amendment1/fits/R4a/states.npz", 4)
    w = st4 + st8[4:]
    xw = C.flat_x(w)
    ok = np.array_equal(xw[:4 * N], C.flat_x(st4)) and np.array_equal(xw[4 * N:], C.flat_x(st8)[4 * N:])
    t["warm_r4_layout"] = entry(ok)
    return t


def leak(root):
    t = {}
    ev = Path(__file__).resolve().parent / "eval104kc.py"
    py = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for sub in ("stageA/fits", "fits", "data", "contracts"):
            if (root / sub).exists():
                shutil.copytree(root / sub, tmp / sub)
        r = subprocess.run([py, "-B", str(ev), "test", str(tmp)], capture_output=True, text=True)
        t["refuses_without_fit_decision"] = entry(r.returncode != 0 and "TEST_LOCKED" in (r.stderr + r.stdout),
                                                  tail=(r.stderr or r.stdout)[-300:])
        if (root / "fit_decision").exists():
            shutil.copytree(root / "fit_decision", tmp / "fit_decision")
            side = tmp / "fit_decision/fit_decision.json.sha256"
            good = side.read_text(encoding="utf-8")
            side.write_text("0" * 64 + "  fit_decision.json\n", encoding="utf-8")
            r = subprocess.run([py, "-B", str(ev), "test", str(tmp)], capture_output=True, text=True)
            t["refuses_tampered_fit_decision"] = entry(r.returncode != 0 and "TEST_LOCKED" in (r.stderr + r.stdout),
                                                       tail=(r.stderr or r.stdout)[-300:])
            side.write_text(good, encoding="utf-8")
            p = tmp / "fits/CR/states.npz"
            with np.load(p) as z:
                d = {k: z[k].copy() for k in z.files}
            d["P_y"][0, 0] += 1e-6
            np.savez(p, **d)
            r = subprocess.run([py, "-B", str(ev), "test", str(tmp)], capture_output=True, text=True)
            t["refuses_changed_states"] = entry(r.returncode != 0 and "TEST_LOCKED" in (r.stderr + r.stdout),
                                                tail=(r.stderr or r.stdout)[-300:])
    return t


def main():
    mode, out, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    t = prefit(root) if mode == "prefit" else leak(root)
    res = {"mode": mode, "tests": t, "passed": all(v["passed"] for v in t.values()), "commit": C.commit()}
    C.write(out, res)
    print(mode, {k: v["passed"] for k, v in t.items()}, flush=True)
    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
