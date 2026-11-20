"""WL104H parameter-level local sensitivity (no optimisation): exact float64 forward-mode Jacobian of the frozen certified ER response
at the R0 raw state, group column-space projections of the noise-unbiased residual, principal angles, singular spectra, FD check, and
integrated-gradient attribution of the WL104D C - A response change.

Usage: python params104h.py <out_dir> [--smoke]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.stats as sst
import torch

import io104h as IO
import comp104h as K

DEV = K.H.dev
RCOND = 1e-10
T15 = float(sst.t.ppf(0.975, IO.BLOCKS - 1))
BANDS = IO.P["response_phase"]["strata"]["band"]


def response_fn(wi, wo):
    def f(z):
        p = K.CL.decode("ER", "cert", z[None])
        return K.native_torch(p, wi, wo).reshape(-1)
    return f


def jacobian(z, wi, wo):
    return torch.func.jacfwd(response_fn(wi, wo))(z)          # [N*3, 64]


def fd_errors(f, J, z0, RW=None, h=1e-4):
    """Central-difference check (protocol): |FD - AD| / max(|AD col|, 1e-4 max col norm) per raw coordinate."""
    RW = torch.ones(J.shape[0], device=J.device, dtype=J.dtype) if RW is None else RW
    norms = torch.linalg.norm(J * RW[:, None], dim=0)
    floor = 1e-4 * float(norms.max())
    errs = []
    for i in range(J.shape[1]):
        e = torch.zeros(J.shape[1], device=J.device, dtype=J.dtype)
        e[i] = h
        with torch.no_grad():
            col = (f(z0 + e) - f(z0 - e)) / (2 * h)
        errs.append(float(torch.linalg.norm((col - J[:, i]) * RW) / max(float(norms[i]), floor)))
    return errs


def basis(J, rows):
    """Orthonormal column-space basis (SVD, relative rcond) of J restricted to rows."""
    U, s, _ = torch.linalg.svd(J[rows], full_matrices=False)
    keep = s > RCOND * s.max() if s.numel() and s.max() > 0 else torch.zeros_like(s, dtype=torch.bool)
    return U[:, keep], s


def proj_stats(J, ra, rb, rows):
    U, s = basis(J, rows)
    pa, pb = U.T @ ra[rows], U.T @ rb[rows]
    return float((pa * pb).sum()), U, s


def jk(full, loo):
    loo = np.asarray(loo, float)
    B = len(loo)
    se = float(np.sqrt((B - 1) / B * ((loo - loo.mean()) ** 2).sum()))
    return {"point": float(full), "se": se, "lo": float(full - T15 * se), "hi": float(full + T15 * se)}


def material(m, ref, raw, groups, lobes, fd):
    sl = slice(m * 20, (m + 1) * 20)
    wi = torch.as_tensor(ref["wi"][sl].reshape(-1, 3), device=DEV, dtype=torch.float64)
    wo = torch.as_tensor(ref["wo"][sl].reshape(-1, 3), device=DEV, dtype=torch.float64)
    a, b = ref["a"][sl], ref["b"][sl]
    w = ref["use"][sl] / ref["q"][sl]
    A = (w[..., None] * 0.5 * (a + b)).sum((1, 2))
    rw = np.repeat((np.sqrt(w) / A[:, None]).reshape(-1), 3)                      # row weight per (lane, channel)
    RW = torch.as_tensor(rw, device=DEV)
    z0 = torch.as_tensor(raw["R0"][m], device=DEV, dtype=torch.float64)
    f = response_fn(wi, wo)
    with torch.no_grad():
        F0 = f(z0)
    J = jacobian(z0, wi, wo)
    res = {}
    if fd:
        errs = fd_errors(f, J, z0, RW)
        res["fd_max_column_relative_error"] = max(errs)
        res["fd_column_relative_error"] = errs
    Jw = J * RW[:, None]
    ra = torch.as_tensor(a.reshape(-1), device=DEV) * RW - F0 * RW
    rb = torch.as_tensor(b.reshape(-1), device=DEV) * RW - F0 * RW
    B = torch.as_tensor(np.concatenate(list(groups.values()), 1), device=DEV)
    JB = Jw @ B
    cols, start = {}, 0
    for g, M in groups.items():
        cols[g] = list(range(start, start + M.shape[1]))
        start += M.shape[1]
    lane_blk = np.repeat(np.tile(np.arange(1024) % IO.BLOCKS, 20), 3)
    probe = np.repeat(np.repeat(np.arange(20), 1024), 3)
    rad = np.repeat(_radius(ref["wi"][sl], ref["wo"][sl]).reshape(-1), 3)
    strata = {f"{band}|all": np.isin(IO.POLAR[probe], ang) if ang != "all" else np.ones_like(probe, bool)
              for band, ang in BANDS.items()}
    strata.update({"all|core": rad < 3, "all|shoulder": (rad >= 3) & (rad < 12), "all|tail": rad >= 12})
    out = {}
    for key, sel in strata.items():
        rows_full = np.nonzero(sel)[0]
        def r2_all_sets(rows):
            rows_t = torch.as_tensor(rows, device=DEV)
            den = float((ra[rows_t] * rb[rows_t]).sum())
            vals = {}
            vals["all"], _, _ = proj_stats(JB, ra, rb, rows_t)
            for g, c in cols.items():
                vals[g], _, _ = proj_stats(JB[:, c], ra, rb, rows_t)
                rest = [i for i in range(64) if i not in c]
                vals[f"all_minus_{g}"], _, _ = proj_stats(JB[:, rest], ra, rb, rows_t)
            return den, vals
        den, vals = r2_all_sets(rows_full)
        loo = []
        for bb in range(IO.BLOCKS):
            loo.append(r2_all_sets(rows_full[lane_blk[rows_full] != bb]))
        rec = {"residual_cross_energy": den}
        for k in vals:
            rec[f"R2_{k}"] = jk(vals[k] / den if den > 0 else np.nan, [v[k] / d if d > 0 else np.nan for d, v in loo])
        for g in cols:
            u_full = (vals["all"] - vals[f"all_minus_{g}"]) / den if den > 0 else np.nan
            rec[f"unique_R2_{g}"] = jk(u_full, [(v["all"] - v[f"all_minus_{g}"]) / d if d > 0 else np.nan for d, v in loo])
        # per-lobe sub-blocks (descriptive, full data)
        rows_t = torch.as_tensor(rows_full, device=DEV)
        lob = {}
        for name, sub in lobes.items():
            for pg, M in sub.items():
                v, _, _ = proj_stats(Jw @ torch.as_tensor(M, device=DEV), ra, rb, rows_t)
                lob[f"{name}.{pg}"] = v / den if den > 0 else None
        rec["lobe_R2"] = lob
        out[key] = rec
    res["strata"] = out
    # singular spectra, null directions, principal angles (all rows)
    allrows = torch.arange(JB.shape[0], device=DEV)
    U_all, s_all = basis(JB, allrows)
    _, s_glob, Vh = torch.linalg.svd(JB, full_matrices=False)
    smax = float(s_glob.max())
    rel = (s_glob / smax).cpu().numpy()
    Vn = Vh.cpu().numpy()
    names = list(cols)
    load = lambda v: {g: float((v[c] ** 2).sum()) for g, c in cols.items()}
    res["global_singular_relative"] = rel.tolist()
    res["null_directions"] = [{"sigma_rel": float(rel[i]), "group_loading": load(Vn[i])} for i in np.nonzero(rel < 1e-6)[0]]
    res["weak_directions"] = [{"sigma_rel": float(rel[i]), "group_loading": load(Vn[i])}
                              for i in np.nonzero((rel >= 1e-6) & (rel < 1e-3))[0]]
    spectra, Q = {}, {}
    for g, c in cols.items():
        Ug, sg = basis(JB[:, c], allrows)
        spectra[g] = (sg / smax).cpu().numpy().tolist()
        Q[g] = Ug
    res["group_singular_relative_to_global_max"] = spectra
    pa = {}
    for i, g in enumerate(names):
        for h in names[i + 1:]:
            if Q[g].shape[1] and Q[h].shape[1]:
                c = torch.linalg.svdvals(Q[g].T @ Q[h]).cpu().numpy()
                pa[f"{g}|{h}"] = c.tolist()
    res["principal_cosines"] = pa
    # integrated gradients for C - A (straight raw path), group attribution projected on Delta_CA
    zA = torch.as_tensor(raw["A"][m], device=DEV, dtype=torch.float64)
    zC = torch.as_tensor(raw["C"][m], device=DEV, dtype=torch.float64)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    tq, wq = (nodes + 1) / 2, weights / 2
    Jbar = torch.zeros_like(J)
    for t, wt in zip(tq, wq):
        Jbar += float(wt) * jacobian(zA + float(t) * (zC - zA), wi, wo)
    with torch.no_grad():
        dCA = f(zC) - f(zA)
    dzB = B.T @ (zC - zA)
    attr = (Jbar @ B) * dzB[None, :]
    total = attr.sum(1)
    res["ig_completeness_relative"] = float(torch.linalg.norm((total - dCA) * RW) / torch.linalg.norm(dCA * RW))
    dW = dCA * RW
    ig = {}
    for key, sel in strata.items():
        rows_t = torch.as_tensor(np.nonzero(sel)[0], device=DEV)
        den = float((dW[rows_t] ** 2).sum())
        ig[key] = {g: float(((attr[rows_t][:, c].sum(1) * RW[rows_t]) * dW[rows_t]).sum() / den) for g, c in cols.items()}
        ig[key]["raw_parameter_change_norm"] = {g: float(torch.linalg.norm(dzB[c])) for g, c in cols.items()}
    res["ig_ca_shares"] = ig
    return res


def _radius(wi, wo):
    m = np.stack([-wi[..., 0], -wi[..., 1], wi[..., 2]], -1)
    return np.degrees(np.arccos(np.clip((m * wo).sum(-1), -1, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, compiled, seen = IO.inputs()
    ref = IO.reference()
    groups, lobes = K.raw_groups()
    mats = [0] if args.smoke else range(len(rows))
    result = {}
    for m in mats:
        r = material(m, ref, raw, groups, lobes, fd=rows[m]["cohort"] == "development")
        if "fd_max_column_relative_error" in r and r["fd_max_column_relative_error"] > 1e-6:
            IO.write(args.out / "stop.json", {"stop": "JACOBIAN_FD_CHECK", "material": rows[m]["tag"],
                                              "fd": r["fd_column_relative_error"]})
            raise IO.Stop("JACOBIAN_FD_CHECK")
        result[rows[m]["tag"]] = r
        print("params material", rows[m]["tag"], round(time.monotonic() - t0, 1), flush=True)
        if args.smoke:
            IO.write(args.out / "smoke.json", {"fd": r.get("fd_max_column_relative_error"),
                                               "ig_completeness": r["ig_completeness_relative"],
                                               "seconds": time.monotonic() - t0})
            return
    IO.write(args.out / "params.json", {"commit": IO.commit(), "protocol_sha256": IO.sha(IO.HERE / "protocol.json"),
                                        "inputs_sha256": seen, "materials": rows, "groups": {g: int(M.shape[1]) for g, M in groups.items()},
                                        "rcond": RCOND, "results": result, "seconds": time.monotonic() - t0})
    print("params done", round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
