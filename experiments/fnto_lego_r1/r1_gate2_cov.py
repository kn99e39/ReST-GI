"""WL42 Gate 2c - excitation coverage of the r1 Stage A incident fields (PC; Mitsuba only for the view-6 sun visibility).

Same measure as the Batch 6 audit check F2 (r1_audit_checks.f2): per training image the direct incident irradiance of
every port (reconstructed exactly from r1_prep: light table x port visible fraction x cos / r^2), area-integrated into a
coarse basis, unit-normalised; SVD; effective rank at 90 / 99 / 99.9 % energy; share of the energy of
  indirect_one_bounce : one Lambertian bounce (albedo 0.5) of the held-out T0 view-6 sun over the r1 T0 coupling graph
  direct_sun_same_view: that sun's direct field
  holdout_fields      : the 120 independent r1 holdout fields (median, q90)
outside the rank-99 span and outside the full span of the training fields.
Bases: 'r1_coarse' = the r1 8 cm coarse ports (declared Gate 2c basis); 'common' = 8 cm voxel x dominant signed axis x
object, built identically for the Batch 6 and the r1 ports, so the Batch 6 fields (stagea/prep_train/portE.npy of
results/fnto_lego, unchanged) are measured in the same basis for the before/after comparison.

    python r1_gate2_cov.py
"""
import json

import numpy as np

import r1_common as C
import fnto_relations as R
import r1_relations as RR


def common_key(pos, nrm, obj):
    ijk = np.floor(pos / 0.08).astype(np.int64) + 1024
    a = np.argmax(np.abs(nrm), 1); ax = a * 2 + (nrm[np.arange(len(nrm)), a] > 0)
    return (((ijk[:, 0] * 2048 + ijk[:, 1]) * 2048 + ijk[:, 2]) * 6 + ax) * 2 + obj.astype(np.int64)


def direct_field(pos, nrm, T, nl, vis, nsamp):
    E = np.zeros(len(pos))
    for k in range(nl):
        L = T[k]; f = vis[k].astype(np.float64) / np.maximum(nsamp, 1)
        if L[0] == 0:
            w = L[1:4] / np.linalg.norm(L[1:4]); E += L[8] * np.clip(nrm @ w, 0, None) * f
        else:
            v = L[1:4][None] - pos; r2 = np.maximum((v * v).sum(1), 1e-8); w = v / np.sqrt(r2)[:, None]
            E += L[8] * np.clip((nrm * w).sum(1), 0, None) / r2 * f
    return E


def svd_stats(X, probes):
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    _, S, Vt = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    en = np.cumsum(S ** 2) / (S ** 2).sum(); k99 = int(np.searchsorted(en, 0.99) + 1)
    out = {"fields": int(X.shape[0]), "basis_dim": int(X.shape[1]), "effective_rank_90_99_999": [int(np.searchsorted(en, q) + 1) for q in (0.9, 0.99, 0.999)],
           "basis_elements_never_excited": int((np.abs(X).sum(0) == 0).sum())}
    for name, v in probes.items():
        v = np.atleast_2d(v); v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        o99 = ((v - (v @ Vt[:k99].T) @ Vt[:k99]) ** 2).sum(1); oall = ((v - (v @ Vt.T) @ Vt) ** 2).sum(1)
        if len(v) == 1:
            out[f"{name}_energy_outside_span_rank99"] = float(o99[0]); out[f"{name}_energy_outside_full_span"] = float(oall[0])
        else:
            out[f"{name}_energy_outside_span_rank99_median_q90"] = np.quantile(o99, [0.5, 0.9]).tolist()
            out[f"{name}_energy_outside_full_span_median_q90"] = np.quantile(oall, [0.5, 0.9]).tolist()
    return out


def r1_fields(A, split, basis, dims):
    d = C.OUT1 / f"stagea/prep_{split}"
    pv = np.load(d / "port_vis.npy", mmap_mode="r"); LT = np.load(d / "lights.npy"); nl = np.load(d / "nlights.npy"); ns = np.load(d / "nsamp.npy")
    pos = A["pos"].astype(np.float64); nrm = A["nrm"].astype(np.float64); area = A["area"].astype(np.float64)
    cfg = sorted(json.loads((C.OUT1 / f"stagea/{split}/configs.json").read_text()), key=lambda c: c["i"])
    out = {b: np.zeros((len(nl), dims[b]), np.float32) for b in basis}
    for j in range(len(nl)):
        p = pos.copy(); bo = cfg[j]["box_offset"]
        if bo is not None:
            p[A["obj"] == 1] += np.asarray(bo)                                   # lights act on the placed box
        E = direct_field(p, nrm, LT[j], int(nl[j]), np.asarray(pv[j]), ns)
        if bo is None:
            E[A["obj"] == 1] = 0
        for b, f in basis.items():
            out[b][j] = np.bincount(f, E * area, out[b].shape[1])
    return out


def indirect_r1(A, basis, dims):
    geo = R.Geo(*C.offsets("T0")); w = np.load(C.OUT / "eval_px/heldout_T0.npz")["v6_light"]
    frac, _ = RR.sun_vis(A, geo, w)
    Es = C.E_SUN * np.clip(A["nrm"].astype(np.float64) @ w, 0, None) * frac
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; rad = 0.5 / np.pi * Es; Eind = np.zeros(len(Es))
    for s in range(0, len(H), 100000):
        h = H[s:s + 100000]; Eind[s:s + 100000] = np.where(h >= 0, rad[np.clip(h, 0, None)], 0).sum(1) * np.pi / H.shape[1]
    area = A["area"].astype(np.float64)
    return {b: (np.bincount(f, Eind * area, dims[b]), np.bincount(f, Es * area, dims[b])) for b, f in basis.items()}


def b6_common(cu):
    """Batch 6 training fields and its own audit indirect field, re-binned into the common basis (read-only)."""
    A6 = np.load(C.OUT / "anchors.npz")
    key = common_key(A6["pos"].astype(np.float64), A6["nrm"].astype(np.float64), A6["obj"])
    f = np.searchsorted(cu, key); area = A6["area"].astype(np.float64)
    pe = np.load(C.OUT / "stagea/prep_train/portE.npy", mmap_mode="r")
    X = np.stack([np.bincount(f, np.asarray(pe[j]).astype(np.float64) * area, len(cu)) for j in range(pe.shape[0])])
    w = np.load(C.OUT / "eval_px/heldout_T0.npz")["v6_light"]
    Vp = np.unpackbits(np.load(C.OUT / "eval_vis.npz")["heldout_T0_v6_T0"])[:len(A6["obj"])].astype(bool)
    Es = C.E_SUN * np.clip(A6["nrm"] @ w, 0, None) * Vp; H = np.load(C.OUT / "rel_T0.npz")["hit"]; rad = 0.5 / np.pi * Es
    Eind = np.zeros(len(Es))
    for s in range(0, len(H), 100000):
        h = H[s:s + 100000]; Eind[s:s + 100000] = np.where(h >= 0, rad[np.clip(h, 0, None)], 0).sum(1) * np.pi / C.K_GATHER
    return X, np.bincount(f, Eind * area, len(cu)), np.bincount(f, Es * area, len(cu))


def main():
    A = RR.load()
    key = common_key(A["pos"].astype(np.float64), A["nrm"].astype(np.float64), A["obj"])
    A6 = np.load(C.OUT / "anchors.npz")
    cu = np.unique(np.concatenate([key, common_key(A6["pos"].astype(np.float64), A6["nrm"].astype(np.float64), A6["obj"])]))
    basis = {"r1_coarse": A["f2c"].astype(np.int64), "common": np.searchsorted(cu, key)}
    dims = {"r1_coarse": len(A["cobj"]), "common": len(cu)}
    tr = r1_fields(A, "train", basis, dims); ho = r1_fields(A, "holdout", basis, dims); ind = indirect_r1(A, basis, dims)
    rec = {"r1": {b: svd_stats(tr[b], {"indirect_one_bounce": ind[b][0], "direct_sun_same_view": ind[b][1], "holdout_fields": ho[b]}) for b in basis}}
    X6, i6, d6 = b6_common(cu)
    rec["batch6_common"] = svd_stats(X6, {"indirect_one_bounce": i6, "direct_sun_same_view": d6})
    g = rec["r1"]["r1_coarse"]["indirect_one_bounce_energy_outside_full_span"]
    rec["gate2c_pass"] = bool(g <= 0.10)
    (C.OUT1 / "gate2").mkdir(parents=True, exist_ok=True); (C.OUT1 / "gate2/coverage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
