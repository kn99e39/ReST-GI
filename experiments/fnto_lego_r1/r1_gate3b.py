"""WL42 Gate 3b - light-table radiometry on the r1 held-out probes (numpy only).

Box pixels (single diffuse material) in the intrinsic-all Stage A receive direct light only (every bounce leaves the
box into free space and is absorbed), so  col = (albedo / pi) x E  with E reconstructed exactly from the light table
and the pixel's own visibility bits (r1_prep): sun S cos, point / spot I cos / r^2 (spot inside its cone by the bit).
Predeclared pass: per-family medians of col_R / (E / pi) (sun, spot, broad, multisun) within +-3 % of each other
(max / min - 1 <= 0.06 around their mean, i.e. every family median within +-3 % of the mean of the medians).

    python r1_gate3b.py
"""
import json

import numpy as np

import r1_common as C


def main():
    d = C.OUT1 / "stagea/prep_holdout"
    px = {k: np.load(d / f"px_{k}.npy", mmap_mode="r") for k in ("pos", "nrm", "col", "dom", "vbits")}
    start = np.load(d / "start.npy"); LT = np.load(d / "lights.npy"); nl = np.load(d / "nlights.npy"); fam = np.load(d / "family.npy")
    cfg = sorted(json.loads((C.OUT1 / "stagea/holdout/configs.json").read_text()), key=lambda c: c["i"])
    ratios = {}
    for j in range(len(nl)):
        s, e = start[j], start[j + 1]; dom = np.asarray(px["dom"][s:e])
        if cfg[j]["box_offset"] is None or not (dom == 1).any():
            continue
        m = dom == 1; p = np.asarray(px["pos"][s:e])[m].astype(np.float64) + np.asarray(cfg[j]["box_offset"])
        n = np.asarray(px["nrm"][s:e])[m].astype(np.float64); n /= np.linalg.norm(n, axis=1, keepdims=True)
        bits = np.asarray(px["vbits"][s:e])[m]; col = np.asarray(px["col"][s:e])[m].astype(np.float64)
        E = np.zeros(len(p))
        for k in range(int(nl[j])):
            L = LT[j, k].astype(np.float64); b = ((bits >> np.uint32(k)) & 1).astype(bool)
            if L[0] == 0:
                w = L[1:4] / np.linalg.norm(L[1:4]); E += b * L[8] * np.clip(n @ w, 0, None)
            else:
                v = L[1:4][None] - p; r2 = (v * v).sum(1); E += b * L[8] * np.clip((n * v).sum(1) / np.sqrt(r2), 0, None) / r2
        ok = E > 1e-3 * max(E.max(), 1e-12)
        if ok.sum() < 50:
            continue
        ratios.setdefault(str(fam[j]), []).append(col[ok, 0] / (E[ok] / np.pi))
    rec = {"per_family": {}}
    for f, r in ratios.items():
        r = np.concatenate(r); rec["per_family"][f] = {"pixels": int(len(r)), "images": len(ratios[f]), "median": float(np.median(r)), "q10_q90": np.quantile(r, [0.1, 0.9]).tolist()}
    meds = np.array([v["median"] for v in rec["per_family"].values()]); mu = meds.mean()
    rec["max_rel_dev_from_mean"] = float(np.abs(meds / mu - 1).max()); rec["pass"] = bool(rec["max_rel_dev_from_mean"] <= 0.03)
    (C.OUT1 / "gate3").mkdir(parents=True, exist_ok=True); (C.OUT1 / "gate3/light_table_radiometry.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
