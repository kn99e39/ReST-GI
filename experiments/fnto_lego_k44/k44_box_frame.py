"""WL44 A1 - box-domain light geometry frame check (numpy only; no kernel, no rendering).

The box is a single diffuse material and the intrinsic-all Stage A box pixels receive direct light only, so
col = (albedo / pi) * E. r1_prep stores box pixel positions in the box's CANONICAL frame (world - box offset);
r1_train.predict / port_incident evaluate point / spot irradiance and directions from those canonical positions,
while the light table holds WORLD positions. This script computes, per probe family, the ratio col / (E / pi) with E
from (a) world positions (what the renderer saw) and (b) canonical positions (what training fed the kernel).
A consistent frame gives one albedo per channel for every family; a frame error spreads the ratio for point / spot
families only (the sun is direction-only).

    python k44_box_frame.py [split ...]      -> results/fnto_lego_k44/A1_box_frame.json
"""
import json
import sys

import numpy as np

import k44_common as K


def irradiance(p, n, T, nl, bits):
    E = np.zeros(len(p))
    for k in range(int(nl)):
        L = T[k].astype(np.float64); b = ((bits >> np.uint32(k)) & 1).astype(bool)
        if L[0] == 0:
            w = L[1:4] / np.linalg.norm(L[1:4]); E += b * L[8] * np.clip(n @ w, 0, None)
        else:
            v = L[1:4][None] - p; r2 = (v * v).sum(1); E += b * L[8] * np.clip((n * v).sum(1) / np.sqrt(r2), 0, None) / r2
    return E


def main(splits):
    rec = {}
    for split in splits:
        d = K.C.OUT1 / f"stagea/prep_{split}"
        px = {k: np.load(d / f"px_{k}.npy", mmap_mode="r") for k in ("pos", "nrm", "col", "dom", "vbits")}
        start = np.load(d / "start.npy"); LT = np.load(d / "lights.npy"); nl = np.load(d / "nlights.npy"); fam = np.load(d / "family.npy")
        cfg = sorted(json.loads((K.C.OUT1 / f"stagea/{split}/configs.json").read_text()), key=lambda c: c["i"])
        acc = {}; offs = []
        n_img = len(nl) if split == "holdout" else min(len(nl), 300)
        for j in range(n_img):
            s, e = start[j], start[j + 1]; dom = np.asarray(px["dom"][s:e]); m = dom == 1
            if cfg[j]["box_offset"] is None or m.sum() < 50:
                continue
            bo = np.asarray(cfg[j]["box_offset"], np.float64); offs.append(np.linalg.norm(bo))
            pc = np.asarray(px["pos"][s:e])[m].astype(np.float64); pw = pc + bo
            n = np.asarray(px["nrm"][s:e])[m].astype(np.float64); n /= np.linalg.norm(n, axis=1, keepdims=True)
            bits = np.asarray(px["vbits"][s:e])[m]; col = np.asarray(px["col"][s:e])[m].astype(np.float64)
            Ew, Ec = irradiance(pw, n, LT[j], nl[j], bits), irradiance(pc, n, LT[j], nl[j], bits)
            ok = Ew > 1e-3 * max(Ew.max(), 1e-12)
            if ok.sum() < 20:
                continue
            a = acc.setdefault(str(fam[j]), {"world": [], "canon": [], "img": 0, "E_canon_over_world": []}); a["img"] += 1
            a["world"].append(col[ok] / (Ew[ok, None] / np.pi)); a["canon"].append(col[ok] / (np.maximum(Ec[ok], 1e-12)[:, None] / np.pi))
            a["E_canon_over_world"].append(Ec[ok] / Ew[ok])
        out = {"box_offset_norm_m_q10_q50_q90": np.quantile(offs, [0.1, 0.5, 0.9]).tolist() if offs else None, "per_family": {}}
        for f, a in acc.items():
            w = np.concatenate(a["world"]); c = np.concatenate(a["canon"]); r = np.concatenate(a["E_canon_over_world"])
            out["per_family"][f] = {"images": a["img"], "pixels": int(len(w)),
                                    "ratio_world_median_rgb": np.median(w, 0).tolist(), "ratio_world_q10_q90_R": np.quantile(w[:, 0], [0.1, 0.9]).tolist(),
                                    "ratio_canon_median_rgb": np.median(c, 0).tolist(), "ratio_canon_q10_q90_R": np.quantile(c[:, 0], [0.1, 0.9]).tolist(),
                                    "E_canon_over_E_world_q10_q50_q90": np.quantile(r, [0.1, 0.5, 0.9]).tolist()}
        rec[split] = out
    K.save("A1_box_frame.json", rec)


if __name__ == "__main__":
    main(sys.argv[1:] or ["holdout", "train"])
