"""WL104I Stage C: local Jacobian identifiability of the original (64) and ER-v1-GF (58) parameterizations at identical responses.

Usage: python ident104i.py <out_dir> [--states fit_states.npz] [--smoke]
  default: R0 / A / C of the 9 materials (original raw z; candidate = from_original(z)).
  --states: fitted A1 (original raw) and B1 (candidate raw) terminal states from fit104i (post-fit diagnostic).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import common104i as CM

GF, K, CL, IO = CM.GF, CM.K, CM.CL, CM.IO
DEV = CM.DEV
EXACT, WEAK6, WEAK3 = 1e-15, 1e-6, 1e-3


def weights(ref, m):
    sl = slice(m * 20, (m + 1) * 20)
    w = ref["use"][sl] / ref["q"][sl]
    A = (w[..., None] * 0.5 * (ref["a"][sl] + ref["b"][sl])).sum((1, 2))
    RW = torch.as_tensor(np.repeat((np.sqrt(w) / A[:, None]).reshape(-1), 3), device=DEV)
    x = torch.as_tensor(ref["wi"][sl].reshape(-1, 3), device=DEV, dtype=torch.float64)
    y = torch.as_tensor(ref["wo"][sl].reshape(-1, 3), device=DEV, dtype=torch.float64)
    return x, y, RW


def spectrum(J):
    s = torch.linalg.svdvals(J)
    rel = (s / s.max()).cpu().numpy()
    return rel, {"exact_nulls": int((rel < EXACT).sum()), "weak_1e-6": int(((rel >= EXACT) & (rel < WEAK6)).sum()),
                 "weak_1e-3": int(((rel >= EXACT) & (rel < WEAK3)).sum()), "min_rel": float(rel.min()),
                 "min_rel_above_exact": float(rel[rel >= EXACT].min()) if (rel >= EXACT).any() else None,
                 "rank_rcond_1e-12": int((rel > 1e-12).sum())}


def colspace(J, rcond=1e-12):
    U, s, _ = torch.linalg.svd(J, full_matrices=False)
    return U[:, s > rcond * s.max()]


def analyse(x, y, RW, z=None, yv=None):
    rec = {}
    if z is not None:
        zt = torch.as_tensor(z, device=DEV, dtype=torch.float64)
        Jo = torch.func.jacfwd(CM.f_original(x, y))(zt) * RW[:, None]
        Fo = CM.f_original(x, y)(zt) * RW
        rel_o, rec["original"] = spectrum(Jo)
        gv = torch.as_tensor(GF.original_gauge_vectors(z), device=DEV, dtype=torch.float64)
        g = (Jo @ gv.T).norm(dim=0) / (Jo.norm() * gv.norm(dim=1) / np.sqrt(64))
        rec["original"]["gauge_vector_response_relative"] = g.cpu().numpy().tolist()
        rec["original"]["singular_relative"] = rel_o.tolist()
        rec["original"]["column_sensitivity_median"] = float((Jo.norm(dim=0) / Fo.norm()).median())
        if yv is None:
            yv = GF.from_original(z[None])[0]
    yt = torch.as_tensor(yv, device=DEV, dtype=torch.float64)
    Jg = torch.func.jacfwd(CM.f_candidate(x, y))(yt) * RW[:, None]
    Fg = CM.f_candidate(x, y)(yt) * RW
    rel_g, rec["candidate"] = spectrum(Jg)
    rec["candidate"]["singular_relative"] = rel_g.tolist()
    groups = GF.raw_groups()
    Bm = torch.as_tensor(np.concatenate(list(groups.values()), 1), device=DEV)
    JB = Jg @ Bm
    cols, start = {}, 0
    for gname, M in groups.items():
        cols[gname] = list(range(start, start + M.shape[1]))
        start += M.shape[1]
    Q = {g: colspace(JB[:, c]) for g, c in cols.items()}
    names = list(cols)
    pc = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if Q[a].shape[1] and Q[b].shape[1]:
                pc[f"{a}|{b}"] = float(torch.linalg.svdvals(Q[a].T @ Q[b]).max())
    rec["candidate"]["max_principal_cosine"] = pc
    rec["candidate"]["group_sensitivity_median"] = {g: float((JB[:, c].norm(dim=0) / Fg.norm()).median()) for g, c in cols.items()}
    rec["candidate"]["column_sensitivity_median"] = float((Jg.norm(dim=0) / Fg.norm()).median())
    if z is not None:
        Uo, Ug = colspace(Jo), colspace(Jg)
        c = torch.linalg.svdvals(Uo.T @ Ug)
        rec["tangent"] = {"rank_original": int(Uo.shape[1]), "rank_candidate": int(Ug.shape[1]),
                          "min_principal_cosine": float(c.min()), "response_gap": float((Fo - Fg).abs().max() / Fo.abs().max())}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--states", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, _, seen = CM.inputs()
    ref = IO.reference()
    res = {}
    mats = [0] if args.smoke else range(len(rows))
    if args.states is None:
        for arm in ("R0", "A", "C"):
            for m in mats:
                x, y, RW = weights(ref, m)
                res[f"{arm}/{rows[m]['tag']}"] = analyse(x, y, RW, z=raw[arm][m])
                print("ident", arm, rows[m]["tag"], res[f"{arm}/{rows[m]['tag']}"]["original"]["exact_nulls"],
                      res[f"{arm}/{rows[m]['tag']}"]["candidate"]["exact_nulls"], round(time.monotonic() - t0, 1), flush=True)
    else:
        with np.load(args.states) as zf:
            st = {k: zf[k] for k in zf.files}
        keys = [str(k) for k in st["start_keys"]]          # WL104I execution fix: per-start states only
        for key in [f"{a}_{k}" for a in ("A1", "B1") for k in keys]:
            for m in mats:
                x, y, RW = weights(ref, m)
                if key.startswith("A1_"):
                    r = analyse(x, y, RW, z=st[key][m].astype(np.float64))
                else:
                    r = analyse(x, y, RW, yv=st[key][m].astype(np.float64))
                res[f"{key}/{rows[m]['tag']}"] = r
        print("ident fitted states", len(res), flush=True)
    IO.write(args.out / "ident.json", {"results": res, "thresholds": {"exact": EXACT, "weak6": WEAK6, "weak3": WEAK3},
                                       "commit": IO.commit(), "protocol_sha256": IO.sha(CM.HERE / "protocol.json"),
                                       "inputs_sha256": seen, "seconds": time.monotonic() - t0})


if __name__ == "__main__":
    main()
