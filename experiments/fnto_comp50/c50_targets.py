"""WL50 Gate 0 - shared residual-learning target for L0 / L1 (protocol fnto_comp50).

For every K44 probe configuration of the worklog-49 split (fit 480 / val 120 of prep2_train, test 120 prep2_holdout):
Yellow pixels (domain A, port material 0) of the Cycles probe image, residual r = col - K44 K_glob - K44 K_loc (the exact
k44_train prediction terms, exact gamma), aggregated per port (mean, pixel count). Also K44's own band + mid at those
pixels (the term the arms replace) and the domain-A port irradiance K44's band / mid read (renderer-consistent incident
field, k44 batch_inputs), stored as float16 fields.

    python c50_targets.py <out dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c50_common as CC  # noqa: E402
import k44_train as K44T  # noqa: E402
import r1_common as C  # noqa: E402
from r1_eval import load_kernels  # noqa: E402

TR = K44T.TR


@torch.no_grad()
def parts(K, P, tab, px, L, nl):
    """glob, loc, band+mid of the K44 prediction (k44_train.predict, im = 0, domain A)."""
    canon, n, wo, port = px["pos"], px["nrm"], px["cam"], px["port"]; world = canon + px["off"]
    mat = P.mat[port]; h = K.feat(canon, mat); rho, kap, alpha = K.bsdf_params(h); p, _ = K.p_q(h); hx = K.proj(h)
    glob = (p * tab["gamma"][0][None]).sum(1)
    Jm = int(nl.max()); Li = L[0][None, :Jm].expand(len(n), -1, -1)
    bits = ((px["vbits"][:, None] >> torch.arange(Jm, device=n.device)[None]) & 1).float() * (torch.arange(Jm, device=n.device)[None] < nl[0]).float()
    typ = Li[..., 0]; v = Li[..., 1:4] - world[:, None]; r2 = (v * v).sum(-1).clamp(min=1e-8)
    w = torch.where((typ == 0)[..., None], F.normalize(Li[..., 1:4], dim=-1), v / r2.sqrt()[..., None])
    cos = (w * n[:, None]).sum(-1).clamp(min=0); E = torch.where(typ == 0, Li[..., 8] * cos, Li[..., 8] * cos / r2) * bits
    loc = K.loc_response(n, wo, rho, kap, alpha, w, E[..., None].expand(-1, -1, 3))
    nb = P.nbr[port]; use = (nb >= 0).float(); nbc = nb.clamp(min=0)
    W = K.band_w(hx, tab["hs"][nbc], n, P.bnrm[nbc], (P.bpos[nbc] - canon[:, None]) / C.R_LOCAL, wo)
    band = torch.einsum("nkbc,nkb->nc", W * (P.nbrA[port] * use)[..., None, None], tab["hB"][torch.zeros_like(port)[:, None], nbc])
    mc = P.mid[port]; usem = (mc >= 0).float(); mcc = mc.clamp(min=0)
    Wm = K.mid_w(hx, tab["hc"][mcc], n, P.cnrm[mcc], (P.cpos[mcc] - canon[:, None]) / C.R_MID, wo)
    midt = (Wm * (P.midA[port] * usem)[..., None] * tab["Ec"][0][mcc][..., None]).sum(1)
    return glob, loc, band + midt


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t0 = time.time()
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); P = TR.Ports(dev); nA = P.nA
    sp = CC.S49.splits(); jobs = [("fit", "train", sp["fit"]), ("val", "train", sp["val"]), ("test", "holdout", sp["test_probe"])]
    fields = np.lib.format.open_memmap(out / "fields_f16.npy", "w+", np.float16, (sum(len(j[2]) for j in jobs), nA))
    rows = {k: [] for k in ("cfg", "split", "port", "res", "k44_bm", "loc", "glob", "col", "count")}; ci = 0; meta = []
    for tag, split, ids in jobs:
        S = K44T.Split2(split)
        for j in ids:
            d = K44T.to_t(S.pixels(int(j)), dev); L, nl, inc = K44T.batch_inputs(S, [int(j)], P, dev)
            fields[ci] = inc[0][0][0].clamp(max=6e4).half().cpu().numpy()
            m = (d["dom"] == 0) & (P.mat[d["port"]] == 0)
            if m.any():
                tab = TR.source_tables(KA, P, 0, *inc[0], exact=True)
                px = {k: v[m] for k, v in d.items()}; g_, l_, bm_ = [], [], []
                for s in range(0, int(m.sum()), 8192):
                    a, b, c = parts(KA, P, tab, {k: v[s:s + 8192] for k, v in px.items()}, L, nl); g_.append(a); l_.append(b); bm_.append(c)
                glob, loc, bm = torch.cat(g_), torch.cat(l_), torch.cat(bm_); col = px["col"]; res = col - glob - loc
                up, inv = torch.unique(px["port"], return_inverse=True); cnt = torch.bincount(inv).float()
                agg = lambda x: torch.zeros(len(up), 3, device=dev).index_add_(0, inv, x) / cnt[:, None]  # noqa: E731
                for k_, x in (("res", res), ("k44_bm", bm), ("loc", loc), ("glob", glob), ("col", col)):
                    rows[k_].append(agg(x).cpu().numpy())
                rows["port"].append(up.cpu().numpy()); rows["count"].append(cnt.cpu().numpy())
                rows["cfg"].append(np.full(len(up), ci)); rows["split"].append(np.full(len(up), {"fit": 0, "val": 1, "test": 2}[tag]))
            meta.append({"row": ci, "split": tag, "prep": split, "config": int(j), "family": str(S.family[int(j)]), "yellow_px": int(m.sum())}); ci += 1
        print(tag, ci, f"{time.time() - t0:.0f}s", flush=True)
    fields.flush()
    np.savez(out / "targets.npz", **{k: np.concatenate(v) for k, v in rows.items()})
    tg = np.load(out / "targets.npz"); spl = tg["split"]
    summ = {}
    for s, nm in ((0, "fit"), (1, "val"), (2, "test")):
        m = spl == s; r, bm, col = tg["res"][m], tg["k44_bm"][m], tg["col"][m]
        summ[nm] = {"rows": int(m.sum()), "pixels": int(tg["count"][m].sum()),
                    "residual_share_of_col_RGB": (r.sum(0) / col.sum(0)).tolist(),
                    "K44_bandmid_vs_residual_relL2_RGB": [float(np.linalg.norm(bm[:, c] - r[:, c]) / np.linalg.norm(r[:, c])) for c in range(3)],
                    "loc_share_of_col_RGB": (tg["loc"][m].sum(0) / col.sum(0)).tolist(), "glob_share_of_col_RGB": (tg["glob"][m].sum(0) / col.sum(0)).tolist()}
    (out / "targets_meta.json").write_text(json.dumps({"configs": meta, "summary": summ, "wall_s": time.time() - t0, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
