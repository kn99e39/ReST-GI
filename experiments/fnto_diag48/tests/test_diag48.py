"""WL48 focused tests (protocol fnto_diag48 tests_before_evidence); GPU, RNA venv. Writes one JSON entry per test.

    python tests/test_diag48.py <out json>
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fnto_dir46"))
import dir46_core as D  # noqa: E402
import dir46_solve as DS  # noqa: E402
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402

R = {}


def check(name, ok, **info):
    R[name] = {"pass": bool(ok), **info}; print(name, R[name], flush=True)


@torch.no_grad()
def main(outp):
    import r1_solve as SV
    from r1_eval import load_kernels
    from probe_b48 import burley, kern, csr
    from probe_c48 import MSStateX
    dev = torch.device("cuda")
    kp = C.OUT1 / "K_k44P/kernels.pt"; sha = hashlib.sha256(kp.read_bytes()).hexdigest()
    frz = json.loads((C.ROOT / "results/fnto_diag45/baseline_freeze_server.json").read_text())["files"]["K_k44P/kernels.pt"]["sha256"]
    check("baseline_k44_sha256_equals_wl45_freeze", sha == frz, sha=sha[:16])
    KA, KB = load_kernels(str(kp), dev); T = SV.Tables(KA, KB, dev); An = np.load(C.OUT1 / "anchors.npz"); nA = int(An["n_A"])
    rng = np.random.default_rng(0)
    # conventions: training camera directions leave the surface; C-requested directions (-d) leave the hit port's front
    pr = C.OUT1 / "stagea/prep2_train"; rows = np.sort(rng.choice(len(np.load(pr / "px_dom.npy", mmap_mode="r")), 200000, replace=False))
    cam = np.load(pr / "px_cam.npy", mmap_mode="r")[rows].astype(np.float32); nr = np.load(pr / "px_nrm.npy", mmap_mode="r")[rows].astype(np.float32)
    fc = float(((cam * nr).sum(1) > 0).mean())
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; p_ = rng.choice(T.N, 20000); k_ = rng.integers(0, C.K_GATHER, 20000); hq = H[p_, k_]; ok = hq >= 0
    dd = -T.dirs[torch.as_tensor(p_[ok], device=dev), torch.as_tensor(k_[ok], device=dev)].float(); fr = float(((dd * T.nrm[torch.as_tensor(hq[ok], device=dev)]).sum(-1) > 0).float().mean())
    cen = torch.as_tensor(D.bin_centres_local(), device=dev); own = bool((M.hemi_bin(cen) == torch.arange(19, device=dev)).all())
    check("direction_conventions", fc > 0.98 and fr > 0.95 and own, train_cam_front=fc, C_request_front=fr, bin_centres_own_bin=own)
    # units / flux normalisation of the 64-direction cosine quadrature
    n = T.nrm[:1000]; W = D.cosine_dirs(n, 64); c = (W * n[:, None]).sum(-1)
    e1 = float(math.pi * torch.ones_like(c).mean(1).mean()); e2 = float(math.pi * c.mean(1).mean())
    check("units_flux_quadrature", abs(e1 - math.pi) < 1e-6 and abs(e2 - 2 * math.pi / 3) / (2 * math.pi / 3) < 0.01, const=e1, cos_weighted=e2, expect=2 * math.pi / 3)
    # linearity of nonlocal queries in the incident field (sun off), finiteness
    S = SV.State(T, H, np.zeros(T.N, np.float32), np.array([0, 0, 1.0], np.float32)); q = torch.as_tensor(rng.choice(nA, 4000, replace=False), device=dev)
    w = DS.upper(T.nrm[q], torch.nn.functional.normalize(torch.randn(len(q), 3, device=dev), dim=-1))
    outs = []; del S.spec; torch.cuda.empty_cache()
    for seed in (1, 2):
        g = torch.Generator(device=dev).manual_seed(seed)
        S.L = torch.rand(S.L.shape, generator=g, device=dev, dtype=torch.float16) * 0.1
        Eind, h = S.aggregates(); S.outgoing(Eind, h); outs.append(sum(D.nonlocal_out(T, S, q, w)))
    g = torch.Generator(device=dev).manual_seed(1); S.L = torch.rand(S.L.shape, generator=g, device=dev, dtype=torch.float16) * 0.1
    g = torch.Generator(device=dev).manual_seed(2); S.L += torch.rand(S.L.shape, generator=g, device=dev, dtype=torch.float16) * 0.1
    Eind, h = S.aggregates(); S.outgoing(Eind, h); o3 = sum(D.nonlocal_out(T, S, q, w)); del S; torch.cuda.empty_cache()
    lin = float((o3 - outs[0] - outs[1]).norm() / o3.norm())
    check("linearity_nonlocal_query", lin < 5e-3 and bool(torch.isfinite(o3).all()), rel_residual=lin, note="fp16 L storage")
    # support tables: represented areas vs member areas within the declared radii; aggregate membership
    pos = torch.as_tensor(An["pos"][:nA], device=dev).float(); area = torch.as_tensor(An["area"][:nA], device=dev).float()
    xi = torch.as_tensor(rng.choice(np.flatnonzero(An["mat"][:nA] == 0), 200, replace=False), device=dev)
    dist = torch.cdist(pos[xi], pos); a5 = (area[None] * (dist < C.R_LOCAL)).sum(1); a30 = (area[None] * (dist < C.R_MID)).sum(1)
    nbrA = torch.as_tensor(An["nbrA"][xi.cpu().numpy()], device=dev).float().clamp(min=0).sum(1); midA = torch.as_tensor(An["midA"][xi.cpu().numpy()], device=dev).float().clamp(min=0).sum(1)
    rb, rm = float((nbrA / a5).median()), float((midA / a30).median())
    f2b = torch.as_tensor(An["f2b"][:nA].astype(np.int64), device=dev); bpos = torch.as_tensor(An["bpos"], device=dev).float()
    md = float((pos - bpos[f2b]).norm(dim=-1).quantile(0.99))
    check("support_tables_alignment", 0.7 < rb < 1.3 and 0.7 < rm < 1.3 and md < 0.03, nbrA_over_area5_median=rb, midA_over_area30_median=rm, member_to_band_centre_q99_m=md)
    # Burley oracle: self exclusion and kernel normalisation on a plane (2 pi int r R dr ~ 1)
    xp = pos[xi[:5]]; k, r = kern(xp, pos[xi[:5]], 0.0122); kk = k * (1 - torch.eye(5, device=dev))
    rr = torch.linspace(2e-4, 16 * 0.0122, 200000, device=dev, dtype=torch.float64); norm = float((2 * math.pi * rr * burley(rr, 0.0122) * (rr < 16 * 0.0122)).sum() * (rr[1] - rr[0]))
    check("burley_oracle_alignment", bool((kk.diagonal() == 0).all()) and 0.95 < norm < 1.01, plane_normalisation=norm)
    # exact query == basis table value at bin centres (query alignment of C2 vs C1)
    G = dict(np.load(C.ROOT / "results/fnto_diag46/ms/geom/geom_T0.npz"))
    SX = MSStateX(T, G, plane_mode="dir", nl="n", loc_in="E", plane_read="exact"); SX.prepare(True)
    qq = torch.as_tensor(rng.choice(nA, 2000, replace=False), device=dev); Wb = D.to_world(T.nrm[qq], D.bin_centres_local())
    j = torch.as_tensor(rng.integers(0, 19, len(qq)), device=dev); wb = Wb[torch.arange(len(qq), device=dev), j]
    ex = SX.k_sender(qq, -wb) - SX.loc_iso[qq] - SX.sun_lobe(qq, wb)
    ref = sum(D.nonlocal_out(T, SX, qq, wb)); qa = float((ex - ref).abs().max() / ref.abs().max().clamp(min=1e-30))
    check("exact_query_alignment", qa < 1e-4, max_rel=qa)
    # paired T0 / T1 pixels: same image pixel and camera ray in both states
    G1 = np.load(C.ROOT / "results/fnto_diag46/ms/geom/geom_T1.npz"); okp = True; npair = []
    for v in range(3):
        a0 = {int(k): i for i, k in enumerate(G[f"v{v}_mk"])}; m1 = G1[f"v{v}_mk"]
        both = [(a0[int(k)], jj) for jj, k in enumerate(m1) if int(k) in a0 and G[f"v{v}_dom"][a0[int(k)]] == 2 and G1[f"v{v}_dom"][jj] == 2]
        i0, i1 = map(np.array, zip(*both)); npair.append(len(i0))
        okp &= bool(np.abs(G[f"v{v}_cam"][i0] - G1[f"v{v}_cam"][i1]).max() < 1e-5)
    check("paired_T0_T1_validity", okp and min(npair) > 1000, pairs=npair)
    R["ALL_PASS"] = all(v["pass"] for v in R.values() if isinstance(v, dict))
    Path(outp).parent.mkdir(parents=True, exist_ok=True); Path(outp).write_text(json.dumps(R, indent=1)); print("ALL_PASS", R["ALL_PASS"])


if __name__ == "__main__":
    main(sys.argv[1])
