"""WL50 focused tests (GPU). python tests/test_c50.py <out json>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import c50_common as CC  # noqa: E402
import r1_common as C  # noqa: E402

R = {}


def check(n, ok, **i):
    R[n] = {"pass": bool(ok), **i}; print(n, R[n], flush=True)


@torch.no_grad()
def main(outp):
    from r1_eval import load_kernels
    dev = torch.device("cuda")
    sha = hashlib.sha256((C.OUT1 / "K_k44P/kernels.pt").read_bytes()).hexdigest(); check("frozen_k44_identity", sha.startswith("30d55da373a01ab8"), sha=sha[:16])
    wl49 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((C.ROOT / "results/fnto_spatial49/gate3_learned_ab").glob("ckpt_*.pt"))}
    check("wl49_checkpoints_present", len(wl49) == 4, hashes=wl49)
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = CC.SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); v = 6
    Vf = vis[f"heldout_T0_v{v}_T0"].astype(np.float32); light = px[f"v{v}_light"]
    val = px[f"v{v}_valid"].astype(bool)
    pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][val].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
    # candidate-off equivalence: K_only, one step, short solve, render
    CC.deactivate(); S0 = CC.SV.State(T, H, Vf, light); E0, h0 = S0.aggregates(); I0 = S0.outgoing(E0, h0); r0 = CC.SV.render_pixels(T, S0, pxd)[0]
    CC.activate(None); S1 = CC.SV.State(T, H, Vf, light); E1, h1 = S1.aggregates(); I1 = S1.outgoing(E1, h1); r1 = CC.SV.render_pixels(T, S1, pxd)[0]
    d_off = [float((I1 - I0).abs().max()), float((r1 - r0).abs().max())]
    CC.deactivate(); Sb = CC.SV.State(T, H, Vf, light); Eb, hb = Sb.aggregates(); Ib = Sb.outgoing(Eb, hb)          # production vs production (CUDA index_add_ is non-deterministic)
    d_prod = float((Ib - I0).abs().max())
    S0.solve(max_it=4); CC.activate(None); S1.solve(max_it=4)
    hd = float(np.max(np.abs(np.array(S0.hist) - np.array(S1.hist)) / np.array(S0.hist)))
    tol = max(1e-5 * float(I0.abs().max()), 4 * d_prod)
    check("candidate_off_equivalence", d_off[0] <= tol and d_off[1] <= 1e-5 * float(r0.abs().max()) and hd < 1e-4,
          outgoing_render_maxabs=d_off, production_vs_production_maxabs=d_prod, tol=tol, resid_hist_max_rel_diff=hd)
    # arms (untrained network): ownership, linearity, zero field, finiteness, render replacement only on Yellow pixels
    geo = CC.Geo(dev); yp = CC.yellow_ports(); net = CC.G3.Net().to(dev); info = {}
    for kind in ("L0", "L1"):
        sub = yp[::40]; arm = CC.ArmOp(kind, net, geo, sub)
        nA = geo.L.nA; g = torch.Generator(device=dev).manual_seed(1)
        Ea, Eb = torch.rand(nA, 3, generator=g, device=dev), torch.rand(nA, 3, generator=g, device=dev)
        na, nb, nab, nz = arm.apply(Ea), arm.apply(Eb), arm.apply(Ea + Eb), arm.apply(torch.zeros(nA, 3, device=dev))
        lin = float((nab - na - nb).abs().max() / nab.abs().max())
        CC.activate(arm); S2 = CC.SV.State(T, H, Vf, light); E2, h2 = S2.aggregates(); I2 = S2.outgoing(E2, h2); r2 = CC.SV.render_pixels(T, S2, pxd)[0]
        notyel = torch.ones(T.N, dtype=torch.bool, device=dev); notyel[arm.ports] = False
        ypix = (pxd["dom"] == 0) & torch.isin(pxd["port"], arm.ports)
        own = float((I2[notyel] - I0[notyel]).abs().max()); pixo = float((r2[~ypix] - r0[~ypix]).abs().max()); tl = 1e-5 * float(I0.abs().max())
        keep_ok = bool((S2.keep[arm.ports] == 0).all() and (S2.keep[notyel] == 1).all())
        ok = lin < 1e-4 and float(nz.abs().max()) == 0.0 and bool(torch.isfinite(nab).all()) and own <= max(tl, 4 * d_prod) and pixo <= 1e-5 * float(r0.abs().max()) and keep_ok
        info[kind] = {"linearity": lin, "nonyellow_port_maxabs": own, "nonyellow_pixel_maxabs": pixo, "keep_ok": keep_ok, "K": arm.K, "yellow_px": int(ypix.sum())}
        check(f"arm_{kind}_ownership_linearity", ok, **info[kind])
    CC.deactivate()
    # residual target decomposition: glob + loc + band/mid == k44_train.predict on one probe configuration
    import c50_targets as TG
    K44T = TG.K44T; TR = TG.TR; P = TR.Ports(dev); S_ = K44T.Split2("train"); j = int(CC.S49.splits()["fit"][0])
    d = K44T.to_t(S_.pixels(j), dev); L, nl, inc = K44T.batch_inputs(S_, [j], P, dev); m = (d["dom"] == 0) & (P.mat[d["port"]] == 0)
    tab = TR.source_tables(KA, P, 0, *inc[0], exact=True); px_ = {k: vv[m][:4096] for k, vv in d.items()}; px_["im"] = torch.zeros_like(px_["port"])
    g_, l_, bm_ = TG.parts(KA, P, tab, px_, L, nl); full = TR.predict(KA, P, tab, 0, px_, L, nl)
    dec = float((g_ + l_ + bm_ - full).abs().max() / full.abs().max())
    check("residual_target_decomposition", dec < 1e-4, rel_maxabs=dec)
    R["ALL_PASS"] = all(x["pass"] for x in R.values() if isinstance(x, dict))
    Path(outp).parent.mkdir(parents=True, exist_ok=True); Path(outp).write_text(json.dumps(R, indent=1)); print("ALL_PASS", R["ALL_PASS"])


if __name__ == "__main__":
    main(sys.argv[1])
