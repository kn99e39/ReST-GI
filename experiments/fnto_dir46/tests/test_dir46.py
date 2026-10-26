"""WL46 focused tests of the directional interface on the frozen K_k44P machinery (GPU; run as a script, JSON report).

    python tests/test_dir46.py <out json> [view=6]

T1 frames      bin centres fall in their own hemi_bin; upper() keeps unit length and n.w > 0; to_world is orthonormal
T2 units       cosine quadrature of a constant radiance gives E = pi; K_loc 'dir' on a constant field = rho c + gloss albedo
T3 zero        zero sun and zero field -> exactly zero rays in every mode
T4 constant    constant incident radiance c on all gathers of sampled ports: local 'dir' diffuse equals local 'E' diffuse
T5 reflection  a single incident sample on a sharp analytic GGX (alpha 0.01): output peak within 2 deg of the mirror direction
T6 linearity   one step on L1, L2 and L1 + L2 (sun off), every mode: residual relative to the norm <= 2e-3 (fp16 storage)
T7 finite      all outputs finite after one production-state step in every mode
T8 frozen      parameter digest of K_k44P identical before and after all tests
T9 identity    DirState(nl='n', loc_in='E') step == r1_solve.State step (same L) within fp16 rounding; converged view S_T0
               pixels == frozen eval prediction within 0.01 dB
T10 sun once   first bounce (L = 0): loc_in 'dir' (both loc_out) gives the production ray values (sun diffuse + lobe once)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dir46_core as D  # noqa: E402
import dir46_solve as DS  # noqa: E402
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402

MODES = {"R0": dict(nl="n", loc_in="E"), "R3": dict(nl="basis", loc_in="E"), "R4": dict(nl="n", loc_in="dir", loc_out="n"),
         "R5": dict(nl="basis", loc_in="dir", loc_out="ray")}


def rel(a, b):
    return float((a - b).float().norm() / max(float(b.float().norm()), 1e-30))


@torch.no_grad()
def main(out, v=6):
    import r1_solve as SV
    from r1_eval import load_kernels
    dev = torch.device("cuda"); rec = {}
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); dig0 = M.digest(KA, KB); T = SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    Vf, w = vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"]
    # T1
    cen = torch.as_tensor(D.bin_centres_local(), device=dev); ids = M.hemi_bin(cen)
    n = torch.nn.functional.normalize(torch.randn(1000, 3, device=dev), dim=-1); W = D.to_world(n, D.bin_centres_local())
    t, b = M.det_frame(n); R = torch.stack([t, b, n], 1)
    wl = torch.nn.functional.normalize(torch.randn(1000, 3, device=dev), dim=-1); up = DS.upper(n, wl)
    rec["T1_frames"] = {"bins_self": bool((ids == torch.arange(19, device=dev)).all()), "frame_orthonormal_err": float((R @ R.transpose(1, 2) - torch.eye(3, device=dev)).abs().max()),
                        "upper_unit_err": float((up.norm(dim=-1) - 1).abs().max()), "upper_min_ndotw": float((n * up).sum(-1).min()),
                        "to_world_ndot_centre_err": float(((n[:, None] * W).sum(-1) - cen[None, :, 2]).abs().max())}
    # T2
    cd = D.cosine_dirs(n[:10], 128); E = math.pi / 128 * torch.ones(10, 128).sum(1)
    rho = torch.full((10, 3), 0.5, device=dev); kap = torch.zeros(10, 2, 3, device=dev); al = torch.full((10, 2), 0.3, device=dev)
    diff = M.DomainKernel.loc_response(n[:10], n[:10], rho, kap, al, cd, torch.full((10, 128, 3), math.pi / 128, device=dev))
    rec["T2_units"] = {"E_const": float(E.mean()), "diffuse_rho_c": float(diff.mean()), "cos_dirs_upper": bool(((n[:10, None] * cd).sum(-1) > 0).all())}
    # T3
    zr = {}
    for k, m in MODES.items():
        S = DS.DirState(T, H, np.zeros(T.N, np.float32), w, **m); S.step(); zr[k] = float(S.L.float().abs().max()); del S
    rec["T3_zero_max"] = zr
    # T5
    nn5 = torch.tensor([[0.0, 0, 1]], device=dev); wi = torch.nn.functional.normalize(torch.tensor([[[0.3, 0.1, 0.9]]], device=dev), dim=-1)
    th = torch.linspace(0, math.pi / 2 * 0.99, 300, device=dev); ph = torch.linspace(0, 2 * math.pi, 600, device=dev)
    TH, PH = torch.meshgrid(th, ph, indexing="ij"); wo = torch.stack([torch.sin(TH) * torch.cos(PH), torch.sin(TH) * torch.sin(PH), torch.cos(TH)], -1).reshape(-1, 3)
    g = M.DomainKernel.loc_response(nn5.expand(len(wo), 3), wo, torch.zeros(len(wo), 3, device=dev), torch.ones(len(wo), 2, 3, device=dev) * torch.tensor([1.0, 0], device=dev)[None, :, None],
                                    torch.full((len(wo), 2), 0.01, device=dev), wi.expand(len(wo), 1, 3), torch.ones(len(wo), 1, 3, device=dev))[:, 0]
    mir = torch.tensor([-wi[0, 0, 0], -wi[0, 0, 1], wi[0, 0, 2]], device=dev)
    rec["T5_peak_err_deg"] = float(torch.rad2deg(torch.arccos((wo[g.argmax()] * mir).sum().clamp(-1, 1))))
    # production state after one step, for T4 / T6 / T7 / T9 / T10 (one State on the GPU at a time; copies on the CPU)
    S0 = SV.State(T, H, Vf, w); S0.step(); Lp = S0.L.cpu(); del S0; torch.cuda.empty_cache()
    t10 = {}
    for k in ("R4", "R5"):
        S = DS.DirState(T, H, Vf, w, nl="n", loc_in="dir", loc_out=MODES[k]["loc_out"]); S.step(); t10[k] = rel(S.L.cpu(), Lp); del S; torch.cuda.empty_cache()
    rec["T10_sun_once_rel"] = t10
    S0 = SV.State(T, H, Vf, w); S0.L = Lp.to(dev); S0.step(); Lp2 = S0.L.cpu(); del S0; torch.cuda.empty_cache()
    Sd = DS.DirState(T, H, Vf, w, nl="n", loc_in="E"); Sd.L = Lp.to(dev); Sd.step(); Ld = Sd.L.cpu(); del Sd; torch.cuda.empty_cache()
    rec["T9_step_identity"] = {"rel": rel(Ld, Lp2), "max_abs": float((Ld.float() - Lp2.float()).abs().max()), "max_val": float(Lp2.float().abs().max())}
    del Ld, Lp2
    # T4: constant field c = 1 on sampled ports' gathers
    q = torch.as_tensor(np.random.default_rng(0).choice(T.N, 2000, replace=False), device=dev)
    St = DS.DirState(T, H, np.zeros(T.N, np.float32), w, nl="n", loc_in="dir"); St.L[q] = 1.0
    Ea, ha = St.aggregates(); St.outgoing(Ea, ha)
    nq = T.nrm[q]; le = D.local_out(T, St, q, nq, "E"); ld = D.local_out(T, St, q, nq, "dir"); lt = torch.as_tensor(D.LUM, device=dev).float()
    rec["T4_constant"] = {"local_dir_over_local_E_lum": float((ld.sum(0) @ lt) / (le.sum(0) @ lt))}
    del St, Ea, ha; torch.cuda.empty_cache()
    # T6 / T7: linearity and finiteness with sun off
    g6 = torch.Generator(); g6.manual_seed(1)
    L1 = (torch.rand(Lp.shape, generator=g6) * Lp.float().abs().max()).half(); L2 = Lp
    lin, fin = {}, {}
    for k, m in MODES.items():
        outs = []
        for L in (L1, L2, (L1.float() + L2.float()).half()):
            S = DS.DirState(T, H, Vf, w, **m); S.L = L.to(dev); S.step(sun_on=False); outs.append(S.L.cpu().float()); fin[k] = bool(torch.isfinite(S.L).all()); del S
            torch.cuda.empty_cache()
        lin[k] = rel(outs[2], outs[0] + outs[1]); del outs
    rec["T6_linearity_rel"] = lin; rec["T7_finite"] = fin
    # T9b: converged view equals the frozen eval
    valid = px[f"v{v}_valid"].astype(bool)
    pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][valid].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
    S0 = SV.State(T, H, Vf, w).solve(); pa = SV.render_pixels(T, S0, pxd)[0].cpu().numpy(); it0 = S0.iters; del S0; torch.cuda.empty_cache()
    Sd = DS.DirState(T, H, Vf, w, nl="n", loc_in="E").solve(); pb = SV.render_pixels(T, Sd, pxd)[0].cpu().numpy(); it1 = Sd.iters; del Sd; torch.cuda.empty_cache()
    ev = np.load(C.OUT1 / "eval_K_k44P/views.npz")[f"heldout_T0_v{v}_S_T0"][:, :3].astype(np.float32); full = np.load(C.OUT1 / "refs/heldout_T0_full.npz")[f"v{v}_rgb"][valid]
    ps = lambda p: float(10 * np.log10(1 / ((C.ldr3(p) - C.ldr3(full)) ** 2).mean()))  # noqa: E731
    rec["T9_converged"] = {"psnr_frozen_eval": ps(ev), "psnr_r1_state_now": ps(pa), "psnr_dirstate_R0": ps(pb), "iters": [it0, it1], "max_abs_pixel_diff": float(np.abs(pa - pb).max())}
    rec["T8_frozen_digest_identical"] = M.digest(KA, KB) == dig0
    ok = (rec["T1_frames"]["bins_self"] and rec["T1_frames"]["frame_orthonormal_err"] < 1e-5 and rec["T1_frames"]["upper_min_ndotw"] > 0
          and abs(rec["T2_units"]["E_const"] - math.pi) < 1e-5 and abs(rec["T2_units"]["diffuse_rho_c"] - 0.5) < 1e-3 and max(zr.values()) == 0
          and rec["T5_peak_err_deg"] < 2 and max(lin.values()) < 2e-3 and all(fin.values()) and rec["T8_frozen_digest_identical"]
          and rec["T9_step_identity"]["rel"] < 2e-3 and max(t10.values()) < 2e-3 and abs(rec["T9_converged"]["psnr_frozen_eval"] - rec["T9_converged"]["psnr_dirstate_R0"]) < 0.01)
    rec["ALL_PASS"] = bool(ok)
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
