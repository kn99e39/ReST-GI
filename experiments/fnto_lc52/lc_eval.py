"""WL52 Stages E / F - single-flight operator evaluation and coupled GI (protocol fnto_lc52).

Arms (identical configuration, ports, fields, hardware):
  ORA   explicit, 2,048 gathers per port (seed A); ORA2 (seed B) gives the oracle noise
  C0    explicit, 128 gathers per port (production budget)
  C0p   explicit PATCH-level transfer from 1,024 gathers (the learned model's representation; isolates patch quantization)
  C1    learned GLT (frozen checkpoint)
Fields (single flight): smooth (random Fourier, per port), hot (5 emitting patches), first (rho / pi E_sun with shadows),
noise (i.i.d. per port). Coupled: Lambertian K (exact for every configuration), sun from the configuration; REF = independent
path-traced GI at the ports (2 x 512 paths, continuous geometry), images through 3 fixed cameras (port lookup).

    python lc_eval.py <ckpt> <out dir> [smoke]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_model as LM  # noqa: E402
import lc_scene as LS  # noqa: E402
import lc_trace as TR  # noqa: E402

DEV = torch.device("cuda")
SETS = {"test_iid": [LS.sample_config(s) for s in range(300000, 300120)],
        "combo": [LS.sample_config(s, n_box=(1, 3), n_arm=2, n_sheet=2) for s in range(500000, 500030)],
        "scale_obj": [LS.sample_config(s, n_box=(8, 12)) for s in range(600000, 600030)],
        "train_cfg": [LS.sample_config(s) for s in range(100000, 100030)]}                   # memorization control (all ports; single flight only)
COUPLED_IID = 40
CLASSES = {"room": ("floor", "wall_x", "wall_y"), "box": ("box",), "arm": ("arm",), "sheet": ("sheet",)}


def rel(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    return float((a - b).norm() / b.norm().clamp(min=1e-12)) if len(b) else None


@torch.no_grad()
def learned_F(model, G, chunk=2048):
    z = model.encode(G.pcen, G.pnrm, G.parea); F = torch.empty(G.S.N, G.S.M, device=DEV)
    for s in range(0, G.S.N, chunk):
        sl = slice(s, s + chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f, _, _ = model.transfer(z, G.pos[sl], G.nrm[sl], G.patch[sl], G.pcen, G.pnrm, G.parea, G.pobj, G.obj[sl])
        F[sl] = f.float()
    return F


def port_smooth(G, seed):
    g = torch.Generator(device=DEV).manual_seed(seed); w = torch.randn(6, 3, 3, generator=g, device=DEV) * 3.0; ph = torch.rand(6, 3, generator=g, device=DEV) * 6.283
    return torch.nn.functional.softplus(2 * (torch.sin(torch.einsum("ni,fci->nfc", G.pos, w) + ph[None])).sum(1))


def fields(G, cfg, Es, seed):
    g = torch.Generator(device=DEV).manual_seed(seed); hot = torch.zeros(G.S.N, 3, device=DEV)
    hp = torch.randperm(G.S.M, generator=g, device=DEV)[:5]; hot[torch.isin(G.patch, hp)] = 1.0
    return {"smooth": port_smooth(G, seed), "hot": hot, "first": G.alb / math.pi * Es[:, None], "noise": torch.rand(G.S.N, 3, generator=g, device=DEV)}


def arms(model, G, cfg):
    t = {}; sync = torch.cuda.synchronize
    sync(); t0 = time.time(); Ho, To = TR.gather(G, 2048, 1000 + cfg["seed"]); Ho2, _ = TR.gather(G, 2048, 2000 + cfg["seed"]); sync(); t["oracles"] = time.time() - t0
    sync(); t0 = time.time(); H0, _ = TR.gather(G, 128, 3000 + cfg["seed"]); sync(); t["C0_gather"] = time.time() - t0
    Hp, _ = TR.gather(G, 1024, 4000 + cfg["seed"]); F0p = math.pi * TR.patch_target(G, Hp)[:, :-1] / 1024
    sync(); t0 = time.time(); F1 = learned_F(model, G); sync(); t["C1_F"] = time.time() - t0
    ap = {"ORA": lambda L: TR.apply_table(Ho, L), "ORA2": lambda L: TR.apply_table(Ho2, L), "C0": lambda L: TR.apply_table(H0, L),
          "C0p": lambda L: F0p @ TR.patch_mean(G, L), "C1": lambda L: F1 @ TR.patch_mean(G, L)}
    return ap, Ho, To, t


def masks(G, Ho, To):
    near = (To < 0.01).any(1); one = torch.ones(G.S.N, 3, device=DEV); Eu = TR.apply_table(Ho, one)[:, 0]
    prior = LM.ff0_total(G.pos, G.nrm, G.pcen, G.pnrm, G.parea); occ = (Eu / prior.clamp(min=1e-9) < 0.7) & (prior > 0.05)
    out = {"all": torch.ones(G.S.N, dtype=torch.bool, device=DEV), "near_contact_lt1cm": near, "strongly_occluded": occ}
    for c, kinds in CLASSES.items():
        out[c] = torch.as_tensor(np.isin([G.S.objs[o]["kind"] for o in G.S.obj], kinds), device=DEV)
    return out


def single_flight(ap, G, cfg, Es, mk):
    F_ = fields(G, cfg, Es, 9000 + cfg["seed"]); res = {}
    for fn, L in F_.items():
        Eo = ap["ORA"](L); r = {"ORA2": {}}; E = {a: ap[a](L) for a in ("ORA2", "C0", "C0p", "C1")}
        for a, Ea in E.items():
            r[a] = {m: rel(Ea, Eo, mm) for m, mm in mk.items()}
            ae = (Ea - Eo).norm(dim=1) / Eo.norm(dim=1).mean().clamp(min=1e-12); r[a]["abs_err_p99_over_meanE"] = float(torch.quantile(ae, 0.99)); r[a]["abs_err_max_over_meanE"] = float(ae.max())
        res[fn] = r
    one = torch.ones(G.S.N, 3, device=DEV); Eo = ap["ORA"](one)[:, 0]
    res["closure_uniform"] = {a: float((G.area * ap[a](one)[:, 0]).sum() / (G.area * Eo).sum()) for a in ("ORA2", "C0", "C0p", "C1")}
    return res


def ldr(x, ex):
    return (x * ex).clamp(0, 1) ** (1 / 2.2)


def psnr(a, b):
    return float(10 * torch.log10(1 / ((a - b) ** 2).mean().clamp(min=1e-12)))


def coupled(ap, G, cfg, Es, mk, want_rho=True):
    out = {}; sol = {}
    for a in ("ORA", "C0", "C0p", "C1"):
        t0 = time.time(); r = TR.solve(G, Es, ap[a]); torch.cuda.synchronize(); sol[a] = r
        out[a] = {"status": r["status"], "iters": r["iters"], "solve_s": time.time() - t0}
        if want_rho and a in ("C0", "C1", "C0p"):
            out[a]["rho_RGB"] = TR.spectral_radius(G, ap[a])
    p1, p2 = TR.path_traced_Eind(G, cfg["sun"], 512, 11 + cfg["seed"]), TR.path_traced_Eind(G, cfg["sun"], 512, 12 + cfg["seed"]); Eref = (p1 + p2) / 2
    out["REF_split_rel_l2"] = rel(p1, p2)
    Ed = Es[:, None]; Lref = G.alb / math.pi * (Ed + Eref); imgs = {}
    for ci, (eye, tgt) in enumerate(TR.CAMS):
        pid = TR.camera_ports(G, eye, tgt, res=256); bg = pid < 0; pidc = pid.clamp(min=0)
        ref = torch.where(bg[..., None], 0.0, Lref[pidc]); ex = 1.0 / torch.quantile(ref[~bg].mean(-1), 0.99).clamp(min=1e-9)
        refi = torch.where(bg[..., None], 0.0, (G.alb / math.pi * Eref)[pidc])
        for a in ("ORA", "C0", "C0p", "C1"):
            if sol[a]["status"] != "converged":
                continue
            La = G.alb / math.pi * (Ed + sol[a]["Eind"]); im = torch.where(bg[..., None], 0.0, La[pidc]); imi = torch.where(bg[..., None], 0.0, (G.alb / math.pi * sol[a]["Eind"])[pidc])
            out[a].setdefault("psnr_full_vs_REF", []).append(psnr(ldr(im, ex), ldr(ref, ex))); out[a].setdefault("psnr_indirect_vs_REF", []).append(psnr(ldr(imi, ex), ldr(refi, ex)))
            if a == "C1" and "C0" in sol and sol["C0"]["status"] == "converged":
                L0 = G.alb / math.pi * (Ed + sol["C0"]["Eind"]); out[a].setdefault("psnr_full_vs_C0", []).append(psnr(ldr(im, ex), ldr(torch.where(bg[..., None], 0.0, L0[pidc]), ex)))
            if ci == 0:
                imgs[a] = ldr(im, ex).cpu().numpy().astype(np.float16)
        imgs.setdefault("REF", ldr(ref, ex).cpu().numpy().astype(np.float16)) if ci == 0 else None
    for a in ("ORA", "C0", "C0p", "C1"):
        if sol[a]["status"] == "converged":
            out[a]["Eind_rel_l2_vs_REF"] = {m: rel(sol[a]["Eind"], Eref, mm) for m, mm in mk.items()}
            if a != "ORA" and sol["ORA"]["status"] == "converged":
                e = (sol[a]["Eind"] - sol["ORA"]["Eind"]).norm(dim=1) / sol["ORA"]["Eind"].norm(dim=1).clamp(min=1e-3 * float(sol["ORA"]["Eind"].norm(dim=1).mean()))
                out[a]["port_rel_err_p99_vs_ORA"] = {m: float(torch.quantile(e[mm], 0.99)) for m, mm in mk.items() if mm.sum() > 20}
    return out, sol, imgs


def main(ckpt, outd, smoke=False):
    outd = Path(outd); outd.mkdir(parents=True, exist_ok=True); model = LM.GLT().to(DEV); model.load_state_dict(torch.load(ckpt, map_location=DEV)); model.eval()
    rec = {"ckpt": str(ckpt), "sets": {}}; t00 = time.time(); imgs_all = {}
    sets = dict(SETS); sets["traj"] = [f for j, nm in enumerate(LS.TRAJ) for f in LS.trajectory(nm, 400000 + j)]
    if smoke:
        sets = {k: v[:2] for k, v in sets.items()}
    for sn, cfgs in sets.items():
        R = []; prev = None
        for i, cfg in enumerate(cfgs):
            S = LS.build(cfg); G = TR.Geo(S, DEV); Es = TR.sun_irr(G, cfg["sun"]); ap, Ho, To, tt = arms(model, G, cfg); mk = masks(G, Ho, To)
            r = {"seed": cfg["seed"], "traj": cfg.get("traj"), "frame": cfg.get("frame"), "N": S.N, "M": S.M, "n_objects": len(cfg["objects"]), "times": tt}
            r["single_flight"] = single_flight(ap, G, cfg, Es, mk)
            do_c = sn not in ("test_iid", "train_cfg") or (sn == "test_iid" and i < COUPLED_IID)
            if do_c:
                r["coupled"], sol, imgs = coupled(ap, G, cfg, Es, mk)
                if (sn == "traj" and cfg["frame"] in (0, 15, 29)) or (sn != "traj" and i < 2):
                    imgs_all[f"{sn}_{cfg['seed']}_{cfg.get('traj', '')}_{cfg.get('frame', '')}"] = imgs
                if sn == "traj":                                                      # temporal tracking vs the oracle solution
                    cur = {a: sol[a]["Eind"] for a in sol if sol[a]["status"] == "converged"}; cur["Ho"] = Ho
                    if prev is not None and prev["traj"] == cfg["traj"] and "ORA" in cur:
                        dO = cur["ORA"] - prev["sol"]["ORA"]; vt = ((Ho != prev["sol"]["Ho"]).float().mean(1) > 0.1)
                        r["temporal"] = {"visibility_transition_ports": int(vt.sum())}
                        for a in ("C0", "C0p", "C1"):
                            if a in cur and a in prev["sol"]:
                                dA = cur[a] - prev["sol"][a]; cc = float(torch.corrcoef(torch.stack([dA.flatten(), dO.flatten()]))[0, 1]) if dO.abs().sum() > 0 else None
                                r["temporal"][a] = {"delta_corr": cc, "delta_rel_l2": rel(dA, dO), "delta_rel_l2_transition": rel(dA, dO, vt) if vt.sum() > 20 else None}
                    prev = {"traj": cfg["traj"], "sol": cur}
            R.append(r); print(sn, i, json.dumps({"sf_first_C1": r["single_flight"]["first"]["C1"]["all"], "sf_first_C0": r["single_flight"]["first"]["C0"]["all"],
                                                   "coupled": {a: (v.get("status"), round(float(np.mean(v["psnr_full_vs_REF"])), 2) if v.get("psnr_full_vs_REF") else None) for a, v in r.get("coupled", {}).items() if isinstance(v, dict)}}), flush=True)
            del G; torch.cuda.empty_cache()
        rec["sets"][sn] = R; (outd / "eval.json").write_text(json.dumps(rec, indent=1))
    np.savez_compressed(outd / "images.npz", **{f"{k}__{a}": v for k, d in imgs_all.items() for a, v in d.items()})
    rec["wall_s"] = time.time() - t00; rec["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9; (outd / "eval.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "smoke")
