"""WL45 Track A3 / A1b / A4 - a physics oracle for Yellow's first-event response that uses NO learned parameter.

For held-out T0 view v, Yellow receiver pixels x (subsample) and the verified port sun field of K_k44P's own input
(E(x') = E_SUN max(0, n'.w) frac(x'), frac = eval_vis_k44 port sun fractions, 0.3 mm rule):
    L_c(x) = s_c * sum_{ports x' in S} R_c(|x - x'|) E(x') A(x'),      R_c Burley with length d_c (A1a fit),
    truncated at 16 d_c (Cycles' disk sampling bound); |x - x'| >= 2 mm.
Source sets S (A1b): 'object' all Lego ports (Cycles' Burley is object-scoped), 'yellow' Yellow ports only,
'component' ports of the receiver port's connected component only.
s_c (one scalar per channel and source set) is least-squares fitted on views 0-7 and scored on views 8-15 against
the first_r1 reference; the frozen K_k44P K-only prediction (eval views.npz) is scored on the SAME test pixels.
Primary comparison A (protocol diag45): display PSNR of oracle vs K_k44P on test-view Yellow pixels, lit / shadow,
with a bootstrap 95 % CI over the 8 test views.
A4: support mass of each channel's profile within 5 / 30 cm, and the rank-64 energy of the receiver x coarse-source
Burley operator (3,000 Yellow receiver ports x 8,980 coarse ports, A = coarse area).

    python a3_burley_oracle.py <a1a_fit.json> <out dir> [px_per_view=12000]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])


def burley(r, d):
    return (torch.exp(-r / d) + torch.exp(-r / (3 * d))) / (8 * np.pi * d * r)


@torch.no_grad()
def main(fit_json, out, npx=12000):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    fit = json.loads(Path(fit_json).read_text())["fit"]; dch = [fit[c]["d_fit_m"] for c in "RGB"]
    A = np.load(C.OUT1 / "anchors.npz"); nA = int(A["n_A"])
    pos = torch.as_tensor(A["pos"][:nA], device=dev).float(); nrm = torch.as_tensor(A["nrm"][:nA], device=dev).float()
    area = torch.as_tensor(A["area"][:nA], device=dev).float(); mat = torch.as_tensor(A["mat"][:nA].astype(np.int64), device=dev)
    comp = torch.as_tensor(A["comp"][:nA].astype(np.int64), device=dev)
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); first = np.load(C.OUT1 / "refs/heldout_T0_first.npz")
    V = np.load(C.OUT1 / "eval_K_k44P/views.npz"); vis = np.load(C.OUT1 / "eval_vis_k44.npz")
    rng = np.random.default_rng(4503); data = {k: [] for k in ("view", "ref", "K", "lit", "obj", "yellow", "component", "dist_lit")}
    for v in range(int(px["nv"])):
        val = px[f"v{v}_valid"].astype(bool); dom = px[f"v{v}_dom"][val]; matp = px[f"v{v}_mat"][val]
        yel = np.flatnonzero((dom == 0) & (matp == 0))
        sel = np.sort(rng.choice(yel, min(npx, len(yel)), replace=False))
        w = torch.as_tensor(px[f"v{v}_light"], device=dev).float()
        frac = torch.as_tensor(vis[f"heldout_T0_v{v}_T0"].astype(np.float32)[:nA], device=dev)
        E = C.E_SUN * (nrm @ w).clamp(min=0) * frac
        EA = E * area; lit_ports = pos[E > 0]
        xr = torch.as_tensor(px[f"v{v}_pos"][val][sel], device=dev).float(); prt = torch.as_tensor(px[f"v{v}_port"][val][sel].astype(np.int64), device=dev)
        res = {k: torch.zeros(len(sel), 3, device=dev) for k in ("obj", "yellow", "component")}
        dl = torch.zeros(len(sel), device=dev)
        for c0 in range(0, len(sel), 96):
            xs = xr[c0:c0 + 96]; r = torch.cdist(xs, pos).clamp(min=2e-3)              # [b, nA]
            ys = (mat == 0).float()[None]; cs = (comp[None] == comp[prt[c0:c0 + 96]][:, None]).float()
            for ci, d in enumerate(dch):
                k = burley(r, d) * (r < 16 * d) * EA[None]
                res["obj"][c0:c0 + 96, ci] = k.sum(1); res["yellow"][c0:c0 + 96, ci] = (k * ys).sum(1); res["component"][c0:c0 + 96, ci] = (k * cs).sum(1)
            dl[c0:c0 + 96] = torch.cdist(xs, lit_ports).min(1).values if len(lit_ports) else 1e9
        data["view"].append(np.full(len(sel), v)); data["ref"].append(first[f"v{v}_rgb"][val][sel].astype(np.float64))
        data["K"].append(V[f"heldout_T0_v{v}_K_only"][sel][:, :3].astype(np.float64))        # views.npz rows = valid pixels in order
        data["lit"].append(px[f"v{v}_vis"][val][sel].astype(bool)); data["dist_lit"].append(dl.cpu().numpy())
        for k in ("obj", "yellow", "component"):
            data[k].append(res[k].cpu().numpy().astype(np.float64))
        print("view", v, len(sel), flush=True)
    d = {k: np.concatenate(x) for k, x in data.items()}
    tr, te = d["view"] < 8, d["view"] >= 8; D = C.ldr3
    rec = {"d_m": dch, "px": int(len(d["ref"])), "source_sets": {}}

    def psnr(p, r, m):
        return float(10 * np.log10(1 / max(((D(p[m]) - D(r[m])) ** 2).mean(), 1e-30)))

    def boot(p, r, m, B=400):
        vs = np.unique(d["view"][m]); out_ = []
        bs = np.random.default_rng(1)
        for _ in range(B):
            pick = bs.choice(vs, len(vs)); mm = np.concatenate([np.flatnonzero(m & (d["view"] == q)) for q in pick])
            out_.append(10 * np.log10(1 / max(((D(p[mm]) - D(r[mm])) ** 2).mean(), 1e-30)))
        return np.quantile(out_, [0.025, 0.975]).tolist()
    for s in ("obj", "yellow", "component"):
        X = d[s]; scale = [float((X[tr, c] * d["ref"][tr, c]).sum() / max((X[tr, c] ** 2).sum(), 1e-30)) for c in range(3)]
        P = X * np.array(scale)[None]
        rec["source_sets"][s] = {"scale_per_channel": scale, **{f"psnr_test_{nm}": psnr(P, d["ref"], te & m) for nm, m in
                                 (("all", np.ones(len(P), bool)), ("lit", d["lit"]), ("shadow", ~d["lit"]))},
                                 "ci95_test_all": boot(P, d["ref"], te), "corr_lum_test": float(np.corrcoef(P[te] @ LUM, d["ref"][te] @ LUM)[0, 1])}
    rec["K_k44P"] = {f"psnr_test_{nm}": psnr(d["K"], d["ref"], te & m) for nm, m in (("all", np.ones(len(d["K"]), bool)), ("lit", d["lit"]), ("shadow", ~d["lit"]))}
    rec["K_k44P"]["ci95_test_all"] = boot(d["K"], d["ref"], te); rec["K_k44P"]["corr_lum_test"] = float(np.corrcoef(d["K"][te] @ LUM, d["ref"][te] @ LUM)[0, 1])
    rec["per_channel_test_psnr"] = {nm: [float(10 * np.log10(1 / max(((D(P_[te, c]) - D(d["ref"][te, c])) ** 2).mean(), 1e-30))) for c in range(3)]
                                    for nm, P_ in (("K_k44P", d["K"]), ("oracle_obj", d["obj"] * np.array(rec["source_sets"]["obj"]["scale_per_channel"])[None]))}
    bins = [0, 0.01, 0.03, 0.10, 0.30, 10]; P = d["obj"] * np.array(rec["source_sets"]["obj"]["scale_per_channel"])[None]
    rec["by_distance_to_lit_port_test"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = te & (d["dist_lit"] >= lo) & (d["dist_lit"] < hi)
        if m.sum() > 50:
            rec["by_distance_to_lit_port_test"].append({"m": [lo, hi], "px": int(m.sum()), "K_psnr": psnr(d["K"], d["ref"], m), "oracle_psnr": psnr(P, d["ref"], m),
                                                         "K_over_ref_lum": float((d["K"][m] @ LUM).sum() / max((d["ref"][m] @ LUM).sum(), 1e-30)),
                                                         "oracle_over_ref_lum": float((P[m] @ LUM).sum() / max((d["ref"][m] @ LUM).sum(), 1e-30))})
    frac = lambda R, dd: 1 - (np.exp(-R / dd) + 3 * np.exp(-R / (3 * dd))) / 4  # noqa: E731
    rec["A4_support_mass_within_5cm_30cm"] = {c: [float(frac(0.05, dd)), float(frac(0.30, dd))] for c, dd in zip("RGB", dch)}
    # A4 rank: Yellow receiver ports x coarse sources
    yp = torch.nonzero(mat == 0)[:, 0]; yp = yp[torch.as_tensor(np.random.default_rng(7).choice(len(yp), 3000, replace=False), device=dev)]
    cpos = torch.as_tensor(A["cpos"], device=dev).float(); carea = torch.as_tensor(A["carea"], device=dev).float(); cobj = torch.as_tensor(A["cobj"].astype(np.int64), device=dev)
    cl = cobj == 0; r = torch.cdist(pos[yp], cpos[cl]).clamp(min=2e-3)
    rec["A4_rank64_energy"] = {}
    for c, dd in zip("RGB", dch):
        Mx = (burley(r, dd) * (r < 16 * dd) * carea[cl][None]).double()
        s_ = torch.linalg.svdvals(Mx); e = torch.cumsum(s_ ** 2, 0) / (s_ ** 2).sum()
        rec["A4_rank64_energy"][c] = {"rank64": float(e[63]), "rank8": float(e[7]), "rank256": float(e[255])}
    (out / "a3_oracle.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({k: v for k, v in rec.items() if k != "by_distance_to_lit_port_test"}, indent=1))
    print(json.dumps(rec["by_distance_to_lit_port_test"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12000)
