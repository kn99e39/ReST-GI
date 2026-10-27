"""WL52 focused tests (GPU, Mitsuba). python tests/test_lc52.py <out json>"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lc_data as LD  # noqa: E402
import lc_model as LM  # noqa: E402
import lc_scene as LS  # noqa: E402
import lc_trace as TR  # noqa: E402

R = {}; DEV = torch.device("cuda")


def check(n, ok, **i):
    R[n] = {"pass": bool(ok), **i}; print(n, R[n], flush=True)


def corner_ff(X, Y):
    return (X / math.sqrt(1 + X * X) * math.atan(Y / math.sqrt(1 + X * X)) + Y / math.sqrt(1 + Y * Y) * math.atan(X / math.sqrt(1 + Y * Y))) / (2 * math.pi)


@torch.no_grad()
def main(outp):
    # C reference accuracy: point at the centre of a plate facing a parallel square (analytic form factor)
    B = LS.Builder(); a = B.obj("box", (0.5, 0.5, 0.5)); s, h = 0.4, 0.3
    B.plane(a, np.array([-0.2, -0.2, 0.0]), np.array([0.4, 0, 0]), np.array([0, 0.4, 0]), 33, 33)                       # receiver plate (+z), odd grid -> a centre port
    b = B.obj("box", (0.5, 0.5, 0.5)); B.plane(b, np.array([-0.2, -0.2, h]), np.array([0, 0.4, 0]), np.array([0.4, 0, 0]), 8, 8)   # source plate facing -z
    S = B.build(); G = TR.Geo(S, DEV); c = int(np.argmin(np.linalg.norm(S.pos - np.array([0, 0, 0]), axis=1)))
    H, _ = TR.gather(G, 1 << 16, 1, ports=torch.tensor([c], device=DEV)); L = (G.obj == 1).float()[:, None].expand(-1, 3).contiguous()
    E = float(TR.apply_table(H, L)[0, 0]); Ea = math.pi * 4 * corner_ff(s / 2 / h, s / 2 / h)
    check("C_reference_analytic_formfactor", abs(E / Ea - 1) < 0.01, E=E, analytic=Ea)
    # constant-radiance normalization and conventions
    cfg = LS.sample_config(42, n_arm=1, n_sheet=1); S = LS.build(cfg); G = TR.Geo(S, DEV); Ho, _ = TR.gather(G, 512, 3)
    one = torch.ones(S.N, 3, device=DEV); e = TR.apply_table(Ho, one)[:, 0]
    check("constant_radiance_normalization", float((e - math.pi * (Ho >= 0).float().mean(1)).abs().max()) < 1e-5)
    # reciprocity (object pairs, area-weighted flux under unit radiance)
    P = torch.zeros(len(S.objs), len(S.objs), device=DEV); w = math.pi / Ho.shape[1]
    for o in range(len(S.objs)):
        Lo = (G.obj == o).float()[:, None]; P[:, o] = torch.zeros(len(S.objs), device=DEV).index_add_(0, G.obj, G.area * TR.apply_table(Ho, Lo)[:, 0])
    big = P > 1e-3 * float(P.max()); Q = (P / P.T.clamp(min=1e-12))[big & big.T]
    check("explicit_reciprocity_object_pairs", bool(((Q > 0.9) & (Q < 1.1)).all()), Q_min=float(Q.min()), Q_max=float(Q.max()), pairs=int(Q.numel()))
    # learned operator contract (untrained network: properties hold by construction)
    torch.manual_seed(0); model = LM.GLT().to(DEV).eval(); import lc_eval as LE
    F = LE.learned_F(model, G); Lb = TR.patch_mean(G, torch.rand(S.N, 3, device=DEV)); Lb2 = TR.patch_mean(G, torch.rand(S.N, 3, device=DEV))
    lin = float((F @ (Lb + 2 * Lb2) - F @ Lb - 2 * (F @ Lb2)).abs().max() / (F @ Lb).abs().max())
    check("learned_zero_superposition_positivity_passivity", float((F @ torch.zeros_like(Lb)).abs().max()) == 0.0 and lin < 1e-5 and bool((F >= 0).all()) and float(F.sum(1).max()) <= math.pi * (1 + 1e-5),
          linearity=lin, max_row_sum_over_pi=float(F.sum(1).max() / math.pi))
    F2 = LE.learned_F(model, G); check("deterministic_learned_eval", float((F2 - F).abs().max()) < 1e-6, maxabs=float((F2 - F).abs().max()))
    H1, _ = TR.gather(G, 128, 9); H2, _ = TR.gather(G, 128, 9); check("deterministic_gather_same_seed", bool(torch.equal(H1, H2)))
    ffm = LM.ff0(G.pos, G.nrm, G.pcen, G.pnrm, G.parea); same_face = (G.pobj[None] == G.obj[:, None]) & ((G.pnrm[None] * G.nrm[:, None]).sum(-1) > 0.999)
    check("ff0_zero_on_own_planar_face", float(ffm[same_face & (G.obj[:, None] < 3)].max()) < 1e-6)
    # K / C composition: C = 0 -> L = rho / pi E_sun; explicit C solve = Neumann series
    Es = TR.sun_irr(G, cfg["sun"]); r0 = TR.solve(G, Es, lambda L: torch.zeros_like(L))
    ap = lambda L: TR.apply_table(Ho, L)  # noqa: E731
    r = TR.solve(G, Es, ap, tol=1e-7); Ln = G.alb / math.pi * Es[:, None] * torch.ones(1, 3, device=DEV); acc = Ln.clone()
    for _ in range(200):
        Ln = G.alb / math.pi * ap(Ln); acc = acc + Ln
    check("KC_composition", float(r0["Eind"].abs().max()) == 0.0 and r["status"] == "converged" and float((r["L"] - acc).abs().max() / acc.abs().max()) < 1e-4,
          neumann_rel=float((r["L"] - acc).abs().max() / acc.abs().max()))
    # geometry transformation: a box moved into the line of sight lowers the receivers it occludes
    base = {"seed": 1, "objects": [("box", {"c": [0.5, 0.5, 0.11], "h": [0.1, 0.1, 0.1], "yaw": 0.0, "albedo": (0.5, 0.5, 0.5)})], "sun": [0.3, 0.3, 0.9]}
    moved = json.loads(json.dumps(base)); moved["objects"][0][1]["c"] = [-0.85, 0.0, 0.11]
    Ew = []
    for cf in (base, moved):
        S2 = LS.build(cf); G2 = TR.Geo(S2, DEV); H2_, _ = TR.gather(G2, 1024, 5); Lw = (G2.obj == 1).float()[:, None].expand(-1, 3).contiguous()    # wall_x emits
        Ew.append(TR.apply_table(H2_, Lw)[:, 0][G2.obj == 0].reshape(64, 64))
    near = Ew[1][4:12, 26:38].mean() / Ew[0][4:12, 26:38].mean()                                # floor just in front of the moved box's new place
    check("visibility_transition_explicit", float(near) < 0.8, floor_irradiance_ratio=float(near))
    # train / test separation
    tr = set(LD.split_seeds("train")); others = set(LD.split_seeds("val")) | set(range(300000, 300120)) | set(range(500000, 500030)) | set(range(600000, 600030)) | set(range(400000, 400006))
    hsh = lambda c: hashlib.sha256(json.dumps(c["objects"], sort_keys=True, default=float).encode()).hexdigest()  # noqa: E731
    trh = {hsh(LS.sample_config(s)) for s in list(tr)[:3000]}; teh = {hsh(LS.sample_config(s)) for s in range(300000, 300120)}
    check("train_test_separation", not (tr & others) and not (trh & teh), seed_overlap=len(tr & others), config_hash_overlap=len(trh & teh))
    Path(outp).parent.mkdir(parents=True, exist_ok=True); Path(outp).write_text(json.dumps(R, indent=1))
    print("ALL PASS" if all(v["pass"] for v in R.values()) else "FAILURES", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
