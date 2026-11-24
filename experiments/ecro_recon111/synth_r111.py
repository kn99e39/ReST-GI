"""WL111 Stage E: synthetic and runtime contracts on the real renderer with the trained candidate (both realizations),
before any final evaluation. Any failure stops the chain (protocol stop rules).

python synth_r111.py ROOT [--smoke]   -> ROOT/synth/synth.json (ALL_PASS True / False)
Unit-level contracts (constant fields, composition, HDR, reset, causality, reprojection, mixed / id / view gates, fused vs
reference) also run in tests_r111.py; here they are repeated with the real guides, real frames and trained weights.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

import common_r111 as C
import engine_r111 as E
import recon_r111 as RC
import arms_r111 as AR
import seq_r111 as SQ

torch, R = E.torch, E.R
W = C.RES
N = W * W
RES = {}


def check(name, cond, **info):
    RES[name] = {"pass": bool(cond), **info}
    print(("PASS " if cond else "FAIL ") + name, {k: v for k, v in info.items() if not isinstance(v, list)}, flush=True)


def rel_l1(a, b):
    return float((a - b).abs().sum() / b.abs().sum().clamp(min=1e-30))


def main(root, smoke):
    out = root / "synth"
    out.mkdir(parents=True, exist_ok=True)
    rt, Bd, B, audit = E.admit(print, replay=False)
    eng = E.Engine(rt, Bd, B)
    rows = SQ.rows()
    eng.state = None
    eng.set_state(rows[0]["geom"], rows[0]["lights"][0])
    dsg = C.read(root / "design.json")
    sub = dsg["sub"]
    sd = lambda test, i: C.seed("synthetic", 0, test, i, smoke=smoke)
    v0 = E.cam_dict(C.P["views"]["V0"])
    camb = dict(v0, origin=[v0["origin"][0] + 0.08] + v0["origin"][1:])
    basis = R.PT.cam_basis
    # E10 material table: every non-sphere material of the SSS case is Lambert (the view gate covers shape 6 only)
    st = np.asarray(rt.mats.states, np.float32)
    lam = [bool(np.allclose(s[24:38], 0.3) and np.allclose(s[38:45], 0.5) and s[45] == 1 and s[46] == 0) for s in st]
    check("E10_non_sphere_materials_lambert", all(l for i, l in enumerate(lam) if i != 2), lambert=lam)
    for ri in (0, 1):
        cand = AR.load_candidate(root / "models" / f"cand_r{ri}.pt", dsg["m"])
        tag = f"_r{ri}"
        g = eng.guides(v0, sub)
        # E1 constant radiance with the trained net on real guides (fused and reference)
        for fused in (True, False):
            cand.fused = fused
            for ess in (1.0, 64.0):
                lo, hi = cand.spatial(torch.full((3, W, W), 0.25, device="cuda"), torch.full((3, W, W), 1.7, device="cuda"),
                                      torch.full((W, W), ess, device="cuda"), g)
                e = max(float((lo - 0.25).abs().max()) / 0.25, float((hi - 1.7).abs().max()) / 1.7)
                check(f"E1_constant{tag}_{'fused' if fused else 'ref'}_ess{int(ess)}", e <= 1e-5, max_rel=e)
        cand.fused = True
        # E2 / E3 composition and HDR range on a real frame plus a 1e4 spike
        comp = eng.frame(v0, 4, sd(2, ri))
        low, high = RC.raw_low_high(comp)
        high = high.clone()
        high[:, 256, 256] = 1e4
        ess = torch.full((W, W), 4.0, device="cuda")
        lo, hi = cand.spatial(low, high, ess, g)
        tot = lo + hi
        check(f"E2_total_composition{tag}", torch.equal(tot, lo + hi))
        mn = min(float(low.min()), float(high.min()))
        check(f"E3_hdr{tag}", bool(torch.isfinite(tot).all()) and float(hi.max()) <= 1e4 * (1 + 1e-6)
              and float(lo.min()) >= mn - 1e-6 and float(hi.min()) >= mn - 1e-6, max_high=float(hi.max()))
        # E13 fused == reference passes with trained logits on a real frame
        cand.fused = False
        lr, hr = cand.spatial(low, high, ess, g)
        cand.fused = True
        e = max(rel_l1(lo, lr), rel_l1(hi, hr))
        check(f"E13_fused_vs_reference_real{tag}", e <= 1e-5, rel_l1=e)
        # E4 reset == S, E5 causality on a real static stream
        S = AR.CandidateArm(cand, basis, False, "S")
        ST = AR.CandidateArm(cand, basis, True, "ST")
        cache = {}
        frames = [eng.frame(v0, 2, sd(4, k)) for k in range(6)]
        outs = []
        for k, c in enumerate(frames):
            outs.append(ST(AR.Frame(eng, c, v0, 2, eng.revision, v0, False, sub), cache).clone())
        ST.reset()
        y_st = ST(AR.Frame(eng, frames[0], v0, 2, eng.revision, v0, False, sub), cache)
        y_s = S(AR.Frame(eng, frames[0], v0, 2, eng.revision, v0, False, sub), cache)
        check(f"E4_reset_equals_S{tag}", rel_l1(y_st, y_s) <= 1e-6, rel_l1=rel_l1(y_st, y_s))
        ST2 = AR.CandidateArm(cand, basis, True, "ST")
        outs2 = []
        for k in range(6):
            c = frames[k] if k < 4 else eng.frame(v0, 2, sd(5, k))
            outs2.append(ST2(AR.Frame(eng, c, v0, 2, eng.revision, v0, False, sub), cache).clone())
        check(f"E5_causal{tag}", all(torch.equal(outs[k], outs2[k]) for k in range(4)) and not torch.equal(outs[5], outs2[5]))
        # E6 / E7 / E8 / E9 / E10 on a real camera move (V0 -> camb)
        ST3 = AR.CandidateArm(cand, basis, True, "ST")
        c0 = eng.frame(v0, 2, sd(6, 0))
        ST3(AR.Frame(eng, c0, v0, 2, eng.revision, None, False, sub), {})
        c1 = eng.frame(camb, 2, sd(6, 1))
        ST3(AR.Frame(eng, c1, camb, 2, eng.revision, v0, False, sub), {})
        acc = ST3.last["accept"]
        j = ST3.g.map.long().clamp(min=0)
        cen1 = eng.centre(camb)
        cen0 = eng.centre(v0)
        g0, g1 = eng.guides(v0, sub), eng.guides(camb, sub)
        import svgf as SV
        x, y, z, _, _ = SV.projection(cen1, v0, W, basis)
        inframe = (z > 0) & (x >= -0.5) & (x < W - 0.5) & (y >= -0.5) & (y < W - 0.5)
        hid_or = eng.hidden_from(v0, cen1[:3], oracle=True)
        dis = (cen1[6] >= 0) & (hid_or | ~inframe)
        pure1 = g1["pure"].reshape(N)
        vis_pure = (cen1[6] >= 0) & ~dis & pure1 & g0["pure"].reshape(N)[j]
        frac = float(acc[vis_pure].float().mean())
        check(f"E6_real_reprojection_acceptance{tag}", frac >= 0.90, accept_fraction=frac)
        ret = float(acc[dis].float().mean()) if dis.any() else 0.0
        check(f"E7_disocclusion{tag}", dis.any() and ret <= 0.005, retained_fraction=ret, disoccluded=int(dis.sum()))
        check(f"E8_mixed_never_accepted{tag}", not bool(acc[~pure1].any()) and not bool(acc[~g0["pure"].reshape(N)[j]].any()))
        check(f"E9_id_change_never_accepted{tag}", not bool(acc[cen1[6] != cen0[6][j]].any()))
        o0, o1 = torch.tensor(basis(v0)[0], device="cuda").float(), torch.tensor(basis(camb)[0], device="cuda").float()
        a, b = o0[:, None] - cen1[:3], o1[:, None] - cen1[:3]
        vcos = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=1e-12)
        check(f"E10_view_gate_real{tag}", not bool(acc[(cen1[6] == 6) & (vcos < 0.995)].any()))
        # E11 negative control (box edit) and E12 light edit: radiance change on matched floor pixels; full reset
        for test, geom2, light2, label in ((11, rows[C.P["sequences"]["Q3"]["edit_to_config"]]["geom"], rows[0]["lights"][0], "E11_geometry_edit"),
                                           (12, rows[0]["geom"], rows[0]["lights"][C.P["sequences"]["Q5"]["light_index"]], "E12_light_edit")):
            eng.set_state(rows[0]["geom"], rows[0]["lights"][0])
            STe = AR.CandidateArm(cand, basis, True, "ST")
            before = [eng.frame(v0, 64 if not smoke else 4, sd(test, 10 + k)) for k in range(2)]
            cache = {}
            for k in range(4):
                STe(AR.Frame(eng, eng.frame(v0, 2, sd(test, k)), v0, 2, eng.revision, v0, False, sub), cache)
            cb = eng.centre(v0)
            eng.set_state(geom2, light2)
            after = [eng.frame(v0, 64 if not smoke else 4, sd(test, 20 + k)) for k in range(2)]
            ca = eng.centre(v0)
            cache = {}
            STe(AR.Frame(eng, eng.frame(v0, 2, sd(test, 30)), v0, 2, eng.revision, v0, True, sub), cache)
            age = STe.last["age"]
            match = (cb[6] == ca[6]) & (cb[6] == 0) & ((cb[3:6] * ca[3:6]).sum(0) >= 0.9)
            lum = lambda cc: C.LUMA[0] * (cc[0] + cc[3] + cc[6]) + C.LUMA[1] * (cc[1] + cc[4] + cc[7]) + C.LUMA[2] * (cc[2] + cc[5] + cc[8])
            lb = torch.stack([lum(x).reshape(N) for x in before])
            la = torch.stack([lum(x).reshape(N) for x in after])
            d = la.mean(0) - lb.mean(0)
            sig = torch.sqrt((lb.var(0) + la.var(0)) / 2 + 1e-30)
            changed = match & (d.abs() > 3 * sig)
            check(f"{label}{tag}", bool((age == 1).all()) and float(changed.float().sum() / match.float().sum().clamp(min=1)) > 0.01,
                  id_normal_matched_floor=int(match.sum()), matched_with_resolved_change=int(changed.sum()),
                  mean_rel_change_matched=float((d[match].abs().mean() / lb.mean(0)[match].mean()).item()),
                  history_age_max_after_edit=float(age.max()))
        eng.set_state(rows[0]["geom"], rows[0]["lights"][0])
        # E13 fused temporal == reference on the real camera move
        h_f, h_r = RC.History(N, "cuda", cand.cfg), RC.History(N, "cuda", cand.cfg)
        hid = eng.hidden_from(v0, cen1[:3])
        for (cam, c, cen, gg, hd) in ((v0, c0, cen0, g0, None), (camb, c1, cen1, g1, hid)):
            l_, h_ = RC.raw_low_high(c)
            a1 = h_f.step(l_.reshape(3, N), h_.reshape(3, N), 2, cen, gg["pure"].reshape(N), cam, 1, basis, hidden=hd, fused=True)
            a2 = h_r.step(l_.reshape(3, N), h_.reshape(3, N), 2, cen, gg["pure"].reshape(N), cam, 1, basis, hidden=hd, fused=False)
        # E13 graph runtime == eager fused path (temporal step + spatial passes) on the same real camera move
        gc = RC.GraphCandidate(cand, N, True)
        h_e = RC.History(N, "cuda", cand.cfg)
        ge = []
        for (cam, c, cen, gg, hd, prv) in ((v0, c0, cen0, g0, None, None), (camb, c1, cen1, g1, hid, v0)):
            l_, h_ = RC.raw_low_high(c)
            pk = dict(gg, centre=cen, hidden=hd if hd is not None else torch.zeros(N, dtype=torch.bool, device="cuda"))
            yg = gc.run(l_, h_, 2, pk, cam, 1, basis).clone()
            al, ah, ess, _, _, _ = h_e.step(l_.reshape(3, N), h_.reshape(3, N), 2, cen, gg["pure"].reshape(N), cam, 1, basis,
                                            hidden=hd, fused=True)
            lo, hi = cand.spatial(al.reshape(3, W, W), ah.reshape(3, W, W), ess.reshape(W, W), gg)
            ge.append(rel_l1(yg, lo + hi))
        check(f"E13_graph_equals_eager{tag}", max(ge) <= 1e-6, rel_l1=ge)
        agree = float((h_f.map == h_r.map).float().mean())
        both = (h_f.map == h_r.map) & (h_f.map >= 0)
        e = float(((h_f.acc - h_r.acc)[:, both].abs() / h_r.acc[:, both].abs().clamp(min=1e-6)).max()) if both.any() else 0.0
        check(f"E13_fused_temporal_real{tag}", agree >= 0.9999 and e <= 1e-5, map_agreement=agree, max_rel=e)
    allp = all(v["pass"] for v in RES.values())
    C.write(out / "synth.json", {"results": RES, "all_pass": allp, "protocol_sha256": C.sha(C.PROTOCOL), "smoke": smoke,
                                 "code_sha256": {f.name: C.sha(f) for f in sorted(C.HERE.glob("*.py"))}})
    print("ALL_PASS", allp, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.root.resolve(), a.smoke)
