"""WL109 Stage C (part 2): GPU contract qualification of C1 on the real frozen geometry with synthetic and learned HIGH tables.

python synth109.py OUT [--smoke]

S-structural (exact): R0 re-implementation identity, LOW identity, replay identity (inner path == original HIGH path), zero
table, determinism, frozen == eager. S-statistical (dev view, 4 spp, 64 replicates): paired expectation, spatial bias power,
M = 1 variance equivalence, M = 2 variance theory, fixed-jitter stream independence (+ power control), known negative and positive
covariance cases. S-quadrature: inner-MC conditional expectation at fixed pixel-centre U against an independent dense
cosine-grid quadrature using closure eval (conditional PDF, throughput and normalization).
"""
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

from common109 import P, DEV_VIEWS, apply_smoke, array_sha, rois, seed_synth, sha, write, HERE
import runtime109 as RT

TABLES = ("learned", "zero", "constant", "piecewise", "directional", "contrast", "positive_cov", "negative_cov")
STAT_TABLES = ("learned", "constant", "piecewise", "directional", "contrast", "positive_cov", "negative_cov")
QUAD_TABLES = ("learned", "constant", "piecewise", "directional")


def make_tables(R, rt, Bd):
    import torch
    PT, S = R.PT, R.S
    N, NB = rt.N, PT.NB
    rs = np.random.default_rng(P["synthetic"]["table_seed"])
    psid = np.asarray(rt.inc.G.PSID)
    b = np.arange(NB)
    zc = ((b // S.NPHI) + 0.5) / S.NZ
    ip = b % S.NPHI
    src = rt.source()[0].double().cpu().numpy()                     # [N, NB, 3] source encoding of the frozen state
    lit = src.sum((1, 2)); lit = lit / lit.max()
    t = {}
    t["learned"] = np.asarray(Bd, np.float32).reshape(N, NB, 3)
    t["zero"] = np.zeros((N, NB, 3), np.float32)
    t["constant"] = np.full((N, NB, 3), 0.5, np.float32)
    t["piecewise"] = (0.3 * np.exp(rs.normal(0, 0.75, (N, 1, 3)))).repeat(NB, 1).astype(np.float32)
    dirv = (0.2 + 1.6 * zc) * np.where(ip % 2 == 0, 0.5, 1.5)
    t["directional"] = (dirv[None, :, None] * np.array([1.0, 0.8, 0.6])[None, None, :]).repeat(N, 0).astype(np.float32)
    box = np.isin(psid, (S.SID_BOXA, S.SID_BOXB))
    t["contrast"] = np.where(box[:, None, None], 5.0, 0.05).repeat(NB, 1).repeat(3, 2).astype(np.float32)
    t["positive_cov"] = (2.0 * lit)[:, None, None].repeat(NB, 1).repeat(3, 2).astype(np.float32)
    t["negative_cov"] = (2.0 * (1.0 - lit))[:, None, None].repeat(NB, 1).repeat(3, 2).astype(np.float32)
    return {k: (v, PT.fl(np.ascontiguousarray(v))) for k, v in t.items()}


def rel_l1(a, b):
    import torch
    den = float(b.abs().sum())
    return float((a - b).abs().sum()) / den if den > 0 else float((a - b).abs().sum())


# ------------------------------------------------------------------------------------------------ structural
def structural(eng, cams, tables, smoke, log):
    import torch
    tol = P["synthetic"]["structural_rel_l1"]
    rows, ok = [], True
    for ti, name in enumerate(TABLES):
        Bt = tables[name][1]
        for di, view in enumerate(DEV_VIEWS):
            for spp in (1, 4):
                seed = seed_synth(ti, 90 + 10 * di + spp, 0, smoke)
                r0 = eng.r0(cams[view], spp, seed, B=Bt)                                  # e1 e2 hl
                c0 = eng.c1_frame(cams[view], spp, seed, 0, orig_hl=True, B=Bt)           # e1 e2 ho
                c1 = eng.c1_frame(cams[view], spp, seed, 1, orig_hl=True, B=Bt)           # e1 e2 hl ho
                c1b = eng.c1_frame(cams[view], spp, seed, 1, orig_hl=True, B=Bt)
                rp = eng.c1_frame(cams[view], spp, seed, 1, orig_hl=True, replay=True, B=Bt)
                # key order: c0 = e1(3) e2(3) ho(3); c1 = e1 e2 hl ho
                row = {"table": name, "view": view, "spp": spp,
                       "reimpl_low_rel_l1": rel_l1(c0[0:6], r0[0:6]), "reimpl_ho_rel_l1": rel_l1(c0[6:9], r0[6:9]),
                       "reimpl_bitwise": bool(torch.equal(c0, r0)),
                       "c1_low_rel_l1": rel_l1(c1[0:6], r0[0:6]), "c1_ho_rel_l1": rel_l1(c1[9:12], r0[6:9]),
                       "c1_low_bitwise": bool(torch.equal(c1[0:6], r0[0:6])),
                       "replay_high_rel_l1": rel_l1(rp[6:9], r0[6:9]), "replay_high_bitwise": bool(torch.equal(rp[6:9], r0[6:9])),
                       "replay_low_bitwise": bool(torch.equal(rp[0:6], r0[0:6])),
                       "deterministic": bool(torch.equal(c1, c1b))}
                checks = [row["reimpl_low_rel_l1"] <= tol, row["reimpl_ho_rel_l1"] <= tol, row["c1_low_rel_l1"] <= tol,
                          row["c1_ho_rel_l1"] <= tol, row["replay_high_rel_l1"] <= tol, row["deterministic"]]
                if name == "zero":
                    row["zero_high_exact"] = bool(torch.count_nonzero(c1[6:9]) == 0 and torch.count_nonzero(r0[6:9]) == 0)
                    checks.append(row["zero_high_exact"])
                row["pass"] = bool(all(checks))
                ok &= row["pass"]
                rows.append(row)
        log("structural", name, all(r["pass"] for r in rows if r["table"] == name))
    # frozen == eager (learned table, dev view 0, 4 spp) for both arms
    view = DEV_VIEWS[0]
    seed = seed_synth(0, 99, 0, smoke)
    fz = {}
    for kind, M in (("R0", 0), ("C1", 1)):
        out = eng.run_frozen(kind, view + "_synth", cams[view], 4, seed, M)
        out = eng.run_frozen(kind, view + "_synth", cams[view], 4, seed, M)        # replayed recording
        fz[kind] = torch.stack([o.torch() for o in out]).view(-1, 512, 512) / 4
    er0 = eng.r0(cams[view], 4, seed)
    ec1 = eng.c1_frame(cams[view], 4, seed, 1, orig_hl=False)
    frozen_row = {"r0_frozen_eq_eager_rel_l1": rel_l1(fz["R0"], er0), "c1_frozen_eq_eager_rel_l1": rel_l1(fz["C1"], ec1),
                  "r0_bitwise": bool(torch.equal(fz["R0"], er0)), "c1_bitwise": bool(torch.equal(fz["C1"], ec1))}
    frozen_row["pass"] = frozen_row["r0_frozen_eq_eager_rel_l1"] <= tol and frozen_row["c1_frozen_eq_eager_rel_l1"] <= tol
    ok &= frozen_row["pass"]
    return {"pass": bool(ok), "rows": rows, "frozen_vs_eager": frozen_row,
            "all_bitwise": bool(all(r["reimpl_bitwise"] and r["c1_low_bitwise"] and r["replay_high_bitwise"] for r in rows))}


# ------------------------------------------------------------------------------------------------ statistical
def regions(aux):
    from scipy.ndimage import binary_dilation
    m = rois(aux)
    sid = aux["sid"]
    valid = aux["valid_frac"] > 0
    box = np.isin(sid, (7, 8))
    edge = binary_dilation(box, iterations=2) & binary_dilation(~box & (sid >= 0), iterations=2) & valid
    return {"all_valid": valid, "sphere_interior": m["sss_interior"], "silhouette": m["silhouette"],
            "box_visibility_edge": edge, "contact": m["contact"], "control": m["control"]}


def statistical(eng, cams, tables, smoke, log):
    import torch
    import stats109 as ST
    cfg = P["synthetic"]["statistical"]
    view = cfg["view"]
    R, spp = cfg["replicates"], cfg["spp"]
    aux = RT.aux(view)
    regs = {k: v for k, v in regions(aux).items() if v.sum() >= cfg["min_region_pixels"]}
    dev = "cuda"
    out, ok = {}, True
    for ti, name in enumerate(STAT_TABLES):
        Bt = tables[name][1]
        frames = {}
        for M in (1, 2):
            L, HO, HC = [], [], []
            for r in range(R):
                c = eng.c1_frame(cams[view], spp, seed_synth(10 + ti, M, r, smoke), M, orig_hl=True, B=Bt).double()
                L.append(c[0:3] + c[3:6]); HC.append(c[6:9]); HO.append(c[9:12])
            frames[M] = {k: torch.stack(v) for k, v in (("L", L), ("HO", HO), ("HC", HC))}
        boot = ST.Boot(R, cfg["bootstrap_draws"], P["seeds"]["bootstrap"] + ti, dev)
        boot2 = ST.Boot(R, cfg["bootstrap_draws"], P["seeds"]["bootstrap"] + 20 + ti, dev)
        rec = {}
        f1, f2 = frames[1], frames[2]
        for rn, mask in regs.items():
            m = {}
            # paired expectation (both M), replicate ROI luminance means
            for M, f in ((1, f1), (2, f2)):
                d = ST.region_luma_means(f["HC"] - f["HO"], mask)
                tt = ST.t_test(d)
                scale = ST.pooled(ST.pix_var(f["L"] + f["HO"]), mask)
                Q = ST.spatial_bias_power(f["HC"] - f["HO"], mask, scale, cfg["Q_groups"]) if scale > 0 else {"point": 0.0, "se": 0.0, "upper95": 0.0}
                m[f"M{M}_paired_t"] = tt
                m[f"M{M}_spatial_bias_power"] = Q
            vho = lambda i: ST.pooled(ST.pix_var(f1["HO"], i), mask)
            if vho(None) > 0:
                m["M1_var_ratio"] = ST.ratio_stat(lambda i: ST.pooled(ST.pix_var(f1["HC"], i), mask), vho, boot)
                # M = 2 theory: Var(HC_2) = C_J + (V_H - C_J) / 2, C_J = Cov(HO, HC_1), V_H = Var(HO), from the M = 1 run
                def pred(i):
                    cj_ = ST.pooled(ST.pix_cov(f1["HO"], f1["HC"], i), mask)
                    vh_ = ST.pooled(ST.pix_var(f1["HO"], i), mask)
                    return cj_ + (vh_ - cj_) / 2
                num = lambda i: ST.pooled(ST.pix_var(f2["HC"], i), mask)
                p2 = num(None) / pred(None)
                draws = np.array([ST._div(num(b), pred(a)) for a, b in zip(boot.idx, boot2.idx)])
                se2 = float(np.nanstd(draws, ddof=1))
                m["M2_var_over_theory"] = {"point": float(p2), "se": se2, "ci95": [p2 - 1.96 * se2, p2 + 1.96 * se2]}
                m["J_share_of_high_variance"] = ST.pooled(ST.pix_cov(f1["HO"], f1["HC"]), mask) / vho(None)
            vt0 = lambda i: ST.pooled(ST.pix_var(f1["L"] + f1["HO"], i), mask)
            m["M1_total_var_ratio"] = ST.ratio_stat(lambda i: ST.pooled(ST.pix_var(f1["L"] + f1["HC"], i), mask), vt0, boot)
            rec[rn] = m
        # gates (thresholds from the protocol)
        g = cfg["gates"]
        fails = []
        for rn, m in rec.items():
            for M in (1, 2):
                if abs(m[f"M{M}_paired_t"]["t"]) > g["paired_t_max"]:
                    fails.append((rn, f"M{M}_paired_t"))
                if m[f"M{M}_spatial_bias_power"]["upper95"] > g["spatial_bias_power_max"]:
                    fails.append((rn, f"M{M}_Q"))
            if "M1_var_ratio" in m:
                # a deviation fails only if it is both material (beyond tolerance) and resolved (beyond z_max SE)
                v = m["M1_var_ratio"]
                if abs(v["point"] - 1) > g["var_equivalence_tol"] and abs(v["point"] - 1) > g["z_max"] * v["se"]:
                    fails.append((rn, "M1_var_equivalence"))
                w = m["M2_var_over_theory"]
                if abs(w["point"] - 1) > g["var_theory_tol"] and abs(w["point"] - 1) > g["z_max"] * w["se"]:
                    fails.append((rn, "M2_var_theory"))
        if name == "negative_cov":
            r_all = rec["all_valid"]["M1_total_var_ratio"]
            if not r_all["ci95"][0] > 1.0:
                fails.append(("all_valid", "negative_case_not_worse"))
        if name == "positive_cov":
            r_all = rec["all_valid"]["M1_total_var_ratio"]
            if not r_all["ci95"][1] < 1.0:
                fails.append(("all_valid", "positive_case_not_better"))
        rec["fails"] = fails
        rec["pass"] = not fails
        ok &= rec["pass"]
        out[name] = rec
        log("statistical", name, "PASS" if not fails else fails)
        del frames
        torch.cuda.empty_cache()
    # CPU/GPU agreement of the pooled-variance arithmetic on the last table's frames is checked in independence()
    return {"pass": bool(ok), "view": view, "regions_pixels": {k: int(v.sum()) for k, v in regs.items()}, "tables": out}


def independence(eng, cams, tables, smoke, log):
    """Fixed pixel-centre camera rays remove J: inner-stream HIGH must be uncorrelated with LOW and the original HIGH."""
    import torch
    import stats109 as ST
    cfg = P["synthetic"]["independence"]
    view, R, spp = cfg["view"], cfg["replicates"], cfg["spp"]
    aux = RT.aux(view)
    mask = aux["valid_frac"] > 0
    out, ok = {}, True
    for ti, name in enumerate(cfg["tables"]):
        Bt = tables[name][1]
        L, HO, HC = [], [], []
        for r in range(R):
            c = eng.c1_frame(cams[view], spp, seed_synth(30 + ti, 0, r, smoke), 1, orig_hl=True, fixed_jitter=True, B=Bt).double()
            L.append(c[0:3] + c[3:6]); HC.append(c[6:9]); HO.append(c[9:12])
        L, HO, HC = torch.stack(L), torch.stack(HO), torch.stack(HC)
        boot = ST.Boot(R, cfg["bootstrap_draws"], P["seeds"]["bootstrap"] + 50 + ti, "cuda")

        def corr(X, Y):
            def f(i):
                c = ST.pooled(ST.pix_cov(X, Y, i), mask)
                return ST._div(c, math.sqrt(ST.pooled(ST.pix_var(X, i), mask) * ST.pooled(ST.pix_var(Y, i), mask)))
            p = f(None)
            se, _ = boot.se(f)
            return {"point": p, "se": se}
        rec = {"corr_HC_L": corr(HC, L), "corr_HC_HO": corr(HC, HO), "power_control_corr_HO_L": corr(HO, L)}
        g = cfg["gates"]
        rec["pass"] = bool(all(abs(rec[k]["point"]) <= g["abs_corr_max"] and abs(rec[k]["point"]) <= g["z_max"] * rec[k]["se"] + 1e-12
                               for k in ("corr_HC_L", "corr_HC_HO")) and
                           abs(rec["power_control_corr_HO_L"]["point"]) > g["z_max"] * rec["power_control_corr_HO_L"]["se"])
        if ti == 0:   # CPU / GPU agreement of the pooled variance arithmetic
            gpu = ST.pooled(ST.pix_var(HO), mask)
            cpu = ST.numpy_pooled_var(HO.cpu().numpy(), mask)
            rec["cpu_gpu_pooled_var_rel_diff"] = abs(gpu - cpu) / abs(cpu)
            rec["pass"] &= rec["cpu_gpu_pooled_var_rel_diff"] <= 1e-9
        ok &= rec["pass"]
        out[name] = rec
        log("independence", name, rec["pass"])
    return {"pass": bool(ok), "tables": out}


# ------------------------------------------------------------------------------------------------ replicate independence
def replicate_independence(eng, cams, tables, smoke, log):
    """Consecutive-seed replicate pairs must be uncorrelated under the protocol seeding (splitmix64 main-stream seed value).
    Power control: the historical seed * 1000003 convention, which the development diagnostic found to correlate
    consecutive pcg_046 replicates, must show a resolved correlation with the same statistic."""
    import torch
    from c1_109 import inner_seed_value
    cfg = P["synthetic"]["replicate_independence"]
    view, R_, spp = cfg["view"], cfg["replicates"], cfg["spp"]
    mnp = RT.aux(view)["valid_frac"] > 0
    mask = torch.as_tensor(mnp, device="cuda")
    mi, dr, rt = eng.mi, eng.dr, eng.rt
    cam = dict(cams[view], res=512)
    out, ok = {}, True
    for ti, name in enumerate(cfg["tables"]):
        Bt = tables[name][1]
        rec = {}
        for conv in ("protocol", "historical"):
            L, HO, HC = [], [], []
            for r in range(R_):
                seed = seed_synth(60 + ti, 0, r, smoke)
                mv = eng.main_seed(seed) if conv == "protocol" else dr.opaque(mi.UInt64, seed * 1000003)
                o = eng.c1.frame_prog(rt.scene, rt.tabs(), rt.light_args(), Bt, mv, eng.in_seed(seed), (cam, 512, spp), 1, orig_hl=True)
                c = eng._to_torch(o, spp).double()
                L.append(c[0:3] + c[3:6]); HC.append(c[6:9]); HO.append(c[9:12])
            X = {"LOW": torch.stack(L), "HO": torch.stack(HO), "HC": torch.stack(HC)}
            X["D"] = X["HC"] - X["HO"]
            row = {}
            for k, v in X.items():
                Xm = v[:, :, mask]
                c = Xm - Xm.mean(0)
                prod = (c[0::2] * c[1::2]).mean((1, 2))           # consecutive-seed pairs
                var = Xm.var(0, unbiased=True).mean()
                excess = float(prod.mean() / var + 1.0 / R_)       # E[prod] = -Var / R under independence
                se = float(prod.std(unbiased=True) / np.sqrt(len(prod)) / var)
                row[k] = {"excess_pair_correlation": excess, "se": se, "z": excess / se if se > 0 else 0.0}
            rec[conv] = row
        g = cfg["gates"]
        prot_ok = all(abs(v["z"]) <= g["z_max"] for v in rec["protocol"].values())
        rec["protocol_pass"] = bool(prot_ok)
        rec["historical_power_control_resolved"] = bool(rec["historical"]["HO"]["z"] > g["z_max"])
        ok &= prot_ok and (rec["historical_power_control_resolved"] or name != cfg["power_control_table"])
        out[name] = rec
        log("replicate_independence", name, {c: {k: round(v["z"], 2) for k, v in rec[c].items()} for c in ("protocol", "historical")})
    return {"pass": bool(ok), "view": view, "tables": out}


# ------------------------------------------------------------------------------------------------ quadrature
def quadrature_values(eng, camera, Bd, pix, n_grid):
    R, rt = eng.R, eng.rt
    mi, dr, PT = R.mi, R.dr, R.PT
    F, U = mi.Float, mi.UInt32
    rt._bind(rt.tabs())
    ray0, _ = PT.gen_rays(dict(camera, res=512), 512, 1, None)
    g = (np.arange(n_grid) + 0.5) / n_grid
    gx, gy = [a.ravel().astype(np.float32) for a in np.meshgrid(g, g, indexing="ij")]
    nd = n_grid * n_grid
    out = np.zeros((len(pix), 3))
    batch = max(1, (1 << 22) // nd)
    GX, GY = F(gx), F(gy)
    for b0 in range(0, len(pix), batch):
        pids = pix[b0:b0 + batch]
        nb = len(pids)
        lane = dr.arange(U, nb * nd)
        pi = lane // nd
        di = lane % nd
        px = dr.gather(U, U(pids.astype(np.uint32)), pi)
        o = mi.Point3f(*[dr.gather(F, ray0.o[k], px) for k in range(3)])
        d = mi.Vector3f(*[dr.gather(F, ray0.d[k], px) for k in range(3)])
        ray = mi.Ray3f(o, d)
        si = rt.scene.ray_intersect(ray)
        act = si.is_valid()
        sid = rt.shape_id(si)
        pid, bn = rt.patch_bin(si, sid, -ray.d)
        mid = dr.gather(U, rt.PMATU, pid, act)
        wi = si.to_local(-ray.d)
        act &= wi.z > 1e-6
        wi = dr.select(act, wi, mi.Vector3f(0, 0, 1))
        act &= ~dr.eq(mid, 2)
        cl = PT.CD.DrClosure83("ER", rt.mats.states, mid)
        wo = mi.warp.square_to_cosine_hemisphere(mi.Point2f(dr.gather(F, GX, di), dr.gather(F, GY, di)))
        f = cl.eval(wi, wo, act)                                           # BSDF x cosine (closure eval, not the sampler)
        w = f * (math.pi / dr.maximum(wo.z, 1e-12))                        # / cosine-hemisphere density
        nr = si.spawn_ray(si.to_world(wo))
        s2 = rt.scene.ray_intersect(nr, act)
        a2 = act & s2.is_valid()
        p2, b2 = rt.patch_bin(s2, rt.shape_id(s2), -nr.d)
        a2 &= s2.to_local(-nr.d).z > 1e-6
        i = (p2 * PT.NB + b2) * 3
        val = [dr.select(a2, w[c] * dr.gather(F, Bd, i + c, a2), 0.) for c in range(3)]
        for c in range(3):
            buf = dr.zeros(F, nb)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, val[c], pi)
            dr.eval(buf)
            out[b0:b0 + nb, c] = np.asarray(buf, np.float64) / nd
    return out


def quadrature(eng, cams, tables, smoke, log):
    import torch
    cfg = P["synthetic"]["quadrature"]
    view = cfg["view"]
    aux = RT.aux(view)
    sid, ok_c = aux["sid"].ravel(), aux["ok"].ravel()
    cand = np.flatnonzero(ok_c & (sid >= 0) & (sid != 6))
    rs = np.random.default_rng(cfg["pixel_seed"])
    pix = np.sort(rs.choice(cand, size=min(cfg["pixels"], len(cand)), replace=False))
    res, ok = {}, True
    for ti, name in enumerate(QUAD_TABLES):
        Bt = tables[name][1]
        q1 = quadrature_values(eng, cams[view], Bt, pix, cfg["grid"])
        q2 = quadrature_values(eng, cams[view], Bt, pix, 2 * cfg["grid"])
        per = []
        for r in range(cfg["frames"]):
            c = eng.c1_frame(cams[view], cfg["spp"], seed_synth(40 + ti, 0, r, smoke), cfg["M"], orig_hl=False, fixed_jitter=True, B=Bt)
            per.append(c[6:9].reshape(3, -1)[:, torch.as_tensor(pix, device=c.device)].double().cpu().numpy().T)
        per = np.stack(per)                                    # [frames, pixels, 3]
        mc = per.mean(0)
        se = per.std(0, ddof=1) / math.sqrt(per.shape[0])
        qerr = np.abs(q2 - q1)
        tol = cfg["z_max"] * se + 2 * qerr + 1e-7
        viol = np.abs(mc - q2) > tol
        z = (mc - q2) / np.maximum(se, 1e-12)
        zz = z[se > 0]
        nu = per.shape[0] - 1
        rec = {"pixels": int(len(pix)), "violation_fraction": float(viol.mean()), "mean_z": float(zz.mean()),
               "mean_z2": float((zz ** 2).mean()), "expected_mean_z2_student": nu / (nu - 2),
               "relative_mean_gap": float((mc - q2).sum() / max(q2.sum(), 1e-30)),
               "quadrature_grid_change_relative": float(qerr.sum() / max(q2.sum(), 1e-30))}
        g = cfg["gates"]
        rec["pass"] = bool(rec["violation_fraction"] <= g["violation_fraction_max"] and abs(rec["mean_z"]) <= g["mean_z_abs_max"]
                           and rec["mean_z2"] <= rec["expected_mean_z2_student"] + g["mean_z2_excess_max"])
        ok &= rec["pass"]
        res[name] = rec
        log("quadrature", name, rec)
    return {"pass": bool(ok), "view": view, "tables": res}


def main(out, smoke):
    import toy109
    if smoke:
        apply_smoke()
    out.mkdir(parents=True, exist_ok=False)
    logf = open(out / "synth.log", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    t0 = time.time()
    result = {"protocol_sha256": sha(HERE / "protocol109.json"), "smoke": smoke,
              "code_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}, "complete": False}
    toy = toy109.run(P["synthetic"]["toy"], P["synthetic"]["toy"]["seeds"])
    result["toy"] = toy
    log("toy", toy["pass"])
    write(out / "synth.json", result)
    R, rt, Bd, audit, _ = RT.admit()
    from gpu109 import Engine
    eng = Engine(R, rt, Bd)
    result["admission"] = audit
    result["versions"] = RT.versions(R)
    cams = RT.cameras()
    tables = make_tables(R, rt, Bd)
    result["table_sha256"] = {k: array_sha(v[0]) for k, v in tables.items()}
    assert result["table_sha256"]["learned"] == audit["table_bytes_sha256"]
    for stage, fn in (("structural", structural), ("replicate_independence", replicate_independence), ("statistical", statistical),
                      ("independence", independence), ("quadrature", quadrature)):
        result[stage] = fn(eng, cams, tables, smoke, log)
        write(out / "synth.json", result)
    result["learned_table_unchanged"] = RT.table_bytes_sha(Bd) == audit["table_bytes_sha256"]
    result["pass"] = bool(toy["pass"] and all(result[s]["pass"] for s in ("structural", "replicate_independence", "statistical",
                                                                         "independence", "quadrature"))
                          and result["learned_table_unchanged"])
    result["seconds"] = time.time() - t0
    result["complete"] = True
    write(out / "synth.json", result)
    log("SYNTH", "PASS" if result["pass"] else "FAIL", round(result["seconds"], 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(Path(a.out).resolve(), a.smoke)
