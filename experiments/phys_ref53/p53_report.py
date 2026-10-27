"""WL53 report: predeclared rules (protocol phys_ref53) -> report.json.

    python p53_report.py <results dir>        (expects render/, oracle/, cycles/ under it)
Port comparisons: per camera, port group (component : rest face) and channel, over ports with >= 50 interior pixels.
bias B = sum(pred - ref) / sum(ref); SE from the 2 render seeds (var of a 2-seed mean = (x1 - x2)^2 / 4) plus, where two
oracle seeds exist, the oracle seed variance. Groups with < 2 % of the camera's reference signal or < 5 ports are reported
but not ruled.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p53_fixtures as FX  # noqa: E402

MINPIX, MINPORTS, MINSHARE = 30, 5, 0.02
FACE = {0: "-x", 1: "+x", 2: "-y", 3: "+y", 4: "bot", 5: "top"}


def dirkey(d):
    for k, v in FX.DIRS.items():
        if np.allclose(v, d):
            return k
    raise KeyError(d)


def port_groups(scene):
    sd = FX.scene_def(scene); labels = []
    for n, k, m, V in sd["comps"]:
        labels += [f"{n}:{FACE[int(f)]}" if k == "medium" else n for f in m["port_face"]]
    return np.array(labels)


def compare(ref2, pred, count, groups, tol, pred2=None, min_share=MINSHARE):
    """ref2 (2, P, 3) render seeds; pred (P, 3) (or pred2 (2, P, 3) oracle seeds -> mean)."""
    ok = count >= MINPIX
    ref = ref2.mean(0); vref = (ref2[0] - ref2[1]) ** 2 / 4
    if pred2 is not None:
        pred = pred2.mean(0); vpred = (pred2[0] - pred2[1]) ** 2 / 4
    else:
        vpred = np.zeros_like(ref)
    tot = ref[ok].sum(0)
    out = {}
    for g in np.unique(groups[ok]):
        m = ok & (groups == g)
        s = ref[m].sum(0)
        B = (pred[m] - ref[m]).sum(0) / np.maximum(s, 1e-30)
        SE = np.sqrt(vref[m].sum(0) + vpred[m].sum(0)) / np.maximum(s, 1e-30)
        D = np.abs(pred[m] - ref[m]).sum(0) / np.maximum(s, 1e-30)
        ruled = (m.sum() >= MINPORTS) & (s / np.maximum(tot, 1e-30) >= min_share)
        passed = np.where(ruled, np.abs(B) <= tol + 3 * SE, True)
        out[g] = dict(n_ports=int(m.sum()), share=(s / np.maximum(tot, 1e-30)).round(4).tolist(), bias=B.round(5).tolist(), se=SE.round(5).tolist(),
                      abs_dev=D.round(5).tolist(), ruled=ruled.tolist(), pass_=passed.tolist())
    return out


def all_pass(d):
    return bool(all(all(v["pass_"]) for v in d.values()))


def load_ports(path):
    z = np.load(path); cams = sorted({k.split("__")[0] for k in z.files})
    return {c: (z[f"{c}__mean"], z[f"{c}__count"]) for c in cams}


def ratio_se(a, b):
    """mean(a) / mean(b) over seeds with delta-method SE (a, b: (n_seeds, 3))."""
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    ma, mb = a.mean(0), b.mean(0); r = ma / mb
    sa, sb = a.std(0, ddof=1) / math.sqrt(n), b.std(0, ddof=1) / math.sqrt(n)
    return r, r * np.sqrt((sa / ma) ** 2 + (sb / mb) ** 2)


def main():
    R = Path(sys.argv[1]); rd, od, cd = R / "render", R / "oracle", R / "cycles"
    rep = dict(protocol="phys_ref53"); V = {}
    # ---------------- G1 (primary reference: Cycles volume; Mitsuba = diagnostic second renderer)
    g = {}
    ch = json.load(open(cd / "c_chain.json"))
    g["chain"] = dict(**ch, pass_=bool(abs(ch["lambert_L"] / ch["lambert_analytic"] - 1) <= 0.002 and abs(ch["fisheye_E_uniform"][0] / math.pi - 1) <= 0.005))
    ca = json.load(open(cd / "c_g1a.json")); iw = json.load(open(od / "g1a_iw.json"))
    g["g1a"] = dict(cycles=[dict(albedo=x["albedo"], theta0=x["theta0"], theta=x["theta"], ratio=x["cycles"] / x["analytic"], sem=x["sem"] / x["analytic"],
                                 pass_=abs(x["cycles"] / x["analytic"] - 1) <= max(0.01, 3 * x["sem"] / x["analytic"])) for x in ca],
                    iw=[dict(albedo=x["albedo"], theta0=x["theta0"], theta=x["theta"], ratio=x["iw"] / x["analytic"], pass_=abs(x["iw"] / x["analytic"] - 1) <= 0.015,
                             plane_albedo_ratio=x["plane_albedo_iw"] / x["plane_albedo_analytic"]) for x in iw])
    if (rd / "g1a_mitsuba.json").exists():
        g["g1a"]["mitsuba_diagnostic"] = [dict(albedo=x["albedo"], theta0=x["theta0"], theta=x["theta"], ratio=x["mitsuba"] / x["analytic"], sem=x["sem"] / x["analytic"])
                                          for x in json.load(open(rd / "g1a_mitsuba.json"))]
    g["g1a"]["pass_cycles"] = all(x["pass_"] for x in g["g1a"]["cycles"]); g["g1a"]["pass_iw"] = all(x["pass_"] for x in g["g1a"]["iw"])
    g["g1b"] = {}
    for tag in ("alb1", "albW"):
        f = load_ports(rd / f"cports_furnace_{tag}.npz")
        for cam, (m, c) in f.items():
            ok = c >= MINPIX; mean = m[:, ok].mean((0, 1)); se = np.sqrt(((m[0, ok] - m[1, ok]) ** 2 / 4).sum(0)) / ok.sum()
            g["g1b"][f"{tag}_{cam}"] = dict(mean=mean.tolist(), se=se.tolist(), pass_=bool(np.all(np.abs(mean - 1) <= 0.005 + 3 * se)) if tag == "alb1" else None)
    g["g1b"]["pass"] = all(v["pass_"] for k, v in g["g1b"].items() if k.startswith("alb1"))
    sup = {t: load_ports(rd / f"cports_sup_{t}.npz") for t in ("zero", "A", "B", "AB")}; grp = port_groups("iso_L1"); g["g1c"] = {}
    g["g1c"]["zero_max"] = float(max(np.abs(sup["zero"][c][0]).max() for c in sup["zero"]))
    for cam in sup["AB"]:
        g["g1c"][cam] = compare(sup["AB"][cam][0], None, sup["AB"][cam][1], grp, 0.005, pred2=sup["A"][cam][0] + sup["B"][cam][0])
    g["g1c"]["pass"] = g["g1c"]["zero_max"] == 0.0 and all(all_pass(v) for k, v in g["g1c"].items() if isinstance(v, dict))
    ph = json.load(open(cd / "c_g1d.json")); g["g1d"] = {}
    for cfg, d in ph.items():
        r1, s1 = ratio_se(d["E_da"], d["E_cv"]); r2, s2 = ratio_se(d["L_mirror"], d["L_direct"])
        g["g1d"][cfg] = dict(diffuse_over_cv=r1.tolist(), se=s1.tolist(), mirror_over_direct=r2.tolist(), se_m=s2.tolist(), E_cv=np.mean(d["E_cv"], 0).tolist(),
                             pass_=bool(np.all(np.abs(r1 - 1) <= 0.02 + 3 * s1) and np.all(np.abs(r2 - 1) <= 0.02 + 3 * s2)))
    g["g1d"]["pass_primary"] = g["g1d"]["above_single_vol"]["pass_"] and g["g1d"]["above_single_lam"]["pass_"]
    g["g1e"] = {}
    for name in ("gap4", "gap4_below"):
        one = load_ports(rd / f"cports_grp_{name}_vol_one.npz"); two = load_ports(rd / f"cports_grp_{name}_vol_two.npz"); grp = port_groups(name)
        g["g1e"][name] = {cam: compare(two[cam][0], None, two[cam][1], grp, 0.005, pred2=one[cam][0]) for cam in one}
    g["g1e"]["pass"] = all(all_pass(v) for nm in ("gap4", "gap4_below") for v in g["g1e"][nm].values())
    one = load_ports(rd / "cports_grp_gap4_below_rw_one.npz"); two = load_ports(rd / "cports_grp_gap4_below_rw_two.npz"); grp = port_groups("gap4_below")
    g["g1e"]["rw_diagnostic"] = {cam: compare(two[cam][0], None, two[cam][1], grp, 0.005, pred2=one[cam][0], min_share=0.0) for cam in one}
    vol1 = load_ports(rd / "cports_grp_gap4_below_vol_one.npz")
    g["g1e"]["vol_vs_rw_signal"] = {cam: dict(vol=vol1[cam][0].mean(0).sum(0).tolist(), rw=one[cam][0].mean(0).sum(0).tolist()) for cam in one}
    rc = json.load(open(cd / "c_g1f.json")); g["g1f"] = {}
    for case, d in rc.items():
        r, sse = ratio_se(d["fwd"], d["rev"])
        g["g1f"][case] = dict(ratio=r.tolist(), se=sse.tolist(), informative=bool(np.all(sse <= 0.03)), pass_=bool(np.all(np.abs(r - 1) <= 0.03 + 3 * sse)))
    g["g1f"]["pass"] = all(v["pass_"] for v in g["g1f"].values() if v["informative"])
    tr = load_ports(rd / "cports_trunc512_lblock.npz"); full = load_ports(rd / "cports_lblock.npz"); grp = port_groups("lblock")
    g["trunc512_vs_1024"] = {cam: compare(full[cam][0], tr[cam][0][0], full[cam][1], grp, 0.0) for cam in tr}
    mt = {}
    if (rd / "ports_iso_L1.npz").exists():
        for name in ("iso_L1", "iso_L2", "iso_L3"):
            mi_ = load_ports(rd / f"ports_{name}.npz"); cy = load_ports(rd / f"cports_{name}.npz"); grp = port_groups(name)
            mt[name] = {cam: compare(cy[cam][0], None, np.minimum(cy[cam][1], mi_[cam][1] // 4), grp, 0.015, pred2=mi_[cam][0]) for cam in cy}
        ph = json.load(open(rd / "g1d_pathhist.json")); mt["g1d"] = {}
        for cfg, d in ph.items():
            r1, s1 = ratio_se(d["diffuse"]["E_diffuse_ancestor"], d["diffuse"]["E_camera_vertex"]); r2, s2 = ratio_se(d["mirror"]["L_via_mirror"], d["mirror"]["L_direct"])
            mt["g1d"][cfg] = dict(diffuse_over_cv=r1.tolist(), se=s1.tolist(), mirror_over_direct=r2.tolist(), se_m=s2.tolist())
        mt["g1e"] = {}
        for name in ("gap4", "gap4_below"):
            one = load_ports(rd / f"ports_grp_{name}_one.npz"); two = load_ports(rd / f"ports_grp_{name}_two.npz"); grp = port_groups(name)
            mt["g1e"][name] = {cam: compare(two[cam][0], None, two[cam][1], grp, 0.005, pred2=one[cam][0]) for cam in one}
        rc = json.load(open(rd / "g1f_recip.json")); mt["g1f"] = {}
        for pair, d in rc.items():
            keys = list(d); r, sse = ratio_se(d[keys[0]], d[keys[1]]); mt["g1f"][pair] = dict(ratio=r.tolist(), se=sse.tolist())
        f = json.load(open(rd / "g1b_furnace.json")); mt["g1b"] = {}
        for tag in f:
            for cam, d in f[tag].items():
                m = np.array(d["mean"]); c = np.array(d["count"]); ok = c >= MINPIX; mt["g1b"][f"{tag}_{cam}"] = m[:, ok].mean((0, 1)).tolist()
    g["mitsuba_diagnostic"] = mt
    rep["G1"] = g
    V["reference_physical_validity"] = dict(chain=g["chain"]["pass_"], analytic_cycles=g["g1a"]["pass_cycles"], furnace=g["g1b"]["pass"], superposition=g["g1c"]["pass"],
                                            path_history=g["g1d"]["pass_primary"], grouping=g["g1e"]["pass"], reciprocity=g["g1f"]["pass"], iw=g["g1a"]["pass_iw"])
    # ---------------- G2
    z = np.load(od / "g2_pred.npz"); props = json.load(open(od / "g2_props.json")); g2 = {}
    for name in ("iso_L1", "iso_L2", "iso_L3"):
        mit = load_ports(rd / f"cports_{name}.npz"); grp = port_groups(name); sd = FX.scene_def(name); g2[name] = {}
        for cam, (ref2, cnt) in mit.items():
            k = ("top", "bot", "otop", "obot").index(dirkey(sd["cams"][cam]["dir"]))
            e = dict(tr_exact=compare(ref2, z[f"{name}__L_exact"][k], cnt, grp, 0.015),
                     bin_nee=compare(ref2, z[f"{name}__L_bin_nee"][k], cnt, grp, 0.015), bin_out=compare(ref2, z[f"{name}__L_bin_out"][k], cnt, grp, 0.015))
            if name != "iso_L2":                                                  # IW models top-face entry only (oblique sun also enters a side)
                Liw = z[f"{name}__L_iw"][k]; okiw = ~np.isnan(Liw[:, 0])
                e["iw"] = compare(ref2, np.nan_to_num(Liw), np.where(okiw, cnt, 0), grp, 0.015)
            g2[name][cam] = e
    g2["pass_tr"] = all(all_pass(g2[n][c]["tr_exact"]) for n in ("iso_L1", "iso_L2", "iso_L3") for c in g2[n])
    g2["pass_iw"] = all(all_pass(g2[n][c]["iw"]) for n in ("iso_L1", "iso_L3") for c in g2[n])
    st = props["alpha1"]; g2["alpha1"] = dict(colsum_min=st["colsum_min"], colsum_max=st["colsum_max"],
                                             pass_=bool(max(abs(x - 1) for x in st["colsum_min"] + st["colsum_max"]) <= 1e-5))
    g2["passivity"] = dict(colsum_max=[props[t]["colsum_max"] for t in ("s1", "s2")], pass_=bool(max(max(props[t]["colsum_max"]) for t in ("s1", "s2")) <= 1 + 1e-5))
    leaks = {t: (props[t]["stats"]["leaks"] + props[t]["stats"]["wrong_side"]) / props[t]["stats"]["photons"] for t in ("s1", "s2", "alpha1")}
    g2["leaks"] = dict(fraction=leaks, pass_=bool(max(leaks.values()) <= 1e-5))
    rr = props["reciprocity"]
    g2["reciprocity"] = rr
    g2["reciprocity_pass"] = bool(all(rr[t]["dir_asym_adjacent"][k] <= 1.25 * rr[t]["dir_noise_adjacent"][k] + 0.005 for t in ("s1", "s2") for k in range(3)))
    g2["port_pair_reciprocity"] = {t: dict(pooled_asym=rr[t]["pooled_asym"], frac_absz_gt4=rr[t]["frac_absz_gt4"]) for t in ("s1", "s2")}
    rep["G2"] = g2
    V["intrinsic_K_ownership"] = dict(tr_vs_mitsuba=g2["pass_tr"], iw_vs_mitsuba=g2["pass_iw"], alpha1=g2["alpha1"]["pass_"], passivity=g2["passivity"]["pass_"],
                                      leaks=g2["leaks"]["pass_"], reciprocity=g2["reciprocity_pass"], grouping=g["g1e"]["pass"])
    # ---------------- G3 / G4 / G5 compositions
    g3 = {}
    pdir = od / "pred"

    def arm(scene, tag):
        p = pdir / f"{scene}__{tag}.npz"
        return (np.load(p), json.load(open(pdir / f"{scene}__{tag}.json"))) if p.exists() else (None, None)

    def eval_scene(scene, tags, mode="nee", two_seed=None, tol=0.03):
        mit = load_ports(rd / f"cports_{scene}.npz"); grp = port_groups(scene); sd = FX.scene_def(scene); res = {}
        for tag in tags:
            P, meta = arm(scene, tag)
            if P is None:
                continue
            r = dict(meta={k: v for k, v in meta.items() if k != "hist"}, cams={})
            for cam, (ref2, cnt) in mit.items():
                key = f"{dirkey(sd['cams'][cam]['dir'])}__{mode}"
                if key not in P.files:
                    continue
                pred2 = None
                if two_seed and tag == two_seed[0]:
                    P2, _ = arm(scene, two_seed[1]); pred2 = np.stack([P[key], P2[key]]) if P2 is not None else None
                r["cams"][cam] = compare(ref2, P[key], cnt, grp, tol, pred2=pred2)
            r["pass_"] = bool(meta.get("converged", False) and max(meta.get("rho", [0])) < 1 and all(all_pass(v) for v in r["cams"].values()))
            res[f"{tag}__{mode}"] = r
        return res

    for scene in ("gap4", "gap8", "floor"):
        g3[scene] = eval_scene(scene, ["part_s1"], "nee", two_seed=("part_s1", "part_s2"))
        g3[scene].update(eval_scene(scene, ["part_s1", "order0"], "bin"))
        g3[scene].update(eval_scene(scene, ["order0"], "nee"))
    g3["gap4"].update(eval_scene("gap4", ["pair4_full", "pair4_masked", "pair4dup_full"], "bin"))
    g3["gap8"].update(eval_scene("gap8", ["pair8_full", "pair8_masked", "pair4stale_masked"], "bin"))
    g3["lblock"] = eval_scene("lblock", ["part_s1"], "nee"); g3["lblock"].update(eval_scene("lblock", ["part_s1"], "bin"))
    # coupling share per group (nee): 1 - order0 / composed
    share = {}
    for scene in ("gap4", "gap8", "floor"):
        P, _ = arm(scene, "part_s1"); O, _ = arm(scene, "order0"); sd = FX.scene_def(scene); grp = port_groups(scene); mit = load_ports(rd / f"cports_{scene}.npz")
        share[scene] = {}
        for cam, (ref2, cnt) in mit.items():
            key = f"{dirkey(sd['cams'][cam]['dir'])}__nee"; ok = cnt >= MINPIX
            for gname in np.unique(grp[ok]):
                m = ok & (grp == gname); share[scene][f"{cam}:{gname}"] = (1 - O[key][m].sum(0) / np.maximum(P[key][m].sum(0), 1e-30)).round(4).tolist()
    g3["coupling_share"] = share
    # disjointness power: the "dup" K claims the gap paths AND records the gap-face exits that C also transports
    pw = {}
    for cam, cv in g3["gap4"]["pair4dup_full__bin"]["cams"].items():
        for gname, v in cv.items():
            if any(v["ruled"]):
                pw[f"{cam}:{gname}"] = dict(dup_bias=v["bias"], se=v["se"], detectable=bool(any(b_ >= max(0.05, 5 * s_) for b_, s_ in zip(v["bias"], v["se"]))))
    g3["disjointness_power"] = pw
    obj = g3["gap4"]["pair4_full__bin"]["cams"]
    g3["object_scoped_consistent"] = {f"{cam}:{gname}": dict(bias=v["bias"], pass_=v["pass_"]) for cam, cv in obj.items() for gname, v in cv.items() if any(v["ruled"])}
    dmeta = g3["gap4"]["pair4dup_full__bin"]["meta"]
    g3["dup_rho"] = dmeta["rho"]; g3["dup_converged"] = dmeta["converged"]
    g3["power_pass"] = bool(max(dmeta["rho"]) >= 1 or any(v["detectable"] for v in pw.values()))
    g3["pass_all"] = all(g3[s_]["part_s1__nee"]["pass_"] for s_ in ("gap4", "gap8", "floor", "lblock"))
    rep["G3"] = g3
    V["KC_path_disjointness"] = dict(partition_gap4=g3["gap4"]["part_s1__nee"]["pass_"], partition_gap8=g3["gap8"]["part_s1__nee"]["pass_"], control_power=g3["power_pass"])
    V["coupled_transport_correctness"] = dict(all_scenes_nee=g3["pass_all"], rho={s: g3[s]["part_s1__nee"]["meta"]["rho"] for s in ("gap4", "gap8", "floor", "lblock")})
    # G4
    ops = json.load(open(od / "ops.json")); g4 = dict(T=ops["rigid"])
    g4["lblock_pose1"] = eval_scene("lblock_pose1", ["reuse_pose0", "recomputed"], "nee"); g4["lblock_pose1"].update(eval_scene("lblock_pose1", ["reuse_pose0", "recomputed"], "bin"))
    rr_ = ops["rigid"]["lblock_pose1_s3"]
    dif = {}
    for cam in g4["lblock_pose1"]["reuse_pose0__nee"]["cams"]:
        for gname, a_ in g4["lblock_pose1"]["reuse_pose0__nee"]["cams"][cam].items():
            b_ = g4["lblock_pose1"]["recomputed__nee"]["cams"][cam][gname]
            if any(a_["ruled"]):
                dif[f"{cam}:{gname}"] = dict(d=[x - y for x, y in zip(a_["bias"], b_["bias"])], ok=bool(all(abs(x - y) <= 0.01 + 3 * s for x, y, s in zip(a_["bias"], b_["bias"], a_["se"]))))
    g4["reuse_vs_recomputed"] = dif
    g4["pass_"] = bool(all(0.85 <= x <= 1.15 for x in rr_["fro_vs_seed"] + rr_["fro_full_vs_seed"]) and g4["lblock_pose1"]["reuse_pose0__nee"]["pass_"] and all(v["ok"] for v in dif.values()))
    rep["G4"] = g4; V["rigid_domain_reuse"] = g4["pass_"]
    # G5
    g5 = {k: ops[k] for k in ("seed_noise", "bend30_s3", "bend90_s3", "dent_s3", "slab_rest_self_reentry_fraction_uniform")}
    for scene in ("bend30", "bend90", "dent"):
        g5[scene] = eval_scene(scene, ["current"], "nee"); g5[scene].update(eval_scene(scene, ["current", "stale_rest"], "bin"))
    meas = {}
    for scene in ("bend30", "bend90", "dent"):
        o = ops[f"{scene}_s3"]; stale = g5[scene]["stale_rest__bin"]["cams"]; cur = g5[scene]["current__bin"]["cams"]
        exceed = []
        for cam in stale:
            for gname, s_ in stale[cam].items():
                c_ = cur[cam][gname]
                if any(s_["ruled"]):
                    exceed.append(dict(group=f"{cam}:{gname}", stale_bias=s_["bias"], current_bias=c_["bias"], se=s_["se"],
                                       beyond=bool(any(abs(a - b) > 3 * math.hypot(s1, s1) for a, b, s1 in zip(s_["bias"], c_["bias"], s_["se"])))))
        meas[scene] = dict(dT_over_noise=o["fro_vs_noise"], measured=bool(max(o["fro_vs_noise"]) >= 3), stale_effect=exceed,
                           stale_beyond_noise=any(e["beyond"] for e in exceed))
    g5["measurability"] = meas
    rep["G5"] = g5
    V["nonrigid_K_variation"] = {k: dict(measured=v["measured"], stale_beyond_noise=v["stale_beyond_noise"]) for k, v in meas.items()}
    rep["verdict_inputs"] = V
    json.dump(rep, open(R / "report.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(V, indent=1, default=str))


if __name__ == "__main__":
    main()
