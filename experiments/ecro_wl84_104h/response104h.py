"""WL104H response-space component decomposition, hierarchical ablation, albedo-matched shape test, C-A decomposition, peak grid.

Usage: python response104h.py <out_dir> [--smoke]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.stats as sst

import io104h as IO
import comp104h as K

LUMA = np.array([0.2126, 0.7152, 0.0722])
REGIONS = ("all", "core", "shoulder", "tail")
EFFECTS = (*K.NAMES, "ATTEN_INTERACTION", "ETA_ATTEN_INTERACTION")
TERM_SHOW = ("O0", "O1", "O2", "SL", "SI0", "SI1", "SI2", "SI3")
CA_PARTS = ("O0", "O1", "O2", "atten", "L_content", "I0_content", "I1_content", "I2_content", "I3_content")


def mirror_radius(wi, wo):
    m = np.stack([-wi[..., 0], -wi[..., 1], wi[..., 2]], -1)
    return np.degrees(np.arccos(np.clip((m * wo).sum(-1), -1, 1)))


def effects(t):
    """Delta F for every intervention and the two interaction diagnostics; t = numpy term dict (arrays [..., 3])."""
    f0 = K.native(t)
    d = {name: spec["f"](t) - f0 for name, spec in K.INTERVENTIONS.items()}
    d["ATTEN_INTERACTION"] = d["ATTEN_OFF"] - d["ATTEN_IN_OFF"] - d["ATTEN_OUT_OFF"]
    d["ETA_ATTEN_INTERACTION"] = d["ETA_ATTEN_OFF"] - d["ETA_OFF"] - d["ATTEN_OFF"]
    return f0, d


def term_dict(raw):
    t = {k: raw[k] for k in ("O0", "O1", "O2", "L", "I0", "I1", "I2", "I3", "U0", "U1", "U2", "U3", "si", "so", "S")}
    return t


def ca_parts(ta, tc):
    """Exact decomposition of F_C - F_A (two-factor symmetric split for S * Y)."""
    parts = {f"O{k}": tc[f"O{k}"] - ta[f"O{k}"] for k in range(3)}
    ybar = 0.5 * ((ta["L"] + sum(ta[f"I{k}"] for k in range(4))) + (tc["L"] + sum(tc[f"I{k}"] for k in range(4))))
    parts["atten"] = (tc["S"] - ta["S"]) * ybar
    sbar = 0.5 * (ta["S"] + tc["S"])
    parts["L_content"] = sbar * (tc["L"] - ta["L"])
    for k in range(4):
        parts[f"I{k}_content"] = sbar * (tc[f"I{k}"] - ta[f"I{k}"])
    return parts


def block_sums(x, masks, blk):
    """x [P, L] -> [R, P, B] sums of x over lanes in region r and block b."""
    out = np.zeros((len(masks), x.shape[0], IO.BLOCKS))
    for r, m in enumerate(masks):
        v = np.where(m, x, 0.0)
        for b in range(IO.BLOCKS):
            out[r, :, b] = v[:, blk == b].sum(1)
    return out


def material_stats(m, ref, states):
    """All lane sums for material m (20 probes)."""
    sl = slice(m * 20, (m + 1) * 20)
    wi, wo = ref["wi"][sl], ref["wo"][sl]
    a, b = ref["a"][sl], ref["b"][sl]
    r_mean = 0.5 * (a + b)
    w = ref["use"][sl] / ref["q"][sl]
    A = (w[..., None] * r_mean).sum((1, 2))                           # el1 normaliser per probe
    W2 = w / A[:, None] ** 2
    W1 = w / A[:, None]
    rad = mirror_radius(wi, wo)
    masks = [np.ones_like(rad, bool), rad < 3, (rad >= 3) & (rad < 12), rad >= 12]
    blk = np.arange(wi.shape[1]) % IO.BLOCKS
    shp = wi.shape[:2]
    terms = {}
    for name, st in states.items():
        raw = K.terms_dr_numpy(st[m], wi.reshape(-1, 3), wo.reshape(-1, 3))
        terms[name] = {k: v.reshape(*shp, *v.shape[1:]) for k, v in raw.items()}
    t0 = term_dict(terms["R0"])
    f0, d = effects(t0)
    out = {}
    dot = lambda x, y: (x * y).sum(-1)
    ra, rb, r = a - f0, b - f0, r_mean - f0
    out["rr"] = block_sums(W2 * dot(ra, rb), masks, blk)
    out["nu"] = block_sums(W2 * dot(a - b, a - b) / 4, masks, blk)
    out["ff"] = block_sums(W2 * dot(f0, f0), masks, blk)
    out["ref_luma"] = block_sums(w * (r_mean @ LUMA), masks, blk)
    out["f_luma"] = block_sums(w * (f0 @ LUMA), masks, blk)
    out["n2"], out["cross"], out["n2m"], out["crossm"], out["shape2"], out["energy"] = ({} for _ in range(6))
    out["matched_ok"] = {}
    out["el1_half"] = {}
    for h, hv in (("a", a), ("b", b)):
        out["el1_half"].setdefault("R0", []).append((W1[..., None] * np.abs(f0 - hv)).sum((1, 2)))
    aR0 = (w[..., None] * f0).sum(1)                                      # [P, 3]
    for name, df in d.items():
        out["n2"][name] = block_sums(W2 * dot(df, df), masks, blk)
        out["cross"][name] = block_sums(W2 * dot(df, r), masks, blk)
        out["energy"][name] = block_sums(w * (df @ LUMA), masks, blk)
        if name in K.INTERVENTIONS:
            fx = f0 + df
            for h, hv in (("a", a), ("b", b)):
                out["el1_half"].setdefault(name, []).append((W1[..., None] * np.abs(fx - hv)).sum((1, 2)))
            aX = (w[..., None] * fx).sum(1)
            ok = np.all(aX >= 0.01 * aR0, axis=1) & np.all(aR0 > 0, axis=1)
            k = np.where(ok[:, None], aR0 / np.where(aX > 0, aX, 1.0), 1.0)
            dm = fx * k[:, None, :] - f0
            out["matched_ok"][name] = ok
            out["n2m"][name] = block_sums(W2 * dot(dm, dm), masks, blk)
            out["crossm"][name] = block_sums(W2 * dot(dm, r), masks, blk)
    names = list(d)
    G = np.zeros((len(REGIONS), 20, len(names), len(names)))
    for i, x in enumerate(names):
        for j, y in enumerate(names[i:], i):
            v = np.stack([(W2 * dot(d[x], d[y]) * mm).sum(1) for mm in masks])
            G[:, :, i, j] = G[:, :, j, i] = v
    out["gram"] = G
    out["gram_names"] = names
    # term contributions (luma, cosine-weighted directional albedo estimates)
    show = {"O0": t0["O0"], "O1": t0["O1"], "O2": t0["O2"], "SL": t0["S"] * t0["L"],
            **{f"SI{k}": t0["S"] * t0[f"I{k}"] for k in range(4)}}
    out["contrib"] = {k: block_sums(w * (v @ LUMA), masks, blk) for k, v in show.items()}
    # term Gram (shape collinearity of the base terms)
    tn = list(show)
    TG = np.zeros((len(REGIONS), 20, len(tn), len(tn)))
    for i, x in enumerate(tn):
        for j, y in enumerate(tn[i:], i):
            v = np.stack([(W2 * dot(show[x], show[y]) * mm).sum(1) for mm in masks])
            TG[:, :, i, j] = TG[:, :, j, i] = v
    out["term_gram"] = TG
    # C - A response change and its exact component decomposition
    ta, tc = term_dict(terms["A"]), term_dict(terms["C"])
    fa, fc = K.native(ta), K.native(tc)
    dca = fc - fa
    parts = ca_parts(ta, tc)
    closure = np.abs(sum(parts.values()) - dca).max() / max(np.abs(dca).max(), 1e-30)
    out["ca_closure"] = float(closure)
    out["ca_n2"] = block_sums(W2 * dot(dca, dca), masks, blk)
    out["ca_parts"] = {k: block_sums(W2 * dot(v, dca), masks, blk) for k, v in parts.items()}
    out["ca_proj"] = {k: block_sums(W2 * dot(v, dca), masks, blk) for k, v in d.items()}
    out["ca_energy"] = block_sums(w * (dca @ LUMA), masks, blk)
    # identities (lane level)
    idt = {}
    for name, tr in terms.items():
        peak = np.abs(tr["native"]).max()
        idt[name] = {"native_vs_native_order": float(np.abs(tr["native"] - tr["native_order_sum"]).max() / peak),
                     "native_vs_numpy_term_sum": float(np.abs(tr["native"] - K.native(term_dict(tr))).max() / peak)}
    out["identity"] = idt
    lanes = {"wi": wi.astype(np.float32), "wo": wo.astype(np.float32), "w": w.astype(np.float32),
             "a": a.astype(np.float32), "b": b.astype(np.float32), "F": f0.astype(np.float32),
             **{k: v.astype(np.float32) for k, v in show.items()}, "dCA": dca.astype(np.float32),
             **{f"D_{k}": d[k].astype(np.float32) for k in ("LAMBERT_OFF", "OUTER_OFF", "INNER_OFF", "ATTEN_OFF", "ETA_OFF")}}
    return out, lanes, peak_stats(m, ref, states)


def peak_stats(m, ref, states):
    sl = slice(m * 20, (m + 1) * 20)
    pwi, pwo = ref["peak_wi"][sl], ref["peak_wo"][sl]
    pa, pb = ref["peak_a"][sl], ref["peak_b"][sl]
    raw = K.terms_dr_numpy(states["R0"][m], pwi.reshape(-1, 3), pwo.reshape(-1, 3))
    t = {k: v.reshape(*pwi.shape[:2], *v.shape[1:]) for k, v in term_dict(raw).items()}
    f0, d = effects(t)
    rl = (0.5 * (pa + pb)) @ LUMA
    ip = rl.argmax(1)
    idx = np.arange(20)
    ref_pk = rl[idx, ip]
    noise = np.abs((pa - pb) @ LUMA)[idx, ip] / 2 / ref_pk
    fl = f0 @ LUMA
    res = {"ref_peak": ref_pk.tolist(), "ref_peak_noise_rel": noise.tolist(), "peak_index": ip.tolist(),
           "R0_at_ref_peak_rel_error": ((fl[idx, ip] - ref_pk) / ref_pk).tolist(),
           "R0_peak_rel_error": ((fl.max(1) - ref_pk) / ref_pk).tolist(),
           "R0_mirror_rel_error": ((fl[:, 0] - rl[:, 0]) / rl[:, 0]).tolist()}
    show = {"O0": t["O0"], "O1": t["O1"], "O2": t["O2"], "SL": t["S"] * t["L"],
            **{f"SI{k}": t["S"] * t[f"I{k}"] for k in range(4)}}
    res["term_share_at_ref_peak"] = {k: ((v @ LUMA)[idx, ip] / fl[idx, ip]).tolist() for k, v in show.items()}
    res["intervention_rel_change_at_ref_peak"] = {k: ((v @ LUMA)[idx, ip] / ref_pk).tolist() for k, v in d.items()}
    return res


# ------------------------------------------------------------------ aggregation and labels
T15 = float(sst.t.ppf(0.975, IO.BLOCKS - 1))


def jk(full, loo):
    loo = np.asarray(loo, float)
    B = len(loo)
    se = float(np.sqrt((B - 1) / B * ((loo - loo.mean()) ** 2).sum()))
    return {"point": float(full), "se": se, "lo": float(full - T15 * se), "hi": float(full + T15 * se)}


def agg(x, sel_m, sel_p, r):
    """x [M, R, P, B] -> totals [B] for the selection."""
    return x[np.ix_(sel_m, [r], sel_p)][:, 0].sum((0, 1))


def loo_of(tot):
    return tot.sum() - tot


def stat_block(S, name, sel_m, sel_p, r, matched=False):
    n2 = agg(S["n2m" if matched else "n2"][name], sel_m, sel_p, r)
    cr = agg(S["crossm" if matched else "cross"][name], sel_m, sel_p, r)
    rr = agg(S["rr"], sel_m, sel_p, r)
    nu = agg(S["nu"], sel_m, sel_p, r)
    f = lambda a, c, q: (c / np.sqrt(max(a * q, 1e-300))) if q > 0 and a > 0 else np.nan
    e = lambda a, c, q: (2 * c - a) / q if q > 0 else np.nan
    full = (n2.sum(), cr.sum(), rr.sum())
    lo = [(n2.sum() - n2[b], cr.sum() - cr[b], rr.sum() - rr[b]) for b in range(IO.BLOCKS)]
    sc = IO.BLOCKS / (IO.BLOCKS - 1)
    res = {"rho": jk(f(*full), [f(*x) for x in lo]),
           "error_change": jk(e(*full), [e(*x) for x in lo]),
           "dmse_over_ref_energy": jk(full[0] - 2 * full[1], [sc * (x[0] - 2 * x[1]) for x in lo]),
           "rr": jk(full[2], [sc * x[2] for x in lo]),
           "n2": float(full[0]), "nu": float(nu.sum()),
           "change_over_noise": float(full[0] / max(nu.sum(), 1e-300)),
           "relative_change_l2": float(np.sqrt(full[0] / max(agg(S["ff"], sel_m, sel_p, r).sum(), 1e-300)))}
    return res


def label(st, per_material=None, need=None, matched_ok=True):
    if not matched_ok or st["rr"]["lo"] <= 0 or not np.isfinite(st["rho"]["point"]):
        return "EVIDENCE_UNQUALIFIED"
    det = st["change_over_noise"] >= 9 and (st["dmse_over_ref_energy"]["lo"] > 0 or st["dmse_over_ref_energy"]["hi"] < 0)
    if not det:
        return "NO_DETECTABLE_COMPONENT_EFFECT"
    rho = st["rho"]
    loc = rho["lo"] > 0.5 or rho["hi"] < -0.5
    if loc and per_material is not None:
        sign = np.sign(rho["point"])
        hits = sum(1 for x in per_material if (x["lo"] > 0.5 if sign > 0 else x["hi"] < -0.5))
        loc = hits >= need
    return "CONDITIONAL_ERROR_LOCALIZATION" if loc else "DETECTABLE_COMPONENT_EFFECT"


def family(name):
    for fam, members in IO.P["interventions"]["families_for_confounding"].items():
        if name in members:
            return fam
    return None


def summarize(S, rows):
    bands = IO.P["response_phase"]["strata"]["band"]
    cohorts = {"development": [i for i, r in enumerate(rows) if r["cohort"] == "development"],
               "held_out": [i for i, r in enumerate(rows) if r["cohort"] == "held_out"]}
    need = {"development": 2, "held_out": 4}
    names = S["gram_names"]
    out = {}
    for cohort, ms in cohorts.items():
        for band, ang in bands.items():
            sel_p = list(range(20)) if ang == "all" else [i for i, t in enumerate(IO.POLAR) if t in ang]
            for r, region in enumerate(REGIONS):
                key = f"{cohort}|{band}|{region}"
                G = S["gram"][np.ix_(ms, [r], sel_p)][:, 0].sum((0, 1))
                cos = G / np.sqrt(np.outer(np.diag(G), np.diag(G)).clip(1e-300))
                block = {"interventions": {}, "cos": cos.tolist(), "names": names}
                for name in EFFECTS:
                    st = stat_block(S, name, ms, sel_p, r)
                    per = [stat_block(S, name, [m], sel_p, r)["rho"] for m in ms]
                    st["per_material_rho"] = per
                    st["energy_change"] = float(agg(S["energy"][name], ms, sel_p, r).sum()
                                                / agg(S["ref_luma"], ms, sel_p, 0).sum())
                    st["ca_projection"] = float(agg(S["ca_proj"][name], ms, sel_p, r).sum()
                                                / max(agg(S["ca_n2"], ms, sel_p, r).sum(), 1e-300))
                    nca = agg(S["ca_n2"], ms, sel_p, r).sum()
                    st["ca_cos"] = float(agg(S["ca_proj"][name], ms, sel_p, r).sum() / np.sqrt(max(nca * st["n2"], 1e-300)))
                    if name in K.INTERVENTIONS:
                        st["label"] = label(st, per, need[cohort])
                        okm = bool(np.all(S["matched_ok"][name][np.ix_(ms, sel_p)]))
                        sm = stat_block(S, name, ms, sel_p, r, matched=True)
                        sm["label"] = label(sm, [stat_block(S, name, [m], sel_p, r, True)["rho"] for m in ms],
                                            need[cohort], okm)
                        sm["qualified_probes"] = int(S["matched_ok"][name][np.ix_(ms, sel_p)].sum())
                        sm["shape_fraction_l2"] = float(np.sqrt(sm["n2"] / max(st["n2"], 1e-300)))
                        st["matched"] = sm
                    block["interventions"][name] = st
                # confounding among different families
                for name in K.NAMES:
                    st = block["interventions"][name]
                    if st["label"] != "CONDITIONAL_ERROR_LOCALIZATION":
                        continue
                    i = names.index(name)
                    rivals = [y for j, y in enumerate(names) if y in K.NAMES and family(y) != family(name)
                              and abs(cos[i, j]) >= 0.9 and block["interventions"][y]["label"] in
                              ("CONDITIONAL_ERROR_LOCALIZATION", "CONFOUNDED_OR_NON_IDENTIFIABLE")]
                    if rivals:
                        st["label"] = "CONFOUNDED_OR_NON_IDENTIFIABLE"
                        st["confounded_with"] = rivals
                        for y in rivals:
                            block["interventions"][y]["label"] = "CONFOUNDED_OR_NON_IDENTIFIABLE"
                            block["interventions"][y].setdefault("confounded_with", []).append(name)
                # hierarchical shares within families
                hier = {}
                for grp, lobes in (("OUTER_OFF", ("O0_OFF", "O1_OFF", "O2_OFF")),
                                   ("INNER_OFF", ("I0_OFF", "I1_OFF", "I2_OFF", "I3_OFF"))):
                    gi = names.index(grp)
                    hier[grp] = {lb: float(G[names.index(lb), gi] / max(G[gi, gi], 1e-300)) for lb in lobes}
                block["hierarchical_share"] = hier
                # base-term contributions and collinearity
                TG = S["term_gram"][np.ix_(ms, [r], sel_p)][:, 0].sum((0, 1))
                block["term_cos"] = (TG / np.sqrt(np.outer(np.diag(TG), np.diag(TG)).clip(1e-300))).tolist()
                fl = agg(S["f_luma"], ms, sel_p, r).sum()
                block["term_luma_share"] = {k: float(agg(v, ms, sel_p, r).sum() / max(fl, 1e-300))
                                            for k, v in S["contrib"].items()}
                # residual resolution and C-A decomposition
                rr = agg(S["rr"], ms, sel_p, r)
                block["residual_energy"] = jk(rr.sum(), [IO.BLOCKS / (IO.BLOCKS - 1) * (rr.sum() - rr[b]) for b in range(IO.BLOCKS)])
                block["reference_noise_energy"] = float(agg(S["nu"], ms, sel_p, r).sum())
                nca = agg(S["ca_n2"], ms, sel_p, r)
                block["ca_shares"] = {}
                for k, v in S["ca_parts"].items():
                    x = agg(v, ms, sel_p, r)
                    block["ca_shares"][k] = jk(x.sum() / max(nca.sum(), 1e-300),
                                               [(x.sum() - x[b]) / max(nca.sum() - nca[b], 1e-300) for b in range(IO.BLOCKS)])
                block["ca_energy"] = float(agg(S["ca_energy"], ms, sel_p, r).sum() / agg(S["ref_luma"], ms, sel_p, 0).sum())
                out[key] = block
    return out


def stack_stats(per_mat):
    """List over materials of material_stats dicts -> arrays [M, R, P, B]."""
    S = {}
    first = per_mat[0]
    for k, v in first.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            S[k] = np.stack([pm[k] for pm in per_mat])
        elif isinstance(v, dict) and k not in ("identity", "el1_half", "matched_ok"):
            S[k] = {n: np.stack([pm[k][n] for pm in per_mat]) for n in v}
    S["matched_ok"] = {n: np.stack([pm["matched_ok"][n] for pm in per_mat]) for n in first["matched_ok"]}
    S["gram"] = np.stack([pm["gram"] for pm in per_mat])
    S["gram_names"] = first["gram_names"]
    S["term_gram"] = np.stack([pm["term_gram"] for pm in per_mat])
    return S


def per_probe_table(per_mat, rows):
    """Per material x probe: residual energy, noise, rho and error change of each intervention (all lanes), el1 changes per half."""
    tab = []
    for m, pm in enumerate(per_mat):
        for p in range(20):
            rr = pm["rr"][0, p].sum()
            nu = pm["nu"][0, p].sum()
            rec = {"material": rows[m]["tag"], "cohort": rows[m]["cohort"], "theta": float(IO.POLAR[p]),
                   "phi": float(IO.AZIMUTHS[p % 2]), "residual_energy": float(rr), "noise_energy": float(nu),
                   "el1_R0_halves": [float(x[p]) for x in pm["el1_half"]["R0"]], "interventions": {}}
            for name in K.NAMES:
                n2, cr = pm["n2"][name][0, p].sum(), pm["cross"][name][0, p].sum()
                rec["interventions"][name] = {
                    "rho": float(cr / np.sqrt(n2 * rr)) if rr > 0 and n2 > 0 else None,
                    "error_change": float((2 * cr - n2) / rr) if rr > 0 else None,
                    "change_over_noise": float(n2 / max(nu, 1e-300)),
                    "del1_halves": [float(pm["el1_half"][name][h][p] - pm["el1_half"]["R0"][h][p]) for h in range(2)],
                    "matched_ok": bool(pm["matched_ok"][name][p])}
            tab.append(rec)
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, compiled, seen = IO.inputs()
    ref = IO.reference()
    precision, ok = IO.reference_precision(ref, len(rows))
    if not ok:
        IO.write(args.out / "stop.json", {"stop": "REFERENCE_PRECISION", "precision": precision})
        raise IO.Stop("REFERENCE_PRECISION")
    mats = [0] if args.smoke else range(len(rows))
    per_mat, peaks, lanes = [], {}, {}
    for m in mats:
        st, ln, pk = material_stats(m, ref, compiled)
        worst = max(v for x in st["identity"].values() for v in x.values())
        if worst > IO.P["decomposition"]["identity_tolerances"]["drjit_native_vs_term_sum_relative_to_peak"] or st["ca_closure"] > 1e-9:
            IO.write(args.out / "stop.json", {"stop": "DECOMPOSITION_IDENTITY", "material": rows[m]["tag"],
                                              "identity": st["identity"], "ca_closure": st["ca_closure"]})
            raise IO.Stop("DECOMPOSITION_IDENTITY")
        per_mat.append(st)
        peaks[rows[m]["tag"]] = pk
        lanes[rows[m]["tag"]] = ln
        print("response material", rows[m]["tag"], round(time.monotonic() - t0, 1), flush=True)
    if args.smoke:
        IO.write(args.out / "smoke.json", {"identity": per_mat[0]["identity"], "ca_closure": per_mat[0]["ca_closure"],
                                           "seconds": time.monotonic() - t0})
        return
    S = stack_stats(per_mat)
    summary = summarize(S, rows)
    np.savez_compressed(args.out / "lanes.npz", **{f"{tag}|{k}": v for tag, d in lanes.items() for k, v in d.items()})
    IO.write(args.out / "response.json", {
        "commit": IO.commit(), "protocol_sha256": IO.sha(IO.HERE / "protocol.json"), "inputs_sha256": seen,
        "reference_precision": precision, "materials": rows, "theta": IO.POLAR.tolist(), "regions": REGIONS,
        "identity": {rows[m]["tag"]: pm["identity"] for m, pm in enumerate(per_mat)},
        "ca_closure": {rows[m]["tag"]: pm["ca_closure"] for m, pm in enumerate(per_mat)},
        "strata": summary, "per_probe": per_probe_table(per_mat, rows), "peak": peaks,
        "seconds": time.monotonic() - t0})
    print("response done", round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
