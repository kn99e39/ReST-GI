"""WL101I qualification contract v2: gate mathematics (numpy/scipy only, no renderer).

Every table quantity is luminance. A "cell" is a slot (pid*64+bin; tables U, V, G) or a (slot, s)
spatial subcell (table S_G).
"""
import math

import numpy as np
from scipy.stats import chi2, norm, poisson

LUMA = np.array([0.2126, 0.7152, 0.0722])
THETA_ABS = 7.0e-4      # 5% of the WL101F near-contact band 0.01403 (image bias <= max slot bias, throughput <= 1)
THETA_REL = 2.5e-4      # 10% of the WL101G additive relative-RMSE margin 0.0025 (relative bias propagates <= max)
T_FLOOR = 1e-3          # luminance floor of the relative criterion (as in WL101H)
TIE_BAND = 6.4e-5       # address tie band: 4 coordinates x 2 sides x 8e-6


def theta(T):
    return np.minimum(THETA_ABS, THETA_REL * np.maximum(np.abs(T), T_FLOOR))


def z_family(n_tests, alpha=0.05):
    return float(norm.isf(alpha / (2 * max(n_tests, 1))))


def div(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


# ---------------------------------------------------------------- geometry of the registered exclusions
def singular_fractions(geom, h_min, R=0.4, center=(0.9, 0.4, 0.4)):
    """Continuous measure fraction of the candidate singular set per slot (all 64 bins of a patch share it)
    and per (slot, s). Floor: disk of radius r_s with sphere gap 2 h_min. Sphere: cap of height 2 h_min."""
    n_patch = len(geom["psid"])
    f_patch, f_sub = np.zeros(n_patch), np.zeros((n_patch, 4))
    r_s = math.sqrt(max(4 * R * h_min - 4 * h_min * h_min, 0.0))   # = R^2 - (R - 2 h_min)^2 without cancellation
    disk = math.pi * r_s * r_s
    cx, cy = center[0], center[1]
    floor = np.flatnonzero(geom["psid"] == 0)
    hit = None
    for pid in floor:
        po, e1, e2 = geom["po"][pid], geom["pe1"][pid], geom["pe2"][pid]
        a = (cx - po[0]) / e1[0] if abs(e1[0]) > 0 else (cy - po[1]) / e1[1]
        b = (cy - po[1]) / e2[1] if abs(e2[1]) > 0 else (cx - po[0]) / e2[0]
        if 0 <= a < 1 and 0 <= b < 1:
            hit = (pid, a, b, np.linalg.norm(e1), np.linalg.norm(e2))
    fp, a, b, l1, l2 = hit
    # The disk must lie inside one spatial subcell for this closed form.
    assert min(abs(a - 0.5) * l1, abs(b - 0.5) * l2, a * l1, (1 - a) * l1, b * l2, (1 - b) * l2) > r_s
    s = int(a >= .5) + 2 * int(b >= .5)
    f_patch[fp] = disk / geom["parea"][fp]
    f_sub[fp, s] = disk / (geom["parea"][fp] / 4)
    cap = 2 * math.pi * R * (2 * h_min)
    bottom = np.flatnonzero((geom["psid"] == 6) & (geom["psph_iz"] == 0))
    for pid in bottom:
        f_patch[pid] = (cap / len(bottom)) / geom["parea"][pid]
        f_sub[pid, 0] = f_sub[pid, 1] = (cap / len(bottom) / 2) / (geom["parea"][pid] / 4)
    return {"r_s_m": r_s, "disk_m2": disk, "cap_m2": cap, "floor_pid": int(fp), "floor_subcell": s,
            "f_slot": np.repeat(f_patch, 64), "f_sub": np.repeat(f_sub, 64, axis=0)}


def band_fraction_upper(geom, b_m):
    planar = geom["psid"] != 6
    with np.errstate(divide="ignore"):
        extra = np.where(planar, 2 * b_m * (1 / np.maximum(geom["l1"], 1e-30) + 1 / np.maximum(geom["l2"], 1e-30)), 0.0)
    return np.repeat(TIE_BAND + extra, 64)


def junction_fraction_upper(geom, b_m, box_b):
    """Analytic upper bound on the fraction of a slot's proposals inside the junction band: for every patch
    side lying on (within b_m of) a junction, b_m / (patch extent across that side). Box B footprint
    boundary on the floor and Box B's bottom row count as junctions."""
    out = np.zeros(len(geom["psid"]))
    c, h, yaw = np.array(box_b["c"]), np.array(box_b["h"]), box_b["yaw"]
    ax = np.array([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    def box_dist(pts):
        loc = (pts - c) @ ax.T
        return np.max(np.abs(loc) - h, axis=-1)
    planes = [(2, 0.0), (2, 3.0), (1, 2.0), (0, -2.0), (0, 2.0)]   # (axis, value): floor, ceiling, back, left, right
    t = np.linspace(0, 1, 65)
    for pid in np.flatnonzero(geom["psid"] != 6):
        po, e1, e2 = geom["po"][pid], geom["pe1"][pid], geom["pe2"][pid]
        sides = [(po + np.outer(t, e1), np.linalg.norm(e2)), (po + e2 + np.outer(t, e1), np.linalg.norm(e2)),
                 (po + np.outer(t, e2), np.linalg.norm(e1)), (po + e1 + np.outer(t, e2), np.linalg.norm(e1))]
        sid = geom["psid"][pid]
        for pts, across in sides:
            touch = False
            for axis, val in planes:
                d = np.abs(pts[:, axis] - val)
                if np.all(np.abs((po + 0.5 * e1 + 0.5 * e2)[axis] - val) < 1e-9):
                    continue                      # the patch's own plane
                touch |= bool(d.max() <= b_m)
            if sid in (0, 8):
                touch |= bool(np.min(np.abs(box_dist(pts))) <= b_m) and sid == 0
                touch |= bool(sid == 8 and np.max(np.abs(pts[:, 2])) <= b_m)
            if touch:
                out[pid] += b_m / across
    return np.repeat(out, 64)


def poisson_upper(k, conf=0.95):
    k = np.asarray(k, np.float64)
    return 0.5 * chi2.ppf(conf, 2 * (k + 1))


# ---------------------------------------------------------------- S3
def s3(packets, geom, h_min, b_m, proposals, alpha=0.05):
    sing = singular_fractions(geom, h_min)
    fb = band_fraction_upper(geom, b_m)
    out = {"singular_geometry": {k: v for k, v in sing.items() if not isinstance(v, np.ndarray)}, "replicas": []}
    for pk in packets:
        k_sub = pk["k_singular"].sum(2)                     # (slot, s)
        acc_sub = pk["n"].sum(2) + k_sub
        expected = acc_sub * sing["f_sub"]
        registered = sing["f_sub"] > 0
        obs_reg, exp_reg = float(k_sub[registered].sum()), float(expected[registered].sum())
        p_two = float(min(1.0, 2 * min(poisson.cdf(obs_reg, exp_reg), poisson.sf(obs_reg - 1, exp_reg)))) if exp_reg > 0 else (1.0 if obs_reg == 0 else 0.0)
        unregistered = int(k_sub[~registered].sum())
        excl = pk["slot_tie"] + pk["slot_boundary_miss"]
        exp_band = proposals * fb
        p_band = poisson.sf(excl - 1, exp_band)
        n_slots = int(np.count_nonzero(pk["n"].sum((1, 2)) + excl))
        band_fail = int(np.sum(p_band < alpha / max(n_slots, 1)))
        out["replicas"].append({
            "singular_observed_registered": obs_reg, "singular_expected_registered": exp_reg, "singular_two_sided_p": p_two,
            "singular_unregistered_cells_count": unregistered,
            "band_observed_total": float(excl.sum()), "band_upper_expected_total": float(exp_band.sum()),
            "band_slots_exceeding_upper_model": band_fail,
            "max_slot_observed_band_fraction": float((excl / proposals).max()),
            "pass": bool(unregistered == 0 and p_two >= alpha and band_fail == 0)})
    out["pass"] = all(r["pass"] for r in out["replicas"])
    out["f_sub_registered_max"] = float(sing["f_sub"].max())
    return out, sing, fb


# ---------------------------------------------------------------- S4
def _tables(pk, level):
    ax = (1, 2) if level == "slot" else 2
    n, v, g = pk["n"].sum(ax), pk["v"].sum(ax), pk["g"].sum(ax)
    L, vL, gL = (pk[k].sum(ax) @ LUMA for k in ("L", "vL", "gL"))
    return n, v, g, div(L, n), div(vL, v), div(gL, g)


def s4(pk, geom, sing, fb, fj, proposals, proxy, z, level="slot", semantics="candidate"):
    """Per-class bias accounting for U, V, G (level 'slot') or S_G (level 'sub').

    proxy: {"L": continuity radiance of the floor singular disk, "model": its model/statistical allowance}.
    """
    ax = (1, 2) if level == "slot" else 2
    n, v, g, U, V, G = _tables(pk, level)
    S = lambda k: pk[k].sum(ax)
    psid = np.repeat(geom["psid"], 64)
    if level == "sub":
        psid = np.repeat(psid[:, None], 4, 1)
    f_s = sing["f_slot"] if level == "slot" else sing["f_sub"]
    excl = (pk["slot_tie"] + pk["slot_boundary_miss"]) / proposals
    rep = (lambda x: x) if level == "slot" else (lambda x: np.repeat(np.asarray(x)[:, None], 4, 1))
    f_obs, f_up, f_jup = rep(excl), rep(fb), rep(fj)
    sh = {"U": (S("sh_n"), S("sh_L"), S("sh_LL")), "V": (S("sh_v"), S("sh_vL"), None), "G": (S("sh_g"), S("sh_gL"), S("sh_gLL"))}
    n_st = pk["n"]
    Lst = div(pk["L"] @ LUMA, n_st)
    tables = {"U": (U, n), "V": (V, v), "G": (G, g)} if level == "slot" else {"S_G": (G, g)}
    out = {}
    for name, (T, W) in tables.items():
        base = "G" if name == "S_G" else name
        active = W > 0
        Tb = T[..., None, None] if level == "slot" else T[..., None]
        dev = np.where(n_st >= 16, np.abs(Lst - Tb), 0).max(axis=ax)
        # singular (candidate exclusion, continuous measure)
        if semantics == "candidate":
            floor_c = (f_s > 0) & (psid == 0)
            Lex = np.where(floor_c, proxy["L"], T)
            model = np.where(floor_c, proxy["model"], dev)
            k = div(f_s, 1 - f_s)
            b_s = k * (Lex - T)
            se_s = np.zeros_like(T)
            B_s = k * (np.abs(Lex - T) + model)
        else:
            b_s = se_s = B_s = np.zeros_like(T)
        # band (tie + boundary miss exclusion, float lattice); radiance from the near-border shell
        cnt, sL, sLL = sh[base]
        Lsh = div(sL, cnt)
        if sLL is None:
            sLL = S("sh_LL") * div(cnt, S("sh_n"))
            var_sh = np.maximum(div(S("sh_LL"), S("sh_n")) - div(S("sh_L"), S("sh_n")) ** 2, 0)
        else:
            var_sh = np.maximum(div(sLL, cnt) - Lsh ** 2, 0)
        se_sh = np.sqrt(div(var_sh, np.maximum(cnt, 1)))
        b_b = np.where(cnt > 0, div(f_obs, 1 - f_obs) * (Lsh - T), 0.0)
        se_b = div(f_obs, 1 - f_obs) * se_sh
        B_b = np.where(cnt > 0, div(np.maximum(f_up, f_obs), 1 - np.maximum(f_up, f_obs)) * (np.abs(Lsh - T) + z * se_sh),
                       np.where(active & (np.maximum(f_up, f_obs) > 0), np.inf, 0.0))
        # junction (included, float lattice): analytic band measure or the observed count's upper bound
        kj = S("k_junction")
        N = {"U": n, "V": v, "G": v}[base]
        frac_j = np.maximum(f_jup, np.where(kj > 0, div(poisson_upper(kj), N), 0.0))
        Lj = np.where(kj > 0, div(S("j_L"), kj), T)
        se_j = np.sqrt(div(np.maximum(div(S("j_LL"), kj) - Lj ** 2, 0), np.maximum(kj, 1)))
        B_j = frac_j * (np.abs(Lj - T) + np.abs(T) + z * se_j)
        # silhouette (included; native visibility relative to its continuous limit), V and G only
        if base in ("V", "G"):
            w, wL = (v, S("vL") @ LUMA) if base == "V" else (g, S("gL") @ LUMA)
            pre = "dv" if base == "V" else "dg"
            Tm = {t: div(wL + S(f"{pre}L{t}"), w + S(f"{pre}{t}")) for t in ("10", "100")}
            d10, d100 = Tm["10"] - T, Tm["100"] - T
            b_sil = -d10 / 0.9
            # Sampling SE of the paired difference: per-sample contributions dw_i (L_i - T) are independent.
            if base == "V":
                ss = S("dvL2_10") - 2 * T * S("nfL10") + T * T * S("nf10")
            else:
                ss = S("dgL2_10") - 2 * T * S("dgdgL10") + T * T * S("dg2_10")
            se10 = div(np.sqrt(np.maximum(ss, 0)), w)
            se_sil = se10 / 0.9
            model_sil = np.abs((d100 - d10) - 0.1 * d10) / 0.9
            B_sil = np.abs(b_sil) + z * se_sil + model_sil
        else:
            b_sil = se_sil = B_sil = model_sil = np.zeros_like(T)
        b = b_s + b_b + b_sil
        se = np.sqrt(se_s ** 2 + se_b ** 2 + se_sil ** 2)
        B = B_s + B_b + B_j + B_sil
        th = theta(T)
        state = np.where(~active, 0, np.where(B <= th, 1, np.where(np.abs(b) - z * se > th, -1, 2)))
        out[name] = {"T": T, "W": W, "b": b, "se": se, "B": B, "theta": th, "state": state,
                     "parts_b": {"singular": b_s, "band": b_b, "silhouette": b_sil},
                     "parts_B": {"singular": B_s, "band": B_b, "junction": B_j, "silhouette": B_sil},
                     "silhouette_linearity_model": model_sil}
    return out


def summarize_s4(res):
    out = {}
    for name, r in res.items():
        st = r["state"]
        bad = (st == 2) | (st == -1)
        parts = np.stack([np.where(np.isfinite(v), v, 1e30) for v in r["parts_B"].values()])
        dom = np.argmax(parts, 0)
        names = list(r["parts_B"])
        out[name] = {"active": int(np.count_nonzero(st)), "pass": int(np.sum(st == 1)), "fail": int(np.sum(st == -1)),
                     "unresolved": int(np.sum(st == 2)),
                     "max_measured_abs_bias": float(np.max(np.abs(r["b"])[st != 0])) if np.any(st != 0) else 0.0,
                     "max_finite_bound": float(np.max(r["B"][np.isfinite(r["B"]) & (st != 0)])) if np.any(st != 0) else 0.0,
                     "dominant_bound_class_of_non_pass": {names[i]: int(np.sum(bad & (dom == i))) for i in range(len(names))},
                     "measured_bias_over_theta_by_class": {k: int(np.sum((st != 0) & (np.abs(v) > r["theta"]))) for k, v in r["parts_b"].items()}}
    return out


# ---------------------------------------------------------------- S5
def s5(new_packets, old_packets, z):
    rows = []
    for new, old in zip(new_packets, old_packets):
        slot_n = np.array_equal(new["n"].sum((1, 2)), old["n"].sum((1, 2)))
        slot_v = np.array_equal(new["v"].sum((1, 2)), old["v"].sum((1, 2)))
        rel = max(float(np.max(div(np.abs(new[k].sum((1, 2)) - old[k].sum((1, 2))), np.abs(old[k].sum((1, 2)))))) for k in ("c", "g", "L", "vL", "gL"))
        moved = int(np.abs(new["n"] - old["n"]).sum() // 2)
        moved_spatial = int(np.abs(new["n"].sum(2) - old["n"].sum(2)).sum() // 2)
        _, _, gN, _, _, SN = _tables(new, "sub")
        _, _, gO, _, _, SO = _tables(old, "sub")
        d = np.abs(SN - SO)
        th = theta(SO)
        exceed = int(np.sum((d > th) & (gO > 0)))
        rows.append({"slot_n_identical": bool(slot_n), "slot_v_identical": bool(slot_v), "slot_real_max_relative": rel,
                     "moved_samples": moved, "moved_across_spatial_subcells": moved_spatial,
                     "max_abs_S_G_change": float(d[gO > 0].max()), "S_G_cells_over_theta": exceed,
                     "pass": bool(slot_n and slot_v and rel <= 1e-6 and exceed == 0)})
    return {"rows": rows, "pass": all(r["pass"] for r in rows)}
