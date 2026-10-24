"""Metrics and the declared descriptive classification of protocol 8dna_dyn_v1 (worklog 43). CPU only (numpy).

    python dyn_eval.py            # reads $EDNA_OUT/{design,refs,neural,static,ladder,cost}, writes $EDNA_OUT/eval/eval.json
"""

from __future__ import annotations

import math

import numpy as np

import dynlib as L
from dyn_stages import STATES, scenes_of, tag

P = L.PROTOCOL
TAU = 0.005
LUM = np.array([0.2126, 0.7152, 0.0722])


def disp(x):
    return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)


def box5(img):
    k = 5
    p = np.pad(img, ((k // 2, k // 2), (k // 2, k // 2), (0, 0)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0), (0, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def erode(m):
    """3x3 erosion: a pixel stays only if all 8 neighbours have the same class (box-filter footprint is pure)."""
    p = np.pad(m, 1, constant_values=False)
    out = np.ones_like(m)
    h, w = m.shape
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out &= p[dy:dy + h, dx:dx + w]
    return out


def mse(a, b, m=None):
    d = (disp(a) - disp(b)) ** 2
    return float(d[m].mean()) if m is not None else float(d.mean())


def psnr(v):
    return float("inf") if v == 0 else -10 * math.log10(v)


def hdr(a, b, m=None):
    m = np.ones(a.shape[:2], bool) if m is None else m
    x, r = a[m], b[m]
    return {"linear_relmse": float(np.mean((x - r) ** 2 / (r ** 2 + 1e-2))),
            "logL1": float(np.mean(np.abs(np.log1p(np.clip(x, 0, None)) - np.log1p(np.clip(r, 0, None))))),
            "display_mae": float(np.mean(np.abs(disp(x) - disp(r))))}


def _gauss(img, sigma=1.5, rad=5):
    x = np.arange(-rad, rad + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    p = np.pad(img, ((rad, rad), (rad, rad), (0, 0)), mode="reflect")
    t = sum(k[i] * p[i:i + img.shape[0], :, :] for i in range(len(k)))
    return sum(k[i] * t[:, i:i + img.shape[1], :] for i in range(len(k)))


def ssim(a, b):
    """SSIM on display images: Gaussian 11x11 window, sigma 1.5, K1 0.01, K2 0.03, mean over channels and pixels."""
    x, y = disp(a).astype(np.float64), disp(b).astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = _gauss(x), _gauss(y)
    sxx, syy, sxy = _gauss(x * x) - mx * mx, _gauss(y * y) - my * my, _gauss(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def finite(*vals):
    return all(v is None or (isinstance(v, float) and math.isfinite(v)) for v in vals)


def response(dn, dg):
    """Linear-RGB change tracking on paired receiver samples (n, 3)."""
    gg = float(np.sum(dg * dg))
    if gg == 0:
        return {"gain": None, "corr": None, "rel_err": None}
    a, b = dn.ravel(), dg.ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {"gain": float(np.sum(dn * dg) / gg), "corr": corr, "rel_err": float(np.sqrt(np.sum((dn - dg) ** 2) / gg))}


def classify(resp, gate_ok, R=None, inputs_finite=True):
    if not inputs_finite or not finite(resp.get("gain"), resp.get("corr"), R):
        return "INVALID_NONFINITE"
    if not gate_ok or resp["gain"] is None:
        return "NOISE_LIMITED"
    g = resp["gain"]
    c = resp["corr"] if resp["corr"] is not None else 0.0  # dN without variance (e.g. fully stale) has no correlation
    if g < 0.4 or g > 2.0 or c < 0.5 or (R is not None and R < 0.4):
        return "NOT_TRACKED"
    if 0.8 <= g <= 1.25 and c >= 0.8 and (R is None or R >= 0.8):
        return "TRACKED"
    return "PARTIAL"


class Exp:
    def __init__(self, exp):
        self.exp = exp
        cl = np.load(L.OUT / "design" / "classes.npz")
        self.cls = {k.split("/", 1)[1]: cl[k] for k in cl.files if k.startswith(exp + "/")}
        self.A, self.B, self.G, self.N, self.Nr = {}, {}, {}, {}, {}
        self.dA, self.dB = {}, {}
        for state, variant in scenes_of(exp):
            t = tag(state, variant)
            self.A[t] = np.load(L.OUT / "refs" / exp / f"{t}_A.npy")
            self.B[t] = np.load(L.OUT / "refs" / exp / f"{t}_B.npy")
            self.G[t] = 0.5 * (self.A[t] + self.B[t])
            self.N[t] = np.load(L.OUT / "neural" / exp / f"{t}_main.npy")
            self.Nr[t] = np.load(L.OUT / "neural" / exp / f"{t}_repeat.npy")
            f = L.OUT / "refs" / exp / f"{t}_direct_A.npy"
            if f.exists():
                self.dA[t] = np.load(f)
                self.dB[t] = np.load(L.OUT / "refs" / exp / f"{t}_direct_B.npy")

    def asset(self, t, lab=1):
        return erode(self.cls[t] == lab)

    def receivers(self, *tags, lab=1):
        m = np.ones(next(iter(self.cls.values())).shape, bool)
        for t in tags:
            m &= self.asset(t, lab)
        return m

    def finite_on(self, m, *tags):
        return all(np.isfinite(arr[t][m]).all() for t in tags for arr in (self.A, self.B, self.N, self.Nr))

    def static(self, t):
        G, N = self.G[t], self.N[t]
        out = {"full": {"psnr": psnr(mse(N, G)), "ssim": ssim(N, G), **hdr(N, G), "neural_repeat_psnr": psnr(mse(N, self.Nr[t])),
                        "ref_A_vs_B_psnr": psnr(mse(self.A[t], self.B[t])),
                        "nonfinite": {k: int((~np.isfinite(a[t])).sum()) for k, a in (("A", self.A), ("B", self.B), ("N", self.N), ("Nr", self.Nr))}}}
        for lab, name in ((1, "asset_pixels"), (2, "second_asset_pixels"), (3, "box_pixels"), (4, "support_pixels")):
            mm = self.asset(t, lab)
            if mm.sum() > 0:
                out[name] = {"n": int(mm.sum()), "psnr": psnr(mse(N, G, mm)), **hdr(N, G, mm),
                             "neural_repeat_psnr": psnr(mse(N, self.Nr[t], mm)), "ref_A_vs_B_psnr": psnr(mse(self.A[t], self.B[t], mm))}
        return out

    def change(self, t0, t1, rec, gref=None, nref=None, Aref=None, Bref=None):
        """dG vs dN on receivers; reference noise from paired A-B differences, neural noise from main vs repeat."""
        gref, nref = gref or self.G, nref or self.N
        Aref, Bref = Aref or self.A, Bref or self.B
        dg = (gref[t1] - gref[t0])[rec]
        dn = (nref[t1] - nref[t0])[rec]
        eab = ((Aref[t1] - Aref[t0]) - (Bref[t1] - Bref[t0]))[rec]  # paired seeds: noise of dG = rms(eab)/2
        gnoise = rms(eab) / 2
        lum_noise = rms(eab @ LUM) / 2
        dn_rep = (self.Nr[t1] - self.Nr[t0])[rec]
        nnoise = rms(dn - dn_rep) / math.sqrt(2)
        r = response(dn, dg)
        gate = rms(dg) > 3 * gnoise and rms(dg) > 3 * nnoise
        lum_g, lum_n = dg @ LUM, dn @ LUM
        strong = np.abs(lum_g) > 3 * lum_noise
        r.update({"rms_dG": rms(dg), "rms_dN": rms(dn), "ref_noise_rms": gnoise, "ref_noise_lum": lum_noise,
                  "neural_noise_rms": nnoise, "signal_gate": bool(gate), "pixels": int(rec.sum()),
                  "sign_agreement": float(np.mean(np.sign(lum_g[strong]) == np.sign(lum_n[strong]))) if strong.any() else None,
                  "strong_pixels": int(strong.sum()), "inputs_finite": self.finite_on(rec, t0, t1)})
        return r

    def changed_region(self, t0, t1, rec):
        d = np.abs(box5(disp(self.G[t1])) - box5(disp(self.G[t0])))
        e = box5((disp(self.A[t1]) - disp(self.A[t0])) - (disp(self.B[t1]) - disp(self.B[t0])))
        s = float(np.std(e[rec])) / 2 if rec.any() else 0.0
        thr = max(2 * TAU, 4 * s)
        return rec & (d.max(-1) > thr), thr, s

    def recovery(self, t0, t1, region):
        if region.sum() == 0:
            return {"pixels": 0, "R": None, "F": None}
        e_stale = mse(self.N[t0], self.G[t1], region)
        e_upd = mse(self.N[t1], self.G[t1], region)
        e_static = mse(self.N[t0], self.G[t0], region)
        den = e_stale - e_static
        return {"pixels": int(region.sum()), "E_stale": e_stale, "E_upd": e_upd, "E_static": e_static,
                "R": None if den <= 0 else float((e_stale - e_upd) / den), "F": float(e_stale / e_upd) if e_upd > 0 else None}


def eval_box(exp: str) -> dict:
    E = Exp(exp)
    res = {"static": {t: E.static(t) for t in E.G}}
    t0 = "T0_red"
    for st in STATES[exp][1:]:
        t1 = f"{st}_red"
        rec = E.receivers(t0, t1, "absent")
        ch = E.change(t0, t1, rec)
        region, thr, sig = E.changed_region(t0, t1, rec)
        rv = E.recovery(t0, t1, region)
        rv.update({"threshold": thr, "sigma": sig})
        ch["class"] = classify(ch, ch["signal_gate"], rv["R"], ch["inputs_finite"])
        fam = {}
        recx = E.receivers(t1, "absent")
        occ = E.change("absent", f"{st}_black", recx)
        occ["class"] = classify(occ, occ["signal_gate"], inputs_finite=occ["inputs_finite"])
        refl = E.change(f"{st}_black", t1, recx)
        refl["class"] = classify(refl, refl["signal_gate"], inputs_finite=refl["inputs_finite"])
        fam["occlusion"], fam["reflection"] = occ, refl
        if E.dA:  # E2: split the reference occlusion change into direct (max_depth 2) and indirect
            Gd = {k: 0.5 * (E.dA[k] + E.dB[k]) for k in E.dA}
            tb = f"{st}_black"
            dgo = (E.G[tb] - E.G["absent"])[recx]
            dgd = (Gd[tb] - Gd["absent"])[recx]
            dno = (E.N[tb] - E.N["absent"])[recx]
            gg = float(np.sum(dgo * dgo))
            share_d = float(np.sum(dgd * dgo) / gg) if gg > 0 else None
            ind = response(dno - dgd, dgo - dgd)
            dnoise = rms(((E.dA[tb] - E.dA["absent"]) - (E.dB[tb] - E.dB["absent"]))[recx]) / 2
            ind_gate = rms(dgo - dgd) > 3 * math.hypot(occ["ref_noise_rms"], dnoise) and rms(dgo - dgd) > 3 * occ["neural_noise_rms"]
            ind.update({"rms_indirect_dG": rms(dgo - dgd), "direct_ref_noise_rms": dnoise, "signal_gate": bool(ind_gate),
                        "note": "assumes 8DNA's direct term equals the reference direct term (both use the real BSDF and current visibility)"})
            ind["class"] = classify(ind, ind_gate, inputs_finite=occ["inputs_finite"])
            fam["occlusion_direct_share"] = share_d
            fam["occlusion_indirect"] = ind
        sup = E.receivers(t0, t1, "absent", lab=4)
        res[f"T0->{st}"] = {"asset_receivers": ch, "recovery": rv, "families": fam,
                            "support_receivers": E.change(t0, t1, sup) if sup.sum() > 100 else None}
    return res


def eval_e3() -> dict:
    E = Exp("E3")
    res = {"static": {t: E.static(t) for t in E.G}}
    rec = E.receivers("T0_red", "T1_red")
    ch = E.change("T0_red", "T1_red", rec)
    region, thr, sig = E.changed_region("T0_red", "T1_red", rec)
    rv = E.recovery("T0_red", "T1_red", region)
    rv.update({"threshold": thr, "sigma": sig})
    ch["class"] = classify(ch, ch["signal_gate"], rv["R"], ch["inputs_finite"])
    res["T0->T1"] = {"seal_receivers": ch, "recovery": rv}
    return res


def eval_static() -> dict:
    out = {}
    pc = {"seal_scene2": 26.61, "teaset_get_scene": 29.20}  # WL22 wavefront rows on the RTX 5080 (own references)
    runs = L.read_json(L.OUT / "static" / "static_runs.json") if (L.OUT / "static" / "static_runs.json").exists() else {}
    for key in ("seal_scene2", "teaset_get_scene"):
        d = L.OUT / "static" / key
        if not (d / "ref_1.npy").exists():
            continue
        n0, n1 = np.load(d / "neural_0.npy"), np.load(d / "neural_1.npy")
        a, b = np.load(d / "ref_0.npy"), np.load(d / "ref_1.npy")
        g = 0.5 * (a + b)
        r = runs.get(key, {})
        out[key] = {"psnr_vs_ref": psnr(mse(n0, g)), "ssim_vs_ref": ssim(n0, g), **hdr(n0, g),
                    "neural_seed_repeat_psnr": psnr(mse(n0, n1)), "ref_A_vs_B_psnr": psnr(mse(a, b)),
                    "pc_rtx5080_psnr_vs_own_ref": pc[key], "params_unchanged": r.get("params_before") == r.get("params_after"),
                    "reference_integrator": r.get("reference_integrator"), "nonfinite": r.get("nonfinite"),
                    "seconds": {k: v for k, v in r.items() if k.endswith("seconds")},
                    "note": "each machine against its own references; not pooled"}
    return out


def _interp(xs, ys, x):
    """log-log interpolation; None outside the sampled range."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    if not (lx.min() <= math.log(x) <= lx.max()):
        return None
    o = np.argsort(lx)
    return float(np.exp(np.interp(math.log(x), lx[o], ly[o])))


def eval_ladder() -> dict:
    out = {}
    tm = L.read_json(L.OUT / "ladder" / "ladder_timing.json")
    for exp in ("E1", "E2"):
        if exp not in tm:
            continue
        E = Exp(exp)
        t1 = "T1_out_red"
        g, a, b = E.G[t1], E.A[t1], E.B[t1]
        rec = E.receivers("T0_red", t1, "absent")
        full = np.ones(rec.shape, bool)
        metrics = {"mse_full": (lambda x, y: mse(x, y), full), "mse_receivers": (lambda x, y: mse(x, y, rec), rec),
                   "relmse_full": (lambda x, y: hdr(x, y, full)["linear_relmse"], full),
                   "relmse_receivers": (lambda x, y: hdr(x, y, rec)["linear_relmse"], rec)}
        floors = {k: 0.25 * f(a, b) for k, (f, _) in metrics.items()}  # error of G itself
        rows = {}
        for k, t in tm[exp].items():
            img = np.load(L.OUT / "ladder" / exp / f"{k}.npy")
            rows[k] = {"seconds": t, **{m: f(img, g) for m, (f, _) in metrics.items()}}
        summary = {}
        nk = [f"neural_{s}" for s in (16, 64, 256, 1024)]
        rk = [f"ref_{s}" for s in (16, 64, 256, 1024, 4096)]
        for m in metrics:
            f0 = floors[m]

            def corr(e):
                return max(e - f0, 1e-3 * f0) if f0 > 0 else e

            nt = np.array([rows[k]["seconds"] for k in nk])
            ne = np.array([corr(rows[k][m]) for k in nk])
            rt = np.array([rows[k]["seconds"] for k in rk])
            re_ = np.array([corr(rows[k][m]) for k in rk])
            e8, t8 = float(ne[-1]), float(nt[-1])
            t_pt, how = _interp(re_, rt, e8), "interpolated"
            if t_pt is None and e8 < re_.min():
                t_pt, how = float(rt[-1] * re_[-1] / e8), "extrapolated (MSE ~ 1/time beyond 4096 spp)"
            elif t_pt is None:
                how = "neural error above every PT rung: PT at 16 spp is already better"
            e_pt = _interp(rt, re_, t8)
            summary[m] = {"floor_of_G": f0, "neural_1024_error_floor_corrected": e8, "neural_1024_error_raw": rows[nk[-1]][m],
                          "neural_1024_seconds": t8, "pt_4096_error_floor_corrected": float(re_[-1]), "pt_4096_error_raw": rows[rk[-1]][m],
                          "pt_seconds_to_same_error": t_pt, "pt_seconds_method": how,
                          "equal_error_time_ratio_pt_over_neural": None if t_pt is None else t_pt / t8,
                          "pt_error_at_equal_time": e_pt, "equal_time_error_ratio_pt_over_neural": None if e_pt is None else e_pt / e8}
        out[exp] = {"rows": rows, "summary": summary, "receiver_pixels": int(rec.sum())}
    return out


def eval_cost() -> dict:
    out = {}
    nr = L.read_json(L.OUT / "neural" / "neural_runs.json")
    cost = L.read_json(L.OUT / "cost" / "cost.json") if (L.OUT / "cost" / "cost.json").exists() else {}
    for exp in ("E1", "E2", "E3"):
        if exp not in nr:
            continue
        res = L.R(L.EXPERIMENTS[exp]["res"])
        rows = {}
        for k, v in nr[exp].items():
            if not isinstance(v, dict) or "counters_4spp" not in v:
                continue
            c = v["counters_4spp"]
            per = {kk: vv / 4.0 / (res * res) for kk, vv in c.items() if "@" not in kk and not kk.endswith("unpatchable")}
            rows[k] = {"per_pixel_per_spp": per, "main_seconds": v.get("main_seconds"), "peak_torch_bytes": v.get("peak_torch_bytes"),
                       "peak_process_mib": v.get("peak_process_mib"), "nonfinite": v.get("nonfinite")}
        out[exp] = {"scenes": rows, "params_unchanged": nr[exp].get("params_unchanged"), "cost": cost.get(exp)}
    return out


def main() -> None:
    ev = {"protocol": P["protocol"], "dry_run": L.DRY, "out": str(L.OUT),
          "E1": eval_box("E1"), "E2": eval_box("E2"), "E3": eval_e3(), "B_static": eval_static(),
          "ladder": eval_ladder() if (L.OUT / "ladder" / "ladder_timing.json").exists() else None, "cost": eval_cost()}
    L.write_json(L.OUT / "eval" / "eval.json", ev)
    for exp in ("E1", "E2"):
        for k, v in ev[exp].items():
            if k.startswith("T0->"):
                a = v["asset_receivers"]
                print(exp, k, a["class"], "gain", a["gain"], "corr", a["corr"], "R", v["recovery"]["R"],
                      "| occ", v["families"]["occlusion"]["class"], "refl", v["families"]["reflection"]["class"])
    print("E3", ev["E3"]["T0->T1"]["seal_receivers"]["class"])


if __name__ == "__main__":
    main()
