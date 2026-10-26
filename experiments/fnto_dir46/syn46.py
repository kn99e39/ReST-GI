"""WL46 class I synthetic directional contract (analytic geometry and BSDFs; NO learned K). The interface math is the
production one: fnto_model.ggx / DomainKernel.loc_response (K_loc), sender value 'iso' = production semantics (value at
the port normal: an emitter reads g(1); a glossy sender reads its isotropic collapse sum kappa / pi E), 'dir' = value
toward the receiver (-d), with the sender's own incident samples. Ground truth by dense numerical integration.

Cases (protocol fnto_dir46 SYN):
  S0 zero field; S1 constant sky (E = pi, glossy directional albedo); S2 equal irradiance, different angular
  distribution (uniform sky vs 3-degree disk at the mirror direction) on a GGX receiver (alpha 0.04 / 0.09);
  S3 directional emitter g_kappa(c) = (kappa + 2)/2 c^kappa seen at a grazing exit angle (kappa 0 / 2 / 8; 2^16 rays,
  isolates axis A); S4 one glossy reflection emitter -> glossy plane -> camera; S5 two glossy bounces emitter -> plane 1
  -> plane 2 -> camera; S6 superposition of two emitters.
Interface variants: iso, dir + 128 cosine samples, dir + 128 GGX importance samples (receiver / sender gathers);
estimates averaged over 32 seeds (S2 / S4 / S5).

    python syn46.py <out json>
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import fnto_model as M  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); F64 = torch.float64
Z = torch.tensor([0.0, 0.0, 1.0], dtype=F64, device=DEV)


def nrm(x):
    return x / x.norm(dim=-1, keepdim=True)


def frame(n):
    a = torch.tensor([1.0, 0, 0], dtype=F64, device=DEV) if abs(float(n[2])) > 0.9 else torch.tensor([0.0, 0, 1], dtype=F64, device=DEV)
    t = nrm(torch.cross(a, n, dim=0)); return t, torch.cross(n, t, dim=0)


def ggx_val(n, wi, wo, alpha):
    """Analytic GGX BRDF value (fnto_model.ggx semantics), float64, n [3] broadcast."""
    return M.ggx(n.expand_as(wi).float(), wi.float(), wo.float(), torch.full(wi.shape[:-1] + (1,), alpha, device=DEV))[..., 0].double()


def cos_samples(n, k, g):
    u1, u2 = torch.rand(k, generator=g, dtype=F64, device=DEV), torch.rand(k, generator=g, dtype=F64, device=DEV)
    r, ph = torch.sqrt(u1), 2 * math.pi * u2; t, b = frame(n)
    d = r[:, None] * torch.cos(ph)[:, None] * t + r[:, None] * torch.sin(ph)[:, None] * b + torch.sqrt(1 - u1)[:, None] * n
    return d, torch.full((k,), math.pi / k, dtype=F64, device=DEV)            # weight w: sum f L w  (cos / pdf / k)


def ggx_is_samples(n, wo, alpha, k, g):
    """Sample half vectors from D(h) cos(h) (GGX), reflect wo; weight = cos(wi) / pdf(wi) / k."""
    u1, u2 = torch.rand(k, generator=g, dtype=F64, device=DEV), torch.rand(k, generator=g, dtype=F64, device=DEV)
    a2 = alpha * alpha; ct = torch.sqrt((1 - u1) / (1 + (a2 - 1) * u1)); st = torch.sqrt(1 - ct * ct); ph = 2 * math.pi * u2
    t, b = frame(n); h = st[:, None] * torch.cos(ph)[:, None] * t + st[:, None] * torch.sin(ph)[:, None] * b + ct[:, None] * n
    wi = 2 * (h @ wo)[:, None] * h - wo
    D = a2 / (math.pi * (ct * ct * (a2 - 1) + 1) ** 2); pdf_h = D * ct; pdf = pdf_h / (4 * (h @ wo).abs().clamp(min=1e-12))
    ci = wi @ n; w = torch.where(ci > 0, ci / pdf.clamp(min=1e-30) / k, torch.zeros_like(ci))
    return wi, w


class Rect:
    def __init__(self, c, u, v, emit=None, alpha=None, kap=0.0):
        self.c = torch.tensor(c, dtype=F64, device=DEV); self.u = torch.tensor(u, dtype=F64, device=DEV); self.v = torch.tensor(v, dtype=F64, device=DEV)
        self.n = nrm(torch.cross(self.u, self.v, dim=0)); self.area = float(4 * torch.cross(self.u, self.v, dim=0).norm())
        self.emit = emit; self.alpha = alpha; self.kap = kap

    def hit(self, o, d):
        dn = d @ self.n; t = ((self.c - o) @ self.n) / torch.where(dn.abs() > 1e-12, dn, torch.full_like(dn, 1e-12))
        p = o + t[..., None] * d; r = p - self.c
        a = (r @ self.u) / (self.u @ self.u); b = (r @ self.v) / (self.v @ self.v)
        ok = (t > 1e-9) & (a.abs() <= 1) & (b.abs() <= 1) & (dn < 0)                       # front side only
        return ok, p, t

    def sample(self, k, g):
        a = torch.rand(k, generator=g, dtype=F64, device=DEV) * 2 - 1; b = torch.rand(k, generator=g, dtype=F64, device=DEV) * 2 - 1
        return self.c + a[:, None] * self.u + b[:, None] * self.v


def emitter_L(E, w_out, kappa):
    c = (w_out @ E.n).clamp(min=0); return (kappa + 2) / 2 * c ** kappa


def irradiance_from_rect_dense(x, n_x, E, kappa, k=1 << 20, seed=0, iso=False):
    g = torch.Generator(device=DEV); g.manual_seed(seed); y = E.sample(k, g); d = y - x; r2 = (d * d).sum(-1); d = d / r2.sqrt()[:, None]
    cr = (d @ n_x).clamp(min=0); ce = (-d @ E.n).clamp(min=0)
    L = (kappa + 2) / 2 * (torch.ones_like(ce) if iso else ce ** kappa)
    return float((L * cr * ce / r2).mean() * E.area)


def main(out):
    rec = {"device": str(DEV)}; g = torch.Generator(device=DEV)
    # S0 / S1 -------------------------------------------------------------------------------------------------------
    n = Z; wo = nrm(torch.tensor([0.4, 0.0, 0.9], dtype=F64, device=DEV))
    dense = []
    for alpha in (0.04, 0.09):
        g.manual_seed(1); wi, w = ggx_is_samples(n, wo, alpha, 1 << 22, g)
        alb = float((ggx_val(n, wi, wo[None].expand_as(wi), alpha) * w).sum()); dense.append(alb)
    rec["S0_zero"] = 0.0 * dense[0]
    rec["S1_constant_sky"] = {"E": math.pi, "glossy_albedo_alpha0.04_truth": dense[0], "glossy_albedo_alpha0.09_truth": dense[1],
                              "iso_E_collapse_kappa_over_pi_E": 1.0, "note": "iso collapse returns kappa E / pi = 1 for unit kappa; truth is the directional albedo"}
    # S2 -------------------------------------------------------------------------------------------------------------
    s2 = {}
    for alpha in (0.04, 0.09):
        wi0 = torch.tensor([-wo[0], -wo[1], wo[2]], dtype=F64, device=DEV); half = math.radians(3.0); omega = 2 * math.pi * (1 - math.cos(half))
        Ld = math.pi / (omega * float(wi0 @ n))                                         # disk radiance with E = pi
        indisk = lambda wi: (wi @ wi0) >= math.cos(half)  # noqa: E731
        g.manual_seed(2); u = torch.rand(1 << 22, generator=g, dtype=F64, device=DEV); vv = torch.rand(1 << 22, generator=g, dtype=F64, device=DEV)
        ct = 1 - u * (1 - math.cos(half)); st = torch.sqrt(1 - ct * ct); t, b = frame(wi0)
        wd = st[:, None] * torch.cos(2 * math.pi * vv)[:, None] * t + st[:, None] * torch.sin(2 * math.pi * vv)[:, None] * b + ct[:, None] * wi0
        truth_disk = float((ggx_val(n, wd, wo[None].expand_as(wd), alpha) * (wd @ n) * Ld).mean() * omega)
        truth_sky = dense[0] if alpha == 0.04 else dense[1]
        est = {"cos128": [], "is128": []}
        for sd in range(32):
            g.manual_seed(100 + sd); wc, wcw = cos_samples(n, 128, g); fc = ggx_val(n, wc, wo[None].expand_as(wc), alpha)
            est["cos128"].append(float((fc * Ld * indisk(wc) * wcw).sum()))
            g.manual_seed(200 + sd); wis, wiw = ggx_is_samples(n, wo, alpha, 128, g); fi = ggx_val(n, wis, wo[None].expand_as(wis), alpha)
            est["is128"].append(float((fi * Ld * indisk(wis) * wiw).sum()))
        s2[f"alpha{alpha}"] = {"truth_sky": truth_sky, "truth_disk": truth_disk, "iso_E_collapse_both": 1.0,
                               "dir_cos128_disk_mean": sum(est["cos128"]) / 32, "dir_is128_disk_mean": sum(est["is128"]) / 32,
                               "rel_err_iso_disk": 1.0 / truth_disk - 1, "rel_err_dir_cos128_disk": sum(est["cos128"]) / 32 / truth_disk - 1,
                               "rel_err_dir_is128_disk": sum(est["is128"]) / 32 / truth_disk - 1,
                               "seed_cv_cos128": float(torch.tensor(est["cos128"]).std() / max(abs(sum(est["cos128"]) / 32), 1e-30)),
                               "seed_cv_is128": float(torch.tensor(est["is128"]).std() / max(abs(sum(est["is128"]) / 32), 1e-30))}
    rec["S2_equal_irradiance_different_angular"] = s2
    # S3 -------------------------------------------------------------------------------------------------------------
    s3 = {}; x = torch.zeros(3, dtype=F64, device=DEV)
    E = Rect((0.0, 0.8, 0.6), (0.3, 0, 0), (0, 0.24, -0.18), emit=True)               # tilted: the receiver sees it at a grazing exit angle
    if float((x - E.c) @ E.n) < 0:
        E = Rect((0.0, 0.8, 0.6), (0, 0.24, -0.18), (0.3, 0, 0), emit=True)
    for kappa in (0, 2, 8):
        truth = irradiance_from_rect_dense(x, Z, E, kappa)
        g.manual_seed(3); d, wgt = cos_samples(Z, 1 << 16, g); ok, p, _ = E.hit(x.expand_as(d), d)
        iso = float(((kappa + 2) / 2 * ok * wgt).sum()); dirv = float((emitter_L(E, -d, kappa) * ok * wgt).sum())
        s3[f"kappa{kappa}"] = {"truth": truth, "iso": iso, "dir": dirv, "rel_err_iso": iso / truth - 1, "rel_err_dir": dirv / truth - 1,
                               "exit_cos_at_receiver": float((x - E.c) @ E.n / (x - E.c).norm())}
    rec["S3_directional_sender"] = s3
    # S4 / S5 --------------------------------------------------------------------------------------------------------
    al = 0.04; kap = 0.9
    Em = Rect((0.0, -0.6, 0.5), (0.15, 0, 0), (0, 0.1, 0.1))
    if float((x - Em.c) @ Em.n) < 0:
        Em = Rect((0.0, -0.6, 0.5), (0, 0.1, 0.1), (0.15, 0, 0))
    P = Rect((0.0, 0.0, 0.0), (0.5, 0, 0), (0, 0.5, 0), alpha=al, kap=kap)
    r_in = nrm(Em.c - x); wo4 = torch.tensor([-r_in[0], -r_in[1], r_in[2]], dtype=F64, device=DEV)    # camera at the mirror direction

    def L_emit(o, d):
        ok, p, _ = Em.hit(o, d); return ok.double()                                        # isotropic emitter, radiance 1

    def glossy_dense(xp, w_o, Lfun, k=1 << 21, seed=5):                                     # truth: area sampling of the emitter
        gg = torch.Generator(device=DEV); gg.manual_seed(seed); y = Em.sample(k, gg); d = y - xp; r2 = (d * d).sum(-1); d = d / r2.sqrt()[:, None]
        ce = (-d @ Em.n).clamp(min=0); cr = (d @ P.n).clamp(min=0)
        return float((kap * ggx_val(P.n, d, w_o[None].expand_as(d), al) * cr * ce / r2).mean() * Em.area)

    truth4 = glossy_dense(x, wo4, None)
    e4 = {"cos128": [], "is128": []}
    for sd in range(32):
        g.manual_seed(300 + sd); wc, wcw = cos_samples(P.n, 128, g); e4["cos128"].append(float((kap * ggx_val(P.n, wc, wo4[None].expand_as(wc), al) * L_emit(x.expand_as(wc), wc) * wcw).sum()))
        g.manual_seed(400 + sd); wi_, wiw = ggx_is_samples(P.n, wo4, al, 128, g); e4["is128"].append(float((kap * ggx_val(P.n, wi_, wo4[None].expand_as(wi_), al) * L_emit(x.expand_as(wi_), wi_) * wiw).sum()))
    rec["S4_one_glossy_reflection"] = {"truth": truth4, "iso_collapse": kap / math.pi * irradiance_from_rect_dense(x, P.n, Em, 0, iso=True) * 2 / 2,
                                       "rel_err_iso": (kap / math.pi * irradiance_from_rect_dense(x, P.n, Em, 0, iso=True)) / truth4 - 1,
                                       "rel_err_dir_cos128": sum(e4["cos128"]) / 32 / truth4 - 1, "rel_err_dir_is128": sum(e4["is128"]) / 32 / truth4 - 1}
    # S5: emitter -> P (plane 1, at origin) -> P2 (plane 2) -> camera
    P2 = Rect((0.0, 1.2, 0.6), (0.4, 0, 0), (0, 0.25, -0.35), alpha=al, kap=kap)
    if float((x - P2.c) @ P2.n) < 0:
        P2 = Rect((0.0, 1.2, 0.6), (0, 0.25, -0.35), (0.4, 0, 0), alpha=al, kap=kap)
    x2 = P2.c; d12 = nrm(x - x2); wo5 = 2 * (d12 @ P2.n) * P2.n - d12                    # camera at the mirror of the direction toward plane 1

    def L_plane1(ys, w_out):                                                                # exact outgoing radiance of plane 1 (emitter area integral)
        k = 1 << 10; gg = torch.Generator(device=DEV); gg.manual_seed(7); e = Em.sample(k, gg)
        d = e[None] - ys[:, None]; r2 = (d * d).sum(-1); d = d / r2.sqrt()[..., None]
        ce = (-d @ Em.n).clamp(min=0); cr = (d @ P.n).clamp(min=0)
        f = kap * ggx_val(P.n, d.reshape(-1, 3), w_out[:, None].expand_as(d).reshape(-1, 3), al).view(d.shape[:2])
        return (f * cr * ce / r2).mean(1) * Em.area

    g.manual_seed(9); wis, wisw = ggx_is_samples(P2.n, wo5, al, 1 << 14, g); ok, yp, _ = P.hit(x2.expand_as(wis), wis)
    Ly = torch.zeros(len(wis), dtype=F64, device=DEV); Ly[ok] = L_plane1(yp[ok], -wis[ok])
    truth5 = float((kap * ggx_val(P2.n, wis, wo5[None].expand_as(wis), al) * Ly * wisw).sum())
    # interface: plane 1 is a grid of ports (5 cm); a ray hitting plane 1 reads its port's sender value
    def port_centre(yp_):
        q = torch.round(yp_ / 0.05) * 0.05; q[:, 2] = 0; return q

    def sender_iso(ys, sd):                                                                 # isotropic collapse: kappa / pi E(port)
        Ev = torch.tensor([irradiance_from_rect_dense(y, P.n, Em, 0, k=1 << 12, seed=sd, iso=True) for y in ys], dtype=F64, device=DEV)
        return kap / math.pi * Ev

    def sender_dir(ys, w_out, sampler, sd):                                                 # K_loc of the port's own 128 incident samples toward w_out
        out = torch.zeros(len(ys), dtype=F64, device=DEV); gg = torch.Generator(device=DEV)
        for i, y in enumerate(ys):
            gg.manual_seed(sd * 7919 + i)
            wi_, ww = (cos_samples(P.n, 128, gg) if sampler == "cos" else ggx_is_samples(P.n, w_out[i], al, 128, gg))
            out[i] = (kap * ggx_val(P.n, wi_, w_out[i][None].expand_as(wi_), al) * L_emit(y.expand_as(wi_), wi_) * ww).sum()
        return out

    e5 = {"iso_cos": [], "dir_cos": [], "dir_is": []}
    for sd in range(32):
        for lab in e5:
            gg = torch.Generator(device=DEV); gg.manual_seed(500 + sd)
            wi_, ww = (cos_samples(P2.n, 128, gg) if lab != "dir_is" else ggx_is_samples(P2.n, wo5, al, 128, gg))
            okh, yh, _ = P.hit(x2.expand_as(wi_), wi_); val = torch.zeros(len(wi_), dtype=F64, device=DEV)
            if okh.any():
                yc = port_centre(yh[okh])
                val[okh] = sender_iso(yc, sd) if lab == "iso_cos" else sender_dir(yc, -wi_[okh], "cos" if lab == "dir_cos" else "is", sd)
            e5[lab].append(float((kap * ggx_val(P2.n, wi_, wo5[None].expand_as(wi_), al) * val * ww).sum()))
    rec["S5_two_glossy_bounces"] = {"truth": truth5, **{f"rel_err_{k}": sum(v) / 32 / truth5 - 1 for k, v in e5.items()},
                                    **{f"seed_cv_{k}": float(torch.tensor(v).std() / max(abs(sum(v) / 32), 1e-30)) for k, v in e5.items()}}
    # S6: superposition (deterministic: same samples)
    g.manual_seed(11); d, wgt = cos_samples(Z, 4096, g)
    E1 = E; E2 = Rect((0.3, 0.6, 0.7), (0.1, 0, 0), (0, 0.1, -0.05))
    a = (emitter_L(E1, -d, 2) * E1.hit(x.expand_as(d), d)[0] * wgt).sum(); b = (emitter_L(E2, -d, 2) * E2.hit(x.expand_as(d), d)[0] * wgt).sum()
    both = ((emitter_L(E1, -d, 2) * E1.hit(x.expand_as(d), d)[0] + emitter_L(E2, -d, 2) * E2.hit(x.expand_as(d), d)[0]) * wgt).sum()
    rec["S6_superposition_rel"] = float((both - a - b).abs() / both.abs())
    crit = {"S0": rec["S0_zero"] == 0, "S6": rec["S6_superposition_rel"] < 1e-5,
            "S2_dir_is_within_2pct": all(abs(v["rel_err_dir_is128_disk"]) <= 0.02 for v in s2.values()),
            "S3_dir_within_2pct": all(abs(v["rel_err_dir"]) <= 0.02 for v in s3.values()),
            "S4_dir_is_within_2pct": abs(rec["S4_one_glossy_reflection"]["rel_err_dir_is128"]) <= 0.02,
            "S5_dir_is_within_2pct": abs(rec["S5_two_glossy_bounces"]["rel_err_dir_is"]) <= 0.02}
    rec["criteria"] = crit; rec["ALL_PASS"] = all(crit.values())
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
