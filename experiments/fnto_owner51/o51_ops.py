"""WL51 frozen K / C loop attribution, reference-semantics counterfactual and energy audit (protocol fnto_owner51).

Diagnostics only. Production r1_solve / K_k44P / the WL50 arms (c50_common) are imported unchanged; edge masks and the SEM
sender override exist only on State instances created here.
  edge mask   valid / spec / L zeroed on the masked (receiver port, gather) edges; receiver class = port class, sender
              class = class of the hit port (Y Yellow, O other opaque Lego, G glass, X box)
  SEM         the value a Yellow port SENDS into C gathers = Cycles diffuse-ancestor operator 0.98 c / pi E_tot
              + K44 kap / pi E_ind (glossy); receiver / render terms unchanged

    python o51_ops.py g1    <out dir>                 masks x per-channel rho x modes (H / L0C / L1C seed 5011), seed-5012 and init-seed replicates
    python o51_ops.py views <out dir>                 held-out T0: one-step prod / SEM, SEM full solves, H full solve
    python o51_ops.py g3    <out dir>                 C closure / reciprocity by class (uniform emission)
    python o51_ops.py g0    <out dir>                 air-chord / source-material proxy shares of the Burley oracle
    python o51_ops.py smoke <out dir>                 2 iterations / 2 views of every stage (profiling)
"""
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_comp50"))
import c50_common as CC  # noqa: E402
import r1_common as C  # noqa: E402
import s49_data as S49  # noqa: E402

PROTO = json.loads((HERE / "protocol" / "owner51.json").read_text())
CK = C.ROOT / "results/fnto_comp50/train"
Y_COLOR = (0.799102783203125, 0.5028861165046692, 0.0295569971203804)   # blend Yellow SSS colour (gate 0 dump)
SSS_W = 0.98                                                            # Mix Fac 0.02 -> SSS weight 0.98
RHO_Y_INT = [SSS_W * c for c in Y_COLOR]                                # Cycles Yellow at a diffuse-ancestor vertex (Lambert)
NAMES = ["Y", "O", "G", "X"]
MASKS = ["none", "noYY", "noYO", "noOY", "noOO", "onlyYY", "noYY_t3mm", "noYY_t3cm", "noCross"]
TBINS = [0.0, 0.003, 0.01, 0.03, 0.10, float("inf")]
LUM = np.array([0.2126, 0.7152, 0.0722]); PXCLS = {"Yellow": [0], "opaque": [1, 3, 5, 6, 7], "glass": [2, 4], "box": [8]}
CH = 1 << 15
SMOKE = False


def setup(dev):
    from r1_eval import load_kernels
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = CC.SV.Tables(KA, KB, dev)
    R = np.load(C.OUT1 / "rel_T0.npz"); H = R["hit"]
    return T, H, R


def classes(T):
    obj, mat = T.obj, T.mat
    return torch.where(obj == 1, 3, torch.where(mat == 0, 0, torch.where((mat == 2) | (mat == 4), 2, 1))).long()


def make_arm(cn, dev, geo, seed=5011):
    if cn == "H":
        return None
    kind = cn[:2]; return CC.ArmOp(kind, CC.load_net(str(CK / f"ckpt_{kind}_seed{seed}.pt"), dev), geo, CC.yellow_ports())


@torch.no_grad()
def edge_keep(S, cls, name, tT=None):
    keep = S.valid.clone()
    if name == "none":
        return keep
    for s in range(0, S.T.N, CH):
        sl = slice(s, s + CH); r = cls[sl][:, None]; snd = cls[S.Hc[sl]]; Yr, Ys = r == 0, snd == 0
        if name == "noYY":
            m = Yr & Ys
        elif name == "noYO":
            m = Ys & ~Yr
        elif name == "noOY":
            m = ~Ys & Yr
        elif name == "noOO":
            m = ~Ys & ~Yr
        elif name == "onlyYY":
            m = ~(Yr & Ys)
        elif name == "noYY_t3mm":
            m = Yr & Ys & (tT[sl] < 0.003)
        elif name == "noYY_t3cm":
            m = Yr & Ys & (tT[sl] < 0.03)
        elif name == "noCross":
            m = (r == 3) != (snd == 3)
        else:
            raise ValueError(name)
        keep[sl] &= ~m
    return keep


@torch.no_grad()
def apply_mask(S, keep):
    S.valid = keep
    for s in range(0, S.T.N, CH):
        sl = slice(s, s + CH); k = keep[sl][..., None].half(); S.spec[sl] *= k; S.L[sl] *= k


def sem_on(S):
    """Diagnostic override of the Yellow SENDER value (Cycles diffuse-ancestor semantics); render terms untouched."""
    T = S.T; yp = torch.as_tensor(CC.yellow_ports(), device=T.dev); rho = torch.tensor(RHO_Y_INT, device=T.dev); orig = S.outgoing

    def outgoing(Eind, h, sun_on=True):
        I = orig(Eind, h, sun_on)
        Et = Eind[yp] + (S.Esun[yp, None] if sun_on else 0)
        I[yp] = rho / math.pi * Et + T.kap[yp].sum(1) / math.pi * Eind[yp]
        return I
    S.outgoing = outgoing; S.sem = True
    return S


def sun_off_state(T, H):
    return CC.SV.State(T, H, np.zeros(T.N, np.float32), np.array([0.0, 0.0, 1.0]))


@torch.no_grad()
def chan_norm(L):
    acc = torch.zeros(3, dtype=torch.float64, device=L.device)
    for s in range(0, L.shape[0], CH):
        acc += L[s:s + CH].float().square().sum((0, 1)).double()
    return acc.sqrt()


@torch.no_grad()
def scale_(L, sc):
    sc = sc.float()
    for s in range(0, L.shape[0], CH):
        L[s:s + CH] = (L[s:s + CH].float() * sc).half()


@torch.no_grad()
def power(S, iters, seed):
    """Per-channel power iteration of the homogeneous composed map; each channel renormalised separately."""
    g = torch.Generator(device=S.T.dev).manual_seed(seed)
    S.L = (torch.rand(S.L.shape, generator=g, device=S.T.dev, dtype=torch.float16) * S.valid[..., None]).half()
    n = chan_norm(S.L); scale_(S.L, torch.where(n > 0, 1e3 / n.clamp(min=1e-30), torch.ones_like(n)))
    rates = []; status = "ok"
    for _ in range(iters):
        n0 = chan_norm(S.L); S.step(sun_on=False); n1 = chan_norm(S.L)
        if not torch.isfinite(n1).all():
            status = "nonfinite"; break
        rates.append(torch.where(n0 > 0, n1 / n0.clamp(min=1e-300), torch.zeros_like(n0)).tolist())
        scale_(S.L, torch.where(n1 > 0, 1e3 / n1.clamp(min=1e-30), torch.ones_like(n1)))
    R = np.array(rates) if rates else np.zeros((0, 3))
    rho = np.median(R[-10:], 0).tolist() if len(R) else [float("nan")] * 3
    return {"rho_RGB": rho, "rho_max": float(np.max(rho)), "rates_RGB": R.tolist(), "status": status, "iters": len(R)}


@torch.no_grad()
def class_flux(S, I_send, cls, tT=None, ch_bins=None):
    """After S.step(): flux gathered at receiver class r from sender class s [4, 4, 3] (port areas), and Y<-Y by hit distance."""
    T = S.T; w = math.pi / C.K_GATHER; F = torch.zeros(4, 4, 3, dtype=torch.float64, device=T.dev); yyb = torch.zeros(len(TBINS) - 1, 3, dtype=torch.float64, device=T.dev)
    for s in range(0, T.N, CH):
        sl = slice(s, s + CH); Lf = S.L[sl].float() * w; snd = cls[S.Hc[sl]]; A = T.area[sl]; rc = cls[sl]
        for k in range(4):
            g = (Lf * (snd == k)[..., None]).sum(1) * A[:, None]                              # [n, 3] gathered from class k
            F[:, k].index_add_(0, rc, g.double())
        if tT is not None:
            ym = (rc == 0)[:, None] & (snd == 0); tt = tT[sl]
            for b in range(len(TBINS) - 1):
                mb = ym & (tt >= TBINS[b]) & (tt < TBINS[b + 1])
                yyb[b] += ((Lf * mb[..., None]).sum(1) * A[:, None]).sum(0).double()
    return F, yyb


@torch.no_grad()
def mode_stats(S, cls, tT, comp, dch, terms_fn=None):
    T = S.T; A = T.area
    Eind, h = S.aggregates(); I = S.outgoing(Eind, h, sun_on=False)
    Pin = torch.zeros(4, 3, dtype=torch.float64, device=T.dev).index_add_(0, cls, (A[:, None] * Eind).double())
    Pout = torch.zeros(4, 3, dtype=torch.float64, device=T.dev).index_add_(0, cls, (math.pi * A[:, None] * I).double())
    r = {"class_share_outgoing_flux_RGB": {NAMES[k]: (Pout[k] / Pout.sum(0).clamp(min=1e-300)).tolist() for k in range(4)}}
    if terms_fn is not None:
        tm = terms_fn(S, Eind, h); ym = cls == 0
        r["yellow_term_share_RGB"] = {k: ((v[ym] * A[ym, None]).sum(0) / (I[ym] * A[ym, None]).sum(0).clamp(min=1e-30)).tolist() for k, v in tm.items()}
    w = (A * I[:, dch]).abs(); ws = torch.sort(w, descending=True).values; N = int((w > 0).sum())
    r["participation_ratio"] = float(w.sum() ** 2 / (N * (w ** 2).sum()).clamp(min=1e-30)) if N else 0.0
    r["top1pct_share"] = float(ws[:max(1, N // 100)].sum() / w.sum()); r["top10pct_share"] = float(ws[:max(1, N // 10)].sum() / w.sum())
    cs = torch.zeros(int(comp.max()) + 1, dtype=torch.float64, device=T.dev).index_add_(0, comp, w.double()); top = torch.sort(cs, descending=True)
    r["top5_component_share"] = (top.values[:5] / cs.sum()).tolist(); r["top5_components"] = top.indices[:5].tolist()
    S.step(sun_on=False)
    F, yyb = class_flux(S, I, cls, tT)
    M = (F / Pout[None].clamp(min=1e-300)); g = Pout / Pin.clamp(min=1e-300)
    r["M_return_fraction_RGB[r][s]"] = M.permute(2, 0, 1).tolist(); r["gain_out_over_in_RGB"] = {NAMES[k]: g[k].tolist() for k in range(4)}
    red = []
    for c in range(3):
        Mc = (M[:, :, c] * g[:, c][None]).cpu().numpy(); ok = (Pout[:, c] > 0).cpu().numpy() & (Pin[:, c] > 0).cpu().numpy()
        red.append(float(np.max(np.abs(np.linalg.eigvals(Mc[np.ix_(ok, ok)])))) if ok.any() else 0.0)
    r["rho_class_reduced_RGB"] = red
    r["YY_gathered_by_hit_distance_RGB"] = {f"[{TBINS[b]},{TBINS[b + 1]})": (yyb[b] / yyb.sum(0).clamp(min=1e-300)).tolist() for b in range(len(TBINS) - 1)}
    return r


@torch.no_grad()
def g1(out, dev):
    import c50_diag as CD
    T, H, R = setup(dev); cls = classes(T); tT = torch.as_tensor(R["t"], device=dev); geo = CC.Geo(dev)
    comp = torch.as_tensor(np.load(C.OUT1 / "anchors.npz")["comp"].astype(np.int64), device=dev).clamp(min=0)
    iters = 2 if SMOKE else 40; rec = {"iters": iters, "rho_class_note": "dominant-channel mode stats on the unmasked prod and SEM runs"}
    masks = MASKS[:2] if SMOKE else MASKS
    conds = [("H", 5011), ("L0C", 5011), ("L1C", 5011)] + ([] if SMOKE else [("L0C", 5012), ("L1C", 5012)])
    for cn, seed in conds:
        t0 = time.time(); arm = make_arm(cn, dev, geo, seed); CC.activate(arm); key = f"{cn}_{seed}"; rr = {}
        mlist = masks if seed == 5011 else ["none"]
        for mk in mlist:
            S = sun_off_state(T, H); apply_mask(S, edge_keep(S, cls, mk, tT)); t1 = time.time()
            p = power(S, iters, 0); p["wall_s"] = time.time() - t1
            if mk == "none" and seed == 5011 and p["status"] == "ok":
                dch = int(np.argmax(p["rho_RGB"])); p["mode"] = mode_stats(S, cls, tT, comp, dch, CD.terms); p["mode"]["dominant_channel"] = "RGB"[dch]
                S2 = sun_off_state(T, H); p["init_seed1"] = {k: v for k, v in power(S2, iters, 1).items() if k != "rates_RGB"}; del S2
            rr[mk] = p; print(key, mk, json.dumps({k: v for k, v in p.items() if k in ("rho_RGB", "status")}), flush=True)
            del S; gc.collect(); torch.cuda.empty_cache()
        if seed == 5011:                                                                          # SEM counterfactual
            S = sem_on(sun_off_state(T, H)); t1 = time.time(); p = power(S, iters, 0); p["wall_s"] = time.time() - t1
            if p["status"] == "ok":
                dch = int(np.argmax(p["rho_RGB"])); p["mode"] = mode_stats(S, cls, tT, comp, dch); p["mode"]["dominant_channel"] = "RGB"[dch]
            rr["SEM"] = p; print(key, "SEM", json.dumps({k: v for k, v in p.items() if k in ("rho_RGB", "status")}), flush=True); del S; gc.collect(); torch.cuda.empty_cache()
        rr["wall_s"] = time.time() - t0; rec[key] = rr
        CC.deactivate(); del arm; gc.collect(); torch.cuda.empty_cache()
        (out / "g1.json").write_text(json.dumps(rec, indent=1))
    rec["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9; (out / "g1.json").write_text(json.dumps(rec, indent=1))


def psnr(p, r):
    return float(10 * np.log10(1 / max(((C.ldr3(p) - C.ldr3(r)) ** 2).mean(), 1e-30)))


@torch.no_grad()
def views(out, dev):
    T, H, R = setup(dev); cls = classes(T); geo = CC.Geo(dev); nA = int(T.dom[0].numel())
    vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    full = np.load(C.OUT1 / "refs/heldout_T0_full.npz"); first = np.load(C.OUT1 / "refs/heldout_T0_first.npz")
    vlist = [6, 15] if SMOKE else list(range(int(px["nv"]))); rec = {"per_view": {}, "pooled": {}, "energy": {}, "flux_first_event": {}, "solves": {}}; acc = {}
    yp = torch.as_tensor(CC.yellow_ports(), device=dev)
    for cn in ("H", "L0C", "L1C"):
        arm = make_arm(cn, dev, geo); CC.activate(arm); t0 = time.time()
        for v in vlist:
            val = px[f"v{v}_valid"].astype(bool); mat = np.where(px[f"v{v}_dom"][val] == 1, 8, px[f"v{v}_mat"][val])
            pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][val].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
            rf, r0 = full[f"v{v}_rgb"][val].astype(np.float64), first[f"v{v}_rgb"][val].astype(np.float64); rind = (rf - r0) @ LUM
            Vf = vis[f"heldout_T0_v{v}_T0"].astype(np.float32); light = px[f"v{v}_light"]
            runs = [("prod", "onestep"), ("SEM", "onestep"), ("SEM", "solve")] + ([("prod", "solve")] if cn == "H" else [])
            for sem, kind in runs:
                S = CC.SV.State(T, H, Vf, light)
                if sem == "SEM":
                    sem_on(S)
                Eind, h = S.aggregates(); I0 = S.outgoing(Eind, h); p0 = CC.SV.render_pixels(T, S, pxd)[0].cpu().numpy()
                if kind == "onestep":
                    if v in (6, 15):
                        A = T.area; ym = cls == 0
                        rec["energy"].setdefault(f"{cn}_{sem}", {})[f"view{v}"] = {
                            "gY_sun_portarea_RGB": (math.pi * (I0[ym] * A[ym, None]).sum(0) / (A[ym] * S.Esun[ym]).sum()).tolist(),
                            "yellow_out_over_lego_sun_RGB": (math.pi * (I0[yp] * A[yp, None]).sum(0) / (A[:nA] * S.Esun[:nA]).sum()).tolist()}
                    S.step()
                    if v in (6, 15):
                        F, _ = class_flux(S, I0, cls); Pout = torch.zeros(4, 3, dtype=torch.float64, device=dev).index_add_(0, cls, (math.pi * T.area[:, None] * I0).double())
                        rec["flux_first_event"].setdefault(f"{cn}_{sem}", {})[f"view{v}"] = {"F_gathered_r_from_s_RGB": F.permute(2, 0, 1).tolist(), "Pout_RGB": Pout.permute(1, 0).tolist()}
                    Eind, h = S.aggregates(); S.outgoing(Eind, h); info = {}
                else:
                    S.solve(); info = {"status": S.status, "iters": S.iters, "last_hist": S.hist[-1] if S.hist else None, "solve_s": S.solve_s}
                    rec["solves"].setdefault(f"{cn}_{sem}", {})[f"view{v}"] = info
                p1 = CC.SV.render_pixels(T, S, pxd)[0].cpu().numpy(); pind = (p1 - p0) @ LUM
                rr = {"psnr_vs_full": psnr(p1, rf) if np.isfinite(p1).all() else None, "finite": bool(np.isfinite(p1).all()), **info}
                for k, ids in PXCLS.items():
                    m = np.isin(mat, ids)
                    if m.sum() > 100 and np.isfinite(p1[m]).all():
                        rr[k] = {"indirect_ratio_to_ref": float(pind[m].sum() / max(rind[m].sum(), 1e-30)), "indirect_corr": float(np.corrcoef(pind[m], rind[m])[0, 1]),
                                 "psnr_vs_full": psnr(p1[m], rf[m])}
                        a = acc.setdefault((cn, sem, kind, k), [0.0, 0.0, [], []]); a[0] += float(pind[m].sum()); a[1] += float(rind[m].sum()); a[2].append(pind[m]); a[3].append(rind[m])
                rec["per_view"].setdefault(f"{cn}_{sem}_{kind}", {})[f"view{v}"] = rr
                print(cn, sem, kind, v, json.dumps({k: (round(x["indirect_ratio_to_ref"], 3) if isinstance(x, dict) else x) for k, x in rr.items() if k in PXCLS or k == "status"}), flush=True)
                del S; gc.collect(); torch.cuda.empty_cache()
        rec.setdefault("cond_wall_s", {})[cn] = time.time() - t0
        CC.deactivate(); del arm; gc.collect(); torch.cuda.empty_cache()
        for (c_, s_, k_, cl), a in acc.items():
            pi_, ri_ = np.concatenate(a[2]), np.concatenate(a[3])
            rec["pooled"].setdefault(f"{c_}_{s_}_{k_}", {})[cl] = {"indirect_ratio_to_ref": a[0] / max(a[1], 1e-30), "indirect_corr": float(np.corrcoef(pi_, ri_)[0, 1])}
        (out / "views.json").write_text(json.dumps(rec, indent=1))
    rec["peak_GB"] = torch.cuda.max_memory_allocated() / 1e9; (out / "views.json").write_text(json.dumps(rec, indent=1))


@torch.no_grad()
def g3(out, dev):
    """C closure for uniform unit radiance per sender class (Yellow split by x median): received flux / emitted flux."""
    T, H, R = setup(dev); cls = classes(T); A = T.area
    ym = cls == 0; xmed = T.pos[ym, 0].median(); c5 = cls.clone() + 1; c5[ym & (T.pos[:, 0] >= xmed)] = 0      # 0 Y_hi, 1 Y_lo, 2 O, 3 G, 4 X
    names = ["Y_hi", "Y_lo", "O", "G", "X"]; Hc = torch.as_tensor(H.astype(np.int64), device=dev); w = math.pi / C.K_GATHER
    recv = torch.zeros(5, 5, dtype=torch.float64, device=dev)                                      # [r, s]
    for s in range(0, T.N, CH):
        sl = slice(s, s + CH); h = Hc[sl]; ok = h >= 0; snd = c5[h.clamp(min=0)]
        for k in range(5):
            recv[:, k].index_add_(0, c5[sl], (A[sl] * w * ((snd == k) & ok).sum(1)).double())
    emit = torch.zeros(5, dtype=torch.float64, device=dev).index_add_(0, c5, (math.pi * A).double())
    Rm = recv / emit[None]; Q = recv / recv.T.clamp(min=1e-300)
    area = {n: float(A[c5 == k].sum()) for k, n in enumerate(names)}
    rec = {"classes": names, "received_over_emitted[r][s]": Rm.tolist(), "captured_fraction_by_sender": Rm.sum(0).tolist(), "reciprocity_Q[r][s]": Q.tolist(),
           "port_area_m2": area, "lego_port_area_m2": float(A[:int(T.dom[0].numel())].sum())}
    (out / "g3_closure.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


@torch.no_grad()
def g0(out, dev):
    L = S49.Lego(dev); _, held = S49.receivers(L); xi = torch.as_tensor(held, device=dev); E = S49.view_fields(L)[:, [6, 15]]
    if SMOKE:
        xi = xi[:50]
    d = S49.dch(); xp, nr = L.pos[xi], L.nrm[xi]; nonY_src = torch.as_tensor(L.mat != 0, device=dev); AE = E * L.area[:, None]
    keys = ["tot", "air", "nonY", "air_in", "tot_in"]; acc = {k: torch.zeros(3, 2, dtype=torch.float64, device=dev) for k in keys}; ch = 8192
    for s0 in range(0, L.nA, ch):
        s1 = min(s0 + ch, L.nA); dv = L.pos[s0:s1][None] - xp[:, None]; r = dv.norm(dim=-1).clamp(min=2e-3)
        air = ((-(dv * L.nrm[s0:s1][None]).sum(-1)) > 0.1 * r) | (((dv * nr[:, None]).sum(-1)) > 0.1 * r)
        notself = xi[:, None] != torch.arange(s0, s1, device=dev)[None]; inn = r < S49.R_IN
        for c in range(3):
            k = S49.burley(r, d[c]) * (r < 16 * d[c]) * notself
            for key, m in (("tot", None), ("air", air), ("nonY", nonY_src[s0:s1][None].expand_as(k)), ("air_in", air & inn), ("tot_in", inn)):
                kk = k if m is None else k * m
                acc[key][c] += (kk @ AE[s0:s1]).sum(0).double()
    rec = {"receivers": int(len(xi)), "views": [6, 15], "d_RGB_m": d, "proxy": PROTO["gate0"]["air_chord_proxy"],
           "air_chord_share_RGB_by_view": (acc["air"] / acc["tot"]).T.tolist(), "air_chord_share_in_support_RGB_by_view": (acc["air_in"] / acc["tot_in"]).T.tolist(),
           "nonyellow_source_share_RGB_by_view": (acc["nonY"] / acc["tot"]).T.tolist(), "in_support_share_RGB_by_view": (acc["tot_in"] / acc["tot"]).T.tolist()}
    (out / "g0_chord.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


def main():
    global SMOKE
    stage, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    if stage == "smoke":
        SMOKE = True
        for f in (g0, g3, g1, views):
            t = time.time(); f(out, dev); print("SMOKE", f.__name__, round(time.time() - t, 1), "s", flush=True)
    else:
        {"g1": g1, "views": views, "g3": g3, "g0": g0}[stage](out, dev)
    (out / f"cost_{stage}.json").write_text(json.dumps({"wall_s": time.time() - t0, "peak_GB": torch.cuda.max_memory_allocated() / 1e9, "gpu": torch.cuda.get_device_name(0)}))


if __name__ == "__main__":
    main()
