"""WL111A Stage B (H2 / H3): operator capacity witnesses on development views (target-informed; never deployable).

For each dev view and spp in (1, 4), on a 256 x 256 crop around the SSS sphere:
  CTAK_oracle  the actual composed CT-AK operator (3 passes x 2 components x 25 softmax logits per pixel, dilations 1, 2,
               4, reflect padding, cross-pixel dependencies through the passes) with free per-pixel logits
  R3_oracle    R3's operator (per component one 15 x 15 softmax kernel per pixel) with free per-pixel logits
Both are noise-independent per-pixel weights fitted to the dev reference on K_FIT independent frames (warm start: the
learned predictor's logits on fit frame 0; Adam, fixed steps) and scored on K_TEST held-out frames. Learned S111 r0 / r1
and R3 are scored on the same held-out frames. Scores use the crop interior (16 px margin) intersected with each ROI.
These witnesses show what each operator family can represent with non-adaptive weights; they are not certificates of
impossibility, and an input-adaptive predictor is a different family.

python oracle_r111a.py [--views Q00,...] [--steps N]   -> OUT/stageB/oracle.json
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as Fn

import common_r111a as A
import common_r111 as C
import engine_r111 as E
import recon_r111 as RC
import r3_r111
import seq_r111 as SQ
import stageA_r111a as SA

M = r3_r111.M
W = C.RES
CROP, MARGIN = 256, 16
K_FIT, K_TEST = 8, 8
SPPS = (1, 4)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def crop_box(aux):
    sph = aux["cover"][6] > 0
    ys, xs = np.nonzero(sph)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (W // 2, W // 2)
    y0 = min(max(cy - CROP // 2, 0), W - CROP)
    x0 = min(max(cx - CROP // 2, 0), W - CROP)
    return y0, x0


def dense_apply(img, logits):
    """R3 operator (model105.apply_kernel, unfused path): img (B,3,h,w), logits (B,225,h,w)."""
    K = M.K
    r = K // 2
    w = torch.softmax(logits, 1)
    B, Cc, H, Wd = img.shape
    p = Fn.unfold(Fn.pad(img, (r, r, r, r), mode="reflect"), K).view(B, Cc, K * K, H, Wd)
    return (p * w[:, None]).sum(2)


def rel_loss(y, t):
    return ((y - t) ** 2 / (t ** 2 + 0.01)).mean()


def fit(params, forward, tgt, steps, lr):
    params = params.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=lr)
    trace = []
    for s in range(steps):
        lo, hi = forward(params)
        loss = rel_loss(lo, tgt["low"]) + rel_loss(hi, tgt["high"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            trace.append([s, float(loss)])
    return params.detach(), trace


def score(outs, ref, var_ref, masks, n):
    """outs: list of (lo, hi) numpy (3,h,w) over n held-out frames -> decomposition per component and ROI."""
    res = {}
    for c in ("low", "high", "total"):
        ys = np.stack([o[0] if c == "low" else o[1] if c == "high" else o[0] + o[1] for o in outs])
        m, v = ys.mean(0), ys.var(0, ddof=1)
        res[c] = {roi: A.decompose(m, v, n, ref[c], var_ref[c], mk) for roi, mk in masks.items() if mk.sum() > 50}
    return res


def main(views, steps):
    out = A.OUT / "stageB"
    out.mkdir(parents=True, exist_ok=True)
    eng, audit = SA.setup()
    cams = {k: E.cam_dict(c) for k, c in SQ.wl105_cameras().items()}
    rec = SA.Recons()
    r3 = rec.r3
    res = {"crop": CROP, "margin": MARGIN, "k_fit": K_FIT, "k_test": K_TEST, "steps": steps, "views": {}}
    for v in views:
        vi = A.DEV_VIEWS.index(v)
        ref_full, var_full, aux = SA.load_devref(v)
        y0, x0 = crop_box(aux)
        ys, xs = slice(y0, y0 + CROP), slice(x0, x0 + CROP)
        inner = np.zeros((CROP, CROP), bool)
        inner[MARGIN:-MARGIN, MARGIN:-MARGIN] = True
        masks = {k: m[ys, xs] & inner for k, m in C.rois(aux).items()}
        ref = {c: ref_full[c][:, ys, xs] for c in ref_full}
        var_ref = {c: var_full[c][:, ys, xs] for c in var_full}
        tgt = {c: torch.from_numpy(ref[c]).float().cuda()[None] for c in ("low", "high")}
        cam = cams[v]
        g4, g8 = eng.guides(cam, 4), eng.guides(cam, 8)
        res["views"][v] = {"box": [y0, x0]}
        for si, spp in enumerate(SPPS):
            t0 = time.perf_counter()
            frames = [eng.frame(cam, spp, A.seed("devset", vi, 50 + si, k)) for k in range(K_FIT + K_TEST)]
            lows, highs, learned = [], [], {"R3": [], "S111_r0": [], "S111_r1": []}
            ess = torch.full((W, W), float(spp), device="cuda")
            with torch.no_grad():
                aux14 = r3_r111.aux14(g8, spp)
                for k, comp in enumerate(frames):
                    lo, hi = RC.raw_low_high(comp)
                    lows.append(lo[:, ys, xs])
                    highs.append(hi[:, ys, xs])
                    if k >= K_FIT:
                        o = rec(comp, spp, g4, g8)
                        for a in learned:
                            learned[a].append(tuple(x[:, ys, xs].double().cpu().numpy() for x in o[a]))
                # warm starts: learned logits on fit frame 0 (full frame, then cropped)
                lo0, hi0 = RC.raw_low_high(frames[0])
                x = RC.features(lo0[None], hi0[None], g4["cover"][None], g4["normal"][None], g4["depth"][None, None],
                                ess.view(1, 1, W, W))
                lg_c = rec.cands["S111_r0"].net.float()(x.contiguous())[:, :, ys, xs].float()
                rec.cands["S111_r0"].net.to(memory_format=torch.channels_last)
                xl = torch.cat([torch.log1p(lo0[None]), aux14], 1)
                xh = torch.cat([torch.log1p(hi0[None]), aux14], 1)
                lg_r = torch.cat([r3.net_low(xl), r3.net_high(xh)], 0)[:, :, ys, xs].float()
            fitL = torch.stack(lows[:K_FIT]).float()
            fitH = torch.stack(highs[:K_FIT]).float()

            def ctak_fwd(p, L=fitL, Hh=fitH):
                pe = p.expand(L.shape[0], -1, -1, -1)
                return RC.spatial_ref(L, Hh, pe)

            def r3_fwd(p, L=fitL, Hh=fitH):
                return (dense_apply(L, p[0:1].expand(L.shape[0], -1, -1, -1)),
                        dense_apply(Hh, p[1:2].expand(Hh.shape[0], -1, -1, -1)))

            pc, tr_c = fit(lg_c, ctak_fwd, tgt, steps, 0.05)
            pr, tr_r = fit(lg_r, r3_fwd, tgt, steps, 0.05)
            cell = {"traces": {"CTAK_oracle": tr_c, "R3_oracle": tr_r}}
            with torch.no_grad():
                tl = torch.stack(lows[K_FIT:]).float()
                th = torch.stack(highs[K_FIT:]).float()
                oc = ctak_fwd(pc, tl, th)
                orr = r3_fwd(pr, tl, th)
                fc = ctak_fwd(pc)
                fr_ = r3_fwd(pr)
            to_np = lambda t: [(t[0][i].double().cpu().numpy(), t[1][i].double().cpu().numpy()) for i in range(t[0].shape[0])]
            cell["test"] = {"CTAK_oracle": score(to_np(oc), ref, var_ref, masks, K_TEST),
                            "R3_oracle": score(to_np(orr), ref, var_ref, masks, K_TEST),
                            **{a: score(learned[a], ref, var_ref, masks, K_TEST) for a in learned}}
            cell["fit"] = {"CTAK_oracle": score(to_np(fc), ref, var_ref, masks, K_FIT),
                           "R3_oracle": score(to_np(fr_), ref, var_ref, masks, K_FIT)}
            res["views"][v][str(spp)] = cell
            log("oracle", v, spp, round(time.perf_counter() - t0, 1), "s",
                {a: round(cell["test"][a]["total"]["primary"]["rel_rmse"], 4) for a in cell["test"]})
            del frames
            torch.cuda.empty_cache()
        C.write(out / "oracle.json", res)
    print("ORACLE_COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default=",".join(A.DEV_VIEWS))
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    main(a.views.split(","), a.steps)
