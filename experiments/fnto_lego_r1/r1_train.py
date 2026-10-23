"""WL42 Stage A r1 training of the intrinsic kernels K_A, K_B (server; torch only).

    python r1_train.py --out <dir> [--steps 24000] [--passivity 0] [--max-steps N]

Same kernel family as Batch 6 (fnto_model.DomainKernel: K_loc, K_glob, K_band, K_mid; linear, non-negative), trained on
the r1 probes (r1_stagea_render / r1_prep). Changes are the corrected inputs only:
  * incident field = sum over the image's lights (sun / point / spot) reconstructed exactly from the light table and the
    per-port visible fraction of S_PORT surface samples (ports) or the per-pixel visibility bits (pixels);
  * K_band sources = 1.5 cm aggregates (area-mean 19-bin incident) over the FULL 5 cm ball via 48 stratified
    representatives weighted by represented area; K_mid sources = 8 cm aggregates over the FULL 30 cm ball (64
    representatives, represented area);
  * pixels are attached to ports by surface identity (r1_map), not by nearest port.
--passivity lam (declared conditional variant r1-P only): adds lam x mean(relu(albedo_eff - 0.98)) over 32,768 sampled
ports, albedo_eff = pi I / E of the port's isotropic response to a uniform unit radiance field (all four components).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import r1_common as C
import fnto_model as M

D = C.OUT1 / "stagea"
NB = C.N_BINS_H


class Split:
    def __init__(self, name):
        od = D / f"prep_{name}"
        self.px = {k: np.load(od / f"px_{k}.npy", mmap_mode="r") for k in ("pos", "nrm", "cam", "col", "dom", "port", "vbits")}
        self.start = np.load(od / "start.npy"); self.n = len(self.start) - 1
        self.lights = np.load(od / "lights.npy"); self.nl = np.load(od / "nlights.npy")
        self.pvis = np.load(od / "port_vis.npy", mmap_mode="r"); self.family = np.load(od / "family.npy")

    def pixels(self, j, sel=None):
        s, e = self.start[j], self.start[j + 1]
        out = {k: np.asarray(v[s:e]) for k, v in self.px.items()}
        return {k: v[sel] for k, v in out.items()} if sel is not None else out


class Ports:
    def __init__(self, dev):
        A = np.load(C.OUT1 / "anchors.npz")
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x)).to(dev, dt)  # noqa: E731
        L = lambda x: T(x, torch.int64)  # noqa: E731
        self.pos, self.nrm, self.area = T(A["pos"]), T(A["nrm"]), T(A["area"])
        self.mat, self.obj = L(A["mat"]), L(A["obj"]); self.nA = int(A["n_A"]); self.N = len(A["obj"])
        self.idx = [torch.arange(0, self.nA, device=dev), torch.arange(self.nA, self.N, device=dev)]
        self.nsamp = T(np.isfinite(A["spos"][..., 0]).sum(1).clip(1, None))
        self.nbr, self.nbrA, self.mid, self.midA = L(A["nbr"]), T(A["nbrA"]), L(A["mid"]), T(A["midA"])
        self.f2b, self.bpos, self.bnrm, self.barea, self.bmat, self.bobj = L(A["f2b"]), T(A["bpos"]), T(A["bnrm"]), T(A["barea"]), L(A["bmat"]), L(A["bobj"])
        self.f2c, self.cpos, self.cnrm, self.carea, self.cmat, self.cobj = L(A["f2c"]), T(A["cpos"]), T(A["cnrm"]), T(A["carea"]), L(A["cmat"]), L(A["cobj"])
        self.Nb, self.Nc = len(A["barea"]), len(A["carea"])
        td, bd = M.det_frame(self.nrm); self.detf = (td, bd)

    def local_mat(self, o):
        return self.mat[self.idx[o]] if o == 0 else torch.zeros_like(self.idx[1])


def light_dirs_E(L, x, n, vis):
    """Lights L [J, 12] (one image), points x [N, 3], normals n [N, 3], visible fraction vis [J, N] ->
    unit directions toward the lights [J, N, 3] and irradiance contributions [J, N]."""
    J = L.shape[0]
    typ = L[:, 0].long()
    d_sun = F.normalize(L[:, 1:4], dim=-1)[:, None].expand(J, x.shape[0], 3)
    v = L[:, None, 1:4] - x[None]; r2 = (v * v).sum(-1).clamp(min=1e-8); d_pt = v / r2.sqrt()[..., None]
    sun = (typ == 0)[:, None, None]
    w = torch.where(sun, d_sun, d_pt)
    cos = (w * n[None]).sum(-1).clamp(min=0)
    E = torch.where((typ == 0)[:, None], L[:, 8:9] * cos, L[:, 8:9] * cos / r2) * vis
    return w, E


def port_incident(P, o, L, nl, pv):
    """Per image of the step: total irradiance E [I, Np_o] and 19-bin incident h [I, Np_o, NB] at the ports of domain o."""
    ii = P.idx[o]; x = P.pos[ii]; n = P.nrm[ii]; td, bd = P.detf[0][ii], P.detf[1][ii]
    Es, Hs = [], []
    for k in range(L.shape[0]):
        J = int(nl[k])
        frac = pv[k, :J].float() / P.nsamp[ii][None]
        w, E = light_dirs_E(L[k, :J], x, n, frac)
        b = M.hemi_bin(torch.stack([(w * td[None]).sum(-1), (w * bd[None]).sum(-1), (w * n[None]).sum(-1)], -1))   # [J, Np]
        h = torch.zeros(x.shape[0], NB, device=x.device)
        h.scatter_add_(1, b.T, E.T)
        Es.append(E.sum(0)); Hs.append(h)
    return torch.stack(Es), torch.stack(Hs)


def source_tables(K, P, o, E, H, weight_n=32768, gen=None, exact=False):
    """Aggregated band (1.5 cm) and mid (8 cm) sources and the gamma estimate of domain o for the step's images."""
    ii = P.idx[o]; a = P.area[ii]
    I = E.shape[0]
    hB = torch.zeros(I, P.Nb, NB, device=E.device).index_add_(1, P.f2b[ii], H * a[None, :, None]) / P.barea[None, :, None].clamp(min=1e-12)
    Ec = torch.zeros(I, P.Nc, device=E.device).index_add_(1, P.f2c[ii], E * a[None]) / P.carea[None].clamp(min=1e-12)
    mat = P.local_mat(o)
    if exact:
        sidx = torch.arange(len(ii), device=E.device); sw = a
    else:
        pr = a * (E.mean(0) + 1e-3 * E.mean().clamp(min=1e-6)); pr = pr / pr.sum()
        n = min(weight_n, len(ii)); sidx = torch.multinomial(pr, n, replacement=True, generator=gen); sw = a[sidx] / (n * pr[sidx])
    _, q = K.p_q(K.feat(P.pos[ii][sidx], mat[sidx]))
    gamma = torch.einsum("s,smc,is->imc", sw, q, E[:, sidx])
    bm = torch.where(P.bobj == 0, P.bmat, 0) if o == 0 else torch.zeros_like(P.bmat)
    cm = torch.where(P.cobj == 0, P.cmat, 0) if o == 0 else torch.zeros_like(P.cmat)
    return {"hB": hB, "Ec": Ec, "gamma": gamma, "hs": K.src_desc(bm), "hc": K.src_desc(cm)}


def predict(K, P, tab, o, px, L, nl):
    canon, n, wo, port, im = px["pos"], px["nrm"], px["cam"], px["port"], px["im"]
    mat = P.mat[port] if o == 0 else torch.zeros_like(port)
    h = K.feat(canon, mat); rho, kap, alpha = K.bsdf_params(h); p, _ = K.p_q(h); hx = K.proj(h)
    out = (p * tab["gamma"][im]).sum(1)
    # K_loc on the image's lights (exact directions, the pixel's own visibility bits)
    Jm = int(nl.max())
    Li = L[im][:, :Jm]                                                      # [N, J, 12]
    bits = ((px["vbits"][:, None] >> torch.arange(Jm, device=n.device)[None]) & 1).float()
    bits = bits * (torch.arange(Jm, device=n.device)[None] < nl[im][:, None]).float()
    typ = Li[..., 0]
    v = Li[..., 1:4] - canon[:, None]; r2 = (v * v).sum(-1).clamp(min=1e-8)
    w = torch.where((typ == 0)[..., None], F.normalize(Li[..., 1:4], dim=-1), v / r2.sqrt()[..., None])
    cos = (w * n[:, None]).sum(-1).clamp(min=0)
    E = torch.where(typ == 0, Li[..., 8] * cos, Li[..., 8] * cos / r2) * bits
    out = out + K.loc_response(n, wo, rho, kap, alpha, w, E[..., None].expand(-1, -1, 3))
    # K_band over the full 5 cm ball (aggregated sources, represented area)
    nb = P.nbr[port]; use = (nb >= 0).float(); nbc = nb.clamp(min=0)
    W = K.band_w(hx, tab["hs"][nbc], n, P.bnrm[nbc], (P.bpos[nbc] - canon[:, None]) / C.R_LOCAL, wo)
    out = out + torch.einsum("nkbc,nkb->nc", W * (P.nbrA[port] * use)[..., None, None], tab["hB"][im[:, None], nbc])
    # K_mid over the full 30 cm ball
    mc = P.mid[port]; usem = (mc >= 0).float(); mcc = mc.clamp(min=0)
    Wm = K.mid_w(hx, tab["hc"][mcc], n, P.cnrm[mcc], (P.cpos[mcc] - canon[:, None]) / C.R_MID, wo)
    out = out + (Wm * (P.midA[port] * usem)[..., None] * tab["Ec"][im[:, None], mcc][..., None]).sum(1)
    return out


def predict_ckpt(K, P, tab, o, px, L, nl, chunk):
    """predict() over pixel chunks with activation checkpointing (amendment 1: K 96 / 128 representatives do not fit 12 GB in
    one pass). Same values and gradients as one predict() call; only memory changes."""
    n = len(px["port"])
    return torch.cat([checkpoint(predict, K, P, tab, o, {k: v[s:s + chunk] for k, v in px.items()}, L, nl, use_reentrant=False)
                      for s in range(0, n, chunk)])


def uniform_albedo_penalty(K, P, o, gen, n=32768):
    """Effective albedo pi I / E of sampled ports under a uniform unit radiance field (E = pi, bins by cosine-weighted
    solid angle), all four components; returns mean(relu(albedo - 0.98))."""
    ii = P.idx[o]; dev = P.pos.device
    s = ii[torch.randint(0, len(ii), (min(n, len(ii)),), device=dev, generator=gen)]
    mat = P.mat[s] if o == 0 else torch.zeros_like(s)
    h = K.feat(P.pos[s], mat); rho, kap, _ = K.bsdf_params(h); p, _ = K.p_q(h); hx = K.proj(h)
    E = math.pi
    loc = rho + kap.sum(1)                                                  # diffuse + glossy (lobe albedo <= 1)
    # global: gamma under uniform E over the whole domain (exact sum of q A over all ports, cheap enough on 32 k samples)
    ss = ii[torch.randint(0, len(ii), (32768,), device=dev, generator=gen)]
    _, q = K.p_q(K.feat(P.pos[ss], P.mat[ss] if o == 0 else torch.zeros_like(ss)))
    gam = (q * (P.area[ss] * len(ii) / len(ss))[:, None, None]).sum(0) * E
    glob = (p * gam[None]).sum(1) * math.pi / E
    cw = torch.tensor(_bin_weights(), device=dev)                           # cosine-weighted share of each bin
    nb = P.nbr[s]; use = (nb >= 0).float(); nbc = nb.clamp(min=0)
    bm = torch.where(P.bobj == 0, P.bmat, 0) if o == 0 else torch.zeros_like(P.bmat)
    W = K.band_w(hx, K.src_desc(bm)[nbc], P.nrm[s], P.bnrm[nbc], (P.bpos[nbc] - P.pos[s][:, None]) / C.R_LOCAL, P.nrm[s])
    band = torch.einsum("nkbc,nk,b->nc", W, P.nbrA[s] * use, cw * E) * math.pi / E
    mc = P.mid[s]; usem = (mc >= 0).float(); mcc = mc.clamp(min=0)
    cm = torch.where(P.cobj == 0, P.cmat, 0) if o == 0 else torch.zeros_like(P.cmat)
    Wm = K.mid_w(hx, K.src_desc(cm)[mcc], P.nrm[s], P.cnrm[mcc], (P.cpos[mcc] - P.pos[s][:, None]) / C.R_MID, P.nrm[s])
    mid = (Wm * (P.midA[s] * usem)[..., None]).sum(1) * E * math.pi / E
    alb = loc + glob + band + mid
    return F.relu(alb - 0.98).mean(), alb


def _bin_weights():
    rng = np.random.default_rng(0); u = rng.random((200000, 2)); r = np.sqrt(u[:, 0]); ph = 2 * np.pi * u[:, 1]
    d = np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(1 - r * r)], 1)
    b = C.hemi_bin_local(d); return (np.bincount(b, minlength=NB) / len(b)).astype(np.float32)


def to_t(d, dev):
    g = lambda k: torch.as_tensor(np.ascontiguousarray(d[k])).to(dev)  # noqa: E731
    return {"pos": g("pos").float(), "nrm": g("nrm").float(), "cam": g("cam").float(), "col": g("col").float(),
            "dom": g("dom").long(), "port": g("port").long(), "vbits": g("vbits").long()}


def batch_inputs(S, js, P, dev):
    L = torch.as_tensor(S.lights[js]).to(dev); nl = torch.as_tensor(S.nl[js]).to(dev).long()
    pv = torch.as_tensor(np.stack([np.asarray(S.pvis[j]) for j in js])).to(dev)
    inc = []
    for o in (0, 1):
        ii = P.idx[o]; E, H = port_incident(P, o, L, nl, pv[:, :, ii[0]:ii[-1] + 1])
        inc.append((E, H))
    return L, nl, inc


def make_kernels(dev):
    A = np.load(C.OUT1 / "anchors.npz"); pa = A["pos"][A["obj"] == 0]; pb = A["pos"][A["obj"] == 1]
    pad = lambda p, f: np.stack([p.min(0) - f * (p.max(0) - p.min(0)), p.max(0) + f * (p.max(0) - p.min(0))])  # noqa: E731
    return M.DomainKernel(pad(pa, 0.02), 8, m_rank=64).to(dev), M.DomainKernel(pad(pb, 0.05), 1, res=64, res_c=16, m_rank=8).to(dev)


@torch.no_grad()
def evaluate(Ks, P, S, dev, max_images=None):
    sq = {}; n = {}; l1 = 0.0; n3 = 0
    for j in range(S.n if max_images is None else min(S.n, max_images)):
        d = to_t(S.pixels(j), dev); d["im"] = torch.zeros_like(d["port"])
        L, nl, inc = batch_inputs(S, [j], P, dev)
        pred = torch.zeros_like(d["col"])
        for o in (0, 1):
            m = d["dom"] == o
            if not m.any():
                continue
            tab = source_tables(Ks[o], P, o, *inc[o], exact=True)
            px = {k: v[m] for k, v in d.items()}
            out = [predict(Ks[o], P, tab, o, {k: v[s:s + 8192] for k, v in px.items()}, L, nl) for s in range(0, int(m.sum()), 8192)]
            pred[m] = torch.cat(out)
        e = ((C.ldr3(pred.cpu().numpy()) - C.ldr3(d["col"].cpu().numpy())) ** 2).mean(1)
        fam = str(S.family[j])
        mat = np.where(d["dom"].cpu().numpy() == 1, 8, P.mat[d["port"]].cpu().numpy())
        for key, mm in (("all", np.ones(len(e), bool)), (f"family_{fam}", np.ones(len(e), bool)), ("yellow", mat == 0), ("glass", (mat == 2) | (mat == 4)),
                        ("reflective", np.isin(mat, [1, 3, 5, 6, 7])), ("box", mat == 8)):
            sq[key] = sq.get(key, 0.0) + float(e[mm].sum()); n[key] = n.get(key, 0) + int(mm.sum())
        l1 += float((torch.log1p(pred.clamp(min=0)) - torch.log1p(d["col"])).abs().sum()); n3 += pred.numel()
    return {"psnr": {k: float(10 * np.log10(n[k] / max(sq[k], 1e-12))) for k in sq if n[k]}, "px": n, "logl1": l1 / max(n3, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--imgs", type=int, default=4); ap.add_argument("--px", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--passivity", type=float, default=0.0); ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16384); ap.add_argument("--train-split", default="train"); ap.add_argument("--holdout-split", default="holdout")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda")
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    S = Split(a.train_split); Hd = Split(a.holdout_split); P = Ports(dev)
    KA, KB = make_kernels(dev); Ks = [KA, KB]
    opt = torch.optim.Adam(list(KA.parameters()) + list(KB.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-5)
    rec = {"args": vars(a), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "params_A": sum(p.numel() for p in KA.parameters()),
           "params_B": sum(p.numel() for p in KB.parameters()), "train_images": S.n, "holdout_images": Hd.n, "log": [], "evals": [],
           "anchors_sha16": C.sha16(C.OUT1 / "anchors.npz")}
    t0 = time.time(); losses = []
    for step in range(a.steps):
        js = rng.choice(S.n, a.imgs, replace=False)
        L, nl, inc = batch_inputs(S, js, P, dev)
        tabs = [source_tables(Ks[o], P, o, *inc[o], gen=gen) for o in (0, 1)]
        parts = []
        for k, j in enumerate(js):
            s, e = S.start[j], S.start[j + 1]
            sel = np.sort(rng.choice(e - s, min(a.px, e - s), replace=False))
            d = to_t(S.pixels(j, sel), dev); d["im"] = torch.full_like(d["port"], k); parts.append(d)
        d = {k: torch.cat([x[k] for x in parts]) for k in parts[0]}
        total = None; cnt = 0
        for o in (0, 1):
            m = d["dom"] == o
            if not m.any():
                continue
            px = {k: v[m] for k, v in d.items()}
            pred = predict_ckpt(Ks[o], P, tabs[o], o, px, L, nl, a.chunk)
            l_ = ((torch.log1p(pred.clamp(min=0)) - torch.log1p(px["col"])) ** 2).sum()
            total = l_ if total is None else total + l_; cnt += int(m.sum())
        loss = total / (3 * cnt)
        if a.passivity > 0:
            pen = sum(uniform_albedo_penalty(Ks[o], P, o, gen)[0] for o in (0, 1))
            loss = loss + a.passivity * pen
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if (step + 1) % (20 if a.max_steps else 200) == 0:
            torch.cuda.synchronize()
            rec["log"].append({"step": step + 1, "loss": float(np.mean(losses[-200:])), "s": time.time() - t0}); print(json.dumps(rec["log"][-1]), flush=True)
        if a.max_steps and step + 1 >= a.max_steps:
            break
        if (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            ev = evaluate(Ks, P, Hd, dev, max_images=40); ev["step"] = step + 1; rec["evals"].append(ev); print("eval", json.dumps(ev), flush=True)
            torch.save({"A": KA.state_dict(), "B": KB.state_dict(), "step": step + 1}, out / "checkpoint.pt")
    rec["train_s"] = time.time() - t0; rec["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.save({"A": KA.state_dict(), "B": KB.state_dict()}, out / "kernels.pt"); rec["digest"] = M.digest(KA, KB)
    ev = evaluate(Ks, P, Hd, dev); ev["step"] = "final"; rec["evals"].append(ev); print("final holdout", json.dumps(ev))
    with torch.no_grad():
        alb = [uniform_albedo_penalty(Ks[o], P, o, gen)[1].sum(1).cpu().numpy() / 3 for o in (0, 1)]
    rec["uniform_albedo"] = {o: {"q50_q90_q99": np.quantile(x, [0.5, 0.9, 0.99]).tolist(), "frac_gt_1": float((x > 1).mean())} for o, x in zip("AB", alb)}
    (out / "train.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
