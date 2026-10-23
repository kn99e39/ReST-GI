"""FNTO Stage A training of the intrinsic optical-domain kernels K_A (Lego) and K_B (box) (8dna26 .venv-cu128).

    python fnto_train.py --out <dir> [--steps N] [--seed 0] [--max-steps N (smoke)]

Supervision: isolated-domain first-intrinsic-event probe renders (fnto_stagea_render / fnto_stagea_prep). Each image has
a fully known incident field (one delta sun x traced visibility, with spatial masks), so every training pair is
(incident field on the domain boundary) -> (outgoing radiance at observed pixels). No full-scene image, no free-space
interreflection and nothing of the held-out T1 edit is ever used. K_A and K_B own disjoint parameters and see only
pixels of their own domain.

Per step: IMGS images x PX pixels. For each image the source coefficients gamma = sum_s A_s q(s) E(s) are estimated
without bias by importance-sampling 32,768 ports per domain with probability ~ A_s x (mean incident irradiance of the
step's images); evaluation uses the exact sum over all ports. Loss: L2 on log(1 + radiance) (RNA's loss).
Adam 1e-3, cosine decay to 1e-5. Final weights are used. Holdout probes (unseen masks / suns / cameras / box placements)
are evaluated at the end and every --eval-every steps (monitoring only; nothing is selected on them).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import fnto_common as C
import fnto_model as M

D = C.OUT / "stagea"


class Split:
    def __init__(self, name):
        od = D / f"prep_{name}"
        self.name = name
        self.px = {k: np.load(od / f"px_{k}.npy", mmap_mode="r") for k in ("pos", "nrm", "cam", "col", "vis", "dom", "port", "pix")}
        self.start = np.load(od / "start.npy"); self.n = len(self.start) - 1
        self.portE = np.load(od / "portE.npy", mmap_mode="r"); self.portbin = np.load(od / "portbin.npy", mmap_mode="r")
        self.suns = np.load(od / "suns.npy")
        self.two = (od / "portE2.npy").exists()
        if self.two:
            self.portE2 = np.load(od / "portE2.npy", mmap_mode="r"); self.portbin2 = np.load(od / "portbin2.npy", mmap_mode="r")
            self.suns2 = np.load(od / "suns2.npy")
        self.ids = np.load(od / "ids.npy")

    def pixels(self, j, sel=None):
        s, e = self.start[j], self.start[j + 1]
        out = {k: np.asarray(v[s:e]) for k, v in self.px.items()}
        if sel is not None:
            out = {k: v[sel] for k, v in out.items()}
        return out


class Ports:
    """Configuration-invariant port tables on the GPU."""

    def __init__(self, dev):
        A = np.load(C.OUT / "anchors.npz")
        T = lambda x, dt=torch.float32: torch.as_tensor(x).to(dev, dt)  # noqa: E731
        self.pos, self.nrm, self.area = T(A["pos"]), T(A["nrm"]), T(A["area"])
        self.mat = T(A["mat"].astype(np.int64), torch.int64); self.obj = T(A["obj"].astype(np.int64), torch.int64)
        self.nbr = T(A["nbr"].astype(np.int64), torch.int64); self.nA = int(A["n_A"]); self.N = len(A["obj"])
        self.idx = [torch.arange(0, self.nA, device=dev), torch.arange(self.nA, self.N, device=dev)]
        self.f2c = T(A["f2c"].astype(np.int64), torch.int64); self.mid = T(A["mid"].astype(np.int64), torch.int64)
        self.cpos, self.cnrm, self.carea = T(A["cpos"]), T(A["cnrm"]), T(A["carea"])
        self.cmat = T(A["cmat"].astype(np.int64), torch.int64); self.cobj = T(A["cobj"].astype(np.int64), torch.int64); self.Nc = len(A["cobj"])

    def local_mat(self, o):
        return self.mat[self.idx[o]] if o == 0 else torch.zeros_like(self.idx[1])


def port_tables(K, P, o, weight=None, n_sample=32768, gen=None):
    """Source side of domain o. gamma = sum_s A_s q(s) E(s) is estimated by importance sampling ports with probability
    proportional to A_s * weight_s (weight = mean incident irradiance of the step's images; unbiased), or exactly over
    all ports when weight is None. Also the band-kernel source descriptors of every port (material only)."""
    ii = P.idx[o]
    mat = P.local_mat(o)
    cm = torch.where(P.cobj == 0, P.cmat, 0) if o == 0 else torch.zeros_like(P.cmat)    # other-domain entries unused
    tab = {"hs": K.src_desc(mat), "hc": K.src_desc(cm)}
    if weight is None:
        sidx = torch.arange(len(ii), device=ii.device); sw = P.area[ii]
    else:
        pr = P.area[ii] * (weight + 1e-3 * weight.mean().clamp(min=1e-6))
        pr = pr / pr.sum()
        n = min(n_sample, len(ii))
        sidx = torch.multinomial(pr, n, replacement=True, generator=gen)
        sw = P.area[ii][sidx] / (n * pr[sidx])
    h = K.feat(P.pos[ii][sidx], mat[sidx])
    _, q = K.p_q(h)
    tab.update({"sidx": sidx, "sw": sw, "q": q})
    return tab


def predict(K, P, tab, o, px, sun):
    """First-intrinsic-event prediction for pixels px (dict of tensors, px["im"] = index into the incident-field stack) of
    domain o. sun = list of incident fields, each (w [I, 3], E_ports [I, Np_dom], bin_ports [I, Np_dom], V_pixel [N]);
    the operator is applied to their SUM (it is linear)."""
    ii = P.idx[o]; off = ii[0]
    canon, n, wo, port, im = px["pos"], px["nrm"], px["cam"], px["port"], px["im"]
    mat = P.mat[port] if o == 0 else torch.zeros_like(port)
    h = K.feat(canon, mat)
    rho, kap, alpha = K.bsdf_params(h)
    p, _ = K.p_q(h)
    hx = K.proj(h)
    nb = P.nbr[port]; use = nb >= 0; nbg = nb.clamp(min=0); nbl = nbg - off
    W = K.band_w(hx, tab["hs"][nbl], n, P.nrm[nbg], (P.pos[nbg] - canon[:, None]) / C.R_LOCAL, wo)    # [N, k, 7, 3]
    As = P.area[nbg] * use
    mc = P.mid[port]; usem = mc >= 0; mcg = mc.clamp(min=0)
    Wm = K.mid_w(hx, tab["hc"][mcg], n, P.cnrm[mcg], (P.cpos[mcg] - canon[:, None]) / C.R_MID, wo) * (P.carea[mcg] * usem)[..., None]
    f2c = P.f2c[ii]
    out = torch.zeros_like(canon)
    for (w, E, b, Vp) in sun:
        Ec = torch.zeros(E.shape[0], P.Nc, device=E.device).index_add_(1, f2c, E * P.area[ii]) / P.carea   # [I, Nc]
        out = out + (Wm * Ec[im[:, None], mcg][..., None]).sum(1)
        wl = w[im]
        Li = (C.E_SUN * (n * wl).sum(-1).clamp(min=0) * Vp)[:, None, None].expand(-1, 1, 3)
        out = out + K.loc_response(n, wo, rho, kap, alpha, wl[:, None], Li)
        gamma = torch.einsum("s,smc,is->imc", tab["sw"], tab["q"], E[:, tab["sidx"]])               # [I, M, 3]
        out = out + (p * gamma[im]).sum(1)
        Wsel = torch.gather(W, 2, b[im[:, None], nbl][..., None, None].expand(-1, -1, 1, 3))[:, :, 0]  # [N, k, 3]
        out = out + (Wsel * (E[im[:, None], nbl] * As)[..., None]).sum(1)
    return out


def to_t(d, dev):
    g = lambda k: torch.as_tensor(np.ascontiguousarray(d[k])).to(dev)  # noqa: E731   (transfer native dtypes, convert on GPU)
    return {"pos": g("pos").float(), "nrm": g("nrm").float(), "cam": g("cam").float(), "col": g("col").float(),
            "vis": g("vis").float(), "port": g("port").long(), "dom": g("dom").long()}


def image_sun(S, js, P, o, dev, which=1):
    """Stacked incident fields of images js for domain o: (w [I, 3], E [I, Np_dom], bin [I, Np_dom])."""
    ii = P.idx[o].cpu().numpy(); lo, hi = int(ii[0]), int(ii[-1]) + 1
    sw, sE, sb = (S.suns, S.portE, S.portbin) if which == 1 else (S.suns2, S.portE2, S.portbin2)
    js = np.atleast_1d(js)
    return (torch.as_tensor(np.stack([sw[j] for j in js]), dtype=torch.float32, device=dev),
            torch.as_tensor(np.stack([np.asarray(sE[j, lo:hi]) for j in js])).to(dev).float(),          # fp16 transfer
            torch.as_tensor(np.stack([np.asarray(sb[j, lo:hi]) for j in js])).to(dev).long())


@torch.no_grad()
def evaluate(Ks, P, S, dev, max_images=None, keep=None):
    """Display-space and HDR metrics of the first-intrinsic-event prediction on a split (per domain and pooled)."""
    tabs = [port_tables(Ks[o], P, o) for o in (0, 1)]                       # exact quadrature at evaluation
    sq, l1, n3, per = 0.0, 0.0, 0, []
    by = {0: [0.0, 0], 1: [0.0, 0]}; kept = {}
    bym = np.zeros((9, 2)); byv = np.zeros((2, 2)); bymv = np.zeros((9, 2, 2))
    for j in range(S.n if max_images is None else min(S.n, max_images)):
        d = to_t(S.pixels(j), dev); pred = torch.zeros_like(d["col"])
        for o in (0, 1):
            m = d["dom"] == o
            if not m.any():
                continue
            px = {k: v[m] for k, v in d.items()}; px["im"] = torch.zeros_like(px["port"])
            w, E, b = image_sun(S, [j], P, o, dev)
            out = []
            for s in range(0, int(m.sum()), 1 << 14):
                sl = {k: v[s:s + (1 << 14)] for k, v in px.items()}
                out.append(predict(Ks[o], P, tabs[o], o, sl, [(w, E, b, sl["vis"])]))
            pred[m] = torch.cat(out)
        dl = (torch.as_tensor(C.ldr3(pred.cpu().numpy())) - torch.as_tensor(C.ldr3(d["col"].cpu().numpy()))) ** 2
        e = dl.mean(1).numpy(); sq += e.sum(); n3 += len(e); per.append(float(10 * np.log10(1 / max(e.mean(), 1e-12))))
        l1 += float((torch.log1p(pred.clamp(min=0)) - torch.log1p(d["col"])).abs().sum())
        dom = d["dom"].cpu().numpy()
        for o in (0, 1):
            by[o][0] += float(e[dom == o].sum()); by[o][1] += int((dom == o).sum())
        mat = np.where(dom == 1, 8, P.mat[d["port"]].cpu().numpy()); vv = d["vis"].cpu().numpy().astype(np.int64)
        np.add.at(bym[:, 0], mat, e); np.add.at(bym[:, 1], mat, 1); np.add.at(byv[:, 0], vv, e); np.add.at(byv[:, 1], vv, 1)
        np.add.at(bymv[:, :, 0], (mat, vv), e); np.add.at(bymv[:, :, 1], (mat, vv), 1)
        if keep is not None and j in keep:
            kept[j] = (pred.cpu().numpy(), d["col"].cpu().numpy(), np.asarray(S.pixels(j)["pix"]))
    rec = {"pooled_psnr": float(10 * np.log10(n3 / sq)), "view_mean_psnr": float(np.mean(per)), "logl1": l1 / (3 * n3),
           "psnr_A": float(10 * np.log10(by[0][1] / max(by[0][0], 1e-12))) if by[0][1] else None,
           "psnr_B": float(10 * np.log10(by[1][1] / max(by[1][0], 1e-12))) if by[1][1] else None, "images": len(per),
           "material_mse": {(C.MATERIALS[i] if i < 8 else "B"): (float(bym[i, 0] / bym[i, 1]) if bym[i, 1] else None) for i in range(9)},
           "material_px": {(C.MATERIALS[i] if i < 8 else "B"): int(bym[i, 1]) for i in range(9)},
           "shadowed_mse": float(byv[0, 0] / max(byv[0, 1], 1)), "lit_mse": float(byv[1, 0] / max(byv[1, 1], 1)),
           "yellow_shadowed_mse": float(bymv[0, 0, 0] / max(bymv[0, 0, 1], 1)), "yellow_shadowed_px": int(bymv[0, 0, 1])}
    return (rec, kept) if keep is not None else rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--imgs", type=int, default=4); ap.add_argument("--px", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--train-split", default="train"); ap.add_argument("--holdout-split", default="holdout")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda")
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    S = Split(a.train_split); H = Split(a.holdout_split); P = Ports(dev)
    KA, KB = M.make_kernels(dev); Ks = [KA, KB]
    params = list(KA.parameters()) + list(KB.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-5)
    rec = {"args": vars(a), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "params_A": sum(p.numel() for p in KA.parameters()), "params_B": sum(p.numel() for p in KB.parameters()),
           "train_images": S.n, "holdout_images": H.n, "log": [], "evals": []}
    t0 = time.time(); losses = []
    for step in range(a.steps):
        js = rng.choice(S.n, a.imgs, replace=False)
        suns = [image_sun(S, js, P, o, dev) for o in (0, 1)]
        tabs = [port_tables(Ks[o], P, o, weight=suns[o][1].mean(0), gen=gen) for o in (0, 1)]
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
            w, E, b = suns[o]
            pred = predict(Ks[o], P, tabs[o], o, px, [(w, E, b, px["vis"])])
            l = ((torch.log1p(pred.clamp(min=0)) - torch.log1p(px["col"])) ** 2).sum()
            total = l if total is None else total + l
            cnt += int(m.sum())
        loss = total / (3 * cnt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if (step + 1) % (20 if a.max_steps else 200) == 0:
            torch.cuda.synchronize()
            rec["log"].append({"step": step + 1, "loss": float(np.mean(losses[-200:])), "s": time.time() - t0, "lr": sched.get_last_lr()[0]})
            print(json.dumps(rec["log"][-1]), flush=True)
        if a.max_steps and step + 1 >= a.max_steps:
            break
        if (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            ev = evaluate(Ks, P, H, dev, max_images=40); ev["step"] = step + 1; rec["evals"].append(ev); print("eval", json.dumps(ev), flush=True)
            torch.save({"A": KA.state_dict(), "B": KB.state_dict(), "step": step + 1}, out / "checkpoint.pt")
    rec["train_s"] = time.time() - t0; rec["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.save({"A": KA.state_dict(), "B": KB.state_dict()}, out / "kernels.pt")
    rec["digest"] = M.digest(KA, KB)
    ev = evaluate(Ks, P, H, dev); ev["step"] = "final"; rec["evals"].append(ev); print("final holdout", json.dumps(ev))
    (out / "train.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
