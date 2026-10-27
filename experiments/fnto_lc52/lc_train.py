"""WL52 training of the learned C1 (GLT, protocol fnto_lc52).

Objective: per receiver, cross-entropy between the explicit 1024-gather hit distribution over [NULL, patches] (target mass on
FF0 = 0 patches moved to NULL) and the model's transfer distribution. Supervises the OPERATOR (geometry -> transfer), never an
illumination. AdamW 3e-4, wd 1e-4, 1k warm-up + cosine, bf16 autocast for encoder / pair head, batch = 4 configurations x 512
receivers (padded batched forward on GPU-resident data; amendment 1), seed 5201. Validation every 2,000 steps on the 200 validation configurations (all stored receivers): CE and the
relative L2 error of E = F Lbar against the explicit target for fixed random smooth patch radiance fields.

    python lc_train.py <data dir> <out dir> [steps=60000]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_data as LD  # noqa: E402
import lc_model as LM  # noqa: E402

DEV = torch.device("cuda")
KEYS = ("pcen", "pnrm", "parea", "pobj", "pos", "nrm", "patch_of", "robj")


def load(d, split):
    out = []
    for f in sorted(Path(d).glob(f"{split}_*.pt")):
        out += torch.load(f)
    return out


def to_dev(sh):
    g = {k: sh[k].to(DEV) for k in KEYS}
    for k in ("pobj", "patch_of", "robj"):
        g[k] = g[k].long()
    return g


def smooth_field(pcen, seed, nf=6):
    g = torch.Generator(device=pcen.device).manual_seed(seed); w = torch.randn(nf, 3, 3, generator=g, device=pcen.device) * 3.0
    ph = torch.rand(nf, 3, generator=g, device=pcen.device) * 2 * math.pi; a = torch.rand(nf, 3, generator=g, device=pcen.device)
    x = (torch.sin(torch.einsum("mi,fci->mfc", pcen, w) + ph[None]) * a[None]).sum(1)
    return torch.nn.functional.softplus(2 * x)


def loss_on(model, g, tgt, sub=None):
    if sub is not None:
        tgt = tgt[sub]; g = dict(g); g.update({k: g[k][sub] for k in ("pos", "nrm", "patch_of", "robj")})
    z = model.encode(g["pcen"], g["pnrm"], g["parea"])
    ffm = LM.ff0(g["pos"], g["nrm"], g["pcen"], g["pnrm"], g["parea"])
    F, Fn, P = model.transfer(z, g["pos"], g["nrm"], g["patch_of"], g["pcen"], g["pnrm"], g["parea"], g["pobj"], g["robj"], ffm=ffm)
    tp = tgt[:, :-1] * (ffm > 0); tn = 1 - tp.sum(1); T = torch.cat([tn[:, None], tp], 1)
    lp = torch.log(P.float().clamp(min=1e-12)); ce = -(T * lp).sum(1).mean()
    return ce, F, T


@torch.no_grad()
def validate(model, val):
    model.eval(); ces, errs, errs_prior = [], [], []
    for i, sh in enumerate(val):
        g = to_dev(sh); tgt = LD.dense_target(sh, DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, F, T = loss_on(model, g, tgt)
        ces.append(float(ce)); L = smooth_field(g["pcen"], 7000 + i); Et = math.pi * tgt[:, :-1] @ L; Ep = F.float() @ L
        errs.append(float((Ep - Et).norm() / Et.norm().clamp(min=1e-12)))
        ffm = LM.ff0(g["pos"], g["nrm"], g["pcen"], g["pnrm"], g["parea"]); errs_prior.append(float(((ffm @ L) - Et).norm() / Et.norm().clamp(min=1e-12)))
    model.train(); return float(np.mean(ces)), float(np.mean(errs)), float(np.mean(errs_prior))


class GPUData:
    """All training configurations resident on the GPU (patch / receiver tensors + dense targets, fp16) - amendment 1."""
    def __init__(self, shards):
        self.c = []
        for sh in shards:
            g = to_dev(sh); g["tgt"] = LD.dense_target(sh, DEV).half(); g["M"] = sh["M"]; self.c.append(g)

    def batch(self, idx, nsub, gen):
        cs = [self.c[i] for i in idx]; B = len(cs); M = max(c["M"] for c in cs); n = nsub
        P = {k: torch.zeros(B, M, w, device=DEV) for k, w in (("pcen", 3), ("pnrm", 3))}; parea = torch.zeros(B, M, device=DEV); pobj = torch.full((B, M), -1, dtype=torch.long, device=DEV)
        pmask = torch.zeros(B, M, dtype=torch.bool, device=DEV); R = {k: [] for k in ("pos", "nrm", "patch_of", "robj")}; T = torch.zeros(B, n, M + 1, device=DEV)
        for b, c in enumerate(cs):
            m = c["M"]; P["pcen"][b, :m] = c["pcen"]; P["pnrm"][b, :m] = c["pnrm"]; parea[b, :m] = c["parea"]; pobj[b, :m] = c["pobj"]; pmask[b, :m] = True
            sub = torch.randperm(len(c["pos"]), generator=gen, device=DEV)[:n]
            for k in R:
                R[k].append(c[k][sub])
            t = c["tgt"][sub].float(); T[b, :, :m] = t[:, :m]; T[b, :, M] = t[:, m]                 # NULL column moved to index M
        return P["pcen"], P["pnrm"], parea, pobj, pmask, {k: torch.stack(v) for k, v in R.items()}, T


def batched_loss(model, d, T):
    pcen, pnrm, parea, pobj, pmask, R, T = d[0], d[1], d[2], d[3], d[4], d[5], T
    z = model.encode_b(pcen, pnrm, parea, pmask)
    ffm = LM.ff0_b(R["pos"], R["nrm"], pcen, pnrm, parea)
    P = model.transfer_b(z, R["pos"], R["nrm"], R["patch_of"], pcen, pnrm, parea, pobj, R["robj"], ffm)        # [B, n, M + 1], column 0 = NULL
    tp = T[..., :-1] * (ffm > 0); tn = 1 - tp.sum(-1); Tt = torch.cat([tn[..., None], tp], -1)
    return -(Tt * torch.log(P.float().clamp(min=1e-12))).sum(-1).mean()


def main(data, out, steps=60000):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(5201); rng = np.random.default_rng(5201); gen = torch.Generator(device=DEV).manual_seed(5201)
    tr, val = load(data, "train"), load(data, "val"); print("train", len(tr), "val", len(val), flush=True)
    t0 = time.time(); D = GPUData(tr); del tr; print("gpu data", round(time.time() - t0, 1), "s", round(torch.cuda.memory_allocated() / 1e9, 2), "GB", flush=True)
    model = LM.GLT().to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log = {"params": LM.n_params(model), "steps": steps, "batch": "4 configs x 512 receivers (padded batched forward, amendment 1)", "curve": []}; best = 1e9; t0 = time.time()
    acc = torch.zeros((), device=DEV); run = []
    for step in range(steps):
        d = D.batch(rng.integers(len(D.c), size=4), 512, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = batched_loss(model, d, d[6])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); acc += loss.detach()
        if (step + 1) % 100 == 0:
            run.append(float(acc) / 100); acc.zero_()
        if (step + 1) % 2000 == 0 or step == 0:
            vce, verr, vprior = validate(model, val); e = {"step": step + 1, "train_ce_last2k": float(np.mean(run[-20:])) if run else None, "val_ce": vce, "val_rel_l2_smooth": verr,
                                                      "val_rel_l2_unoccluded_prior": vprior, "lr": sched.get_last_lr()[0], "wall_s": time.time() - t0,
                                                      "gpu_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
            log["curve"].append(e); print(json.dumps(e), flush=True)
            if vce < best:
                best = vce; torch.save(model.state_dict(), out / "ckpt_best.pt"); log["best"] = e
            torch.save(model.state_dict(), out / "ckpt_last.pt"); (out / "train.json").write_text(json.dumps(log, indent=1))
    log["wall_s"] = time.time() - t0; (out / "train.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60000)
