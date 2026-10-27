"""WL52 learned-C supervision (protocol fnto_lc52). Per geometry configuration: 1024 explicit cosine gathers per receiver
(independent seed) -> the receiver's hit distribution over source patches + NULL. Training / validation store R random
receivers per configuration and their sparse hit distribution over [patches, NULL] (CSR; column M = NULL). Candidates are
every patch with positive analytic FF0 (computed at training / inference time); target mass on FF0 = 0 patches (< 1 %,
measured) is moved to NULL there.

    python lc_data.py kselect <out dir>            the (failed) top-k coverage rule on 100 TRAINING configurations, kept as record
    python lc_data.py make <out dir> <split>       shards for train / val
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_model as LM  # noqa: E402
import lc_scene as LS  # noqa: E402
import lc_trace as TR  # noqa: E402

SPLITS = {"train": (100000, 3000), "val": (200000, 200), "test_iid": (300000, 120)}
R_REC = 1024
K_TARGET = 1024


def split_seeds(name):
    s0, n = SPLITS[name]; return list(range(s0, s0 + n))


def config(seed):
    return LS.sample_config(seed)


@torch.no_grad()
def make_one(cfg, k, dev, R=R_REC, seed_off=0):
    S = LS.build(cfg); G = TR.Geo(S, dev)
    rec = torch.as_tensor(np.sort(np.random.default_rng(cfg["seed"] + 77).choice(S.N, min(R, S.N), replace=False)), device=dev) if R else torch.arange(S.N, device=dev)
    H, _ = TR.gather(G, K_TARGET, 10_000_019 + cfg["seed"] + seed_off, ports=rec); cnt = TR.patch_target(G, H) / K_TARGET    # [R, M + 1]
    out = {"pcen": G.pcen.cpu(), "pnrm": G.pnrm.cpu(), "parea": G.parea.cpu(), "pobj": G.pobj.cpu().int(), "pos": G.pos[rec].cpu(), "nrm": G.nrm[rec].cpu(),
           "patch_of": G.patch[rec].cpu().int(), "robj": G.obj[rec].cpu().int(), "seed": cfg["seed"], "N": S.N, "M": S.M}
    if k:
        cand, ff = LM.candidates(G.pos[rec], G.nrm[rec], G.pcen, G.pnrm, G.parea, k); pc = torch.gather(cnt[:, :-1], 1, cand)
        out["coverage"] = float((pc.sum() / (1 - cnt[:, -1]).sum().clamp(min=1e-9)).cpu())
    nz = cnt.nonzero(); out["crow"] = torch.searchsorted(nz[:, 0].contiguous(), torch.arange(len(rec) + 1, device=dev)).cpu().int()
    out["col"] = nz[:, 1].cpu().short(); out["val"] = cnt[nz[:, 0], nz[:, 1]].cpu().half()
    return out


def dense_target(sh, dev):
    """CSR -> dense [R, M + 1] on dev."""
    R = len(sh["pos"]); crow = sh["crow"].long(); rows = torch.repeat_interleave(torch.arange(R), crow[1:] - crow[:-1])
    t = torch.zeros(R, sh["M"] + 1, device=dev); t[rows.to(dev), sh["col"].long().to(dev)] = sh["val"].float().to(dev); return t


def kselect(out, dev):
    seeds = split_seeds("train")[:100]; res = {}
    for k in (64, 128, 192, 256):
        c = [make_one(config(s), k, dev)["coverage"] for s in seeds]; res[k] = {"mean": float(np.mean(c)), "p05": float(np.percentile(c, 5)), "min": float(np.min(c))}
        print(k, res[k], flush=True)
    chosen = next((k for k in (64, 128, 192, 256) if res[k]["mean"] >= 0.99), None)
    (Path(out) / "kselect.json").write_text(json.dumps({"rule": "smallest k in {64,128,192,256} with mean non-NULL target mass coverage >= 0.99 on train seeds 100000-100099", "coverage": res, "k": chosen,
                                                       "outcome": "no k met the rule -> candidate set = every patch with positive analytic FF0 (dense; coverage measured separately)"}, indent=1))
    print("k =", chosen)


def make(out, split, dev, k=0):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); seeds = split_seeds(split); t0 = time.time(); shard, idx = [], 0
    for i, s in enumerate(seeds):
        shard.append(make_one(config(s), k, dev))
        if len(shard) == 500 or i == len(seeds) - 1:
            torch.save(shard, out / f"{split}_{idx:02d}.pt"); idx += 1; shard = []; print(split, i + 1, round(time.time() - t0, 1), flush=True)
    (out / f"{split}_meta.json").write_text(json.dumps({"split": split, "seeds": [seeds[0], seeds[-1]], "n": len(seeds), "R": R_REC, "gathers": K_TARGET, "s": time.time() - t0}))


if __name__ == "__main__":
    dev = torch.device("cuda"); Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "kselect":
        kselect(sys.argv[2], dev)
    else:
        make(sys.argv[2], sys.argv[3], dev)
