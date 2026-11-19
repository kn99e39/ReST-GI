"""WL107 O2 diagnostic, matched to the WL105 R3 training recipe."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from common107 import P, PROTOCOL, SPPS, VIEWS, read, sha, write
import data105 as D
import model105 as M
from train105 import D_aux, validate


def signed_weights(logits):
    """The sole relaxation: a zero-sum signed residual to R3's weights."""
    return logits.softmax(dim=1) + (logits - logits.mean(dim=1, keepdim=True)) / (M.K * M.K)


def apply_signed(image, logits):
    r = M.K // 2
    batch, channels, height, width = image.shape
    w = signed_weights(logits)
    if not torch.is_grad_enabled() and height == width == 512:
        # Work in row strips to bound diagnostic inference's temporary workspace.
        pad = F.pad(image, (r, r, r, r), mode="reflect")
        out = []
        for start in range(0, height, 64):
            stop = min(start + 64, height)
            patch = F.unfold(pad[:, :, start:stop + 2 * r], M.K).view(batch, channels, M.K ** 2, stop - start, width)
            out.append((patch * w[:, None, :, start:stop]).sum(2))
        return torch.cat(out, dim=2)
    patch = F.unfold(F.pad(image, (r, r, r, r), mode="reflect"), M.K)
    return (patch.view(batch, channels, M.K ** 2, height, width) * w[:, None]).sum(2)


class SignedDiagnostic(M.Reconstructor):
    def __init__(self):
        super().__init__("R3")

    def forward(self, comp, aux):
        low, high, _ = M.features(comp, aux)
        lo = apply_signed(low, self.net_low(torch.cat((torch.log1p(low), aux), 1)))
        hi = apply_signed(high, self.net_high(torch.cat((torch.log1p(high), aux), 1)))
        return lo + hi, {"low": lo, "high": hi}


def train(data, out, admit, trust):
    assert read(admit)["repeat_bitwise"], "Frozen replay must pass before O2"
    assert read(trust)["preliminary_pass"], "Preliminary reference precision gate"
    out.mkdir(parents=True, exist_ok=False)
    seed = P["seeds"]["training"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dm = read(data / "run_manifest.json")
    names = {split: sorted(k.split("/")[1] for k in dm["views"] if k.startswith(split + "/"))
             for split in ("train", "val", "eval")}
    assert tuple(map(len, names.values())) == (40, 8, 6)
    assert set(names["eval"]) == set(VIEWS)
    assert not (set(names["train"]) & set(names["val"])) and not (set(names["train"]) & set(names["eval"]))
    ts = D.TrainSet(data / "train", names["train"], 4, "cuda")
    val = []
    for n in names["val"]:
        base, target, _, noisy = D.load_view(data / "val" / n, 4)
        val.append((base, target, noisy))
    net = SignedDiagnostic().cuda()
    assert M.n_params(net) == 1055624
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000, eta_min=1e-5)
    stamp = time.monotonic()
    log = {"seed": seed, "params": M.n_params(net), "loss": [], "val": [],
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "data_manifest": sha(data / "run_manifest.json"), "protocol": sha(PROTOCOL)}
    for it in range(20000):
        if time.monotonic() - stamp > 1800:
            raise TimeoutError("O2 training budget exceeded")
        comp, tgt, aux = ts.batch(gen, 16, 128)
        _, pred = net(comp, aux)
        loss = M.rel_mse(pred["low"], tgt[:, :3] + tgt[:, 3:6]) + M.rel_mse(pred["high"], tgt[:, 6:9])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0:
            log["loss"].append([it, float(loss)])
            assert np.isfinite(log["loss"][-1][-1])
        if (it + 1) % 2000 == 0:
            vals = validate(net, val, "cuda")
            log["val"].append({"iter": it + 1, "RMSE": vals})
            print("O2", it + 1, log["loss"][-1][-1], vals, flush=True)
            write(out / "train.json", log)
    weights = out / "O2.pt"
    torch.save(net.state_dict(), weights)
    write(out / "train.json", {**log, "weights_sha256": sha(weights), "seconds": time.monotonic() - stamp})


def evaluate(data, weights, out, trust):
    assert read(trust)["preliminary_pass"], "Preliminary reference precision gate"
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    net = SignedDiagnostic().cuda().eval()
    net.load_state_dict(torch.load(weights, map_location="cuda"))
    manifest = {"weights_sha256": sha(weights), "files": {}, "params": M.n_params(net)}
    with torch.no_grad():
        for view in VIEWS:
            source = data / "eval" / view
            with np.load(source / "aux.npz") as f:
                base = {k: f[k] for k in f.files}
            for spp in SPPS:
                aux = torch.from_numpy(D_aux(base, spp))[None].cuda()
                old_aux = aux.clone()
                for rep in range(8):
                    f = f"noisy_spp{spp}_r{rep}.npz"
                    with np.load(source / f) as z:
                        nz = z["comp"]
                    comp = torch.from_numpy(nz)[None].cuda()
                    y, parts = net(comp, aux)
                    if view == VIEWS[0] and spp == 4 and rep == 0:
                        assert torch.equal(y, net(comp, aux)[0])
                    assert torch.equal(aux, old_aux) and np.array_equal(comp[0].cpu().numpy(), nz)
                    pred = {k: v[0].cpu().numpy() for k, v in parts.items()}
                    assert np.isfinite(y.cpu().numpy()).all()
                    name = f"{view}_spp{spp}_r{rep}.npz"
                    np.savez_compressed(out / name, O2=y[0].cpu().numpy(), **pred)
                    manifest["files"][name] = sha(out / name)
            print("O2 evaluated", view, flush=True)
        for _ in range(10):
            net(comp, aux)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial = torch.cuda.memory_allocated()
        ms = []
        for _ in range(50):
            start = time.perf_counter()
            net(comp, aux)
            torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - start))
        manifest["diagnostic_latency_ms_median"] = float(np.median(ms))
        manifest["diagnostic_peak_extra_mib"] = (torch.cuda.max_memory_allocated() - initial) / 2**20
    write(out / "manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("train")
    for field in ("data", "output", "replay", "reference"):
        t.add_argument(field, type=Path)
    e = sub.add_parser("eval")
    for field in ("data", "weights", "output", "reference"):
        e.add_argument(field, type=Path)
    a = p.parse_args()
    if a.stage == "train":
        train(a.data, a.output, a.replay, a.reference)
    else:
        evaluate(a.data, a.weights, a.output, a.reference)
