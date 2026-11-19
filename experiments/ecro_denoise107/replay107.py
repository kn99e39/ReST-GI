"""Reproduce all WL105 frames, with an immutable paired input to each inference."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common107 import SPPS, VIEWS, read, sha, total, write
from train105 import D_aux
import model105 as model


def verify(baseline, anchor, output):
    from capture107 import admit

    _, manifest, count = admit(baseline, anchor)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model.FUSED = True
    nets = {}
    for arm in ("R1", "R2", "R3"):
        net = model.Reconstructor(arm).cuda()
        net.load_state_dict(torch.load(baseline / "models" / f"{arm}.pt", map_location="cuda"))
        nets[arm] = net.eval()
    maxima = {a: 0.0 for a in ("R0", *nets)}
    with torch.no_grad():
        for view in VIEWS:
            root = baseline / "data/eval" / view
            with np.load(root / "aux.npz") as a:
                aux = {k: a[k] for k in a.files}
            for spp in SPPS:
                guide = torch.from_numpy(D_aux(aux, spp))[None].cuda()
                guide_before = guide.clone()
                for rep in range(manifest["reps"]):
                    with np.load(root / f"noisy_spp{spp}_r{rep}.npz") as z:
                        comp_np = z["comp"]
                    comp = torch.from_numpy(comp_np)[None].cuda()
                    outs = {"R0": total(comp_np)}
                    for arm, net in nets.items():
                        y = net(comp, guide)[0]
                        if view == VIEWS[0] and spp == 4 and rep == 0:
                            assert torch.equal(y, net(comp, guide)[0]), arm
                        outs[arm] = y[0].cpu().numpy()
                    assert np.array_equal(comp_np, comp[0].cpu().numpy())
                    assert torch.equal(guide, guide_before)
                    with np.load(baseline / "eval" / f"{view}_spp{spp}_r{rep}.npz") as z:
                        for arm, y in outs.items():
                            ref = z[arm].astype(np.float64)
                            ratio = float(np.abs(y - ref).sum() / np.abs(ref).sum())
                            maxima[arm] = max(maxima[arm], ratio)
            print("replayed", view, flush=True)
    assert max(maxima.values()) <= 1e-6, maxima
    representative = baseline / "data/eval/V0"
    with np.load(representative / "aux.npz") as a:
        aux = torch.from_numpy(D_aux({k: a[k] for k in a.files}, 4))[None].cuda()
    with np.load(representative / "noisy_spp4_r0.npz") as n:
        comp = torch.from_numpy(n["comp"])[None].cuda()
    r3 = nets["R3"]
    with torch.no_grad():
        for _ in range(10):
            r3(comp, aux)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial = torch.cuda.memory_allocated()
        ms = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            r3(comp, aux)
            torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - start))
    o0_cost = {"latency_ms_median": float(np.median(ms)), "latency_ms_p95": float(np.quantile(ms, 0.95)),
               "peak_extra_mib": (torch.cuda.max_memory_allocated() - initial) / 2**20,
               "params": model.n_params(r3)}
    write(output, {"files_verified": count, "views": VIEWS, "relative_l1_max": maxima,
                   "input_immutable": True, "repeat_bitwise": True, "archived_cost": manifest["cost"],
                   "measured_O0_cost": o0_cost,
                   "anchor_sha256": sha(anchor), "device": torch.cuda.get_device_name(0)})


if __name__ == "__main__":
    verify(*(Path(x) for x in sys.argv[1:]))
