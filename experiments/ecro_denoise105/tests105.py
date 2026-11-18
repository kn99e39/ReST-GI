"""WL105 focused tests. python tests105.py [DATA MODELS]  (model-level tests need no data; data/model tests run when given)."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model105 as M  # noqa: E402


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)
    if not ok:
        raise SystemExit(1)


def model_tests(device):
    g = torch.Generator().manual_seed(1)
    comp = torch.rand(1, 9, 48, 48, generator=g).to(device) * 3
    aux = torch.rand(1, M.AUX, 48, 48, generator=g).to(device)
    low, high, total = M.features(comp, aux)
    check("LOW + HIGH = TOTAL", torch.equal(low + high, total))
    for arm in ("R1", "R2", "R3"):
        m = M.Reconstructor(arm).to(device).eval()
        before = comp.clone()
        with torch.no_grad():
            y1, _ = m(comp, aux)
            y2, _ = m(comp, aux)
        check(f"{arm} input unmodified", torch.equal(before, comp))
        check(f"{arm} deterministic inference", torch.equal(y1, y2))
        const = torch.full((1, 9, 48, 48), 0.7, device=device)
        with torch.no_grad():
            yc, _ = m(const, aux)
        exp = 3 * 0.7 if arm == "R1" else 2 * 0.7 + 0.7
        check(f"{arm} constant region exact", float((yc - exp).abs().max()) < 1e-5, float((yc - exp).abs().max()))
        hdr = comp * 500
        with torch.no_grad():
            yh, _ = m(hdr, aux)
        lo, hi, tot = M.features(hdr, aux)
        check(f"{arm} HDR preserved (finite, no clamp, bounded by local input range)",
              bool(torch.isfinite(yh).all()) and float(yh.max()) > 1.0 and float(yh.max()) <= float(tot.max()) * (2 if arm != "R1" else 1) + 1e-3,
              (float(yh.max()), float(tot.max())))
    if device == "cuda":
        img = torch.rand(2, 3, 67, 53, generator=g).to(device) * 5
        lg = torch.randn(2, M.K * M.K, 67, 53, generator=g).to(device) * 3
        M.FUSED = False
        with torch.no_grad():
            ref = M.apply_kernel(img, lg)
        from kpn_triton105 import apply_kernel_fused
        fused = apply_kernel_fused(img, lg)
        err = float(((fused - ref).abs() / ref.abs().clamp_min(1e-6)).max())
        check("fused kernel application == unfold reference", err < 1e-5, err)
    print("model tests ok")


def data_tests(data):
    man = json.loads((data / "run_manifest.json").read_text())
    seeds = {"train": set(), "val": set(), "eval": set()}
    for key in man["views"]:
        split, name = key.split("/")
        for f in (data / split / name).glob("noisy_*.npz"):
            seeds[split].add(int(np.load(f)["seed"]))
        seeds[split] |= set(man["views"][key]["target_seeds"])
    check("seed isolation (train/val/eval disjoint)", not (seeds["train"] & seeds["eval"]) and not (seeds["val"] & seeds["eval"]) and not (seeds["train"] & seeds["val"]),
          {k: len(v) for k, v in seeds.items()})
    check("frozen renderer gates", man["gates"].get("G1_replay", {"pass": True})["pass"] and man["gates"]["G3_determinism"] and man["gates"]["G2_composition_max_abs"] == 0)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_tests(dev)
    if len(sys.argv) > 1:
        data_tests(Path(sys.argv[1]).resolve())
    print("ALL TESTS PASSED")
