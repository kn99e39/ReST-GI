"""Development prototype (not evidence): convergence, active fraction and cost of the coverage-moment constraint on
the frozen R3 LOW logits of validation views (no evaluation view is read)."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ecro_denoise105"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model105 as M, data105 as D, tilt106 as T
from train105 import D_aux
dev = "cuda"
data = Path(sys.argv[1]); models = Path(sys.argv[2])
iters = int(sys.argv[3]) if len(sys.argv) > 3 else T.ITERS
m = M.Reconstructor("R3").to(dev); m.load_state_dict(torch.load(models / "R3.pt", map_location=dev)); m.eval()
for name in ("Q00", "Q03", "Q06"):
    base, target, _, nz = D.load_view(data / "val" / name, 1)
    aux = torch.from_numpy(D_aux(base, 4))[None].to(dev)
    comp = torch.from_numpy(nz[4][0])[None].to(dev)
    low, high, total = M.features(comp, aux)
    with torch.no_grad():
        zl = m.net_low(torch.cat([torch.log1p(low), aux], 1))
        for _ in range(3):
            T.constrain_logits(zl, aux[:, :9], iters)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        for _ in range(5):
            T.constrain_logits(zl, aux[:, :9], iters)
        torch.cuda.synchronize(); dt = 1e3 * (time.perf_counter() - t0) / 5
        st = {}
        T.constrain_logits(zl, aux[:, :9], iters, stats=st)
    q = torch.quantile(st["residual"], torch.tensor([0.5, 0.99, 0.999, 1.0], device=dev)).tolist()
    print(name, f"iters {iters} active {st['active_fraction']:.3f} tilted {st['tilted_fraction']:.4f} maxS {st['max_own_surfaces']} ms {dt:.1f} resid {q} peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
