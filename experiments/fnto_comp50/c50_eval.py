"""WL50 Gates 1 / 3 / 4 - the unchanged r1_eval protocol (K_only first event, S_T0 coupled T0, D_stale / E_upd / E_relfull
T1, signed change, spectral radius) with the Yellow band + mid of K_k44P replaced by an arm (c50_common.ArmState /
arm_render_pixels); H = no arm (ArmState with keep = 1, identical to production - tested).

    python c50_eval.py <cond: H | L0 | L1> <ckpt or -> <out dir>
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c50_common as CC  # noqa: E402
import r1_common as C  # noqa: E402
import r1_eval as RE  # noqa: E402


class A:
    pass


def main(cond, ckpt, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t0 = time.time()
    os.environ["R1_EVAL_VIS"] = "eval_vis_k44.npz"
    arm = None; info = {"cond": cond, "ckpt": ckpt}
    if cond in ("L0", "L1"):
        geo = CC.Geo(dev); net = CC.load_net(ckpt, dev); arm = CC.ArmOp(cond, net, geo, CC.yellow_ports())
        info.update({"arm_ports": int(len(arm.ports)), "arm_K": arm.K, "arm_weights_nonzero": arm.n_elements, "arm_build_s": arm.t_build_s})
    CC.activate(arm); torch.cuda.reset_peak_memory_stats()
    a = A(); a.kernels = str(C.OUT1 / "K_k44P/kernels.pt"); a.out = str(out); t1 = time.time()
    RE.run(a)
    info.update({"eval_wall_s": time.time() - t1, "total_wall_s": time.time() - t0, "arm_apply": dict(CC.TIMING), "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
                 "gpu": torch.cuda.get_device_name(0)})
    (out / "cond.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info))


if __name__ == "__main__":
    main(*sys.argv[1:4])
