"""WL109 frame engine: original R0 frames (rt101.RT._frame_prog, unchanged), C1 frames, frozen timing programs, memory.

All frames return per-pixel component MEANS as float32 torch tensors [C, H, W] on the GPU (sum / spp, as WL105 frame_components).
R0 keys: e10 e11 e12 e20 e21 e22 hl0 hl1 hl2. C1 keys are sorted names from C1.keys(M, orig_hl).
Main-stream seed value = common109.launch_seed(seed) (splitmix64; see there), passed to the unchanged frame program.
"""
import random
import time

import numpy as np
import torch

from c1_109 import C1, inner_seed_value
from common109 import launch_seed

RES = 512


class Engine:
    def __init__(self, R, rt, Bd):
        self.R, self.rt, self.Bd = R, rt, Bd
        self.mi, self.dr = R.mi, R.dr
        self.c1 = C1(R, rt)
        self._frozen = {}

    # ------------------------------------------------------------------ seeds
    def main_seed(self, seed, k=0):
        return self.dr.opaque(self.mi.UInt64, launch_seed(seed, k))

    def in_seed(self, seed):
        return self.dr.opaque(self.mi.UInt64, inner_seed_value(seed))

    @staticmethod
    def cam(camera):
        return dict(camera, res=RES)

    def _to_torch(self, out, spp):
        self.dr.eval(out)
        t = torch.stack([o.torch() for o in out])
        if spp == 1:
            t = t.clone()
        return (t / spp).view(len(out), RES, RES)

    # ------------------------------------------------------------------ eager frames (data)
    def r0(self, camera, spp, seed, B=None):
        rt = self.rt
        out = rt._frame_prog(rt.scene, rt.tabs(), rt.light_args(), self.Bd if B is None else B, self.main_seed(seed),
                             (self.cam(camera), RES, spp))
        return self._to_torch(out, spp)

    def r0_blocks(self, camera, spp, seed, b=32, B=None):
        """High-spp R0 component means [9,H,W] float64 (numpy): spp / b launches of b spp, launch k seeded launch_seed(seed, k)."""
        rt = self.rt
        acc = None
        for k in range(spp // b):
            out = rt._frame_prog(rt.scene, rt.tabs(), rt.light_args(), self.Bd if B is None else B, self.main_seed(seed, k),
                                 (self.cam(camera), RES, b))
            self.dr.eval(out)
            t = torch.stack([o.torch().double() for o in out])
            acc = t if acc is None else acc + t
        return (acc / spp).view(9, RES, RES).cpu().numpy()

    def c1_frame(self, camera, spp, seed, M, orig_hl=True, replay=False, fixed_jitter=False, B=None):
        rt = self.rt
        out = self.c1.frame_prog(rt.scene, rt.tabs(), rt.light_args(), self.Bd if B is None else B, self.main_seed(seed),
                                 self.in_seed(seed), (self.cam(camera), RES, spp), M, orig_hl=orig_hl, replay=replay,
                                 fixed_jitter=fixed_jitter)
        return self._to_torch(out, spp)

    # ------------------------------------------------------------------ frozen programs (timing)
    def frozen(self, kind, view, camera, spp, M=0):
        key = (kind, view, spp, M)
        if key not in self._frozen:
            rt, c1, cam = self.rt, self.c1, self.cam(camera)
            if kind == "R0":
                fn = lambda scene, tab, lp, Bd, sd, si: rt._frame_prog(scene, tab, lp, Bd, sd, (cam, RES, spp))
            elif kind == "C1":
                fn = lambda scene, tab, lp, Bd, sd, si: c1.frame_prog(scene, tab, lp, Bd, sd, si, (cam, RES, spp), M)
            else:
                raise KeyError(kind)
            self._frozen[key] = self.R.maybe_freeze(fn)
        return self._frozen[key]

    def run_frozen(self, kind, view, camera, spp, seed, M=0, compose=False):
        rt = self.rt
        out = self.frozen(kind, view, camera, spp, M)(rt.scene, rt.tabs(), rt.light_args(), self.Bd, self.main_seed(seed),
                                                      self.in_seed(seed))
        self.dr.eval(out)
        if compose:
            t = torch.stack([o.torch() for o in out])
            img = (t[0:3] + t[3:6] + t[6:9]) * (1.0 / spp)
            return img
        return out

    def sync(self):
        self.R.sync()

    def timed(self, kind, view, camera, spp, seed, M=0, compose=False):
        self.sync()
        t0 = time.perf_counter()
        self.run_frozen(kind, view, camera, spp, seed, M, compose)
        self.sync()
        return 1e3 * (time.perf_counter() - t0)

    def watermark(self, kind, view, camera, spp, seed, M=0):
        """Peak Dr.Jit CUDA allocation (bytes) during one warm frame, after the malloc cache is flushed."""
        dr = self.dr
        self.sync()
        dr.flush_malloc_cache()
        dr.detail.malloc_clear_statistics()
        self.run_frozen(kind, view, camera, spp, seed, M)
        self.sync()
        return int(dr.detail.malloc_watermark(dr.detail.AllocType.Device))


def timing_schedule(slots, rounds, seed):
    """Randomized interleaved order of timing slots per round (frozen RNG)."""
    rs = random.Random(seed)
    order = []
    for r in range(rounds):
        s = list(slots)
        rs.shuffle(s)
        order += [(r, x) for x in s]
    return order


def comp_to_parts(comp, keys):
    """Component means [C,H,W] -> dict of named RGB parts (float64 torch)."""
    idx = {k: i for i, k in enumerate(keys)}
    def rgb(prefix):
        return torch.stack([comp[idx[f"{prefix}{c}"]] for c in range(3)]).double()
    parts = {"E1": rgb("e1"), "E2": rgb("e2")}
    parts["LOW"] = parts["E1"] + parts["E2"]
    if "hl0" in idx:
        parts["H"] = rgb("hl")
    if "ho0" in idx:
        parts["HO"] = rgb("ho")
    return parts
