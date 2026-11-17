"""WL101 common: WL100 case contexts / checkpoints, the FROZEN pre-optimization ECRO dynamic lifecycle (arm A),
the same-checkpoint full-state rebuild (arm C) and timing helpers.

Arm A calls the unchanged production functions (pt86 / pt100 / ecro86 / model86) in exactly the order of the WL86 Stage 4 /
WL97 E5 edit lifecycle: set_geom -> selective state update -> source encoding -> respond (GeoState + factors + apply + host copy)
-> table upload -> LOW + lookup frame. Nothing in this module changes a frozen source file.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'ecro_material100'))
sys.path.insert(0, str(EXP / 'ecro86'))
import common100 as C100  # noqa: E402
import pt100  # noqa: E402  (initialises mitsuba cuda_ad_rgb through pt86)
PT = pt100.PT
mi, dr = PT.mi, PT.dr
import torch  # noqa: E402
import ecro86 as EC  # noqa: E402
import model86 as M  # noqa: E402
S = PT.S
ROOT100 = EXP.parent / 'results' / 'ecro_material100'
dev = torch.device('cuda')
FROZEN_SOURCES = ['ecro86/pt86.py', 'ecro86/ecro86.py', 'ecro86/model86.py', 'ecro86/scene86.py',
                  'ecro_material100/pt100.py', 'ecro_material100/common100.py', 'fnto_compiled83/closure83_dr.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_hashes():
    return {f: sha(EXP / f) for f in FROZEN_SOURCES}


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


def configs(case):
    return json.loads((ROOT100 / 'configs.json').read_text())[case]


def load_model(case):
    ck = ROOT100 / case / 'model.pt'
    m = M.ECRO().to(dev)
    m.mat_states = torch.as_tensor(C100.materials(case), device=dev)
    m.load_state_dict(torch.load(ck, map_location=dev))
    return m.eval(), sha(ck)


def moved_boxes(g0, g1):
    return [b for b in 'AB' if json.dumps(g0[b], sort_keys=True) != json.dumps(g1[b], sort_keys=True)]


class Timer:
    """Synchronised part timers (wall) + Dr.Jit kernel time of the bracketed part."""

    def __init__(self):
        self.t = {}; self.g = {}

    def __call__(self, name, fn):
        sync(); dr.kernel_history()
        t0 = time.perf_counter()
        r = fn()
        sync()
        self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
        self.g[name] = self.g.get(name, 0.0) + sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        return r


# ------------------------------------------------------------------------------------------------ arm A: frozen lifecycle
class FrozenECRO:
    """Pre-optimization runtime, exactly the production call sequence."""

    def __init__(self, case, geom, model):
        self.case, self.model = case, model
        self.ctx = pt100.Ctx(geom, case)
        self.geom = geom
        self.st = EC.build_state(self.ctx)

    def edit(self, geom, lights, T, full_state=False):
        """Geometry edit (and the current light) -> table B (device Float) ready for the frame. Returns diagnostics."""
        mv = moved_boxes(self.geom, geom)
        T('geometry_update', lambda: self.ctx.set_geom(geom))
        if full_state:
            st = T('relation_update', lambda: EC.build_state(self.ctx)); nd = st.hit.size
        else:
            st, nd = T('relation_update', lambda: EC.update_state(self.ctx, self.st, mv))
        self.st, self.geom = st, geom
        lt = T('lights', lambda: PT.Lights(lights))
        s = T('source', lambda: self.ctx.source(lt))
        B, f = T('generator_apply', lambda: EC.respond(self.model, st, s))
        Bd = T('table_upload', lambda: PT.fl(B))
        return {'B': B, 'Bd': Bd, 'lt': lt, 's': s, 'dirty_rays': int(nd), 'moved': mv}

    def frame(self, lt, Bd, cam, res, spp, seed, T):
        acc, Tr = T('frame', lambda: self.ctx.render(lt, 'ER', cam, res, spp, seed, 'ecro', B=Bd, b=spp))
        return acc, Tr


def cam_of(case, res):
    return dict(C100.P['datasets'][case]['camera'], res=res)
