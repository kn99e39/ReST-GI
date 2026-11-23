"""WL110 deterministic rigid-body animation (numpy only).

The pose is a pure function of simulation time t (seconds). Every pose lies on the straight centre / yaw interpolation segment
between two WL100 held-out test configurations, which is exactly the segment WL101's V_SSS_ANIM workload sampled at 17 poses
(k / 16). The live animation evaluates the same segment continuously through a smooth ping-pong

    u(t) = 0.5 - 0.5 cos(2 pi t / period),   geom(t) = lerp(cfg_from, cfg_to, u(t)),

so u(0) = 0 (cfg_from), u(period / 2) = 1 (cfg_to), and velocity is zero at both ends. With quantize = Q > 0, u is rounded to the
grid k / Q (Q = 16 reproduces the V_SSS_ANIM poses bit for bit, using seq101._lerp_geom unchanged).

Light and camera are fixed (light 0 of test configuration 0, the WL101 rule for geometry-only edits; WL100 dataset camera).
"""
import hashlib, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'ecro_dyn101'))
import seq101 as Q  # noqa: E402  (WL101 workloads; numpy only)

# case -> (from test config, to test config). sss: the WL101 V_SSS_ANIM segment (cfg 2 -> cfg 1). near: consecutive configs of
# the WL101 V_NEAR_TOUR (cfg 0 -> cfg 1); its intermediate poses were never used by WL101 (secondary, descriptive mode only).
SEGMENTS = {'sss': (2, 1), 'near': (0, 1)}


def geom_hash(g):
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()[:16]


class Animation:
    def __init__(self, case, period_s=6.0, quantize=0):
        rows = Q._rows(case)
        a, b = SEGMENTS[case]
        self.case, self.period, self.quantize = case, float(period_s), int(quantize)
        self.g0, self.g1 = rows[a]['geom'], rows[b]['geom']
        self.light = rows[0]['lights'][0]
        self.segment = {'from_test_config': a, 'to_test_config': b}

    def u(self, t):
        u = 0.5 - 0.5 * math.cos(2.0 * math.pi * t / self.period)
        if self.quantize:
            u = round(u * self.quantize) / self.quantize
        return u

    def pose(self, t):
        u = self.u(t)
        if self.quantize:          # identical floating-point path to seq101 V_SSS_ANIM (k / 16)
            return Q._lerp_geom(self.g0, self.g1, round(u * self.quantize) / self.quantize), u
        return Q._lerp_geom(self.g0, self.g1, u), u

    def describe(self):
        return {'case': self.case, 'segment': self.segment, 'period_s': self.period, 'quantize': self.quantize,
                'function': 'u(t) = 0.5 - 0.5 cos(2 pi t / period); geom = seq101._lerp_geom(cfg_from, cfg_to, u) (centre and yaw linear)',
                'from_geom': self.g0, 'to_geom': self.g1, 'light': self.light}


def segment_validity(case, n=257):
    """scene86.geom_ok (room bounds, sphere clearance, box-box clearance; the WL86 sampling rule) on n poses of the segment.
    Imports scene86 lazily (it needs the Mitsuba stack only through scene86's module constants)."""
    sys.path.insert(0, str(EXP / 'ecro86'))
    import scene86 as S
    an = Animation(case)
    oks = [bool(S.geom_ok(Q._lerp_geom(an.g0, an.g1, k / (n - 1)))) for k in range(n)]
    return {'poses': n, 'geom_ok_all': all(oks), 'geom_ok_fraction': sum(oks) / n,
            'rule': 'scene86.geom_ok (WL86 training-geometry rule: box corners in room bounds, centre clearance from the sphere, box-box clearance)'}


def v_sss_anim_poses():
    """WL101 V_SSS_ANIM event list (geom, light, pose id) after its warm-up, for replay against the archived run."""
    case, start, ev = Q.workload('V_SSS_ANIM')
    return case, start, ev
