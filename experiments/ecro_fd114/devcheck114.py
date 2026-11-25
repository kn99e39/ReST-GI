"""WL114 development check (throwaway seeds, never evidence): deep tracer identities and timing at small sizes."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss113'))
import common113 as C  # noqa: E402
import tracer114 as T  # noqa: E402

PT, mi, dr = T.PT, T.mi, T.dr
SEED = 91140000


def rel(a, b):
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30))


def main(what):
    geom, lc = C.fixture('F0'); light = PT.Lights(lc)
    ctx = T.Ctx114(geom, 'C')
    cam = dict(C.V0_CAM, res=64)
    out = {}
    if 'cam' in what:
        t0 = time.perf_counter(); f, _ = ctx.full_pass(light, cam, 64, 32, SEED, b=32); t1 = time.perf_counter()
        d, dg = ctx.deep_pass(light, cam, 64, 32, SEED, 24, b=32, compact=False); t2 = time.perf_counter()
        dc, dgc = ctx.deep_pass(light, cam, 64, 32, SEED, 24, b=32, compact=True); t3 = time.perf_counter()
        out['cam_e1e2h_vs_full'] = {k: rel(d[k], f[k]) for k in ('e1', 'e2', 'h')}
        out['cam_tail_vs_full'] = rel(d['c7_12'] + d['c13_24'], f['tail'])
        out['compact_vs_not'] = {k: rel(dc[k], d[k]) for k in d}
        out['compactions'] = dgc['compactions']
        out['times'] = [t1 - t0, t2 - t1, t3 - t2]
        d96, dg96 = ctx.deep_pass(light, cam, 64, 32, SEED, 192, b=32, compact=True)
        out['deep192_bucket_lum'] = {k: float(T.lum(v).mean()) for k, v in d96.items()}
        out['deep192_cap'] = {k: dg96[k] for k in ('cap_alive', 'cap_throughput_lum', 'compactions')}
        rr, dgr = ctx.deep_pass(light, cam, 64, 32, SEED, 192, b=32, rr_from=24, compact=True)
        out['rr_bucket_lum'] = {k: float(T.lum(v).mean()) for k, v in rr.items()}
        out['rr_e_identical'] = {k: rel(rr[k], d96[k]) for k in ('e1', 'e2', 'h', 'c7_12', 'c13_24')}
    if 'con' in what:
        t0 = time.perf_counter(); a4 = ctx.construct4(light, 8, SEED + 1, b=8); t1 = time.perf_counter()
        ad, dg = ctx.construct_deep(light, 8, SEED + 1, 5, b=8); t2 = time.perf_counter()
        out['con_w'] = rel(ad['U_w'], a4['U_w'])
        out['con_f2'] = rel(sum(ad[f'U_f2{c}'] for c in range(3)), sum(a4[f'U_f2{c}'] for c in range(3)))
        out['con_fh'] = rel(sum(ad[f'U_fh{c}'] for c in range(3)), sum(a4[f'U_fh{c}'] for c in range(3)))
        out['con_cnt_disagree'] = float(ad['cnt_disagree'].sum())
        ad23, dg23 = ctx.construct_deep(light, 8, SEED + 1, 23, b=8); t3 = time.perf_counter()
        out['con23_f2fh_vs_5'] = rel(sum(ad23[f'U_{n}{c}'] for n in ('f2', 'fh') for c in range(3)), sum(ad[f'U_{n}{c}'] for n in ('f2', 'fh') for c in range(3)))
        out['con23_deep_over_d6'] = float(sum(ad23[f'U_{n}{c}'].sum() for n in T.DEEP for c in range(3) if f'U_{n}{c}' in ad23) /
                                          sum(ad23[f'U_{n}{c}'].sum() for n in ('f2', 'fh') for c in range(3)))
        out['con_times'] = [t1 - t0, t2 - t1, t3 - t2]
        out['con23_diag'] = dg23
    print(json.dumps(out, indent=1, default=float), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or ['cam', 'con'])
    import os; sys.stdout.flush(); os._exit(0)
