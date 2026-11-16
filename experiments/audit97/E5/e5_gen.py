"""WL97 E5: sub-steps of the ECRO-class generator (model86.ECRO.factors, OWN s8601) on CK2, N = 1,270 (own copy of the factor
construction split into steps, for timing only; the full production call is timed alongside as the control).
  python e5_gen.py <R_own> <CODE> <R92A> <out.json>"""
import json, os, sys, time
import numpy as np
R_OWN, CODE, R92A, OUT = sys.argv[1:5]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.join(CODE, 'experiments', 'ecro_mat94'))
import mat94 as MT  # noqa: E402
import fx92  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
import ecro86 as EC  # noqa: E402
import scene86 as S  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

PT, S_, A, FX, ctx, lt = MT.ctx_ck2(R_OWN)
MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
st = EC.build_state(ctx)
model = MT.load_frozen(R92A, 8601)
G = ctx.G; N = G.N; NB = S.NB; K = S.K
t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(np.asarray(a).astype(dt)), device=EC.dev)


class GS:
    pass


gs = GS(); gs.N = N
gs.cen, gs.nrm, gs.tan, gs.area, gs.mat = t(G.PC, np.float32), t(G.PN, np.float32), t(G.PT, np.float32), t(G.PAREA, np.float32), t(G.PMAT, np.int64)
gs.hit = t(st.hit, np.int64).view(N, K); gs.dbin = t(st.dbin, np.int64).view(N, K)
m = model


def tokens():
    Ft = m.form_factors(gs)
    opn = 1.0 - Ft.sum(1, keepdim=True)
    x = torch.cat([gs.cen * 0.5, gs.nrm, gs.tan, torch.log(gs.area).view(N, 1) + 3.0, m.mat_mlp(m.mat_states[gs.mat]), opn], 1)
    tk = m.inp(x)
    for l in range(m.rounds):
        mm = Ft @ m.msg[l](tk)
        gm = m.glb[l](tk).mean(0, keepdim=True).expand(N, -1)
        tk = tk + m.upd[l](torch.cat([tk, mm, gm], 1))
    return tk


def heads_bin(tk):
    be = torch.cat([m.bin_dirs, m.bin_emb], 1)
    tb = torch.cat([tk.view(N, 1, -1).expand(N, NB, -1), be.view(1, NB, 16).expand(N, NB, 16)], 2)
    c = Fn.softplus(m.head_c(tb)).view(N, NB, 3, m.R)
    a = Fn.softplus(m.head_a(tb)).view(N, NB, 3, m.R)
    return c, a


def heads_glob(tk):
    return Fn.softplus(m.head_Q(tk)).view(N, 3, m.RHO, m.R), Fn.softplus(m.head_P(tk)).view(N, 3, m.R, m.RHO)


def meas(fn, n=20):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    g = sum((e.time_range.end - e.time_range.start) for e in prof.events() if 'CUDA' in str(e.device_type)) / n / 1e3
    return {'wall_med_ms': float(np.median(ts)), 'wall_q25': float(np.percentile(ts, 25)), 'wall_q75': float(np.percentile(ts, 75)), 'gpu_ms': g, 'n': n}


with torch.no_grad():
    tk = tokens()
    f = model.factors(gs)
    c, a = heads_bin(tk); Qm, Pm = heads_glob(tk)
    out = {'N': N, 'R': m.R, 'RHO': m.RHO, 'rounds': m.rounds, 'apps': os.popen('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader').read().strip(),
           'uptime': os.popen('uptime').read().strip(),
           'split_equals_production': {'c': float((c - f['c']).abs().max()), 'a': float((a - f['a']).abs().max()), 'Q': float((Qm - f['Q']).abs().max()), 'P': float((Pm - f['P']).abs().max())},
           'factors_full_production': meas(lambda: model.factors(gs)), 'form_factors': meas(lambda: m.form_factors(gs)), 'tokens_incl_form_factors': meas(tokens),
           'heads_per_bin_c_a': meas(lambda: heads_bin(tk)), 'heads_global_Q_P': meas(lambda: heads_glob(tk)),
           'global_only_generator(tokens+Q,P)': meas(lambda: heads_glob(tokens()))}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=0))
