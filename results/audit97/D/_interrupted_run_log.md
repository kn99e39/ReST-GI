# Track D: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List audit-relevant directories] ls docs/audits 2>/dev/null; ls docs/worklogs | sed -n '/8[6-9]_\|9[0-9]/p'; ls experiments/ecro_util96a experiments/gm1_c3urx1_91 experiments/gm1_explicit87 experiments/ecro86; ls results/ecro_util96a results/gm1_c3urx1_91 results/gm1_explicit87 results/ecro86 results/wl96_calc 2>/dev/null | head -150
  -> 86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
88_WORKLOG_GM1_DIRECTIONAL_RELATION_FEASIBILITY.md
89_WORKLOG_GM1_SENDER_RELATION_QUALIFICATION_AND_DISCRIMINATION.md
90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md
91_WORKLOG_GM1_C3URX1_HELDOUT_VALIDATION_ ...[cut]... riments/gm1_explicit87:
__pycache__
analyze87.py
devchk87.py
econ87.py
fx87.py
protocol
pt87.py
quad87.py
run87.py
run87.sh
tests87.py
viz87.py
xport87.py
results/ecro86:
chain.status
chain.status.run1
chain_frozen_run1.status
code
results
watch.out
watch_run1.out

results/ecro_util96a:
T96A.npz
chain.status
cost_s1.0.json
final.json
gpu.txt
s0.json
s1.json
tails.json
tails.npz

results/gm1_c3urx1_91:
chain.status
gpu_trace.csv
results

results/gm1_explicit87:
chain.status
code
gpu_util_trace.txt
results

results/wl96_calc:
ck2_support.py
ck2_support_E.npy
ring_alloc.json
ring_alloc.py
scratch
RUN [List audit-relevant directories] cd /c/Projects/ReST-GI && ls docs/audits 2>/dev/null; ls docs/worklogs | sed -n '/8[6-9]_\|9[0-9]/p'; ls experiments/ecro_util96a experiments/gm1_c3urx1_91 experiments/gm1_explicit87 experiments/ecro86; ls results/ecro_util96a results/gm1_c3urx1_91 results/gm1_explicit87 results/ecro86 results/wl96_calc 2>/dev/null | head -150
RUN [Inspect cost artifacts and code sizes] wc -l experiments/ecro_util96a/w96a.py experiments/gm1_c3urx1_91/e2e91.py experiments/gm1_explicit87/econ87.py experiments/ecro86/econ86.py && cat results/ecro_util96a/cost_s1.0.json | head -c 6000; echo; cat results/ecro_util96a/gpu.txt | head -30; ls results/wl96_calc/scratch; ls results/gm1_c3urx1_91/results results/gm1_explicit87/results results/ecro86/results | head -120
  -> 819 experiments/ecro_util96a/w96a.py
  466 experiments/gm1_c3urx1_91/e2e91.py
  201 experiments/gm1_explicit87/econ87.py
  179 experiments/ecro86/econ86.py
 1665 total
{
 "s": 1.0,
 "N": 1270,
 "gpu": "NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB",
 "foreign_before": [],
 "frame_spp": 64,
 "res" ...[cut]...  %, 3546 MiB
17:07:03 87 %, 3546 MiB
17:07:08 83 %, 3546 MiB
17:07:13 97 %, 3546 MiB
17:07:18 98 %, 3546 MiB
17:07:23 18 %, 2660 MiB
17:07:28 75 %, 2660 MiB
critic_cost
critic_phys
d1
d2
d3
d4
results/ecro86/results:
amend1
contracts
econ
final.json
logs
masks
models
qualify.json
qualify_frozen.json
renders
stage1.json
tests
viz

results/gm1_c3urx1_91/results:
econ
final91.json
lists
logs
tables
tests91.json
viz
w_choice.json

results/gm1_explicit87/results:
e3_admission.json
e3_choice.json
econ
final87.json
lists
logs
masks
renders
stage1_87.json
stage2_87.json
stage3_87.json
tables
tests
viz
RUN [Inspect cost artifacts and code sizes] cd /c/Projects/ReST-GI && wc -l experiments/ecro_util96a/w96a.py experiments/gm1_c3urx1_91/e2e91.py experiments/gm1_explicit87/econ87.py experiments/ecro86/econ86.py && cat results/ecro_util96a/cost_s1.0.json | head -c 6000; echo; cat results/ecro_util96a/gpu.txt | head -30; ls results/wl96_calc/scratch; ls results/gm1_c3urx1_91/results results/gm1_explicit87/results results/ecro86/results | head -120
RUN [Show remaining cost json and code outline] python -c "
import json
d=json.load(open('results/ecro_util96a/cost_s1.0.json'))
print(json.dumps({k:v for k,v in d.items() if k not in ('arms',)},indent=1)[:5000])
"; grep -n "def \|^# \|^class " experiments/ecro_util96a/w96a.py
  -> {
 "s": 1.0,
 "N": 1270,
 "gpu": "NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB",
 "foreign_before": [],
 "frame_spp": 64,
 "res": 512,
 "components": {
  "q_table_s": 0.0242000799626112,
  "one_order_s": 0.0011352160945534706,
  "tail_3_orders_s": 0.003293312154710293,
  "lambda24_app ...[cut]... ain():
542:        def __init__(self):
547:        def _take(self):
551:        def part(self, name):
555:        def done(self):
569:    def lam_apply(rho, L2):
576:        def __init__(self, arm):
582:        def _table(self, lt_, ck):
595:        def _frame(self, lt_, seed, ck):
598:        def edit(self, geom, lt_, seed):
613:        def source(self, lt_, seed):
640:        def summ(rows):
651:            def tm(fn, n=20):
688:# ------------------------------------------------------------------------------------------------ verdict
689:def final(R, outdir):
740:def figures(R, outdir, ext):
RUN [Show remaining cost json and code outline] cd /c/Projects/ReST-GI && python -c "
import json
d=json.load(open('results/ecro_util96a/cost_s1.0.json'))
print(json.dumps({k:v for k,v in d.items() if k not in ('arms',)},indent=1)[:5000])
"; grep -n "def \|^# \|^class " experiments/ecro_util96a/w96a.py
RUN [Read w96a.py header and explicit operator] sed -n 1,135p experiments/ecro_util96a/w96a.py
RUN [Read w96a.py Stage 3 cost code] sed -n 486,690p experiments/ecro_util96a/w96a.py
  -> """WL96A (protocol ecro_util96a): learned higher-order transport utility ceiling on CK2 'none' (learning-free diagnostics).
Production code (pt86 / pt86a / ecro86 / model86 / quad87 / xport87 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.

  python w96a.py stage0 <R> <R95A>      ...[cut]...  T_o(L2 + L3 + L4) (last gather split by the material of s1); orders for the tests."""
    import torch
    import xport87 as X
    L2 = torch.as_tensor(np.asarray(L2, np.float32), device=X.dev)
    L3 = op.apply(L2); L4 = op.apply(L3); L5 = op.apply(L4)
    g = op.gather(L2 + L3 + L4)
    tp = op.scatter(g * mp); to = op.scatter(g * (1.0 - mp))
    return tp.double().cpu().numpy(), to.double().cpu().numpy(), (L3, L4, L5)


def realizations(ctx):
    import fx92
    import xport87 as X
    E87 = fx92.E87
    yield 'deployed', X.realization_us(86001), E87['seed_ray']
    for d in range(R_POOL):
  -> # ------------------------------------------------------------------------------------------------ Stage 3 lifecycle
_KH = {'exec_ms': 0.0, 'compile_ms': 0.0, 'misses': 0}


def _install_kh():
    import pt86 as PTm
    dr = PTm.dr
    orig = dr.kernel_history

    def kh(*a, **k):
        h = orig( ...[cut]... n ('edit', 'source')}
    out['U'] = U
    out['boundary_ok'] = bool(all(0.95 <= out['arms'][a][k]['parts_sum_over_wall'] <= 1.05 for a in out['arms'] for k in ('edit', 'source')))
    jd(out, os.path.join(R, 'results', f'cost_s{s}.json'))
    print('COST DONE', s, 'boundary_ok', out['boundary_ok'], flush=True)
    if s == 1.0 and max(U['ZERO'].values()) >= 1.15 and not DEV:
        print('COST EXTEND', flush=True)


# ------------------------------------------------------------------------------------------------ verdict
def final(R, outdir):
    s0 = jl(os.path.join(R, 'results', 's0.json'))
RUN [Read w96a.py header and explicit operator] cd /c/Projects/ReST-GI && sed -n 1,135p experiments/ecro_util96a/w96a.py
RUN [Read w96a.py Stage 3 cost code] cd /c/Projects/ReST-GI && sed -n 486,690p experiments/ecro_util96a/w96a.py
RUN [Read quad87.py] wc -l experiments/gm1_explicit87/*.py experiments/ecro86/*.py experiments/ecro_mat94/*.py experiments/ecro86a/*.py && cat experiments/gm1_explicit87/quad87.py
  -> 588 experiments/gm1_explicit87/analyze87.py
    71 experiments/gm1_explicit87/devchk87.py
   201 experiments/gm1_explicit87/econ87.py
    38 experiments/gm1_explicit87/fx87.py
   147 experiments/gm1_explicit87/pt87.py
   338 experiments/gm1_explicit87/quad87.py
   274 experiments/gm1_explicit87/run8 ...[cut]...         # inside a patch-frame bin and above the local horizon
        cv = dr.select(keep, val * E * (dr.gather(F, qs['invom'], slot) * (1.0 / (P * Mq))), 0.0)
        dr.eval(cv, slot); dr.sync_thread()                                    # evaluate BEFORE the scatter (lazy-scatter defect)
        for c in range(3):
            dr.scatter_reduce(dr.ReduceOp.Add, buf[c], cv[c], slot)
        dr.eval(buf)
    dr.sync_thread()
    q = np.stack([np.asarray(x, np.float64) for x in buf], 1).reshape(N, NB, 3)
    ql = np.stack([np.asarray(x, np.float64) for x in bl], 1)
    return q + ql[:, None, :]
RUN [Read quad87.py] cd /c/Projects/ReST-GI && wc -l experiments/gm1_explicit87/*.py experiments/ecro86/*.py experiments/ecro_mat94/*.py experiments/ecro86a/*.py && cat experiments/gm1_explicit87/quad87.py
RUN [Read xport87.py and ecro86.py] cat experiments/gm1_explicit87/xport87.py && sed -n 1,206p experiments/ecro86/ecro86.py
  -> """WL87 explicit (learning-free) patch x bin transport on the WL86 geometry state.

Primary comparator EXP (receiver gather on the deployed state):
  L_1 = q                                           once-scattered table (quad87.qtable)
  L_{k+1}[i, bo] = sum_r W_ray[m_i, r, bo] L_k[hit_r, dbin_r]   ...[cut]... 'camera2'] if opt.get('--cam') == '2' else None)
        out = os.path.join(root, 'results', 'renders'); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), B=B.astype(np.float32), **o)
        json.dump({'cfg': cfg, 'model': mtag, 'variant': variant, 'seed': seed, 'spp': spp, 'timing': T_, 'gpu_mem_mib': PT.gpu_mem()},
                  open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
        print(tag, json.dumps(T_), flush=True)
    elif cmd == 'contracts':
        contracts(root, args[0])


if __name__ == '__main__':
    main()
    os._exit(0)
RUN [Read xport87.py and ecro86.py] cd /c/Projects/ReST-GI && cat experiments/gm1_explicit87/xport87.py && sed -n 1,206p experiments/ecro86/ecro86.py
RUN [Read model86.py and pt86 outline] cat experiments/ecro86/model86.py && grep -n "def " experiments/ecro86/pt86.py
  -> """WL86 ECRO v0 operator (torch): geometry-conditioned, source-independent, linear and non-negative.

Geometry state (no radiance): patch tables, frozen ER states of the patch materials, stratified hemisphere state rays
(hit patch, outgoing bin at the hit patch) -> generator network -> response fact ...[cut]... =False):
208:        def add(name, c):
213:        def adds(key, v):
288:    def accum(out, index, n):
300:    def render(self, lights, ref, cam, res, spp, seed, mode='cam', B=None, plain=False, b=32, kmax=None):
327:    def field(self, lights, spp, seed, kmax=D - 1, b=128):
353:    def state_rays(self, idx=None):
378:    def source(self, lights):
404:    def gbuffer(self, cam, res):
418:    def light_vis(self, p, n, lights):
429:    def hit_info(self, o, d, active):
434:    def seg_clear(self, a, b, active):
441:def pack(acc, res, spp):
453:def config(name):
459:def gpu_mem():
463:def main():
RUN [Read model86.py and pt86 outline] cd /c/Projects/ReST-GI && cat experiments/ecro86/model86.py && grep -n "def " experiments/ecro86/pt86.py
RUN [Read pt86 Ctx init, set_geom, render, state_rays, source] sed -n 84,141p experiments/ecro86/pt86.py && sed -n 286,404p experiments/ecro86/pt86.py
  -> class Ctx:
    """One Mitsuba scene of the family + Dr.Jit tables of the current geometry; set_geom() edits the boxes in place."""

    def __init__(self, geom, bsdfs=None, emitters=None):
        self.mats = Mats()
        g = S.Geom(geom)
        df = {'type': 'diffuse'}
        bs = bsdfs or [df] ...[cut]... occ = self.scene.ray_test(r, on)
            e = dr.select(on & ~occ, cs * dr.rcp(d2), 0.0)
            bn = self.bin_of(dr.dot(wl, t), dr.dot(wl, bv), dr.dot(wl, n))
            ch = [e * lights.rgb[l].x, e * lights.rgb[l].y, e * lights.rgb[l].z]
            slot = pid * NB + bn
            dr.eval(ch, slot); dr.sync_thread()
            sl = np.asarray(slot)
            for c in range(3):
                s[:, c] += np.bincount(sl, weights=np.asarray(ch[c], np.float64), minlength=self.N * NB)
        return (s / PSRC).reshape(self.N, NB, 3).astype(np.float32)

    def gbuffer(self, cam, res):
RUN [Read pt86 Ctx init, set_geom, render, state_rays, source] cd /c/Projects/ReST-GI && sed -n 84,141p experiments/ecro86/pt86.py && sed -n 286,404p experiments/ecro86/pt86.py
RUN [Read econ87.py and list econ results] cat experiments/gm1_explicit87/econ87.py; ls results/gm1_explicit87/results/econ results/gm1_c3urx1_91/results/econ results/ecro86/results/econ
  -> """WL87 Stage 3: quality-matched economics of all arms (run ALONE on the GPU; protocol stage3).

  python econ87.py <R87> <R86> [--dev]
Arms: PT, PT_RR (physical path tracing), ECRO_s8601 (frozen learned table), EXP (explicit same-state transport), TF<b> (physical table
fill at b samples per slot),  ...[cut]... ef_files'] = [os.path.basename(f) for f in fns]
        R[e] = Re
        json.dump(R, open(os.path.join(out, 'econ87_partial.json'), 'w'), indent=1)
    R['foreign_after'] = foreign()
    json.dump(R, open(os.path.join(out, 'econ87.json'), 'w'), indent=1)
    print('ECON_DONE', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
results/ecro86/results/econ:
econ86.json

results/gm1_c3urx1_91/results/econ:
cold91_C3UR.json
cold91_C3URX1.json
cold91_PT.json
cold91_R0.json
ladder91.json
life91.json
scale91.json

results/gm1_explicit87/results/econ:
econ87.json
econ87_partial.json
RUN [Read econ87.py and list econ results] cd /c/Projects/ReST-GI && cat experiments/gm1_explicit87/econ87.py; ls results/gm1_explicit87/results/econ results/gm1_c3urx1_91/results/econ results/ecro86/results/econ
RUN [Summarize econ87.json] python - <<'EOF'
import json
d=json.load(open('results/gm1_explicit87/results/econ/econ87.json'))
print({k:d[k] for k in d if k not in ('E1','E2','E3')})
for e in d['edits']:
    R=d[e]
    print('==',e)
    print(' cold',json.dumps(R['cold']))
    print(' edit_median',json.dumps(R['edit_median']))
    print(' edits', [ (x['cfg'],x['dirty_rays'],round(x['q_s']*1e3,1),round(x['q_gpu_ms'],2),round(x['source_s']*1e3,1),round(x['source_gpu_ms'],2), round(x['ecro_inference_s']*1e3,1), round(x['exp_ap
  -> {'foreign_before': '', 'gpu': 'NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB', 'versions': {'mitsuba': '3.5.1', 'drjit': '0.4.6', 'torch': '2.3.1+cu121'}, 'edits': ['E1', 'E2', 'E3'], 'foreign_after': ''}
== E1
 cold {"first_process": true, "model_load_s": 0.0689708162099123, "scene_build_s": 0 ...[cut]... 8.9, 7.0, 0.2497), (16, 40.0, 5.7, 28.8, 11.1, 0.1412), (64, 55.7, 21.3, 29.0, 26.8, 0.0733), (256, 226.3, 85.3, 114.4, 112.0, 0.0381), (1024, 905.9, 341.5, 454.7, 451.1, 0.0183), (4096, 3628.7, 1367.3, 1819.7, 1808.7, 0.009), (16384, 14558.5, 5469.4, 7282.8, 7273.5, 0.0053)]
 view2 {'PT': {'trace_s': 0.09067055396735668, 'eval_s': 0.13480274751782417, 'gpu_ms': 121.41382287722081, 'wall_s': 0.22547674924135208, 'launches': 1, 'spp': 64}, 'EXP': {'trace_s': 0.029793884605169296, 'eval_s': 0.02759651280939579, 'gpu_ms': 22.0906239496544, 'wall_s': 0.0573941208422184, 'launches': 1, 'spp': 64}}
RUN [Summarize econ87.json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/gm1_explicit87/results/econ/econ87.json'))
print({k:d[k] for k in d if k not in ('E1','E2','E3')})
for e in d['edits']:
    R=d[e]
    print('==',e)
    print(' cold',json.dumps(R['cold']))
    print(' edit_median',json.dumps(R['edit_median']))
    print(' edits', [ (x['cfg'],x['dirty_rays'],round(x['q_s']*1e3,1),round(x['q_gpu_ms'],2),round(x['source_s']*1e3,1),round(x['source_gpu_ms'],2), round(x['ecro_inference_s
RUN [Read e2e91.py first part] sed -n 1,200p experiments/gm1_c3urx1_91/e2e91.py
  -> """WL91 end-to-end lifecycle, quality ladders and bounded scaling (run ALONE on the GPU; protocol 'economics' / 'scaling').

  python e2e91.py life <R91> [--dev]            warm end-to-end pipelines (edits, source updates, amortisation sequence), memory, ray counts
  python e2e91.py cold <R91> <arm> ...[cut]... s'), 'wall_raw_s': med(rows, 'wall_raw_s'), 'events_with_compilation': int(sum(r['kernel_cache_misses'] > 0 for r in rows)),
            'compile_s_total': float(sum(r['compile_s'] for r in rows)), 'cpu_s': med(rows, 'cpu_s'), 'gpu_kernel_ms': med(rows, 'gpu_kernel_ms'),
            'parts': {k: float(np.median([r['parts'].get(k, 0.0) for r in rows])) for k in keys},
            'wall_minus_render_s': float(np.median([r['wall_s'] - r['parts'].get('render', 0.0) for r in rows])),
            'parts_sum_over_wall': float(np.median([r['parts_sum_s'] / r['wall_s'] for r in rows])), 'n': len(rows)}
RUN [Read e2e91.py first part] cd /c/Projects/ReST-GI && sed -n 1,200p experiments/gm1_c3urx1_91/e2e91.py
RUN [Read e2e91.py second part] sed -n 200,466p experiments/gm1_c3urx1_91/e2e91.py
  -> def life(R):
    R8.inject_choice(R)
    fx91.fx88.setup('W1_T0')
    out = os.path.join(R, 'results', 'econ'); os.makedirs(out, exist_ok=True)
    ex = FX['W1']
    ctx = PT.Ctx(ex['T0']); lt = PT.Lights(FX['light_sets'][ex['light_set']])
    mat = R8.materials(ctx)
    quad = Q.Quad(ctx.u_state_np ...[cut]... row.items() if not isinstance(v, (dict, list))}), flush=True)
        del cand
        torch.cuda.empty_cache()
    json.dump({'rows': rows, 'foreign': foreign()}, open(os.path.join(out, 'scale91.json'), 'w'), indent=1)
    print('SCALE_DONE', flush=True)


if __name__ == '__main__':
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    cmd = sys.argv[1]
    if cmd == 'life':
        life(sys.argv[2])
    elif cmd == 'cold':
        cold(sys.argv[2], sys.argv[3])
    elif cmd == 'ladder':
        ladder(sys.argv[2], sys.argv[3])
    elif cmd == 'scale':
        scale(sys.argv[2])
    os._exit(0)
RUN [Read e2e91.py second part] cd /c/Projects/ReST-GI && sed -n 200,466p experiments/gm1_c3urx1_91/e2e91.py
RUN [Summarize life91.json and scale91.json] python - <<'EOF'
import json
d=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
print({k:d[k] for k in d if k!='arms'})
for a,A in d['arms'].items():
    print('==',a)
    for k in ('edit','source_single_light','source_two_lights','frame'):
        s=A[k]
        if s is None: print(k,None); continue
        print(' ',k,'wall %.1f raw %.1f cpu %.1f gpuK %.1f noRender %.1f n=%d comp_ev=%d comp_tot=%.2f sum/wall %.3f'%(s['wall_s']*1e3,s['wall_raw_s']*1e3,s['cpu_s']*1e3,s['gpu_kern
  -> {'foreign_before': '', 'gpu': 'NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB', 'frame_spp': 64, 'foreign_after': ''}
== C3URX1
  edit wall 124.7 raw 124.7 cpu 129.0 gpuK 22.4 noRender 69.4 n=7 comp_ev=0 comp_tot=0.00 sum/wall 0.996
      {'geometry_update': 30.78, 'probe_buffer': 1.28, 'relati ...[cut]... d': 23369}
{'s': 1.0, 'kappa_s': 16, 'N': 1270, 'C3UR_build_s': 0.10983, 'unique_relation_rays': 1300480, 'landed_rays': 1052482, 'bytes': {'relation': 41615360, 'probe_buffer': 25259568, 'scratch_radiance': 12629784, 'scratch_incident_cells': 62423040}, 'q_table_s': 0.0254, 'X1_source_eval_s': 0.01139, 'X1_first_gather_s': 0.00275, 'apply_one_order_s': 0.00278, 'C3URX1_table_J2_s': 0.01603, 'C3UR_table_J2_s': 0.00615, 'C3URX1_table_J4_s': 0.02162, 'C3UR_table_J4_s': 0.01039, 'C3URX1_table_J8_s': 0.03005, 'C3UR_table_J8_s': 0.02017, 'C3URX1_update_s': 0.01839, 'relation_rays_retraced': 93633}
RUN [Summarize life91.json and scale91.json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
print({k:d[k] for k in d if k!='arms'})
for a,A in d['arms'].items():
    print('==',a)
    for k in ('edit','source_single_light','source_two_lights','frame'):
        s=A[k]
        if s is None: print(k,None); continue
        print(' ',k,'wall %.1f raw %.1f cpu %.1f gpuK %.1f noRender %.1f n=%d comp_ev=%d comp_tot=%.2f sum/wall %.3f'%(s['wall_s']*1e3,s['wall_raw_s']*1e3,
RUN [List WL96 calc scratch and plan dirs] ls -la results/wl96_calc/scratch/* | head -60; ls docs/plans/wl96_spatio_angular_carrier | head -50
  -> results/wl96_calc/scratch/critic_cost:
total 8
drwxr-xr-x 1 dna10 197609    0 Nov 15 22:37 .
drwxr-xr-x 1 dna10 197609    0 Nov 15 22:37 ..
-rw-r--r-- 1 dna10 197609 2800 Nov 15 22:37 dirty.py

results/wl96_calc/scratch/critic_phys:
total 17
drwxr-xr-x 1 dna10 197609    0 Nov 15 22:37 .
drwxr-xr-x 1 ...[cut]... 2:37 ..
-rw-r--r-- 1 dna10 197609  3378 Nov 15 22:37 fbc.py
-rw-r--r-- 1 dna10 197609  5680 Nov 15 22:37 photon.py
-rw-r--r-- 1 dna10 197609  4737 Nov 15 22:37 portal2.py
-rw-r--r-- 1 dna10 197609  2117 Nov 15 22:37 rankM.py
-rw-r--r-- 1 dna10 197609   368 Nov 15 22:37 res_exact_L0.json
-rw-r--r-- 1 dna10 197609   301 Nov 15 22:37 res_exact_L1.json
-rw-r--r-- 1 dna10 197609   648 Nov 15 22:37 res_fbc_L0.json
-rw-r--r-- 1 dna10 197609   653 Nov 15 22:37 res_fbc_L1.json
README.md
critic_cost.md
critic_physics.md
designer1_free.md
designer2_lobe_state.md
designer3_basis.md
designer4_continuous.md
RUN [List WL96 calc scratch and plan dirs] cd /c/Projects/ReST-GI && ls -la results/wl96_calc/scratch/* | head -60; ls docs/plans/wl96_spatio_angular_carrier | head -50
RUN [Read WL96 cost critic materials] cat results/wl96_calc/scratch/critic_cost/dirty.py; wc -c docs/plans/wl96_spatio_angular_carrier/*; cat docs/plans/wl96_spatio_angular_carrier/critic_cost.md
  -> <persisted-output>
Output too large (29.4KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bmmopmeqk.txt

Preview (first 2KB):
import sys, json, math, numpy as np
sys.path.insert(0, r'c:/Projects/ReST-GI/experiments/ecro8 ...[cut]... ,np.inf),tn)
        tf=np.where(par,np.where(np.abs(oo)<=h[k],np.inf,-np.inf),tf)
        lo=np.maximum(lo,tn);hi=np.minimum(hi,tf)
    return (lo<hi-1e-9)&(hi>1e-6)&(lo<1-1e-6)
A=ck['geom']['A']
tot=0;front=0;dirtyA=0
recvA=(G.PSID[rpid]==S.SID_BOXA) if hasattr(G,'PSID') else None
for i in range(0,len(cells),32):
    c=cells[i:i+32][:,None,:]
    d=rp[None]-c
    ff=((d@pn)>1e-9)&((np.einsum('ijk,jk->ij',-d,rn))>1e-9)
    hit=obb_hit(c+0*d,d,A,1.0)
    tot+=d.shape[0]*d.shape[1]; front+=ff.sum(); dirtyA+=(ff&hit).sum()
onA=np.isin(rpid,np.where(G.PSID==S.SID_BOXA)[0])
...
</persisted-output>
RUN [Read WL96 cost critic materials] cd /c/Projects/ReST-GI && cat results/wl96_calc/scratch/critic_cost/dirty.py; wc -c docs/plans/wl96_spatio_angular_carrier/*; cat docs/plans/wl96_spatio_angular_carrier/critic_cost.md
RUN [Read critic_cost.md with truncated lines] sed -n 40,80p results/wl96_calc/scratch/critic_cost/dirty.py; grep -n "" docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-900 | head -120
  -> for i in range(0,len(cells),32):
    c=cells[i:i+32][:,None,:]
    d=rp[None]-c
    ff=((d@pn)>1e-9)&((np.einsum('ijk,jk->ij',-d,rn))>1e-9)
    hit=obb_hit(c+0*d,d,A,1.0)
    tot+=d.shape[0]*d.shape[1]; front+=ff.sum(); dirtyA+=(ff&hit).sum()
onA=np.isin(rpid,np.where(G.PSID==S.SID_BOXA)[0])
print(' ...[cut]... coped)
112:
113:| item | value | source | regime |
114:|---|---|---|---|
115:| ECRO per edit | 76–78 ms, of which operator 11 ms | [MEAS WL86:214] | scene86 |
116:| explicit 4-order recurrence | 10–11 ms | [MEAS WL87:175] | scene86, same machine |
117:| R0 sweeps | 5.6–6.1 ms | [MEAS WL89 / WL91] | GM1 room, N 1,270 |
118:| C3UR sweeps | 10.6–11.6 ms | [MEAS WL89 / WL91] | GM1 room, N 1,270 |
119:| source-only update | ECRO 13–18 ms vs EXP 51–64 ms (q 42–54 ms wall, 2 ms GPU) | [MEAS WL87:175, 281] | scene86 |
120:| fidelity | learned 0 better / 3 worse / 36 equal | [MEAS WL87:346] | scene86 |
RUN [Read critic_cost.md with truncated lines] cd /c/Projects/ReST-GI && sed -n 40,80p results/wl96_calc/scratch/critic_cost/dirty.py; grep -n "" docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-900 | head -120
RUN [Read critic_cost.md section 2.2 onward] sed -n 120,200p docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-1200
  -> | fidelity | learned 0 better / 3 worse / 36 equal | [MEAS WL87:346] | scene86 |
| projection at orders ≥ 2 | P2plus ≤ 0.0009 | [MEAS WL90:176] | GM1 |

### 2.2 Amdahl ceiling: perfect zero-cost resolvent replacing the measured sweeps [INF from MEAS WL91 C3URX1 lifecycle]

| lifecycle metric | measu ...[cut]... IABLE (pending a cheap NMF confirmation). Comparator: none (dominated by M64 / q).

### d2 LSTO / C1 (lobe state + learned transfer)

- **Explicit?** No; it is learned ray casting (WL52 class) or a low-rank PRT transfer.
- **Fatal.**
  - 2.6 GB of generated factors per geometry at rank 256.
  - Per light, Σ_c E_cl · T_c is about 6.6e8 MAC, more than DQ's 6 M ER evaluations.
  - Dominated in memory, per-light compute and accuracy.
- **Verdict.** NOT VIABLE on both counts.

### d2 SRR-LB / C2 (DQ atoms + relay continuations + camera continuation + learned Λ_θ)

- **Explicit?** Yes, apart from Λ.
RUN [Read critic_cost.md section 2.2 onward] cd /c/Projects/ReST-GI && sed -n 120,200p docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-1200
RUN [Read WL96A protocol stage 3 sections] python - <<'EOF'
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
print(list(p.keys()))
for k in p:
    if 'stage3' in k.lower() or 'cost' in k.lower() or 'decision' in k.lower() or 'verdict' in k.lower() or 'lifecycle' in k.lower():
        print(k, json.dumps(p[k],indent=1))
EOF
  -> ['protocol', 'version', 'title', 'brief', 'batch_class', 'question', 'preserved', 'families', 'stage_0_contract', 'stage_1', 'stage_2', 'stage_3', 'verdict', 'stops', 'not_authorised', 'execution', 'pre_freeze_exposure']
verdict {
 "LEARNED_RESOLVENT_UTILITY": {
  "ACCURACY_HEADROOM": "S1_HEADROOM and ACCURACY_HEADROOM_CONFIRMED, COST criterion not met",
  "COST_HEADROOM": "COST criterion met, accuracy not (S1_NO_HEADROOM, or S1_HEADROOM without confirmation)",
  "BOTH": "both",
  "NO_MEANINGFUL_HEADROOM": "S1_NO_HEADROOM (or S1_HEADROOM not confirmed by Stage 2) and COST criterion not met",
  "INCONCLUSIVE": "S1_INCONCLUSIVE and COST not met, or a validity gate fails after Stage 0"
 },
 "note": "evidence classification only; headroom is not viability, trainability or generalization; no implementation or training permission"
}
RUN [Read WL96A protocol stage 3 sections] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
print(list(p.keys()))
for k in p:
    if 'stage3' in k.lower() or 'cost' in k.lower() or 'decision' in k.lower() or 'verdict' in k.lower() or 'lifecycle' in k.lower():
        print(k, json.dumps(p[k],indent=1))
EOF
RUN [Read WL96A protocol stage_3 and related] python - <<'EOF'
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
for k in ('question','stage_3','stops','execution','pre_freeze_exposure','preserved'):
    print('==',k); print(json.dumps(p[k],indent=1)[:6000])
EOF
  -> == question
"Given a fixed, physically accurate transport prefix, is there meaningful representability or lifecycle-cost headroom for a learned higher-order ECRO operator over explicit recurrence? Accuracy headroom and runtime headroom are separated. A positive result is not evidence of trainabilit ...[cut]...  and the pool sizes."
== preserved
[
 "ER and its convention (eval(wi = viewer, wo = light) = f cos(theta_wo))",
 "ECRO model86 / ecro86 / pt86 / pt86a / quad87 / xport87 / mat94 / an94 source files and every checkpoint (own_s8601 / own_s8602 WL92A loaded read-only)",
 "the explicit comparator EXP (xport87.XOp, quad87.Quad, deployed realization 86001, J = 4 orders)",
 "CK2 geometry, materials (default map 'none'), light LCK2, camera, masks94",
 "CKF_none field reference (2 x 4,096 spp, seeds 94301 / 94302) and REF_ER_CK_none images (4 x 8,192 spp)",
 "WL86-WL96 history and verdicts"
]
RUN [Read WL96A protocol stage_3 and related] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
for k in ('question','stage_3','stops','execution','pre_freeze_exposure','preserved'):
    print('==',k); print(json.dumps(p[k],indent=1)[:6000])
EOF
RUN [Grep WL96A worklog for cost claims] grep -n -i "stage 3\|lifecycle\|amdahl\|U(\|cost\|144\|generator\|cpu" docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | cut -c1-700 | head -80
  -> 6:- **Question.** Given a fixed, physically accurate transport prefix, is there meaningful representability or lifecycle-cost headroom for a learned higher-order ECRO operator over explicit recurrence? Accuracy and runtime headroom are kept separate. A positive result would not have shown trainabili ...[cut]... ITY = NO_MEANINGFUL_HEADROOM** (CK2 `none`; frozen mapping; Stage 1 S1_NO_HEADROOM, Stage 2 not triggered, COST criterion not met).
231:  - lifecycle sources rng 96500; random Λ factors 96600.
237:  - `cost_s1.0.json` 103b775ed571ed26;
243:  - figW3 (lifecycle; colour-inconsistent chain version, kept) and figW3_lifecycle_posthoc (one colour per component, figure-only fix).
247:  - Lifecycle 19 s alone, GPU 18–26 % (CPU-bound phases of the pipeline).
250:  - Utilisation by stage: tails 7 %, one render 23 %, one extension field half 3 % (CPU-bound accumulation), Stage 2 fit 98 %, lifecycle 15 %.
RUN [Grep WL96A worklog for cost claims] cd /c/Projects/ReST-GI && grep -n -i "stage 3\|lifecycle\|amdahl\|U(\|cost\|144\|generator\|cpu" docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | cut -c1-700 | head -80
RUN [Read WL96A worklog B4 through verdict; hash inputs] sed -n 118,210p docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | cut -c1-1500; sha256sum results/ecro_util96a/cost_s1.0.json results/gm1_c3urx1_91/results/econ/life91.json results/gm1_c3urx1_91/results/econ/scale91.json results/gm1_explicit87/results/econ/econ87.json results/ecro86/results/econ/econ86.json | cut -c1-90
  -> - box A e3_o: no field headroom (0.004 deployed vs bound 0.016).
  - S1 e3_o, deployed: flagged (0.043 vs 0.024).
  - S1 e3_o, pooled: not flagged (0.016 vs 0.024).

### B4. Lifecycle cost (Stage 3; `cost_s1.0.json`)

Setup:
- Hardware: RTX 3080 Ti 12 GB, run alone (no foreign process).
- Scene: N = ...[cut]... ESOLVENT_UTILITY = NO_MEANINGFUL_HEADROOM** (CK2 `none`; frozen mapping; Stage 1 S1_NO_HEADROOM, Stage 2 not triggered, COST criterion not met).

103b775ed571ed26212d4589a3c3b58765fe1b6441d45093eabaff9e09e389aa *results/ecro_util96a/cos
2f3fc40b5afb28cd4d570eb04ca746f256e5ed6105c130599b465c3f144a6498 *results/gm1_c3urx1_91/re
e158b848a563ddb9658faa221226b1d6fba01d552cc03976641ff6d1d1045d1b *results/gm1_c3urx1_91/re
db9d932ff8ee90dc7048de91810c54d430df5f72cc799c8a1101f181db841dba *results/gm1_explicit87/r
b543a6b1146e83061200d2bf63bf1db983aedbd0e12c9cd088961193adf065c8 *results/ecro86/results/e
RUN [Read WL96A worklog B4 through verdict; hash inputs] cd /c/Projects/ReST-GI && sed -n 118,210p docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | cut -c1-1500; sha256sum results/ecro_util96a/cost_s1.0.json results/gm1_c3urx1_91/results/econ/life91.json results/gm1_c3urx1_91/results/econ/scale91.json results/gm1_explicit87/results/econ/econ87.json results/ecro86/results/econ/econ86.json | cut -c1-90
RUN [Read econ86.py] sed -n 1,179p experiments/ecro86/econ86.py
  -> """WL86 Stage 4: measured end-to-end economics, ECRO vs full frozen-ER path tracing (run ALONE on the GPU).

  python econ86.py <root> <model_tag> [--dev]
For E1 (primary) and E2: time-to-quality ladders of the plain path tracer (events <= 6) and of plain ECRO (LOW + table lookup) against the
pooled ...[cut]... gpu)
                rp, rc = p['wall_s'].get('region_err'), c['wall_s'].get('region_err')
                row['region_matched'] = bool(rp is not None and rc is not None and all(rc[k] <= E['region_tolerance'] * rp[k] for k in rp)) if rp and rc else None
            Re['matched'][name] = row
    json.dump(R, open(os.path.join(out, 'econ86.json'), 'w'), indent=1)
    print(json.dumps({e: {k: {kk: vv for kk, vv in v.items() if 'speedup' in kk or kk == 'region_matched'} for k, v in R[e]['matched'].items()} for e in ('E1', 'E2')}), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
RUN [Read econ86.py] cd /c/Projects/ReST-GI && sed -n 1,179p experiments/ecro86/econ86.py
RUN [Summarize econ86.json and grep WL86 worklog cost claims] python - <<'EOF'
import json
d=json.load(open('results/ecro86/results/econ/econ86.json'))
print({k:d[k] for k in d if k not in ('E1','E2')})
for e in ('E1','E2'):
    R=d[e]
    print('==',e,'g',R.get('g_stationary'))
    print(' cold',json.dumps(R['cold']))
    print(' edit_median',json.dumps(R['edit_median']))
    print(' pt_edit',R['pt_edit_median'], 'fixed', R['ecro_fixed_wall_s'],R['ecro_fixed_wall_full_state_s'],R['pt_fixed_wall_s'])
    print(' edits',[(x['cfg'],x['dirty_rays'],x['selecti
  -> {'foreign_before': '', 'gpu': 'NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB', 'versions': {'mitsuba': '3.5.1', 'drjit': '0.4.6', 'torch': '2.3.1+cu121'}, 'model': 'ecro_s8601', 'foreign_after': ''}
== E1 g 0.008659190174597585
 cold {"model_load_s": 0.06822939589619637, "scene_build_s": 0.1812 ...[cut]... rhead 40 ms wall above the path tracer's.
258:- The quality level g was not reached by either method on the ladder (≥ 4,096 spp needed); the economic comparison there is unmeasured.
271:| END-TO-END ECONOMICS | **NOT TESTED** (gated by Stage 3) | Descriptive: 9× per-frame GPU and 3–6× edit-to-first-frame wall at 0.05 / 0.03 with region-matched errors; g not reached; bias floor in E2's C |
290:- **Visuals.** `results/evaluation/86/` (manifest with sha256): references and regions, signed change, higher-order term and its change for E1 and E2, unseen sources, error by region, training, economics.
RUN [Summarize econ86.json and grep WL86 worklog cost claims] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/ecro86/results/econ/econ86.json'))
print({k:d[k] for k in d if k not in ('E1','E2')})
for e in ('E1','E2'):
    R=d[e]
    print('==',e,'g',R.get('g_stationary'))
    print(' cold',json.dumps(R['cold']))
    print(' edit_median',json.dumps(R['edit_median']))
    print(' pt_edit',R['pt_edit_median'], 'fixed', R['ecro_fixed_wall_s'],R['ecro_fixed_wall_full_state_s'],R['pt_fixed_wall_s'])
    print(' edits',[(x['cfg'],
RUN [Grep WL87 worklog cost claims] grep -n -i "source-only\|q table\|42\b\|2 ms\|overhead\|untested\|utility\|1\.25\|edit-to\|per edit\|econom" docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-900 | head -50
  -> 9:  - H1 (learned utility): explicit transport represents the response, but learning saves measurable computation.
79:  - Measures: WL86 E_H with the WL86 thresholds (F1 stationary ≤ 0.03 with capture ≤ 0.25; F2 A / C / D ≤ 0.06 at T1; F3 slope 0.75–1.25 and correlation ≥ 0.8 in D; F4 unseen sources ...[cut]...  학습 operator와 같거나 나은 fidelity를, 비슷한 edit 비용과 같은 frame 비용으로 냅니다. H1은 source-only 갱신에서만 성립합니다. H2는 출력 표현이 아니라 수신측 512-ray relation의 sharp-family 표본 배치에 한정해 부분적으로 성립하며, 학습과 explicit 모두에 공통입니다. 학습 operator 연구를 지지하는 측정은 source-only 갱신 비용 하나이고, 이번 결과는 explicit physical baseline을 이 workload의 기준점으로 세웁니다. 이후 아키텍처 선택은 하지 않습니다 |
353:- 여기에 4의 요소가 함께 있습니다. 학습 효용은 source-only 갱신에서만 측정되었고, sharp 표현성은 state 표본에 의존해 미결입니다.
358:- **Commits.** freeze `60efecb`, results and this report in the closing commit. Parent: WL86 `7786d09` + `63972c8`, results `b97027d`; WL86A `46593f0` / `1bd7df1` / `f6483ab`, `bd81442`.
RUN [Grep WL87 worklog cost claims] cd /c/Projects/ReST-GI && grep -n -i "source-only\|q table\|42\b\|2 ms\|overhead\|untested\|utility\|1\.25\|edit-to\|per edit\|econom" docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-900 | head -50
RUN [Read WL87 economics section] sed -n 262,292p docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-1200; sed -n 310,335p docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-900
  -> - C3 halving은 3/3 family case에서 성립합니다.

**경제성 (Stage 3, RTX 3080 Ti 단독, 512², T1).**

- **per-edit 고정비용(wall).** PT 36 ms, ECRO 69–78 ms, EXP 89–95 ms, TF64 152 ms, TF256 334 ms, TF1024 1.2 s, TF4096 4.6 s, C3 0.43–0.58 s.
  - TF16의 3.4 s는 첫 호출 kernel 컴파일이 warm 측정에 섞인 측정 결함입니다. GPU 시간은 9 ms 입니다.
- * ...[cut]... ESOLVED QUESTIONS

- source만 바뀌는 갱신에서 EXP의 q 비용(42–54 ms)이 줄어들 수 있는지는 시험하지 않았습니다. 이 비용은 사전 고정한 MIS 예산이 지배합니다. 따라서 학습의 유일한 측정 이점이 구현 예산에 얼마나 의존하는지 모릅니다.
- C3와 TF의 비용(0.4–4.6 s)은 이번 구현(torch index_add, numpy 재표본)에서 측정한 값입니다. 최적화된 sender-side 계산의 비용은 측정하지 않았습니다.
- pooled state가 남기는 E2 C의 체계적 결손(−0.024)의 원인은 분리하지 않았습니다. 후보는 수신 patch의 공간 평균, outgoing bin 평균, 512-방향 공유 offset 규칙입니다.
- E3의 C 영역은 바닥 footprint뿐이라 panel mirror-cone 수신자가 없습니다. 따라서 미검사 조건에서의 sharp-receiver 검증은 약합니다.
- scene 규모, 재질 수, area light, non-rigid edit에 대한 scaling은 측정하지 않았습니다. ECRO는 family-specific 학습이고 EXP는 W가 재질과 state 방향 전용입니다.
RUN [Read WL87 economics section] cd /c/Projects/ReST-GI && sed -n 262,292p docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-1200; sed -n 310,335p docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-900
RUN [Check server CPU, env and audit dir] ssh -o BatchMode=yes -o ConnectTimeout=15 LabServer63 'nproc; lscpu | grep -E "Model name|Thread|Core|Socket|MHz"; cat ~/rest_x95a/env.sh; ls ~/rest_x97 2>/dev/null; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ls ~/rest_x96a | head -30'
  -> 32
Model name:                              Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz
Thread(s) per core:                      2
Core(s) per socket:                      8
Socket(s):                               2
CPU(s) scaling MHz:                      44%
CPU max MHz:                             3000.0000
CPU min MHz:                             800.0000
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
B
C1
C2
E3
E4
MAP
 17:51:05 up 108 days, 12:34, 12 users,  load average: 0.59, 0.38, 0.21
0 %, 1370 MiB
chain.out
chain.status
code
data
env.sh
figw3_posthoc.py
results
RUN [Write and run CPU-only OpenMP probe on server] mkdir -p /c/Projects/ReST-GI/experiments/audit97/D /c/Projects/ReST-GI/results/audit97/D /c/Projects/ReST-GI/docs/audits/wl97_evidence_reliability && cat > /c/Projects/ReST-GI/experiments/audit97/D/omp_cpu_probe.py <<'EOF'
"""WL97 audit track D: CPU-only probe of the WL96A 'generator' CPU-time anomaly (no GPU: CUDA_VISIBLE_DEVICES must be empty).

Hypothesis under test: model86.GeoState converts the int16 hit / uint8 dbin state arrays (N * K = 650,240 elements) to int64 with torch
on the CPU (to
  -> omp_cpu_probe.py: Python script, ASCII text executable
omp_cpu_probe.py: Python script, ASCII text executable
{
 "torch": "2.3.1+cu121",
 "num_threads": 16,
 "num_interop": 32,
 "parallel_info": "ATen/Parallel:\n\tat::get_num_threads() : 16\n\tat::get_num_interop_threads() : 32\nOpenMP 201511 (a.k.a ...[cut]...  },
  "numpy_conv": {
   "conv_wall_ms": 13.579909689724445,
   "conv_cpu_ms": 13.609637000000063,
   "work_wall_ms": 60.12773886322975,
   "work_cpu_ms": 60.11971299999974,
   "event_wall_ms": 120.9994563832879,
   "event_cpu_ms": 75.10260150000025
  }
 },
 "work_slowdown_ms": -5.444697104394436,
 "extra_event_cpu_ms": 530.3953139999998,
 "median_torch_conv_1thread": {
  "conv_wall_ms": 22.581085562705994,
  "conv_cpu_ms": 22.59760150000023,
  "work_wall_ms": 52.67441924661398,
  "work_cpu_ms": 52.669275000000404,
  "event_wall_ms": 120.81888783723116,
  "event_cpu_ms": 77.65436950000026
 }
}
RUN [Run second CPU-only probe with warm main thread] cat > /c/Projects/ReST-GI/experiments/audit97/D/omp_cpu_probe2.py <<'EOF'
"""WL97 audit track D: CPU-only probe 2 (no GPU).  Does the OpenMP worker spin that follows the torch CPU dtype conversion slow a
single-threaded workload on a main thread that was ALREADY busy (as in the WL96A edit event, where the q table follows the generator)?
The main thread is kept busy for 0.4 s before every event (single-core turbo state), then: conversion, 5 consecutive fixed work chunks.

  CUDA_VISIBLE_DEVICES= 
  -> omp_cpu_probe2.py: Python script, ASCII text executable
{
 "num_threads": 16,
 "chunk_iters": 94693,
 "loadavg": [
  0.9462890625,
  0.53076171875,
  0.283203125
 ],
 "torch": {
  "chunk_median_ms": [
   18.6585895717144,
   14.213627204298973,
   12.356169521808624,
   11.298500001430511,
   11.288 ...[cut]... _ms": 611.118127000001,
  "conv_wall_ms": 2.136416733264923
 },
 "numpy": {
  "chunk_median_ms": [
   11.3157257437706,
   11.256856843829155,
   11.263780295848846,
   11.265961453318596,
   11.269461363554,
   11.266853660345078,
   11.26297377049923,
   11.2506914883852
  ],
  "first3_chunks_sum_median_ms": 33.9243970811367,
  "last3_chunks_sum_median_ms": 33.860765397548676,
  "event_wall_ms": 95.94381041824818,
  "event_cpu_ms": 95.94395500000097,
  "conv_wall_ms": 5.845354869961739
 },
 "first3_slowdown_ratio": 1.3332647926302401,
 "last3_slowdown_ratio": 1.0025530203764152
}
schedutil
0
RUN [Read mat94.ctx_ck2 and scene config] grep -n "def ctx_ck2" -A 40 experiments/ecro_mat94/mat94.py | head -80; grep -n "^SC\b\|^SC =\|SC = \|DEV =\|def load_frozen" -A 6 experiments/ecro_mat94/mat94.py | head -60
  -> 188:def ctx_ck2(R):
189-    """Ctx of the CK2 scene (geometry, light set, camera of the WL93D addendum rev. 2), default material map."""
190-    import fx92
191-    fx92.inject(R)
192-    PT, S, A = install()
193-    FX = fx92.FX
194-    FX[SC['tag']] = {'T0': SC['geom'], 'T1': SC['geom'], 'moved':  ...[cut]... checker', 'scene_ck2.json')
26:SC = json.load(open(SCENE_FN))
27-TAG = SC['tag'] + '_T1'
28-ALB = {3: 0.7, 4: 0.5, 5: 0.6, 6: 0.3, 7: 0.8, 8: 0.2, 9: 0.4, 10: 0.9}
29-NEW_IDS = [6, 7, 8, 9, 10]
30-TARGET_RECTS = list(range(6, 12))            # box A faces
31-PANEL_FACE = 6                               # outward normal -x (towards the panel)
32-LAMBERT_RECTS = [1, 2, 3, 4] + list(range(6, 18))
--
209:def load_frozen(R92A, seed):
210-    import torch
211-    import train93 as TR
212-    return extend_buffer(TR.load_own(R92A, int(seed), torch.device('cuda')))
213-
214-
215-def load_cand(R, tag):
RUN [Read mat94.ctx_ck2 and scene config] cd /c/Projects/ReST-GI && grep -n "def ctx_ck2" -A 40 experiments/ecro_mat94/mat94.py | head -80; grep -n "^SC\b\|^SC =\|SC = \|DEV =\|def load_frozen" -A 6 experiments/ecro_mat94/mat94.py | head -60
RUN [Read CK2 scene and scene86 helpers] cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -c 1500; echo; grep -n "def light_ok\|def box_axes\|def obb\|class Geom\|def box_matrix\|def geom_ok\|def valid" -A 14 experiments/ecro86/scene86.py | head -120; grep -n "^import\|^from" experiments/ecro86/scene86.py; grep -n "def inject" -A 20 experiments/ecro_attr92/fx92.py | head -40
  -> {
 "tag": "CK2",
 "targets": [
  "A"
 ],
 "light_set": "LCK2",
 "camera": {
  "origin": [
   0.3,
   -1.6,
   1.3
  ],
  "target": [
   -1.3,
   0.45,
   1.0
  ],
  "fov_deg": 32.0,
  "res": 512
 },
 "geom": {
  "A": {
   "c": [
    -1.08,
    0.6,
    1.0
   ],
   "h": [
    0.2,
    0.45,
    0.45 ...[cut]...  = P87['e3_candidates'][name] if name in P87['e3_candidates'] else P87['e3_dev_candidate'][name]
33-    FX['E4'] = {'T0': c['T0'], 'T1': c['T1'], 'moved': c['moved'], 'light_set': 'L4', 'role': 'WL92 held-out condition (WL87 candidate ' + name + ')'}
34-    FX['light_sets']['L4'] = [c['light']]
35-    for k, v in P92['e4']['sources'].get(name, P92['e4']['sources']['E3b']).items():
36-        FX['light_sets'][k] = v
37-    return name
38-
39-
40-def cfg_ls(tag):
41-    """'E2_T1' or 'E1_T1_LMIX' -> (geometry, light-set name, edit name, cfg)."""
42-    p = tag.split('_')
43-    e, t = p[0], p[1]
RUN [Read CK2 scene and scene86 helpers] cd /c/Projects/ReST-GI && cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -c 1500; echo; grep -n "def light_ok\|def box_axes\|def obb\|class Geom\|def box_matrix\|def geom_ok\|def valid" -A 14 experiments/ecro86/scene86.py | head -120; grep -n "^import\|^from" experiments/ecro86/scene86.py; grep -n "def inject" -A 20 experiments/ecro_attr92/fx92.py | head -40
RUN [Read mat94.install] sed -n 34,60p experiments/ecro_mat94/mat94.py; grep -n "def install" -A 45 experiments/ecro_mat94/mat94.py | head -90
  -> VDEF = {'none': None, 'U07': ('u', 3), 'U03': ('u', 6), 'bx75': ('c', 3, 4), 'bx38': ('c', 6, 7), 'bx38i': ('c', 7, 6)}
SEEDS = [8601, 8602]
ARMS_FROZEN = ['C0', 'C2_src', 'C2_rcv', 'C2_both', 'C2_srcS1', 'C2_srcNonS1']
FREF_SPP, FREF_SEEDS = 4096, (94301, 94302)
IREF_SPP, ASPP = 8192, 4096
_INST =  ...[cut]... tched_trace'] = g['trace']
175-    s2 = _patch_src(_INST['orig_field'], 'ctx', 'pt86a.field_src', probe_k1=True)
176-    g2 = dict(A.__dict__)
177-    exec(compile(s2, 'pt86a.field_src[mat94 per-patch material lookup]', 'exec'), g2)
178-    A.field_src = g2['field_src']; _INST['patched_field'] = g2['field_src']
179-    _INST['patch_sha'] = {'trace': hashlib.sha256(s1.encode()).hexdigest(), 'field_src': hashlib.sha256(s2.encode()).hexdigest()}
180-    return PT, S, A
181-
182-
183-def set_pmat(PT, ctx, pm):
184-    ctx.G.PMAT[:] = pm
185-    ctx.PMATU = PT.U(np.asarray(pm, np.uint32))
186-
187-
RUN [Read mat94.install] cd /c/Projects/ReST-GI && sed -n 34,60p experiments/ecro_mat94/mat94.py; grep -n "def install" -A 45 experiments/ecro_mat94/mat94.py | head -90
RUN [Check geometric validity of timing-only edits] cat > experiments/audit97/D/edit_validity.py <<'EOF'
"""WL97 audit track D: geometric validity of the timing-only edits (numpy only; scene86 imported read-only).
WL96A Stage 3: box A translated by +0.15 m in y (and back).  WL96 cost critic (dirty.py): box A shifted by -0.3 m in y.
Checks: scene86.geom_ok (room bounds, sphere and box-box clearance), box A vs every stationary rectangle (OBB corner / segment test)."""
import copy, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.absp
  -> {
 "CK2_T0": {
  "A_center": [
   -1.08,
   0.6,
   1.0
  ],
  "geom_ok": true,
  "N": 1270,
  "light_ok_LCK2": true,
  "A_corner_min": [
   -1.3450013752003283,
   0.1251653884340611,
   0.55
  ],
  "A_corner_max": [
   -0.8149986247996719,
   1.0748346115659388,
   1.45
  ],
   ...[cut]...     (pc, Y, Z, ph[0], ph[1], 6, 7, 1),                             # panel +x
39-    ]
40-
41-
42-def box_axes(yaw):
43-    c, s = math.cos(yaw), math.sin(yaw)
44-    return np.array([c, s, 0.0]), np.array([-s, c, 0.0]), Z.copy()
45-
46-
47-def box_faces(box, mat):
48-    """Faces f = 2 * axis + (positive side): -x', +x', -y', +y', -z, +z; in-plane axes with a1 x a2 = outward normal."""
49-    c = np.array(box['c'], np.float64); h = box['h']; ax = box_axes(box['yaw'])
50-    pairs = {(0, 1): (1, 2), (0, 0): (2, 1), (1, 1): (2, 0), (1, 0): (0, 2), (2, 1): (0, 1), (2, 0): (1, 0)}
51-    out = []
RUN [Check geometric validity of timing-only edits] cd /c/Projects/ReST-GI && cat > experiments/audit97/D/edit_validity.py <<'EOF'
"""WL97 audit track D: geometric validity of the timing-only edits (numpy only; scene86 imported read-only).
WL96A Stage 3: box A translated by +0.15 m in y (and back).  WL96 cost critic (dirty.py): box A shifted by -0.3 m in y.
Checks: scene86.geom_ok (room bounds, sphere and box-box clearance), box A vs every stationary rectangle (OBB corner / segment test)."""
import copy, json, os, sys
import numpy as np
HERE = os
RUN [Re-run WL96 critic dirty-pair script read-only] python results/wl96_calc/scratch/critic_cost/dirty.py > results/audit97/D/dirty_rerun.txt 2>&1; cat results/audit97/D/dirty_rerun.txt; sha256sum results/wl96_calc/scratch/critic_cost/dirty.py | cut -c1-16
  -> cells 672 recv pts 19136 pairs 12859392
front pairs 9093120 frac 0.7071189679885332
front pairs blocked by box A (current pose) 4295980 frac of front 0.47244290188626126
front pairs to box A receivers 163584 frac 0.017989864864864865
dirty (old or new pose, 0.3 m y-shift) frac of front, excl. receivers on A 0.6044616149352477
101af5646fb28b7a
RUN [Re-run WL96 critic dirty-pair script read-only] cd /c/Projects/ReST-GI && python results/wl96_calc/scratch/critic_cost/dirty.py > results/audit97/D/dirty_rerun.txt 2>&1; cat results/audit97/D/dirty_rerun.txt; sha256sum results/wl96_calc/scratch/critic_cost/dirty.py | cut -c1-16
RUN [Check server code layout and hashes against freeze commit] ssh -o BatchMode=yes LabServer63 'ls ~/rest_x96a/code ~/rest_x96a/code/experiments | head -60; ls ~/rest_x96a/results | head; cat ~/rest_x96a/env.sh; tail -5 ~/rest_x96a/chain.status; cd ~/rest_x96a/code && sha256sum experiments/ecro_util96a/w96a.py experiments/ecro86/ecro86.py experiments/ecro86/pt86.py experiments/gm1_explicit87/quad87.py experiments/gm1_explicit87/xport87.py experiments/ecro86/model86.py | cut -c1-16,65-'; cd /c/Projects/ReST-GI && git show a1718f7:experiments/ecro_util96a/w9
  -> /home/nd/rest_x96a/code:
AGENTS.md
COMMIT
experiments

/home/nd/rest_x96a/code/experiments:
8dna_deformation_replication
baseline_sup67
dynamic_transport_failure
dynamic_transport_validity
ecro86
ecro86a
ecro_attr92
ecro_attr95a
ecro_cf92b
ecro_exp94b
ecro_local93d
ecro_mat94
ecro_mat94r
ecro_own92a ...[cut]... 09:01Z FINAL NO_MEANINGFUL_HEADROOM
2026-11-15T17:09:01Z CHAIN_DONE
afb80dd3200d24d4  experiments/ecro_util96a/w96a.py
0ef09a86939745ed  experiments/ecro86/ecro86.py
912e68acd314d79e  experiments/ecro86/pt86.py
cfa9c5c4e854526c  experiments/gm1_explicit87/quad87.py
aba6e5765c22143c  experiments/gm1_explicit87/xport87.py
72a728b81e490fb7  experiments/ecro86/model86.py
afb80dd3200d24d4
afb80dd3200d24d4
0ef09a86939745ed
912e68acd314d79e
cfa9c5c4e854526c
aba6e5765c22143c
72a728b81e490fb7
 experiments/ecro_util96a/figw3_posthoc.py | 29 +++++++++++++++++++++++++++++
 1 file changed, 29 insertions(+)
RUN [Run relation-state invalidation correctness check on server] cat > /c/Projects/ReST-GI/experiments/audit97/D/edit_state_check.py <<'EOF'
"""WL97 audit track D: correctness (NOT timing) of the relation-state cache invalidation on the timing-only edits.
Frozen code imported read-only from ~/rest_x96a/code (commit a1718f7); run with PYTHONDONTWRITEBYTECODE=1; own run root as <R>.

  python edit_state_check.py <own R> <out.json>
For CK2 and the edits used by WL96A Stage 3 (box A +0.15 m y and back) and the WL96 critic (-0.3 m y):
  selective update (ecro86.up
  -> edit_state_check.py: Python script, ASCII text executable
0 %, 602 MiB
3458020, 592 MiB
{
 "N": 1270,
 "K": 512,
 "relation_rays": 650240,
 "edits": {
  "plus0.15y": {
   "dirty_rays_retraced": 49936,
   "dirty_fraction": 0.07679625984251968,
   "records_changed": 29881,
   "selective_equals_full":  ...[cut]... 
   "digest": "d97fa96d1cbafe5e"
  },
  "minus0.3y": {
   "dirty_rays_retraced": 52590,
   "dirty_fraction": 0.08087782972440945,
   "records_changed": 47979,
   "selective_equals_full": true,
   "full_equals_fresh_scene": true,
   "digest": "cc7090d8078dc5ce"
  },
  "back_to_T0_b": {
   "dirty_rays_retraced": 52590,
   "dirty_fraction": 0.08087782972440945,
   "records_changed": 47979,
   "selective_equals_full": true,
   "full_equals_fresh_scene": true,
   "digest": "d97fa96d1cbafe5e"
  }
 },
 "digest_T0": "d97fa96d1cbafe5e",
 "move_and_return_identical": true
}
--- files newer in rest_x96a:
RUN [Fetch outputs, check hashes, grep WL91 worklog cost claims] scp -q LabServer63:rest_x97/D/edit_state_check.json LabServer63:rest_x97/D/omp_cpu_probe.json LabServer63:rest_x97/D/omp_cpu_probe2.json results/audit97/D/ && ssh -o BatchMode=yes LabServer63 'sha256sum ~/rest_x96a/results/cost_s1.0.json ~/rest_x91/results/econ/life91.json ~/rest_x91/results/econ/scale91.json 2>/dev/null | cut -c1-16,65-; ls ~/rest_x191 2>/dev/null | head -3; ls -d ~/rest_x1* ~/rest_gm187 2>/dev/null' ; grep -n -i "lifecycle\|0\.57\|0\.90\|scal\|amortis\|overhead\|compil\|kernel
  -> 103b775ed571ed26  /home/nd/rest_x96a/results/cost_s1.0.json
chain.out
chain.status
code
/home/nd/rest_gm187
/home/nd/rest_x191
/home/nd/rest_x191dev
45:- Any quality ladder or end-to-end lifecycle of C3URX1.
82:- Also recorded: component timers, process CPU time and Dr.Jit kernel time.
83:- Events:  ...[cut]... d draw (86001) and realisations 89201–89216. Renders 89600 / 89601. New order fields 91460. Lifecycle events 918xx, cold 919xx, sources 91720, ladders 91600.
435:  - `results/gm1_c3urx1_91/`: chain status, GPU trace, tests91, final91, econ (life, cold × 4, ladder, scale), table metadata, render lists, logs.
437:- **Invalid or excluded.** None in the evidence run. Kernel compilation is excluded from warm walls by the frozen protocol and reported raw alongside.
438:- **Non-evidence.** The dev chain and kernel-variant probes on the dev geometry (`~/rest_x191dev`), and the full-scale `tabs` probe.
RUN [Fetch outputs, check hashes, grep WL91 worklog cost claims] cd /c/Projects/ReST-GI && scp -q LabServer63:rest_x97/D/edit_state_check.json LabServer63:rest_x97/D/omp_cpu_probe.json LabServer63:rest_x97/D/omp_cpu_probe2.json results/audit97/D/ && ssh -o BatchMode=yes LabServer63 'sha256sum ~/rest_x96a/results/cost_s1.0.json ~/rest_x91/results/econ/life91.json ~/rest_x91/results/econ/scale91.json 2>/dev/null | cut -c1-16,65-; ls ~/rest_x191 2>/dev/null | head -3; ls -d ~/rest_x1* ~/rest_gm187 2>/dev/null' ; grep -n -i "lifecycle\|0\.57\|0\.90\|scal\|amortis
RUN [Verify server-side hashes of timing artifacts] ssh -o BatchMode=yes LabServer63 'sha256sum ~/rest_x191/results/econ/life91.json ~/rest_x191/results/econ/scale91.json ~/rest_gm187/results/econ/econ87.json ~/rest_ecro86/results/econ/econ86.json 2>&1 | cut -c1-16,65-; grep -n "COST\|cost" ~/rest_x96a/chain.out | head; ls ~/rest_x96a/results/logs | head -20'
  -> 2f3fc40b5afb28cd  /home/nd/rest_x191/results/econ/life91.json
e158b848a563ddb9  /home/nd/rest_x191/results/econ/scale91.json
db9d932ff8ee90dc  /home/nd/rest_gm187/results/econ/econ87.json
b543a6b1146e8306  /home/nd/rest_ecro86/results/econ/econ86.json
C1.log
C1.rc
FIN.log
FIN.rc
N1.log
N1.rc
R_p_p_00.log
R_p_p_00.rc
R_p_p_01.log
R_p_p_01.rc
R_p_p_02.log
R_p_p_02.rc
R_p_p_03.log
R_p_p_03.rc
R_p_p_04.log
R_p_p_04.rc
R_p_p_05.log
R_p_p_05.rc
S0.log
S0.rc
RUN [Inspect WL96A cost stage log and GPU trace] ssh -o BatchMode=yes LabServer63 'cat ~/rest_x96a/results/logs/C1.log | tail -12; grep -n "C1\|N1" ~/rest_x96a/chain.status'; cd /c/Projects/ReST-GI && awk '$1>="17:08:20" && $1<="17:09:10"' results/ecro_util96a/gpu.txt; grep -n "cost\|C1" experiments/ecro_util96a/run96a.sh | head
  -> R0 144.13 88.53
LAM24 154.63 85.08
LAM64 156.5 85.26
ZERO 138.85 84.89
COST DONE 1.0 boundary_ok True
19:2026-11-15T17:08:35Z START N1
20:2026-11-15T17:08:39Z END N1 rc=0
23:2026-11-15T17:08:39Z START C1
24:2026-11-15T17:08:58Z END C1 rc=0
17:08:23 46 %, 1775 MiB
17:08:28 18 %, 890 MiB
17:08:33 18 %, 890 MiB
17:08:38 0 %, 292 MiB
17:08:43 1 %, 1442 MiB
17:08:49 18 %, 1574 MiB
17:08:54 26 %, 1578 MiB
17:08:59 0 %, 292 MiB
68:run C1 "$A cost $R $R92A 1.0 $DEV"
69:if grep -q '^COST EXTEND' $O/logs/C1.log || [ -n "$DEV" ]; then   # dev exercises the scaling path regardless
71:  run C15 "$A cost $R $R92A 1.5 $DEV"
72:  run C20 "$A cost $R $R92A 2.0 $DEV"
RUN [Compute ceilings and sensitivities from recorded JSON] cat > experiments/audit97/D/ceilings.py <<'EOF'
"""WL97 audit track D: recomputation of every cost ratio / Amdahl ceiling from the recorded JSON artifacts (pure Python + numpy, no GPU).
Inputs (read-only): results/ecro_util96a/cost_s1.0.json, results/gm1_c3urx1_91/results/econ/{life91,scale91}.json,
results/gm1_explicit87/results/econ/econ87.json, results/ecro86/results/econ/econ86.json.  Output: results/audit97/D/ceilings.json.
Every number below is arithmetic on recorded values; conventions ar
  -> Exit code 2
/usr/bin/bash: -c: line 70: unexpected EOF while looking for matching `''
RUN [Compute ceilings and sensitivities from recorded JSON] cd /c/Projects/ReST-GI && cat > experiments/audit97/D/ceilings.py <<'EOF'
"""WL97 audit track D: recomputation of every cost ratio / Amdahl ceiling from the recorded JSON artifacts (pure Python + numpy, no GPU).
Inputs (read-only): results/ecro_util96a/cost_s1.0.json, results/gm1_c3urx1_91/results/econ/{life91,scale91}.json,
results/gm1_explicit87/results/econ/econ87.json, results/ecro86/results/econ/econ86.json.  Output: results/audit97/D/ceilings.json.
Every number below is arithmetic on recor
WRITE c:\Projects\ReST-GI\experiments\audit97\D\ceilings.py
WRITE c:\Projects\ReST-GI\experiments\audit97\D\show.py
RUN [Run ceilings computation and print sections A-D] python experiments/audit97/D/ceilings.py && python experiments/audit97/D/show.py A_WL96A B_WL96_critic_from_WL91 C_source_only_work_convention D_conventions
  -> written c:\Projects\ReST-GI\results\audit97\D\ceilings.json
===== A_WL96A
R0_edit_ms: 144.1319
R0_source_ms: 88.53
R0_edit_parts_ms:
  geometry_update: 30.1848
  prefix_order: 2.0244
  relation_coefficients: 4.9531
  relation_update: 18.1444
  render: 59.5132
  source_q_table: 25.6825
  t ...[cut]... 
  fixed_ms: 0.0
  prefix_ms: 2.0505
  tail_ms: 3.3108
  U_ZERO: 2.6147
  note: only the operator applications (1 prefix + 3 tail orders)
  U_LAM24_no_generator: 2.43
  U_LAM64_no_generator: 2.3379
edit_to_frame_wall_as_recorded:
  fixed_ms: 138.6901
  prefix_ms: 2.0505
  tail_ms: 3.3108
  U_ZERO: 1.0235
  note: WL96A R0 edit parts
  U_LAM24_no_generator: 1.0224
  U_LAM64_no_generator: 1.0218
edit_to_table_wall_no_frame:
  fixed_ms: 79.1769
  prefix_ms: 2.0505
  tail_ms: 3.3108
  U_ZERO: 1.0408
  note: no frame
  U_LAM24_no_generator: 1.0388
  U_LAM64_no_generator: 1.0377
RUN [Run ceilings computation and print sections A-D] cd /c/Projects/ReST-GI && python experiments/audit97/D/ceilings.py && python experiments/audit97/D/show.py A_WL96A B_WL96_critic_from_WL91 C_source_only_work_convention D_conventions
RUN [Print sensitivity, work and overhead sections] python experiments/audit97/D/show.py E_sensitivity F_work G_WL86 C_overhead_WL87 | cut -c1-400 | head -260
  -> ===== E_sensitivity
lights_WL91_R0:
  q_ms_1: 25.1313
  q_ms_2: 40.642
  render_ms_1: 54.4132
  render_ms_2: 97.0603
  sweeps_ms_1: 5.5638
  sweeps_ms_2: 5.7377
  ceiling_remove_3_of_4_orders_1light: 1.0513
  ceiling_remove_3_of_4_orders_2lights: 1.0309
  n_two_light_events: 1
orders_R0_w ...[cut]...       wall_ms: 223.8271
      python_trace_ms: 112.0979
      gpu_kernel_ms: 85.0592
      kernel_share: 0.38
  source_only_wall:
    ecro_ms: 13.4953
    exp_ms: 53.019
    exp_q_ms_2lights_single_unwarmed_call: 43.1665
    ratio_ecro_over_exp: 0.2545
E3:
  q_wall_ms: 25.8662
  q_gpu_kernel_ms: 1.9377
  q_kernel_share: 0.0749
  state_selective_wall_ms: 19.1653
  state_selective_gpu_kernel_ms: 0.085
  source_enc_wall_ms: 5.074
  source_enc_gpu_kernel_ms: 0.022
  ecro_inference_wall_ms: 10.5022
  exp_apply_wall_ms: 10.4826
  table_frame_by_spp:
    4:
      wall_ms: 35.8964
RUN [Print sensitivity, work and overhead sections] cd /c/Projects/ReST-GI && python experiments/audit97/D/show.py E_sensitivity F_work G_WL86 C_overhead_WL87 | cut -c1-400 | head -260
RUN [Check 'plain' render mode and architecture constants] grep -n "plain" experiments/ecro86/pt86.py | head -20; grep -n "channels_R\|global_rank\|token_dim\|message_rounds" -r experiments/ecro86/protocol/*.json | head; grep -n "\"P\"\|\"Mq\"\|M_ray\|seed_q" experiments/gm1_explicit87/protocol/*.json | head
  -> 202:    def trace(self, ray, N, rng, sd, ref, kmax, lights, mode, B=None, probe_y=None, probe_ok=None, plain=False):
233:                add('tot' if plain else 'hl', hl)
234:            if k == 2 and not plain and mode == 'cam':
260:                if plain:
300:    def render(self, lights, ref, ca ...[cut]... max=None):
316:            out, _ = self.trace(ray, N, rng, sd, ref, kmax, lights, mode, B=B, plain=plain)
468:    if cmd == 'render':                       # render <root> <cfg> <ER|P> <seed> <spp> <tag> [--ls=] [--plain] [--kmax=] [--res=] [--cam=2]
475:        acc, T_ = ctx.render(Lights(FX['light_sets'][ls]), ref, cam, res, spp, seed, 'cam', plain='--plain' in sys.argv,
experiments/ecro86/protocol/ecro86.json:55: "architecture": {"token_dim": 128, "channels_R": 16, "global_rank": 24, "message_rounds": 3,
36:  "M_ray": 65536,
41:  "P": 16,
42:  "Mq": 1024,
43:  "seed_q": 87003,
366:  "P": {
RUN [Check 'plain' render mode and architecture constants] cd /c/Projects/ReST-GI && grep -n "plain" experiments/ecro86/pt86.py | head -20; grep -n "channels_R\|global_rank\|token_dim\|message_rounds" -r experiments/ecro86/protocol/*.json | head; grep -n "\"P\"\|\"Mq\"\|M_ray\|seed_q" experiments/gm1_explicit87/protocol/*.json | head
RUN [Grep WL96 worklog cost claims] grep -n -i "amdahl\|1\.09\|1\.16\|60 %\|dirty\|generator\|11 ms\|decorative\|overhead\|cost" docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | cut -c1-700 | head -40
  -> 18:     - one with a cost / scaling / explicit-comparator lens.
19:  4. **Synthesis and verdict** (this worklog) by the supervising session. It verified the cost anchors (WL91:228-246; WL87:175, 281) and the canon lines.
20:- **Agent outputs** are kept verbatim in [docs/plans/wl96_spatio_angular_car ...[cut]... d uncontended on the same GPU, with a full-scale utilisation probe, at N = 1,270 / 2,802 / 5,080. Then U_max = lifecycle / (lifecycle − t_sweeps + t_Λ [+ t_generator per edit]) for edit→frame and source→frame.
233:| U_max ≥ 1.25 at the largest N, H_acc within the bound | COST_ONLY scaling hypothesis |
236:**Separable, optional (comparator selection, not architecture).** DQ g4 vs photons 40 k / 160 k vs ESC-R r4 on the same workloads, scored against PT e2_p (not the in-house oracle), with dirty-cache counts and ms per edit for the box edit and the slab rotation.
264:  - cost critic: `dirty.py`.
RUN [Grep WL96 worklog cost claims] cd /c/Projects/ReST-GI && grep -n -i "amdahl\|1\.09\|1\.16\|60 %\|dirty\|generator\|11 ms\|decorative\|overhead\|cost" docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | cut -c1-700 | head -40
RUN [Read cand91 table timing boundaries] grep -n "def table" -A 22 experiments/gm1_c3urx1_91/cand91.py | head -60; grep -n "def c3ur_table" -A 12 experiments/gm1_c3urx1_91/cand91.py
  -> 97:    def table(self, ctx, lt, orders=False, J_=J, timing=None):
98-        """B (torch [N, NB, 3]) for the current geometry and the given lights; orders=True also returns [L2..L5]."""
99-        T_ = timing if timing is not None else {}
100-        sync(); t0 = time.perf_counter()
101-        rad  ...[cut]... nce': self.n * 3 * 4,
115-                'scratch_incident_cells': self.rel.G.N * R9.NC * 3 * 4}
116-
117-
118-def c3ur_table(rc, q, orders=False, J_=J):
119-    """The preserved C3UR table recurrence (rel88.recurrence semantics) on torch."""
118:def c3ur_table(rc, q, orders=False, J_=J):
119-    """The preserved C3UR table recurrence (rel88.recurrence semantics) on torch."""
120-    L = torch.as_tensor(np.asarray(q, np.float32), device=dev)
121-    ords = []
122-    for _ in range(J_):
123-        L = rc.apply(L); ords.append(L)
124-    B = sum(ords)
125-    return (B, ords) if orders else B
RUN [Read cand91 table timing boundaries] cd /c/Projects/ReST-GI && grep -n "def table" -A 22 experiments/gm1_c3urx1_91/cand91.py | head -60; grep -n "def c3ur_table" -A 12 experiments/gm1_c3urx1_91/cand91.py
RUN [Read WL96A final() decision logic and final.json] sed -n 689,740p experiments/ecro_util96a/w96a.py; python -c "
import json
f=json.load(open('results/ecro_util96a/final.json'))
print({k:(v if not isinstance(v,(dict,list)) else '...') for k,v in f.items()})
print(json.dumps(f.get('cost',f.get('stage3','')),indent=1)[:1500])
"
  -> def final(R, outdir):
    s0 = jl(os.path.join(R, 'results', 's0.json'))
    ext = os.path.exists(os.path.join(R, 'results', 's1_ext.json'))
    s1 = jl(os.path.join(R, 'results', 's1_ext.json' if ext else 's1.json'))
    o1 = s1['decision']['outcome']
    s2 = None
    acc_conf = False
    if o1 == ...[cut]... s: c['U'] for s, c in costs.items()}, 'LEARNED_RESOLVENT_UTILITY': v}
    jd(res, os.path.join(R, 'results', 'final.json'))
    print('FINAL', 'DEV' if DEV else v, flush=True)
    try:
        figures(R, outdir, ext)
    except Exception as e:                                                       # figures never change the verdict
        print('FIG_ERROR', repr(e), flush=True)


def figures(R, outdir, ext):
{'stage0': True, 'S1': 'S1_NO_HEADROOM', 'S1_ext': False, 'S2': None, 'accuracy_confirmed': False, 'cost_met': False, 'U': '...', 'LEARNED_RESOLVENT_UTILITY': 'NO_MEANINGFUL_HEADROOM'}
""
RUN [Read WL96A final() decision logic and final.json] cd /c/Projects/ReST-GI && sed -n 689,740p experiments/ecro_util96a/w96a.py; python -c "
import json
f=json.load(open('results/ecro_util96a/final.json'))
print({k:(v if not isinstance(v,(dict,list)) else '...') for k,v in f.items()})
print(json.dumps(f.get('cost',f.get('stage3','')),indent=1)[:1500])
"
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
