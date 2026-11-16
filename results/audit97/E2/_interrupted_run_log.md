# Track E2: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List relevant worklogs, code and result directories] ls docs/worklogs | grep -E "^9[4-6]" ; ls experiments/ecro_attr95a experiments/ecro_mat94 experiments/gm1_explicit87 experiments/ecro86 experiments/ecro86a; ls results/ecro_attr95a | head -50; ls docs/audits 2>/dev/null; ls experiments/audit97 results/audit97 2>/dev/null
RUN [Count lines and list WL95A headings] wc -l docs/worklogs/95A_* experiments/ecro_attr95a/*.py experiments/ecro_mat94/*.py experiments/gm1_explicit87/*.py experiments/ecro86/*.py experiments/ecro86a/*.py && grep -n "^#\|^##" docs/worklogs/95A_*.md | head -80
  -> Exit code 2
94A_WORKLOG_ECRO_MATERIAL_FIDELITY_RETENTION_SMOKE.md
94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md
94C_WORKLOG_ECRO_PANEL_MEDIATED_QUERY_SUPERVISION_SMOKE.md
94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md
94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
9 ...[cut]... py
protocol
pt86.py
run86.sh
scene86.py
tests86.py
train86.py
viz86.py
watch86.sh

experiments/ecro86a:
analyze86a.py
attrib86a.py
d2_86a.py
d3_86a.py
diag86a.py
protocol
pt86a.py
run86a.sh
run86a2.sh
viz86a.py
viz86a2.py

experiments/ecro_attr95a:
a95.py
protocol
run95a.sh
wl95calc

experiments/ecro_mat94:
an94.py
c94.py
mat94.py
posthoc94.py
protocol
run94.sh
viz94.py

experiments/gm1_explicit87:
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
an.json
chain.status
deliv.npz
gpu.txt
main.json
ops95a.npz
s0.json
  -> 213 docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
   628 experiments/ecro_attr95a/a95.py
   379 experiments/ecro_mat94/an94.py
   310 experiments/ecro_mat94/c94.py
   578 experiments/ecro_mat94/mat94.py
    36 experiments/ecro_mat94/posthoc94.py
   168 experiments ...[cut]... ability & Flight Attribution Smoke (2026-11-15)
3:## Origin and scope
17:## A. Implementation Fact
49:## B. Measurement
53:### B1. Sender (CK2 S1 field, 42 patches; deterministic; `main.json`)
68:### B2. Delivery: exactly-2-event S1-mediated exitance (`an.json`)
88:### B3. Integrated ROI (CK2 ROI_T, 2 × 4,096 spp per arm; noise bound about 0.0083 per arm, so a difference needs > about 0.0167)
101:### B4. Compute (RTX 3080 Ti 12 GB, LabServer63)
116:## C. Observation
140:## D. Interpretation
157:## E. Architecture Verdict (predeclared rules)
183:## Unresolved questions
190:## F. Reproducibility
RUN [List relevant worklogs, code and result directories] cd /c/Projects/ReST-GI && ls docs/worklogs | grep -E "^9[4-6]" ; ls experiments/ecro_attr95a experiments/ecro_mat94 experiments/gm1_explicit87 experiments/ecro86 experiments/ecro86a; ls results/ecro_attr95a | head -50; ls docs/audits 2>/dev/null; ls experiments/audit97 results/audit97 2>/dev/null
RUN [Count lines and list WL95A headings] cd /c/Projects/ReST-GI && wc -l docs/worklogs/95A_* experiments/ecro_attr95a/*.py experiments/ecro_mat94/*.py experiments/gm1_explicit87/*.py experiments/ecro86/*.py experiments/ecro86a/*.py && grep -n "^#\|^##" docs/worklogs/95A_*.md | head -80
RUN [Read WL95A worklog] sed -n 1,213p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
  -> # Worklog — WL95A: Sharp Sender Representability & Flight Attribution Smoke (2026-11-15)

## Origin and scope

- **Brief.** "WL95A — Sharp Sender Representability & Flight Attribution Smoke" (user). It is a suffix of WL95 and a bounded, learning-free attribution batch.
- **Proposition tested.** "The ...[cut]... d ~/wl95_calc ~/rest_x92a`. The chain runs `a95.py stage0 | main | an` and `mat94.py render`. Extension to 64 draws re-runs `main` with `A95_EXTEND=1`, triggered by the predeclared rule after the 16-draw pilot.
- **Outputs.**
  - `results/ecro_attr95a/`: s0.json, main.json, an.json, deliv.npz (per-patch exitance per arm, deployed and per draw), ops95a.npz, chain.status, gpu.txt.
  - `results/evaluation/95A/`, with a sha256 manifest:
    - figA1 senders;
    - figA2 delivery;
    - figA3 box-A footprint maps (shared scale);
    - figA4 ROI renders and signed errors (shared scale, one exposure).
RUN [Read WL95A worklog] cd /c/Projects/ReST-GI && sed -n 1,213p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
RUN [Read scene86.py] cat experiments/ecro86/scene86.py
RUN [Read pt86.py] cat experiments/ecro86/pt86.py
  -> """WL86 transport module (Dr.Jit on Mitsuba 3.5.1 ray intersection): reference path tracer, LOW / ECRO renderer, field probes,
geometry state rays, source encoding, geometric masks.

Estimator (as WL85): at each surface vertex next-event estimation to every light, then BSDF sampling; no Russian roul ...[cut]... geom, ls = config(cfg)
        ls = opt.get('--ls', ls)
        ctx = Ctx(geom)
        lt = Lights(FX['light_sets'][ls])
        t0 = time.perf_counter()
        fa, ca = ctx.field(lt, spp // 2, seed)
        fb, cb = ctx.field(lt, spp // 2, seed + 500)
        out = os.path.join(root, 'results', 'fields'); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), fa=fa, fb=fb, ca=ca, cb=cb)
        print(tag, 'field', round(time.perf_counter() - t0, 2), 's', 'valid', float((ca > 0).mean()), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
  -> """WL86 scene family, patch / direction-bin discretisation and sampling rules (numpy only; no renderer import).

Room x,y in [-2,2], z in [0,3], open towards -y. Stationary: floor (ER B), ceiling + 3 walls (Lambert), panel (ER S1), sphere (ER S2).
Movable: box A and box B (Lambert), each given by ce ...[cut]... ntinue
        return [{'pos': [float(v) for v in pos], 'rgb': [FX['light_intensity']] * 3}]


def light_arrays(lights):
    """-> pos [2,3], rgb [2,3], axis [2,3], cos_cut [2] (point light: cos_cut = -2; unused slot: rgb = 0)."""
    pos = np.zeros((2, 3)); rgb = np.zeros((2, 3)); axis = np.tile(Z, (2, 1)); cc = np.full(2, -2.0)
    for i, l in enumerate(lights):
        pos[i] = l['pos']; rgb[i] = l['rgb']
        if 'aim' in l:
            a = np.array(l['aim'], np.float64) - pos[i]; axis[i] = a / np.linalg.norm(a); cc[i] = math.cos(math.radians(l['cone_deg']))
    return pos, rgb, axis, cc
RUN [Read scene86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/scene86.py
RUN [Read pt86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/pt86.py
RUN [Read pt86a.py] cat experiments/ecro86a/pt86a.py
RUN [Outline mat94.py] grep -n "def \|^class \|CKF\|e2_p\|ctx_ck2\|CK2\|ROI_T" experiments/ecro_mat94/mat94.py | head -120
  -> """WL86A diagnostic sources for the frozen WL86 transport module (pt86 is imported unchanged; nothing here is a runtime input of ECRO).

Field probes (radiance leaving a surface point made of 2..5 scattering events, exactly pt86.Ctx.field) under controlled incident fields:
  'true'            the li ...[cut]... e irradiance in the bin of its direction); `visible` = fraction of 256 patch points it reaches."""
    G = ctx.G
    j = int(bm['pid'])
    w = np.array(bm['w'], np.float64)
    loc = np.array([w @ G.PT[j], w @ G.PB[j], w @ G.PN[j]])
    iz = int(np.clip(math.floor(loc[2] * S.NZ), 0, S.NZ - 1)); ip = int(np.clip(math.floor((math.atan2(loc[1], loc[0]) + math.pi) * S.NPHI / (2 * math.pi)), 0, S.NPHI - 1))
    frac = 1.0 if bm.get('half') is None else 0.5
    s = np.zeros((ctx.N, NB, 3), np.float32)
    s[j, iz * S.NPHI + ip] = bm['E'] * loc[2] * frac
    return s, iz * S.NPHI + ip, float(loc[2])
  -> 1:"""WL94 (protocol ecro_mat94): per-patch material machinery, Stage A tests, Stage B frozen material counterfactuals on CK2.
11:  python mat94.py fref   <R> <variant>                            CK2 field reference, two halves (results/fields/CKF_<v>.npz)
12:  python mat94.py iref   <R> <variant> <r ...[cut]... ': front_, 'sid': sid_,
414:def pre(R, R92A, R92B, R93D, RCK):
419:    PT, S, A, FX, ctx, lt = ctx_ck2(R)
441:    def hemi(k):
462:    for nm, geom, lts in (('CK2', None, lt), ('E2_T1', FX['E2']['T1'], PT.Lights(FX['light_sets'][FX['E2']['light_set']]))):
487:    e0 = (f0['e2_p'] + f0['e2_o'])[tg].astype(np.float64).sum((1, 2)); e1 = (f1['e2_p'] + f1['e2_o'])[tg].astype(np.float64).sum((1, 2))
508:    # frozen reproduction: WL93D E2 T1 tables (default map) and addendum CK2 tables
555:    V = {'roi_pixels': int(M['ROI_T'].sum()), 'diff_pixels': int(M['DIFF'].sum()), 'diff_rule': M['DIFF_rule'],
RUN [Read pt86a.py] cd /c/Projects/ReST-GI && cat experiments/ecro86a/pt86a.py
RUN [Outline mat94.py] cd /c/Projects/ReST-GI && grep -n "def \|^class \|CKF\|e2_p\|ctx_ck2\|CK2\|ROI_T" experiments/ecro_mat94/mat94.py | head -120
RUN [Read mat94.py CK2 context and field reference code] sed -n 1,60p experiments/ecro_mat94/mat94.py && sed -n 120,310p experiments/ecro_mat94/mat94.py
  -> """WL94 (protocol ecro_mat94): per-patch material machinery, Stage A tests, Stage B frozen material counterfactuals on CK2.

Per-patch materials are consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors); new Lambert rows are a ...[cut]... ds=np.array(FREF_SEEDS), **out)
    jd({'variant': v, 'spp_per_half': spp, 'seeds': FREF_SEEDS, 'seconds': time.time() - t0, 'patch_sha256': _INST['patch_sha'],
        'pmat_digest': pm_digest(ctx.G.PMAT)}, fn.replace('.npz', '.json'))
    print('FREF', v, round(time.time() - t0, 1), 's', flush=True)


def iref(R, v, r):
    PT, S, A, FX, ctx, lt = ctx_ck2(R)
    set_pmat(PT, ctx, variant_pmat(ctx.G, v))
    cam = FX['camera']; spp = 32 if DEV else IREF_SPP; seed = 86100 + int(r)
    acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], spp, seed, 'cam', b=32)
    otag = f'REF_ER_CK_{v}_{TAG}_r{r}'
RUN [Read mat94.py CK2 context and field reference code] cd /c/Projects/ReST-GI && sed -n 1,60p experiments/ecro_mat94/mat94.py && sed -n 120,310p experiments/ecro_mat94/mat94.py
RUN [Read quad87.py] cat experiments/gm1_explicit87/quad87.py
  -> """WL87 discrete ER scattering on the WL86 patch x bin state (learning-free; frozen ER closures, Dr.Jit).

Notation (patch-local frame (t, b, n) of scene86; 64 equal-solid-angle bins, DOMEGA = 2 pi / 64; eval(wi, wo) = f(wi, wo) cos(wo)).
ER f is reciprocal (f(a, b) = f(b, a): symmetric GGX lobes, s ...[cut]...         # inside a patch-frame bin and above the local horizon
        cv = dr.select(keep, val * E * (dr.gather(F, qs['invom'], slot) * (1.0 / (P * Mq))), 0.0)
        dr.eval(cv, slot); dr.sync_thread()                                    # evaluate BEFORE the scatter (lazy-scatter defect)
        for c in range(3):
            dr.scatter_reduce(dr.ReduceOp.Add, buf[c], cv[c], slot)
        dr.eval(buf)
    dr.sync_thread()
    q = np.stack([np.asarray(x, np.float64) for x in buf], 1).reshape(N, NB, 3)
    ql = np.stack([np.asarray(x, np.float64) for x in bl], 1)
    return q + ql[:, None, :]
RUN [Read quad87.py] cd /c/Projects/ReST-GI && cat experiments/gm1_explicit87/quad87.py
RUN [Find local_matrices and fx92.inject] grep -rn "def local_matrices" -A 25 experiments/ecro_attr92/*.py | head -60; grep -n "def inject" -B3 -A 25 experiments/ecro_attr92/fx92.py | head -70
RUN [Read model86, ecro86 build_state, CK2 scene and fixture] cat experiments/ecro86/model86.py; sed -n 1,80p experiments/ecro86/ecro86.py; cat experiments/ecro_local93d/addendum_checker/scene_ck2.json; python -c "
import json;P=json.load(open('experiments/ecro86/protocol/ecro86.json'));print(json.dumps(P['fixture'],indent=0)[:3000]);print(P['representation'])"
  -> experiments/ecro_attr92/diag92.py:133:def local_matrices():
experiments/ecro_attr92/diag92.py-134-    import fx92
experiments/ecro_attr92/diag92.py-135-    import quad87 as Q
experiments/ecro_attr92/diag92.py-136-    import xport87 as X
experiments/ecro_attr92/diag92.py-137-    E87 = fx92.E87
experi ...[cut]... ved': c['moved'], 'light_set': 'L4', 'role': 'WL92 held-out condition (WL87 candidate ' + name + ')'}
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
44-    return FX[e][t], (p[2] if len(p) > 2 else FX[e]['light_set']), e, f'{e}_{t}'
45-
46-
47-def mask_file(R, R86, tag):
48-    p = tag.split('_')
  -> """WL86 ECRO v0 operator (torch): geometry-conditioned, source-independent, linear and non-negative.

Geometry state (no radiance): patch tables, frozen ER states of the patch materials, stratified hemisphere state rays
(hit patch, outgoing bin at the hit patch) -> generator network -> response fact ...[cut]... s_z': 4, 'bins_phi': 16, 'bins_note': '64 equal-solid-angle bins of the patch-local hemisphere (4 rings in cos(theta) x 16 azimuth sectors), piecewise constant', 'state_rays_per_bin': 8, 'source_points_per_patch': 16, 'table_seed': 86001, 'state': 'per patch 512 stratified hemisphere rays (8 per incident bin): hit patch, outgoing bin at the hit patch, segment (for the dirty test); patch tables', 'budget': {'table_floats': 243840, 'state_bytes_max': 20000000, 'model_parameters_max': 1000000}, 'not_changed_after_freeze': 'bins, patches, channel / rank sizes, network, loss, schedule, data sizes'}
RUN [Find local_matrices and fx92.inject] cd /c/Projects/ReST-GI && grep -rn "def local_matrices" -A 25 experiments/ecro_attr92/*.py | head -60; grep -n "def inject" -B3 -A 25 experiments/ecro_attr92/fx92.py | head -70
RUN [Read model86, ecro86 build_state, CK2 scene and fixture] cd /c/Projects/ReST-GI && cat experiments/ecro86/model86.py; sed -n 1,80p experiments/ecro86/ecro86.py; cat experiments/ecro_local93d/addendum_checker/scene_ck2.json; python -c "
import json;P=json.load(open('experiments/ecro86/protocol/ecro86.json'));print(json.dumps(P['fixture'],indent=0)[:3000]);print(P['representation'])"
RUN [Read a95.py first half] sed -n 1,330p experiments/ecro_attr95a/a95.py
  -> """WL95A (protocol ecro_attr95a): sharp sender representability and flight attribution smoke on CK2 'none' (learning-free diagnostics).
Production code (model86 / ecro86 / pt86 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.

  python a95.py stage0 <R> <R92A>     contract tests,  ...[cut]... 'passive_all': bool(all(p['passive'] for p in per)), 'nonneg': bool(min(p['min_entry'] for p in per) >= 0),
                           'mass_mean': float(np.mean([p['mass'] for p in per]))}
    out['sender'] = sm; out['operator'] = om
    # propagation: deployed + pooled draws
    rw0 = rw_of(u0, S)
    TAB = [k_ for k_ in snd]                                   # table senders
    RAY = ['X1c', 'X1']
    ex = {}; tabs_mean = {}; tabs_dep = {}

    def deliver(hit, dbin, seg, rw, store):
        res = {}
        for k_ in TAB:
            Dt = receive(flight(hit, dbin, rw, snd[k_], N, S), Mrcv)
RUN [Read a95.py first half] cd /c/Projects/ReST-GI && sed -n 1,330p experiments/ecro_attr95a/a95.py
RUN [Read a95.py second half (metrics)] sed -n 330,520p experiments/ecro_attr95a/a95.py
  -> Dt = receive(flight(hit, dbin, rw, snd[k_], N, S), Mrcv)
            res[k_] = Dt
        for k_, cen in (('X1c', True), ('X1', False)):
            L, nr = x1_rays(PT, ctx, lt, hit, seg, cen)
            res[k_] = receive(flight_ray(L, rw, N, S), Mrcv)
            store['rays_S1'] = nr
        retu ...[cut]... 02 or r['map_rel_l1_vs_A4'] > 0.10 + 0.02
                    or r['E_H'] > r['E_H_A4'] + r['tol'])
        r16 = 'INADEQUATE_ON_TESTED_WORKLOAD' if clear_fail('A1') and clear_fail('A2') else 'INCONCLUSIVE'
    # NEXT_DESIGN_TARGET
    comp = {}
    for sd in SEEDS:
        dS = img[f'dep_A0_s{sd}']['E_H'] - floor; dF = img['dep_A4']['E_H'] - floor; dG = img[f'G_s{sd}']['E_H'] - floor
        t_ = max(tol(f'dep_A0_s{sd}', 'floor'), tol('dep_A4', 'floor'), tol(f'G_s{sd}', 'floor'))
        comp[sd] = {'dS': dS, 'dF': dF, 'dG': dG, 'tol': t_}
    allc = lambda f: all(f(c) for c in comp.values())
RUN [Read a95.py second half (metrics)] cd /c/Projects/ReST-GI && sed -n 330,520p experiments/ecro_attr95a/a95.py
RUN [Inspect server environment and inputs] ssh -o ConnectTimeout=15 LabServer63 'cat ~/rest_x95a/env.sh; ls ~/rest_x95a ~/rest_x95a/results ~/rest_x96a ~/rest_x96a/code 2>&1 | head -60; ls -la ~/rest_x97 2>&1; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv; nproc; free -g | head -2; ls -la ~/rest_x94/results/fields/ | head; sha256sum ~/rest_x94/results/fields/CKF_none.npz'
  -> export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
/home/nd/rest_x95a:
chain.out
chain.status
code
data
env.sh
results

/ ...[cut]...        120
total 33644
drwxr-xr-x  2 nd nd    4096 Nov 14 10:47 .
drwxr-xr-x 12 nd nd    4096 Nov 14 11:10 ..
-rw-r--r--  1 nd nd     329 Nov 14 10:46 CKF_U03.json
-rw-r--r--  1 nd nd 5732154 Nov 14 10:46 CKF_U03.npz
-rw-r--r--  1 nd nd     328 Nov 14 10:46 CKF_U07.json
-rw-r--r--  1 nd nd 5735822 Nov 14 10:46 CKF_U07.npz
-rw-r--r--  1 nd nd     329 Nov 14 10:47 CKF_bx38.json
-rw-r--r--  1 nd nd 5736189 Nov 14 10:47 CKF_bx38.npz
-rw-r--r--  1 nd nd     330 Nov 14 10:47 CKF_bx38i.json
3a6238a8a42d1db467c0e3d0bcfef7b290c62ead78de9a9e0c5399af72ffaa67  /home/nd/rest_x94/results/fields/CKF_none.npz
RUN [Check frozen code checkout and WL95A run directory layout] ssh LabServer63 'cd ~/rest_x96a/code && cat COMMIT; du -sh experiments; ls experiments | head -80; ls ~/rest_x95a/results/fields ~/rest_x95a/data 2>&1 | head; ls -la ~/rest_x95a/results/fields/ | head -5; ls ~/rest_x95a/results/pre | head; ls ~/rest_x97/B ~/rest_x97/C1 ~/rest_x97/MAP'
  -> a1718f7
28M	experiments
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
ecro_own92a
ecro_q94c
ecro_s94d
ecro_stage93
ecro_stage93a
ecro_stage93b
ecro_s ...[cut]... on
server_digest_population.py

/home/nd/rest_x97/C1:

/home/nd/rest_x97/MAP:
blobs.tgz
blobs_rest_ecro86.tsv
blobs_rest_ecro86a.tsv
blobs_rest_gap90.tsv
blobs_rest_gm187.tsv
blobs_rest_rel88.tsv
blobs_rest_rel89.tsv
blobs_rest_x191.tsv
blobs_rest_x92.tsv
blobs_rest_x92a.tsv
blobs_rest_x92b.tsv
blobs_rest_x93.tsv
blobs_rest_x93a.tsv
blobs_rest_x93b.tsv
blobs_rest_x93c.tsv
blobs_rest_x93d.tsv
blobs_rest_x93d_ck2.tsv
blobs_rest_x94.tsv
blobs_rest_x94b.tsv
blobs_rest_x94c.tsv
blobs_rest_x94d.tsv
blobs_rest_x94r.tsv
blobs_rest_x95a.tsv
blobs_rest_x96a.tsv
norm_rest_ecro86.tsv
norm_rest_ecro86a.tsv
RUN [Inspect ER closure API and path setup module] wc -l experiments/fnto_compiled83/closure83_dr.py && grep -n "def \|class " experiments/fnto_compiled83/closure83_dr.py && grep -n "def init" -A 12 experiments/fnto_compiled81/layered81.py && cat experiments/ecro_local93d/ops93d.py | head -40; ssh LabServer63 'cat ~/rest_x95a/code/COMMIT; ls ~/rest_x95a/code'
  -> 202 experiments/fnto_compiled83/closure83_dr.py
29:def _mi():
34:def compiled_state(arm, p):
45:def expand_state(st):
52:class DrClosure83:
55:    def __init__(self, arm, state, idx=None):
79:    def _s(self, mu):
91:    def _refr(self, w):
99:    def _lobe(self, k, wi, wo, want_eval, want_pdf):
121 ...[cut]... in(HERE, 'protocol', 'ecro_local93d.json')))
FAMS = ['own', 'T0']
SEEDS = [8601, 8602]
ARMS = ['C0', 'C1_src', 'C1_rcv', 'C1_both', 'C2_src', 'C2_rcv', 'C2_both']
NB = 64


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


# ------------------------------------------------------------------------------------------------ configurations
def cfgs(R, R86):
    import prep93 as PR
    PR.inject(R)
    base = jl(os.path.join(R, 'results', 'configs.json'))['configs']
    C = [dict(c) for c in base]
c152120
AGENTS.md
COMMIT
docs
experiments
results
RUN [Inspect ER closure API and path setup module] cd /c/Projects/ReST-GI && wc -l experiments/fnto_compiled83/closure83_dr.py && grep -n "def \|class " experiments/fnto_compiled83/closure83_dr.py && grep -n "def init" -A 12 experiments/fnto_compiled81/layered81.py && cat experiments/ecro_local93d/ops93d.py | head -40; ssh LabServer63 'cat ~/rest_x95a/code/COMMIT; ls ~/rest_x95a/code'
RUN [Read ER closure implementation] sed -n 1,202p experiments/fnto_compiled83/closure83_dr.py
  -> """WL83 Dr.Jit runtime for the WL82 closures C0 / E / ER (semantics = experiments/fnto_compiled82/closure82.py, torch reference).

Compiled state per material (69 floats, identical layout for every arm; unused fields are 1):
  lam 24 (c * 8 + r) | au 7 | av 7 | f0 7 | sampling probs 8 | s 15 (c * 5  ...[cut]...      if self.arm == 'ER':
            tx, ty = wr.x * self.eta, wr.y * self.eta
            r2 = tx * tx + ty * ty
            wex = mi.Vector3f(tx, ty, dr.sqrt(dr.maximum(1.0 - r2, 0.0)))
            ok_exit = ~refr | ((r2 < 1.0) & (wr.z > 0.0))
            wr = dr.select(refr, wex, wr)
        wo = dr.select(dr.eq(sel, 0), mi.warp.square_to_cosine_hemisphere(u2), wr)
        f, pdf = self.eval_pdf(wi, wo, active)
        valid = active & ok_exit & (wo.z > EPS) & (pdf > 0)
        w = dr.select(valid, f / dr.select(valid, pdf, 1.0), 0.0)
        return wo, dr.select(valid, pdf, 0.0), w, valid
RUN [Read ER closure implementation] cd /c/Projects/ReST-GI && sed -n 1,202p experiments/fnto_compiled83/closure83_dr.py
RUN [Inspect M64 build config and original delivery artifacts] grep -n "M_ray\|M_dense\|seed_ray\|seed_dense" -r experiments/gm1_explicit87/protocol/*.json experiments/ecro_attr92/fx92.py experiments/gm1_explicit87/fx87.py | head; sed -n 1,22p experiments/ecro_attr92/fx92.py; cat experiments/gm1_explicit87/fx87.py; grep -n "def realization_us" -A 8 experiments/gm1_explicit87/xport87.py; python - <<'EOF'
import numpy as np, json
z=np.load('results/ecro_attr95a/deliv.npz')
for k in z.files: print(k, z[k].shape)
a=json.load(open('results/ecro_attr95a/an.json')
  -> experiments/gm1_explicit87/protocol/gm1_explicit87.json:36:  "M_ray": 65536,
experiments/gm1_explicit87/protocol/gm1_explicit87.json:37:  "seed_ray": 87001,
experiments/gm1_explicit87/protocol/gm1_explicit87.json:38:  "M_dense": 16384,
experiments/gm1_explicit87/protocol/gm1_explicit87.json:39:  "se ...[cut]... 6,
"Rs": 0.024175660096948215,
"Rs_se": 0.034055179336480924,
"Ra": 0.23188168674623544,
"Ra_se": 0.046620248374830725,
"Rp": 0.1571067426298993,
"Rp_se": 0.06009806987301328,
"Rq_incident": 0.06643679185456292,
"Rq_incident_se": 0.0053191013018510916,
"Rq_outgoing": 0.16544489489167252,
"Rq_outgoing_se": 0.046560463778183264,
"E_X1_pool": 0.05129950858780274,
"E_M64_spool_pool": 0.44028793796393745,
"E_X1c_pool": 0.20840625121770204,
"E_q256_pool": 0.37385114610937453,
"E_A4_pool": 0.4871664042591579,
"verdict": "ANGULAR_PROJECTION_EVIDENCED",
"near_boundary_extend": false
}
RUN [Inspect M64 build config and original delivery artifacts] cd /c/Projects/ReST-GI && grep -n "M_ray\|M_dense\|seed_ray\|seed_dense" -r experiments/gm1_explicit87/protocol/*.json experiments/ecro_attr92/fx92.py experiments/gm1_explicit87/fx87.py | head; sed -n 1,22p experiments/ecro_attr92/fx92.py; cat experiments/gm1_explicit87/fx87.py; grep -n "def realization_us" -A 8 experiments/gm1_explicit87/xport87.py; python - <<'EOF'
import numpy as np, json
z=np.load('results/ecro_attr95a/deliv.npz')
for k in z.files: print(k, z[k].shape)
a=json.load(open('resu
RUN [Create my server workspace with a private copy of the frozen code and verify provenance hashes] ssh LabServer63 'mkdir -p ~/rest_x97/E2 && cd ~/rest_x97/E2 && mkdir -p code results logs && cp -r ~/rest_x95a/code/experiments code/ && cp ~/rest_x95a/code/COMMIT code/COMMIT && ls code/experiments/ecro86/data | head; ls ~/rest_x95a/results/wl95_calc | head; sha256sum ~/rest_x95a/results/ops95a.npz ~/rest_x95a/results/deliv.npz ~/rest_x95a/results/senders.npz; diff -rq ~/rest_x95a/code/experiments/ecro_attr95a ~/rest_x96a/code/experiments/ecro_attr95a; for d in ecro86 ecro86a gm1_explicit87 ecr
  -> mats86.json
mats86.npz
S1_ops.npz
wl95_calc.json
wl95_calc2.json
d4d02e5a2f423ce97749ad73287bc2c35f7907f60c03d1941a626778b90b0c70  /home/nd/rest_x95a/results/ops95a.npz
f2324a32effc80ca2124596716e39f79b71eb1e14caf3ad10b82e44b8a6b3064  /home/nd/rest_x95a/results/deliv.npz
5d365d19d6dffadcc223603b42bbbe2b3b399d26f7dcf78c570db68dcd51cbb6  /home/nd/rest_x95a/results/senders.npz
Only in /home/nd/rest_x95a/code/experiments/ecro_attr95a/wl95calc: __pycache__
diffdone
f2324a32effc80ca2124596716e39f79b71eb1e14caf3ad10b82e44b8a6b3064 *results/ecro_attr95a/deliv.npz
d4d02e5a2f423ce97749ad73287bc2c35f7907f60c03d1941a626778b90b0c70 *results/ecro_attr95a/ops95a.npz
WRITE c:\Projects\ReST-GI\experiments\audit97\E2\e2_chain.py
RUN [Check module import side effects] sed -n 1,40p experiments/ecro_attr92/diag92.py | head -60; grep -n "^import\|^from\|open(.*'w'\|savez\|makedirs" experiments/gm1_explicit87/xport87.py experiments/ecro86a/pt86a.py experiments/ecro_local93d/ops93d.py experiments/ecro_attr92/fx92.py | head -30
  -> """WL92 diagnostics on the frozen ECRO v0 family (stages S0-S5 of protocol ecro_attr92). Nothing here is a deployable candidate.

  python diag92.py prov     <R> <R86>                 provenance hashes (checkpoints, data, references, environment)
  python diag92.py split    <R> <R86> <tag>           ...[cut]...  exist_ok=True)
experiments/ecro_local93d/ops93d.py:183:    np.savez(os.path.join(R, 'results', 'ops', f'{fam}_s{seed}.npz'), idx=idx, **out)
experiments/ecro_local93d/ops93d.py:412:    os.makedirs(os.path.join(R, 'results', 'tables'), exist_ok=True)
experiments/ecro_local93d/ops93d.py:413:    np.savez_compressed(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'), **out)
experiments/ecro_local93d/ops93d.py:501:        os.makedirs(os.path.join(a[0], 'results', d), exist_ok=True)
experiments/ecro_attr92/fx92.py:6:import json, os, sys
experiments/ecro_attr92/fx92.py:12:import fx87  # noqa: E402
RUN [Check module import side effects] cd /c/Projects/ReST-GI && sed -n 1,40p experiments/ecro_attr92/diag92.py | head -60; grep -n "^import\|^from\|open(.*'w'\|savez\|makedirs" experiments/gm1_explicit87/xport87.py experiments/ecro86a/pt86a.py experiments/ecro_local93d/ops93d.py experiments/ecro_attr92/fx92.py | head -30
NOTE: Now run the checks stage on the server.
RUN [Run sanity, identity and analytic Lambert checks on the server] scp -q experiments/audit97/E2/e2_chain.py LabServer63:~/rest_x97/E2/ && ssh LabServer63 'cd ~/rest_x97/E2 && file e2_chain.py && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && . ~/rest_x95a/env.sh && (time $PY e2_chain.py checks ~/rest_x97/E2) > logs/checks.log 2>&1; tail -150 logs/checks.log'
  -> e2_chain.py: Python script, ASCII text executable
889 MiB, 0 %
Traceback (most recent call last):
  File "/home/nd/rest_x97/E2/e2_chain.py", line 591, in <module>
    checks(E)
  File "/home/nd/rest_x97/E2/e2_chain.py", line 530, in checks
    ana[j] = lambert_polygon_irradiance(yy, G.PN[p], V).mean()
  File "/home/nd/rest_x97/E2/e2_chain.py", line 461, in lambert_polygon_irradiance
    E_ += th * (cr / s[:, None]) @ n
ValueError: operands could not be broadcast together with shapes (2304,) (2304,3) 

real	0m1.788s
user	0m5.301s
sys	0m5.361s
NOTE: Now run the checks stage on the server.
RUN [Run sanity, identity and analytic Lambert checks on the server] cd /c/Projects/ReST-GI && scp -q experiments/audit97/E2/e2_chain.py LabServer63:~/rest_x97/E2/ && ssh LabServer63 'cd ~/rest_x97/E2 && file e2_chain.py && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && . ~/rest_x95a/env.sh && (time $PY e2_chain.py checks ~/rest_x97/E2) > logs/checks.log 2>&1; tail -150 logs/checks.log'
RUN [Check line endings of the script] wc -l experiments/audit97/E2/e2_chain.py && file experiments/audit97/E2/e2_chain.py && ssh LabServer63 'wc -l ~/rest_x97/E2/e2_chain.py; head -c 300 ~/rest_x97/E2/e2_chain.py | od -c | head -5'
  -> 605 experiments/audit97/E2/e2_chain.py
experiments/audit97/E2/e2_chain.py: Python script, ASCII text executable
605 /home/nd/rest_x97/E2/e2_chain.py
0000000   "   "   "   W   L   9   7       a   u   d   i   t   ,       t
0000020   r   a   c   k       E   2   :       i   n   d   e   p   e   n
0000040   d   e   n   t       r   e   p   r   o   d   u   c   t   i   o
0000060   n       o   f       t   h   e       W   L   9   5   A       d
0000100   e   l   i   v   e   r   y       c   h   a   i   n       (   P
RUN [Check line endings of the script] cd /c/Projects/ReST-GI && wc -l experiments/audit97/E2/e2_chain.py && file experiments/audit97/E2/e2_chain.py && ssh LabServer63 'wc -l ~/rest_x97/E2/e2_chain.py; head -c 300 ~/rest_x97/E2/e2_chain.py | od -c | head -5'
RUN [Inspect doubled blank lines] sed -n 1,14p experiments/audit97/E2/e2_chain.py | cat -A | cut -c1-80; grep -c '^$' experiments/audit97/E2/e2_chain.py
  -> """WL97 audit, track E2: independent reproduction of the WL95A delivery chain (P
$
Own implementation of: relation drawing (state rays), hit patch / outgoing bin, 
encoding, a physical q table, the 2 x 2 (direction x position) exact / projected
Primitives taken from the frozen code (read-only copy at <R>/code, commit c15212
mat94.ctx_ck2), Mitsuba ray intersection, ctx.shape_id, the ER closure (closure8
M64 matrix (ops95a.npz key A4 = quad87 enc_matrix row 1). a95.py and an94.py are
$
  python e2_chain.py checks <R>            sanity / identity / analytic Lambert 
  python e2_chain.py senders <R>           source encodings, tables, q table, ow
  python e2_chain.py flight <R> <D> <tag> <mode> <seed0>   D relation draws (mod
  python e2_chain.py aref <R> <log2 nper> <nbatch> <seed0> <tag> [mat] area-form
"""$
import hashlib, json, math, os, sys, time$
60
RUN [Inspect doubled blank lines] cd /c/Projects/ReST-GI && sed -n 1,14p experiments/audit97/E2/e2_chain.py | cat -A | cut -c1-80; grep -c '^$' experiments/audit97/E2/e2_chain.py
RUN [Fix operator precedence bug and rerun checks] python - <<'EOF'
p='experiments/audit97/E2/e2_chain.py'
s=open(p,newline='').read()
s=s.replace("E_ += th * (cr / s[:, None]) @ n","E_ += th * ((cr / s[:, None]) @ n)")
open(p,'w',newline='\n').write(s)
EOF
scp -q experiments/audit97/E2/e2_chain.py LabServer63:~/rest_x97/E2/ && ssh LabServer63 'cd ~/rest_x97/E2 && . ~/rest_x95a/env.sh && (time $PY e2_chain.py checks ~/rest_x97/E2) > logs/checks.log 2>&1; tail -150 logs/checks.log'
  -> {
 "cos_convention_lambert_mat3": {
  "albedo": 0.699999988079071,
  "max_rel_dev_vs_rho_over_pi_cos_second_arg": 1.785595660219741e-07,
  "max_rel_dev_vs_cos_first_arg": 95.72412295812694
 },
 "cos_convention_lambert_mat4": {
  "albedo": 0.5,
  "max_rel_dev_vs_rho_over_pi_cos_second_arg": 1.2883038 ...[cut]... here_flight": 0.497720676551959,
  "area_estimator": 0.498643331695348,
  "all54_total_hemi_over_area": 0.9998198583548565,
  "area_total_rel_se": 0.0007595727784679142,
  "per_patch_max_rel_dev_hemi_vs_area": 0.024677939192139675
 },
 "CKF_none": {
  "sha256": "3a6238a8a42d1db467c0e3d0bcfef7b290c62ead78de9a9e0c5399af72ffaa67",
  "spp_per_half": 4096,
  "seeds": [
   94301,
   94302
  ],
  "pmat_equal_default": true,
  "e2_p_on_panel_sum": 0.0
 },
 "wl95a_deliv_sha256": "f2324a32effc80ca2124596716e39f79b71eb1e14caf3ad10b82e44b8a6b3064"
}
DONE checks

real	0m11.547s
user	0m45.456s
sys	0m55.172s
RUN [Fix operator precedence bug and rerun checks] cd /c/Projects/ReST-GI && python - <<'EOF'
p='experiments/audit97/E2/e2_chain.py'
s=open(p,newline='').read()
s=s.replace("E_ += th * (cr / s[:, None]) @ n","E_ += th * ((cr / s[:, None]) @ n)")
open(p,'w',newline='\n').write(s)
EOF
scp -q experiments/audit97/E2/e2_chain.py LabServer63:~/rest_x97/E2/ && ssh LabServer63 'cd ~/rest_x97/E2 && . ~/rest_x95a/env.sh && (time $PY e2_chain.py checks ~/rest_x97/E2) > logs/checks.log 2>&1; tail -150 logs/checks.log'
RUN [Build source encodings, sender tables, own q table and own M64] ssh LabServer63 'cd ~/rest_x97/E2 && . ~/rest_x95a/env.sh && (time $PY e2_chain.py senders ~/rest_x97/E2) > logs/senders.log 2>&1; tail -90 logs/senders.log; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
  -> {
 "light_visibility_fraction_on_panel": {
  "s16": 1.0,
  "pool": 1.0
 },
 "s16_vs_production": {
  "max_abs": 2.1456228967764446e-05,
  "rel_l1": 1.157507030907849e-06,
  "sum_mine": 277.6780535854181,
  "sum_prod": 277.6777321714908
 },
 "ops95a_sha256": "d4d02e5a2f423ce97749ad73287bc2c35f7907f60 ...[cut]... 3603b42bbbe2b3b399d26f7dcf78c570db68dcd51cbb6",
  "A4_table_max_abs_diff": 0.0007110265707979124,
  "A4_table_rel_l1": 1.1573978826184483e-06,
  "q256_vs_my_q": {
   "flux_rel_l1": 0.04281486325863393,
   "energy": 0.9897546765414444
  },
  "M64_spool_vs_mine": {
   "flux_rel_l1": 0.012224671984510594,
   "energy": 1.0029055345151419
  }
 },
 "incident": {
  "s16_energy_by_ring": [
   0.9999999999999998,
   0.0,
   0.0,
   0.0
  ],
  "spool_energy_by_ring": [
   1.0,
   0.0,
   0.0,
   0.0
  ]
 },
 "total_s": 20.964889526367188
}
DONE senders

real	0m21.869s
user	0m25.509s
sys	0m7.061s
602 MiB
RUN [Run the deployed-realization flight and a 64-draw timing test] ssh LabServer63 'cd ~/rest_x97/E2 && . ~/rest_x95a/env.sh && $PY e2_chain.py deployed ~/rest_x97/E2 > logs/deployed.log 2>&1; tail -3 logs/deployed.log; (time $PY e2_chain.py flight ~/rest_x97/E2 64 test64 shared 9719000) > logs/flight_test.log 2>&1; tail -8 logs/flight_test.log'
  -> deployed hits [4380]
DONE deployed
flight test64 32 / 64 0.9 s
flight test64 64 / 64 1.7 s
DONE flight

real	0m2.467s
user	0m8.617s
sys	0m8.176s
RUN [Launch detached chain: 2048 shared draws, 2048 independent draws, area-form reference] ssh LabServer63 'cd ~/rest_x97/E2 && cat > run_main.sh <<"EOF"
#!/bin/bash
cd ~/rest_x97/E2
. ~/rest_x95a/env.sh
rm -f done_main
$PY e2_chain.py flight ~/rest_x97/E2 2048 shared2048 shared 9720000 > logs/flight_shared.log 2>&1
$PY e2_chain.py flight ~/rest_x97/E2 2048 indep2048 indep 9740000 > logs/flight_indep.log 2>&1
$PY e2_chain.py aref ~/rest_x97/E2 24 16 9750000 main > logs/aref_main.log 2>&1
touch done_main
EOF
chmod +x run_main.sh && file run_main.sh && setsid nohup ./run_main.sh > logs/
  -> run_main.sh: Bourne-Again shell script, ASCII text executable
launched
NOTE: Meanwhile, write the analysis script.
WRITE c:\Projects\ReST-GI\experiments\audit97\E2\e2_analyze.py
RUN [Check chain progress] ssh LabServer63 'cd ~/rest_x97/E2 && ls; tail -2 logs/flight_shared.log logs/flight_indep.log logs/aref_main.log 2>/dev/null; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
  -> code
e2_chain.py
logs
results
run_main.sh
1423 MiB, 7 %
RUN [Diagnose the detached chain] ssh LabServer63 'cd ~/rest_x97/E2 && ls logs results; cat logs/run_main.out; pgrep -u nd -af "e2_chain|run_main" | head'
  -> logs:
aref_main.log
checks.log
deployed.log
flight_indep.log
flight_shared.log
flight_test.log
run_main.out
senders.log

results:
checks.json
flight_deployed.npz
flight_indep2048.json
flight_indep2048.npz
flight_shared2048.json
flight_shared2048.npz
flight_test64.json
flight_test64.npz
ref_boxA.npz
 ...[cut]... me/nd/NeuralShaderDemo/external/8dna26/.venv/bin/python e2_chain.py aref /home/nd/rest_x97/E2 24 16 9750000 main
3467972 /usr/sbin/tailscaled be-child ssh --login-shell=/bin/bash --uid=1010 --gid=1010 --groups=1010,27,100,988 --local-user=nd --home-dir=/home/nd --remote-user=dna10165@gmail.com --remote-ip=100.70.31.19 --has-tty=false --tty-name= --force-v1-behavior --cmd=cd ~/rest_x97/E2 && ls logs results; cat logs/run_main.out; pgrep -u nd -af "e2_chain|run_main" | head
3467980 /bin/bash -c cd ~/rest_x97/E2 && ls logs results; cat logs/run_main.out; pgrep -u nd -af "e2_chain|run_main" | head
RUN [Copy flight outputs locally and run the analysis] mkdir -p results/audit97/E2 && scp -q LabServer63:~/rest_x97/E2/results/{checks.json,senders.json,senders_e2.npz,flight_deployed.npz,flight_shared2048.npz,flight_shared2048.json,flight_indep2048.npz,flight_indep2048.json,ref_boxA.npz,wl95a_deliv_boxA.npz} results/audit97/E2/ && ls -la results/audit97/E2 && python experiments/audit97/E2/e2_analyze.py results/audit97/E2
  -> total 15426
drwxr-xr-x 1 dna10 197609       0 Nov 16 02:57 .
drwxr-xr-x 1 dna10 197609       0 Nov 16 02:57 ..
-rw-r--r-- 1 dna10 197609    3502 Nov 16 02:57 checks.json
-rw-r--r-- 1 dna10 197609    9905 Nov 16 02:57 flight_deployed.npz
-rw-r--r-- 1 dna10 197609     173 Nov 16 02:57 flight_indep2048 ...[cut]... 0.443 32:1.417/0.434 64:1.417/0.430 128:1.417/0.428 256:1.417/0.428 512:1.417/0.428 1024:1.417/0.428 2048:1.417/0.428
   convergence X1c 1:1.070/0.735 2:1.070/0.610 4:1.070/0.512 8:1.070/0.435 16:1.070/0.375 32:1.070/0.334 64:1.070/0.307 128:1.070/0.299 256:1.070/0.294 512:1.070/0.291 1024:1.070/0.291 2048:1.070/0.291
   convergence X1 1:1.011/0.713 2:1.011/0.584 4:1.011/0.471 8:1.011/0.373 16:1.011/0.292 32:1.011/0.225 64:1.011/0.178 128:1.011/0.142 256:1.011/0.119 512:1.011/0.104 1024:1.011/0.101 2048:1.011/0.099
   deployed percentile (binned ratio, X1 ratio): 0.09521484375 0.27490234375
RUN [Copy flight outputs locally and run the analysis] cd /c/Projects/ReST-GI && mkdir -p results/audit97/E2 && scp -q LabServer63:~/rest_x97/E2/results/{checks.json,senders.json,senders_e2.npz,flight_deployed.npz,flight_shared2048.npz,flight_shared2048.json,flight_indep2048.npz,flight_indep2048.json,ref_boxA.npz,wl95a_deliv_boxA.npz} results/audit97/E2/ && ls -la results/audit97/E2 && python experiments/audit97/E2/e2_analyze.py results/audit97/E2
RUN [Check area-reference progress] ssh LabServer63 'cd ~/rest_x97/E2 && ls done_main 2>/dev/null; cat logs/aref_main.log | tail -5; cat logs/run_main.out | tail -3; ps -o etime= -p $(pgrep -u nd -f "e2_chain.py aref" | head -1) 2>/dev/null; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
  -> done_main
aref main seconds 74.0 total irradiance 30.71966161719677 rel SE of total 0.00016251547178572291 median per-patch rel SE 0.0004751884312642403
DONE aref
      00:00
890 MiB, 27 %
RUN [Re-run the analysis with the area-form reference] scp -q LabServer63:~/rest_x97/E2/results/aref_main.npz results/audit97/E2/ && python experiments/audit97/E2/e2_analyze.py results/audit97/E2 2>&1 | grep -v "^REFERENCE" | sed -n '/^AREA/,$p'
  -> AREA {
 "samples_per_patch": 268435456,
 "total": 15.359830808598385,
 "rel_se_total": 0.00016251547178572291,
 "own_half_split_noise": 0.00046103645114957767,
 "expected_abs_rel_l1_own_noise": 0.0005225794022155165,
 "CKF_ref_vs_area": {
  "W_T": {
   "ratio": 0.9851067872555973,
   "abs_rel_l1": 0 ...[cut]... 96/0.418 32:1.396/0.409 64:1.396/0.406 128:1.396/0.405 256:1.396/0.405 512:1.396/0.405 1024:1.396/0.405 2048:1.396/0.405
   convergence X1c 1:1.054/0.730 2:1.054/0.605 4:1.054/0.505 8:1.054/0.423 16:1.054/0.359 32:1.054/0.314 64:1.054/0.284 128:1.054/0.270 256:1.054/0.262 512:1.054/0.259 1024:1.054/0.254 2048:1.054/0.254
   convergence X1 1:0.996/0.712 2:0.996/0.585 4:0.996/0.470 8:0.996/0.371 16:0.996/0.287 32:0.996/0.215 64:0.996/0.161 128:0.996/0.118 256:0.996/0.088 512:0.996/0.061 1024:0.996/0.055 2048:0.996/0.050
   deployed percentile (binned ratio, X1 ratio): 0.09521484375 0.27490234375
RUN [Re-run the analysis with the area-form reference] cd /c/Projects/ReST-GI && scp -q LabServer63:~/rest_x97/E2/results/aref_main.npz results/audit97/E2/ && python experiments/audit97/E2/e2_analyze.py results/audit97/E2 2>&1 | grep -v "^REFERENCE" | sed -n '/^AREA/,$p'
WRITE c:\Projects\ReST-GI\experiments\audit97\E2\e2_extra.py
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
