# Track E1: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List relevant worklogs, experiment code and results] ls docs/worklogs | grep -E '^(9[2-6]|86|87)' ; ls experiments/ecro_attr95a experiments/ecro86 experiments/ecro_mat94 experiments/gm1_explicit87 experiments/ecro86a; ls results/ecro_attr95a 2>/dev/null; ls docs/audits 2>/dev/null; ls experiments/audit97 results/audit97 2>/dev/null
RUN [Inspect server run directories and GPU state] ssh -o ConnectTimeout=15 LabServer63 'ls ~/rest_x95a ~/rest_x95a/results ~/rest_x92a/results/models 2>&1 | head -80; cat ~/rest_x95a/env.sh; ls ~/rest_x97 2>&1; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'
  -> Exit code 2
86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md
86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md
86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
92A_WORKLOG_ECRO_PHYSICAL_COMPONENT_SUPERVISION.md
92B_WORKLOG_ECRO_OWN_SHARP_AND_SOURCE_ATTRIBU ...[cut]... py
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
  -> /home/nd/rest_x92a/results/models:
own_s8601.json
own_s8601.pt
own_s8602.json
own_s8602.pt
sum_s8601.json
sum_s8601.pt
sum_s8602.json
sum_s8602.pt

/home/nd/rest_x95a:
chain.out
chain.status
code
data
env.sh
results

/home/nd/rest_x95a/results:
an.json
deliv.npz
eval95a
fields
lists
logs
main.json
ops95a.npz
pre
renders
s0.json
senders.npz
tables
wl94d
wl95_calc
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
ls: cannot access '/home/nd/rest_x97': No such file or directory
memory.used [MiB], memory.total [MiB], utilization.gpu [%]
1 MiB, 12288 MiB, 0 %
RUN [List relevant worklogs, experiment code and results] cd /c/Projects/ReST-GI && ls docs/worklogs | grep -E '^(9[2-6]|86|87)' ; ls experiments/ecro_attr95a experiments/ecro86 experiments/ecro_mat94 experiments/gm1_explicit87 experiments/ecro86a; ls results/ecro_attr95a 2>/dev/null; ls docs/audits 2>/dev/null; ls experiments/audit97 results/audit97 2>/dev/null
RUN [Inspect server run directories and GPU state] ssh -o ConnectTimeout=15 LabServer63 'ls ~/rest_x95a ~/rest_x95a/results ~/rest_x92a/results/models 2>&1 | head -80; cat ~/rest_x95a/env.sh; ls ~/rest_x97 2>&1; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'
RUN [Measure file sizes and list WL95A headings] wc -l docs/worklogs/95A_*.md experiments/ecro_attr95a/a95.py experiments/ecro86/model86.py experiments/ecro_mat94/mat94.py experiments/ecro_mat94/an94.py experiments/ecro86/pt86.py experiments/ecro86/scene86.py experiments/gm1_explicit87/quad87.py experiments/gm1_explicit87/xport87.py experiments/ecro86/ecro86.py && grep -n "^#\|^##" docs/worklogs/95A_*.md | head -60
  -> 213 docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
   628 experiments/ecro_attr95a/a95.py
   137 experiments/ecro86/model86.py
   578 experiments/ecro_mat94/mat94.py
   379 experiments/ecro_mat94/an94.py
   498 experiments/ecro86/pt86.py
   237 experiments/ecro86/s ...[cut]... ability & Flight Attribution Smoke (2026-11-15)
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
RUN [Measure file sizes and list WL95A headings] cd /c/Projects/ReST-GI && wc -l docs/worklogs/95A_*.md experiments/ecro_attr95a/a95.py experiments/ecro86/model86.py experiments/ecro_mat94/mat94.py experiments/ecro_mat94/an94.py experiments/ecro86/pt86.py experiments/ecro86/scene86.py experiments/gm1_explicit87/quad87.py experiments/gm1_explicit87/xport87.py experiments/ecro86/ecro86.py && grep -n "^#\|^##" docs/worklogs/95A_*.md | head -60
RUN [Read the WL95A worklog] cat docs/worklogs/95A_*.md
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
RUN [Read the WL95A worklog] cd /c/Projects/ReST-GI && cat docs/worklogs/95A_*.md
RUN [Read first half of a95.py] sed -n 1,330p experiments/ecro_attr95a/a95.py
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
RUN [Read first half of a95.py] cd /c/Projects/ReST-GI && sed -n 1,330p experiments/ecro_attr95a/a95.py
RUN [Read model86.py and ecro86.py] cat experiments/ecro86/model86.py && sed -n 1,206p experiments/ecro86/ecro86.py
RUN [Read scene86.py] cat experiments/ecro86/scene86.py
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
  -> """WL86 ECRO v0 operator (torch): geometry-conditioned, source-independent, linear and non-negative.

Geometry state (no radiance): patch tables, frozen ER states of the patch materials, stratified hemisphere state rays
(hit patch, outgoing bin at the hit patch) -> generator network -> response fact ...[cut]... 'camera2'] if opt.get('--cam') == '2' else None)
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
RUN [Read model86.py and ecro86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/model86.py && sed -n 1,206p experiments/ecro86/ecro86.py
RUN [Read scene86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/scene86.py
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
RUN [Read pt86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/pt86.py
RUN [List mat94.py definitions] grep -n "^def \|^class \|^[A-Z_]* = " experiments/ecro_mat94/mat94.py | head -80
  -> 19:HERE = os.path.dirname(os.path.abspath(__file__))
23:DEV = '--dev' in sys.argv
24:P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_mat94.json')))
25:SCENE_FN = os.path.join(HERE, '..', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')
26:SC = json.load(open(SCENE_FN))
27:TAG = SC['tag ...[cut]... ant_pmat(G, v):
99:def train_pmat_B(G, idx):
109:def val_pmat_ck(G, idx):
121:def pm_digest(pm):
126:def _patch_src(fn, obj, label, probe_k1=False):
142:def install():
183:def set_pmat(PT, ctx, pm):
188:def ctx_ck2(R):
202:def extend_buffer(model):
209:def load_frozen(R92A, seed):
215:def load_cand(R, tag):
226:def load_spec(R, R92A, spec):
231:def mq_ext():
238:def arm_tables(model, gs, s, Mn, s1, arms):
260:def ck_tables(R, R92A, spec, otag, arms=None):
289:def fref(R, v):
305:def iref(R, v, r):
318:def render(R, listfn):
342:def masks(PT, S, ctx, cam):
414:def pre(R, R92A, R92B, R93D, RCK):
RUN [List mat94.py definitions] cd /c/Projects/ReST-GI && grep -n "^def \|^class \|^[A-Z_]* = " experiments/ecro_mat94/mat94.py | head -80
RUN [Read mat94.py infrastructure and CK2 scene definition] sed -n 1,340p experiments/ecro_mat94/mat94.py; cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -60
  -> """WL94 (protocol ecro_mat94): per-patch material machinery, Stage A tests, Stage B frozen material counterfactuals on CK2.

Per-patch materials are consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors); new Lambert rows are a ...[cut]... 
    print('RENDER_DONE', listfn, flush=True)


{
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
    0.45
   ],
   "yaw": 0.15
  },
  "B": {
   "c": [
    1.4,
    1.5,
    0.5
   ],
   "h": [
    0.3,
    0.25,
    0.5
   ],
   "yaw": 0.0
  }
 },
 "lights": [
  {
   "pos": [
    -1.45,
    -0.6,
    1.7
   ],
   "rgb": [
    30.0,
    30.0,
    30.0
   ]
RUN [Read mat94.py infrastructure and CK2 scene definition] cd /c/Projects/ReST-GI && sed -n 1,340p experiments/ecro_mat94/mat94.py; cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -60
RUN [Read second half of a95.py] sed -n 330,628p experiments/ecro_attr95a/a95.py
  -> Dt = receive(flight(hit, dbin, rw, snd[k_], N, S), Mrcv)
            res[k_] = Dt
        for k_, cen in (('X1c', True), ('X1', False)):
            L, nr = x1_rays(PT, ctx, lt, hit, seg, cen)
            res[k_] = receive(flight_ray(L, rw, N, S), Mrcv)
            store['rays_S1'] = nr
        retu ...[cut]...  + e3 (identical across arms); floor = reference table; '
                 f'G_s = reference e2 + own learned global (diagnostic); shared error scale {E_MAX:.3g}, one exposure', fontsize=9)
    fn = os.path.join(outd, 'figA4_renders.png'); fig.savefig(fn, dpi=70, bbox_inches='tight'); plt.close(fig); files.append(fn)
    return files


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'stage0':
        stage0(a[1], a[2])
    elif cmd == 'main':
        main(a[1], a[2])
    elif cmd == 'an':
        an(a[1], a[2])
    os._exit(0)
RUN [Read second half of a95.py] cd /c/Projects/ReST-GI && sed -n 330,628p experiments/ecro_attr95a/a95.py
RUN [Find M64 definition, fx92.inject, load_own, ops93d path setup] grep -rn "def local_matrices" -A40 experiments/*/diag92.py | head -80; grep -n "def inject" -A25 experiments/*/fx92.py; grep -n "def load_own" -A15 experiments/*/train93.py; sed -n 1,60p experiments/ecro_local93d/ops93d.py
  -> 133:def local_matrices():
134-    import fx92
135-    import quad87 as Q
136-    import xport87 as X
137-    E87 = fx92.E87
138-    quad = Q.Quad(X.realization_us(86001), E87['M_ray'], E87['seed_ray'], E87['M_dense'], E87['seed_dense'])
139-    return quad, quad.enc_matrix()                       #  ...[cut]...  [c for c in C if c['kind'] == 'train_rep'][:2] + [c for c in C if c['kind'] == 'val'][:4] + [c for c in C if c['kind'] == 'sharp']
    part = {}
    for kind in ('train_rep', 'val'):
        ids = [c['cfg'] for c in C if c['kind'] == kind]
        for k, i in enumerate(ids):
            part[str(i)] = ids[(k + 1) % len(ids)]
    tags = {c['tag']: c['cfg'] for c in C if c['kind'] == 'sharp'}
    for t, i in tags.items():
        o = t[:-1] + ('1' if t.endswith('0') else '0')
        if o in tags:
            part[str(i)] = tags[o]
    out = {'configs': C, 'partner': part, 'e6': e6, 'dev': DEV}
RUN [Find M64 definition, fx92.inject, load_own, ops93d path setup] cd /c/Projects/ReST-GI && grep -rn "def local_matrices" -A40 experiments/*/diag92.py | head -80; grep -n "def inject" -A25 experiments/*/fx92.py; grep -n "def load_own" -A15 experiments/*/train93.py; sed -n 1,60p experiments/ecro_local93d/ops93d.py
RUN [Read quad87.py (M64 and q table)] cat experiments/gm1_explicit87/quad87.py
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
RUN [Read quad87.py (M64 and q table)] cd /c/Projects/ReST-GI && cat experiments/gm1_explicit87/quad87.py
RUN [Read the WL95A run script and fx92 header] cat experiments/ecro_attr95a/run95a.sh; cat experiments/ecro_attr92/fx92.py | sed -n 1,25p; ls experiments | tr '\n' ' '
  -> #!/bin/bash
# WL95A chain (one detached chain): links (read-only) -> Stage 0 -> main (senders, draws, deliveries, tables) -> renders -> analysis
# [-> rule-based extension to 64 draws -> re-render pooled tables -> analysis].
#   setsid nohup bash run95a.sh <R> <W94> <W94D> <WL95CALC> <R92A> [--dev]  ...[cut]... nk66 fnto_rank75 fnto_repr68 fnto_reuse58 fnto_rule57 fnto_sel56 fnto_smoke47 fnto_spatial49 fnto_src73 fnto_srcid78 fnto_sym79 fnto_visual_review fnto_walker71c full_recompute_benchmark gm1_c3urx1_91 gm1_char85 gm1_explicit87 gm1_gap90 gm1_relation88 gm1_relation89 neural_recompute_cost phys_ref53 radiometric_relation_state_prototype relational_residual_prototype rna_baseline_quality rna_reproduction rna_selective_reconstruction_lego rna_surface_ownership rna_surface_ownership_b2 rna_surface_ownership_lego selective_refit_oracle server_setup spt_lego sunreuse70 surface_owned_radiance xbench69
RUN [Read the WL95A run script and fx92 header] cd /c/Projects/ReST-GI && cat experiments/ecro_attr95a/run95a.sh; cat experiments/ecro_attr92/fx92.py | sed -n 1,25p; ls experiments | tr '\n' ' '
RUN [Check audit dirs and stored WL95A sender/operator numbers] ls experiments/audit97 results/audit97 docs/audits/wl97_evidence_reliability 2>&1; python -c "
import json
m=json.load(open('results/ecro_attr95a/main.json'))
print(m.keys())
print(json.dumps(m['sender'],indent=0)[:3500])
print(json.dumps(m['operator'],indent=0))
print(m['counts']['deployed_rays_to_S1'], m['D_used'])
"; ls -la results/ecro_attr95a
  -> docs/audits/wl97_evidence_reliability:

experiments/audit97:
B
C2
MAP

results/audit97:
B
C2
MAP
dict_keys(['timing', 'counts', 'sender', 'operator', 'D_pilot', 'D_used', 'T_DEPLOYED_ID', 'gpu_mem_peak_mib_torch', 'tables'])
{
"A1": {
"vs_M64_s16": {
"rel_l1": 0.9572351144740621,
"energy": 0.45 ...[cut]... 969,
"passive_all": false,
"nonneg": true,
"mass_mean": 0.8349744492310064
}
}
9995 64
total 4565
drwxr-xr-x 1 dna10 197609       0 Nov 15 06:08 .
drwxr-xr-x 1 dna10 197609       0 Nov 16 02:45 ..
-rw-r--r-- 1 dna10 197609   26886 Nov 15 06:08 an.json
-rw-r--r-- 1 dna10 197609    1557 Nov 15 06:08 chain.status
-rw-r--r-- 1 dna10 197609 4213168 Nov 15 06:08 deliv.npz
-rw-r--r-- 1 dna10 197609     412 Nov 15 06:08 gpu.txt
-rw-r--r-- 1 dna10 197609    6001 Nov 15 06:08 main.json
-rw-r--r-- 1 dna10 197609  373656 Nov 15 06:08 ops95a.npz
-rw-r--r-- 1 dna10 197609    2324 Nov 15 06:08 s0.json
RUN [Check audit dirs and stored WL95A sender/operator numbers] cd /c/Projects/ReST-GI && ls experiments/audit97 results/audit97 docs/audits/wl97_evidence_reliability 2>&1; python -c "
import json
m=json.load(open('results/ecro_attr95a/main.json'))
print(m.keys())
print(json.dumps(m['sender'],indent=0)[:3500])
print(json.dumps(m['operator'],indent=0))
print(m['counts']['deployed_rays_to_S1'], m['D_used'])
"; ls -la results/ecro_attr95a
RUN [Inspect server code checkout and stored artifacts] ssh -o ConnectTimeout=15 LabServer63 'cd ~/rest_x95a; cat code/COMMIT 2>/dev/null; ls code code/experiments | head -50; ls -la results/ | head -40; ls -la results/fields | head; cat chain.status | tail -20; ls ~/rest_x96a/code; cat ~/rest_x96a/code/COMMIT 2>/dev/null; sha256sum results/senders.npz results/main.json results/ops95a.npz ~/rest_x92a/results/models/own_s860?.pt | cut -c1-16,65-'
  -> c152120
code:
AGENTS.md
COMMIT
docs
experiments
results

code/experiments:
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
ecro_s ...[cut]... 00 rc=0
2026-11-14T21:07:59Z END R_e_r95a_e_01 rc=0
2026-11-14T21:07:59Z END R_e_r95a_e_02 rc=0
2026-11-14T21:07:59Z END R_e_r95a_e_03 rc=0
2026-11-14T21:07:59Z END R_e_r95a_e_04 rc=0
2026-11-14T21:07:59Z END R_e_r95a_e_05 rc=0
2026-11-14T21:07:59Z START N2
2026-11-14T21:08:08Z END N2 rc=0
2026-11-14T21:08:08Z CHAIN_DONE
AGENTS.md
COMMIT
experiments
a1718f7
5d365d19d6dffadc  results/senders.npz
28d104991bcad6f7  results/main.json
d4d02e5a2f423ce9  results/ops95a.npz
739e7eb50b25d329  /home/nd/rest_x92a/results/models/own_s8601.pt
cee9a09a289fca8c  /home/nd/rest_x92a/results/models/own_s8602.pt
RUN [Locate training data format and OWN training code] sed -n 1,60p experiments/ecro86/gen86.py; grep -n "data\b\|np.load\|\.npz" experiments/ecro86/train86.py | head -30; grep -n "np.load\|npz\|def " experiments/ecro_own92a/*.py | head -60
  -> """WL86 response supervision: random geometries of the family x random point lights -> geometry state, source encoding, target field.

  python gen86.py <root> <split> <start> <count>
Per geometry (file <root>/data/<split>/g<idx>.npz): geom / lights (json), patch tables, state rays (hit, dbin), and  ...[cut]... npz'))
experiments/ecro_own92a/tests92a.py:15:def pre(R, R86, R92):
experiments/ecro_own92a/tests92a.py:33:                ref = np.load(os.path.join(R86, 'results', 'renders', f'EC_{sd}_ecro_{tag}_r0.npz'))['B']
experiments/ecro_own92a/tests92a.py:62:def f2(R):
experiments/ecro_own92a/tests92a.py:69:    have = {sp: len(glob.glob(os.path.join(R, 'data', sp + '_f2', 'g*.npz'))) for sp in ('train', 'val', 'trainhq')}
experiments/ecro_own92a/viz92a.py:17:def col(n):
experiments/ecro_own92a/viz92a.py:21:def sq_cbar(fig, im, ax, vmax, signed, label):
experiments/ecro_own92a/viz92a.py:27:def main():
RUN [Locate training data format and OWN training code] cd /c/Projects/ReST-GI && sed -n 1,60p experiments/ecro86/gen86.py; grep -n "data\b\|np.load\|\.npz" experiments/ecro86/train86.py | head -30; grep -n "np.load\|npz\|def " experiments/ecro_own92a/*.py | head -60
RUN [Read OWN training objective and data generation] sed -n 72,235p experiments/ecro_own92a/own92a.py
  -> def kept_files(R, R86, sp, limit=None):
    ex = {n for n, _, _ in jl(os.path.join(R, 'results', 'exclusion.json'))['excluded'].get(sp, [])}
    fns = [f for f in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))) if os.path.basename(f) not in ex]
    return fns[:limit]


# ----------------- ...[cut]... son.dumps(rec), flush=True)
    res = {'seed': seed, 'arm': arm, 'steps': steps, 'params': sum(p.numel() for p in model.parameters()), 'init_sha256': init_hash,
           'train_s': time.time() - t0, 'items': len(items), 'val_items': len(val), 'nonfinite_steps': nonfinite,
           'gpu_mem_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'log': log}
    torch.save(model.state_dict(), os.path.join(out, f'{tag}.pt'))
    jd(res, os.path.join(out, f'{tag}.json'))
    print('FINAL', tag, json.dumps({k: res[k] for k in ('train_s', 'nonfinite_steps', 'init_sha256')}), log[-1], flush=True)
RUN [Read OWN training objective and data generation] cd /c/Projects/ReST-GI && sed -n 72,235p experiments/ecro_own92a/own92a.py
RUN [Locate training data on server] ssh -o ConnectTimeout=15 LabServer63 'ls ~ | grep rest_ ; ls ~/rest_ecro86/data; ls ~/rest_ecro86/data/train | wc -l; ls ~/rest_ecro86/data/train | head -3; du -sh ~/rest_ecro86/data/train; cat ~/rest_x92a/results/exclusion.json | head -c 600; ls ~/rest_x92a/results ~/rest_x92a/data; df -h ~ | tail -1; cat ~/rest_x92a/results/models/own_s8601.json | head -c 500'
  -> rest_8dna_baseline
rest_arch77a
rest_ato74
rest_ato74_design
rest_ato74_dev
rest_attrib71b
rest_b4rep
rest_closure80d
rest_cmp72
rest_cmp72_dev
rest_compiled80c
rest_compiled81
rest_compiled81_dev
rest_compiled82
rest_compiled82_dev
rest_compiled82_probe
rest_compiled83
rest_compiled83_dev
rest_comp ...[cut]... _f2.json
tests_pre.json
timing92a.json
viz
/dev/mapper/ubuntu--vg-home   11T  2.0T  7.9T  21% /home
{
 "seed": 8601,
 "arm": "own",
 "steps": 24000,
 "params": 735664,
 "init_sha256": "939ab4702706dc073711f6faa7cf1b7a3c8ad9071c0b4e04771a8499559979f7",
 "train_s": 2287.977981567383,
 "items": 928,
 "val_items": 55,
 "nonfinite_steps": 0,
 "gpu_mem_peak_mib": 629.33642578125,
 "log": [
  {
   "step": 500,
   "objective": 0.37052319806814193,
   "L_sum": 0.18512723835557698,
   "L_single": 0.08671487126871943,
   "L_global": 0.09868108770251274,
   "grad_norm_mean": 0.36131913597509263,
   "grad_
RUN [Inspect closure interface, field split, material table and protocol] grep -n "def \|class " experiments/fnto_compiled83/closure83_dr.py | head -40; sed -n 150,200p experiments/ecro86a/pt86a.py; python - <<'EOF'
import numpy as np
z=np.load('experiments/ecro86/data/mats86.npz')
print(z.files); print(z['desc']); print(z['lambert'], z['albedo']); print(z['states'].shape)
import json
P=json.load(open('experiments/ecro86/protocol/ecro86.json'))
print(P['representation']); print(P['fixture']['panel']); print(P['training']['light_distribution']); print(P['architecture']
  -> 29:def _mi():
34:def compiled_state(arm, p):
45:def expand_state(st):
52:class DrClosure83:
55:    def __init__(self, arm, state, idx=None):
79:    def _s(self, mu):
91:    def _refr(self, w):
99:    def _lobe(self, k, wi, wo, want_eval, want_pdf):
121:    def eval_pdf(self, wi, wo, active=True, wan ...[cut]... #97)'}
{'x': [-1.6, 1.6], 'y': [-1.6, 1.6], 'z': [1.0, 2.8]}
{'token_dim': 128, 'channels_R': 16, 'global_rank': 24, 'message_rounds': 3, 'description': 'patch features (centroid, frame, area, frozen ER state through an MLP, openness) -> tokens; 3 rounds of message passing over the state-ray form-factor matrix + global mean; heads: c (incident bin -> 16 channels), a (16 channels -> outgoing bin), global encoder Q / decoder P (rank 24), all softplus (>= 0), per colour channel. Application: single learned flight through the state rays (current visibility) + low-rank global term (model86.py).'}
RUN [Inspect closure interface, field split, material table and protocol] cd /c/Projects/ReST-GI && grep -n "def \|class " experiments/fnto_compiled83/closure83_dr.py | head -40; sed -n 150,200p experiments/ecro86a/pt86a.py; python - <<'EOF'
import numpy as np
z=np.load('experiments/ecro86/data/mats86.npz')
print(z.files); print(z['desc']); print(z['lambert'], z['albedo']); print(z['states'].shape)
import json
P=json.load(open('experiments/ecro86/protocol/ecro86.json'))
print(P['representation']); print(P['fixture']['panel']); print(P['training']['light_distribution']
RUN [Read ER closure implementation] sed -n 1,175p experiments/fnto_compiled83/closure83_dr.py
  -> """WL83 Dr.Jit runtime for the WL82 closures C0 / E / ER (semantics = experiments/fnto_compiled82/closure82.py, torch reference).

Compiled state per material (69 floats, identical layout for every arm; unused fields are 1):
  lam 24 (c * 8 + r) | au 7 | av 7 | f0 7 | sampling probs 8 | s 15 (c * 5  ...[cut]...   return self.eval_pdf(wi, wo, active, want_eval=False)[1]

    # ---------------------------------------------------------------- sampling
    def sample(self, wi, u1, u2, active=True):
        """Returns wo, pdf, weight (= eval / pdf), valid."""
        mi = _mi()
        import drjit as dr
        active = mi.Bool(active) & (wi.z > EPS)
        sel = mi.UInt32(0)
        acc = mi.Float(self.pr[0])
        for k in range(1, 8):
            sel = dr.select(u1 >= acc, mi.UInt32(k), sel)
            acc = acc + self.pr[k]
        au = mi.Float(1.0); av = mi.Float(1.0)
        for k in range(7):
RUN [Read ER closure implementation] cd /c/Projects/ReST-GI && sed -n 1,175p experiments/fnto_compiled83/closure83_dr.py
RUN [Read train86 Item, S1 material state, server status] sed -n 1,60p experiments/ecro86/train86.py; python - <<'EOF'
import numpy as np
z=np.load('experiments/ecro86/data/mats86.npz')
st=z['states'][1]
print('lam',st[:24].reshape(3,8)); print('au',st[24:31]); print('av',st[31:38]); print('f0',st[38:45]); print('pr',st[45:53]); print('s',st[53:68].reshape(3,5)); print('eta',st[68])
EOF
ssh LabServer63 'ls ~/rest_x97/; ls ~/rest_ecro86/data/val | wc -l; ls ~/rest_x95a/code/experiments/ecro_mat94/ ~/rest_x95a/code/experiments/ecro86/ | head -40; nvidia-
  -> """WL86 ECRO v0 training (response supervision; no test configuration, no image, no source identity).

  python train86.py <root> <seed> [--split=train] [--steps=N] [--tag=name]
Writes <root>/results/models/<tag>.pt (state dict) and <tag>.json (log, train-subset / validation field errors).
"""
impor ...[cut]...  0.11237707 0.00103582]
s [[0.32625163 0.8591727  0.99585885 0.99666    0.99668604]
 [0.82599354 0.99549    0.99941516 0.9996387  0.99999666]
 [0.72180027 0.97965634 0.99610895 0.99911255 0.9999943 ]]
eta 2.4955008
B
C1
C2
E4
MAP
64
/home/nd/rest_x95a/code/experiments/ecro86/:
__pycache__
analyze86.py
data
diag_v3.py
econ86.py
ecro86.py
gen86.py
masks86.py
mats86.py
model86.py
protocol
pt86.py
run86.sh
scene86.py
tests86.py
train86.py
viz86.py
watch86.sh

/home/nd/rest_x95a/code/experiments/ecro_mat94/:
__pycache__
an94.py
c94.py
mat94.py
posthoc94.py
protocol
run94.sh
viz94.py
1 MiB, 0 %
RUN [Read train86 Item, S1 material state, server status] cd /c/Projects/ReST-GI && sed -n 1,60p experiments/ecro86/train86.py; python - <<'EOF'
import numpy as np
z=np.load('experiments/ecro86/data/mats86.npz')
st=z['states'][1]
print('lam',st[:24].reshape(3,8)); print('au',st[24:31]); print('av',st[31:38]); print('f0',st[38:45]); print('pr',st[45:53]); print('s',st[53:68].reshape(3,5)); print('eta',st[68])
EOF
ssh LabServer63 'ls ~/rest_x97/; ls ~/rest_ecro86/data/val | wc -l; ls ~/rest_x95a/code/experiments/ecro_mat94/ ~/rest_x95a/code/experiments/e
RUN [Create E1 audit dirs; check autocrlf and flight helper] mkdir -p /c/Projects/ReST-GI/experiments/audit97/E1 /c/Projects/ReST-GI/results/audit97/E1 && cd /c/Projects/ReST-GI && git config core.autocrlf; ls experiments/audit97/B | head; sed -n 60,130p experiments/ecro_stage93b/eval93b.py 2>/dev/null | head -5; grep -n "def flight" -A12 experiments/ecro_stage93b/eval93b.py
  -> true
heldout_disjointness.py
panel_coverage.py
regen_population.py
server_digest_population.py
    import train93 as TR
    out = {}; ok = True
    os.makedirs(os.path.join(R, 'results', 'models'), exist_ok=True)
    for a in ('T0', 'TQ'):
        for s in SEEDS:
149:def flight(rw, gs, L):
150-    import fx92
151-    S = fx92.S
152-    NB, K, KAPPA = S.NB, S.K, S.KAPPA
153-    Ls, N = L.shape[0], gs.N
154-    ok = (gs.hit >= 0).view(1, N * K, 1)
155-    idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
156-    return ((L.reshape(Ls, N * NB, 3)[:, idx] * ok).view(Ls, N, NB, KAPPA, 3) * rw.view(1, 1, NB, KAPPA, 1)).sum(3)
157-
158-
159-def l1s(X, T):
160-    return [float(np.abs(X - T).sum()), float(T.sum())]
161-
WRITE c:\Projects\ReST-GI\experiments\audit97\E1\e1_audit.py
EDIT c:\Projects\ReST-GI\experiments\audit97\E1\e1_audit.py
RUN [Deploy E1 audit script to server and start the conventions stage] cd /c/Projects/ReST-GI/experiments/audit97/E1 && cat > run_e1.sh <<'EOF'
#!/bin/bash
# WL97 track E1 runner (LabServer63). Usage: bash run_e1.sh <stage>   (stage: conv | sender | train)
# Reads frozen code and artifacts read-only; writes only under ~/rest_x97/E1.
set -u
. ~/rest_x95a/env.sh
export PYTHONDONTWRITEBYTECODE=1
export E1_CODE=$HOME/rest_x95a/code/experiments      # frozen checkout c152120 (WL95A freeze), read-only
export E1_ROOT=$HOME/rest_x97/E1
export E1_ORIG=$HOME/rest_x95a/result
  -> e1_audit.py: Python script, ASCII text executable
run_e1.sh:   Bourne-Again shell script, ASCII text executable
syntax ok
e1_audit.py: Python script, ASCII text executable
run_e1.sh:   Bourne-Again shell script, ASCII text executable
1359 MiB, 4 %
started
EDIT c:\Projects\ReST-GI\experiments\audit97\E1\e1_audit.py
RUN [Deploy E1 audit script to server and start the conventions stage] cd /c/Projects/ReST-GI/experiments/audit97/E1 && cat > run_e1.sh <<'EOF'
#!/bin/bash
# WL97 track E1 runner (LabServer63). Usage: bash run_e1.sh <stage>   (stage: conv | sender | train)
# Reads frozen code and artifacts read-only; writes only under ~/rest_x97/E1.
set -u
. ~/rest_x95a/env.sh
export PYTHONDONTWRITEBYTECODE=1
export E1_CODE=$HOME/rest_x95a/code/experiments      # frozen checkout c152120 (WL95A freeze), read-only
export E1_ROOT=$HOME/rest_x97/E1
export E1_ORIG=$HOME/rest_x95a/result
RUN [Wait for conventions stage and show output] ssh LabServer63 'cd ~/rest_x97/E1; for i in $(seq 1 60); do [ -f done_conv ] && break; sleep 3; done; cat done_conv; tail -c 6000 log_conv.txt'
  -> 0
{
 "bins": {
  "agreement_own_vs_production": 1.0,
  "sum_WT": 3.141592653589793,
  "pi": 3.141592653589793,
  "WT_ring_values": [
   0.01227184630308513,
   0.03681553890925539,
   0.06135923151542565,
   0.0859029241215959
  ],
  "W94_first16": [
   0.006135923151542565,
   0.006135923151542565, ...[cut]... 16,
    0.8484556138747041,
    0.7788401922028456
   ]
  }
 ],
 "frame": {
  "panel_hits": 284,
  "max_abs_diff_to_local_vs_patch_frame": 2.980123703366644e-08
 },
 "workload": {
  "light_pos": [
   -1.45,
   -0.6,
   1.7
  ],
  "light_rgb": [
   30.0,
   30.0,
   30.0
  ],
  "panel_plane_x": -1.5,
  "light_height_above_panel_plane_m": 0.050000000000000044,
  "cos_incidence_at_S1_patch_centres_min_med_max": [
   0.02616223799263027,
   0.040658447519801966,
   0.12038585308576932
  ],
  "S1_ids": [
   1088,
   1129
  ],
  "BOXA_ids": [
   1130,
   1183
  ],
  "rho_boxA": 0.5
 }
}
E1_DONE conv
RUN [Run the sender stage on the server and wait] ssh LabServer63 'cd ~/rest_x97/E1; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; (setsid nohup bash run_e1.sh sender > /dev/null 2>&1 &); for i in $(seq 1 110); do [ -f done_sender ] && break; sleep 5; done; cat done_sender; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; tail -c 1500 log_sender.txt'
  -> 602 MiB, 0 %
0
2255 MiB, 2 %

     1.0871563993417208,
     0.41931735617620236,
     0.031119492430604603,
     2.226782514565456
    ],
    [
     0.024110339528768624,
     1.0805075832149527,
     0.41976102014586597,
     0.0318896542523124,
     2.196129381346986
    ],
    [
     0.0261900392 ...[cut]... 51,
     0.8579080108927154,
     0.10818510869243254,
     3.08223465597944
    ],
    [
     0.036903061320526426,
     1.198153268817529,
     0.8656004872886292,
     0.10774883519756454,
     3.1285191405055643
    ],
    [
     0.027124431138361514,
     1.1820714466503288,
     0.8603615496238938,
     0.10882486097235859,
     3.086071225700753
    ],
    [
     0.01766964414902377,
     1.1876045271118059,
     0.8625296921718906,
     0.10845550143640272,
     3.085449047791374
    ]
   ]
  }
 },
 "seconds": 32.2455039024353,
 "gpu_mem_peak_mib_torch": 263.6337890625
}
E1_DONE sender
RUN [Fetch sender results and print first sections] mkdir -p /c/Projects/ReST-GI/results/audit97/E1 && cd /c/Projects/ReST-GI/results/audit97/E1 && scp -q LabServer63:~/rest_x97/E1/results/conv.json LabServer63:~/rest_x97/E1/results/sender.json LabServer63:~/rest_x97/E1/results/sender_e1.npz . && python - <<'EOF'
import json
o=json.load(open('sender.json'))
def show(k, d=None):
    print('==',k); print(json.dumps(o[k] if d is None else d, indent=0)[:6000])
for k in ('tables','source','state','inputs','learned','q','m64'):
    show(k)
EOF
  -> == tables
{
"u_state_equal": true,
"u_src_equal": true
}
== source
{
"max_abs_diff_rect_patches": 2.1456228967764446e-05,
"max_prod": 43.16370391845703,
"rel_l1_S1": 1.157507030907849e-06,
"S1_irradiance_by_ring": [
1.0,
0.0,
0.0,
0.0
],
"S1_nonzero_incident_bins": [
12,
13,
14,
 ...[cut]... 
"exitance_exact_cos_over_WT_exitance_own_q": 0.7693776268445895,
"own_q_energy_by_ring": [
0.6353334622941703,
0.24041776345456184,
0.09420951386211494,
0.03003926038915299
]
}
== m64
{
"own_split_half_flux_l1_half": 0.004668217258935072,
"own_vs_stored_A4_flux_l1": 0.01170904380551408,
"own_vs_stored_A4_flux_l1_by_incident_ring": [
0.045218365690752854,
0.008599978983887927,
0.005484651403627302,
0.012482931901409283
],
"ownM64_s16_vs_stored_A4": {
"rel_l1": 0.0442217416729695,
"energy": 0.9705862649307203,
"peak": 0.933765821593371,
"max_ratio": 0.933765821593371
}
}
RUN [Print passivity results] cd /c/Projects/ReST-GI/results/audit97/E1 && python - <<'EOF'
import json
o=json.load(open('sender.json'))
p=o['passivity']
for k in ('stored_A4_M64','own_M64','A1','A2','A3','ck2_energy_out_over_in'):
    print('==',k, json.dumps(p[k]))
for sd in ('A0_s8601','A0_s8602'):
    d=dict(p[sd]); pb=d.pop('per_incident_bin_max'); pf=d.pop('per_incident_bin_fraction_gt1')
    print('==',sd); print(json.dumps(d,indent=0))
    import numpy as np
    print('per-bin max (4x16):'); print(np.round(np.array(p
  -> == stored_A4_M64 {"albedo_max": 0.8246719744363133, "albedo_by_ring_mean": [0.6521693066281774, 0.5808187860067946, 0.6435593434377757, 0.6631612139136097], "albedo_by_ring_max": [0.6858009309981697, 0.7033228682069392, 0.7988324118410901, 0.8246719744363133]}
== own_M64 {"albedo_max_WT": 0.8262928 ...[cut]... 0.6  2.76 0.26 0.03 0.67 0.89 1.09 1.46
  1.33 0.67]
 [0.44 1.14 2.15 1.64 0.81 1.11 0.1  0.58 1.47 1.14 0.75 1.3  0.61 1.51
  1.32 0.48]
 [0.74 0.31 0.67 0.81 0.95 0.26 0.28 1.1  1.13 0.82 0.77 0.88 1.04 1.31
  1.18 1.27]]
per-bin frac>1 (4x16):
[[0.   0.   0.   0.   0.   0.   0.   0.   0.85 0.9  0.67 0.   0.71 0.6
  0.29 0.  ]
 [0.   0.1  0.   0.72 0.16 0.   0.   0.8  0.   0.   0.   0.   0.14 0.59
  0.13 0.  ]
 [0.   0.07 0.48 0.21 0.   0.07 0.   0.   0.41 0.03 0.   0.39 0.   0.48
  0.18 0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.05 0.07 0.   0.   0.   0.02 0.24
  0.1  0.37]]
RUN [Print sender metrics, delivery and redraw results] cd /c/Projects/ReST-GI/results/audit97/E1 && python - <<'EOF'
import json
o=json.load(open('sender.json'))
m=o['sender_metrics']
def row(k,r,w='WT'):
    d=m[k][r][w]; return f"{d['rel_l1']:.3f}/{d['energy']:.3f}/{d['peak']:.3f}"
print('sender | vs own_q | vs stored_q256 | vs stored A4 | vs ownM64_s16   (WT)')
for k in m:
    print(k.ljust(20), ' | '.join(row(k,r) if r in m[k] else '   -   ' for r in ('own_q','stored_q256','stored_A4_M64s16','ownM64_s16')))
print('--- W94 as coded')
for k in ('A
  -> sender | vs own_q | vs stored_q256 | vs stored A4 | vs ownM64_s16   (WT)
own_q                   -    | 0.026/0.999/0.998 | 0.249/1.030/1.198 | 0.284/1.061/1.283
stored_q256          0.026/1.001/1.002 |    -    | 0.246/1.031/1.202 | 0.282/1.063/1.287
stored_A4_M64s16     0.241/0.971/0.832 | 0.239 ...[cut]... 88530529077,
0.6528127205523956
],
"min": [
0.23342320174566628,
0.8254945660038111,
0.707127599205209,
2.3737788033674594,
1.629879731637998,
0.6527120002381436
],
"max": [
0.3479423352737167,
1.1399762052166489,
1.017667051707029,
3.294643734976936,
2.420590485768881,
0.6529051999629424
],
"sd": [
0.02917069374909183,
0.06700870239011428,
0.07411394708121506,
0.21907704210013362,
0.1844565888187429,
4.176240995647084e-05
]
}
}
{'timing_q_s': 10.623933553695679, 'timing_m64_s': 6.241497039794922, 'timing_boxa_ref_s': 2.422598361968994, 'seconds': 32.2455039024353}
RUN [Add a debug stage cross-checking the two box-A estimators and run it] cd /c/Projects/ReST-GI/experiments/audit97/E1 && python - <<'EOF'
p='e1_audit.py'; s=open(p,newline='').read()
# parametrise the two box-A estimators (closure state, constant-radiance test mode)
s=s.replace("def boxa_ref(seed, launches, SG=512):","def boxa_ref(seed, launches, SG=512, state=None, const=False, hist=None):")
s=s.replace("""    pn = np.cross(a1, a2)
    area = 4 * h1 * h2
    cl = closure_s1()
    V = lambda x: mi.Vector3f(float(x[0]), float(x[1]), float(x[2]))
    acc = np.zeros(nb
  -> ok
10
e1_audit.py: Python script, ASCII text executable
0
 0.039, 0.039, 0.038, 0.036, 0.032, 0.027, 0.023, 0.022, 0.02], [0.022, 0.024, 0.025, 0.029, 0.034, 0.038, 0.04, 0.041, 0.042, 0.041, 0.038, 0.034, 0.029, 0.024, 0.024, 0.022], [0.024, 0.025, 0.025, 0.029, 0.034, 0.038, 0.041, 0.042, 0.042,  ...[cut]... 18, 0.02, 0.021, 0.022, 0.022, 0.023, 0.011, 0.003], [0.0, 0.0, 0.0, 0.001, 0.002, 0.006, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.016, 0.015, 0.007, 0.002], [0.0, 0.0, 0.0, 0.001, 0.002, 0.006, 0.008, 0.009, 0.01, 0.01, 0.011, 0.011, 0.011, 0.009, 0.004, 0.001], [0.0, 0.0, 0.0, 0.001, 0.002, 0.005, 0.006, 0.006, 0.006, 0.007, 0.007, 0.007, 0.007, 0.005, 0.002, 0.001], [0.0, 0.0, 0.0, 0.001, 0.002, 0.003, 0.003, 0.003, 0.004, 0.004, 0.004, 0.004, 0.004, 0.003, 0.001, 0.001], [0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.0]]
E1_DONE debug
RUN [Show debug totals for const/Lambert/S1 cases] ssh LabServer63 'cd ~/rest_x97/E1; grep -E "^(const|lambert07|s1) " log_debug.txt | cut -c1-400; python3 - <<EOF
import json
o=json.load(open("results/debug.json"))
for k in o:
    print(k, [round(x,4) for x in o[k]["area_first9"]], [round(x,4) for x in o[k]["hemi_first9"]], o[k]["hist_area_sum"], o[k]["hist_hemi_sum"])
EOF'
  -> const {'area_total': 11.815619727076841, 'hemi_total': 13.466965641631246, 'hemi_over_area': 1.1397595684947577, 'hemi_vs_area': {'ratio': 1.1397595684947577, 'abs_rel_l1': 0.14056999680225798, 'corr': 0.9984952122470995}}
lambert07 {'area_total': 10.377050908503827, 'hemi_total': 12.241156045294835 ...[cut]... 1004367672, 'corr': 0.5252688807770016}}
const [1.147, 1.1719, 1.1483, 1.1412, 1.1529, 1.146, 0.7108, 0.7113, 0.7136] [1.2941, 1.349, 1.2937, 1.3315, 1.3755, 1.3313, 0.8984, 0.9231, 0.8995] 11.81561972707684 13.466965641631248
lambert07 [0.5677, 1.0173, 1.6963, 0.3437, 0.5094, 0.6955, 0.1506, 0.202, 0.254] [0.7672, 1.3276, 2.0154, 0.4981, 0.7485, 0.978, 0.2405, 0.3298, 0.399] 10.377050908503826 12.241156045294836
s1 [0.1553, 0.3119, 0.6375, 0.0835, 0.1296, 0.1888, 0.0465, 0.0626, 0.0796] [0.7253, 1.0779, 1.5264, 0.852, 1.252, 1.5478, 1.0419, 0.9521, 1.6406] 6.239828476054655 15.318780075412722
RUN [Check PCG32 lane correlation and an independent numpy form factor] cat > /tmp/e1_rngchk.py <<'EOF'
import sys, os, math
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, os.path.expanduser('~/rest_x97/E1'))
os.environ.setdefault('E1_CODE', os.path.expanduser('~/rest_x95a/code/experiments'))
for k, v in (('E1_ROOT', '~/rest_x97/E1'), ('E1_ORIG', '~/rest_x95a/results'), ('E1_R92A', '~/rest_x92a'), ('E1_R94', '~/rest_x94'), ('E1_R86', '~/rest_ecro86')):
    os.environ.setdefault(k, os.path.expanduser(v))
import e1_audit as E
mi, dr, F, U = E.mi,
  -> draw 0 mean 0.49999428894705034 var*12 1.0001902026959841 corr(lane) 2.8128098132144553e-05 corr(adjacent lanes) -0.0002635540730513355 corr lane vs lane+512 -8.270518404059895e-05
draw 1 mean 0.5000224105561983 var*12 0.9999158016837684 corr(lane) -0.0003032973874357854 corr(adjacent lanes) -0.0002 ...[cut]... r*12 1.0000906405960654 corr(lane) -0.00010053410878336201 corr(adjacent lanes) -0.00019661045792422873 corr lane vs lane+512 -0.01949904284513718
corr draw0-draw1 -0.0005512072863146822 d0-d2 -0.0013435750956898296 d1-d3 0.0007812744856170745
chi2-ish (d0,d1) 197.89990234375 dof 255
chi2-ish (lane%512, d0) 47.22998046875
chi2-ish (lane//512, d0) 127.31640625
numpy hemisphere form factor patch0 0.82347875 -> rho*pi*F 1.293517395693653
numpy area form factor patch0 0.8240680527685614 -> rho*F*pi 1.2944430703178793 PN [-0.98877108 -0.14943813  0.        ] PC [-1.23292278  0.27348105  0.7       ]
RUN [Fix own area-estimator visibility defect and rerun the debug stage] cd /c/Projects/ReST-GI/experiments/audit97/E1 && python - <<'EOF'
p='e1_audit.py'; s=open(p,newline='').read()
old="""        ok = (cx > 1e-6) & (cy > 1e-6)
        ray = mi.Ray3f(x + nx * 1e-4, om); ray.maxt = r - 2e-4
        ok &= ~ctx.scene.ray_test(ray, ok)
"""
new="""        ok = (cx > 1e-6) & (cy > 1e-6)
        # visibility segment between the two OFFSET end points (x + 1e-4 n_x, y + 1e-4 n_y). A first version shot the ray from the offset
        # origin along the un-offset direction wi
  -> ok
e1_audit.py: Python script, ASCII text executable
0
tail: cannot open 'log_debug.txt' for reading: No such file or directory
RUN [Wait for the debug stage and show output] ssh LabServer63 'cd ~/rest_x97/E1; sleep 20; for i in $(seq 1 60); do [ -f done_debug ] && break; sleep 3; done; cat done_debug; tail -8 log_debug.txt | cut -c1-1200'
  -> 0
const {'area_total': 13.470175340641429, 'hemi_total': 13.466965641631246, 'hemi_over_area': 0.9997617180972768, 'hemi_vs_area': {'ratio': 0.9997617180972768, 'abs_rel_l1': 0.0015158392061613763, 'corr': 0.999998615991477}, 'area_first9': [1.2926731960955173, 1.3483849504368735, 1.2943697810438088 ...[cut]... 74294558623174, 0.8864435789186862, 1.2573565133994717, 1.673047382582216, 0.9504626397590629, 1.1968709782596814, 1.5406448609103094], 'hemi_first9': [0.7253112888144986, 1.0778538375232738, 1.526353300842791, 0.8519861455252532, 1.2520177531736914, 1.5477518889783748, 1.041905573045339, 0.952140519478832, 1.640598483437091], 'hist_area_sum': 15.35950197131216, 'hist_hemi_sum': 15.318780075412722}
numpy_form_factor_patch0 {'hemisphere_rho_pi_F': 1.293517395693653, 'area_rho_pi_F': 1.2944430703178789, 'drjit_area_const': 1.2926731960955173, 'drjit_hemi_const': 1.2940610993647421}
E1_DONE debug
RUN [Rerun sender stage with fixed reference and print delivery results] ssh LabServer63 'cd ~/rest_x97/E1; rm -f done_sender; (setsid nohup bash run_e1.sh sender > /dev/null 2>&1 &); sleep 5; for i in $(seq 1 110); do [ -f done_sender ] && break; sleep 5; done; cat done_sender; tail -3 log_sender.txt | cut -c1-300' && cd /c/Projects/ReST-GI/results/audit97/E1 && scp -q LabServer63:~/rest_x97/E1/results/sender.json LabServer63:~/rest_x97/E1/results/sender_e1.npz LabServer63:~/rest_x97/E1/results/debug.json . && scp -q LabServer63:~/rest_x97/E1/results/sender_v0_buggy
  -> 0
 "gpu_mem_peak_mib_torch": 263.6337890625
}
E1_DONE sender
{
"own_area_split_half_noise": 0.003698075175307045,
"own_hemi_split_half_noise": 0.030054376497028246,
"own_hemi_vs_own_area": {
"ratio": 1.0057813297598484,
"abs_rel_l1": 0.023081362657162954,
"corr": 0.9992163065077116
},
"store ...[cut]... 1': {'stored_sender_max_abs_diff': 0.0, 'stored_sender_max': 43.28993948837463, 'stored_sender_rel_l1': 0.0, 'single_flight_term_own_vs_production_apply_rel_l1': 3.9132473181826704e-08, 'rw_buffer_max_abs_diff': 4.656520932733166e-10, 'factor_min': [1.931203000250445e-13, 5.226645338751723e-09]}, '8602': {'stored_sender_max_abs_diff': 0.0, 'stored_sender_max': 126.34958188606781, 'stored_sender_rel_l1': 0.0, 'single_flight_term_own_vs_production_apply_rel_l1': 3.9358065057957195e-08, 'rw_buffer_max_abs_diff': 4.656520932733166e-10, 'factor_min': [4.479996645060042e-15, 1.826640783075817e-10]}}
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
