# Track E3: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List experiment directories and file sizes] ls experiments/ecro86 experiments/ecro86a experiments/ecro_local93d experiments/ecro_local93d/addendum_checker experiments/ecro_mat94 experiments/ecro_attr95a 2>&1 | head -150; wc -l experiments/ecro86/*.py experiments/ecro86a/*.py experiments/ecro_mat94/*.py; ls docs/audits 2>&1; ls experiments/audit97 results/audit97 2>&1
RUN [Show CK2 scene JSON and ecro86 protocol] cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -c 6000; echo; echo ----; python -c "
import json
p=json.load(open('experiments/ecro86/protocol/ecro86.json'))
print(list(p.keys()))
for k in p:
    s=json.dumps(p[k])
    print(k, s[:1500])
"
  -> Exit code 2
experiments/ecro86:
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

experiments/ecro86a:
analyze86a.py
attrib86a.py
d2_86a.py
d3_86a.py
diag86a.py
protoco ...[cut]... cro86a/d2_86a.py
   147 experiments/ecro86a/d3_86a.py
   216 experiments/ecro86a/diag86a.py
   213 experiments/ecro86a/pt86a.py
   102 experiments/ecro86a/viz86a.py
    80 experiments/ecro86a/viz86a2.py
   379 experiments/ecro_mat94/an94.py
   310 experiments/ecro_mat94/c94.py
   578 experiments/ecro_mat94/mat94.py
    36 experiments/ecro_mat94/posthoc94.py
   168 experiments/ecro_mat94/viz94.py
  5145 total
ls: cannot access 'docs/audits': No such file or directory
ls: cannot access 'experiments/audit97': No such file or directory
ls: cannot access 'results/audit97': No such file or directory
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
    0.45 ...[cut]... D if F3 passes on both edits and F5 shows the update matters; CONDITIONALLY if F3 passes but F5 does not, or on one edit; INDETERMINATE if not resolvable", "END-TO-END ECONOMICS": "stage4.verdict", "GM1 ARCHITECTURE VIABILITY": "SUPPORTED only if all of the above are SUPPORTED within the declared scope (one room family, rigid box edits, three ER materials, point / spot sources, 512^2); otherwise the weakest label with its cause"}
scope_limits "one scene family with fixed stationary geometry and materials; material generalisation, non-rigid edits, area lights, other resolutions are NOT TESTED"
RUN [List experiment directories and file sizes] cd /c/Projects/ReST-GI && ls experiments/ecro86 experiments/ecro86a experiments/ecro_local93d experiments/ecro_local93d/addendum_checker experiments/ecro_mat94 experiments/ecro_attr95a 2>&1 | head -150; wc -l experiments/ecro86/*.py experiments/ecro86a/*.py experiments/ecro_mat94/*.py; ls docs/audits 2>&1; ls experiments/audit97 results/audit97 2>&1
RUN [Show CK2 scene JSON and ecro86 protocol] cd /c/Projects/ReST-GI && cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -c 6000; echo; echo ----; python -c "
import json
p=json.load(open('experiments/ecro86/protocol/ecro86.json'))
print(list(p.keys()))
for k in p:
    s=json.dumps(p[k])
    print(k, s[:1500])
"
RUN [Read scene86.py] cat experiments/ecro86/scene86.py
RUN [Read pt86.py] cat experiments/ecro86/pt86.py
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
RUN [Read scene86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/scene86.py
RUN [Read pt86.py] cd /c/Projects/ReST-GI && cat experiments/ecro86/pt86.py
RUN [Read pt86a.py] cat experiments/ecro86a/pt86a.py
RUN [Grep mat94.py for CKF field generation] grep -n "CKF\|e2_p\|def \|none\|field" experiments/ecro_mat94/mat94.py | head -150
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
  -> 7:  (iii) the physical references: pt86.Ctx.trace (camera / ECRO lookup renders) AND pt86a.field_src (patch x bin field references) get the
11:  python mat94.py fref   <R> <variant>                            CK2 field reference, two halves (results/fields/CKF_<v>.npz)
33:VARIANTS = ['none', 'U07',  ...[cut]... field'](ctx, A.Src(lt, 'true', ()), 64, 94902, b=64, split=True)
475:    fo2 = _INST['orig_field'](ctx, A.Src(lt, 'true', ()), 64, 94902, b=64, split=True)
476:    fp = _INST['patched_field'](ctx, A.Src(lt, 'true', ()), 64, 94902, b=64, split=True)
484:    f0 = A.field_src(ctx, A.Src(lt, 'true', ()), 256, 94903, b=128, split=True)
486:    f1 = A.field_src(ctx, A.Src(lt, 'true', ()), 256, 94903, b=128, split=True)
487:    e0 = (f0['e2_p'] + f0['e2_o'])[tg].astype(np.float64).sum((1, 2)); e1 = (f1['e2_p'] + f1['e2_o'])[tg].astype(np.float64).sum((1, 2))
534:    for v in ('none', 'bx38', 'bx75'):
RUN [Read pt86a.py] cd /c/Projects/ReST-GI && cat experiments/ecro86a/pt86a.py
RUN [Grep mat94.py for CKF field generation] cd /c/Projects/ReST-GI && grep -n "CKF\|e2_p\|def \|none\|field" experiments/ecro_mat94/mat94.py | head -150
RUN [Read mat94.py setup and fref] sed -n 1,135p experiments/ecro_mat94/mat94.py; sed -n 135,200p experiments/ecro_mat94/mat94.py; sed -n 285,305p experiments/ecro_mat94/mat94.py
  -> """WL94 (protocol ecro_mat94): per-patch material machinery, Stage A tests, Stage B frozen material counterfactuals on CK2.

Per-patch materials are consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors); new Lambert rows are a ...[cut]... (lt, 'true', ()), spp, sd, b=128, split=True)
        for k, x in t.items():
            out[f'{k}_{h}'] = x
    fn = os.path.join(R, 'results', 'fields', f'CKF_{v}.npz'); os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, PMAT=ctx.G.PMAT, spp=spp, seeds=np.array(FREF_SEEDS), **out)
    jd({'variant': v, 'spp_per_half': spp, 'seeds': FREF_SEEDS, 'seconds': time.time() - t0, 'patch_sha256': _INST['patch_sha'],
        'pmat_digest': pm_digest(ctx.G.PMAT)}, fn.replace('.npz', '.json'))
    print('FREF', v, round(time.time() - t0, 1), 's', flush=True)


def iref(R, v, r):
RUN [Read mat94.py setup and fref] cd /c/Projects/ReST-GI && sed -n 1,135p experiments/ecro_mat94/mat94.py; sed -n 135,200p experiments/ecro_mat94/mat94.py; sed -n 285,305p experiments/ecro_mat94/mat94.py
RUN [List relevant worklogs and WL96 scratch] ls docs/worklogs | grep -E "^(9[4-6]|87)" ; wc -l docs/worklogs/95A_* docs/worklogs/96_* docs/worklogs/96A_* 2>/dev/null; ls results/wl96_calc results/wl96_calc/scratch | head -80
RUN [Read quad87.py (M64 definition)] cat experiments/gm1_explicit87/quad87.py | head -200
  -> 87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
94A_WORKLOG_ECRO_MATERIAL_FIDELITY_RETENTION_SMOKE.md
94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md
94C_WORKLOG_ECRO_PANEL_MEDIATED_QUERY_SUPERVISION_SMOKE.md
94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md
94_WORKLOG_ECRO_MATERIAL_ ...[cut]... ITY_AND_FLIGHT_ATTRIBUTION.md
95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md
96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md
96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
  213 docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
  267 docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
  263 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md
  743 total
results/wl96_calc:
ck2_support.py
ck2_support_E.npy
ring_alloc.json
ring_alloc.py
scratch

results/wl96_calc/scratch:
critic_cost
critic_phys
d1
d2
d3
d4
  -> """WL87 discrete ER scattering on the WL86 patch x bin state (learning-free; frozen ER closures, Dr.Jit).

Notation (patch-local frame (t, b, n) of scene86; 64 equal-solid-angle bins, DOMEGA = 2 pi / 64; eval(wi, wo) = f(wi, wo) cos(wo)).
ER f is reciprocal (f(a, b) = f(b, a): symmetric GGX lobes, s ...[cut]... igin point: local shading frame, local cosine, local horizon;
    normalised by the bin's mean valid solid angle, as the oracle field pt86.Ctx.field averages over valid directions) - plus dense
    tables (built lazily; flat patch-frame approximation, used only by the ENC / C3 controls)."""

    def __init__(self, us, M_ray, seed_ray, M_dense=None, seed_dense=None):
        self.states, self.lam, self.alb = mats()
        self.dirs = ray_dirs(us)
        cosr = self.dirs[:, 2]
        self.W = np.zeros((6 + S.SPH_NZ, K, NB, 3), np.float64)
        for m in range(6):
            if self.lam[m]:
RUN [List relevant worklogs and WL96 scratch] cd /c/Projects/ReST-GI && ls docs/worklogs | grep -E "^(9[4-6]|87)" ; wc -l docs/worklogs/95A_* docs/worklogs/96_* docs/worklogs/96A_* 2>/dev/null; ls results/wl96_calc results/wl96_calc/scratch | head -80
RUN [Read quad87.py (M64 definition)] cd /c/Projects/ReST-GI && cat experiments/gm1_explicit87/quad87.py | head -200
RUN [Read WL95A worklog first half] sed -n 1,120p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
  -> # Worklog — WL95A: Sharp Sender Representability & Flight Attribution Smoke (2026-11-15)

## Origin and scope

- **Brief.** "WL95A — Sharp Sender Representability & Flight Attribution Smoke" (user). It is a suffix of WL95 and a bounded, learning-free attribution batch.
- **Proposition tested.** "The ...[cut]... CK2's grazing (ring-0) incidence, every material-level rank-16 operator loses the S1 lobe almost completely: peak 0.018 (NMF16) and 0.002 (L1-NMF16), with energy 0.46 and 0.11. Rank 32 keeps 0.43 of the peak at energy 0.67. The learned senders keep 0.04 and 0.13 of the peak.
  - The learned per-patch S1 operators are non-passive for some incident bins (directional albedo up to 2.3 / 3.1). The bound operators and M64 are passive (≤ 0.96).
  - The WL95 validation-field bounds (rank-16 field rel L1 0.36–0.54) do **not** carry over to this incidence. Ring 0 holds only 4.3 % of training irradiance.
RUN [Read WL95A worklog first half] cd /c/Projects/ReST-GI && sed -n 1,120p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
RUN [Read rest of WL95A and grep WL96 claims] sed -n 120,213p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md; grep -n "1\.015\|0\.068\|1\.486\|0\.487\|area-form\|area form\|deterministic" docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | head -60
  -> - The WL95 validation-field bounds (rank-16 field rel L1 0.36–0.54) do **not** carry over to this incidence. Ring 0 holds only 4.3 % of training irradiance.
- **Delivery, rank arms.** NMF16 delivers 0.63× M64's box-A flux (pooled) with a map 0.65 away from M64's. L1-NMF16 delivers 0.16× M64's. Rank  ...[cut]... ess beyond explicit ER + transport: **nothing in expectation**. All four designers concede this. Its only claims are amortisation (cost) and a deterministic tail.
140:  - Agreement of 0.068 against n0 0.059 leaves up to about 0.03 of unresolved oracle bias.
174:  - Photons reach per-draw 0.03–0.07 with 40–160 k rays per light per facet. DQ is deterministic with no per-light rays, but is geometry-quadratic.
210:1. **V0 contract (STOP gate).** A Dr.Jit deterministic e2_p oracle with real scene visibility vs path-traced e2_p on every PT-referenced light: |ratio − 1| ≤ 2 SE and rel L1 ≤ n0 + 2 SE.
RUN [Read rest of WL95A and grep WL96 claims] cd /c/Projects/ReST-GI && sed -n 120,213p docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md; grep -n "1\.015\|0\.068\|1\.486\|0\.487\|area-form\|area form\|deterministic" docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | head -60
RUN [Read WL96 verified facts] sed -n 36,60p docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md; sed -n 134,145p docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
  -> - **What fixes sharp delivery.** In every design, the mechanism that fixes sharp first-flight delivery is learning-free and explicit: G1 class, a comparator under current canon.
- **The only learned element that survives** is one broad ≥3-event resolvent, written four ways. It carries no sharp spati ...[cut]... 95A learned senders were non-passive.
- **Oracle validation is one configuration.**
  - The CK2 grazing omni light, Lambert receivers, no shadow edge across the panel.
  - The other lights are oracle-only.
  - Agreement of 0.068 against n0 0.059 leaves up to about 0.03 of unresolved oracle bias.
- **Image-level relevance on CK2 is unresolved** (§A).

**Cost / scaling.**
- **Amdahl ceiling (cost critic §2.2; anchors verified).** Assume a perfect, zero-cost learned resolvent that replaces the whole 10.6 ms sweep stage. The best lifecycle ratios are then:
  - edit→frame 124.7 / 114.1 = **1.09×**;
RUN [Read WL96 verified facts] cd /c/Projects/ReST-GI && sed -n 36,60p docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md; sed -n 134,145p docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
RUN [Check local results and server environment] ls results/ecro_mat94 2>/dev/null | head; ls results/ecro_attr95a | head -30; ssh -o ConnectTimeout=15 LabServer63 'ls ~/rest_x94/results/fields/ | head -20; ls ~/rest_x97 2>/dev/null; cat ~/rest_x95a/env.sh; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; ls ~/rest_x96a/code/experiments | head -80'
  -> chain.status
final.json
gpu.txt
models
pop
posthoc94.json
pre.json
reused_refs.txt
select94.json
stageB.json
an.json
chain.status
deliv.npz
gpu.txt
main.json
ops95a.npz
s0.json
CKF_U03.json
CKF_U03.npz
CKF_U07.json
CKF_U07.npz
CKF_bx38.json
CKF_bx38.npz
CKF_bx38i.json
CKF_bx38i.npz
CKF_bx75.json
CKF ...[cut]... a61
fnto_diag45
fnto_diag48
fnto_dir46
fnto_direction64
fnto_drift71a
fnto_fia76
fnto_fia77
fnto_k55
fnto_lc52
fnto_lego
fnto_lego_k44
fnto_lego_r1
fnto_orbit80
fnto_owner51
fnto_persist59
fnto_rank66
fnto_rank75
fnto_repr68
fnto_reuse58
fnto_rule57
fnto_sel56
fnto_smoke47
fnto_spatial49
fnto_src73
fnto_srcid78
fnto_sym79
fnto_visual_review
fnto_walker71c
full_recompute_benchmark
gm1_c3urx1_91
gm1_char85
gm1_explicit87
gm1_gap90
gm1_relation88
gm1_relation89
neural_recompute_cost
phys_ref53
radiometric_relation_state_prototype
relational_residual_prototype
rna_baseline_quality
rna_reproduction
RUN [Check local results and server environment] cd /c/Projects/ReST-GI && ls results/ecro_mat94 2>/dev/null | head; ls results/ecro_attr95a | head -30; ssh -o ConnectTimeout=15 LabServer63 'ls ~/rest_x94/results/fields/ | head -20; ls ~/rest_x97 2>/dev/null; cat ~/rest_x95a/env.sh; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; ls ~/rest_x96a/code/experiments | head -80'
RUN [Read the ER closure module] wc -l experiments/fnto_compiled83/*.py && cat experiments/fnto_compiled83/closure83_dr.py
  -> 163 experiments/fnto_compiled83/bench83.py
   33 experiments/fnto_compiled83/closure83.py
  202 experiments/fnto_compiled83/closure83_dr.py
  164 experiments/fnto_compiled83/compiler83.py
  179 experiments/fnto_compiled83/data83.py
  135 experiments/fnto_compiled83/eval83.py
   14 experiments/fnto_c ...[cut]...      if self.arm == 'ER':
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
RUN [Read the ER closure module] cd /c/Projects/ReST-GI && wc -l experiments/fnto_compiled83/*.py && cat experiments/fnto_compiled83/closure83_dr.py
RUN [Read rest of quad87.py and mq_ext] sed -n 200,330p experiments/gm1_explicit87/quad87.py; grep -n "def mq_ext" -A 8 experiments/ecro_mat94/mat94.py
  -> if self.lam[m]:
                self.W[m] = (self.alb[m] / math.pi * cosr * DOMEGA / KAPPA)[:, None, None]
            else:
                fb = fbar(self.states[m], self.dirs, M_ray, seed_ray + 10 * m)
                self.W[m] = fb * (cosr * DOMEGA / KAPPA)[:, None, None]
        self.omega_sph = ...[cut]...                              # inside a patch-frame bin and above the local horizon
        cv = dr.select(keep, val * E * (dr.gather(F, qs['invom'], slot) * (1.0 / (P * Mq))), 0.0)
231:def mq_ext():
232-    import diag92 as DG
233-    _, Mq = DG.local_matrices()
234-    lam = Mq[3].astype(np.float64) / 0.7
235-    return np.concatenate([Mq] + [(lam * ALB[i])[None].astype(Mq.dtype) for i in NEW_IDS])
236-
237-
238-def arm_tables(model, gs, s, Mn, s1, arms):
239-    """Single flight per arm (WL93D arm definitions + C2_srcS1 / C2_srcNonS1), the deployed global term, and sender-split C0 / C2_rcv.
RUN [Read rest of quad87.py and mq_ext] cd /c/Projects/ReST-GI && sed -n 200,330p experiments/gm1_explicit87/quad87.py; grep -n "def mq_ext" -A 8 experiments/ecro_mat94/mat94.py
RUN [Grep a95.py metric definitions] grep -n "def \|W_T\|n0\|ratio\|rel\b\|corr" experiments/ecro_attr95a/a95.py | head -120
  -> 27:WT = ZB * DOM                                         # corrected cos dOmega (4 rings x 16 sectors)
32:def jd(o, fn):
37:def jl(fn):
41:def setup(R):
47:def exitance(T, W):
52:def ref_fields(R):
61:def rw_of(u, S):
67:def relation(ctx, u):
73:def flight(hit, dbin, rw, Bs1, N, S):
82:def flight_ra ...[cut]... z, mj):
555:    for a_, t in zip(ax[:3], ('S1 sender field flux rel L1 (CK2, W_T)', 'energy ratio', 'peak retention')):
560:    ax[3].set_title('operator max directional albedo (W_T)', fontsize=8); ax[3].legend(fontsize=7)
561:    fig.suptitle(f"WL95A senders on CK2 'none' S1 field; q split-half rel L1 {sm['q_split_half']['rel_l1_half']:.3f}", fontsize=10)
567:        dep = [D.get(f'dep|{k}|{mn}', {}).get('ratio', np.nan) for k in arms]
568:        pol = [D[f'pool|{k}|{mn}']['ratio'] for k in arms]
576:        ax[i, 1].set_title(f'{mn}: abs rel L1 vs reference e2_p (W_T exitance)', fontsize=8)
RUN [Grep a95.py metric definitions] cd /c/Projects/ReST-GI && grep -n "def \|W_T\|n0\|ratio\|rel\b\|corr" experiments/ecro_attr95a/a95.py | head -120
RUN [Read a95 exitance and metric definitions; inspect deliv.npz] sed -n 20,62p experiments/ecro_attr95a/a95.py; sed -n 400,416p experiments/ecro_attr95a/a95.py; python -c "
import numpy as np
z=np.load('results/ecro_attr95a/deliv.npz'); print({k:z[k].shape for k in z.files})"
  -> S1 = np.arange(1088, 1130)
BOXA = np.arange(1130, 1184)
PFACE = np.arange(1130, 1139)
SEEDS = [8601, 8602]
NB = 64
DOM = 2 * math.pi / NB
ZB = ((np.arange(NB) // 16) + 0.5) / 4
WT = ZB * DOM                                         # corrected cos dOmega (4 rings x 16 sectors)
W94 = ((np.arange(NB) / ...[cut]...   o['corr'] = float(np.corrcoef(a, r)[0, 1])
    return o


def jk(draws, ref, mask, fn):
{'dep|A1': (1270,), 'dep|A2': (1270,), 'dep|A3': (1270,), 'dep|A4': (1270,), 'dep|A0_s8601': (1270,), 'dep|A0_s8602': (1270,), 'dep|M64_spool': (1270,), 'dep|q256': (1270,), 'dep|X1c': (1270,), 'dep|X1': (1270,), 'draws|A1': (64, 1270), 'draws|A2': (64, 1270), 'draws|A3': (64, 1270), 'draws|A4': (64, 1270), 'draws|A0_s8601': (64, 1270), 'draws|A0_s8602': (64, 1270), 'draws|M64_spool': (64, 1270), 'draws|q256': (64, 1270), 'draws|X1c': (64, 1270), 'draws|X1': (64, 1270), 'ref_a': (1270,), 'ref_b': (1270,)}
RUN [Read a95 exitance and metric definitions; inspect deliv.npz] cd /c/Projects/ReST-GI && sed -n 20,62p experiments/ecro_attr95a/a95.py; sed -n 400,416p experiments/ecro_attr95a/a95.py; python -c "
import numpy as np
z=np.load('results/ecro_attr95a/deliv.npz'); print({k:z[k].shape for k in z.files})"
RUN [Check code drift between commits and print material states] git diff --stat 7786d09 a1718f7 -- experiments/ecro86 experiments/ecro86a experiments/fnto_compiled83 experiments/fnto_compiled82 experiments/fnto_compiled81 | tail -5; python -c "
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=200)
z=np.load('experiments/ecro86/data/mats86.npz', allow_pickle=True)
print({k:z[k].shape for k in z.files})
st=z['states']
for m in range(st.shape[0]):
    s=st[m]
    print('mat',m,'lambert',z['lambert'][m],'alb',z['albedo'][m])
    print
  -> experiments/ecro86a/run86a.sh                |  37 ++++
 experiments/ecro86a/run86a2.sh               |  31 +++
 experiments/ecro86a/viz86a.py                | 102 ++++++++++
 experiments/ecro86a/viz86a2.py               |  80 ++++++++
 18 files changed, 1620 insertions(+), 21 deletions(-)
{'states' ...[cut]... .74227 0.11389]
 [0.3     1.5     0.5     0.5     0.5     0.5     0.5     0.5     0.5     0.3    ]
 [0.3     1.5     0.5     0.5     0.5     0.5     0.5     0.5     0.5     0.3    ]
 [0.3     1.5     0.5     0.5     0.5     0.5     0.5     0.5     0.5     0.3    ]]
d3c5be854d80665b11f476eff0333a0232828293d31b28a6069f4dbd81371b3f *experiments/ecro86/data/mats86.npz
977ba122ab3232e7d428155c0963cff2f7764ce56704274832ee12d37b09de75 *experiments/ecro_local93d/addendum_checker/scene_ck2.json
ecca67246ef013d798fa4bbec2cd93aef4f0430b43d51429b6953821ce2451b5 *experiments/ecro86/protocol/ecro86.json
RUN [Check code drift between commits and print material states] cd /c/Projects/ReST-GI && git diff --stat 7786d09 a1718f7 -- experiments/ecro86 experiments/ecro86a experiments/fnto_compiled83 experiments/fnto_compiled82 experiments/fnto_compiled81 | tail -5; python -c "
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=200)
z=np.load('experiments/ecro86/data/mats86.npz', allow_pickle=True)
print({k:z[k].shape for k in z.files})
st=z['states']
for m in range(st.shape[0]):
    s=st[m]
    print('mat',m,'lambert',z['lambert'][m],'alb'
RUN [Copy frozen code into own server dir and verify hashes] git diff --stat 7786d09 a1718f7 -- experiments/ecro86 experiments/fnto_compiled83 experiments/fnto_compiled82 experiments/fnto_compiled81 | tail -8; grep -n "def init" -A 14 experiments/fnto_compiled81/layered81.py | head -40; ssh LabServer63 'mkdir -p ~/rest_x97/E3 && cd ~/rest_x97/E3 && mkdir -p code/experiments out && for d in ecro86 fnto_compiled81 fnto_compiled82 fnto_compiled83; do cp -r ~/rest_x96a/code/experiments/$d code/experiments/; done; find code -name __pycache__ -prune -exec rm -r
  -> experiments/ecro86/analyze86.py             |  3 ++
 experiments/ecro86/diag_v3.py               | 37 +++++++++++++++++
 experiments/ecro86/protocol/amendment1.json | 11 +++++
 experiments/ecro86/run86.sh                 |  9 +++++
 experiments/ecro86/tests86.py               | 63 ++++++++++++++++++ ...[cut]... ac11b3210"
 },
 "pmat_digest": "3d894e73cef701a4"
}-rw-r--r-- 1 nd nd 5730422 Nov 14 10:46 /home/nd/rest_x94/results/fields/CKF_none.npz
32
               total        used        free      shared  buff/cache   available
Mem:             125           4          12           0         109         120
912e68acd314d79ef0cc270784d99f79ed8024caf856527a127f31e6ccd6c983 *experiments/ecro86/pt86.py
b75a2d9cbae1f738429e16fa32095a1c3cdf346bff5f5e55d5576ad14ce39e69 *experiments/ecro86/scene86.py
3eee59095fc77fdb7a965d1a14a124230efa40fda05a2c4744e260b64c904614 *experiments/fnto_compiled83/closure83_dr.py
RUN [Copy frozen code into own server dir and verify hashes] cd /c/Projects/ReST-GI && git diff --stat 7786d09 a1718f7 -- experiments/ecro86 experiments/fnto_compiled83 experiments/fnto_compiled82 experiments/fnto_compiled81 | tail -8; grep -n "def init" -A 14 experiments/fnto_compiled81/layered81.py | head -40; ssh LabServer63 'mkdir -p ~/rest_x97/E3 && cd ~/rest_x97/E3 && mkdir -p code/experiments out && for d in ecro86 fnto_compiled81 fnto_compiled82 fnto_compiled83; do cp -r ~/rest_x96a/code/experiments/$d code/experiments/; done; find code -name __py
RUN [Write independent ER port and geometry modules] mkdir -p /c/Projects/ReST-GI/experiments/audit97/E3 /c/Projects/ReST-GI/results/audit97/E3 /c/Projects/ReST-GI/docs/audits/wl97_evidence_reliability && cd /c/Projects/ReST-GI/experiments/audit97/E3 && cat > er97.py <<'EOF'
"""WL97 audit track E3: independent float64 port of the frozen ER closure (written from the formula in the docstring of
experiments/fnto_compiled83/closure83_dr.py; validated against the production DrClosure83 by val97_dr.py / cmp97.py).

eval(wi, wo) = f(wi, wo) * cos(theta_o
  -> 1.0 [0.01227185 0.03681554 0.06135923 0.08590292]
1130 [-1.2778  0.5701  1.    ] [-0.9888 -0.1494  0.    ] [0.45 0.45]
1139 [-0.8822  0.6299  1.    ] [ 0.9888  0.1494 -0.    ] [0.45 0.45]
1148 [-1.0128  0.1551  1.    ] [ 0.1494 -0.9888  0.    ] [0.2  0.45]
1157 [-1.1472  1.0449  1.    ] [-0.1494  0.9888  0.    ] [0.45 0.2 ]
1166 [-1.08  0.6   0.55] [ 0.  0. -1.] [0.45 0.2 ]
1175 [-1.08  0.6   1.45] [ 0. -0.  1.] [0.2  0.45]
RUN [Write and run production-side validation on server] cd /c/Projects/ReST-GI/experiments/audit97/E3 && cat > val97_dr.py <<'EOF'
"""WL97 audit track E3 - production-side facts (Dr.Jit / Mitsuba, frozen pt86 scene + DrClosure83; nothing here is audited measurement code).

  python val97_dr.py <CODE> <OUT>
    CODE : directory holding experiments/ (private copy of the frozen code)      OUT : output directory (own)
Writes OUT/val_dr.json and OUT/val_pairs.npz:
  T_GEOM   geo97 patch tables == scene86.Geom tables for patches 1088..1183
  T_FRAME  Mitsu
  -> Exit code 2
/usr/bin/bash: -c: line 133: unexpected EOF while looking for matching `''
WRITE c:\Projects\ReST-GI\experiments\audit97\E3\val97_dr.py
WRITE c:\Projects\ReST-GI\experiments\audit97\E3\run_e3.sh
RUN [Upload and run production-side validation] cd /c/Projects/ReST-GI/experiments/audit97/E3 && ls && file *.py *.sh && scp -q er97.py geo97.py val97_dr.py run_e3.sh LabServer63:rest_x97/E3/ && ssh LabServer63 "cd ~/rest_x97/E3 && file run_e3.sh val97_dr.py && chmod +x run_e3.sh && ./run_e3.sh val97_dr.py /home/nd/rest_x97/E3/code /home/nd/rest_x97/E3/out 2>&1 | tail -130"
  -> __pycache__
er97.py
geo97.py
run_e3.sh
val97_dr.py
er97.py:     Python script, ASCII text executable
geo97.py:    Python script, ASCII text executable
val97_dr.py: Python script, ASCII text executable
run_e3.sh:   Bourne-Again shell script, ASCII text executable
run_e3.sh:   Bourne-Again shell script, ASCII text executable
val97_dr.py: Python script, ASCII text executable
Traceback (most recent call last):
  File "/home/nd/rest_x97/E3/val97_dr.py", line 26, in <module>
    fx, sc = geo97.load(CODE)
  File "/home/nd/rest_x97/E3/geo97.py", line 15, in load
    sc = json.load(open(os.path.join(repo, 'experiments', 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
FileNotFoundError: [Errno 2] No such file or directory: '/home/nd/rest_x97/E3/code/experiments/ecro_local93d/addendum_checker/scene_ck2.json'
RUN [Copy scene JSON and rerun validation] rm -rf /c/Projects/ReST-GI/experiments/audit97/E3/__pycache__; ssh LabServer63 "cd ~/rest_x97/E3 && mkdir -p code/experiments/ecro_local93d/addendum_checker && cp ~/rest_x96a/code/experiments/ecro_local93d/addendum_checker/scene_ck2.json code/experiments/ecro_local93d/addendum_checker/ && sha256sum code/experiments/ecro_local93d/addendum_checker/scene_ck2.json && ./run_e3.sh val97_dr.py /home/nd/rest_x97/E3/code /home/nd/rest_x97/E3/out 2>&1 | tail -130"
  -> 977ba122ab3232e7d428155c0963cff2f7764ce56704274832ee12d37b09de75  code/experiments/ecro_local93d/addendum_checker/scene_ck2.json
   0.0
  ],
  "t_local_of_+z": [
   0.0,
   1.0,
   0.0
  ],
  "n_local_of_+x": [
   0.0,
   0.0,
   1.0
  ],
  "hit_shape_ids": [
   5
  ],
  "hit_dist_err": 4.8428773880 ...[cut]... _base": 1157,
    "analytic_visible_frac": 0.0,
    "mismatch": 204,
    "mismatch_excluding_abs_cos_r_below_1e-3": 204,
    "cos_s_min": 0.09086939028339786
   },
   {
    "face_base": 1166,
    "analytic_visible_frac": 0.17913246154785156,
    "mismatch": 213,
    "mismatch_excluding_abs_cos_r_below_1e-3": 185,
    "cos_s_min": 0.09379738757512457
   },
   {
    "face_base": 1175,
    "analytic_visible_frac": 0.1788644790649414,
    "mismatch": 617,
    "mismatch_excluding_abs_cos_r_below_1e-3": 163,
    "cos_s_min": 0.09397417311108516
   }
  ],
  "n_per_face": 2097152,
  "pass": false
 }
}
RUN [Refine visibility test and rerun] cd /c/Projects/ReST-GI/experiments/audit97/E3 && python - <<'PYEOF'
import re
s = open('val97_dr.py').read()
old = s[s.index("tot_mis = 0; per = []"):s.index("res['T_VIS'] = vis")]
new = '''bc = np.array(sc['geom']['A']['c'], float); bh = np.array(sc['geom']['A']['h'], float)
cy, sy = math.cos(sc['geom']['A']['yaw']), math.sin(sc['geom']['A']['yaw'])
bax = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0, 0, 1.0]])


def inside_len(a, b):
    """Length of the part of segment a -> b that lies inside b
  -> val97_dr.py: Python script, ASCII text executable
{"max_abs_err": 0.0, "pass": true, "N": 1270, "panel_pmat": [1], "boxA_pmat": [4], "panel_psid": [5], "boxA_psid": [7]}
True True [1.0, 0.0, 0.0]
{"max_rel_vs_rho_over_pi_cos_SECOND_arg": 0.0, "max_rel_vs_rho_over_pi_cos_FIRST_arg": 952.4815569928409 ...[cut]... 913246154785156,
"visible_but_occluded": 28,
"hidden_but_unoccluded": 185,
"hidden_but_unoccluded_max_inside_len_m": 9.929602020286504e-05,
"cos_s_min": 0.09379738757512457
},
{
"face_base": 1175,
"analytic_visible_frac": 0.1788644790649414,
"visible_but_occluded": 455,
"hidden_but_unoccluded": 162,
"hidden_but_unoccluded_max_inside_len_m": 9.863482969572844e-05,
"cos_s_min": 0.09397417311108516
}
],
"n_per_face": 2097152,
"note": "ray test: origin a + 1e-4 w, maxt = dist - 2e-4; a hidden pair is reported unoccluded only if its path inside box A is shorter than these epsilons",
"pass": false
}
RUN [Add grazing diagnostics to visibility test and rerun] cd /c/Projects/ReST-GI/experiments/audit97/E3 && python - <<'PYEOF'
s = open('val97_dr.py').read()
s = s.replace("""'hidden_but_unoccluded': int(hid_but_clear.sum()),""", """'visible_but_occluded_max_cos_r': float(cr[vis_but_occ].max()) if vis_but_occ.any() else 0.0,
                'visible_but_occluded_cos_r_weight_share': float(cr[vis_but_occ].sum() / cr[ana].sum()) if ana.any() else 0.0,
                'hidden_but_unoccluded': int(hid_but_clear.sum()),""")
s = s.replace("""    bad_vis += in
  -> {'face_base': 1130, 'analytic_visible_frac': 1.0, 'visible_but_occluded': 0, 'visible_but_occluded_max_cos_r': 0.0, 'visible_but_occluded_cos_r_weight_share': 0.0, 'hidden_but_unoccluded': 0, 'hidden_but_unoccluded_max_inside_len_m': 0.0, 'cos_s_min': 0.09013126190353088}
{'face_base': 1139, 'analyt ...[cut]... 5, 'cos_s_min': 0.09379738757512457}
{'face_base': 1175, 'analytic_visible_frac': 0.1788644790649414, 'visible_but_occluded': 455, 'visible_but_occluded_max_cos_r': 0.0010042152463842376, 'visible_but_occluded_cos_r_weight_share': 2.1180768442392876e-06, 'hidden_but_unoccluded': 162, 'hidden_but_unoccluded_max_inside_len_m': 9.863482969572844e-05, 'cos_s_min': 0.09397417311108516}
True
total 200164
drwxr-xr-x 2 nd nd      4096 Nov 15 17:53 .
drwxr-xr-x 4 nd nd      4096 Nov 15 17:52 ..
-rw-r--r-- 1 nd nd      4238 Nov 15 17:55 val_dr.json
-rw-r--r-- 1 nd nd 204950746 Nov 15 17:55 val_pairs.npz
WRITE c:\Projects\ReST-GI\experiments\audit97\E3\cmp97.py
RUN [Compare independent ER port with production closure] cd /c/Projects/ReST-GI/experiments/audit97/E3 && scp -q cmp97.py LabServer63:rest_x97/E3/ && ssh LabServer63 "cd ~/rest_x97/E3 && ./run_e3.sh cmp97.py /home/nd/rest_x97/E3/out 2>&1 | tail -20"
  -> uniform|mat0 {'n': 1046494, 'median_rel': 9.982872121802668e-08, 'p999_rel': 5.102658255129298e-07, 'max_rel': 9.396661346962302e-07, 'sum_ratio_port_over_production': [1.0000000397966988, 1.0000000471098844, 1.00000003472406], 'max_abs_over_max_value': 4.745917707971303e-07, 'production_max': 1.251 ...[cut]... 44694822407e-07, 'p999_rel': 6.253055614942093e-07, 'max_rel': 9.930317816165543e-07, 'sum_ratio_port_over_production': [1.0000000632302901, 1.0000000603206485, 1.0000000618692972], 'max_abs_over_max_value': 4.0241840827320505e-07, 'production_max': 53.995216369628906}
graze_any|mat4 {'n': 1047562, 'median_rel': 4.0368269676354674e-08, 'p999_rel': 9.695599135440923e-08, 'max_rel': 9.968309510044215e-08, 'sum_ratio_port_over_production': [1.0000000403079783, 1.0000000403079783, 1.0000000403079783], 'max_abs_over_max_value': 8.69957046407001e-08, 'production_max': 0.039788730442523956}
pass True
WRITE c:\Projects\ReST-GI\experiments\audit97\E3\quad97.py
RUN [Run table-quadrature validation test on server] cd /c/Projects/ReST-GI/experiments/audit97/E3 && sed -i 's/for nz, nphi in ((512, 4096), (1024, 8192)):/for nz, nphi in ((256, 2048), (512, 4096)):/; s/bref = br\[.1024x8192.\]/bref = br["512x4096"]/; s/br\[.512x4096.\] \* wt/br["256x2048"] * wt/; s/(br\[.512x4096.\] - bref)/(br["256x2048"] - bref)/' quad97.py && grep -n "256x2048\|512x4096" quad97.py && file quad97.py && scp -q quad97.py LabServer63:rest_x97/E3/ && ssh LabServer63 "cd ~/rest_x97/E3 && nvidia-smi --query-gpu=memory.used,utilizat
  -> 153:    bref = br["512x4096"]
155:    res['brute_self_convergence'] = {'albedo_max_rel': float(((br["256x2048"] * wt[None, :, None]).sum(1) / alb_ref - 1).abs().max()),
156:                                     'bin_max_abs_over_bin_max': float((br["256x2048"] - bref).abs().max() / bref.max())}
quad9 ...[cut]... rectional_albedo": [
  [
   0.8453334381448553,
   0.788269950064184,
   0.8332153345071022
  ],
  [
   0.6096695518714912,
   0.6652551469129193,
   0.6562052128935376
  ],
  [
   0.8033033617469748,
   0.7621609243818411,
   0.7963741435364265
  ],
  [
   0.7431557603823223,
   0.7275290896509828,
   0.7467910193313062
  ],
  [
   0.8680095585964891,
   0.8058259558176907,
   0.8583355913913405
  ],
  [
   0.8395904803815999,
   0.7872687509355502,
   0.8325484654135218
  ]
 ]
}
[This command modified 1 file you've previously read: experiments\audit97\E3\quad97.py. Call Read before editing.]
RUN [Show table test summary] ssh LabServer63 "cd ~/rest_x97/E3 && python3 -c \"
import json
j=json.load(open('out/tabtest.json'))
for k,v in j.items():
    if k not in ('albedo_times_cos','directional_albedo'): print(k, v)
\""
  -> lambert_bin_max_rel {'192x128': 0.2335812316484981, '384x256': 0.05492278518644067, '768x512': 0.03713355530853124}
test_dirs [[-0.29389, 0.95513, 0.03674], [-0.79603, 0.59702, 0.0995], [-0.75175, 0.65778, 0.04698], [-0.99038, 0.1238, 0.0619], [-0.78601, 0.61758, 0.02807], [-0.99683, 0.0712, 0.0356] ...[cut]... 6982280961265045, 's': 0.0150604248046875}
hv_384x256 {'albedo_max_rel_vs_brute': 0.0004563501179376539, 'bin_L1_rel_vs_brute': 0.0006518976407141601, 'bin_max_abs_over_bin_max': 0.0003684706670842226, 's': 0.01803731918334961}
hv_768x512 {'albedo_max_rel_vs_brute': 0.000127894856120081, 'bin_L1_rel_vs_brute': 0.0002145830592551408, 'bin_max_abs_over_bin_max': 0.00017156143916064845, 's': 0.06373190879821777}
hv_1536x1024 {'albedo_max_rel_vs_brute': 3.482015450217979e-05, 'bin_L1_rel_vs_brute': 6.494361600525447e-05, 'bin_max_abs_over_bin_max': 2.4697398831661017e-05, 's': 0.27715492248535156}
RUN [Launch tables and coarse delivery jobs detached] ssh LabServer63 "cd ~/rest_x97/E3 && cat > job1.sh <<'EOF'
#!/bin/bash
cd ~/rest_x97/E3
C=/home/nd/rest_x97/E3/code; O=/home/nd/rest_x97/E3/out
rm -f job1.done
./run_e3.sh quad97.py tables \$C \$O a 16 384 256 8
./run_e3.sh quad97.py tables \$C \$O c 32 384 256 32
./run_e3.sh quad97.py deliver \$C \$O s8r12 8 12 --tables=\$O/tables_a.npz
./run_e3.sh quad97.py deliver \$C \$O s4r12 4 12 --tables=\$O/tables_a.npz
touch job1.done
EOF
chmod +x job1.sh && setsid nohup ./job1.sh > job1.log 2>&1 & echo
  -> started
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
