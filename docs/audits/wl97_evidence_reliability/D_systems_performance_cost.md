# WL97 Track D - Systems and performance audit of the cost / lifecycle claims (WL86 - WL96A)

Independent reviewer report, batch WL97 "ReST-GI Historical Evidence Reliability Audit", track D.
Date 2026-11-16. Repository `c:/Projects/ReST-GI`, branch `Selective_Recompute`, HEAD `efcf86a`.
This track is static inspection and arithmetic on recorded artifacts. **No GPU timing was re-run by this track** (a separate track re-times);
every "ms" below is a number recorded by the original batches, recombined by me, unless it is marked *audit measurement*.
The run was resumed after an interruption of a first attempt of the same track; its scripts / outputs were re-checked (section 7.3) and kept.

Legend: **[REC]** recorded value read from a result JSON, **[ARITH]** my arithmetic on recorded values, **[CODE]** read from frozen code,
**[AUDIT-MEAS]** a measurement of my own (CPU-only or numpy), **[INF]** inference.

---

## 1. Scope and method

Claims audited (every performance / cost statement that supports a negative conclusion):

| id | claim (source) | raw artifact | code |
|---|---|---|---|
| C1 | WL86 economics: 9x per-frame GPU, 3-6x edit-to-first-frame wall vs path tracing; "ECRO per edit 76-78 ms, operator 11 ms" | `results/ecro86/results/econ/econ86.json` | `experiments/ecro86/econ86.py`, `ecro86.py` |
| C2 | WL87: "learned edge only source-only update cost"; ECRO 13-18 ms vs EXP 51-64 ms; q table 42-54 ms wall vs 2 ms GPU | `results/gm1_explicit87/results/econ/econ87.json` | `experiments/gm1_explicit87/econ87.py`, `quad87.py`, `xport87.py` |
| C3 | WL91 lifecycle: C3URX1 0.57-0.90x C3UR | `results/gm1_c3urx1_91/results/econ/life91.json` | `experiments/gm1_c3urx1_91/e2e91.py`, `cand91.py` |
| C4 | WL91 bounded scaling (N 1,270 / 2,802 / 5,080) | `.../econ/scale91.json` | `e2e91.py scale` |
| C5 | WL96 cost critic: Amdahl ceilings 1.09x edit->frame, 1.16x source->frame; "generator ~11 ms => per-edit ratio <= 1, a guaranteed loss" | `docs/plans/wl96_spatio_angular_carrier/critic_cost.md` section 2 (derived from life91.json, WL86 / WL87) | - |
| C6 | WL96 cost critic: DQ / CIS-ER pair cache, "60 % dirty pairs" for a 0.3 m box shift => ~60 ms => 136-140 ms per edit (1.8x ECRO) | `results/wl96_calc/scratch/critic_cost/dirty.py` | same |
| C7 | WL96A Stage 3: R0 144.1 / 88.5 ms, tail orders 3.3 ms, U(ZERO) 1.04x => COST criterion (U >= 1.25) not met | `results/ecro_util96a/cost_s1.0.json` | `experiments/ecro_util96a/w96a.py:488-685` |
| C8 | WL96A Stage 3: U(LAM, edit) 0.92-0.93 ("the generator makes edits slower") | same | same, `experiments/ecro86/model86.py:49-58` |
| C9 | WL96A worklog line 194: "WL91's GM1 scaling shows the tail orders growing slower than the q table and relation build ... so no scaling hypothesis reaches 1.25x" | `scale91.json` | - |
| C10 | timing-only geometry edits (WL96A box A +-0.15 m; WL96 critic 0.3 m shift) | `scene_ck2.json`, `w96a.py:562` | `scene86.py`, `ecro86.py:70-78` |

Method per claim: protocol text -> timed code path (what is inside each timer, synchronisation, compilation handling) -> raw JSON ->
recomputation of every ratio -> sensitivity to the convention -> inference check. Server-side hashes of the four timing JSON files were
compared with the local copies (identical, section 7.1).

---

## 2. What the timers contain (verified from code)

### 2.1 WL96A Stage 3 (`w96a.py:521-685`) [CODE]

- **Synchronisation.** Every component boundary calls `dr.sync_thread()` **and** `torch.cuda.synchronize()` (`sync()`, l.533-534; `Clock.part`, l.551-553).
  Parts are contiguous differences of one `perf_counter`; `parts_sum_over_wall` = 1.000 in all 8 (arm, lifecycle) cells [REC]. No un-synchronised gap.
- **Compilation.** Dr.Jit kernel history is drained at every boundary; `backend_time` of cache-miss kernels is subtracted (l.497-505, 547-549).
  All timed rows: `events_with_compilation` = 0, `wall_raw_s` = `wall_s` [REC]. Two untimed warm events of each kind precede (l.633-636). Verified.
- **Inside an edit event** (R0): `ctx.set_geom` (Mitsuba vertex update + scene parameter update + patch tables), `EC.update_state` (dirty-ray selection in
  numpy, re-trace of the dirty rays, host arrays copied), `X.XOp(...)` (numpy int64 conversion + host->device upload of the N x K relation),
  `Q.qtable` (Dr.Jit, **re-traced in Python at every call**, result downloaded to float64 numpy), upload of q, 4 torch orders, table hand-over
  (torch -> Dr.Jit), one 512^2 x 64 spp `'ecro'` table-lookup frame (`plain=True`).
  Host<->device transfers and numpy conversions **are inside** the boundary (q download / upload, state upload).
- **Inside a source event:** q table + orders + hand-over + frame; the relation operator stays resident.
- **Arms** are measured **sequentially** (R0, LAM24, LAM64, ZERO; l.629), not interleaved; n = 6 edits, 16 sources; only medians are stored (no per-event rows, no dispersion).
- **LAM arm:** `Lambda` = two random rank-rho tensors over D = N x NB (l.566-573); the edit event additionally builds `M86.GeoState` and calls the WL92A
  OWN checkpoint's `model.factors(gs)` whose output is discarded (l.603-608). ZERO: `B = L2` only.
- **Memory / rays.** `torch.cuda.max_memory_allocated` per arm (peak, reset per arm); `process_gpu_mem` is one `nvidia-smi` reading at the end (after `del p`,
  `empty_cache`), i.e. not a peak. `render_rays_nominal_per_frame = res^2 * spp * 4` is a nominal count, not a traced-ray counter. Dirty rays 49,936 of 650,240
  reproduce exactly in my independent state check (section 5).

### 2.2 WL91 (`e2e91.py`, `cand91.py`) [CODE]

Same clock design (WL96A copied it): synchronised parts, compilation excluded and reported raw; 7 edits, 15 single-light + **1** two-light source events,
4 frames per arm; per-event rows are stored (edit wall SD 1.0-2.5 ms [ARITH]).

### 2.3 WL86 / WL87 (`econ86.py:33-38`, `econ87.py:38-44`) [CODE]

`timed()` synchronises before and after and sums Dr.Jit `execution_time` as "GPU ms" (Dr.Jit kernels only: torch work has no GPU-ms figure).
**Compilation is not excluded** in these two scripts (the WL87 worklog itself flags TF16 3.4 s as a first-call compilation defect).
Per-component timers are separate calls (not one lifecycle clock).

- `EC.respond` (`ecro86.py:82-87`) = `GeoState` construction (CPU dtype conversion + upload) + `model.factors` + `model.apply` + **download to numpy**.
  The WL86 / WL87 "operator / inference 10.5-11.8 ms" is this whole call.
- WL87 source-only update (`econ87.py:136-141`): **one un-repeated call per edit** with the two-light set LMIX. The explicit arm's `exp_apply`
  **re-constructs `X.XOp`** (a geometry-only object that a source update does not invalidate) and downloads the table to numpy;
  the learned arm reuses resident factors but **re-constructs `GeoState`**.

### 2.4 CPU / GPU work distribution [REC + ARITH] (`results/audit97/D/addendum.json`, `ceilings.json`)

| lifecycle (R0, WL96A) | wall | Dr.Jit kernel ms | torch orders (sync wall) | remainder (Python tracing, numpy, host transfers, scene update) | process CPU / wall |
|---|---|---|---|---|---|
| edit -> frame | 144.1 ms | 25.5 (17.7 %) | 5.3 (3.7 %) | **113.3 ms (78.6 %)** | 1.03 |
| source -> frame | 88.5 ms | 24.3 (27.4 %) | 5.4 (6.1 %) | **58.9 ms (66.5 %)** | 1.02 |

Component level (WL87 E1-E3, one light) [REC]: q table 25.9-29.6 ms wall with 1.94-2.00 ms kernel (kernel share **6.7-7.7 %**);
selective state update 19.2-21.4 ms wall with 0.06-0.09 ms kernel (**0.3-0.4 %**); source encoding 5.1-5.4 ms with 0.02-0.08 ms kernel;
table frame: wall 35.5 / 39.9 / 55.3 / 223.8 ms at 4 / 16 / 64 / 256 spp with ~29 ms of Python tracing constant up to 64 spp
(kernel 1.5 / 5.7 / 21.3 / 85.1 ms).
The GPU utilisation trace of the WL96A cost stage is 18-26 % (`results/ecro_util96a/gpu.txt`, 17:08:43-17:08:54), consistent.

**Reading.** Every lifecycle is bound by one saturated Python thread. At N = 1,270 the wall numbers describe the prototype's driver overhead,
not transport work. This is the central systems fact behind findings D-03 / D-04 / D-09.

---

## 3. Findings

Each finding: id, category, material, claim affected, original vs audit number, evidence, decision-level impact, proposed verdict, confidence.

### D-01 - WL96A Stage 3 numbers and timing boundary reproduce (confirmation; not material)

- Claim: C7. U recomputed from `cost_s1.0.json`: ZERO 1.0380 / 1.0429, LAM24 0.9321 / 1.0405, LAM64 0.9210 / 1.0383 (edit / source) = recorded.
  Analytic ceiling R0 / (R0 - tail) = 1.0235 / 1.0389; the measured ZERO arm is 2.0 ms (edit) / 0.3 ms (source) faster than "R0 - tail"
  (mostly a -1.6 ms render drift between sequential arms) -> U carries about +-0.015 of arm-to-arm drift [ARITH].
- Boundary: synchronised, contiguous, compilation excluded, 0 compile events, warm-ups present, same GPU, `foreign_before = []` [CODE, REC].
- Server copy `~/rest_x96a/results/cost_s1.0.json` sha256 `103b775ed571ed26` = local = worklog; server `w96a.py` = `git show a1718f7:` (afb80dd3200d24d4).
- Impact: the frozen decision "COST criterion (U >= 1.25) not met" is what the recorded data say. Margin 0.21 >> drift 0.015.
- Verdict for the measurement: **VERIFIED** (arithmetic and code; not re-timed by this track). Confidence high.

### D-02 - U(LAM, edit) = 0.92-0.93 is dominated by an implementation side effect (implementation_defect; material for that number, not for the frozen verdict)

- Claim: C8 (WL96A worklog B4; later summarised as "generator makes edits slower").
- Original: LAM24 / LAM64 edit 154.6 / 156.5 ms vs R0 144.1 ms (deficit 10.5 / 12.4 ms); U 0.932 / 0.921.
- Decomposition [REC + ARITH]: tail saved -3.31; Lambda apply +0.21 / +0.29; "generator" part +8.36 / +8.37 (isolated `model.factors` on a resident
  `GeoState`: 5.30 ms -> **3.07 ms is GeoState construction**); **q table part +5.57 / +7.02 ms although the q table code and inputs are identical**.
  The q inflation appears **only in LAM edit events** (31.3 / 32.7 ms); in LAM *source* events the q part is 24.39 / 24.34 ms = R0 24.37 ms.
  Process CPU per edit: 580 / 727 ms vs 149 ms (R0) and 144 ms (ZERO); cpu / wall 3.75 / 4.64 vs 1.03.
- Cause [CODE + AUDIT-MEAS + INF]: `model86.GeoState` (`model86.py:54-58`) calls `torch.as_tensor(np_array, dtype=torch.long, device=cuda)` on the
  int16 `hit` / uint8 `dbin` arrays (650,240 elements each): torch converts the dtype **on the CPU with its OpenMP pool (16 threads)** before the upload.
  The explicit arm converts with numpy (`xport87.py:54-56`, `hit.astype(np.int64)`) - no OpenMP region.
  CPU-only probe on the same server (`omp_cpu_probe.py`, no GPU): torch conversion event 605 ms CPU vs 75 ms with the numpy conversion, same wall
  (the worker team spin-waits) - reproduces the 580-727 ms signature. Probe 2 (main thread kept warm): the ~35 ms of single-threaded Python work
  that follows the conversion is 1.33x slower (+11 ms), later chunks unaffected (1.00x). Probe 1 (cold main thread) did **not** show the slowdown
  (-5 ms; frequency-scaling confound, governor `schedutil`). So: CPU-time anomaly reproduced (high confidence); the q table slowdown mechanism is
  plausible and of the right size but **not demonstrated on the actual pipeline** (medium confidence; the decisive check - the cost stage with
  `OMP_NUM_THREADS=1` or a numpy conversion - is a GPU timing run outside this track's remit).
- Audit number: removing the two artefacts (q inflation, GeoState construction) gives U(LAM24 / LAM64, edit) = **0.987 / 0.984** [ARITH]; break-even generator
  cost is 3.1 ms. That is inside the +-0.015 arm drift: "edit parity", not "8 % slower".
- Decision impact: none on the frozen COST criterion (needs >= 1.25; best case ~0.99-1.02). Material for any statement that a learned generator
  *costs* 7-8 % per edit, and for the WL96A worklog's reading "the generator path runs multi-threaded CPU work" (it is a dtype conversion, not the generator).
- Verdict for "U(LAM, edit) = 0.92-0.93": **IMPLEMENTATION_DEFECT**. Confidence: high that the number is artefact-dominated (CPU signature + q part
  only in edit events are recorded facts); medium on the exact mechanism.

### D-03 - "No lifecycle-cost headroom" is convention-bound: the same recorded numbers give ceilings from 1.02 to 2.6 (invalid_inference if generalised; material)

- Claim: C7 / P0-4 (cost half), C5.
- The removable work is 3 explicit orders = **3.3 ms** (1.1 ms per order) [REC]. Whether that is "headroom" depends entirely on the denominator.
  Zero-cost-resolvent ceilings U_ZERO from recorded numbers only [ARITH] (`ceilings.json` D_conventions, `addendum.json` edit_conventions):

| convention (all at N = 1,270, one light, J = 4) | fixed part | U_ZERO | >= 1.25? |
|---|---|---|---|
| **frozen**: edit -> frame, wall, 512^2 x 64 spp | 138.7 ms | 1.02 (measured arm 1.04) | no |
| **frozen**: source -> frame, wall, 64 spp | 83.1 ms | 1.04 | no |
| source -> frame, wall, 16 / 4 spp (WL87 E2 table-frame ladder) | 64.5 / 60.1 ms | 1.05 / 1.05 | no |
| source -> frame, wall, 256 spp | 248.4 ms | 1.01 | no |
| source -> table, wall, no frame | 24.6 ms | 1.12 | no |
| edit -> table, wall, no frame | 79.2 ms | 1.04 | no |
| source -> frame, Dr.Jit kernel ms for q and frame, 64 spp | 23.5 ms | 1.13 | no |
| source -> frame, kernel ms, 4 spp | 3.7 ms | **1.57** | yes |
| source -> table, kernel ms, no frame | 2.2 ms | **1.78** | yes |
| transport operator only (prefix + tail) | 0 | **2.61** | yes |
| edit -> frame 64 spp, kernel ms for q / state / frame, scene update 30.2 ms kept at wall | 58.7 ms | 1.05 | no |
| edit -> table, kernel ms, scene update kept at wall | 37.2 ms | 1.08 | no |
| edit -> table, kernel ms, scene update excluded | 7.0 ms | **1.37** | yes |

  (torch orders and the 5 ms relation-coefficient upload have no kernel timer and are kept at synchronised wall in the "kernel" rows.)
- Sensitivities [ARITH]: **orders** (R0 wall): J = 8 / 16 / 32 -> source->frame 1.09 / 1.20 / 1.41, edit->frame 1.06 / 1.12 / 1.25.
  **Lights** (WL91 R0; one two-light event only): q 25.1 -> 40.6 ms, frame 54.4 -> 97.1 ms, sweeps 5.6 -> 5.7 ms; ceiling 1.05 -> 1.03 (the summed field makes
  the sweeps light-independent, so more lights *lower* the ceiling in this design). **Resolution**: not measured (512^2 only); a higher resolution only enlarges the frame term.
- Answer to brief question (1): on this prototype the ceiling is set by (a) the 64-spp frame convention (58 ms wall, 21 ms kernel), (b) the q table's
  Python overhead (24 ms wall, 2 ms kernel) and (c) the 30 ms scene update. It reaches >= 1.25 only when the frame is dropped or reduced to <= 4 spp
  **and** driver overhead is excluded (and, for edits, the scene update too). With the frozen frame at kernel time it is still 1.13.
  So "no headroom" is **both** a property of the tested design at this size (the explicit tail is 3 x 1.1 ms; the learned part replaces only that) **and** a
  property of the overhead profile (the ratio's value). It is not a property of "the transport problem" in general.
- Decision impact: the frozen verdict stands under its frozen convention. Statements that the learned higher-order part is "decorative" *in general*
  (WL96 critic section 2.3) rest on one convention at one size.
- Verdict: P0-4 cost half **CONDITIONALLY_VALID** (CK2, N 1,270, one point light, J = 4, 512^2 x 64 spp plain table frame, wall incl. Python overhead,
  resolvent-only design with the explicit q prefix kept). Confidence high (arithmetic), medium on cross-batch combinations (WL87 scene86 kernel ms
  used with WL96A CK2 walls; same patch family and N, same GPU).

### D-04 - The WL96A / WL96 ceilings apply to the *resolvent-only* design, not to a full learned operator (invalid_inference risk; material for scoping)

- Claim: P0-4 / P0-5 scoping.
- In WL96A the LAM / ZERO arms keep the explicit q table (24 ms wall) and one explicit order; the learned part replaces orders >= 3 only [CODE l.582-593].
  A full learned operator of the ECRO form (source encoding + apply; factors resident) replaces q **and** all orders on a source update.
  Recorded numbers [REC]: WL87 source encoding 5.1-5.4 ms wall (0.02-0.08 ms kernel), WL96A isolated ECRO apply 0.77 ms, vs explicit q 24.4 + 4 orders 5.4 ms.
  [ARITH] source -> frame at the frozen convention: 88.5 / (58.5 + 0.2 + 5.1 + 0.8) = **1.37** (with WL87's recorded application incl. the GeoState rebuild,
  5.4-7.4 ms: 1.24-1.28); work-only (kernel / resident) source -> table: 6.5 ms explicit vs 0.8 ms learned (ratio 0.12).
- This is the WL87 finding ("learned edge only source-only update cost") restated on WL96A numbers: it survives removing implementation overhead from
  both arms (see D-05). It says nothing about quality (ECRO's quality deficit is outside this track).
- Decision impact: P0-4 must not be read as "a learned operator has no lifecycle-cost headroom"; it is "a learned *higher-order resolvent behind an explicit
  first-scatter prefix* has none on CK2". Verdict for the broader reading: **UNTESTED_INFERENCE**. Confidence high.

### D-05 - WL87 source-only comparison: both arms carry avoidable overhead; direction robust, magnitude not (evaluation_defect + statistical_weakness; material for magnitude only)

- Claim: C2. Original [REC]: ECRO 13.2 / 13.5 / 18.3 ms vs EXP 51.7 / 53.0 / 64.4 ms (ratio 0.25-0.28); q 41.9-53.6 ms.
- Defects [CODE]: (i) n = 1 un-repeated call per edit, two-light LMIX set (so "q 42-54 ms" is a two-light number; one light is 25.9-29.6 ms);
  (ii) the explicit arm rebuilds `XOp` and downloads the table inside `exp_apply` (9.8-10.8 ms vs 4 x 1.1 = 4.5 ms for resident orders in WL96A);
  (iii) the learned arm rebuilds `GeoState` inside `ecro_application` (5.4-7.4 ms vs 0.77 ms isolated apply), and its source encoding is a CPU `np.bincount`;
  (iv) q is 92-93 % Python tracing; (v) compilation not excluded in this script.
- Audit numbers [ARITH]: overhead-free (kernel / resident) one light: explicit 2.0 + 4.5 = 6.5 ms vs learned 0.02 + 0.77 = 0.8 ms (ratio 0.12).
  The learned advantage is **not** an artefact of the q table's wall overhead (as WL96 critic section 2.1 "q 42-54 ms wall, 2 ms GPU" suggests): it grows in
  ratio and shrinks in absolute terms (<= ~6-8 ms) when overhead is removed.
- Decision impact: "learned edge only on source-only updates" holds in direction; the 3.7-4x wall ratio is implementation-dependent, and against a 58 ms
  frame the absolute saving is small. The WL87 worklog's own unresolved-question note already concedes the dependence on the MIS budget.
- Verdict: **CONDITIONALLY_VALID**. Confidence medium-high.

### D-06 - WL96 critic Amdahl table: 1.09 reproduces; 1.16 / 1.98 mix edit-event sweeps into the source lifecycle; "generator ~11 ms => guaranteed loss" compares unlike quantities (evaluation_defect; minor numerically, material for the wording)

- Claim: C5 (`critic_cost.md:123-134`).
- [ARITH from `life91.json`, C3URX1]: edit -> frame 124.66 / (124.66 - 10.57) = **1.093** (ok); edit without render 1.179 (ok).
  Source -> frame: the critic divides the *source* lifecycle (75.81 ms) by "sweeps 10.6 ms", which is the **edit** event's sweep time; the source event's own
  sweeps are 9.40 ms -> **1.142** (not 1.16); source without render **1.79** (not 1.98).
- "Generator of about 11 ms per edit ... the per-edit ratio is <= 1, a guaranteed loss": the 11 ms is `EC.respond` in WL86 (GeoState CPU conversion +
  factors + apply + download, section 2.3). WL96A's own isolated generator is 5.30 ms. With 5.3 ms against the C3UR-class sweeps the critic used
  (10.6 ms): 124.66 / (124.66 - 10.57 + 5.30) = **1.044**; keeping one of four orders: 1.02; with 11 ms: 0.997 [ARITH]. Not a "guaranteed loss": parity within noise.
- The critic's anchor table mixes scene86 (WL86 / WL87) and GM1 (WL91) numbers but labels the regimes; same GPU in all four JSON files [REC].
- Decision impact: none on a 1.25 threshold; the qualitative "<= 1, guaranteed loss" and "decorative" wording is stronger than the numbers.
- Verdict: ceilings **CONDITIONALLY_VALID** (convention as in D-03); the generator sub-claim **INVALID_COMPARISON**. Confidence high.

### D-07 - WL96 critic "60 % dirty pairs => ~60 ms => 136-140 ms per edit" counts pairs that need no re-test and assumes a linear ms-per-ray rate the data contradict (evaluation_defect; material for the DQ / CIS-ER / SRR-LB budget predictions)

- Claim: C6 (`critic_cost.md:64-80`; WL96 worklog l.114, l.153).
- Re-run of the critic's script unchanged: 0.6045 (reproduces) [AUDIT-MEAS, `dirty_rerun.txt`].
- Defect 1 (counted quantity): the script counts front-facing S1-cell x receiver pairs whose segment crosses box A at the old **or** the new pose (union).
  A pair that crosses box A at **both** poses stays blocked whatever the other occluders are: its cached visibility cannot change.
  Independent recomputation, same cells / points / OBB test (`dirty_xor.py`) [AUDIT-MEAS]:

| -0.3 m shift (critic's edit) | pairs | share of the 9.09 M front-facing pairs |
|---|---|---|
| union (critic's "dirty"), static receivers | 5,391,278 | 59.3 % (critic 60 %) |
| blocked at both poses (no re-test needed) | 3,586,239 | 39.4 % |
| visibility w.r.t. box A can change (XOR) | 1,805,039 | **19.9 %** |
| XOR + pairs with a receiver on the moving box | 1,968,623 | **21.6 %** |
| rays strictly needed (blocked before, not crossing the new pose; + moved receivers) | 873,325 | 9.6 % |

  (+0.15 m, the WL96A edit: union 47.9 %, XOR 8.6 %, XOR + moved 10.4 %.) The corrected count (0.9-2.0 M) lies **inside** designer d2's "<= about 1-2 M rays"
  estimate, which the critic called "badly underestimated".
- Defect 2 (rate): "1.05 M rays plus ER evaluation in 11.5 ms => 5.5 M re-tests ~ 60 ms" assumes time proportional to rays. WL91 `scale91.json` shows the
  exact-source evaluation flat at 11.1-14.0 ms from 0.26 M to 2.1 M landed rays (linear fit: 12.1 ms intercept, 0.22 ns per ray, r = 0.13) [REC + ARITH]:
  the 11.5 ms is launch overhead, not a per-ray rate.
- Audit numbers at the critic's own rate [ARITH]: 21.6 ms (XOR + moved) or 9.6 ms (strictly needed) instead of 60.8 ms -> per-edit total 1.28x or 1.12x
  ECRO's 77 ms instead of 1.79x. At the flat measured rate the increment is smaller still (extrapolation beyond 2.1 M rays, weak).
- Decision impact: the predicted failures of d4 FA5 (<= ECRO + 25 %) and d3 FC / d2 F2e on the designers' favourable workload are **not supported** as stated
  (FA5 becomes borderline-to-pass). These are budgets of *explicit* carriers (comparator selection), so the effect on the learned-operator verdict of WL96 is
  indirect; it removes one of the listed reasons (WL96 worklog l.153).
- Note: the 47 % "blocked at the CK2 pose" figure is specific to CK2, where box A stands next to the panel (worst case by construction); the actual
  512-ray relation state has 7.7-8.1 % dirty rays for the same edits (section 5).
- Verdict: **INVALID_COMPARISON** (the counted set is not the set the cost claim needs) with an unsupported rate assumption. Confidence high for the counts,
  medium for the ms conversion.

### D-08 - WL91 lifecycle ratios reproduce; they are explicit-vs-explicit and overhead-bound (confirmation + dataset_limitation; not material to the learned-operator chain)

- Claim: C3. [ARITH from `life91.json`] C3URX1 / C3UR: edit 0.902, source 0.822, edit without render 0.837, source without render 0.572 = "0.57-0.90x".
  Boundary as in 2.2; kernel share 17-28 % of edit / source walls; one two-light event only.
- Verdict: **VERIFIED** (arithmetic and boundary; not re-timed) as a prototype measurement; **CONDITIONALLY_VALID** as a statement about the methods.

### D-09 - N-scaling statement in WL96A (l.194) is not what scale91.json shows; the ceiling rises with N (invalid_inference; material for the "no scaling hypothesis" sentence only)

- Claim: C9, and brief question (5).
- `scale91.json` (kappa_s 8; N 1,270 -> 5,080) [REC]: q table 24.9 -> 41.5 ms (**1.67x**), one order 2.67 -> 4.54 ms (**1.70x**), C3UR J4 table 11.2 -> 18.9 ms (1.70x),
  full relation build 52 -> 229 ms (4.4x). The tail orders do **not** grow slower than the q table (they grow marginally faster); they do grow slower
  than a *full* relation build, which no lifecycle event performs.
- All components grow 1.7x for 4x N: a linear fit puts **72-77 % of each N = 1,270 timing in an N-independent intercept** (q 18.8 of 24.9 ms; one order
  2.06 of 2.67 ms) [ARITH]. The N = 1,270 lifecycles are therefore in the overhead-dominated regime, and the measured ceiling **rises** with N:
  source -> frame, C3UR J4 sweeps, remove 3 of 4 orders: 1.10 / 1.12 / 1.14 at N 1,270 / 2,802 / 5,080 (same values from q + 4 x `apply_one_order_s`; scale91 has no R0-class order timing, whose order is ~2.4x cheaper at N 1,270);
  without the frame 1.30-1.31 at every N; per-patch slopes alone give an asymptote of 1.30-1.32 (source -> table); a linear extrapolation with a constant
  54 ms frame crosses 1.25 near N ~ 4 x 10^4 (three points, 8x beyond the measured range: weak) [ARITH].
- Explicit order cost and a rank-rho Lambda apply are both linear in N at fixed K and rho, so N by itself does not change the structural ratio; what can
  change it is the number of orders (D-03), K / bins for sharper materials, per-light tables, and the overhead profile. None of these was measured.
- Decision impact: "runtime competitiveness absent at N = 1,270" is supported; "no scaling hypothesis reaches 1.25x" is not supported by scale91
  (nor refuted): **UNTESTED_INFERENCE**. The frozen scaling-extension rule (U_ZERO >= 1.15) was correctly not triggered. Confidence high for the growth
  ratios, low for any extrapolation.

### D-10 - Fairness of the LAM arm as a model of the proposed design's cost (evaluation_defect; not material to the verdict)

- Brief question (3). [CODE]
  - Random rank-rho factors: fair for the apply cost (dense einsum over D = 81,280 x rho x 3; 11.7 M / 31.2 M MAC vs 124.8 M per explicit order; measured 0.16 / 0.24 ms).
  - Generator = OWN `model.factors`: it generates ECRO's *own* factors (R 16 local channels + rank-24 global), not rank-24 / 64 Lambda factors over D; output
    discarded. A plausible stand-in of unknown bias; it carries the D-02 artefact; and it is a **full** regeneration on every edit, while the explicit arm
    gets a **selective** (7.7 % dirty-ray) update - an asymmetry the protocol does not discuss (an incremental or amortised generator is not represented).
  - Prefix stand-in = one binned order (2.0 ms): cheaper than a real sharp prefix, i.e. conservative in the direction that favours the learned arm, as the protocol states.
  - Source events: LAM = ZERO within 0.4 ms (1.038-1.043): the source-side numbers are a fair ceiling.
- Verdict: LAM is a fair *upper-bound proxy* for source events and for the Lambda apply; its edit number is not a fair cost estimate of the design (D-02).
  Frozen verdict unaffected. **CONDITIONALLY_VALID**. Confidence medium-high.

### D-11 - Statistical basis of the cost numbers (statistical_weakness; not material)

- WL96A: 6 edits / 16 sources, medians only, no per-event rows, sequential arms, one process, one day; ZERO vs (R0 - tail) drift up to 2 ms.
  WL87 source-only: n = 1. WL91: one two-light event. WL86 / WL87: compilation not excluded.
- All decisive margins (1.04 vs 1.25) are an order of magnitude larger than the observed drift (1-2 %). Not material to the frozen decisions; material
  only where a difference of a few ms is interpreted (D-02, D-06).

### D-12 - WL86 economics reproduce as arithmetic; they are ECRO-vs-path-tracing, descriptive (confirmation)

- Claim: C1. [ARITH from `econ86.json`] resident frame GPU speedup 8.8-9.6x; edit-to-first-frame wall 3.07-3.93x at 0.05 and 4.80-5.58x at 0.03; ECRO fixed
  per-edit wall 75.7 / 78.3 ms, of which `respond` 11.3 / 11.8 ms, selective state 20.8 / 21.4 ms wall with 0.06 ms kernel. Selective = full state in all edits [REC].
- The comparison is against path tracing, not against explicit same-state transport, and g was not reached; WL86 itself labels economics NOT TESTED / descriptive.
- Verdict: **CONDITIONALLY_VALID** (descriptive, prototype walls). Not a negative-chain claim.

---

## 4. Answers to the brief's specific questions

1. **Problem property or prototype property?** Both, separably (D-03, D-04, D-09). Recorded numbers give a zero-cost ceiling of 1.02-1.04 under the frozen
   convention, 1.13 with q and frame at GPU-kernel time, 1.57 with a 4-spp frame at kernel time, 1.78 source->table at kernel time, 2.6 for the transport
   operator alone; edit lifecycles reach 1.25 only if the 30 ms scene update is also excluded (1.37). 67-79 % of each frozen lifecycle is non-kernel time.
   In absolute terms the removable explicit work is 3.3 ms at N = 1,270.
2. **LAM edit q table 25.7 -> 31-33 ms, CPU 580-727 ms.** Explained by the torch CPU dtype conversion in `model86.GeoState` (OpenMP team spin), see D-02.
   U(LAM, edit) = 0.92-0.93 is **not trustworthy as a cost of the design**; corrected ~0.985 (parity). The COST-criterion outcome is unaffected.
3. **Is LAM a fair cost model?** Fair for the apply and for source events; not for the edit generator (stand-in generator of a different operator, full
   regeneration vs selective explicit update, artefact-laden). See D-10.
4. **Timing-only edits.** Geometrically valid and the cached relation state is exactly correct after them (section 5), but **no T1 reference exists** for the
   +-0.15 m (WL96A) or 0.3 m (WL96 critic) poses - the WL96A protocol says so explicitly. The cost numbers are quality-free. Implicit quality assumptions:
   U compares R0 with an arm *assumed* to match R0's image (ZERO renders only L2; LAM uses random factors); R0's own J = 4 accuracy at the edited pose is
   assumed from T0 (Stage 1 is static). The critic's DQ estimate assumes its quadrature quality at T1 without a reference.
5. **Transfer from N = 1,270 / one light.** Not established (D-09, D-03). The measurements sit in the overhead-dominated regime (72-77 % N-independent
   intercept); the ceiling rises with N within the measured range and falls with the number of lights in this summed-field design; J, K, bin count,
   area lights, per-light tables, non-rigid edits and resolution were not measured.

---

## 5. Special checks owned by this track

| check | result | evidence |
|---|---|---|
| WL96A cost ceilings recomputed | recorded U reproduce; analytic 1.0235 / 1.0389; frozen criterion not met with margin 0.21 | `ceilings.json` A_WL96A, `addendum.json` spot |
| WL96 critic ceilings recomputed | 1.093 ok; 1.16 -> 1.142 and 1.98 -> 1.79 with the source event's own sweeps; generator "guaranteed loss" not supported (1.02-1.04 with the isolated 5.3 ms) | `ceilings.json` B_WL96_critic_from_WL91 |
| Lifecycle timing boundaries (WL96A, WL91) | synchronised (Dr.Jit + torch), contiguous, compilation excluded and zero in timed rows, warm-ups present; host transfers and numpy conversions are inside the boundary | `w96a.py:533-559, 598-615`; `e2e91.py` |
| Lifecycle timing boundaries (WL86, WL87) | synchronised; compilation NOT excluded; component-wise timers; source-only n = 1 with XOp / GeoState rebuilds inside | `econ86.py:33-38`, `econ87.py:38-44, 136-141` |
| Timing-only edits: geometric validity | CK2 T0, +0.15 m and -0.3 m poses pass `scene86.geom_ok` and `light_ok`; box A intersects no stationary rectangle (clearance 0.17 / 0.19 / 0.14 m) | `edit_validity.json` [AUDIT-MEAS, numpy] |
| Timing-only edits: cache invalidation correctness | selective `update_state` = full `build_state` = state of a fresh scene at the new pose, record for record; move-and-return restores the T0 digest; dirty rays 49,936 (7.68 %) for +-0.15 m = the WL96A recorded median; 52,590 (8.09 %) for -0.3 m | `edit_state_check.json` [AUDIT-MEAS, server, frozen code a1718f7, correctness only, no timing] |
| Timing-only edits: physical reference at T1 | none exists (protocol states it); cost claims are quality-free | `ecro_util96a.json` stage_3.edit |
| Same hardware | all four timing JSON files: RTX 3080 Ti, driver 595.71.05, no foreign process | JSON `gpu`, `foreign_before` |
| Memory accounting | torch peaks 251 / 355 / 350 / 306 MiB consistent with the factor sizes (LAM64 P + Q = 119 MiB); process 1,452 MiB is an end-of-run reading, not a peak | `cost_s1.0.json` |
| Ray counts | relation 650,240 (N x 512), q shadow rays 20,320 per light (plus 20.8 M ER evaluations not counted as rays), frame count nominal | `cost_s1.0.json` rays |

---

## 6. Priority-0 assessment within this track's lens

| P0 claim | assessment (cost / systems lens only) | proposed verdict |
|---|---|---|
| P0-1 learned S1 sender error / passivity | no cost content; not assessed | - |
| P0-2 M64 sender bias | no cost content; not assessed | - |
| P0-3 exact-direction ER evaluation closes the CK2 delivery error | cost side only: WL96A times a *binned* order as the prefix stand-in (2 ms); the cost of the exact prefix on CK2 was not timed there (WL91 X1: ~11.5 ms source evaluation, flat in ray count, GM1 scene). No cost claim depends on it except "only lowers U". | not assessed (accuracy claim) |
| P0-4 no meaningful accuracy or lifecycle-cost headroom for the learned broad higher-order resolvent on CK2 | **cost half**: measurement verified (D-01); holds under the frozen convention with a large margin; convention-bound (D-03); the LAM edit number is artefact-laden but irrelevant to the outcome (D-02); applies to the resolvent-only design, not to a full learned operator (D-04); not shown to hold at other N / J / lights (D-09). | **CONDITIONALLY_VALID** |
| P0-5 negatives justify rejecting the tested ECRO mechanisms but not a general impossibility | cost lens: consistent with the second half - every cost conclusion is a prototype / workload statement; several cost sentences over-reach ("guaranteed loss", "decorative", "no scaling hypothesis reaches 1.25x", "60 % dirty pairs") and must not be carried as general facts. The first half (rejection of the tested resolvent on cost grounds, on CK2, under the frozen rule) is supported. | **CONDITIONALLY_VALID** (general readings: UNTESTED_INFERENCE) |

Classification of each cost claim: C1 CONDITIONALLY_VALID; C2 CONDITIONALLY_VALID; C3 VERIFIED (prototype measurement); C4 CONDITIONALLY_VALID
(overhead-dominated regime); C5 ceilings CONDITIONALLY_VALID, generator sub-claim INVALID_COMPARISON; C6 INVALID_COMPARISON; C7 VERIFIED as a measurement,
CONDITIONALLY_VALID as "no headroom"; C8 IMPLEMENTATION_DEFECT; C9 UNTESTED_INFERENCE; C10 CONDITIONALLY_VALID (valid timing edits, no quality reference).

---

## 7. Not verified, and why

- **No GPU re-timing** (by assignment). Every wall / kernel number is the originally recorded one. In particular the D-02 mechanism was not confirmed on the
  real cost stage (needs one ~20 s GPU run with `OMP_NUM_THREADS=1`; it could change only the *explanation* of the LAM edit deficit, not the COST outcome).
- The Dr.Jit `KernelHistory` flag is on during all WL91 / WL96A timings; whether it adds per-launch overhead was not measured (equal for all arms) [INF].
- What the 30 ms `geometry_update` consists of (Mitsuba / OptiX acceleration update vs Python table rebuild) was not decomposed; no kernel figure exists for it.
- `econ88` / `econ89` lifecycle scripts, WL91 `cold` / `ladder` outputs, and the WL86 / WL87 time-to-quality ladders were not audited beyond the numbers above.
- Other WL96 critic / designer cost figures (2.6 GB of generated factors, MAC counts, alpha^-4 pair scaling, 13.6 M pairs for a rotating slab) were not recomputed.
- Cross-batch combinations (WL87 scene86 kernel ms with WL96A CK2 walls) assume comparable per-component cost on the two scenes (same patch family,
  N and GPU); not measured jointly.
- Whether the WL91 lifecycle edits (W1 T0 <-> T1) have quality references was not re-checked (they are the batch's approved workload).
- Resolution sensitivity: no recorded number other than 512^2 exists.
- I did not open any other track's report or directory.

### 7.3 Status of the interrupted first attempt's material

Re-used after checks: `ceilings.py` / `ceilings.json` (12 values recomputed independently in `addendum.py`: all agree to 4 digits), `edit_validity`,
`edit_state_check` (dirty-ray count equals the WL96A recorded median), `omp_cpu_probe*` (JSON re-read; conclusions drawn here are mine), `dirty_rerun.txt`.
No `*_buggy_*` file and no unfinished server job existed (`~/rest_x97/D/` holds only the finished probe and state-check outputs; no process of this track running).

---

## 8. Reproducibility appendix

Local (Python 3.11, numpy only), from the repository root:

```
python experiments/audit97/D/ceilings.py            # -> results/audit97/D/ceilings.json   (all ratios / conventions / fits from the 5 recorded JSON files)
python experiments/audit97/D/show.py [section ...]  # compact print of ceilings.json
python experiments/audit97/D/edit_validity.py       # -> edit_validity.json   (geometry checks of the timing-only edits)
python results/wl96_calc/scratch/critic_cost/dirty.py > results/audit97/D/dirty_rerun.txt   # critic's script, unchanged, read-only
python experiments/audit97/D/dirty_xor.py           # -> dirty_xor.json       (union / both / XOR dirty-pair counts)
python experiments/audit97/D/addendum.py            # -> addendum.json        (spot-checks, LAM decomposition, edit conventions, scaling, DQ ms)
```

Server (LabServer63, `~/rest_x97/D/`; frozen code read-only from `~/rest_x96a/code`, commit a1718f7):

```
CUDA_VISIBLE_DEVICES= $PY omp_cpu_probe.py omp_cpu_probe.json      # CPU only, seed 97004
CUDA_VISIBLE_DEVICES= $PY omp_cpu_probe2.py omp_cpu_probe2.json    # CPU only
PYTHONDONTWRITEBYTECODE=1 $PY edit_state_check.py ~/rest_x97/D/run edit_state_check.json   # correctness only, < 1 min, ~0.6 GB GPU, no timing
```

sha256 prefixes (16 hex):

| file | sha256 |
|---|---|
| results/ecro_util96a/cost_s1.0.json (= server ~/rest_x96a/results) | 103b775ed571ed26 |
| results/gm1_c3urx1_91/results/econ/life91.json (= server ~/rest_x191) | 2f3fc40b5afb28cd |
| results/gm1_c3urx1_91/results/econ/scale91.json (= server) | e158b848a563ddb9 |
| results/gm1_explicit87/results/econ/econ87.json (= server ~/rest_gm187) | db9d932ff8ee90dc |
| results/ecro86/results/econ/econ86.json (= server ~/rest_ecro86) | b543a6b1146e8306 |
| experiments/ecro_util96a/w96a.py (= a1718f7 = server) | afb80dd3200d24d4 |
| experiments/gm1_c3urx1_91/e2e91.py | 31388871a835732b |
| experiments/gm1_explicit87/econ87.py | 1149e23b49026784 |
| experiments/gm1_explicit87/quad87.py | cfa9c5c4e854526c |
| experiments/ecro86/econ86.py | 5ba5b7c240ca03ac |
| experiments/ecro86/ecro86.py | 0ef09a86939745ed |
| experiments/ecro86/model86.py | 72a728b81e490fb7 |
| results/wl96_calc/scratch/critic_cost/dirty.py | 101af5646fb28b7a |
| docs/plans/wl96_spatio_angular_carrier/critic_cost.md | dfdfec5158c7bf61 |
| results/audit97/D/ceilings.json | d20dad759dbe29ac |
| results/audit97/D/addendum.json | 259474a5fa0a58b6 |
| results/audit97/D/dirty_xor.json | 559c8741dbb7b084 |
| results/audit97/D/edit_state_check.json | 2cf9aeb37e0aa84a |
| results/audit97/D/edit_validity.json | e8f419a93a005f61 |
| results/audit97/D/omp_cpu_probe.json / omp_cpu_probe2.json | 82a897ab9f1e57cb / 1e0b66ea0c7c67c8 |
| experiments/audit97/D/ceilings.py / addendum.py / dirty_xor.py | 3a079e81ea443549 / 98fbcd5702d5164f / eb76507bec4a4485 |

Key evidence lines: `experiments/ecro_util96a/w96a.py:533-559` (clock), `:562` (timing-only edit), `:566-573` (random Lambda), `:582-593` (table),
`:598-611` (edit event, generator), `:629-638` (arm order, warm-ups), `:677-681` (U); `experiments/ecro86/model86.py:54-58` (torch CPU dtype conversion);
`experiments/gm1_explicit87/xport87.py:54-56` (numpy conversion); `experiments/ecro86/ecro86.py:70-87` (selective update, respond);
`experiments/gm1_explicit87/econ87.py:38-44, 95-106, 136-141`; `experiments/gm1_explicit87/quad87.py:268-338` (q table, re-traced per call);
`docs/plans/wl96_spatio_angular_carrier/critic_cost.md:64-80, 113-134`; `docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md:121-164, 194`.
