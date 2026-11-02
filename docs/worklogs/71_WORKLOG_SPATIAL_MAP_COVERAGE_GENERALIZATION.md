# Worklog — Spatial-Map Coverage and Cross-Source Generalization, WL68 Follow-up (2026-11-02)

## Authorisation and scope

On 2026-11-02 the user issued a pasted brief: one bounded follow-up to WL68, on the cross-source and deformation generalization of the existing geometry-conditioned spatial map of the WL59 P intrinsic-K family.

**Central question.** Does the unchanged WL59 geometry-conditioned spatial map generalize physically valid collided transport when source coverage is controlled? Or does the failure persist in a way that justifies reconsidering the mapping architecture?

**The batch has four stages:**

1. Spatial-map identifiability diagnosis.
2. A cross-source synthetic teacher control.
3. A controlled physical coverage experiment with these arms:
   - A: frozen WL59;
   - B: WL68 fitted;
   - C: sparse coverage;
   - D: expanded coverage;
   - E: coefficient oracle.
4. An independent physical evaluation on previously unscored ports and replicas.

**The batch may change only:**

- training populations;
- supervision-coverage conditions;
- diagnostic fitting and evaluation infrastructure;
- synthetic teacher-generalization controls;
- necessary qualified references.

**The batch preserves:**

- the WL59 family and its parameterisation;
- positivity and passivity;
- A/O and spatial semantics;
- geometry features;
- K/C ownership;
- the ballistic/collided split;
- normalisation;
- historical checkpoints;
- the original eligibility thresholds.

**Not authorised:**

- rank, width or depth changes;
- feature or architecture changes;
- threshold changes or production-loss changes;
- tuning historical checkpoints;
- a new learned mechanism or spatial heuristic;
- hyperparameter sweeps;
- production K/C changes;
- coupled GI;
- a successor.

**Coordination.** A concurrent session (rest-gi-d2, WL69 follow-up) claimed Worklog 70, LabServer69 and `experiments/sunreuse70`. By agreement this batch is **Worklog 71**. It runs on LabServer63 under `~/rest_cover71`, with code in `experiments/fnto_cover71/` and results in `results/fnto_cover71/`. It does not use LabServer69's GPU or touch the other batch's paths.

- **Starting HEAD:** 7a646f4.
- **Population status:** the WL66/WL68 populations (WL66 slots 0–3, WL68 slots 4–7, cf3) and every model scored on them are development evidence here, never fresh confirmation.

## Stage 1: identifiability diagnosis (development evidence, before the protocol)

`stage1_71.py` (commit edde411) ran on LabServer63 on the WL68 inputs, fits and references; output is `results/fnto_cover71/stage1/diag.json`. Every population used here was scored in WL66/WL68, so this is development evidence.

**Observations:**

- **Source identity is position only.** At rest every one of the 1,600 top-face ports has the same non-position local features (displacement, normal, area ratio, thickness, normal spread, face). The network can distinguish rest sources only by absolute position (rest and current xyz), plus slab-edge effects that reach it through neighbourhood means. At pt0 there are 277 distinct non-position vectors on the top face and at cf3 209. WL68's ports 1123, 1025 and 2707 share their non-position features with 1,316 pt0 ports.
- **Physical responses look source-translation-similar.** On the development references the collided flux within 1 cm of the source is 0.90 at every rest port tested, both fitting and WL68-new (within 2 cm it is 0.995). At pt0 dent ports it falls to 0.80–0.87.
- **The frozen WL59 model reproduces this.** P_5902 gives 0.895–0.903 within 1 cm at every tested port.
- **The WL68 fits lose it at new ports.** At the fitting ports they give 0.91–0.92. At the new ports they give:

  | Fit | Port 1025 | Port 2707 |
  |---|---|---|
  | net_all_5902 | 0.76 | 0.34 |
  | net_all_5901 | 0.73 | 0.80 |
  | scratch | 0.69 | 0.52 |

  Port 2707 shares none of its receivers with the fitting sources; 1025 also shares none. 2148 shares 23% and 2422 shares 75%.
- **The fit changes T almost everywhere.** The relative change of T against the init checkpoint is 0.80–0.95 (all-group) or 0.89–1.07 (net group), with median nearly flat in rest-xy distance to the nearest fitting source. The node-state singular values grow 8–35×.

**Hypotheses, not established:**

- **H-cov.** With 4 sources per geometry the objective admits many position-specific maps. More sources spread over positions, faces and dent states would constrain the map toward a source-independent rule.
- **H-map.** The map cannot represent the physical rule's position-independence and its geometry dependence together.

Stage 2 and Stage 3 test the coverage hypothesis directly.

## Prospective design (before evidence)

The design document is [`docs/plans/SPATIAL_MAP_COVERAGE71_V1.md`](../plans/SPATIAL_MAP_COVERAGE71_V1.md) and the frozen contract is `experiments/fnto_cover71/protocol/cover71_v1.json`. In summary:

- **Spatial-only arms.** These arms train every parameter except A/O. The A/O are fixed to the WL68 `net_all_5902` bases, which passed WL68's held-out oracle, so the experiment isolates the geometry→T map.
- **Coverage levels.** Only two:
  - S4: the WL66 fitting ports.
  - S16: S4, plus the 4 WL68 held-out ports (now development data), plus 8 new ports chosen by a geometry-only rule. 2 of the new ports lie inside the pt0 dent on the top face and 1 below it.
  
  Training geometries are rest and pt0.
- **Budget accounting.**
  - C: S4 at 4 replicas.
  - D1: S16 at 4 replicas, i.e. fixed per-source precision and 4× the photons.
  - D2: S16 at replica 0 only, i.e. a fixed total budget equal to C.
  - J (joint A/O on S16) is labelled separately.
- **Teacher generalization control.** Students start from P_5901 with fixed A/O and are trained on sparse or expanded responses of two known members: TA (WL68 `net_all_5902`) and TB (the P_5902 map with fixed A/O). They are scored noise-free:
  - at the final ports (rest, cf2);
  - at 64 random ports;
  - on the cf2−rest signed change.
- **Final held-out.**
  - Ports: 8 previously unscored ports (2475, 1769, 2536, 1419, 960, 2462, 2389, 1116), selected by rule. 3 top ports lie in the cf2 dent (dz −4.49 / −2.34 / −0.25 mm), 1 bottom port is below its centre and 4 are controls.
  - Geometry: cf2, the WL57 confirmation pose never used for probe references and the farthest dent from pt0. It is the archived WL57 geometry (sha 7f8d3c0f…); the fresh LabServer63 build matches its features bitwise, and β differs by relative L2 0.105, the same Monte Carlo quadrature difference WL68 saw.
  - References: rest and cf2, 4 replicas each, new seeds 713xxxxxx, with the signed deformation.
- **Decision.** The outcome rules, primary-candidate selection and thresholds are in the protocol and are unchanged from WL59/WL68.

### Pre-evidence engineering facts (disclosed)

- **Training-port targets.** The targets were revised once before any response existed. The first dev port run placed all 8 new training ports outside the pt0 dent (dz = 0), so 3 targets were moved into or below the dent (commit 1942bb2). The final-port targets were not changed.
- **Dev bug fixes.** Three bugs were fixed in the dev root (`~/rest_cover71_dev`) before the protocol commit:
  - The seed formula in the code differed from the protocol, which produced colliding seeds.
  - Two equivalence tests asserted float32 equality. float32 reductions are nondeterministic run to run, so the tests now assert float64 identity: values and gradients ≤ 1e-10 (measured 1e-14).
  - A perturbed-weight passivity test hit float32 sigmoid saturation. It is now seeded and uses the WL68 1+1e-6 tolerance.
- **Dev pipeline smoke.** The dev root ran the full pipeline with 0.5M-photon packets, 40-step fits and a 5-iteration oracle. No metric value from it is used, and its decision label carries no information.
- **Export file-name collision.** The dev export failed because `refs/manifest.json` was copied into `final/` under the name of the final manifest. Reference records are now copied as `refs_manifest.json` and `refs_complete.json`.
- **Dev throughput and memory with two concurrent fits.** Measured on LabServer63:

  | Fit | Seconds per step | Peak GPU memory |
  |---|---|---|
  | S4 | 0.05 | 1.06 GB |
  | S16 spatial | 0.06–0.08 | 3.35 GB |
  | Joint | 0.25 | 3.72 GB |

  Evidence fits therefore run in two concurrent lanes (protocol `execution`), so per-run seconds are contended timings.
- **Dev reference packets** (0.5M photons) took 1.6–2.1 s each, dominated by fixed per-packet overhead.
- **Idle GPU.** The GPU sat idle from the end of the dev chain (05:27 UTC) to 07:43 UTC because the session watching it restarted. This was lost time only.

## Execution

**Commits:**

| Commit | Content |
|---|---|
| 4431fe8 | Protocol and code frozen; protocol sha256 87cacb56… |
| 853c582 | Evidence port identities |
| f91197e, `amendments[1]` | Amendment 1 |

**Server chain.** The evidence chain ran as a single server chain on LabServer63 (`~/rest_cover71`): pre (ports, tests) → ports marker → preflight → generation → bundle → fit targets → post tests → 12 fits in 2 lanes → evaluation → export.

- The port marker records the frozen code commit server-side, so the chain did not wait for the session (user instruction, 2026-11-02).
- Pre tests passed 6/6 and post tests 2/2.
- The ports are identical to the dev run (`results/fnto_cover71/pre/ports.json`, sha d1057a30…).

**Reference preflight.** By rule: 32M chunk, 3.37 s per packet, peak 9.53 GB; the 16M chunk took 5.29 s.

**Amendment 1 (execution only).** Generation stopped after 175 valid packets.

- **Cause.** The 32M→16M OOM fallback retried inside the `except` handler. There the traceback still held the failed attempt's tensors, so the second fallback (rest slot 9, bin 5, rep 7) ran out of memory again.
- **Change.** The retry now runs after the handler exits, with `gc.collect()` and `empty_cache()`.
- **Unchanged.** Estimator, seeds, rules and packets. The failed packet was regenerated from its own seed.

**Generation result.** All 3,072 packets passed every gate. 3,045 used the 32M chunk and 27 fell back to 16M. Total packet time was 2.92 h.

**User-observed GPU duty during generation.** About 43% mean SM, oscillating between ~10% and ~100% every ~3 s, with one CPU core at 100%. The cause is the fixed WL57 fast walker: a dense ~1 s burst, then a ~2 s Python-loop tail over few remaining photons. The 32M chunk's memory prevents running a concurrent job on the 12 GB card.

The user decided to leave it as is and to restructure the walker for GPU efficiency after WL71 (a pending, separate task). WL68's fitting objective was not affected.

**Fits.** All 12 runs completed. Every physical run converged by the WL68 rule. The noise-free teacher students had not met the last-decile rule, which the adequacy rule does not require. Times below are contended (two concurrent lanes):

| Run | Seconds | Peak memory |
|---|---|---|
| C | 1,028–1,057 | 1.06 GB |
| D1 / D2 | 1,607–1,616 | 3.35 GB |
| J | 5,248–5,260 | 3.72 GB |
| Teacher students | 1,040–1,386 | — |

Evaluation took 1,005 s with a peak of 3.02 GB. The chain finished on 2026-11-02 at 14:36 UTC.

## Final report (2026-11-02)

The evaluation follows the frozen protocol; nothing was rerun or reselected.

- **Exports:** `results/fnto_cover71/final/`, including the full `analysis.json`.
- **Reviewer copies:** `results/evaluation/71/`.
- **Metric format:** "a / f / c" means coarse collided / fine16 / coarse-contrast relative L2. Each value is the upper 95% jackknife bound, worst channel, pooled per group.
- **Thresholds:** 0.10 / 0.15 / 0.25 (WL59/WL68). Deformation: error ≤ 0.35 and corr ≥ 0.90 (WL59 dynamic).

### Implementation Fact

- **Family.** Every fitted model is an experimental copy of the unchanged WL59 P family: 157,024 parameters, same features, rank, width, A/O and spatial semantics. The fixed WL68 `net_all_5902` angular bases were installed by rule.
- **Passivity.** All network candidates pass the WL59 zero/linearity/cache/positivity tests at rest, pt0, cf3 and cf2. The maximum column sum is ≤ 0.9990.
- **Untouched.** No production checkpoint, production K/C, feature, threshold or objective was changed.
- **Batched restriction.** `family71` is certified against `family68` in float64: values and gradients ≤ 1e-10, measured 1e-14.
- **Precision.**
  - All 16 final matrices are qualified (signal/SE ≥ 10 in every channel).
  - The signed-deformation matrices at the 4 displaced or underlying ports (20, 21, 22, 25) are qualified, with signal/SE 380–4,192.
  - The 4 far controls (23, 24, 26, 27) have no measurable change (|signal/SE| ≤ 0.4) and are excluded by rule. The 4-matrix minimum is met.
- **Primary.** By rule the primary candidate is D1_5902: final training J 0.03677 against 0.03677 for D1_5901, the lower by 2e-6.

### Measurement

**Physical arms.** Values are a / f / c, upper bound, worst channel. Dev groups are labelled development.

| Candidate | S16 rest | S16 ports 4–15, rest | FIN_rest | FIN_cf2 | DEV cf3, WL68 ports | DEF_cf2 err / corr |
|---|---|---|---|---|---|---|
| frozen P_5902 | .174/.255/.92 | .198/.254/.93 | .153/.255/.90 | .151/.256/.90 | .155/.258/.95 | .41 / .91 |
| WL68 net_all_5902 (sparse joint) | .520/.573/.90 | .625/.659/.97 | .362/.427/.82 | .361/.428/.87 | .562/.606/.95 | 1.44 / .58 |
| C_5901 / C_5902 (S4) | .456/.477/1.10, .437/.478/1.01 | .545/.548/1.17, .524/.549/1.09 | .456/.503/1.05, .317/.374/.96 | .457/.482/1.06, .318/.375/1.01 | .528/.558/1.11, .493/.550/.93 | .97/.59, 1.26/.80 |
| **D1_5902 (primary)** | **.068/.091/.20** | .077/.092/.21 | .167/.235/.44 | .170/.236/.45 | .057/.094/.16 | .91 / .88 |
| D1_5901 | .068/.090/.20 | .077/.091/.21 | .156/.195/.47 | .160/.202/.48 | .053/.090/.16 | 1.37 / .81 |
| D2_5901 / D2_5902 (S16, fixed budget) | .068/.090/.20 | .077/.091–.092/.21 | .166/.219/.49, .162/.235/.48 | .170/.227/.49, .164/.236/.49 | .053–.056/.09/.16 | 1.28/.83, .97/.86 |
| J_5901 / J_5902 (joint, S16) | .069/.089/.17 | .078/.090/.17 | .170/.231/.33, .163/.241/.39 | .171/.232/.34, .166/.251/.44 | .058/.09/.14 | .73/.89, 1.41/.82 |
| **Oracle**, fixed WL68 A/O, admissible T on final references (labelled) | — | — | .062/.087/.175 | .062/.087/.176 | — | — |

**Decision inputs:**

- S_fit passes: S16 rest .068/.091/.20 and pt0 .070/.091/.20.
- S_cross fails, S_state fails and S_def fails.
- **improves = true.** D1_5902 against C_5902 on FIN_rest and FIN_cf2: contrast .44 / .45 against .96 / 1.01, and collided error .167 / .170 against .317 / .318. Both are below 0.8× in both groups.
- **adequate = true:** the primary converged, both S16 teacher students recover their teachers at their training sources, and the references are qualified.

**Final ports: distance to the nearest same-face training port against error** (rest; cf2 is nearly identical). Values are a / c.

| Slot | Position (mm), face | Distance to S4 / S16 (mm) | C_5902 | D1_5902 | frozen |
|---|---|---|---|---|---|
| 20 | (16, 9), top, cf2 dent | 15.0 / 12.7 | .14 / .50 | .06 / .19 | .16 / .89 |
| 21 | (−4, 4), top | 12.5 / 10.6 | .05 / .43 | .04 / .28 | .06 / .68 |
| 22 | (16, 21), top, dent | 27.5 / 14.6 | .16 / .69 | .07 / .25 | .17 / .96 |
| 23 | (−14, 9), top | 23.0 / 17.7 | .07 / .27 | .08 / .23 | .16 / .89 |
| 24 | (−29, −29), top corner | 39.5 / 22.5 | .73 / 1.37 | **.48 / .83** | .23 / .97 |
| 25 | (16, 9), bottom, below dent | 19.5 / 12.5 | .23 / 1.26 | .07 / .19 | .17 / .90 |
| 26 | (14, −16), bottom | 14.1 / 10.3 | .29 / 1.19 | .07 / .38 | .05 / .75 |
| 27 | (−29, 29), bottom corner | 47.8 / 27.5 | .53 / 1.45 | **.18 / .53** | .22 / .94 |

**Signed deformation per port, DEF cf2 − rest** (error / corr):

| Port | frozen | D1_5902 |
|---|---|---|
| 20 | .40 / .92 | .74 / .92 |
| 21 | .77 / .67 | 1.66 / .83 |
| 22 | .70 / .79 | 1.11 / .81 |
| 25 | .40 / .92 | 1.18 / .83 |

D1 per-channel slope is 1.07 / 0.98 / 1.46.

**Teacher generalization** (noise-free; worst channel, a / f / c):

| Student | Training sources | FIN rest | FIN cf2 | R64 | DEF FIN (err / corr) |
|---|---|---|---|---|---|
| TA_S4 (TA = WL68 net_all_5902) | .003/.011/.009 | .42/.52/1.04 | .43/.50/1.07 | .59/.70/1.24 | 1.51 / .73 |
| TA_S16 | .005/.016/.008 | .17/.25/.81 | .18/.29/.85 | .28/.39/.68 | .64 / .88 |
| TB_S4 (TB = P_5902 map + fixed A/O) | .000/.001/.001 | .007/.014/.044 | .008/.016/.052 | .057/.091/.45 | .096 / .996 |
| TB_S16 | .001/.001/.001 | .001/.002/.005 | .004/.011/.019 | .048/.073/.37 | .092 / .998 |

- **Initial errors.** The initial students are far off at FIN: contrast 1.55 for TA and 1.89 for TB. Every test is informative.
- **R64 composition.** R64 has 12 side-face ports and 32 ports within 1 cm of a slab edge; no training source lies on either.
- **Generalization result.** `teacher_generalizes` is false: TA fails everywhere off the training set, and TB fails R64 contrast.

**WL59 historical full fields** (inspected in WL59–68, not fresh). Coarse mean error:

| Model | rest | pt0 | cf3 |
|---|---|---|---|
| frozen | .062–.066 | | |
| WL68 net_all_5902 | .57–.85 | | |
| C_5902 | .45–.94 | | |
| D1_5902 | .24 / .18 / .18 | .24 / .18 / .18 | .25 / .18 / .18 |
| J_5901 | .25 / .22 / .15 | | |

No fitted model is F-path eligible.

### Observation

1. **Source coverage changes the fitted map substantially.** On the independent final ports, the expanded spatial-only fits (D1, D2) reduce pooled contrast from ~1.0 (C) to 0.44–0.49 and collided error from 0.32–0.46 to 0.16–0.17, on both rest and the unseen geometry cf2. They also exceed the sparse WL68 joint fits. They still miss all three thresholds.
2. **Generalization is local in position.** The six final ports within 10–18 mm of an S16 training source on the same face reach a ≤ .085 and c .19–.38 (D1_5902); several pass individually. The two ports 22–28 mm away, both in uncovered corners, fail badly: slot 24 at .48 / .83 is worse than frozen WL59 in collided error. These two dominate the pooled failure.
3. **Precision is not the limit.** D2 (same total photons as C, spread over 16 sources at one replica each) equals D1 (4× photons) within seed spread.
4. **Joint A/O training on S16 (J) behaves like D1,** with slightly lower contrast on the final groups (.33–.44).
5. **The fixed angular bases are adequate.** With fixed WL68 A/O, admissible spatial coefficients reach .062 / .087 / .175 on both final groups (labelled oracle). The shortfall is in the geometry→T map.
6. **Geometry-state change works at trained positions but not at new ones.**
   - At trained positions it reaches cf3 (development, WL68 ports 4–7, now S16 sources) at .053–.058 / .09 / .14–.16.
   - At new positions FIN_cf2 ≈ FIN_rest.
   - The signed deformation at the dent ports is not reproduced: D1 error .74–1.66 with slope up to 1.46. These errors are larger than frozen WL59's (.40–.77), though correlation is similar (.81–.92).
7. **Teachers: a smooth same-family rule is learnable from sparse data, a position-specific one is not.**
   - TB, WL59's smooth map, is recovered at the final ports and on deformation from S4 alone. It is still not recovered at side and edge ports (R64 contrast .37–.45).
   - TA, the WL68-fitted map, is not recovered off the training sources even from S16.
8. **Expanded coverage reduces the full-field regression but does not remove it.** The fitted maps still regress on the historical full fields (D1 .18–.25 against .062 frozen), less than the sparse fits (.45–.94).

### Interpretation

Each statement is limited to this population: one homogeneous-medium slab family, training geometries rest and pt0, 16 training sources on top and bottom faces, and final ports on rest and cf2.

- **Coverage supports an identifiability explanation for part of the WL68 failure.** Adding sources moves the same unchanged map from failing everywhere off the fitting ports to near-threshold behaviour within ~1–2 cm of a supervised source. At equal photon budget it is the number and spread of sources, not label precision, that matters.
- **The learned rule is not source-independent.** Its accuracy decays with distance from supervised positions, and it collapses in uncovered regions. This matches Stage 1:
  - the map conditions on absolute position;
  - with sparse physical supervision it converges to position-specific solutions rather than the translation-similar rule the physical responses exhibit.
- **The learning procedure can generalize a smooth rule but the physical fits do not land on one.** TB students learn a smooth same-family rule from 4 sources, so the procedure and map *can* generalize when the target is such a rule. The physical fits do not reach such a solution from 16 sources, and the teacher built from a physical fit (TA) is not identifiable from 16 sources either.
- **Not separated:** whether denser or more systematic coverage would close the remaining gaps, or whether no smooth member of the family fits the physical responses to threshold.
- **Signed deformation is not learned from two training geometries,** even where static accuracy is near threshold.
- **The fitted family is not a reusable K by the brief's definition.** It still requires supervised sources near every queried position, so it cannot be called reusable.

### Unresolved Question

1. Does a smooth, source-independent member of the unchanged WL59 family exist that meets the static thresholds on the physical probes? TB shows smooth members are learnable; WL68 and WL71 show the physically fitted members are position-specific. Neither establishes existence or non-existence.
2. How fast does error fall with source density? Two coverage levels cannot answer this, and it is untested for side and edge ports and for more training geometries.
3. Why is signed deformation worse than frozen WL59 at the dent ports while static error near the dent is close to threshold?
4. Is the coverage gain robust to the fitting-port selection? There is one S16 layout.

### Architecture Verdict

- **Outcome B** by the predeclared mapping: S_fit passes, improves, but S_cross, S_state and S_def fail. Expanded source coverage improves the mapping substantially, but the cross-source, geometry-state and dynamic gates remain unsatisfied.
- **Family status:** the WL59 geometry-conditioned spatial map is **not demonstrated reusable**. Its generalization is supervision-local, and deformation is not learned. A limitation of the map is not established either: the controls show the procedure can generalize a smooth same-family rule.
- **Not claimed:** no adoption, successor, coupled GI or F/S claim.

**Completion question.** *Does the unchanged WL59 geometry-conditioned spatial map generalize physically valid collided transport when source coverage is controlled, or does the failure persist in a way that justifies reconsidering the mapping architecture?*

- Under controlled coverage it generalizes **partially and locally**: to the unseen geometry state, and to new ports close to supervised sources.
- The failure **persists** at uncovered positions and for signed deformation.
- Its form, position-specific solutions learned from physical supervision, is the evidence relevant to the mapping's absolute-position conditioning.
- Whether that justifies reconsidering the mapping architecture, or calls for denser coverage, is a user decision. This batch does not separate the two.

### Reproducibility

- **Commits:** 4431fe8 (frozen protocol and code), 853c582 (ports), f91197e (amendment 1), and this report's commit.
- **Protocol:** `experiments/fnto_cover71/protocol/cover71_v1.json`, sha 87cacb56… at freeze; `amendments[1]` added later.
- **Inputs:**
  - WL68 inputs, manifest sha 2abf3ba6…, read-only;
  - WL68 fit targets, 3c21b619… (identical to the S4 part of `fit_targets71`);
  - cf2 geometry, 7f8d3c0f….
- **Ports:** `ports.json`, sha d1057a30….
- **References:** seeds `713000000 + g·1e6 + slot·1e4 + bin·100 + rep`; 3,072 packets in `refs_manifest.json` and `refs_complete.json`.
- **Server:** LabServer63, RTX 3080 Ti 12 GB, torch 2.3.1+cu121, Mitsuba `cuda_ad_rgb` with the user-space OptiX path.
- **Optimizer:** the WL68 `net` optimizer, 20,000 steps, 2 concurrent lanes.
- **Determinism:** fits are not bitwise reproducible (CUDA `index_add`).
- **Run identity:** checkpoint hashes are in `final/run_checkpoints.sha256`, and run files are on the server under `~/rest_cover71/runs`.
