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
