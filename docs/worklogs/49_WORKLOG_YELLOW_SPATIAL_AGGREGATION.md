# Worklog — Bounded Yellow Nonlocal Spatial Aggregation Experiment (2026-10-27)

Protocol `fnto_spatial49` + amendments 1–6 (`experiments/fnto_spatial49/protocol/spatial49.json`). Authorisation: the
user brief "ReST-GI — Worklog 49: Bounded Yellow Nonlocal Spatial Aggregation Experiment" (2026-10-27).

**Machine.** First batch on LabServer69, the main server since 2026-10-27 (RTX 3090 24 GB; bpy in docker
`restgi/bpy-runtime:22.04`). The inputs migrated from LabServer63 were verified: all 15 worklog-45 freeze sha256 values
are equal, including K_k44P `30d55da3…`.

**Frozen.** K_k44P, the production K / C code, support radii (5 / 30 cm), global rank, relations, anchors, references and
worklogs 1–48. Nothing in production was touched. The only new mechanism is one aggregation candidate (DASQ), run in
isolated experiments.

**Answer to the completion question.**
- **The present in-support aggregation imposes a real limitation on Yellow nonlocal transport.** With every fitting
  control that is valid, the best in-support shadow error on the Burley oracle stays at R 0.56 / G 0.23 / B 0.22 (test
  fields). The causes are measured:
  - **R:** the representative sub-selection. The 96 + 128 representatives carry only 28 % of the red in-support shadow
    response; the rest is unobserved by any function of the features.
  - **B and G:** within-aggregate resolution. An atom collision proves an irreducible loss of 0.22 for B and 0.05–0.07
    for G.
- **One bounded change, receiver-centred distance-adaptive source quadrature (DASQ), removes most of it.**
  - Oracle space, ideal-weight quadrature: R 0.51 → 0.044, G 0.36 → 0.054, B 0.33 → 0.12.
  - Matched learned comparison (held-out receivers × test fields): R 0.32 → 0.043, G 0.19 → 0.067, B 0.26 → 0.19–0.20,
    both seeds.
  - Real-scene Yellow first-event fidelity against the Cycles references: +0.65 dB [0.48, 0.82] (8 / 8 views, both
    seeds, T0 and T1). DASQ reaches the oracle's own fidelity (24.8 dB).
- **Costs and limits.**
  - DASQ needs 2.2–2.8× the existing element budget. A budget-matched variant was infeasible.
  - The learned comparison's predeclared fair-state rule is not met by L1, whose validation loss fluctuates within
    ±10 % of a level 12× below L0's. Formally the learned comparison is INCONCLUSIVE.
  - Coupled-solve integration was not authorised.
- **The Gate 0 direct Cycles measurement settles worklog 48's open question.** For one sun (view 6), Cycles' Yellow
  first-event exit flux is 3.9 / 1.38 / 0.056× the sun power incident on the whole Lego mesh (R / G / B), replicated
  within 0.6 %. The reference itself is non-passive.

---

## IMPLEMENTATION FACT

**Spatial information that changed (candidate only).**

| | Existing K_band / K_mid aggregation | DASQ candidate |
|---|---|---|
| Elements per receiver | 96 representative 1.5 cm band aggregates within 5 cm, plus 128 representative 8 cm mid cells within 30 cm, each weighted by its represented area | Partition of every Lego port within 30 cm of the receiver (self excluded) into octree elements |
| Element size | Fixed (1.5 cm / 8 cm) | Cell diagonal ≤ α × distance; leaves are single ports |
| Support boundary | Representative-level | Decided per element by centroid distance (amendment 4, like K44's mid cells) |
| Feature | Area-mean irradiance | Area-mean irradiance, via prefix sums over a Morton-ordered port list |

Both keep the same support (30 cm), object scope, field input (scalar irradiance; worklog 47), output readout (isotropic
scalar) and linearity in the field. The candidate is removable: `experiments/fnto_spatial49/dasq.py`.

**Code** (`experiments/fnto_spatial49/`).

| File | Role |
|---|---|
| `s49_data.py` | Lego tables; K44 probe / view fields (r1_train incident semantics); oracle in-support / tail split; aggregates; VRAM-adaptive chunks |
| `g0_exitance_bake.py`, `g0_check.py` | Cycles Yellow exitance bake (DIFFUSE direct, vertex colours) and white-Lambertian incident bake; reference checks |
| `g1_capacity.py`, `g1b_fitcontrols.py` | Existing-aggregation capacity: A0, A1, A1v, A1nn, A1g, atom collision |
| `dasq.py`, `g2_candidate.py` | Candidate and its comparison (B2′, B2d; coverage diagnostic) |
| `g3_learned.py` | Matched learned L0 / L1 (identical 9,155-parameter network) |
| `g4_real.py` | Real-scene Yellow vs first_r1 |
| `panels49.py` | Panels |
| `tests/test_s49.py` | Focused tests |

**Gates completed:** 0, 1 (+ 1b), 2, 3 and 4 (first-event only). Composition (coupled solve) is NOT APPLICABLE: it would
require modifying production K / C.

**Tests** (ALL_PASS before each evidence step; JSONs in `results/fnto_spatial49/tests/`):
- frozen K44 identity;
- train / test separation;
- object-scope membership;
- zero field;
- linear superposition;
- finite / nonnegative;
- source-receiver index, support boundary and self exclusion against brute force;
- aggregation normalisation (Lego aggregates; atom projection keeps every aggregate mean);
- probe fields;
- DASQ disjoint / no-self / uniform-field exact (4e-8);
- learned-arm tables (finite, zero field → 0, linearity 1e-4, one code path for both arms).

**Amendments** (all committed before the evidence they affect).

| # | Committed before | Change |
|---|---|---|
| 1 | any output | Crash fix (CUDA tensor → numpy) |
| 2 | — | The predeclared A1 (ridge 1e-6) over-fits: fit-field error 0.009–0.04 vs validation 1.2–15. It is kept as a failed control. Valid controls are added: A1v (validation-selected ridge) and A1nn (nonnegative LS). Decision rule unchanged, applied with A1*. |
| 3 | any candidate evaluation | DASQ specification |
| 4 | any Gate 2 error | Element-level support membership. Straddling-node refinement produced 17–25 k elements per receiver; the measured counts made B1 (≤ 224) and B2 (≤ 448) infeasible. B2′ = α 1.0 (2.2× median / 2.8× q95 budget) becomes the decision condition; B2d = α 0.7 is a diagnostic. |
| 5 | — | Port areas sum to 1.5× the mesh area. A white-Lambertian bake gives the mesh-consistent incident power. |
| 6 | the reruns | Two fixes. (a) Gate 4 bug: references indexed by valid-pixel rank. The run-1 numbers (all ≈ 11 dB) are invalid and preserved. (b) Gate 3 fair-state check moved from the training minibatch to validation loss, as the protocol says. Both arms retrained identically for 20,000 steps (the 4,000-step run is preserved). |

Further engineering commits made no logic change: receiver chunking; VRAM-adaptive chunks; a line restored after the
amendment-6 edit.

---

## MEASUREMENT

All errors are in-support (r < 30 cm) relative L2 against the Burley oracle, on shadow receivers, pooled over the test
fields (120 holdout probe configurations + 16 T0 view sun fields), unless stated otherwise.

### Gate 0 — reference readiness

**Checks.**
- Object scope: all 8 materials in the sources.
- s_c provenance: worklog 45 'obj', fitted on T0 views 0–7 and scored on 8–15.
- Splits disjoint.
- Self exclusion matches brute force to 9e-8; the self port carries 2.5 % of the receiver response.
- Readouts are consistent: isotropic scalars for oracle and arms; K44 is read at the camera.

**Direct Cycles exitance** (view 6 sun, 32 spp × 2 seeds, CPU in docker).

| | R | G | B |
|---|---|---|---|
| Yellow first-event exit flux (W) | 17.39 / 17.50 | 6.14 / 6.16 | 0.247 / 0.248 |
| Replicate difference | 0.6 % | 0.2 % | 0.4 % → **valid** |
| / incident on the whole Lego mesh (4.442 W, white bake) | **3.92–3.94** | **1.38–1.39** | 0.056 |
| / incident on Yellow mesh (2.146 W) | 8.1–8.2 | 2.86–2.87 | 0.115 |
| / injected (0.98 × Yellow incident) | 8.3 | 2.9 | 0.12 |

- Mesh vs port areas: Lego 27.0 vs 40.8 m², Yellow 11.0 vs 15.9 m².
- Port-based incident power is 5.06 W (Lego) and 2.16 W (Yellow).

### Gate 1 — existing aggregation capacity

| Condition (224 features per receiver) | R | G | B | Note |
|---|---|---|---|---|
| A0 ideal-weight quadrature | 0.508 | 0.362 | 0.329 | view_T0 alone: 0.356 / 0.280 / 0.264 |
| A1 predeclared ridge 1e-6 | 17.9 | 2.36 | 7.87 | over-fit (val 5.9 / 1.2 / 15); **invalid control** |
| A1g (+64 global PCA features) | 7.9 | 2.0 | 6.4 | same failure |
| **A1v** (val-selected λ 0.1 / 1e-3 / 1e-4) | **0.560** | 0.413 | 0.417 | relaxed: per-receiver free weights |
| **A1nn** (nonnegative LS) | 0.709 | **0.227** | **0.220** | relaxed |
| **A1\*** (lower validation error) | **0.56 (A1v)** | **0.23 (A1nn)** | **0.22 (A1nn)** | decision: all ≥ 0.10 → **proceed** |
| Atom collision e_coll (irreducible for any function of the features) | 0.015 | 0.069 | **0.225** | view_T0: 0.009 / 0.050 / 0.17 |
| Coverage: in-support shadow response inside represented aggregates (view_T0) | **0.29** | 0.53 | 0.69 | uniform field: 0.28 / 0.52 / 0.73 |

- Tail share of the full shadow response (beyond 30 cm, K_glob's part): R 0.24–0.25; G / B 0.
- Distance to light (A0, R, view_T0): 0.25 / 0.31 / 0.48 / 0.59 for 0–1 / 1–3 / 3–10 / 10–30 cm.

### Gate 2 — DASQ vs existing aggregation (same oracle / fields / receivers)

| Condition | Elements median / q95 | quad R / G / B | A1v R / G / B | A1nn R / G / B |
|---|---|---|---|---|
| B0 existing | 224 / 224 | 0.508 / 0.362 / 0.329 | 0.560 / 0.413 / 0.417 | 0.709 / 0.227 / 0.220 |
| **B2′ DASQ α 1.0** | 482 / 620 (max 703) | **0.044 / 0.054 / 0.122** | 0.268 / 0.379 / 0.382 | **0.084 / 0.043 / 0.046** |
| B2d DASQ α 0.7 (diagnostic) | 1079 / 1450 | 0.031 / 0.040 / 0.116 | 0.317 / 0.424 / 0.427 | 0.046 / 0.023 / 0.034 |

- Decision (B2′ vs B0, same fit type):
  - quad: relative gains 0.91 / 0.85 / 0.63, absolute 0.46 / 0.31 / 0.21 → **pass**;
  - A1*: 0.85 / 0.81 / 0.79 → **pass**.
- Red, view_T0 shadow, by distance: DASQ quad 0.03 / 0.03 / 0.04 / 0.10 vs A0 0.25 / 0.31 / 0.48 / 0.59.
- view_T1, B2′: quad 0.033 / 0.061 / 0.151; A1nn 0.021 / 0.009 / 0.018.
- A1v is poor on DASQ because the features (≈ 480) exceed the fit fields (480), so the problem is under-determined. A1nn
  is the valid fit there.
- B1 / B2 (≤ 224 / 448 elements) are infeasible: counts saturate at 482 / 620 for α ≥ 1.
- Cost: octree 0.28 s; traversal 0.03 s / 2000 receivers; ideal weights 1.1 s / channel; features 0.2 s for all 752
  fields.

### Gate 3 — matched learned comparison

Identical network (9,155 parameters), descriptors, objective, budget and seeds. Errors on held-out receivers (1000) ×
test fields.

| Arm | Seed | R | G | B | view_T1 R / G / B | Val loss change (last 10 %) | Fair state |
|---|---|---|---|---|---|---|---|
| L0 existing elements (224) | 4911 | 0.319 | 0.194 | 0.261 | 0.260 / 0.149 / 0.237 | 0.3 % | yes |
| L0 | 4912 | 0.316 | 0.192 | 0.259 | 0.256 / 0.147 / 0.243 | 1.3 % | yes |
| **L1 DASQ elements (482)** | 4911 | **0.044** | **0.067** | **0.198** | 0.031 / 0.062 / 0.255 | 14 % | **no** |
| **L1** | 4912 | **0.043** | **0.067** | **0.188** | 0.030 / 0.058 / 0.239 | 15 % | **no** |

- Relative gain L1 vs L0: R 0.86, G 0.65–0.66, B 0.24–0.27 → the "L1 better" rule **passes for both seeds**.
- Fair-state rule: **not met by L1**, so the comparison is formally INCONCLUSIVE.
  - L1's validation curve plateaus at 0.014–0.017 with ±10 % fluctuation; L0's sits at 0.186.
  - The losing arm (L0) is converged.
- Run 1 (4,000 steps, preserved): L0 0.37 / 0.27–0.28 / 0.32–0.33; L1 0.05–0.07 / 0.09–0.10 / 0.22–0.24. Same
  direction.
- Red by distance, held-out receivers, view_T0: L1 0.02 / 0.03 / 0.04 / 0.08 vs L0 0.16 / 0.22 / 0.36 / 0.47.
  Shadow ratio 1.004 vs 0.85; corr 0.999 vs 0.92.
- Training 49 s per run (20k steps); tables: L0 0.1 s, L1 2.8 s (3000 receivers); peak 13.1 GB.

### Gate 4 — real-scene Yellow first-event (Cycles first_r1)

8,000 Yellow pixels per view common to T0 / T1; views 8 / 12 use all Yellow pixels. Test views 8–15. All PSNRs are
display ldr3.

| T0 | All | Lit | Shadow | Shadow 3–30 cm | Rel L2 (linear) | Lum ratio | Corr | Per channel R / G / B |
|---|---|---|---|---|---|---|---|---|
| K44 (historical, camera readout) | 24.29 | **27.79** | 23.35 | 21.62 | 0.303 | 0.968 | 0.906 | 22.47 / 23.13 / 31.86 |
| Oracle | 24.80 | 24.32 | 25.03 | 23.94 | 0.278 | 0.967 | 0.903 | 24.51 / 24.27 / 25.77 |
| L0 (s4911 / s4912) | 23.99 / 23.99 | 23.98 / 23.91 | 23.99 / 24.03 | 22.85 / 22.95 | 0.313 | 0.957 | 0.89 | 23.26 / 23.28 / 25.93 |
| **L1 (s4911 / s4912)** | **24.78 / 24.73** | 24.06 / 24.02 | **25.14 / 25.07** | **24.12 / 24.06** | 0.284 | 0.985 | 0.90 | 24.52 / 24.19 / 25.79 |

- **L1 − L0**, per-view PSNR:
  - T0: +0.65 dB, 95 % CI [0.49, 0.82] (seed 4911) and [0.48, 0.80] (seed 4912), 8 / 8 views;
  - T1: +0.66 / +0.65 dB, 8 / 8 views.
- T1 pooled values are within 0.2 dB of T0.
- View 14 is an outlier: K44 27.8 dB, while oracle / L0 / L1 are 19.6–20.0 dB.
- **Edit T0 → T1**, Yellow pixels (reference change RMS 0.021), corr / slope:

  | Condition | corr | slope |
  |---|---|---|
  | K44 | 0.855 | 0.745 |
  | Oracle | 0.924 | 0.968 |
  | L0 | 0.906 | 0.913–0.918 |
  | L1 | 0.909–0.911 | **1.010–1.016** |

- Composition: NOT APPLICABLE (production K / C would have to change).
- Cost: 37,519 receiver ports; 53 s; peak 9.4 GB.

### Resources

| Item | Amount | Limit |
|---|---|---|
| GPU (RTX 3090), all runs incl. failed / OOM attempts | ≈ 31 min | 3 h |
| … of which Gates 0–2 | ≈ 22 min | 45 min |
| CPU renders (bakes) | ≈ 6 min | 1 h |
| Wall time of experimental execution | ≈ 1 h (17:14–18:10 KST) | 5 h |

The largest single cost was Gate 2's NNLS on 1600-element B2d (≈ 15 min of the 16-min Gate 2 run).

## OBSERVATION

- **Panels** (`results/evaluation/49/01_yellow_view{8,12}.png`, one absolute error scale M = 1.39 and one change scale).
  - K44 shows grainy, overly flat shadowed Yellow.
  - L0 under-reaches into shadow regions.
  - L1 and the oracle reproduce the reference's smooth red falloff.
  - The error maps of L1 and the oracle are nearly identical.
- **Capacity plot** (`02_capacity_and_learning.png`).
  - The existing aggregation's red error grows with distance from the light. DASQ (quadrature and learned) stays flat
    up to 10 cm.
  - B remains the hardest channel for every condition. The DASQ ideal quadrature plateaus at 0.12; nonnegative fitting
    reaches 0.03–0.05.
- **Failures and defects** (all preserved):
  - A1 / A1g over-fit;
  - the straddling-node explosion of the first DASQ rule;
  - Gate 2 OOM ×2 and Gate 1 crash;
  - the Gate 4 reference-index bug (run 1);
  - L1 not meeting the 2 % plateau rule.

## INTERPRETATION

- **Information insufficiency** (direct evidence).
  - B / G: atom collisions show that no function of the existing features can reproduce the response. Within-aggregate
    resolution loss is 0.22 (B) and 0.05–0.07 (G) on test fields.
  - R: 72 % of the in-support shadow response lies in regions the representatives do not read. Optimal per-receiver
    weights (A1v 0.56) do no better than ideal weights (0.51). Unobserved information cannot be inferred from the read
    aggregates.
- **Approximation limitation.** With full coverage, DASQ's distance-adaptive elements bring the ideal quadrature to
  0.04–0.12. The remaining B error (0.12) comes from the element-level support boundary and within-element field
  variation near shadow edges; nonnegative fitting halves it.
- **Optimisation / supervision.**
  - The relaxed per-receiver fits bound what any weight model could do on the existing features. A learned shared
    network (L0) is better than A1*, because shared smoothness generalises better than free per-receiver weights, but
    it still sits at 0.32 / 0.19 / 0.26.
  - L1's advantage survives learning with an identical network and objective, so it is not a parameter-count effect
    (parameters are equal; only the element / query count differs, 2.2×).
  - Formally the learned comparison is inconclusive by the predeclared fair-state rule. The losing arm is converged,
    and L1's fluctuation band is an order of magnitude below L0.
- **Angular-readout artefacts.** Not applicable within the oracle comparisons: every arm and the oracle are isotropic
  scalars. K44 is read at the camera, so worklog 48's normal-direction artefact does not enter.
- **Renderer-reference limitation.**
  - The real-scene gain (+0.65 dB) is far smaller than the oracle-space gain, because the oracle itself caps fidelity
    at 24.8 dB: Burley approximation, object-scoped semantics, no local glossy, and port-level evaluation.
  - L1 reaches that cap, and K44 beats every oracle-based arm on lit pixels. The local K_loc fidelity (27.8 dB lit) is
    a separate, existing strength.
  - The direct bake confirms that the reference is energetically non-passive for Yellow (3.9× in red). Under Cycles
    semantics, reference fidelity and physical passivity are therefore in conflict. That is a contract question this
    batch does not change.
- **Composition.** Not evaluated, so first-event gains do not establish coupled-solve stability or end-to-end image
  quality.

## UNRESOLVED QUESTION (consequential)

1. Can DASQ be brought to the existing element budget, or is the 2.2–2.8× query count inherent? The axis-aligned octree
   is not surface-adaptive; a surface-aware hierarchy was not tested. This decides computational viability.
2. How does a DASQ-based Yellow nonlocal term behave inside the coupled solve? Stability, interaction with the
   reference's non-passive Yellow energy, and edit response remain open. This needs production-contract changes, which
   are not authorised.
3. How much of the remaining real-scene gap is oracle vs Cycles? L1 ≈ oracle, so further gains require a better target
   than the Burley oracle, or Cycles-supervised training.
4. The B-channel residual (0.19 learned): element-level support boundary vs shadow-edge variation is not separated.

## ARCHITECTURE VERDICT

| Item | Status | Basis |
|---|---|---|
| A. Existing aggregation information sufficiency | **NOT SUPPORTED** | Valid fits ≥ 0.22 in every channel; irreducible collisions (B 0.22, G 0.05–0.07); red coverage 29 % |
| B. Candidate approximation advantage | **SUPPORTED** (at 2.2–2.8× element budget) | quad 0.51 / 0.36 / 0.33 → 0.044 / 0.054 / 0.12; budget-matched variant infeasible |
| C. Learned-model advantage | **INCONCLUSIVE** by the predeclared fair-state rule (L1 val fluctuation 14–15 %) | Direction unambiguous: L1 0.04 / 0.07 / 0.19–0.20 vs converged L0 0.32 / 0.19 / 0.26, both seeds, identical network |
| D. Real-scene transfer | **CONDITIONALLY SUPPORTED** | +0.65 dB [0.48, 0.82], 8 / 8 views, T0 and T1; capped by the oracle target (L1 ≈ oracle) |
| E. Static transport fidelity | **CONDITIONALLY SUPPORTED** for the Yellow nonlocal (shadow / 3–30 cm) term | Shadow 25.1 vs K44 23.4 dB; K44's lit (K_loc) fidelity is not reproduced by any oracle-based arm; first-event only |
| F. Edit-response fidelity | **CONDITIONALLY SUPPORTED** | First-event Yellow change slope 1.01 (L1) vs 0.91 (L0) / 0.75 (K44); corr 0.91 |
| G. Numerical stability | **NOT APPLICABLE / INCONCLUSIVE** | No coupled solve run (not authorised) |
| H. Computational viability | **CONDITIONALLY SUPPORTED** | 2.2–2.8× elements; O(elements) per field via prefix sums; traversal 15 µs per receiver; L1 tables 2.8 s / 3000 receivers vs 0.1 s; same network size |
| I. Renderer energy / reference validity | Reference **VALID** for fidelity; **NOT passive** (measured) | Direct Cycles bake: Yellow exit 3.9 / 1.38 / 0.056× the Lego-mesh incident power, replicated within 0.6 % |

**On retaining, developing or deferring.** The evidence does not support retaining the existing in-support aggregation
unchanged for Yellow: its information loss is measured and partly irreducible. The proposed alternative is supported in
oracle space and transfers to the real scene, conditionally. Its learned advantage is formally inconclusive, its cost is
2.2–2.8× in elements, and its coupled-solve behaviour is untested. Whether to develop it further (e.g. budget
reduction, solve integration) or defer is the next human decision. The candidate is not adopted.

## REPRODUCIBILITY

- **Commits** (protocol before evidence; each amendment before its evidence):
  - 89067d4 — protocol + Gate 0 / 1
  - 2a9f1e8 — amendment 1
  - 9bab6ee — amendment 2
  - 8d297f7 — amendments 3–4
  - 7b3450e — DASQ test update
  - amendment 5 commit
  - Gate 2 chunking commits
  - e097abe — Gate 3 / 4 code
  - 9304391 — adaptive chunks
  - 012ca60 — amendment 6
  - the g4 line-restore commit
  - panels49 commit
  - this worklog's commit
- **Frozen hashes:** K_k44P `30d55da373a01ab8…` (= worklog 45 freeze); the 15 / 15 freeze files verified on 69.
- **Environment:** LabServer69, RTX 3090 24 GB, CUDA torch 2.8.0+cu128 (RNA venv); Blender / Cycles 3.5.1 in docker
  `restgi/bpy-runtime:22.04` (CPU).
- **Seeds:**

  | Item | Seed |
  |---|---|
  | Fit / val split | 4901 |
  | Fit receivers | 4902 |
  | Held-out receivers | 4903 |
  | Gate 4 pixels | 4904 |
  | Bakes | 4901 / 4902 (32 spp), 4903 (white, 8 spp), 4900 (1-spp timing smoke, not evidence) |
  | Gate 3 training | 4911 / 4912 |
- **Data:**
  - fit = 480 prep2_train configurations;
  - val = 120 prep2_train configurations;
  - test = 120 prep2_holdout configurations + 16 T0 view sun fields (+ 16 T1);
  - receivers: 2000 fit + 1000 held-out Yellow ports.
- **Server scripts:** `~/rest_fnto_r1/s49_bake.sh`, `s49_g01.sh`, `s49_g2.sh`, `s49_chain34.sh`, `s49_chain34b.sh`,
  `s49_chain4c.sh`; status / logs in `~/rest_fnto_r1/launcher/s49_*`.
- **Outputs:**
  - `results/fnto_spatial49/` — gate0_reference, gate1_existing_capacity, gate2_candidate, gate3_learned_ab (L0 / L1
    checkpoints; run-1 checkpoints in `run1_ckpt/`), gate4_real_scene, cache, tests;
  - `results/evaluation/49/` — 3 panels, JSON copies and a manifest.

No later batch or worklog 50 is started.
