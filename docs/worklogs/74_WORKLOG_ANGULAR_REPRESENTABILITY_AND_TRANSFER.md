# Worklog — Angular Representability and Transfer Validation (2026-11-05)

## Authorisation and scope

The user's brief "Worklog 74 — Angular Representability and Transfer Validation" (2026-11-05) asks whether the fixed global angular interface of the intrinsic-K operator limits boundary-source transport fidelity.

**Hypothesis under test.** A shared rank-8 angular factorisation may be sufficient, but the frozen WL68 bases do not suit every source condition.

**Arms:**
- **G0:** fixed WL68 A/O at rank 8.
- **G1:** globally shared, refitted rank-8 A/O.
- **G2:** globally shared, refitted rank-16 A/O. G2 is a diagnostic capacity bound and is not deployable.

**Conditional stages.** If G1 passes a preregistered gate, a bounded learned-K test follows in the same batch:
- WL72 Candidate A, retrained with the frozen G1 basis;
- evaluated on a new independent confirmation population against the frozen WL73 AR controls.

**Preserved:**
- K/C ownership, first-exit semantics and ballistic B;
- 16 incoming / 64 outgoing bins;
- positivity, passivity, linearity and zero input;
- materials and reference definitions;
- the WL59 gates;
- all historical results.

**Not authorised:**
- production changes;
- a new spatial representation;
- geometry-conditioned A/O or a joint network;
- an angular-resolution change or an unrestricted rank sweep;
- threshold tuning;
- confirmation labels in any fit;
- held-out checkpoint selection;
- adoption or a successor.

WL73 data and every previously scored confirmation port are development evidence. The batch runs on LabServer63, root `~/rest_ato74`, branch `Selective_Recompute`, in the isolated `experiments/fnto_ato74`.

## Stage 0 design (pre-evidence)

The protocol is `experiments/fnto_ato74/protocol/ato74_v1.json`.

**Operator.** K[i,j,a,o,c] = d[i,a] Σ_r A[r,a,c] T[i,j,r,c] O[r,o,c], with A = softmax over mixtures and O = softmax over outgoing bins.
- T ≥ 0, with Σ_j T ≤ 1, enforced by Euclidean projection onto the capped simplex.
- Positivity, linearity, zero input and Σ_{j,o} K ≤ d ≤ 1 therefore hold structurally at any rank.
- `family68.predict` hard-codes rank 8 in its coarse path, so `ato74.predict_m` / `objective_m` are rank-generic copies. They are certified equal to family68 at rank 8, in float64.
- The test also checks that duplicated rank-16 bases with duplicated T reproduce the rank-8 operator to ≤ 1e-12. G2 therefore contains the G0 operator class.

**Objective.** The WL68 objective J, relative per source and channel: mean of (native + fine16 + coarse)/3 + coarse contrast.

**Fitting population.** S16-R at rest and pt0, using the means of fitting replicas 0–3 (WL73 `fit_targets73.pt`). Confirmation labels are never used.

**Optimisers.**
- **G1/G2:** block-coordinate descent with 30 rounds. Each round runs 30 warm-started FISTA steps on the per-source T, then 30 Adam steps (lr 0.01) on the shared logits. A 1,000-step FISTA polish of T follows. Float32.
- **G0:** the same 1,000-step polish, then 2,000 further steps as a convex convergence check.

**Restarts.** G1 and G2 each run 3 restarts:
- restart 0 starts from the WL68 bases (G2: duplicated WL68 + N(0, 0.01));
- restarts 1 and 2 start from N(0, 1) logits with fixed seeds.

**Stage 1 scoring.** On the same training sources, against held-out replicas 4–7.

**Stage 2.** For every frozen basis, the WL73 confirmation sources (development evidence) are refitted per source and scored:
- per-source T is fitted on replica 0 (float64 FISTA, 1,500 steps, then 1,500 more as a check);
- scoring uses replicas 1–3.

The estimators are the unchanged eval68 ones (unbiased cross-replica, jackknife, upper 95 %).

**Stage 3 gate** (predeclared, per G1 restart). A restart passes when all of the following hold:
- **valid:** structural positivity and passivity.
- **training boundary:** on TRAIN_NEW_rest and TRAIN_NEW_pt0, either abs_coarse ≤ 0.10 and con_coarse ≤ 0.25, or both metrics ≤ 0.8 × G0.
- **dev boundary:** on BND_DEV (= BND_NEAR + BND_FAR + UNS), G1's upper bound is below G0's lower bound for both metrics.
- **interior:** ≤ 1.25 × G0 on TRAIN_KEEP_rest/pt0, INT_DED, REM_NEAR and FP_rest.

The gate passes when at least 2 of 3 restarts pass. The Stage 3 basis is the passing restart with the lowest fitting objective.

**Classes:**
- **A:** the gate passes and G1 meets the WL59 gates on the boundary groups.
- **B:** G2 restarts are ≤ 0.8 × every G1 restart.
- **C:** none of the improvements holds.
- **D:** the restart spread is > 10 %, or a convex problem still moves > 1 % when its budget is doubled.

The verdict mapping is in the protocol.

**Stage 3** (only if the gate passes). A74_5901/5902: the WL73 AR recipe exactly (W+UV + L_sym, P_590x, S16-R, 20,000 steps), with the frozen G1 A/O.

**Stage 4** (only if Stage 3 ran). The geometry-only population was frozen now, before any evidence:
- **Exclusions:** every previously scored source position and its exact images (WL66–72 referenced, S16-R, all 29 WL73 confirmation ports).
- **Static rest ports (slots 100–109):** TBE, TBC, SM, SR and INT, each near and far: ports 3485, 224, 3814, 43, 410, 260, 871, 1568, 1716, 976.
- **Pose:** cf1, chosen by the WL73 pose rule with te0 added to the probed poses. The archived file is sha256 `dcf780ae…`, from LabServer69.
- **Footprint ports (slots 110–115):** 2059, 2070, 2403, 1690, 2468, 2033.
- **References:** 1,408 packets with K = 4 and 32M photons, using the unchanged walker.
- **Decision:** three-valued, as in WL73. A74 vs AR on BND4_FAR and BND4_NEAR, with regression measured on INT4 and TRAIN_KEEP.
- **Confound:** cf1 lies 4.4 mm from the pt0 training centre, so DEF4 is a near-training-pose check.

**Limits stated before evidence:**
- Oracle values are achieved optimisation results, not certified lower bounds.
- G0/G1/G2 differ in parameter count and optimisation budget.
- Oracle T are source-specific latent variables, never a learned rule.
- Volume results say nothing about glossy transport.

**Pre-freeze dev smoke (`~/rest_ato74_dev`; not evidence).** It found three defects, all fixed before freezing:
- **The rank-16 duplication test was too strict.** It compared float32 softmax outputs with a 1e-12 tolerance, which float32 cannot meet. The softmax is now computed in float64, and the identity holds to 1.7e-16.
- **History handling.** `ato74.fista_T` treated the family68 FISTA history as a list.
- **Dev path.** Stage 3/4 is now forced in dev only, so the full path is exercised.

After the fixes the whole chain completed:
- 4/4 tests;
- 7 bases;
- Stage 1, Stage 2 and the decision;
- forced Stage 3 fits;
- 1,408 Stage 4 packets;
- eval4 and export.

Its labels carry no information, because it ran 2 rounds of 5 iterations.

## Execution log

**2026-11-05.** The design was frozen at aa80fac. `run74.sh` started on LabServer63 at 06:12 UTC. The dev smoke re-check passed. In the evidence chain, `ports4` (Stage 4 lists and the cf1 identity: node exact, β rel L2 0.105) and the tests passed.

The first two bases (G0_r0, G1_r0) then stayed in their final T polish for more than 55 minutes at 100 % GPU. The 30 G1 rounds had taken about 5 minutes. No basis file had been written. The chain was stopped at 07:13 UTC.

**Amendment 1** (pre-evidence, disclosed in protocol `amendments`).

**Cause of the stall.** In float32, FISTA's backtracking condition cannot be met once the objective reaches its float32 / nondeterministic-reduction noise floor, so the step size never settles. The 5-iteration dev smoke never reached that floor.

**Changes:**
- All Stage 1 basis fits now run in float64, as Stage 2 already did.
- A scratch float64 run of G0 on the fitting data showed slow convex convergence: J fell by 12 % between 200 and 600 polish steps. This is non-evidence: only the fitting-replica objective, no held-out score. The budgets were therefore raised:
  - polish: 1,000 → 3,000;
  - G0 receives the same total T iterations as G1/G2 (3,900), with a convergence check of the same budget again;
  - Stage 2: 1,500 → 3,000 iterations, plus the same again as its check.
- Rule D is unchanged.

**2026-11-05, OOM restart (execution only).** In the first float64 attempt, G1_r0 ran out of GPU memory (CUDA OOM). With two lanes, float64 training targets for both geometries exceed 12 GB. G0_r0 continued alone and finished at 07:58 UTC.

A detached relauncher (`relaunch_lanes1.sh`) then reran the wrapper with `R74_LANES=1`:
- Completed outputs were skipped.
- No rule, budget or result changed.
- GPU utilisation was 99–100 % with one lane, so a second lane would not have added throughput.

**Chain timing** (07:58–15:58 UTC, one lane):

| Stage | Duration |
|---|---|
| Bases | G0 2,424 s; G1 2,541–2,551 s; G2 2,178–2,179 s |
| Stage 2 | 1,784–2,224 s per basis |

FP64 throughput on the RTX 3080 Ti is about 1/64 of FP32. The float64 choice of amendment 1 therefore made every T solve slow, although GPU utilisation read 99 %.

A float32 solver with a float32-aware backtracking tolerance would have been faster, but it would have been a change to the reused family68 solver. This trade-off is recorded as a lesson, not as a change.

## Final report (2026-11-06)

The evaluation follows the frozen protocol (aa80fac) plus amendment 1, which changed the precision and the budgets before any evidence. Nothing was rerun or reselected.

All values are worst-channel upper 95 % bounds (eval68 pooled).
- **Stage 1:** S16-R training sources. T is fitted on fitting replicas 0–3 and scored on held-out replicas 4–7.
- **Stage 2:** WL73 confirmation sources (development). T is fitted on replica 0 and scored on replicas 1–3.

Every number is a LABELLED ORACLE: per-source T is latent fitted data, not a learned rule.

### Implementation Fact

- **Operator.** All three arms use the same operator, objective, constraints, references and estimators:
  - G0: the fixed WL68 rank-8 A/O.
  - G1: a refit shared rank-8 A/O.
  - G2: a refit shared rank-16 A/O.
- **Validity.** All 7 bases pass the structural checks: A columns and O rows sum to 1, all values are nonnegative, and T ≥ 0 with column sums ≤ 1, so Σ_{j,o} K ≤ d.
- **Parameters.**
  - Shared logits: 1,920 at rank 8, 3,840 at rank 16.
  - Per-source latent T: 2.95M floats at rank 8, 5.90M at rank 16, for the 32 training matrices.
- **Convex adequacy.** G0's T problem changed by 1.6e-16 when its budget was doubled. Stage 2 T problems changed by ≤ 1.1e-5. The per-source convex parts are converged.
- **Restarts.** G1 and G2 each had 3 restarts: restart 0 from the WL68 bases (for G2, duplicated WL68 plus noise), restarts 1 and 2 from random logits.
- **Stages 3 and 4 did not run,** because the gate failed.

### Measurement

**1. Fitting objective J** (fitting replicas, both geometries):

| Basis | J |
|---|---|
| G0 | 0.0431 |
| G1_r0 / r1 / r2 | 0.0381 / 0.0726 / 0.0838 |
| G2_r0 / r1 / r2 | 0.0146 / 0.0203 / 0.0212 |

Restart spread: G1 120 %, G2 45 %.

**2. Stage 1, training sources** (held-out replicas 4–7; abs_coarse / abs_fine16 / con_coarse). pt0 equals rest to 3 decimals for every basis on the boundary sources, which lie far from the dent.

| Basis | TRAIN_NEW (4 boundary sources) | TRAIN_KEEP_rest (12 interior) |
|---|---|---|
| G0 | 0.111 / 0.117 / 0.270 | 0.064 / 0.088 / 0.188 |
| G1_r0 | 0.113 / 0.127 / 0.222 | 0.084 / 0.103 / 0.166 |
| G1_r1 | 0.136 / 0.181 / 0.243 | 0.122 / 0.176 / 0.265 |
| G1_r2 | 0.148 / 0.191 / 0.245 | 0.125 / 0.190 / 0.282 |
| G2_r0 | 0.084 / 0.096 / 0.129 | 0.044 / 0.069 / 0.088 |
| G2_r1 | 0.098 / 0.108 / 0.137 | 0.059 / 0.086 / 0.101 |
| G2_r2 | 0.093 / 0.119 / 0.137 | 0.069 / 0.101 / 0.108 |
| WL59 gate | 0.10 / 0.15 / 0.25 | |

**3. Stage 2, WL73 confirmation sources (development).** BND_DEV = BND_NEAR + BND_FAR + UNS, 15 matrices.

| Basis | BND_DEV | BND_FAR | INT_DED | REM_NEAR | FP_te0 |
|---|---|---|---|---|---|
| G0 | 0.093 / 0.110 / 0.258 | 0.104 / 0.114 / 0.267 | 0.072 / 0.088 / 0.178 | 0.062 / 0.086 / 0.121 | 0.083 / 0.089 / 0.192 |
| G1_r0 | 0.103 / 0.120 / 0.231 | 0.119 / 0.123 / 0.228 | 0.097 / 0.104 / 0.156 | 0.076 / 0.100 / 0.103 | 0.101 / 0.103 / 0.160 |
| G1_r1 | 0.106 / 0.179 / 0.233 | 0.113 / 0.178 / 0.246 | 0.144 / 0.177 / 0.279 | 0.125 / 0.173 / 0.239 | 0.138 / 0.171 / 0.280 |
| G1_r2 | 0.113 / 0.192 / 0.236 | 0.122 / 0.192 / 0.247 | 0.156 / 0.190 / 0.288 | 0.148 / 0.190 / 0.258 | 0.149 / 0.184 / 0.293 |
| G2_r0 | 0.076 / 0.092 / 0.157 | 0.087 / 0.093 / 0.167 | 0.053 / 0.069 / 0.086 | 0.045 / 0.068 / 0.068 | 0.058 / 0.069 / 0.092 |
| G2_r1 | 0.079 / 0.104 / 0.154 | 0.086 / 0.103 / 0.158 | 0.074 / 0.086 / 0.096 | 0.070 / 0.086 / 0.080 | 0.078 / 0.086 / 0.104 |
| G2_r2 | 0.083 / 0.116 / 0.154 | 0.088 / 0.116 / 0.153 | 0.082 / 0.102 / 0.109 | 0.080 / 0.104 / 0.082 | 0.082 / 0.100 / 0.102 |

**4. Per context** (median per-port ratio to G0, all Stage 1 + 2 matrices; con_coarse / abs_coarse):

| Context | G1_r0 | G2 (three restarts) |
|---|---|---|
| TBE | ×0.86 / ×0.90 | ×0.56–0.58 / ×0.73–0.74 |
| TBC | ×0.88 / ×1.14 | ×0.51–0.56 / ×0.85–0.94 |
| SM | ×0.76 / ×1.02 | ×0.47–0.52 / ×0.74–0.94 |
| SR | ×0.82 / ×1.11 | ×0.50–0.56 / ×0.81–0.96 |
| SV | ×0.94 / ×1.06 | ×0.63–0.66 / ×0.73–0.95 |
| interior confirmation | ×0.86 / ×1.21 | ×0.52–0.63 / ×0.69–1.21 |
| interior training | ×0.95 / ×1.16 | ×0.61–0.75 / ×0.86–1.01 |

**5. Per-port structure at G0.** Interior rest confirmation ports fall into a few classes with identical oracle errors (to 3 decimals) in every basis:
- **Contrast about 0.237**, abs about 0.101: ports 80, 83, 84 and 62. Their sub-cell index within the 1 cm coarse cell is a corner, (3,3), (0,3) or (3,0).
- **Contrast about 0.13**, abs about 0.065: ports 76, 79, 81 and 82. These sit in edge sub-cells, (3,2) or (2,3).
- **Contrast about 0.113:** ports 77 and 78, sub-cell (1,3).

Interior ports in corner sub-cells therefore have G0 contrast errors as large as the boundary ports (0.26–0.32). G2 reduces every class to about 0.07–0.12.

**6. Decision (predeclared).**
- The Stage 3 gate is NOT passed. 0/3 G1 restarts pass:
  - G1_r0 improves contrast 18 % on TRAIN_NEW but not abs_coarse (0.113 vs 0.111);
  - its BND_DEV upper bounds are not below G0's lower bounds in abs_coarse;
  - it regresses on interior TRAIN_KEEP and INT_DED;
  - r1 and r2 regress everywhere.
- Formal rows:

  | Rule | Result |
  |---|---|
  | G1 better than G0 | NO |
  | G2 better than G0 | NO: 20 % in both metrics is not met on abs_coarse for 2/3 restarts, nor on BND_DEV abs (0.076–0.083 vs 0.8 × 0.093 = 0.074) |
  | G2 better than G1 | NO, for the same reason |
  | Optimisation uncertainty | YES (G1 restart spread 120 %) |

- Classes: **C (neither improves sufficiently by the predeclared rule) and D (optimisation uncertainty).**
- Descriptively, every G2 restart meets all three WL59 gates on TRAIN_NEW and BND_DEV (G2_r0 and r1 also pass the full Stage 3 gate rules; r2 misses the REM_NEAR interior rule). It also lowers interior errors.

### Observation

1. **Refitting the shared rank-8 basis does not remove the boundary shortfall.**
   - From the WL68 initialisation (G1_r0), the fitting objective drops 12 % and contrast improves 10–18 % on the boundary groups. Absolute and fine16 errors do not improve, and interior errors rise 15–35 %: a redistribution of fixed rank-8 capacity.
   - From random initialisations, rank 8 converges to much worse bases (J +68 % to +94 % vs G0).
2. **Rank 16 changes the attainable class qualitatively.**
   - Every G2 restart, including both random ones, reaches a lower fitting objective (by 51–66 %) than the best rank-8 basis.
   - Contrast falls by about half in every context: boundary, interior and the te0 footprint.
   - Absolute errors fall 10–30 %, depending on the restart.
   - All boundary groups then meet the WL59 gates as an oracle.
3. **The G0 contrast error is not boundary-specific.** Interior ports at coarse-cell corner sub-positions show the same G0 contrast (0.237) as boundary ports. The per-port oracle error depends on the source's position relative to the 1 cm coarse grid.
4. **The per-source convex problems are converged.** The uncertainty is confined to the shared-basis fit at rank 8 (restart spread 120 %). At rank 16 the spread is 45 %, and every restart lies far below every rank-8 restart.
5. **Training-source and development-source oracle results agree in direction and size.** Stage 1 (held-out replicas of the fitting sources) and Stage 2 (other sources) show the same G0 / G1 / G2 ordering.

### Interpretation

**A. Does refitting a shared rank-8 A/O improve physical representability?**
Not materially. The best rank-8 refit trades contrast for absolute and interior accuracy, and fails the predeclared gate. The rank-8 shared basis is also hard to optimise: the random restarts land far worse.

**B. Does rank 16 add meaningful physical representability beyond rank 8?**
Yes, descriptively, and robustly across restarts. Contrast roughly halves everywhere, and every boundary oracle group meets the WL59 gates. The predeclared 20 %-in-both-metrics rule is not met for the absolute metric on 2/3 restarts, so the formal class is not B. G2 is a diagnostic capacity bound with twice the latent T per source, not a deployable result.

**C. Is the rank-8 ceiling reliable or optimisation-limited?**
- The per-source convex parts are converged.
- The rank-8 shared-basis fit is optimisation-sensitive (spread 120 %), so the true rank-8 optimum is not certified. G1_r0 is the best rank-8 point found.
- Every rank-16 restart beats every rank-8 restart by a large margin, including random against WL68-initialised. A capacity limit at rank 8 is therefore the better-supported reading, but optimisation cannot be excluded as a contributor.

**D. Does the improved basis benefit an actual learned K?**
Not tested. The gate failed, so Stages 3 and 4 did not run.

**E. What portion of the remaining failure is spatial mapping / generalisation?**
- In WL73, the learned models' BND_FAR contrast was 0.87–1.30, against this batch's G0 oracle of 0.267 on the same sources: about 3.3–4.9×.
- The angular interface therefore bounds the learned far-boundary contrast at about 0.27 (G0) or about 0.16 (G2). Most of the learned error, about 70–80 % of the contrast bound, lies above that bound and is spatial mapping / generalisation.
- Near supervised sources (WL73 BND_NEAR, 0.42–0.51) the learned error is closer to the oracle, but still about 1.6–1.9× above it.

**F. Retaining the global ATO factorisation.**
- On this fixture, a single global angular basis shared by all sources can represent the tested first-exit responses at the WL59 quality level as an oracle, but only with more than 8 mixtures. At rank 8, the frozen WL68 basis is close to the best attainable compromise found.
- The angular interface is a real secondary constraint, concentrated in directional contrast and not specific to boundaries. It does not explain the dominant learned-K failure, which is spatial transfer.

### Unresolved Question

- **Rank-8 optimum.** Whether a better rank-8 optimum exists, given the optimisation sensitivity.
- **Minimal rank.** The minimal rank between 8 and 16 at which the WL59 gates are met (no sweep was authorised).
- **Learned K at rank 16.** Whether a rank-16 learned K would transfer better or worse, since the spatial map would have to produce twice the mixtures.
- **Coarse-grid dependence.** Why the oracle error depends on the source's sub-position within the 1 cm coarse cell. A plausible mechanism is that T has no incoming-bin index, so an incoming-direction-dependent lateral shift of the near-field exit footprint across coarse-cell boundaries can only be expressed through global mixtures. This is not tested.
- **Beyond this fixture.** Glossy or heterogeneous media are untested.

### Architecture Verdict

**Per the predeclared mapping: ANGULAR CHANGE DOES NOT ADDRESS PRIMARY ERROR (class C) with INCONCLUSIVE elements (class D: rank-8 basis optimisation uncertainty).**

Descriptively, and not as a formal verdict:
- **FIXED BASIS LIMIT SUPPORTED** is not established: the rank-8 refit gives no robust gain.
- **HIGHER ANGULAR CAPACITY INDICATED** holds for oracle representability. Rank 16 meets the WL59 gates on every boundary and development group in all three restarts and halves contrast everywhere, but misses the predeclared 20 % absolute-error margin on two restarts.
- The dominant learned-K error remains spatial mapping (E).

No architecture is adopted, and no successor is authorised.

### Reproducibility

- **Code:** `experiments/fnto_ato74/`, protocol `protocol/ato74_v1.json`. Frozen at aa80fac; amendment 1 at 1bce529 (float64, budgets). The execution-only relauncher (`relaunch_lanes1.sh`, `R74_LANES=1`) is on the server.
- **Server:** LabServer63, root `~/rest_ato74` (code_commit.txt `1bce529`).
- **Inputs:** the WL73 references and fit_targets73 (read-only), and the WL68 net_all_5902 logits. The Stage 4 lists and the cf1 geometry (sha256 `dcf780ae…`) are frozen but were not used.
- **Restart seeds:** G1 7410–7412, G2 7420–7422.
- **Outputs:**
  - `results/fnto_ato74/`: `stage1/scores.json`, `stage2/*.json`, `decision.json`, `final/` (bases.csv, per_matrix.csv, ato74.png/pdf, summary, manifest);
  - reviewer copies in `results/evaluation/74/`.
- **Not exported:** the bases themselves (logits and latent T), which stay on the server at `~/rest_ato74/bases/` (7 files).
