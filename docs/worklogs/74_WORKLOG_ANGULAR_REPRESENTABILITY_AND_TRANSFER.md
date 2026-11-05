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
