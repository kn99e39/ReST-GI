# Worklog — ECRO OWN Sharp Fidelity and Source Regression: Causal Attribution and Independently Qualified Validation (WL92B) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL92B ECRO OWN Sharp Fidelity and Source Regression: Causal Attribution and Independently Qualified Validation" (user, 2026-11-13).
- **Context.**
  - WL92A's verdict stays **D_STOPPED**: E4 failed Q4 (SNR 3.9 vs 4.0) and remains non-admitted.
  - Its component-ownership improvement is valid within its measured scope.
- **Question.** After physically identifiable component supervision improves ECRO's learned transport ownership, what mechanism limits its remaining sharp-response fidelity and source / geometry generalisation, and is the improvement reproducible on an independently qualified edit?
- **No learning.** Every model parameter is frozen: WL86 ecro_s8601 / s8602; WL92A sum / own seeds 8601 / 8602. No training, fine-tuning, objective, rank, encoding, bin, ray-budget, ER or threshold change.
- **Where.** Branch `Selective_Recompute`, LabServer63. `experiments/ecro_cf92b/`, `results/ecro_cf92b/`, `results/evaluation/92B/`. Server `~/rest_x92b`, reading `~/rest_ecro86`, `~/rest_x92`, `~/rest_x92a`.
- **Terminology.** Oracle substitutions are non-deployable diagnostics. A counterfactual improvement is never counted as a learned-model improvement. Explicit transport is a null comparator only.

## Design (`experiments/ecro_cf92b/protocol/ecro_cf92b.json`)

**Reproduction first.** The WL92A tables of every model (E2 T1 / T0, E1 LSPOT, E4 T1), the rendered E_H values and the validation component errors must be reproduced. Otherwise STOP.

**Counterfactuals.** Each replaces exactly one quantity of a frozen model's table. The original output is always evaluated alongside.

| Name | Replaced | Kept from the model | Question |
|---|---|---|---|
| X_L2 | single-flight term → path-traced 2-event field | global term | share of error in the single-flight term |
| X_L3 | global term → path-traced 3–5-event field | single-flight term | composition / compensation |
| X_M1 | first local use → exact frozen-ER bin matrix M64 applied to the deployed encoding | flight, second local use, global | source-side learned local response |
| X_M2 | second local use → M64 on the model's own first-flight irradiance | first stage, global | receiver-side learned local response |
| X_M12 | both local uses of the single-flight term → M64 | global | learned local response as a whole |
| X_Q | first stage → exact once-scattered table q (no encoding) | flight, second use, global | with X_M1: source information (encoding vs exact q) |
| X_SHQ | source encoding → 1,024-point estimator | everything | source-estimator sensitivity |

- X_M*, X_Q: diagnostic compositions (sphere patches use the flat-frame M64, as WL87 declared).
- X_L2, X_L3: a component replaced by its physical target.

**Attribution rules (fixed before evidence).**
- f(X) = (E_orig − E_X) / (E_orig − E_floor), with E_floor = E_H of the path-traced table.
- DOMINANT: f ≥ 0.5 for both OWN seeds.
- H_local: X_M12 dominant in both E2 cases (region C).
- H_global: X_L3 dominant in both E2 cases.
- H_source: on E1 LSPOT stationary pixels, X_SHQ or the encoding effect (X_M1 → X_Q) ≥ 0.5 for both OWN seeds.
- H_geometry: mean relative flux-norm change of the learned panel `a cᵀ` between E2 T0 and T1 ≥ 0.10 for both OWN seeds. The panel is intrinsically unchanged; a change alone is not a K/C violation.
- H_interact: no dominant counterfactual, but at least two with f ≥ 0.25 for both OWN seeds in a case.
- Also reported: seed-gap closure, the level / directional split of the reading, and the stage-wise field errors (Bq vs q, encoding vs q).

**Held-out E5 (fixed before any candidate radiance).**
- **Candidates.** Ordered, from `default_rng([92550, k])`: T0 from the training distribution; one box moved by ≥ 0.6 m.
- **Lights.** L1 (even k) / L2 (odd k). These are WL86 test lights, excluded from every training light at data generation, so light disjointness holds for all six models. Held-out sources: LMIX, LSPOT.
- **Disjointness.** Every box centre of T0 / T1 at least 0.15 m from the same box of all 1,120 WL86 data files.
- **Geometry rule.** A, C, D, E ≥ 1,000 px and C_footprint ≥ 1,000 px.
- **References.** For the first three geometry-admissible candidates, in order: REF-ER / REF-P 8 × 8,192 spp at T0 / T1 and a truth split.
- **Admission.**
  - WL86 Q1–Q4, Q6, Q7 unchanged (Q4 at SNR 4.0).
  - Sharp family: light → panel → receiver 2-event share of the region-C higher-order reading ≥ 0.08 at T1, with SNR ≥ 4 against its half-split noise.
- **No admission.** If no candidate is admitted, no held-out claim is made.
- **E5 criteria.** OWN consistent advantage: OWN ≤ SUM − max(noise, 0.005) at T1 and T0 for both seeds. Plus NO_REGRESSION and DYNAMIC (WL92A definitions).

**Outcome mapping.**
- A: one dominant mechanism, E5 admitted, consistent advantage, no regression.
- B: one dominant mechanism, E5 admitted, but OWN not consistently better.
- C: interacting or several mechanisms (E5 admitted).
- D: no candidate admitted, baseline not reproduced, or no mechanism isolated.
- Mechanism findings are reported under every outcome.

## Pre-freeze development (non-evidence)

- **E5 feasibility (geometry only).** Of 3,000 generated candidates, 5 satisfy the per-box disjointness against all 1,120 files. Box B, which rests on the floor, is the bottleneck. The population was therefore set to 20,000 candidates.
- **Dev chain** (`~/rest_x92bdev`; 64-spp renders and references, 64-point "high-quality" encoding, disjointness disabled, E5 evaluation branch forced through with a dev-only override that the evidence chain never sets): every stage rc = 0.
  - Reproduction PASS.
  - Tests PASS: counterfactual identity, linearity / non-negativity, event split of E1 LSPOT, disjointness.
  - The analysis and visuals ran through all branches, including E5.
- **Disclosure.** The dev chain renders the real E1 / E2 / E4 conditions with the real models at 64 spp, so noise-dominated counterfactual numbers on those inspected conditions were seen. No rule, threshold, counterfactual definition or candidate rule was changed after that. Only two dev-only overrides were added (`FORCE_E5` in the chain, `WL92B_DEV_FORCE_E5` in the analysis), both inactive without the dev flag / variable.
