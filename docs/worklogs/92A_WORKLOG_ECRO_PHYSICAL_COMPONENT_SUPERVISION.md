# Worklog — ECRO Physical Component Supervision: Controlled Generator Learning and Held-out Sharp Interreflection (WL92A) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93 ECRO Physical Component Supervision: Controlled Generator Learning and Held-out Sharp Interreflection" (user, 2026-11-13). By the user's instruction it is run and recorded as **WL92A**, not 93.
- **Context.**
  - Human decision 2026-11-13: ECRO is the central learned-global-transport development base.
  - WL92's verdicts are unchanged: gate `STOP_INTERVENTION_G_rep` (the designated OWN-k16 fit did not converge).
  - WL92's separately converged OWN-r1 fixed-geometry fits (E2 T1 / T0 region-C E_H 0.041 / 0.024) are descriptive prior evidence, not a gate result. This batch is a new, independently controlled learning experiment.
- **Question.** Under an unchanged ECRO representation, does physically exclusive component supervision produce a demonstrable improvement in learned sharp interreflection and dynamic geometry generalisation over the original SUM-only training contract?
- **Preserved.** WL84 ER; the WL86 architecture (735,664 parameters; R 16; global rank 24; 1,270 × 64 table; 512 state rays; 16-point source encoding; LOW + lookup readout) and its training contract; WL86 data, checkpoints, references, thresholds and verdicts; WL86A–WL92 code and results (imported and called unchanged).
- **Changed only.** The training-target decomposition (L₂ / L₃₊) and the objective; instrumentation and evaluation.
- **Where.** Branch `Selective_Recompute`, LabServer63. `experiments/ecro_own92a/`, `results/ecro_own92a/`, `results/evaluation/92A/`. Server `~/rest_x92a`, reading `~/rest_ecro86` and `~/rest_x92`.
- **Terminology.** Path tracing and explicit transport are physical reference / null comparator only. No external neural baseline; no Neural Full Recompute claim.

## Contract (`experiments/ecro_own92a/protocol/ecro_own92a.json`)

**Arms.** FROZEN (WL86 checkpoints, reference only), SUM (fresh control), OWN (intervention). Seeds 8601 / 8602 for SUM and OWN.

**Objectives (fixed before any training outcome).**
- den = t.mean(over bins) + 0.05 · t.mean(over everything) — the `model86.loss_fn` denominator of the total target t.
- SUM: L = mean(((B_single + B_global − t) / den)²) = `model86.loss_fn`.
- OWN: L = SUM term + mean(((B_single − t₂) / den)²) + mean(((B_global − t₃) / den)²), weights 1 / 1 / 1 (the WL92 OWN objective, unchanged).
- t = stored WL86 target; t₂ = regenerated exactly-2-event part with the WL86 generator seeds and samples; t₃ = t − t₂ (no clamping).

**Training.** The train86 contract: Adam, lr 1e-3, warm-up 500, cosine to 1e-5, 24,000 steps, one geometry (4 lightings) per step, gradient clip 10; init and data order from the seed. The initial state dict must be identical across SUM and OWN for each seed (T_INIT). No hyperparameter search, no repeated seeds.

**Training population and held-out disjointness.**
- The WL86 exclusion rule (box within 0.15 m, light within 0.3 m) applied to E4's boxes and lights, for both arms.
- Removes 96 of 1,024 training geometries, 9 of 64 validation, 2 of 32 high-quality training geometries (light rule dominant: four E4 lights × four lights per geometry).
- **Disclosed confound.** The fresh SUM / OWN population differs from the frozen checkpoints' (9.4 % fewer training geometries). The primary comparison is SUM vs OWN on the identical population; the frozen checkpoints are a reference. T_CTL requires the fresh SUM's validation error within 15 % of the frozen checkpoint's on the same reduced validation set.

**Component targets.** WL86A split tracer, WL86 generator seeds and sample counts (256; 512 + 512). T_F2_SUM: for every lighting of every used file, the sum of the regenerated parts against the stored target, relative L1 ≤ 1e-3 and flux within 1e-3.

**Held-out E4.**
- WL87 candidate E3b, chosen by WL92 from geometry only. No radiance existed before this batch.
- References: REF-ER / REF-P 8 × 8,192 spp at T0 / T1; sources LMIX4 / LSPOT4 4 × 8,192 spp.
- Admission: WL86 stage-0 criteria (Q1–Q4, Q6, Q7) with unchanged thresholds. Not admitted → no held-out verdict (outcome D for that question).

**Renders.** 2 × 4,096 spp, WL86 ECRO render seeds 86600 / 86601. Ten configurations: E1 T0 / T1, E2 T0 / T1, E1 LMIX / LSPOT, E4 T0 / T1, E4 LMIX4 / LSPOT4. Controls: STALE (T0 state, current source) on E2 T1 and E4 T1; ORACLE table and explicit X_EXP on E4.

**Criteria (prospective).**

| Criterion | Rule |
|---|---|
| OWNERSHIP_IMPROVED | both seeds: OWN validation field relative L1 of B_single vs L₂ ≤ 0.75 × SUM's, and of B_global vs L₃₊ ≤ 0.75 × SUM's |
| PANEL_LOCAL_IMPROVED (mechanism) | both seeds: panel `a cᵀ` mirror-bin flux retention OWN ≥ 1.5 × SUM (E4 T1 geometry) |
| HELDOUT_SHARP | E4 admitted; both seeds: E4 T1 region-C E_H of OWN ≤ 0.06 and ≤ SUM − max(noise bound, 0.005) |
| NO_REGRESSION | both seeds: no WL86 gate (F1, F2 A / D, F4) passed by the same-seed SUM is lost; stationary and D E_H of OWN ≤ 1.25 × SUM + 0.002 on every T1 and source configuration |
| DYNAMIC | both seeds: F3 (region D) not lost on E1, E2, E4; region-C edit response on E4 reported |
| SOURCE | F4 on E4 LMIX4 / LSPOT4 |

**Outcome mapping.**
- D STOPPED: a test or control fails (T_REPRO, T_F2_SUM, T_INIT, T_DISJOINT, T_CTL), a non-finite loss occurs, or E4 is not admitted.
- C NOT SUPPORTED: OWNERSHIP_IMPROVED fails.
- A SUPPORTED: OWNERSHIP_IMPROVED, HELDOUT_SHARP, NO_REGRESSION and DYNAMIC all hold.
- B PARTIALLY SUPPORTED: otherwise.
- Hypotheses H1–H5 are mapped descriptively (protocol).
- Two seeds give limited confidence; no robustness claim beyond them.

**Economics.**
- Offline: target regeneration and training wall per arm.
- Runtime: factor generation, application, transfer, state and source per model on E4. The architecture is identical by construction.
- Quality ladder on E4 T1 only if outcome A.

## Pre-freeze development (non-evidence)

- **Dev chain** (`~/rest_x92adev`; 64-spp references and renders, 32-spp targets, 48 training geometries, 300 steps). Its numbers are not interpreted.
- **Pre-tests on the real inputs** (deterministic; re-run in the evidence chain):
  - frozen checkpoints reproduce the WL86 tables exactly;
  - E4 state: selective = full = fresh scene, null edit identical;
  - exclusion margins hold (minimum kept distances 0.153 m box, 0.302 m light).
- **Dev chain completed end to end** (every stage rc = 0; dev stop labels T_F2_SUM and T_CTL are the expected consequences of the reduced sample counts and the 300-step training).
  - T_INIT: identical initial state dicts for SUM and OWN per seed.
  - T_LINEAR, T_OWNERSHIP_CODE, T_REPLAY pass for every model.
  - No non-finite step.
  - The dev-run measurements are not interpreted (300 steps, 48 geometries, 64-spp references).
- **Fixes from the dev run (before the freeze).**
  - The timing stage listed its own process as foreign.
  - The ladder recorded no GPU kernel time (kernel-history flag not set).
- **Throughput (dev, four trainings concurrently):** about 100 ms per step for both objectives. Hence about 40 min per training at 24,000 steps; OWN costs no measurable extra time per step. The full component-target regeneration runs in the WL86 shard layout (8 × 128 train, 4 × 16 val, 4 × 8 high-quality), concurrently with the E4 references. Expected chain wall: about 1 h 45 min.
