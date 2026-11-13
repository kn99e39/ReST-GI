# Worklog — ECRO Source/Receiver Factor Separation Smoke Test (WL93B) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93B Source/Receiver Factor Separation Smoke Test" (user, 2026-11-13).
- **Batch class.** Suffix batch continuing the WL93 architecture question with WL93A evidence. It reuses the existing WL93 tied, untied and capacity-matched implementations at the WL93A smoke scale; it adds no new architecture class or mechanism.
- **Preserved.**
  - WL93 STOPPED-INCONCLUSIVE (its Amendment 1 is still pending approval).
  - WL93A NO_SIGNAL_AT_SMOKE_SCALE (freeze `888ceb6`).
  - All earlier checkpoints and artifacts.
- **Question.** At smoke scale, does source / receiver factor separation alleviate the source-stage versus whole-transport accuracy conflict seen in WL93A, beyond parameter-capacity effects?
- **Locations.**
  - Branch `Selective_Recompute`, LabServer63.
  - Code: `experiments/ecro_stage93b/`. It reuses `ecro_stage93/model93.py` (unchanged), `ecro_stage93a` (references, targets, training contract, metrics) and the WL92 render / reference tools.
  - Results: `results/ecro_stage93b/` and `results/evaluation/93B/`.
  - Server root: `~/rest_x93b`. The WL93A targets and references are read-only links to `~/rest_x93a/data`.

## Protocol (`experiments/ecro_stage93b/protocol/ecro_stage93b.json`)

**Inspection disclosure.** WL93A had already measured and inspected T0 / TQ. The quality of U0 / UQ / C0 / CQ has never been observed: WL93 stopped before training, and its dev chain is not evidence.

**Arms.** The model code is `model93.Local` (WL93 freeze `c30c797` definitions).

| Arm | Factors | Objective | Origin |
|---|---|---|---|
| T0 | tied | OWN | WL93A checkpoint (verified) |
| TQ | tied | OWN + WL93 source-stage (w = 1) | WL93A checkpoint (verified) |
| U0 | untied (receiver heads copied from the source heads) | OWN | trained here |
| UQ | untied | OWN + source-stage | trained here |
| C0 | capacity-matched tied (shared factor softplus(head + head2), head2 hidden layers from seed + 9300, zero output layer) | OWN | trained here |
| CQ | capacity-matched tied | OWN + source-stage | trained here |
| R_T0 / R_TQ | tied | as T0 / TQ | retrained here: reproduction controls and run-to-run variation |

**Training contract.** The WL93A contract is unchanged:
- Data: the WL93A 128-geometry subset, its B0 training targets and its training-validation targets.
- Schedule: 2,000 steps; lr 3e-4 → 1e-5 cosine with 200 warmup steps; clip 10; data order rng(seed + 93); seeds 8601 / 8602.
- The global term is frozen to WL92A own_s<seed> and is identical across arms (tested).

**Reproduction gate.**
- R_X must match the WL93A X within 2 % relative on validation whole, single-vs-L2 and source-stage (training-validation target) relative L1. Otherwise STOP: historical controls not reproduced.
- The reused checkpoints must load with the recorded state hashes (T_REUSE).

**Accounting** (validation primary; train-representative and E2 for context; WL93A B5 references; g00010 in every full-population number).

| Stage | Measure |
|---|---|
| Source local use | Bq vs B5. Primary: relative L1 vs the B5 mean (SRC). Also the cross-product relative L2 M (the square root of a ratio of two unbiased cross-product estimates, not itself exactly unbiased), flux / directional split and peak retention |
| Receiver local use | a_r c_r applied to the first flight of the physical q̄_B5, vs t2 (exact frozen-ER floor reported). Also the receiver use on the model's own first flight, vs M64 |
| Single flight | S vs t2 (SINGLE), plus cross-product L2 with the f2 / f2b halves |
| Global (frozen) | G vs t − t2 |
| Whole | S + G vs t (WHOLE, primary), plus cross-product L2 with the f / fb halves |
| Operators | E2 T1 panel source / receiver matrices vs M64: mirror retention, and the source − receiver difference |

**Uncertainty.**
- Paired geometry bootstrap of pooled differences (2,000 resamples, seed 93002).
- Run-to-run variation r_m from R_X vs X.
- margin_m = max(2 SE, r_m).

**Criteria (per seed s).** Let gap_s = WHOLE(TQ) − WHOLE(T0) and gain_s = SRC(T0) − SRC(TQ).
- **(a) Source benefit preserved.** SRC(UQ) ≤ SRC(T0) − 0.5 gain_s, and SRC(UQ) ≤ SRC(TQ) + margin.
- **(b) Whole improved vs TQ.** WHOLE(UQ) ≤ WHOLE(TQ) − max(margin, 0.5 gap_s), and SINGLE(UQ) ≤ SINGLE(TQ) + margin.
- **(c) Whole improved vs CQ.** WHOLE(UQ) ≤ WHOLE(CQ) − max(margin, 0.5 gap_s).
- **(d) No compensating regression.** Rendered stationary and D E_H ≤ 1.25 × T0 + 0.002 on E2 T0 / T1, E5k995 T0 / T1 and E1 LSPOT, and no WL86 gate lost vs T0.

**Labels** (first match wins):
1. SHARING_SIGNAL: a, b, c, d for both seeds.
2. CAPACITY_EXPLANATION: a and b for both seeds, c fails for at least one seed, and CQ meets b for both seeds.
3. TRADEOFF_REMAINS: a for both seeds, b fails for both seeds.
4. NO_ADVANTAGE: b fails for both seeds, a fails for at least one seed, and U0 does not improve whole over T0.
5. INCONCLUSIVE: anything else, or a control failure.

Contrasts U0−T0, UQ−U0, CQ−C0, C0−T0 are reported descriptively. They are not additive shares.

**Renders (inspected conditions only, diagnostic).**
- Tags: E2 T0 / T1, E1 T1 LSPOT, E5k995 T0 / T1, and the pattern diagnostic.
- Models: 6 arms × 2 seeds, plus WL92A own (which checks reproduction of the WL93A renders).
- Sampling: 2 × 4,096 spp, seeds 86600 / 86601.

**Pattern diagnostic (visual only; never in the verdict).**
- **Interface limit.** The source interface (`scene86.light_arrays`) admits at most **two lights**. A multi-cell checkerboard is therefore not expressible without changing the source contract, so the diagnostic is reduced to a **2 × 2 checker**.
- **Layout.** Two hard-edged cone lights (the existing cone class, as in LSPOT) illuminate the two diagonal 40 cm cells of a 2 × 2 cell block on the panel. The cell edges lie on the 20 cm patch grid. The two other cells stay unlit.
- **Flux.** Equal to the E2 light's panel-incident flux.
- **Reference.** New REF-ER, 8 × 8,192 spp.
- **Representability gate (before any render).** From the deployed encoding, every lit-cell patch must be ≥ 0.3 × the lit maximum and every other panel patch ≤ 0.05 × it.
- **Not run.** No above-Nyquist stress pattern.
- **Caveat.** The cone source class was never used in training.

**Tests.** T_REUSE, T_INIT, T_GLOBAL, T_OWNERSHIP, T_CAPACITY (including non-degeneracy after training), T_TARGETS, T_SPLIT (geometry content hash), T_DET, T_RENDER_REF.

## Dev chain (before the freeze; not evidence)

`~/rest_x93bdev` with `--dev`: 100 steps, 64 spp, the full WL93A references.
- Every stage ran: reuse, pattern, tests, 12 trainings, accounting, operators, render tables, LPAT reference, renders, analysis and visuals.
- Expected dev-only failures: the reproduction gate and render-reference consistency fail at 100 steps / 64 spp.
- T_REUSE passes. Pre and post tests pass. Parameter counts are tied 405,424, untied = tiedcm 487,952. Non-degeneracy C0 vs T0 is 0.22.
- **Pre-freeze correction.** The first pattern layout (lights straight in front of the cells, cone radius 0.17 m) failed the representability gate: the moved box B at T1 completely shadows the second cell (all four patch levels 0).
  - Corrected layout: the lights sit 0.8 m toward −y (oblique, clearing box B) with cone radius 0.13 m.
  - The gate then passes. Lit patches are at 0.58–1.0 of the maximum, dark patches at 0, panel flux ratio vs the E2 light 1.000, region C 1,527 px.
  - This was decided before any evidence run; the only model outputs on the pattern were dev 100-step / 64-spp ones.
- A close-up figure of region C was added to the visuals.

**Probes.** The stage costs are known from WL93A and WL92B, which use identical code paths:
- training: 4 concurrent runs at ≈ 82 % GPU, ≈ 3.5 min per batch of 4 (12 runs ≈ 11 min);
- accounting (18 models × 73 configurations): ≈ 2.6 min (dev measurement, same size);
- REF-ER: pool 6 (8 × 8,192 spp);
- renders: pool 5 (168 renders).
