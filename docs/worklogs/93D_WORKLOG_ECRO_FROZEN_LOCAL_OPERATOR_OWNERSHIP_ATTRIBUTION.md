# Worklog — ECRO Frozen Local Operator Ownership Attribution (WL93D) (2026-11-14)

## Origin and scope

- **Brief.** "ReST-GI — WL93D Frozen Local Operator Ownership Attribution" (user, 2026-11-14).
- **Batch class.** Suffix batch. It is a mechanism attribution on frozen checkpoints, informing the next architecture decision.
  - No training, no new architecture, no parameter change.
  - IDTS / LSIT are not implemented.
- **Preserved.** These earlier results stay as they were and are not reinterpreted:
  - WL93 STOPPED-INCONCLUSIVE, with Amendment 1 pending.
  - WL93A NO_SIGNAL_AT_SMOKE_SCALE.
  - WL93B SHARING_SIGNAL.
  - WL93C NO_SIGNAL_AT_SMOKE_SCALE.
  - The dual-architecture verification report.
- **Question.** Does geometry-dependent specialisation of ECRO's learned local operator make a measurable causal contribution to sharp interreflection failure, beyond generic representation error, optimisation variance and component co-adaptation?
- **Where.**
  - Branch `Selective_Recompute`, LabServer63.
  - Code in `experiments/ecro_local93d/`. It reuses the WL93 prep / E6 rule, the WL93A references, the WL93B / C accounting helpers and the WL92 render / reference tools, all unchanged.
  - Results in `results/ecro_local93d/` and `results/evaluation/93D/`.
  - Server root `~/rest_x93d`.

## Protocol (`experiments/ecro_local93d/protocol/ecro_local93d.json`)

**Hypotheses.**
- H_G: functional geometry dependence of the learned local operator contributes materially to sharp error.
- H_R: intrinsic representation / optimisation error dominates.
- H_C: co-adaptation prevents attribution by isolated interventions.

**Models (frozen, hash-checked).**
- **own** (primary): WL92A own_s8601 / 8602. The ECRO local factors (c, a) feed both the single flight and the global term.
- **T0** (secondary): WL93A T0_s8601 / 8602. Tied Local; its global term is the frozen own.
- Identity checks:
  - own: file sha256 against WL92B `repro.json`.
  - T0: state sha256 against WL93A `final_sha256`.

**Effective operator.**
- A_i = Σ_r a cᵀ, a 64 × 64 × 3 operator in patch-local bins.
- The physical control is M64, the frozen-ER bin matrix.

**Material groups (stationary patches).**
- S1 panel: 42 patches.
- S2 sphere: 32 patches.
- B floor: every 4th patch.
- LAM: a Lambert negative control of material-3 patches (index rank % 13).

**Task A — functional invariance.** All probes are applied to the same patch-local bins under both geometries.
- Probes:
  - canonical one-hot bins, measured in the WL92 flux-transfer norm;
  - material-level common workload bases: flux-weighted PCA (99 % energy, cap 48) of training-distribution incident fields from the 16 WL93A train-representative geometries × 4 lights. Source use: s16. Receiver use: flight(q_B5).
  - The bases are identical for every geometry.
- Metrics:
  - E_phys: error against M64.
  - D_geo: drift over geometry pairs. The pairs are E2, E5k995 and E6 T0↔T1, plus validation derangement pairs (k, k+1).
  - D_seed: seed-to-seed difference.
  - S_geo: the geometry-varying share of the squared physical error.
  - Error by incident ring.

**Task C — incident-subspace attribution.**
- On evaluation incident fields, the error is split into in-span and out-of-span parts relative to the fixed material-level basis.
- Geometry drift is split the same way.
- Geometry-specific subspaces are secondary.

**Task B — interventions.**
- Arms:
  - C0: deployed operator.
  - C1_src / C1_rcv / C1_both: the partner-geometry learned operator — T0's operator for T1 conditions, the derangement partner for validation.
  - C2_src / C2_rcv / C2_both: frozen M64. C2_both = WL92B X_M12 = WL93C ORA.
- Only the single-flight local uses change. Geometry, state rays, visibility, illumination, source encoding and the global term are fixed; the global term uses the deployed factors.
- Outputs:
  - tables for the validation population (55, g00010 included), train-representative and sharp configurations;
  - renders for E2 / E5k995 / E6: T1 for all 7 arms; T0 for C0 and the C2 arms, so each arm has its own edit response.

**Held-out (Task E).**
- WL93's E6 was frozen in the WL93 protocol (rng 93550, WL86 stage-0 incl. Q4 SNR 4, sharp family ≥ 0.08) and never generated. It is qualified here, unchanged, with `prep93.py`.
- Budget: the frozen 3 geometry-admissible candidates.
- Qualification happens before any model output on E6. If no candidate is admitted, conclusions are limited to inspected conditions.

**Criteria (per family and seed; S1; workload criteria must hold for both the source-use and receiver-use bases).**
- **G1:** D_geo ≥ 0.10 and ≥ 2 × D_geo(LAM).
- **G2:** S_geo ≥ 0.25.
- **G3:** rendered region-C |C1_both − C0| ≥ max(noise, 0.005) on ≥ 2 T1 conditions, and validation single-flight |C1_both − C0| ≥ 2 SE (paired bootstrap, seed 93004).
- **Labels** (each evaluated independently; several may hold):
  - **GEOMETRY_SPECIALIZATION_SUPPORTED:** G1 ∧ G2 ∧ G3, both seeds.
  - **REPRESENTATION_OR_OPTIMIZATION_DOMINANT:** E_phys ≥ 0.30 ∧ S_geo < 0.25 (both seeds), and G3 not met.
  - **COADAPTATION_LIMITED:** any of —
    - C2_both is not better than C0 in region C in at least half of the T1 × seed cells;
    - validation single flight under C2_both is not better than C0 by 2 SE, for both seeds;
    - the C2 interaction |d_both − (d_src + d_rcv)| > max(|d_src|, |d_rcv|), for both seeds.
  - **INCONCLUSIVE:** a control failure, no label, or the two families disagree between G and R.

**Tests.**
- T_CKPT.
- T_M64: canonical-probe response.
- T_IDENTITY: C0 through the substitution path equals the deployed output.
- T_SWAP / T_ISOLATION: per-use factor digests.
- T_GLOBAL: identical in every arm.
- T_FRAME: stationary patch frames identical across paired geometries.
- T_USED: doubling the swapped factors or M64 changes the output with the expected scaling.
- T_DET.
- T_REPRO:
  - C2_both = WL92B X_M12;
  - own deployed renders reproduce the WL93A renders.

## Dev chain (before the freeze; not evidence) and a disclosed pre-freeze exposure

**Run.** `~/rest_x93ddev`, `--dev`.
- 12 configurations: 2 train-representative, 4 validation, the sharp pairs.
- E6 forced to the first geometry-admissible candidate, plumbing only.
- 64 spp.

**Stages.** Every stage ran end to end: checkpoint check, E6 qualification loop, configurations, tests, fields, operators, functional analysis, interventions, render tables, renders, analysis, visuals.
- One code bug was fixed: an indexing error in the per-ring error (weighting applied after slicing).
- Results:
  - CKPT PASS and TESTS PASS.
  - The dev-only render reproduction failure was expected at 64 spp.
  - No dev E6 candidate was admitted at 64 spp (expected).

**Disclosed exposure.** The dev chain printed **Task A functional values computed on the frozen checkpoints for the 6-geometry dev subset plus the sharp pairs** (own S1 canonical: E_phys ≈ 0.80, D_geo ≈ 0.045–0.053, S_geo ≈ 0.003). It also printed the resulting dev label line.
- **Why this matters.** Task A uses deterministic frozen operators, so these are real (subset) values, not plumbing noise.
- **What was already frozen.** Every criterion and threshold was written in the protocol **before** the dev run (the protocol file predates the dev launch). None was changed afterwards; the only post-dev changes are the ring-metric bug fix and this disclosure.
- **What it means for the reader.** The evidence run's Task A values should be read as having been previewed on a subset. Task B / C values (interventions, renders at 4,096 spp, E6) were not previewed. E6 was not qualified in dev.

**Timing.**
- Accounting: 12 configurations in < 1 min; the full 79 are expected to take ≈ 6 min.
- Functional analysis: ≈ 1 min.
- Renders: ≈ 8 s each at 4,096 spp, pool 5, 264 renders in total.
- E6 references: ≈ 20 min per candidate (WL92B measured path), pool 6.
