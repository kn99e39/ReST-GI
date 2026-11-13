# Worklog — ECRO Dual-Stage Physical Supervision for a Shared Local Operator (WL93C) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93C Dual-Stage Physical Supervision for a Shared Local Operator" (user, 2026-11-13).
- **Batch class.** Suffix batch. It continues the WL93 / WL93A / WL93B local-operator question. Only the supervision of the existing tied local operator changes: no new representation, no untied factors, no rank / width / encoding / global change.
- **Preserved and not reinterpreted.**
  - WL93 STOPPED-INCONCLUSIVE (Amendment 1 pending).
  - WL93A NO_SIGNAL_AT_SMOKE_SCALE.
  - WL93B SHARING_SIGNAL.
- **Question.** Can ECRO's physically shared local operator learn accurate source and receiver responses at the same time under direct stage supervision? Does that reduce the whole-transport regression seen in WL93A?
- **Where.**
  - Branch `Selective_Recompute`, LabServer63.
  - Code `experiments/ecro_stage93c/`. It reuses `ecro_stage93/model93.py` unchanged, plus `ecro_stage93a` / `ecro_stage93b` helpers.
  - Results in `results/ecro_stage93c/` and `results/evaluation/93C/`.
  - Server root `~/rest_x93c`, with read-only links to the WL93A targets and references.

## Protocol (`experiments/ecro_stage93c/protocol/ecro_stage93c.json`)

**Inspection disclosure.** The WL93A / WL93B results were inspected before this protocol. That includes the exact-local floor of 0.0998: the receiver use on the physical first flight compared with t2. The oracle composition and TR / TQR have never been evaluated on this population. WL92B evaluated X_M12 on E2 / E4 tables only.

**Physical local operator.**
- M64 is the quad87 dense frozen-ER bin matrix per material, in the flat patch frame; sphere patches use the flat-frame matrix, as WL87 declared.
- The exact use is out[i, bo] = Σ_bi E[i, bi] M64[m_i, bi, bo].
- The flight is the deployed state-ray gather with the model's fixed ray weights. These weights are identical in every arm.

**Task A: oracle feasibility.** Non-deployable and involves no learning.
- **ORA_s** = M64(flight(M64(s16))) + G_s. This applies the exact operator at both uses on the deployed encoding, with the frozen global term of WL92A own_s<seed>.
- **Attribution variants:**
  - ORA_SRC: exact source use, with T0's learned receiver use.
  - ORA_RCV: T0's learned source use, with the exact receiver use.
  - ORA_PHYS: the exact receiver use on the first flight of the physical B5 field, plus G.
- **Accounting.** WL93B-style, on validation (55 × 4, g00010 included; primary), train-representative and E2. Signed level errors of the single-flight and global terms are reported to read compensation.
- **Renders.** ORA_s is rendered on E2 T0 / T1, E5k995 T0 / T1 and E1 LSPOT.
- **Gate ORACLE_FEASIBLE.** For both seeds:
  - WHOLE(ORA) ≤ WHOLE(TQ) − max(2 SE, 0.5 gap), and
  - SINGLE(ORA) ≤ SINGLE(TQ) − 2 SE.
  - Here gap = WHOLE(TQ) − WHOLE(T0), and SE comes from a paired geometry bootstrap (2,000 resamples, seed 93003).
  - If the gate fails: the verdict is ORACLE_COMPOSITION_LIMITED and there is no training.
- **Limiting factor** (reported in every case):
  - LOCAL_OPERATOR if the exact composition's single flight is not better than T0's.
  - GLOBAL_COMPENSATION if the single flight is better but the whole is not.
  - INTERACTION if the seeds disagree.

**Task B: only if ORACLE_FEASIBLE.**
- **Arms.** All arms share the same tied operator.
  - T0 and TQ are the WL93A checkpoints, hash-verified.
  - TR and TQR are trained here.
- **Objective.** OWN + w_q l_q + w_r l_r, with w_q, w_r ∈ {0, 1}, fixed now.
  - l_q is the WL93 source-stage loss.
  - l_r = mean(((R_θ − R*)/den(R*))²), using the WL93 normalisation.
  - R_θ is the tied receiver use on the first flight of the physical training target q.
  - R* = M64(flight(q)) is the physically correct receiver response. It is built from q, never from the model's own first flight.
- **Training contract.** The WL93A contract, unchanged: 128 geometries, the B0 training targets, 2,000 steps, the same schedule and order, seeds 8601 / 8602.
- **Criteria** (per seed; margins 2 SE; training is deterministic on this path, per WL93B):
  - a — source preserved: SRC(TQR) ≤ SRC(T0) − 0.5 gain, and SRC(TQR) ≤ SRC(TQ) + margin.
  - r — receiver improved: RECV(TQR) better than both T0 and TQ by at least the margin.
  - b — whole improved vs TQ: WHOLE(TQR) ≤ WHOLE(TQ) − max(margin, 0.5 gap), and SINGLE not worse than TQ.
  - s — no sharp regression: region-C E_H ≤ 1.25 × TQ + 0.002 on E2 / E5; stationary / D ≤ 1.25 × T0 + 0.002 (also on E1 LSPOT); no WL86 gate lost.
- **Labels.**
  - DUAL_SUPERVISION_SIGNAL: a, r, b and s hold for both seeds.
  - STAGE_ONLY_IMPROVEMENT: a and r hold for both seeds, but b or s fails.
  - NO_SIGNAL_AT_SMOKE_SCALE: r fails for both seeds, or a and b both fail for both seeds.
  - INCONCLUSIVE: anything else, or a control failure.
  - ORACLE_COMPOSITION_LIMITED: the Task A gate failed.

**Tests.**
- T_REUSE.
- T_FLIGHT: the stand-alone flight equals the first flight inside `Local.single`.
- T_ORACLE: ORA equals WL92B's X_M12 counterfactual of own_s<seed> on E2 T1.
- T_RECV_TARGET: R* is linear, non-negative, maps zero to zero, and is model-independent. Its correspondence to the path-traced 2-event field is reported.
- T_TIED, T_GLOBAL, T_INIT, T_TARGETS, T_DET.

## Dev chain (before the freeze; not evidence)

`~/rest_x93cdev`, run with `--dev`: 5 accounting configurations, 100 steps, 64 spp. Task B was forced only to exercise the plumbing. Values were not printed, and no dev quality value was read.

- **Stages that ran.** Reuse, tests, oracle accounting, oracle render tables and renders, TR / TQR training, post tests, accounting, arm tables and renders, analysis and visuals.
- **Pre and post tests pass:**
  - ORA vs WL92B X_M12: 1.0e-7.
  - R* scaling: 0. R* superposition: 4.6e-7. R* is non-negative and maps zero to zero.
  - Flight equality, tied identity, global identity and determinism are all exact.
  - Init equivalence: 1.7e-7.
- **R* correspondence (a property of the target, not a model result).** On E2 T1, M64(flight(q)) differs from the path-traced 2-event field t2 by a relative L1 of 0.064.

**Timing.**
- Training (4 concurrent): 9.8 s per 100 steps, about 3.3 min per 2,000-step run; GPU peak 0.75 GB per process.
- Accounting: dominated by fixed setup. WL93B accounted 73 configurations × 18 models in 2.6 min.
- Renders: the WL93B path, pool 5.
