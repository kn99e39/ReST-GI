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

## Freeze and execution

- **Freeze and run.** Commit `cab6e18`, deployed to `~/rest_x93c` with `git archive` (LF). One detached setsid chain on LabServer63, 13:50:20–14:04:35 UTC (14 min).
- **Checks.**
  - T_REUSE passed (the WL93A T0 / TQ hashes match).
  - The pre-training tests passed: ORA vs WL92B X_M12 differs by 1.0e-7, and R* is linear, non-negative and maps zero to zero. On E2 T1, R* differs from the path-traced 2-event field by a relative L1 of 0.064.
- **Task A.** Oracle accounting, then the gate: **ORACLE_FEASIBLE**, limiting factor **LOCAL_OPERATOR** (both seeds).
- **Task B.** TR / TQR were trained (4 concurrent runs, 2,000 steps each, about 4 min) and the post-training tests passed. Accounting, render tables and renders followed.
- **Analysis.** Done with the frozen `analyze93c.py`. No amendment and no post hoc rule.

## Measurements

### Task A: oracle composition (validation, 55 × 4, g00010 included; seed 8601 / 8602)

| Arm | Source vs B5 | Receiver on physical first flight vs t2 | Single vs t2 | Whole vs t | Signed single / global level |
|---|---|---|---|---|---|
| T0 | 0.899 / 0.927 | 0.1069 / 0.1072 | 0.1022 / 0.1042 | 0.0762 / 0.0771 | −0.012 / −0.013 (s8601), −0.007 / −0.005 (s8602) |
| TQ | 0.738 / 0.790 | 0.1096 / 0.1104 | 0.1097 / 0.1116 | 0.0803 / 0.0806 | −0.016 / −0.013, −0.010 / −0.005 |
| **ORA** (exact both uses, deployed encoding) | 0.495 | **0.0998** (= exact floor) | 0.1033 | 0.0776 / 0.0774 | **+0.0015** / −0.013, +0.0015 / −0.005 |
| ORA_SRC (exact source, T0 receiver) | 0.495 | 0.1069 / 0.1072 | 0.1101 / 0.1104 | 0.0796 / 0.0795 | |
| ORA_RCV (T0 source, exact receiver) | 0.899 / 0.927 | 0.0998 | 0.0997 / 0.1033 | 0.0756 / 0.0772 | |
| ORA_PHYS (physical B5 field, exact receiver) | (reference) | 0.0998 | 0.0998 | 0.0760 / 0.0758 | −0.0002 |

**Gate.** Paired 2-SE margins in parentheses.
- ORA − TQ: whole −0.0027 (0.0015) and −0.0032 (0.0012); single −0.0064 (0.0027) and −0.0083 (0.0020). **FEASIBLE.**
- ORA − T0: single +0.0010 (0.0072) and −0.0010 (0.0035); whole +0.0015 and +0.0004. The single is not better than T0, so the limiting factor is **LOCAL_OPERATOR**.

**Per geometry.**
- ORA beats T0 on single for 49 / 55 and 47 / 55 geometries, and on whole for 45 / 55 and 42 / 55. The median single difference is −0.0036 / −0.0029.
- The pooled tie comes from g00010. There, ORA's single is 0.464 vs T0's 0.289 / 0.393: the deployed-encoding source discrepancy that WL93A identified as genuine on that geometry.
- High-confidence subgroup (previously inspected, secondary): ORA single 0.0964 vs T0 0.0987 / 0.0987; whole 0.0725 / 0.0719 vs 0.0728 / 0.0723.

**Compensation reading.**
- The exact composition removes the learned single flight's level deficit: signed level −1.2 % / −0.7 % becomes +0.15 %.
- The frozen global term keeps its own deficit (−1.3 % / −0.5 %).
- The whole-table signed level moves toward zero (−0.0125 → −0.0055 for s8601; −0.0062 → −0.0019 for s8602).
- So there is no level-compensation conflict with the global term. The table-level limit is the single-flight accuracy of the composed local operator (64 bins, 512 state rays, 16-sample source encoding) relative to the path-traced 2-event field. ORA_PHYS shows a floor of 0.0998 for the single flight, against T0's 0.102 / 0.104.

**Oracle renders** (diagnostic, 2 × 4,096 spp, common random numbers). Region-C E_H:

| Condition | ORA | T0 | TQ |
|---|---|---|---|
| E2 T1 | **0.030 / 0.033** | 0.076 / 0.042 | 0.046 / 0.035 |
| E2 T0 | 0.059 / 0.058 | 0.067 / 0.049 | 0.056 / 0.046 |
| E5k995 T0 | **0.015 / 0.014** | 0.034 / 0.024 | |
| E5k995 T1 | 0.031 / 0.031 | 0.032 / 0.025 | |

- E1 LSPOT stationary: ORA 0.021 / 0.030 vs T0 0.026 / 0.037.
- Stationary and D are similar to TQ, slightly above T0 on E2 D (0.0039 / 0.0041 vs 0.0034 / 0.0032).
- WL86 gates: all kept.
- Edit tracking: the E2 region-C slope is 0.995 / 1.000, against T0 0.57 / 0.75 and TQ 0.72 / 0.92.

### Task B: training smoke (validation)

| Arm | Source | Receiver (physical first flight vs t2) | Receiver fit to M64 on validation q | Single | Whole |
|---|---|---|---|---|---|
| T0 | 0.899 / 0.927 | 0.1069 / 0.1072 | 0.046 / 0.049 | 0.1022 / 0.1042 | 0.0762 / 0.0771 |
| TQ | 0.738 / 0.790 | 0.1096 / 0.1104 | 0.034 / 0.037 | 0.1097 / 0.1116 | 0.0803 / 0.0806 |
| TR | 0.886 / 0.927 | **0.1046 / 0.1043** | 0.028 / 0.028 | 0.1029 / 0.1058 | 0.0771 / 0.0784 |
| TQR | 0.762 / 0.803 | 0.1071 / 0.1077 | 0.028 / 0.029 | 0.1084 / 0.1098 | 0.0798 / 0.0801 |

**Contrasts.** 2-SE margins in parentheses.

| Contrast | Seed 8601 | Seed 8602 |
|---|---|---|
| TR − T0 receiver | −0.0023 (0.0002) | −0.0030 (0.0003) |
| TR − T0 whole | +0.0009 (0.0005) | +0.0013 (0.0014) |
| TQR − TQ source | +0.024 (0.016) | +0.013 (0.005) |
| TQR − TQ receiver | −0.0025 | −0.0027 |
| TQR − TQ whole | −0.0004 (0.0006) | −0.0006 (0.0006) |
| TQR − TQ single | −0.0013 | −0.0019 |
| TQR − T0 receiver | +0.0003 (0.0004) | +0.0005 (0.0006) |

**Criteria** (both seeds): a fails, r fails, b fails, s passes. The frozen mapping gives **NO_SIGNAL_AT_SMOKE_SCALE**, because r fails for both seeds.

**Arm renders** (region-C E_H).

| Arm | E2 T1 | E2 T0 | E5k995 T1 | E5k995 T0 |
|---|---|---|---|---|
| TR | 0.075 / 0.062 | 0.078 / 0.075 | 0.044 / 0.043 | |
| TQR | 0.042 / 0.035 | 0.054 / 0.047 | 0.030 / 0.024 | 0.018 / 0.022 |

- TR is worse than T0 in region C at E2 T1 for seed 8602 and on both E2 T0 seeds.
- TQR is close to TQ. It shows no sharp regression, and every gate is kept.

## Observations

- The exact physical local operator at both uses is computationally feasible: it reproduces WL92B X_M12, and it is linear and non-negative.
  - Its table-level single / whole are clearly better than TQ.
  - They are only equal to the learned tied T0 in the pooled metric, which g00010 dominates through the deployed encoding; per geometry it is better on most geometries.
  - It is markedly better on rendered sharp readers (E2 T1, E5 T0) and on sharp edit tracking.
- Receiver-only physical supervision is learnable in the tied operator: the receiver response improves on both seeds, without loss of source response. It does not improve whole transport, and it degrades rendered sharp readers.
- With both stage targets, the tied operator keeps most of the source gain (85 % / 91 % of TQ's) but loses almost all of TR's receiver gain. Its receiver response returns to T0's level, and its whole error stays at TQ's level.

## Interpretation

At smoke scale, the shared tied operator does not satisfy the source-stage and receiver-stage physical targets at the same time:
- each target alone moves its own stage;
- jointly, the source objective dominates and the receiver gain disappears;
- neither single-stage nor joint supervision reduces the WL93A whole-transport regression.

The non-deployable oracle shows that the exact frozen-ER operator at both uses would be a better sharp-reader transport than any learned tied arm, and that the frozen global term is not a level-compensation obstacle. The table-level headroom over T0 is small and limited by the composed single-flight discretization and by the deployed source encoding on g00010; that is the LOCAL_OPERATOR mapping.

The smoke therefore separates two things that table metrics conflate:
- **Representability / optimization.** The learned tied operator cannot reach the physical operator under these objectives at this budget.
- **Metric sensitivity.** Rendered sharp readers respond strongly to the physical operator, while pooled whole-table relative L1 is dominated by broad transport.

None of this establishes generalization (E2 / E5 were inspected) or the necessity of untying.

## Unresolved questions

- Why does the receiver gain vanish under joint supervision? Candidates are optimization conflict, objective scaling, or a representation limit of one rank-16 tied factor pair for two different incident-field distributions. Weights were fixed by protocol and not varied.
- Would a longer schedule (full scale) change the joint result?
- How much of the oracle's sharp-reader advantage survives with a learned operator that matches M64 on the panel?
- Can the deployed encoding's discrepancy on g00010-type configurations be reduced without changing the encoding contract?
- The frozen global term was never trained here. Its interaction with a physically accurate single flight under joint training is untested.

## Visuals (`results/evaluation/93C/`, manifest with hashes)

- `fig1_oracle_accounting.png`: Task A stage / integrated relative L1 and signed levels for own / T0 / TQ / ORA / ORA_SRC / ORA_RCV / ORA_PHYS.
- `fig2_training_accounting.png`: Task B arms.
- `fig3_E2_T1_sharp_maps.png`, `fig3_E5k995_T1_sharp_maps.png`: signed higher-order error of T0 / TQ / TR / TQR / ORA per seed, on one shared scale.

## Reproducibility

- **Code / protocol.** Freeze `cab6e18`.
- **Server.** `~/rest_x93c`:
  - `chain.status`
  - `results/`: oracle, oracle_gate, account, contrasts, final93c, tests, models TR / TQR (+ WL93A T0 / TQ copies), tables, renders
- **Inputs.**
  - WL93A data via read-only links: B5 references, B0 training targets.
  - WL86 / WL92A / WL92B inputs, hashed in `chain.status`.
- **Local copies.** `results/ecro_stage93c/`, `results/evaluation/93C/`.
- **Seeds.**
  - Training: 8601 / 8602, order rng(seed + 93).
  - Bootstrap: 93003.
  - Renders: 86600 / 86601.
- **Checkpoint state hashes.** `models/*.json` (`final_sha256`).
- **Compute.** About 14 min wall.

## Final report (Korean, for the owner)

**Implementation Fact (구현 사실)**
- 이전 판정은 모두 보존했다. 동결 `cab6e18`로 실행했다(14분).
- Task A — 비배포 오라클(ORA):
  - frozen-ER 정확 국소 연산자를 single-flight의 두 사용처에 모두 적용하고, 동결 전역 항을 더했다.
  - WL92B X_M12와 1e-7 이내로 일치한다.
  - 귀인용 변형 3개를 함께 계산했다: 정확 source만, 정확 receiver만, 물리 1차 비행.
- Task B — 같은 tied 연산자에서 TR(receiver 감독)과 TQR(source + receiver 감독)을 WL93A 계약으로 학습했다.
  - receiver 타깃은 R* = M64(flight(물리 q))다. 모델 자신의 1차 비행은 쓰지 않았다.
  - 가중치는 사전 고정했다(1).
  - T0 / TQ는 WL93A 체크포인트를 재사용했다.

**Measurement (측정)**
- 오라클 게이트: FEASIBLE. ORA는 TQ 대비 whole −0.0027 / −0.0032, single −0.0064 / −0.0083이다.
- T0 대비 single은 동등하다(+0.0010 / −0.0010) → 제한 요인 LOCAL_OPERATOR.
- TR: receiver가 −0.0023 / −0.0030으로 개선됐고, whole은 +0.0009 / +0.0013으로 악화했다.
- TQR:
  - source는 TQ 이득의 85 / 91 %를 유지했다.
  - receiver는 T0 수준으로 돌아갔다.
  - whole은 TQ 대비 −0.0004 / −0.0006으로 유의하지 않다.
- 판정: **NO_SIGNAL_AT_SMOKE_SCALE** (두 시드 모두 r 실패).

**Observation (관찰) — 질문별 답**
1. **오라클 합성은 물리적·계산적으로 가능한가?** 가능하다. TQ보다 table 수준에서 낫고, sharp 영역 렌더는 모든 학습 처리군보다 크게 낫다(E2 T1 영역 C 0.030 / 0.033 vs T0 0.076 / 0.042). sharp 편집 추적 기울기는 ≈1.0이다. 풀링 기준으로는 T0와 동등한데, 이는 g00010(배포 인코딩 불일치) 때문이다. 기하별로는 47–49 / 55에서 single이 우세하다.
2. **receiver 감독이 receiver 정확도를 높이나?** 단독(TR)으로는 높인다. 결합(TQR)에서는 이득이 사라진다.
3. **결합 감독이 source 개선을 유지하나?** 대부분 유지한다(85–91 %). 다만 TQ 대비 차이가 2SE를 넘는다.
4. **TQ 대비 whole을 개선하나?** 유의하게 개선하지 않는다.
5. **sharp 영역 충실도를 유지하나?** TQR은 TQ 수준을 유지한다(회귀 없음). TR은 sharp 영역을 악화시킨다.
6. **동결 전역 항으로 설명되지 않는 것은?** 수준 보상 충돌은 없다(ORA가 whole 부호 오차를 0 쪽으로 옮긴다). 남은 한계는 합성된 single-flight의 이산화(64 bin, 512 ray, 16-샘플 인코딩)와 g00010의 인코딩 불일치다.
7. **대규모 평가나 독립 설계 단계를 정당화하나?** 이 smoke의 동결 판정은 대규모 평가를 정당화하는 신호를 주지 않는다. 오라클이 보여준 sharp 영역 여지는 소유자의 다음 결정을 위한 정보로만 남긴다.

**Interpretation (해석)**
- 공유 tied 연산자는 이 규모에서 두 단계의 물리 타깃을 동시에 만족하지 못한다. 결합하면 source 목적이 우세해져 receiver 이득이 사라진다.
- 정확한 물리 연산자는 sharp 영역에서는 큰 여지를 보이지만, 풀링 whole-table 지표에서는 여지가 작다. 지표의 민감도가 다르다.
- 일반화와 untied의 필요성은 입증되지 않았다.

**Unresolved Question (미해결 질문)**
- 결합 시 receiver 이득이 사라지는 원인: 최적화 충돌인가, 목적 스케일인가, 표현 한계인가?
- 대규모 학습에서는 결과가 달라지는가?
- 학습 연산자가 패널에서 M64에 근접하면 sharp 영역 이점이 얼마나 남는가?
- g00010 유형의 인코딩 불일치는 줄일 수 있는가?
- 전역 항과 함께 학습하면 상호작용은 어떠한가?

**Architecture Verdict (아키텍처 판정)**
- **NO_SIGNAL_AT_SMOKE_SCALE** (오라클 FEASIBLE, 제한 요인 LOCAL_OPERATOR). feasibility 결론일 뿐이며 채택이 아니다. 후속 배치는 없다.

**Reproducibility (재현성)**
- 동결 `cab6e18`, LabServer63 `~/rest_x93c`. 로컬 사본은 `results/ecro_stage93c/`, `results/evaluation/93C/`.
