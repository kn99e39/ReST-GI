# Worklog — ECRO Source-Stage Target Reliability and Supervision Smoke Test (WL93A) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93A Source-Stage Target Reliability and Supervision Smoke Test" (user, 2026-11-13).
- **Batch class.** Suffix batch of WL93 under the AGENTS.md numbering policy. It resolves the target-reliability uncertainty that stopped WL93 before training; there is no new mechanism.
- **Question.** Can a reliably estimated physical source-stage target produce a measurable, independently validated improvement in ECRO's learned local response, under a small, controlled tied-factor training experiment?
- **Not in scope.** No generalization claim, no decision on untied factors, no adoption, no full-scale run, no successor.
- **Where.**
  - Branch `Selective_Recompute`, LabServer63.
  - Code `experiments/ecro_stage93a/` (reuses `ecro_stage93/model93.py`, `train93.py`, `prep93.py` unchanged).
  - Results `results/ecro_stage93a/`, `results/evaluation/93A/`.
  - Server root `~/rest_x93a`, reading `~/rest_ecro86`, `~/rest_x92`, `~/rest_x92a`, `~/rest_x92b`, `~/rest_x93`.

## Frozen WL93 history (preserved)

- WL93 ended **STOPPED-INCONCLUSIVE / STOP_TARGET_NOT_IDENTIFIABLE** at its identifiability preflight (frozen commit `c30c797`, closing commit `56be753`).
  - Validation: encoding-through-ER vs q 0.508 against the bound 0.485 / 0.483.
  - E2: pass (0.362).
- WL93 Amendment 1 (stop-path analysis crash fix) keeps its provenance and its **pending-approval** status.
- No WL93 file is modified. WL93's verdict and gate are not re-judged here. H_TARGET below is an assessment only.

## WL93A protocol (`experiments/ecro_stage93a/protocol/ecro_stage93a.json`)

**Invariants.**
- Unchanged: ECRO generator; frozen ER materials and path semantics; the deployed 16-sample source encoding; local rank 16; widths; the geometry interface; tied source / receiver factors; the global branch frozen to WL92A own_s<seed>; WL92A checkpoints and split; WL86 thresholds.
- No runtime oracle.
- **What changes:** only offline target estimation / uncertainty accounting, and source-stage supervision in one arm.

**Target q.** `quad87.qtable` with an unchanged estimator: P stratified points per patch × Mq one-sample MIS directions per point on ER patches; Lambert exact per point. Only the budget varies, never the quantity.

| Budget | P | Mq | Note | Probe s / geometry (4 lights) |
|---|---|---|---|---|
| B0 | 16 | 1,024 | WL93 / WL87 budget | 0.39 |
| B1 | 16 | 4,096 | directional × 4 | 0.41 |
| B2 | 64 | 1,024 | spatial × 4 | 0.81 |
| B3 | 64 | 4,096 | | 0.88 |
| B4 | 256 | 1,024 | spatial × 16 | 2.42 |
| B5 | 256 | 4,096 | evaluation reference | 2.74 |

- **Axes.** The spatial axis (P) and the directional axis (Mq) are varied separately.
- **Realizations.** Four independent realizations per budget and configuration.
- **Seeds.** Evaluation reference 93,100,000 + 1,000 cfg + 40 budget + 4 r + lighting. Training targets 93,500,000 + 4 idx + l. Training-validation targets 93,600,000 + 4 idx + l. The ranges are pairwise disjoint and disjoint from WL93 and WL87 (tested).

**Configurations (deterministic, by position only).**
- **Training subset:** the WL92A training population sorted by name → every 7th → first 128.
- **Train-representative:** every 8th of the subset (16 × 4 lights).
- **Validation:** all 55 × 4, including WL93's difficult g00010 / g00039; nothing excluded.
- **Sharp:** E2 T0 / T1.

**Task A metrics.**
- **M(X):** pooled debiased relative L2 on panel patches, sqrt(Σ(X − qa)(X − qb) / Σ qa qb), with qa / qb the means of two disjoint pairs of the four B5 realizations. It is unbiased for ‖X − q‖² / ‖q‖² when X is independent of the reference.
- **Uncertainty:** pairing uncertainty U (max − min over the 3 pairings) and a 2,000-resample geometry bootstrap.
- **Secondary:** relative L1 vs the B5 mean (with its floor), the flux / directional split, panel peak-bin retention, per-geometry values, top-5 shares and the predeclared high-confidence subgroup (B5 per-geometry noise ≤ 0.10, always next to the full population).
- **Three quantities kept distinct:**
  1. independent target disagreement N_b (single-realization relative L2 from the unbiased across-realization variance);
  2. the frozen-ER-through-encoding discrepancy M(enc_q(s16)). This is an operator / encoding discrepancy and is **not** called an information-theoretic irreducible error;
  3. WL92A OWN's learned discrepancy M(Bq_own).

**Rules (fixed before evidence).**
- **H_TARGET.** SUPPORTED if, on the pooled validation panel, |relL1(enc vs WL93's stored q) − relL1(enc vs B5 mean)| ≥ 0.05 while the B5 mean's floor ≤ 0.02. NOT SUPPORTED if < 0.02. Otherwise INCONCLUSIVE.
- **Training budget.** The cheapest of B0–B4 with N_b ≤ 0.10 on the train-representative panel. Else B5 if N_B5 ≤ 0.20. Else STOP TARGET_UNRELIABLE.
- **TARGET_RELIABLE.** δ = 0.10 × mean_s M(Bq_own init) on the validation panel (WL93's STAGE effect size). RELIABLE iff a training budget exists AND U(Bq_own) ≤ δ/3 for both seeds. Otherwise STOP before training.
- **The 0.5 ratio.** WL93's 0.5-ratio gate is neither reused nor replaced by a more permissive threshold.

**Task B (smoke).**
- **Arms.** T0 (WL93 OWN objective) and TQ (+ the WL93 source-stage loss, w = 1, den_q unchanged). Both use `model93.Local(own, 'tied')`.
- **Shared between arms.** Same start (WL92A own_s<seed>), same frozen global term, the 128-geometry subset, data order rng(seed + 93), 2,000 steps, lr 3e-4 → 1e-5 cosine (warmup 200), clip 10, identical parameter count.
- **Targets.** q is one realization at the training budget. Validation is monitored with independent training-validation targets. Evaluation is against the independent B5 reference.
- **Seeds.** 8601 / 8602.

**Criteria.**
- dM_s = M(TQ_s) − M(T0_s) on the validation panel; SE from a paired geometry bootstrap; S = pairing spread.
- **LEARNING_SIGNAL:** RELIABLE and, for both seeds, dM_s ≤ −max(δ/2, 2 SE, S); TQ validation whole and single-vs-L2 relative L1 ≤ 1.05 × T0; no non-finite step.
- **NO_SIGNAL:** RELIABLE, stable, and the threshold is met for neither seed.
- **INCONCLUSIVE:** anything else.
- **Verdict mapping:** FEASIBLE_FOR_FULL_SCALE_EVALUATION / NO_SIGNAL_AT_SMOKE_SCALE / TARGET_UNRELIABLE / INCONCLUSIVE. These are feasibility assessments only.

**Renders (diagnostic only).**
- E2 T0 / T1, E1 T1 LSPOT, E5k995 T0 / T1.
- T0 / TQ / own92a × seeds; 2 × 4,096 spp, seeds 86600 / 86601.
- E_H against the existing references.
- No superiority claim.

**Tests.**
- Arms: T_INIT (T0 / TQ state dicts identical, = WL92A own ≤ 1e-5), T_PARAMS, T_GLOBAL.
- Target: T_QGEN (B0 + WL87 seed reproduces the stored E2_T1 q; every budget runs), T_SEEDS.
- Data: T_SPLIT (train subset vs validation compared by **geometry content hash**; see below).
- T_DET.

## Dev chain and full-scale probes (before the freeze; not evidence)

**Dev chain** (`~/rest_x93adev`, `--dev`: Mq 64, 24 / 3 / 6 configurations, 100 steps, 64 spp). Every stage ran: configurations, references, stage tables, Task A, targets, tests, training, evaluation, render tables, renders, analysis, visuals.

One defect was found and fixed:
- **Problem.** T_SPLIT compared train and validation by file name. The `train/` and `val/` directories reuse the names g00000…, so different geometries were reported as overlapping and the test failed.
- **Fix.** The test now compares geometry content hashes. Afterwards every test passes.
- **Dev numbers.** Not inspected as evidence: Mq 64 on 6 validation geometries is not the protocol.

**Full-scale probes.**
- **qtable cost per budget.** On one training geometry (timing only), see the table above.
- **Chain probe** (6 configurations, timing / GPU only):

| Stage | Time | GPU | Peak memory |
|---|---|---|---|
| B0–B3 concurrently, plus 2 stage-table jobs | 33 s | 30 % | 8.1 GB |
| B4 alone | 55 s (9 s / configuration) | 17 % | 2.7 GB |
| B5 alone | 64 s | 27 % | 9.4 GB |

- **Why P = 256 is CPU-bound.** Each P = 256 call is dominated by CPU-side estimator work: per-call valid-solid-angle evaluation of sphere points, which depends on the sampled points. That part cannot be cached without changing the estimator.
- **Chosen concurrency.**
  - B0–B3 run concurrently with the stage tables.
  - B4 runs in 3 shards (2.7 GB each).
  - B5 runs alone (≈ 13 min). Two B5 processes would exceed the 12 GB card.
  - Target generation runs 2 concurrent jobs, or 1 if the training budget is B5.
- **Training.** Taken from the WL93 probe: 4 concurrent processes give GPU 82 %, so 2,000 steps take ≈ 4 min per run. With 128 geometries the RAM is small.
- **Renders.** 60 renders at ≈ 8 s each, pool 5.

## Freeze and execution

- **Freeze.** Commit `888ceb6`, deployed to `~/rest_x93a` (`git archive`, LF). One detached setsid chain, 07:22:40–07:51:12 UTC (28.5 min) on LabServer63 (RTX 3080 Ti 12 GB).
- **S1 references.** 07:22–07:43: B0–B5 × 4 realizations × 73 configurations, plus stage tables for enc / own.
- **S2 Task A.** RELIABLE; training budget B0.
- **S3–S5.**
  - Targets at B0 (training seeds 93.5M range, training-validation seeds 93.6M range).
  - Tests pre PASS.
  - Training: 4 concurrent runs, 207–211 s each, no non-finite step.
  - Tests post PASS.
- **S6–S7.** Stage evaluation, render tables and 60 diagnostic renders.
- **Analysis.** The verdict was produced by the frozen `analyze93a.py`. No amendment and no rerun.
- **Post hoc diagnostics (labelled, after the verdict).** `posthoc93a.py` and `vizpost93a.py`.

## Task A measurements: target reliability

**(1) Independent target disagreement N_b.** Single-realization relative L2, panel patches, pooled.

| Group | B0 (16, 1,024) | B1 (16, 4,096) | B2 (64, 1,024) | B3 (64, 4,096) | B4 (256, 1,024) | B5 (256, 4,096) |
|---|---|---|---|---|---|---|
| train-representative (16 × 4) | 0.095 | 0.087 | 0.028 | 0.023 | 0.0096 | 0.0077 |
| validation (55 × 4) | 1.97 | 1.90 | 1.18 | 0.74 | 0.33 | 0.49 |
| sharp E2 T0 / T1 | 0.029 | 0.015 | 0.012 | 0.006 | 0.006 | 0.003 |

- **Spatial noise dominates.** Quadrupling P (B0 → B2) cuts the train-representative noise 3.4×. Quadrupling Mq (B0 → B1) cuts it 1.1×. The spatial points per patch carry the noise: light visibility, cone and 1/d² variation within a patch.
- **Validation is one geometry.** The pooled validation noise comes from **g00010**. Its single-realization noise is 2.67 at B0 and still **0.66 at B5**. Every other validation geometry is ≤ 0.06 at B5, and 54 of 55 are in the predeclared high-confidence subgroup. The non-monotone B4 / B5 values come from g00010's heavy-tailed noise.
- **Bias across budgets.** Train-representative and sharp: ≤ 0.016 (consistent with one physical quantity). Validation: 0.18–0.34 at B2–B4, an artefact of g00010's heavy tail on the debiased estimator.

**(2), (3) Discrepancies against the independent B5 reference.**
- (2) is the frozen-ER-through-encoding discrepancy (enc).
- (3) is the learned WL92A OWN discrepancy.
- M is debiased relative L2 with a 95 % geometry bootstrap; rel L1 is against the B5 mean.

| | train-rep M | validation M | validation M, high-confidence (54) | validation rel L1 | sharp M |
|---|---|---|---|---|---|
| enc | 0.25 [0.22, 0.32] | 2.79 [0.36, 3.51] | 0.56 | 0.495 | 0.37 |
| OWN s8601 | 0.71 | 0.86 [0.66, 0.96] | 0.67 | 0.923 | 0.91 |
| OWN s8602 | 0.71 | 1.12 [0.65, 1.32] | 0.67 | 0.939 | 0.80 |

- **Concentration in g00010.** It carries 98 % of the pooled validation enc numerator and 71 % / 83 % of OWN's. On g00010, M(enc) is 3.77, far above that geometry's own B5 noise of 0.66.
- **Split by kind of error.**
  - enc retains the reference peak bin (0.95–1.06) and has small flux error (0.07–0.14).
  - OWN keeps only 0.16–0.35 of the peak, with flux error 0.19–0.55 and directional error 0.47–0.64.
- **H_TARGET: NOT SUPPORTED** (frozen rule).
  - enc vs WL93's stored q: 0.508. enc vs the B5 mean: 0.495. The change is 0.013 < 0.02. The B5-mean floor is 0.023.
  - OWN: 0.970 / 0.966 → 0.923 / 0.939.
  - **Correction to WL93's post hoc reading.** WL93's post hoc diagnostic attributed the pooled gate value mainly to target Monte Carlo noise. Against the 16×-larger-budget B5 reference, the pooled relative L1 barely moves, and g00010's encoding discrepancy stays far above its target noise. So, on this evidence, the reference uncertainty of the WL93 budget does not materially change the gate quantity; g00010 is a genuine large encoding / operator discrepancy, not mainly a target-noise artefact. WL93's numbers and verdict are unchanged.
- **TARGET_RELIABLE: RELIABLE** (frozen rule).
  - Training budget B0 (train-representative N_B0 0.095 ≤ 0.10).
  - δ = 0.0987.
  - Pairing uncertainty U(OWN) 0.0069 / 0.0049 ≤ δ/3 = 0.033.

## Task B measurements: training smoke (T0 vs TQ, tied, 128 geometries, 2,000 steps)

**Controls (tests pass).**
- Identical initial state dicts, equal to WL92A own (≤ 1.7e-7).
- 405,424 trainable parameters in both arms.
- Global term identical (exact).
- Base frozen.
- Seed sets pairwise disjoint.
- Train / validation geometries disjoint by content hash.
- Deterministic evaluation.
- Training targets and evaluation references use independent seeds.

**Primary rule (validation panel, debiased relative L2 vs B5).**

| Seed | M(T0) | M(TQ) | dM | 95 % CI | Threshold −max(δ/2, 2 SE, S) | Signal | Whole / single-vs-L2 ratio TQ/T0 | NOREG |
|---|---|---|---|---|---|---|---|---|
| 8601 | 0.780 | 0.831 | +0.051 | [−0.073, 0.097] | −0.120 | no | 1.054 / 1.073 | fail |
| 8602 | 1.012 | 1.136 | +0.125 | [−0.092, 0.191] | −0.211 | no | 1.046 / 1.071 | fail |

- **Classification: NO_SIGNAL. Verdict: NO_SIGNAL_AT_SMOKE_SCALE** (frozen mapping).
- **Other measurements** (T0 → TQ per seed; WL92A own init in parentheses):
  - Validation relative L1 vs B5: 0.899 → 0.738 and 0.927 → 0.790 (own 0.923 / 0.939).
  - Panel flux error: 0.298 → 0.163 and 0.324 → 0.159.
  - Directional error: 0.467 → 0.376 and 0.464 → 0.402.
  - Peak retention: 0.315 → 0.484 and 0.330 → 0.454.
  - Train-representative M: 0.687 → 0.628 and 0.697 → 0.614.
  - Sharp E2 M: 0.831 → 0.586 and 0.758 → 0.612.
  - Learned panel a cᵀ mirror-bin retention on E2 T1 vs frozen-ER M64: 0.237 → 0.311 and 0.237 → 0.321 (own 0.204 / 0.211).
  - Validation Bq vs independent training-validation targets: panel relative L1 0.927 → 0.810 and 0.960 → 0.839.
- **Target-noise fitting: not detected.** On train-representative geometries, the fit to the model's own training target equals M against the independent B5 reference (TQ 0.625 vs 0.628 and 0.615 vs 0.614; T0 0.685 vs 0.687).
- **Whole-table cost of TQ.** TQ raises the validation whole-table relative L1 by 4.6–5.4 % and the single-flight-vs-L2 relative L1 by 7.1–7.3 %, so the NOREG clause fails for both seeds.

**Post hoc descriptive (labelled; verdict, rules and criteria unchanged).**
- **Per geometry.** TQ's M is lower on 54 / 55 (s8601) and 51 / 55 (s8602) validation geometries, and its relative L1 is lower on 55 / 55 for both seeds. Median dM is −0.123 / −0.116.
- **g00010 drives the pooled metric.** It alone accounts for 78 % / 88 % of the total absolute change of the pooled numerator, with TQ worse there.
- **High-confidence subgroup.** This subgroup (B5 noise ≤ 0.10, 54 geometries, only g00010 excluded) was declared for Task A and is applied to Task B post hoc. There, dM is −0.053 [−0.084, −0.034] and −0.069 [−0.097, −0.046]. That exceeds δ/2 (0.049) and 2 SE for both seeds. NOREG would still fail, so even this post hoc view does not meet LEARNING_SIGNAL.
- **Protocol design limitation (disclosed).** The predeclared reliability check measured only the pairing spread U of the reference. It did not detect that one validation geometry with a heavy-tailed target dominates the pooled L2 metric.

**Rendered smoke (diagnostic only; 2 × 4,096 spp, common random numbers).** Region-C E_H, T0 → TQ per seed:

| Configuration | s8601 | s8602 | own92a s8601 / s8602 |
|---|---|---|---|
| E2 T1 | 0.076 → 0.046 | 0.042 → 0.035 | 0.077 / 0.040 |
| E2 T0 | 0.067 → 0.056 | 0.049 → 0.046 | |
| E5k995 T0 | 0.034 → 0.021 | 0.024 → 0.022 | |
| E5k995 T1 | 0.032 → 0.031 | 0.025 → 0.023 | |

- **E1 LSPOT stationary:** 0.026 → 0.021 and 0.037 → 0.031.
- **Stationary and D E_H on E2 / E5:** unchanged or slightly higher with TQ (for example E5 T1 s8601 stationary 0.0044 → 0.0052).
- **Reading.** These smoke renders are diagnostic only; no rendered superiority is claimed.

## Visuals (`results/evaluation/93A/`, manifest with hashes)

- `fig1_target_reliability.png`: target disagreement per budget and group; the three discrepancies against B5 with bootstrap intervals; per-geometry WL93-budget noise vs encoding discrepancy.
- `fig2_training_smoke.png`: validation M per arm and seed; dM with the predeclared thresholds; validation stage error during training.
- `fig3_E2T1_render_smoke.png`: E2 T1 signed higher-order error of own92a / T0 / TQ per seed on one shared scale (diagnostic).
- `fig4_posthoc_per_geometry_dM.png` (POST HOC): per-geometry dM and d relative L1 vs the geometry's B5 target noise.

## Reproducibility

- **Code and protocol.** Freeze `888ceb6`. Post hoc scripts `posthoc93a.py` and `vizpost93a.py` are in the closing commit.
- **Server.** `~/rest_x93a` (`chain.status`, `results/`: configs, taskA, evalB, final93a, posthoc93a, tests, models with checkpoints, stage tables, references `data/qref`, targets `data/train_q` / `data/val_q`, renders).
- **Local copies.** `results/ecro_stage93a/` (JSON, logs, model JSONs), `results/evaluation/93A/`.
- **Estimator.** quad87.qtable unchanged; budgets B0–B5 as in the protocol; 4 realizations each.
- **Seeds.**
  - Evaluation reference: 93,100,000 + 1,000 cfg + 40 budget + 4 r + l.
  - Training targets: 93,500,000 + 4 idx + l.
  - Training-validation targets: 93,600,000 + 4 idx + l.
  - Data order: rng(seed + 93).
  - Renders: 86600 / 86601.
- **Training data.** WL92A training population [::7][:128] (file list in `configs.json`).
- **Checkpoint final sha256 (prefix).** T0_s8601 879a5b977939, T0_s8602 0345a36f3dfe, TQ_s8601 e9e6eecbaa8a, TQ_s8602 2ef6ef6b1c45. The initial hash equals the WL92A own-derived Local state (tested).
- **Compute.** 28.5 min wall: references 20.5 min (P = 256 budgets CPU-side bound); training 3.5 min for four concurrent runs; evaluation and renders 7 min. Probes and the dev chain (`~/rest_x93aprobe`, `~/rest_x93adev`) are not evidence.

## Final report (Korean, for the owner)

**Implementation Fact (구현 사실)**
- WL93의 STOPPED-INCONCLUSIVE 판정과 Amendment 1(승인 대기)은 그대로 보존했다. 이번 배치는 별도 프로토콜 `ecro_stage93a`(동결 `888ceb6`)로 실행했다.
- Task A: once-scattered 타깃 q를 6개 예산에서 생성했다.
  - 예산은 공간 축 P(16 / 64 / 256)와 방향 축 Mq(1,024 / 4,096)를 따로 변화시켰다.
  - 예산마다 독립 실현 4개, 구성 73개(train-rep 16, 검증 55 전체, E2 T0 / T1).
- Task B: tied T0 / TQ 두 처리군을 학습했다.
  - 공통 조건: WL92A OWN에서 시작, 전역 분기 동결, 128개 기하, 2,000 step, seed 8601 / 8602.
  - 학습 타깃과 평가 기준은 독립 시드를 쓴다.
- 진단용 렌더: E2, E1 LSPOT, E5.
- 동결 전 dev 점검에서 분할 테스트 결함 1건을 고쳤다. train과 val의 파일명이 같아 생긴 오판으로, 기하 내용 해시 비교로 바꿨다.

**Measurement (측정)**
- 타깃 잡음은 공간 점 수(P)가 지배한다. train-rep 기준 B0 0.095 → B2 0.028 → B5 0.008이고, Mq를 4배로 늘려도 효과는 1.1배뿐이다.
- 검증 풀링 잡음은 g00010 한 기하가 결정한다. 이 기하는 B5에서도 0.66이며, 나머지 54개는 0.06 이하다.
- H_TARGET은 NOT SUPPORTED다. enc의 WL93 q 대비 0.508과 B5 평균 대비 0.495의 차이는 0.013이다.
- TARGET_RELIABLE은 RELIABLE이다(학습 예산 B0, U 0.007 / 0.005 ≤ 0.033).
- Task B의 1차 지표(검증 풀링 debiased L2)에서 dM은 +0.051 / +0.125로, TQ가 오히려 나쁘다. 임계 −0.120 / −0.211에 미달해 두 시드 모두 신호가 없다.
- whole-table / single 항은 TQ에서 4.6–7.3 % 악화했다. NOREG 실패.
- 판정: **NO_SIGNAL_AT_SMOKE_SCALE**.

**Observation (관찰)**
- 2차 지표는 두 시드 모두 TQ가 개선됐다.
  - 검증 rel L1: 0.90 / 0.93 → 0.74 / 0.79.
  - flux 오차는 약 절반으로 줄었다.
  - peak 보존: 0.32 / 0.33 → 0.48 / 0.45.
  - E2 sharp M: 0.83 / 0.76 → 0.59 / 0.61.
  - 학습된 패널 연산자의 거울 bin 보존: 0.24 → 0.31 / 0.32.
- 학습 타깃 잡음을 맞춘 흔적은 없다. 자기 학습 타깃 적합도와 독립 B5 기준 오차가 같다.
- 사후 기술(판정 불변):
  - TQ는 검증 기하 54 / 55, 51 / 55개에서 M이 낮고, rel L1은 55 / 55에서 낮다.
  - 풀링 변화의 78 / 88 %는 g00010이 차지하며, 이 기하에서는 TQ가 나쁘다.
  - g00010을 뺀 고신뢰 부분군에서는 dM −0.053 / −0.069로 CI가 0을 포함하지 않는다. 이 부분군은 Task A용으로 선언한 것을 사후에 적용한 것이다.
- 진단용 렌더에서 E2 T1 영역 C E_H는 0.076 / 0.042 → 0.046 / 0.035로 낮아졌다. 정지 영역은 약간 악화했다. 통계적 우위는 주장하지 않는다.

**Interpretation (해석)**
- 동결 규칙상 이 결과는 소규모 smoke에서 source-stage 감독의 이득을 입증하지 못했다. 원인은 두 가지다.
  - 1차 풀링 지표를 g00010 한 기하가 지배한다. 이 기하는 B5에서도 타깃이 두꺼운 꼬리 잡음을 갖고, TQ가 오히려 나빠진다.
  - whole-table과 single 항이 5–7 % 악화했다.
- 기하별로는 일관된 stage 수준 개선이 관찰된다. 다만 고정 용량에서 OWN 목적과 상충하는 비용이 있다.
- 사전 신뢰성 검사(쌍 spread U)는 단일 기하의 꼬리 지배를 잡지 못했다. 프로토콜 설계의 한계로 공개한다.
- WL93 사후 해석("게이트 값은 주로 타깃 잡음")은 B5 기준으로 지지되지 않는다. g00010의 인코딩 불일치는 실제로 크다.
- 이 결과만으로 대규모 실험을 정당화하는 판정은 나오지 않는다(NO_SIGNAL_AT_SMOKE_SCALE). 결정은 소유자에게 있다.

**Unresolved Question (미해결 질문)**
- g00010 유형 구성의 물리적 성격은 무엇인가? 근접 광원, 스침각, 콘 경계, 1/d² 특이 등이 후보이며, 왜 B5에서도 잡음이 큰가?
- 배포 인코딩(16 샘플)이 g00010 같은 구성에서 생기는 큰 불일치를 감독으로 줄일 수 있는가?
- whole-table 비용이 감독 가중치(w = 1, 고정) 때문인가, tied 용량 때문인가?
- tied와 untied의 필요성, 실제 장면 일반화, 대규모 학습에서의 행동은 측정하지 않았다.

**Architecture Verdict (아키텍처 판정)**
- **NO_SIGNAL_AT_SMOKE_SCALE** (feasibility 평가일 뿐이다).
- 채택 없음, 대규모 실행 없음, 후속 배치 없음.

**Reproducibility (재현성)**
- 동결 `888ceb6`, LabServer63 `~/rest_x93a`.
- 로컬 사본: `results/ecro_stage93a/`, `results/evaluation/93A/`.
- 시드, 예산, 체크포인트 해시는 위 Reproducibility 절을 참조한다. 소요 28.5분.
