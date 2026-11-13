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

## Evidence run 1 (freeze `7247d30`): stopped by the operator, no evidence produced

- 2026-11-13 00:20–00:51 UTC. Pre-tests passed. The component-target stage then ran eight split-tracer shards concurrently.
- That exhausted the 12 GB GPU: Dr.Jit entered an allocation-cache flush loop with GPU utilisation 0 % and 100 % CPU per process. Six files were written in 30 min.
- **This is a defect of my execution plan.** The full-scale probe before the freeze ran one tracer alone; it did not probe eight concurrently.
- The chain was stopped by PID. Its status, output and GPU trace are kept as `chain_run1.status`, `chain_run1.out`, `gpu_trace_run1.csv`. All its data were deleted; it produced no target, reference or model that is used.
- **Execution fix (no research change).** Target shards run at most five at a time, then the E4 references at six. The split tracer keeps the WL86 batch size (128 samples per launch), because the launch partition determines the samples.
- **Probe of the fix.** Four concurrent full-spp shards (3 files each): 6.7 GB peak, no flush warning, about 10 s per file per process, regenerated sum vs stored target 1.8e-4.

## Evidence run 2 (commit `c40226d` = freeze `7247d30` + execution fix; no protocol amendment)

- **Chain.** LabServer63, 2026-11-13 00:57–02:26 UTC (89 min). Every stage rc = 0. GPU utilisation ≈ 97–100 % in the target and training stages; peak 8.4 GB.
- **Outcome by the frozen mapping: `D_STOPPED`.** The only stop reason is that **E4 was not admitted**. Every test and control passed.

### Tests and controls

| Test | Result |
|---|---|
| T_REPRO | frozen checkpoints = stored WL86 tables (difference 0) |
| T_STATE (E4) | deterministic; selective = full = fresh scene; null edit identical |
| T_DISJOINT | kept training data ≥ 0.153 m (boxes) / 0.302 m (lights) from E4; no E4 radiance before this batch |
| T_F2_SUM | 928 + 55 + 30 files, every lighting: regenerated sum vs stored target ≤ 2.1e-4 relative L1, flux within 1e-3 |
| T_INIT | identical initial state dicts for SUM and OWN per seed |
| T_CTL | fresh SUM validation relative L1 = 1.06 / 1.05 × frozen (≤ 1.15) |
| T_LINEAR, T_OWNERSHIP_CODE, T_REPLAY | pass for all six models |
| Numerical | no non-finite step; no gradient clipping (max norm 0.28–0.36 SUM, 1.05–1.06 OWN) |

### E4 admission (WL86 stage-0 criteria, unchanged thresholds)

| Criterion | Value | Result |
|---|---|---|
| Q1 direct change in A | 0.639 | pass |
| Q2 indirect change, stationary | 0.0086 | pass |
| Q3 region C change; sharp / moderate share | 0.363; 0.96 (5,937 px) | pass |
| **Q4 broad multi-bounce change in D** | change 0.0101, higher-order part 0.0047 against 4 × floor 0.0048 (SNR 3.9) | **fail** |
| Q6 higher-order share | 0.229 | pass |
| Q7 reference uncertainty | 0.0037 | pass |

- The sharp-reader criteria hold; the failing clause is the resolvability of the broad higher-order change (0.0047 against 0.0048).
- By the protocol no held-out verdict is given. E4 values below are **descriptive**. No other edit was substituted and no threshold was changed.

### Training

| Run | Wall | Final objective | Whole term | Single vs L₂ term | Global vs L₃₊ term | Peak memory |
|---|---|---|---|---|---|---|
| sum_s8601 / s8602 | 2,291 / 2,297 s | 0.070 / 0.063 | (same) | 0.054 / 0.046 (unsupervised) | 0.049 / 0.041 (unsupervised) | 638 MiB |
| own_s8601 / s8602 | 2,288 / 2,289 s | 0.135 / 0.122 | 0.068 / 0.062 | 0.035 / 0.032 | 0.031 / 0.028 | 629 MiB |

(Last 500 training steps; 928 geometries; 24,000 steps; four runs concurrently.)

### Field level (validation geometries of the reduced population, 55 × 4 lightings)

| Model | Whole relative L1 | Whole debiased RMS | Single vs L₂ | Global vs L₃₊ | Panel patches: whole / single vs L₂ | SUM-term objective |
|---|---|---|---|---|---|---|
| ecro_s8601 / s8602 (frozen) | 0.075 / 0.074 | 0.101 / 0.094 | 0.190 / 0.196 | 0.196 / 0.207 | 0.21 / 0.53–0.59 | 0.062 / 0.059 |
| sum_s8601 / s8602 | 0.080 / 0.078 | 0.105 / 0.102 | 0.210 / 0.196 | 0.218 / 0.200 | 0.26 / 0.74, 0.23 / 0.58 | 0.060 / 0.064 |
| own_s8601 / s8602 | 0.077 / 0.078 | 0.113 / **0.157** | **0.104 / 0.105** | **0.092 / 0.092** | 0.20 / 0.32, 0.20 / 0.30 | 0.070 / **0.095** |

- OWNERSHIP_IMPROVED: OWN / SUM ratio single 0.50 / 0.54, global 0.42 / 0.46 → **pass** (≤ 0.75, both seeds).
- The whole-table relative L1 is unchanged. The squared-error measures of the whole table are worse for own_s8602 (debiased RMS 0.157, SUM-term objective 0.095 vs 0.064).

### Learned local operator (panel S1, flux norm, E4 T1 geometry; mean over patches and colours)

| Model | Mirror-bin retention | Residual | Mass | Panel `Bq` flux / exact q |
|---|---|---|---|---|
| frozen | 0.11 / 0.08 | 0.89 / 0.90 | 0.52 / 0.44 | 0.58 / 0.58 |
| SUM | 0.028 / 0.078 | 0.96 / 0.91 | 0.27 / 0.51 | 0.45 / 0.49 |
| OWN | **0.207 / 0.212** | 0.80 / 0.81 | 0.84 / 0.80 | 1.10 / 1.12 |

- PANEL_LOCAL_IMPROVED: ratio 7.4 / 2.7 → **pass**.
- Reference points (WL92): rank-16 material NMF retains 0.47–0.54; frozen ER = 1.

### Rendered fidelity (E_H, 2 × 4,096 spp; region-C noise bound 0.006 / 0.013 / 0.010 for E1 / E2 / E4)

| Condition | frozen | SUM | OWN |
|---|---|---|---|
| E4 T1, region C (held-out, not admitted) | 0.018 / 0.019 | 0.025 / 0.024 | **0.013 / 0.012** |
| E4 T0, region C | 0.034 / 0.036 | 0.048 / 0.041 | **0.020 / 0.017** |
| E2 T1, region C (inspected) | 0.072 / 0.063 | 0.065 / 0.065 | 0.077 / **0.040** |
| E2 T0, region C | 0.060 / 0.059 | 0.060 / 0.064 | 0.066 / 0.055 |
| E1 T1, region C | 0.013 / 0.011 | 0.012 / 0.013 | 0.011 / 0.013 |
| stationary, every T0 / T1 | 0.003–0.006 | 0.004–0.006 | 0.003–0.006 |
| E4 T1 panel object | 0.020 / 0.021 | 0.029 / 0.027 | 0.015 / 0.014 |

- E4 controls: ORACLE table 0.0096 (T1) / 0.011 (T0); explicit X_EXP 0.027 / 0.014. STALE region C on E4 T1: OWN 0.089 / 0.094, SUM 0.059 / 0.064 (the state update matters 7× for OWN).
- HELDOUT_SHARP, numerically: OWN 0.013 / 0.012 ≤ SUM 0.025 / 0.024 − 0.010 and ≤ 0.06. Formally **not passed**, because E4 is not admitted.
- WL86 gates: F1, F2 (A, D), F3 (D) and F4 hold for every model in every condition. F2 in E2 region C fails for every model except own_s8602 (0.040).

### Component attribution (region C reading; fractions of h)

| Case | Model | Panel family delivered | Single − L₂ | Global − L₃₊ | Total |
|---|---|---|---|---|---|
| E4 T1 (panel family share 0.03) | SUM | 0.08 / 0.17 | −0.098 / −0.092 | +0.129 / +0.114 | +0.032 / +0.022 |
| | OWN | 0.79 / 1.26 | −0.008 / −0.007 | +0.006 / +0.004 | −0.002 / −0.002 |
| E2 T1 (share 0.18) | SUM | 0.07 / 0.10 | −0.186 / −0.167 | +0.105 / +0.084 | −0.081 / −0.083 |
| | OWN | 0.25 / 0.64 | −0.130 / −0.069 | +0.035 / +0.027 | −0.095 / −0.042 |
| E2 T0 (share 0.12) | SUM | 0.02 / 0.03 | −0.145 / −0.138 | +0.058 / +0.045 | −0.087 / −0.093 |
| | OWN | 0.14 / 0.51 | −0.081 / −0.065 | −0.010 / −0.008 | −0.091 / −0.073 |

- Field level on the conditions: single vs L₂ 0.07–0.13 (OWN) against 0.15–0.21 (SUM); global vs L₃₊ 0.05–0.08 against 0.17–0.21.

### Edit response (slope / correlation of the higher-order change)

| Region | SUM | OWN |
|---|---|---|
| E4 C | 0.95, 0.98 / 0.93, 0.99 | **1.00, 0.99 / 0.98, 0.99** |
| E2 C | 0.63, 0.98 / 0.60, 0.97 (FAIL) | 0.60, 0.91 (FAIL) / 0.78, 0.93 (PASS) |
| E1 C | 0.23, 0.36 / 0.34, 0.53 (FAIL) | 0.68, 0.89 / 0.60, 0.86 (FAIL) |
| D (F3) on E1 / E2 / E4 | PASS | PASS |

- DYNAMIC: no F3 lost → pass. The E4 region-C response is "improved" by the rule for both seeds (descriptive, E4 not admitted).

### Sources (stationary E_H; F4 rule 0.04)

| Source | SUM | OWN |
|---|---|---|
| E1 LMIX | 0.006 / 0.006 | 0.005 / 0.005 |
| E1 LSPOT | 0.026 / 0.022 | 0.029 / **0.036** |
| E4 LMIX4 | 0.004 / 0.004 | 0.004 / 0.004 |
| E4 LSPOT4 | 0.017 / 0.020 | 0.012 / 0.014 (region C 0.025 / 0.031 vs 0.052 / 0.052) |

- NO_REGRESSION: **fails** on one clause. own_s8602 E1 LSPOT stationary 0.036 > 1.25 × 0.022 + 0.002. No gate is lost (F4 still holds).

### Costs

- **Offline.**
  - Component-target regeneration: 16 min of the chain (five concurrent shards) for 1,013 files. This is a one-time cost of decomposing existing data; a new data generator would emit the split from the same tracer pass.
  - Training wall identical: 2,288–2,297 s per run with four concurrently.
- **Runtime (identical architecture; E4, warm medians).** State full / selective 33 / 17 ms; source encoding 4.6 ms; factors 7.2 ms, application 1.1 ms, table transfer 2.3 ms for every model. Checkpoint 2.97 MB. torch peak 222 MiB.
- **Quality ladder.** Not run: it is conditional on outcome A.

### Verdict (frozen mapping)

| Criterion | Result |
|---|---|
| OWNERSHIP_IMPROVED | **PASS** (ratios 0.42–0.54) |
| PANEL_LOCAL_IMPROVED | **PASS** (retention 0.21 vs 0.03 / 0.08) |
| HELDOUT_SHARP | **not passed (E4 not admitted)**; numerically OWN 0.013 / 0.012 vs SUM 0.025 / 0.024 |
| NO_REGRESSION | **FAIL** (one source configuration, one seed) |
| DYNAMIC | PASS |
| SOURCE | F4 passes for both arms on E4 sources |
| **Outcome** | **D STOPPED** (E4 not admitted). Hypothesis label by the code: H2 |

- **Counterfactual, not a verdict.** Had E4 been admitted, the NO_REGRESSION failure alone would map the outcome to B (PARTIALLY SUPPORTED).

## 최종 보고

### Implementation Fact

- **번호.** 사용자 브리프 "WL93"을 사용자 지시에 따라 **WL92A**로 수행했습니다.
- **변경 범위.** ECRO 구조·용량·runtime interface는 그대로입니다(735,664 parameter, R 16, global rank 24, 1,270 × 64 table, 512 state ray, 16-point encoding). 바뀐 것은 학습 목표의 분해(L₂ / L₃₊)와 목적함수뿐입니다.
- **arm.**
  - SUM(새 control): WL86 목적함수.
  - OWN: WL92 OWN 목적함수 = 전체 table 항 + B_single vs L₂ + B_global vs L₃₊, 가중치 1 / 1 / 1, 증거 전에 고정.
  - seed 8601 / 8602, 초기값과 데이터 순서가 두 arm에서 동일함을 검증했습니다.
- **component target.** WL86 생성기의 seed와 표본으로 재생성했고, 모든 lighting에서 저장 target과 2.1e-4 이내로 일치합니다.
- **학습 population.** E4 held-out을 지키려고 WL86 배제 규칙을 E4에 적용해 학습 96 / 검증 9 / hq 2개 형상을 뺐습니다(두 arm 동일, 공개된 confound). 새 SUM은 frozen 대비 검증 오차 1.05–1.06배로 계약을 재현합니다.
- **E4 reference.** WL87 후보 E3b. 새 reference를 생성했고 WL86 admission 기준으로 판정했습니다.
- **run 1 정지.** shard 8개 동시 실행으로 GPU 메모리가 고갈되어 증거 없이 중단했습니다. 제 실행 계획의 결함이며, 동시 실행 수만 고쳐 run 2를 돌렸습니다.
- **freeze / 결과 커밋.** freeze `7247d30`, 실행 수정 `c40226d`, amendment 없음.

### Measurement

**학습 비용.** 두 arm의 학습 시간은 같습니다(2,288–2,297 s). OWN의 추가 비용은 component target 재생성(이번 chain에서 16분, 1회)뿐입니다. runtime(factor 7.2 ms, application 1.1 ms)과 checkpoint 크기(2.97 MB)도 동일합니다.

**SUM vs OWN 직접 비교 (seed 8601 / 8602).**

| 측정 | SUM | OWN |
|---|---|---|
| 검증 single vs L₂ 상대 L1 | 0.210 / 0.196 | **0.104 / 0.105** |
| 검증 global vs L₃₊ 상대 L1 | 0.218 / 0.200 | **0.092 / 0.092** |
| 검증 전체 table 상대 L1 | 0.080 / 0.078 | 0.077 / 0.078 |
| 검증 전체 table debiased RMS | 0.105 / 0.102 | 0.113 / **0.157** |
| panel 국소 행렬 mirror-bin 유지 | 0.028 / 0.078 | **0.207 / 0.212** |
| E4 T1 region C E_H (held-out, 미승인) | 0.025 / 0.024 | **0.013 / 0.012** |
| E4 T0 region C E_H | 0.048 / 0.041 | **0.020 / 0.017** |
| E2 T1 region C E_H (원래 실패 사례, 점검 조건) | 0.065 / 0.065 | 0.077 / **0.040** |
| E2 T0 region C E_H | 0.060 / 0.064 | 0.066 / 0.055 |
| E4 region C edit 응답 (slope, corr) | 0.95, 0.98 / 0.93, 0.99 | 1.00, 0.99 / 0.98, 0.99 |
| E1 LSPOT 정지 화소 E_H | 0.026 / 0.022 | 0.029 / **0.036** |
| E4 LSPOT4 정지 화소 / region C | 0.017, 0.052 / 0.020, 0.052 | 0.012, 0.025 / 0.014, 0.031 |

- **E4 admission.** Q4(broad 영역의 higher-order 변화 해상도) 하나가 SNR 3.9 대 기준 4로 탈락했습니다. sharp 관련 기준(Q1, Q3)과 noise 기준은 통과했습니다.
- **E4 대조군.** oracle table 0.0096, explicit X_EXP 0.027(T1). STALE은 OWN 0.089 / 0.094로, OWN에서 state 갱신의 효과가 7배입니다.

### Observation

1. OWN은 두 seed 모두에서 component 배분을 회복했습니다(component 오차 약 절반). 전체 table 상대 L1은 변하지 않았습니다.
2. E4에서는 SUM의 보상 패턴(single −0.09~−0.10, global +0.11~+0.13)이 OWN에서 거의 사라졌습니다(±0.01). region C 오차도 절반으로 줄었습니다. 다만 E4의 region C에서 panel-sender family 비중은 0.03 / 0.06으로 작아서, 개선은 주로 일반적인 component 보상 제거에서 옵니다.
3. 학습된 panel 국소 행렬의 mirror lobe 유지가 7배 / 2.7배 늘었지만, 0.21로 frozen ER(1)과 재질 rank-16 상한(0.47–0.54)보다 아직 낮습니다.
4. 원래 실패 사례인 E2 T1 region C는 seed에 따라 갈렸습니다. 8602는 기준 아래(0.040)로 내려왔고, 8601은 오히려 나빠졌습니다(0.077). panel-sender family 전달은 0.25 / 0.64로 늘었지만 완전하지 않습니다.
5. E1 LSPOT(hard cone)에서 seed 8602의 정지 화소 오차가 커졌습니다(0.036, F4 기준 0.04 이내). E4 LSPOT4에서는 반대로 OWN이 더 좋습니다.

### Interpretation

- component 감독은 고정된 ECRO 표현 안에서 학습되는 generator의 **물리적 소유 배분을 바꿀 수 있습니다**. 이는 WL92가 남긴 "학습 계약" 쪽 가설을 지지하는 측정입니다.
- 그 효과는 새 형상(E4)의 sharp 수신자에서 렌더 오차 감소로 나타났습니다. 다만 E4가 admission을 통과하지 못해 **사전 held-out 판정은 없습니다**.
- 원래 실패 지점(E2 panel-sender 경로)은 국소 sharp 응답이 아직 불완전해(mirror 유지 0.21) seed에 따라 해결되거나 되지 않았습니다. component 소유 개선과 sharp 국소 응답 회복은 같은 것이 아닙니다.
- 두 seed만으로는 robustness를 주장할 수 없습니다.

### Unresolved Question

- E4는 broad 영역 기준 하나(SNR 3.9 / 4)로 미승인입니다. 이 조건으로 held-out 판정을 내릴지는 연구 책임자의 결정입니다(WL88의 같은 유형 결정은 "amendment 없음"이었습니다).
- E2 T1에서 seed 간 차이(0.077 vs 0.040)의 원인. 두 seed로는 분리할 수 없습니다.
- 전체 table의 제곱오차가 own_s8602에서 나빠진 원인(검증 SUM 항 0.095 vs 0.064). 일부 검증 lighting의 큰 오차로 추정하지만 측정하지 않았습니다.
- E1 LSPOT 회귀가 seed 고유인지 OWN 계약 때문인지.
- panel mirror lobe 유지 0.21이 학습 예산, generator 구조, tied factor 중 무엇에 묶여 있는지.

### Architecture Verdict

범위: WL86 방 family, rigid box edit, frozen ER 세 재질, 점·cone 광원, 512², 학습 seed 2개.

| 질문 | 답 |
|---|---|
| 1. OWN이 single-flight / global 기여를 물리적으로 회복했나 | **예 (field 수준, 두 seed)**: component 오차 0.42–0.54배. E4에서 보상이 ±0.01로 감소. E2에서는 부분적 |
| 2. sharp panel 방향 응답을 회복했나 | **부분적**: mirror-bin 유지 0.21(SUM 0.03 / 0.08, frozen 0.08–0.11). 재질 rank-16 상한과 frozen ER에는 못 미침 |
| 3. E4로 일반화됐나 | **수치상 예(서술)**: region C 0.013 / 0.012 vs 0.025 / 0.024. **사전 판정 없음** (E4 미승인) |
| 4. signed 동적 edit 응답이 개선됐나 | E4 C: slope 0.98–1.00 vs 0.93–0.95 (서술). E1 C 크게 개선, E2 C는 seed별로 혼재. D의 F3는 두 arm 모두 통과 |
| 5. source 일반화가 개선됐나 | E4 source는 같거나 좋음. E1 LSPOT에서 seed 8602 회귀(0.036 vs 0.022) → NO_REGRESSION 실패 |
| 6. 추가 감독 / 학습 비용 | 학습 시간·runtime·저장 동일. component target 재생성 16분(1회) |
| 7. ECRO 학습 계약 변경을 지지하나 | 사전 규칙으로는 **판정 불가(D)**. component 수준 성공은 측정됐지만, held-out 판정 부재, 한 source 회귀, E2 원 실패의 seed 의존 해소 때문에 전체 시스템 성공으로 확립되지 않았습니다. 변경 여부는 책임자 결정입니다 |

**Completion outcome: D (INCONCLUSIVE / STOPPED)**: E4 admission 실패(Q4, SNR 3.9).
- component-level success(소유 회복, panel 국소 응답 개선)와 full-system success(held-out 판정, 회귀 없음)를 구분합니다. 전자는 측정됐고, 후자는 확립되지 않았습니다.
- E4가 승인됐다고 가정해도 NO_REGRESSION 실패 때문에 B(PARTIALLY SUPPORTED)에 해당합니다(가정, 판정 아님).
- 어떤 아키텍처나 학습 계약도 채택하지 않았습니다. WL86 / WL92 판정은 그대로입니다.

### Reproducibility

- **Commits.** freeze `7247d30`; execution fix `c40226d` (run 2); results in the closing commit. Branch `Selective_Recompute`. Parents: WL92 `5e672ff` / `9212a55`; WL86 `7786d09` / `63972c8` / `b97027d`.
- **Environment.** LabServer63, RTX 3080 Ti 12 GB, driver 595.71.05, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Protocol.** `experiments/ecro_own92a/protocol/ecro_own92a.json` (v1, sha256 14fae630…, no amendment).
- **Exact objectives.** See "Contract" above (`own92a.objective`).
- **Data.** WL86 train / val / trainhq minus `results/exclusion.json` (928 / 55 / 30). Component targets `~/rest_x92a/data/*_f2` (checks in `results/ecro_own92a/data/*_f2/check_*.json`).
- **Seeds.** Training 8601 / 8602 (init sha256 939ab470… / ebb33894…, equal across arms). E4 REF-ER 92610–92617, REF-P 92630–92637, sources 92650–92653 (8,192 spp). Oracle field 92450 (4,096 spp). Truth split 92400 / 92405. Renders 86600 / 86601 (4,096 spp).
- **Inputs reused.** WL92 `e4_choice.json`, E4 masks, D / W fields of E1 / E2 (hashes in the chain status line `INPUTS`). WL86 checkpoints (sha256 9d332cd6… / 4d993121…).
- **Server.** `~/rest_x92a` (data, models, renders, logs, GPU trace; run 1 status kept).
- **Local.** `results/ecro_own92a/` (chain status runs 1 and 2, GPU traces, final92a / fieldeval / tests / exclusion / admission / local / timing, model logs, render lists, logs). Visuals: `results/evaluation/92A/` (seven figures, manifest with sha256).
- **Invalid or excluded.** Run 1 (stopped, no evidence). Non-evidence: the dev chain (`~/rest_x92adev`) and probes.
- **Throughput (dev, four trainings concurrently):** about 100 ms per step for both objectives. Hence about 40 min per training at 24,000 steps; OWN costs no measurable extra time per step. The full component-target regeneration runs in the WL86 shard layout (8 × 128 train, 4 × 16 val, 4 × 8 high-quality), concurrently with the E4 references. Expected chain wall: about 1 h 45 min.
