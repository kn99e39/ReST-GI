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

## Evidence run (freeze `b29d8ff`; no amendment)

- **Chain.** LabServer63, 2026-11-13 03:47:57–04:40:56 UTC (53 min). Every stage rc = 0. Mean GPU utilisation 87 %, peak 6.5 GB. No rerun.
- **Outcome by the frozen mapping: C (multiple interacting limitations).** H_interact and H_geometry are SUPPORTED, and E5 was admitted.

### Reproduction and tests

- **Reproduction PASS.**
  - Checkpoint hashes are recorded in `repro.json`.
  - Recomputed WL92A tables of all six models on E2 T1 / T0, E1 LSPOT and E4 T1 are identical.
  - Rendered E_H matches `final92a.json` to 0.
  - Validation component errors on the 55 WL92A validation geometries match `fieldeval.json`.
- **Tests PASS.**
  - T_CF_IDENTITY: every counterfactual with the replaced quantity set back reproduces the model to 0.
  - T_LINEAR: scaling 0, all tables ≥ 0.
  - T_EVENT: E1 LSPOT split vs a fresh oracle field, flux 0.99995 and relative L1 0.044, below the half-split noise of 0.062.
  - T_DISJOINT: E5 candidates are 0.152–0.169 m from the nearest training box.

### Counterfactual attribution, region C, OWN seeds 8601 / 8602 (rendered E_H, 2 × 4,096 spp; floor = path-traced table)

| Case | Floor | ORIG | X_L2 | X_L3 | X_M1 | X_M2 | X_M12 | X_Q | X_SHQ |
|---|---|---|---|---|---|---|---|---|---|
| E2 T1 | 0.013 | 0.077 / 0.040 | 0.022 / 0.019 | 0.100 / 0.058 | 0.019 / 0.024 | 0.093 / 0.056 | 0.030 / 0.033 | 0.021 / 0.023 | 0.077 / 0.039 |
| E2 T0 | 0.010 | 0.066 / 0.055 | 0.016 / 0.015 | 0.060 / 0.050 | 0.030 / 0.033 | 0.089 / 0.075 | 0.059 / 0.058 | 0.024 / 0.027 | 0.066 / 0.055 |

Accounted fraction f (OWN 8601 / 8602):

| Case | X_L2 | X_L3 | X_M1 | X_M2 | X_M12 | X_Q | X_SHQ | encoding effect (X_M1 → X_Q) |
|---|---|---|---|---|---|---|---|---|
| E2 T1 C | 0.86 / 0.77 | −0.36 / −0.68 | 0.91 / 0.59 | −0.26 / −0.59 | 0.74 / 0.25 | 0.87 / 0.63 | 0.01 / 0.01 | −0.04 / 0.04 |
| E2 T0 C | 0.89 / 0.88 | 0.11 / 0.11 | 0.64 / 0.49 | −0.41 / −0.46 | 0.13 / −0.06 | 0.74 / 0.62 | 0.00 / −0.01 | 0.10 / 0.13 |

- SUM, by contrast: X_L2 accounts −0.08 / 0.18 (E2 T1) and 0.44 / 0.62 (E2 T0); X_L3 −1.37 / −1.12. In SUM the single-flight deficit and the global over-prediction cancel, so neither component alone isolates the error.
- **OWN seed gap on E2 T1 C** (0.037). Fraction closed: X_Q 0.96, X_L2 0.92, X_M12 0.91, X_M1 0.86, X_L3 −0.14, X_SHQ 0.00.
- **Reading split (region C, E2 T1; OWN 8601 / 8602).**
  - Single-flight error: level −0.25 / −0.15 of h; directional +0.009 / +0.018.
  - Global term: +0.075 / +0.058.
  - Panel-sender family delivered: 25 % / 64 %.
- **Stage-wise field errors.**
  - Learned once-scattered table on panel patches: relative L1 1.14 / 1.07, flux 0.66 / 1.07 of exact q (SUM 0.20–0.24; frozen 0.24–0.34).
  - Encoding through the exact matrix vs exact q on panel patches: 0.36 (point lights) / 0.36 (LSPOT). All patches: 0.03 (point) vs 0.15 (LSPOT).

### E1 T1 LSPOT (inspected source condition)

| | Floor | ORIG | X_L2 | X_L3 | X_M12 | X_Q | X_SHQ |
|---|---|---|---|---|---|---|---|
| stationary, own_s8601 / s8602 | 0.0059 | 0.029 / 0.036 | 0.020 / 0.029 | **0.013 / 0.011** | 0.021 / 0.030 | 0.020 / 0.032 | 0.027 / 0.034 |
| stationary, sum_s8601 / s8602 | 0.0059 | 0.026 / 0.022 | 0.028 / 0.024 | 0.018 / 0.016 | 0.029 / 0.025 | 0.029 / 0.022 | 0.025 / 0.021 |

- Accounted, OWN stationary: X_L3 0.71 / 0.82; X_SHQ 0.09 / 0.07; encoding effect 0.05 / 0.04 → **H_source NOT SUPPORTED**.
- OWN seed gap (0.007): X_L3 closes 0.82.
- Region C, own_s8602: reading +0.075 of h, of which global +0.14 and single +0.03 (over-prediction).

### Learned local matrices (flux norm)

- **Panel `a cᵀ`, relative change E2 T0 → T1** (panel intrinsically unchanged, box B moved):
  - frozen 0.34 / 0.44, SUM 0.30 / 0.28, **OWN 0.145 / 0.146**;
  - floor 0.03–0.05, sphere 0.02–0.03, walls ≤ 0.008.
  - E4 T0 → T1: 0.03–0.07. E1 T1 → E2 T1: OWN 0.14 / 0.16.
- **Seed-to-seed difference at the same configuration (panel):** frozen 0.54, SUM 1.76, OWN 0.51.
- **Panel mirror-bin retention:** OWN 0.20–0.21 in every configuration; SUM 0.03 / 0.08–0.10; frozen 0.08–0.14.

### Hypotheses (frozen rules)

| Hypothesis | Label | Basis |
|---|---|---|
| H_local (X_M12 dominant in both E2 cases) | INCONCLUSIVE | 0.74 / 0.25 (T1), 0.13 / −0.06 (T0) |
| H_global (X_L3) | NOT SUPPORTED as the cause in E2 | replacing the global term worsens E2 T1 (f −0.36 / −0.68): it compensates |
| H_source (E1 LSPOT) | NOT SUPPORTED | X_SHQ ≤ 0.09, encoding effect ≤ 0.05 |
| H_geometry | SUPPORTED (variation present) | panel change 0.145 / 0.146 ≥ 0.10 |
| H_interact | SUPPORTED | no dominant local / global counterfactual, several with f ≥ 0.25 |
| Reported, not a hypothesis rule | single-flight term X_L2 dominant (0.77–0.89); source-side once-scattered stage X_Q dominant (0.62–0.87) and X_M1 0.49–0.91; receiver-side exact X_M2 worsens (−0.26 to −0.59) | |

### Held-out E5 (independently qualified)

- **Population.** Candidates k = 0 … 7,850 were generated; 9 are disjoint from all training boxes; 3 are geometry-admissible (995, 5,882, 7,850).
- **Candidate 995 admitted on the first reference stage.**
  - Box A moved; light L2.
  - Q1 0.955; Q2 0.018; Q3 0.307 (sharp / moderate share 0.83, 4,980 px).
  - **Q4 change 0.026, higher-order part 0.0133 vs floor 0.0010 (SNR 13)**.
  - Q6 0.23; Q7 0.0024.
  - Sharp family 0.098 ± 0.0009.

| Model | T1 C | T0 C | T1 stationary | T1 D | LMIX stationary | LSPOT stationary | C edit slope, corr |
|---|---|---|---|---|---|---|---|
| ecro_s8601 / s8602 | 0.022 / 0.024 | 0.037 / 0.030 | 0.004 / 0.004 | 0.004 / 0.003 | 0.007 / 0.008 | 0.021 / 0.036 | 1.00, 0.97 / 0.84, 0.96 |
| sum_s8601 / s8602 | 0.028 / 0.026 | 0.028 / 0.030 | 0.004 / 0.005 | 0.004 / 0.005 | 0.008 / 0.008 | 0.022 / **0.016** | 0.76, 0.97 / 0.89, 0.98 |
| own_s8601 / s8602 | **0.033** / 0.027 | **0.034** / **0.017** | 0.004 / 0.004 | 0.004 / 0.003 | 0.007 / 0.007 | 0.023 / **0.026** | 0.88, 0.97 / 0.75, 0.98 |

- Region-C noise bound 0.007 (T1) / 0.008 (T0). Controls: ORACLE 0.006 / 0.010; X_EXP 0.030 / 0.017. STALE region C 0.05–0.06 for every model.
- **WL86 gates F1–F4 pass for every model** (all region-C errors ≤ 0.037 against the 0.06 rule).
- **Component attribution on E5 (region C reading).** OWN single − L₂ −0.056 / −0.044, global −0.004 / −0.004; panel family delivered 15 % / 51 %. SUM: single −0.14 / −0.15, global +0.09 / +0.10, delivered 2 %.

| Criterion | Result |
|---|---|
| OWN_CONSISTENT_ADVANTAGE | **FAIL**: own_s8601 worse at T1 (0.033 vs 0.028) and T0 (0.034 vs 0.028); own_s8602 better at T0 (0.017 vs 0.030, margin 0.008) and equal at T1 (0.027 vs 0.026) |
| NO_REGRESSION | **FAIL**: own_s8602 LSPOT stationary 0.026 vs SUM 0.016 (same pattern as E1 LSPOT) |
| DYNAMIC | pass (no F3 lost) |

## 최종 보고

### Implementation Fact

- **재학습 없음.** 6개 checkpoint를 모두 고정한 채 진단했습니다: WL86 frozen, WL92A SUM, WL92A OWN, 각 seed 8601 / 8602.
- **재현.** WL92A의 table, 렌더 E_H, 검증 component 오차를 정확히(차이 0) 재현했습니다.
- **반사실 7종.** 각각 한 가지 양만 교체하고 원 출력과 함께 평가했습니다. 대상은 E2 T1 / T0, E1 LSPOT, 그리고 승인되지 않은 진단 조건인 E4입니다.
- **국소 행렬.** 학습된 `a cᵀ`를 5개 configuration에서 frozen ER M64와 비교했습니다.
- **E5.** 학습 데이터 1,120개와 모든 상자가 0.15 m 이상 떨어진 기하 후보를 순서대로 생성했습니다. WL86 test 광원(L1 / L2, LMIX, LSPOT)을 썼고, WL86 admission(Q4 SNR 4.0 포함)과 sharp family 조건으로 자격을 판정했습니다. 후보 995가 첫 순서에서 승인됐습니다.
- **freeze.** `b29d8ff`, amendment 없음.

### Measurement

- **E2 sharp 영역 (OWN, rendered E_H).**
  - single-flight 항을 정답 L₂로 바꾸면 오차의 77–89 %가 사라집니다.
  - single-flight 항 안에서는 source 측 once-scattered 단계를 정확한 q로 바꾸는 것(X_Q)이 62–87 %, 학습된 source 측 국소 응답을 frozen ER M64로 바꾸는 것(X_M1)이 49–91 %를 설명합니다.
  - 수신 측 국소 응답만 정확하게 바꾸면(X_M2) 오히려 26–59 % 나빠집니다.
  - global 항을 정답으로 바꾸면 E2 T1에서 나빠집니다(−36 / −68 %).
- **E2 T1 seed 차이 (0.077 vs 0.040).**
  - X_Q가 차이의 96 %를 메웁니다.
  - 학습된 panel once-scattered 에너지가 exact q 대비 0.66배(8601) 대 1.07배(8602)입니다.
  - region C에서 single 항의 level 오차는 −0.25 / −0.15이고, 방향 배치 오차는 +0.01–0.02에 그칩니다.
- **E1 LSPOT (OWN 정지 화소).** global 항 교체(X_L3)가 71 / 82 %를 설명합니다. 고밀도 source encoding(X_SHQ)은 9 / 7 %, encoding 정보 효과는 5 / 4 %입니다.
- **국소 행렬의 기하 의존성.** panel 재질은 그대로인데 상자만 옮겨도(E2 T0→T1) OWN의 panel `a cᵀ`가 14.5 % 바뀝니다. SUM은 28–30 %, frozen은 34–44 %입니다. 같은 configuration에서 seed 간 차이는 OWN 51 %, SUM 176 %입니다.
- **E5 (승인된 독립 held-out).**

| | SUM 8601 / 8602 | OWN 8601 / 8602 |
|---|---|---|
| T1 region C | 0.028 / 0.026 | 0.033 / 0.027 |
| T0 region C | 0.028 / 0.030 | 0.034 / **0.017** |
| LSPOT 정지 화소 | 0.022 / 0.016 | 0.023 / **0.026** |
| single − L₂ / global − L₃₊ (region C reading) | −0.14 / +0.09, −0.15 / +0.10 | −0.06 / −0.004, −0.04 / −0.004 |

### Observation

1. OWN에서 남은 sharp 오차는 single-flight 항에 몰려 있습니다. SUM에서는 single 결손과 global 과대가 서로 상쇄되어, 어느 한 항만 바꾸면 오히려 나빠집니다.
2. single-flight 항 안에서 binding인 단계는 **source 측 once-scattered 응답**입니다. 학습된 panel의 1차 산란 응답은 field 기준 상대 L1이 1 이상(1.07–1.14)으로 방향 배치가 크게 틀립니다. 수신자에서는 이것이 patch 에너지 결손(level)으로 나타납니다.
3. 수신 측 국소 응답을 물리적으로 정확하게 바꾸면 결과가 나빠집니다. tied factor(같은 `a cᵀ`)가 source 측 결손을 수신 측에서 일부 보정하고 있다는 뜻입니다.
4. LSPOT 회귀는 global 항이 cone 광원(학습에 없던 source class)에서 과대 예측하는 데서 옵니다. source encoding은 원인이 아닙니다. 같은 패턴이 E5 LSPOT의 own_s8602에서 재현됐습니다.
5. 독립 E5에서는 모든 모델이 WL86 기준을 통과했지만, OWN이 SUM보다 일관되게 좋지는 않았습니다. component 배분(보상 제거)은 E5에서도 재현됐습니다.

### Interpretation

- OWN은 component 배분을 물리적으로 바꿨고, 그 덕분에 오차 위치를 분리할 수 있게 됐습니다. **남은 오차는 generator가 sharp panel의 source 측 1차 산란 응답을 기하별로 일관되게 만들지 못하는 데서 옵니다**(seed 차이와 기하 변동의 대부분이 여기에 있음).
- 다만 이 응답은 tied factor로 수신 측과 global 항에 얽혀 있어서, 한 단계만 정확하게 바꾸는 반사실조차 일관된 개선을 주지 않습니다(X_M12, X_M2). 그래서 사전 규칙으로는 단일 지배 원인이 아니라 **상호작용(C)**으로 분류됩니다.
- source encoding은 이 workload의 점광원과 cone 광원 모두에서 binding이 아닙니다. 반면 학습에 없던 광원 class에 대한 global 항의 외삽은 별도 실패 모드입니다.
- 반사실 개선은 oracle 데이터를 쓴 진단이며, 학습 모델의 개선으로 계산하지 않습니다.

### Unresolved Question

- source 측 국소 응답만을 독립적으로(tied 해제) 학습하면 일관되게 좋아지는지. 학습이 필요해 이번 범위 밖입니다.
- 학습된 panel 응답의 seed 간 차이(51 %)가 기하 의존 변동(14.5 %)보다 큰 이유. 최적화 경로인지 데이터 부족인지 분리하지 않았습니다.
- cone 광원에서 global 항이 과대 예측하는 기제(학습 분포 밖 source의 global factor 응답). 측정은 했지만 내부 원인은 분리하지 않았습니다.
- E5 하나, seed 둘로는 robustness를 주장할 수 없습니다.

### Architecture Verdict

범위: WL86 방 family, rigid box edit, frozen ER 세 재질, 점·cone·2-광원 source, 512², 학습 seed 2개, 고정 checkpoint.

| 질문 | 답 | 근거 수준 |
|---|---|---|
| 1. OWN 8601 vs 8602의 E2 차이 | source 측 once-scattered panel 응답의 에너지 차이(0.66 vs 1.07× exact q). X_Q가 seed 차이의 96 % 해소 | 반사실 귀속 |
| 2. E1 LSPOT 회귀 | learned global 항의 과대 예측(X_L3 71 / 82 %). E5 LSPOT에서도 재현 | 반사실 귀속 + 독립 재현 |
| 3. 국소 각 응답의 몫 | source 측 교체가 49–91 %. 두 국소 사용을 모두 교체하면 −6–74 %로 불일관. 수신 측만 교체하면 악화. **H_local INCONCLUSIVE** | 반사실 귀속(tied factor 교락) |
| 4. source encoding의 충분성 | 충분함. LSPOT에서 X_SHQ ≤ 9 %, encoding 효과 ≤ 5 %. **H_source NOT SUPPORTED** | 반사실 귀속 |
| 5. 보상 / 기하 누출 | 보상: SUM에서 강함, OWN에서 E2에 일부 남음. 기하 변동: OWN panel 14.5 % (**H_geometry SUPPORTED**, 단 seed 차이가 더 큼). K/C 위반 여부는 tied factor 때문에 판정 불가 | 측정 + 규칙 |
| 6. 독립 held-out에서 OWN의 일관된 우위 | **NOT SUPPORTED**: E5에서 8601은 SUM보다 나쁘고, 8602는 T0에서만 좋으며, LSPOT 회귀가 재현됨 | 독립 승인된 held-out |
| 7. 증거가 지지하는 개입 | 학습된 **source 측 국소 응답의 기하 조건부 생성**과 **tied-factor 소유(같은 `a cᵀ`의 이중 사용)**, 그리고 별도로 **학습 분포 밖 source에서의 global 항**. source interface 변경은 지지되지 않음. 구체적 설계는 책임자 결정이며, 구현하지 않았습니다 | 상호작용 증거 |

**Completion outcome: C (multiple interacting limitations established; E5 qualified).**
- WL92A의 D_STOPPED 판정은 그대로이고, E4는 계속 미승인입니다.
- OWN은 component 소유 개선으로는 유효하지만, 독립 workload에서 sharp fidelity의 일관된 우위는 확립되지 않았습니다.
- 다음 아키텍처 결정이 다룰 수 있는 항목: 국소 각 응답(source 측) × tied-factor 소유 × 기하 조건부 generator 동작의 **상호작용**, 그리고 미학습 source class에서의 global 항. source interface는 해당하지 않습니다.
- 어떤 개입도 채택하거나 시작하지 않았습니다.

### Reproducibility

- **Commits.** freeze `b29d8ff`; results in the closing commit. Branch `Selective_Recompute`. Parents: WL92A `7247d30` / `c40226d` / `ce87188`; WL92 `5e672ff` / `9212a55`.
- **Environment.** LabServer63, RTX 3080 Ti 12 GB, driver 595.71.05, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Protocol.** `experiments/ecro_cf92b/protocol/ecro_cf92b.json` (v1, sha256 db6c39fc…, no amendment).
- **Checkpoints.** sha256 in `results/ecro_cf92b/results/repro.json`.
- **Seeds and samples.**
  - Counterfactual renders 86600 / 86601 at 4,096 spp.
  - E1 LSPOT split 92400 / 92405 (2 × 2,048 spp); its oracle field 92460 (4,096 spp).
  - E5 candidate rng [92550, k].
  - E5 references for candidate k: REF-ER 92710 + r + 100 k and REF-P 92730 + r + 100 k (8 × 8,192 spp); sources 92750 + r (4 × 8,192 spp); oracle field 92460.
- **Inputs reused.** D / W fields of E2 / E1 (WL92) and E4 (WL92A); E4 masks; hashes in the chain status `INPUTS` line.
- **Server.** `~/rest_x92b`.
- **Local.** `results/ecro_cf92b/` (chain status, GPU trace, final92b, repro, tests, local92b, E5 candidates and admission, counterfactual table metadata, render lists, logs). Visuals: `results/evaluation/92B/` (seven figures, manifest with sha256).
- **Invalid or excluded.** None. Non-evidence: the dev chain (`~/rest_x92bdev`).
