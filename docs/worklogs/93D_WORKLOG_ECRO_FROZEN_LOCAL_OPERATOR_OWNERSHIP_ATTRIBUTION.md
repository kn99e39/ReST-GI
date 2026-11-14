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

## Freeze and execution

- **Freeze.** Commit `4f3fc56`, deployed to `~/rest_x93d` with `git archive` (LF).
- **Chain.** One detached setsid chain on LabServer63 (RTX 3080 Ti), 05:40:58–06:59:27 UTC (78 min). No amendment; the analysis was produced by the frozen `analyze93d.py`.
- **T_CKPT: PASS.**
  - own: file sha256 matches WL92B `repro.json`.
  - T0: state sha256 matches WL93A.
- **E6 qualification** (frozen WL93 rule, budget 3 geometry-admissible candidates k = 28 / 1956 / 3940; references 8 × 8,192 spp per candidate, about 20 min each):
  - **No candidate was admitted.**
  - All three failed SHARP_FAMILY: share 0.000 / 0.015 / 0.027, against ≥ 0.08.
  - k = 3940 also failed Q2.
  - Q4 passed for all three.
  - The budget is exhausted. **Conclusions are limited to the inspected E2 / E5k995 conditions.**
- **Other controls.**
  - T_TESTS: PASS. This covers identity substitution ≤ 1e-6, the global term identical, per-use digests, the frame identity of stationary patches, T_USED scaling, determinism, and C2_both = WL92B X_M12.
  - Render reproduction (own C0 vs WL93A I93A_own92a): PASS.

## Measurements

**Task A — functional invariance.**
- Probes: canonical one-hot, and material-level workload bases. The S1 source basis has dimension 35 and the receiver basis 31, both at 99 % energy. All values are relative to the M64 response.
- Values are given as own s8601 / s8602; T0 in brackets.

| Material | E_phys (canonical) | E_phys (workload src / rcv) | D_geo (all pairs) | D_geo (E2 / E5 edit pairs) | D_seed | S_geo (geometry share of squared error) |
|---|---|---|---|---|---|---|
| **S1 panel** | 0.80 / 0.81 [0.79 / 0.81] | 0.79–0.81 / 0.75 [0.78–0.82 / 0.75] | 0.040–0.054 [0.061–0.084] | 0.093–0.108 [0.12–0.14] | 0.26–0.29 [0.30–0.32] | **0.002–0.004** [0.004–0.008] |
| S2 sphere | 0.59–0.61 | 0.69–0.73 / 0.60–0.64 | 0.021–0.036 | 0.015–0.023 | 0.24–0.29 | 0.001–0.002 |
| B floor | 0.29–0.31 | 0.27–0.29 / 0.30–0.33 | 0.072–0.088 | 0.060–0.082 | 0.16–0.18 | 0.03–0.05 |
| LAM (negative control) | 0.09–0.10 | 0.10–0.11 / 0.10–0.11 | 0.013–0.015 | 0.012–0.013 | 0.066–0.070 | 0.008–0.012 |

- E_common is the error of the geometry-mean operator. For S1 it is 0.788–0.807, essentially equal to E_phys: the physical error is almost entirely geometry-common.
- S1 error by incident ring, from normal to grazing (own s8601): 0.98 / 0.88 / 0.83 / 0.76.

**Task C — incident-subspace attribution.** Relative to |M x|. Evaluation fields: validation, E2 and E5. Basis: the fixed material-level training-distribution basis.

| Group : use | Out-of-span energy share of evaluation fields | e_in | e_out | Geometry drift in / out |
|---|---|---|---|---|
| S1 : source | **0.65** | 0.58–0.69 | 0.59–0.63 | 0.039–0.095 / 0.026–0.049 |
| S1 : receiver | 0.027 | **0.31–0.34** | 0.09 | 0.048–0.091 / 0.004–0.010 |
| S2 : source | 0.027 | 0.50–0.53 | 0.08–0.09 | 0.022–0.032 / 0.004–0.007 |
| S2 : receiver | 0.13 | 0.28–0.31 | 0.11–0.15 | 0.019–0.028 / 0.002–0.005 |

**Task B — interventions on the validation population.**
- 55 geometries, g00010 included. Single-flight relative L1; paired difference to C0 with SE in brackets.

| Family / seed | C0 | C1_src | C1_rcv | C1_both | C2_src | C2_rcv | C2_both |
|---|---|---|---|---|---|---|---|
| own s8601 | 0.1039 | +0.0009 (0.0007) | +0.0007 (0.0001) | +0.0016 (0.0008) | **+0.0073 (0.0028)** | **−0.0035 (0.0003)** | −0.0007 (0.0029) |
| own s8602 | 0.1053 | +0.0012 (0.0010) | +0.0008 (0.0001) | +0.0019 (0.0010) | +0.0052 (0.0011) | −0.0017 (0.0004) | −0.0020 (0.0011) |
| T0 s8601 | 0.1022 | +0.0001 (0.0002) | +0.0007 (0.0001) | +0.0007 (0.0002) | +0.0079 (0.0034) | −0.0026 (0.0003) | +0.0010 (0.0035) |
| T0 s8602 | 0.1042 | +0.0012 (0.0010) | +0.0008 (0.0001) | +0.0021 (0.0011) | +0.0062 (0.0016) | −0.0009 (0.0004) | −0.0010 (0.0017) |

- Whole-table relative L1 follows the same pattern. For own s8601: C0 0.0767, C1_both 0.0776, C2_src 0.0800, C2_rcv 0.0760, C2_both 0.0776.

**Task B — rendered region-C E_H.** 2 × 4,096 spp, common random numbers; seed 8601 / 8602.

| Condition | Family | C0 | C1_both | C2_src | C2_rcv | C2_both |
|---|---|---|---|---|---|---|
| E2 T1 | own | 0.077 / 0.040 | 0.084 / 0.043 | **0.019 / 0.024** | 0.093 / 0.056 | 0.030 / 0.033 |
| E2 T1 | T0 | 0.076 / 0.042 | 0.081 / 0.035 | **0.018 / 0.021** | 0.085 / 0.056 | 0.030 / 0.033 |
| E5k995 T1 | own | 0.033 / 0.027 | 0.034 / 0.024 | **0.016 / 0.017** | 0.050 / 0.041 | 0.031 / 0.031 |
| E5k995 T1 | T0 | 0.032 / 0.025 | 0.037 / 0.030 | **0.018 / 0.013** | 0.049 / 0.041 | 0.031 / 0.031 |
| E2 T0 | own | 0.066 / 0.055 | (= C0) | **0.030 / 0.033** | 0.089 / 0.075 | 0.059 / 0.058 |
| E5k995 T0 | own | 0.034 / 0.017 | (= C0) | **0.017 / 0.014** | 0.038 / 0.020 | 0.015 / 0.014 |

**Criteria (frozen).** For both families and both seeds:
- **G1 fails.** D_geo(S1, workload) is 0.047–0.084, below 0.10, although it exceeds 2 × LAM.
- **G2 fails.** S_geo is 0.003–0.008, far below 0.25.
- **G3 fails.**
  - The region-C |C1_both − C0| changes fall within max(noise, 0.005) on fewer than 2 conditions.
  - The validation |C1_both − C0| is ≥ 2 SE only for T0 s8601 and own s8601, not for both seeds of a family.
- **R condition holds:** E_phys 0.75–0.82 ≥ 0.30 and S_geo < 0.25.
- **COADAPTATION holds.** C2_both validation single flight is not better than C0 by 2 SE for both seeds. The C2 interaction term is met for s8602 only.

**Labels.**
- own (primary): **REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED.**
- T0 (secondary): the same.
- No stop, no family conflict.

## Observation

- **Geometry dependence is real but small.**
  - The learned S1 operator changes between geometry pairs by 4–8 % of the physical response overall, and by 9–14 % on the E2 / E5 box-move pairs. That is consistent with WL92B's 14.5 % in a different norm.
  - The change is 3–6 × the Lambert control.
  - It is 3–6 × smaller than the seed-to-seed difference (26–32 %).
  - It carries < 1 % of the squared physical error. The operator's error (≈ 80 % on S1) is common to all geometries.
- **Swapping in the other geometry's learned operator (C1) barely changes transport.** Validation single flight changes by +0.1 to +0.2 points (about 1–2 % of the error, small but systematic toward worse). Region-C renders change within reference noise.
- **Most of the S1 error lies inside the training-distribution incident span.**
  - Receiver use: 97 % of evaluation energy is in span, and the in-span error is 0.31–0.34.
  - Source use: the evaluation source fields at the panel lie 65 % outside the training span, and the error is equally large in and out of span (0.58–0.69 / 0.59–0.63).
- **The physical source use (C2_src) gives by far the best rendered sharp readers** in every inspected condition and seed: E2 T1 0.018–0.024 against 0.040–0.077 for C0. The pooled validation single-flight error, however, is worse (+0.5 to +0.8 points).
- **The physical receiver use (C2_rcv) shows the opposite:** pooled validation is better (−0.1 to −0.35 points) and rendered sharp readers are worse.
- **C2_both lies in between and does not beat C0 on the pooled validation.** The stage substitutions therefore have opposite-signed effects on pooled table metrics and on sharp-reader renders.

## Interpretation

- **H_G: not supported as a material contributor** on the measured population and inspected conditions.
  - The functional geometry dependence of the learned local operator exists, but it is small relative to both its physical error and its seed variance.
  - Replacing it with the other geometry's operator does not measurably change sharp transport.
  - This rejects, for this lineage, the causal premise that the principal ECRO failure is geometry information encoded in the local operator (the IDTS core hypothesis as stated in the dual-architecture verification). It does not test IDTS as an architecture.
- **H_R: supported.** The learned operator is about 80 % wrong on S1 in a geometry-common way, and the error is mostly inside the training span (receiver use). This points to representation / optimisation (cf. WL92: the rank-16 class ceiling plus learning beyond it), not geometry specialisation or extrapolation.
- **H_C: supported.** Exact physical substitution at one use improves one metric family and degrades the other. The rendered sharp readers favour a physical source use with the learned receiver and global terms. The pooled table favours a physical receiver use with the learned source.
- The learned receiver / global terms appear co-adapted to the learned source's errors. This is an inference: the opposite-signed outcomes are measured, but the compensation mechanism is not isolated.
- **Practical note for the owner (not an adoption).** On the inspected sharp conditions the single largest measured lever is the source-side local use. This is consistent with WL92B (exact q closes 96 % of the OWN seed gap) and WL93A (source supervision helps sharp renders). The pooled table metric does not reflect it.

## Unresolved questions

- No never-inspected sharp-reader held-out exists. Three E6 candidates failed the sharp-family clause, so every render-level statement is limited to inspected E2 / E5.
- Why does the physical source use improve sharp renders while worsening pooled single-flight error? Candidates: learned receiver / global compensation, the g00010-type encoding discrepancy weighted into the pool, or region weighting. Not isolated.
- Only two seeds. D_seed exceeds D_geo, so seed-level structure could hide geometry effects at other operating points.
- The probe basis covers only 35 % of evaluation source-field energy at the panel. The training-distribution light set does not span validation source fields well there.
- Results are confined to the tied ECRO lineage (WL92A own, WL93A T0). Untied (WL93B) and dual-supervised (WL93C) operators were not analysed.

## Visuals (`results/evaluation/93D/`, manifest with hashes)

- `fig1_functional.png`: E_phys, D_geo, D_seed and S_geo per material, probe and seed, with the thresholds.
- `fig2_subspace.png`: in- and out-of-span error and drift.
- `fig3_interventions_tables.png`: validation single / whole per arm.
- `fig4_interventions_renders.png`: region-C E_H per arm on E2 / E5 T1.
- `fig5_E2_T1_maps.png`, `fig5_E5k995_T1_maps.png`: signed error maps for own C0 / C1_both / C2_src / C2_rcv / C2_both on one shared scale.

## Reproducibility

- **Code / protocol:** freeze `4f3fc56`.
- **Server:** `~/rest_x93d` (`chain.status`; `results/`: ckpt93d, tests93d, functional93d, account93d, contrasts93d, final93d, ops/*.npz, fields93d.npz, E6 candidates / admissions / references, tables, renders).
- **Local copies:** `results/ecro_local93d/` (JSON, logs), `results/evaluation/93D/`.
- **Checkpoints:**
  - WL92A own_s8601 / s8602: file sha256 as in WL92B repro.json.
  - WL93A T0_s8601 / s8602: state sha256 as recorded.
- **Seeds:**
  - bootstrap 93004;
  - renders 86600 / 86601;
  - E6 rng 93550, references 93710 / 93730 + r + 100k.
- **Probe-basis configurations:** WL93A train-representative 16 geometries.
- **Evaluation:** WL93A validation 55, plus E2 / E5k995 T0 / T1.
- **Compute:** 78 min wall, of which E6 qualification took 62 min.

## Final report (Korean, for the owner)

**Implementation Fact (구현 사실)**
- 동결 체크포인트만으로 분석했다. 학습, 파라미터 변경, 새 아키텍처는 없다.
  - 대상은 WL92A own과 WL93A T0, 각 시드 2개이며 해시를 확인했다.
- 구현한 것은 진단 코드와 분석 코드다.
  - 유효 연산자 추출, 고정 프로브(canonical과 재질 수준 workload 기저), 교체 개입 C0 / C1 / C2(사용처별), 표 수준 회계와 렌더, 테스트.
- 전역 항, 기하, 가시성, 조명은 모든 arm에서 고정했다. 테스트로 확인했다.
- E6 held-out은 동결 WL93 규칙과 예산으로 판정했다. 후보 3개 모두 sharp-family 기준을 통과하지 못해 admit되지 않았다.
- dev 실행에서 Task A 부분집합 값이 동결 전에 출력됐다. 기준은 그 전에 고정돼 있었고 바꾸지 않았다. 이 사실은 공개했다.

**Measurement (측정)**
- S1 국소 연산자 오차(E_phys)는 0.75–0.82다.
- 기하 간 변화(D_geo)는 0.04–0.08이다. 편집 쌍만 보면 0.09–0.14로 Lambert 대조군의 3–6배다. 시드 간 변화(0.26–0.32)는 이보다 3–6배 크다.
- 기하 변화가 물리 오차에서 차지하는 몫(S_geo)은 0.003–0.008이다.
- 다른 기하의 연산자로 바꿔도(C1_both) 검증 single은 +0.07–0.21 포인트 바뀐다. 렌더 영역 C 변화는 잡음 이내다.
- source 사용만 M64로 바꾸면(C2_src):
  - 렌더 영역 C: E2 T1 0.018–0.024(C0는 0.040–0.077), E5 T1 0.013–0.018(C0는 0.025–0.033).
  - 검증 single은 +0.5–0.8 포인트 나빠진다.
- receiver 사용만 M64로 바꾸면(C2_rcv) 정반대다.
- 판정: 두 모델군 모두 **REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED**.

**Observation (관찰)**
- 기하 의존 변화는 실재하지만 작다. 오차의 99 % 이상이 모든 기하에 공통이다.
- receiver 사용에서는 오차 대부분이 학습 분포의 입사 공간 안에 있다(0.31–0.34).
- source 사용에서는 평가 입사장 에너지의 65 %가 학습 공간 밖에 있지만, 오차는 공간 안팎에서 비슷하다.
- 물리 source 사용은 sharp 렌더를 크게 개선하고, 풀링 표 지표는 악화시킨다.

**Interpretation (해석)**
- H_G(기하 특화가 주원인)는 이 계열에서 지지되지 않는다. 이는 IDTS 핵심 인과 가설을 반박하는 결과지만, IDTS 아키텍처 자체를 시험한 것은 아니다.
- H_R(표현·최적화 오차가 지배)은 지지된다.
- H_C(공적응)도 지지된다. 한쪽 사용만 물리 연산자로 바꾸면 표 지표와 렌더가 서로 반대로 움직인다. 보상 메커니즘 자체는 분리하지 못했다.
- 점검 조건에서 가장 큰 측정 지렛대는 source 쪽 국소 사용이다. WL92B, WL93A와 일치한다.

**Unresolved Question (미해결 질문)**
- 본 적 없는 sharp held-out이 없다. E6 후보 3개가 모두 실패했다.
- 물리 source 사용이 렌더와 풀링 표 지표에서 반대로 작용하는 원인을 분리하지 못했다.
- 시드가 2개뿐이다.
- 학습 광원 분포가 패널 source 입사장을 충분히 덮지 못한다.
- untied와 이중 감독 연산자는 분석하지 않았다.

**Architecture Verdict (아키텍처 판정)**
- **REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED** (두 모델군 모두).
- GEOMETRY_SPECIALIZATION_SUPPORTED는 G1, G2, G3 모두 미충족이다.
- 채택은 없다. IDTS 채택 아님. 후속 배치는 없다.

**Reproducibility (재현성)**
- 동결 `4f3fc56`, LabServer63 `~/rest_x93d`, 78분 소요.
- 체크포인트 해시, 시드, 산출물 위치는 위 Reproducibility 절을 참조한다.
