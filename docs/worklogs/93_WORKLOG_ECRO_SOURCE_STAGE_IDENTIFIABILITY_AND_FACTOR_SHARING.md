# Worklog — ECRO Source-Stage Identifiability and Local Factor Sharing (WL93) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93 ECRO Source-Stage Identifiability and Local Factor Sharing" (user, 2026-11-13).
- **Batch class.** Regular numbered batch under the AGENTS.md numbering policy (2026-11-13). It introduces a new mechanism (source-stage supervision and untied source / receiver local factors) and a new architecture decision question. It is not a WL92B evidence-gap suffix.
- **Question.** Can ECRO achieve reproducible sharp interreflection fidelity with a physically shared local response operator once its source-stage response is explicitly supervised? Or does it need source / receiver factor separation beyond supervision and capacity effects?
- **Hypotheses.**
  - H_IDENTIFIABILITY: stronger source-stage supervision improves the learned local response and sharp interreflection without untying.
  - H_SHARING: tied factors prevent independently accurate source and receiver responses even with identifiable source-stage supervision.
  - H_INTERACTION: neither alone suffices; their combination gives a reproducible improvement.
- **Fixed.**
  - ECRO global branch: frozen. The global prediction is identical across arms for the same seed.
  - Source encoding, ranks / widths, ray budget, ER materials and WL86 thresholds are unchanged.
  - No explicit GI, no source-class handling, no adoption, no follow-up batch.
- **Where.** Branch `Selective_Recompute`, LabServer63 (main). Code `experiments/ecro_stage93/`; results `results/ecro_stage93/`, `results/evaluation/93/`. Server root `~/rest_x93`, reading `~/rest_ecro86`, `~/rest_x92`, `~/rest_x92a`, `~/rest_x92b`.

## WL92B rule reconciliation (required before WL93 evidence; WL92B files unchanged)

**Written rule.** WL92B's frozen protocol (`attribution_rules.H_interact`) reads: "no dominant counterfactual, but at least two with f ≥ 0.25 for both OWN seeds in a case".

**Implemented rule.** `analyze92b.py` (line 110) implements the "no dominant" clause as `not (H_local SUPPORTED or H_global SUPPORTED)`. That checks only X_M12 and X_L3 for dominance.

**Under the written rule.**
- On region C of both E2 cases, X_L2 (single-flight term) and X_Q (exact once-scattered table) are DOMINANT (f ≥ 0.5 for both OWN seeds).
- H_interact is therefore INCONCLUSIVE, not SUPPORTED.
- With H_interact not SUPPORTED, only H_geometry is SUPPORTED. H_geometry is not a dominance hypothesis, so the frozen outcome mapping has no matching clause. The code falls through to **D**; the official record is **C**.

**Re-derivation.** `prep93.py recon92b` re-derives this from `~/rest_x92b/results/final92b.json` into `results/ecro_stage93/recon92b.json`. The dev chain re-derived: official H_interact SUPPORTED / outcome C; written rule INCONCLUSIVE / D (fall-through).

**Materiality for WL93.** Not material:
- No measurement, counterfactual value, E5 admission or E5 criterion changes.
- The mechanism finding that WL93 builds on is unaffected: the OWN error sits in the single-flight term (X_L2 dominant) and within it in the source-side once-scattered stage (X_Q dominant).
- WL93 thresholds and criteria do not depend on WL92B's outcome label.

**Interpretation going forward.** WL92B's "outcome C" should be read as "the single-flight / source-stage counterfactuals dominate, with co-adapted tied factors and a geometry-dependent panel operator" (outcome D under its written mapping). WL92B's worklog, protocol and results are preserved unchanged.

## Design (`experiments/ecro_stage93/protocol/ecro_stage93.json`)

**Architecture (model93.py).**

The table is B = S_θ(s) + G_frozen(s).

- **G_frozen.** The global term (3–5 events) of WL92A `own_s<seed>`, evaluated with that model's own frozen factors (c, a, P, Q). It is identical for every arm of a seed (test T_GLOBAL_IDENTITY) and is never trained.
- **S_θ.** The single-flight term a_r · (c_r · flight(a_s · (c_s · s))) from a trainable copy of the OWN generator (trunk and local heads). The copy's global heads are frozen and unused.
- **Disclosed change in every arm.** The single-flight local factors are decoupled from the global term's factors. This is needed to hold the global prediction identical across arms. "Tied" refers to source / receiver tying inside the single-flight term.
- **Modes.**
  - **tied:** c_s = c_r, a_s = a_r (the WL86 / WL92A operator).
  - **untied:** the source uses (head_c, head_a); the receiver uses duplicated heads (head_c2, head_a2); the trunk is shared.
  - **tiedcm:** capacity-matched tied control. One shared (c, a) = softplus(head(tb) + head2(tb)), where head2 has the same architecture as the local head. head2's hidden layers are freshly initialised from `torch.Generator` (seed = run seed + 9300) with the nn.Linear default bound; its output layer is zero. The result is a 2×-wide tied head with exactly the untied arm's trainable parameter count.
- **Pre-freeze correction (dev chain).**
  - The first tiedcm design averaged the pre-activations of two copy-initialised heads.
  - That design is symmetric: both copies receive identical Adam updates, so the pair stays identical forever. In the dev chain it reproduced the tied arm bit-for-bit (validation errors of C0 / CQ equal to T0 / TQ to every printed digit, with the same E_H), making it a degenerate control.
  - It was replaced before any evidence run.
  - The extra parameters of tiedcm (fresh hidden layers, zero output) are necessarily initialised differently from untied's copies. This is disclosed.
  - A post-training test now requires tiedcm and untied to differ from tied.
- **Dev chain bug fix.** The first dev training had zero trainable parameters: the OWN base was frozen before being deep-copied. The copy is now made trainable explicitly. The tests require n_trainable > 0, untied = tiedcm > tied, and a frozen base and frozen global heads.
- **Initialisation.** Every arm equals WL92A OWN at step 0 (test T_INIT_EQUIV). For untied the duplicated heads are copies; for tiedcm the added head outputs zero.

**Arms (2 seeds each, 8601 / 8602, initialised from the same-seed WL92A OWN).**

| Arm | Factors | Objective |
|---|---|---|
| T0 | tied | OWN (WL92A objective) |
| TQ | tied | OWN + source-stage |
| U0 | untied | OWN |
| UQ | untied | OWN + source-stage |
| C0 | tiedcm (capacity control for U0) | OWN |
| CQ | tiedcm (capacity control for UQ) | OWN + source-stage |

**Objective.**
- OWN = mean(((S + G − t)/den)²) + mean(((S − t2)/den)²) + mean(((G − (t − t2))/den)²), with den = t.mean(bins) + 0.05 · t.mean(all). The last term is constant because G is frozen.
- Source stage: + w · mean(((Bq − q)/den_q)²), w = 1 (fixed). Bq = a_s · (c_s · s) is the single-flight term's source-side once-scattered table.
- q is the offline physical once-scattered table: `quad87.qtable` (P 16, Mq 1,024), seed 93000 + 4·idx + lighting, same geometry and light. It is never used at runtime.

**Training.**
- Population: the WL92A population (WL86 train / val minus the WL92A E4 exclusion; 928 / 55 geometries × 4 lights).
- 8,000 steps; fresh Adam; lr 3e-4 → 1e-5 cosine after a 200-step warmup; clip 10; one geometry (4 lightings) per step.
- Data order from `default_rng(seed + 93)`: identical for every arm of a seed.
- No extension and no seed replacement.

**Identifiability preflight (STOP_TARGET_NOT_IDENTIFIABLE).**
- Computed on the 55 validation geometries and on E2 T0 / T1.
- The irreducible discrepancy of the exact frozen-ER operator applied through the deployed 16-sample encoding (panel-patch relative L1 of `enc_q(s16)` vs q) must be ≤ 0.5 × WL92A OWN's learned discrepancy (Bq vs q on panel patches), for both seeds.
- Rationale: if the deployed encoding cannot carry the target, supervising toward it cannot be distinguished from fitting noise.

**Held-out E6 (qualified before any training).**
- **Candidates.** Ordered from `default_rng([93550, k])`, using WL92B's E5 generation rule.
- **Lights.** WL86 test lights L1 / L2 by parity.
- **Disjointness.** Every box centre is ≥ 0.15 m from all 1,120 WL86 data files.
- **Geometry rule.** A, C, D, E and C_footprint ≥ 1,000 px.
- **References.** REF-ER / REF-P 8 × 8,192 spp (seeds 93710 / 93730 + r + 100k) and the diag92 truth split, for the first three geometry-admissible candidates in order.
- **Admission.** WL86 stage-0 Q1–Q4 (Q4 at SNR 4.0), Q6, Q7, plus the sharp family: the light → panel → receiver 2-event share of the region-C higher-order reading ≥ 0.08 with SNR ≥ 4.
- **No admission.** If no candidate is admitted: STOPPED_WORKLOAD_UNQUALIFIED, no training, no held-out claim.
- **E5.** WL92B candidate 995 is re-evaluated as an inspected (not independent) workload.

**Evaluation.**
- Tags: E2 T0 / T1, E1 T1 LSPOT, E5k995 T0 / T1 / LMIX / LSPOT, E6 T0 / T1 / LMIX / LSPOT.
- Models: 12 arm models, plus WL92A own / sum for context and reproduction.
- Renders: 2 × 4,096 spp (seeds 86600 / 86601, common random numbers).
- Controls on E6: an ORACLE table and the explicit X_EXP.

**Criteria (fixed now).**
- **ADV(X, Y).** On E6 region C at T1 and at T0, for both seeds: E_H(X) ≤ E_H(Y) − max(noise_X, noise_Y, 0.005).
- **STAGE(X, Y).** Panel-patch Bq-vs-q relative L1 on E6 T1 and E2 T1: X ≤ 0.9 × Y, for both seeds.
- **NOREG(X).** Against the same-seed T0:
  - no WL86 gate passed by T0 is lost (F1; F2 in A and D; F4);
  - stationary and D E_H ≤ 1.25 × T0 + 0.002 on E2 / E5 / E6 T0 and T1;
  - the same bound on stationary E_H for every source configuration.
- **DYN(X).** F3 (region D) is not lost against T0 on E2, E5 and E6.
- **PASS_sup(X, Y)** = ADV ∧ STAGE ∧ NOREG ∧ DYN. **PASS_share(X, Y)** = ADV ∧ NOREG ∧ DYN.

**Labels.**
- INTERACTION_SUPPORTED: PASS_sup(UQ, T0) ∧ PASS_share(UQ, CQ) ∧ ¬PASS_sup(TQ, T0) ∧ ¬PASS_share(U0, C0).
- SHARING_SUPPORTED: PASS_share(UQ, CQ) ∧ PASS_share(UQ, TQ), unless INTERACTION applies.
- IDENTIFIABILITY_SUPPORTED: PASS_sup(TQ, T0) ∧ ¬PASS_share(UQ, CQ).
- NOT_SUPPORTED: none of PASS_sup(TQ, T0), PASS_sup(UQ, U0), PASS_share(U0, C0), PASS_share(UQ, CQ), PASS_sup(UQ, T0) holds.
- INCONCLUSIVE: any other combination.
- STOPPED-INCONCLUSIVE: a test, control or reproduction fails, the target is not identifiable, or E6 is not admitted.
- Labels do not authorise adoption.

**Tests.**
- T_INIT_EQUIV: every arm at step 0 = WL92A own on E2 T1 (≤ 1e-5).
- T_GLOBAL_IDENTITY: the global term is identical.
- T_TIED: tensor identity for tied / tiedcm; for untied, a source-only gradient leaves the receiver heads at zero gradient.
- T_LINEAR: zero, scaling, superposition and non-negativity.
- T_QTARGET: the stored E2 T1 q is reproduced.
- T_DET.
- T_REPRO: WL92A own re-rendered on E2 T0 / T1 equals WL92A's own renders in region-C E_H (≤ 1e-9).

**Chain (run93.sh, one detached setsid chain).**
1. Inputs and the WL92B reconciliation.
2. Validation q, then the identifiability gate.
3. Training q in parallel with E6 generation.
4. E6 references and admission, in candidate order.
5. E6 source references and oracle fields.
6. Pre-training tests.
7. 12 trainings.
8. Post-training tests.
9. Tables, local matrices, timing.
10. Renders.
11. Analysis, visuals.

## Dev chain and full-scale probes (before the freeze; not evidence)

**Dev chain** (`~/rest_x93dev`, `--dev`: Mq 64, 48 training geometries, 200 steps, 64 spp, E6 forced to the first geometry-admissible candidate).
- Every stage ran end to end: reconciliation, identifiability gate, q generation, E6 generation / references / admission, tests, training, tables, renders, analysis, visuals.
- Two defects were found and fixed before any evidence run:
  - **Zero trainable parameters.** The base was frozen before being copied.
  - **Degenerate capacity control.** The averaged copy pair gave C0 ≡ T0 bit-for-bit (see the design section).
- After the fixes, the tests pass:
  - init equivalence 1.7e-7;
  - global identity exact;
  - untied receiver-head gradient 0 under a source-only loss;
  - n_trainable: tied 405,424, untied = tiedcm 487,952;
  - trained C0 vs T0 single-term difference 2.0e-3 / 2.2e-3, U0 vs T0 0.05 / 0.15 (non-degenerate).

**Full-scale probes** (`~/rest_x93probe`, RTX 3080 Ti 12 GB, 125 GB RAM, 32 cores).

| Stage | Measurement | Chosen setting |
|---|---|---|
| q generation (Mq 1,024) | ≈ 0.35 s per geometry × 4 lights; 6 concurrent shards: GPU 31 %, 4.4 GB (CPU / IO-bound, short) | 8 concurrent shards of 128 (≈ 3 min total) |
| training (300-step probe, full step cost) | 1 process: 46 ms/step, GPU 33 %, 1.2 GB; 4 processes: GPU 82 %, 5.6 GB, 8.4 s per run-equivalent; 6 processes: GPU 89 %, 8.7 GB, 8.7 s (no throughput gain) | 4 concurrent (≈ 15 min per 8,000-step run; 12 runs ≈ 45–50 min); RSS 2.2 GB with 93 items, so ≈ 13 GB per process at full population (52 GB for 4, within 121 GB available) |
| E6 references / split / sources / oracle field | identical code and concurrency as WL92B (pool 6: 32 REF renders at 8,192 spp ≈ 19.5 min per candidate; split ≈ 30 s; sources and oracle fields ≈ 5 min) | pool 6, up to three candidates |
| renders | identical path to WL92B (pool 5); single render 8 s at 4,096 spp | pool 5 (≈ 360 renders) |
| T_REPRO | WL92A own_s8601 E2 T1 table via the WL93 path, rendered at 4,096 spp seed 86600, vs WL92A's render: max \|hl\| difference 6.0e-8 (float32 ULP), 1,893 region-C pixels differ, region-C single-half E_H difference 4.2e-9 | tolerance set to 1e-6 (the pre-written 1e-9 would fail on ULP-level table differences; 1e-6 is three orders below the 0.005 decision margin) |


## Freeze and execution

- **Freeze.** Commit `c30c797` (protocol v1, code, worklog pre-freeze sections), deployed to `~/rest_x93` with `git archive` (LF). One detached setsid chain, started 2026-11-13 06:42:56 UTC.
- **S0.**
  - Inputs copied and hashed (identical hashes to the dev chain).
  - WL92B reconciliation reproduced: official H_interact SUPPORTED / outcome C; written rule INCONCLUSIVE / D fall-through.
- **S1.**
  - Validation q: 55 geometries × 4 lights, Mq 1,024, 17 s.
  - Identifiability gate: **FAIL** → `STOP STOP_TARGET_NOT_IDENTIFIABLE` at 06:44:01 UTC.
- **Not run (by protocol).** q for the training set, E6 generation / references / admission, tests, training, tables and renders.
  - Under the protocol, E6 was not generated and no model was trained or rendered.
  - The other entries in the stop list of `final93.json` are consequences of the stop: E6 unqualified, tests / reproduction / training absent.
- **Analysis crash on the stop path.** The frozen `analyze93.py` raised a TypeError in STAGE when no diagnostics existed: an empty dict was returned instead of None. The chain ended `CHAIN_FAILED` after the STOP. See Amendment 1.

## WL93 Amendment 1 (non-scientific execution fix, appended after the stop)

- **Change.** In `analyze93.py`, the STAGE value extraction now returns None when diagnostics are absent: `(((m or {}).get('diag') or {}).get('T1') or {}).get(...)`. That was the intended behaviour; the original expression could return `{}`.
- **What is unchanged.** No rule, threshold, criterion or label mapping. The stop was decided by the chain before analysis.
- **Rerun.** `analyze93.py` (log `A_final_amend1.log`) and `viz93.py` were rerun. Label: **STOPPED-INCONCLUSIVE**, primary cause STOP_TARGET_NOT_IDENTIFIABLE.
- **Status.** Applied by the agent to record the stop; the amendment awaits explicit owner approval under the numbering policy. Without it, the label follows directly from the chain status line `STOP STOP_TARGET_NOT_IDENTIFIABLE`.

## Result: identifiability preflight (frozen rule)

Values are panel-patch relative L1 against the physical once-scattered table q (Mq 1,024).

| Set | Exact frozen-ER through deployed encoding (irreducible) | WL92A OWN learned Bq | Bound 0.5 × learned | Result |
|---|---|---|---|---|
| validation, seed 8601 | 0.508 | 0.970 | 0.485 | FAIL |
| validation, seed 8602 | 0.508 | 0.966 | 0.483 | FAIL |
| E2 T1 | 0.362 | 1.140 / 1.068 | 0.534 | pass |
| E2 T0 | 0.362 | 1.118 / 1.084 | 0.542 | pass |

- **Gate.** The rule requires every set and seed to pass. Validation fails by 0.023–0.025 (irreducible / learned = 0.52–0.53 vs 0.50).
- **Label.** **STOPPED-INCONCLUSIVE** (STOP_TARGET_NOT_IDENTIFIABLE). No hypothesis (H_IDENTIFIABILITY / H_SHARING / H_INTERACTION) was tested.
- **Not authorised by this result.** No architecture decision, no threshold change, no rerun and no successor.

## Post hoc descriptive diagnostic (labelled; gate, label and rules unchanged)

`posthoc93.py` covers the same 55 validation geometries × 4 lights (log `P_posthoc.log`, `posthoc_ident.json`). It reproduces the gate values exactly (0.5079 / 0.9704 / 0.9658).

- **The pooled failure is concentrated in a few geometries.**
  - Per geometry, the irreducible term has median 0.358 (IQR 0.344–0.385).
  - The median irreducible / learned ratio is 0.36, and 52 of 55 geometries meet the 0.5 ratio individually.
  - Geometry g00010 carries 19.7 % of the pooled panel q mass with an irreducible term of 0.99. g00039 (8.4 % of the mass) is at 0.68.
- **Those geometries have a noisy target.**
  - A second independent q realisation (seed + 500,000) differs from q by 0.91 (g00010) and 0.56 (g00039), against a median of 0.043.
  - Across geometries, the irreducible term correlates with this target noise (r = 0.92).
  - Pooled q-vs-q is 0.29. Averaging the two q realisations barely lowers the pooled irreducible term (0.508 → 0.496), which suggests heavy-tailed rather than Gaussian target noise on those geometries.
- **Level is right; placement is not.** Pooled panel flux enc_q / q is 1.005, so the irreducible term is directional placement, not level.
- **Reading.** Descriptively, the frozen gate measured two things at once:
  - the deployed encoding's directional information loss (≈ 0.29–0.42 on typical geometries, which pass);
  - the Monte Carlo noise of the Mq 1,024 once-scattered target on a few panel-illumination configurations, which dominates the pooled value.
- **Not shown.** This does not show that the target is identifiable. It also does not license changing the gate, excluding geometries or rerunning. A noisy q would also be a noisy supervision target on those geometries.

## Visuals

`results/evaluation/93/fig1_identifiability_gate.png` (manifest with hashes):
- **(a)** The frozen gate per set and seed.
- **(b)** Post hoc, per geometry: irreducible vs learned, against the 0.5 line; marker area is the share of panel q, colour is target noise.
- **(c)** Post hoc: irreducible term vs target noise.

No model was trained or rendered, so no image comparison exists. The dev chain plumbing figures (`~/rest_x93dev`) are not evidence and are not exported.

## Reproducibility

- **Code and protocol.** Frozen commit `c30c797`. Amendment 1 and the post hoc diagnostic / visual (`posthoc93.py`, `vizgate93.py`) are in the closing commit.
- **Server.** LabServer63 (RTX 3080 Ti 12 GB); root `~/rest_x93` (`chain.status`, `results/ident.json`, `recon92b.json`, `posthoc_ident.json`, `final93.json`, logs).
- **Local copies.** `results/ecro_stage93/`, `results/evaluation/93/`.
- **Inputs (read-only).** `~/rest_ecro86` (WL86 data / masks / references), `~/rest_x92` (E4 choice), `~/rest_x92a` (OWN checkpoints, f2 targets, exclusion), `~/rest_x92b` (final92b, E5 inputs, CF tables).
- **Seeds.** q seed 93000 + 4·idx + lighting; post hoc second realisation + 500,000.
- **Compute.** About 1 GPU-minute of evidence (gate) plus about 5 minutes post hoc. The dev chain and probes (`~/rest_x93dev`, `~/rest_x93probe`) are not evidence.

## Final report (Korean, for the owner)

**Implementation Fact (구현 사실)**
- 아키텍처와 프로토콜을 동결 커밋 `c30c797`로 고정한 뒤 실행했다.
  - 전역 분기는 WL92A own에서 동결했고, single-flight 로컬 연산자만 학습한다.
  - 처리군: tied / untied / 용량 대조 tiedcm × source-stage 감독 유무 = 6 arm × 2 seed.
- 동결 전 dev 체인에서 결함 두 개를 발견해 고쳤다.
  - 학습 파라미터가 0개였다. 동결된 base를 복사했기 때문이다.
  - 복사-평균 방식의 tiedcm은 Adam에서 대칭이 깨지지 않아 T0와 비트 단위로 같았다. 0으로 초기화한 추가 헤드로 바꿨다(파라미터 수 487,952로 untied와 동일).
- WL92B 규칙 불일치:
  - 구현은 X_M12 / X_L3의 지배성만 검사했고, 작성된 규칙은 모든 counterfactual을 대상으로 한다.
  - 작성 규칙을 따르면 H_interact는 INCONCLUSIVE, 결과는 D(매핑 fall-through)다. 공식 기록은 SUPPORTED / C다.
  - 측정값은 바뀌지 않으므로 WL93에는 영향이 없다. WL92B 파일은 수정하지 않았다.
- 본 체인은 식별성 사전 게이트에서 프로토콜대로 정지했다. 학습용 q, E6, 학습, 렌더는 실행되지 않았다.
- 정지 경로에서 분석 스크립트가 크래시했다. Amendment 1(비과학적 실행 수정)로 고쳤고, 소유자 승인을 기다린다.

**Measurement (측정)**
- 검증 세트 55개 기하 × 4 조명, 패널 패치 기준:
  - 배포 인코딩을 통한 정확한 frozen-ER 연산자 vs q = 0.508.
  - WL92A OWN 학습 Bq vs q = 0.970 / 0.966.
  - 기준 0.485 / 0.483을 넘었다(비율 0.52–0.53). FAIL.
- E2 T1 / T0: 0.362, 기준 0.534 / 0.542. 통과.
- 라벨: **STOPPED-INCONCLUSIVE (STOP_TARGET_NOT_IDENTIFIABLE)**.

**Observation (관찰, 사후 기술 — 판정 불변)**
- 기하별 비가역 항의 중앙값은 0.358, 비율 중앙값은 0.36이다. 55개 중 52개는 개별적으로 0.5 비율을 충족한다.
- 풀링 실패는 소수의 기하에 집중돼 있다.
  - g00010: 패널 q 질량의 19.7 %, 비가역 항 0.99, 독립 q 재실현과의 차이 0.91.
  - g00039: 질량 8.4 %, 비가역 항 0.68, 재실현 차이 0.56.
- 비가역 항과 q 자체의 MC 잡음 사이 상관은 r = 0.92다.
- 패널 flux 비는 1.005다. 수준(level)은 맞고, 방향 배치(placement)가 손실된다.

**Interpretation (해석)**
- 동결 게이트는 두 가지를 함께 측정했다.
  - 배포 인코딩의 방향 정보 손실(일반 기하에서 약 0.29–0.42).
  - 일부 패널 조명 구성에서 Mq 1,024 once-scattered 타깃이 갖는, 꼬리가 두꺼운 MC 잡음.
  - 풀링 값은 후자가 지배한다.
- 이는 타깃이 식별 가능하다는 증거가 아니다. 게이트 변경, 기하 제외, 재실행의 근거도 아니다. 잡음이 큰 q는 해당 기하에서 감독 타깃으로도 잡음이 크다.
- H_IDENTIFIABILITY / H_SHARING / H_INTERACTION 중 어느 것도 시험되지 않았다.

**Unresolved Question (미해결 질문)**
- once-scattered 타깃의 잡음이 큰 구성(g00010, g00039 유형)이 물리적으로 어떤 경우인지(스침각, 집광 패널 조명 등) 확인되지 않았다.
- 배포 인코딩 자체의 방향 정보 손실(약 0.36)이 감독으로 개선할 여지를 얼마나 제한하는지 측정되지 않았다.
- 저잡음 타깃에서 source-stage 감독과 factor 분리가 어떤 효과를 내는지 측정되지 않았다.
- Amendment 1 승인 여부는 소유자가 결정한다.

**Architecture Verdict (아키텍처 판정)**
- STOPPED-INCONCLUSIVE.
- 공유(tied) 로컬 연산자에 대해서도, 분리(untied)에 대해서도 결정하지 않는다.
- 채택 없음, 후속 배치 승인 없음. 결정은 소유자에게 있다.

**Reproducibility (재현성)**
- 동결 커밋 `c30c797`, LabServer63 `~/rest_x93`. 로컬 사본은 `results/ecro_stage93/`, `results/evaluation/93/`.
- q 시드는 93000 + 4·idx + lighting, 사후 재실현은 + 500,000. 자세한 내용은 위 Reproducibility 절을 참조.
