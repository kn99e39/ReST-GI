# Worklog — GM1 Intermediate Transport Representation Fidelity-Gap Attribution and Feasibility (2026-11-12)

## Origin and scope

- **Brief.** "ReST-GI — GM1 Intermediate Transport Representation Fidelity-Gap Attribution and Feasibility" (user, 2026-11-12). One bounded diagnostic batch.
- **Question.** Is the common sharp fidelity floor of the converged explicit relations (R0 / R1 / C3UR, W1 R_mirror E_H ≈ 0.038 against the oracle table's ≈ 0.014) caused primarily by intermediate spatial / angular projection? What evidence bears on keeping or reconsidering the present explicit representation?
- **Primary hypothesis (prospective).** The gap is information discarded when the sender radiance is projected onto the patch × 64-bin table before each gather, not relation sampling variance.
- **Preserved.**
  - WL84 ER and the WL86–WL89 code, protocols, results and verdicts (imported, never edited).
  - R0 = WL87 EXP; C3UR = WL89, kept as the explicit relation development baseline.
  - The WL89 workloads (W1 = V3, W2 = X2), references, masks, oracle tables and order fields.
  - WL89's H1 verdict is not reinterpreted.
- **Not done.** No learned operator, no new architecture, no GM2, no adoption. The diagnostics are not deployable representations.
- **Where.** Branch `Selective_Recompute`, LabServer63. `experiments/gm1_gap90/`, `results/gm1_gap90/`, `results/evaluation/90/`. Server `~/rest_gap90`, reading `~/rest_rel89`.

## A. What the code computes (verified against the sources)

| Path | Code | What it does |
|---|---|---|
| Table recurrence (every WL87–WL89 relation) | `xport87.recurrence`, `rel88.recurrence` | L₁ = q, L_{k+1} = G(P L_k), B = L₂ + … + L₅ |
| Projection P at the sender | implicit in the table read `L[hit, dbin]` (`xport87.XOp.gather`, `rel89.SendRelR.apply`) | the gather reads the hit patch's table entry: patch-constant in space, constant over a 0.098 sr direction bin. Applied before **every** gather, so it compounds through the orders |
| Source projection q | `quad87.qtable` | once-scattered radiance at 16 points per patch, bin-averaged by 1,024 MIS samples. Against the path-traced order-1 field: relative L1 0.013, flux 1.0007 (WL89). q is therefore an accurate patch × bin *projection* of the exact 1-event field |
| Gather G at the receiver | per-ray frozen-ER weights W (`quad87.Quad`), exact local direction, sphere rows per band | writes the receiver's own patch × bin entry |
| Camera readout | `pt86` mode `ecro` | LOW (e1, e2 path-traced) + B[patch(x₂), bin(x₂ → x₁)] |
| Oracle table | `pt86.Ctx.field` | receiver patch × bin average of the path-traced 2–5-event radiance; **no P at any sender** |

- Table and oracle carry the same event family (2–5 events at the receiver; no next-event at the probe vertex) and use the same readout.
- With relation sampling converged, the differences that remain are:
  - (i) P at the senders, at each of the four gathers;
  - (ii) the receiver quadrature (W, receiver point sampling);
  - (iii) the q estimator.
- WL89's attribution of the gap to "relation discretisation" was an inference from converged − oracle. It did not separate (i) from (ii) / (iii).

## B. Prior evidence that shaped the design (disclosed)

- WL89, already inspected: converged R0 / R1 / C3UR 0.038 against ORACLE 0.014.
- Deployed R0 with the exact first-gather radiance (repaired RAYSEND, RS0) was 0.023, against R0 0.060.
- Every diagnostic below is new and was fixed before being computed.

## C. Diagnostics (frozen in `protocol/gm1_gap90.json`)

**Exact-read operators.** They run on a pooled R0 relation: 64 fresh realisations, WL87 weights.
- For every state ray, the gather reads a path-traced k-event radiance instead of the table entry. The tracer is the pt86 estimator split by event count.
- The radiance is optionally projected at the read:

  X(a, s)[k] = G(P_{a,s} Lx_k), k = 1…4

- a = sub-bins per direction bin (0 = exact direction); s = sub-patches per patch (0 = exact point).
- 4 samples per read, averaged over the valid ones, as the table average.

| Operator | Meaning |
|---|---|
| X(0,0) = EE | no intermediate projection |
| X(1,0) | angular projection only (exact point, bin-averaged direction) |
| X(0,1) | spatial projection only (patch-averaged point, exact direction) |
| X(1,1) | the table's projection applied **once** to the exact field |
| M_m (m = 0…4) | first m gathers exact, the rest by the pooled table operator. M₀ = table recurrence; M₄ = EE |
| P1only, P2plus, EA1, SE1 | projection at order 1 only / at orders ≥ 2 only; angular / spatial projection at order 1 only |
| Fixed counterfactual grid (no sweep) | angular 4 / 16 sub-bins, spatial 4 / 16 sub-patches, joint (4,4), (16,16). Single-read projections: a lower bound of the loss of a refined table recurrence, since compounding is excluded |

**Rules** (primary: W1 T1, R_mirror; E_H with common random numbers across arms).

| Rule | Definition |
|---|---|
| Gap | G₀ = E_H(M₀) − E_H(ORACLE). It must exceed 2 × its noise, else stop |
| Isolation | E_H(EE) − E_H(ORACLE) ≤ 0.25 G₀ → ISOLATED; ≤ 0.5 G₀ → PARTIAL; else NOT ISOLATED (another shared approximation) |
| Factor | with J = X(1,1) − EE, A = X(1,0) − EE, S = X(0,1) − EE: ANGULAR if A ≥ ⅔ J and S ≤ ⅓ J; SPATIAL in the mirrored case; otherwise BOTH / INTERACTING |
| Order | m* = smallest m with M_m within 0.25 G₀ of the oracle; first-order share = (P1only − EE) / J; compounding = (M₀ − X(1,1)) / G₀ |
| Field level | first table order whose error exceeds 2 × truth noise on the sharp sender classes |
| Ladder | smallest grid point within 0.25 J of EE |

**Feasibility probe.** Interpreted only if isolation holds and the first gather carries the loss (m* = 1, or first-order share ≥ 0.5).
- R0X1 / C3URX1: the first gather reads the exact once-scattered radiance per relation ray. That is one deterministic next-event evaluation per ray per source; the relation is unchanged and source-independent. Later orders use the table operator.
- Evaluated on the deployed draw and the 16 WL89 realisations.
- X1 is MATERIAL if the paired C3URX1 − C3UR interval in R_mirror is < 0.
- X1 is ECONOMICALLY PLAUSIBLE if its measured per-edit cost is ≤ 1.25× and its source-only update ≤ 2× C3UR's. Costs are measured warm and alone, with compilation excluded.
- The probe is a diagnostic proof of feasibility of one mechanism. It is not a candidate and is not adopted.

**Tests** (gate; any failure stops):
- per-order tracer = pt86 estimator on camera rays;
- sub-bin / sub-patch samples stay inside the read's cell;
- pooled exact gather vs path-traced order fields (flux within 3 %);
- single-read table projection of the 1-event field = pooled table gather of q (flux within 2 %);
- X1 plumbing ≤ 1e-5;
- relation digests unchanged by the exact reads;
- replay ≤ 1e-5.

## Pre-freeze development (non-evidence; dev geometry Wk via `~/rest_rel89dev`)

- **Dev tests: all pass first run.**
  - Tracer vs pt86: 1e-7.
  - Exact gather flux per order: 1.004 / 1.003 / 1.001 / 0.994.
  - Projection-once vs table gather of q: 1.001.
  - Plumbing: 4e-9 / 0.
- **Scale probe of the pooled exact-read stage.** 3.3 s per realisation for the 10 variants (about 22 M probe paths), after a 20 s first-call compilation. At 64 realisations that is about 4 min per configuration; three configurations run concurrently.
- **Dev chain end to end.** See the evidence section for the freeze.
- **Dev chain end to end** (`WL90_DEV=1`, dev geometry Wk through the WL89 dev root, 64 spp, 4 pooled realisations). Every stage completed in 2 min: tests, order field, pooled exact-read tables, feasibility tables, renders, costs, analysis, visuals. Dev values are noise-dominated and are not evidence.
- **Implementation optimisation of the probe-ray buffers** (same operator; tests and plumbing re-verified).
  - Buffers are now built on the GPU from the resident relation segments (torch → Dr.Jit directly), and the per-ray radiance is returned as a torch tensor.
  - Buffer rebuild on the dev geometry: 70–90 ms → 1.1 ms (C3UR) / 7 ms (R0).
  - Exact once-scattered radiance per source: about 10 ms for about 0.6 M rays; the q table takes 25 ms.

## Evidence run (freeze `51b93b8`; LabServer63 `~/rest_gap90`; 2026-11-12 07:05–07:31 UTC; one chain, no rerun, no amendment)

| Stage | Wall | Mean GPU | Outcome |
|---|---|---|---|
| tests ‖ W2 T1 order field ‖ pooled exact-read tables (3 configurations, 64 realisations × 10 variants, 222–242 s each) | 4.7 min | 26 % | **GATE TESTS PASS** (7 / 7); tables finite, non-negative |
| feasibility tables (17 draws × 4 arms × 2 configurations) | 0.5 min | 22 % | plumbing 6e-9 / 0 |
| renders (~390, 4,096 spp, WL89 render seeds) | 19.2 min | 80 % | — |
| costs (alone; no foreign GPU process) | 0.2 min | — | measured |
| analysis, visuals | 0.2 min | — | — |

- The exact-read and feasibility stages are bound by Python / host work: per-realisation quadrature (1.3 s) and numpy transfers of the per-ray radiance.
- GPU utilisation during the renders was 80 %.

**Tests (evidence parameters).**

| Test | Result |
|---|---|
| Tracer vs pt86 | 2e-7 |
| Sub-cell membership | 1.0000 (16 sub-bins: 0.99999) |
| Pooled exact gather vs path-traced order fields, flux | 1.0009 / 1.0001 / 1.0014 / 0.9988 |
| Projection-once of the 1-event field vs table gather of q | flux 0.9992, relative L1 0.007 |
| Replay | 0 |
| Plumbing | 3e-9 / 0 |
| Digests | unchanged |

## Final report (Korean)

### Implementation Fact

- **A. 코드가 실제로 계산하는 것 (소스 확인).**
  - WL87–WL89의 모든 relation은 L₁ = q, L_{k+1} = G(P L_k), B = L₂ + … + L₅를 계산합니다.
  - P는 gather가 hit patch의 table entry `L[hit, dbin]`를 읽는 순간에 생기는 암묵적 사영입니다. 공간은 patch 상수, 방향은 0.098 sr bin 상수입니다.
  - P는 **네 번의 gather마다** 적용되므로, order를 거치며 누적될 수 있습니다.
- **B. 물리 reference가 재는 것.**
  - oracle table은 path-traced 2–5 event radiance의 수신 patch × bin 평균입니다. sender 쪽에는 사영이 전혀 없습니다.
  - event family와 camera readout은 table과 같습니다.
- **C. 진단 구현.**
  - 진단은 pooled R0 relation(64개 새 realization, WL87 weight) 위에서 돌립니다.
  - state ray마다 table 대신 path-traced k-event radiance를 읽습니다. tracer는 pt86 estimator를 event 수로 분해한 것이고, pt86과 2e-7 수준에서 일치합니다.
  - 읽는 순간에만 선택적으로 사영합니다. X(a, s)에서 a는 방향 sub-bin 수, s는 sub-patch 수, 0은 exact입니다.
  - 추가 진단: hybrid M_m(앞의 m개 gather만 exact), order별 조합(P1only, P2plus, EA1, SE1), 고정 해상도 grid.
- **Feasibility probe.**
  - R0X1 / C3URX1은 relation을 바꾸지 않고(source-independent 유지) first gather만 relation ray별 exact once-scattered radiance로 읽습니다. ray당 결정론적 next-event 1회입니다.
  - 이후 order는 기존 table operator를 씁니다.
  - probe ray buffer는 GPU-resident입니다.
- **범위 밖.** 진단 operator(exact read, hybrid)는 배포 가능한 표현이 아닙니다. 채택한 것은 없습니다.

### Measurement

**공통 gap 재확인 (W1 T1 R_mirror E_H, 4,096 spp, 같은 render seed).**

| operator | E_H |
|---|---|
| M₀ (64개 realization으로 수렴한 table recurrence) | 0.0363 |
| ORACLE | 0.0137 |

- gap G₀ = 0.0226이고, arm 간 CRN noise는 0.0001입니다.
- pooled table의 split-half noise는 field 수준 0.7–1.0 %입니다.

**Isolation.**
- 중간 사영이 없는 operator EE는 0.0135로, ORACLE과의 차이가 −0.0002입니다. gap의 100.9 %가 사라집니다.
- W1 T0에서는 97 %, W2 T1에서는 102 %가 사라집니다.
- 세 configuration × 네 region이 모두 ISOLATED입니다.
- 따라서 수신측 quadrature, relation, readout 등 다른 공유 근사는 gap에 기여하지 않습니다.

**요인 분해 (R_mirror E_H − EE).**

| | W1 T1 | W1 T0 | W2 T1 (held-out) |
|---|---|---|---|
| 한 번 사영 J = X(1,1) − EE | +0.0207 | +0.0309 | +0.0170 |
| **angular only** A = X(1,0) − EE | **+0.0226** | **+0.0339** | **+0.0177** |
| spatial only S = X(0,1) − EE | −0.0009 | −0.0011 | −0.0027 |
| order 1에서만 사영 (P1only − EE) | +0.0198 | +0.0311 | +0.0170 |
| order ≥ 2에서만 사영 (P2plus − EE) | +0.0009 | +0.0007 | −0.0004 |
| 누적 (M₀ − X(1,1)) | +0.0021 | −0.0003 | −0.0001 |

- 규칙 판정: **ANGULAR** (A ≥ ⅔ J, S ≤ ⅓ J), 세 configuration 모두.
- patch 평균(공간 사영)은 E_H를 늘리지 않습니다.

**Order.**
- m* = 1: first gather만 exact로 바꾸면(M₁) 0.0146 / 0.0159 / 0.0149로, ORACLE과 0.001–0.002 차이입니다.
- first-order share는 0.96 / 1.01 / 1.00이고, 누적 몫은 gap의 9 % / −1 % / 0 %입니다.
- field 수준: W1 T1에서 table 오차가 truth noise의 2배를 넘는 첫 곳은 **panel의 order 2**입니다.

  | panel order 2 | field relative L1 |
  |---|---|
  | M₀ | 0.137 |
  | EE | 0.053 |
  | angular only | 0.148 |
  | spatial only | 0.089 |
  | truth noise | 0.043 |

- 다른 class와 다른 order는 noise 수준입니다.

**고정 해상도 grid (single-read, EE 대비 잔여, W1 T1 / T0 / W2 T1).**

| 사영 해상도 | EE 대비 잔여 |
|---|---|
| angular 4 sub-bin (256 방향) | +0.0038 / +0.0063 / +0.0016 (J의 18 % / 20 % / 9 % → 규칙 충족) |
| angular 16 (1,024 방향) | +0.0007 / +0.0012 / −0.0031 |
| spatial 4 / 16 | 변화 없음 (≤ 0.0002, 일부 음수) |
| joint (4,4) / (16,16) | +0.0028 / +0.0006 (W1 T1) |

**Sharp 밖의 영역.**
- C_ss, stationary, D의 gap은 0.0002–0.0012로 작고, 모두 사영 없는 operator에서 사라집니다.
- broad 영역 D는 요인이 섞여 있습니다(BOTH, m* 2–4). 다만 절대값이 0.0003 이하입니다.

**Feasibility probe (deployed relation 하나, 16개 realization).**

| W1 T1 R_mirror | E_H mean ± SD | signed | 통과율 |
|---|---|---|---|
| R0 | 0.047 ± 0.015 | −0.034 | 81 % |
| R0X1 | 0.051 ± 0.015 | −0.010 | 75 % |
| C3UR | 0.040 ± 0.005 | −0.027 | 100 % |
| **C3URX1** | **0.024 ± 0.005** | −0.001 | 100 % |

- paired 차이:
  - C3URX1 − C3UR: T1 **−0.016 [−0.020, −0.012]** (16 / 16 개선), T0 −0.021 [−0.028, −0.014].
  - R0X1 − R0: +0.003 [−0.006, +0.012]. 편향은 줄지만 E_H는 개선되지 않습니다.
- WL89의 RS0 0.023은 deployed draw 하나의 값이었습니다.
- plumbing(table radiance로 first gather 재현)은 6e-9 / 0입니다.

**측정 비용 (RTX 3080 Ti 단독, warm median).**

| 단계 | ms |
|---|---|
| q table | 26.1 |
| relation ray별 exact L1 (C3UR, 약 0.53 M ray) | 10.4 |
| relation ray별 exact L1 (R0) | 10.7 |
| probe-ray buffer 재구성 (C3UR) | 1.2 |
| probe-ray buffer 재구성 (R0) | 7.2 |

| 갱신 비용 | C3URX1 | C3UR | 비율 |
|---|---|---|---|
| source-only | 20.6 ms | 36.7 ms | 0.56× |
| per-edit (geometry 갱신 31.8 ms 포함, WL89 값 사용) | 68.2 ms | 83.6 ms | 0.82× |

- 추가 상태: ray buffer 12.6 MB, ray별 radiance 6.3 MB(일시). q table(1 MB)은 필요 없어집니다.

**투영 (측정 아님).** table 자체를 a × s배로 세분하는 경우입니다.
- table entry와 sender relation ray가 a × s배가 됩니다. 256 방향이면 4×, 1,024 방향이면 16×입니다.
- 이것은 single-read 하한에 근거한 투영입니다. 세분한 table recurrence를 실제로 돌린 측정이 아닙니다.

### Observation

1. 공통 fidelity floor는 relation 표본 문제가 아닙니다. 수신측 quadrature 문제도 아닙니다. **sender radiance의 방향 bin 사영 하나**로 설명됩니다. 사영을 없애면 ORACLE과 구별되지 않습니다.
2. 손실은 **first gather**에 있습니다. 즉 once-scattered radiance L₁의 방향 사영입니다. order 2 이상의 사영과 그 누적은 R_mirror에 거의 기여하지 않습니다.
3. 공간 patch 평균은 이 workload의 rendered sharp 오차에 기여하지 않습니다. field 수준의 panel order 2에는 작은 영향이 있지만 readout에는 나타나지 않습니다.
4. 오차 지도(fig1)에서 M₀와 angular-only는 R_mirror 영역 안의 같은 위치에 오차를 보입니다. spatial-only와 M₁은 ORACLE과 같은 모양입니다.
5. first gather를 exact로 바꾸는 효과는 sender-side relation(C3UR)에서만 E_H 개선으로 나타납니다.
   - R0에서는 편향은 줄지만 E_H는 그대로입니다. sharp한 exact radiance를 수신측 512 방향으로 읽으면 분산이 커지기 때문입니다.
   - C3UR은 각 ray가 자기 원점·방향의 radiance를 실어 나르므로 이 분산이 없습니다.

### Interpretation

- **C. 분리된 오차 성분.**
  - 측정으로 분리된 것: relation sampling(pooled로 제거), 수신측 quadrature / readout(EE = ORACLE), 방향 사영, 공간 사영, order별 위치, 누적.
- **D. 남은 confound와 한계.**
  - ladder는 single-read 사영입니다. 세분한 table을 recurrence에 넣었을 때의 누적은 측정하지 않았습니다(누적 자체가 작다는 것은 현 해상도에서 측정).
  - ORACLE과 EE의 E_H floor(약 0.014)에는 두 table의 추정 noise가 섞여 있습니다. W2 T1에서 signed는 +0.009와 −0.007로 갈리지만 E_H는 같습니다.
  - workload는 단일 family입니다: S1 slab, 점광원, α ≈ 0.05 재질.
  - sphere sender의 C3UR frame 근사는 분리하지 않았습니다.
- **E. 기존 표현의 충분성.**
  - 출력 table(patch × 64 bin)은 충분합니다. ORACLE이 sharp F2를 통과합니다(0.014).
  - 한계는 **중간 표현**에 있습니다. 구체적으로 "once-scattered sender radiance를 64 bin table에 담아 first gather에 넘기는 것" 하나입니다.
  - order ≥ 2의 중간 table은 이 workload에서 한계가 아닙니다.
- **F. 경제적 타당성.**
  - gap을 줄이는 데 table 세분(4–16× 상태, 투영)이 필요하지 않습니다.
  - first gather를 relation ray별 exact 평가로 바꾸면 deployed C3UR에서 R_mirror가 0.040에서 0.024로 내려갑니다(수렴 table floor 0.036보다 낮고, ORACLE은 0.014).
  - 측정 비용은 오히려 감소합니다(per-source 0.56×, per-edit 0.82×). q table(26 ms)이 ray별 평가(10 ms)로 대체되기 때문입니다.
  - 남은 0.024 − 0.014는 deployed relation의 sampling 몫입니다(C3URX1 SD 0.005).
- **추론 표시.**
  - "R0에서 X1이 E_H를 못 줄이는 이유는 분산"이라는 설명은 signed와 SD 측정에 기댄 추론입니다.
  - 다른 재질 sharpness나 area light에서의 거동은 측정하지 않았습니다.

### Unresolved Question

- sender가 더 sharp하거나 덜 sharp한 재질일 때, 또는 광원이 area light일 때 "first gather 방향 사영이 지배"가 유지되는지. area light에서는 ray별 exact L₁이 결정론적이지 않습니다.
- 두 번 이상의 sharp bounce가 지배하는 workload에서 order ≥ 2 사영이 커지는지. 이번 C_ss(S1 → S1)는 gap이 0.001로 작았습니다.
- C3URX1의 잔여(0.024 vs 0.014)가 relation ray 수로 어떻게 줄어드는지. 측정하지 않았습니다.
- C3URX1은 16개 realization과 2개 configuration에서만 평가했습니다. held-out source / edit(W2, LMIX, LSPOT)에서는 평가하지 않았습니다.

### Architecture Verdict

범위: W1 = V3 (T0 / T1), held-out edit W2 T1, frozen ER, 점광원, 512², patch × 64-bin table.

| 항목 | 판정 |
|---|---|
| COMMON GAP VERIFIED | **SUPPORTED** (G₀ 0.0226, noise 0.0001) |
| TRANSPORT FAMILY COMPARABILITY | **SUPPORTED** (EE = ORACLE: E_H 차이 −0.0002; order별 flux 0.999–1.001) |
| INTERMEDIATE PROJECTION ISOLATED | **ISOLATED** (gap의 97–102 %, 세 configuration, 네 region) |
| DOMINANT FACTOR | **ANGULAR** (A ≈ J, S ≤ 0; 세 configuration) |
| FIRST DIVERGENCE | **first gather (L₁의 사영)**: m* = 1, first-order share 0.96–1.01, 누적 ≤ 9 % |
| COUNTERFACTUAL (고정 grid) | angular 4 sub-bin에서 J의 9–20 % 잔여. spatial 세분 효과 없음 |
| PRIMARY HYPOTHESIS | **SUPPORTED** (사전 규칙) |
| OUTPUT TABLE SUFFICIENCY | **SUPPORTED** (ORACLE sharp F2 통과) |
| INTERMEDIATE TABLE SUFFICIENCY | first gather의 L₁ 방향 사영: **NOT SUFFICIENT** (binding). order ≥ 2: 측정 범위에서 sufficient |
| FEASIBILITY (X1 probe) | MATERIAL **true** (C3URX1 − C3UR −0.016 [−0.020, −0.012]). ECONOMICALLY PLAUSIBLE **true** (per-edit 0.82×, per-source 0.56×). R0X1은 E_H 개선 없음 |

**Completion answer.**
- 공통 sharp fidelity floor는 측정 범위에서 **중간 방향(angular) 정보 손실**이 일차 원인입니다.
- 그것도 first gather에 넘기는 once-scattered sender radiance의 bin 사영 하나에 집중되어 있습니다. 공간 사영, 고차 order 사영, 누적, relation 표본, 수신측 근사는 원인이 아닙니다.
- 표현을 유지할지 재검토할지에 관한 증거는 다음과 같습니다.
  - 출력 table과 order ≥ 2의 중간 table은 충분합니다.
  - first-gather 입력만이 한계입니다.
  - 그 한계는 table 세분 없이, relation ray별 exact 평가로 비용 증가 없이 대부분 제거됩니다(진단 probe).
- 어떤 표현 변경이나 아키텍처도 채택하지 않았습니다. WL89의 H1 판정은 그대로입니다.

### Reproducibility

- **Commits.** freeze `51b93b8`; results in the closing commit. Branch `Selective_Recompute`. Parents: WL89 `7515a29` / `5d6bb5c` / `0d646a9`.
- **Environment.** LabServer63, RTX 3080 Ti 12 GB, driver 595.71.05, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Protocol.** `experiments/gm1_gap90/protocol/gm1_gap90.json` (v1, no amendment).
- **Seeds.**
  - Pooled realisations 90301–90364 (quadrature 90001 + 1000 · i, reads 90003 + 7919 · i + 101 · variant).
  - W2 T1 order field 90460.
  - Renders 89600 / 89601 (the WL89 render seeds).
  - Feasibility: the WL89 deployed draw and realisations 89201–89216 with the WL89 quadrature / sender seeds.
  - Sources 90720; tests 909xx.
- **Reused from WL89 (`~/rest_rel89`).** W1 choice (sha256 213a3ccc…), REF-ER, masks, oracle tables F89, order fields ORD_W1.
- **Server.** `~/rest_gap90` (tables, renders, logs, GPU trace).
- **Local.**
  - `results/gm1_gap90/`: chain status, GPU trace, tests90, final90, econ90, table metadata, split-half table noise, logs.
  - `results/evaluation/90/`: five figures plus a manifest with sha256.
- **Invalid or excluded.** None in the evidence run.
- **Limits of the noise estimate.** The arm-to-arm noise quoted is the render-seed noise. The pooled-table estimation noise is given separately (split halves, 0.7–1.0 % field level).
