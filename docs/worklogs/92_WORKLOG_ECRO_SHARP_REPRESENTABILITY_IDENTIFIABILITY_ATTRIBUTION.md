# Worklog — ECRO Sharp Interreflection: Representability, Identifiability and Learning-Failure Attribution (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — ECRO Learned Global Transport: Sharp Interreflection Representability, Identifiability and Learning Failure Attribution" (user, 2026-11-13). One bounded hypothesis-discrimination batch.
- **Context.** The human decision of 2026-11-13 restores ECRO as the central learned-global-transport development base. WL86's frozen negative verdict stands.
- **Question.** Does the existing ECRO operator have the representational and informational capacity to reproduce sharp directional interreflection, and, if so, does physically identifiable supervision improve learned generalisation without changing the architecture?
- **Primary hypothesis (prospective).** The sharp-response failure is caused at least partly by non-identifiable learned component responsibilities, not solely by an insufficient operator representation. Alternatives: H_rep, H_info, H_learn, H_id, H_interact.
- **Preserved.** WL84 ER; the WL86 code, checkpoints, data, references, masks, thresholds and verdicts; WL86A and WL87–WL91 (imported unchanged). No in-family arm changes the architecture, the ranks (R 16, global 24), the 1,270 × 64 table, the 512 state rays or the 16-point source encoding.
- **Not done.** No second operator family, no capacity change, no explicit residual, no GM2, no adoption.
- **Where.** Branch `Selective_Recompute`, LabServer63. `experiments/ecro_attr92/`, `results/ecro_attr92/`, `results/evaluation/92/`. Server `~/rest_x92`, reading `~/rest_ecro86`.
- **Terminology.** Explicit transport and path tracing are physical comparators and diagnostics. No external neural baseline is run; no Neural Full Recompute claim is made.

## ECRO v0 as recovered from the implementation

Source: `experiments/ecro86/model86.py`, `train86.py`, `gen86.py`, `pt86.py` (read, not reconstructed from worklogs).

**Operator.**
- `u = c·s`, `Bq = a·u`, `E1` = flight of `Bq` through the 512 state rays with weight cos dΩ / κ, `B = a·(c·E1) + a·(P·Σ_j area_j Q_j·u_j)`.
- `c`, `a` are per patch, per colour, 64 bins × 16 channels, non-negative. `Q`, `P` are per patch, rank 24, non-negative.
- The local matrix `M_n = a_n c_nᵀ` (incident bin → outgoing bin, non-negative rank ≤ 16) is used with **tied factors** in three places: source → once-scattered radiance `Bq`; first-flight irradiance `E1` → outgoing radiance of the single-flight term; encoder and decoder of the global term.

**Physical correspondence of the terms.**

| Model quantity | Path family it can represent | Path-traced counterpart |
|---|---|---|
| `Bq` | once-scattered outgoing radiance | L₁ (the table q) |
| single-flight term `a·(c·E1)` | exactly 2 scattering events, all senders | L₂ |
| global term | 3–5 scattering events | L₃ + L₄ + L₅ |
| `B` | the WL86 target | L₂ + … + L₅ |

- The mapping is exclusive: every path has exactly one event count, so the two additive terms have disjoint physical targets.
- The implementation does not enforce it. `train86.py` supervises only the sum `B` (`model86.loss_fn`). Nothing prevents the global term from carrying two-event light or the single-flight term from carrying less than L₂.

**Training contract.** 1,024 geometries × 4 point lights, 256-spp path-traced field targets; 24,000 steps of Adam, one geometry per step; loss = squared error relative to the target's patch mean over bins plus 5 % of the field mean; seeds 8601 / 8602.

## Design of this batch (`experiments/ecro_attr92/protocol/ecro_attr92.json`)

**Stages.**
- **S0 Baseline and provenance.** Hashes of checkpoints, data, references. The frozen checkpoints must reproduce the tables stored with the WL86 renders, and the re-rendered tables the WL86 E_H values.
- **S1 Truth by family.** Path-traced tables of exactly-2-event radiance (last sender on the panel / elsewhere) and 3–5-event radiance (WL86A tracer, unchanged), two halves of 2,048 spp. Reader weights W_R (Σ_pixels hl = Σ W_R·B).
- **S2 Sharp deficit.** Truth contribution, each learned term's contribution, residual, compensation sign, on E2 T1, E2 T0, E1 T1.
- **S3 Ladder of the 2-event term.** Tables = (variant of the 2-event term) + (path-traced L₃₊). Only the 2-event term changes.

| Variant | Meaning | What it isolates |
|---|---|---|
| ORA2 | path-traced L₂ | must equal the oracle table |
| RS | exact once-scattered radiance per deployed ray | the deployed rays without any bin projection (non-deployable for ECRO) |
| EXP | exact 64-bin q through the deployed rays | sender bin projection |
| ENCQ | q from the deployed encoding through M64 | source encoding |
| ENCB | the ECRO single-flight form with exact full-rank local matrices (R = 64) | everything the deployed interface allows, without the rank limit (non-deployable capacity control) |
| R16 | the same form with rank-16 non-negative factors of M64 | an in-family point with exclusive ownership |
| LRN | the frozen generators' single-flight term | what was learned |

  - Pooled variants (16 state realisations) are a non-deployable adequate-sampling control.
- **Local operator.** M64 = bin-averaged frozen-ER response per material. Singular spectrum, Eckart–Young rank-16 bound, rank-16 NMF, and the learned `a cᵀ` of the frozen generators, all in the flux-transfer norm.
- **S4 Fixed-geometry free-factor fits** (non-deployable). All four factor tables free on one geometry; 192 fit lights, 32 validation lights; objectives SUM (the WL86 objective) and OWN (SUM + component ownership); initialisations: frozen generators, untrained generators, the rank-16 construction; 8,000 steps; a convergence rule.
- **S5 Identifiability.** Across the SUM fits and the frozen generators: same objective value with different allocations?
- **Gate, then at most one intervention.**

**Gate (level = the frozen WL86 rule F2: E_H ≤ 0.06 in region C).**
- G_id: component targets valid (tests).
- G_info: ENCB + L₃₊ meets the level in both failing cases (E2 T1, E2 T0), deployed state draw.
- G_rep: the better of R16 + L₃₊ and the single-flight term of the converged OWN-k16 fit + L₃₊ meets the level in both failing cases.
- The intervention runs only if all three hold. Otherwise the intervention path stops; S0–S5 are the evidence.

**Conditional intervention.**
- Control: the WL86 contract (frozen checkpoints; a fresh rerun for the E4 comparison).
- Intervention: objective OWN with the path-traced 2-event target regenerated from the WL86 generator's own seeds and samples; f₃ = f − f₂. Same architecture, data, steps, optimiser, seeds.
- Evaluation: WL86 gates F1–F4 and the component attribution on E1 / E2 (inspected), a new condition E4 and held-out sources.
- E4: the WL87 candidates E3b / E3c, defined on 2026-11-11 and never rendered, chosen by a geometry-only rule that requires a sharp-reader class; admitted by the unchanged WL86 stage-0 criteria.

**Interpretation rules (prospective).**
- *Failure to fit*: a converged fit stays above a constructed in-family point.
- *Failure to optimise*: a converged fit ends above the objective value of a constructed in-family point (a better point exists and was not found).
- *Failure to generalise*: fixed-geometry factors reach a level the generator does not.
- *Information limit*: ENCB + L₃₊ misses the level although ORA2 meets it.
- A failed optimiser is not treated as a proof of insufficient capacity: the Eckart–Young bound and the constructed points carry the capacity statements.

## Prior evidence used as constraints (not re-measured unless stated)

- WL86: oracle table E_H 0.014 in E2 T1 region C; frozen ECRO 0.063 / 0.072.
- WL86A: the single flight from panel senders delivers 1–18 % of its family; the global term compensates; free-factor fit indeterminate (64 lights, over-fitting).
- WL87: explicit transport on ECRO's own state and encoding: EXP 0.049, ENC 0.054, RAYSEND 0.059, sender-side C3 0.024 in E2 T1 region C.
- WL90 / WL91: the first-gather angular projection of the once-scattered radiance is the explicit floor on the moving-sharp workload; exact first gather removes it in expectation.
- These bound what any operator on the deployed interface can reach. They do not by themselves explain the learned failure.

## Pre-freeze development (non-evidence)

- **Design refinement from the local-operator dev run.** The plain Frobenius norm of M64 is dominated by the grazing ring (f ∝ 1 / (4 cos cos)): a rank-16 fit in that norm spends every channel on 16 grazing incident bins. The analysis and the rank-16 construction therefore use the flux-transfer matrix X[bi, bo] = M64 z_bi z_bo, whose rows sum to the directional albedo. This concerns frozen-ER physics only and was fixed before the freeze.
- **Dev chain** (`~/rest_x92dev`; 64 spp, 6 fit lights, 1,000 fit steps, 48 training geometries, 300 training steps; the intervention branch forced so that every command runs): completed end to end, every stage rc = 0, tests pass. All dev numbers are noise-dominated and are not interpreted. The dev local-operator stage (deterministic frozen-ER quantities, re-measured in the evidence run) was seen.
- **Full-scale probes** (`probe92.py`, LabServer63, alone):
  - Component-target regeneration: the WL86A split tracer with the WL86 generator seeds reproduces `pt86.Ctx.field` to 1e-7 and the stored float16 training target to 1.7e-4 (two lightings of `train/g00000`). The planned T_F2_SUM identity test (≤ 1e-3) is therefore attainable; the dev failure came from the reduced dev sample count only.
  - Fixed-geometry fit at the full batch (16 lights, all four tables free, OWN objective): 19.7 ms per step, 680 MiB peak; 8,000 steps ≈ 2.6 min per fit, 21 fits in a pool of 4.
  - Field tracer: about 2.9 s per lighting at 256 spp; fit data ≈ 15 min per geometry (three in parallel).
  - Expected chain wall: about 1 h without the intervention, about 2 h with it.

## Evidence run (frozen; no amendment)

- **Freeze commit** `5e672ff`. Deployed with `git archive` (LF); deployed protocol hash = committed (sha256 b558babd…).
- **Chain.** One detached chain on LabServer63, 2026-11-12 21:57:30–23:16:13 UTC (78.7 min). Every stage rc = 0. Mean GPU utilisation 90 %, peak 8.0 GB of 12 GB. No rerun.
- **Outcome by the frozen gate: `STOP_INTERVENTION_G_rep`.** The tests passed and G_id and G_info hold; G_rep does not. The conditional intervention, E4 references and all training were therefore **not run**. E4 (= WL87 candidate E3b) was chosen from geometry only (C_mirror 5,060 px; training exclusion 12 / 1 / 1 files) and has **no radiance**: it remains untouched.

### Baseline reproduction and tests

- Checkpoints `ecro_s8601` / `ecro_s8602` (sha256 9d332cd6… / 4d993121…), training data 1,024 files (listing hash in `prov.json`), WL86 `final.json` (sha256 59b0cece…).
- The frozen checkpoints reproduce the tables stored with the WL86 renders exactly (difference 0). Region-C E_H recomputed and re-rendered: E2 T1 0.0717 / 0.0634, E2 T0 0.0603 / 0.0588, E1 T1 0.0132 / 0.0110 (identical to WL86).

| Test | Result |
|---|---|
| T_REPRO_TABLE | 0 |
| T_SPLIT_SUM | family parts sum to the stored oracle field: flux 1.0002–1.0003; region-C reading −0.9 %, −0.5 %, +0.1 % of h (z −1.5, −0.9, +3.4; all < 4) |
| T_OWNERSHIP_CODE | single-flight + global = B to 2e-7; all parts ≥ 0 |
| T_FORM | ECRO form at R 64 vs the explicit bin-quadrature form: 0.018–0.021 |
| T_LINEAR | frozen, R16 and R64 factor sets: superposition / scaling ≤ 7e-7, zero → 0, outputs ≥ 0 |
| T_STATE | deterministic; selective = full = fresh scene; null edit identical |
| T_DISJOINT | fit / validation lights ≥ 0.308 m from every test light; E4 chosen without radiance |

### Physical attribution (frozen generators; region C read through the near-mirror panel)

Shares of the path-traced higher-order reading h of the region (seeds 8601 / 8602):

| Case | Truth: 2 events via panel / 2 events other / 3–5 events | Single − L₂ | Global − L₃₊ | Total | Panel-sender family delivered |
|---|---|---|---|---|---|
| E2 T1, C | 0.18 / 0.36 / 0.46 | −0.168 / −0.175 | +0.076 / +0.096 | −0.092 / −0.079 | 14 % / 18 % |
| E2 T0, C | 0.12 / 0.41 / 0.47 | −0.127 / −0.151 | +0.038 / +0.067 | −0.089 / −0.084 | 1.6 % / 3.0 % |
| E1 T1, panel | 0.01 / 0.37 / 0.62 | −0.009 / −0.014 | +0.050 / +0.047 | +0.041 / +0.033 | 58 % / 31 % |
| E2 T1, D | 0.01 / 0.51 / 0.48 | −0.016 / −0.022 | +0.005 / +0.017 | −0.011 / −0.004 | 47 % / 54 % |

- Field level (all patches): single-flight vs L₂ relative L1 0.16–0.17; global vs L₃₊ 0.16–0.19.
- Learned once-scattered table `Bq` against the exact q table, panel patches: relative L1 0.90–1.11, flux 0.16–0.72 (floor 0.24–0.27 / 0.86–0.92; Lambert 0.04–0.24).

### Local operator (flux-transfer norm; frozen ER)

| Material | Eckart–Young rank-16 residual (R / G / B) | Rank for 99 % energy | Rank-16 NMF: residual / mirror-bin retention | Frozen generators' `a cᵀ`: residual / mirror-bin retention / mass |
|---|---|---|---|---|
| floor B | 0.03 / 0.03 / 0.04 | 8 | 0.03–0.04 / 0.96–0.97 | 0.33–0.34 / 0.80–0.84 / 1.08 |
| panel S1 | 0.52 / 0.46 / 0.46 | 46–59 | 0.49–0.54 / 0.47–0.54 | **0.88–0.89 / 0.11–0.14 / 0.68–0.76** |
| sphere S2 | 0.20 / 0.34 / 0.23 | 31–42 | 0.22–0.36 / 0.60–0.71 | 0.63–0.64 / 0.32–0.34 / 1.15–1.17 |
| Lambert | rank 1 (exact) | 1 | exact | 0.13–0.32 / 1.0–1.1 |

### Ladder of the 2-event term (+ path-traced L₃₊), region C E_H, 2 × 4,096 spp (rule 0.06; noise bound 0.013 / 0.010)

| Variant | E2 T1 | E2 T0 | E1 T1 |
|---|---|---|---|
| ORA2 (path-traced L₂) | 0.013 | 0.010 | 0.008 |
| RS (exact radiance per deployed ray) | 0.048 | 0.044 | 0.010 |
| EXP (exact q, 64 bins, deployed rays) | 0.041 | 0.048 | 0.010 |
| ENCQ (q from the deployed encoding) | 0.045 | 0.053 | 0.010 |
| **ENCB (ECRO form, exact local matrix, R 64)** | **0.045** | **0.052** | 0.010 |
| **R16 (ECRO form, material rank-16 NMF)** | **0.084** | **0.073** | 0.010 |
| pooled 16 realisations: EXP / ENCB / R16 | 0.027 / 0.027 / 0.064 | 0.022 / 0.023 / 0.056 | 0.008 / 0.009 / 0.008 |
| explicit full recurrence (WL87 EXP) | 0.049 | 0.055 | 0.011 |
| frozen ECRO B (8601 / 8602) | 0.072 / 0.063 | 0.060 / 0.059 | 0.013 / 0.011 |
| learned single-flight + true L₃₊ | 0.124 / 0.128 | 0.083 / 0.098 | 0.017 / 0.021 |
| true L₂ + learned global | 0.052 / 0.064 | 0.028 / 0.041 | 0.020 / 0.020 |

- The signed reading of every variant is negative in E2 (ENCB −0.054 / −0.076 of h; pooled −0.009 / −0.023): the deployed-draw shortfall is the WL86A sampling limit, about half of ENCB's error.

### Fixed-geometry free-factor fits (non-deployable; 192 fit / 32 validation lights; 8,000 steps)

Region-C E_H (rendered): single-flight term + true L₃₊ / whole table. Convergence = relative change of the objective over the last 500 steps (rule ≤ 2 %).

| Fit | E2 T1 | conv. | E2 T0 | conv. | E1 T1 |
|---|---|---|---|---|---|
| OWN-f8601 | 0.029 / 0.027 | yes (1.0 %) | 0.020 / 0.024 | no (2.2 %) | 0.011 / 0.023 |
| **OWN-k16** (gate) | 0.036 / 0.036 | **no (6.1 %)** | 0.017 / 0.013 | **no (11.6 %)** | 0.011 / 0.011 |
| OWN-r1 | 0.041 / 0.044 | yes (1.9 %) | 0.024 / 0.021 | yes (1.7 %) | 0.013 / 0.018 |
| SUM-f8601 | 0.102 / 0.043 | yes | 0.095 / 0.033 | yes | 0.022 / 0.014 |
| SUM-f8602 | 0.111 / 0.033 | yes | 0.102 / 0.024 | yes | 0.023 / 0.023 |
| SUM-r1 | 0.074 / 0.027 | no | 0.095 / 0.018 | no | 0.106 / 0.031 |
| SUM-r2 | 0.078 / 0.024 | no | 0.098 / 0.016 | no | 0.105 / 0.031 |

Component behaviour (fit lights / validation lights):

| | SUM, frozen init | SUM, untrained init | OWN (all inits) |
|---|---|---|---|
| single-flight flux / L₂ flux | 0.82–0.87 | 1.36–1.37 | 0.996–1.002 |
| validation single vs L₂, relative L1 | 0.24–0.27 | 0.38–0.49 | 0.10–0.14 |
| validation sum, relative L1 | 0.069–0.094 | 0.083–0.108 | 0.076–0.100 |
| panel family delivered, E2 T1 C | 31–40 % | 77–81 % | 57–74 % |
| panel family delivered, E2 T0 C | 1.8–2.5 % | 51–52 % | 17–44 % |

- Validation SUM objective (frozen generator → fits): E2 T1 0.0157 → SUM 0.014–0.024, OWN 0.019–0.046; E2 T0 0.037 → 0.052–0.065 / 0.060–0.072; E1 T1 0.059 → 0.065–0.085 / 0.071–0.075. On fit lights the fits reach 0.016–0.033. On E2 T0 and E1 T1 the free tables generalise over sources worse than the frozen generator (fixed-geometry over-fitting); for the workload light they are better.
- Contracts of every fitted factor set: superposition 0, zero 0, outputs ≥ 0.

### Gate and verdict mapping (frozen rules)

| Gate | Value | Result |
|---|---|---|
| G_id (T_SPLIT_SUM, T_OWNERSHIP_CODE, T_FORM) | pass | **holds** |
| G_info: ENCB + L₃₊ ≤ 0.06 in both failing cases | 0.045 / 0.052 | **holds** |
| G_rep: better of R16 and the converged OWN-k16 fit ≤ 0.06 | R16 0.084 / 0.073; OWN-k16 not converged | **fails** |

- Identifiability rule (SUM fits, same objective within 10 % and allocation ≥ 2×): allocation ratio 2.4–28×, objective spread 1.26–1.67× → the non-identifiability criterion is **not met** (code label "IDENTIFIABLE BY THE RULE"). The objective spread comes mainly from the untrained-init SUM fits, which are not converged.
- Verdicts by `analyze92.py final`: A INCONCLUSIVE; B SUPPORTED; C INCONCLUSIVE (frozen single + L₃₊ over the best gate point R16: 1.14–1.52×, mixed around 1.25); D: component targets valid, SUM criterion not met; E and F NOT TESTED; completion code 1.
- **Protocol wording defect (disclosed, no re-mapping).** Completion code 1 is defined as "representation / interface insufficient (G_info or G_rep fails)". Here G_rep fails because the in-family point was **not established** (material rank-16 construction insufficient; the geometry-specific gate fit unconverged), while verdict A is INCONCLUSIVE by the same protocol. Code 1 must be read as "G_rep not established", not as a demonstrated representation limit.

## 최종 보고

### Implementation Fact

- **ECRO v0 구조를 코드에서 복원했습니다** (`model86.py`, `train86.py`).
  - 국소 행렬 `M_n = a_n c_nᵀ`(patch·색별 64 → 64 bin, 비음수 rank ≤ 16)가 source → `Bq`, 첫 flight → single-flight 항, global 항의 encoder / decoder에 **tied factor**로 세 번 쓰입니다.
  - single-flight 항 = 정확히 2-event(L₂), global 항 = 3–5-event(L₃₊)로 물리 경로가 **배타적**으로 대응합니다.
  - 학습은 합 B만 감독합니다. component 소유는 강제되지 않습니다.
- **새 진단 코드**(`experiments/ecro_attr92/`): event family 분리 truth, reader-weighted 분해, 2-event 항 ladder(ECRO 형식 R 64 / R 16 포함), flux-transfer norm 국소 연산자 분석, 고정 기하 free-factor fit(SUM / OWN), 식별성 규칙, 게이트, 조건부 intervention 경로.
- **freeze** `5e672ff`, amendment 없음. ECRO 구조·rank·bin·state ray·source encoding·ER·threshold는 바꾸지 않았습니다.
- **게이트 결과 `STOP_INTERVENTION_G_rep`**. intervention 학습, E4 reference, 재학습은 실행하지 않았습니다. E4(WL87 후보 E3b)는 기하 규칙으로만 선택했고 radiance가 없어 미사용 held-out으로 남아 있습니다.

### Measurement

**기준 재현.** frozen checkpoint가 WL86 table을 정확히 재현합니다(차이 0). region C E_H: E2 T1 0.072 / 0.063, E2 T0 0.060 / 0.059, E1 T1 0.013 / 0.011. 테스트 전부 통과.

**성분 귀속 (E2 T1, region C; h 대비 비율).**
- truth: panel 경유 2-event 0.18, 기타 2-event 0.36, 3–5-event 0.46.
- 학습된 single-flight 항 −0.17, global 항 +0.08 / +0.10. panel-sender family 전달 14 % / 18 % (E2 T0 1.6 % / 3.0 %).
- 학습된 panel 국소 행렬 `a cᵀ`: mirror bin flux 유지 11–14 %, 질량 0.68–0.76. 학습된 `Bq`의 panel flux는 정확한 q의 0.16–0.72배.

**국소 연산자 (frozen ER, flux-transfer norm).**
- floor B: rank 16으로 잔차 0.03–0.04.
- panel S1: Eckart–Young rank-16 잔차 0.46–0.52, 99 % 에너지에 rank 46–59 필요. 재질 단위 rank-16 NMF는 mirror bin flux의 47–54 %만 유지.

**2-event 항 ladder (region C E_H, 규칙 0.06).**

| 변형 | E2 T1 | E2 T0 |
|---|---|---|
| path-traced L₂ | 0.013 | 0.010 |
| ENCB (ECRO 형식, 정확한 국소 행렬 R 64, 배포 interface) | 0.045 | 0.052 |
| R16 (ECRO 형식, 재질 rank-16 NMF) | 0.084 | 0.073 |
| ENCB, 16 realization pooled | 0.027 | 0.023 |
| frozen ECRO | 0.072 / 0.063 | 0.060 / 0.059 |
| 학습 single-flight + 참 L₃₊ | 0.124 / 0.128 | 0.083 / 0.098 |

**고정 기하 free-factor fit (비배포 oracle; region C E_H, single-flight + 참 L₃₊).**
- OWN 목적(component 소유 감독): E2 T1 0.029–0.041, E2 T0 0.017–0.024. 수렴한 것은 OWN-f8601(E2 T1), OWN-r1(E2 T1, E2 T0).
- 게이트 지정 fit OWN-k16: 0.036 / 0.017이지만 수렴 규칙 미충족(6.1 % / 11.6 %).
- SUM 목적(WL86 목적): 표 전체 E_H 0.016–0.043(기준 이하), 그러나 single-flight + 참 L₃₊는 0.074–0.111.
- component 배분: SUM은 초기값에 따라 single-flight flux / L₂가 0.82–0.87(frozen init) 또는 1.36–1.37(무학습 init). OWN은 모든 init에서 0.996–1.002.
- 고정 기하 fit은 workload 광원에서는 generator보다 좋지만, 검증 광원 평균 SUM 목적은 E2 T0 / E1 T1에서 generator보다 나쁩니다(0.052–0.085 vs 0.037 / 0.059).

### Observation

1. 배포 interface(512 state ray, 64-bin source encoding)를 그대로 두고 정확한 국소 행렬을 쓰면 실패 사례에서 규칙을 통과합니다(0.045 / 0.052). 정보 손실은 기준 이하로 남고, 그 오차의 약 절반은 배포 state draw의 표본 한계입니다(pooled 0.027 / 0.023).
2. 같은 ECRO 형식에서 **재질 단위 rank-16 bin factor**는 sharp panel의 mirror lobe를 잃어 규칙을 넘습니다(0.084 / 0.073).
3. 그러나 **patch별·기하별 rank-16 factor**를 고정 기하에서 맞추면 규칙 아래로 내려갑니다(OWN 0.017–0.041, SUM 표 0.016–0.043). 그중 수렴 규칙을 충족한 점도 있습니다(OWN-r1 0.041 / 0.024).
4. 학습된 generator의 panel 국소 행렬은 mirror lobe 대부분을 잃었고(유지 11–14 %), panel-sender family를 1.6–18 %만 전달합니다. global 항이 양의 방향으로 보상합니다.
5. SUM 목적에서는 component 배분이 초기값에 따라 크게 달라지지만, 사전 식별성 규칙(목적값 10 % 이내 + 배분 2배 이상)은 목적값 차이(1.26–1.67배, 주로 미수렴 fit)로 충족되지 않았습니다.

### Interpretation

- **정보(B).** 배포 interface 정보만으로 sharp 수신자의 규칙 수준에 도달할 수 있습니다. 실패의 주원인은 interface가 아닙니다. 다만 배포 state draw의 표본 한계가 여유를 절반가량 소비합니다.
- **표현(A).** 사전 게이트 기준으로는 확립되지 않았습니다. 재질 단위 rank-16 분해는 부족하지만, 고정 기하의 patch별 factor는 충분해 보입니다(사후 서술 관찰이며 게이트 판정에는 쓰지 않음). 즉 rank-16 한계가 문제인지는 "factor를 기하별로 얼마나 정확히 생성할 수 있는가"와 분리되지 않습니다.
- **학습(C).** 같은 연산자 계열을 기하별로 맞추면 generator보다 sharp 응답이 좋다는 점은, 실패가 고정 기하 표현보다 generator의 기하 조건부 예측(일반화)과 학습 쪽에 있음을 시사합니다. 사전 규칙상 판정은 INCONCLUSIVE입니다.
- **식별성(D).** component 목표는 물리적으로 유효하고 ECRO factorization에 직접 대응하며, OWN 감독은 고정 기하에서 소유를 회복합니다. SUM 계약이 비식별적이라는 사전 기준은 충족되지 않았습니다. 초기값 의존 배분은 서술적 관찰로만 남깁니다.
- 이 해석들은 하나의 원인으로 수렴하지 않습니다: generator의 sharp 국소 응답 붕괴, 배포 state 표본 한계, SUM 계약의 배분 자유도가 함께 관찰됩니다.

### Unresolved Question

- OWN-k16 fit이 수렴했을 때도 규칙 아래에 머무는지(미수렴 값 0.036 / 0.017).
- 학습된 generator가 component 소유 감독으로 panel 국소 응답과 panel-sender family를 회복하는지: intervention은 게이트에 의해 실행되지 않았습니다.
- 고정 기하 fit의 source 일반화 손실(E2 T0 / E1 T1 검증 목적 악화)이 과적합인지, factor 자유도 자체의 문제인지.
- 배포 state draw의 표본 한계를 어떻게 다룰지(512 ray 예산은 고정).
- E4(held-out)의 sharp fidelity와 edit 응답은 측정하지 않았습니다.

### Architecture Verdict

범위: WL86 방 family, rigid box edit, frozen ER 세 재질, 점광원, 512², patch × 64-bin table.

| 항목 | 판정 | 근거 |
|---|---|---|
| A. Representation capacity | **INCONCLUSIVE** | G_rep 미확립: 재질 rank-16 구성 0.084 / 0.073 실패, 게이트 fit 미수렴. 고정 기하 patch별 fit은 서술적으로 규칙 이하 |
| B. Input information sufficiency | **SUPPORTED** | ENCB 0.045 / 0.052 ≤ 0.06 (pooled 0.027 / 0.023) |
| C. Learning / optimisation adequacy | **INCONCLUSIVE** (사전 규칙) | frozen single-flight 대비 게이트 점 비율 1.14–1.52배로 1.25 경계 혼재. 서술적으로 모든 고정 기하 fit이 generator보다 sharp 응답이 좋음 |
| D. Physical component identifiability | component 목표 **유효**, OWN으로 고정 기하에서 소유 회복; SUM 계약의 비식별성은 **INCONCLUSIVE**(사전 기준 미충족) | 배분 비율 2.4–28배, 목적값 차이 1.26–1.67배 |
| E. Held-out sharp fidelity / edit response | **NOT TESTED** | intervention 미실행(게이트) |
| F. Runtime / training economics | **NOT TESTED** | runtime 경로 변경 없음; 이번 batch 진단 chain 78.7분(GPU 평균 90 %) |

**Completion answer.**
- 기존 ECRO 연산자 계열은 배포 interface의 정보만으로 sharp 방향 상호반사를 규칙 수준까지 재현할 정보 용량을 가집니다(B SUPPORTED).
- 표현 용량은 사전 게이트로 확립되지 않았습니다. 재질 단위 rank-16은 부족하고, 기하별 factor는 고정 기하에서 충분해 보이지만 게이트 fit이 수렴 규칙을 충족하지 못했습니다.
- 따라서 물리적으로 식별 가능한 감독이 일반화를 개선하는지는 **이 batch에서 검정하지 않았습니다**.
- 사전 completion 코드는 1(G_rep 실패)입니다. 이는 "표현 한계가 입증됨"이 아니라 "표현 적합성이 확립되지 않음"으로 읽어야 합니다(프로토콜 문구 결함, 공개).
- 다음 결정이 정당하게 바꿀 수 있는 항목: 사전 규칙으로 확립된 것은 없습니다(**none without further evidence**). 서술적 증거는 학습 / 감독 계약과 generator의 기하 조건부 국소 응답 쪽을 가리키며, state 표본 한계가 부차 요인입니다. 연산자 표현 계열 교체를 요구하는 증거는 없습니다.
- ECRO v0의 WL86 판정은 그대로입니다. 어떤 아키텍처 전환도 승인되지 않았습니다.

### Reproducibility

- **Commits.** freeze `5e672ff`; results in the closing commit. Branch `Selective_Recompute`. Parents: WL86 `7786d09` / `63972c8` / `b97027d`, WL86A `46593f0` / `1bd7df1` / `f6483ab`.
- **Environment.** LabServer63, RTX 3080 Ti 12 GB, driver 595.71.05, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Protocol.** `experiments/ecro_attr92/protocol/ecro_attr92.json` (v1, sha256 b558babd…, no amendment).
- **Frozen inputs.** `~/rest_ecro86` (checkpoints, data, REF-ER, masks, oracle fields); hashes in `results/ecro_attr92/results/prov.json`.
- **Seeds.** Truth split 92400 / 92405 (2 × 2,048 spp); importance 86600 (1,024 spp); NMF 92100; pooled realisations 86001, 92201–92215; fit lights 92990 (fit 92500+, validation 92700+ / 92800+); fit optimiser torch 0, batch 1; untrained inits 92001 / 92002; renders 86600 / 86601 (4,096 spp, common random numbers with WL86).
- **Server.** `~/rest_x92` (fields, tables, fits, renders, logs, GPU trace).
- **Local.** `results/ecro_attr92/` (chain status, GPU trace, final92 / core92 / gate / tests92 / prov / local / e4_choice, table and fit metadata, render lists, logs). Visuals: `results/evaluation/92/` (eight figures, manifest with sha256).
- **Invalid or excluded.** None. Non-evidence: the dev chain (`~/rest_x92dev`) and probes.
