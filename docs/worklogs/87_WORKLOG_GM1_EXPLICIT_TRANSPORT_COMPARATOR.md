# Worklog — GM1 Learning-free Global Transport Comparator and Architecture Discrimination (2026-11-11)

## Origin and scope

- **Brief.** "WL87 — Learning-free Global Transport Comparator and Architecture Discrimination" (user, 2026-11-11). One bounded implementation and experiment batch.
- **Question.** At the same patch × bin representation, frozen ER scattering, geometry state and camera readout, does the learned global response operator of WL86 (ECRO v0) give a quality-matched advantage over explicitly computed physical transport, and what representation limits remain?
- **Hypotheses.**
  - H0 (explicit sufficiency): an explicit operator on the same state reaches comparable fidelity at practical cost.
  - H1 (learned utility): explicit transport represents the response, but learning saves measurable computation.
  - H2 (representation bottleneck): the state cannot carry sharp transport even with physical coefficients.
- **Frozen.**
  - WL84 ER closures. The WL86 code (`experiments/ecro86/`) is imported unchanged.
  - Checkpoints `ecro_s8601` / `ecro_s8602`, the WL86 representation, references, masks, thresholds and verdicts.
  - The WL86A diagnostics and WL86B study.
- **Not done.** No learned operator is trained or tuned. No H-P, subspace or successor architecture is built. No new branch or worktree.
- **Where.**
  - Branch `Selective_Recompute`; LabServer63.
  - `experiments/gm1_explicit87/`, `results/gm1_explicit87/`, `results/evaluation/87/`.
  - Server `~/rest_gm187`, which reads `~/rest_ecro86` and `~/rest_ecro86a`.
- **Comparator terminology.** All comparators are physical recomputations with the same frozen ER (path tracing, table fill, explicit transport). No external neural baseline is run. No Neural Full Recompute or Native Reuse claim is made.

## Deployed WL86 contract, verified against the code

| Item | Deployed value (file) |
|---|---|
| Transport | b = S_ER(e + C_G b), q = S_ER e, T = S_ER C_G. Reference: next-event estimation at ≤ 6 camera vertices, no Russian roulette (`pt86.Ctx.trace`) |
| Event convention | A next-event contribution at camera vertex k has k scattering events: e1 = q, e2 = T q, h = events 3–6 |
| Table B | `[1270, 64, 3]`: outgoing radiance of **2–5** events per patch and outgoing bin (`pt86.Ctx.field`, probe mode: no next-event estimation at the probe vertex, k = 2…5) |
| Readout | At the second camera vertex: `throughput(x1) × B[patch(x2), bin(x2 → x1)]` (`pt86` mode `ecro`). Camera paths therefore get 3–6 events from the table, and q at x2 (= e2) is not in B |
| Bins | 4 rings uniform in cos θ × 16 azimuth sectors of the patch-local frame (t, b, n) = equal solid angle 2π / 64. `bin_of` and `dir_in_bin` are consistent |
| State | Per patch 512 rays = 8 per incident bin, uniform within the bin. **One draw of offsets `default_rng(86001).random((4, 1, 512))` is shared by all patches**. Each ray records its hit patch (front face only) and the outgoing bin at the hit |
| ER | `eval(wi, wo) = f(wi, wo) cos(wo)`; f is reciprocal (symmetric GGX lobes and s(μ) factors; test T_SYM, max relative difference 1.7e-5). The Lambert walls and boxes are ER states with `lam_0 = albedo`, so f = ρ / π exactly |
| ECRO source encoding | Patch-average direct irradiance per incident bin from 16 fixed points (`pt86.Ctx.source`) |

There is no discrepancy with the WL86 / WL86A worklogs. One clarification: the per-patch state rays are not stratified independently per patch. All patches use the same 512 local directions, and only the origin point differs per ray.

## Comparators (frozen in `protocol/gm1_explicit87.json`)

**Primary: EXP (C1, same-state receiver gather).**

- L₁ = q_d is the once-scattered radiance of the actual lights, projected onto patch × outgoing bin.
  - 16 stratified points per patch, each with a shadow ray and the exact light direction.
  - Lambert patches: ρ / π · E, exact.
  - ER patches: the bin average over outgoing directions uses 1,024 MIS-mixture samples per point (½ ER sampling, ½ uniform hemisphere). It is computed in the point's shading frame and binned in the patch frame.
- L_{k+1}[i, b_o] = Σ_r W_ray[m_i, r, b_o] · L_k[hit_r, dbin_r] over the 512 deployed state rays of patch i.
  - W_ray = (1/ΔΩ) ∫_{b_o} f(v, ω_r) dv · cos ω_r · ΔΩ / 8, at each ray's **exact** local direction ω_r.
  - Estimated with 65,536 MIS-mixture samples per (material, ray direction). Lambert rows are exact.
  - W depends only on the materials and the realisation's directions, not on the geometry. It is built once (initialisation).
- **Curved sphere patches.** The 8 patches of a sphere band are rotations about z, so W has one row set per band (4 extra rows).
  - Each row is evaluated at that ray's origin point: local shading frame, local cosine (0 below the local horizon), and outgoing directions above the local horizon.
  - Each row is normalised by the bin's mean valid solid angle, because the oracle field averages over valid directions. q of sphere patches is normalised the same way.
  - The C2-ENC and C3 controls use a flat patch-frame table for sphere receivers. This approximation is declared.
- B_exp = L₂ + L₃ + L₄ + L₅ = Σ_{j=1}^{4} T_d^j q_d. It is read by the unchanged WL86 LOW + lookup renderer with the WL86 ECRO render seeds (86600 / 86601).
  - Because the seeds are common random numbers, EXP − ECRO differences come from the tables alone.
- No renormalisation, clamping or energy correction.

**Diagnostic controls (same output table, same readout).**

| Control | Changes | Isolates |
|---|---|---|
| C2-BINQ | each ray's W replaced by the mean over the 8 rays of its incident bin (same quadrature samples) | incident-direction information inside the bin (scattering quadrature) |
| C2-ENC | L₁ from ECRO's own 16-point encoding s | source projection at the first event (identical input to ECRO) |
| C2-RAYSEND | first gather reads the exact once-scattered radiance at each ray's hit point | sender projection of the 1-event family (light → panel → receiver) |
| C3 | sender-side deposition: per order 650,240 rays drawn in proportion to emitted power, traced, deposited at the receiver's dense incident cell | receiver-side sampling placement. This is **not** an identical-state comparison: there is a separate relation per order and per source |
| POOL | T_d averaged over 16 independent state realisations (8,192 rays per patch) | convergence diagnostic. Not matched budget and not a candidate |
| TF\<b\> | path-traced 2–5-event table at b samples per slot (`pt86.Ctx.field`), with independent seeds and the full fill cost counted | is the speed-up the table's or the learning's? |
| PT / PT_RR | the frozen-ER path tracer (re-measured), and the same tracer with Russian roulette before vertices ≥ 3 (survival min(0.95, max throughput)) | physical full-path comparators |

## Prospective rules (summary; full text in the protocol)

- **Stage 0 gate.** These tests must all pass, otherwise `STOPPED_INVALID_COMPARISON`:
  - WL86 recomputation and re-render readout identity;
  - ER reciprocity; the MIS estimator against an independent uniform estimator;
  - Lambert normalisation; q against analytic irradiance;
  - torch against a numpy loop; zero / no-visibility, linearity, non-negativity, furnace;
  - state identity under null edit, move-and-return and selective = full;
  - realisation reproducibility; RR expectation; seed disjointness.
- **Stage 1.**
  - Measures: WL86 E_H with the WL86 thresholds (F1 stationary ≤ 0.03 with capture ≤ 0.25; F2 A / C / D ≤ 0.06 at T1; F3 slope 0.75–1.25 and correlation ≥ 0.8 in D; F4 unseen sources ≤ 0.04).
  - Also: field-level and per-order errors against path-traced truth, and the region-C reading of the light → panel → Lambert family (WL86A definitions).
  - Classification: NUMERICALLY_INVALID / ACCURATE_UNDER_FROZEN_REPRESENTATION / DISCRETISATION_LIMITED / INSUFFICIENTLY_RESOLVED.
- **Stage 2.**
  - 16 independent realisations (seeds 87201–87216), same rule and budget as the deployed state, plus the deployed state.
  - Per realisation: E_H and signed error per region, T1 − T0 change, panel hit counts, missed important relations, BIN / RAY family readings. Mean, SD and 95 % t interval.
  - Classes:
    - A (systematic): the pooled-state error keeps ≥ 50 % of the realisation-mean error, and that mean differs from 0.
    - B (variance): realisation SD ≥ |mean| and the pooled error ≤ 50 % of it.
    - C (receiver sampling): C3 halves the region-C signed error in ≥ 2 of 3 family cases.
    - D: a mixture of these.
- **Stage 3.**
  - Measured alone on the GPU. Ladders 4–16,384 spp × 3 replicates for every arm at T1.
  - Quality: relative L1 against the pooled REF-ER on stationary pixels at 0.10 / 0.05 / 0.03 / g. g is evaluated only if the pooled reference noise is ≤ g / 2.
  - WL86 region rule: region-matched if A, C, D are each ≤ 1.1 × the path tracer's error. Per-region time-to-quality is reported for C and D.
  - Matching uses log-log interpolation only, with no extrapolation.
  - Per-edit, source-only, cold and memory costs are measured, with amortisation over V = 1 / 4 / 16 frames.
- **Untouched condition E3.**
  - E3 is the first of three predeclared candidates whose geometry-only masks give ≥ 1,000 pixels in A, C, D and E, and whose geometry passes the WL86 admissibility checks.
  - No radiance of any candidate existed before the choice.
  - E3 needs new references (8 × 8,192 spp REF-ER and REF-P, seeds 87100+ / 87120+) and the WL86 admission criteria Q1–Q4, Q6, Q7. If E3 is not admitted it is reported, and the batch does not stop.
  - The dev smoke used a plumbing-only stand-in (E3dev: E1 geometry, another light) so that no E3-candidate radiance existed before the freeze.

## Pre-freeze development (non-evidence)

All development ran on LabServer63 in `~/rest_gm187dev`. It used a dev geometry from the WL86 dev split (seed 86300) with a dev light, or the plumbing-only E3dev stand-in. No E1 / E2 result value was read. No radiance of an E3 candidate was computed.

1. **ER sampler consistency.** Three independent directional-albedo estimators agree for B, S1 and S2 at three angles: uniform hemisphere, ER importance sampling and the MIS mixture.
   - The first per-bin test failure came from the MIS estimator's own noise at the dev sample count (about 64 samples per bin).
   - The test was rewritten to include both estimators' split-half noise before any evidence existed.
2. **Lazy-scatter defect, again.** The first GPU-resident q table came out 1.9× too large on every class. The cause was Dr.Jit 0.4.6 scattering unevaluated expressions (the WL84 "lazy scatter 2×" defect). Values and indices are now evaluated before every scatter.
3. **Cost of q.** The first q implementation (numpy round trips) took 0.85 s per call. The GPU-resident one takes 25 ms. This was done so that the comparator is not handicapped by implementation overhead.
   - Per call, after the first: state 30 ms, q 25 ms, four-order recurrence 10 ms.
   - The material W tables take 0.4–1.3 s once per realisation (geometry-independent).
4. **Dev geometry against a 512-spp path-traced field** (flux ratio / relative L1, which includes truth noise):
   - EXP: floor 0.98 / 0.10, panel 1.01 / 0.09, walls 1.00 / 0.06, boxes 1.00 / 0.05.
   - Sphere: 0.98 / 0.16 with the flat-frame approximation, and 0.95 / 0.12 with the curved-patch rows (item 6).
   - Per order (walls), flux ratios are 0.99–1.00.
5. **End-to-end dev smoke of the chain** (low spp, E3dev). Every stage ran and the visuals were produced. One analysis crash on the dev stand-in's geometry lookup was fixed.
6. **Independent code review** (one subagent, before the freeze). The following were fixed before the freeze:
   - **Curved sphere patches** (above). Before the fix, W used the patch-centre frame for the sphere rays: wrong cosine and lobe orientation up to about 40° from the local frame, and no local horizon. The oracle field has no such approximation.
   - **Common random numbers.** The table renders used 64-spp launches against ECRO's 32-spp launches, so only half the samples were shared. Renders now use 32.
   - **Verdict mapping.** It was documented but not coded; it is now implemented (`analyze87.py verdict`).
   - **Analysis rules brought in line with the protocol:**
     - INSUFFICIENTLY_RESOLVED requires every failure to be within 1.5 × noise.
     - DISCRETISATION_LIMITED requires the localisation evidence; otherwise FAILING_UNLOCALISED.
     - The receiver-sampling class uses the rendered errors.
     - The systematic class requires the same sign.
     - The capture criterion is included in the realisation pass fraction.
     - The region errors of a first ladder point are kept.
     - C3 GPU time uses wall time as a proxy.
     - Second-view amortisation is reported.
     - Per-realisation quadrature seeds are used.
   - C2-BINQ was redefined as the in-bin mean of the per-ray weights (same samples) instead of a separate dense table.

## Evidence run (freeze `60efecb`; LabServer63; 2026-11-11 08:38–10:15 UTC)

One detached chain. No step was rerun and nothing was changed after the freeze.

| Stage | Wall | Mean GPU | Outcome |
|---|---|---|---|
| E3 choice (geometry only) | 2 s | — | **E3a** chosen: first candidate, A 10,006 / C 5,693 / D 69,410 / E 20,379 pixels |
| Stage 0: tests ‖ E3 references (16 REF-ER + 16 REF-P, 8,192 spp) ‖ E3 oracle fields ‖ per-order truth fields | 20.9 min | 97 % | **TESTS PASS**; **E3 ADMITTED** |
| Explicit tables + controls (8 jobs) | 0.2 min | 20 % | finite, non-negative |
| 16 realizations × 6 configurations | 0.9 min | 24 % | — |
| 1,040 renders (4,096 spp) | 21.8 min | 95 % | — |
| Stage 1 / 2 analysis | 2 min | — | **ACCURATE_UNDER_FROZEN_REPRESENTATION** |
| Stage 3 economics (alone; no foreign GPU process before or after) | 49.6 min | 37 % | measured |

The economics stage is a single timing process. Its GPU utilisation is low by construction: small-spp ladder points are bound by Python launch overhead. It ran alone so that its times are uncontended.

## Final report (Korean)

### A. CURRENT ASSESSMENT

- **기존 불확실성.**
  - WL86의 ECRO는 LOW + table lookup으로 path tracing 대비 per-frame 9배의 이득을 보였습니다. 그러나 이것이 학습 덕분인지, table 표현 덕분인지 분리된 적이 없습니다.
  - WL86A는 sharp 실패의 원인으로 두 가지를 제시했습니다. 하나는 학습된 single flight의 결손이고, 다른 하나는 512-ray state relation의 sampling 한계입니다. 둘은 분리되지 않았습니다.
  - WL86B는 같은 state 위의 learning-free 계산이 결정적 공백이라고 지목했습니다.
- **보존한 것.** WL84 ER, WL86 checkpoints, reference, mask, threshold, verdict(WL86 NOT SUPPORTED, WL86A 귀속)는 그대로이고 어떤 것도 재학습하거나 수정하지 않았습니다.
- **이 실험이 구분할 수 있는 것.**
  - 같은 state, 같은 ER, 같은 readout 위에서 table을 학습으로 만드는 것과 물리 계산으로 만드는 것의 차이(fidelity, 비용).
  - table 표현 자체의 렌더 이득.
  - sharp 결손 가운데 state 표본(placement, variance)의 몫과 학습의 몫.

### B. IMPLEMENTATION FACT

- **주 비교기 EXP.** 같은 512-ray state(hit patch, hit에서의 outgoing bin)를 사용합니다.
  - 각 ray의 정확한 국소 방향에서 frozen ER로 계산한 weight W_ray = (1/ΔΩ)∫_{b_o} f(v, ω_r)dv · cos ω_r · ΔΩ/8 을 씁니다(65,536 MIS 샘플).
  - L₁ = q는 실제 광원의 1-event 방사휘도를 patch × bin으로 사영한 것입니다(점 16개, MIS 1,024).
  - L_{k+1} = T_d L_k를 네 번 적용하고 B = L₂ + … + L₅ 입니다. 2–5 event로, WL86 table 정의와 같습니다.
  - camera readout은 WL86 그대로입니다: LOW + B[patch(x2), bin(x2 → x1)], 3–6 event. q는 B에 포함되지 않습니다(e2와 중복 없음).
  - 구 패치는 band별 행을 ray 원점의 국소 frame, cosine, horizon으로 계산하고 유효 solid angle로 정규화했습니다(oracle field와 같은 정의).
  - 재정규화나 clamp는 없습니다.
- **구현.**
  - W는 재질과 state 방향에만 의존하므로 한 번 계산합니다(1–2.6 s).
  - edit당 비용(warm median): geometry 32 ms, selective state 19–21 ms, q 26–30 ms(GPU 2 ms), 4-order recurrence 10–11 ms(torch), 전송 1 ms.
- **대조군.**
  - C2-BINQ: bin 안의 입사 방향 정보를 제거합니다.
  - C2-ENC: ECRO의 16-point encoding s를 입력으로 씁니다. ECRO와 입력이 완전히 같습니다.
  - C2-RAYSEND: 첫 gather에서 정확한 sender 방사휘도를 읽습니다.
  - C3 sender-side deposition: order당 650,240 ray, 같은 출력 table을 씁니다. 동일-state 비교가 아닙니다.
  - POOL: 16 realization 평균 T입니다. 진단용이며 matched budget이 아닙니다.
  - TF: 물리 table fill, 16–4,096 spp/slot.
  - PT, PT_RR: Russian roulette는 vertex 3부터 적용합니다.
- **E3.**
  - 사전 규칙(geometry-only mask)으로 E3a를 선택했습니다.
  - reference는 새로 렌더했고, WL86 기준 Q1–Q4, Q6, Q7을 통과해 admitted 되었습니다. g = 0.0064 입니다.
  - E3의 C 영역은 전부 panel 반사 footprint(바닥)이고, panel mirror-cone 수신자는 0입니다.

### C. MEASUREMENT

**정확성 (Stage 0, 모든 gate PASS).**

- WL86 E_H 재계산 차이는 0입니다. 재렌더 readout 차이는 4e-9 입니다.
- ER reciprocity 1.2e-5, MIS 대 uniform per-bin 추정기 ≤ 2.2 %(노이즈 내).
- Lambert 정규화 4e-16, q 대 해석적 조도 ≤ 0.19 %, torch 대 numpy loop 5e-7.
- zero / no-visibility 0, 선형성 1.5e-7, furnace 6e-7.
- null edit, move-and-return, selective = full 은 B가 비트 동일입니다.
- RR 대 PT 기대값 0.02 % 이내.
- 보고용(이산화 진단): 512-ray quadrature의 directional albedo 오차 median은 panel S1 5.3 %(max 18 %), B 1.0 %, S2 2.0 % 입니다.

**충실도 (E_H, T1. EXP / ECRO s8601 / s8602 / oracle table).**

| 영역 | E1 | E2 | E3 (미검사 조건) |
|---|---|---|---|
| stationary | 0.0040 / 0.0056 / 0.0043 / 0.0032 | 0.0041 / 0.0048 / 0.0040 / 0.0025 | 0.0038 / 0.0047 / 0.0041 / 0.0030 |
| A | 0.0092 / 0.0126 / 0.0113 / 0.0075 | 0.0046 / 0.0060 / 0.0069 / 0.0033 | 0.0090 / 0.0097 / 0.0089 / 0.0085 |
| **C** | 0.0113 / 0.0132 / 0.0110 / 0.0074 | **0.0491** / 0.0717 / 0.0634 / 0.0137 | 0.0052 / 0.0051 / 0.0063 / 0.0050 |
| D | 0.0033 / 0.0049 / 0.0036 / 0.0028 | 0.0041 / 0.0043 / 0.0034 / 0.0024 | 0.0050 / 0.0059 / 0.0052 / 0.0043 |
| panel object | 0.0185 / 0.0400 / 0.0339 / 0.0121 | 0.0390 / 0.0555 / 0.0515 / 0.0115 | 0.0104 / 0.0118 / 0.0104 / 0.0080 |

- EXP는 WL86 gate를 모두 통과합니다(E1, E2, E3).
  - F1 stationary 0.0038–0.0041, capture ≤ 0.02.
  - F2 A, C, D ≤ 0.06. E2 C는 0.049로 noise bound 0.013 범위 안에 있습니다.
  - F3 D slope 0.97–1.01, corr ≥ 0.99.
  - F4 LMIX 0.0039, LSPOT 0.0069. 같은 조건에서 ECRO는 0.026 / 0.035 입니다.
- 학습 대 explicit. 36개 region-class 비교(3 edit × T0/T1 × 6 class) 결과입니다.
  - ECRO가 두 seed 모두 noise 이상으로 나은 곳: **0곳**.
  - 나쁜 곳: **3곳** (E1 T1 panel, E2 T1 C, E2 T1 panel).
  - 나머지는 noise 내입니다. E3에서는 둘이 noise 내로 같습니다.
- 대조군(E2 T1 C, E_H / signed).

  | arm | E_H | signed |
  |---|---|---|
  | EXP | 0.049 | −0.049 |
  | BINQ | 0.046 | −0.046 |
  | ENC | 0.054 | −0.053 |
  | RAYSEND | 0.059 | −0.058 |
  | **C3** | **0.024** | **−0.005** |

  - E2 T0도 같은 패턴입니다: EXP 0.055, C3 0.021.
  - 입사 bin quadrature(BINQ), source 사영(ENC), 1-event sender 사영(RAYSEND)은 결손을 설명하지 못합니다.
  - 같은 ray 예산의 sender-side 배치(C3)가 결손의 약 90 %를 제거합니다.
- field 수준(T1 전체 relative L1 / flux). EXP 0.043–0.051 / 0.99–1.00, ECRO 0.045–0.056, oracle noise 0.020–0.026.
  - panel table은 EXP 0.08–0.10, ECRO 0.20–0.23 입니다.
  - 차수별 EXP 오차는 truth noise와 비슷합니다. 2-event만 약 2 × noise입니다.
- **signed edit response (T1 − T0, slope / corr).**

  | arm | D | C (E1 / E2 / E3) |
  |---|---|---|
  | EXP | 0.97–1.01 / ≥ 0.99 | 0.86 / 0.98 / 0.96 |
  | ECRO | 0.80–0.97 | 0.38 / 0.63 / 0.96 |

**State realization (Stage 2; 16개 독립 realization, 같은 규칙과 예산).**

| | E2 T1 C E_H (mean, SD, range) | signed mean [95 % CI] | deployed | pooled 8,192 ray | C3 |
|---|---|---|---|---|---|
| EXP | 0.044, 0.018, 0.024–0.077 | −0.024 [−0.042, −0.006] | −0.049 | −0.024 (E_H 0.030) | −0.005 |
| ECRO (같은 state) | 0.062, 0.012 | −0.058 [−0.065, −0.051] | −0.070 | — | — |

- E2 C에서 EXP의 realization 통과율(≤ 0.06)은 75 % 입니다. E1과 E3에서는 100 % 입니다.
- deployed state는 분포의 나쁜 꼬리에 있습니다: signed −0.049, realization 평균 −0.024.
- **family (light → panel → Lambert).**
  - 중요 relation 중 하나도 hit하지 못한 비율은 56–62 %(가중 38–65 %)이며, 모든 realization에서 비슷합니다.
  - 중요 patch당 panel hit는 min 0, median 13–20 입니다.
  - BIN read / truth: deployed 0.73–0.97, realization 평균 0.88–1.07(SD 0.13–0.30).
  - patch-level relative RMS: BIN 0.41–0.69, RAY 0.39–0.97.
- 분류(사전 규칙, region C).
  - E2 T1: D_MIXTURE (A systematic + C receiver sampling). pooled가 평균 오차의 100 %를 유지하고 CI는 0을 제외합니다.
  - E2 T0: C_RECEIVER_SAMPLING.
  - E1: D_MIXTURE.
  - E3: C 오차가 noise 수준이라 규칙상 C만 성립합니다.
  - C3 halving은 3/3 family case에서 성립합니다.

**경제성 (Stage 3, RTX 3080 Ti 단독, 512², T1).**

- **per-edit 고정비용(wall).** PT 36 ms, ECRO 69–78 ms, EXP 89–95 ms, TF64 152 ms, TF256 334 ms, TF1024 1.2 s, TF4096 4.6 s, C3 0.43–0.58 s.
  - TF16의 3.4 s는 첫 호출 kernel 컴파일이 warm 측정에 섞인 측정 결함입니다. GPU 시간은 9 ms 입니다.
- **per-frame.** 모든 table arm이 같은 LOW + lookup 렌더러를 씁니다(같은 spp에서 GPU 비용 약 1/5.5). matched quality 0.05–0.03의 edit-to-first-frame GPU는 PT 대비 6.6–8.2× 입니다(ECRO, EXP). 두 번째 view(64 spp)는 PT / table 4.0× wall, 5.6× GPU 입니다.
- **matched quality, edit-to-first-frame wall, PT 대비 배수.**

  | 수준 | ECRO | EXP | TF64 | TF256 | PT_RR |
  |---|---|---|---|---|---|
  | 0.10 | 1.9–2.1 | 1.6–1.8 | 1.1–1.3 | 0.6–0.7 | 0.8–1.1 |
  | 0.05 | 3.7–4.4 | 3.2–4.1 | 2.3–3.3 | 1.3–2.0 | 0.8 |
  | 0.03 | 5.0–5.3 | 4.7–5.2 | 3.8–4.6 | 2.5–3.5 | 0.8 |
  | g | 3.9 / 4.6 / **미도달** | 5.5 / 5.1 / 4.3 | 4.5 / 5.3 / 3.5 | 5.4 / 5.4 / 4.7 | 0.8 / 0.75 / 미도달 |

  - g에서 E1과 E2의 region 규칙은 ECRO와 EXP 모두 실패합니다(C 영역 오차가 PT보다 큼). E3에서는 ECRO와 PT_RR이 g에 미도달이고, EXP와 TF는 도달하며 region-matched 입니다.
- **ECRO / EXP 비용비.**
  - edit-to-first-frame: 0.10–0.03 수준에서 0.85–0.98 (ECRO가 2–15 % 빠름), g 수준에서 1.11–1.40 (EXP가 빠름).
  - source만 바뀌는 갱신: ECRO 13–18 ms(factor 재사용), EXP 51–64 ms(q 42–54 ms + recurrence 10 ms). 비율 0.25–0.28 입니다.
- **region별 time-to-quality.**
  - E2 C 0.05: PT 2,440 spp, EXP 5,485 spp(bias floor 0.049 근처), ECRO 미도달(floor 0.072), C3 224, TF4096 190.
  - E2 C 0.03: PT 7,412 spp, C3 1,470, TF1024 1,525, TF4096 683. EXP / ECRO / TF256은 미도달입니다.
  - D 영역에서는 모든 table arm이 PT의 약 0.6× spp로 같습니다.
- **amortisation (V frames, 0.03).**
  - V = 1: PT 1.4–2.5 s, ECRO 0.27–0.48 s, EXP 0.29–0.49 s.
  - V = 16: PT 21–40 s, ECRO 3.3–6.5 s, EXP 3.3–6.3 s.
- **cold.** PT 첫 렌더 21 s(컴파일), table 0.7–0.9 s. W tables 2.6 s(첫 프로세스).
- **memory.** state 17.6 MB, W 3.9 MB, ECRO factors 42.9 MB(735,664 params), table 1 MB, process 3.9–4.9 GB.

### D. OBSERVATION

1. 같은 state에서 물리 계산으로 만든 table은 학습한 table과 같거나 더 정확합니다. 차이는 sharp 수신자에서 나타납니다: E1 panel 0.019 대 0.034–0.040, E2 C 0.049 대 0.063–0.072.
2. 학습 operator 고유의 오차가 따로 보입니다.
   - E1 panel 과대예측(+0.038, WL86A의 global term)은 EXP에 없습니다. EXP는 반대 부호의 작은 결손(−0.013)을 보입니다.
   - C 영역의 edit response slope(E1 0.38)도 EXP에서는 0.86입니다.
   - LSPOT 편향(0.026–0.035)도 EXP에서는 0.007입니다.
3. EXP에도 E2 C 결손이 남아 있습니다(hl/h 0.93). 그 크기는 realization에 따라 0.024–0.077로 흔들립니다.
4. 16배 ray(pooled)에서도 평균 결손의 상당 부분이 남습니다. 반면 같은 예산의 sender-side 배치는 이를 −0.005로 줄입니다.
5. broad 영역(D, stationary)에서는 EXP, ECRO, TF, oracle이 모두 노이즈 수준에 가깝습니다.
6. 렌더 이득은 table 사용 여부로 결정됩니다. table을 누가 만들었는지와는 무관하게 per-frame 비용이 같습니다.

### E. INTERPRETATION

- **물리 표현(patch × bin table).**
  - oracle, TF4096, C3가 E2 C를 0.014–0.024로 표현합니다. 출력 표현은 sharp family를 담을 수 있습니다(WL86 R1과 일치).
- **explicit 계산.**
  - 같은 state에서 학습 없이 WL86 gate를 전부 통과합니다.
  - 미검사 조건 E3에서도 학습 operator와 동등합니다.
  - 측정 범위에서 H0(explicit sufficiency)를 지지합니다.
- **학습 근사.**
  - fidelity 측면의 추가 효용은 관측되지 않았습니다.
  - 오히려 학습은 sharp 수신자, 미학습 source class, C edit response에서 추가 편향을 만듭니다.
  - 측정된 유일한 이점은 geometry가 고정된 상태에서 source만 바뀔 때의 table 갱신 비용입니다(3.5–3.9×, 절대값 약 40–50 ms).
  - geometry edit 후 첫 프레임까지의 비용 차이는 ≤ 15 %로, 사전 기준 1.25×에 못 미칩니다.
  - 이는 측정이며, 다른 source 수나 scene 규모로 일반화되지 않습니다.
- **table 기반 렌더링.**
  - WL86의 per-frame 9× 이득은 LOW + lookup 구조에서 나옵니다. EXP, TF64, TF256이 같은 이득을 가집니다.
- **state 재사용.**
  - geometry-dependent state(17.6 MB)는 ECRO와 EXP가 공유합니다. selective update 차이는 1 ms 수준입니다.
- **sharp 한계의 소재.**
  - 남은 결손은 receiver-gather relation의 표본 배치에 속합니다. 학습 operator가 아니라 공통 state의 몫입니다.
  - 512 ray를 수신측 고정 방향으로 쓰면 light → panel → Lambert 관계의 56–62 %가 표본되지 않습니다.
  - 이 결손은 같은 예산의 sender-side 배치나 조밀한 physical table fill로 사라집니다. 다만 현재 구현에서는 둘 다 edit당 0.4–4.6 s로 비쌉니다.
  - "더 조밀한 수신측 state가 필요하다"는 결론은 지지되지 않습니다. pooled 16×도 평균 결손의 일부를 남기기 때문입니다.
- **추론 표시.** 위 내용 가운데 "학습 operator 고유의 오차"는 같은 state, 같은 seed의 쌍 비교에서 나온 측정입니다. 그 원인 메커니즘(WL86A의 single flight와 global term)은 기존 귀속에 기댄 추론입니다.

### F. UNRESOLVED QUESTIONS

- source만 바뀌는 갱신에서 EXP의 q 비용(42–54 ms)이 줄어들 수 있는지는 시험하지 않았습니다. 이 비용은 사전 고정한 MIS 예산이 지배합니다. 따라서 학습의 유일한 측정 이점이 구현 예산에 얼마나 의존하는지 모릅니다.
- C3와 TF의 비용(0.4–4.6 s)은 이번 구현(torch index_add, numpy 재표본)에서 측정한 값입니다. 최적화된 sender-side 계산의 비용은 측정하지 않았습니다.
- pooled state가 남기는 E2 C의 체계적 결손(−0.024)의 원인은 분리하지 않았습니다. 후보는 수신 patch의 공간 평균, outgoing bin 평균, 512-방향 공유 offset 규칙입니다.
- E3의 C 영역은 바닥 footprint뿐이라 panel mirror-cone 수신자가 없습니다. 따라서 미검사 조건에서의 sharp-receiver 검증은 약합니다.
- scene 규모, 재질 수, area light, non-rigid edit에 대한 scaling은 측정하지 않았습니다. ECRO는 family-specific 학습이고 EXP는 W가 재질과 state 방향 전용입니다.
- TF16의 wall 시간은 측정 결함(컴파일 포함)으로 무효입니다. GPU 9 ms 기준으로는 경쟁력이 있을 수 있으나 측정되지 않았습니다.

### G. ARCHITECTURE VERDICT (scope: one room family, rigid box edits, three frozen ER materials, point / cone sources, 512², E1 / E2 historical, E3 untouched)

| 항목 | 판정 (사전 mapping) | 근거 |
|---|---|---|
| EXPLICIT OPERATOR CORRECTNESS | **SUPPORTED** | Stage 0 gate 전부 통과, 유한·비음수, flux 0.99–1.00 |
| FROZEN STATE ADEQUACY | **CONDITIONALLY SUPPORTED** | deployed state로 모든 gate 통과(ACCURATE). 그러나 E2에서 realization 통과율 75 % < 90 %, broad는 통과 |
| SHARP TRANSPORT REPRESENTABILITY | **INDETERMINATE** | E2 C에서 pass/fail이 realization마다 다르고(0.024–0.077), deployed와 pooled는 통과. 체계적 잔여 결손(pooled −0.024)과 표본 분산이 함께 있음. sender-side 배치는 같은 예산으로 해소 |
| BROAD TRANSPORT FIDELITY | **SUPPORTED** | F1, F2-D, F3-D 모든 edit 통과 |
| LEARNED OPERATOR ADDITIONAL UTILITY | **SUPPORTED (사전 규칙) — 근거는 source만 바뀌는 갱신 비용 하나** | fidelity 효용 없음(우세 0 / 열세 3 / 36). edit-to-first-frame ≤ 15 % 차이(기준 미달). source-only table 갱신 3.5–3.9× 저렴(약 40–50 ms) |
| QUALITY-MATCHED ECONOMICS | EXP **CONDITIONALLY** (E3 SUPPORTED); ECRO **CONDITIONALLY** (E3에서 g 미도달); TF64 / TF256 **CONDITIONALLY**; PT_RR / C3 / TF1024 / TF4096 **NOT SUPPORTED**; TF16 **NOT SUPPORTED, wall 측정 결함** | 0.10–0.03에서 table arm은 PT 대비 2–5×. g에서 E1 / E2 region 규칙은 C 오차로 실패 |
| DYNAMIC EDIT VALIDITY | **SUPPORTED** (EXP) | F3 모든 edit 통과, C 영역 slope 0.86–0.98 |
| GM1 ARCHITECTURE IMPLICATION | 조건부 | 측정 범위에서 H0이 성립합니다. 학습 없는 동일-state 계산이 학습 operator와 같거나 나은 fidelity를, 비슷한 edit 비용과 같은 frame 비용으로 냅니다. H1은 source-only 갱신에서만 성립합니다. H2는 출력 표현이 아니라 수신측 512-ray relation의 sharp-family 표본 배치에 한정해 부분적으로 성립하며, 학습과 explicit 모두에 공통입니다. 학습 operator 연구를 지지하는 측정은 source-only 갱신 비용 하나이고, 이번 결과는 explicit physical baseline을 이 workload의 기준점으로 세웁니다. 이후 아키텍처 선택은 하지 않습니다 |

**Completion outcome.**
- 1 (EXPLICIT SUFFICIENCY SUPPORTED, 이 workload와 0.10–0.03 및 g 수준 한정)에 해당합니다.
- 여기에 4의 요소가 함께 있습니다. 학습 효용은 source-only 갱신에서만 측정되었고, sharp 표현성은 state 표본에 의존해 미결입니다.
- ECRO v0의 WL86 verdict와 WL86A 귀속은 그대로입니다.

### H. REPRODUCIBILITY

- **Commits.** freeze `60efecb`, results and this report in the closing commit. Parent: WL86 `7786d09` + `63972c8`, results `b97027d`; WL86A `46593f0` / `1bd7df1` / `f6483ab`, `bd81442`.
- **Protocol.** `experiments/gm1_explicit87/protocol/gm1_explicit87.json` (no amendment).
- **Frozen inputs.** `experiments/ecro86/data/mats86.npz` (ER states, sha256 in `mats86.json`). Checkpoints `~/rest_ecro86/results/models/ecro_s8601.pt` / `ecro_s8602.pt`. WL86 references, masks and fields in `~/rest_ecro86`. WL86A importance and split fields in `~/rest_ecro86a`.
- **Seeds.**
  - Quadrature 87001 (+1000 per realization), dense 87002, q 87003, C3 87004.
  - Realizations 87201–87216 (deployed 86001).
  - Renders 86600 / 86601 (CRN with WL86 ECRO renders), E3 oracle renders 86400 / 86401.
  - E3 REF-ER 87100–87107, REF-P 87120–87127, E3 oracle field 87450, order fields 87460.
  - Ladders 87600 + 97·spp + rep (RR +7, tables +50), TF 87700+, tests 879xx.
- **Environment.** RTX 3080 Ti 12 GB, driver 595.71.05, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Server.** `~/rest_gm187`: tables, fields, 1,040 renders, references, logs, GPU trace; 1.8 GB.
- **Local copy.** `results/gm1_explicit87/`: chain status, GPU trace, all JSON (tests, E3 choice / admission, stage1–3, econ, final), logs, render / table metadata.
- **Visuals.** `results/evaluation/87/` (manifest with sha256).
- **Invalid or amended.** No amendment.
  - TF16 warm wall time includes first-call kernel compilation. This is a measurement defect of the cost loop: TF16 was the first budget measured. Its wall verdict is invalid; its GPU time and quality stand.
  - Pre-freeze dev runs: see above.
