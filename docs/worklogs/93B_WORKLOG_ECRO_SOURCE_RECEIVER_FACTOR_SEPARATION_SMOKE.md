# Worklog — ECRO Source/Receiver Factor Separation Smoke Test (WL93B) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93B Source/Receiver Factor Separation Smoke Test" (user, 2026-11-13).
- **Batch class.** Suffix batch continuing the WL93 architecture question with WL93A evidence. It reuses the existing WL93 tied, untied and capacity-matched implementations at the WL93A smoke scale; it adds no new architecture class or mechanism.
- **Preserved.**
  - WL93 STOPPED-INCONCLUSIVE (its Amendment 1 is still pending approval).
  - WL93A NO_SIGNAL_AT_SMOKE_SCALE (freeze `888ceb6`).
  - All earlier checkpoints and artifacts.
- **Question.** At smoke scale, does source / receiver factor separation alleviate the source-stage versus whole-transport accuracy conflict seen in WL93A, beyond parameter-capacity effects?
- **Locations.**
  - Branch `Selective_Recompute`, LabServer63.
  - Code: `experiments/ecro_stage93b/`. It reuses `ecro_stage93/model93.py` (unchanged), `ecro_stage93a` (references, targets, training contract, metrics) and the WL92 render / reference tools.
  - Results: `results/ecro_stage93b/` and `results/evaluation/93B/`.
  - Server root: `~/rest_x93b`. The WL93A targets and references are read-only links to `~/rest_x93a/data`.

## Protocol (`experiments/ecro_stage93b/protocol/ecro_stage93b.json`)

**Inspection disclosure.** WL93A had already measured and inspected T0 / TQ. The quality of U0 / UQ / C0 / CQ has never been observed: WL93 stopped before training, and its dev chain is not evidence.

**Arms.** The model code is `model93.Local` (WL93 freeze `c30c797` definitions).

| Arm | Factors | Objective | Origin |
|---|---|---|---|
| T0 | tied | OWN | WL93A checkpoint (verified) |
| TQ | tied | OWN + WL93 source-stage (w = 1) | WL93A checkpoint (verified) |
| U0 | untied (receiver heads copied from the source heads) | OWN | trained here |
| UQ | untied | OWN + source-stage | trained here |
| C0 | capacity-matched tied (shared factor softplus(head + head2), head2 hidden layers from seed + 9300, zero output layer) | OWN | trained here |
| CQ | capacity-matched tied | OWN + source-stage | trained here |
| R_T0 / R_TQ | tied | as T0 / TQ | retrained here: reproduction controls and run-to-run variation |

**Training contract.** The WL93A contract is unchanged:
- Data: the WL93A 128-geometry subset, its B0 training targets and its training-validation targets.
- Schedule: 2,000 steps; lr 3e-4 → 1e-5 cosine with 200 warmup steps; clip 10; data order rng(seed + 93); seeds 8601 / 8602.
- The global term is frozen to WL92A own_s<seed> and is identical across arms (tested).

**Reproduction gate.**
- R_X must match the WL93A X within 2 % relative on validation whole, single-vs-L2 and source-stage (training-validation target) relative L1. Otherwise STOP: historical controls not reproduced.
- The reused checkpoints must load with the recorded state hashes (T_REUSE).

**Accounting** (validation primary; train-representative and E2 for context; WL93A B5 references; g00010 in every full-population number).

| Stage | Measure |
|---|---|
| Source local use | Bq vs B5. Primary: relative L1 vs the B5 mean (SRC). Also the cross-product relative L2 M (the square root of a ratio of two unbiased cross-product estimates, not itself exactly unbiased), flux / directional split and peak retention |
| Receiver local use | a_r c_r applied to the first flight of the physical q̄_B5, vs t2 (exact frozen-ER floor reported). Also the receiver use on the model's own first flight, vs M64 |
| Single flight | S vs t2 (SINGLE), plus cross-product L2 with the f2 / f2b halves |
| Global (frozen) | G vs t − t2 |
| Whole | S + G vs t (WHOLE, primary), plus cross-product L2 with the f / fb halves |
| Operators | E2 T1 panel source / receiver matrices vs M64: mirror retention, and the source − receiver difference |

**Uncertainty.**
- Paired geometry bootstrap of pooled differences (2,000 resamples, seed 93002).
- Run-to-run variation r_m from R_X vs X.
- margin_m = max(2 SE, r_m).

**Criteria (per seed s).** Let gap_s = WHOLE(TQ) − WHOLE(T0) and gain_s = SRC(T0) − SRC(TQ).
- **(a) Source benefit preserved.** SRC(UQ) ≤ SRC(T0) − 0.5 gain_s, and SRC(UQ) ≤ SRC(TQ) + margin.
- **(b) Whole improved vs TQ.** WHOLE(UQ) ≤ WHOLE(TQ) − max(margin, 0.5 gap_s), and SINGLE(UQ) ≤ SINGLE(TQ) + margin.
- **(c) Whole improved vs CQ.** WHOLE(UQ) ≤ WHOLE(CQ) − max(margin, 0.5 gap_s).
- **(d) No compensating regression.** Rendered stationary and D E_H ≤ 1.25 × T0 + 0.002 on E2 T0 / T1, E5k995 T0 / T1 and E1 LSPOT, and no WL86 gate lost vs T0.

**Labels** (first match wins):
1. SHARING_SIGNAL: a, b, c, d for both seeds.
2. CAPACITY_EXPLANATION: a and b for both seeds, c fails for at least one seed, and CQ meets b for both seeds.
3. TRADEOFF_REMAINS: a for both seeds, b fails for both seeds.
4. NO_ADVANTAGE: b fails for both seeds, a fails for at least one seed, and U0 does not improve whole over T0.
5. INCONCLUSIVE: anything else, or a control failure.

Contrasts U0−T0, UQ−U0, CQ−C0, C0−T0 are reported descriptively. They are not additive shares.

**Renders (inspected conditions only, diagnostic).**
- Tags: E2 T0 / T1, E1 T1 LSPOT, E5k995 T0 / T1, and the pattern diagnostic.
- Models: 6 arms × 2 seeds, plus WL92A own (which checks reproduction of the WL93A renders).
- Sampling: 2 × 4,096 spp, seeds 86600 / 86601.

**Pattern diagnostic (visual only; never in the verdict).**
- **Interface limit.** The source interface (`scene86.light_arrays`) admits at most **two lights**. A multi-cell checkerboard is therefore not expressible without changing the source contract, so the diagnostic is reduced to a **2 × 2 checker**.
- **Layout.** Two hard-edged cone lights (the existing cone class, as in LSPOT) illuminate the two diagonal 40 cm cells of a 2 × 2 cell block on the panel. The cell edges lie on the 20 cm patch grid. The two other cells stay unlit.
- **Flux.** Equal to the E2 light's panel-incident flux.
- **Reference.** New REF-ER, 8 × 8,192 spp.
- **Representability gate (before any render).** From the deployed encoding, every lit-cell patch must be ≥ 0.3 × the lit maximum and every other panel patch ≤ 0.05 × it.
- **Not run.** No above-Nyquist stress pattern.
- **Caveat.** The cone source class was never used in training.

**Tests.** T_REUSE, T_INIT, T_GLOBAL, T_OWNERSHIP, T_CAPACITY (including non-degeneracy after training), T_TARGETS, T_SPLIT (geometry content hash), T_DET, T_RENDER_REF.

## Dev chain (before the freeze; not evidence)

`~/rest_x93bdev` with `--dev`: 100 steps, 64 spp, the full WL93A references.
- Every stage ran: reuse, pattern, tests, 12 trainings, accounting, operators, render tables, LPAT reference, renders, analysis and visuals.
- Expected dev-only failures: the reproduction gate and render-reference consistency fail at 100 steps / 64 spp.
- T_REUSE passes. Pre and post tests pass. Parameter counts are tied 405,424, untied = tiedcm 487,952. Non-degeneracy C0 vs T0 is 0.22.
- **Pre-freeze correction.** The first pattern layout (lights straight in front of the cells, cone radius 0.17 m) failed the representability gate: the moved box B at T1 completely shadows the second cell (all four patch levels 0).
  - Corrected layout: the lights sit 0.8 m toward −y (oblique, clearing box B) with cone radius 0.13 m.
  - The gate then passes. Lit patches are at 0.58–1.0 of the maximum, dark patches at 0, panel flux ratio vs the E2 light 1.000, region C 1,527 px.
  - This was decided before any evidence run; the only model outputs on the pattern were dev 100-step / 64-spp ones.
- A close-up figure of region C was added to the visuals.

**Probes.** The stage costs are known from WL93A and WL92B, which use identical code paths:
- training: 4 concurrent runs at ≈ 82 % GPU, ≈ 3.5 min per batch of 4 (12 runs ≈ 11 min);
- accounting (18 models × 73 configurations): ≈ 2.6 min (dev measurement, same size);
- REF-ER: pool 6 (8 × 8,192 spp);
- renders: pool 5 (168 renders).

## Freeze and execution

- **Freeze and deployment.** Commit `03fbc84` was deployed to `~/rest_x93b` (`git archive`, LF).
- **Chain.** One detached setsid chain on LabServer63 (RTX 3080 Ti), 09:36:46–10:07:33 UTC (31 min).
- **Stage results.**
  - T_REUSE passed: the WL93A T0 / TQ state hashes match the recorded hashes.
  - The pattern gate passed.
  - Tests passed before and after training.
  - 12 training runs (4 concurrent, about 3.5 min per batch).
  - Accounting and operators completed.
  - Render tables, the LPAT reference (8 × 8,192 spp) and 184 renders completed.
- **Analysis.** Produced by the frozen `analyze93b.py`. No amendment, no rerun, no post hoc rule.
- **Control results.**
  - **Reproduction.** R_T0 / R_TQ reproduce the WL93A checkpoints' validation metrics to every printed digit, because training is deterministic on this path. Run-to-run variation is r_m = 0, so the margins equal 2 SE.
  - **Render-reference consistency.** Re-rendered WL92A own vs the WL93A renders differ by at most 1.3e-10 in region-C E_H.
  - **Global term.** Identical across arms (exact).
  - **Parameters.** Tied 405,424; untied = tiedcm 487,952.
  - **Non-degeneracy.** Post-training C0 vs T0 is > 1e-4.

## Measurements (validation, 55 geometries × 4 lights, g00010 included)

**Stage and integrated relative L1.** Seed 8601 / 8602.

| Arm | Source stage vs B5 (SRC) | Source peak retention | Receiver use on physical first flight vs t2 (floor 0.0998) | Single flight vs t2 | Whole vs t |
|---|---|---|---|---|---|
| WL92A own (init) | 0.923 / 0.939 | 0.28 / 0.30 | 0.1079 / 0.1072 | 0.1039 / 0.1053 | 0.0767 / 0.0775 |
| T0 | 0.899 / 0.927 | 0.31 / 0.33 | 0.1069 / 0.1072 | 0.1022 / 0.1042 | 0.0762 / 0.0771 |
| TQ | 0.738 / 0.790 | 0.48 / 0.45 | 0.1096 / 0.1104 | 0.1097 / 0.1116 | 0.0803 / 0.0806 |
| U0 | 1.067 / 1.092 | 0.17 / 0.21 | 0.1192 / 0.1768 | 0.0960 / 0.0965 | **0.0735 / 0.0737** |
| UQ | 0.737 / 0.808 | 0.49 / 0.47 | **0.1046 / 0.1055** | 0.1035 / 0.1057 | 0.0770 / 0.0775 |
| C0 | 0.899 / 0.927 | 0.31 / 0.33 | 0.1068 / 0.1072 | 0.1022 / 0.1042 | 0.0761 / 0.0771 |
| CQ | 0.740 / 0.790 | 0.48 / 0.45 | 0.1095 / 0.1106 | 0.1097 / 0.1117 | 0.0802 / 0.0806 |

The frozen global term's error is identical across arms: 0.0920 / 0.0915.

**Primary contrasts.** Paired geometry bootstrap; 95 % CI in brackets.

| Contrast | Seed 8601 | Seed 8602 |
|---|---|---|
| UQ − TQ, SRC | −0.0015 [−0.019, 0.014] | +0.0176 [−0.008, 0.037] |
| UQ − TQ, WHOLE | −0.0032 [−0.0044, −0.0025] | −0.0031 [−0.0039, −0.0025] |
| UQ − TQ, SINGLE | −0.0062 [−0.0087, −0.0045] | −0.0059 [−0.0079, −0.0043] |
| UQ − CQ, WHOLE | −0.0032 [−0.0044, −0.0024] | −0.0031 [−0.0038, −0.0025] |
| TQ − T0, WHOLE (the WL93A regression) | +0.0041 [0.0026, 0.0065] | +0.0036 [0.0027, 0.0046] |
| U0 − T0, WHOLE | −0.0027 [−0.0034, −0.0021] | −0.0033 [−0.0049, −0.0023] |
| U0 − T0, SRC | +0.168 [0.114, 0.218] | +0.166 [0.083, 0.245] |
| UQ − U0, WHOLE | +0.0035 [0.0021, 0.0056] | +0.0038 [0.0024, 0.0056] |
| C0 − T0, WHOLE | −0.00001 | 0.00000 |
| CQ − TQ, WHOLE | −0.00007 [−0.00016, 0.00008] | −0.00004 [−0.00019, 0.00007] |

**Criteria.** Both seeds meet a, b, c and d. b_CQ is false for both seeds, because CQ does not improve on TQ. U0 improves whole over T0.

**Label: SHARING_SIGNAL** (frozen mapping).

**Per-geometry distributions.**
- UQ is better than TQ on WHOLE for 55 / 55 geometries, both seeds; median −0.0028 / −0.0028.
- UQ is better than TQ on SINGLE for 53 / 55 and 51 / 55 geometries.
- On SRC, UQ is better than TQ for 49 / 55 and 45 / 55 geometries (median −0.016 / −0.013). The pooled SRC gap for seed 8602 (+0.018) comes from a few high-mass geometries.
- UQ vs T0 on WHOLE: better for 29 / 55 and 27 / 55 geometries; median −0.0002 / +0.00003, i.e. parity.
- U0 vs T0: WHOLE better for 55 / 55 geometries, but SRC worse for 54 / 55.

**g00010** (full-population member; whole relative L1 0.24–0.35 for every arm, versus a pooled value near 0.077).

| Metric | T0 | TQ | U0 | UQ | CQ |
|---|---|---|---|---|---|
| Whole (s8601 / s8602) | 0.256 / 0.334 | 0.319 / 0.351 | 0.238 / 0.294 | 0.290 / 0.330 | 0.323 / 0.348 |
| Source stage | 0.844 / 1.084 | 0.823 / 1.028 | 0.801 / 0.913 | 0.883 / 1.117 | 0.846 / 1.009 |

- UQ reduces the TQ whole regression on g00010 as well.
- Its source stage there is worse than TQ for both seeds, and worse than T0 for both seeds.
- WL93A showed that g00010's target is uncertain even at B5 (noise 0.66). Its source-stage values carry that reference uncertainty. Its whole / single values use the WL86 / WL92A targets, which do not carry it.

**High-confidence subgroup** (previously inspected; secondary; 54 geometries without g00010). WHOLE:

| | T0 | TQ | UQ | CQ | U0 |
|---|---|---|---|---|---|
| s8601 / s8602 | 0.0728 / 0.0723 | 0.0758 / 0.0756 | 0.0730 / 0.0728 | 0.0757 / 0.0756 | 0.0704 / 0.0696 |

- SRC: TQ 0.726 / 0.755, UQ 0.715 / 0.762.

**Operators** (E2 T1 panel vs frozen-ER M64).
- Mirror-bin retention, source / receiver:

| Arm | Source | Receiver |
|---|---|---|
| T0 | 0.24 | 0.24 |
| TQ | 0.31 / 0.32 | 0.31 / 0.32 |
| U0 | 0.16 | 0.29 / 0.25 |
| UQ | 0.31 / 0.32 | 0.29 |

- Relative source − receiver operator difference on the (intrinsically single) panel material: U0 0.72 / 0.68, UQ 0.57 / 0.68 (tied arms 0 by construction).

**Renders** (inspected conditions, diagnostic, 2 × 4,096 spp, common random numbers).
- **Region-C E_H, E2 T1:** T0 0.076 / 0.042, TQ 0.046 / 0.035, UQ 0.056 / 0.041, U0 0.057 / 0.042, CQ 0.047 / 0.035.
- **Region-C E_H, E2 T0:** T0 0.067 / 0.049, TQ 0.056 / 0.046, UQ 0.066 / 0.051.
- **Region-C E_H, E5k995 T0:** T0 0.034 / 0.024, TQ 0.021 / 0.022, UQ 0.020 / 0.022.
- **Region-C E_H, E5k995 T1:** T0 0.032 / 0.025, TQ 0.031 / 0.023, UQ 0.042 / 0.028.
- **E1 LSPOT stationary (source class):** T0 0.026 / 0.037, TQ 0.021 / 0.031, UQ 0.020 / 0.033, **U0 0.044 / 0.047**. U0 fails the WL86 F4 bound of 0.04; UQ passes.
- **Gates and dynamics.** UQ: every WL86 gate passed by T0 is kept, F3 PASS on E2 / E5, no stationary / D regression beyond 1.25 × T0 + 0.002. Edit slope in region D: UQ 0.81 / 0.86 (E2), T0 0.87 / 0.91.

**Pattern diagnostic (LPAT 2 × 2 checker; visual only; cone class unseen in training).**
- Region-C E_H: T0 0.56 / 0.61, TQ 0.51 / 0.51, UQ 0.53 / 0.48, CQ 0.51 / 0.51, U0 0.61 / 0.59, WL92A own 0.59 / 0.61.
- Stationary E_H: 0.34–0.45 for every arm.
- **What the images show.** The reference's reflected spot appears on the left wall next to the panel as a localized bright blob. Every arm instead produces a vertical strip along the wall: the spot placement is lost, giving negative error at the spot and positive error below it.
- **Arm differences.** Small and not consistent across seeds.
- **Close-up limitation.** Region C has two separate components (left wall and sphere), so the bounding-box "close-up" is nearly full-frame.

## Observations

- **UQ keeps TQ's source-stage gain.**
  - Pooled SRC is within 2 SE of TQ on both seeds; better on 49 / 55 and 45 / 55 geometries.
  - Peak retention matches TQ.
  - UQ has the best receiver use on the physical first flight of all arms, closest to the exact frozen-ER floor.
- **UQ removes most of TQ's whole / single regression.**
  - 79 % / 87 % of the WHOLE gap is recovered, on every validation geometry.
  - UQ does not improve whole over T0; it reaches parity with it.
- **The capacity-matched control does not use its added parameters.**
  - The tiedcm arms (additive head with zero output initialisation) stay within 1e-5 of the tied arms on WHOLE after 2,000 steps.
  - So "UQ beats CQ" is, in practice, the same evidence as "UQ beats TQ" at this budget.
  - It shows that the parameter count alone does not explain the effect for this control form; it does not exclude a capacity effect under a control that actually uses its capacity.
- **Untying without supervision (U0) gives the best whole fit, through a non-physical decomposition.**
  - Its source stage moves away from the physical response, to worse than the WL92A init.
  - Its receiver use on the physical first flight degrades (0.12 / 0.18).
  - Its source / receiver operators diverge strongly.
  - It fails the cone-source gate.

## Interpretation

At smoke scale and on the validation population, source / receiver factor separation removes most of the whole-transport cost that source-stage supervision causes in the tied operator, while keeping the source-stage gain. By the frozen mapping this is a **SHARING_SIGNAL**, consistent across both seeds and every validation geometry. Four qualifications bound it:

1. **Capacity control.** The capacity-matched control was functionally inert at this budget, so capacity is ruled out only for this control form.
2. **No net whole gain.** The whole-transport result is a removal of the regression, not a gain over the unsupervised tied T0.
3. **Interpretation trade-off.** Untying lets the model fit whole transport by separating source and receiver roles. Without physical supervision it uses that freedom for a non-physical compensating decomposition (U0). With supervision (UQ) the source stage stays physical and the receiver use is the most accurate. Even so, the two learned panel operators differ by 57–68 %, which weakens the reading of a single shared intrinsic material response.
4. **Mixed renders.** Rendered sharp-region results are mixed (TQ best on E2 T1, UQ worse on E5 T1 seed 8601). The pattern diagnostic shows a shared failure to localize reflected spots from an unseen cone source; it does not separate the arms.

## Unresolved questions

- Would a capacity control that actually uses its capacity — for example symmetric-breaking copies, or a wider shared head — recover the UQ effect?
- Does the effect survive full-scale training (8,000 steps, 928 geometries) and an independently qualified held-out edit? E6 was never qualified.
- Why does UQ's source stage lag on g00010 while its whole improves?
- What is the trade-off between role separation and a shared intrinsic material operator, for ownership and reuse under K/C contracts?
- Why do all arms lose spot localisation under the cone pattern: source class, encoding, or local operator?
- Interaction with the frozen global term (never trained here), loss weights and other seeds is untested.

## Visuals (`results/evaluation/93B/`, manifest with hashes)

- `fig1_stage_accounting.png`: stage and integrated relative L1 per arm and seed, with retrained R_T0 / R_TQ marked.
- `fig2_per_geometry_differences.png`: per-geometry UQ − TQ, UQ − CQ and U0 − T0 for the source stage and whole; g00010 marked.
- `fig3_E2T1_sharp_maps.png`: E2 T1 reference / arm images and signed errors, one shared exposure and error scale.
- `fig4_pattern_LPAT.png`, `fig4b_pattern_LPAT_closeup.png`: LPAT pattern diagnostic (reference, tied, untied, capacity control, differences).
- `fig5_operators.png`: source / receiver panel mirror retention and source − receiver difference.

## Reproducibility

- **Code and protocol.** Freeze `03fbc84`.
- **Server.** `~/rest_x93b` (`chain.status`, `results/`: reuse, lpat, tests, account (with per-configuration sums), operators, final93b, models incl. R_*, tables, renders incl. REF_ER_E2_T1_LPAT_r0–7).
- **Inputs.** WL93A data via read-only links (`~/rest_x93a/data/{qref,train_q,val_q}`); WL86 / WL92A / WL92B inputs hashed in `chain.status`.
- **Local copies.** `results/ecro_stage93b/`, `results/evaluation/93B/`.
- **Seeds.** Training 8601 / 8602, data order rng(seed + 93), tiedcm head2 Generator(seed + 9300). Bootstrap 93002. Renders 86600 / 86601. LPAT reference 93810 + r.
- **Training.** WL93A contract: 128 geometries, 2,000 steps, 207–211 s per run with 4 concurrent.
- **Checkpoint state hashes.** In `models/*.json` (`final_sha256`); T0 / TQ equal WL93A.

## Final report (Korean, for the owner)

**Implementation Fact (구현 사실)**
- WL93 / WL93A 판정은 그대로 두고, 동결 `03fbc84`로 실행했다.
- WL93A T0 / TQ 체크포인트를 재사용하고 해시를 검증했다.
- 기존 WL93 구현으로 U0 / UQ(untied)와 C0 / CQ(용량 대조 tied)를 WL93A smoke 계약 그대로 학습했다: 128 기하, 2,000 step, 시드 2개, 같은 타깃과 순서, 전역 출력 동일.
- 재현 대조 R_T0 / R_TQ를 학습했다. WL93A 수치와 완전히 일치한다(학습이 결정적).
- 단계별 회계: source 사용, 물리 1차 비행 위의 receiver 사용, single, 동결 global, whole.
- 진단 렌더를 만들었다.
- 패턴 진단:
  - 광원 인터페이스가 최대 2개라서 체커보드를 2×2(대각선 두 셀을 콘 스폿으로 조명)로 축소했다.
  - 동결 전 첫 배치는 상자 B가 셀 하나를 가려 표현 가능성 검사에서 실패했다. 광원을 비스듬히 비추도록 옮겨 수정했다.

**Measurement (측정)**
- 검증 whole: T0 0.0762 / 0.0771, TQ 0.0803 / 0.0806, UQ 0.0770 / 0.0775, CQ 0.0802 / 0.0806, U0 0.0735 / 0.0737.
- source 단계: T0 0.899 / 0.927, TQ 0.738 / 0.790, UQ 0.737 / 0.808, U0 1.067 / 1.092.
- UQ − TQ whole: −0.0032 / −0.0031이고 CI가 0을 포함하지 않는다. TQ 회귀의 79 / 87 %를 회복했다.
- 판정: **SHARING_SIGNAL** (두 시드 모두 a–d 충족).

**Observation (관찰) — 질문별 답**
1. UQ는 TQ의 source 개선을 유지한다. 2SE 이내이고, 기하별로 49 / 55, 45 / 55에서 UQ가 우세하며, peak 보존도 같다. 물리 1차 비행 위의 receiver 오차는 UQ가 전 처리군 중 가장 낮다.
2. UQ는 TQ의 whole / single 회귀를 대부분 제거한다(55 / 55 기하). 다만 T0보다 좋지는 않고 동등 수준이다.
3. CQ 대비 우위는 있다. 그러나 CQ는 TQ와 1e-4 이내로 같다. 0-초기화 추가 헤드가 사실상 쓰이지 않았으므로, 용량 설명은 이 형태의 대조에 한해서만 배제된다.
4. 두 시드에서 일관되고, whole은 모든 검증 기하에서 일관된다. source 단계의 시드 8602 풀링 차이(+0.018)는 일부 고질량 기하 때문이다.
5. g00010: UQ가 whole 회귀를 줄이지만(0.290 / 0.330 vs TQ 0.319 / 0.351), source 단계는 TQ와 T0보다 나쁘다. 모든 처리군에서 큰 오차가 남는 기하이며, source 기준 자체도 불확실하다(B5 잡음 0.66).
6. 패턴 진단(학습에 없던 콘 광원): 모든 처리군이 반사 스폿의 위치를 잃고 세로 띠로 번진다. 처리군 간 차이는 작고 시드 간에 일관되지 않는다. 의미 있는 차이는 드러나지 않았다.
7. 미시험 항목은 Unresolved questions 절에 정리했다(대규모 학습, held-out, 실제로 용량을 쓰는 대조, 손실 가중치, 전역 항과의 상호작용 등).
- 추가: U0(감독 없는 untied)는 whole이 가장 좋다. 그러나 source가 초기보다 나빠지고 receiver도 악화하며 두 연산자가 크게 갈라진다. 콘 광원 게이트 F4도 실패한다. 비물리적 보상 분해다.

**Interpretation (해석)**
- smoke 규모에서 factor 분리는 source 감독의 whole 비용을 대부분 없애면서 source 이득을 유지한다(SHARING_SIGNAL).
- 단서는 네 가지다.
  - 용량 대조가 실질적으로 작동하지 않았다.
  - whole에서 순이득은 없다(T0와 동등).
  - 역할을 분리하면 공유된 고유 재질 연산자라는 해석이 약해진다(두 패널 연산자가 57–68 % 다르다).
  - 렌더 결과는 섞여 있다.
- 이 결과는 독립적으로 자격을 갖춘 대규모 실험을 검토할 근거일 뿐이며, 그 실험을 승인하지 않는다. 결정은 소유자에게 있다.

**Unresolved Question (미해결 질문)**
- 실제로 용량을 쓰는 대조에서도 효과가 남는가?
- 대규모 학습과 held-out에서 재현되는가?
- 역할 분리와 공유 재질 연산자(K / C 소유 계약) 사이의 상충은 어떻게 정리되는가?
- 콘 패턴에서 스폿 위치를 잃는 원인은 무엇인가?
- g00010에서 source 단계가 뒤처지는 이유는 무엇인가?

**Architecture Verdict (아키텍처 판정)**
- **SHARING_SIGNAL** — smoke 규모 feasibility 평가일 뿐이며, 아키텍처 검증이나 채택이 아니다. 후속 배치는 없다.

**Reproducibility (재현성)**
- 동결 `03fbc84`, LabServer63 `~/rest_x93b`, 31분.
- 로컬 사본: `results/ecro_stage93b/`, `results/evaluation/93B/`.
- 시드, 해시, 입력 해시는 위 Reproducibility 절을 참조한다.
