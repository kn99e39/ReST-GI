# Worklog — GM1 Source-independent Directional Relation Qualification and Architecture Discrimination (2026-11-12)

## Origin and scope

- **Brief.** "ReST-GI — GM1 Source-independent Directional Relation Qualification and Architecture Discrimination" (user, 2026-11-12). One bounded batch.
- **Aim.** Preserve the WL88 negative result. On a fresh, independently qualified moving-sharp workload, test whether a source-independent sender-side relation (C3UR) gives physically accurate and economically justified dynamic sharp multi-bounce transport, relative to R0 (WL87 EXP) and R1 (WL88 SAM).
- **Hypotheses (prospective).**
  - H0: the E2 sender-side advantage does not survive.
  - H1: C3UR improves sharp fidelity or time-to-quality without source-specific reconstruction.
  - H2: relation construction / update cost outweighs the benefit.
- **Preserved and unchanged.**
  - WL86 / WL87 / WL88 code, protocols, results and verdicts. They are imported, never edited.
  - WL88 stays STOPPED_WORKLOAD_UNQUALIFIED. Wc / Wh / Wi are not used; Wi is not promoted.
  - Frozen ER. R0 = WL87 EXP.
  - R1 / R1M = WL88 SAM with the WL88 relation parameters, bit-for-bit (no tuning).
- **Not done.** No learned global model, no H-P / subspace, no GM2, no new branch or worktree.
- **Where.**
  - Branch `Selective_Recompute`, LabServer63.
  - `experiments/gm1_relation89/`, `results/gm1_relation89/`, `results/evaluation/89/`.
  - Server `~/rest_rel89`; it reads `~/rest_gm187` (WL87 tables, integration test).
- **Terminology.** All comparators are physical recomputations with the same frozen ER (path tracing, table fill, explicit relations). No external Neural Full Recompute claim is made.

## Design evidence from WL88 and the qualification contract

**WL88 Q5 inconsistency (reviewed).**
- WL88 measured the Q5 SNR floor on its 4-replica qualification partition.
- Q7 and Q8 projected the noise to the 8-replica final references, and WL86 qualified with 8 replicas.
- WL89 removes the inconsistency differently: the qualification partition has the **same budget as the final evaluation references** (8 × 8,192 spp per T, family-tagged REF-ER, seeds 89140+).
- Every floor (half-difference of the replica halves) is therefore measured at the evaluation budget, and "change ≥ 4 × floor" directly means "resolvable in the evaluation".

**Thresholds.**
- The thresholds are the WL86 / WL88 values, unchanged.
- C_min 0.02 is one third of the F2 threshold 0.06: the smallest sharp-region change whose mis-tracking can move E_H across F2.
- The family-share minimum 0.25 requires the targeted S1 → S1 (C_ss) and moving-S1 (R_mirror) families to carry a material part of the change of the higher-order radiance.
- Q8 requires the reference noise in the sharp regions to be ≤ F2 / 3.

**Fresh workload.**
- WL88's candidates showed that the sharp families carried most of the change but the absolute changes were small.
- WL89 uses a larger glossy S1 slab (half 0.05 × 0.55 × 0.50 m) with fresh poses and lights.

**Screening partition (disclosed).**
- Family-tagged REF-ER, 2 × 1,024 spp per T, seeds 89900–89901, on seven configurations (V1–V6, X1). Masks are geometry-only.
- Results:
  - V4 is geometry-inadmissible.
  - V5 and V6 have R_mirror < 1,000 pixels.
  - X1 (held-out edit) fails `geom_ok` at T1 (too close to box B). It is replaced by **X2**, checked by geometry only, with no radiance.
- Projected final SNRs (this partition, crude two-replica floors):

  | Candidate | C_ss change | C_ss snr | R_mirror | D higher-order snr | H2 share |
  |---|---|---|---|---|---|
  | V3 | 0.027 | 11 | 3,244 px | 8.7 | 0.81 |
  | V2 | 0.029 | — | 3,229 px | 8.5 | — |
  | V1 | 0.022 | — | — | 8.4 | — |

- The screening partition was used only to choose and order the candidates (V3, V2, V1; at most three) and is never reused.
- No relation or method output was computed on any candidate before the freeze.

## Candidates and controls (frozen in `protocol/gm1_relation89.json`)

**Candidates.**

| Arm | Construction | Unique traced rays per build | Role |
|---|---|---|---|
| R0 | WL87 EXP: receiver gather on the deployed 512-ray state | 650,240 | preserved baseline |
| **C3UR** | 8 rays per (sender patch, outgoing bin) cell, cos-weighted in the bin, deposited at the receiver's WL87 dense incident cell; one relation reused for all orders | 650,240 | primary new candidate; geometry-only |
| R1 | WL88 SAM | — | frozen comparator |
| R1M | WL88 SAM, matched-ray variant | — | matched-budget comparator |

C3UR details:
- Per-ray stream keyed by the global ray id, so a selective update (rays starting on the mover, or whose segment meets the old / new mover volume) is bit-identical to a rebuild.
- Application aggregates incident radiance per (receiver, cell) and scatters per material by one contraction. This is an implementation choice; test T_C3UR_FORM checks it equals the per-ray form.
- Sphere receivers use the flat patch-centre frame of the dense table, as in WL87 C3 (declared).

**Budget points.**
- 1× unique rays: R0 / C3UR / R1M.
- 2×: POOL2 (R0) and C3UR16 (C3UR with 16 rays per cell).
- R1 is reported at its own budget.

**Controls.**
- C3: WL87 power-proportional, fresh per order, source-dependent diagnostic.
- C3U: sender-uniform, fresh per order.
- Converged relations for target identity: R0_INF (64 realisations), R1_INF (16 × m 16), C3UR_INF (4 × 64 rays per cell).
- RS0 / RS1: **repaired** RAYSEND. Exact radiance is traced from offset origins and validated before use: table-radiance mode must reproduce the first gather; exact traces must hit the recorded sender ≥ 99 %.
- ORACLE: path-traced table, 4,096 spp per slot.
- PT, TF256 / TF1024: physical full-path and table-fill comparators.
- REF-P: ER approximation, reported separately.

## Prospective rules (summary)

- **Stage 0.**
  - Geometry screening, then the qualification partition, with candidates in order. None admitted: WORKLOAD_UNQUALIFIED; no candidate is evaluated and there is no rescue.
  - tests89 = the WL88 tests on the WL89 W1, plus the C3UR and RAYSEND tests. Any failure: STOPPED_INVALID_COMPARISON.
- **Stage 1 (W1 T0 / T1).**
  - E_H and signed error per region; WL86 F1–F3 thresholds, plus F2 / F3 on the sharp regions.
  - Field and per-order errors against path-traced order fields.
  - Validity gate: tests, finite / non-negative tables, target identity (signed ≤ 0.01 + noise, flux within 3 %), RAYSEND repair. A failure stops the batch before economics.
  - Error decomposition per sharp region: sampling (deployed − converged), relation discretisation (converged − oracle), output representation (oracle − reference), first-gather sender projection (arm − RS).
  - Energy-weighted coverage and missing important relations for R0 / C3UR / R1.
- **Stage 2.**
  - 16 realisations each of R0, C3UR and R1: paired differences with 95 % intervals and pass fractions.
  - Held-out edit W2 = X2 and held-out sources LMIX89 / LSPOT89, with relations reused (F4, F2 on sharp regions ≥ 1,000 px, F3 on W2).
  - Edit and source-reuse validity.
- **Stage 3 (alone on the GPU).**
  - Per-edit lifecycle stages A–G, warm, compilation excluded.
  - Source-only update and multi-source totals; edit-to-first-frame and sustained-frame cost at matched quality (ladders 4–16,384 spp × 3; PT, R0, C3UR, R1, R1M, C3, TF).
  - Memory.
  - Scaling in N (s = 1 / 1.5 / 2), peaked coverage (second glossy box), order (1 / 2 / 4 / 8) and source count (1 / 4 / 16).
- **Verdict mapping.**
  - H0 / H1 / H2 and the item verdicts are computed by `analyze89.verdict`, which also attributes failures.
  - Sharp improvement means the paired C3UR − R0 interval is < 0 in a sharp region with no region > 0.
  - Quality-matched advantage means ≥ 1.25× faster edit-to-first-frame at a sharp level, or reaching a level R0 does not.

## Pre-freeze development (non-evidence; `~/rest_rel89dev`)

1. **Screening.** As above. Seven parallel screening processes first exhausted the GPU and were restarted as two processes; no data was lost or reused.
2. **Dev chain end to end** (`WL89_DEV=1`, W1 = W2 = the dev geometry Wk, 64 spp): every stage completed, from geometry screening through qualification, tests, tables, realisations, renders, stage-1 validity gate, economics, scaling, analysis, verdict and visuals.
   - Dev tests: all WL88 tests plus the C3UR tests and the repaired RAYSEND pass.
     - Aggregated contraction vs per-ray form 3e-7.
     - Known-field flux ratio 0.993.
     - Selective = full, null edit and move-and-return identical.
     - RAYSEND table mode 4e-7; exact traces hit the recorded sender 99.9996 %.
   - Dev numbers are noise-dominated and are not evidence.
3. **Implementation optimisation of the C3UR update** (same operator; digests unchanged, selective = full re-verified). Ray records stay GPU-resident and the coefficients (landed rays, slots, weights) are computed in torch.
   - Warm C3UR update on Wk: 53 → 14 ms.
   - R0 state + coefficients: 28 ms. R1 update: 124 ms.

## Evidence run, part 1 (freeze `7515a29`; LabServer63 `~/rest_rel89`; 2026-11-12 03:55–04:35 UTC)

| Step | Outcome |
|---|---|
| Geometry screening | V3, V2, V1 admissible |
| Qualification of V3 (16 renders, 8 × 8,192 spp per T, seeds 89140+) | **ADMITTED**, all of Q1–Q8 |
| W1 = V3; evaluation references, oracle fields, order fields | complete: REF-ER W1 2 × 8, W2 2 × 8, held-out sources 2 × 4, REF-P 4; sha256 in `results/amend1/reference_manifest.json` |
| tests | **GATE TESTS FAIL → `STOPPED_INVALID_COMPARISON`** (04:35:33) |
| verdict in the stop branch | crashed (analysis bug: it expected stage-1 output) → `CHAIN_FAILED` |

**V3 qualification values.**

| Criterion | Value |
|---|---|
| Q1 A | 0.426 |
| Q2 Bset | 0.0145 |
| Q3 C_ss | 0.0234 (13,466 px; H2 share 1.08) |
| Q4 R_mirror | 0.134 (3,244 px; G share 1.43) |
| Q5 D | 0.0153, higher-order ratio 0.45, higher-order SNR 5.5 |
| Q6 H share | 0.25 |
| Q7 uncertainty | 0.0025 |
| Q8 sharp reference noise | 0.0103 |

- The family "shares" exceed 1 because the predeclared share is Σ|Δfamily| / Σ|Δh|, and family changes partly cancel inside h.
- Candidates V2 and V1 were never rendered.

**The single failing gate.** WL88's frozen `tests88.T_EDIT`, `selective_equals_full`, for R1 (SAM) on V3. Every other WL88 test passed, as did every WL89 test (C3UR and repaired RAYSEND).

**No candidate-quality evidence existed or was inspected at the stop.** `results/tables` was empty; no relation table or candidate render had been computed. The original failure artifacts are preserved in `results/orig_stop/`: chain status, both test JSONs, the logs and the freeze commit.

## Failure attribution and the human decision

**First diagnosis (this session, before the human decision).**
- Exactly 2 R1 area samples differ, in `dbin` only, and both are invalid.
- I attributed this to undefined data in inactive lanes. **That attribution was wrong**; see below.
- I asked for a decision, and the user approved option 2 conditionally (brief "Controlled Post-freeze Correctness Amendment and Conditional Evidence Chain Resumption"): amend only if code- and execution-level evidence proves a false positive from non-semantic invalid-lane data, otherwise stop.

**Code-level audit.** These are the read paths of the area-sample arrays:

| Read path | What it reads |
|---|---|
| `rel88.SAM._finish` (`rel88.py:301-310`) | builds every operator tensor (`a_recv`, `a_idx`, `a_w`, `a_cell`, groups) from **valid** samples only |
| `SAM.update` | copies stored arrays by index without reading values; re-traces dirty samples |
| `SAM.nbytes` | sizes only |
| `SAM.digest` | **the only read of invalid entries** |
| RAYSEND (`run88.raysend`, `rel89.raysend_fixed`) | masks by validity |
| coverage (`run88` / `run89.real`) | masks by validity |
| `run88.pair_energy` | a separate diagnostic trace; invalid entries are multiplied by an exact 0 |

The u-ray part (state, MIS factors) does not read area samples at all.

**Execution-level audit** (`audit89.py`; `results/amend1/audit89.json`; frozen evidence parameters).
1. **The two entries.** Pairs (206 floor ← 1260 sphere) and (1053 box B ← 1268 sphere), sample 2.
   - Both are invalid with weight 0 in the selective and the full relation; `cell` and segment are identical; only `dbin` differs (37 vs 51, 54 vs 53).
   - Both are **traced** grazing rays (sender cosine 0.014 / 0.0007) that miss the sender patch.
   - The stored `dbin` is the bin at the first surface hit **beyond the sample segment**:
     - segment lengths 1.02 / 1.35 m (`segment_check.txt`);
     - at T0 the first hit is the back wall (t 4.94 / 4.80 m);
     - at T1 it is the moved slab (t 2.68 / 2.66 m).
   - Validity ("first hit is the sender at y") depends only on geometry inside [x, y]: a flat sender plane is crossed once, at y, and a convex sphere is met only near y. So changes beyond y cannot change validity or contribution, and the segment-based dirty rule correctly did not flag these samples.
2. **Operational state.** The operational digest, every operator tensor, the u state, all segments and the validity mask are bitwise equal between selective and full.
3. **Transport outputs.** Deterministic CPU evaluation of the unchanged operator, selective vs full, for a random field, the W1 q and the 4-order recurrence: bitwise equal, finite, and self-repeat equal.
   - The GPU recurrence differs by 1.6e-7 relative, within its own repeat floor of 3.2e-7: GPU `index_add_` is atomic.
4. **Null edit and move-and-return.** Equal under both digests.
5. **Invalid-lane perturbation** of all 157,538 invalid entries, three ways:
   - `dbin` random;
   - `dbin` 255 and `cell` int32 max;
   - `w` NaN and `cell` random.

   In all three, the operator tensors, the CPU outputs (finite) and a subsequent selective edit T1 → T0 are bitwise unchanged against an unperturbed twin.
   - The first audit run had a test bug: it compared against a relation built on a different scene object, and a fresh `Ctx(T1)` differs from a `set_geom(T0 → T1)` scene in mesh-vertex bits. It is preserved as `audit89_run1_testbug.json`.

Conclusion: the STOP was caused solely by a test that hashed non-semantic stored values of invalid samples. **AUDIT PASS.**

## POST-FREEZE AMENDMENT 1 (protocol `amendments[0]`)

| | Predicate |
|---|---|
| **Old** | `tests88.T_EDIT`: equality of `rel88.SAM.digest()`, a sha256 over the u state, MIS factors, pair list and the **full** arrays valid / dbin / cell / w of all area samples |
| **New** | `tests89.T_R1_EDIT_OPERATIONAL`: equality of `audit89.op_digest`, a sha256 over the u state (hit, dbin, segments), MIS factors, pair list, validity mask, dbin / cell / w of **valid** samples and all area segments; plus bitwise equality of every operator tensor, and null edit / move-and-return under the same digest. Bitwise, no tolerance |

- This changes the test implementation, not transport semantics. The excluded values are never read by the operator, and perturbing them changes nothing.
- `tests88.T_EDIT` still runs and its FAIL is reported. Every other gate is unchanged.
- Also committed: the stop-branch verdict crash fix (`analyze89.verdict`, analysis only).
- Unchanged: candidates, sampling, MIS, material rule, workloads, qualification, references, seeds, thresholds and evaluation.
- **Resume point.** The chain resumes with RESUME at the Stage 0 test gate. Qualification, references and truth fields are reused (sha256 manifest).

## Evidence run, part 2 (amendment `5d6bb5c`; resumed 2026-11-12 06:01 UTC; `CHAIN_DONE` 06:36:46)

| Stage | Wall | Mean GPU | Outcome |
|---|---|---|---|
| Stage 0 test gate (re-run) | 1.5 min | 3 % | **GATE TESTS PASS** (tests88.T_EDIT still FAIL, reported; T_R1_EDIT_OPERATIONAL PASS, strict-invalid entries differing 2) |
| tables (6) + realisations (W1 T0 / T1) | 4.8 min | 9 % | finite, non-negative |
| renders (~310, 4,096 spp) | 14.1 min | 89 % | — |
| stage 1 (validity gate) | 0.4 min | — | tests, finite / non-negative, **target identity**, **RAYSEND repair**: all true |
| economics (alone) | 12.4 min | 36 % | measured; no foreign GPU process before or after |
| scaling (alone) | 1.0 min | 21 % | measured; **update points include first-call compilation** (see defects) |
| stage 2 / 3, verdict, visuals | 0.3 min | — | — |

- The part-1 GPU trace (03:55–04:35) was overwritten when the watcher restarted; it is not available. Part 2: `gpu_util_trace_part2.txt`.
- The table and realisation stages are CPU-bound (numpy / Python relation bookkeeping), at 9 % GPU.

**Measurement defects.**
- `econ89` checked R1 selective = full with the strict WL88 digest, so its `selective_equals_full.R1 = false` is the same non-semantic artefact. R1's operational equivalence comes from tests89 and the audit.
- `econ89.scale` timed the edit update without a warm-up call, so the scaling *update* times (C3UR / R1 / R0) include first-call compilation and are invalid as warm measures. The scaling *build* and *application* times are warm.

## Final report (Korean)

### Implementation Fact

- **A. 원래 frozen test 결과 (보존).**
  - 2026-11-12 04:35:33 UTC에 `GATE TESTS FAIL → STOPPED_INVALID_COMPARISON`이 났습니다(`results/orig_stop/`).
  - 실패 항목은 WL88 frozen test `tests88.T_EDIT`(R1 selective = full) 하나였습니다. 그 시점에는 candidate 품질 결과가 하나도 없었습니다(tables 비어 있음).
  - stop 분기 verdict가 crash해 `CHAIN_FAILED`가 됐습니다.
- **B. Amendment 근거와 범위.**
  - 사용자가 조건부로 승인했습니다.
  - 코드 감사: 무효 sample 값을 읽는 곳은 `SAM.digest` 하나뿐이고, 연산 경로는 모두 valid mask를 거칩니다.
  - 실행 감사(`audit89.json`):
    - 차이는 무효 grazing sample 2개의 저장 bin뿐이었습니다. 이 bin은 segment(1.02 / 1.35 m) 너머 표면의 값입니다(T0 벽 4.9 m, T1 움직인 slab 2.7 m).
    - operator tensor와 CPU 결정론적 출력이 bit 단위로 같았습니다.
    - 무효 entry 157,538개에 random / 255 / NaN 교란을 넣어도 출력과 이후 update가 불변이었습니다.
  - 그래서 판정 기준만 "연산에 쓰이는 state의 bit 단위 동일성"으로 바꿨습니다(POST-FREEZE AMENDMENT 1, `5d6bb5c`). 원래 FAIL은 계속 실행하고 보고합니다.
  - 후보, 파라미터, workload, reference, 임계값, 평가는 바꾸지 않았습니다.
  - 정정 사항이 두 가지 있습니다.
    - 제 첫 진단("비활성 lane 쓰레기 값")은 틀렸고, 실행 감사로 바로잡았습니다.
    - 감사 1차 실행에는 test 버그가 있었습니다(다른 scene 객체끼리 비교). 그 결과는 보존했습니다.
- **C. 정정된 정확성 결과.**
  - T_R1_EDIT_OPERATIONAL PASS(null edit, selective = full, move-and-return). 그 밖의 WL88 / WL89 test도 모두 PASS입니다.
  - stage-1 validity 4개 항목이 모두 true입니다.
    - target identity: R0_INF / R1_INF / C3UR_INF의 region signed 차이가 noise 이내입니다. W1 T1 R_mirror 수렴 E_H는 0.038 / 0.038 / 0.038입니다.
    - 수리한 RAYSEND: table 모드 오차 ≤ 1e-5, exact trace의 sender 적중률 ≥ 99 %.
- **D. 구현 사실.**
  - C3UR은 (sender patch, outgoing bin)마다 8 ray를 bin 안에서 cos 가중으로 쏘고, receiver의 dense incident cell에 deposit합니다. relation 하나를 모든 order에 재사용합니다.
  - geometry와 재질만으로 만들고, ray별 key stream으로 selective update가 full rebuild와 bit 단위로 같습니다. 추적 ray는 R0와 같은 650,240개입니다.
  - R0(WL87 EXP)와 R1 / R1M(WL88 SAM)은 바꾸지 않았습니다. 추적 ray: R1 1.05 M, R1M 0.73 M.

### Measurement

**Workload.** V3(큰 S1 slab, yaw −0.10 → 0.45)가 독립 qualification partition(T마다 8 × 8,192 spp)에서 Q1–Q8을 모두 통과했습니다.

| 항목 | 값 |
|---|---|
| C_ss 변화 | 0.023 |
| R_mirror 변화 | 0.134 |
| D 고차 SNR | 5.5 |
| sharp reference noise | 0.010 |

**W1 fidelity (deployed draw, T1, 4,096 spp, common random numbers).**

| 방법 | stationary | C_ss | R_mirror | D |
|---|---|---|---|---|
| R0 | 0.0047 | 0.0109 | 0.0598 | 0.0043 |
| C3UR | 0.0039 | 0.0080 | 0.0416 | 0.0033 |
| R1 | 0.0043 | 0.0093 | 0.0527 | 0.0038 |
| R1M | 0.0052 | 0.0106 | 0.0576 | 0.0045 |
| C3 | 0.0035 | 0.0074 | 0.0366 | 0.0031 |
| ORACLE | 0.0028 | 0.0059 | 0.0137 | 0.0026 |

- 모든 arm이 W1의 F1 / F2 / F3를 통과합니다. R0의 R_mirror는 기준 0.06에 0.0002 차이로 통과했습니다.

**16개 독립 relation draw (W1 T1).**

| 방법 | R_mirror E_H (pass) | C_ss E_H | stationary E_H |
|---|---|---|---|
| R0 | 0.047 ± 0.015 (81 %) | 0.0105 ± 0.0027 | 0.0056 ± 0.0015 |
| C3UR | 0.040 ± 0.005 (100 %) | 0.0079 ± 0.0005 | 0.0040 ± 0.0001 |
| R1 | 0.040 ± 0.009 (100 %) | 0.0097 ± 0.0025 | — |

- 모든 region 통과율: R0 81 %, C3UR 100 %, R1 100 %.

**Paired 차이 (95 % 구간).**

| 비교 | T1 | T0 |
|---|---|---|
| C3UR − R0, C_ss | −0.0026 [−0.0041, −0.0010] | −0.0020 [−0.0032, −0.0007] |
| C3UR − R0, R_mirror | −0.0077 [−0.0167, +0.0013] (0 포함) | −0.0067 [−0.0128, −0.0006] |
| C3UR − R0, stationary | −0.0016 [−0.0024, −0.0009] | — |
| C3UR − R0, D | −0.0017 [−0.0025, −0.0009] | — |
| C3UR − R1, C_ss | −0.0018 (구간 < 0) | — |
| C3UR − R1, R_mirror | −0.0003 [−0.0068, +0.0062] | — |

**오차 분해 (W1 T1 R_mirror, signed; noise 0.014).**

| 성분 | R0 | R1 | C3UR |
|---|---|---|---|
| sampling (deployed − converged) | −0.031 | −0.018 | −0.001 |
| relation discretisation (converged − ORACLE) | −0.024 | −0.025 | −0.025 |
| first-gather sender projection | −0.049 | −0.023 | — |
| output representation (ORACLE − REF) | −0.005 (공통) | | |

**Relation coverage (17 draws).** C3UR의 이점은 pair coverage가 아닙니다.

| 방법 | 중요 relation 누락 에너지 | 에너지 coverage |
|---|---|---|
| R0 | 0.28 | 0.71 |
| C3UR | 0.21 | 0.78 |
| R1 | 0.003 | 0.996 |

**Edit response.**
- W1: 모든 arm이 F3를 통과합니다. R_mirror slope: R0 1.12, C3UR 0.86, R1 0.82.
- W2(held-out edit), C_ss: C3UR 0.79 / 0.93, C3 0.80 / 0.95, ORACLE 0.87 / 0.98은 PASS입니다. R0 0.67 / 0.88, R1 0.71 / 0.92, R1M 0.66 / 0.89는 slope가 0.75 미만이라 **FAIL**입니다.
- W2 R_mirror는 resolvable하지 않습니다.
- W2 T1 R_mirror E_H: R0 0.051, C3UR 0.038, R1 0.044.

**Held-out sources (relation 재사용, 단일 draw).**
- 모든 arm이 F4와 sharp F2를 통과합니다.
- LMIX89 R_mirror: C3UR 0.023 대 R0 0.017로, C3UR이 더 나쁩니다.
- LSPOT89 R_mirror: 0.038 대 0.037.

**경제성 (RTX 3080 Ti 단독, warm median).**

| per-edit 비용 (렌더 전) | ms |
|---|---|
| R0 | 92 |
| C3UR | 86 |
| R1 | 192 |
| C3 | 510 |
| PT | 36 |
| TF256 | 322 |
| TF1024 | 1,183 |

- C3UR과 R0의 per-edit 구성:

  | 단계 | C3UR | R0 |
  |---|---|---|
  | relation update | 14.3 ms | 26.6 ms (discovery 20.3 + coefficient 6.3) |
  | 적용 (4 order) | 11.6 ms | 6.1 ms |
  | geometry | 31.8 ms | 31.8 ms |
  | q | 26.6 ms | 26.6 ms |
  | 전체에서 relation update 몫 | 17 % | 29 % |

- R1의 relation update는 125 ms로 per-edit의 65 %입니다.
- source-only 갱신: R0 32, R1 34, C3UR 38, C3 451 ms.
- quality-matched edit-to-first-frame, C3UR / R0:
  - 대부분의 수준에서 1.03–1.12×입니다(stationary 0.10–0.03, C_ss 0.10 / 0.06, R_mirror 0.10, D).
  - 예외는 R_mirror 0.06입니다. R0 deployed draw가 14,432 spp(12.95 s)를 쓰는 반면 C3UR은 347 spp(0.39 s)로, **33×** 차이입니다.
- PT / C3UR: stationary 1.7–5.2×, R_mirror 18×.
- 품질 floor(ladder 최소): R_mirror R0 0.060, C3UR 0.042, R1 0.053, PT 0.024.
- memory: C3UR 20.8 MB, R0 17.6 MB, R1 35.4 MB.

**Scaling (s = 1 / 1.5 / 2, N 1,270–5,080; warm build / apply).**
- C3UR build ∝ N^1.03, bytes ∝ N^1.0, 4-order apply ∝ N^0.38.
- R1 build ∝ N^2.15, pair 수 ∝ N^1.99.
- R0 state ∝ N^0.4.
- glossy B를 추가해도 같은 경향입니다.
- update scaling 값은 첫 호출 컴파일이 섞인 측정 결함이라 무효입니다.

### Observation

1. C3UR의 가장 뚜렷한 측정 효과는 **realization 분산 감소**입니다.
   - E_H SD: stationary 0.0001 대 0.0015, C_ss 0.0005 대 0.0027, R_mirror 0.005 대 0.015.
   - deployed draw의 sampling 성분(−0.001)이 수렴 relation과 거의 같습니다.
2. 평균 fidelity는 C3UR, R1, C3가 같은 수렴 floor(R_mirror 0.038)에 모이고, ORACLE(0.014)과의 간격은 모든 relation이 공유합니다.
3. W2 held-out edit의 C_ss edit response는 C3UR(와 C3, ORACLE)만 F3를 통과하고, 수신측 R0 / R1 / R1M은 실패합니다(단일 draw).
4. 오차 지도(fig1)에서 C3UR의 오차는 방의 모서리, 즉 수직 patch가 만나는 곳에 집중되어 보입니다. R0는 R_mirror 영역 안에 집중됩니다.
5. LMIX 광원의 R_mirror에서 C3UR이 R0보다 나쁩니다(0.023 대 0.017, 단일 draw).
6. 33× quality-matched 이득은 R_mirror 0.06 한 수준에서만 나옵니다. 그 수준이 R0 deployed draw의 bias floor(0.0598)와 거의 같습니다.

### Interpretation

- **Relation 발견 (coverage).**
  - C3UR은 중요 relation을 R0보다 조금 덜 놓칠 뿐입니다(0.21 대 0.28).
  - 그런데도 이득이 있는 것은 sender cell마다 ray를 균등하게 할당해, 밝은 sender cell의 에너지가 항상 전달되기 때문으로 추론됩니다. 그래서 분산이 작습니다.
  - 이것은 WL88 E2 Task A의 "sender 배치" 귀속과 일치합니다(추론).
- **표현 한계.**
  - 남은 sharp 오차는 relation discretisation이고, 모든 구성이 공유합니다: 수렴 relation 0.038, ORACLE 0.014.
  - 따라서 relation을 더 촘촘히 표본해도 이 workload의 R_mirror를 ORACLE 수준으로 낮출 수 없습니다. 중간 order의 patch × bin 표현이 그 바닥을 만듭니다.
- **Quadrature와 sampling.**
  - R0와 R1의 deployed 오차 중 sampling 몫이 큽니다(R_mirror −0.031 / −0.018).
  - R1은 coverage를 거의 채우지만, 분산은 C3UR보다 큽니다(R_mirror SD 0.009 대 0.005).
- **Source 의존성.**
  - C3UR은 source 없이 만들어지고, 3개 광원 조건에서 digest가 변하지 않았습니다.
  - held-out 광원에서 gate는 통과했지만, LMIX의 R_mirror는 R0보다 나빴습니다(단일 draw). source 분포에 따른 sender 표본 효율 차이일 수 있으나, 분리하지 않았습니다.
- **비용.**
  - C3UR의 per-edit 비용은 R0보다 7 % 작습니다(update는 0.54×, 적용은 1.9×).
  - 두 방법 모두 per-edit 비용은 geometry 갱신(32 ms)과 q(27 ms)가 지배합니다. relation 구성과 반복 적용은 지배적이지 않습니다.
  - R1은 relation 구성이 지배하고(65 %), 규모에 따라 거의 제곱으로 늘어납니다.
  - quality-matched 이점은 R0의 sampling floor가 걸리는 sharp 수준에서만 1.25×를 넘습니다. 나머지 수준에서는 같은 비용입니다.
- **H 판정.**
  - 사전 mapping으로 **H1 SUPPORTED**입니다. 근거는 C_ss paired 개선(구간 < 0, 모든 region에서 악화 없음)과 R_mirror 0.06 수준의 quality-matched 이득이고, sharp 손실은 없습니다.
  - 다만 C_ss 개선의 절대값은 작고(0.0026, 모든 arm이 기준을 크게 통과하는 영역), T1의 R_mirror 개선 구간은 0을 포함합니다.
  - quality-matched 이득은 R0 deployed draw의 floor 근접에서 나옵니다.
  - 따라서 이 결과는 "같은 비용에서 C3UR이 더 안정적(저분산)이고 같거나 나은 정확도를 낸다"로 읽는 것이 측정에 맞습니다. "큰 정확도 / 비용 우위"는 측정되지 않았습니다.

### Unresolved Question

- 남은 sharp 오차(R_mirror 0.038 vs ORACLE 0.014)는 relation discretisation(중간 order 표현)의 몫입니다. relation 구성으로는 줄지 않습니다.
- LMIX에서 C3UR이 R_mirror로 R0보다 나쁜 이유를 분리하지 않았습니다(단일 draw).
- C3UR 오차가 방 모서리(수직 patch 접합부)에 집중되는 원인은 추론뿐입니다. dense cell-centre deposition과 grazing 입사가 후보입니다.
- W2의 C_ss F3 결과는 단일 draw입니다. realization 분포로 확인하지 않았습니다.
- workload는 단일 장면 family이고 edit 유형도 하나(G 회전, 그리고 W2 평행이동+회전)라, 일반화 범위가 좁습니다.
- scaling update 시간은 측정 결함으로 무효입니다.

### Architecture Verdict

범위: qualified W1 = V3(회전하는 S1 slab), held-out edit W2 = X2, held-out 광원 LMIX89 / LSPOT89, frozen ER, 512², patch × 64-bin table.

| 항목 | 판정 |
|---|---|
| WORKLOAD QUALIFICATION | **SUPPORTED** (V3, 독립 partition, Q1–Q8) |
| EXPERIMENTAL VALIDITY | **SUPPORTED**. POST-FREEZE AMENDMENT 1 이후 기준. 원래 frozen 결과는 STOPPED_INVALID_COMPARISON이며 보존 |
| C3UR PHYSICAL FIDELITY | **SUPPORTED** (W1 F1–F3, W2, held-out 광원 gate) |
| R0 / R1 PHYSICAL FIDELITY | **CONDITIONALLY SUPPORTED** (W2 C_ss F3 실패) |
| C3UR SAMPLING / RELATION QUALITY | **SUPPORTED** (C_ss paired 개선, 악화 region 없음, 통과율 100 %). R_mirror T1 개선 구간은 0 포함 |
| SOURCE-INDEPENDENT REUSE | **SUPPORTED** (구성에 source 없음, digest 불변, held-out 광원 gate 통과). LMIX R_mirror는 R0보다 나쁨 |
| DYNAMIC EDIT VALIDITY | **SUPPORTED** (C3UR null / selective = full / move-and-return 동일, W1 / W2 F3) |
| RELATION CONSTRUCTION ECONOMICS | **SUPPORTED** (C3UR / R0 update 0.54×) |
| OPERATOR APPLICATION ECONOMICS | **CONDITIONALLY SUPPORTED** (1.9×; 절대 11.6 ms) |
| QUALITY-MATCHED ADVANTAGE | **SUPPORTED by rule**: R_mirror 0.06 한 수준(33×, R0 floor 근접). 그 밖은 1.03–1.12× |
| SCALING | **SUPPORTED by rule**: build N^1.03. update 지수는 측정 결함 |
| H0 / H1 / H2 | **H1 SUPPORTED**, H0 / H2 NOT SUPPORTED (사전 mapping) |

**Completion answer.**
- 원래 STOP은 비의미 test false positive(무효 sample의 저장 bin) 하나가 원인이었습니다.
- 이를 정정한 뒤, qualified workload에서 source-independent sender relation(C3UR)은 측정 범위에서 다음을 보였습니다.
  - R0와 같은 ray 수와 약간 낮은 per-edit 비용에서 relation 분산을 크게 줄였습니다.
  - C_ss와 broad 영역은 작지만 유의하게 개선했습니다.
  - R_mirror는 평균 개선이었지만, T1 구간은 0을 포함합니다.
  - W2 sharp edit response를 통과했습니다.
- 남은 sharp 오차는 모든 relation이 공유하는 discretisation 바닥이며, 비용은 relation이 아니라 geometry 갱신과 source 사영이 지배합니다.
- C3UR, R1, 어떤 아키텍처도 채택하지 않았습니다.

### Reproducibility

- **Commits.** freeze `7515a29`, amendment `5d6bb5c`, results in the closing commit. Branch `Selective_Recompute`.
- **Environment.** LabServer63, RTX 3080 Ti 12 GB, driver 595.71.05, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Protocol.** `experiments/gm1_relation89/protocol/gm1_relation89.json` (v1 + `amendments[0]`).
- **Seeds.**
  - Screening 89900–89901; qualification 89140+.
  - REF-ER 89100+, W2 89180+, sources 89160+, P 89120+.
  - ORACLE / order fields 89450 / 89460.
  - C3UR 89007 (realisation + 1000 · i); realisations 89201–89216; converged 89301–89348 / sender 89011.
  - Pair energy 89009; renders 89600 / 89601; ladders 89600 + 97 · spp + rep; TF 89700+; sources 89720.
  - R1 / R1M use the WL88 relation seeds.
- **Server.** `~/rest_rel89`. Original stop artifacts in `results/orig_stop/`. Amendment evidence in `results/amend1/` (audit89.json, the run-1 test-bug JSON, segment_check.txt, reference_manifest.json).
- **Local.**
  - `results/gm1_relation89/`: chain status, part-2 GPU trace, JSON (w_screen, wq_V3, tests88 / 89, stage1–3, econ, scale, final89), logs, orig_stop, amend1.
  - `results/evaluation/89/`: seven figures plus a manifest with sha256.
- **Invalid or excluded.**
  - The original stop branch (preserved).
  - Audit run 1 (test bug).
  - Scaling update times (compilation).
  - econ R1 strict-digest check.
  - The part-1 GPU trace (overwritten).
  - Screening partition (design only).
