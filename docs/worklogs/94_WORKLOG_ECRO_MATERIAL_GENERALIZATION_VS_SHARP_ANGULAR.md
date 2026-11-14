# Worklog — ECRO Material Generalization vs. Sharp Angular Representation (WL94) (2026-11-14)

## Origin and scope

- **Brief.** "WL94 — ECRO Material Generalization vs. Sharp Angular Representation: Controlled Attribution and Conditional Small-Scale Training" (user, 2026-11-14).
- **Batch class.** The brief names it WL94 (regular number). Under the 2026-11-13 numbering policy it could also be read as a suffix of WL93D, since it directly resolves the checkerboard addendum's residual question. The owner decides; it was not renumbered.
- **Question.** Is the frozen ECRO's under-response to reflected albedo in the sharp-panel reflection caused by material–scene entanglement in training (H-M), by an angular representation limit (H-A), or by both?
- **Preserved, unchanged:**
  - the WL93D labels (REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED);
  - the addendum's revision-2 evidence;
  - the WL92A own s8601 / s8602 checkpoints and all historical renders.
- **Code.** [experiments/ecro_mat94/](../../experiments/ecro_mat94/)
  - `mat94.py`: material routing, Stage A tests, Stage B.
  - `c94.py`: Stage C.
  - `an94.py`: metrics, gate and labels.
  - `viz94.py`: figures.
  - `run94.sh`: the chain.
  - `posthoc94.py`: post hoc, labelled.
- **Protocol.** [protocol/ecro_mat94.json](../../experiments/ecro_mat94/protocol/ecro_mat94.json). Freeze commit `e347904`.
- **Where.**
  - Server: LabServer63 `~/rest_x94`, one detached chain, 10:44–11:11 UTC (27 min).
  - Local results: `results/ecro_mat94/`.
  - Figures: `results/evaluation/94/`.

## Implementation Fact (verified in code before the design)

**Training materials are fixed per object.** For every WL86 training item the assignment is:
- floor B (id 0), panel S1 (1), sphere S2 (2);
- walls and ceiling Lambert 0.7 (3), box A Lambert 0.5 (4), box B Lambert 0.6 (5).

The model's only material input is the 69-d frozen state `mat_states[PMAT[patch]]`, passed through `mat_mlp`. Ids 3 / 4 / 5 differ only in their three albedo entries. Across the whole training set the network therefore saw exactly one albedo per object.

**Where the reflected checker comes from in an ECRO render.** The renderer is exact at the camera vertex. The higher-order term is `hl = throughput(panel) × B[box patch, bin(box → panel)]`, so:
- The reflected amplitude is set by the **learned B table at the Lambert box patches**, read in the bins that face the panel.
- The panel's sharp lobe enters only upstream: through the panel's learned **sender** (once-scattered light delivered to the box) and through the global term.

**Material routing.** A per-patch material map is consumed identically by four paths:
- the learned operator (buffer rows appended with the WL86 constructor; parameters unchanged at 735,664);
- M64;
- the camera / ECRO renderer (run-time patch of `pt86.Ctx.trace`);
- the patch × bin field reference (the same patch in `pt86a.field_src`).

**Test results.** All 13 Stage A tests passed:
- T_FIELD_E2: the exactly-2-event reference at box-A patches scales with albedo to 8e-8.
- T_ID_FIELD: rel L1 3.7e-8, equal to the original-vs-original float-atomic floor.
- T_TABLES_REPRO / T_CK_REPRO: WL93D and addendum tables reproduced to 1e-7.
- T_REF_REPRO: the addendum bx38 reference reproduced to 8e-7 of the image mean, so the 12 addendum references for none / bx38 / bx75 were reused.
- T_SRCS1: linearity of the new arm, 4e-8.
- T_MODEL_INPUT and T_CKPT: passed.

**Linearity audit (protocol).**
- The exactly-2-event reference at a box-A patch is exactly proportional to its albedo, because box A is convex.
- The 3..5-event reference is not proportional, because paths can revisit the box. It is measured, never imposed.
- Model components are compared with their own physical components:
  - S (single learned flight) against e2 = e2_p + e2_o;
  - G (learned global term) against e3.

**Dev-run fixes before the freeze (disclosed in the protocol).**
- The T_ID_FIELD criterion was changed from a per-cell max to whole-table rel L1. Float atomics alone change about 18 % of cells between two original runs, so the per-cell max was ill-conditioned.
- At field_src k = 1, the probe point's material is now `PMAT[pid0]` when the hit lies on the probe patch's own shape. Before the fix, `patch_bin` rounded 6 of 10,368 box-A cells to the neighbouring checker cell.
- The dev figures (32 spp, 60-step dev candidates) were opened to check layout. This exposure is disclosed.

**Anticipated gate outcome.** The addendum had already shown frozen C0 compression on CK2, so the gate outcome RUN_C was anticipated (disclosed).

## Design (frozen)

**Stage B: frozen counterfactuals.** CK2 geometry, light and camera are unchanged. Six arrangements on box A:

| variant | arrangement | what it tests |
|---|---|---|
| none | default (0.5) | baseline |
| U07 | all 0.7 | the walls' training-vocabulary state on new patches |
| U03 | all 0.3 | new value |
| bx75 | checker 0.7 / 0.5 | vocabulary values |
| bx38 | checker 0.3 / 0.8 | new values |
| bx38i | bx38 inverted | identical value histogram, permuted layout |

- **References.**
  - Patch × bin field: 2 × 4,096 spp, with common random numbers across variants.
  - Images: 4 × 8,192 spp.
- **Arms.**
  - C0, C2_src, C2_rcv, C2_both (WL93D);
  - new: C2_srcS1, which uses M64 at the S1 panel sender only (isolates the sharp sender).
- **Metrics.** Log-response amplitude (slope) and location (corr) of S, G and B, measured over:
  - box-A patches;
  - the panel-facing face vs the other faces;
  - panel-facing bins vs other bins.
- **Further splits:**
  - error decomposition into a baseline-carried part and a response part;
  - sender split (panel sender vs others);
  - receivers of the box material (S1 vs non-S1 patches);
  - image readers: the sharp ROI_T and a diffuse reader DIFF (floor / wall pixels with view factor to box A ≥ 0.10; 9,039 px).

**Gate.** RUN_C unless the frozen material response is already correct, defined as amplitude in [0.85, 1.15] and corr ≥ 0.9.

**Stage C: paired smoke.**
- Both candidates fine-tune from own_s8601 with the extended material buffer (WL93A–C smoke lineage):
  - A: original fixed materials.
  - B: per-geometry, per-Lambert-rectangle albedo drawn from {0.2, 0.4, 0.5, 0.6, 0.7, 0.9}.
- Held constant: the same 96 geometries, lights and target seeds (common random numbers), the OWN objective, 2,000 steps, the same order and the same initialisation (T_INIT: identical sha256).
- Budget: 8.3 % of the steps and 9.4 % of the geometries of a full run.
- Held out:
  - values 0.3 / 0.8;
  - within-rectangle patterns;
  - the CK2 geometry and light, by margin;
  - WL86 validation geometries:
    - V_orig: original materials;
    - V_ck: per-patch 0.3 / 0.8 checkers on every Lambert rectangle.
- T_LEAK passed: no 0.3 / 0.8 and no within-rectangle variation in any training map. Train and val are different WL86 splits; 22 file names coincide across the splits, but the geometries come from different seeds.

**Fit validity.**
- F1: no non-finite step.
- F2: V_orig rel L1 ≤ 1.10 × the initial value.
- F3: no divergence.

**Outcome rules.**
- MR_C: material response restored.
- SR_C: sharp ROI error reduced by ≥ 20 %.
- SSR: the sharp-sender residual. For each seed, the untextured ROI E_H(C0) − E_H(C2_srcS1) must be ≥ max(2 × noise, 0.005) and form ≥ 30 % of E_H(C0).
- Labels follow the protocol mapping.

## Measurement

### Stage B — frozen own (new counterfactual measurements)

**Table level, box-A patches, C0.** Log-response amplitude (corr), seeds s8601 / s8602.

| component | U07 (0.7 vocab) | U03 (0.3 new) | bx75 | bx38 | bx38i |
|---|---|---|---|---|---|
| B = S + G | 0.66 / 0.64 | 0.10 / 0.41 | 0.69 (0.96) / 0.68 (0.90) | 0.30 (0.91) / 0.48 (0.96) | 0.34 (0.91) / 0.49 (0.96) |
| S, learned receiver | 0.74 / 0.73 | **0.01** / 0.37 | 0.75 / 0.74 | 0.25 / 0.46 | 0.31 / 0.48 |
| S, exact receiver (C2_rcv) | 1.00 / 0.97 | 1.00 / 0.98 | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| G, learned global | 0.60 / 0.58 | 0.16 / 0.42 | 0.63 / 0.62 | 0.32 / 0.48 | 0.36 / 0.48 |

- Reference S/N: the median |rho| / noise over box-A patches is 990–3,950, so references are not noise-limited.
- Sender-side substitution (C2_src, C2_srcS1) leaves the material amplitude unchanged (±0.02).

**Faces and bins (C0, B amplitude).**
- Panel face vs other faces: bx38 0.22 / 0.31 (s8601), 0.45 / 0.49 (s8602).
- Panel-facing bins vs other bins: bx38 0.33 / 0.30 and 0.55 / 0.48.

**Textured error decomposition at box-A patches (bx38, C0).**
- Response term 0.30 / 0.24 against baseline-carried term 0.065 / 0.068.
- The untextured baseline error is 0.06 / 0.07.

**Sender split, untextured.** The single flight delivered from the panel sender (e2_p) is 10 % of S over box A and 59 % on the panel face.
- Its error is 0.59 / 0.87 with the learned receiver.
- It is 0.59 / 0.91 with the exact receiver.
- The non-panel senders have errors of 0.05 / 0.06.

**Receivers of the box material.** C0 amplitude on S1 panel patches is 0.19–0.38. Non-S1 receivers are near 0 or erratic (−1.37 to 0.17), which is noise-dominated.

**Images (C0).**

| | s8601 | s8602 |
|---|---|---|
| Untextured ROI_T E_H, C0 | 0.047 | 0.042 |
| Untextured ROI_T E_H, C2_srcS1 | 0.031 | 0.019 |
| Noise bound | 0.0084 | 0.0084 |
| Textured ROI_T E_H, C0 | 0.062–0.106 | 0.030–0.122 |

Texture-response slope (corr):

| reader | U07 | bx75 | bx38 |
|---|---|---|---|
| ROI_T (sharp) | 0.36 / 0.47 | 0.41 / 0.52 | 0.28 (0.76) / 0.46 (0.94) |
| DIFF (diffuse) | 0.41 / 0.42 | 0.41 / 0.43 | 0.69 / 0.31, reference noise 0.41 rel, noise-limited |

- With the exact receiver plus the learned global (C2_rcv), the ROI bx38 slope is 0.39 / 0.63.

**Gate and SSR_frozen.**
- Gate: MR_frozen is false for both seeds, so **RUN_C**.
- SSR_frozen, s8601: diff 0.0162, share 0.34, against 2 × noise = 0.0167. **Fails by 0.0005.**
- SSR_frozen, s8602: diff 0.0226, share 0.54. **Passes.**

### Stage C — paired smoke, seed 8601 (paired-smoke evidence)

**Fit.** Both runs took 108–110 s at 656 MB peak; T_INIT passed and there were no non-finite steps.

| | A (fixed) | B (diverse) |
|---|---|---|
| V_orig whole rel L1 (init 0.0778) | 0.0768, F2 pass | **0.0879, F2 FAIL** (limit 0.0856; 0.0964 at step 1,000) |
| L_sum first / last 200 steps | 0.055 / 0.054 | 0.42 / 0.069 |
| V_ck whole rel L1 (init 0.305) | 0.315 | **0.106** |

FIT fails, so the outcome is **INCONCLUSIVE_SMOKE_FIT**. By the frozen rule the seed 8602 pair was not run, and the budget was not extended.

**Material response (C0, B amplitude).**
- CK2 box A, A / B:

  | variant | A | B |
  |---|---|---|
  | bx38 | 0.29 | 1.06 |
  | bx38i | 0.34 | 1.06 |
  | U03 | 0.13 | 0.99 |
  | U07 | 0.64 | 1.05 |
  | bx75 | 0.67 | 1.08 |

- B's checker corr is 0.99.
- In B every component responds: S 1.02–1.10 and learned G 0.97–1.08.
- V_ck Lambert population: init 0.42, A 0.40, B 1.05 (corr 1.00, 22,200 patches). S1 receivers: 0.53 / 0.50 / 1.00.
- The MR_C cells all pass, but the rule is conditional on FIT.

**Baseline and sharp reader.**
- B's untextured box-A error is 0.105, against A's 0.058.
- ROI_T E_H, C0, A / B:

  | | A | B |
  |---|---|---|
  | untextured | 0.039 | 0.087 (signed −0.086) |
  | bx38 | 0.066 | 0.095 |
  | bx38i | 0.049 | 0.081 |

  SR_C fails: B is worse.
- ROI texture-response slope: A 0.36 / 0.44 / 0.57, B 0.66 / 0.68 / 0.67 (bx38 / bx38i / bx75).
- DIFF texture-response slope: A 0.43 / 0.42 (U07 / bx75), B 0.98 / 0.96. U03 goes from −0.05 to 0.92.
- B untextured SSR_C: diff 0.0098, share 0.11. Fails.

**Labels (frozen mapping).** **H-M INCONCLUSIVE, H-A INCONCLUSIVE** (`results/ecro_mat94/final.json`).

### Post hoc (descriptive, after the labels; `posthoc94.py`, `results/ecro_mat94/posthoc94.json`)

Untextured box-A error, split by face × bin group (B rel L1, signed):
- **The error is a face effect, not a bin effect.**
  - On the panel face, panel-facing bins and other bins have equal error:
    - own s8601: 0.251 (−0.24) vs 0.267 (−0.26);
    - own s8602: 0.177 vs 0.186;
    - B: 0.409 (−0.41) vs 0.374 (−0.37).
  - The other faces have 0.04–0.07 in both bin groups.
  - The pre-registered "panel-facing bins" contrast (0.15 vs 0.056) mainly reflects that panel-facing bins sit on the panel face.
- **On the panel face, 61 % of B is the higher-order part** (box ↔ panel paths).
  - Learned G there is biased low: own −0.17 / −0.11, A −0.02, **B −0.46**. This is B's dark reflection.
  - The panel-sender single flight there carries 59 % of S. Its error is 0.53–0.84 under the learned receiver and the same under the exact receiver.
  - C2_srcS1 turns the S bias on that face from −0.34 to +0.13 (s8601), but its rel L1 stays at 0.40.

## Observation

1. **Material response compression is not sharp-specific.**
   - The frozen operator compresses the albedo response at Lambert box patches in every component (S and G), on every face and in every bin group.
   - The compression appears equally in the sharp reader (ROI_T) and in the diffuse reader (DIFF).
   - The exact receiver restores S exactly.
   - A vocabulary state on new patches (0.7) still gets only 0.66–0.74. Values below the vocabulary range are almost ignored (U03: 0.01–0.37).
2. **Material diversification alone restores the material response in the same architecture within 2,000 steps.**
   - This holds for all components, including the learned global term.
   - It holds on held-out values (0.3 / 0.8) and held-out layouts (checkers), at CK2 and over 24 validation geometries.
   - In images, the diffuse reader's texture response goes from about 0.42 to 0.92–0.98.
3. **The same smoke run lost baseline fit.**
   - It lost fit on original materials (+13 % V_orig; F2 fails).
   - It lost fit specifically on the panel face of the box, whose higher-order input is panel-mediated (G bias −0.46).
   - As a result the sharp reflection became darker. ROI E_H worsened in every variant, although the reflected texture slope improved (0.36 → 0.66).
4. **A sharp-coupled error exists with in-distribution materials.**
   - Untextured, the panel sender's once-scattered delivery to the box is 59–91 % wrong, independent of the receiver.
   - Substituting the exact panel sender removes 34 % (s8601; at the noise limit) and 54 % (s8602) of the untextured ROI error.
   - The error sits in what the sharp panel delivers to the box (sender lobe and panel-mediated global term). It does **not** sit in the box's outgoing directional bins.

## Interpretation

**Physical correctness.** Material routing, reference proportionality, reproduction of WL93D and the addendum, and reference S/N all hold. The Stage B reference differences are therefore valid targets.

**Model behaviour (frozen).**
- The under-response of the reflected checker decomposes into two parts.
  - Material-response compression, which is generic. It appears in Lambert receiver and global terms, and in sharp and diffuse readers alike.
  - A separate sharp-coupled baseline error at the panel-lit face, located in the panel sender and the panel-mediated global term.
- At box-A patches, the textured error is dominated by the response term (0.24–0.30 vs 0.07).

**Paired smoke.**
- Descriptively, it supports the premise of H-M: the existing architecture *can* express the correct material response when the training distribution decorrelates material from object, even in its global term.
- But the protocol's minimum baseline-fit criterion failed for B. The run is INCONCLUSIVE_SMOKE_FIT, and H-M cannot be labelled SUPPORTED.
- At this budget the material gain came with a loss on the sharp-coupled baseline, so the sharp reflection error was **not** reduced.

**What this does not establish.**
- Whether a longer or better-matched run keeps both: correct material response and baseline sharp fit.
- Whether the sharp-coupled error is a representation limit. WL92's representability control remains INCONCLUSIVE, and no capacity was varied.
- **Architectural inference (bounded).** Only this: the reflected-albedo compression is not, by itself, evidence for an angular representation deficit, because it is generic and is removable by training distribution alone. The residual sharp-specific deficit concerns sharp-panel-delivered incident light. That is consistent with WL93D's C2_src finding and WL92's panel-lobe retention, not with the box's outgoing angular structure.

## Unresolved Question

- Does a material-diverse run that also retains baseline fit (F2) reduce the sharp ROI error? The smoke stopped by rule; seed 8602 was not run.
- The panel-face global term worsened sharply under diversification (−0.46). Is that budget-limited re-fitting or a capacity trade-off? Not isolated.
- SSR_frozen differs between seeds: s8601 misses the noise rule by 0.0005. The sharp-sender residual is real in the table-level sender split for both seeds, but the image rule is not met for both.
- Non-S1 receivers of the box material are noise-dominated at this reference budget.
- The non-sharp image control (DIFF) is noise-limited for bx38 / bx38i (reference noise 0.41 / 0.14 rel). U07 / U03 / bx75 are usable.

## Architecture Verdict

- **H-M: INCONCLUSIVE.** The reason is INCONCLUSIVE_SMOKE_FIT: B failed F2, the baseline fit on original materials, at 1.13× vs the 1.10× limit.
  - Descriptively, material diversification restored the material response (amplitude 0.29 → 1.06) within the same architecture.
  - The failure was a short-run fit failure, not a representation-class failure.
- **H-A: INCONCLUSIVE.**
  - SSR_frozen holds for s8602 only.
  - Descriptively, a sharp-sender / panel-mediated baseline deficit is present with in-distribution materials and is not reduced by diversification at smoke scale.
  - Representation-class limitation is not tested.
- No architecture is adopted or changed. There is no WL95 and no successor. The architecture decision returns to the owner and the supervising assistant.

## Reproducibility

- **Commit and protocol.** Freeze `e347904` (protocol sha256 prefix 99e199f3847939e8). The post hoc script `posthoc94.py` is committed with the results.
- **Server.** LabServer63 `~/rest_x94`; `chain.status`; `results/`: pre, fields, tables, renders, models, pop, an, eval94. The dev root is `~/rest_x94dev` (not evidence).
- **Checkpoints.** own_s8601 `739e7eb5…`, own_s8602 `cee9a09a…` (file sha256, = WL92B repro.json). Candidate init sha256 `a85ddb42…` for both A and B.
- **Scene.** `experiments/ecro_local93d/addendum_checker/scene_ck2.json` (unchanged).
- **Reused references.** `results/ecro_mat94/reused_refs.txt` (12 files with sha256).
- **Material maps.** `results/ecro_mat94/select94.json`:
  - per-file digests for A / B / orig / ck;
  - B albedo histogram over rectangles: 0.2: 275, 0.4: 247, 0.5: 255, 0.6: 257, 0.7: 246, 0.9: 256.
- **Seeds.**
  - Field references: 94301 / 94302.
  - Images: 86100 + r.
  - Arm renders: 86600 + r.
  - Targets: 940000000 / 941000000 + 16 × idx + light.
  - Assignments: 94100 / 94200 + idx.
  - Data order: rng(8695).
- **Sample counts.** Fields 2 × 4,096; images 4 × 8,192; arm renders 2 × 4,096; train targets 256; validation 2 × 512.
- **Budget.** 2,000 steps × 2 candidates, 96 geometries.
- **Cost.** 27 min wall in total; training 108–110 s per candidate; GPU mean 69 % over the chain samples (79 % of samples > 50 %).
- **Figures** (`results/evaluation/94/`, manifest with sha256):
  - fig0: references and regions;
  - fig1: Stage B amplitudes;
  - fig2: direction / sender / receivers;
  - fig3_images_<variant>: REF / frozen arms / A / B, with higher-order close-ups, signed error and texture response on shared absolute scales;
  - fig4: training;
  - fig5: Stage C amplitudes.

## Final report (Korean, for the owner)

**1. Implementation Fact (구현 사실)**
- 학습 재질은 객체마다 고정돼 있습니다. 상자 A는 항상 0.5, 벽은 0.7입니다.
- ECRO 렌더에서 반사된 체커 진폭은 Lambert 박스 패치의 학습 B가 정합니다. 패널의 sharp 로브는 sender(박스로 보내는 빛)와 전역 항을 통해서만 들어옵니다.
- 패치별 재질은 학습 연산자, M64, 렌더러, 장 레퍼런스가 똑같이 소비합니다. Stage A 테스트 13개가 모두 통과했습니다.
- 동결 전 dev에서 고친 것은 둘입니다. 장 레퍼런스의 k=1 프로브 재질, 그리고 field 동일성 지표입니다. dev 그림을 열람한 사실과 함께 공개했습니다.

**2. Measurement (측정)**
- **Stage B(동결 own).** 박스 A의 학습 receiver S 응답 진폭(s8601 / s8602):
  - 어휘 상태 0.7: 0.74 / 0.73
  - 새 값 0.3: 0.01 / 0.37
  - 체커 0.3 / 0.8: 0.25 / 0.46
  - 정확 receiver: 1.00
  - 학습 G: 0.16–0.63
- 압축은 면이나 bin과 무관하고, sharp 판독(ROI)과 확산 판독(DIFF)에서 비슷합니다. 예: U07 0.36–0.47 대 0.41–0.42.
- 패널 sender를 거친 단일비행은 receiver와 무관하게 0.59–0.91 틀립니다.
- 패널 sender만 정확히 바꾸면 미텍스처 ROI 오차가 34 % / 54 % 줄어듭니다. s8601은 노이즈 규칙에 0.0005 차이로 미달했습니다.
- **Stage C(own_s8601에서 2,000 step, 96 기하).** B(다양화)는 재질 진폭을 회복했습니다.
  - CK2 기준 bx38 0.29 → 1.06, U03 0.13 → 0.99.
  - 검증 모집단 0.40 → 1.05.
  - 확산 판독의 텍스처 기울기 0.42 → 0.96.
- 그러나 원래 재질 검증 오차가 0.0778 → 0.0879로 늘어 F2(≤ 0.0856)를 넘었습니다.
- 패널 면 전역 항이 −0.46으로 어두워져 sharp ROI 오차가 0.066 → 0.095로 악화됐습니다.

**3. Observation (관찰)**
- 반사 체커의 진폭 압축은 sharp 전용 현상이 아닙니다. Lambert receiver와 전역 항 전반에 걸친 재질 응답 압축입니다.
- 재질 분포만 다양화해도, 같은 아키텍처의 짧은 학습에서 그 압축이 사라졌습니다.
- 이와 별도로, 학습 분포 재질 그대로인 미텍스처 장면에도 sharp 패널이 공급하는 입사 결손이 있습니다. 위치는 sender 로브와 패널 매개 전역 항입니다.
- 박스의 출사 방향 bin 구조에는 결손이 없습니다(사후 분해: 면 효과이지 bin 효과가 아님).

**4. Interpretation (해석)**
- 물리 정확성은 확인됐습니다.
- 모델 행동으로 보면 결손은 두 가지가 공존합니다. 하나는 일반적인 재질 응답 압축이고, 다른 하나는 sharp 패널 결합 기준선 오차입니다.
- 짝 smoke는 서술적으로는 H-M의 전제를 지지합니다. 아키텍처가 재질 응답을 표현할 수 있다는 것입니다.
- 하지만 기준선 적합 기준을 지키지 못했으므로 판정은 INCONCLUSIVE_SMOKE_FIT입니다. 이 예산에서는 재질 응답을 얻는 대신 sharp 결합 기준선을 잃었습니다.
- 아키텍처 추론은 이 한도 안에서만 합니다. 반사 체커 압축 자체는 각도 표현 결손의 증거가 아닙니다. 남는 sharp 결손은 패널이 공급하는 입사 쪽에 있습니다(WL93D C2_src, WL92 패널 로브와 일치).

**5. Unresolved Question (미해결 질문)**
- 기준선 적합을 유지하는 다양화 학습이 sharp 오차를 줄일 수 있는지는 확인하지 못했습니다(규칙상 중단, seed 8602 미실행).
- 패널 면 전역 항의 악화가 예산 부족 때문인지 용량 trade-off 때문인지 분리하지 못했습니다.
- SSR 판정이 시드 간에 불일치합니다.
- DIFF bx38 / bx38i는 노이즈 한계입니다.

**6. Architecture Verdict (아키텍처 판정)**
- **H-M: INCONCLUSIVE.** 원인은 INCONCLUSIVE_SMOKE_FIT이고, 재질 응답 회복은 서술적으로 관찰됐습니다. 이는 짧은 학습의 적합 실패이지 표현 클래스의 실패가 아닙니다.
- **H-A: INCONCLUSIVE.** SSR은 s8602만 통과했습니다. 표현 클래스 한계는 시험하지 않았습니다.
- 채택, 아키텍처 변경, 후속 배치는 없습니다. 결정은 owner에게 돌려드립니다.

**7. Reproducibility (재현성)**
- freeze `e347904`, LabServer63 `~/rest_x94`, 27분 소요, 그림은 `results/evaluation/94/`.
- 체크포인트 해시, 시드, 재질 맵 다이제스트는 위 Reproducibility 절에 있습니다.
