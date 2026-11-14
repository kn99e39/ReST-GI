# Worklog — WL93D Post-hoc Qualitative Addendum: Patch-aligned Checkerboard Albedo (2026-11-14)

> **POST-HOC QUALITATIVE ADDENDUM to WL93D (non-numbered).** This is not part of the frozen WL93D protocol, evidence or labels, and nothing here changes them. WL93D's labels stand unchanged: REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED ([93D worklog](93D_WORKLOG_ECRO_FROZEN_LOCAL_OPERATOR_OWNERSHIP_ATTRIBUTION.md)). Every number below is descriptive and post hoc. No gate, label or verdict is computed.

- Brief: "ReST-GI — WL93D Checkerboard Qualitative Addendum" (Non-Numbering / Post-hoc Qualitative Evaluation).
- Code: [experiments/ecro_local93d/addendum_checker/](../../experiments/ecro_local93d/addendum_checker/).
- Protocol and amendments: [protocol_checker93d.json](../../experiments/ecro_local93d/addendum_checker/protocol_checker93d.json).
- Commits: 29f394b (protocol and code), a8a5340 (A1–A3, before any full-budget render), 18c35ba (A4, post hoc).
- Server root: LabServer63 `~/rest_x93d_ck`, about 16 GPU-min in total.
- Results: `results/ecro_local93d_checker/`. Reviewer images: `results/evaluation/93D/addendum_checker/`.

## What was asked

Render a physically valid, patch-aligned checkerboard albedo on a Lambertian surface that the near-mirror panel reflects. Compare the path-traced reference with the frozen WL93D arms C0, C2_src, C2_rcv and C2_both on E2 T1. The texture must never silently change the physical material while leaving the model unaware of the change. Nothing may be trained, and no checkpoint or operator may change.

## Implementation Fact

**How the texture enters each path.** The texture is a per-patch **material-id map** (`PMAT`). Each of the three paths consumes it identically:

| path | how it sees the texture |
|---|---|
| learned ECRO (frozen WL92A own s8601 / s8602) | Its only material input is `mat_mlp(mat_states[PMAT[patch]])` ([model86.py:95](../../experiments/ecro86/model86.py#L95)). A per-patch id is therefore a native, model-visible input. No learned weight changes. |
| M64 (frozen-ER dense bin matrices) | `Mn = Mq[PMAT[patch]]`, as in WL93D. |
| physical reference / LOW / ECRO-lookup renderer (pt86) | WL86 looks up the material per *shape* (`SMAT[sid]`). One lookup was inserted at run time: at a back-wall hit, the material id is `PMAT[patch_bin(hit)]`. This is a source patch applied in-process; `pt86.py` itself is unchanged. |

**Scene.** E2 T1. The texture sits on the back wall (y = 2 m), whose 16 × 12 patches are each exactly 25 cm × 25 cm. A cell boundary is therefore always a patch boundary.

**Variants.** All three are labelled **material-distribution tests**.

| id | cells | materials | why it is a material-distribution test |
|---|---|---|---|
| ck38 | 25 cm (1 patch) | Lambert 0.3 / 0.8. Two new rows built with the WL86 `lambert_state` constructor and appended to the renderer's material table and to the model's `mat_states` buffer. | Outside the Lambert range seen in training (0.5–0.7). |
| ck75 | 25 cm | Existing ids 3 (walls, 0.7) and 4 (box A, 0.5). | Training-vocabulary states, but box material placed on wall patches. |
| ck38L (post hoc, A4) | 1 m (4 × 4 patches, still patch-aligned) | Same as ck38. | Same as ck38. |

**M64 rows for ids 6 / 7.** These are `Mq[3] × albedo / 0.7`. This is admissible because the Lambert rows are exactly proportional to albedo (T_mq, rel 5.7e-8).

**Preflight.** All nine tests pass for ck38 and for ck38L (`results/pre/pre*.json`):

| test | result |
|---|---|
| T_state | constructor reproduces mats86 ids 3–5 bit-exactly |
| T_buffer | each checkpoint's `mat_states` equals mats86 bit-exactly; the first 6 rows are unchanged after extension |
| T_mq | Lambert rows proportional to albedo (rel 5.7e-8) |
| T_closure | the frozen ER closure is linear in albedo (eval rel 8.9e-8; identical sampled directions; weights equal albedo) |
| T_identity | with the checker off, the patched renderer equals the original (rel 1.4e-7, equal to the original-vs-original float-atomic floor) |
| T_tables | with the checker off, the B tables reproduce the WL93D tables (C0 exact; C2 arms ≤ 1.0e-7) |
| T_corr | the material id at 64,998 back-wall pixels equals the analytic checker from the hit position: 0 mismatches |
| T_e2e | kmax = 1 CRN renders: textured / untextured direct radiance = albedo / 0.7 at 36,139 lit pixels (max abs error 3.2e-7) |
| T_aware | the model input changes on exactly the 192 back-wall patches and nowhere else |

**Visibility rule (geometry only, ideal mirror).** VISIBLE. 94 % of the camera-visible panel pixels (5,171 / 5,479) reflect the back wall, hitting 14 patches (7 even, 7 odd). This set is called **ROI_W**. Of region C's 3,974 pixels, 3,041 lie in ROI_W.

**Renders.**
- Reference: REF-ER, 4 × 8,192 spp per variant. Seeds 86100–86103 are those of the WL86 untextured E2 T1 references r0–r3, which gives common random numbers. Lambert lobe choice and sampling do not depend on albedo, so the paths are identical and only the weights differ.
- Arms: 2 × 4,096 spp. Seeds 86600 / 86601 match the WL93D untextured arm renders (CRN).
- GPU about 100 % during the renders.

**Amendments.**
- A1–A3 were made after the 32-spp dev smoke and before any full-budget render:
  - A1: T_identity uses a tolerance, because `accum` uses float atomics.
  - A2: T_e2e excludes silhouette pixels.
  - A3: presentation scale only.
- A4 is post hoc (see Observation 1). It added ck38L and a reflection-parity metric.
- The 25 cm results are kept unchanged.

## Measurement

The sheet shows one exposure for the whole addendum, keyed to the untextured reference median (Reinhard, gamma 2.2). The error and response maps each use one absolute signed-sqrt scale across the whole addendum: 0.133 for the higher-order error and 0.163 for the panel-pixel texture response.

**Table 1 — reflection parity in ROI_W.** Median of textured / untextured per checker parity of the mirror cell (even / odd):

| | ck38 (25 cm) | ck38L (1 m) | ck75 (25 cm) |
|---|---|---|---|
| ideal mirror (albedo ratio) | 0.43 / 1.14 | 0.43 / 1.14 | 1.00 / 0.71 |
| reference LOW (events 1–2) | 0.84 / 0.85 | 0.76 / 0.93 | 0.90 / 0.89 |
| reference higher-order h | 0.77 / 0.81 | 0.72 / 0.90 | 0.86 / 0.85 |
| C0 hl (s8601 / s8602) | 0.78 / 0.80 · 0.76 / 0.79 | 0.75 / 0.89 · 0.75 / 0.87 | 0.97 / 0.97 · 0.97 / 0.97 |
| C2_src hl | 0.77 / 0.80 · 0.76 / 0.78 | 0.75 / 0.89 · 0.75 / 0.87 | 0.98 / 0.98 · 0.97 / 0.97 |
| C2_rcv hl | 0.80 / 0.83 · 0.80 / 0.83 | 0.76 / 0.92 · 0.77 / 0.92 | 0.93 / 0.91 · 0.93 / 0.92 |
| C2_both hl | 0.80 / 0.83 · 0.81 / 0.84 | 0.77 / 0.93 · 0.78 / 0.92 | 0.92 / 0.91 · 0.93 / 0.92 |

**Table 2 — region E_H (WL86 definition), textured vs untextured on the same pixels.** Seed s8601; s8602 is in `metrics_checker93d.json`.

| arm | ck38 C (unt.) | ck38L C | ck75 C | ck38 ROI_W (unt.) | ck75 ROI_W |
|---|---|---|---|---|---|
| C0 | 0.075 (0.077) | 0.081 | 0.042 | 0.064 (0.061) | 0.059 |
| C2_src | 0.036 (0.021) | 0.035 | 0.078 | 0.034 (0.026) | 0.100 |
| C2_rcv | 0.075 (0.093) | 0.082 | 0.063 | 0.065 (0.076) | 0.054 |
| C2_both | 0.028 (0.031) | 0.028 | 0.028 | 0.027 (0.026) | 0.035 |

The noise bound in ROI_W is 0.014–0.017.

**Table 3 — texture response in ROI_W.** Textured − untextured higher-order term, luminance, pixel level, slope / corr vs the reference. Reference half-split noise is 3 % (ck38), 2 % (ck75) and 3 % (ck38L) of |Δh|.

| arm | ck38 s8601 · s8602 | ck38L s8601 · s8602 | ck75 s8601 · s8602 |
|---|---|---|---|
| C0 | 0.93 / 0.95 · 1.04 / 0.91 | 0.91 / 0.96 · 1.02 / 0.93 | 0.19 / 0.77 · 0.31 / 0.77 |
| C2_src | 1.02 / 0.96 · 1.08 / 0.91 | 0.95 / 0.96 · 1.04 / 0.93 | 0.17 / 0.71 · 0.31 / 0.78 |
| C2_rcv | 0.80 / 0.95 · 0.82 / 0.95 | 0.83 / 0.98 · 0.83 / 0.98 | 0.51 / 0.94 · 0.52 / 0.90 |
| C2_both | 0.85 / 0.96 · 0.83 / 0.95 | 0.86 / 0.98 · 0.83 / 0.98 | 0.57 / 0.96 · 0.54 / 0.92 |

The back wall seen directly gives slope 0.85–0.97 and corr ≥ 0.995 for every arm and variant. This is not a test of the model: there the albedo multiplies the arm's lookup through the renderer's physical throughput at x1.

**Table 4 — table-level diagnostic of the back-wall patches** (`sg_ck38.json`, `sg_ck75.json`; textured / untextured ratio of sums per material id; S = single flight, G = learned global term):

| | albedo ratio | C0 S | C0 G | C2_rcv S | G share of B (untextured) |
|---|---|---|---|---|---|
| ck38 id 6 (0.3) | 0.43 | 0.62 · 0.59 | 0.58 · 0.60 | 0.430 · 0.428 | 0.53 |
| ck38 id 7 (0.8) | 1.14 | 0.97 · 0.93 | 0.98 · 0.98 | 1.147 · 1.143 | 0.53 |
| ck75 id 4 (0.5) | 0.71 | 0.89 · 0.83 | 0.87 · 0.84 | 0.717 · 0.714 | 0.53 |
| ck75 id 3 (0.7) | 1.00 | 1.00 · 0.99 | 1.00 · 1.01 | 1.004 · 1.000 | 0.53 |

## Observation

1. **The 25 cm checker is not resolved in the panel reflection, even in the physical reference.**
   - In the reference LOW the ck38 parity ratios are 0.84 vs 0.85, where an ideal mirror gives 0.43 vs 1.14.
   - The S1 lobe (α 0.026–0.075) at a mean mirror distance of 1.82 m averages several 25 cm cells.
   - The geometric visibility rule passed because it assumes an ideal mirror. The panel close-ups in figCK0 show only a soft gradient.
   - With 1 m cells (ck38L), a partial two-tone reflection appears: 0.76 vs 0.93.
   - The 25 cm sheets therefore test the arms' response to a blurred mixture of cells, not to individually resolved cells.
2. **ck38 and ck38L (new states 0.3 / 0.8).**
   - In the reflection, every arm follows the reference texture response closely: slope 0.80–1.08, corr 0.91–0.98.
   - In ck38L the arms' parity contrast is 0.14–0.16, against the reference higher-order 0.18.
3. **ck75 (in-vocabulary ids 3 / 4).**
   - The reference higher-order term in ROI_W drops by about 15 %.
   - The arms whose receiver use stays learned barely move: C0 and C2_src change by 2–3 %, slope 0.17–0.31.
   - The M64-receiver arms move about halfway: slope 0.51–0.57.
   - As a result, C2_src's E_H in region C rises from 0.021 untextured to 0.078 textured, and C0's falls from 0.077 to 0.042: the arm ordering changes with the wall material.
4. **Table level (Table 4).**
   - The M64 receiver reproduces the albedo ratio exactly in S (0.430 / 1.147 / 0.717).
   - The learned local operator compresses the albedo response: id 6 S 0.59–0.62 vs 0.43; id 4 0.83–0.89 vs 0.71. For id 7 its S ratio (0.93–0.97) points the opposite way from the albedo ratio (1.14).
   - The learned global term G, which carries 53 % of B on the back wall and is kept learned in every arm, behaves the same way: 0.58–0.60 / 0.98 / 0.84–0.87.
5. **The learned operator does see the texture.** Its per-patch inputs change exactly on the textured patches. In every variant its table and its rendered higher-order term change in the direction of the albedo, except at the 0.8 cells, where the table ratio is 0.93–0.98.

## Interpretation

- The checkerboard is a valid, model-consistent material-distribution test here. The interface is the per-patch material id, which the frozen ECRO, M64 and the patched reference all consume identically (tests above). No retraining or architecture change was needed. The texture is not an image-space effect, and the model is not unaware of it.
- Within this scene and these two checkpoints, the learned components respond to a back-wall albedo change in a compressed way. This applies both to the local receiver use and to the global term. Exact albedo scaling appears only where M64 replaces the receiver use, and even there only in S: G stays learned.
- The 0.3 / 0.8 case looks well tracked in the rendered reflection because the reflection blurs the cells. The blurred average of the compressed per-patch ratios (about 0.79) happens to match the average of the true per-patch ratios. ck38L, with partially resolved cells, shows that the arms' contrast is 75–85 % of the reference.
- The ck75 failure to follow the reference is consistent with Table 4:
  - id 4 patches respond 0.83–0.89 instead of 0.71;
  - id 3 patches are unchanged in both S and G (1.00);
  - so the blurred average stays near 0.93–0.97 while the reference falls to about 0.85.
  This is a descriptive account from table ratios; it is not an isolated attribution. Why a training-vocabulary state on wall patches is compressed more than the two out-of-range states is not resolved here.
- On the WL93D question: the C2_src advantage in region C seen in WL93D (untextured) does not carry over to ck75, where C2_src is the worst arm in region C. It remains in ck38 and ck38L.
  - This qualifies the WL93D observation as dependent on the receiving surface's material distribution.
  - It does not change WL93D's frozen labels.

## Unresolved Question

- Why does the learned receiver and global response to the in-vocabulary id 4 compress more (0.83–0.89 vs 0.71) than its response to the out-of-range 0.3 state? Possible readings are a co-adapted wall/box context in the token features, and the albedo encoding being one of 69 state entries passed through `mat_mlp`. Neither is tested here.
- How much of the reference's ROI_W change at the id 3 cells is blur from neighbouring id 4 cells, and how much is multi-bounce coupling? Separating them would need a per-cell reference table (truth fields), which was not generated.
- Would a sharper, smaller-α panel resolve 25 cm cells? That would change transport, so it is out of scope.
- The WL93A T0 family and E5k995 were not rendered.

## Architecture Verdict

None. This qualitative addendum computes no label. WL93D's frozen labels and evidence are unchanged. No architecture, document role or canonical contract changes, nothing is adopted, and no successor is authorised.

## Reproducibility

- `bash run_ck.sh <R> ~/rest_ecro86 ~/rest_x92a ~/rest_x93a ~/rest_x93d` (one detached chain; add `--dev` for the 32-spp smoke; finished jobs resume by return code). Run `python diag_sg93d.py <R> ~/rest_x92a ~/rest_x93a <variant>` for Table 4.
- Inputs:
  - frozen checkpoints `rest_x92a/results/models/own_s860{1,2}.pt`, checked through T_buffer and the T_tables reproduction of WL93D;
  - WL86 untextured references `rest_ecro86/results/renders/REF_ER_E2_T1_r0-3`;
  - WL93D untextured arm renders `rest_x93d/results/renders/I93D_own_s*_E2_T1_r{0,1}`;
  - WL86 masks `rest_ecro86/results/masks/E2.npz`.
- The trace patch hash is in each `pre*.json` and in each reference render's JSON.
- Images, with sha256 in `manifest_checker93d.json` (`results/evaluation/93D/addendum_checker/`):

| file | contents |
|---|---|
| `figCK0_scene_references.png` | untextured and three textured references: full frame, panel close-ups with ROI_W (cyan) and region C (yellow), higher-order term |
| `figCK1_{ck38,ck75,ck38L}_own_s{8601,8602}_sheet.png` | comparison sheets, columns REF / C0 / C2_src / C2_rcv / C2_both. Rows: full frame (textured LOW + arm higher-order), higher-order only, panel close-up, signed error vs the reference (full and close-up; the reference column shows its half-split noise), and the texture response textured − untextured |
| `figCK2_cell_response.png` | per-patch texture response, arm vs reference, in ROI_W and on the directly seen wall |
| `metrics_checker93d.json`, `sg_ck38.json`, `sg_ck75.json` | all numbers above |

## Final report (Korean)

**Implementation Fact.**
- 체커는 뒷벽(25 cm 패치 16×12)의 **패치별 material id 맵**입니다. 세 경로가 같은 맵을 같은 방식으로 소비합니다.
  - 학습 ECRO: `mat_states[PMAT]`
  - M64: `Mq[PMAT]`
  - 물리 레퍼런스 렌더러: 런타임 소스 패치로 재질 조회 1줄을 패치 기준으로 바꿨습니다. pt86.py는 수정하지 않았습니다.
- 재학습, 체크포인트 변경, 연산자 변경은 없었습니다.
- preflight 9개 테스트가 모두 통과했습니다. 여기에는 WL93D 테이블 재현, 픽셀 단위 대응(불일치 0), 직접광 알베도 비 end-to-end 검사(오차 3e-7)가 포함됩니다.
- 모든 variant는 **material-distribution test**입니다. 0.3/0.8은 학습 범위 밖이고, 0.7/0.5는 박스 재질을 벽에 놓은 경우입니다.

**Measurement / Observation.**
- (1) 25 cm 체커는 물리 레퍼런스에서도 패널 반사에 분해되지 않습니다. 반사 LOW 비가 0.84/0.85로, 이상 거울의 0.43/1.14와 거리가 멉니다. 원인은 S1 로브 α 0.026–0.075와 반사 거리 1.8 m입니다. 그래서 사후(A4)로 1 m 셀(ck38L)을 추가했고, 여기서는 0.76/0.93으로 부분적으로 보입니다.
- (2) 0.3/0.8에서는 모든 arm이 반사 영역의 텍스처 응답을 대체로 따라갑니다(slope 0.80–1.08, corr 0.91–0.98). 다만 1 m 셀에서 arm의 대비는 레퍼런스의 75–85 %입니다.
- (3) 0.7/0.5에서는 레퍼런스 고차항이 약 15 % 줄어듭니다. 이에 비해 C0와 C2_src는 2–3 %만 줄고(slope 0.17–0.31), M64 수신 arm은 절반 정도(0.51–0.57) 반응합니다. 그 결과 C2_src의 region C E_H가 0.021에서 0.078로 나빠져, WL93D에서 보였던 C2_src 우위가 이 재질 분포에서는 사라집니다.
- (4) 테이블 수준에서 M64 수신은 S에서 알베도 비를 정확히 재현합니다(0.430/1.147/0.717). 반면 학습 로컬 연산자와 학습 global 항 G(벽 B의 53 %)는 응답이 압축됩니다(id 4: 0.83–0.89 vs 0.71). 0.8 셀에서는 학습 응답이 오히려 반대 방향입니다(0.93–0.98 vs 1.14).

**Interpretation.**
- 텍스처는 모델이 인식하는 유효한 재질 분포 테스트였습니다.
- 학습 구성요소(수신 사용과 global 항)는 벽 알베도 변화에 압축된 응답을 보입니다.
- 0.3/0.8에서 잘 맞아 보인 것은 반사 블러가 셀을 평균한 결과입니다.
- 이상은 서술적 해석이고, 원인을 분리 검증한 것은 아닙니다.

**Unresolved Question.**
- id 4(학습 어휘)가 0.3(범위 밖)보다 더 압축되는 이유
- id 3 셀 변화 중 블러와 다중 반사가 각각 차지하는 비중
- 더 날카로운 패널에서의 결과. 전송이 바뀌므로 범위 밖입니다.
- T0 계열과 E5는 렌더하지 않았습니다.

**Architecture Verdict.** 없음. WL93D의 동결 라벨과 증거는 바뀌지 않았고, 채택하거나 후속 승인한 것도 없습니다.

**Reproducibility.** 커밋 29f394b / a8a5340 / 18c35ba, 서버 63 `~/rest_x93d_ck`, 이미지는 `results/evaluation/93D/addendum_checker/`에 있습니다.
