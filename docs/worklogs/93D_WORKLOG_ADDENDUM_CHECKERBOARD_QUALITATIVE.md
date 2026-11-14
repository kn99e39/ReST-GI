# Worklog — WL93D Post-hoc Qualitative Addendum: Checkerboard Reflected by the Sharp Panel (2026-11-14, revision 2)

> **POST-HOC QUALITATIVE ADDENDUM to WL93D (non-numbered).** Not part of the frozen WL93D protocol, evidence or labels, and nothing here changes them. WL93D's labels stand: REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED ([93D worklog](93D_WORKLOG_ECRO_FROZEN_LOCAL_OPERATOR_OWNERSHIP_ATTRIBUTION.md)). Every number below is descriptive and post hoc; no gate, label or verdict is computed.

- Code: [experiments/ecro_local93d/addendum_checker/](../../experiments/ecro_local93d/addendum_checker/)
- Scene: [scene_ck2.json](../../experiments/ecro_local93d/addendum_checker/scene_ck2.json)
- Protocol and revision history: [protocol_checker93d.json](../../experiments/ecro_local93d/addendum_checker/protocol_checker93d.json)
- Revision-2 code commit: d8d092f
- Server root: LabServer63 `~/rest_x93d_ck2`; the full chain took 11 min, the scene iteration about 2 min
- Results: `results/ecro_local93d_checker/`. Reviewer images: `results/evaluation/93D/addendum_checker/`

## Revision history (why this report replaces the first one)

**Revision 1 is withdrawn.** It used a back-wall checker on E2 T1.
- The S1 panel lobe at about 1.8 m did not resolve 25 cm cells, so the checker could not be seen in the reflection, even in the physical reference. The reference LOW ratios were 0.84 / 0.85 against the ideal 0.43 / 1.14.
- Its visibility rule was geometric only (an ideal-mirror ray landing on the wall). It passed, and the reflection was never checked on a render.
- A post hoc 1 m variant (A4) was added instead of stopping.

**The owner reopened the addendum** with these instructions:
- change the scene composition;
- check on actual renders that the checker is visibly and correctly reflected by the sharp panel, and adjust object placement until it is;
- replace the earlier report and renders.

Revision 2 does exactly that. Revision 1's code, results and report remain in git history (commits 29f394b, a8a5340, 18c35ba, 8c11bea); they are not repeated here.

## Implementation Fact

**How the texture reaches each path.** The texture is a per-patch material-id map, `PMAT`, and all three paths consume it identically. No learned weight, checkpoint or operator changed. The model's input changes on exactly the 54 checker patches (T_aware).
- **Learned ECRO.** Its only material input is `mat_states[PMAT[patch]]` ([model86.py:95](../../experiments/ecro86/model86.py#L95)). The frozen WL92A own s8601 / s8602 checkpoints are used.
- **M64.** `Mn = Mq[PMAT[patch]]`.
- **Physical reference / LOW / ECRO-lookup renderer.** At every surface hit the material id is `PMAT[patch_bin(hit)]`. This is a run-time source patch of `pt86.Ctx.trace`; `pt86.py` itself is unchanged.

**Scene (changed; the camera is not a model input).**

| element | setting | constraint check |
|---|---|---|
| Room, panel S1, sphere, materials | WL86 | unchanged |
| Box A (checker) | centre (−1.08, 0.60, 1.00), half sizes (0.20, 0.45, 0.45), yaw 0.15 | inside the training box rule, `geom_ok` = True; corner x = −1.345 against the rule's −1.35, so the panel-facing face is about 0.22 m from the panel plane; largest allowed face, 30 cm patches |
| Box B | (1.4, 1.5, 0.5), out of view | — |
| Light | one point light at (−1.45, −0.6, 1.7), intensity 30 | inside the training light distribution, `light_ok` = True |
| Camera | (0.3, −1.6, 1.3) → (−1.3, 0.45, 1.0), fov 32° | views the panel at about 45° |

**Scene iteration.** Every step was checked on an actual render (`checker93d.py probe`: textured and untextured at the same seed). The images are in `scene_iteration/`.

| probe | change | what the render showed |
|---|---|---|
| p01 | yaw 45° box at about 0.5 m | blurred blobs |
| p02 | box closer, yaw 20° | still blurred |
| p03 / p04 | largest face, yaw 0.15, face at about 0.22 m | the 3 × 3 checker is clearly visible in the reflection |
| p05 | tighter framing | final framing |

At p03 / p04 the reference LOW ratio in the reflection is 0.79 / 1.36, against the ideal 0.6 / 1.6, at a median mirror distance of 0.36 m. That is about the minimum distance the training box rule allows. The remaining softness of cell edges is the frozen S1 lobe itself.

**Variants.** Both are **material-distribution tests**. An untextured counterpart (box A uniform, id 4) is rendered in the same scene with the same seeds, so all differences use common random numbers.

| variant | materials | why a material-distribution test | note |
|---|---|---|---|
| bx38 | Lambert 0.3 / 0.8 (new rows from the WL86 constructor) | outside the training Lambert range 0.5–0.7 | — |
| bx75 | ids 3 (0.7) / 4 (0.5) | training vocabulary, but wall material on box patches | contrast only 1.4 : 1 |

**Preflight: 9 / 9 PASS** (`results/pre/pre.json`).

| test | result |
|---|---|
| T_state | constructor bit-exact |
| T_buffer | `mat_states` buffer bit-exact |
| T_mq | Lambert rows proportional to albedo |
| T_closure | closure linear in albedo |
| T_identity | patched renderer equals the original, untextured E2 T1: rel ≤ 2e-7, the float-atomic floor |
| T_tables | WL93D tables reproduced |
| T_corr | material id at 86,042 box pixels equals an independent face / cell computation: 0 mismatches |
| T_e2e | direct-light ratio = albedo / 0.5 at 31,173 pixels, max error 3.8e-7 |
| T_aware | model input changes on exactly the 54 checker patches |

**ROI_T.** The panel pixels whose ideal-mirror ray hits a checker patch: 58,035 pixels, 15 patches.

**Renders.**
- Reference: REF-ER, 4 × 8,192 spp per variant (none / bx38 / bx75), seeds 86100–86103.
- Arms: own s8601 / s8602 × C0, C2_src, C2_rcv, C2_both, at 2 × 4,096 spp, seeds 86600 / 86601.
- GPU 83–94 % during the reference renders.

## Measurement

Exposure and scales:
- One exposure for every image, keyed to the untextured reference median.
- One absolute signed-sqrt scale each, shared across the whole addendum: 0.62 for the higher-order error and 0.34 for the panel-pixel texture response.

**Table 1 — Reflection in the reference.**

| | bx38 | bx75 |
|---|---|---|
| Pixel correlation of log ratio, ideal-mirror prediction vs reference LOW, in ROI_T | 0.81 (slope 0.46) | 0.81 (slope 0.47) |
| Ratio per mirror-cell material, reference LOW | 0.79 / 1.36 | 1.32 / 1.09 |
| Ratio per mirror-cell material, reference higher-order h | 0.83 / 1.36 | 1.36 / 1.14 |
| Ratio per mirror-cell material, ideal | 0.6 / 1.6 | 1.4 / 1.0 |

**Table 2 — The arms' higher-order term hl in ROI_T, per mirror-cell material** (textured / untextured, median; s8601 · s8602):

| arm | bx38: 0.3 cells / 0.8 cells | bx75: 0.7 cells / 0.5 cells |
|---|---|---|
| reference h | 0.83 / 1.36 | 1.36 / 1.14 |
| C0 | 1.00 / 1.12 · 0.84 / 1.09 | 1.17 / 1.07 · 1.17 / 1.05 |
| C2_src | 0.99 / 1.11 · 0.88 / 1.07 | 1.17 / 1.08 · 1.17 / 1.06 |
| C2_rcv | 0.96 / 1.20 · 0.85 / 1.17 | 1.21 / 1.08 · 1.20 / 1.06 |
| C2_both | 0.94 / 1.22 · 0.87 / 1.13 | 1.21 / 1.08 · 1.20 / 1.07 |

**Table 3 — Texture response in ROI_T** (textured − untextured higher-order term, slope / corr against the reference; reference half-split noise 5 % (bx38) / 3 % (bx75) of |Δh|), and **E_H in ROI_T** (textured, with untextured in brackets; noise bound 0.008–0.010):

| arm | bx38 response s8601 · s8602 | bx75 response | bx38 E_H s8601 · s8602 | bx75 E_H |
|---|---|---|---|---|
| C0 | 0.28 / 0.76 · 0.46 / 0.94 | 0.41 / 0.70 · 0.52 / 0.89 | 0.076 (0.047) · 0.046 (0.042) | 0.070 · 0.030 |
| C2_src | 0.33 / 0.78 · 0.34 / 0.95 | 0.46 / 0.75 · 0.45 / 0.89 | 0.063 (0.031) · 0.054 (0.019) | 0.048 · 0.044 |
| C2_rcv | 0.39 / 0.93 · 0.63 / 0.96 | 0.49 / 0.79 · 0.62 / 0.88 | 0.069 (0.048) · 0.047 (0.046) | 0.067 · 0.035 |
| C2_both | 0.47 / 0.97 · 0.48 / 0.96 | 0.55 / 0.86 · 0.53 / 0.89 | 0.053 (0.033) · 0.046 (0.023) | 0.046 · 0.038 |

On the box seen directly, every arm gives slope 0.96–1.01 and corr ≥ 0.999. This is not a model test: there the albedo multiplies the arm's lookup through the renderer's physical throughput at x1.

**Table 4 — Table-level ratios of the checker patches** (textured / untextured sums per material, s8601 · s8602; S = single flight, G = learned global term; G carries 49–52 % of B on these patches):

| | ideal | C0 S | C0 G | C2_rcv S |
|---|---|---|---|---|
| bx38, 0.3 cells | 0.6 | 0.94 · 0.78 | 0.88 · 0.77 | 0.600 · 0.598 |
| bx38, 0.8 cells | 1.6 | 1.35 · 1.28 | 1.28 · 1.25 | 1.599 · 1.594 |
| bx75, 0.7 cells | 1.4 | 1.30 · 1.28 | 1.25 · 1.26 | 1.399 · 1.396 |
| bx75, 0.5 cells (box's own id) | 1.0 | 1.00 · 0.99 | 1.00 · 0.99 | 0.999 · 0.996 |

## Observation

1. **The physical reference clearly shows the box checker in the panel reflection.**
   - It is visible in the full radiance, in events 1–2 (LOW) and in the higher-order term (events ≥ 3); see figCK0 rows 1–3.
   - It lies where an ideal mirror predicts (pixel correlation 0.81, figCK0 row 4).
   - Its contrast is softened by the S1 lobe (slope 0.46–0.47 in log ratio).
2. **Every arm places the reflected texture response on the correct cells.** The patterns in figCK1 row 6 coincide with the reference, with corr 0.70–0.97.
3. **Every arm's amplitude is compressed.**
   - Response slopes are 0.28–0.63 (bx38) and 0.41–0.62 (bx75).
   - The darkening of the 0.3 cells is almost absent for C0 s8601 (0.997 vs reference 0.83).
   - The arms' hl ratios sit between 0.84 and 1.22, against the reference's 0.83 / 1.36.
4. **The checker raises the higher-order error in ROI_T for every arm.**
   - For example, C0 s8601 goes from 0.047 to 0.076, and C2_src s8601 from 0.031 to 0.063.
   - The WL93D (untextured) observation that C2_src is best on sharp readers holds in this scene untextured (0.031 / 0.019).
   - With the checker, C2_both is best or near best, and no single arm is best for both seeds and both variants.
5. **At the table level, the M64 receiver use reproduces the albedo ratio exactly in S** (0.600 / 1.599 / 1.399). The learned local operator (C0 S) and the learned global term G compress it: 0.78–0.94 vs 0.6 and 1.25–1.35 vs 1.6. Because G stays learned in every arm, even C2_rcv's total B ratio is 0.68–0.75 / 1.43.

## Interpretation

- **The test is valid.** The texture is consumed identically and model-awarely by all three paths (9 / 9 preflight). It is visibly and correctly reflected by the sharp panel in the physical reference, so the comparison now tests what the brief asked for.
- **On this scene and these two checkpoints, the frozen model localises a reflected albedo pattern correctly but under-responds in amplitude.** The table-level split (Table 4) locates the compression in both learned parts:
  - the local receiver use, which M64 fixes in S;
  - the learned global term, which no arm replaces.
- The compression appears both for new states (0.3 / 0.8) and for training-vocabulary states (0.7 / 0.5 on box patches).
- This is a descriptive account from table ratios and renders, not an isolated attribution. It agrees in direction with revision 1's table-level finding on the back wall.

## Unresolved Question

- Why do the learned local and global terms compress albedo response? Candidate readings include an albedo encoding that is one of 69 state entries through `mat_mlp`, and per-material training (each training patch's material was fixed per object). Neither is tested.
- How much of the arms' remaining rendered under-response (Table 2) beyond the table-level compression comes from bin-specific B values toward the panel, rather than bin sums? This is not separated.
- The WL93A T0 family was not rendered.

## Architecture Verdict

None. No label is computed. WL93D's frozen labels and evidence are unchanged. Nothing is adopted and no successor is authorised. The addendum's status (closed or not) is the owner's decision.

## Reproducibility

- **Commands.**
  - Full chain: `bash run_ck.sh <R> ~/rest_x92a ~/rest_x93a ~/rest_x93d`. This is one detached chain covering preflight, tables, S / G diagnostic, references, arm renders and viz. Add `--dev` for the 32-spp smoke.
  - Scene iteration: `python checker93d.py probe <R> bx38 <spp> <out>`.
- **Inputs.**
  - Frozen checkpoints `rest_x92a/results/models/own_s860{1,2}.pt`, checked by T_buffer and T_tables.
  - WL93D tables `rest_x93d/results/tables/I_E2_T1.npz` (T_tables only).
  - The scene JSON, whose sha256 is logged in `chain.status`.
  - Trace patch hash: in `pre.json` and in every reference render's JSON.
- **Images and data** (`results/evaluation/93D/addendum_checker/`; sha256 in `manifest_checker93d.json`):

| file | contents |
|---|---|
| `figCK0_scene_references.png` | untextured / bx38 / bx75 references: full frame with ROI_T, panel-reflection close-ups of LOW and of the higher-order term, and the reflection-correspondence maps (ideal-mirror prediction next to the reference) |
| `figCK1_{bx38,bx75}_own_s{8601,8602}_sheet.png` | comparison sheets, columns REF / C0 / C2_src / C2_rcv / C2_both. Rows: full frame (reference LOW + arm higher-order), higher-order only, reflection close-up, signed error vs the reference (full and close-up), texture response |
| `figCK2_cell_response.png` | per-patch texture response, arm vs reference |
| `metrics_checker93d.json`, `sg_bx38.json`, `sg_bx75.json` | all numbers |
| `scene_iteration/p01–p05*.png` | the scene-iteration renders |

## Final report (Korean)

**Implementation Fact.**
- 장면을 바꿨습니다. 체커는 박스 A의 각 면(3×3 패치, 셀 30 cm)에 깔았고, 박스는 학습 박스 분포 경계 안에서 패널 평면으로부터 약 0.22 m까지 붙였습니다.
- 카메라는 패널을 약 45°로 보도록 옮겼습니다. 카메라는 모델 입력이 아닙니다. 조명 1개는 학습 분포 안에 두었습니다.
- 배치는 실제 렌더 p01–p05를 직접 보면서 조정했습니다.
- 텍스처는 패치별 material id이고, ECRO, M64, 물리 레퍼런스가 같은 방식으로 소비합니다.
- preflight 9/9 PASS입니다. 박스 픽셀 대응 불일치 0, 직접광 비 오차 3.8e-7이며, 재학습이나 체크포인트·연산자 변경은 없었습니다.

**Measurement / Observation.**
- (1) 레퍼런스에서 박스 체커가 패널 반사에 분명히 보입니다. LOW와 고차항 모두에서 보이고, 이상 거울 예측과의 픽셀 상관은 0.81입니다.
- (2) 모든 arm이 반사된 텍스처 응답의 위치는 맞힙니다(corr 0.70–0.97).
- (3) 진폭은 압축됩니다. slope는 0.28–0.63입니다. 0.3 셀의 어두워짐이 C0 s8601에서는 거의 없습니다(1.00, 레퍼런스 0.83).
- (4) 체커가 들어가면 모든 arm에서 ROI 고차항 오차가 커집니다. 예를 들어 C0는 0.047→0.076, C2_src는 0.031→0.063입니다.
- (5) 테이블 수준에서 M64 수신 S는 알베도 비를 정확히 재현합니다(0.600/1.599/1.399). 반면 학습 로컬 연산자와 학습 global 항 G는 압축됩니다(0.78–0.94 vs 0.6, 1.25–1.35 vs 1.6).

**Interpretation.**
- 이번에는 브리프가 요구한 "반사에 보이는 체커"로 비교했습니다.
- 동결 모델은 반사된 알베도 패턴의 위치는 맞히지만 진폭 응답이 압축됩니다. 압축은 학습 수신 사용과 학습 global 항 양쪽에 있습니다.
- 이상은 서술적 해석이며, 원인을 분리 검증한 것은 아닙니다.

**Unresolved Question.**
- 학습 항이 알베도 응답을 압축하는 원인
- 렌더된 응답 부족 중 bin별 B가 차지하는 몫
- T0 계열은 렌더하지 않았습니다.

**Architecture Verdict.** 없음. WL93D의 동결 라벨은 그대로이고, 애드엔덤 종결 여부는 오너가 결정합니다.

**Reproducibility.** 코드 커밋 d8d092f, 서버 63 `~/rest_x93d_ck2`, 이미지는 `results/evaluation/93D/addendum_checker/`에 있습니다.
