# WL97 Phase 2 - J3: P0-3 (exact-direction / hit-point ER evaluation) and the WL95A FLIGHT_ATTRIBUTION label

Adversarial adjudication of findings A-05, B-07, C2-03, C2-04, C2-07, C2-08, E2-F4, E2-F5, E2-F6, E2-F7, E3-04, E3-05, E3-06,
E3-08 and F0-11. Read-only audit: nothing was trained, tuned, modified, moved or deleted. Date 2026-11-16.

## 1. Own checks (two scripts, both on stored data only)

| check | script | output | input |
|---|---|---|---|
| J3-1 frozen FLIGHT_ATTRIBUTION rule re-derived with the alternatives auditors raised; single-draw paired statistics; budget curve | `experiments/audit97/J3/j3_flight_rule.py` (local, numpy) | `results/audit97/J3/j3_flight_rule.json` | `results/ecro_attr95a/deliv.npz` sha256 f2324a32effc80ca (same file B, C2, E3 used) |
| J3-2 paired image statistic for the WL95A CK2 ROI_T renders: per-replicate pairing x reference halves, unpaired cross-seed control, 16x16-pixel block bootstrap (2,000) | `experiments/audit97/J3/j3_img_paired.py` (server, CPU, ~1 min) | `results/audit97/J3/j3_img_paired.json` (copy of `~/rest_x97/J3/j3_img_paired.json`) | `~/rest_x95a/results/renders/{CK_none_*_X_CK2_T1_r0,r1, REF_ER_CK_none_CK2_T1_r0..r3}.npz`, `~/rest_x95a/results/pre/masks94.npz` (ROI_T) |

J3-1 reproduces WL95A's stored rule values exactly (E0 0.4645, Rs 0.0242 +- 0.0341, Ra 0.2319 +- 0.0466, Rp 0.1571 +- 0.0601,
E(X1,pool) 0.0513, label ANGULAR_PROJECTION_EVIDENCED). J3-2 reproduces WL95A B3 / C2 E_H values (floor 0.01184, pooled A4 +0.01261, ...).

### 1.1 Flight attribution under alternatives (J3-1)

| E0 definition | E0 | Rs | label under the frozen rule |
|---|---|---|---|
| frozen: deployed A4 draw 86001 | 0.464 | 0.024 | ANGULAR_PROJECTION_EVIDENCED |
| mean single-draw A4 error | 0.702 | 0.262 | MIXED |
| median single-draw A4 error | 0.667 | 0.227 | MIXED |
| each stored draw as "deployed", leave-one-out pool | - | median Rs/E0 0.34 | ANGULAR 23, MIXED 31, SAMPLING_DOMINANT 10 of 64 |
| same-arm sampling term: mean single-draw M64_spool - pooled M64_spool | - | 0.221 | (not a rule quantity) |

Ra (0.232) is independent of the draw choice. The 10 SAMPLING_DOMINANT cases are draws with E0 > 0.93, where the E0-scaled
threshold (0.25 E0) stops counting the fixed Ra: a threshold artefact, not evidence against the angular bias.

### 1.2 Single deployed-budget realisation vs pooled (J3-1; box A abs rel L1 vs e2_p, n0 = 0.059)

- Single draw (512 rays per patch): X1 mean 0.737 / median 0.617; A4 0.761 / 0.726; M64_spool 0.720 / 0.665.
  Paired X1 - A4 = -0.024 +- 0.055 (X1 better on 69 % of draws); X1 - M64_spool = +0.017 +- 0.055 (62.5 %). No average gain; heavy tail.
- Deployed draw: X1 0.764 / 0.627 (ratio / L1) vs A4 1.208 / 0.524.
- Mean of D random draws (200 subsets): X1 0.738 / 0.614 / 0.471 / 0.342 / 0.260 / 0.181 / 0.111 at D = 1 / 2 / 4 / 8 / 16 / 32 / 64;
  M64_spool 0.722 / 0.632 / 0.554 / 0.523 / 0.505 / 0.502 / 0.500. The binned arm plateaus at 0.50 (bias); X1 halves it at D ~ 16.

### 1.3 Image level, paired (J3-2; ROI_T, E_H normalised by total radiance; floor E_H 0.0118; protocol tol 0.0167)

| arm | dE_H vs floor | replicate SD (paired) | replicate SD (unpaired, cross-seed) | block-bootstrap SE |
|---|---|---|---|---|
| pooled X1 | +0.0006 | 1.3e-5 | 1.5e-5 | 2.2e-4 |
| pooled X1c | +0.0009 | 1.9e-5 | 2.0e-5 | 5.0e-4 |
| pooled q256 | +0.0068 | 2.6e-5 | 2.6e-5 | 6.9e-4 |
| pooled M64_spool | +0.0104 | 4.1e-5 | 4.1e-5 | 7.9e-4 |
| pooled A4 | +0.0126 | 4.3e-5 | 4.3e-5 | 8.9e-4 |
| deployed A4 | +0.0034 | 6.2e-6 | 1.0e-5 | 5.2e-4 |
| deployed A0 s8601 / s8602 | +0.0130 / +0.0354 | 2.5e-5 / 1.9e-5 | - | 1.2e-3 / 1.7e-3 |
| G s8601 / s8602 | +0.0133 / +0.0032 | 3.3e-5 / 1.9e-5 | - | 1.1e-3 / 4.8e-4 |

Arm-vs-arm: pooled A4 - pooled X1 +0.0120 (bootstrap SE 8.7e-4; reference-half / replicate values 0.0114-0.0115);
pooled q256 - pooled X1c +0.0058 (9.7e-4); pooled X1c - pooled X1 +0.0003 (4.9e-4, not resolved); deployed A4 - pooled X1 +0.0028 (5.4e-4).

Key point: even the UNPAIRED cross-seed replicate scatter of dE_H is <= 4e-5. The protocol's noise bound (0.0083) is the half-split L1 of
the per-pixel noise, dominated by the reference (0.0077), i.e. a noise magnitude, not the uncertainty of the aggregate E_H. The
tolerance 0.0167 is therefore not a resolution limit: it is ~400-3000x the replicate SD and ~10-35x a conservative block-bootstrap SE.
Every arm difference >= 0.003 in the table is resolved.

NEXT_DESIGN_TARGET under a paired tolerance (0.005 materiality floor, or 2 x bootstrap SE ~ 0.002-0.003):
s8601 dS 0.0130 / dF 0.0034 / dG 0.0133 -> no rule category (dG - dS = 0.0003; DIRECTIONAL needs dF >= 0.0133 - tol);
s8602 dS 0.0354 -> LEARNED_SENDER on this seed only. With pooled dF (0.0126) and/or pooled dS (0.0104 / 0.0446): s8601 could read
DIRECTIONAL_CARRIER but s8602 stays LEARNED_SENDER. UNDETERMINED in every variant; the reason is seed disagreement, not render noise.

## 2. Rulings

| finding | ruling | basis |
|---|---|---|
| A-05 | UPHELD | Deterministic 1.015 / 0.068 corroborated by E2 (area form 0.994 / 0.029), E3, C2-08 and WL96's designers; group numbers present in `results/audit97/A/a1_bins_m64_deliv.json` (walls/ceiling X1 pooled 1.228 +- 0.395). Scope restrictions correct. |
| B-07 | UPHELD_WITH_MODIFICATION | Frozen-rule and alternative-E0 numbers reproduced exactly (J3-1). Modification: the "pooled X1 error 1.08 at D = 1" depends on which draws are taken; random-subset means are 0.74 / 0.47 / 0.26 / 0.11 at D = 1 / 4 / 16 / 64. |
| C2-03 | UPHELD_WITH_MODIFICATION | Reproduced (own leave-one-out: 23 / 31 / 10 vs C2 22 / 30 / 12 with self-inclusion). Modification: the SAMPLING_DOMINANT draws arise from the E0-scaled threshold no longer counting the draw-independent Ra (E0 > 0.93); the angular bias itself is draw-independent. |
| C2-04 | UPHELD_WITH_MODIFICATION | Per-draw statistics reproduced. Modification: at single draw X1 has no average gain (paired -0.024 +- 0.055 vs A4, +0.017 +- 0.055 vs M64_spool) but a better median (0.617 vs 0.665-0.726); the q256 -> X1c confound was disclosed in WL95A:152 and partly re-attributed in WL96:63-64. |
| C2-07 | UPHELD_WITH_MODIFICATION | Paired analysis confirms (J3-2). Modification: the tolerance is not a standard error of E_H at all; even unpaired replicates scatter by <= 4e-5, so "60-80x" understates the gap vs replicate SD and lies within the range vs a conservative block bootstrap (10-35x). NEXT_DESIGN_TARGET stays UNDETERMINED (seed disagreement). |
| C2-08 | UPHELD | Oracle-reference agreement corroborated by three separate code paths (C2 rerun, E2 area form, A deterministic). Resolution ~3-5 % (E3-08). |
| E2-F4 | UPHELD | Consistent with A, E3, WL96 deterministic values and the stored 64-draw pool. |
| E2-F5 | UPHELD | Stored-draw values agree (J3-1). Already disclosed in WL95A:132; the P0-3 wording omits it. |
| E2-F6 | UPHELD_WITH_MODIFICATION | Not material: 0.111 vs an expected ~0.17 is ~1.5 x WL95A's own reported SE (0.041); no rule or decision moves (E(X1,pool) 0.05 or ~0.115 vs 0.5 E0 = 0.232). Percentile claim rests on E2's simulation only. Compatible with E3-05 (residual is noise; it is a low noise draw). |
| E2-F7 | UPHELD_WITH_MODIFICATION | Code confirms X1c = point evaluation at the patch centre (`a95.py:90-104,134`). The confound was partly disclosed (WL95A:152), but its line "only X1c -> X1 isolates spatial position" is wrong because a patch-constant carrier is a patch average, not a centre sample; WL96:63-64 partly corrected it. Rp is not in the label rule, so the label is unaffected. Image level: pooled X1c - X1 = +0.0003, unresolved. |
| E3-04 | UPHELD_WITH_MODIFICATION | X1c ~0.27 confirmed (stored 0.268, A 0.268, E2 0.269). Modification: "both parts needed" holds for closure to reference noise, but direction carries ~70 % and a patch-AVERAGE anchor with exact direction already reaches 1.034 / 0.111 (E2). |
| E3-05 | UPHELD | Reproduced pooled values exactly. |
| E3-06 | UPHELD | Reproduced (J3-1). |
| E3-08 | UPHELD | No verdict impact (effects 0.27-0.49 vs reference resolution ~0.06). |
| F0-11 | UPHELD_WITH_MODIFICATION | First half (expectation only) upheld. Second half modified: the stored renders DO show image-level relevance in expectation (pooled A4 - pooled X1 = +0.0120 E_H, ~14 block-bootstrap SE; ROI E_H 0.0245 -> 0.0124, floor 0.0118). Relevance at the deployed single-draw budget is UNTESTED (no single-draw X1 render; deployed A4 only +0.0034). |

## 3. Answers to the cluster questions

(i) Closure is an expectation-level statement. At the tested ray budget (one relation draw, 512 rays per patch) the Monte Carlo
exact estimator gives no average gain over the binned table, only a better median, with a heavy tail; it beats the binned plateau
by ~15 % at 4 draws and ~50 % at 16. A deterministic exact quadrature (A, E3, WL96 designers) closes to reference noise, but its cost on
CK2 was never measured. So "substantially closes" is true for the bias term and for estimators with >= ~16x the relation budget; for
a deployable estimator at the tested budget it is not shown (untested for low-variance exact estimators).

(ii) 0.111 was a low draw within its stated SE. No decision changes.

(iii) Step attribution: the angular (outgoing-direction) share is robust and, if anything, understated by Ra (q256 -> X1c also
moves the incident state and emission to the patch centre, which worsens X1c; E2's factorial gives 0.288 for direction alone).
The position share Rp = 0.157 is mostly the centre point-sampling artefact (genuine sub-patch position 0.082). The label: the frozen
rule correctly yields ANGULAR_PROJECTION_EVIDENCED for draw 86001 (no implementation defect), but it is draw-dependent
(23 / 31 / 10 of 64) and MIXED for the mean or median single draw. Defensible reading: "angular projection BIAS evidenced
(draw-independent, expectation level); relation-sampling VARIANCE not excluded and of comparable size at the deployed budget (MIXED
for a typical draw)". Not INCONCLUSIVE: the bias half is robust; the "not by relation sampling" half is not.

(iv) Image level: arm differences are resolved under any paired or replicate statistic (J3-2). The binned vs exact difference in
expectation is +0.0120 E_H; incident projection +0.0037 (pooled M64_spool -> q256); outgoing + centre +0.0058; position <= 0.0003.
NEXT_DESIGN_TARGET stays UNDETERMINED, but because the two learned-sender seeds disagree, not because of noise.

(v) P0-3 matters in the CK2 ROI image in expectation (it roughly halves the ROI error to the oracle-table floor). Whether it matters
for any single-realisation estimator at the deployed budget is untested.

## 4. P0 verdicts

- **P0-3: CONDITIONALLY_VALID.** Holds for box-A delivery and the CK2 ROI_T image in expectation (pooled over 64 relation draws of
  512 rays per patch, or deterministic quadrature), on one scene / one light / grazing sharp panel / rigid static geometry. Exact
  outgoing direction carries most of the effect; the hit-point adds a second-order 0.08. It does NOT hold for a single relation draw
  at the deployed budget (no average gain, heavy tail). Other receivers (walls/ceiling, sphere, box B) are unresolved. The cost of a
  deployable low-variance exact estimator on CK2 is untested.
- **P0-2, sub-claim "not eliminated by relation sampling": VERIFIED** on the tested workload (pooled M64 1.495-1.544; deterministic
  no-sampling 1.486 in WL96; E2 4,096 draws 1.506-1.529). The sub-claim does not extend to "sampling is not an error source at the
  deployed budget".
- **WL95A FLIGHT_ATTRIBUTION label: CONDITIONALLY_VALID** as the frozen-rule output for draw 86001; as an estimator attribution the
  defensible label is MIXED (bias angular, variance sampling).

## 5. Affected statements

| # | worklog | original claim | defect or limit | corrected interpretation | dependents | action |
|---|---|---|---|---|---|---|
| 1 | WL95A:179 (Answer 2) | first-flight error caused by directional and positional projection, "not by relation sampling" | label draw-dependent; Rs counts for a typical draw; positional share mostly a centre artefact | the systematic bias is outgoing-direction projection; sub-patch position is second order; relation-sampling variance is a comparable co-contributor at the deployed budget | WL96 design premise; P0-2/P0-3 wording | QUALIFY |
| 2 | WL95A:167-171 | FLIGHT_ATTRIBUTION = ANGULAR_PROJECTION_EVIDENCED (Rs does not count) | E0 = one favourable single draw; 23/31/10 of 64 under alternative deployed draws; MIXED under mean/median E0 | frozen-rule output stands for draw 86001; estimator-level reading MIXED (bias angular, variance sampling) | WL96:61 | QUALIFY |
| 3 | WL95A:147, :171 | "further by the exact position (Rp = 0.157)"; "sender-side spatial projection also counts" | X1c is a centre point sample, not a patch projection | genuine patch-average -> hit-point effect 0.082 (E2); image level <= 0.0003 | WL96:63-64 (partly corrected) | QUALIFY |
| 4 | WL95A:149 | loss is the patch- and bin-constant outgoing representation of a position-dependent lobe, not relation sampling | position term second order; sampling claim holds for the bias only | the bin-constant outgoing representation is the dominant bias; patch constancy is minor | WL96 carrier designs | QUALIFY |
| 5 | WL95A:131 | exact hit point reaches 1.02 / 0.111, at reference noise plus SE | 64-draw pool is a low draw (expected ~0.17) | still at noise within the stated SE; closure confirmed by 4,096 draws and deterministic quadrature | WL96:52 | PRESERVE |
| 6 | WL95A:133-138, :166, :175, :178 | image effects "below the 2 x 4,096 spp render resolution"; "render resolution cannot rank the components" | tol 0.0167 is a per-pixel noise magnitude, not an E_H uncertainty (replicate SD <= 4e-5) | the renders resolve every difference >= 0.003; frozen-rule outcomes remain rule outputs; NEXT_DESIGN_TARGET UNDETERMINED stands for seed disagreement | WL96:69, :161, :216; RANK16 criterion (iv) | RETRACT (resolution wording); PRESERVE (labels) |
| 7 | WL95A:172-175 | NEXT_DESIGN_TARGET = UNDETERMINED (render cannot rank) | stated reason wrong | UNDETERMINED in every paired variant because s8601 and s8602 disagree | - | PRESERVE label, QUALIFY reason |
| 8 | WL96:52 | pooled X1 reaches the reference (1.022 / 0.111) | expectation-level only | add "pooled over 64 draws; a single draw has no average gain" | P0-3 | QUALIFY |
| 9 | WL96:61-64 | angular projection evidenced; dominant mechanism outgoing quantisation; X1c -> X1 mostly emission anchor | anchor share itself mostly centre sampling | dominant mechanism confirmed (E2 ~70 %) | WL96 designs | PRESERVE |
| 10 | WL96:69 | image-level relevance on CK2 not demonstrated (+0.0127 below 0.0167) | stored renders resolve it under paired/replicate statistics | demonstrated in expectation (pooled A4 - pooled X1 +0.0120, ~14 SE); untested for a single deployed draw | WL96 "every premise rests on delivery metrics" | QUALIFY |
| 11 | WL96:161, :216 | the 0.0167 tolerance / 2 x 4,096 spp "cannot resolve" a 0.013 effect | conflates tolerance with resolution | 2 x 4,096 spp resolves 0.013 by > 10 bootstrap SE | WL96A (already uses a paired statistic) | RETRACT |
| 12 | P0-3 wording | exact-direction and hit-point evaluation substantially closes the CK2 delivery error | expectation-level; budget-dependent | as in section 4 | - | QUALIFY |

## 6. Unresolved

- Image-level effect of a single-draw (deployed-budget) X1 or any deployable exact estimator: no such render exists.
- Cost of a deterministic / low-variance exact sender estimator on CK2 at matched budget.
- Delivery to receivers other than box A (walls/ceiling, sphere, box B) with an exact sender.
- E2-F6 percentile (3rd) rests on one auditor's simulation, which has a heavier tail than the stored draws.
