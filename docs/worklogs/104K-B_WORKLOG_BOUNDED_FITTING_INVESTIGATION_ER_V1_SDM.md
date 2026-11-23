# Worklog 104K-B — Exhaustive bounded fitting investigation for ER-v1-SDM (2026-11-23 KST)

## Authorization, classification and freeze

The owner's brief asks how far the **unchanged** ER-v1-SDM representation (7 GGX lobes + Lambert, 8 reciprocal per-component 5-knot gates, 178 parameters per material) can be improved by fitting alone: optimization, angular training coverage, a peak-sensitive objective and independent start selection, with a conditional combination and real-scene validation.

- **Batch class.** Suffix batch (dependent follow-up of WL104K / WL104K-A), owner-named WL104K-B. No lobe, gate, knot, GGX function or parameter count changed; nothing is adopted.
- **Branch and host.** `ecro_lr3_adapter`, local RTX 5080. GPU jobs of this batch ran one at a time after the full-scale probe (below); no other job of this batch shared the GPU.
- **Freeze.** Stage A code `92aa4b5` (pre-freeze diagnostic of archived states only). Protocol and implementation `edc85d7` before any new data or fit: [protocol](../../experiments/ecro_wl84_104kb/protocol.json), [code](../../experiments/ecro_wl84_104kb/).
- **Pre-freeze development (in the protocol).** Modules were written and adversarially reviewed by parallel agents, then smoke-tested on scratch roots only (64-lane data, 5–300-step fits, fake evaluation roots). A full-size `train_coverage` precision probe (nothing saved) and full-size fit probes were run. Measured during probes: one 8-start fit takes 101.6 ms/step at 98 % GPU and 4.5 GB; two concurrent fits oversubscribe memory, so fits ran sequentially. The only physical results seen before the freeze were Stage A's, which use archived states.
- **Stage A informed two design choices before the freeze, both recorded in the protocol:**
  - the O1 method;
  - the shoulder term of the E2 objective, because scene first-bounce lanes lie mostly 3–24° from the mirror direction.
- **Amendment 1 (owner-approved after the result was seen).** It changes only the O0 reproduction qualification rule: [amendment1.json](../../experiments/ecro_wl84_104kb/amendment1.json), committed `62cfad7`. Its outcome is reported below. The original-rule decision is preserved.
- **Execution recoveries (disclosed, non-scientific).**
  - **First reboot (06:06).** The PC rebooted during the scene replays. Partial replicate files were preserved in `scenes_interrupted/`, and the scenes were replayed from replicate 0 with identical seeds and code. Rerun vs partial files: max abs difference 1.9e-6 on values ~22 (hard_1) and 4.8e-7 on ~5.8 (iid_1), i.e. ≤ 1e-7 relative; not bitwise equal (float atomics).
  - **Second interruption (crash before 17:53).** The block_186 replay had produced no replicate for more than 5 h before the crash; the cause is unidentified. It was replayed alone from replicate 0 at 324–385 s per replicate.
  - **iid_1 analysis crash.** The iid_1 analysis failed on a division by zero, because O0's noise-debiased RMS is clipped at 0. A fix commit `c8bf685` reports such contrasts as `UNRESOLVED_NOISE_FLOOR`, never labelled. The decision treats a scene in that state as `INCONCLUSIVE_NOISE_FLOOR`, never as SUPPORTED.
  - **`decision104kb.py` changed after the dev decision.** Options to read the amended labels were added, so the file's code hash differs from the value recorded in `dev.json`. No rule changed.

## Implementation Fact

- **Representation.** Every arm fits exactly the WL104K ER-v1-SDM response: `common104k.eval_sdm` / `decode`, with gates projected to [0, 1]. Pre-fit contracts passed:
  - the objective at the archived WL104K states equals WL104K's own to 0.0;
  - the Adam path is bitwise equal to `fit104k.train` over 50 steps;
  - E2 gradients match finite differences to ≤ 6.9e-7;
  - the L-BFGS box and budget logic, and the synthetic truth, are exact.
  All 8 fitted arms passed reciprocity (≤ 4e-15), passivity (pointwise envelope and MC albedo), budget and the 178-parameter check.
- **Materials.** The nine WL104D materials, each fitted separately on its own angular data. The development / held-out cohort marks which materials were inspected historically; it is not zero-shot material generalisation.
- **Starts.** 8 per material: generic + 104911–104917, i.e. the WL104K starts 1–4 plus 4 new ones.
- **New data (all qualified; [data.json](../../results/ecro_wl84_104kb/primary/data/data.json)).** All three lane sets use a scene-independent refined proposal: half the lanes from the WL83 defensive mixture, half from 8 radial annuli around the mirror direction (0, .25, .5, 1, 2, 3, 4.5, 7, 12°). Incidence is stratified-jittered per material: 10 polar strata over 1–88°, 2 azimuths. Every set uses a new initstate and a sequence range ≥ 2^44, disjoint from all earlier batches. All passed the WL104D precision, support and isolation rules.

  | Set | Walk-sequence start | Lanes per probe | Walks per lane | Peak grid (0–8°, 305 points per probe) |
  |---|---|---|---|---|
  | train_coverage | 2^44 | 1024 | 8192 | — |
  | dev | 2^45 | 1024 | 16384 | 65,536 walks |
  | test | 2^48 | 2048 | 32768 | 131,072 walks |

- **Arms.** The control is O0: the WL104K Adam method on the WL104J fit set. Each intervention changes exactly one thing relative to O0.

  | Arm | What changes vs O0 | Training data | Budget |
  |---|---|---|---|
  | O1 | Optimizer: batched projected L-BFGS (memory 20, Armijo, box on gates; a unit freezes after 3 failed line searches) | original | 12,000 evaluations |
  | D | Coverage only | train_coverage | 12,000 steps |
  | E2 | Objective only: el1 + ealb + e_core (< 3°) + e_shoulder (3–12°), region-normalised relative L1, weights 1 | original | 12,000 steps |
  | E1 | No new fit: per material, the start with the lowest dev criterion (el1 + ealb + e_core + e_shoulder on the independent dev set) | — | — |
  | F | Conditional combination (eligibility from dev only) | — | — |

  Synthetic detection controls S_O0 / S_O1 / S_D / S_E2 use an exactly representable truth: GF(R0) with a sharp isotropic outer lobe (α = 0.03) and the WL104K SP gates.
- **Leak lock.** The test set opened only after the hashed dev decision existed. The lock also checks that every fit, the protocol, the code and the data hashes are unchanged. The leak contract passed 6 / 6.
- **Scenes.** The existing WL104K fixed-proposal replay was used, with 49 closures in groups of 8. Identity gates passed in all scenes.
  - Archived: iid_1_L2, hard_1_L1 (the development failure case), iid_159_L2.
  - New, predeclared before any candidate result: N1 = hard_53_L2 (sharp coat over sharp base) and N2 = block_186_L2 (sharp coat over broad base). Both passed the WL104G reference precision gate at 12 × 4096 spp; no fallback was used.
  - HIGH is held fixed from R0, so this is not an end-to-end ECRO validation.

## Measurement

### Stage A — functional attribution of the four archived hard_1 P fits (12k states)
Reference: fresh walks, 65,536 per direction, 0.28 % median half-noise at the mirror.

- **Peak height.** Every start overshoots the mirror-peak height over 14–42° incidence:
  - starts 3–4 (the "good" starts): +1.9 to +6.0 %;
  - starts 1–2 (the "bad" starts): +3.8 to +10.4 %.
  The overshoot grows with incidence.
- **Region signed error (averaged over incidence).**

  | Region | Starts 1–2 | Starts 3–4 |
  |---|---|---|
  | Core (< 3°) | +4.7 / +6.6 % | +2.5 / +2.4 % |
  | Shoulder (3–12°) | ≈ 0 | −1.0 / −1.1 % |
  | Tail (12–30°) | −2.6 to −3.5 % (all starts) | |

- **Good-vs-bad parameter differences.**
  - 90–98 % of each difference lies in the sloppy subspace of the Gauss–Newton response metric (eigenvalue < 1e-6 λ_max); this metric is a surrogate.
  - Along the straight path between a bad and a good state, the actual training objective rises from 0.088 to 0.14–0.19. The states sit in separate basins.
  - The good starts have the slightly lower training objective: 0.0872 vs 0.0881 / 0.0893.
- **Local identifiability.** An actual ±5 % peak change along the least-curvature direction costs +13 to +212 % training objective. Locally the peak is well constrained by the objective.
- **Attribution.** Grouped integrated gradients and group swaps spread the 1.7–4.6 % peak difference over outer amplitude, Lambert / inner split, roughness and gates O0 / O1 / O2. The contributions are large and cancelling; this is not a unique physical attribution, and it is path dependent.
- **Scene lanes.** On the hard_1 scene, reference first-bounce energy by mirror offset is: < 3° 5.1 %, 3–12° 43 %, 12–24° 31 %, > 24° 21 %. P's signed excess is start-dependent only in the core (+3.4 / +5.1 vs +1.2 / +1.0 %).

### Optimization (O0, O1)

| Arm | Stationary units (< 0.5 % change over the last 2,000 steps) | Median final objective | Units terminated early | Time | GPU memory |
|---|---|---|---|---|---|
| O0 | 52.8 % | 0.0247 | — | 1,108 s | 4.5 GB |
| O1 | 4.2 % | 0.0325 | 70 / 72 | 960 s | 4.5 GB |

- O0 projected-gradient median: 0.058.
- O1 terminated units used a median of 4,552 of the 12,000 evaluations. No O1 unit ended below O0; per-material median worsening was +12 to +181 %.

### Independent angular test (O0-paired, 8 starts, material-cluster bootstrap; all contrasts resolved on both reference halves)

| X vs O0 | Peak-profile error | Peak-height error | Across-start peak SD | el1 | ealb | eshape |
|---|---|---|---|---|---|---|
| O1 | +47.5 % [+13, +98] | +176 % | ×9,000 (diverged units) | +40.5 % | +36.7 % | +40.3 % |
| D | −4.7 % [−11.8, +4.1] | −6.0 % | −17.5 % [−33.8, +3.5] | +0.7 % | +25.3 % | −0.4 % |
| E2 | **−53.7 % [−64.0, −31.8]** | **−55.7 %** | −17.8 % [−32.9, −5.4] | +9.9 % [+3.5, +21.9] | +53.5 % | +11.7 % |

- **Per-start consistency.** E2's peak-profile change on test is −48.7 to −55.1 % in all 8 starts (dev: −47.9 to −53.1 %). D's ranges from −0.6 to −6.7 %.
- **Cohorts.** E2's peak-profile change is −57.7 % (development) and −50.8 % (held out).
- **Signed mirror-peak bias** (mean over starts and test probes).
  - O0: hard_1 +9.0 %, iid_159 +4.1 %, hard_76 +5.0 %, hard_53 +4.4 %.
  - E2: hard_1 +3.9 %, iid_159 0.0 %, hard_76 +1.7 %, hard_53 −0.1 %.
- **Across-start SD of peak height, hard_1.** O0 2.15 pp; E2 1.83 pp; D 1.65 pp.
- **Broad preservation (vs O0).** D passes every rule. E2 fails el1 (+9.9 %), the comparison with P_hist (+9.8 %), tail L1 and the per-material rule (iid_0 +44 %). O1 fails all rules.
- **Secondary historical test (WL104D Amendment 1).** Reported in `test.json`.
- **Historical baselines vs O0.** P_hist: el1 +0.5 %, peak-profile −1.5 %. A_hist: el1 +148 %.
- **Synthetic detection controls (truth exactly representable, 4 starts).**

  | Control | Result | Passed |
  |---|---|---|
  | SC_D | peak-profile ratio vs S_O0 0.67 | yes |
  | SC_E2 | ratio 0.48 | yes |
  | SC_E1 | selected start ≤ median start in 8 / 9 materials | yes |
  | SC_O1 | objective ×1.56, el1 ×2.2 | no |

  Even S_O0 does not recover the exactly representable truth: pooled truth-referenced peak-profile error is 0.88–3.5 summed over 180 probes, i.e. about 0.5–2 % per probe, and differs by start.
- **E1 selection on O0.**
  - Not predictive: Spearman ρ > 0 in 6 / 9 materials, median ρ 0.24.
  - Not effective: the selected starts' pooled peak-profile error is −3.0 % vs the median start (the rule needs ≤ −10 %). It is ≤ the training-objective selection, and el1 is preserved.
  - On O1 (descriptive), selection is predictive (median ρ 0.83) and effective (−19.9 %); this reflects O1's diverged starts.
- **Stage F.** No mechanism met dev eligibility (dev peak-profile change: E2 −51.6 % but el1 +9.1 %; D −4.2 %; O1 +44 %), so **F was NOT_RUN**.

### Scenes — relative change of objects stable RMS vs the O0 start-mean (jackknife t(11) 95 % CI)

| Scene | O1 TOTAL | D TOTAL | E2 TOTAL | E1-selected TOTAL |
|---|---|---|---|---|
| hard_1_L1 (development regression case) | +10.8 % REGRESSED | **−13.7 % IMPROVED** | +9.9 % REGRESSED | −8.1 % IMPROVED |
| iid_1_L2 (development) | UNRESOLVED (noise floor) | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| iid_159_L2 (independent) | −15.1 % IMPROVED | +17.6 % REGRESSED | +15.5 % REGRESSED | −5.9 % IMPROVED |
| hard_53_L2 (new N1, independent) | +43.7 % REGRESSED | +9.9 % REGRESSED | +36.3 % REGRESSED | −7.1 % IMPROVED |
| block_186_L2 (new N2, independent) | +83.8 % REGRESSED | −2.9 % NEUTRAL | −0.3 % NEUTRAL | +2.1 % [+0.0, +4.2] NEUTRAL |

- **iid_1 first-bounce (E1 event).** D −7.0 %, E1-selected −12.5 %, O1 +50.0 %, E2 +71.3 %. iid_1's O0 TOTAL error is 0.0016 (A: 0.055), at the replicate noise floor.
- **E1-selected start vs the mean of the other 7 starts.** Similar to the table: hard_1 −9.1 %, iid_159 −6.7 %, hard_53 −8.1 %, block_186 +2.4 %.
- **Historical states (descriptive).**
  - P_hist vs the matched O0 starts: ≈ 0 in every scene.
  - A_hist vs O0: hard_1 −2.9 %, iid_159 +48.6 %, hard_53 +62.8 %, block_186 +611 %.
- **Scene verdicts.** O1, D and E2: SCENE_NOT_SUPPORTED, each with a regression in ≥ 1 scene. E1: INCONCLUSIVE_NOISE_FLOOR. E1 would be SUPPORTED under the rule if iid_1 were resolved: no regression, and improvements in iid_159 and hard_53.

### Qualification

- **O0 reproduction (frozen rule).**
  - Under the frozen rule, every angular label is **UNQUALIFIED**. Reproduction of WL104K P failed: 24 / 36 units above 1e-4 and 2 above 1e-2 (iid_0 start 1 +4.84 %, block_186 start 2 −1.82 %). [decision.json](../../results/ecro_wl84_104kb/primary/decision.json) carries this.
  - Measured labels (not qualified): O1 REGRESSION (control not demonstrated), D INEFFECTIVE, E2 TRADE_OFF, E1 SELECTION_NOT_EFFECTIVE.
- **Amendment 1 → NOT_CONSISTENT.** The labels stay UNQUALIFIED: [result](../../results/ecro_wl84_104kb/primary/amendment1/result.json), [amended decision](../../results/ecro_wl84_104kb/primary/amendment1/decision_amendment1.json). The components were measured as follows:

  | Same-method comparison | Max relative difference | 95th percentile |
  |---|---|---|
  | Batch 4 vs archive and vs each other | 0 (bitwise identical) | 0 |
  | Batch 8 vs O0 | 0 (bitwise identical) | 0 |
  | Batch size 8 vs 4 (R8 starts 1–4 vs R4a / R4b) | 4.84e-2 | 9.25e-3 |
  | O0 vs archive (the target) | 4.84e-2 | 9.25e-3 (identical distribution) |

  - The max criterion passed. The q95 criterion failed (9.25e-3 > 2 × 3.5e-3) because the pooled rerun distribution was diluted by the exact-zero run-to-run components.
  - The predeclared criterion is applied as written. It did not anticipate that the code is deterministic run to run.

## Observation

- **The code is deterministic.** It reproduces itself bitwise at a fixed batch size. The O0–WL104K discrepancy is exactly the batch-size effect: GPU reduction order differs between 4 and 8 starts per batch and is amplified to a different basin. It is not a method difference.
- **E2 halves the angular peak error in every start and cohort**, and removes most of the systematic mirror-peak overshoot (hard_1 +9.0 → +3.9 %). It does so at +10 % broad el1, +54 % albedo error and +12 % shape error, and it worsens 3 of 5 scenes.
- **D leaves angular peak error essentially unchanged** and preserves el1, but raises the albedo error by 25 %. It improves hard_1 (−13.7 %) and worsens iid_159 (+17.6 %) and hard_53 (+9.9 %).
- **Dev-criterion start selection (E1)** does not predict the angular peak error, yet improves the selected start's image error in hard_1, iid_159 and hard_53 (−6 to −8 %) and is neutral in block_186.
- **O1 stalls** on the L1 objective's kinks: 70 / 72 units froze. It also failed its synthetic control.
- **Across-start peak spread** falls by at most ~18 % (D, E2). No arm reaches the −25 % stability rule.

## Interpretation

- **Peak inaccuracy has two separable components.**
  - A systematic peak-height bias. The energy-weighted objective barely weights it, and a region-normalised objective (E2) largely removes it.
  - A start-dependent spread. It comes from separate basins (Stage A barrier) and is not reduced by better coverage, a different objective or a quasi-Newton optimizer.
- **Scene-critical fidelity is not peak-height fidelity.** Removing the bias with E2 moves error into the shoulder, tail and albedo. The scene first bounce weights those regions heavily (hard_1: 74 % of first-bounce energy lies 3–24° off mirror). E2 therefore loses in scenes despite the angular peak gain. The descriptive WL104K-A link between the hard_1 regression and the peak overshoot is not causal in the sense that fixing the overshoot fixes the image.
- **Start choice matters more for images than any of the tested fitting changes.** The dev selection functional (el1 + ealb + region L1) picks starts with better image error in three of four resolved scenes. This is selection among existing basins, not identification of a correct response.
- **Representation adequacy remains unresolved.**
  - With an exactly representable synthetic truth, the O0 contract still misses the peak by about 0.5–2 % per probe, so a residual can be a fitting limit rather than capacity.
  - On physical data, no arm reached accurate, start-stable peaks together with broad fidelity. The evidence neither shows that the 7-GGX family is sufficient nor shows that it must be replaced.

## Unresolved Question

- Whether a single objective can reduce the peak bias without moving error into the shoulder, tail and albedo. Only one predeclared weighting was tested, with no sweep.
- What drives the basin structure, and whether more starts with dev selection converge to a stable image-quality ceiling.
- Why D raises albedo error while preserving el1, and why its scene effect changes sign between scenes.
- The block_186 replay stall before the crash: cause unknown.
- The iid_1 TOTAL comparison is below the replay noise floor at 12 × 1024 spp.
- The q95 component of Amendment 1 and the label qualification are left as they are. Any further rule change is an owner decision.

## Architecture Verdict

All angular labels are **UNQUALIFIED** under both the frozen rule and Amendment 1; the measured labels are given as technical evidence.

| Item | Verdict / evidence class |
|---|---|
| Optimization intervention (O1) | Measured REGRESSION; synthetic control failed (CONTROL_NOT_DEMONSTRATED); scenes NOT_SUPPORTED — independent angular + scene + synthetic |
| Sampling / coverage intervention (D) | Measured INEFFECTIVE for peak (−4.7 %, n.s.); broad preserved; control passed; scenes NOT_SUPPORTED (mixed sign) — independent angular + scene + synthetic |
| Objective intervention (E2) | Measured TRADE_OFF: peak −54 % in every start; broad not preserved; control passed; scenes NOT_SUPPORTED — independent angular + scene + synthetic |
| Independent selection criterion (E1) | Angular: not predictive, not effective. Scenes: improved 3 / 4 resolved, none regressed; verdict INCONCLUSIVE_NOISE_FLOOR (iid_1) — independent angular + scene |
| Combined fitting candidate (F) | NOT_RUN (no dev-eligible mechanism) |
| Peak stability across starts | Not achieved by any arm (best −18 %, rule −25 %) — independent angular |
| Broad angular fidelity | Preserved only by D; E2 +10 % el1 — independent angular |
| Scene fidelity | No fitting intervention supported; start selection is the only consistently non-regressing mechanism — independent scene |
| Remaining representation uncertainty | UNRESOLVED: fitting limits shown on an exactly representable synthetic truth; no evidence that the lobe family must change — synthetic control + independent angular |

Classification of the conclusions:
- **Synthetic controls:** SC_D, SC_E2 and SC_E1 pass (detection demonstrated); SC_O1 fails.
- **Historical hard_1 (descriptive only):** Stage A basins and attribution.
- **Not adopted:** ER-v1-SDM is not adopted.

## Reproducibility

- **Commits.**
  - Stage A: `92aa4b5`
  - Freeze: `edc85d7`
  - Amendment 1 and first resume: `62cfad7`
  - Noise-floor execution fix and second resume: `c8bf685`
  - The branch also received unrelated WL110A commits and a merge (`3e027ec`) from another session; nothing in this batch depends on them.
- **Raw results.** [results/ecro_wl84_104kb/primary](../../results/ecro_wl84_104kb/primary): `data/`, `fits/`, `dev_eval/`, `test_eval/`, `scenes/`, `scenes_new/`, `scenes_interrupted/`, `amendment1/`, `contracts/`, `decision.json`, `status.txt`, `logs/`. Stage A: [results/ecro_wl84_104kb/stageA](../../results/ecro_wl84_104kb/stageA).
- **Data hashes.**
  - train_coverage `d5b51363…`
  - dev `7729e7b8…`
  - dev_peak `18384e47…`
  - test `221f7b2a…`
  - test_peak `e14cef1f…`
- **Environment.** `external/8dna26/.venv-cu128`: Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1, Dr.Jit 0.4.6.
- **Run commands.** `bash chain.sh <root>`, then `resume_a1.sh` and `resume_b186.sh`.
- **Compute.** Fits ≈ 1,100 s per 8-start arm and ≈ 530 s per 4-start synthetic arm. Data generation ≈ 6 min. Amendment reruns ≈ 37 min. Scene replays ≈ 20–50 min per scene (5–8.5 min per 4096-spp replicate).
- **Reviewer package.** [evaluation/104K-B/](../../evaluation/104K-B/) (index.html, figures, manifest).

## Reviewer figures ([evaluation/104K-B/index.html](../../evaluation/104K-B/index.html))

- Stage A (descriptive, archived states):
  - [peak height vs incidence per start](../../evaluation/104K-B/stageA_hard1_mirror_peak_height_vs_incidence_per_start.png)
  - [core / shoulder / tail signed error](../../evaluation/104K-B/stageA_hard1_core_shoulder_tail_signed_error_per_start.png)
  - [straight-path barrier and 5 % cost](../../evaluation/104K-B/stageA_hard1_good_vs_bad_straight_path_objective_peak_and_5pct_cost.png)
  - [grouped integrated-gradient attribution](../../evaluation/104K-B/stageA_hard1_grouped_integrated_gradient_attribution_bad_vs_good.png)
- Optimization and qualification:
  - [objective curves](../../evaluation/104K-B/optimization_training_objective_curves_per_arm_hard1_block186_hard53_iid0.png)
  - [L-BFGS early termination](../../evaluation/104K-B/optimization_lbfgs_early_termination_and_final_objective_vs_adam.png)
  - [O0 reproduction and Amendment 1](../../evaluation/104K-B/qualification_O0_reproduction_of_WL104K_P_final_objective_and_amendment1.png)
- Independent angular test:
  - [per-start pooled errors](../../evaluation/104K-B/test_pooled_peak_profile_error_and_el1_per_start_all_arms.png)
  - [contrasts vs O0 with thresholds](../../evaluation/104K-B/test_relative_change_vs_O0_with_bootstrap_CI_and_predeclared_thresholds.png)
  - [per material](../../evaluation/104K-B/test_peak_profile_error_per_material_and_start_all_arms.png)
  - [regions](../../evaluation/104K-B/test_region_errors_core_shoulder_tail_grazing_per_arm.png)
  - [signed peak height vs incidence](../../evaluation/104K-B/test_mirror_peak_height_signed_error_vs_incidence_per_material_O0_E2_D.png)
  - [instability](../../evaluation/104K-B/test_peak_height_instability_across_starts_per_material.png)
  - [ring residual 0–8°](../../evaluation/104K-B/test_ring_residual_profile_0_to_8deg_hard1_hard53_block186.png)
  - [stage F eligibility](../../evaluation/104K-B/dev_stage_F_eligibility_vs_predeclared_rule.png)
  - [historical secondary test](../../evaluation/104K-B/historical_WL104D_secondary_test_vs_independent_test_contrasts.png)
- Selection and controls:
  - [S_dev vs test peak error](../../evaluation/104K-B/selection_E1_dev_criterion_vs_test_peak_error_per_material.png)
  - [selection bars](../../evaluation/104K-B/selection_E1_selected_vs_median_vs_training_selected_bars.png)
  - [synthetic controls](../../evaluation/104K-B/synthetic_controls_truth_referenced_errors_and_pass_rules.png)
- Scenes:
  - [relative change vs O0](../../evaluation/104K-B/scenes_relative_change_vs_O0_objects_total_and_first_bounce.png)
  - [per-start RMS](../../evaluation/104K-B/scenes_objects_stable_rms_per_start_per_arm_total_and_first_bounce.png)
  - [peak-pixel concentration](../../evaluation/104K-B/scenes_peak_pixel_concentration_per_start.png)
  - Start-mean images with absolute-scale error maps:
    - [iid_1](../../evaluation/104K-B/scene_iid_1_L2_start_mean_images_and_absolute_error_maps.png)
    - [hard_1](../../evaluation/104K-B/scene_hard_1_L1_start_mean_images_and_absolute_error_maps.png)
    - [iid_159](../../evaluation/104K-B/scene_iid_159_L2_start_mean_images_and_absolute_error_maps.png)
    - [hard_53](../../evaluation/104K-B/scene_hard_53_L2_start_mean_images_and_absolute_error_maps.png)
    - [block_186](../../evaluation/104K-B/scene_block_186_L2_start_mean_images_and_absolute_error_maps.png)

Additional observation from the signed-peak figure (descriptive): O0's hard_1 mirror-peak bias grows from about +3 % at 5° incidence to +20–25 % at 70–80°, then turns to −25 % at 88°. E2 flattens it below ~3 % up to 65° but not at 75–85°.
