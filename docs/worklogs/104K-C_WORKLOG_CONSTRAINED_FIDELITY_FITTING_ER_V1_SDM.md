# Worklog 104K-C — Constrained fidelity fitting and ER-v1-SDM closure (2026-11-24 KST)

## Authorization, classification and freeze

The owner's brief, "WL104K-C — Constrained Fidelity Fitting and ER-v1-SDM Closure", authorised one final bounded fitting-mechanism investigation. The representation was unchanged ER-v1-SDM: 7 GGX lobes + Lambert, 8 reciprocal per-component 5-knot gates, 178 parameters per material. The brief's stages were:

- Stage A: baseline / numerical qualification.
- Stage B: ONE physically motivated constrained method.
- Stage C: synthetic exactly representable truth.
- Stage D: a new disjoint angular test plus scenes.
- Stage E: architecture boundary assessment.

**Classification and setup**
- **Batch class.** Suffix batch, a dependent follow-up of WL104K / WL104K-A / WL104K-B, owner-named WL104K-C. No lobe, gate, knot, GGX function or parameter count changed. Nothing is adopted.
- **Branch and host.** `ecro_lr3_adapter`, local RTX 5080, one GPU process at a time.
- **GPU cap.** The owner asked that this batch use at most 50 % of the GPU, valid for this task only. [gpu_duty.ps1](../../experiments/ecro_wl84_104kc/gpu_duty.ps1) suspends and resumes this batch's python processes, which are identified by their command line. Two further measures apply:
  - The controller estimates the batch's own GPU share as total utilisation minus the utilisation measured in late off-windows (other applications ran concurrently, mean 21 %). It targets 0.45.
  - Torch memory fraction 0.5.

  Over the 10.6 h run the batch's own share had median 0.22, mean 0.25 and 90th percentile 0.43 (18,157 control windows; [log](../../results/ecro_wl84_104kc/gpu_duty/duty_log.csv)). Suspension does not change numerics: the Stage A refits are bitwise equal to the unthrottled WL104K / WL104K-B states (below).

**Freeze points**
- **Stage A.** Protocol and diagnostics `ed69ee1`, frozen before the Stage A runs: [protocol_stageA.json](../../experiments/ecro_wl84_104kc/protocol_stageA.json).
- **Main protocol and implementation.** `02274e9`, frozen before any independent evidence: [protocol.json](../../experiments/ecro_wl84_104kc/protocol.json). It contains the CR method, controls, synthetic arms, the new test_c set, the new scenes N3 / N4, labels, decision rules and stops.
- **`cab31e7`, before any independent evidence.** The test evaluation additionally stores ring-residual profiles, ring radii and probe polar angles in `arrays.npz`. These feed figures only; no metric, rule or label changed.

**Design history (disclosed in the protocol)**
- CR was designed on the WL104K-B DEV set only. A design run on DEV, `CR_v1`, gave the following against O0:
  - peak-profile error −6.5 %, core −6.8 %, peak height −8.8 %;
  - el1 +0.2 %, ealb +1.0 %, eshape +0.2 %, shoulder −1.7 %, tail +0.9 %, grazing +0.7 %;
  - across-start instability +1.2 %;
  - 95.8 % of units feasible.
- An earlier smoke setting, lr0 3e-3, violated the constraints by 57–139 % at once. lr0 was therefore set to 3e-4 (O0's terminal rate) before the freeze. No constant was changed after the DEV run.
- One 800-step pre-freeze probe was launched with a relative script path. The GPU supervisor did not match it, so it ran unthrottled for about 100 s. It was a design probe and produced no evidence.

**Execution recoveries (non-scientific, disclosed)**
- **Owner-requested pause, about 02:05–02:32.** The owner asked to pause. The supervisor was stopped and all batch processes were suspended (NtSuspendProcess), then resumed when the owner said to continue. Stage A R4 was running and finished bitwise equal to the archive.
- **Figure-script commit `ae82533` while Stage A ran.** The main chain therefore recorded `ae82533` as its start commit. `figures104kc.py` is not part of any evaluation-code hash.
- **N4 never attempted in the first pass.** The main chain's "already attempted" test for new-scene references searched the whole reference manifest for the scene name. The predeclared `order` list matched, so the first pass skipped the predeclared N4 `test83_block_161_L1` and the fallbacks, and ran with N3 only.
  - The fix is in a separate copy, [chain_n4fix.sh](../../experiments/ecro_wl84_104kc/chain_n4fix.sh), which checks the manifest's `scenes` record (`5f1d046`, `b66d5f2`, `8c7d0b2`).
  - `chain.sh` was briefly edited in place and restored byte-identical within about two minutes. The running bash reads scripts by offset, and its parser was inside the scene loop, which it had already read, so no command was affected.
  - [chain_recover_n4.sh](../../experiments/ecro_wl84_104kc/chain_recover_n4.sh) waited for the first pass to exit and preserved its decision as `decision_first_pass_before_N4_recovery.json`. It then ran only the N4 reference, the N4 replay / analysis and the decision.
  - N4 qualified. No protocol, data, rule or evaluation code changed. The first-pass and final decisions carry the same labels.
- **WL104K-B test lock bypassed by design.** `decision104kb.py` changed in WL104K-B (`c8bf685`), so WL104K-B's own test-lock guard refuses to open its test set. WL104K-C reads that set as regression evidence with the guard off, after authenticating every file against `evaluation/104K-B/manifest.json`.

## Implementation Fact

- **CR, constrained refinement** ([fit104kc.py](../../experiments/ecro_wl84_104kc/fit104kc.py)).
  - **Objective.** Per unit (material, start), minimise the mean < 3° specular-core error F, relative to the warm start.
  - **Constraints.** For every training probe (20) and k ∈ {el1, ealb, e_shoulder (3–12°), e_tail (≥ 12°)}: g_k,p ≤ max((1 + τ_k) g_k,p(x0), g_k,p(x0) + δ_k), with τ = 2 / 5 / 5 / 5 % and δ_ealb = 2e-4.
  - **Solver.** Powell-Hestenes-Rockafellar augmented Lagrangian, solved by Adam: 6000 steps, 12 outer iterations, ρ 10 → ×5 up to 1e4.
  - **Returned state.** The best feasible checkpoint; the warm start if none is feasible. Gate knots are projected to [0, 1] after every step.
  - **Data.** Training data only (train_original). Constraint multipliers are produced by the method, not tuned.
- **Controls** (same 6000-step continuation from the same warm start unless noted):

  | Arm | What it is |
  |---|---|
  | O0C | Original objective (compute control) |
  | E2W | WL104K-B E2 weighted sum |
  | CR_R4 | CR from the 4-start (historical) warm start for starts 1–4 |
  | E2_B | WL104K-B E2 fit, read from the archive |
  | P_hist | WL104K archive, descriptive |

  The warm start is the Stage A R8 refit, bitwise equal to WL104K-B O0, at a fixed batch shape of 72 units (9 materials × 8 starts).
- **Synthetic arms.** CR_S / O0C_S / E2W_S start from S_O0r (train_original lanes). CR_SD / O0C_SD start from S_Dr (WL104K-B coverage lanes). All have 36 units, the same constants, and the WL104K-B exactly representable truth: GF(R0), a sharp isotropic outer lobe α = 0.03, SP gates, noise seed 1041340.
- **New independent angular set, test_c** ([data.json](../../results/ecro_wl84_104kc/primary/data/data.json)).
  - Direction seed 1043330.
  - Lanes: 2048 per probe × 32,768 walks, sequence start 2^50.
  - Peak grid: 0–8°, 131,072 walks, sequence start 2^52.
  - Qualified: precision, support, isolation against train_original, WL104K-B train_coverage, dev and test (0 shared directions), and a registry with no overlap. Generation took 547 s.
- **Contracts.** Pre-fit, 10 / 10 ([prefit.json](../../results/ecro_wl84_104kc/primary/contracts/prefit.json)):
  - response identity with `fit104kb.Data` = 0.0;
  - E2 identity;
  - region partition;
  - AL gradient vs float64 finite differences ≤ 7e-10;
  - warm start feasible at F = 1;
  - zero learning rate returns the warm start in every arm;
  - synthetic targets bitwise equal to WL104K-B;
  - warm-R4 layout.

  Fitted, all 9 arms pass the unchanged `tests104kb.py --fitted`: reciprocity ≤ 5e-16, pointwise envelope, MC albedo / passivity, 178 parameters, gates in [0, 1], budget. Leak, 3 / 3: test_c refuses to open without the hashed fit decision, with a tampered one, or with changed states.
- **Scenes.** `scenes104kb.FixedKB` fixed-proposal replay: e1, frozen-R0 e2, TOTAL = e1 + e2 + R0 HIGH. **This is not end-to-end ECRO.** Identity gates passed in all 7 scenes.
  - New, predeclared and never rendered before: N3 `test83_hard_76_L1` and N4 `test83_block_161_L1`. Both passed the WL104G precision gate at 12 × 4096 spp; no fallback was used.
  - Regression (previously examined): hard_1_L1, iid_159_L2, hard_53_L2, block_186_L2, iid_1_L2. CR / O0C were replayed; O0 / E2_B / P_hist were read from authenticated WL104K-B replicates, and O0 starts 0–1 were replayed again as a determinism check (passed).

## Measurement

### Stage A — numerical qualification ([stageA.json](../../results/ecro_wl84_104kc/primary/stageA/stageA.json)): BASELINE_QUALIFIED

**Reproduction**

| Check | Result |
|---|---|
| Q1: R4 (historical 4-start batching) vs WL104K P | 36 / 36 units bitwise equal |
| Q2: R8 vs WL104K-B O0 | 72 / 72 bitwise equal |
| Q4: S_O0r / S_Dr vs WL104K-B S_O0 / S_D | 36 / 36 each bitwise equal |

**Attribution**
- The 36 units in which WL104K-B O0 differs from the archive are exactly the units in which R8 differs from R4. Unexplained: 0.
- The batch-shape difference in final objective has median 0.06 % and max 4.8 % (DEV: median |difference| / across-start SD 0.017, max 0.70; one unit above 0.5).
- **Mechanism.** Every forward quantity (decoded parameters, lane responses, group terms, objective) is bitwise identical for any layout. Only the gradient differs with batch shape, through the reduction over units: max relative 5.0e-5 at shape 36 and 7.9e-6 at shape 9.
- At a fixed shape of 72, swapping other units' content (seeds 104918–104921) or reversing start order leaves every step bitwise identical (36 / 36 and 72 / 72 units). The L72 trajectory equals the WL104K-B O0 history.
- At shapes 36 and 9, trajectories diverge from step 3–17. After 500 steps the objective differs by a median 1.6 % and a max 29 %.
- **Frozen protocol.** All arms are compared paired, at fixed shape 72 (physical) and 36 (synthetic). The batch-size numerical null for test contrasts is |R4 − O0| and |CR_R4 − CR|; for e_prof it is 0.20 %.

### Training (fit decision hashed before test_c; [fit_decision.json](../../results/ecro_wl84_104kc/primary/fit_decision/fit_decision.json))

- **CR feasibility.** The feasible fraction per checkpoint rose 0.06 → 0.96: 0.06, 0, 0.19, 0.47, 0.69, 0.74, 0.89, 0.94, 0.97, 0.96, 0.96, 0.96.
- **CR result.** 69 / 72 units moved from the warm start. The returned training core error has median F 0.945. Units with F ≤ 0.9: **16 / 72 (22 %)**.
- **Change in training terms** (median over units): core −5.5 %, ealb +3.9 %, tail +0.9 %, el1 +0.3 %, shoulder +0.2 %.
- **Convergence indicators.** F change over the last two checkpoints: median 0.06 %. Final ρ reached its maximum of 1e4. The median projected Lagrangian gradient is 17.2, i.e. not stationary in the KKT sense.
- **Controls.** O0C's criterion changed by 0.006 % over the last two checkpoints; E2W's by 0.04 %.
- **Costs** (wall clock under the 50 % cap, shared GPU):

  | Run | Wall time |
  |---|---|
  | Each 8-start physical continuation (CR / O0C / E2W / CR_R4) | 1258–1296 s |
  | Each synthetic arm | 597–616 s |
  | Stage A R4 (12k steps, 4-start batching) | 4231 s |
  | Stage A R8 | 2532 s |
  | Stage A S_O0r / S_Dr | 1202 / 1198 s |
  | test_c evaluation | 32 s |

### Independent angular test_c (paired with O0 over 8 starts; median and 95 % material-cluster bootstrap; [test.json](../../results/ecro_wl84_104kc/primary/test_eval/test.json))

Pooled relative change vs O0 (%):

| Metric | CR | O0C | E2W | E2_B |
|---|---|---|---|---|
| peak-profile e_prof | **−7.0** [−10.5, −4.6] | +0.02 [−0.04, 0.06] | −50.9 [−60.5, −32.4] | −51.5 [−61.5, −32.5] |
| peak height e_h | −8.9 [−14.1, −6.5] | +0.06 | −56.3 | −55.6 |
| core L1 (< 3°) | −7.3 [−10.7, −4.6] | +0.03 | −53.3 | −53.9 |
| shoulder L1 (3–12°) | −1.5 [−4.4, −0.1] | −0.14 | −19.4 | −18.8 |
| tail L1 (≥ 12°) | +0.9 [0.6, 1.5] | −0.19 | **+13.4** | **+20.1** |
| broad el1 | +0.15 [−0.04, 0.38] | −0.12 | +3.0 [0.1, 8.5] | **+9.3** |
| grazing el1 | +0.6 [0.1, 1.0] | −0.15 | −2.4 | +1.8 |
| albedo ealb | +0.8 [−0.1, 2.0] | −0.24 | **+55.4** [3.5, 103] | **+55.6** |
| shape eshape | +0.09 | −0.11 | +4.3 | **+10.8** |
| across-start peak-height SD | +1.3 [−4.6, 4.9] | −0.1 | **−42.9** | −23.2 |

- **Numerical null.** CR's e_prof change is resolved: |−7.0 %| > 3 × the 0.20 % null. O0C's change is not.
- **Labels** (predeclared):
  - **CR: PRESERVED_NO_PEAK_GAIN.** BROAD_PRESERVED holds on all 8 clauses, including el1 vs P_hist −0.45 % and no material el1 > +10 %. PEAK_GAIN (≤ −15 %) is not met; COMPUTE_CONTROL vs O0C (≤ −15 %) is not met.
  - **O0C: PRESERVED_NO_PEAK_GAIN.**
  - **E2W: TRADE_OFF.** PEAK_GAIN and STABILITY_GAIN hold; it fails ealb, tail and the material-el1 clause.
  - **E2_B: TRADE_OFF.** It fails el1, ealb, eshape, tail, vs-P_hist and material clauses.
- **Peak-grid ring residual** (F / R − 1, mean of 20 probes, median over starts). O0 overshoots the mirror peak by up to ≈ 10 % (hard_1), 7 % (iid_159) and 6 % (hard_76, hard_53). CR lowers the overshoot by roughly 1 point at 0–1°; E2_B lowers it by half or more ([figure](../../evaluation/104K-C/test_c_peak_ring_residual_profile_per_material_O0_CR_E2B_Phist_line.png)).

### Synthetic exactly representable truth (test_c lanes, noise-free truth)

Ratio of error to the warm start:

| Arm | e_prof | e_h | core L1 | el1 | ealb |
|---|---|---|---|---|---|
| CR_S | 0.92 (per start 0.88–0.97) | 0.87 | 0.92 | 1.015 | 1.036 |
| O0C_S | 0.98 | 0.98 | 0.98 | 0.97 | 0.97 |
| E2W_S | 0.68 | 0.60 | 0.69 | 1.066 | 1.046 |
| CR_SD | 0.89 | 0.85 | 0.88 | 1.027 | 1.011 |

- **Recovery rules.** SYN_RECOVERY_CR_S, CR_S vs O0C_S and CR_SD all fail. The rule is e_prof ≤ 0.8 at el1 ≤ 1.05.
- **Constraint feasibility of the truth.** The truth parameters satisfy the arm's own training bounds in only **5 / 36 units** (CR_S) and **3 / 36** (CR_SD).
- **Data sufficiency** (descriptive). CR_SD's e_prof against the truth is 0.59 × CR_S's.
- **Warm start vs truth.** The warm start S_O0r is far from the truth in the core: pooled test e_prof 1.59 and e_h 2.50, versus the coverage-trained S_Dr's 0.96 / 1.48.

### Scenes (relative change of the start-mean objects stable RMS vs O0; replicate jackknife t(11) 95 % CI; [figure](../../evaluation/104K-C/scenes_objects_rms_change_e1_e2_total_all_scenes_dotplot.png))

| Scene | CR TOTAL | CR e1 | CR e2 | O0C TOTAL | CR vs O0C TOTAL | E2_B TOTAL |
|---|---|---|---|---|---|---|
| **N3 hard_76_L1 (new)** | **−1.52 [−1.78, −1.26] IMPROVED** | −1.56 | +1.09 | +0.03 | −1.55 IMPROVED | −1.68 IMPROVED |
| **N4 block_161_L1 (new)** | **−0.27 [−0.34, −0.21] IMPROVED** | −3.21 | +2.76 | −0.22 IMPROVED | −0.05 NEUTRAL | +3.52 REGRESSED |
| hard_1_L1 | −1.27 [−1.61, −0.93] IMPROVED | −1.31 | +1.95 | −0.00 | −1.27 IMPROVED | +9.86 REGRESSED |
| **iid_159_L2** | **+5.25 [5.16, 5.35] REGRESSED** | +1.50 | **−19.2** | −0.40 IMPROVED | +5.67 REGRESSED | +15.5 REGRESSED |
| hard_53_L2 | +0.56 NEUTRAL | +2.93 REGRESSED | −21.0 | −0.04 IMPROVED | +0.60 NEUTRAL | +36.3 REGRESSED |
| block_186_L2 | +1.75 [0.25, 3.25] NEUTRAL | −0.26 | +3.1 | +0.26 | +1.48 | −0.3 NEUTRAL |
| iid_1_L2 | UNRESOLVED_NOISE_FLOOR (TOTAL, e2) | +1.96 | — | unresolved | unresolved | unresolved |

- **Absolute objects RMS for iid_1** (noise-debiased TOTAL; O0's e2 is clipped at 0): O0 0.00162, CR 0.00181, O0C 0.00161, E2_B 0.00458, P_hist 0.00158.
- **Scene verdicts** (predeclared):
  - **CR: SCENE_NOT_SUPPORTED.** iid_159 TOTAL is REGRESSED, even though both new independent scenes are IMPROVED.
  - **O0C: SCENE_SUPPORTED_IMPROVED.** Effects are ≤ 0.4 %.
  - **E2_B: SCENE_NOT_SUPPORTED** (4 scenes regressed).
- **Costs.**
  - Scene replays and analyses: 18–73 min per scene under the cap (6 arms × 12 replicates for new scenes; CR / O0C plus a determinism check for merged scenes).
  - New references: 133 s for N3 and about 2.3 min for N4.
  - Wall clock: Stage A 01:46–04:35, main chain 04:36–11:07, N4 recovery 11:08–12:23.

### Decision ([decision.json](../../results/ecro_wl84_104kc/primary/decision.json))

| Item | Result |
|---|---|
| qualified | true |
| Optimization | converged by the predeclared rule (feasible 0.958 ≥ 0.9, F change 0.0006 ≤ 0.005) |
| Objective feasibility | **LIMITED_FEASIBLE_IMPROVEMENT** (22 % < 50 %) |
| Synthetic recovery | all three rules fail |
| Independent angular (CR) | **PRESERVED_NO_PEAK_GAIN** |
| Stability | STABILITY_GAIN only for E2W |
| Scene (CR) | **SCENE_NOT_SUPPORTED** |
| **Q1** | **NO_UNDER_TESTED_CONDITIONS** |

### Post hoc, descriptive only (after every label and the decision; [posthoc104kc.py](../../experiments/ecro_wl84_104kc/posthoc104kc.py))

These analyses change no rule, state or label.

- **Which constraint excludes the synthetic truth** ([json](../../results/ecro_wl84_104kc/primary/posthoc/truth_constraint_attribution.json)).
  - **Only ealb.** The truth violates the per-probe albedo bound in 31 / 36 (CR_S) and 33 / 36 (CR_SD) units; median worst-probe excess 86–88 %. It violates el1, shoulder or tail in no unit.
  - **Mean comparison.** On average the truth has *lower* training errors than the warm start in every term:
    - median probe ratio el1 0.77, ealb 0.78, shoulder 0.37, tail 0.86, core 0.07;
    - mean ealb 4.8e-4 vs 7.4e-4.
  - **Per-probe excess.** On individual probes, however, the warm start has fitted the noisy albedo targets better than the truth does.
- **Signed cancellation in iid_159** ([json](../../results/ecro_wl84_104kc/primary/posthoc/scene_signed_cancellation.json); raw start-mean luminance on the objects mask, not noise-debiased).
  - In O0, the e1 error and the remainder error (TOTAL − e1 = fixed e2 + R0 HIGH) are negatively correlated pixelwise, at −0.43. Every other scene lies between −0.06 and +0.05.
  - CR weakens the correlation to −0.34 and raises the raw e1 RMS by 2.8 %; raw TOTAL RMS goes 0.00393 → 0.00421.
  - E2_B removes the correlation (−0.02); its raw TOTAL RMS is 0.00487.

## Observation

- Constrained refinement under the frozen preservation bounds moves the specular core of almost every unit (69 / 72). Every broad, albedo, shoulder and tail metric stays within tolerance on independent data, so preservation transfers from training to test_c. The core gain stays small: about −7 % on test, −5.5 % on training.
- The same extra compute with the original objective (O0C) changes nothing measurable. The CR gain is therefore attributable to the constrained objective, not to extra steps.
- The weighted-sum continuation (E2W) reproduces WL104K-B's E2 trade-off from a different warm start: core error halves, and albedo +55 % and tail +13 % breach the rules. It also makes peak height start-stable (−43 % across-start SD), which no other arm does.
- In synthetic data the predeclared feasible set excludes the exact truth in 31 / 36 units, through the per-probe albedo bound alone. CR's recovery (e_prof 0.92) is far from the truth, while E2W reaches 0.68 at el1 1.07.
- The scene effects of CR are small and of mixed sign:
  - improved: N3, N4 and hard_1, by 0.3–1.5 %;
  - regressed: iid_159, by 5.3 %, although its e2 RMS falls 19 %;
  - in hard_53 e1 regresses by 2.9 % while TOTAL is neutral.

## Interpretation

**Supported**
- Under the frozen constants, constrained refinement finds a "do-no-harm" core improvement that generalises to an independent angular set. Its size is far below the 15 % PEAK_GAIN rule. Training feasibility reaches 96 %, but only 22 % of units obtain ≥ 10 % core reduction.
- **Synthetic case.** Here the representation can express the truth exactly, yet the predeclared constraint set excludes that truth in most units: the per-probe albedo bound is relative to a warm start that has partly fitted target noise. So, at least in the synthetic case, the limited gain is a property of the **constraint formulation relative to a noisy warm start**, not of the representation's capacity. Whether the same holds for the physical materials is not measured, because physical truth parameters do not exist.
- **iid_159 regression.** It is consistent with removal of a pre-existing **signed cancellation** between the first-bounce error and the fixed e2 / HIGH error. Evidence: e1 +1.5 % alone cannot produce +5.3 % TOTAL, and the e1–remainder correlation is −0.43 → −0.34. This analysis is post hoc and on raw, not debiased, RMS. A TOTAL-image regression can therefore come from a locally more accurate BSDF. The scene rule, which scores TOTAL only, cannot distinguish the two.

**Not supported**
- A fitting-only route by which ER-v1-SDM improves sharp-peak accuracy by the predeclared margin while preserving broad, albedo and scene fidelity. Neither the constrained method (insufficient peak gain, scene regression) nor the weighted-sum controls (broad / albedo trade-off) achieve it.

## Unresolved Question

- **Formulation or representation?** Is the residual peak overshoot of the physical fits a representation limit, or a consequence of noise-relative per-probe bounds? A noise-aware bound, for example relative to a noise-debiased target error, was not predeclared and was not run.
- **KKT stationarity.** CR's projected Lagrangian gradient (median 17) shows the method stopped at the budget at ρ_max rather than at a KKT point. A larger budget or a different solver might change the 22 % fraction; this was not tested (bounded batch).
- **Scene scoring.** Scenes use fixed-proposal replay with R0 HIGH. A BSDF change that removes a compensating error can lower TOTAL fidelity, and the TOTAL-only rule mixes BSDF accuracy with cancellation. End-to-end ECRO is not measured.
- **Start stability.** The across-start peak instability that E2W removes is untouched by CR. Its cause remains unattributed.

## Architecture Verdict

**ER-v1-SDM is not adopted.** Conclusions, kept separate:

| Question | Conclusion |
|---|---|
| Numerical qualification | **QUALIFIED.** Baseline reproduced bitwise. The WL104K-B discrepancy is fully explained by batch-shape gradient reduction; paired fixed-shape protocol frozen. |
| Optimization adequacy | **Converged by the predeclared rule.** Not KKT-stationary (ρ at maximum, median projected gradient 17). |
| Objective feasibility | **LIMITED_FEASIBLE_IMPROVEMENT** (16 / 72 units ≥ 10 % core reduction). |
| Synthetic representation adequacy | **Not demonstrated by CR** (e_prof 0.92 / 0.89, rules fail). The constraint set excludes the representable truth in 31–33 / 36 units via ealb, so the synthetic failure reflects the constraint formulation and is not evidence of insufficient capacity. |
| Independent angular fidelity | **CR PRESERVED_NO_PEAK_GAIN** (−7.0 % peak, all broad clauses preserved). **E2W / E2_B TRADE_OFF.** |
| Scene fidelity | **CR SCENE_NOT_SUPPORTED** (iid_159 +5.3 % TOTAL; both new independent scenes improved, by 1.5 % and 0.3 %). Fixed-proposal replay, not end-to-end ECRO. |
| Across-start stability | No gain for CR; STABILITY_GAIN only for E2W, a trade-off arm. |
| Remaining architecture uncertainty | Representation adequacy **UNRESOLVED**. No experiment here isolates a capacity limit. Fitting failure alone is not taken as insufficiency (WL104K-C invariant). |

**Completion question 1.** "Can ER-v1-SDM be fitted to achieve improved sharp-peak accuracy while preserving broad angular, albedo and real-scene fidelity?" Under the tested conditions, **no** (NO_UNDER_TESTED_CONDITIONS). The constrained method preserved broad and albedo fidelity on independent angular data but improved the peak by only 7 % and regressed one scene. The weighted-sum objectives improved the peak by about 50 % but broke broad / albedo preservation.

**Completion question 2.** "Does the remaining evidence justify preserving this representation for integration work, or opening a separate representation-design investigation?" Evidence statements only; the decision belongs to the owner:
- (a) No tested fitting route meets the joint peak-plus-preservation criterion.
- (b) The available evidence does not isolate a representation capacity limit:
  - the synthetic truth is exactly representable but excluded by the constraint set;
  - WL104K-B's E2 shows the representation can halve the peak error at a broad / albedo cost.
- (c) Scene-level utility of BSDF-only improvements is confounded by signed cancellation with fixed e2 / HIGH errors.
- (d) Gated sampling / pdf for ER-v1-SDM is still undesigned (WL104K-A), and integration readiness is unchanged.

## Reproducibility

- **Commits.**
  - `ed69ee1`: Stage A freeze.
  - `02274e9`: main freeze.
  - `cab31e7`: figure arrays only.
  - `ae82533`: figure script; recorded as the main-chain start commit.
  - `5f1d046`, `b66d5f2`, `8c7d0b2`: N4 execution recovery.
  - The worklog commit adds the post hoc script, package script and figure script edits.
- **Protocol sha256.** Recorded in every result JSON; [protocol.json](../../experiments/ecro_wl84_104kc/protocol.json).
- **Seeds.**
  - test_c: directions 1043330; walks 1043331 / 1043332.
  - Scenes: N3 1071900, N4 1072900 (masks at seed + 990).
  - Bootstrap 1043350; tests 1043360; fitted-contract RNG 1041360 / 1041361.
  - Synthetic noise 1041340 (WL104K-B).
  - Swap content seeds 104918–104921.
- **Batching.** Physical fits 72 units (start-major, 9 × 8); synthetic 36; Stage A R4 in the historical 4-start layout.
- **Budgets.** Stage A 12,000 steps. Continuations 6000 steps, 500 inner steps per outer iteration.
- **Environment.** Local RTX 5080 (16 GB) under the 50 % own-share cap; `external/8dna26/.venv-cu128` (torch cu128, Mitsuba 3.5.1 / Dr.Jit for scenes).
- **Run.**
  - [chainA.sh](../../experiments/ecro_wl84_104kc/chainA.sh), then [chain_after_A.sh](../../experiments/ecro_wl84_104kc/chain_after_A.sh) → [chain.sh](../../experiments/ecro_wl84_104kc/chain.sh), then [chain_recover_n4.sh](../../experiments/ecro_wl84_104kc/chain_recover_n4.sh) → [chain_n4fix.sh](../../experiments/ecro_wl84_104kc/chain_n4fix.sh).
  - Post hoc: `posthoc104kc.py <root>` and `posthoc104kc.py <root> scenes`.
  - Figures: `figures104kc.py`; package: `package104kc.py`.
- **Raw results.** `results/ecro_wl84_104kc/`, 225 files. Hashes in [evaluation/104K-C/manifest.json](../../evaluation/104K-C/manifest.json). Status log: [status.txt](../../results/ecro_wl84_104kc/primary/status.txt).
- **Monte Carlo uncertainty.**
  - Angular: 95 % material-cluster bootstrap over 8 starts and 9 materials, plus the batch-size numerical null.
  - Scenes: replicate jackknife t(11) over 12 × 4096-spp replicates. Reference readiness: WL104G half-width rule.
  - Synthetic: noise-free truth.

## Reviewer figures ([evaluation/104K-C/index.html](../../evaluation/104K-C/index.html))

**Stage A**
- `stageA_batch_layout_bitwise_identity_by_computation_bar.png`: which computations stay bitwise identical under each batch layout.
- `stageA_batch_shape_first_divergence_step_histogram.png`: step at which trajectories diverge under a different batch shape.
- `stageA_dev_batch4_vs_batch8_difference_vs_across_start_sd_scatter.png`: batch-size difference vs across-start SD on DEV.

**Training**
- `training_CR_feasibility_and_core_error_by_checkpoint_line.png`: CR feasibility and training core error per checkpoint.
- `training_CR_returned_core_error_ratio_per_material_start_heatmap.png`: returned core-error ratio per material and start.

**Independent angular test (test_c)**
- `test_c_pooled_change_vs_O0_per_metric_CR_O0C_E2W_E2B_dotplot.png`: pooled change vs O0 per metric, with numerical null and rule thresholds.
- `test_c_{e_prof,el1,ealb}_per_material_per_start_O0_CR_E2B_Phist_dots.png`: per-material, per-start values.
- `test_c_peak_ring_residual_profile_per_material_O0_CR_E2B_Phist_line.png`: peak overshoot vs offset from the mirror direction.

**Synthetic**
- `synthetic_truth_recovery_ratio_vs_warm_start_bar.png`: error against the noise-free truth, relative to the warm start.

**Scenes**
- `scenes_objects_rms_change_e1_e2_total_all_scenes_dotplot.png`: change of objects RMS per event and scene.
- `scenes_{e1,total}_absolute_error_maps_O0_CR_E2B_shared_scale.png`: absolute HDR error, one shared sqrt scale with a real-unit colour bar.
- `scenes_{e1,total}_relative_signed_error_maps_O0_CR_E2B_shared_scale.png`: (X − reference) / reference, one shared symmetric scale.
