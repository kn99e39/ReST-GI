# Worklog — CT-AK Cold-frame Reconstruction Refinement (WL111A, 2026-11-26)

> Append-only execution record. **Batch class: suffix WL111A.** It is a bounded follow-up to WL111: it attributes
> WL111's residual cold-frame quality deficit and corrects it, under WL111's preserved contracts. Owner brief
> "WL111A — CT-AK Cold-frame Reconstruction Refinement", 2026-11-26.
> - Branch `Denoise_Reconstructor`; no new branch or worktree.
> - Denoising Track only. ECRO transport and renderer work belong to another session and branch, and this batch does
>   not touch them.
> - WL111's evidence and its NOT_SUPPORTED verdict are preserved unchanged.

## Authorization and scope (owner brief)

**Question.** What limits the cold-frame (no temporal history) quality of the lightweight CT-AK reconstructor? Can one
evidence-supported correction improve its quality–performance frontier over R3 without losing its low reconstruction
cost?

**Stages.**
- A: baseline recovery, including bias / variance separation.
- B: causal attribution among four hypotheses:
  - H1, guide information (4×4 vs 8×8 sub-pixel guides);
  - H2, structured operator (3 × 5×5 à-trous vs a dense 15×15 kernel);
  - H3, the compact shared predictor;
  - H4, cold / accumulated training-distribution interference.
- C: select ONE intervention, declared before final training; INCONCLUSIVE if no actionable bottleneck exists.
- D: one candidate S111A, at least two realizations, frozen protocol.
- Final evaluation:
  - arms R0, R3, S111 (both realizations) and S111A, plus any required matched control;
  - views: historical V0/H1–H5 and K1–K6, newly sealed views, and qualified edited geometry states, all as cold
    frames;
  - targets T_ECRO (primary) and T_PHYSICAL (descriptive);
  - Routes A and B against R3 at 4 spp in both realizations, measured cost, absolute contract C, reviewer artifacts in
    `evaluation/111A/`.

**Fixed.** ECRO (checkpoint, tables, response, sampling, renderer, geometry update), C1, the R3 checkpoint, both WL111
CT-AK checkpoints, and the evaluation masks and references. No temporal history, and no repair of WL111's camera-motion
failure.

**Resources.** LabServer70 RTX 3090, the owner's current allocation, still at the owner's 250 W power limit and clock
reduction. No CPU caps of my own. Primary timing is single-process and records clock, power and thermal state.

## Inputs read

- **WL111 code and evidence.** Code `experiments/ecro_recon111/`, package `evaluation/111/`. Commits 24211b7, 0c56218,
  d6a6525, a89cfed; package 5bc399d; final report d253f75.
- **R3 = WL105 `Reconstructor('R3')`.** Two separate component U-Nets (widths 32/64/96/128 × 1/√2) predict per-pixel
  15×15 softmax kernels.
  - `net_low` sees log1p(LOW) + 14 aux channels (8×8 guides).
  - `net_high` sees log1p(HIGH) + aux.
  - 1,055,624 parameters.
  - Training: 40 T-views, spp ∈ {4, 16, 32} uniformly, 4 fixed noisy replicates per (view, spp), all cold (single
    frame). Loss: the WL105 relative MSE on LOW + HIGH. 20k iterations, batch 16 × 128². So R3 at 1–2 spp is outside
    its training distribution.
- **CT-AK = WL111 `CompactNet(1.0)`.** One shared trunk on [log1p LOW, log1p HIGH, cover(9), normal(3), depth,
  log2 ESS], with 4×4 guides. It predicts 3 passes × 2 components × 25 logits (5×5, dilations 1, 2, 4).
  - 194,102 parameters.
  - Training: the same 40 T-views as a 12-level spp ladder × 2 replicates. spp ∈ {1: .25, 2: .25, 4: .25, 8: .1,
    16: .1, 32: .05}. Synthetic age maps: 35 % age 1, 30 % uniform age 2–64, 35 % spatially varying.
  - Loss: relative MSE on LOW + HIGH + TOTAL. Same steps, batch and schedule as R3. Seeds 17950001 / 17950002.
- **Contract differences R3 vs CT-AK, all confounds of the WL111 comparison.** Guide resolution, operator, predictor
  (separate vs shared), spp distribution, age distribution, loss (CT-AK adds a TOTAL term), replicate data (fixed 4
  vs ladder 2).

## Plan for Stages A–B (development data only; written before any WL111A measurement)

**Development data.**
- The 8 WL105 validation views Q00–Q07. They are held out from both R3's and CT-AK's training.
- They are not WL111 or WL111A evaluation views. Historical V0/H1–H5, K1–K6, the new sealed views and the edited states
  are used only in the final evaluation.
- Dev references: new T_ECRO references of the frozen renderer, 64 independent blocks × 1,024 spp per view. They store
  per-component (LOW, HIGH, TOTAL) mean and variance of the mean. These are needed for an expectation-error estimate.
  The WL105 4,096-spp targets store no variance.

**Stage A (baseline recovery).**
- **Identities.** Frozen renderer admission (G0), R3 and S111 checkpoint hashes, and a replay of one WL111 static item
  (R3 / S0 / S1 frame-1 metrics) under the WL111A code path.
- **Realization sets.** On each dev view at spp 1, 2, 4, 8: N = 24 independent frames, from seeds of a WL111A family
  distinct from every WL111 family. R0, R3, S111 r0 and r1 reconstruct each frame (cold: ESS = spp, no history).
- **Per arm, view, spp, ROI and component (LOW, HIGH, TOTAL).**
  - MSE.
  - Variance: per-pixel sample variance across the N frames.
  - Expectation error:
    `|mean_N − T|² − Var(T mean) − Var(y)/N`, i.e. the squared bias of the reconstruction expectation, debiased for
    both noise terms.
  - Signed luminance bias, gradient ratio.
- ROIs are WL111's (`common_r111.rois`). Contact is separate.

**Stage B (attribution). Discriminating controls, cheapest first.**
- **H4 (training distribution) and H1 (guides): one matched 2×2 factorial.**
  - Factors: age distribution {mixed as WL111, cold-only: every crop at age 1} × guides {sub 4, sub 8}.
  - Everything else matches WL111: architecture, data source (the WL111 ladder; sub-8 guides regenerated for the same
    views), spp distribution, loss, steps, batch, schedule, and the same two seeds. So initialization matches for the
    same sub; with sub 8 only the guide values change, not the input channel count.
  - Cell (mixed, sub 4) is the frozen S111 itself. Three new cells × 2 seeds = 6 trainings.
  - Evaluation: the Stage A dev realization sets (cold).
- **H2 (operator) capacity witness.**
  - On dev crops at 4 spp (and 1 spp), the actual composed CT-AK operator is optimized directly: the 150 logits per
    pixel, with cross-pixel dependencies through the three passes. R3's dense 15×15 operator (225 logits per pixel per
    component) is optimized the same way.
  - Both oracles use noise-independent per-pixel weights fitted to the reference on K_fit = 8 realizations, and are
    scored on K_test = 8 held-out realizations. This prevents per-noise overfitting.
  - These are capacity witnesses of a non-adaptive family, not deployable results and not certificates. An
    input-adaptive predictor can do better than or differently from them.
- **H3 (predictor).** Compare the learned predictor with the CT-AK oracle and the factorial arms, per component (LOW,
  HIGH) and ROI. An oracle-to-predictor gap is not attributed to capacity alone; training and information confounds
  will be stated.

**Selection rule (fixed now, applied in Stage C).**
- A hypothesis is actionable if its matched control reduces the dev cold-frame primary MSE at 4 spp, pooled over the 8
  views, by ≥ 10 % relative to S111, in both seeds.
  - For H2 the comparison is the CT-AK oracle against the R3 oracle: H2 is actionable only if the CT-AK operator's
    held-out witness error is ≥ 10 % worse than R3's.
- Among actionable hypotheses, the one with the largest pooled reduction is selected, subject to its predeclared cost
  consequence. Cost futility is declared in Stage C.
- If none is actionable, the batch stops as INCONCLUSIVE.

## Stage A, first run: stopped by a zero-reference view (2026-11-26 22:19 KST)

- **Completed before the stop.** Identities passed: renderer admission G0, R3 and S111 hashes, S111 hashes equal to
  their WL111 training logs. The WL111 static-item replay (V0, 4 spp, replicate 0, frame 1) is exact for R3, S0 and S1
  in every ROI. All 8 dev references were built.
- **The stop.** The devset step stopped at Q05 with a division by zero in the relative metrics.
- **Cause.**
  - Q05 renders an empty image. Its WL105 4,096-spp target is all zero as well, so the camera sees no geometry. This is a
    pre-existing property of the WL105 validation set, not a renderer or GPU fault.
  - A fresh 4-spp frame confirms it: the mean is 0.0 for Q05, against 1.036 for Q04 and 0.772 for Q06, both equal to their
    WL105 targets.
- **Correction (a code defect, before any attribution evidence).**
  - Q05 is excluded from the dev set (`DEV_EMPTY`). It keeps its index, so the other views' seeds do not move.
  - A zero-reference ROI now yields no relative metric instead of an error.
  - The devset step writes its results after every cell.
  - Dev views are therefore Q00–Q04 and Q06–Q07 (7 views). The devset rerun recomputes every cell with the same seeds.

## Execution notes, 2026-11-27 (no evidence affected)

- **Stage A rerun.** It completed at 11:00 KST. All seven dev views produced results; the values are reported in the
  Stage A results section.
- **Stage B smoke.** The first smoke stopped at the guide step. My consistency check compared a camera in
  `engine.cam_dict` form (with `res`) against the manifest form (without it). The cameras are identical. The check was
  fixed (`75c5274`), the smoke passed, and Stage B started at 11:03. The sub-8 guides were written at 11:03:49.
- **Device move (owner request, 11:10).** The owner asked me to move to the RTX 3080 Ti once it is free, minding its
  12 GB of VRAM. It is held by another project's container (`vism_wl119_train_B2`), so I wait.
  - A host monitor (`tiwatch_r111a.sh`, no GPU use) writes `use_ti` after the 3080 Ti has had no compute process for
    120 s.
  - Chain v2 (`chainB2_r111a.sh`) replaced the v1 wrappers. The three running wave-1 training containers were kept, not
    duplicated.
  - Device choice per step:
    - before `use_ti`: the RTX 3090, up to 3 concurrent trainings (6.6 GB each);
    - after `use_ti`: the RTX 3080 Ti, one process at a time (two trainings would need 13.2 GB), each under
      `guard_r111.py --gpu-uuid <3080 Ti>`, which yields to any other process.
  - Each run's device is logged in `devices.txt`.
  - Mixed devices change only floating-point details of development runs. Primary timing will be measured on one stated
    device.
- **CUDA out-of-memory handling (owner request, 11:22).** Chain v3 (`chainB3_r111a.sh`) replaced the v2 wrapper. No v2
  launch was in flight, and the wave-1 containers were kept.
  - Any step whose log shows a CUDA OOM is retried after 120 s, up to 3 times. OOM retries do not count as failures.
    The OOM log is kept as `<step>_oom<k>.log`.
  - Training batch sizes are WL111's and never change.
  - Inside `oracle_r111a.py`, a fit that runs out of memory restarts with chunked gradient accumulation over the fit
    frames (8 → 4 → 2 → 1). The objective is the same frame mean, so only floating-point summation order changes.
    The chunk used is recorded per cell.

## Stage A results (development data; 7 dev views, 64 independent cold frames per view × spp)

- **Identities and replay.** Renderer admission G0 passes. R3 and both S111 checkpoints match their hashes. The WL111
  static-item replay is exact. Stage B's paired rerun reproduces Stage A's S111 cells bitwise (primary MSE identical).
- **Bias / variance, pooled primary ROI, TOTAL.** "bias" is the root of the debiased squared expectation error, "share"
  is bias² / MSE; all values are relative.

  | spp | R3 RMSE / bias (share) | S111 r0 | S111 r1 | MSE S111 / R3 |
  |---|---|---|---|---|
  | 1 | 2.30 % / 1.02 % (0.20) | 2.43 / 1.57 (0.42) | 2.57 / 1.64 (0.41) | 1.11 / 1.24 |
  | 2 | 1.72 / 0.86 (0.25) | 1.97 / 1.38 (0.49) | 2.06 / 1.42 (0.47) | 1.31 / 1.44 |
  | 4 | 1.32 / 0.75 (0.32) | 1.61 / 1.20 (0.56) | 1.67 / 1.22 (0.54) | 1.48 / 1.60 |
  | 8 | 1.04 / 0.66 (0.40) | 1.33 / 1.04 (0.62) | 1.36 / 1.04 (0.58) | 1.62 / 1.70 |

- **The deficit is systematic.** At every spp the variance of S111 equals R3's (std ≈ 1.07–1.09 % at 4 spp), and the
  whole MSE excess is squared expectation error. The ratio grows with spp as noise falls.
- **Where it sits (4 spp).** MSE S111 / R3, with bias R3 → S111:

  | ROI | MSE S111 / R3 | Bias R3 → S111 |
  |---|---|---|
  | Material boundary | 1.8–2.8 | 5.8 → 9.0–11.2 % |
  | Contact | 2.0–2.7 | 3.8 → 7.7–9.5 % |
  | Silhouette | 1.4–1.5 | |
  | Control | 1.2–1.3 | 0.18 → 0.30–0.34 % |
  | SSS interior | 0.8–0.9 | S111 better |

- **By component.** The excess is mostly in LOW: bias share 0.59–0.63 against R3's 0.37. HIGH differs little.

## Stage B results (development data, paired with Stage A seeds)

**Matched H1 × H4 factorial.**
- **Setup.**
  - Same architecture, data stream, loss, schedule and seeds as WL111.
  - Initial weights equal WL111's in all 6 runs (hash check).
  - Training time 2,190–2,524 s each, 3 concurrent on the RTX 3090.
- **Primary MSE at 4 spp, relative to S111 of the same seed:**

  | Cell | r0 | r1 |
  |---|---|---|
  | C4 (cold-only, sub 4) | 1.000 | 0.790 |
  | M8 (mixed, sub 8) | 1.015 | 1.103 |
  | C8 (cold-only, sub 8) | 0.977 | 0.901 |
  | R3 (reference) | 0.675 | — |

- **Other spp.** The same pattern holds at 1, 2 and 8 spp: no cell improves both seeds by ≥ 10 %.
- **Effect at the localized ROIs (4 spp).** Material boundary C4 0.91 / 0.58, C8 0.95 / 0.63. Contact C4 0.76 / 1.02,
  C8 0.88 / 0.70. These are seed-dependent, and every cell stays 1.3–2.3× R3 on boundary ROIs.
- **Seed effect.** The seed-to-seed difference within a cell (up to 0.21) is as large as any factor effect.

**Operator capacity witnesses (H2).**
- **Setup.** 7 dev views, 256² crops around the sphere, 8 fit + 8 held-out frames, 600 Adam steps. Both oracles stayed in
  chunk 8: no OOM.
- **Overfitting.** Both non-adaptive oracles overfit the fit frames (fit RMSE 0.0–0.9 %). Their held-out variance is
  inflated, R3's 225-tap oracle most: held-out std 2.66 % vs CT-AK oracle 1.42 % at 4 spp primary.
- **Held-out expectation error at 4 spp.** "Oracle" values are CT-AK / R3; "learned" values are S111 r0 / r1 / R3.

  | ROI | Oracle bias | Learned bias |
  |---|---|---|
  | Primary | 0.59 / 0.75 % | 1.17 / 1.00 / 0.83 % |
  | Silhouette | 2.12 / 2.12 % | 5.15 / 5.92 / 4.04 % |
  | Material boundary | 1.76 / 1.87 % | 6.62 / 8.10 / 4.31 % |

  At 1 spp the ordering is the same.
- **Reading.** The composed CT-AK operator represents boundary expectations as well as R3's dense operator (bias equal
  or lower in every ROI). Its held-out total error is also lower: primary 1.53 % vs 2.77 %. So H2 is not actionable by
  the predeclared rule. The witnesses are not certificates, and their overfitting differs.

**Status against the predeclared selection rule.**
- No matched control reduces dev primary MSE at 4 spp by ≥ 10 % in both seeds: H1 no, H4 no. H2 is not actionable.
- **H3 has no matched control yet.** My Stage B plan assessed it only through the oracle gap, which is large: S111's
  boundary bias is 3–5× what the same operator family can represent. Neither guides nor training distribution closes
  that gap.
- **Observation, not yet evidence for a mechanism.** CT-AK's predictor has one full-resolution 3×3 convolution (the stem)
  and reaches full resolution otherwise only through bilinear upsampling and 1×1 layers. R3's U-Nets have four
  full-resolution 3×3 convolutions per component.
- **Decision point (owner).** Either stop here as INCONCLUSIVE by the rule as written, or run one additional labelled
  Stage B control for H3 with a rule fixed before it runs.

## Stage B addition: H3 matched control (owner approval 2026-11-27; added after the Stage B results, rule fixed now)

The owner chose to run one additional Stage B control for H3 instead of stopping as INCONCLUSIVE. It is labelled as an
addition made after the Stage B results were seen.

- **Arm P4 (`recon_r111a.FullResNet`).**
  - It is WL111's CompactNet with the full-resolution 1×1 fusion layer replaced by two 3×3 convolutions (48 → 32 → 32
    channels). The predictor then has three full-resolution 3×3 layers instead of one.
  - Trunk, head, inputs, the 150-logit output and the spatial operator are unchanged.
  - About +21.5k parameters (CompactNet 194,102).
- **Contract.** S111's own contract: mixed ages, sub-4 guides, same data stream, loss, schedule, steps, batch and seeds.
  The initial weights necessarily differ from WL111's, because the module set changes.
- **Execution.** RTX 3080 Ti only (owner), one process at a time under `guard_r111.py --gpu-uuid <3080 Ti>`, with CUDA OOM
  retries. A smoke run (200 steps) gates the run.
- **Evaluation.** The Stage A dev realization sets, paired: same seeds, label `devset_H3`.
- **Rule, fixed before the run.**
  - H3 is actionable if P4 reduces the dev cold-frame primary MSE at 4 spp, pooled over the 7 views, by ≥ 10 % relative
    to S111 of the same seed, in both seeds.
  - If H3 is actionable, it is the selected mechanism for Stage C. Stage C then still has to declare cost consequences
    and futility before final training.
  - If H3 is not actionable, the batch stops as INCONCLUSIVE.
  - ROI and component effects are reported descriptively.

### H3 control result (2026-11-27 13:41 KST; RTX 3080 Ti; 215,638 parameters; 889 s / 860 s per training)

- **Primary MSE, P4 relative to S111 of the same seed:**

  | spp | r0 | r1 | P4 / R3 |
  |---|---|---|---|
  | 1 | 0.858 | 0.756 | 0.94–0.96 |
  | 2 | 0.819 | 0.724 | 1.04–1.07 |
  | 4 | **0.792** | **0.690** | 1.10–1.17 |
  | 8 | 0.767 | 0.671 | 1.14–1.24 |

- **Rule.** At 4 spp both seeds are ≥ 10 % lower, so **H3 is actionable**, and it is the selected mechanism for
  Stage C.
- **What changed.**
  - The reduction is systematic error: bias 1.20 / 1.22 % → 0.99 / 0.88 % at 4 spp.
  - It comes mainly from LOW (P4 / S111 0.78 / 0.67); HIGH changes less (0.86 / 0.89).
- **ROIs at 4 spp, P4 / S111.**
  - Silhouette 0.73 / 0.74, now 1.01–1.07× R3.
  - Material boundary 0.75 / 0.46, still 1.28–1.37× R3.
  - Contact 0.45 / 0.78.
  - Control 0.86 / 0.97.
  - SSS interior 0.87 / 1.09, still better than R3.
- **Not closed.** P4 does not close the gap to R3 at equal spp: 1.10–1.17× at 4 spp. Material-boundary bias stays
  6.8–7.5 %.
- **Interpretation.** Within this evidence, the cold-frame boundary bias of CT-AK is mainly a predictor limitation: the
  depth of full-resolution processing.
  - The operator witness had already shown that the same operator can represent lower boundary bias.
  - Guide resolution and training distribution did not reduce it.
- **Confound.** P4 adds capacity (+11 % parameters) and full-resolution receptive field at once. This control does not
  separate them.
