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
