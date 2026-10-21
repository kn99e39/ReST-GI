# Worklog — Architecture Batch 4: Oracle-Guided Selective Reconstruction on the Diffuse-Interaction Lego (2026-10-18 – 2026-10-21)

## SESSION QUESTION

Architecture Batch 4 (protocol `experiments/rna_selective_reconstruction_lego/protocol/b4_v1.json`, declared
2026-10-18 before any render, oracle or training; Mechanism 2 only):

> After a physical geometry edit, does fully recomputing the complete learned state of the oracle
> transport-dirty units (D\*) — and only those — recover current-geometry radiance as well as recomputing
> every unit, while the still-valid units (V\*) stay bit-identical and valid?

**Short answer: CONDITIONAL (predeclared outcome rules).** Five of the six positive criteria pass; the
whole-image criterion misses by 0.05 dB.

- **Stale GI is real and lands on stationary receivers.** The oracle marks 7 stationary Lego units plus the
  moved box as transport-dirty (D\* = {0, 2, 3, 6, 7, 8, 13, B}; 8 of 17 units, 39% of the counted surface,
  99% of the change energy). Keeping the T0 state costs 2.66 dB on the whole image and 11.5× the full-recompute
  error on the physically changed surfaces.
- **Selective recomputation removes it.** On the physically changed surfaces it recovers R = 0.98 of the
  frozen-vs-full error gap.
- **Ownership holds under a real edit.** The 9 V\* units are bit-identical to SO_T0 (135 tensors). Composed
  outputs change on exactly the predicted support (1,478,240 px = D\*-owned + D\* PU bands; 0 outside).
  V\*-owned error is 1.03× full recompute's.
- **The miss.** B_sel is −0.313 dB below A_full (criterion ≥ −0.26 dB). 69% of the pooled gap comes from one
  re-learned unit (unit 3), a weakly dirty unit whose re-learned state is worse than both A_full's (1.54×) and
  its own frozen T0 state (1.40×). Single seed per arm; the cause is not isolated.

## RELATION TO EARLIER WORK

- **Worklog 36 (Batch 3):** L16 owned units on the native Lego, static Case A. Its partition, unit specs,
  trainer, PU composer and REINIT mechanism are reused unchanged.
- **Worklogs 32–35:** ownership contract, RNA baseline audit, boundary attribution, native RNA qualification.
  None is reinterpreted.
- **Lab-meeting smoke (same worklog number, separate regime):** `results/evaluation/37/smoke_meeting/`
  (protocol `b4_smoke_meeting_v1` + amendment 1; 128-image T1 subset). Its numbers are not compared with the
  main batch below without that label.
- **Code:** `experiments/rna_selective_reconstruction_lego/`. **Outputs:**
  `results/rna_selective_reconstruction_lego/`. **Exports:** `results/evaluation/37/`.

## PROTOCOL AND EXECUTION

**Scene (unchanged from the protocol).** The official RNA Lego (A) plus a saturated-red diffuse box (B) under
the raised bucket. T0 → T1 translates B by +0.44 m; A, materials and light are unchanged. The only light is
the generator's directional sun, so the change on A is indirect (colour bleeding and occlusion of bounce
light).

**Arms (all 250 epochs, seed 0, unchanged trainer/loss/schedule, 400 training images per state).**

| arm | learned state evaluated on T1 |
|---|---|
| C frozen (`SO_T0`) | 17 owned units trained on T0, not updated |
| A_full | all 17 units re-initialised and trained on T1 |
| B_sel | SO_T0 with the 8 D\* units re-initialised and trained on T1; V\* frozen (`requires_grad` False) |
| RNA_T0 / RNA_T1 | original upstream RNA HQ per state (static representability controls) |

Primary composition is PU (h 4.5 mm); hard selection is reported in `eval.json` and differs by ≤ 0.015 dB.

**Amendments.**
- **1 (2026-10-18, post-hoc, user decision):** pilot P4 failed as written (owner distance on depth-edge
  pixels); P4 is evaluated on non-depth-edge pixels. Recorded before training data existed.
- **2–4 (2026-10-19/20, execution only, prospective for every arm):**
  - **2:** RNA_T0 paused for a user reboot right after an epoch-89 checkpoint and resumed; the project folder
    was renamed NeuralShaderDemo → ReST-GI, so configs are read from path-only copies (`config_restgi/`).
  - **3:** a Windows NVIDIA driver fault (nvlddmkm 13 "Graphics FECS Exception: UCODE Fatal Error", 14 "GPU
    Reset Required", 2026-10-20 00:53) killed the resumed RNA_T0. Chain v5 did not check return codes and
    started A_full; that start was stopped during data loading (no epoch completed), set aside as
    `A_full_aborted_tdr_0056` and is not evidence. RNA_T0 resumed again from epoch 104. Chain v6 stops on any
    failed step.
  - **4:** RNA_T1 paused at the user's request right after its epoch-33 checkpoint and resumed; the final
    evaluation ran afterwards with its unchanged command.
- **Consequences:** RNA_T0 (2 resumptions) and RNA_T1 (1 resumption) are labelled "resumed" (weights,
  optimiser, scheduler and epoch restored; shuffle RNG not). **SO_T0, A_full and B_sel each ran uninterrupted
  from start to end**, so their quality and wall-clock are unaffected.

## EVIDENCE

### Pilot and oracle (`oracle/oracle.json`; 32 oracle views, 512², 2 × 2048 spp, LabServer63 renders)

- **Pilot b4_v1.1:** all pass.
  - P1: indirect change z = 4.14.
  - P2: units with CI lower bound > τ: {0, 2, 3, 6, 7, 8, 13}.
  - P3: colour bleeding moves with B, R−G +0.0103 near T1 and −0.0094 near T0, both CIs exclude 0.
  - P4 (amended): owner distance q99.9 0.70 / 0.71 / 0.85 mm; correspondence 0.999999 / 1.0.
  - P5: 17 / 17 units observed.
- **D\*** = {0, 2, 3, 6, 7, 8, 13, 16 (B)}:
  - dirty area fraction 0.389;
  - energy fraction in D\* 0.991;
  - ambiguous unit 11 (CI contains τ; kept in V\*, as the rule says).
- **Sensitivity (reported, not used):** D\* at τ/2 = 12 units; at 2τ = {0, 2, 6, 7, 8, 16}. Units 3
  (RMS 0.0068) and 13 (0.0098) are dirty at τ but not at 2τ.
- **Cross-machine check:** PC vs server render RMS 0.0018 on the same oracle views. Training data and held-out
  references are all PC renders; the oracle views are server renders and are used only for D\*.

### Qualification (held-out T1/T0, 16 views, 1024², 4096-spp references, display PSNR)

| check | value | rule | |
|---|---|---|---|
| RNA_T0 (held-out T0) | 33.57 dB | ≥ 32.44 | pass |
| RNA_T1 (held-out T1) | 33.49 dB | ≥ 32.44 | pass |
| A_full-PU vs RNA_T1 | −0.12 dB | ≥ −0.5 | pass |
| A_full-PU logL1 / RNA_T1 | 1.03× | ≤ 1.10 | pass |
| discriminative edit: frozen / A_full MSE on changed A | 11.46× | ≥ 1.25 | pass |

Static control: SO_T0-PU on held-out T0 is 33.24 dB, −0.32 dB vs RNA_T0 (worklog 36: −0.32).

### Outcome (held-out T1, PU) — `results/evaluation/37/verdict.json`

| positive criterion | value | rule | |
|---|---|---|---|
| B_sel − A_full, view-mean PSNR | **−0.313 dB** | ≥ −0.26 | **fail** |
| changed-region recovery R | 0.979 | ≥ 0.8 | pass |
| V\*-owned MSE, B_sel / A_full | 1.032 | ≤ 1.10 | pass |
| V\* parameters bit-identical to SO_T0 | 135 / 135 tensors; units changed = D\* | exact | pass |
| output change vs frozen inside predicted support | 1,478,240 changed = predicted; 0 outside | exact | pass |

Negative rule (gap < −1.0 dB or R < 0.5): not met. |D\*| = 8 < 15: not degenerate. **Case: CONDITIONAL.**

### Regions (held-out T1, display PSNR from region MSE)

| region | px | RNA_T1 | C frozen | A_full | B_sel |
|---|---|---|---|---|---|
| whole image (pooled) | 3.86 M | 33.39 | 30.10 | 33.15 | 32.76 |
| physically changed A | 287 k | 32.23 | 23.06 | 33.66 | 32.80 |
| A with changed visibility | 58 k | 32.99 | 21.10 | 35.16 | 33.56 |
| D\*-owned | 1.47 M | 33.07 | 27.75 | 33.86 | 32.98 |
| V\*-owned | 2.39 M | 33.60 | 32.63 | 32.77 | 32.63 |
| physically unchanged A | 3.20 M | 33.41 | 32.78 | 32.88 | 32.51 |
| moved box B | 156 k | 36.11 | 25.27 | 40.75 | 40.20 |

View-mean PSNR: RNA_T1 33.49, A_full 33.38, B_sel 33.07, frozen 30.72. Per-view B_sel − A_full ranges from
−1.28 to +0.20 dB (13 of 16 views negative).

### Recompute cost (RTX 5080, WSL, alone on the GPU, uninterrupted)

| | A_full (17 units) | B_sel (8 units) | ratio |
|---|---|---|---|
| wall-clock incl. data setup | 8,763 s (2.43 h) | 6,878 s (1.91 h) | 0.785 |
| mean s/epoch | 32.8 | 26.0 | 0.79 |
| trained parameters | 11,586,006 | 5,275,776 | 0.455 |
| peak torch memory | 1.93 GiB | 1.07 GiB | |

Time does not scale with the trained-parameter fraction: each epoch still reads every training image. That
fixed cost was not separated here. Inference: PU composition 0.27–0.29 s/view (prototype loop).

## OBSERVATION (`results/evaluation/37/`)

- **`01_oracle_units`, `02_oracle_views`:** the counted change concentrates on the base plate around both box
  positions, with both signs on each side: light leaves where the box was and arrives where it is. P3 measures
  the colour part as R−G moving with the box. The visibility-changed pixels excluded from the oracle are small
  patches next to the box footprints.
- **`03_arms_view6`, `03_arms_view15`:**
  - Frozen transfer shows the stale red bleeding at the old box position and lacks it at the new one. Its
    error map is dark exactly over the changed region.
  - A_full and B_sel remove it and are visually indistinguishable at full frame.
  - |B_sel − A_full| is spread over the D\* units at a low level, not concentrated at seams.
- **`04_crops`:**
  - The strongest-change crop (view 6, plate next to the box at T1): B_sel ≈ A_full ≈ reference. Frozen keeps
    the T0 appearance, a bright, reddish-lit plate where the reference is now darkened by the box.
  - The V\* crop: identical to frozen by construction.
  - The largest-B_sel-error window (view 15): the yellow body behind the gears. RNA_T1, A_full and B_sel all
    show blotchy orange texture there, so this is a baseline-level error rather than one specific to
    selection.

## ATTRIBUTION

**Measured, descriptive (written after the result; single seed per arm):**
- **Where the gap is.** The pooled B_sel − A_full gap is −0.39 dB:
  - 77% of it comes from D\*-owned pixels and 23% from V\*-owned pixels.
  - **Unit 3 alone accounts for 69%.** Its B_sel MSE is 1.54× A_full's.
  - The other D\* units are 0.85–1.14× A_full.
- **Unit 3 is weakly dirty.** Its oracle RMS change is 0.0068 (τ = 0.005; not dirty at 2τ). Its frozen T0
  state was only 1.10× A_full on T1, and the re-learned state is 1.40× its own frozen state. Unit 13 behaves
  similarly (frozen 0.99× A_full). For both, the edit-induced change is small relative to the model's own
  error floor (MSE ~7×10⁻⁴, display RMS ~0.027).
- **Post-hoc counterfactual (not a protocol outcome):** with unit 3 kept frozen, the pooled gap would be
  −0.18 dB.
- **V\* units are radiometrically valid at this scale.** Their frozen T1 error is 1.00–1.09× their own static
  T0 error (pixel-weighted 1.02×). The V\* share of the gap is therefore model-to-model variation between
  SO_T0's and A_full's independently trained units (e.g. A_full is better on units 9 and 10 and worse on 5,
  12 and 14), not stale transport.
- **The training monitor plateaus early.** B_sel's training-time val PSNR plateaus near 32.5 dB from about
  epoch 100, while A_full climbs to 32.8. The monitor is whole-image, so frozen V\* error bounds it. It does
  not show per-unit convergence.

**Candidate causes for unit 3 (untested):**
- **Optimisation/initialisation variance of a single re-learned unit:** B_sel re-initialises only D\*, so
  unit 3's draws differ from A_full's.
- **Unit data/support:** the smoke runs also re-learned unit 3 poorly (MSE 1.6–1.7×10⁻³ at 128 images).
- **Interaction with frozen PU neighbours** during training.

**What this batch does and does not show:**
- **Shown (one scene, one edit, one seed):**
  - D_T ≠ D_G: 7 of the 8 dirty units are stationary receivers, so a mover-only rebuild would leave most
    stale GI in place.
  - D_T is sparse at unit granularity for this edit.
  - Complete-state recomputation of the dirty units removes the stale GI.
  - Ownership is exact under a real edit.
- **Not shown:**
  - Whether the −0.31 dB whole-image cost is systematic or single-run variance.
  - A recompute cost that beats full recompute by more than 0.79× wall-clock.
  - Any scene-level selector: D\* is an oracle from ground-truth radiance.

## NEXT DECISION (recommendation; not implemented)

The verdict hinges on 0.05 dB and one weakly dirty unit, so the next step should resolve that before any
selector work:

1. **Seed replicate of the decisive comparison.** Repeat B_sel and A_full with two more seeds, unchanged
   protocol, and report the gap distribution and unit-3 spread. Under the 2026-10-21 placement rule this is a
   long job for LabServer63. It must use PC-rendered data only, and both arms must run on the same machine.
2. **Then decide between two questions:**
   - a weakly-dirty-unit policy (recompute vs keep), which must be predeclared and must not move τ post hoc;
   - the relation-observability / candidate-coverage question of the 2026-10-19 direction record.

Neither is authorised by this worklog.

## REPOSITORY

| purpose | commit |
|---|---|
| scene builder, protocol b4_v1 | `20768b2` |
| ownership adapter, oracle, train/eval wrappers, tests | `a68dcb7` |
| amendment 1 (P4) | `42bde74` |
| local chains, machine-bias check | `a303f7b`, `091987f` |
| smoke protocol, chain v3, smoke result | `05f5943`, `4ba3c25`, `0541b79` |
| smoke amendment 1, chain v4, smoke report | `63a08df`, `1a6efaf`, `35bd0f2`, `b56c5b3` |
| chain v5 + amendment 2 | `40a8373` |
| chain v6 + amendment 3 (SO arms A_full, B_sel, core eval) | `7c2c0a9` |
| chain v7 + amendment 4 (RNA_T1, final eval) | `176f900` |
| verdict script, exports, this worklog | this session's commit |

SO_T0 and the smoke arms were trained at commits up to `1a6efaf` (recorded paths use the pre-rename folder
name). Evaluation records hash their sources in `results/evaluation/37/manifest.json` and `verdict.json`.
