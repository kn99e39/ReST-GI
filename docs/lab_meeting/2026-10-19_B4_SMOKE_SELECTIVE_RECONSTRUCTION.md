# Lab Meeting 2026-10-19 — Early Evidence: Selective Neural GI Reconstruction (Smoke Test)

**Status:** preliminary smoke test. It is not the Architecture Batch 4 result, which is still running (Worklog 37).
**Protocols:**
- `experiments/rna_selective_reconstruction_lego/protocol/b4_smoke_meeting_v1.json`;
- amendment 1, which adds the full-recompute arm. It was declared before that arm was trained.

**Full numbers and figures:** `results/evaluation/37/smoke_meeting/` (README, `smoke_metrics.json`, `01`–`03` PNG).

## Question

After a geometry edit, does re-learning **only the oracle-dirty Surface-Owned Neural Transport Units** recover the
edited scene's indirect illumination, compared with reusing the unchanged T0 model and with recomputing every unit?
Every other unit must stay bit-identical.

**Answer: preliminary positive** (held-out views):
- **Against frozen:** on physically changed persistent surfaces the error drops by 86% (23.1 → 31.6 dB). All 16 views improve, and the CI excludes 0.
- **Against full recompute:** on the same surfaces the two are statistically indistinguishable (31.6 vs 31.8 dB).
- **Whole image:** selective 32.1 dB against 30.2 dB for full recompute under the same T1 data budget.
- **Preservation:** the 9 unselected units are bit-identical.
- **Recompute cost:** 0.76× the full-recompute wall-clock, measured under identical conditions. The trainer was not optimised for selectivity, so this ratio is conservative.

## Setup

**Scene (Diffuse Interaction Variant A):**
- **A** = the official RNA Lego, unchanged.
- **B** = a red Lego-plastic box under the raised bucket.
- **Edit** T0 → T1: B is translated by 0.44 m, from left to right. Materials are identical in T0 and T1.

**Representation:** the qualified Surface-Owned RNA units:
- 16 L16 units on A plus 1 unit for B, each owning a complete triplane and MLP;
- PU composition at the boundaries;
- canonical (T0-placement) frames for the moved box.

**Oracle target set D\*:**
- D\* = {0, 2, 3, 6, 7, 8, 13, B}: 8 of 17 units, 46% of the parameters.
- It comes from ground-truth T0/T1 renders: a visibility-matched, noise-debiased radiance change above τ = 0.005 (32 views, 4096 spp).
- It was fixed before any training.

**Arms:**
- **C — frozen transfer:** the T0 model (SO_T0: full 400-image T0 set, 250 epochs), evaluated on T1 without updates.
- **B — selective:** SO_T0 with the 8 D\* units cold re-initialised and trained alone; all other units have `requires_grad False`.
- **A — full recompute:** all 17 units trained from scratch.

B and A share the same budget:
- the same **128-image T1 subset**;
- 250 epochs, i.e. 32% of the formal optimisation steps;
- the unchanged trainer, loss, optimiser and schedule;
- the same CPU cores and the same concurrent render.

**Evaluation:** 16 held-out T1 views (4096-spp path-traced references) that no training saw. Regions were fixed in advance from the paired T0/T1 references.

## Results (held-out T1, PU composition)

| region | C frozen | **B selective** | A full | B vs C | B vs A |
|---|---|---|---|---|---|
| **physically changed A surfaces** (287 k px) | 23.06 dB | **31.61 dB** | 31.79 dB | −86% MSE, 16/16 views | not distinguishable (CI includes 0) |
| D\*-owned (re-learned units) | 27.75 dB | 31.36 dB | 31.18 dB | better | slightly better |
| moved box B | 25.27 dB | 37.05 dB | 36.20 dB | better | slightly better |
| V\*-owned (reused units) | 32.63 dB | 32.62 dB | 29.66 dB | +0.16% MSE (PU band only) | **better: reuse keeps 400-image quality** |
| physically unchanged A surfaces | 32.78 dB | 31.96 dB | 29.80 dB | **worse (+21% MSE)** | better |
| whole image | 30.10 dB | **32.10 dB** | 30.18 dB | better | better |

**State preservation:**
- Changed units = exactly D\*; the other 9 units are bit-identical.
- Composed outputs changed on 1,478,240 pixels, exactly the predicted support (D\*-owned pixels plus 9,781 V\* PU-band pixels), and **0 outside** it.

**Recompute cost** (same subset, epochs, cores and concurrent render):

| | selective | full | ratio |
|---|---|---|---|
| wall-clock | 68.0 min | 89.8 min | **0.76** |
| median epoch | 14.9 s | 18.8 s | 0.79 |
| trained parameters | 5.28 M | 11.59 M | 0.46 |
| share of training pixels owned by trained units | 39% | 100% | 0.39 |

- **These are recompute times** (updating the learned state after the edit).
- **Frame rendering (inference) time does not differ** after recomputation: both arms are the same 17-unit composed model. Measured: frozen 0.52 s/view, selective 0.41 s, full 0.44 s.
- **Not measured: incremental re-rendering for a fixed camera.** Only the predicted 38% of pixels changed under selective recomputation (100% under full). Only this pixel share is known; no time has been measured.
- **0.76 is conservative.** The unchanged trainer still runs the frozen units' forward pass and validates every epoch. The parameter and pixel shares (0.39–0.46) are the idealised range; a trainer that reaches them has not been built or measured.
- Not included: the oracle's own cost and data generation.

## Visual

`results/evaluation/37/smoke_meeting/01_meeting_comparison.png`:
- held-out view 6, showing T0 reference, T1 reference, frozen, selective, full and the three error maps;
- a crop at the strongest ground-truth change.
- The frozen model still paints B's old **red reflection and shadow ghost** on the sand plate where B stood at T0. Selective and full recomputation both remove it.

## Post-hoc: the frozen error is mostly not out-of-distribution

This analysis was not predeclared; details are in `results/evaluation/37/smoke_meeting/README.md`. It splits the frozen error on physically changed surfaces into three parts:

| part | share of frozen error |
|---|---|
| **visibility same as T0** (inputs in-distribution) | **63%** |
| visibility changed (shadow head extrapolates) | 29% |
| surfaces not corresponded to the same T0 view | 8% |

- **The dominant part is target shift, not OOD.** The query inputs are unchanged; the indirect light they should map to changed.
- Only recomputing the owning units' learned state repairs it.
- Selective recomputation repairs all three parts as well as full recomputation does.

## Interpretation and limits

**What this shows:**
- Under oracle target selection, re-learning the complete state of only the dirty units recovers the changed GI as well as recomputing everything does, on the surfaces that changed.
- The reused units are untouched and keep the quality of their larger T0 training set.
- Recomputation is cheaper even with an unoptimised trainer.

This is preliminary evidence of effective selective Neural Transport State reconstruction under oracle target selection.

**Costs and caveats:**
- **Inside the re-learned units, quality drops on surfaces that did not change** (+21% MSE vs frozen). These units re-learn everything from only 128 T1 images.
- **The whole-image advantage over full recompute is a limited-data effect.** Full recompute must re-learn the valid units from the same 128 images. With all 400 T1 images (the formal batch), that gap is expected to shrink.

**Not shown:**
- parity under the full data budget;
- convergence;
- original-RNA representability on T1;
- cost under exclusive conditions;
- automatic target selection (Mechanism 1). The dirty set here is an oracle.

**Protocol note:** the GT pilot's identity criterion P4 failed as written. The cause was a mis-specified tolerance on anti-aliased depth-edge pixels. The batch proceeded under a post-hoc amendment (b4_v1.1, user decision); see `protocol/b4_v1_amendment1.json`.

**Main batch:** the full 400-image arms are running:
- full recompute (A_full);
- selective reconstruction (B_sel);
- frozen transfer (C);
- original RNA controls on T0 and T1.

The final verdict comes in Worklog 37.
