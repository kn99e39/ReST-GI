# Lab Meeting 2026-10-19 — Early Evidence: Selective Neural GI Reconstruction (Smoke Test)

**Status:** preliminary smoke test. It is not the Architecture Batch 4 result, which is still running (Worklog 37).
**Protocol:** `experiments/rna_selective_reconstruction_lego/protocol/b4_smoke_meeting_v1.json`, committed before the run.

## Question

After a geometry edit, does re-learning **only the oracle-dirty Surface-Owned Neural Transport Units** recover the
edited scene's indirect illumination better than reusing the unchanged T0 model, while every other unit stays bit-identical?

**Answer: preliminary positive** (all numbers are on held-out views):
- **Physically changed persistent surfaces:** display MSE −86%, from 23.1 to 31.6 dB. All 16 views improved, and the 90% CI excludes 0.
- **Unselected units:** all 9 have bit-identical parameters.
- **Whole image:** 30.1 → 32.1 dB.

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
- It was chosen before any training.

**Arms:**
- **C — frozen transfer:** the T0 model (SO_T0: full 400-image T0 set, 250 epochs) evaluated on T1 without updates.
- **B — selective update:** start from the same T0 model; cold re-initialise the 8 D\* units and train only them. All other units have `requires_grad False`.
  - Trainer, loss, optimiser and per-epoch schedule are unchanged from the formal protocol.
  - 250 epochs on a **128-image T1 subset**, i.e. 32% of the formal optimisation steps.
  - Wall-clock: 68 min (14.9 s/epoch), overlapped with a CPU render, so this is **not** a cost measurement.

**Evaluation:**
- 16 held-out T1 views (4096-spp path-traced references) that no training saw.
- Regions are fixed in advance from the paired T0/T1 references.

## Results (held-out T1, PU composition)

| region | pixels | frozen MSE (PSNR) | selective MSE (PSNR) | change | 90% CI of Δ MSE | views improved |
|---|---|---|---|---|---|---|
| **physically changed A surfaces** | 287 k | 0.00494 (23.06 dB) | 0.00069 (31.61 dB) | **−86%** | [−0.00504, −0.00303] | **16/16** |
| D\*-owned | 1.47 M | 0.00168 | 0.00073 | −56% | — | — |
| moved box B | 156 k | 0.00298 | 0.00020 | −93% | — | — |
| V\*-owned (preserved units) | 2.39 M | 0.000546 | 0.000547 | +0.16% (PU band only) | [+3e-7, +1.4e-6] | — |
| physically unchanged A surfaces | 3.20 M | 0.000527 | 0.000636 | **+21%** | — | — |
| whole image | 3.86 M | 30.10 dB | 32.10 dB | −37% MSE | [−0.00059, −0.00016] | 13/16 |

Whole-image HDR logL1 (no composition): 0.0113 (frozen) → 0.0107 (selective).

**State preservation:**
- Changed units = exactly D\*; the other 9 units are bit-identical (parameter digests).
- Composed outputs changed on 1,478,240 pixels, exactly the predicted support:
  - 1,468,459 pixels owned by D\* units;
  - 9,781 pixels in V\* units' PU bands;
  - 0 pixels outside the predicted support.

## Visual

- `results/evaluation/37/smoke_meeting/01_meeting_comparison.png`:
  - held-out view 6, showing T0 reference, T1 reference, frozen, selective and both error maps;
  - a crop at the strongest ground-truth change.
  - The frozen model still paints B's old **red reflection and shadow ghost** on the sand plate where B stood at T0. The selective update removes it.
- `02_mask_and_regions.png`: the updated-unit mask and region bars.
- `smoke_metrics.json`, `manifest.json`: numbers and hashes.

## Interpretation and limits

**What this shows:** under oracle target selection, re-learning the complete state of only the dirty units removes most of the stale indirect light on persistent surfaces, and leaves the reused units untouched. This is preliminary evidence of effective selective Neural Transport State reconstruction under oracle target selection.

**Cost visible already:**
- Physically unchanged parts **inside** the re-learned units got worse: +21% MSE on unchanged A pixels.
- This is expected with only 128 T1 images (dirty units cover 39% of the surface). The formal 400-image run will show whether it is a data-budget effect or something more.

**Not shown:**
- parity with full recompute (arm A_full is still running);
- convergence;
- original-RNA representability on T1;
- any speed-up or system cost;
- automatic target selection (Mechanism 1). The dirty set here is an oracle.

**Protocol note:** the GT pilot's identity criterion P4 failed as written. The cause was a mis-specified tolerance on anti-aliased depth-edge pixels. The batch proceeded under a post-hoc amendment (b4_v1.1, user decision); see `protocol/b4_v1_amendment1.json`.

**Main batch:** the full 250-epoch, 400-image arms are running:
- full recompute (A_full);
- selective reconstruction (B_sel);
- frozen transfer (C);
- original RNA controls on T0 and T1.

The final verdict comes in Worklog 37.
