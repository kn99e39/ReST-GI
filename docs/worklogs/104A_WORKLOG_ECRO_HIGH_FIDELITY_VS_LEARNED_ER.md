# Worklog 104A — ECRO v0 HIGH-transport fidelity against its learned-ER source (2026-11-18)

## Authorization and freeze, 2026-11-18 KST

The user's brief asks whether ECRO v0 adds a significant, systematic sharp-interreflection energy-magnitude or spatial-distribution error beyond its frozen learned-ER material source. WL104 is the preserved baseline. The batch qualifies the architecture; a correction is allowed only for a localized implementation or estimator defect, and no redesign, learned-ER change or denoiser is permitted.

- **Batch identity.** This is a suffix of WL104 under AGENTS.md. It resolves WL104's unresolved HIGH-only comparison, and no WL104A exists on any branch.
- **Where it runs.** Branch `ecro_wl84_scene` (WL104 line; `Denoise_Reconstructor` and the main branch are untouched), on the local RTX 5080 by the owner's answer.
- **Concurrent work.** `origin/ecro_lr3_adapter` has independently advanced to WL101G audit, WL101H and WL101I. Those are separate work and are not merged here.

**Baseline recovered.** The WL104 commits are `8a91ee24`, `9e1c05c5` and `1a8c66d1` (full hashes in the protocol). All three checkpoints match WL104's recorded hashes, and nothing is retrained.

Development checks against WL104:
- Learned tables regenerated from the checkpoints equal WL104's evaluation tables bit for bit.
- ECRO's 4-spp frame replays WL104's stored frame to 1.2e-7.
- **The two components measure the same events.** ECRO HIGH is `thr · B[patch, bin]` read at camera hit 2, where B is the learned outgoing radiance of 2–5 events at the probe. The tracer's h is next-event estimation at camera hits 3–6. Both are camera events 3–6.
- **Paired LOW.** With a shared seed, ECRO's and the tracer's e1 + e2 differ by 3.4e-10 in their sums.
- **Layered tracer.** It needs the layered plugin's per-launch PASS_SEED (the WL84 convention). With it, the tracer's layered TOTAL matches Mitsuba LAY7 (WL104) at an object mean ratio of 0.9996.

**Design** ([protocol](../../experiments/ecro_wl84_104a/protocol/wl104a.json)). A replicate is 4,096 spp per arm. In each replicate the ECRO, field-table and learned-ER tracer arms share the seed, so the comparison is paired.

Comparisons:
- **Primary, representation (P1):** ECRO HIGH vs learned-ER tracer HIGH.
- **Secondary, material (S1):** learned-ER HIGH and TOTAL vs layered, reported without a verdict.
- **Diagnostics:**
  - D1: the camera readout of the exact field table (the training-target quantity, two independent 16,384-spp estimates) vs learned-ER HIGH. This is the patch/bin projection.
  - D2: ECRO vs that field readout. This is the learned approximation.
- **Controls:**
  - C1: paired LOW;
  - C2: depth truncation, from WL104 halves;
  - C3/A2: tracer totals vs WL104 Mitsuba ER7 / LAY7.

**ROIs**, from geometry before any comparison:
- primary: objects (9,887 px), sphere_lower (1,715 px, shading normal z < 0) and floor_near (5,710 px, within 0.60 m of the sphere axis or 0.25 m of the cube footprint);
- secondary: sphere, cube and full frame.

**Metrics:**
- **Energy:** the signed HIGH energy bias relative to the reference's ROI TOTAL luminance (also relative to HIGH). The interval is a delete-one jackknife over replicates. Field arms add the A/B table-noise term.
- **Spatial:** the stable relative RMS of the RGB HIGH difference, from an unbiased cross-replicate mean square (A × B form for the independent field tables), with a jackknife interval.
- **Descriptive:** relative L1 with its half-split noise floor.

Relative L1 values are never added or subtracted in quadrature.

**Thresholds** (fixed before the pilot):
- τ_E = 1 % of ROI TOTAL. This is below WL104's hard_1 object ratio gap of 1.4 points.
- τ_S = 2 % relative RMS. This is a practically significant fraction of the learned-ER material error, which is 2.7–3.5 % object relative L1.
- Precision target: a 95 % half-width of at most τ/4 for both metrics.

**Verdict rule, per image × primary ROI unit on P1:**
- NOT SUPPORTED if any unit's lower bound exceeds a threshold.
- SUPPORTED if every unit's upper bound is below both thresholds.
- INCONCLUSIVE otherwise, or on any gate failure.

If NOT SUPPORTED, D1 and D2 are tested by the same rule in each failing unit. A correction is allowed only if a control or structural test shows a defect.

**Replicate count.**
- An 8-replicate pilot records interval half-widths only.
- R is then the smallest of 32, 64, 128 and 256 whose projected half-widths meet both targets; 256 is the cap.
- A target the final data misses is reported as unresolved.

**Tests.** `tests104a.py` checks:
- unbiasedness of the stable mean square, including removal of shared table noise in the A × B form;
- coverage of the energy-ratio jackknife;
- the |x| bounds and the RMS floor.

The jackknife interval of the U-statistic mean square is conservative (coverage 0.995). This was accepted before freeze because it can only push a unit toward INCONCLUSIVE.

**Smoke disclosure.** The pipeline smoke (256 spp, 8 replicates, 1,024-spp field tables; non-evidence) ran end to end. Its analysis printed a verdict string, NOT SUPPORTED. Thresholds, ROIs, metrics and rules had already been written to the protocol file before the smoke and were not changed afterwards. No smoke number is used.
