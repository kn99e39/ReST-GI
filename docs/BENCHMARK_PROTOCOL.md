# Benchmark Protocol — Common Workload Suite, References, Metrics and Timing

**Current as of:** 2026-11-24 (established by WL110). **Role:** the standing comparison contract that every future ReST-GI milestone evaluation reuses, so external comparisons are not redesigned per batch. A batch protocol may add workloads or tighten rules prospectively; it may not loosen them after results. Baselines: [registry](EXTERNAL_BASELINE_REGISTRY.md). Decision rules: [Contribution Approval Gate](CONTRIBUTION_APPROVAL_GATE.md). Mandatory correctness gate: [SSS near-contact gate](SSS_NEAR_CONTACT_GATE.md). First executed instance: WL110 (`experiments/bench111/protocol/bench111.json`, `bench111_lr3.json`).

## 1. Workload suite

| ID | Workload | Category | Resolution | Output semantics | Primary reference | Status |
|---|---|---|---|---|---|---|
| W1 | WL100 SSS cfg0: homogeneous SSS sphere on a Lambert floor in the ECRO room; views V0 + held-out H1, H2 (H3–H5 defined) | A, E, near contact | 512² | FULL radiance | REF_FULL (same-semantics PT, kmax 256, RR, 2 × 32,768 spp) | executed WL110 |
| W2 | WL100 near-mirror panel (GGX α 0.01, F0 1) | A, C | 512² | FULL | REF_FULL | executed WL110 |
| W3 | W1 with both boxes moved (cfg0 → cfg1 geometry), sphere fixed | G | 512² | FULL; edit → first frame | REF_FULL at T1 | executed WL110 |
| W5 | W1 with the WL100 second light | H | 512² | FULL; light change → first frame | REF_FULL at the new light | executed WL110 |
| W4 | Bitterli Living Room 3, TransGI conversion (glass removed), authored camera V0 | A, B, C, D, F | 640 × 360 (Table I resolution 1920 × 1080 optional) | GI = FULL − DI (CRN) | Ref-65: Mitsuba `path` depth 65, 2 × 65,536 spp (LabServer69, sha256 `62625251…`) | executed WL110 |
| W6 (future) | One rigid LR3 edit (TransGI's own M1–M3 scenarios) | G | 640 × 360 | GI | Ref-65 at T1 | needs ECRO LR3 edit support (WL102 stage 4 was not run) |
| W7 (future) | SSS contact family: sphere translation, lift-off / contact formation, repeated contact geometries | G, E | 512² | FULL | REF_FULL per state | required by the SSS gate; not yet built |
| W8 (future) | Non-rigid / self-contact | GM3 | — | — | — | out of scope until GM3 |

Fixture states are constructed only through the frozen WL100 configuration rows (`results/ecro_material100/configs.json`), so ECRO, the path tracer and the reference use bit-identical geometry, materials, lights and cameras.

## 2. References

1. **Primary = full-depth physical reference.** Truncation error is never removed from a primary score. For the ECRO fixtures it is the same-semantics path tracer (pt111) at kmax 256 with Russian roulette from event 4 (unbiased), validated against the independent pt100 code path at equal depth (gate G_ADAPT). For LR3 it is Ref-65.
2. **Depth-matched reference** (ECRO's D = 6 / Ref-7) is diagnostic only. It separates representation error from truncation.
3. **Component references** (E1, E2, higher-order; GI / DI) are diagnostic.
4. **Uncertainty.** Each reference is two independent halves. Per-pixel reference variance = (h0 − h1)² / 4 enters the debiased error. Half-vs-half agreement and the D6-vs-full energy are reported for every state.
5. **Semantics.** The SSS fixture uses the native WL100 contact semantics. The F6 contact-ring defect is immaterial at V0 (WL101H). Contact-valid semantics are a pending owner decision and must be declared whenever used; native and contact-valid references are never mixed.

## 3. Metrics (fixed before results)

- **Primary E.** Relative debiased HDR RMSE against the full-depth reference over the full frame: `sqrt(max(mean_rep MSE − mean ref variance, 0)) / mean(ref)`, i.e. the expected single-frame error at that operating point. 95 % bootstrap intervals over replicates (≥ 8 independent seeds for ≤ 64 spp).
- **LR3.** GI LDR RMSE with `clip(x,0,1)^(1/2.2)` is reported alongside E, because every historical LR3 number (paper Table I, WL102, TransGI reproduction) uses it.
- **Secondary.** Relative HDR L1, signed energy bias, LDR RMSE / L1 / PSNR, SSIM (descriptive), ROI errors: sphere, interior, silhouette, contact disk, near, flank, off-target, near-mirror panel, mixed-coverage boundary pixels.
- **SSS near contact.** Signed near bias, near-minus-flank excess (Rec.709 luminance), radial signed profile in nine floor rings (0–0.5 m).
- **Temporal.** For non-temporal estimators, frame-to-frame instability equals replicate variance (reported through E). Methods with history (TransGI probes, SVGF, NRC) must report steady-state and post-edit response frames separately.
- **Seeds.** Hash → splitmix64 launch seeds. Consecutive `seed·1000003` streams of the ECRO PCG are correlated, so they are not used for replicates.

## 4. Timing (fixed before results)

- **Host.** Main, RTX 5080 16 GB, driver 610.88, Windows 11. The desktop compositor shares the GPU; this is disclosed. Historical RTX 3090 / 3080 Ti timings are labelled cross-hardware and are never put on the same frontier axis without that label.
- **Exclusive windows.** No other Python / CUDA worker process, and median GPU utilisation ≤ 10 % over 3 s right before and after each block. A failed check repeats the block (at most 3 attempts), then flags it contaminated.
- **Static frame.** From the call with all resident state ready to the image in GPU memory, synchronised before and after. The denoiser counts for PTDN. Presentation is excluded.
- **Order and statistics.** 10 warm-up calls, then 5 rounds in randomised order with 20 frames per round (5 for > 64 spp). First call (compile / record) reported separately. Median, p95, p99, mean, with a bootstrap CI of the median.
- **Edit → first frame.** From new geometry / light parameters to the first frame in GPU memory. Arms alternate T0 → T1 → T0. ECRO reports selective, full-rebuild and stale control; PT reports BVH update + frame.
- **Lifecycle accounting.** Separate one-time costs (training-data generation, training, precomputation, checkpoint), per-edit costs (relation / BVH update, source, generator, retraining if invalidated) and per-frame costs. Neural Full Recompute is the native rebuild lifecycle; path-traced re-render is never a stand-in for it.

## 5. Frontier construction

Per workload and view (never averaged across workloads):
- Each arm contributes its spp ladder as (median ms, E with CI) points.
- The comparator frontier is the lower envelope over all valid executed comparators. Interpolation is piecewise linear in (log ms, log E) between executed points, never extrapolated.
- The strongest comparator is identified separately at equal time and at equal quality.
