# Worklog — Learned Material Rules: Rendering Qualification and End-to-End Utility Closure (2026-11-10)

## Origin

**Brief.** WL84 (user, 2026-11-10). The batch resolves whether WL83's sole learned-ER rendering failure reflects reference unreliability or a mechanism deficiency, and measures uncontended rendering economics.

**Not changed.** The closures, compiler, parameters, scenes, cameras, lights, thresholds and image selection are all unchanged. WL83 OUTCOME D stays immutable.

**Where.** Branch `Selective_Recompute`; LabServer63 (idle at start); `experiments/fnto_compiled84/`, `results/fnto_compiled84/`, `results/evaluation/84/`, server `~/rest_compiled84` (dev `~/rest_compiled84_dev`).

## Design (frozen in `protocol/compiled84.json`)

**Images (predeclared).** All images use the WL83 arms: C0_DF, E_DF, ER_DF, E_L and ER_L (seed-8301 learned states).
- test83_iid_1_L2: the sole WL83 ER_L failure. It was reference-limited: uncertainty 0.0265 > 0.25 × error 0.1019. Its highlight error was 0.0952 against the 0.10 threshold.
- test83_iid_0_L2: a passing control under the same grazing light.
- test83_hard_1_L1: ER_L passed here. It is the E_L genuine-failure negative baseline.

**A. Audit.** The stored WL83 64-spp renders are reproduced bit-identically with their WL83 seeds through the WL83 path. Their errors are recomputed and the material-state hashes recorded.

**B. Independent reference.** The unchanged layered-reference plugin renders two independent halves of 32768 spp, i.e. 65536 spp, 16× WL83.
- The budget is fixed; there is no adaptive stopping.
- Seeds 410000+ are disjoint from WL83's.
- Cumulative levels (8192 / 16384 / 32768 spp per half) are stored and reported.

**C. Evaluation.**
- The WL81 Stage F rule is unchanged. It is applied to the stored WL83 candidates against the new reference.
- The time-matched reference spp comes from the uncontended GPU time of stage D.
- Per-image verdicts are PASS / FAIL / INDETERMINATE.
- Eight 64-spp replicates per arm (ER_L, E_L, new seeds) measure candidate Monte Carlo variability. They are descriptive.

**D. Economics (alone, foreign GPU processes recorded).**
- Per 64 spp: ray generation, trace, GPU execution, kernel GPU time, readback and wall, plus first-call and binding times. The WL83 pipeline is timed for comparison.
- Time-to-quality ladders vs the new reference: REF 64–4096 spp, closures 16–1024 spp.
- ECONOMICS_FAVOURABLE iff, for every image, ER_L's GPU time per 64 spp ≤ the layered reference's / 1.10, and the layered reference needs ≥ 1.10× ER_L@64's GPU time to reach ER_L@64's image error.

**Outcomes.**

| Outcome | Condition |
|---|---|
| E | audit fails or contention |
| C | target image INDETERMINATE |
| B | any ER_L FAIL |
| D | ER_L PASS on all three, economics unfavourable |
| A | ER_L PASS on all three, economics favourable |

## Pre-freeze probes (full scale) and corrections

1. **WL83 render path dominated by host work.** Full-scale probe of the WL83 path (layered reference, 2048 spp):
   - wall 28–34 s, of which numpy camera-ray generation 21–26 s, host readback 5 s, trace 0.8–6.4 s and GPU only 0.39 s;
   - GPU utilisation 3–4 %.

   WL83's render wall times were therefore mostly host-side ray generation, not material or tracing cost.
2. **GPU-resident pipeline for new renders** (`render_g`). It generates rays on the GPU (same camera and per-sample uniform pixel jitter; Dr.Jit PCG32 instead of numpy) and accumulates per pixel on the GPU.
   - First version (float64 `block_sum`, pixel-major lanes): 5× the path-tracing GPU time.
   - Final version (float32 scatter-add, sample-major lanes, float64 across launches): GPU 0.81 s for 4096 spp, equal to path tracing alone.
   - Statistically equal to the WL83 path: mean ratio 0.99983, equal distance to the WL83 reference.
   - Wall 7.2 s vs about 65 s.

   The WL83 path is kept for the bit-identical reproduction and as a timing comparison.
3. **Audit tolerance.** The errors recomputed from the float32-stored renders differ from WL83's float64 values by up to 8e-9, so the tolerance is 1e-6. The real criterion, bit-identical reproduction, held for every image and arm in the dev probe.
4. **Seeds.** Mitsuba's `sampler.seed` is uint32 and render seeds are multiplied by 7919, so the WL84 seeds were moved to 410000–480000 (disjoint from WL83's maximum of about 383,600).
5. **Contention check.** The check had counted its own process; it is now excluded.
6. **Dev probe of the whole chain** (reduced budgets; outcome not interpreted): 3 min 8 s, every stage ran.
   - Expected full run: about 12 min.
   - GPU utilisation is limited by Python tracing in the timing stages (2–25 %); the reference stage is GPU-resident.
