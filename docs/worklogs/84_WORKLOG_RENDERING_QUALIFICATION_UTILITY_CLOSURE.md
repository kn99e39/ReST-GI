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

## Run 1 (`30a2e58`) — INVALID — and amendment 1

**Run 1 completed in 9.5 min.** The audit reproduced the WL83 renders bit-identically, and the frozen mapping returned OUTCOME B: ER_L "FAIL" on all three images. Two anomalies:
- every stored candidate had error about 0.50 against the new reference;
- the old and new references differed by 0.50 (relative L1).

**Cause.** `render_g` scatter-added the **unevaluated** integrator output. In Dr.Jit 0.4.6 that sums every sample twice:
- new / WL83 reference = 2.00008 and 1.99997;
- replicate / stored candidate = 2.001;
- isolated test: lazy scatter 2.0, scatter after `dr.eval(Lr)` 1.0.

The pre-freeze probe had validated this variant's GPU time but not its values. **Every run-1 quantity produced by `render_g` is invalid** (references, replicates, time-matched renders, time-to-quality). Run 1's OUTCOME B is void; it is a measurement-validity failure. Run 1 is preserved on the server in `results_run1_invalid`.

**Amendment 1** (`protocol/amendment1_render_g_eval.json`, post-evidence, labelled):
- `dr.eval(Lr)` before the scatter;
- new gate A2 (`validate_g`) before any reference:
  - the analytic Lambert plane ratio must be within 1e-3;
  - `render_g` and the WL83 path must agree within 0.5 % (object-summed ratio) on every WL84 scene at 1024 spp.
- Dev check of the fix:
  - Lambert ratio 0.999996;
  - layered ratios 0.99915 / 1.00039 / 1.00171.
- A first 4-se version of the gate was replaced by the 0.5 % ratio before any run-2 evidence. Its standard error ignored the RGB-channel correlation, giving z 8.7 on the hard material. The conservative z is reported instead.
- Nothing else changed: images, arms, budgets, seeds, rules and the outcome mapping.

## Run 2 (`10d1f92`, LabServer63, 2026-11-10 09:59–10:09 UTC, sequential single-process stages, no foreign GPU process)

| Stage | Wall | Mean GPU | Result |
|---|---|---|---|
| A audit | 32 s | 5 % | REPRODUCED (all 18 stored WL83 renders bit-identical; errors re-derived to ≤ 1e-8) |
| A2 GPU-pipeline validation | 72 s | 4 % | PASS: Lambert ratio 0.999996; layered ratios 0.99915 / 1.00039 / 1.00171 (conservative z −0.49 / 0.45 / 1.54) |
| B three references | 95–102 s each | 28–39 % | 65536 spp each |
| C replicates | 10 s per image | 0–7 % | 8 × 64 spp per arm |
| D economics | 185 s | 9 % | alone |
| C evaluation, visuals, verdict | 16 s | — | **OUTCOME A** |

GPU utilisation stays low because Dr.Jit 0.4.6 re-traces Python per launch: about 0.4–1.2 s per call, independent of spp.

## Final report

### Implementation Fact

- **Unchanged:**
  - learned ER / E closures (WL83 compiler seed 8301 states; hashes in `audit84.json`);
  - scenes, cameras, lights, integrator (path, max depth 4);
  - the layered-reference plugin;
  - WL81 Stage F metrics and thresholds;
  - the stored WL83 64-spp candidate renders, which are the evaluated candidates.
- **New:**
  - an independent layered reference per image: two halves × 32768 spp, seeds disjoint from WL83;
  - 8 replicate 64-spp renders per learned arm;
  - an uncontended timing harness splitting ray generation, trace, GPU execution, kernel GPU time and readback;
  - a GPU-resident render path for new renders (`render_g`), validated against the WL83 path and an analytic solution.
- **Failed attempt preserved:** run 1 (`30a2e58`), invalid because `render_g` summed each sample twice. Amendment 1 fixed it.

### Measurement

**Reference qualification.**

| Image | WL84 uncertainty (65536 spp) | WL83 uncertainty (4096 spp) | Old-vs-new reference rel. L1 | Highlight-mask Jaccard |
|---|---|---|---|---|
| test83_iid_1_L2 | 0.0066 | 0.0265 | 0.028 | 0.993 |
| test83_iid_0_L2 | 0.0040 | 0.0157 | 0.017 | 0.994 |
| test83_hard_1_L1 | 0.0039 | 0.0155 | 0.016 | 0.961 |

- The uncertainty drops 4×, i.e. by √16.
- The old-vs-new differences are consistent with the old reference's noise.
- Every arm's error is now ≥ 4× the reference uncertainty, so **no image is reference-limited**.

**Rendering fidelity** (stored WL83 64-spp candidates vs REF84; WL81 Stage F rule; time-matched reference from uncontended GPU time).

| Image | Arm | Image L1 | Highlight L1 | Time-matched REF (spp) | Verdict | Replicates: highlight L1 (8 seeds) |
|---|---|---|---|---|---|---|
| test83_iid_1_L2 | ER_L | 0.0995 | 0.0930 | 0.235 (44) | **PASS** | 0.0928 ± 0.0011, 8 / 8 ≤ 0.10 |
| | E_L | 0.0957 | 0.0896 | 0.247 (41) | PASS | 0.0894 ± 0.0018, 8 / 8 |
| | C0_DF | 0.1115 | 0.1044 | 0.256 | FAIL (highlight) | — |
| test83_iid_0_L2 | ER_L | 0.0884 | 0.0756 | 0.142 (41) | **PASS** | 0.0759 ± 0.0010, 8 / 8 |
| | E_L | 0.0815 | 0.0714 | 0.144 | PASS | 0.0729 ± 0.0008, 8 / 8 |
| test83_hard_1_L1 | ER_L | 0.0849 | 0.0326 | 0.140 (39) | **PASS** | 0.0328 ± 0.0008, 8 / 8 |
| | E_L | 0.1367 | 0.1346 | 0.143 | **FAIL** (highlight) | 0.1359 ± 0.0007, 0 / 8 |
| | C0_DF | 0.1684 | 0.2095 | 0.148 | FAIL | — |

- The masks from the old reference give the same highlight errors to within 0.0004.

**Sampling correctness.**
- The stored renders are reproduced bit-identically.
- Replicate image errors lie in a narrow band (ER_L 0.098–0.102 on the target image). The stored render is typical of its seed distribution.
- The GPU path agrees with the WL83 path to within 0.17 %.

**GPU execution economics** (uncontended RTX 3080 Ti, driver 595.71.05, kernel-history GPU time per 64-spp image):

| Image | REF | ER_L | E_L | C0_DF |
|---|---|---|---|---|
| test83_iid_1_L2 | 14.7 ms | 10.1 ms | 9.6 ms | 8.9 ms |
| test83_iid_0_L2 | 15.3 ms | 9.9 ms | 9.6 ms | 9.0 ms |
| test83_hard_1_L1 | 16.4 ms | 10.1 ms | 9.6 ms | 9.0 ms |

- REF / ER_L = 1.45–1.63×. Closure binding takes 2.3–2.7 ms.

**Time-to-quality** (GPU, REF ladder interpolated): the layered reference needs **6.3× / 2.8× / 2.9×** ER_L@64's GPU time to reach ER_L@64's image error (0.099 / 0.089 / 0.086).
- ER_L's error flattens at about 0.035–0.043 (1024 spp), a bias floor.
- The reference passes ER_L at about 1024 spp (about 150–250 ms GPU per image).
- ECONOMICS_FAVOURABLE: **true** on all three images.

**End-to-end wall clock** (GPU-resident path):

| Image | REF / 64 spp | ER_L / 64 spp |
|---|---|---|
| test83_iid_1_L2 | 0.48 s | 0.48 s |
| test83_iid_0_L2 | 0.89 s | 0.89 s |
| test83_hard_1_L1 | 1.30 s | 1.29 s |

- About 85–95 % of the wall time is Python tracing: 0.40 / 0.81 / 1.22 s per call, identical for every arm.
- In wall time, time-to-quality favours ER_L by only about 1.0–1.7×.
- The WL83 pipeline costs 1.2–2.1 s per 64-spp image, of which about 0.66 s is numpy ray generation. WL83's render wall times were therefore dominated by host-side ray generation, not by material evaluation.

### Observation

1. **WL83's sole ER_L failure was a reference-qualification artefact.** With a 16× reference:
   - the uncertainty is 4× lower;
   - ER_L passes test83_iid_1_L2 under the unchanged rule, and also passes both controls;
   - the margin on the target is small (highlight 0.093 vs 0.10) but stable across 8 independent seeds (sd 0.001).
2. **E_L's hard-material highlight failure is genuine and reproducible:** 0.135–0.137, 0 / 8 replicates pass. ER_L passes the same image with 0.033.
3. **Inside the renderer, the closures cut GPU time per spp by about 1.5×.** They reach a 64-spp-equivalent quality level 2.8–6.3× faster than the layered reference. The wall clock, however, is set by Python tracing, so no wall-clock advantage exists at 64 spp in this harness.
4. **The closures have a bias floor of about 0.035–0.04 image L1.** Beyond about 1024 spp the layered reference is more accurate per unit GPU time.

### Interpretation

**The primary hypothesis is supported on this suite.** WL83's rendering-gate failure for learned ER came from insufficient reference qualification, not from an identified scattering deficiency.

**Learned ER's local-rendering cost advantage is real in GPU execution and moderate (about 1.5× per spp), and it is limited to quality levels above its bias floor.** In this Python-traced Mitsuba harness the advantage does not reach wall time.

### Unresolved Question

- **Suite size.** Only 3 of the 10 WL83 images were re-qualified. The other 7 ER_L images passed in WL83 with qualified references, but their time-matched comparisons used concurrent GPU timing (closure 0.08–0.10 vs time-matched reference 0.13–0.23).
- **Wall-clock economics** under a production renderer (frozen functions / megakernel / C++ BSDF) are untested.
- **Scope.** The bias floor and its effect on high-quality targets; generality to other scenes, light transport and dynamic GI; FNTO integration.

### Architecture Verdict

| Entry | Verdict |
|---|---|
| Reference Qualification | **QUALIFIED** (uncertainty 0.004–0.007, ≤ 1/4 of every arm's error; old-vs-new consistent) |
| Rendering Fidelity | **ER_L PASS** on all three images (target margin 0.007 in highlight error, stable over 8 seeds). E_L FAIL on the hard image (genuine). C0_DF FAIL on 2 / 3 |
| Sampling Correctness | **PASS** (bit-identical reproduction; replicate stability; GPU path validated) |
| GPU Execution Economics | **FAVOURABLE** (1.45–1.63× per spp; 2.8–6.3× time-to-quality at the 64-spp quality level) |
| End-to-End Rendering Economics | **NOT FAVOURABLE in wall time in this harness** (equal per-64-spp wall; 1.0–1.7× time-to-quality); limited by Python tracing, not by the closure |
| Learned Local Material Rule Viability | **Supported within the tested scope** (one coated-conductor family, 3 images, direct lighting + limited indirect, static scene, ≤ ~1024 spp quality targets) |
| FNTO Integration Readiness | **NOT TESTED** |

- **WL84 OUTCOME A** (frozen mapping, run 2). WL83 OUTCOME D remains the historical verdict.
- Nothing is adopted into canonical FNTO. Whether Learned Material Rules remain a ReST-GI component candidate is a human decision.

### Reproducibility

- **Commits:** freeze `30a2e58` (run 1, invalid); amendment 1 `10d1f92` (run 2); report in the closing commit.
- **Server:** `~/rest_compiled84` (run 2 `results/`, run 1 `results_run1_invalid/`, `chain.status`, `chain_run1_invalid.status`, `watch.out`).
- **Local copy:** `results/fnto_compiled84/results/`. It contains `verdict84.json`, `eval84.json`, `econ/econ84.json`, `audit/audit84.json` + `validate_g.json`, reference timing JSONs and logs. The large `.npy` images (references, replicates) stay on the server.
- **Seeds:** references 410000+, time-matched 450000+, replicates 460000+, economics 470000+.
- **Environment:** RTX 3080 Ti, driver 595.71.05, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6.
- **Visuals:** `results/evaluation/84/` (manifest with sha256):
  - references and candidates with error maps on one shared scale;
  - time-to-quality;
  - replicate highlight errors.
