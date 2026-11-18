# Worklog 101B — ECRO low-SPP reconstruction

## Authorization and frozen plan, 2026-11-18 KST

This bounded implementation directly follows [WL101A](101A_WORKLOG_ECRO_SSS_LOW_SPP_ARTIFACT_ATTRIBUTION.md) under the same learned transport and physical estimator. Its suffix follows the evidence-gap numbering rule. Work is in the existing ReST-GI checkout on `ecro_lr3_adapter`. WL101A's frozen LOW-only INCONCLUSIVE verdict remains part of its historical record.

The selected reconstruction is joint-RGB SVGF: temporal reprojection, luminance moments and variance-guided à-trous filtering. The filter processes the composed noisy HDR image because LOW and image-space HIGH share samples. The pipeline avoids demodulation of the SSS image. Its albedo parameter does not describe a diffuse surface factor. [Schied et al., 2017](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/)

WL101 has center-ray first-hit position, normal, shape/material and camera data, and it owns current geometry and lighting. It provides no camera motion vectors or history interface. First-hit queries supply guides on the existing BVH. For camera-only motion, project the world position into the previous camera and test visibility compatibility. The whole radiance history resets on geometry or light edits, since first-hit correspondence alone cannot certify unchanged indirect light. The original transport sources and reference arrays are read-only.

The 512² primary evaluation uses the original SSS checkpoint, fixed camera, archived six-event references and existing eight-state WL101 SSS tour. Camera motion is a separate check with a return to the archived camera. The frozen protocol owns the input identities, tests, reconstruction settings, quality thresholds and latency gate. [Protocol](../../experiments/ecro_sss101b/protocol/reconstruction.json)

### Integration preflight before primary evidence

Triton 2.3.1 rejected repeated `constexpr` annotations on the unrolled tap weights. The correction uses regular compile-time assignments with the same stencil. The first running preflight reproduced the original composed 1/4-spp inputs exactly and recovered the archived geometric guides. Nearest-pixel position/normal matching alone retained history on 1.55% of 7,553 independently occluded or off-screen pixels. Before freezing the primary run, the candidate added a previous-eye segment visibility query on camera movements. A separate closest-hit query scores that rejection. Static frames and history resets on geometry edits do not trace the extra motion query. The preflight used raw data and a constant-color input for the filter controls.

## Primary evidence, 2026-11-18 KST

### Implementation fact

The experimental wrapper invokes WL101's original selective edit lifecycle and frame program. A Triton kernel adds the native E1, E2 and HL channels and divides by SPP. The reconstruction uses two history buffers for RGB, luminance moments and age, then estimates uncertainty and filters the joint image in three 5×5 à-trous passes. A bounded sixteen-frame average is used for valid history. Object ID, geometric normal and distance to the tangent plane guide the spatial weights. The history and spatial filter operate in HDR without clipping. The online pass has no physical-reference inputs.

The frozen run reproduced both composed frames bit for bit at its paired 1- and 4-spp seeds and matched the archived cfg0 response table exactly. The live guide's shape-ID mismatch was zero. Position and normal RMS differences versus the archived guide were 1.05e-7 m and 4.33e-8. Constant color stayed constant and reset initialized RGB to the current raw frame. In the camera preflight, the added visibility query reduced history retention on independently disoccluded pixels from 1.55% to 0 of 7,553. Camera-only motion uses cached per-view recordings in the timing sequence.

### Measurement — latency and memory

Device-resident RGB completion on LabServer63 RTX 3080 Ti. The geometry tour uses 32 measured warm frames per arm and SPP. End-to-end wall includes the original geometry, relation and source updates, response evaluation, ray tracing and joint RGB. The reconstructed arm adds guide generation and reconstruction. Readback and file writing are excluded.

| SPP | Raw edit median / p95 (ms) | Reconstructed edit median / p95 (ms) | Reconstruction GPU median / p95 (ms) | Warm camera-only median / p95 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 15.20 / 16.23 | 15.87 / 17.81 | 0.242 / 0.243 | 5.95 / 6.11 |
| 4 | 23.63 / 23.86 | 24.30 / 24.94 | 0.242 / 0.244 | 14.23 / 14.37 |

The spatial/temporal kernel span excludes guide generation. Combined measured end-to-end overhead is about 0.67–0.68 ms. The reconstruction and guide buffers hold 27.0 MiB. Peak PyTorch allocation increases by 27.9 MiB at 1 spp and 28.0 MiB at 4 spp. These allocation deltas cover the measured geometry tour. They do not measure total driver or renderer memory. No foreign GPU process was observed at the tour timing boundaries. The camera-only column includes its extra visibility and reprojection work, without a geometry or response update. The 4-spp raw edit path already exceeds 20 ms on the measured stack.

### Measurement — static and edit quality

Every geometry state used its archived 2 × 4096-spp six-event physical reference. For every SPP there were four independent repeats. Static cold denotes the first frame. Static steady aggregates frames 17–24. The geometry tour aggregates eight current-state frames per repeat, including the first frame at each rigid edit. Values are pooled relative HDR RMSE over the listed frame/ROI. Raw and reconstructed values use the same noisy frame.

| SPP and sequence | Full raw → reconstructed | Sphere raw → reconstructed | Sphere boundary raw → reconstructed | Floor raw → reconstructed | Contact/shadow raw → reconstructed |
|---|---:|---:|---:|---:|---:|
| 1, cold | 22.86% → 2.52% | 77.64% → 10.53% | 55.47% → 14.67% | 53.02% → 5.74% | 95.61% → 9.75% |
| 1, steady | 22.83% → 1.19% | 77.30% → 5.86% | 54.64% → 9.33% | 53.03% → 2.20% | 95.22% → 4.12% |
| 1, geometry tour | 23.08% → 2.56% | 77.68% → 10.95% | 55.78% → 15.15% | 52.40% → 5.57% | 94.88% → 9.65% |
| 4, cold | 11.41% → 1.65% | 38.71% → 7.73% | 28.09% → 11.63% | 26.49% → 3.54% | 47.69% → 5.93% |
| 4, steady | 11.42% → 0.87% | 38.64% → 4.35% | 27.47% → 6.91% | 26.56% → 1.64% | 47.74% → 3.17% |
| 4, geometry tour | 11.54% → 1.71% | 38.80% → 8.35% | 28.22% → 12.63% | 26.18% → 3.47% | 47.41% → 6.13% |

Tour full-frame HDR RMSE falls from 1.254 to 0.139 at 1 spp and from 0.627 to 0.093 at 4 spp. Relative HDR L1 falls from 23.21% to 2.45% and from 11.93% to 1.44%. Sphere LDR RMSE on the same tour falls from 0.1624 to 0.0167 and from 0.0805 to 0.0106. The frozen improvement checks for full color in the sphere, floor and contact/shadow all pass. [Per-frame and ROI metrics](../../evaluation/101B/metrics.json)

Contact/shadow gradient RMS stays close to the physical reference. Reconstructed/reference ratios are 0.983 and 1.034 at steady state and 0.971 and 0.939 during geometry edits for 1 and 4 spp. The sphere boundary retains excessive gradient magnitude. Its corresponding steady ratios are **1.361 and 1.255**, and edit ratios **1.521 and 1.451**, above the frozen 1.20 ceiling. Its gradient relative RMSE falls strongly versus raw but remains 0.737 / 0.574 steady and 0.941 / 0.861 at edits. Gradient comparisons use adjacent pairs that cross or touch the ROI.

### Measurement — temporal stability and disocclusion

Steady temporal variance on the sphere falls to 0.0135% / 0.0265% of raw at 1 / 4 spp. On the floor it falls to 0.0085% / 0.0142% of raw in the same eight-frame windows. Every geometry-tour reconstruction is bitwise equal to its same-frame forced-reset control. Across measured edits, mean full-frame signed-change HDR RMSE falls from 1.772 to 0.175 at 1 spp and from 0.884 to 0.114 at 4 spp. The signed-change slope is 0.978 / 0.986 full-frame and 0.535 / 0.558 on the sphere. Thus a change can be attenuated by the current spatial reconstruction or inherited estimator even when earlier radiance contributes zero. The small local GI changes are a descriptive measurement on this tour.

For camera A→B, 7,553 current pixels were independently occluded or out of the prior field of view. For B→A, 6,627 were. None retains direct temporal history in any repeat or SPP. On return to A their mean HDR RMSE is 0.175 / 0.126 after reconstruction and 0.190 / 0.130 with a same-frame spatial reset. The spatial filter can still use neighboring pixels with valid history. The measured difference from that reset on the disoccluded region is about 0.052 / 0.027 linear HDR RMS, while the physical-reference error is lower in every repeat than the same-frame spatial reset.

### Observation

The raw fine-grained noise on the sphere, floor and walls largely disappears. The first reconstructed frames, including the geometry edits, retain broad mottling in the sphere and shadows, strongest at 1 spp. Static accumulation makes the sphere interior and floor visibly smoother. The edge remains irregular and its contrast differs from the physical silhouette. The geometry animation shows a current shadow at each edit and no carryover of the preceding shadow. [1-spp stills](../../evaluation/101B/static_1spp.png) · [4-spp stills](../../evaluation/101B/static_4spp.png) · [1-spp geometry tour](../../evaluation/101B/tour_1spp.gif) · [4-spp geometry tour](../../evaluation/101B/tour_4spp.gif)

### Interpretation

After steady reconstruction, the sphere's reference-noise-corrected stable error is **5.25% / 3.89%** relative HDR RMS at 1 / 4 spp. The inter-replica stochastic RMS is **2.25% / 1.60%**. In the boundary ring, the corresponding stable errors are **8.98% / 6.63%**, with stochastic RMS **2.42% / 1.87%**. In the sphere interior the remaining error is mostly stochastic. The stable full-render HIGH residual measured by WL101A is 1.05% for the sphere and 1.59% at the boundary. The fixed filter cannot correct that underlying learned response.

A post-hoc attribution control compared the same four reconstructed steady frames to the independent, seed-paired WL101A converged ECRO predictions. The stable discrepancy against that unchanged renderer is **5.04% / 3.64% on the sphere and 8.66% / 6.26% at its edge**. The same-scale residual panels show a dark silhouette ring added by reconstruction, while the pre-existing HIGH residual is concentrated around contact and seams. These are separate errors. The suggested mechanism for the added ring is filtering random subpixel coverage with a single center-ray object/normal guide. The control isolates the reconstruction pipeline as the source of the extra residual. It does not separately manipulate guide coverage, temporal weighting or spatial taps. [Signed residual maps](../../evaluation/101B/residual_attribution.png) · [Post-hoc numbers](../../evaluation/101B/residual_attribution.json)

### Unresolved question

The result qualifies the sampled homogeneous SSS scene and its finite-depth physical reference. Camera motion was tested for two warmed viewpoints, with independent reference scoring on return to the archived view. Continuous new viewpoints may trigger new WL101 recordings and were not timed. The experiment does not establish quality of moving-camera reconstruction from independently converged references at other views. Global reset is deliberately conservative for transport-changing edits. Reusing radiance through continuous deformation, bending or self-contact remains untested.

### Architecture verdict

**The frozen combined quality-and-latency objective is not met at either SPP.** At 1 spp the geometry edit path meets 20 ms and stochastic grain is strongly suppressed. The sphere boundary fails the quality gate both after reset and at steady state. At 4 spp it still fails, and the unfiltered 4-spp edit path already exceeds 20 ms. The measured filter implementation itself costs approximately 0.24 ms on the GPU. Rigid edits meet the history-validity criterion, but resetting history leaves each animated frame with spatial-only reconstruction and its remaining edge errors. The limiting fidelity mechanism is the added reconstruction bias near the silhouette, with an independently measured pre-existing HIGH error at contact. The run stops with this single candidate.

### Reproducibility

- Renderer and protocol freeze `75e16561f9f87bd154785e06cb9cbe1a7bcefabe`. Offline primary scoring `a96ed2d244c1a8e1e4d860d777ce6c44a09df6ff`. The post-hoc control is `residual_review.py` in the experiment directory. No filter settings changed after the primary freeze.
- Runtime is the original LabServer63 WL101 deployment, Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, Triton 2.3.1, NumPy 1.26.4, Python 3.10.21, RTX 3080 Ti and driver-matched process OptiX 595.71.05. No transport code was rebuilt. GPU timings were independently collected for this batch, separate from WL101's RTX 3090 and WL101A's static-render timings.
- `run.py WL101_ROOT FRESH_RAW_ROOT COMMIT` uses the staged WL101 model, renderer, archive guide and archived response table identity. Use the existing virtual environment, set `DYN101_MI_DIR` to the staged Mitsuba package, `DYN101_MI_VERSION=3.8.0`, and prepend the matched OptiX path to `LD_LIBRARY_PATH`. No physical reference is passed to reconstruction.
- `analyze.py RAW_ROOT WL101_REF_ROOT REVIEWER_ROOT` verifies reference and output hashes and generates metrics, images and animations. `residual_review.py RAW_ROOT WL101_REF_ROOT WL101A_RAW_ROOT REVIEWER_ROOT` reproduces the marked post-hoc comparison. The eight original reference pairs, seeds, per-frame HDR outputs, history ages, disocclusion masks and raw timings are under `results/ecro_sss101b/` in this project. The server copy is in the same subpath under the existing WL101A compute workspace. [Run manifest](../../evaluation/101B/run_manifest.json) · [Reviewer gallery](../../evaluation/101B/index.html)
