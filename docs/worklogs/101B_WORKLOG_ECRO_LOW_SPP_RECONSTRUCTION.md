# Worklog 101B — ECRO low-SPP reconstruction

## Authorization and frozen plan, 2026-11-18 KST

This bounded implementation directly follows [WL101A](101A_WORKLOG_ECRO_SSS_LOW_SPP_ARTIFACT_ATTRIBUTION.md) under the same learned transport and physical estimator. Its suffix follows the evidence-gap numbering rule. Work is in the existing ReST-GI checkout on `ecro_lr3_adapter`. WL101A's frozen LOW-only INCONCLUSIVE verdict remains part of its historical record.

The selected reconstruction is joint-RGB SVGF: temporal reprojection, luminance moments and variance-guided à-trous filtering. The filter processes the composed noisy HDR image because LOW and image-space HIGH share samples. The pipeline avoids demodulation of the SSS image. Its albedo parameter does not describe a diffuse surface factor. [Schied et al., 2017](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/)

WL101 has center-ray first-hit position, normal, shape/material and camera data, and it owns current geometry and lighting. It provides no camera motion vectors or history interface. First-hit queries supply guides on the existing BVH. For camera-only motion, project the world position into the previous camera and test visibility compatibility. The whole radiance history resets on geometry or light edits, since first-hit correspondence alone cannot certify unchanged indirect light. The original transport sources and reference arrays are read-only.

The 512² primary evaluation uses the original SSS checkpoint, fixed camera, archived six-event references and existing eight-state WL101 SSS tour. Camera motion is a separate check with a return to the archived camera. The frozen protocol owns the input identities, tests, reconstruction settings, quality thresholds and latency gate. [Protocol](../../experiments/ecro_sss101b/protocol/reconstruction.json)

### Integration preflight before primary evidence

Triton 2.3.1 rejected repeated `constexpr` annotations on the unrolled tap weights. The correction uses regular compile-time assignments with the same stencil. The first running preflight reproduced the original composed 1/4-spp inputs exactly and recovered the archived geometric guides. Nearest-pixel position/normal matching alone retained history on 1.55% of 7,553 independently occluded or off-screen pixels. Before freezing the primary run, the candidate added a previous-eye segment visibility query on camera movements. A separate closest-hit query scores that rejection. Static frames and history resets on geometry edits do not trace the extra motion query. The preflight used raw data and a constant-color input for the filter controls.
