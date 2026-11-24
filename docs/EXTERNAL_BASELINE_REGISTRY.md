# External Baseline Registry and Workload Compatibility

**Current as of:** 2026-11-24 (WL111). **Role:** one current record of every baseline ReST-GI is compared with: what it is, where it runs, what it supports, and on which workloads a comparison is qualified. Measurements and execution history live in the owning Worklogs; this document links them. The benchmark contract (references, metrics, timing) is in the [benchmark protocol](BENCHMARK_PROTOCOL.md); the decision rules are in the [Contribution Approval Gate](CONTRIBUTION_APPROVAL_GATE.md).

Baseline roles and the binding Neural Full Recompute terminology stay owned by the [evidence strategy](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md#mandatory-baseline-terminology-and-comparison-gate). An unavailable implementation is never evidence that a method is inferior.

## Classification vocabulary

| Class | Meaning |
|---|---|
| QUALIFIED | Same physical scene and output semantics, authenticated implementation, qualified reference, comparable timing on the benchmark host |
| QUALIFIED_WITH_LIMITATIONS | As above but one declared limitation (historical / cross-hardware timing, a configuration mismatch between the quality and timing runs, single seed) |
| UNQUALIFIED | Runs, but the comparison contract fails (reference, semantics or timing not comparable) |
| UNSUPPORTED | The method's published scope excludes the workload (material, edit or lighting class) |
| IMPLEMENTATION_UNAVAILABLE | No runnable implementation in this project's stacks |
| REPRODUCTION_REQUIRED | An implementation exists but running it here needs reproduction or engine-integration work beyond configuration |

## Registry

| Baseline | Version / provenance | Evidence class | Materials | Geometry edits | Lighting | Training / preprocessing | Runtime stack (this project) | Known discrepancies |
|---|---|---|---|---|---|---|---|---|
| **Path tracing, same-semantics (pt111)** | WL111 benchmark adapter around the frozen ECRO scene code (pt100 statements in a full-depth symbolic loop, RR from event 4) | Internal physical comparator; validated against the independent pt100 path (G_ADAPT) | Everything the ECRO fixtures contain (Lambert, ER near-mirror, homogeneous SSS walk, ballistic) | Any (BVH update) | Any point light | None | Mitsuba 3.9.1 / Dr.Jit 1.5.0, Main RTX 5080 | Python-traced but frozen (`dr.freeze`) megakernel; not Mitsuba's C++ integrator |
| **Path tracing, Mitsuba stock `path`** | Mitsuba 3.9.1 (Main), 3.6.4 (references, LabServer69) | Official renderer | Mitsuba BSDFs; dielectrics removed in the LR3 conversion | Any | Any | None | Main | GI = full(65) − full(2) with common random numbers (TransGI / WL102 convention) |
| **Path tracing + OptiX AI denoiser** | Mitsuba 3.9.1 `OptixDenoiser` (HDR model, albedo + normal guides, no temporal history) | Official renderer + vendor denoiser | as PT | as PT | as PT | Pretrained vendor network, no per-scene training | Main | Single-frame; a temporal (flow-guided) mode exists but was not used (static camera, no history); SVGF (WL101B) is an internal candidate, not this baseline |
| **Cycles 3.5.1** | Official Blender build (CUDA) | Physical reference for WL53–WL70 fixtures | Volume RTE (validated), BSSRDF (path-history-dependent semantics, WL51) | Any | Any | None | LabServer69 | Not used as a practical competitor in WL111 |
| **8DNA** (arXiv 2604.25129) | Official `lwwu2/8dna26` @ `4a2157c`, unmodified | Official release | Per-asset baked surfaces and homogeneous volumes (all ≥ 2-event paths inside the asset) | Native reuse: translation and 90° rotations exact, general rotation approximate, scale wrong; internal / non-rigid edit needs a full re-bake | Lighting-independent; envmap / area / directional at render time | Re-bake 1.76 h (teaset, RTX 5080, WL27); 2 h 37 min (slab, RTX 3090, WL69) | `external/8dna26/.venv-cu128` (Main), LabServer63/69 | max_depth 64 crashes (empty batch); demo depth 10 / RR 5 used. Static medium bias on the C1 slab (WL69: ineligible). No asset for the ECRO room fixtures |
| **RNA** (arXiv 2312.09398) | Official `66b5b098` + official custom-Cycles bpy 3.5.1 | Official release; Lego qualified COMPARABLE (WL35) | Rough materials credible; poor on near-mirror nickel (WL22); position-AOV failure at transparent medium boundaries (WL69) | None natively; an edit needs a refit / retrain | Directional (sun) conditioning only; area light via a project inference adapter | Lego HQ 6.0 h; teaset rebuild 3.24 h (WL27) | WSL on Main; LabServer63 | Production renderer not released, so performance claims are blocked (WL35) |
| **TransGI** (arXiv 2506.09909, TVCG 2025) | Independent reproduction `TransGI_Reproduction` (owner's project), adopted configuration B24 L40 (D_vis, LeakyReLU, t = 0.15 probe spacing, object-balanced batches, lazy Adam, latent init 1e-4) | Independent reproduction; no official code found (search 2026-11-24) | Opaque surfaces incl. glossy / mirror (extra specular bounce); glass and participating media excluded by the paper | Rigid object motion without retraining (dynamic probes, SH rotation) | Moving lights and environment rotation | Per scene: data generation ≈ 40 min, training 7–22 min (200k iterations) | `C:\Projects\TransGI_Reproduction\.venv` + Mitsuba 3.9.1 for the fast frame (Main); LabServer69 | Not the authors' Falcor engine (Mitsuba / Dr.Jit / PyTorch / tiny-cuda-nn); paper's tone map, resolution and camera inferred; LR3 Table I not met by the reproduction (formulation floor above the paper's total); the timed fast frame omits the adopted quality options (prefiltered albedo, 256-spp mirror DI) |
| **Neural Radiance Caching (NRC)** (Müller et al. 2021) | NVIDIA RTXGI 2.0 / `NVIDIA-RTX/NRC` library (C++, D3D12 / Vulkan; marked experimental) | Official library exists | General path-traced scenes | Any (online training) | Any | Online, per frame | **Not integrated**: needs an engine / API integration this project does not have | REPRODUCTION_REQUIRED |
| **Neural Radiosity** (Hadadan et al. 2021) | `external/neural-radiosity` (official code present on Main) | Official release, not exercised beyond WL69's scope check | Surfaces only (no medium transport, WL69) | Retraining | Retraining | Per scene | Main | Not run in WL111 (no qualified workload chosen) |
| **DDGI / SHaRC** | RTXGI 2.0 (C++ engine SDK) | Official SDK | General | Any | Any | None / online | Not integrated | REPRODUCTION_REQUIRED |

### Internal comparators and ablations (ReST-GI side)

| Comparator | Role | Status |
|---|---|---|
| **Canonical ReST-GI** = ECRO v0 global transport (LOW E1 + E2 physical, learned HIGH E3–E6 patch × bin table, D = 6) + learned ER local closure | The method under test | [Architecture](Architecture.md); WL100 checkpoints, WL101 optimised runtime |
| ECRO converged (4,096 spp) | Bias floor of the canonical estimator | WL111 measured |
| ECRO stale table | Invalidation control after an edit | WL101, WL111 |
| Same-checkpoint full rebuild (F) vs selective update (S) | Within-method reuse control | WL101 (ECRO), WL56 / WL59 (earlier lines) |
| EXP / C3UR / C3URX1 (learning-free explicit transport on the same state) | Isolates what learning adds | WL87–WL91: explicit ≥ learned in quality; learned utility only source-only updates |
| CAK / exact domain-selective oracle | Earlier FNTO-line physical-oracle baselines | WL55, WL56, WL69, WL70 |
| ER-v1-SDM | Closed candidate, **not adopted** (WL104K-C) | Not a baseline here |

## Workload compatibility matrix (WL111)

Categories follow the brief: A diffuse / glossy GI, B layered materials, C sharp specular and interreflection, D textures, E media / SSS, F complex meshes, G rigid edits, H lighting / camera changes.

| Workload | Categories | ReST-GI (ECRO) | PT (same semantics / stock) | PT + OptiX | TransGI | 8DNA | RNA | NRC |
|---|---|---|---|---|---|---|---|---|
| **W1 SSS cfg0** (WL100 room, SSS sphere on floor; V0 + held-out H1, H2) | A, E, near contact | QUALIFIED (canonical checkpoint) | QUALIFIED (pt111, G_ADAPT) | QUALIFIED | UNSUPPORTED (no media) | UNQUALIFIED (no asset of this room; slab-medium bias, WL69) | UNSUPPORTED (medium AOV failure) | REPRODUCTION_REQUIRED |
| **W2 near-mirror panel** (WL100 near cfg0) | A, C | QUALIFIED | QUALIFIED | QUALIFIED | REPRODUCTION_REQUIRED (no scene conversion / training for this room) | UNQUALIFIED (no asset) | UNQUALIFIED (near-mirror weakness, no asset) | REPRODUCTION_REQUIRED |
| **W3 SSS rigid box edit** (cfg0 → cfg1 boxes, sphere fixed) | G | QUALIFIED (selective lifecycle, stale control) | QUALIFIED | QUALIFIED | UNSUPPORTED | Native reuse would apply only to separate assets; none exist | Retrain only | REPRODUCTION_REQUIRED |
| **W5 SSS light change** | H | QUALIFIED (source + generator update) | QUALIFIED | QUALIFIED | UNSUPPORTED | — | Directional only | REPRODUCTION_REQUIRED |
| **W4 Living Room 3** (Bitterli; 640 × 360; GI semantics; Ref-65) | A, B (roughplastic), C (mirror, conductor), D, F (507k triangles) | QUALIFIED_WITH_LIMITATIONS: WL102 quality (same Ref-65), historical RTX 3090 timing on an unoptimised path | QUALIFIED (Mitsuba stock path) | QUALIFIED | QUALIFIED_WITH_LIMITATIONS (independent reproduction; timed fast frame ≠ adopted quality options) | UNQUALIFIED (no per-object assets; envelope rule) | IMPLEMENTATION_UNAVAILABLE for this scene (directional only, no renderer) | REPRODUCTION_REQUIRED |
| TransGI's other five scenes | A, C, D, F, G, H | NOT_RUN (adapter qualified for LR3 only) | — | — | Reproduction results exist (owner's project) | — | — | — |
| CK2 checkerboard (WL21B) | C, D | Historical: rel. L1 0.028 / ROI 0.057 (descriptive) | Historical reference | NOT_RUN | — | Historical 0.071 / 0.226 (official, final 30 epochs) | — | — |
| Teaset T0–T3 (WL22 / 27 / 67) | B (rough conductor), G | No K / no ECRO run | Historical: 64-spp PT beats every neural arm | NOT_RUN | — | Historical; Neural Full Recompute 1.76 h | Historical; rebuild 3.24 h | — |
| C1 slab-over-floor (WL55 / 69 / 70) | E, G | Earlier FNTO line (CAK), not canonical ECRO | Cycles historical | — | UNSUPPORTED | Ineligible (static bias) | UNSUPPORTED | — |
| Lego (WL35) | A, B, D | Not canonical ECRO | — | — | — | — | QUALIFIED (COMPARABLE) | — |
| Non-rigid deformation, self-contact, sphere translation / contact formation | GM3 | **NOT_RUN** (not in the fixture family; out of scope for this batch) | — | — | Unsupported (paper) | Needs re-bake | Needs retrain | — |

Every unsupported or unqualified cell above is an absence of a valid comparison, not a measured win or loss.

## Executed baseline runs (WL111, RTX 5080 "Main")

| Run | What was executed | Authentication / reproduction check |
|---|---|---|
| PT (pt111) on W1 / W2 / W3 / W5 | spp ladder 1–1,024 (8–16 replicates), timing 1–256 spp | G_ADAPT: equals the independent pt100 code path at depth 6 (means within 3.5e-4, RMSE / noise 1.00–1.08) |
| PT + OptiX on W1 / W2 / W3 / W5 | 1–64 spp, guides from the same samples | denoiser device path == host path (0.0 difference) |
| Mitsuba `path` + OptiX on LR3 | 1–256 spp, GI by CRN | Ref-7 vs Ref-65 reproduces WL102 (0.0963) |
| TransGI (reproduction) on LR3 | WL102 model (repro check), adopted L40 checkpoint with adopted / fast-frame evaluation options, fast-frame timing at 640 × 360 and 1080p | WL102 model GI LDR RMSE 0.0768 on Main vs 0.0775 on LabServer69 |
| Canonical ECRO on W1 / W2 / W3 / W5 | 1–64 spp + converged, selective / full / stale edits, light change | WL101 runtime; selective == full; T0 → T1 → T0 restores state and table bit-exactly |
| ECRO on LR3 | not re-run (WL102 historical) | — |
| 8DNA, RNA, NRC, Neural Radiosity | not run (classes above) | — |
