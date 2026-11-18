# Worklog 104 — ECRO trained and rendered on the WL84 scene with a learned-ER LOW (2026-11-18)

## Authorization and freeze, 2026-11-18 KST

The user asked whether the WL84 images, originally rendered by path tracing with the learned ER closure, could be trained and rendered with the current ECRO, and authorized training. Before freezing, the owner made four decisions:
- **Machine:** the local RTX 5080. Inputs that exist only on LabServer63 were copied read-only.
- **LOW material:** learned ER. This is WL84's ER_L, compiler seed 8301.
- **Depth:** D = 6 is kept. New 6-event and full-depth references are rendered. The WL84 depth-4 reference is used only as context.
- **Scope:** the three WL84 images, with error maps.

This is a new workload, training run and evaluation for the existing architecture. It therefore gets the regular number **WL104** under AGENTS.md; the owner may reclassify. Work is on branch `ecro_wl84_scene`, created from `Denoise_Reconstructor` at `04a93e7`.

**What is reused unchanged:**
- ECRO v0 (`model86.py`, 735,664 parameters).
- The WL102 triangle-mesh adapter (`pt102`, `mesh102`).
- The WL100 / WL102 training recipe (`train102.main`).
- The WL84 GPU reference renderer (`render84.render_g`).
- The WL83 plugins: `cmp83` for the learned ER closure and `lay81` for the layered physical BSDF.

The learned-ER state of each image's material is rebuilt from the copied WL83 fits. The three states equal WL84's audited hashes bit for bit.

**How the WL84 scene becomes an ECRO workload:**
- **Geometry.** The geometry is the WL84 scene (render81 pose 0) with one change: the adapter is triangle based, so the analytic sphere and rectangles are converted to meshes. The rectangles become two-triangle quads. The sphere becomes a fixed 128 × 256 UV sphere with exact vertex positions and normals.
  - All new arms and references use the same mesh scene.
  - Gate G2 compares its depth-4 layered render with WL84's analytic 65,536-spp reference.
- **Patches.** The WL102 patch rule gives 991 patches.
- **Physical dispatcher.**
  - On the sphere and cube: the `cmp83` learned-ER closure. Its compiled state is also each of those patches' encoder row, which is ECRO's native material input.
  - On the floor and wall: Mitsuba diffuse 0.8 with WL100's Lambert row.
- **Camera.** Camera rays follow the WL81 formula.
- **Models.** There is one model per material. Each is trained on WL102-sized datasets of white point lights in free space. The two WL84 lights are held out and appear only in a table-level test split.

**References** (all on the mesh scene, two independent halves each):
- **Final:** layered full depth (max_depth 65).
- **For decomposition:**
  - layered 6-event;
  - learned-ER 6-event;
  - learned-ER full depth (the WL84 method without depth truncation);
  - ECRO's own tracer at D = 6 with components.
- **Gate G3** checks that tracer against the Mitsuba learned-ER 6-event reference before any training.

**Error decomposition:**
- **Representation:** ECRO vs learned-ER 6-event.
- **Material:** learned-ER vs layered at 6 events.
- **Truncation:** layered 6-event vs layered full depth.

**Error maps** share one absolute scale for the whole worklog, with square-root display.

The [protocol](../../experiments/ecro_wl84_104/protocol/wl104.json) freezes inputs and hashes, seeds, gates G0–G4, the collapse retry rule, metrics and display. No adoption or superiority verdict is planned.

### Development checks before freeze

- **Inputs.** All 51 copied inputs hash equal to the LabServer63 originals (five spot-checked on the server). The three learned-ER states match WL84's audit.
- **Camera.** Adapter camera rays match the WL81 formula to 1.2e-7.
- **Mesh form.** Mitsuba 3.5.1 treats `rectangle` as analytic, not a mesh. This is why the floor and wall became quads.
- **Patch layout.** 991 patches with a 25 cm median side. Clipping slivers reach 4.7e-11 m². WL102's run 1 collapse makes this a known risk, and the frozen collapse retry rule covers it.
- **Physical-path smoke** (256 spp): the ECRO tracer and Mitsuba depth 7 agree in object mean to 0.15%.
- **Tessellation smoke** (2,048 spp, depth 4): the mesh and analytic scenes have the same error against WL84's reference (3.82% vs 3.81% object rel L1, noise-dominated).
- **Cost.** A 64-spp field item takes about 1 s per light. A 2,048-spp layered render takes about 12.5 s.
- **Pipeline smoke.** The smoke run used reduced counts and 300 training steps and is non-evidence. It found one defect, and the light rule was corrected before freeze.
  - **Defect.** WL102 drew lights in the scene bounding box. Here that box reaches y = 3 m behind the one-sided back wall, because the floor extends that far. Lights behind the wall produce all-zero field targets, so the loss divides 0 by 0 and every model stopped at step 1 on a non-finite loss.
  - **Correction.** Lights are drawn in front of the wall, in [-3, 3] × [-3, 1.5] × [0, 3] m. Generation also stops if any light gives an all-zero target.
  - **Rerun.** The rerun completed every stage. No model collapsed. G3 passed even at smoke counts. G2 is not meaningful at smoke counts, because its noise expectation assumes 2 × 16,384 spp.
- **Display (pre-freeze).** The worklog-wide error maximum is dominated by 4-spp fireflies, so square-root display hid the converged errors. The maps therefore use a logarithmic display, sign(x)·log(1+|x|/m0)/log(1+M/m0) with m0 = M/1000. It keeps one absolute shared scale with real-unit colorbars.

### Execution amendment 1 — launch batching, before any evidence was inspected

The owner noticed slow progress and low resource use in the first evidence chain (freeze `8a91ee2`). nvidia-smi showed about 0 % GPU utilisation. The cause is Dr.Jit 0.4.6, which re-traces the Python program on every launch: with 64 samples per launch, reference rendering was CPU-bound, and successive 32,768-spp halves slowed from 99 s to 216 s. The chain was stopped during the first image's references, with 4 of 10 halves written. Nothing was inspected. The partial directory is kept as `results/ecro_wl84_104/evidence_aborted_b64/` and is not used.

Benchmark wall times by samples per launch:

| Workload | Small batch | Larger batch |
|---|---|---|
| Layered reference, 4,096 spp | 10.3 s (64) | 2.9 s (1,024) |
| Learned-ER reference, 4,096 spp | 32.4 s (64) | 3.7 s (1,024) |
| Layered reference, 8,192 spp | 3.9 s (2,048) | 3.8 s (4,096) |
| ECRO tracer, 1,024 spp | 221 s (8) | 7.0 s (256) |
| ECRO tracer, 4,096 spp | 8.2 s (512) | 43 s (1,024, VRAM full) |
| Dataset field, 64 spp | 2.65 s (16) | 0.87 s (64) |

New samples per launch:

| Workload | Before | After |
|---|---:|---:|
| References | 64 | 2,048 |
| ECRO tracer components and converged ECRO frames | 8 | 512 |
| Dataset fields | 16 | 64 |
| ECRO 64-spp frames | 8 | 64 |

Only the launch layout, and with it the per-launch RNG streams, changes. Estimators, spp, seed bases, gates, dataset sizes, the training recipe and the analysis are unchanged. The chain restarts from the beginning. [Amendment](../../experiments/ecro_wl84_104/protocol/amendment1_execution.json)
