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

## Evidence, 2026-11-18 KST

### Implementation fact

The chain at `9e1c05c` (freeze `8a91ee2` + amendment 1) ran 19:40–20:12 KST on the local RTX 5080.

| Stage | Time |
|---|---|
| Pre-gates | 2 s |
| References, per image | about 3.3 min |
| Datasets, per material | 59 s |
| Training, per model (24,000 steps) | about 6.1 min, GPU about 85 % |
| Evaluation, per image | 3–5 s |
| Analysis | 6 s |

All three models trained with the first frozen seed, 10401. None collapsed; final validation whole / single / global relative L1 by material:

| Material | Whole | Single | Global |
|---|---:|---:|---:|
| hard_1 | 0.193 | 0.211 | 0.302 |
| iid_0 | 0.179 | 0.187 | 0.308 |
| iid_1 | 0.192 | 0.203 | 0.323 |

No architecture, recipe or patch-rule change was made.

| Gate | Result |
|---|---|
| G0 | 51 input hashes, 3 learned-ER state hashes, both scene PLYs: pass |
| G1 | camera 1.2e-7: pass |
| G4 | N = 991: pass |
| G2 tessellation (mesh vs WL84 analytic, depth 4) | object rel L1 0.0116 / 0.0069 / 0.0069 vs limits 0.0171 / 0.0103 / 0.0102; mean ratio 0.9999–1.0005: pass |
| G3 physical path (ECRO tracer D = 6 vs Mitsuba depth 7, learned ER) | object rel L1 0.0089 / 0.0079 / 0.0072 vs limits 0.0131 / 0.0123 / 0.0112; object and full mean ratios 1.0000 ± 0.0001: pass |

### Measurement

All values are relative L1 against the layered full-depth reference (2 × 32,768 spp). Multi-seed arms give the mean over seeds; the seed SD is in the metrics. The ROIs are:
- objects: sphere and cube, 9,887 px;
- highlight: object pixels at ≥ 3 × the object median, 3,636 / 3,199 / 376 px;
- grazing: 998 px.

| Image | Arm | Full | Objects | Highlight | Grazing | Object mean ratio |
|---|---|---:|---:|---:|---:|---:|
| iid_1_L2 | ECRO 4 spp (20 seeds) | 0.262 | 0.286 | 0.281 | 0.382 | 0.970 |
| | ECRO 64 spp (8) | 0.094 | 0.086 | 0.082 | 0.095 | 0.967 |
| | ECRO 2 × 1,024 spp | 0.019 | 0.037 | 0.035 | 0.026 | 0.968 |
| | Learned-ER PT full depth, 64 spp (8) | 0.107 | 0.102 | 0.093 | 0.194 | 0.966 |
| | Learned-ER PT full depth, 2 × 32,768 | 0.008 | 0.035 | 0.035 | 0.044 | 0.966 |
| iid_0_L2 | ECRO 4 spp | 0.261 | 0.262 | 0.237 | 0.594 | 1.008 |
| | ECRO 64 spp | 0.091 | 0.085 | 0.078 | 0.162 | 1.008 |
| | ECRO 2 × 1,024 spp | 0.018 | 0.031 | 0.029 | 0.043 | 1.008 |
| | Learned-ER PT 64 spp | 0.103 | 0.092 | 0.077 | 0.265 | 1.009 |
| | Learned-ER PT converged | 0.008 | 0.027 | 0.025 | 0.049 | 1.009 |
| hard_1_L1 | ECRO 4 spp | 0.144 | 0.207 | 0.060 | 0.355 | 0.989 |
| | ECRO 64 spp | 0.043 | 0.068 | 0.028 | 0.094 | 0.988 |
| | ECRO 2 × 1,024 spp | 0.011 | 0.041 | 0.025 | 0.035 | 0.988 |
| | Learned-ER PT 64 spp | 0.048 | 0.089 | 0.029 | 0.118 | 1.002 |
| | Learned-ER PT converged | 0.005 | 0.030 | 0.025 | 0.027 | 1.002 |

**Decomposition** (converged images, object rel L1; full-frame values in the metrics):

| Term | iid_1 | iid_0 | hard_1 |
|---|---:|---:|---:|
| Total: ECRO vs layered full depth | 0.037 | 0.031 | 0.041 |
| Learned-ER material: ER 6-event vs layered 6-event | 0.035 | 0.027 | 0.030 |
| Representation: ECRO vs learned-ER 6-event | 0.018 | 0.015 | 0.023 |
| Truncation: layered 6-event vs full depth | 0.010 | 0.006 | 0.007 |
| LOW check: ECRO E1 + E2 vs tracer E1 + E2 | 0.015 | 0.014 | 0.010 |

- **Truncation** is at the reference noise level: the half-difference of the layered references is 0.014 / 0.008 / 0.008.
- **The LOW check** is noise-level, since it is the same estimator.
- **Representation noise floor.** The "converged" ECRO mean still carries noise: its half-difference is 0.026 / 0.026 / 0.017, so the noise of the mean is about 0.013 / 0.013 / 0.009. In quadrature, the representation error above that noise is about 0.012 / 0.007 / 0.022. This is an approximate inference.
- **HIGH-only comparison is unresolved.** The learned hl vs the tracer's physical E3–E6 gives 0.127 / 0.125 / 0.107. It is dominated by the 2 × 4,096-spp tracer's own HIGH noise (half-difference 0.112 / 0.131 / 0.058), so it does not resolve the HIGH error.
- **Learned-ER material error at full depth** is 0.035 / 0.027 / 0.030, the same as at 6 events.

**WL84 context** (depth 4, analytic scene, against WL84's own reference):
- The stored WL83 ER_L 64-spp render gives object 0.0995 / 0.0884 / 0.0849 and highlight 0.093 / 0.076 / 0.033.
- The 8 WL84 replicates average the same values.
- The iid_1_L2 highlight value 0.093 reproduces WL84's.

**Table level** (the two held-out WL84 lights, 2 × 256-spp field targets): whole relative L1 0.143 / 0.133 / 0.143.

**Timing** (descriptive, desktop GPU):
- ECRO table: 4.5 ms (source 2.2, factors 1.9, apply 0.4).
- ECRO frame wall: 22 ms at 4 spp, 24 ms at 64 spp.
- Learned-ER full-depth path tracing at 64 spp: 0.67 s wall but only 5–6 ms GPU. Both walls are dominated by Dr.Jit 0.4.6 Python tracing, so this is not a fair cost comparison.

### Observation

Sources: [error maps](../../evaluation/104/error_maps_test83_hard_1_L1.png) (one per image) · [decomposition](../../evaluation/104/decomposition_maps.png) · [bars](../../evaluation/104/error_bars.png) · [training](../../evaluation/104/training_validation.png) · [gallery](../../evaluation/104/index.html)

- **Representation error (ECRO minus learned-ER 6-event), hard_1_L1:**
  - The lower hemisphere of the sphere is darker.
  - The cube's side faces show a blocky, patch-sized signed pattern.
  - The floor directly under and beside the sphere is brighter, a contact-region excess like the one WL101C reported for the SSS fixture.
- **Representation error, iid_1 and iid_0:** the same patterns are weaker.
- **Material error (learned-ER vs layered):** concentrated in the specular highlight rings on the sphere and in a tone shift on the cube faces.
- **Truncation:** visually noise.
- **At 64 spp:** ECRO frames show less grain on the floor and in the shadows than learned-ER path tracing at 64 spp, because deep paths come from the table rather than from sampling.

### Interpretation

- **The current ECRO trains and renders on the WL84 scene without architectural change.** All three models converged with the frozen seed, and the adapter paths matched their physical references.
- **Converged error is led by the learned-ER material, not by ECRO.** Against the layered full-depth reference the converged ECRO error is 3.1–4.1% on the objects. Most of it is the learned-ER material's own error (2.7–3.5%), which ECRO inherits through its LOW.
- **ECRO's own representation error is smaller and image-dependent.** After removing ECRO's residual noise it is roughly 0.7–2.2%, largest on the hard material. There ECRO is also 1.4% darker on the objects than learned-ER path tracing.
- **Depth truncation at 6 events is negligible** in this open scene.
- **Operating point.** At an equal 64 spp, ECRO has a lower object error than full-depth learned-ER path tracing (0.068–0.086 vs 0.089–0.102). The difference is largest at grazing pixels.
  - This is a variance effect of replacing sampled deep paths with the table.
  - It is not a converged-fidelity advantage: converged path tracing is 0.2–1.1 points better on the objects.
  - Any cost comparison is unqualified here.

### Unresolved question

- **HIGH-only error not resolved.** A higher-spp physical HIGH component would be needed.
- **Representation not attributed.** The blocky cube pattern and the contact excess are not separated into generator error and patch/bin discretization.
- **Converged ECRO is still noisy.** The 2 × 1,024-spp ECRO mean retains noise of the order of its smaller representation errors.
- **Scope.**
  - One light per image, three materials, one model per material.
  - No geometry or light edit.
  - No generalization across materials: each model saw only its own material.
- **Costs.** Fair costs need a frozen or traced-once renderer on both sides.

### Verdict

Measurement only, as frozen: there is no adoption or superiority claim. Comparisons with path tracing are path-traced re-rendering with the same learned-ER material.

### Reproducibility

- **Commits:**
  - freeze `8a91ee2`;
  - amendment 1 `9e1c05c`;
  - the report commit adds this section, the reviewer artefacts and the roadmap line.
- **Command:** `<venv-cu128 python> chain104.py results/ecro_wl84_104/evidence evaluation/104`, from `experiments/ecro_wl84_104`. It is resumable; completion is `chain.json` with `complete: true`.
- **Raw data** (gitignored), under `results/ecro_wl84_104/`:
  - `evidence/`: chain status, logs, gates, references, datasets, models and evaluations;
  - `inputs63/`: LabServer63 copies, hashed in the protocol;
  - `scene/`: the meshes;
  - `smoke/` and `evidence_aborted_b64/`: non-evidence.
- **Checkpoints:** SHA-256 values are in `evaluation/104/metrics.json` under `training`.
- **Display:** one error scale M = 6.26 linear-HDR luminance (the maximum over all maps, from 4-spp fireflies), logarithmic display with floor M/1000, real-unit colorbars.
