# Worklog — ECRO Near-Mirror and SSS Dataset Training (2026-11-16)

> Append-only execution record. This is the user's separate request to train the pre-RB-DTO-E ECRO on near-mirror and SSS data and produce figures including error maps. No RB-DTO-E implementation or adoption is involved.

## Authorisation, batch class and base

The user authorises dataset generation, training and figure/error-map creation for near-mirror and SSS, together or separately. The implementation uses **two separate datasets and two fresh models**, preserving the pre-RB-DTO-E **ECRO v0 neural architecture** (`model86.py`) and the maintained **WL92A OWN objective**. This is regular **WL100**: an independent new material-coverage training/evaluation question including nonlocal volumetric material physics, not a renumbering of WL93E or the ongoing RB-DTO-E work. Starting commit: `9b8266d8b6b897f88f3af60f947e270ab3d2f6d3`, branch `Selective_Recompute`. No worktree/branch is created. Old worklogs/checkpoints/results and model/renderer source are preserved.

New code/configuration: `experiments/ecro_material100/`. Raw outputs: `results/ecro_material100/`. Reviewer figures: `evaluation/100/`. Execution is local on RTX 5080 16 GiB, using the existing `external/8dna26/.venv-cu128/Scripts/python.exe` (Mitsuba 3.5.1, Dr.Jit 0.4.6, PyTorch 2.8.0+cu128). No project code is uploaded to a server.

## Architecture and learned ownership

- Unchanged learned layers and parameter count **735,664**, 3 message rounds, local channel rank 16, global rank 24, 1,270 patches, 64 direction bins (4×16), 8 state rays/bin, 16 source samples/patch, nonnegative single/global response factors, source-independent linear application, original nearest patch/bin readout.
- The material **buffer** is replaced/extended for the new fixed material vocabulary; no learned layer is enlarged. The same renderer material-state table is supplied to the model. The SSS row encodes fixed RGB albedo and mean free path; it is explicitly a typed fixed-medium feature row, **not** an ER surface-BSDF interpretation. No general medium/material compiler is claimed.
- The source is current boundary direct irradiance. Training targets are physical outgoing boundary fields of 2–5 optical-domain interactions, split into exactly 2 and ≥3. The original OWN objective is whole + single + global squared relative losses, weights 1/1/1 and the original common denominator.
- Camera rendering computes physical interactions 1–2 and adds ECRO's learned higher-order lookup (≥3 camera interactions). **This does not make ECRO a learned local SSS solver:** local material physics and the first two interactions stay explicit. The dataset tests ECRO's existing global transport response in a scene containing real SSS.

## Physical material adapters

**Near-mirror.** CK2-derived checker scene, panel a single isotropic GGX with direct alpha=.01, F0=1 and neutral coefficient .85. Box A has .3/.8 Lambert 3×3 patch checker. Original other ER/Lambert materials remain. Box positions/yaw and lights vary under fixed configuration seeds. The entire material/texture is in the **new training dataset**, unlike WL93E's frozen-checkpoint OOD diagnostic. Existing `pt86.py` and the compiled ER/GGX source are imported unchanged.

**SSS.** The sphere of radius .4 m at (.9,.4,.4) is an actual **homogeneous, index-matched RTE medium**, sigma_t=12/m, scattering albedo RGB (.92,.80,.65), isotropic phase. Other materials are Lambert. The renderer uses an adjoint entry→first-exit random walk, including all internal collisions and ballistic exit, with point-light NEE at each volume collision. A full internal excursion is **one optical-domain interaction**, not one physical volume collision. Air/free-space remains explicit. Point-light connections through the sphere include Beer attenuation in a separately counted additional domain interaction, not an opaque-shadow approximation. The maximum 128 internal collisions leaves residual weight bounded by .92^128 <2.4e-5. No Burley BSSRDF, diffuse look-alike, radiance bake or image-space SSS shortcut is used.

The finite depth is six optical-domain interactions for both datasets, matching the old ECRO depth at the domain level. SSS internal collision counts must not be compared to historical surface-event counts without this label. The current adapters do not implement deformation, a moving medium, multiple media, refractive boundaries or heterogeneous volume. No such capability is inferred.

## Development and preflight before the freeze

Throwaway physical-only development uses train index 99999 (not in any data split), 128×128/32 spp, inspected directly before model training. The near-mirror checker is visible in the panel; the back/side-lit SSS sphere has a coloured transmitted/scattered response. No model error is used for scene selection.

Development-only fixes: Dr.Jit loop states initially changed from Point3f/Color3f to generic array types; explicit type preservation fixed this before data generation. The isolated-sphere validation's empty shape pointers were replaced with its known single-shape ID. No primary data/checkpoint existed at those fixes.

Preflight (`preflight/preflight.json`) passes 8 checks:

1. SSS first-domain source against **independent native Mitsuba homogeneous volpath**, same isolated sphere, point light and camera: custom/native RGB total **.97685 / .98789 / .99330**, within the prospectively specified 5% energy tolerance. Native reference half-noise relative RMS **1.81%**. This is a bounded low-density qualification, not an exact or dense-medium reference claim; the up-to-2.3% difference is disclosed rather than calibrated away.
2. SSS finite/nonnegative.
3. Native comparison noise below its 5% preflight bound.
4. Actual renderer ballistic shadow connection through a sphere diameter versus independent Lambert×Beer: relative error **1.22e-5**; direct unattenuated connection is zero, so no double counting.
5. Unchanged parameter count 735,664.
6. Near-mirror panel visible.
7. Model/renderer material-feature tables identical.
8. Zero/superposition: zero exact; relative linearity discrepancy **6.84e-8**.

Sampling/resource probes are throwaway (index 99997). Batched field generation costs around 1.1 s near and 4.4 s SSS for 64 spp in a single batch; large batches amortise Python/graph overhead. Dataset field batch is 64 spp and image batch 32 spp, fixed before the primary generation. These are setup measurements, not uncontended performance or contribution claims.

## Frozen data/training/evaluation protocol

The machine protocol is `experiments/ecro_material100/protocol.json`; exact configurations and source hashes are saved in `configs.json` / `freeze_manifest.json` **before primary generation**.

- **Per dataset:** train 128 independent geometries ×4 point-light configurations (**512** response fields), validation 16×4 (**64**), test 8×2 (**16**). Separate configuration seeds 100100/100200/100300. Geometry/light configuration hashes are disjoint. These are bounded newly generated material datasets, not the original 1,024-geometry WL86 population.
- Train fields: 64 spp. Validation: 2×128 spp, test: 2×256 spp; independent target halves retained. Float32 target storage avoids half-precision overflow on rare sharp paths. All targets are generated independently for the new materials; old S1 target fields are not reused.
- Separate fresh fixed fits, seed 10001 each; 24,000 steps; Adam 1e-3→1e-5, 500-step warmup, gradient clipping 10; final-step checkpoint, no held-out selection or tuning. One seed/dataset: no seed-robustness claim.
- Fixed figures: test configurations 0/1/2, lighting 0; 512×512, physical reference 2×4096 spp (seeds 100901/100902 +10×index), independent predicted readout 4096 spp (100903 +10×index), no denoise/clamp. Shared exposure and absolute error-map scales per dataset. Reference, physical LOW, ECRO full and higher-order components are retained independently.
- Metrics: full-frame/material-ROI HDR PSNR, relative RMSE/L1, reference-half noise and predicted higher-order fraction/error. Difference maps are not interpreted as model error below the measured reference/readout noise. Higher-order rare-path convergence is reported explicitly, especially near-mirror point-light caustics.
- No image-fitting loss, target use at inference, geometry-specific checkpoint, test threshold changes, hyperparameter sweep, architecture expansion, RB-DTO-E transport, production adoption or automatic successor.

## Execution (append below)

Primary data generation/training/rendering runs as a hidden detached local process with logs and stage completion markers. Near training may overlap independent SSS target generation. Wall times include any contention and are not presented as an economic advantage. Final counts, training completion, actual figures, quantitative quality, limitations and content hashes are appended after execution.

### Data generation completed; target-noise disclosure

All primary datasets completed with the frozen counts. Near generation: train 72.6 s, validation 27.2 s, test 14.0 s. SSS generation (overlapping near training): train 109.8 s, validation 58.4 s, test 36.5 s. Each case has 512/64/16 train/validation/test lighting fields over 128/16/8 geometries. The first generated file has shape `[4,1270,64,3]`; nonzero finite targets and atomic completion files were verified. The fixed training runs started successfully; no data was regenerated using model/test outcomes.

Independent target-half noise (`dataset_noise.json`), estimated for the average of the two halves:

| Case / split | Relative L1 half-noise | Relative RMS half-noise |
|---|---|---|
| Near validation | .1011 | .9657 |
| Near test | .08082 | .9263 |
| SSS validation | .05799 | .06781 |
| SSS test | .04574 | .05252 |

The sharp point-light boundary fields have **extreme rare-path/heavy-tail RMS noise**, even though their L1 noise is about 8–10%. Training is a fixed-budget material baseline, not a prospective proof of converged near-mirror field fidelity. No training-budget/protocol change or rescue is made after this measurement. Final high-spp image noise is measured separately; neither a large target-field RMS error nor a flat training curve alone can be attributed to ECRO representation. SSS target noise is much lower. These limitations must accompany the figures.

Reference configuration semantics are also checked against the [official Mitsuba homogeneous-medium documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html): single-scattering albedo, extinction per scene unit, default isotropic phase. The actual independent check uses the installed 3.5.1 native volpath runtime; the documentation is not a substitute for that measured check.

### Both fixed training runs completed

Both models completed **24,000 steps** with finite loss/gradients and successful process exit. No restart, tuning, best-test checkpoint selection or architecture change occurred. The final step is the delivered checkpoint.

| Dataset | Train wall | Parameters | Validation whole relative L1 | Test whole / single / global relative L1 |
|---|---|---|---|---|
| Near-mirror | **577.46 s (9.62 min)** | 735,664 | .10119 | **.08195 / .09448 / .10685** |
| SSS | **534.32 s (8.91 min)** | 735,664 | .05766 | **.04723 / .06099 / .06081** |

These are boundary-field errors against the averaged finite-sample targets, **not rendered-image errors** or a noise-debiased fidelity verdict. Near test L1 target noise is .08082 and its heavy-tail RMS noise is .9263; SSS test L1 target noise is .04574. In particular the near whole-field error being close to L1 target noise is not proof of perfect learning or a representation ceiling. Independent high-spp held-out image evaluation follows.

Near training overlapped SSS dataset generation and SSS training; timings are recorded contended walls, not a quality-matched GPU speed comparison with WL92A. Checkpoints and complete learning curves are `near/model.pt`, `sss/model.pt` and each case's `training.json` under the raw root. Their source hashes certify the unchanged original model implementation.

## Held-out rendering and figures completed

The main chain ended successfully at **2026-11-16 11:51:52 UTC / 20:51:52 KST** (about 20 minutes from the start of generation, excluding development and final plotting). All training, generation and rendering subprocesses returned zero. Six held-out scene/light cases produced **12 independent reference halves + 6 prediction images**, with native reference/readout arrays, masks, tables and per-frame JSONs retained. Near reference/prediction stages took 69.46/13.59 s; SSS reference/prediction stages 231.08/24.20 s. These are observed execution walls, not a quality-matched performance claim.

Figures use a **single exposure per dataset**, shared across all its reference, prediction, full, higher-order and ROI views. Error maps show mean absolute RGB error in linear HDR, with **one common absolute range per dataset** across all its tests/components. The display maximum is the pooled 99.5th percentile; outliers are clipped for display only. Metrics use every included pixel without clipping. The reference is the mean of two 4096-spp halves; the 4096-spp prediction uses its own independent readout seed and computes physical LOW at inference (it does not borrow reference LOW).

### Quantitative measurement

Material ROI is the primary panel for near and sphere for SSS. Relative RMSE is RGB RMS error divided by RMS reference. The reference noise column is independent half difference /2, an SEM estimate for the mean reference, not a confidence interval.

| Dataset / test | Full-frame rel RMSE | Material-ROI rel RMSE | Reference ROI noise | h ROI rel RMSE | h ROI noise | Physical h share in ROI |
|---|---|---|---|---|---|---|
| Near 0 | 51.09% | **8.33%** | **8.16%** | 22.10% | 21.67% | 43.71% |
| Near 1 | 36.45% | **15.98%** | **15.87%** | 35.45% | 35.20% | 48.28% |
| Near 2 | 51.72% | **20.10%** | **20.02%** | 46.94% | 46.76% | 42.98% |
| SSS 0 | .489% | **1.856%** | 1.021% | 3.646% | 1.903% | 43.96% |
| SSS 1 | .435% | **2.080%** | 1.156% | 3.894% | 2.060% | 46.34% |
| SSS 2 | .652% | **1.472%** | .814% | 3.129% | 1.633% | 38.95% |

**Near reference limitation:** full-frame raw RMS is strongly driven by rare bright point-light paths, and the panel's reference noise almost equals its raw prediction discrepancy. This batch does **not** qualify complete converged near-mirror GI or isolate that discrepancy as neural error. The same-scale noise-control images expose the rare pixels. Checker location/contrast is visible in both actual rendered images; the remaining structured differences and heavy tails must be distinguished rather than hidden by denoising. No sample was removed, no clamping/denoising was added, and no model/scene was selected after observing error.

**SSS scope:** the model matches this finite-depth custom physical reference at the listed ROI error. The isolated-sphere independent native-volpath check itself differs by up to 2.3% in total RGB energy under a 5% qualification bound. Therefore the 1.47–2.08% figure errors must **not** be promoted to independently established absolute SSS accuracy of that precision. Prediction readout and reference MC noise also remain. The learned h contribution is substantial in these figures, but local SSS physics and LOW are still physical, not learned by ECRO.

### Post-hoc diagnostics (not primary gates)

After observing heavy-tailed reference noise, an additional descriptive cross-half estimate was recorded: `mean((prediction−refA)*(prediction−refB)) / mean(refA*refB)`, square-rooted only when positive. Independent reference variance cancels in expectation; prediction readout noise remains. The resulting material-ROI estimates are near **1.67/1.94/1.85%**, SSS **1.55/1.73/1.23%**. These are labelled **post-hoc**, have no confidence interval, and do not establish exact fidelity or a supported architecture. The primary raw errors/noise above remain unchanged.

A separate **post-hoc physical-only control**, fixed test-0 geometry/light/camera, replaces the SSS sphere by opaque Lambert using the same RGB parameter (.92,.80,.65). This is not matched emergent reflectance/energy and is **not a model comparison or training input**. On 46,541 camera-facing pixels in the opaque source shadow, opaque direct source radiance is zero while the actual volume's mean direct source response is **.12012 linear HDR**. The visual explicitly shows light entering elsewhere and reaching the shadowed front through internal scattering. The control changes no checkpoint, target, primary metric, seed selection or protocol condition. Its source/config hash and arrays are retained.

## Visual observation and interpretation

The actual generated overview, full/higher-order/ROI sheets, same-scale reference-noise controls, fixed learning curves and SSS/opaque control were inspected directly. At alpha .01 the reflected checker appears in both near reference and prediction. Error maps reveal structured cell-level differences plus many reference bright outliers; they do not justify saying the near-mirror error is entirely representation error. For SSS the prediction retains the coloured sphere response; remaining differences include contact/occlusion regions and low-amplitude spatial bias. The physical shadow control verifies that this is actual nonlocal volume response rather than merely a brown diffuse surface.

This completes the user's requested **training and figure generation**, within the declared bounded material families. It is **not** a whole-neural local-SSS method: explicit material evaluation and first-two-interaction tracing are part of the original ECRO contract. It is also not a broad near-mirror/SSS generalisation result, a dynamic reuse result, a scaling/economic comparison, a headroom claim or a replacement of RB-DTO-E. One seed/material, narrow scene distributions, low-density SSS validation, finite outer depth and the near reference noise limit remain open.

## Architecture verdict

**NOT APPLICABLE / no adoption decision.** Both requested fixed ECRO models were actually trained and evaluated, but no prospective architecture-success rule was specified or inferred. Historical ECRO/WL93E/RB-DTO-E outcomes and canonical Architecture.md remain unchanged. No successor, extra training rescue or production integration is started.

## Reproducibility and delivered artifacts

- New source/protocol: `experiments/ecro_material100/`. Original `model86.py`, `pt86.py`, `scene86.py`, `own92a.py` and `closure83_dr.py` match the pre-generation SHA-256 values; the final manifest checks them.
- Raw response datasets, fixed configurations, freeze/preflight, target-half noise, complete training logs and checkpoint files: `results/ecro_material100/`. Case checkpoints: `near/model.pt`, `sss/model.pt` (about 2.96 MB each). The full physical/model render arrays and metadata are in each case's `renders/`; tables are inferred from weights, current geometry and current source only.
- Re-run into a **fresh** output root: `run100.py freeze ROOT`, then the hidden/detached `pipeline100.py ROOT`. The same command line with `train100.py ROOT near|sss` runs the fixed fits on already generated data. `viz100.py ROOT evaluation/100` uses the local analysis venv from WL93E. `control100.py ROOT` reproduces the explicitly post-hoc physical SSS control. No server upload or new environment installation is required.
- Per-render reference seeds/settings and source/config hashes are retained; per-model training.json includes actual steps, parameter count, tests, timings, PyTorch/GPU, peak **PyTorch allocator** memory (~661 MiB, not total process/GPU memory) and checkpoint SHA-256. CUDA atomics may prevent bitwise rerender/retraining identity; deterministic configuration and seed provenance are not a bitwise reproducibility claim.

Reviewer entry: [evaluation/100/index.html](../../evaluation/100/index.html).

| Artifact | Contents |
|---|---|
| [near_overview.png](../../evaluation/100/near_overview.png) | Three fixed near held-outs, reference / trained ECRO / absolute Error Map, 2550×2550 |
| [sss_overview.png](../../evaluation/100/sss_overview.png) | Three fixed SSS held-outs, reference / trained ECRO / absolute Error Map, 2550×2550 |
| `near_test{0,1,2}_comparison.png`, `sss_test{0,1,2}_comparison.png` | Full and material ROI; reference / prediction / error plus physical h / learned hl / h error, 4080×1700 |
| [near_noise_control.png](../../evaluation/100/near_noise_control.png), [sss_noise_control.png](../../evaluation/100/sss_noise_control.png) | Error versus independent reference-half noise on identical absolute scales |
| [sss_physical_nonlocal_control.png](../../evaluation/100/sss_physical_nonlocal_control.png) | Physical-only opaque/SSS direct-source comparison and positive difference |
| [training_curves.png](../../evaluation/100/training_curves.png) | All fixed-step OWN component objectives; no checkpoint-selection curve |
| `*_test*_reference_512.png`, `*_prediction_512.png`, `*_error_512.png` | Standalone native image assets |
| [metrics100.json](../../evaluation/100/metrics100.json), [manifest100.json](../../evaluation/100/manifest100.json) | Full quantitative values, noise, training, configuration/source/checkpoint/data/artifact SHA-256 and execution provenance |

Final delivery verification: 30 PNG files passed image-integrity checks; all gallery links and 32 reviewer-artifact hashes resolved; original source hashes, both finite 735,664-parameter/24,000-step checkpoint hashes/material buffers, and the raw input-material snapshot were verified. No requested artifact is pending. No historical model, reference, Worklog or Architecture.md was modified.
