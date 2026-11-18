# Worklog 101A — ECRO SSS low-SPP artifact attribution

## Authorization and freeze, 2026-11-18

The user authorized a controlled attribution follow-up to WL100 and WL101 on WL101 SSS cfg0/test-0. This uses the available suffix **WL101A** because the scientific question directly follows the observed artifact under the existing architecture. An isolated branch starts at completed WL101 so the active WL102 checkout and ongoing long-path experiments can continue independently.

The protocol fixes the scene, camera, checkpoint, reference identities, renderer stack, seed schedule, regions, error estimators, display scales and dominance rule before evidence execution. The sole optimized renderer is the original WL101 arm B on LabServer69. Its original published arm A and B arrays, gbuffer and two six-event reference halves are read-only inputs. No new physical reference renders are planned. Hashes distinguish the original server's CRLF source bytes from the equivalent LF git content.

The learned table and image-space HIGH are different objects. The table is conditioned on fixed geometry and source. The readout samples camera and first-event paths, which can change both lookup addresses and weights between seeds. The study measures both table identity and image-space HIGH variance. Matching the physical reference seeds at 4096 spp supports an event-by-event LOW control and an independent-pair estimate of stable HIGH error against the custom physical reference. WL100's native-volpath qualification remains a separate bound on physical interpretation.

Protocol and tools are under [experiments/ecro_sss101a](../../experiments/ecro_sss101a/). Raw artifacts go to `results/ecro_sss101a/` and reviewer artifacts to [evaluation/101A](../../evaluation/101A/).

### Pre-execution checksum erratum

The local input check caught two omitted characters in the initial frozen string for the archived arm-B 4096-spp prediction. The corrected string matches the original server file and the independently copied local bytes. The file itself, the reference halves, the sampling protocol and the decision criteria were already fixed. No WL101A renders had been launched.

### Archived input verification

The five copied WL101 cfg0 arrays match their server SHA-256 values. The published arm A and arm B 4096-spp images use seed 101500. Their full-image relative L1 discrepancy is 2.75e-7. The corresponding physical LOW discrepancy is 1.76e-7 and the learned HIGH discrepancy is 7.77e-7. These are historical estimator-equivalence checks on the exact scene. The new run adds the fresh replay and paired physical LOW controls.

The fixed center-ray masks contain 262,144 full-frame pixels, 49,660 sphere pixels, 45,692 interior pixels, 3,968 inner-boundary pixels, 72,657 floor pixels and 38,071 floor contact/shadow pixels. The partition and subset checks passed. Synthetic known-input tests validated component variance and covariance, finite-replicate bias correction, the inverse-SPP slope and the HDR display transfer.

Against the WL101 average reference, the archived independent arm-B prediction's sphere-relative RMSE is 1.858% full, 1.487% LOW and 3.651% HIGH. The corresponding reference-mean noise estimates are 1.031%, 0.868% and 1.910%. The unpaired LOW discrepancy includes independent prediction and reference sampling. These archived values alone cannot resolve systematic LOW or quantify the 32-spp artifact.

### Execution status

Automatic approval review rejected deployment of the experiment scripts and launch on LabServer69 because it required explicit authorization for the code transfer and persistent remote artifacts. The rejected launch did not send code or start GPU work. The original WL101 inputs were read and copied into the isolated local worktree. The local analysis also passed an end-to-end synthetic check for the full seed ladder, verdict logic and six figure outputs.

The new render sweep, matched 4096-spp replays, latency measurements and artifact verdict are pending approval. The gate on the archived data passed. There is no WL101A sampling fraction or systematic component attribution from the archived data alone.

### Execution amendment 1 — LabServer63

The user selected LabServer63 for execution. It has an RTX 3080 Ti 12 GiB and driver 595.71.05. The server has the same Python 3.10.21, Torch 2.3.1+cu121 and NumPy 1.26.4 environment. The original WL101 source, checkpoint, configuration, archived reference arrays and exact Mitsuba 3.8.0 / Dr.Jit 1.3.1 distribution are staged in a separate run directory. This change supersedes the LabServer69 launch status above.

The cross-machine admission gate precedes the seed ladder. It checks fresh arm-B replay against the saved arm-B image at the original seed, plus fresh physical LOW against each saved reference half with its paired seed. Gates apply to each ROI. This checks the exact custom ray tracer and accumulation that the study uses. WL53's historic LabServer63 native CUDA film defect concerned a different renderer path and motivates this explicit control. Timings are reported only for the chosen GPU.

#### Startup correction before evidence

The first LabServer63 launch exited 139 during scene creation. Its output directory is empty. The system OptiX library resolves to 570.195.03 while the running driver is 595.71.05. The existing matching user library in `~/nvlibs/optix595` passed analytic sphere intersection at 1.6 m with Mitsuba 3.8.0 and Dr.Jit 1.3.1. The revised launch sets the library path for its own process only and records the resolved library and SHA-256. The original logs and empty attempt directory are retained. The compatibility gate still precedes the complete sweep.

## Completed evidence — 2026-11-18 KST

### Implementation fact

The original WL101 arm B traced a resident-table 512² frame for 16 independent seeds at each of 1, 4, 16, 32, 128 and 512 spp. The same renderer replayed seed 101500 and the two archived reference seeds at 4096 spp per run. The seed-matched physical LOW uses the same sampled rays, first two optical-domain interactions and launch layout as the archived six-event reference. The renderer dumps its own E1, E2 and learned HIGH contributions. Their reconstruction agrees with the native full image to 1.63e-8 relative L1 across recorded frames.

Geometry, illumination, camera and checkpoint passed identity checks. A second source and response evaluation gave the same bits. The response table was bit-identical to archived arm B and stayed bit-identical through the run. The seed-101500 re-render reproduced every archived arm-B component exactly, and the repeated 32-spp seed produced the same component images. Against the corresponding physical reference half, the paired LOW relative L1 was at most 1.21e-6 over both halves and every ROI. The largest individual event deviation was 3.38e-6 (contact/shadow E1). Both process exits were 0. The two physical references were reused.

### Measurement

The table reports the mean single-render error over 16 seeds at each SPP. All errors use linear HDR RGB. Relative RMSE is divided by the corresponding mean physical reference RMS. Timing is the median synchronized static-frame wall with the table resident and image readback included. No foreign GPU process was detected at the sampled timing boundaries.

| SPP | Full HDR RMSE | Full relative L1 | Sphere relative RMSE | Floor relative RMSE | Frame wall median (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.23990 | 23.056% | 77.119% | 53.114% | 13.46 |
| 4 | 0.61999 | 11.820% | 38.619% | 26.495% | 16.90 |
| 16 | 0.31086 | 6.005% | 19.399% | 13.287% | 47.43 |
| 32 | 0.21995 | 4.264% | 13.735% | 9.397% | 87.39 |
| 128 | 0.10958 | 2.136% | 7.022% | 4.829% | 335.41 |
| 512 | 0.05505 | 1.077% | 3.720% | 2.400% | 1,327.71 |

The 32-spp budget sums full seed variance, estimated stable squared bias and reference-mean noise. Full seed variance is LOW variance plus image-space HIGH variance plus twice their covariance. The shares below keep the covariance explicit. Intervals use the frozen seed-pair and spatial-block bootstrap, conditional on this scene and the two reference halves.

| ROI | 32-spp full relative RMSE | LOW share [95% interval] | HIGH share | Covariance share | All seed variance | Stable bias share | Reference noise share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 4.050% | 50.20% [50.13%, 50.26%] | 13.67% | 35.19% | 99.05% | 0.44% | 0.50% |
| Sphere | 13.735% | 49.90% [49.65%, 50.13%] | 16.88% | 32.08% | 98.85% | 0.59% | 0.56% |
| Sphere interior | 15.346% | 49.57% [49.42%, 49.69%] | 17.08% | 32.61% | 99.26% | 0.16% | 0.57% |
| Sphere boundary | 9.869% | 51.47% [50.42%, 52.39%] | 15.89% | 29.56% | 96.93% | 2.57% | 0.50% |
| Floor | 9.397% | 60.59% [60.38%, 60.77%] | 10.21% | 28.06% | 98.85% | 0.66% | 0.49% |
| Contact/shadow | 16.992% | 49.30% [48.82%, 49.69%] | 14.50% | 33.99% | 97.80% | 1.65% | 0.55% |

Using directly observed average single-render MSE as the denominator gives LOW shares of 50.10% on sphere, 60.52% on floor and 49.23% in contact/shadow. Splitting covariance evenly gives 65.94%, 74.62% and 66.30%, respectively. That split is a sensitivity calculation. The separate covariance term is the primary result.

LOW variance decreased at each SPP step. Log-variance slopes are −0.997 [−0.999, −0.995] on sphere, −1.012 [−1.013, −1.011] on floor and −1.011 [−1.013, −1.008] in contact/shadow. The rates are close to inverse SPP. The exact −1 slope is outside those bootstrap intervals. The measured HIGH readout also decreases with SPP. Its sphere slope is −1.000 to three decimals [−1.001, −0.999]. Metrics for each component, ROI and SPP, including ensemble-mean bias, reference noise, signed RGB, per-seed errors and time IQR, are in the reviewer data. [Metrics](../../evaluation/101A/metrics.json)

### Observation

The common-exposure progression shows grain decreasing on the sphere and floor as SPP rises. The 32-spp separated images show seed variation in both LOW and HIGH. With matched high-SPP samples, the signed full and HIGH residual maps have the same structure at the sphere contact, the lower silhouette and scene seams. The contact area is overbright on average. The LOW residual map is visually uniform at the same signed scale. [Progression](../../evaluation/101A/progression.png) · [Component maps](../../evaluation/101A/attribution.png) · [Sphere crops](../../evaluation/101A/sphere_crops.png) · [Contact crops](../../evaluation/101A/contact_shadow_crops.png)

### Interpretation

The prominent grain is stochastic. The largest individual variance term is physical LOW. The fixed learned response can produce seed-dependent image values because its lookup addresses and throughput weights depend on the same sampled camera and first-event paths that contribute to LOW. This shared readout explains why LOW and HIGH cannot be treated as independent noise sources. The positive covariance is 28–34% of the 32-spp MSE budget across the sphere, floor and contact regions.

The independent paired differences separate the stationary residual from stochastic readout. Stable relative RMSE below comes from cross-multiplying the two matched ECRO-minus-reference difference images. Full values use full-reference RMS, HIGH values use physical HIGH RMS. Noise estimates describe the mean reference, derived from the half difference.

| ROI | Stable full RMSE [95% interval] | Full reference-mean noise | Stable HIGH RMSE | HIGH reference-mean noise |
|---|---:|---:|---:|---:|
| Full | 0.269% [0.239%, 0.300%] | 0.287% | 1.854% | 1.409% |
| Sphere | 1.054% [0.772%, 1.319%] | 1.031% | 2.844% | 1.910% |
| Sphere interior | 0.618% [0.491%, 0.742%] | 1.165% | 1.574% | 2.042% |
| Sphere boundary | 1.590% [1.119%, 2.046%] | 0.700% | 4.924% | 1.466% |
| Floor | 0.762% [0.584%, 0.931%] | 0.656% | 2.546% | 1.524% |
| Contact/shadow | 2.182% [1.621%, 2.744%] | 1.261% | 4.357% | 1.850% |

The raw relative RMSE of the average matched LOW difference is below 0.002% in every ROI, including contact/shadow. The stable residual is therefore assigned to the learned HIGH response and its readout under the shared WL100/WL101 physical definition. The matched pairs control physical-path sampling. Finite physical HIGH sampling still contributes to uncertainty, especially in the interior. The intervals quantify spatial and seed variation within this fixture and do not measure independent SSS-model error.

### Unresolved question

The experiment cannot separate learned weight error from patch/bin discretization or other HIGH readout approximation. The two reference halves implement the same finite-depth homogeneous transport as WL100/WL101. The independent native-volpath qualification in WL100 differs by up to 2.3% in total RGB energy in its separate isolated-sphere test, so this study does not certify the absolute accuracy of the physical SSS model. One camera, scene and checkpoint were tested.

### Architecture verdict

**INCONCLUSIVE under the frozen LOW-only dominance rule.** The rule requires a lower 95% LOW share above one half in both sphere and floor. Floor passes. The sphere estimate is 49.90% and its interval crosses one half. The stochastic origin of the 32-spp grain is resolved independently, with 98.85% of measured error due to seed variation in both primary regions. The systematic floor after convergence is localized primarily to HIGH, with the largest listed full-render relative error in the contact/shadow ROI.

The data justify investigating denoising at this operating point before changing the representation. Such an investigation should evaluate the full readout under the measured LOW–HIGH correlation and measure the persistent HIGH error separately. This batch implements no denoiser and grants no successor authorization.

### Reproducibility

- Run and analysis commit `09ad2e6a3b85565c0087fef81d7ad52a96a668fb`. Original protocol and both execution records are available alongside the isolated scripts. [Experiment](../../experiments/ecro_sss101a)
- LabServer63 RTX 3080 Ti 12 GiB, NVIDIA 595.71.05, OptiX 595.71.05, Python 3.10.21, Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, NumPy 1.26.4, Matplotlib 3.10.9 for analysis. The local standalone panels use the same fixed transfer and limits with NumPy 2.3.5 and Matplotlib 3.10.6.
- Raw evidence remains on LabServer63 under `~/rest_sss101a/results/ecro_sss101a/qualified63/`. It contains 96 sweep frames, three 4096-spp replays, frozen source and response state, all analysis maps and the run manifest. The original references and renderer dependencies are staged under `~/rest_sss101a/wl101/`. Process logs and exit markers are under `~/rest_sss101a/qualified63/`.
- Set `DYN101_MI_DIR` to the staged `wl101/external/mi_3.8.0` and `DYN101_MI_VERSION=3.8.0`. Add `$HOME/nvlibs/optix595` ahead of other process libraries. With the existing server environment run `run.py WL101_ROOT FRESH_RAW_ROOT 09ad2e6a3b85565c0087fef81d7ad52a96a668fb`. Invoke `analyze.py FRESH_RAW_ROOT FRESH_REVIEWER_ROOT WL101_REFERENCE_DIR 09ad2e6a3b85565c0087fef81d7ad52a96a668fb` and `export_panels.py FRESH_RAW_ROOT FRESH_REVIEWER_ROOT/panels` after successful rendering.
- The manifest records the full SPP/seed inventory, software and source identities, input hashes, timing flags, table and configuration fingerprints and per-file raw hashes. The representative 32-spp image is seed 10163200 from the untouched WL101 estimator. [Run manifest](../../evaluation/101A/run_manifest.json) · [Reviewer gallery](../../evaluation/101A/index.html)
