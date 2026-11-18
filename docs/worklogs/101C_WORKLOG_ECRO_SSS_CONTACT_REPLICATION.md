# Worklog 101C — Raw ECRO SSS contact artifact replication

## Authorization and freeze, 2026-11-18 KST

WL101C is a suffix follow-up to WL101A and WL101B under the same checkpoint and physical contract. Work takes place in the existing ReST-GI checkout on `ecro_lr3_adapter`. The target is the reported excess brightness on the visible floor next to the sphere's support point. The independent new condition is the RNG sequence. Eight fresh raw 4096-spp renders use the original WL101 optimized frame path on the qualified LabServer63 stack.

The retained WL101A raw outputs already provide the two reference-seed-matched 4096-spp pairs and the independent WL101 prediction replay. The two archived physical reference halves and the geometry buffer remain the references for this study. The matched runs serve as historical controls only. WL101A's low-SPP dominance verdict and WL101B's reconstruction-induced silhouette result are preserved.

The primary intensity tests use the visible front half of the floor, a radius of 0.18 m about the sphere's geometric support point, and a separate 0.26–0.40 m flank. No new residual image was used to choose these regions. The fixed geometry has 1,318 near pixels and 6,042 flank pixels. WL101A's broad contact/shadow mask remains a second measurement region. The protocol freezes radial bins, maxima, shared display, seeds, noise accounting and the simultaneous direction/ownership criterion. [Protocol](../../experiments/ecro_sss101c/protocol/contact.json)

## Completed evidence, 2026-11-18 KST

### Implementation fact

The original `rt101.render_hq` called the existing `RT._frame_prog` and `pt100.Ctx.trace` in ECRO mode. Each frame used 4,096 spp in 128 launches of 32 spp. The run packed the original E1, E2 and learned HL channels into float32 after float64 accumulation, and recorded LOW and full alongside the original components. This execution path has no reconstruction. The historical physical renders provide the corresponding E1, E2 and six-event remainder H.

The run checked geometry, light, camera, configs, renderer sources, checkpoint and archive hashes before sampling. Its response table was bit-identical to WL101A and archived arm B and stayed bit-identical through all eight seeds. Maximum `LOW + HIGH` versus recorded full relative L1 was **2.34e-8**. Every fresh NPZ was checked offline. The frozen evaluator passed known-input controls for HIGH, LOW and noise-compatible outcomes and checked the contact mask counts. Metrics used unclipped RGB. Render and analysis exited successfully.

### Measurement

All values below compare the mean of eight new 4,096-spp ECRO renders with the mean of the original independent physical halves. **Stable** is the cross product from the frozen 4+4 ECRO split against distinct reference halves. Noise is estimated for the mean physical reference. Percentages use each component's own reference RMS. A negative stable LOW cross estimate is retained in the machine results.

| Region and component | Absolute HDR RMSE | Relative HDR RMSE | Stable relative HDR RMS | Reference-mean noise |
|---|---:|---:|---:|---:|
| Full frame, full | 0.02172 | 0.400% | 0.259% | 0.287% |
| Full frame, LOW | 0.00899 | 0.188% | below resolved floor | 0.177% |
| Full frame, HIGH | 0.01843 | 2.336% | 1.844% | 1.409% |
| WL101A contact/shadow, full | 0.02208 | 2.539% | 2.156% | 1.261% |
| WL101A contact/shadow, LOW | 0.00634 | 1.337% | below resolved floor | 1.240% |
| WL101A contact/shadow, HIGH | 0.02064 | 4.739% | 4.349% | 1.850% |
| Near floor, full | 0.07635 | 26.213% | 26.134% | 1.946% |
| Near floor, LOW | 0.00195 | 3.361% | below resolved floor | 3.101% |
| Near floor, HIGH | 0.07633 | 32.295% | 32.224% | 2.117% |

The near floor's signed full bias is **(+0.06855, +0.07092, +0.06325)** in linear HDR RGB. The broad contact/shadow bias is **(+0.00914, +0.00899, +0.00872)**. In the narrow region, full mean luminance rises from **0.26373 to 0.33360 (+26.49%)**. HIGH rises from **0.21633 to 0.28625 (+32.32%)**, while the LOW difference is **−0.0000475**. The table reports independent whole-render Welch intervals for the signed difference.

| Endpoint, linear HDR luminance | Full difference [95% interval] | LOW difference [95% interval] | HIGH difference [95% interval] |
|---|---:|---:|---:|
| Near ECRO minus physical | +0.06986 [0.06849, 0.07124] | −0.000047 [−0.000208, 0.000113] | +0.06991 [0.06883, 0.07099] |
| Excess of near-minus-flank contrast | +0.06545 [0.06294, 0.06796] | +0.000018 [−0.000351, 0.000386] | +0.06543 [0.06346, 0.06741] |

All eight seeds give positive HIGH near bias and positive HIGH near-minus-flank excess versus each physical half. Across the 16 pairings the near bias varies only **+0.06977 to +0.07010**. The HIGH reference-mean uncertainty proxy for this scalar is 0.000094. The one-sided, family-adjusted lower bounds for full positivity are **+0.06599** at near and **+0.05836** for concentration. For HIGH−LOW the corresponding bounds are **+0.06855 and +0.06308**, and for HIGH+LOW they are **+0.06599 and +0.05836**. The HIGH+LOW bound agrees with the full bound by additivity.

The floor near/flank mean-luminance ratio is **0.759 ECRO vs 0.581 physical for HIGH**, and **0.544 vs 0.433 for full**. The largest fixed annular mean HIGH residual is **+0.07817** in the 0.10–0.14 m band. Profile points use bin midpoints. The first visible floor pixel in any band lies at radius 0.076 m. The largest mask-aware 5×5 HIGH residual in the near ROI is **+0.15538** at image pixel (281,426), world floor position approximately (0.9824,0.3444) m. The corresponding peak in the total residual is +0.15474. Comparing maxima of the two HIGH radiance images within the same mask gives **0.53116 ECRO vs 0.41398 physical**. These maxima describe the observed spatial footprint. They do not enter the gate. [Profile](../../evaluation/101C/floor_intensity_profile.png) · [Full measurements](../../evaluation/101C/metrics.json)

### Observation

All eight raw signed-HIGH crops show bright lobes on the floor just below both sides of the sphere. The darkest contact shadow in the original can still be darker than the flank in absolute luminance. Its central under-sphere region is filled in excessively relative to the physical reference. The WL101A seed-matched HIGH near bias was **+0.06994 / +0.07005**, and the earlier independent seed 101500 gives **+0.06995**. Its paired LOW near differences were smaller than 1.3e-7 in magnitude. The newly sampled image and the historical matched controls have the same localized footprint under the fixed signed scale. [Contact comparison](../../evaluation/101C/contact_crops.png) · [Eight independent crops](../../evaluation/101C/independent_seeds.png)

### Interpretation

In the near ROI, stable full MSE is 0.0057942. The signed component estimates are LOW −4.34e-7, HIGH 0.0058011, and the sum of the two LOW–HIGH cross terms is −6.50e-6. The small negative LOW estimate is consistent with finite independent sampling. The new unpaired LOW image retains visible pixel-level reference and prediction noise, but its mean and concentration intervals contain zero. HIGH accounts for the reproducible positive contact concentration. The same excess is present in the original ECRO frame path with complete reconstruction bypass. WL101B's dark reconstruction-induced silhouette bias is a separate measured residual.

### Unresolved question

This is an image-space result from the fixed checkpoint, scene, camera and finite E1–E6 physical definition. The experiment does not separate the learned response operator from patch/bin discretization or image readout. It does not infer an angular representation failure. The two physical halves bound sampling uncertainty for the archived transport definition. WL100's limited native-volpath qualification remains separate from this inference.

### Architecture verdict

**PERSISTENT HIGH** under the frozen WL101C rule. The original 4,096-spp ECRO path has a reproducible systematic HIGH overbrightness near the SSS floor contact in cfg0. The physical LOW component does not show a resolved same-direction concentration at this precision. This closes the bounded replication without a change to the model, estimator, or architecture.

### Reproducibility

- Render/protocol commit `7cd154ffb81d99587e47838da47a0072dbad07ba`. Scorer commit `436a1f2d2fa7234f835323f63bcbbc5032d1d058`. Seeds **10181001–10181008**. The physical halves retain seeds **101100/101101**. Historical seed **101500** is held out of the new eight-seed inference.
- Original scene fingerprint `a18c76efd49245dfb3f50d6d215e50d1b70e87854bac943126b60b09a50b71cb`. Checkpoint `2d9cb5805ecff8a6aae6639ddca7a3bcd0c84b0eaa9b2a3282fe127df9284f42`. The run manifest includes the config, response, renderer, protocol and archive hashes.
- LabServer63 RTX 3080 Ti, NVIDIA driver and OptiX 595.71.05, Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, Python 3.10.21, NumPy 1.26.4. The same staged WL101 runtime is used. No foreign GPU process was detected at the render boundaries. Synchronized rendering and readback took **84.64 s total**, **10.58 s median**, range **10.569–10.606 s**. The eight NPZ outputs total **96,696,942 bytes**. These are quality-run costs.
- Raw outputs and logs are under `results/ecro_sss101c/` in the existing checkout. Existing reference halves and geometry are in `results/ecro_sss101a/wl101_inputs/`, and the historical ECRO comparison is in `results/ecro_sss101a/qualified63/`. Use `render_contact.py WL101_ROOT FRESH_RAW_DIR COMMIT` on the recorded stack. Use `analyze_contact.py RAW_DIR WL101A_INPUTS WL101A_RAW REVIEWER_DIR` for offline scoring. It reuses WL101A's tone curve with exposure 0.1938553403 and the same numeric signed cap ±0.0905639470, displayed here as Rec.709 luminance. The eight-frame mean is an analysis aggregate and each visual seed is recorded at 4,096 spp. [Run manifest](../../evaluation/101C/run_manifest.json) · [Gallery](../../evaluation/101C/index.html)
