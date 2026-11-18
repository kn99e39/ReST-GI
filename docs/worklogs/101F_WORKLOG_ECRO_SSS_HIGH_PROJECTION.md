# Worklog 101F — SSS HIGH projection and resolution attribution

## Diagnostic scope and prospective contract

This investigation follows [WL101E](101E_WORKLOG_ECRO_SSS_PHYSICAL_ORACLE_RECOVERY.md) on `ecro_lr3_adapter`. WL101E recovered a physically qualified uniform area–solid-angle table and found +0.07015 near-floor and +0.06260 near-minus-flank signed HIGH residuals. The experiment below tests the role of the projection within the current address space. The learned checkpoint, learned table, scene, material, light, camera, LOW, event ownership, original references, masks and earlier evidence retain their original definitions.

**A. Actual physical query.** A hook at the active event-2 gather records the native interaction, exact outgoing direction, original slot, prior RGB throughput and post-prefix PCG state. A pinned five-event suffix uses that exact interaction. Its E2–E5 is multiplied by the recorded throughput, and the paired full-camera replay compares against physical E3–E6. Source visibility, ballistic event ownership and volume walks continue in the original tracer. The matched query-only control uses the archived seeds and continuation state exclusively for validation. The addressing check reconstructs position and direction in the existing coordinates. The original direction layout remains four rings by sixteen sectors, including the sphere ring-zero extension qualified in WL101E.

**B. Same-layout projection.** Four new camera streams sample the fixed near and flank masks with equal *regional* mass, original pixel/subpixel camera distribution and independent physical suffix RNG. The primary diagnostic projection P is the componentwise throughput-weighted mean of the traced suffix samples at each original address. An unweighted conditional query projection Q uses the same samples and same regional weights. U→Q measures the shift in the sampled coordinate distribution relative to WL101E; Q→P measures the additional within-cell throughput covariance. These effects are estimated against identical held-out camera streams, with no physical reference pixel values used in the construction. Independent WL101E-geometry proposals supply an audit of uniform within-cell spatial and directional usage. Empty weighted cells inherit the corresponding WL101E value. The held-out test accounts explicitly for inherited transport weight. Off-target full-frame metrics describe this fallback as well as the changed cells.

**C. Conditional resolution.** Only if the projected table still leaves a resolved material residual, test subdivisions S (2 × 2 within the equal-area position coordinates), A (2 × 2 within the solid-angle bin coordinates), and SA. Each uses the same training queries and throughput weights. Unsupported subcells inherit the common parent P and the held-out coverage gate applies. Cases with unresolved support or precision are held. The factorial contrast compares spatial, angular and joint resolution within this specific weighting and pair of subdivision factors. The diagnostic projection depends on the current camera and target regions, so its success alone would establish same-layout capability in this fixture and would leave reusable transport unproven.

The [protocol](../../experiments/ecro_sss101f/protocol/attribution.json) freezes all seeds, support and precision requirements and stopping/decision rules. The fitted samples are whole independently seeded fields. Each frozen camera stream is paired across variants and references. Near and flank contrast occurs before confidence intervals. All image metrics share the unmodified WL101C radiance and display definitions.

### Development checks before freeze

The exact-path smoke at seed 10202101 captured 3,570 active HIGH queries at 64² × 1 spp. Splitting physical paths at their HIGH query reproduces LOW exactly and HIGH to 4.58e-8 relative RGB L1 with maximum absolute error 9.54e-7. Reconstructed query position/direction differ by at most 3.77e-7 m / 1.78e-6 in direction vector norm. Replaying query addresses and both generator state words at this seed gives exact equality. An analytic synthetic control also verifies the weighted numerator/denominator, native subcell values and explicit parent fallback.

## Frozen run results

### Gate A passed — actual camera queries reproduce physical HIGH

The paired 4,096-spp query control includes 935,551,945 / 935,540,376 nonzero active query paths in the two original streams. Relative HDR RGB L1 versus the physical HIGH images is 2.24e-6 / 2.32e-6. Across both streams the largest signed difference at either main endpoint is 7.51e-7. The observed gather reproduces the archived learned and uniform HIGH readouts to relative L1 below 5.33e-9. LOW relative L1 is at most 5.28e-9 versus the archived learned streams and 6.50e-7 versus the paired physical references. This is a path/event and readout consistency check using the exact continuation state. Independent physical suffix RNG is used for estimation in Gate B.

### Gate B passed — same-layout projection explains most near-floor excess

Separate prefix/suffix seeds generated 60,293,120 candidate paths through the 7,360 original near and flank pixels and traced 53,225,225 valid, positive-throughput query suffixes. Each replica ranges from 3,385 to 3,420 populated original slots. The smallest held-out native-weight fraction across the two target regions, seeds and table arms is 99.9923%; all fractions use the held-out camera's luminance-weighted throughput denominator. The independent uniform geometry audit accepted 10,262,087 of 10,403,840 proposals, with 23 numerical ties and zero unclassified addresses. It only estimates the within-cell distribution. The baseline table and image values come from the four archived WL101E replicas.

The within-cell uniform-versus-query histograms have throughput-averaged total variation of 0.272 spatial, 0.209–0.210 angular and 0.397–0.398 joint. The query density versus throughput-weighted histograms differ by just 0.000136–0.000142 joint. Per RGB channel, P−Q is the within-cell covariance of prefix throughput and suffix radiance divided by mean throughput. Its per-cell luminance RMS (weighted by queried energy) is 0.00355–0.00527, while the correspondingly weighted signed mean is between −1.50e-5 and −0.79e-5. The signed regional readouts below measure the net effect of that covariance.

Changing only to the held-out projection P reduces the mean near-floor excess by 0.05195 (74.1%; paired 95% CI for the reduction [0.05053, 0.05338]) and excess contrast by 0.02623 (41.9%; [0.02529, 0.02717]). Q has almost identical means. The additional throughput weight in the per-cell target, Q minus P, reduces near bias by 1.12e-5 [0.42e-5, 1.82e-5] and contrast by 1.10e-5 [−0.07e-5, 2.27e-5]. The dominant projection effect is switching from the uniform area–solid-angle measure to actual camera query positions and directions on the target regions. The extra correlation of suffix radiance with prefix throughput has little effect on these endpoints. P retains +0.01820 near and +0.03637 near minus flank, including a −0.01817 flank residual. Both excesses exceed their frozen material-equivalence bands, so Gate C ran.

### Gate C passed — spatial subdivision accounts for the material refinement gain

| HIGH arm | Near-floor signed excess [95% interval] | Near-minus-flank signed excess [95% interval] |
| --- | ---: | ---: |
| Learned | +0.07000 [0.06932, 0.07067] | +0.06554 [0.06466, 0.06641] |
| WL101E uniform U | +0.07015 [0.06871, 0.07159] | +0.06260 [0.06169, 0.06351] |
| Query density Q, original layout | +0.01821 [0.01779, 0.01864] | +0.03638 [0.03595, 0.03681] |
| Throughput weighted P, original layout | +0.01820 [0.01776, 0.01864] | +0.03637 [0.03594, 0.03680] |
| Spatial S | +0.00410 [0.00370, 0.00450] | +0.00818 [0.00776, 0.00860] |
| Angular A | +0.01801 [0.01750, 0.01852] | +0.03599 [0.03553, 0.03645] |
| Joint SA | +0.00401 [0.00347, 0.00455] | +0.00800 [0.00755, 0.00845] |

Linear HDR luminance. Intervals use four sampled field replicas and two paired legacy camera/reference streams. The unit is a whole field or camera stream. Variants share the same query samples within each replica, so the refinement contrasts retain that covariance. The maximum target-interval half-width for the measured candidate residuals and contrasts is below 0.00145; the frozen limits are 0.01052 and 0.00939. The finest joint variant retained at least 99.9549% native transport weight across target regions and held-out streams. The finest joint tables have 18,881–18,920 populated subcells, spatial has 9,805–9,841, and angular has 7,143–7,200.

Spatial subdivision reduces the P residual by +0.01410 [0.01407, 0.01413] near and +0.02819 [0.02813, 0.02826] in contrast. The resulting S and SA intervals lie inside both 20%-of-U material bands. Angular subdivision gives +0.00019 and +0.00038 improvement, both below the predeclared material threshold. The signed interaction P−S−A+SA is +0.000102 [0.000066, 0.000138] near and +0.000210 [0.000150, 0.000270] in contrast. Adding direction subdivisions on top of S further reduces excess by only +0.000088 and +0.000174. No material gain requires joint refinement.

### Spatial observation and secondary errors

The positive contact lobes in the U residual weaken with the same-layout projection and weaken again with spatial subdivision. Direction-only and joint residuals look like P and S respectively around the original hotspot. Across the 0.10–0.14 m band the signed excess is +0.07789 for U, +0.02617 for P, +0.01142 for S, +0.02607 for A and +0.01142 for SA. S's band interval is [0.01081, 0.01202]. In the contact crop and radial plot a positive local residual remains. Within the broad contact/shadow mask P's signed mean is +0.00062 and its relative RMSE is 4.87%, reflecting cancellation across pixels. S gives +0.00379 and 3.38% there. [Shared-scale images and profiles](../../evaluation/101F/index.html)

| ROI relative RGB HDR RMSE | Uniform U | Weighted P | Spatial S | Angular A | Joint SA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Near floor | 32.33% | 9.99% | 4.96% | 9.95% | 4.93% |
| Flank | 3.10% | 6.44% | 2.55% | 6.51% | 2.59% |
| Broad contact/shadow | 4.96% | 4.87% | 3.38% | 5.45% | 3.90% |
| Sphere silhouette band | 4.35% | 7.66% | 7.01% | 7.68% | 7.41% |
| Full frame | 2.06% | 2.34% | 2.12% | 2.41% | 2.21% |

The target-region projections alter shared table entries, and away from the target regions they include significant uniform or parent fallback weight. Native (without fallback) weight on the full frame is 25.5–25.6% for P and 20.7–20.8% for S. At the sphere boundary those fractions are 50.2–53.8% and 37.9–39.1%, respectively. Off-target quality is descriptive. The sphere-boundary dark bands seen with P and S coexist with the target-region improvement. [Region and radial metrics](../../evaluation/101F/metrics.json)

### Interpretation and architecture verdict

**REFINEMENT_ISOLATED in this comparison.** For near-floor excess, the same-layout projection accounts for 74.05% of U, and spatial subdivision at fixed weighting and angular layout accounts for another 20.10%, leaving 5.84%. For near-minus-flank excess these shares are 41.90%, 45.04% and 13.06%. Thus the dominant near-floor error comes from projection mismatch, and within-patch variation explains the material residual left by this projection. This spatial mechanism is present at both tested direction resolutions. The improvement from the tested directional subdivision and its interaction is small. The paired replay excludes a material contribution from optical-event or lookup instrumentation mismatch in these comparisons. The RGB decomposition closes to 0 in float64.

**Defer the deployable representation decision.** This comparison establishes what the fixed layout can recover under one independently sampled, camera-conditioned projection and what these spatial subdivisions add. It leaves open whether another single projection on the coarse layout can meet both targets without a particular view or ROI, where the small residual in the strongest radial band comes from, and whether a spatial refinement can preserve the off-target silhouette, source independence, geometry-edit reuse and lifecycle cost. Source encoding, supervision, and generator correctness remain separate questions. No contract change follows from the camera-conditioned controls.

### Reproducibility and cost

- Execution commit `518e6ad59d01b1e120cbe2558538afe36abd90c6` on `ecro_lr3_adapter`. Protocol SHA-256 `03d538a9905f880d7e050a77aab75066f95fd6d5bedd618677d2e0281b106b23`. The [run manifest](../../evaluation/101F/run_manifest.json) records the scene fingerprint, frozen checkpoint and learned table hashes, every original source file and both physical reference hashes. Baseline WL101E manifest SHA-256 `0fb45d7b9ac46c204c393299a77180c74d9f043f44100361980afe0f5427c0c9`.
- LabServer63 RTX 3080 Ti, NVIDIA/OptiX 595.71.05, Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, Python 3.10.21 and NumPy 1.26.4. Camera/reference seeds 101100 and 101101, 4,096 spp each in launches of 32. Diagnostic estimate uses 2,048 spp per selected pixel in launches of 32, prefix seeds 10202201–10202204 and suffix seeds 10202301–10202304. Uniform geometry audit uses seed 10202401 and 128 proposals per slot in launches of 8. The frozen run completed without a process interruption.
- The full two-stream path control took 142.21 s, the uniform audit 2.37 s, and the four independently sampled, nested field estimates 64.65 s. Batched held-out readout took 49.60 s for two projection arms × four fields × two streams, and 50.42 s for three refinement arms × four fields × two streams. Each phase shares its prefixes. These are synchronized experimental stage timings and include any kernel compilation in the phase. Per-variant arrays store dense RGB float32 responses of 0.93 MiB per field for P or Q, 3.72 MiB for S or A and 14.88 MiB for SA, plus Boolean native support. Observed GPU memory at the phase probes was at most 6,398 MiB for A, 7,522 MiB for B and 7,554 MiB for C, with allocator caches resident. These timings and memory observations describe the diagnostic workflow.
- GPU run command in the established LabServer63 staging workspace, with the same interpreter and OptiX environment as WL101E:

  ```sh
  python experiments/ecro_sss101f/run_attribution.py wl101 \
    results/ecro_sss101f/primary 518e6ad59d01b1e120cbe2558538afe36abd90c6
  ```

  The runner creates the output directory. To regenerate the review from the project root:

  ```sh
  python experiments/ecro_sss101f/report.py results/ecro_sss101f/primary \
    results/ecro_sss101e/recovery2 results/ecro_sss101a/wl101_inputs \
    results/ecro_sss101a/qualified63 evaluation/101F
  ```

  [Evidence index](../../evaluation/101F/manifest101f.json) covers `results/ecro_sss101f/`, including physical queries, uniform geometry counts, four sampled usage packets, weighted sums and counts, all table variants and native support masks, held-out readouts, exact source and output hashes, and the development smoke. Offline recomputation from the recorded per-cell physical sums exactly reproduces all five diagnostic table families and support masks. The original WL101C–E evidence is preserved.
