# Worklog 101E — Physically qualified ECRO SSS HIGH Oracle

## Authorization and physical contract, 2026-11-18 KST

This bounded follow-up addresses the failed first-surface gate of [WL101D](101D_WORKLOG_ECRO_SSS_ORACLE_HIGH_ATTRIBUTION.md) on the existing `ecro_lr3_adapter` checkout. It owns a diagnostic physical table estimator. WL100's supervising sampler, the trained checkpoint, B theta, the camera estimator, the physical reference, and earlier worklogs retain their original definitions.

**Physical quantity.** Each of the existing 1,270 × 64 cells represents mean RGB outgoing radiance for its exposed surface-area × solid-angle domain. The unchanged camera prepends one physical interaction to the E2–E5 suffix, so the queried contribution belongs to camera E3–E6. Sampling is uniform in patch surface area and in azimuth and cosine coordinate within the lookup preimage. The first surface sampler uses the local physical front convention of the tracer. Planar patches use the established positive hemisphere. The sphere's lowest ring includes geometrically front-facing directions with negative patch-frame cosine because the existing camera quantizer clamps them to that ring. The partition and camera lookup are fixed.

**Membership and normalization.** A geometry-only native intersection against the intended shape generates the first interaction, with its original normal and shading frame. The full scene, material and original transport program govern source visibility and every subsequent path event. Free-space support excludes the shared interior Floor/Box B interface by its analytic footprint. Sphere/Floor point tangency has zero area, and positive-gap points remain part of their requested domain. This defines the sampled measure before any radiance is calculated. Accepted samples enter the same verified cell in numerator and denominator, with no cosine or flux conversion. Numerical address ties and empty cells have separate counters and an explicit camera-reachability gate. The field is independent of B theta and reference pixels.

**Inference.** Any difference between the learned and physical table after the identical readout is the generated-table-side term, including supervision, source encoding, generator and application. The physical-table residual includes fixed representation and readout. The [protocol](../../experiments/ecro_sss101e/protocol/recovery.json) freezes these quantities, validation, masks, seeds and decision rules before opening Oracle camera images.

### Qualification before primary evidence

The development probe first exposed two runtime issues in the diagnostic attachment code. A default interaction has zero-length fields for this Dr.Jit version, and OptiX permits one shader binding table per materialized kernel. The corrected hook initializes typed interaction buffers and evaluates each isolated native intersection before merging them. This keeps the assembled first-interaction geometry separate from the original full-scene continuation. Both failed development logs are preserved locally with the successful qualification.

The qualification at seed 10200901 passed. The RGB address-coded camera test reproduced its queried IDs to 1.10e-7 and linearity to 2.38e-7. Zero and LOW invariance passed. The same pinned suffix under probe and camera mode gave identical E2 and E3–E5 (maximum 0), and the ordinary unpinned native suffix agreed within 4.17e-7 relative L1 on 2,013 unambiguously exposed matched rays. Independent polygon clipping finds five partly covered floor patches and 0.108 m² of excluded area. The counts from the 16 proposals/slot qualification deviate at most 1.09 binomial standard deviations from that analytic area law.

#### Primary qualification interruption and numerics correction

The first frozen execution stopped at the membership gate in the first 1,024-proposal field. It saw 184 reported ties and one unclassified native miss. No physical comparison or admitted Oracle table had been produced. A read-only replay of the original random proposals isolated launch 29, slot 32675 (ceiling patch 510, bin 35). At area coordinate `u1=0.9999998807907104`, the float32 world point rounded to exactly `x=2.0` m on the ceiling edge. The triangle query missed. There was no wrong-surface hit. The diagnostic packet retains its coordinates, seed, and flags.

The rerun accounts for planar native misses within four float32 epsilons of the area parameter boundary as numerical ties. This is narrower than the original declared coordinate tolerance. It does not reroute samples, assign energy, modify the estimator's intended open surface domain, or increase the frozen tie-rate threshold. The original failed primary log and its replay are retained. The rerun uses the same two references, four field seeds, and all frozen acceptance rules.

## Evidence from the corrected frozen run

The experiment-specific hook and its geometry-only replay are in `experiments/ecro_sss101e/oracle_field.py` and `run_recovery.py`. The hook pins the requested native first surface before calling the original five-event tracer. The WL100 supervisor used a 1 mm position acceptance test and a front gate based on the requested sample, and accumulated accepted samples at their nominal addresses. That procedure let nearby surfaces contribute to a different address. WL101E estimates the mean over the exposed camera lookup domain and verifies the actual address and local front before accumulation.

| Field audit across four independent tables | Count |
| --- | ---: |
| Uniform proposals, 1,024 per slot per table | 332,922,880 |
| Accepted at the requested slot | 328,378,987 |
| Buried points or exact tangencies | 2,812,157 |
| Outside the physical front | 1,730,958 |
| Numerical ties | 778 |
| Unclassified misses or wrong-slot admissions | 0 |
| Accepted sphere ring-zero samples with negative patch-frame cosine | 136,113 |

Numerical ties account for 0.000237% of eligible proposals (frozen cap 0.005%). The largest native attachment error is 5.82e-7 m. Floor coverage agrees with the independent clipped-polygon area to 2.85 binomial standard deviations or better across the four tables. The camera's 64-bin lookup verifies the surface and directional membership before the suffix runs. The tracer's accepted count equals the independent membership indicator in every entry, and the stored E2 plus E3–E5 means reproduce the stored combined RGB field within 2.39e-7 in float32. Raw RGB sample variance and accepted counts are retained per replica and slot. Across supported slots, the minimum accepted count is 156 in a single replica and 692 in the four replicas combined.

The same 576 slots were empty in all four replicas (patches 1220–1228, all 64 directions at the buried Box B underside). On both frozen 4,096-spp camera streams, the unit marker confined to those slots was exactly zero on every pixel and channel. The empty support therefore contributes zero to these comparisons. This marker measures camera transport weight for the empty slots; the substitution measures the radiance difference for the supported field.

### Measurement and uncertainty

Four independent fields were each read on the same two archived 4,096-spp camera streams at 512 × 512. Each stream uses its matching physical six-event reference and archived learned render. The eight substitutions recover E1 and E2 bit for bit against the learned render; relative L1 against the matching reference LOW components is at most 6.49e-7. All images and calculations use undenoised linear HDR. The raw per-pixel decomposition closes exactly in float64.

| Fixed endpoint | Total, learned − physical | Generated-table side, learned − Oracle | Representation/readout side, Oracle − physical |
| --- | ---: | ---: | ---: |
| Near floor, 1,318 pixels | +0.07000 [0.06932, 0.07067] | −0.00016 [−0.00160, +0.00129] | +0.07015 [0.06871, 0.07159] |
| Near minus flank, 6,042 flank pixels | +0.06554 [0.06466, 0.06641] | +0.00294 [0.00200, 0.00387] | +0.06260 [0.06169, 0.06351] |

Entries are mean signed linear luminance with approximate crossed 95% intervals. WL101C measured +0.06991 and +0.06543 for the two total endpoints independently. Mean near-floor HIGH is 0.21633 physical, 0.28633 learned and 0.28649 Oracle. At the flank the learned-to-Oracle signed term is −0.00309 [−0.00493, −0.00126]. Consequently the substitution reduces the excess near-minus-flank contrast by 4.48%, primarily through a brighter flank. The near-floor excess is essentially unchanged.

The two independent table halves differ by 1.65% of table RMS when measured as half their difference. After camera propagation the half differences in mean luminance are 0.000262 for the near floor and 0.000335 for near minus flank. Table mean standard errors are 0.000455 and 0.000293, respectively. The paired camera/reference standard errors in the representation/readout term are 0.000048 and 0.000059. All attribution half-widths are at most 0.00145, below the frozen limits of 0.01748 and 0.01636 for the two endpoints. The independent unit for the field contribution is a whole independently sampled table. The camera contribution retains the pairing and has two independent streams. The approximate intervals carry these finite-sample limitations.

| Region | Pixels | Learned signed luminance | Oracle signed luminance | Relative HDR RGB RMSE, learned → Oracle |
| --- | ---: | ---: | ---: | ---: |
| Full frame | 262,144 | −0.003149 | +0.000743 | 2.262% → 2.056% |
| Sphere | 49,660 | +0.001180 | +0.002838 | 3.303% → 2.997% |
| Sphere boundary | 3,968 | +0.003327 | +0.010751 | 5.093% → 4.354% |
| Broad contact/shadow | 38,071 | +0.009075 | +0.011912 | 4.689% → 4.958% |
| Near floor | 1,318 | +0.069996 | +0.070152 | 32.314% → 32.333% |
| Flank | 6,042 | +0.004459 | +0.007551 | 2.668% → 3.100% |

Signed luminance is relative to the matched mean physical HIGH. Relative RMSE is the RGB error RMS divided by the physical RGB RMS on the same pixels. [Metrics](../../evaluation/101E/metrics.json) retain the signed intervals, RGB and absolute errors, and noise estimates for every region and radial band.

### Observation, attribution and unresolved mechanisms

At the original exposure the learned and Oracle crops show the same broad excess beneath the sphere. On a common signed scale the two residuals share bright lobes on either side of the closest contact. The radial profile puts the largest Oracle excess at 0.10–0.14 m (+0.07789 versus learned +0.07829). The full-frame RMS improves, while the broad contact and flank RMS increase. [Aligned images and radial profile](../../evaluation/101E/index.html).

**Frozen verdict: MIXED across the two endpoints, dominated by representation/readout. The hotspot persists.** The physical-table residual accounts for essentially all of the near-floor excess and 95.52% of the measured excess concentration. The generated-table-side near-floor effect is unresolved. Its small positive contrast term is resolved and includes supervision, source encoding, generator and response application together. Removing first-hit contamination in the diagnostic field is sufficient to qualify the substitution. The earlier contamination's individual contribution to the learned table remains unmeasured.

The residual is conditional on this scene, on the fixed patch/bin partition and lookup, and on the uniform area–solid-angle mean used for the physical cell. Within-cell position and direction variation and the camera's nonuniform throughput-weighted lookup distribution remain coupled. The test leaves their separate contributions unidentified, along with the relative roles of the generated-table stages. It establishes a residual for this physical projection and readout; whether a different projection into the same layout could improve this endpoint remains open.

### Exact reproducibility

- Protocol and initial estimator freeze `52db8cf2f3d7cb78b31088e0b76328a55de105ec`. Final execution source `fc5fa68ecd5adea206f76349f9ed60177afc5753` on `ecro_lr3_adapter`. Protocol SHA-256 `6fba28e397ca8428eb2e659a937a0c59bd99d37949ac408017cc627ce03a8ada`.
- Scene fingerprint `a18c76efd49245dfb3f50d6d215e50d1b70e87854bac943126b60b09a50b71cb`. Original checkpoint SHA-256 `2d9cb5805ecff8a6aae6639ddca7a3bcd0c84b0eaa9b2a3282fe127df9284f42`. Learned table array SHA-256 `d7e40954304bfc2b03f679af0ef6a39876f21ffc21d3a07fe3dfc3de7bd18543`. The [run manifest](../../evaluation/101E/run_manifest.json) gives renderer, configuration, each physical reference, the frozen pixel map, the WL101A source manifest and all Oracle artifact hashes.
- Four field seeds `10201001`–`10201004`, each 1,024 proposals per cell in 32-proposal launches; qualification seed `10200901`, 16 proposals per cell. Original camera/reference seeds `101100` and `101101`, each 4,096 spp in 32-spp launches. Each Oracle field is independent of the camera/reference streams.
- LabServer63, GeForce RTX 3080 Ti, driver/OptiX 595.71.05, Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, Python 3.10.21 and NumPy 1.26.4. Sampling and CPU synchronization totals were 25.60 s for four tables, 21.26 s for the two empty-slot checks and 84.79 s for eight substitutions. The corrected run's field qualification took 9.64 s and suffix validation took 0.68 s. These stage measurements exclude loading, independent range checks, file compression and transfers. During corrected camera rendering the GPU reported 4,064 MiB in use. The successful cold development preflight took about 240 s for its first suffix compilation and validation.
- Render with `experiments/ecro_sss101e/run_recovery.py wl101 results/ecro_sss101e/recovery2 fc5fa68ecd5adea206f76349f9ed60177afc5753` using the pinned interpreter and OptiX paths from the WL101A LabServer63 setup. The runner creates its output directory. One launch ended before sampling because its result directory had been pre-created; `recovery.log` retains that error. The successful replay is `recovery2.log`. The original interrupted primary and its geometry replay are `primary/` and `diagnostic/`. All of these live under [the project's raw result directory](../../results/ecro_sss101e/).
- Regenerate the review under the project root:

  ```sh
  python experiments/ecro_sss101e/analyze_recovery.py \
    results/ecro_sss101e/recovery2 \
    results/ecro_sss101a/wl101_inputs \
    results/ecro_sss101a/qualified63 \
    evaluation/101C/metrics.json evaluation/101E
  ```

  The four response tables and unsmoothed readouts are in `results/ecro_sss101e/recovery2/`, together with the half tables, empty-slot marker images and `analysis_maps.npz`. [The evidence index](../../evaluation/101E/manifest101e.json) identifies the local files by SHA-256. This experiment yields an attribution for the fixed readout; the architecture contract remains in force.
