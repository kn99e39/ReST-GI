# Worklog 104A — ECRO v0 HIGH-transport fidelity against its learned-ER source (2026-11-18)

## Authorization and freeze, 2026-11-18 KST

The user's brief asks whether ECRO v0 adds a significant, systematic sharp-interreflection energy-magnitude or spatial-distribution error beyond its frozen learned-ER material source. WL104 is the preserved baseline. The batch qualifies the architecture; a correction is allowed only for a localized implementation or estimator defect, and no redesign, learned-ER change or denoiser is permitted.

- **Batch identity.** This is a suffix of WL104 under AGENTS.md. It resolves WL104's unresolved HIGH-only comparison, and no WL104A exists on any branch.
- **Where it runs.** Branch `ecro_wl84_scene` (WL104 line; `Denoise_Reconstructor` and the main branch are untouched), on the local RTX 5080 by the owner's answer.
- **Concurrent work.** `origin/ecro_lr3_adapter` has independently advanced to WL101G audit, WL101H and WL101I. Those are separate work and are not merged here.

**Baseline recovered.** The WL104 commits are `8a91ee24`, `9e1c05c5` and `1a8c66d1` (full hashes in the protocol). All three checkpoints match WL104's recorded hashes, and nothing is retrained.

Development checks against WL104:
- Learned tables regenerated from the checkpoints equal WL104's evaluation tables bit for bit.
- ECRO's 4-spp frame replays WL104's stored frame to 1.2e-7.
- **The two components measure the same events.** ECRO HIGH is `thr · B[patch, bin]` read at camera hit 2, where B is the learned outgoing radiance of 2–5 events at the probe. The tracer's h is next-event estimation at camera hits 3–6. Both are camera events 3–6.
- **Paired LOW.** With a shared seed, ECRO's and the tracer's e1 + e2 differ by 3.4e-10 in their sums.
- **Layered tracer.** It needs the layered plugin's per-launch PASS_SEED (the WL84 convention). With it, the tracer's layered TOTAL matches Mitsuba LAY7 (WL104) at an object mean ratio of 0.9996.

**Design** ([protocol](../../experiments/ecro_wl84_104a/protocol/wl104a.json)). A replicate is 4,096 spp per arm. In each replicate the ECRO, field-table and learned-ER tracer arms share the seed, so the comparison is paired.

Comparisons:
- **Primary, representation (P1):** ECRO HIGH vs learned-ER tracer HIGH.
- **Secondary, material (S1):** learned-ER HIGH and TOTAL vs layered, reported without a verdict.
- **Diagnostics:**
  - D1: the camera readout of the exact field table (the training-target quantity, two independent 16,384-spp estimates) vs learned-ER HIGH. This is the patch/bin projection.
  - D2: ECRO vs that field readout. This is the learned approximation.
- **Controls:**
  - C1: paired LOW;
  - C2: depth truncation, from WL104 halves;
  - C3/A2: tracer totals vs WL104 Mitsuba ER7 / LAY7.

**ROIs**, from geometry before any comparison:
- primary: objects (9,887 px), sphere_lower (1,715 px, shading normal z < 0) and floor_near (5,710 px, within 0.60 m of the sphere axis or 0.25 m of the cube footprint);
- secondary: sphere, cube and full frame.

**Metrics:**
- **Energy:** the signed HIGH energy bias relative to the reference's ROI TOTAL luminance (also relative to HIGH). The interval is a delete-one jackknife over replicates. Field arms add the A/B table-noise term.
- **Spatial:** the stable relative RMS of the RGB HIGH difference, from an unbiased cross-replicate mean square (A × B form for the independent field tables), with a jackknife interval.
- **Descriptive:** relative L1 with its half-split noise floor.

Relative L1 values are never added or subtracted in quadrature.

**Thresholds** (fixed before the pilot):
- τ_E = 1 % of ROI TOTAL. This is below WL104's hard_1 object ratio gap of 1.4 points.
- τ_S = 2 % relative RMS. This is a practically significant fraction of the learned-ER material error, which is 2.7–3.5 % object relative L1.
- Precision target: a 95 % half-width of at most τ/4 for both metrics.

**Verdict rule, per image × primary ROI unit on P1:**
- NOT SUPPORTED if any unit's lower bound exceeds a threshold.
- SUPPORTED if every unit's upper bound is below both thresholds.
- INCONCLUSIVE otherwise, or on any gate failure.

If NOT SUPPORTED, D1 and D2 are tested by the same rule in each failing unit. A correction is allowed only if a control or structural test shows a defect.

**Replicate count.**
- An 8-replicate pilot records interval half-widths only.
- R is then the smallest of 32, 64, 128 and 256 whose projected half-widths meet both targets; 256 is the cap.
- A target the final data misses is reported as unresolved.

**Tests.** `tests104a.py` checks:
- unbiasedness of the stable mean square, including removal of shared table noise in the A × B form;
- coverage of the energy-ratio jackknife;
- the |x| bounds and the RMS floor.

The jackknife interval of the U-statistic mean square is conservative (coverage 0.995). This was accepted before freeze because it can only push a unit toward INCONCLUSIVE.

**Smoke disclosure.** The pipeline smoke (256 spp, 8 replicates, 1,024-spp field tables; non-evidence) ran end to end. Its analysis printed a verdict string, NOT SUPPORTED. Thresholds, ROIs, metrics and rules had already been written to the protocol file before the smoke and were not changed afterwards. No smoke number is used.

## Evidence, 2026-11-18 KST

### Implementation fact

The run at freeze commit `e83b5b3` ran on the local RTX 5080 from 20:41 to 21:12 KST, and the analysis completed.

**Gates and sampling:**

| Item | Result |
|---|---|
| A0 baseline | 3 checkpoints, bitwise tables, 4-spp replays and ROI counts: pass |
| Field tables | 2 × 16,384 spp per image, 43 s each |
| Pilot | half-widths only; R = 32 selected by the rule |
| Replicates | 32 per image, 4,096 spp per arm |
| A1 LOW pairing, every replicate | max relative sum difference 3.9e-10, max pixel difference 1.9e-6: pass |
| A2 tracer totals vs WL104 Mitsuba ER7 / LAY7 | object and full mean ratios 0.9998–1.0001; object rel L1 below every limit: pass |
| Precision targets | met in all 9 image × primary-ROI units |

**Instrumentation only:** paired renders through the unchanged pt102 tracer and readout, field tables from the unchanged `field_components`, and the PASS_SEED loop for the layered tracer. **Unchanged:** models, tables, materials, scene, patch layout and D. No correction was implemented.

**Gate resource use.** GPU utilisation was low, because Dr.Jit 0.4.6 traces each launch in Python on one core. The owner noticed. Parallelizing mid-run was judged not worth it with about 10 minutes left; it will be designed in before the next freeze.

### Measurement

Energy biases are signed and relative to the reference's ROI TOTAL luminance. RMS is the stable relative RMS. Brackets are 95 % intervals.

**Primary units (P1 = ECRO HIGH − learned-ER HIGH):**

| Image | ROI | P1 energy | P1 RMS | D1 projection energy / RMS | D2 learned energy / RMS | Unit result |
|---|---|---:|---:|---|---|---|
| iid_1_L2 | objects | +0.21 % [0.21, 0.21] | 0.81 % | −0.02 % / 0.26 % | +0.23 % / 0.73 % | below |
| iid_1_L2 | sphere_lower | **−1.28 %** [−1.31, −1.26] | **2.65 %** [2.59, 2.70] | −0.14 % / 1.99 % | **−1.15 %** / 1.35 % | significant |
| iid_1_L2 | floor_near | +0.33 % | 0.50 % | +0.10 % / 0.38 % | +0.24 % / 0.33 % | below |
| iid_0_L2 | objects | +0.00 % | 0.52 % | −0.06 % / 0.38 % | +0.07 % / 0.23 % | below |
| iid_0_L2 | sphere_lower | −0.71 % [−0.72, −0.70] | 1.41 % | −0.07 % / 0.94 % | −0.64 % / 0.84 % | below |
| iid_0_L2 | floor_near | +0.31 % | 0.65 % | +0.05 % / 0.54 % | +0.26 % / 0.33 % | below |
| hard_1_L1 | objects | −0.93 % [−0.94, −0.93] | 1.58 % | −0.88 % / 1.56 % | −0.05 % / 0.59 % | below |
| hard_1_L1 | sphere_lower | **−2.14 %** [−2.15, −2.12] | **3.16 %** [3.14, 3.17] | −0.99 % [−1.05, −0.93] / **2.36 %** | **−1.15 %** / **2.12 %** | significant |
| hard_1_L1 | floor_near | −0.78 % | **2.42 %** [2.42, 2.43] | −0.66 % / **2.38 %** | −0.11 % / 0.31 % | significant (RMS) |

Relative to HIGH itself, the P1 energy is −9.9 % (iid_1 sphere_lower), −9.4 % (hard_1 sphere_lower), −7.8 % (hard_1 floor_near) and −4.8 % (hard_1 objects).

**Secondary ROIs.** hard_1 cube: P1 −1.59 % energy and **8.97 %** RMS; D1 −2.21 % and 9.20 %; D2 +0.62 % and 2.47 %. Full frame: P1 within ±0.26 % energy and ≤ 1.11 % RMS in every image.

**Energy additivity.** Energy bias is a linear functional, so P1 = D1 + D2 holds exactly for the point estimates (hard_1 sphere_lower: −2.14 = −0.99 − 1.15). RMS and L1 are not additive and are not combined.

**Material (S1, learned-ER vs layered, no verdict):**
- **HIGH:** energy within −0.70 to +0.38 % of TOTAL and RMS ≤ 1.64 % in every ROI.
- **TOTAL:** objects −3.39 % (iid_1), +0.87 % (iid_0) and +0.15 % (hard_1); RMS 2.7–3.9 %. The learned-ER material error sits mainly in LOW, the direct coated-conductor reflection, not in HIGH.

**Controls:**
- **C1, paired LOW:** 0.00 % energy and RMS everywhere.
- **C2, truncation of events ≥ 7** (WL104 halves, energy / TOTAL):
  - iid: −0.02 to +0.16 %.
  - hard_1: ER65 − ER7 is +0.40 / +0.40 % (objects), +0.41 / +0.40 % (sphere_lower) and +0.28 % (floor_near). Layered gives the same.
  - This is an energy-level effect that WL104's L1-based view could not resolve. It is part of the frozen D = 6 contract and separate from P1.

**Table level** (learned table vs mean field table, all slots):
- relative L1 0.086 / 0.073 / 0.041, against a field half-difference of 0.018 / 0.018 / 0.009;
- learned / field energy ratio 1.019 / 1.012 / 0.990.

### Observation

[HIGH maps](../../evaluation/104A/high_maps_test83_hard_1_L1.png) (one per image) · [interreflection crops](../../evaluation/104A/interreflection_crops.png) · [intervals](../../evaluation/104A/intervals.png) · [gallery](../../evaluation/104A/index.html)

- **hard_1_L1, P1 vs D1.** The P1 map and the D1 map are nearly the same picture:
  - blocky, patch-sized signed regions on the cube faces;
  - the patch boundaries of the sphere's lat-long-like layout;
  - a broad deficit on the floor around and under the sphere.
- **hard_1_L1, D2.** The learned-approximation map is spatially smooth. It shows a coherent deficit on the sphere's lower hemisphere and smaller smooth lobes on the floor and cube.
- **Material (S1).** The map is noise-like except for the highlight ring on the sphere.
- **Noise.** The half-split noise map is far below these structures.
- **iid images.** The same layout appears with smaller amplitudes; the sphere-lower deficit in D2 is the common element.

### Interpretation

Direct evidence:
- **The hypothesis is falsified.** Given a fixed learned-ER material, ECRO v0 introduces a practically significant and precisely resolved HIGH error. It shows in sphere_lower energy (iid_1: −1.28 % of TOTAL; hard_1: −2.14 %) and in sphere_lower and floor_near spatial distribution (hard_1: 3.16 % and 2.42 % RMS). It is not established in iid_0 or in the object-integrated energy.
- **No implementation or estimator defect.**
  - LOW pairs exactly.
  - The tracer totals match independent Mitsuba references.
  - The event ranges agree by construction and by the gates.
  - The hard_1 object darkening that WL104 saw (−0.93 % of TOTAL in HIGH here) is not an accounting error.
- **Two mechanisms, in different places.**
  - **(i) Patch/bin projection (D1).** Reading the exact field through the 991-patch × 64-bin table already reproduces most of the hard_1 spatial error: floor_near and cube RMS, about half of the sphere_lower energy, and nearly all of the object-level energy deficit.
  - **(ii) The learned approximation (D2).** The generator under-predicts the field on the sphere's lower hemisphere by about 1.15 % of TOTAL in both iid_1 and hard_1, with a smooth spatial signature.

Inference:
- **(i)** is the same class of camera-query vs uniform-measure projection error that WL101F found for the SSS fixture.
- **(ii)** looks like limited generator fidelity for the sphere-facing-floor slots, under the frozen recipe.

### Unresolved question

- **What D1 contains.** It mixes patch/bin averaging with the field's probe semantics (the 1 mm membership test of `field_components`), and this batch does not separate them.
- **D1 / D2 RMS intervals are too narrow.** They condition on the two realized field tables. Their table noise is removed from the point estimate by the A × B form but is not in the interval, so near-threshold D1/D2 RMS calls (hard_1 sphere_lower D2 2.12 %) are less certain than shown. The energy intervals include a table-noise term.
- **Untested correction routes.** Whether D2 would shrink with more training data, another seed or capacity is untested; those changes are not authorized.
- **Scope.** One light per material, static scene.

### Architecture verdict

**NOT SUPPORTED** for the frozen ECRO v0 baseline. Closure **C**: a significant ECRO-specific HIGH error is established.
- Its parts are the patch/bin projection, which is an architecture-level representation choice, and the learned approximation, which is a training or capacity limitation.
- Correcting either needs a change to the representation (layout or projection) or to training or capacity. Each requires an owner decision.
- No candidate was implemented, because no localized implementation defect was found.

ECRO's fidelity to learned ER and learned ER's fidelity to the physical layered BSDF are separate results:
- S1 shows the material error is mostly in LOW.
- ECRO does not reproduce the physical transport better than its source.

### Reproducibility

- **Commits.** Branch `ecro_wl84_scene`, freeze `e83b5b3`; the report commit adds this section, `evaluation/104A` and the roadmap line.
- **Stack.** Local RTX 5080, `external/8dna26/.venv-cu128` (Python 3.11.9, torch 2.8.0+cu128, Mitsuba 3.5.1 / Dr.Jit 0.4.6).
- **Checkpoints.** WL104 sha256 values are listed in the protocol.
- **Seeds.**
  - paired arms: 1050000 + 10000 i + r;
  - layered: 1060000 + 10000 i + r;
  - pilot: 1070000 + 10000 i + r;
  - field tables: 1080000 + 10 i + {0, 1};
  - R = 32, 4,096 spp per arm and replicate, 512 per launch.
- **Commands**, from `experiments/ecro_wl84_104a`:
  - `<venv> tests104a.py`
  - `<venv> run104a.py results/ecro_wl84_104a/evidence`
  - `<venv> analyze104a.py results/ecro_wl84_104a/evidence evaluation/104A`
- **Raw data** (gitignored), under `results/ecro_wl84_104a/evidence/`: `status.json` with gates, the pilot and A1; `masks.npz`; `field/`; `pilot/`; `rep/<image>/r###.npz` with every arm component per replicate; the logs.
- **Non-evidence.** `results/ecro_wl84_104a/smoke/` and `dev104a.json`.
- **Display.** The difference-map scale is in `evaluation/104A/evaluation_manifest.json`.
