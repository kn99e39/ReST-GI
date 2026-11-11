# Worklog — ECRO Sharp-Response Failure Attribution (WL86A) (2026-11-11)

## Origin and scope

- **Brief.** "WL87 — ECRO Sharp-Response Failure Attribution and Conditional Single-Mechanism Validation" (user, 2026-11-11). By the user's instruction it is run as **WL86A**, not 87.
- **Kind of batch.** One bounded causal investigation of the sharp directional failure of WL86's ECRO v0: baseline verification → error localisation → input-information and learning attribution → attribution gate → at most one admissible intervention → independent validation.
- **Question.** Does ECRO v0 fail on sharp directional interreflection because physically distinct transport conditions become indistinguishable to its runtime inputs, or because its learned response mapping fails despite sufficient input information?
- **Frozen.** WL84 ER; the WL86 checkpoints `ecro_s8601` / `ecro_s8602`; the WL86 representation, splits, references, thresholds and verdicts. WL86's code (`experiments/ecro86/`) is imported unchanged.
- **Where.** Branch `Selective_Recompute`; LabServer63; `experiments/ecro86a/`, `results/ecro86a/`, `results/evaluation/86A/`; server `~/rest_ecro86a` (reads `~/rest_ecro86`).
- **Not done.** No ER change, no capacity / bin / patch / rank change, no new architecture, no sharp-path solver.

## Deployed WL86 runtime, verified against the code

| Item | Deployed value (file) |
|---|---|
| Patches / incident bins / outgoing bins | 1,270 / 64 / 64 (`scene86.py`: 4 rings in cos θ × 16 azimuth sectors, piecewise constant) |
| Source encoding | patch-average direct irradiance per incident bin from 16 fixed sample points per patch per light; the bin is the one containing the light direction at each point (`pt86.Ctx.source`) |
| Geometry state | 512 fixed stratified rays per patch (8 per incident bin): hit patch and outgoing bin at the hit patch (`pt86.Ctx.state_rays`) |
| Generator inputs | patch centroid, frame, area, frozen ER state, openness, the state-ray form-factor matrix (`model86.ECRO.factors`) |
| Operator | `B = a·(c·E1 + P·Σ area Q·u)`, `u = c·s`, `E1` = one flight of `a·u` through the state rays; R = 16 channels, global rank 24; linear in `s` |
| Readout | at the second camera-path vertex: `throughput(x1) × B[patch(x2), bin(x2 → x1)]`, nearest patch and bin (`pt86.Ctx.trace`, mode `ecro`) |

A source-linear operator on this encoding has exactly one response per (patch, incident bin) cell.

## Stage A — baseline verification (stored WL86 data, independent code)

- `diag86a.py baseline` recomputes every WL86 Stage 1 / Stage 3 value from the stored renders: maximum difference from `stage1.json` / `final.json` is 0.
- The failing quantity is a **signed deficit**, present before and after the edit:

| Case | H share | Oracle table E_H | Learned E_H (8601 / 8602) | Learned signed error | Reference noise |
|---|---|---|---|---|---|
| E2 T1, region C | 0.709 | 0.0137 | 0.0717 / 0.0634 | −0.070 / −0.061 | 0.009 |
| E2 T0, region C | 0.621 | 0.0096 | 0.0603 / 0.0588 | −0.060 / −0.056 | 0.007 |
| E1 T1, region C | 0.396 | 0.0074 | 0.0132 / 0.0110 | −0.010 / −0.007 | 0.004 |
| E2 T1, panel object | 0.668 | 0.0115 | 0.0555 / 0.0515 | −0.050 / −0.037 | 0.008 |
| E1 T1, panel object | 0.847 | 0.0121 | 0.0400 / 0.0339 | **+0.039 / +0.033** | 0.008 |
| E2 T1, sphere object | 0.383 | 0.0072 | 0.0106 / 0.0100 | −0.004 / −0.002 | 0.003 |
| E2 T1, region D | 0.239 | 0.0024 | 0.0043 / 0.0034 | −0.004 / −0.002 | 0.002 |

- The failure exceeds reference and render noise by 6–8× and repeats over both seeds. The sign differs between the two edits on the panel.

## Stage B — error localisation (stored WL86 data)

- **Exact field → render identity.** For a region R, `Σ_pixels hl = Σ_{patch, bin} W_R·B`, with `W_R` the accumulated throughput of the LOW bounce. It reproduces the stored oracle renders to 0.03 % and the reference higher-order term to 0.5 %. The signed rendered error of a table therefore decomposes exactly over sender patches and outgoing bins.
- **What region C reads (E2 T1):** back wall 62 %, box B 27 %, floor 5 %; 89 % in the near-normal ring of outgoing bins. The panel reads the table through a near-mirror bounce, so a few sender patches carry most of the weight.
- **Where the deficit is** (seed 8601 / 8602, share of total radiance):

| Case | Total signed | Back wall | Box B | Level part | Directional part |
|---|---|---|---|---|---|
| E2 T1, C | −0.071 / −0.062 | −0.042 / −0.027 | −0.028 / −0.035 | −0.069 / −0.058 | −0.002 / −0.004 |
| E2 T0, C | −0.058 / −0.055 | −0.056 / −0.057 | 0.000 / +0.003 | −0.056 / −0.053 | −0.002 / −0.002 |
| E1 T1, panel | +0.038 / +0.032 | +0.041 / +0.033 | −0.003 / −0.002 | +0.036 / +0.030 | +0.002 / +0.002 |

- The error is a **level error of whole Lambert patches** (their table is 13–20 % low on E2, 6–22 % high on E1), not a redistribution over outgoing bins. The patches are the back-wall patches left of centre (x −1.1 … −0.6, z 0.1 … 0.9) and, at E2 T1, the face of box B that faces the panel.
- **Field level against render level.** Over the whole back wall the learned table has relative L1 0.044–0.045 (the oracle's own half-split noise is 0.043) and flux ratio 0.995–1.006. The rendered deficit is 7–10 % because the mirror readout concentrates on a few patches.
- **Sphere and broad controls:** signed errors ≤ 0.004, no dominant sender.

**Prospective hypothesis H_A** (written before any Stage C–D measurement). The mis-levelled Lambert patches are where the light → panel (sharp) → patch family lands. Within one incident bin (about 0.35 rad) the specular beam of the panel moves by more than a patch, so an operator that sees only the 64-bin encoding spreads that family; patches inside the true beam come out low, neighbours high.

## Stages C–E as frozen (`experiments/ecro86a/protocol/ecro86a.json`)

**Controlled incident fields** (path-traced tables of the 2–5-event outgoing radiance, same probe as WL86's oracle field; 2 × 2,048 spp):

| Variant | Incident field | Isolates |
|---|---|---|
| TRUE | the workload's light | validation of the diagnostic tracer against the stored oracle field |
| ANG_PANEL / ANG_SPHERE / ANG_BROAD | same irradiance, position and visibility; direction redrawn uniformly inside its incident bin, on the panel / the sphere / all broad patches only | angular binning, by sender class |
| ENC_PANEL | panel patches lit by the re-emitted encoding (patch-average irradiance per bin, uniform direction in the bin) | spatial + angular binning at the sharp sender |
| ENC_ALL | every patch lit by the re-emitted high-quality encoding | an incident field with the **same encoding** as the light: what any operator that sees only the encoding cannot tell from the light |
| ENC16_ALL | as ENC_ALL with the deployed 16-point encoding | the estimator on top of the binning |

**Identical-encoding pairs.** Collimated beams on one patch of the E2 T1 geometry (panel S1, sphere S2, floor B, a Lambert wall patch): two directions inside the same incident bin with equal E cos θ (polar and azimuthal pairs), and the two halves of a patch with doubled E. Each pair has identical encodings by construction; the frozen operator must return identical tables for both.

**Frozen-operator evaluations.** 16-point against 1,024-point encoding through the frozen operators; field error by table-patch class on the high-quality training subset and on validation.

**Rules** (written before any Stage C–D measurement).

- Validation: TRUE field against the stored oracle field (flux within 1 %, importance-weighted difference within 4 × noise); pair encodings identical; operator outputs identical.
- Counterexample: identical encodings and path-traced responses differing by ≥ 0.2 relative L1 and ≥ 4 × noise.
- Source angular encoding SUPPORTED (dominant) if ENC_ALL explains ≥ 50 % of the learned signed error with the same sign in ≥ 2 of the 3 primary cases (E2 T1 C, E2 T0 C, E1 T1 panel) and a sharp-sender counterexample exists; WEAKENED if < 20 % in all three.
- Sender specificity: ANG_PANEL ≥ 2/3 of ENC_ALL's effect, ANG_SPHERE and ANG_BROAD each < 20 %.
- Source estimator SUPPORTED if the 1,024-point encoding changes the frozen operator's error by ≥ 25 % of it in ≥ 2 primary cases; WEAKENED if < 10 % in all.
- Learned mapping WEAKENED as the cause if the learned tables agree with the encoding-equivalent field at the important patches (patch correlation ≥ 0.7, explained fraction ≥ 0.5); SUPPORTED if ENC_ALL explains < 20 % (inputs suffice, mapping fails).
- Geometry state: the generator inputs contain the patch tables of every surface, so the box poses are determined; implementation identity is re-checked; SUPPORTED only with an identical-state / different-response pair.
- **Intervention gate.** If the supported cause is the angular source encoding, it lies in the frozen input contract: no admissible intervention exists (more bins, another encoding, a sharp-path solver are excluded) → stop at attribution, no retraining, no E3.

## Pre-freeze development (non-evidence)

- Stage A and Stage B were run once on the stored WL86 data (results above); they motivated H_A.
- Two dev runs of the whole chain at 64 spp checked the plumbing only; their values are noise-dominated and were not interpreted.
  - A collimated beam from infinity is blocked by the room; a beam is now emitted by a virtual projector 0.25 m from its patch.
  - Re-emitting only the 4 strongest bins of a patch lost 1.1 % of the encoding; one of the 16 strongest bins is now drawn in proportion to its irradiance (coverage 1.000).
  - Half-patch beams are defined on planar patches only.
- Checked in the dev run: pair encodings identical (difference 0), frozen-operator outputs identical for every pair (difference 0), box centres recoverable from the patch tables (error 1e-15 m).
- Expected wall time of the evidence chain: about 30 min (6 concurrent GPU processes for fields and beams).

## Evidence runs (LabServer63, 2026-11-11)

| Run | Commit | Wall | Content |
|---|---|---|---|
| Chain 1 | `46593f0` | 5.7 min (05:17–05:23 UTC) | Stage A, Stage B, controlled incident fields (3 configurations × 7 variants, 2 × 2,048 spp), 30 beam fields (2 × 1,024 spp), frozen-operator evaluations, 24 diagnostic renders (2 × 2,048 spp), attribution |
| Chain 2 (amendment 1) | `1bd7df1` | 3.3 min (05:30–05:33 UTC) | truth split by event order and last sender, operator terms, seen / unseen patch-level error, free-factor oracle fit |
| Amendment 2 | `f6483ab` | 1 min | state-relation adequacy for the sharp-sender flight |

- The chains were far shorter than the 30-min estimate: the field probes ran at about 37 s per variant. Mean GPU use of the first pool was 25 % (six short processes, compile-bound for the first half); the render pool 43–51 %.
- Nothing was trained except the free-factor oracle fit, a non-deployable diagnostic. The WL86 checkpoints, references and verdicts are untouched.

## Stage C result — input information (frozen rules)

**Encoding-equivalent incident fields do not reproduce the failure.** P is the importance-weighted signed error of a table in the region, as a share of total radiance; the diagnostic tables are compared with the TRUE diagnostic table.

| Case | Learned P (8601 / 8602) | ENC_ALL | ENC_PANEL | ANG_PANEL | ANG_BROAD | ENC_ALL explains |
|---|---|---|---|---|---|---|
| E2 T1, C | −0.071 / −0.062 | +0.000 | +0.002 | −0.002 | +0.007 | −0.2 % |
| E2 T0, C | −0.058 / −0.055 | +0.002 | +0.005 | +0.006 | +0.006 | −2.6 % |
| E1 T1, panel | +0.038 / +0.032 | −0.002 | −0.003 | −0.009 | −0.001 | −4.9 % |

- Noise of these P values: the four independent half-fields of the true table (stored a, b; TRUE a, b) scatter by 1.3 % of the region-C reading, i.e. about 0.009 of total radiance per half-field. No diagnostic variant comes within a factor of 3 of the learned error.
- Rendered with the WL86 estimator (E_H in region C; threshold 0.06): TRUE 0.017, ENC_ALL **0.025**, ENC_PANEL 0.023, ANG_PANEL 0.027 on E2 T1; 0.012 / 0.016 / 0.020 / 0.023 on E2 T0. An incident field with the same encoding as the light costs about 0.01 of E_H; the learned tables cost 0.063 / 0.072.
- Patch agreement between the learned and the encoding-equivalent deviations: correlation 0.23, 0.04, −0.03.
- **Source estimator.** The 16-point encoding differs from the 1,024-point one by 13–16 % (relative L1 over bins), but the frozen operator's table changes by 0.1–0.2 % and its region error by 1.2 %, 0.3 % and 4.6 % of itself.

**Classification by the frozen rules.**

| Candidate cause | Label | Basis |
|---|---|---|
| Source estimator (16 points) | WEAKENED | effect < 10 % of the learned error in all three cases |
| Source spatial / angular encoding | WEAKENED for the failing cases | ENC_ALL explains < 20 % in all three; H_A is falsified |
| Learned mapping | SUPPORTED (inputs suffice, mapping fails) | rule: ENC_ALL explains < 20 % |
| Camera readout | amplifier, not a defect | field-level error on the sender shapes at the oracle noise, importance-weighted error 7–10 % |
| Validation rule | **FAILED as written** on E2 T1 C | noise proxy was one signed draw (0.0013); with the four half-fields the TRUE − stored difference is 0.9 σ (post hoc) |

**Identical-encoding beam pairs.** Encodings are identical (difference 0) and the frozen operator returns identical tables (difference 0) for every pair.

- As written (entry-wise relative L1 ≥ 4 × noise) **no pair qualifies**: the response to one beam is spread over 81,280 entries and the entry-wise distance is at 1.4–2.1 × its noise.
- Post hoc, with patch-level sums: on the panel the azimuthal pair differs by 0.94 (noise 0.26) and the polar pair by 0.62 (noise 0.22); region C would read 24 / 30 (two halves) from beam a and 78 / 77 from beam b. Sphere 0.13–0.26 (noise 0.08), floor 0.11–0.19 (noise 0.055), Lambert wall 0.075 (noise 0.053; cross-validated distance 0.000).
- So the 64-bin encoding does lose information for a narrow beam on the sharp panel. This fits WL86's narrow-spot contract errors (0.78–0.89). It is not what fails under the workload's point light.

## Amendment 1 — which part of the mapping fails (`protocol/amendment1.json`, post hoc)

**Truth split by family against the operator's terms** (the frozen operator is the exact sum of its single learned flight and its global term).

| Case | Truth: 2 events via panel | 2 events, other senders | ≥ 3 events | Operator reads through its single flight from panel senders | Error of the single flight | of which panel senders | Error of the global term |
|---|---|---|---|---|---|---|---|
| E2 T1, C | 0.188 | 0.351 | 0.461 | 0.028 / 0.036 | −0.123 / −0.127 | −0.116 / −0.110 | +0.050 / +0.064 |
| E2 T0, C | 0.120 | 0.409 | 0.472 | 0.002 / 0.004 | −0.078 / −0.093 | −0.074 / −0.073 | +0.017 / +0.035 |
| E1 T1, panel | 0.014 | 0.364 | 0.622 | 0.008 / 0.004 | −0.006 / −0.010 | −0.005 / −0.008 | +0.041 / +0.039 |
| E2 T1, D (control) | 0.007 | 0.509 | 0.485 | 0.003 / 0.004 | −0.004 / −0.005 | −0.001 / −0.001 | +0.001 / +0.004 |

(Truth columns: share of what the region reads. Error columns: share of total radiance, seeds 8601 / 8602.)

- **The single learned flight from panel senders is almost absent.** The light → panel → patch family is 19 % (E2 T1) and 12 % (E2 T0) of what region C reads; the operator delivers 0.2–3.6 % through it.
- **The global term compensates without geometry.** It adds +0.02 … +0.06 in E2 and the same +0.04 in E1, where the family is only 1.4 % of the reading. That is the over-prediction on E1's panel and explains the opposite signs of the two edits.
- Rule `term_localisation`: SINGLE FLIGHT (its error is 1.9 × and 1.5 × the total error in the two E2 cases; in E1 the error is the global term's).

**Seen against unseen geometries** (broad patches, debiased patch-level RMS error by form factor to the panel front).

| Panel form factor | Training subset (8601 / 8602) | Validation (8601 / 8602) | Seen / unseen |
|---|---|---|---|
| 0.03–0.10 | 0.060 / 0.057 | 0.151 / 0.108 | 0.40 / 0.52 |
| ≥ 0.10 | 0.096 / 0.084 | 0.174 / 0.111 | 0.55 / 0.75 |

- Rule `coverage_generalisation`: INDETERMINATE (between 1/3 and 2/3). There is an error on seen geometries and a larger one on unseen ones.
- Patch-level target noise is 0.008–0.011 at 1,024 spp (about 0.02 at the training sample count): 5–10× below the 13–20 % level errors. Supervision noise is WEAKENED as the controlling cause.
- Field error by table-patch class (debiased relative RMS, training subset / validation): Lambert walls 0.05 / 0.07; floor 0.09 / 0.09; sphere 0.13 / 0.15; **panel 0.28 / 0.31**; box B 0.07–0.08 / 0.17–0.21. The panel's own outgoing table is poor on seen geometries too.

**Free-factor oracle fit** (E2 T1 geometry, 64 random point lights, test light L2 held out; non-deployable).

| Operator | Signed error read by region C under L2 (of the higher-order reading; 8601 / 8602) | Residual / frozen | Held-out-light field error |
|---|---|---|---|
| Frozen generator | −0.100 / −0.087 | 1 | 0.060 / 0.059 |
| Free global tables (P, Q) | −0.064 / −0.059 | 0.64 / 0.68 | 0.055 / 0.053 |
| All factor tables free | −0.062 / −0.042 | 0.62 / 0.49 | 0.072 / 0.070 |

- Rule `operator_family_capacity`: INDETERMINATE. Fitting the operator's own factor tables to this very geometry removes a third to a half of the deficit and leaves the rest; freeing every table over-fits (held-out-light field error rises).

## Amendment 2 — can the state relation carry the sharp-sender flight? (`protocol/amendment2.json`, post hoc)

For the light → panel → Lambert patch family, the receiver irradiance through the deployed 512 state rays, reading the exact panel radiance (RAY) or the path-traced panel radiance averaged per outgoing bin (BIN), against the path-traced truth.

| Case | Truth reading | RAY | BIN | Patch-level relative RMS (RAY / BIN) | Learned single flight (8601 / 8602) | Rays hitting the panel per important patch (min / median / max of 512) |
|---|---|---|---|---|---|---|
| E2 T1, C | 0.181 | 0.144 (−20 %) | 0.153 (−16 %) | 0.50 / 0.38 | 0.025 / 0.033 | 0 / 13 / 249 |
| E2 T0, C | 0.116 | 0.093 (−20 %) | 0.086 (−26 %) | 0.93 / 0.47 | 0.002 / 0.003 | 0 / 17 / 83 |
| E1 T1, panel | 0.013 | 0.013 | 0.012 | 0.36 / 0.37 | 0.008 / 0.004 | 4 / 14 / 46 |

- Rule: **SAMPLING-LIMITED** in the two cases where the family is resolvable (> 2 % of the reading). E2 T1 passes the limit only marginally (reading off by 20.4 % against 20 %; patch RMS 0.496 against 0.5); E2 T0 is clear (patch RMS 0.93). Truth noise at patch level is 0.03–0.04.
- Even an exact sender radiance, read through the deployed rays, places this family patch by patch with 50–90 % error: a Lambert patch sees the panel through a median of 13–17 of its 512 rays, and the panel's reflected lobe is narrow.
- The learned operator carries 14–18 % of the truth (E2 T1) and 1–3 % (E2 T0), i.e. far less than the 74–84 % the relation could carry in aggregate.

## Final report

### Implementation fact

- Diagnostics only, on the frozen WL86 operators: exact recomputation; importance decomposition of the rendered error over sender patches and bins; seven controlled incident fields per configuration; identical-encoding beam pairs; frozen-operator evaluations with 16- and 1,024-point encodings; truth split by path family against the operator's two terms; seen / unseen patch-level errors; a free-factor oracle fit; a state-relation adequacy test.
- No intervention was implemented and no model was retrained. No E3 was generated: the internal gate did not admit an intervention.

### Observation

1. The failing metric is a static signed deficit of what the panel reads (hl / h 0.90–0.91 on E2), present before and after the edit, and an excess on E1's panel.
2. It is carried by a handful of Lambert patches on the back wall and on the facing side of box B whose whole table is 13–20 % low (6–22 % high on E1). Averaged over the wall the table is as accurate as the oracle's own noise; the near-mirror readout of the panel exposes those patches.
3. Those patches are where the direct light reflected by the panel lands. Path tracing puts 19 % / 12 % of region C's higher-order reading in that two-event family; the operator's single learned flight from panel senders delivers almost none of it.
4. The operator's global term adds a geometry-blind amount instead (+0.02 … +0.06), which under-fills E2 and over-fills E1.
5. Replacing the light by an incident field with the same encoding leaves the truth unchanged within noise in these cases; the source encoding is not what is missing here. For a narrow beam on the panel it does lose information.
6. The deployed state rays, reading exact panel radiance, reproduce the family to −16 … −26 % in aggregate and with 40–90 % patch-level error.

### Causal attribution

| Candidate cause | Classification | Supporting | Weakening / confound |
|---|---|---|---|
| Source estimator (16 points) | WEAKENED | — | 1–5 % effect on the frozen operator's error |
| Source spatial / angular encoding | WEAKENED for the workload failure; information loss exists for narrow beams on the sharp sender (post hoc statistic) | beam pairs: 2.8× different region-C reading at identical encodings | ENC_ALL explains −5 … 0 %; rendered E_H 0.025 against 0.06 |
| Geometry-state encoding | Inputs complete (box poses recoverable to 1e-15 m; state identity re-checked). The **state relation** used for the learned flight is SAMPLING-LIMITED for the sharp sender (one case marginal) | amendment 2 | aggregate reading still 74–84 % of the truth, so sampling alone does not explain a 1–18 % delivery |
| Learned mapping | **SUPPORTED** as the proximate cause: single flight from panel senders almost absent, global term compensates | term split; free fit improves only partly | cannot be separated from the sampling limit: a patch-level signal with 50–90 % error is one a squared-error fit would suppress (inference, not tested) |
| Supervision noise | WEAKENED | — | patch-level target noise 5–10× below the error; squared-error loss unbiased |
| Coverage / generalisation | INDETERMINATE | unseen error 1.3–2.5× the seen error | sizeable error on seen geometries; panel table 0.28 on the training subset |
| Operator-family capacity | INDETERMINATE | free tables leave 49–68 % | fit is on 64 lights of one geometry; over-fitting when all tables are free |
| Camera readout | amplifier, not a defect | importance concentrated on few patches | — |

- **Interacting causes.** The evidence does not support a single cause. It supports: the learned single flight for the sharp sender fails (proximate), in a setting where the state relation that would have to carry it is under-sampled (structural), with the global term absorbing the family in a geometry-blind way.
- **Intervention gate.** No bounded admissible cause is supported. The state-ray budget and the bins are frozen; supervision noise and the source estimator are weakened; retraining longer or differently would not be attributable. The batch stops at attribution.

### Unresolved questions

- Whether a denser state relation alone would make the single flight learnable is not tested (it changes the frozen budget).
- Optimisation against operator-family capacity is not separated: the free-factor fit is indeterminate and no retraining was done.
- Amendment 2's classification is marginal on E2 T1 and rests on two cases.
- The region-C edit-response slope (0.36–0.63 in WL86) was not re-measured by family; its attribution to the same family is an inference from the static result.
- The sharp-receiver side (the panel's own table, error 0.28–0.31) matters little for these regions and was not analysed further.
- The beam counterexample statistic and both amendments are post hoc; the Stage C validation rule failed as written.

### Architecture verdict

| Item | Verdict | Scope |
|---|---|---|
| OUTPUT REPRESENTABILITY | **SUPPORTED** | WL86 oracle projection, reproduced exactly (E_H 0.014 in E2 region C) |
| SOURCE INPUT SUFFICIENCY | **CONDITIONALLY SUPPORTED** | sufficient for the point-light workload failure (encoding-equivalent field: E_H 0.025, explains ≤ 5 %); not for narrow directional sources on the sharp sender (post hoc) |
| GEOMETRY STATE SUFFICIENCY | **CONDITIONALLY SUPPORTED** | generator inputs determine the geometry; the 512-ray relation is sampling-limited for the sharp-sender flight |
| LEARNED MAPPING ADEQUACY | **NOT SUPPORTED** | single flight from the sharp sender delivers 1–18 % of the truth; global term compensates blindly |
| SUPERVISION / OPTIMIZATION ADEQUACY | **INDETERMINATE** | noise weakened; coverage and capacity / optimisation not separated |
| SHARP EDIT RESPONSE | **NOT SUPPORTED** | unchanged WL86 measurement; attributed (by inference) to the same family |
| INTERVENTION GENERALIZATION | **NOT TESTED** | no intervention admitted by the gate; no E3 |
| ECRO v0 ARCHITECTURE VIABILITY | **NOT SUPPORTED** (WL86 verdict unchanged) | cause now localised; a diagnostic result does not upgrade it |

**Completion outcome.** The cause is identified at mechanism level and is not a single isolated variable: outcome 4 (cause not reducible to one factor) for the learning side, with an element of outcome 3 (the frozen state relation, not the source encoding, is inadequate for sharp-sender flights). No recovery was attempted. The next architecture decision is the human owner's.

### Living documents

- `docs/Architecture.md` is not changed.
- `docs/RESEARCH_ROADMAP.md`: one sentence added to the WL86 evidence record.

### Reproducibility

- **Commits.** Freeze `46593f0`; amendment 1 `1bd7df1`; amendment 2 `f6483ab`; results in the closing commit. Parent baseline: WL86 `7786d09` + `63972c8`, results `b97027d`.
- **Protocol.** `experiments/ecro86a/protocol/ecro86a.json`, `amendment1.json`, `amendment2.json`.
- **Frozen inputs.** `~/rest_ecro86` on LabServer63 (references, oracle fields, masks, checkpoints `ecro_s8601.pt` / `ecro_s8602.pt`, training data); ER states `experiments/ecro86/data/mats86.npz`.
- **Seeds.** Importance walk 86600; diagnostic fields 86900–86965; beams 87100+; renders 87300–87301; split 87400 / 87405; fit lights 86990, fields 87500+ / 87700+; q-field 87800 / 87805.
- **Environment.** RTX 3080 Ti 12 GB, driver 595.71.05, Mitsuba 3.5.1, Dr.Jit 0.4.6, torch 2.3.1.
- **Server.** `~/rest_ecro86a` (fields, beams, renders, logs).
- **Local copy.** `results/ecro86a/` (`stageA/baseline.json`, `stageB/localize.json`, `results/attribution.json`, `attribution2.json`, `attribution3.json`, `indist.json`, `model.json`, `fit_*.json`, logs).
- **Visuals.** `results/evaluation/86A/` (manifest with sha256): error by sender, patch maps, beam pair, rendered error, family reading, panel flight on the back wall.
- **Invalid or amended items.** Two 64-spp dev runs (plumbing only). Stage C validation rule failed as written (noise proxy); the counterexample rule did not fire as written; both are reported with post hoc statistics. Amendments 1 and 2 are post hoc.
