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
