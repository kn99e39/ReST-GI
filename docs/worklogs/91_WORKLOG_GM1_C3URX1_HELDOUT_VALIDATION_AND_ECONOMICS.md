# Worklog — GM1 C3URX1 Held-out Validation and End-to-end Economic Qualification (2026-11-12)

## Origin and scope

- **Brief.** "ReST-GI — GM1 C3URX1 Held-out Validation and End-to-end Economic Qualification" (user, 2026-11-12). One bounded candidate-evaluation batch.
- **Question.** Does C3URX1 provide a source-independent, physically valid and economically useful first-gather transport mechanism under the qualified GM1 edit / source workload, and is the evidence sufficient to adopt it as the explicit development baseline?
- **Primary hypothesis (prospective).** The benefit of the exact first-gather directional radiance survives held-out conditions and gives useful end-to-end economics.
- **Preserved.**
  - WL84 ER; the WL86–WL90 code, protocols, results and verdicts (imported, never edited).
  - C3UR (WL89) is the preserved explicit development baseline; X1 is a WL90 diagnostic.
  - The WL89 workloads (W1 = V3, W2 = X2), sources (LMIX89, LSPOT89), references, masks, oracle tables and thresholds F1–F4.
- **Not done.** No tuning of sampling, ray counts, bins, ER or thresholds. No learned model, no new representation, no area-light estimator, no GM2. Nothing is adopted automatically.
- **Where.** Branch `Selective_Recompute`, LabServer63. `experiments/gm1_c3urx1_91/`, `results/gm1_c3urx1_91/`, `results/evaluation/91/`. Server `~/rest_x191`, reading `~/rest_rel89` and `~/rest_gap90`.
- **Terminology.** Comparators are physical recomputations with the frozen ER: explicit relations, oracle table and path tracing. External learned baselines are UNMEASURED; no Neural Full Recompute claim is made.

## Candidate (isolated path, `cand91.C3URX1`)

**Mechanism.**
- Relation: the unchanged WL89 C3UR relation (`rel89.SendRelR`). 8 rays per (sender patch, outgoing bin) cell, 650,240 unique rays, reused for all orders, source-independent, with a bit-identical selective update.
- First gather (WL90 X1): L₂[h, b_o] = Σ over landed rays of fD[cell, b_o] · w · Lx₁(y, v).
  - Lx₁ is the exact once-scattered radiance leaving the ray's own origin point in its own direction, under the **current** lights.
  - It costs one probe intersection and one shadow ray per light, per landed ray.
- Later orders: the unchanged C3UR table operator. B = L₂ + … + L₅, read by the unchanged LOW + lookup renderer. No q table is used.

**State.**
- Persistent: the relation plus a geometry-dependent probe-ray buffer, GPU-resident and rebuilt on build / update.
- Nothing source-dependent is stored. Every `table()` call evaluates the lights it is given.
- A geometry stamp makes a stale probe buffer raise.

**Relation to WL90.**
- The class is a consolidation of the WL90 diagnostic functions.
- `tests91` requires equality with those functions and with the stored WL90 tables (≤ 1e-5).

## Prior evidence (disclosed) and what is new

**Inspected before this batch (WL90).**
- C3URX1 vs C3UR on W1 T0 / T1: the deployed draw and 16 realisations (R_mirror 0.024 vs 0.040 at T1).
- Component costs.
- W1 is therefore a **reproduction control**.

**Never computed before the freeze.**
- C3URX1 on W2 (T0 / T1).
- C3URX1 under LMIX89 / LSPOT89.
- Any pooled C3URX1.
- Any quality ladder or end-to-end lifecycle of C3URX1.

## Prospective protocol (`protocol/gm1_c3urx1_91.json`)

**Conditions.**

| Role | Condition |
|---|---|
| Reproduction | W1 T0 / T1 |
| Held-out edit | W2 T0 / T1 |
| Held-out sources | W1 T1 + LMIX89 (two coloured lights) and W1 T1 + LSPOT89 (cone) |

**Draws.** The WL89 deployed draw and realisations 89201–89216, paired across arms.

**Arms.**
- R0, C3UR, C3URX1: all 17 draws.
- R0X1: deployed draw only (limited diagnostic).
- C3UR_POOL / C3URX1_POOL: 16 draws pooled, to separate relation sampling from first-gather projection.
- ORACLE table.
- Frozen-ER path tracing (in the ladders).

**Fidelity.**
- E_H / signed error per region per draw against the WL89 REF-ER.
- Gates: F1, F2 (sharp regions gated at ≥ 1,000 pixels), F3 on both edits, F4 on the sources.
- Realisation pass fractions; paired C3URX1 − C3UR differences with 95 % intervals.
- Per-order field errors against path-traced order fields. New order fields are computed for the two source conditions.

**Held-out rules.**
- *advantage survives*, per held-out condition: the paired interval is < 0 in R_mirror. For LSPOT89, whose R_mirror mask is below 1,000 pixels, the largest gated sharp region is used.
- *no harm*: no region of any condition has an interval > 0, and no gate that C3UR passes is lost.
- HELD-OUT MECHANISM VALIDITY:
  - SUPPORTED: no harm, gates pass (fraction ≥ 0.9), W2 F3 not failing, advantage survives in ≥ 2 of 3.
  - CONDITIONAL: advantage survives in exactly 1.
  - NOT SUPPORTED: otherwise.

**End-to-end economics (alone on the GPU).**
- Every arm's complete pipeline is executed with one synchronised timer per event: geometry update, relation maintenance, source projection or exact first-gather evaluation, transport orders, transfers, table hand-over and a 64-spp frame.
- Also recorded: component timers, process CPU time and Dr.Jit kernel time.
- Events: 6 warm edits; 16 source updates (LMIX89, LSPOT89, 14 random lights); one executed amortisation sequence (edit + 4 sources × 4 views), compared with the prediction from the medians.
- Cold lifecycle: a fresh process per arm.
- Ray counts and memory, including scratch buffers.

**Quality ladders.**
- 3 configurations: W1 T1, W2 T1, W1 T1 + LMIX89.
- 3 draws for the table arms, plus path tracing.
- Tiers: stationary 0.10 / 0.05 / 0.03; R_mirror 0.10 / 0.06 / 0.04; C_ss 0.10 / 0.05.
- Time-to-tier = measured fixed lifecycle cost + render wall by log-log interpolation. A tier above the ladder floor is NOT REACHED.

**Utility rule.**
- (a) No slowdown > 1.25× against C3UR at any common tier.
- (b) A sharp-tier gain (reaching a tier C3UR does not, or ≥ 1.25× faster) in ≥ 2 of 3 configurations, with ≥ 2 of 3 draws each.
- (c) ≥ 1.25× against path tracing at every reached tier.
- SUPPORTED: a, b and c. CONDITIONAL: a and c with a partial b. A single favourable crossing is not a general speed-up.

**Scaling (bounded, cost only, warmed).**
- Patch resolution s 1 / 1.5 / 2; relation size κ_s 4 / 8 / 16; orders 2 / 4 / 8; source count 1 / 4 / 16.
- Two descriptive relation-size fidelity points (κ_s 4 / 16, single draw, never used in a verdict).

**Adoption evidence.** SUFFICIENT iff implementation QUALIFIED, W1 REPRODUCED, held-out validity SUPPORTED, source / edit validity SUPPORTED, lifecycle cost ≤ 1.25× C3UR, and utility not NOT SUPPORTED. This is an evidence statement; adoption is the owner's decision.

## Pre-freeze development (non-evidence; dev geometry Wk through the WL89 / WL90 dev roots)

- **Dev tests: all pass.**
  - Candidate vs WL90 reference functions: 7e-9.
  - Selective = full, including segments, probe buffer and table (3e-9).
  - Null edit and move-and-return identical.
  - The stale-geometry guard raises.
  - Two lights = sum of single lights.
  - Order fluxes: 0.993–0.999.
- **Dev chain end to end: completed** (tests, tables, renders at 64 spp, economics, analysis, visuals). Dev verdicts are meaningless at 64 spp and are not reported.
- **Timing-boundary finding (fixed before the freeze).**
  - The first executed amortisation sequence disagreed with its prediction (path tracing 48.9 s vs 3.7 s; C3URX1 1.93 s vs 1.08 s).
  - Cause, measured from the Dr.Jit kernel history and an IR diff: Dr.Jit emits allocation-history-dependent variants of the same program. The variants have the same operations in a different variable order (9,726 / 9,727 operations, different hash). A new `Lights` object, a second camera or a geometry update can produce a new variant.
  - Every new OptiX variant is compiled once: about 21 s for the path-tracing kernel, about 1 s for the table-lookup kernel. Identical later calls hit the cache (0.21 s).
  - Consequence for the protocol: warm event walls, ladder render walls and scaling timings exclude the compilation time measured in the kernel history, identically for all arms. The raw wall, the compilation time and the number of affected events are recorded. Cold lifecycles keep compilation.
  - The exclusion removes a cost that falls mostly on path tracing, so it is conservative for the candidate.
  - After the change the executed dev sequence equals its prediction: C3URX1 1.097 / 1.087 s, C3UR 1.145 / 1.150 s, R0 1.133 / 1.132 s, path tracing 3.745 / 3.712 s.
- **Disclosure: dev timings seen before the freeze** (dev geometry Wk, full patch count, 2 edits / 3 sources).
  - Edit to frame: C3URX1 124 ms, C3UR 136 ms, R0 152 ms, path tracing 248 ms.
  - Source update to frame: 76 / 93 / 85 / 227 ms.
  - The tiers, the utility rule and the lifecycle rule were written before these runs and were not changed after them.
- **Full-scale probe.** One full-scale `tabs` job (W1 T0, 17 draws × 3 arms, the reproduction control; outputs not inspected for fidelity, probe root deleted): 32 s wall, mean GPU utilisation 9 %, peak 3.6 GB. The stage is CPU-bound table assembly and short. The long stages (4,096-spp renders in a pool of 6, order fields, ladders) use the WL89 / WL90 render path at the scale already run there.

## Evidence run (frozen; no amendment)

- **Freeze commit** `7a8544e` (candidate, protocol v1, tests, pipelines, analysis, this worklog up to the previous section). Deployed with `git archive` (LF); the deployed protocol hash equals the committed one (sha256 1cf84cfd…).
- **Chain.** One detached chain on LabServer63, 2026-11-12 08:40:17–09:35:57 UTC (55.7 min). Every stage rc = 0. No stop branch, no rerun, no amendment.
- **GPU.** Mean utilisation 69 % over the chain (≈ 95 % in the render stage; the economics stages run alone by design). Peak 10.4 GB of 12 GB.
- **Tests (`tests91`, full scale): all pass.**

| Test | Result |
|---|---|
| Candidate = WL90 functions / stored WL90 tables | 7e-9 / 1.7e-7 (C3UR 2.4e-7) |
| Zero light, no visibility, finite, non-negative | 0, 0, true, min 0 |
| Current-source dependence | two lights = sum 5e-7; A after B = A 5e-9; A ≠ B |
| Relation source-independent | digest unchanged; build / update take no light |
| Replay | 7e-9 |
| Edit | selective = full (digest, segments, probe buffer; table 7e-9); null edit and move-and-return identical; fresh T1 scene 7e-7; stale-geometry guard raises |
| Order flux (orders 2–5) | 1.000 / 0.997 / 0.994 / 0.992 |

### Fidelity (E_H against the WL89 REF-ER; 16 paired relation draws)

**R_mirror, mean ± SD over the draws.**

| Condition | R0 | C3UR | C3URX1 | C3URX1 − C3UR [95 %] | C3URX1 better |
|---|---|---|---|---|---|
| W1 T0 (reproduction) | 0.054 ± 0.010 | 0.048 ± 0.007 | 0.027 ± 0.009 | −0.0205 [−0.0275, −0.0136] | 13 / 16 |
| W1 T1 (reproduction) | 0.047 ± 0.015 | 0.040 ± 0.005 | 0.024 ± 0.005 | −0.0160 [−0.0197, −0.0123] | 16 / 16 |
| W2 T0 | 0.035 ± 0.010 | 0.031 ± 0.005 | 0.018 ± 0.002 | −0.0133 [−0.0162, −0.0104] | 16 / 16 |
| W2 T1 (held-out edit) | 0.040 ± 0.011 | 0.037 ± 0.005 | 0.027 ± 0.007 | −0.0100 [−0.0153, −0.0046] | 12 / 16 |
| W1 T1 + LMIX89 (held-out source) | 0.019 ± 0.006 | 0.017 ± 0.003 | 0.018 ± 0.006 | +0.0010 [−0.0017, +0.0037] | 8 / 16 |
| W1 T1 + LSPOT89 (held-out source; 834 px, ungated) | 0.051 ± 0.022 | 0.054 ± 0.009 | 0.027 ± 0.005 | −0.0266 [−0.0320, −0.0211] | 16 / 16 |

**Deployed draw, pooled draws and oracle (R_mirror).**

| Condition | C3UR deployed | C3URX1 deployed | C3UR pooled | C3URX1 pooled | ORACLE |
|---|---|---|---|---|---|
| W1 T0 | 0.047 | 0.023 | 0.047 | 0.013 | 0.014 |
| W1 T1 | 0.042 | 0.027 | 0.038 | 0.015 | 0.014 |
| W2 T0 | 0.032 | 0.017 | 0.030 | 0.011 | 0.010 |
| W2 T1 | 0.038 | **0.053** | 0.036 | 0.013 | 0.016 |
| LMIX89 | 0.023 | 0.019 | 0.012 | 0.013 | 0.010 |
| LSPOT89 | 0.038 | 0.030 | 0.053 | 0.022 | 0.023 |

- The W2 T1 deployed draw of C3URX1 (0.053, signed +0.044) is worse than its C3UR counterpart and lies above the largest of the 16 realisations (0.039).

**Decomposition of the R_mirror error (E_H differences).**

| Condition | C3URX1 sampling (draw mean − pooled) | C3UR sampling | First-gather projection (C3UR pooled − C3URX1 pooled) | Remaining (C3URX1 pooled − ORACLE) |
|---|---|---|---|---|
| W1 T0 | 0.0139 | 0.0009 | 0.0335 | −0.0006 |
| W1 T1 | 0.0091 | 0.0014 | 0.0236 | +0.0009 |
| W2 T0 | 0.0077 | 0.0015 | 0.0195 | +0.0006 |
| W2 T1 | 0.0140 | 0.0015 | 0.0224 | −0.0025 |
| LMIX89 | 0.0053 | 0.0052 | −0.0009 | +0.0035 |
| LSPOT89 | 0.0049 | 0.0005 | 0.0309 | −0.0006 |

**Other regions.**
- stationary_all: C3URX1 − C3UR is −0.0001 to −0.0007 in all six conditions, every interval below 0.
- C_ss: the paired difference is within ±0.0004 in five conditions. Under LSPOT89 it is **+0.0025 [+0.0005, +0.0045]** (C3URX1 better in 4 / 16).
  - LSPOT89 C_ss: C3UR 0.0162 ± 0.0013, C3URX1 0.0188 ± 0.0043 (largest 0.030); deployed 0.016 vs 0.028; pooled 0.0147 vs 0.0134; ORACLE 0.0140.
- LMIX89 R_mirror signed error: C3URX1 +0.0096 [+0.0046, +0.0146] over the draws and +0.0096 pooled; C3UR +0.0003; ORACLE 0.0000.

**Gates F1 / F2 / F4.** C3URX1 passes every applicable gate in every condition on all 16 draws and on the deployed draw (pass fraction 1.0). C3UR likewise. R0 passes F2 R_mirror at W1 T1 on 13 / 16.

**F3, edit response (slope / correlation; realisation pass counts of 16).**

| Edit | Region | R0 | C3UR | C3URX1 |
|---|---|---|---|---|
| W1 | D | 1.06 / 0.97 PASS; 16 | 0.97 / 0.98 PASS; 16 | 0.90 / 0.97 PASS; 16 |
| W1 | C_ss | 0.77 / 0.95 PASS; 8 | 0.87 / 0.96 PASS; 14 | 0.92 / 0.97 PASS; 14 |
| W1 | R_mirror | 1.12 / 0.97 PASS; 9 | 0.86 / 0.98 PASS; 16 | 0.88 / 0.98 PASS; 13 |
| W2 | D | 0.96 / 0.93 PASS; 16 | 0.99 / 0.95 PASS; 16 | 0.91 / 0.94 PASS; 16 |
| W2 | C_ss | 0.67 / 0.88 **FAIL**; 11 | 0.79 / 0.93 PASS; 12 | 0.65 / 0.83 **FAIL**; 13 |
| W2 | R_mirror | INDETERMINATE; 0 | INDETERMINATE; 0 | INDETERMINATE; 0 |

- W2 R_mirror is not resolvable against the reference noise for any arm (16 / 16 indeterminate).

**Limited and descriptive items (never used in a verdict).**
- R0X1, deployed draw only, R_mirror against R0: W1 T0 0.024 vs 0.051; W1 T1 0.023 vs 0.060; W2 T0 0.039 vs 0.047; W2 T1 0.042 vs 0.051; LMIX89 0.016 vs 0.017; LSPOT89 **0.094 vs 0.037**. One draw; WL90 measured no mean gain over 16 draws (+0.003 [−0.006, +0.012]).
- Relation size at W1 T1, single draws, R_mirror: C3URX1 κ_s 4 / 8 / 16 = 0.022 / 0.027 / 0.021; C3UR 0.023 / 0.042 / 0.038.
- Per-order table fields against path-traced order fields (deployed draw, first table order, all patches): C3UR / C3URX1 0.116 / 0.116, 0.123 / 0.126, 0.113 / 0.114, 0.102 / 0.103 and 0.215 / 0.219 (W1 T0, W1 T1, W2 T1, LMIX89, LSPOT89). LSPOT89 floor patches: 0.264 / 0.307.

### Held-out rules (prospective)

| Rule | Outcome |
|---|---|
| Advantage survives, W2 T1 (R_mirror) | **yes** (−0.0100 [−0.0153, −0.0046]) |
| Advantage survives, LMIX89 (R_mirror) | **no** (interval contains 0) |
| Advantage survives, LSPOT89 (rule region C_ss; R_mirror has 834 < 1,000 px) | **no** (+0.0025 [+0.0005, +0.0045]) |
| No harm | **violated**: LSPOT89 C_ss interval above 0 |
| Held-out gates pass (fraction ≥ 0.9) | yes (1.0 / 1.0 / 1.0) |
| W2 F3 not failing (deployed) | **violated**: C_ss FAIL |

- Survives in 1 of 3. HELD-OUT MECHANISM VALIDITY = **NOT SUPPORTED**.
- With the ungated LSPOT89 R_mirror instead of the rule region the count would be 2 of 3; the harm and the W2 F3 conditions would still fail, so the verdict does not depend on that region choice.

### End-to-end economics (alone on the GPU; warm walls exclude kernel compilation)

**Warm lifecycle, medians (7 edits, 15 single-light sources, 4 frames; 512², 64-spp frame).**

| | C3URX1 | C3UR | R0 | Path tracing |
|---|---|---|---|---|
| Edit to frame, ms | 124.7 | 138.2 | 145.4 | 250.1 |
| Edit without render, ms | 69.4 | 83.0 | 89.8 | 30.9 |
| Source update to frame, ms | 75.8 | 92.2 | 85.4 | 213.1 |
| Source update without render, ms | 21.4 | 37.4 | 31.0 | 0.0 |
| Two-light source without render, ms | 27.5 | 52.6 | 46.6 | 0.0 |
| Frame, ms | 55.9 | 54.6 | 54.0 | 215.7 |
| Executed sequence (1 edit + 4 sources × 4 views): measured / predicted, s | 1.10 / 1.10 | 1.16 / 1.16 | 1.15 / 1.13 | 3.75 / 3.69 |
| Same sequence raw (compilation inside), s | 1.98 | 1.16 | 1.15 | 48.97 |
| Edit CPU ms / Dr.Jit kernel ms | 129 / 22 | 143 / 24 | 148 / 24 | 254 / 113 |

**Edit components, ms.**

| Component | C3URX1 | C3UR | R0 |
|---|---|---|---|
| Geometry update | 30.8 | 30.4 | 31.2 |
| Relation update (46,923 rays, 7.2 %; R0 46,319) | 14.5 | 14.6 | 19.9 + 6.0 coefficients |
| Probe buffer | 1.3 | n/a | n/a |
| Source: exact first-gather evaluation / q table | 11.5 | 26.3 | 26.6 |
| Transport orders | 10.6 | 11.4 | 5.6 |
| Table hand-over | 0.2 | 0.2 | 0.2 |
| Render | 55.1 | 55.2 | 55.3 |

- Component sums equal the event wall (ratio 1.00).
- Events with compilation among the timed events: one C3URX1 frame (0.73 s, excluded); none for the other arms. Inside the executed sequence: C3URX1 0.88 s, path tracing 45.2 s.
- Lifecycle ratios C3URX1 / C3UR: edit 0.90, edit without render 0.84, source 0.82, source without render 0.57.
- End-to-end validity on the executed pipeline: selective = full after the edit sequence; relation digest constant across the 16 source updates.

**Rays and memory.**
- Source update, per light: C3URX1 526,467 probe intersections + 526,467 shadow rays; C3UR / R0 20,320 q-table shadow rays.
- Persistent state: C3URX1 relation 20.8 MB + probe buffer 12.6 MB; C3UR relation 20.8 MB + q table 1.0 MB; R0 state 17.6 MB + W 3.9 MB + q table 1.0 MB.
- Scratch: C3URX1 radiance 6.3 MB; incident cells 62.4 MB (both sender-relation arms).
- Process GPU memory: 3.33 / 3.41 / 3.27 / 3.16 GB (C3URX1 / C3UR / R0 / path tracing). torch peak 289 / 253 / 145 / 70 MiB.

**Cold lifecycle (fresh process, compilation included).**

| | C3URX1 | C3UR | R0 | Path tracing |
|---|---|---|---|---|
| Scene + material table + relation build, s | 0.23 + 0.62 + 0.42 | 0.21 + 0.62 + 0.35 | 0.24 + 1.32 + 0.28 | 0.22 |
| First source + frame, s (compilation) | 1.11 (0.87) | 1.11 (0.83) | 1.07 (0.84) | 20.72 (20.48) |
| Total to first frame, s | 2.38 | 2.28 | 2.90 | 20.94 |
| First edit + frame, s (compilation) | 1.20 (0.96) | 1.14 (0.88) | 1.12 (0.85) | 20.88 (20.60) |

**Time to tier, edit to frame, s (deployed draw / the two other draws; NR = tier not reached within the ladder).**

| Configuration | Tier | C3URX1 | C3UR | R0 | Path tracing |
|---|---|---|---|---|---|
| W1 T1 | R_mirror 0.10 | 0.12 / 0.12 / 0.12 | 0.14 / 0.14 / 0.14 | 0.16 / 0.16 / 0.17 | 2.44 |
| W1 T1 | R_mirror 0.06 | 0.23 / 0.23 / 0.24 | 0.41 / 0.45 / 0.46 | NR | 7.28 |
| W1 T1 | R_mirror 0.04 | 0.67 / 0.60 / 0.67 | NR | NR | 17.57 |
| W2 T1 | R_mirror 0.10 | 0.12 / 0.12 / 0.12 | 0.13 / 0.13 / 0.13 | 0.14 / 0.14 / 0.14 | 1.61 |
| W2 T1 | R_mirror 0.06 | **0.74** / 0.23 / 0.18 | 0.24 / 0.30 / 0.23 | 0.44 / NR / 0.41 | 5.58 |
| W2 T1 | R_mirror 0.04 | **NR** / 0.81 / 0.42 | 1.45 / NR / 0.90 | NR | 15.65 |
| LMIX89 | R_mirror 0.06 | 0.25 / 0.23 / 0.24 | 0.27 / 0.25 / 0.27 | 0.26 / 0.25 / 0.29 | 3.41 |
| LMIX89 | R_mirror 0.04 | 0.55 / 0.48 / 0.52 | 0.65 / 0.51 / 0.68 | 0.53 / 0.53 / 0.90 | 7.74 |
| W1 T1 | stationary 0.03 | 0.31 / 0.31 / 0.31 | 0.32 / 0.32 / 0.33 | 0.33 / 0.33 / 0.33 | 1.68 |
| W1 T1 | C_ss 0.05 | 0.18 / 0.18 / 0.18 | 0.19 / 0.19 / 0.19 | 0.20 / 0.20 / 0.20 | 1.17 |

**Utility rule.**
- (a) No slowdown > 1.25× against C3UR at a common tier: **violated**. W2 T1 deployed draw, R_mirror 0.06: 0.74 s vs 0.24 s (3.1×). The same draw does not reach R_mirror 0.04, which its C3UR counterpart reaches.
- (b) Sharp-tier gain in ≥ 2 of 3 configurations with ≥ 2 of 3 draws: **met**. W1 T1 3 / 3 draws, W2 T1 2 / 3, LMIX89 1 / 3.
- (c) ≥ 1.25× against path tracing at every reached tier: **met**. 1.8–5.5× on the broad tiers, 7.6–31× on the sharp tiers.
- QUALITY-MATCHED ECONOMIC UTILITY = **NOT SUPPORTED** (rule a).
- LIFECYCLE COST (≤ 1.25× C3UR) = **SUPPORTED** (0.57–0.90×).

**Bounded scaling (cost only, warmed).**

| Case | Unique / landed rays | Relation build, ms | Exact source evaluation, ms | First gather, ms | C3URX1 table (4 orders), ms | C3UR q table + table, ms | Update, ms | Relation + probe buffer, MB |
|---|---|---|---|---|---|---|---|---|
| s 1.0, κ_s 8 (N 1,270) | 0.65 M / 0.53 M | 52 | 13.2 | 2.7 | 21.0 | 24.9 + 11.2 | 14.9 | 20.8 + 12.6 |
| s 1.5 (N 2,802) | 1.43 M / 1.16 M | 125 | 14.0 | 3.4 | 24.2 | 30.3 + 13.6 | 18.5 | 45.9 + 27.9 |
| s 2.0 (N 5,080) | 2.60 M / 2.11 M | 229 | 12.1 | 4.5 | 27.8 | 41.5 + 18.9 | 24.7 | 83.2 + 50.6 |
| κ_s 4 | 0.33 M / 0.26 M | 34 | 11.1 | 2.6 | 20.7 | 25.1 + 10.9 | 13.8 | 10.4 + 6.3 |
| κ_s 16 | 1.30 M / 1.05 M | 110 | 11.4 | 2.8 | 21.6 | 25.4 + 10.4 | 18.4 | 41.6 + 25.3 |

- Transport orders 2 / 4 / 8 at the base size: C3URX1 15.7 / 21.0 / 29.0 ms; C3UR table 5.9 / 11.2 / 19.2 ms plus the 24.9 ms q table.
- Source count 1 / 4 / 16: C3URX1 25 / 78 / 314 ms; C3UR 35 / 132 / 517 ms.
- The exact source evaluation stays at 11–14 ms from 0.26 M to 2.1 M landed rays: at these sizes it is not bound by the ray count.

### Verdicts (predeclared mapping; no amendment)

| Item | Verdict |
|---|---|
| CANDIDATE IMPLEMENTATION | **QUALIFIED** (all tests pass) |
| W1 REPRODUCTION | **REPRODUCED** (W1 T1 R_mirror −0.0160 [−0.0197, −0.0123]; equality with WL90) |
| HELD-OUT MECHANISM VALIDITY | **NOT SUPPORTED** (survives 1 of 3; harm in LSPOT89 C_ss; W2 deployed C_ss F3 FAIL) |
| SOURCE / EDIT VALIDITY | **SUPPORTED** |
| LIFECYCLE COST | **SUPPORTED** (0.57–0.90× C3UR) |
| QUALITY-MATCHED ECONOMIC UTILITY | **NOT SUPPORTED** (rule a: 3.1× slowdown on the W2 T1 deployed draw) |
| BASELINE ADOPTION EVIDENCE | **INSUFFICIENT** (failing: held-out mechanism validity, quality-matched utility) |
| FINAL NOVELTY EVIDENCE | INSUFFICIENT by scope (not evaluated) |

- C3URX1 is not adopted. C3UR remains the preserved explicit development baseline. The WL89 and WL90 verdicts are unchanged.

## 최종 보고

### Implementation Fact

- **후보.** `cand91.C3URX1`은 WL89 C3UR relation을 그대로 쓰고, first gather만 relation ray별 exact once-scattered radiance(현재 광원, ray당 probe 교차 1회 + 광원당 shadow ray 1회)로 읽습니다. 이후 order는 C3UR table operator입니다.
- **상태.** relation과 geometry 의존 probe buffer만 유지합니다. source 의존 상태는 저장하지 않습니다. geometry stamp가 낡은 probe buffer 사용을 막습니다.
- **고정.** κ_s 8, 64 bin, ER, 임계값, 광원·geometry 정의, draw seed는 WL89 / WL90 값 그대로입니다. 조정한 것은 없습니다.
- **freeze.** `7a8544e`에서 후보·프로토콜·판정 규칙을 고정한 뒤 held-out 실험을 실행했습니다. amendment는 없습니다.
- **타이밍 경계(freeze 전에 고정).** Dr.Jit이 변수 할당 이력에 따라 같은 프로그램의 kernel 변형을 만들고, 새 OptiX 변형마다 한 번 컴파일합니다(path tracing kernel 약 21 s, table-lookup kernel 약 1 s). warm 시간은 kernel history로 측정한 컴파일 시간을 모든 arm에 동일하게 제외하고, raw 시간을 함께 기록했습니다. cold lifecycle은 컴파일을 포함합니다.
- **기존 실험.** WL86–WL90 코드·결과·판정은 수정하지 않았습니다.

### Measurement

**정확성 테스트.** 전부 통과했습니다. WL90 함수·저장 table과 동일(7e-9 / 1.7e-7), zero / no-visibility 0, 유한·비음수, 두 광원 = 합(5e-7), relation의 source 독립, selective = full, null edit·move-and-return 동일, replay 7e-9.

**R_mirror E_H (16개 paired draw, 평균 ± SD).**

| 조건 | C3UR | C3URX1 | C3URX1 − C3UR [95 %] | pooled C3UR / C3URX1 / ORACLE |
|---|---|---|---|---|
| W1 T0 (재현) | 0.048 ± 0.007 | 0.027 ± 0.009 | −0.0205 [−0.0275, −0.0136] | 0.047 / 0.013 / 0.014 |
| W1 T1 (재현) | 0.040 ± 0.005 | 0.024 ± 0.005 | −0.0160 [−0.0197, −0.0123] | 0.038 / 0.015 / 0.014 |
| W2 T0 | 0.031 ± 0.005 | 0.018 ± 0.002 | −0.0133 [−0.0162, −0.0104] | 0.030 / 0.011 / 0.010 |
| W2 T1 (held-out edit) | 0.037 ± 0.005 | 0.027 ± 0.007 | −0.0100 [−0.0153, −0.0046] | 0.036 / 0.013 / 0.016 |
| LMIX89 (held-out source) | 0.017 ± 0.003 | 0.018 ± 0.006 | +0.0010 [−0.0017, +0.0037] | 0.012 / 0.013 / 0.010 |
| LSPOT89 (held-out source, 834 px) | 0.054 ± 0.009 | 0.027 ± 0.005 | −0.0266 [−0.0320, −0.0211] | 0.053 / 0.022 / 0.023 |

- **배포 draw.** W2 T1에서 C3URX1 0.053(signed +0.044)으로 C3UR 0.038보다 나쁩니다. 16개 realization의 최댓값(0.039)보다도 큽니다.
- **C_ss.** 다섯 조건에서 차이는 ±0.0004 이내입니다. LSPOT89에서는 +0.0025 [+0.0005, +0.0045]로 C3URX1이 나쁩니다(C3UR 0.0162 ± 0.0013, C3URX1 0.0188 ± 0.0043, 배포 draw 0.016 vs 0.028, pooled 0.0147 vs 0.0134).
- **gate.** F1 / F2 / F4는 C3URX1이 모든 조건, 모든 draw에서 통과했습니다.
- **F3 (edit 응답, 배포 draw).** W1은 세 region 모두 통과. W2는 D 통과, C_ss **FAIL**(slope 0.65, corr 0.83; C3UR 0.79 / 0.93 통과, R0 0.67 / 0.88 FAIL), R_mirror는 모든 arm에서 INDETERMINATE입니다. W2 C_ss realization 통과 수는 C3URX1 13, C3UR 12, R0 11(각 16 중)입니다.

**end-to-end 비용 (warm 중앙값, 512², 64 spp frame).**

| | C3URX1 | C3UR | R0 | path tracing |
|---|---|---|---|---|
| edit → frame, ms | 124.7 | 138.2 | 145.4 | 250.1 |
| edit (렌더 제외), ms | 69.4 | 83.0 | 89.8 | 30.9 |
| source → frame, ms | 75.8 | 92.2 | 85.4 | 213.1 |
| source (렌더 제외), ms | 21.4 | 37.4 | 31.0 | 0.0 |
| 실행 시퀀스 (edit 1 + source 4 × view 4): 실측 / 예측, s | 1.10 / 1.10 | 1.16 / 1.16 | 1.15 / 1.13 | 3.75 / 3.69 |
| cold, 첫 frame까지, s | 2.38 | 2.28 | 2.90 | 20.94 |

- source 단계: exact first-gather 평가 11.5 ms vs q table 26.3 ms. ray 수는 광원당 probe 526,467 + shadow 526,467 대 q table shadow 20,320입니다.
- 메모리: C3URX1은 probe buffer 12.6 MB와 scratch radiance 6.3 MB가 추가됩니다. 프로세스 GPU 메모리는 3.33 vs 3.41 GB입니다.
- 실행 파이프라인에서 selective = full, source 16회 동안 relation digest 불변을 확인했습니다.

**time-to-tier (edit → frame, s; 배포 draw / 나머지 두 draw).**

| 구성 | tier | C3URX1 | C3UR | path tracing |
|---|---|---|---|---|
| W1 T1 | R_mirror 0.06 | 0.23 / 0.23 / 0.24 | 0.41 / 0.45 / 0.46 | 7.28 |
| W1 T1 | R_mirror 0.04 | 0.67 / 0.60 / 0.67 | 미도달 | 17.57 |
| W2 T1 | R_mirror 0.06 | **0.74** / 0.23 / 0.18 | 0.24 / 0.30 / 0.23 | 5.58 |
| W2 T1 | R_mirror 0.04 | **미도달** / 0.81 / 0.42 | 1.45 / 미도달 / 0.90 | 15.65 |
| LMIX89 | R_mirror 0.04 | 0.55 / 0.48 / 0.52 | 0.65 / 0.51 / 0.68 | 7.74 |
| W1 T1 | stationary 0.03 | 0.31 / 0.31 / 0.31 | 0.32 / 0.32 / 0.33 | 1.68 |

**scaling (비용만).** patch 해상도 s 1 → 2에서 C3URX1 table 21.0 → 27.8 ms, update 14.9 → 24.7 ms, relation + probe buffer 33.4 → 133.8 MB입니다. source 16개: 314 ms vs C3UR 517 ms. exact source 평가는 landed ray 0.26 M–2.1 M 범위에서 11–14 ms로 거의 일정합니다.

### Observation

- **편향 제거는 gap이 있는 조건 전부에서 재현됩니다.** pooled C3URX1은 여섯 조건 모두에서 ORACLE과 −0.0025 ~ +0.0035 이내입니다. first-gather 사영 항은 W1, W2, LSPOT89에서 0.020–0.034입니다.
- **LMIX89에는 제거할 gap이 없습니다.** 사영 항이 −0.0009이고, pooled C3UR(0.012)이 이미 ORACLE(0.010)에 가깝습니다.
- **단일 draw에서는 표본 항이 커집니다.** R_mirror의 (draw 평균 − pooled) 항은 C3URX1 0.005–0.014, C3UR 0.0005–0.005입니다. 여섯 조건 모두에서 C3URX1 쪽이 큽니다.
- **draw 이상치가 판정을 갈랐습니다.** W2 T1 배포 draw(R_mirror +0.044 signed)와 LSPOT89 C_ss(배포 draw 0.028, 최대 0.030)가 그 사례입니다. realization 평균의 signed 오차는 0 근처입니다(W2 T1 +0.0024 [−0.0051, +0.0098]).
- **LMIX89 R_mirror에는 작은 양의 signed 잔여가 있습니다.** C3URX1 +0.0096 [+0.0046, +0.0146], pooled에서도 +0.0096입니다. C3UR과 ORACLE은 0 근처입니다.
- **비용은 모든 측정에서 C3UR 이하입니다.** edit 0.90×, source 0.82×, 렌더 제외 source 0.57×입니다. 넓은 region tier에서는 고정비 차이만큼(약 1.05–1.1×) 빠릅니다.
- **path tracing 대비.** 넓은 tier 1.8–5.5×, sharp tier 7.6–31×입니다. 컴파일을 포함한 raw 시퀀스는 path tracing 48.97 s, C3URX1 1.98 s입니다.

### Interpretation

- exact first gather는 WL90에서 분리한 방향 사영 편향을 held-out edit와 cone 광원에서도 제거합니다. 기대값 수준의 메커니즘은 측정 범위에서 유지됩니다.
- 대신 κ_s 8의 단일 relation draw에서는 ray별 point 표본이 그대로 드러나 draw 간 변동이 남고, 일부 draw는 C3UR보다 나쁩니다. q table은 patch × bin 평균이라 이 변동을 편향으로 바꿔 숨깁니다. 이것은 측정된 분해(표본 항, pooled 값)와 일치하는 해석이며, 변동의 원인 ray를 직접 분리해 측정하지는 않았습니다.
- 경제성 판정이 NOT SUPPORTED인 이유는 비용이 아니라 품질의 draw 의존성입니다. lifecycle 비용 규칙은 통과했고, 실패한 것은 "어떤 공통 tier에서도 C3UR 대비 1.25× 초과 감속 없음" 규칙의 한 draw입니다.
- 사전 규칙이 배포 draw 한 개에 민감하다는 점은 사실이지만, 규칙은 증거 전에 고정했고 변경하지 않았습니다. 판정은 규칙대로입니다.

### Unresolved Question

- C3URX1의 draw 간 변동과 이상치가 relation 크기(κ_s)에 따라 어떻게 줄어드는지. 이번에는 κ_s 4 / 16의 단일 draw 서술 점만 있고(0.022 / 0.021), 조정은 금지 범위였습니다.
- 이상치 draw의 원인이 되는 ray(광원 근접, glossy lobe 정점 등)가 무엇인지. 분리 측정하지 않았습니다.
- LMIX89 R_mirror의 양의 signed 잔여(+0.0096)의 원인. 귀속하지 않았습니다.
- W2 R_mirror edit 응답은 reference noise 때문에 어떤 arm에서도 판정할 수 없습니다.
- area light, 더 sharp하거나 덜 sharp한 sender, 다른 scene family, 외부 learned baseline(Neural Full Recompute)은 측정하지 않았습니다.

### Architecture Verdict

범위: WL89 workload family(W1 = V3, held-out edit W2, held-out source LMIX89 / LSPOT89), frozen ER, 점광원·cone 광원, 512², patch × 64-bin table, κ_s 8.

| 구분 | 판정 |
|---|---|
| 진단 재현 (diagnostic success) | **REPRODUCED**: W1 T1 R_mirror −0.0160 [−0.0197, −0.0123], WL90 table과 동일 |
| 구현 정확성 | **QUALIFIED**: 모든 테스트 통과 |
| source / edit 유효성 | **SUPPORTED**: relation source 독립, 현재 광원 의존, selective = full |
| held-out mechanism validity | **NOT SUPPORTED**: 우위 유지 1 / 3, LSPOT89 C_ss 악화, W2 배포 draw C_ss F3 FAIL |
| lifecycle 비용 | **SUPPORTED**: C3UR의 0.57–0.90× |
| quality-matched economic utility | **NOT SUPPORTED**: 규칙 (a) 위반(W2 T1 배포 draw 3.1× 감속). (b), (c)는 충족 |
| baseline 채택 증거 | **INSUFFICIENT**: held-out validity와 utility 미충족 |
| 최종 ReST-GI novelty 증거 | 범위상 **INSUFFICIENT**(평가하지 않음: 외부 learned baseline 미측정, scene family 1개) |

**Completion answer.**
- C3URX1은 측정 범위에서 source-independent하고 물리적으로 유효합니다(테스트, gate 전부 통과).
- 경제적으로는 lifecycle 비용이 C3UR보다 낮지만, 품질이 relation draw에 의존해 사전 규칙의 quality-matched 유용성을 충족하지 못했습니다.
- held-out 조건에서 우위는 W2 T1에서만 규칙상 유지됐습니다.
- 따라서 explicit development baseline으로 채택하기에 증거는 **충분하지 않습니다**. C3URX1은 채택하지 않았고, C3UR이 보존된 baseline입니다.

### Reproducibility

- **Commits.** freeze `7a8544e`; results in the closing commit. Branch `Selective_Recompute`. Parents: WL89 `7515a29` / `5d6bb5c` / `0d646a9`, WL90 `51b93b8` / `2448b35`.
- **Environment.** LabServer63, RTX 3080 Ti 12 GB, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Protocol.** `experiments/gm1_c3urx1_91/protocol/gm1_c3urx1_91.json` (v1, no amendment; sha256 1cf84cfd…).
- **Seeds.** Relation draws: the WL89 deployed draw (86001) and realisations 89201–89216. Renders 89600 / 89601. New order fields 91460. Lifecycle events 918xx, cold 919xx, sources 91720, ladders 91600.
- **Reused.** From `~/rest_rel89`: W1 choice (sha256 213a3ccc…), REF-ER, masks, oracle tables, order fields. From `~/rest_gap90`: stored WL90 tables for the equality test.
- **Server.** `~/rest_x191` (tables 99 MB, renders 1.6 GB, fields, logs, GPU trace).
- **Local.**
  - `results/gm1_c3urx1_91/`: chain status, GPU trace, tests91, final91, econ (life, cold × 4, ladder, scale), table metadata, render lists, logs.
  - `results/evaluation/91/`: eight figures plus a manifest with sha256.
- **Invalid or excluded.** None in the evidence run. Kernel compilation is excluded from warm walls by the frozen protocol and reported raw alongside.
- **Non-evidence.** The dev chain and kernel-variant probes on the dev geometry (`~/rest_x191dev`), and the full-scale `tabs` probe.
