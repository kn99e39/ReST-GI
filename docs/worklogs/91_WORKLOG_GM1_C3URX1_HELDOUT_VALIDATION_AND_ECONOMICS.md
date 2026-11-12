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
