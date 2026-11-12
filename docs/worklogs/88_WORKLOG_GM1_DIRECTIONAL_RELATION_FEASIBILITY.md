# Worklog — GM1 Source-independent Directional Transport Relation Feasibility (2026-11-12)

## Origin and scope

- **Brief.** "WL88 — Source-independent Directional Transport Relation Feasibility" (user, 2026-11-12). One bounded GM1 architecture-discrimination batch: transport representation and relation construction, no learned global operator.
- **Central hypothesis (falsifiable).** A reusable geometry/material-dependent directional transport relation can capture the sharp path families missed by the deployed receiver-side sampling state, without exhaustive or source-dependent transport reconstruction.
- **Questions.**
  - Q1 information: does the relation carry the sharp contributions?
  - Q2 ownership and reuse: is it source-independent, reusable across sources, and does it update correctly after geometry edits?
  - Q3 economics: is construction/update or repeated application the dominant cost?
- **Frozen.**
  - WL84 ER closures.
  - The WL86 code (`experiments/ecro86`) and the WL87 code (`experiments/gm1_explicit87`), both imported unchanged.
  - WL86 / WL87 references, tables, thresholds and verdicts; ECRO checkpoints (not used here).
- **Not done.**
  - No learned operator, no H-P / subspace factorisation, no GM2.
  - No change to WL87 code or results.
  - No new branch or worktree.
- **Where.**
  - Branch `Selective_Recompute`, LabServer63 (main server).
  - `experiments/gm1_relation88/`, `results/gm1_relation88/`, `results/evaluation/88/`.
  - Server `~/rest_rel88`, which reads `~/rest_ecro86` (E2 references / masks) and `~/rest_gm187` (WL87 tables, integration test).
- **Terminology.** All comparators are physical recomputations with the same frozen ER. No external neural baseline is run and no Neural Full Recompute / Native Reuse claim is made.

## A. The WL87 mechanism, reconstructed from the code

The baseline R0 is WL87 EXP (`xport87.py`, `quad87.py`).

1. **Relation discovery.**
   - Per patch, 512 rays: 8 per incident bin, uniform inside the bin.
   - One offset draw is shared by every patch: the same local directions, with different origin points.
   - Each ray stores its front-face hit patch, the outgoing bin at the hit and its segment.
   - Built from geometry only (`pt86.Ctx.state_rays`); selective update re-traces rays whose segment meets the old / new moved volume or that start on the mover.
2. **Scattering coefficients.**
   - W_ray = fbar(ω_r)[b_o] · cos ω_r · ΔΩ / 8 at each ray's exact local direction (65,536 MIS samples; Lambert exact).
   - Sphere rows are per band, at the ray's origin frame.
   - Material-only: one set per realisation, geometry-independent.
3. **Source projection.** q = L₁ = the once-scattered table of the actual lights (`quad87.qtable`; 16 points × 1,024 MIS per patch). It is the only source-dependent quantity.
4. **Repeated transport.** L_{k+1} = gather over the 512 rays of L_k[hit, dbin] × W. Four orders; B = L₂…L₅.
5. **Camera readout.** WL86 LOW (e1, e2 path-traced) plus B[patch(x₂), bin(x₂ → x₁)], i.e. 3–6 events.

**Information discarded.**
- Incident radiance is sampled at 512 directions per patch (all patches share the same 512 local directions).
- Senders are represented as patch × 64-bin piecewise-constant tables (a sharp lobe is averaged over a bin of 0.098 sr; S1 has α ≈ 0.05).
- Receivers are represented patch-constant.

**Shared across sources and edits.**
- Relation and W are reused across sources. Only q and the recurrence are recomputed.
- After an edit: dirty rays plus a new q and recurrence.

**Where C3 differs from EXP (`xport87.sender_side`).** Seven variables change at once:

| | Variable | EXP | C3 |
|---|---|---|---|
| (a) | ray placement | receiver side | sender side |
| (b) | allocation | uniform | proportional to the current source power per (patch, bin) — source-dependent |
| (c) | relation per order | one relation for all orders | a fresh relation for each of the four orders |
| (d) | total rays | 650,240 | 4 × 650,240 |
| (e) | receiver scattering | per-ray W | the WL87 dense table, 8 × 8 cells per bin, cell centre |
| (f) | sphere receivers | curved rows | flat frame |
| (g) | accumulation | — | numpy float64 |

- WL87 matched only the per-order ray count.
- WL88 isolates these variables with controls on the historical E2 case and the new workload (Task A factors below).
- Under reciprocity, receiver gather and sender deposition estimate the **same** discrete operator for the same piecewise-constant table. This is verified (target identity, below), so any difference between them is estimator variance or allocation, not a different physical target.

## B. Workload (frozen in `protocol/gm1_relation88.json`)

- **Family.**
  - The WL86 room, with box A turned into a moving glossy mover G carrying the frozen S1 ER state (a bit-identical copy; the physical P reference uses S1's layered descriptor).
  - G is a thin slab (half sizes 0.05 / 0.42 / 0.38 m) that faces the stationary S1 panel.
  - Box B (Lambert), sphere S2 and floor B are unchanged. No material outside the frozen ER set is used.
- **What it exposes.**
  - Sharp-to-sharp transport between two S1 surfaces (panel ↔ G), with G moving.
  - The moving sharp surface changes the table entries of stationary surfaces lit through it.
  - The panel is a sharp reader of those entries.
- **Regions** (`masks88.py`, geometry / visibility only):
  - C_ss: stationary broad receivers in a two-specular-reflection (S1 → S1) footprint that exists in exactly one of T0 / T1. Their 3-event camera paths read the table at the S1 vertex.
  - R_mirror: stationary panel pixels whose mirror cone sees G, or a point whose G-involving footprint changes.
  - A / D / E: as in WL86.
  - Footprints come from specular "mirror photons" from the light (16 M per configuration, 2 cm voxels, 2-voxel dilation). No BSDF value or radiance is used.
- **Candidates.**
  - Three W1 candidates in a fixed order: Wc, Wh, Wi.
  - Held-out edit W2 = Wj (translation + rotation, its own light).
  - Held-out sources on W1 T1: LMIX88 (two coloured lights) and LSPOT88 (a cone aimed at the panel / G region).
  - Development geometry: Wk.
- **Geometry-only screening during design.** Pixel counts only:
  - Wc: A 4,216 / C_ss 5,184 / R_mirror 2,864 / D 32,296 / E 39,580.
  - Wh and Wi: also above 1,000 in every rule region.
  - Wj (W2): C_ss 3,512 / R_mirror 3,308.
- **Radiance seen during design.** One 64-spp, 256² reference look render of Wc (T0 / T1) was made to confirm the scene is lit and the slab is visible (dev seed 88990; reference estimator only). No relation or method output was ever computed on Wc, Wh, Wi or Wj before the freeze.
- **Qualification (prospective, in the chain).**
  - Uses an independent qualification partition: 4 × 8,192 spp REF-P and family-tagged REF-ER per T, seeds 88140+ / 88150+. These are never used for final evaluation.
  - Criteria (Q1–Q8):
    - WL86 A / Bset / D / H-share / uncertainty;
    - C_ss: the ≥ 2-S1-vertex family H2 carries ≥ 25 % of the absolute change of the higher-order radiance, above 4 × noise;
    - R_mirror: G-involving families carry ≥ 25 % of the change;
    - the projected final REF-ER noise of h in the sharp regions is ≤ 0.02 (one third of the F2 threshold).
  - The first admitted candidate becomes W1. If none is admitted: `STOPPED_WORKLOAD_UNQUALIFIED`; Task A on E2 still runs.
- **Path-family accounting.**
  - `pt88.render_fam` is the pt86 'cam' estimator with identical random-number use. Test T_FAMILY requires e1 / e2 / h to equal pt86 within 1e-5.
  - It adds exclusive classes of h by the S1 / S2 vertices among x₂…x_k: Hb, Hm, H1P, H1G, H2. They sum to h. H2G is the G-involving subset of H2.

## C. The candidate R1 (SAM: sender-area multiple-importance relation)

- **Physical quantity.** The same discrete operator T as R0:
  - L_{k+1}[i, b_o] = (1/A_i) ∫_x ∫_ω fbar_i(ω)[b_o] · cos ω · L̃_k(first hit(x, ω), −ω) dω dx,
  - where L̃_k is the piecewise-constant patch × bin table.
- **Proposals.** Two proposals per flat receiver patch i, combined by the balance heuristic:
  - u: the deployed receiver-uniform state (κ_u per incident bin; density κ_u / ΔΩ).
  - a: for every front-facing pair (i, p), where p is a patch of a **peaked** material, m samples are drawn: x uniform on i, y stratified uniform on p, and ω = (y − x)/|y − x|. The density is m · d² / (A_p cos_y).
  - The integrand is split by first hit. The "first hit is peaked p" term is estimated by both proposals with weight 1/D, where D = κ_u/ΔΩ + m d²/(A_p cos_y). The remainder is estimated by u alone (weight 1/(κ_u/ΔΩ), i.e. R0's weight).
  - With no peaked material, R1 is exactly R0 (test T_REDUCE).
- **Material rule (one rule, material response only).**
  - A material is peaked if the median over the 64 bin-centre incident directions of max_bo fbar(ω)[b_o] / (a(ω)/π) exceeds κ = 8. A Lambertian scores 1.
  - Dev values: B 4.93, S1 37.95, S2 16.13. The rule therefore selects S1 (panel and G) and S2 (sphere).
  - No object label and no threshold chosen from transport errors.
- **Geometry and visibility.**
  - Pairs are culled conservatively by plane tests (exact for flat-flat pairs). A pair with any uniform hit but no area samples would bias the estimator, so test T_CULL requires zero such hits.
  - Visibility: the first hit must be p, on its front face.
- **Scattering at area samples.**
  - ER receivers use a dense fbar table, 16 × 16 cells per incident bin, cell centre, patch frame (4,096 MIS samples per cell). This is material-only, built once and cached.
  - Lambert receivers use ρ/π.
  - The u rays keep R0's exact per-ray W.
  - Curved (sphere) receivers keep the uniform proposal only. This is a declared interface limitation; their area samples would need per-sample frame integration.
- **Stored state.**
  - R0's state plus a per-ray MIS multiplier.
  - Per area sample: (receiver, sender, dbin, dense cell, weight cos/D, validity, segment).
  - Nothing source-dependent: test T_SOURCE_INDEPENDENT checks the construction signature and that the digest is identical across contexts.
- **Lifecycle.**
  - Source change: nothing is rebuilt; q and the recurrence are recomputed.
  - Geometry edit:
    - u rays are re-traced by the WL86 dirty rule;
    - the pair list is rebuilt (cheap geometry);
    - area samples are re-traced for new pairs, pairs touching a moved patch, and samples whose segment meets the old / new moved volume.
  - The per-sample random stream is keyed by (i, p, j), so selective update is bit-identical to a full rebuild (test T_EDIT).
- **Expected size (dev geometry).** About 102 k pairs, 408 k area samples (~65 % valid), relation ≈ 36 MB, build ≈ 0.3 s first call, apply ≈ 8–13 ms for four orders.

## D. Comparators and controls

| Arm | Definition | Role |
|---|---|---|
| R0 | WL87 EXP, deployed realisation | main baseline (integration test against the frozen WL87 table, ≤ 1e-4) |
| R1 | SAM, κ_u 8, m 4 | the candidate |
| R1M | SAM, κ_u 4 (half the state, reweighted), m 4 | matched traced-ray budget |
| R1m1 / R1m16 | SAM, m 1 / 16 | area-budget ladder |
| POOL2 / POOL4 / R0_POOL16 / R0_INF | R0 averaged over 2 / 4 / 16 / 64 realisations | R0 ray-budget ladder; converged R0 |
| R1_POOL16 / R1_INF | R1 over 16 realisations (m 4 / m 16) | R1 variance / converged R1 |
| RS0 / RS1 | first gather with the exact once-scattered radiance at each sample's hit | sender projection |
| R0D | R0 rays with the WL87 dense-cell scattering | scattering quadrature |
| R0F | independent realisation per order | correlation across orders |
| C3 | WL87 sender-side, power-proportional, fresh per order | source-dependent diagnostic (R2), reconstructed per source |
| C3U / C3UR | sender-side, 8 rays per (patch, bin), fresh per order / one reused relation | allocation / fresh-relation factors; C3UR is a geometry-only sender relation (control, not a candidate) |
| C3U_INF | sender-uniform, κ_s 64 × 4 per order | converged sender relation (target identity) |
| ORACLE | path-traced table, 4,096 spp per slot | output-representation ceiling |
| PT, TF64/256/1024 | frozen-ER path tracing; physical table fill | physical full-path and table-fill comparators (R3) |
| REF-ER / REF-P | 8 × 8,192 spp frozen-ER / 4 × 8,192 physical material | evaluation reference / ER approximation (kept separate) |

## E. Prospective rules (summary; full text in the protocol)

- **Stage 0.**
  - tests88 gate (any failure: `STOPPED_INVALID_COMPARISON`); workload qualification as above.
  - **Target identity**: R0_INF, R1_INF and C3U_INF must agree (region signed difference ≤ 0.01 + noise, field flux within 3 %). Otherwise the comparison changed its physical target.
- **Stage 1 (W1 T0 / T1, E2 T0 / T1).**
  - E_H and signed error per region.
  - WL86 thresholds F1 (stationary ≤ 0.03, capture ≤ 0.25) and F2 (A / C / D ≤ 0.06), applied prospectively also to C_ss / R_mirror. F3 (dH slope 0.75–1.25, correlation ≥ 0.8) on D, C_ss, R_mirror.
  - Field and per-order errors against path-traced order fields (orders 1–5), including the footprint patch set.
  - Coverage: energy-weighted coverage of (flat receiver, peaked sender) relations and missed important relations (≥ 1 % of the receiver's peaked-sender irradiance; pure area estimator, m 64).
  - Task A factor table; limit attribution per sharp region (oracle / converged relation / C3).
  - R1 class: SUFFICIENT / SHARP_SUFFICIENT_OTHER_FAIL / INSUFFICIENTLY_RESOLVED / INSUFFICIENT.
- **Stage 2.**
  - 16 realisations of R0 and R1 (seeds 88201–88216): mean, SD, 95 % interval, pass fraction, paired R1 − R0.
  - Held-out edit W2 and held-out sources, with the relation built once per geometry and reused.
  - Source-independence and edit-validity evidence.
- **Stage 3 (alone on the GPU).**
  - Per-edit lifecycle stages A–G.
  - Source-only update and 1 / 4 / 16-source reuse; 1 / 4 / 16 views.
  - Quality ladders 4–16,384 spp × 3 at W1 T1 (PT, R0, R1, R1M, C3, TF), matched by log-log interpolation only.
  - Memory.
  - Scaling: patch refinement s 1 / 1.5 / 2, a second glossy surface (box B as S1), orders 1 / 2 / 4 / 8, and 1–16 lights.
  - Warm numbers exclude first-call compilation: every timed operation is run once untimed first. This addresses the WL87 TF16 measurement defect.
- **Verdicts.** Ten items and outcomes A–F, by the predeclared mapping (`analyze88.verdict`).

## Pre-freeze development (non-evidence)

All development ran on LabServer63 in `~/rest_rel88dev`, on the dev geometry Wk or on geometry-only screening.

1. **Material peakedness** (material-only): B 4.93, S1 37.95, S2 16.13 (median Π). This fixed the rule threshold κ = 8.
2. **Screening.** Mask pixel counts of eight configurations (geometry only), from which the candidates were chosen.
3. **rel88 smoke on Wk T1, against a 256-spp truth field** (flux / rel L1):
   - R1 with m = 0 equals R0 exactly; zero missing pairs.
   - R0: panel 1.005 / 0.159, G 1.031 / 0.114.
   - R1 (κ8 m4): panel 1.018 / 0.140, G 1.030 / 0.115.
   - Both include truth noise.
4. **Dev chain** (`WL88_DEV=1`: W1 = W2 = Wk, low spp). It exercises every stage. E2 dev numbers at 64 spp were not inspected.
5. **Defects found and fixed by the dev chain (before the freeze).**
   - **Area-sample random streams depended on lane order.** Dr.Jit's PCG32 adds `arange(size)` to `initseq`, so a sample's stream changed when a subset was re-traced. The null edit and selective = full were not identical (test T_EDIT failed).
     - Fix: `initseq = key − arange`, giving one stream per (i, p, j).
     - After the fix: subset vs full trace identical, null edit identical, selective = full.
   - **T_FRAMES false failure.** Box B's bottom face is coincident with the floor, so the probe ray hit the floor.
     - Fix: lanes that hit another shape are excluded and counted.
     - Box-face tangent frames are admitted through S1's measured azimuthal isotropy (0.8 % rel L1).
   - **GPU OOM with three concurrent table jobs.** Table jobs now run two at a time and realisation jobs one at a time; the CUDA cache is emptied before C3.
   - **Missing per-geometry area table** in the pair-energy estimator.
   - **Light count.** pt86 holds at most two lights, so the source-count scaling now times one q per source.
6. **Implementation optimisation of the R1 update** (same operator; the digests are unchanged by construction and selective = full holds).
   - Changes:
     - pairs between unmoved patches keep their facing result;
     - area segments stay resident on the GPU;
     - the u-ray MIS factors are computed in torch.
   - Warm R1 update on Wk went from 193 to 129 ms (full rebuild 138 ms; R0 state 20 ms).
   - About 43 % of the area samples touch the moving glossy slab, so a G edit re-traces them by definition.
7. **Dev chain end to end** (`WL88_DEV=1`, Wk): every stage completed (`CHAIN_DONE`). Dev numbers are noise-dominated at 64 spp and are not evidence.

## Evidence run (freeze `26e13f2`; LabServer63 `~/rest_rel88`; 2026-11-11 18:05 UTC)

One detached chain.

| Step | Wall | Mean GPU | Outcome |
|---|---|---|---|
| Geometry screening of Wc / Wh / Wi | 0.8 min | — | all three geometry-admissible (Wc: A 4,104 / C_ss 5,285 / R_mirror 2,840 / D 32,490 / E 39,338 pixels) |
| Qualification partition (16 renders per candidate, 8,192 spp), candidates in order | 33.8 min | 84 % | **no candidate admitted** → `STOPPED_WORKLOAD_UNQUALIFIED` (18:40:14) |
| tests88 + E2 order / oracle fields | 1.1 min | — | **TESTS PASS** (13 / 13 gated) |
| E2 tables, realisations, renders (the protocol's "Task A on the historical E2 still runs" branch) | 9 min | 75 % (renders) | completed |
| Stage 1 analysis | — | — | **crashed**: the analysis code looked for the W1 masks, which do not exist in the stopped branch → `CHAIN_FAILED` (18:50:21) |

The GPU then stood idle until the next session (2026-11-12 03:09 UTC). The PC had been shut down at the user's request.

**Post-freeze analysis fix (user decision 2026-11-12: accept the stop by the frozen rule; no amendment).**
- `analyze88.py` now analyses only the configurations that exist (W configurations only if W1 was admitted), and `verdict` has a stopped branch.
- `viz88.py` adds the qualification figure.
- No rule, threshold, seed or data changed.
- The fix was run as `post88.sh` (copied to the experiment directory): stage1, stage2 (E2 realisations only), verdict, visuals.

### Qualification (independent partition, 4 × 8,192 spp per T; criteria frozen)

| Criterion | Wc | Wh | Wi |
|---|---|---|---|
| Q1 A direct change (≥ 0.05) | 0.388 ✓ | 0.351 ✓ | 0.402 ✓ |
| Q2 Bset indirect change (≥ 0.005) | 0.0070 ✓ | 0.0054 ✗ (snr) | 0.0060 ✓ |
| Q3 C_ss change (≥ 0.02) / H2 share of Δh (≥ 0.25) | 0.0166 ✗ / 0.995 | 0.0169 ✗ / 0.68 | 0.0207 ✓ / 0.88 |
| Q4 R_mirror change / G-family share | 0.066 ✓ / 0.91 | 0.050 ✓ / 0.82 | 0.067 ✓ / 0.64 |
| Q5 D change ≥ 0.005 and higher-order change ≥ 4 × floor | 0.0147; Δh 0.00667 vs 4 × 0.00181 ✗ | 0.0138; 0.00704 vs 4 × 0.00186 ✗ | 0.0161; 0.00722 vs 4 × 0.00192 ✗ |
| Q6 H share (≥ 0.05) | 0.241 ✓ | 0.240 ✓ | 0.233 ✓ |
| Q7 projected final uncertainty (≤ 0.01) | 0.0026 ✓ | 0.0030 ✓ | 0.0026 ✓ |
| Q8 projected sharp REF noise (≤ 0.02) | 0.0088 ✓ | 0.0090 ✓ | 0.0085 ✓ |

- Wi fails only Q5: the higher-order change of the broad control region D is 3.76 × its floor against the frozen 4 ×.
- Q7 and Q8 project the noise to the 8-replica final references. Q5 uses the floor of the 4-replica qualification partition itself; WL86 qualified with 8 replicas.
- This is an inconsistency of the WL88 protocol design, recorded as such. By user decision it is not amended.

### Historical E2 (Task A; descriptive — this is not evidence on the WL88 hypothesis, whose qualified workload does not exist)

**Correctness and target identity.**
- T_WL87: R0 equals the frozen WL87 EXP table to 1.0e-7.
- T_UNBIASED: flux ratios R0 0.9995 and R1 1.0002 against a pure area estimator; pooled relative L1 0.036 / 0.006.
- T_EDIT: null edit, selective = full and move-and-return are bit-identical.
- Converged relations agree on E2 T1 region C: R1_INF − R0_INF −0.009, C3U_INF − R0_INF −0.001, against a noise bound of 0.013; field flux within 3 %.
- The receiver-side, area-proposal and sender-side constructions therefore estimate the same discrete operator.

**Region C, E_H (signed), 4,096-spp renders with common random numbers.**

| Arm | E2 T1 | E2 T0 |
|---|---|---|
| R0 (WL87 EXP, deployed) | 0.049 (−0.049) | 0.055 (−0.055) |
| R0_POOL16 / R0_INF (64 realisations) | 0.030 (−0.023) / 0.025 (−0.007) | — |
| R1 (SAM) | 0.044 (−0.038) | 0.036 (−0.034) |
| R1_POOL16 / R1_INF | 0.026 (−0.016) / 0.027 (−0.017) | — |
| R1M (matched rays, 618 k vs 650 k) | 0.062 (−0.057) | 0.046 (−0.046) |
| C3 / C3U / C3UR | 0.024 (−0.005) / 0.023 (−0.003) / 0.026 (+0.002) | 0.021 / 0.022 / 0.019 |
| ORACLE table | 0.017 (+0.003) | 0.011 |

**Task A factors (signed change, T1 / T0; noise about 0.013 / 0.010).**

| Factor | Signed change | Reading |
|---|---|---|
| Power allocation (C3 − C3U) | −0.003 / −0.008 | within noise |
| Fresh sender relation per order (C3U − C3UR) | −0.004 / +0.004 | within noise |
| Sender vs receiver placement at equal rays and equal dense quadrature (C3UR − R0D) | **+0.050 / +0.053** | dominant factor |
| Scattering quadrature (R0D − R0) | +0.0002 / +0.0002 | negligible |
| 4× work (POOL4 − R0) | +0.014 / +0.018 | — |
| First-gather sender projection (RS0 − R0) | −0.009 / +0.003 | within noise |
| R0F − R0 | +0.059 / +0.041 | **confounded** (see below) |

R0F uses four non-deployed realisations, and the deployed realisation lies in the bad tail of the distribution (WL87). Its effect therefore mixes per-order freshness with realisation identity.

**R1 information (16 realisations, E2 T1).**
- Energy coverage of (flat receiver, peaked sender) relations: R0 0.59, R1 0.993.
- Important relations never sampled: 68 % → 16 % by count; 40 % → 0.6 % by energy.
- Region C E_H: R0 0.041 ± 0.017 (pass 81 %), R1 0.030 ± 0.008 (pass 100 %).
- Signed: −0.023 → −0.016. Paired R1 − R0: −0.011 [95 %: −0.016, −0.005], R1 better in 14 / 16.
- Stationary and D: R1 − R0 at most 1e-4.
- Edit response in C: slope R0 0.98, R1 0.86, R1M 0.78 (correlation ≥ 0.97); all pass F3. R0 and R1 pass F1–F3 on E2.
- R1's relation on E2: 73,282 pairs, 147,738 valid area samples, 943 k traced rays, 31 MB.

**Invalid control.**
- RS1 (exact first-gather radiance for R1) is invalid. The stored area-sample segment starts at the unoffset receiver point, so the exact-radiance trace self-intersects and the area part reads almost nothing (E_H 0.13).
- R1 itself is unaffected, because its own trace offsets the origin. RS1 was not rerun.
- **Not run (stopped):** Stage 3 economics and scaling, held-out sources / edit, all W1 / W2 evidence.

## Final report (Korean)

### A. CURRENT ASSESSMENT

- **WL87이 세운 것.**
  - 같은 512-ray 수신측 state 위의 explicit 계산(EXP)이 WL86 gate를 모두 통과했고, 학습 operator 이상으로 정확했습니다.
  - E2의 sharp 결손(region C)은 state realization에 따라 크게 흔들렸습니다.
  - sender-side 배치(C3)가 같은 ray 수로 그 결손을 거의 없앴습니다. 다만 C3는 source마다, order마다 relation을 새로 만드는 구조입니다.
- **WL88이 시험하려던 것.**
  - 움직이는 sharp 표면이 있는 새 GM1 workload에서 시험하려 했습니다.
  - 질문은 source에 의존하지 않고 geometry와 재질만으로 만든 relation(R1, SAM)이 sharp 다중 반사 정보를 실용적 비용으로 보존하는지였습니다.
- **실제로 일어난 일.**
  - 새 workload 후보 세 개가 모두 사전 고정한 자격 기준을 통과하지 못했습니다.
  - batch는 규칙대로 `STOPPED_WORKLOAD_UNQUALIFIED`로 끝났습니다.
  - 이 batch가 내놓는 것은 과거 calibration 사례 E2에서의 Task A 분해와 R1의 기술적(descriptive) 측정뿐입니다.

### B. IMPLEMENTATION FACT

- **R0.** WL87 EXP 그대로입니다. 고정된 WL87 table과 1e-7 수준에서 일치합니다.
- **R1 (SAM).** source에 의존하지 않는 relation입니다.
  - 기본은 R0의 수신측 uniform ray입니다.
  - 여기에 peaked 재질(S1, S2)인 sender patch와 수신 flat patch로 이루어진 모든 front-facing pair에 대해 area sample m = 4개를 더합니다.
  - 두 표본은 balance heuristic으로 결합합니다(D = κ/ΔΩ + m d²/(A_p cos_y)).
  - peaked 판정은 재질 응답만으로 하는 단일 규칙입니다: median Π > 8. 값은 B 4.9, S1 38, S2 16입니다.
  - ER 수신자의 area sample은 16 × 16 cell dense fbar로 산란합니다. 구 수신자는 uniform만 씁니다(선언한 한계).
  - 저장하는 것은 hit, bin, cell, MIS weight, segment이고 source 정보는 없습니다.
  - geometry edit 때는 dirty ray와 dirty pair만 다시 추적합니다. sample별 난수 key 덕분에 selective update가 full rebuild와 bit 단위로 같습니다.
  - E2에서의 규모: pair 73k, 추적 ray 943k(R0는 650k), 31 MB.
- **Task A 대조군.** C3, C3U(균등 할당 sender), C3UR(균등 할당에 relation 하나를 전 order에 재사용, geometry-only), R0D, R0F, POOL, RAYSEND, 수렴 relation(R0_INF, R1_INF, C3U_INF).
- **새 workload.**
  - 방의 box A를 S1 재질의 얇은 glossy slab으로 바꿔 패널(S1) 앞에서 회전시켰습니다.
  - mask(C_ss: S1→S1 두 번 반사 footprint, R_mirror: 패널 reader)는 거울 photon을 이용한 geometry-only 계산입니다.
  - 자격 판정은 독립 partition의 family-tagged reference로 했습니다.

### C. MEASUREMENT

- **Workload 자격.** 후보 세 개 모두 불합격입니다.
  - 공통 실패는 Q5입니다. broad 대조 영역 D의 고차 변화가 노이즈 대비 3.7–3.8배로, 기준 4배에 못 미쳤습니다.
  - Wc와 Wh는 C_ss 변화량(0.017 < 0.02)도 미달했고, Wh는 Q2도 미달했습니다.
  - Wi는 Q5 하나만 실패했습니다.
  - 의도한 sharp family의 에너지 자체는 기준을 넉넉히 넘었습니다. C_ss 변화 중 H2 몫은 0.68–0.995, R_mirror 변화 중 G가 관여한 몫은 0.64–0.91입니다.
- **정확성 (E2).**
  - gated test 13개가 모두 통과했습니다.
  - 세 가지 구성의 수렴 relation이 같은 operator로 모입니다(target identity PASS).
- **E2 region C.**
  - deployed R0는 −0.049입니다. 64개 realization으로 수렴시키면 −0.007, oracle table은 +0.003입니다. 따라서 deployed 결손은 표본 오차이고, 다른 물리 목표를 계산한 탓이 아닙니다.
  - 16개 pooled에서 −0.023이 남은 것(WL87)은 수렴이 느리다는 뜻이지 체계적 편향의 증거가 아닙니다.
- **Task A.**
  - C3의 우위는 power 비례 할당(−0.003)에서도, order마다 relation을 새로 만드는 것(−0.004)에서도 오지 않습니다.
  - 같은 ray 수, 같은 dense quadrature 조건에서 geometry-only sender relation을 전 order에 재사용한 C3UR이 C3와 같은 결과를 냅니다(+0.002, E_H 0.026). 즉 sender 쪽 배치 자체가 지배 요인입니다(+0.050).
  - 산란 quadrature(R0D − R0)는 0.0002로 무관합니다.
- **R1 (E2, 기술적 측정).**
  - peaked relation 에너지 coverage가 0.59에서 0.993으로 올랐습니다.
  - region C E_H는 0.041 ± 0.017에서 0.030 ± 0.008로 줄었습니다. paired 차이는 −0.011 [−0.016, −0.005]이고, 16개 중 14개 realization에서 R1이 나았습니다. realization 통과율은 81 %에서 100 %가 됐습니다.
  - deployed 기준으로는 T0에서 noise 이상 개선(0.055 → 0.036), T1에서는 noise 이내(0.049 → 0.044)입니다.
  - coverage가 거의 완전한데도 signed −0.016이 남습니다. R1_INF도 −0.017인데, R0_INF와의 차이(−0.009 ± 0.013)는 noise 이내라 원인을 분리하지 못했습니다.
  - 추적 ray 수를 맞춘 R1M(κ 4)은 오히려 나빴습니다. T1 0.062로 F2를 넘었습니다(단일 draw).
  - C 영역 edit response slope는 0.86(R0 0.98)이지만 F3는 통과합니다.
- **비용.** Stage 3(경제성, scaling)는 실행되지 않았습니다. dev geometry에서의 비증거 값만 있습니다: R1 warm update 129 ms(full rebuild 138 ms), R0 state update 20 ms.
- **Reference 불확실도.** qualification partition 기준으로 최종 reference noise를 환산하면 stationary 0.003, sharp 영역 0.009로 충분했습니다(Q7, Q8 통과).

### D. OBSERVATION

1. 회전하는 glossy slab은 의도한 대로 두 번 반사 footprint와 패널 reader의 변화를 만듭니다(시각화 fig0). 그런데 이 장면에서 slab의 회전이 broad 영역에 주는 고차 변화는 작습니다. 기준을 4 × 8,192 spp에서 판정하면 노이즈 대비 4배에 닿지 않습니다.
2. E2에서 수신측 relation의 결손은 "샘플이 닿지 않은 sharp relation"과 직접 연결됩니다. R0는 중요 relation 에너지의 40 %를 표본하지 못하고, R1은 0.6 %만 놓칩니다.
3. 그럼에도 R1의 region C 개선은 sender-side 배치(C3, C3UR)보다 작습니다: 0.030 대 0.023–0.026.
4. 시각화는 `results/evaluation/88/`에 있습니다: qualification(fig0), E2 오차 지도(fig2), realization 분포(fig3), Task A 요인(fig4).

### E. INTERPRETATION

- **Workload 자격 실패의 성격.**
  - sharp family가 없어서 실패한 것이 아닙니다. 사전 기준의 broad 대조 조건(Q5)과 C_ss 변화량 조건(Q3)에서 실패했습니다.
  - Q5의 noise floor를 qualification partition 크기(4 replica)로 계산한 것은 Q7 / Q8의 최종 환산과 일관되지 않습니다. WL88 protocol 설계의 결함입니다.
  - 사용자 결정에 따라 사후 수정하지 않았습니다. 따라서 "이 workload에서 sharp relation 가설을 시험할 수 없었다"가 정확한 결과입니다.
- **Relation 발견의 한계 (E2).**
  - 수신측 고정 방향 표본의 결손은 표본 배치의 분산(heavy tail)입니다. 수렴 relation은 같은 operator로 모입니다.
  - source 의존 할당은 필요하지 않습니다. geometry-only sender relation(C3UR)이 같은 효과를 냅니다.
  - 이것은 WL87의 "C3 우위 = source 의존 재구성"이라는 해석 가능성을 E2에서 좁힌 측정입니다. 단일 draw이고, 단일 장면입니다.
- **표현 한계.** E2에서 oracle table과 수렴 relation이 모두 region C를 통과하므로, patch × 64-bin 표현은 이 family를 담을 수 있습니다(WL87과 일치). 새 workload의 sharp-to-sharp family에 대해서는 측정하지 못했습니다.
- **Quadrature와 sampling.** 산란 quadrature(R0D)는 무관합니다. R1의 이점은 coverage에서 오지만, 남은 signed 결손의 원인은 분리되지 않았습니다.
- **Source 의존성.** E2에서 C3의 이점은 source 의존성에서 오지 않았습니다. 새 workload에서의 source 재사용은 측정하지 않았습니다.
- **비용.** 측정하지 않았습니다. relation 발견과 반복 적용 중 무엇이 지배하는지는 답하지 못했습니다.

### F. UNRESOLVED QUESTIONS

- 새 sharp-to-sharp, moving-sharp workload에서 R0, R1, C3의 정확도와 비용. 이번 batch의 핵심 질문이며, workload 미자격으로 측정되지 않았습니다.
- E2에서 R1이 coverage를 거의 채우고도 남기는 signed 결손(−0.016)의 원인. R1_INF와 R0_INF의 차이가 noise 이내라 분리되지 않았습니다.
- C3UR(geometry-only sender relation)의 우위가 realization 분포로도 성립하는지. 이번에는 단일 draw입니다.
- R0F 효과에서 order마다 새 relation을 쓰는 효과와 deployed realization의 꼬리 효과가 섞여 있습니다.

### G. ARCHITECTURE VERDICT

범위는 GM1 방 family, 회전하는 S1 glossy slab 후보 세 개, frozen ER 재질, 점광원입니다. E2는 과거 calibration 사례입니다.

| 항목 | 판정 |
|---|---|
| WORKLOAD QUALIFICATION | **NOT SUPPORTED**: `STOPPED_WORKLOAD_UNQUALIFIED`. 세 후보 모두 사전 기준 미달(공통 Q5; Wi는 Q5만) |
| PHYSICAL CORRECTNESS | **SUPPORTED**: gated test 13개 통과, E2 target identity 통과 |
| SHARP RELATION INFORMATION SUFFICIENCY | **NOT TESTED** (qualified workload 없음). E2 기술값: coverage 0.59 → 0.99, realization E_H 0.041 → 0.030 |
| SOURCE-INDEPENDENT REUSE | **NOT TESTED** (구성상 source 입력 없음은 test로 확인) |
| DYNAMIC EDIT VALIDITY | **NOT TESTED** (null, selective = full, move-and-return identity는 test로 확인) |
| RELATION CONSTRUCTION ECONOMICS | **NOT TESTED** |
| OPERATOR APPLICATION ECONOMICS | **NOT TESTED** |
| QUALITY-MATCHED END-TO-END ECONOMICS | **NOT TESTED** |
| REPRESENTATION SCALABILITY | **NOT TESTED** |
| GM1 ARCHITECTURE IMPLICATION | outcome A–F 분류 불가 (stopped). E2 Task A에서 C3 우위는 source 의존 할당이 아니라 sender 쪽 배치에 귀속됨(단일 draw) |

candidate는 채택하지 않았고, 어떤 아키텍처도 선택하지 않았습니다.

### H. REPRODUCIBILITY

- **Commits.** freeze `26e13f2`. The post-freeze analysis fix and results are in the closing commit.
- **Protocol.** `experiments/gm1_relation88/protocol/gm1_relation88.json` (v1, no amendment; the stopped-branch analysis fix changes no rule).
- **Environment.** LabServer63, RTX 3080 Ti 12 GB, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Seeds.**
  - Qualification REF-ER 88140+ and REF-P 88150+.
  - E2 tables: R0 seeds as WL87; area 88005; sender relation 88007.
  - Realisations 88201–88216; converged 88301–88348 (R0), area +500 (R1), sender 88011.
  - Pair energies 88009; E2 renders 86600 / 86601 (common random numbers with WL86 / WL87); E2 oracle and order fields 88450 / 88460.
- **Server.** `~/rest_rel88` (qualification renders, E2 tables / renders, logs, `post88.sh`).
- **Local.**
  - `results/gm1_relation88/`: chain status, GPU trace, JSON (w_screen, wq_*, tests88, stage1 / stage2, final88), logs, render metadata.
  - `results/evaluation/88/`: four figures plus a manifest with sha256.
- **Invalid or excluded.**
  - RS1 control (self-intersection, see above).
  - The chain's first stage1 attempt (crash; fixed post-freeze without a rule change).
  - Stage 3 and every W1 / W2 stage: not run (stopped).
