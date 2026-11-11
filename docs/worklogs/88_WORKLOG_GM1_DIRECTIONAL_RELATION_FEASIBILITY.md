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
