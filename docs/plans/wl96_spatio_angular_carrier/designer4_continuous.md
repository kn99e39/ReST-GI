# WL96 designer D4: spatio-angular transport carrier architecture (design-only report)

## Scope, conduct and labels

- **Read-only.** I edited no repository file, trained nothing and ran no GPU or renderer job.
- **Scratch work.** I ran small CPU numpy calculations, all under `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\` (file list in §9).
- **Labels.**
  - [FACT]: verified in code or docs (path:line).
  - [MEAS]: a measured value, with its source.
  - [MEAS-S]: a value from **my scratch CPU model**. The model uses a numpy port of ER, simplified CK2 geometry (box A's panel-facing face only, no occlusion, omni light) and no renderer. These are design-scale estimates, never renderer measurements.
  - [INF]: inference. [HYP]: untested.
- **No novelty claims.** Every mechanism below belongs to a known class: VPL / virtual-spherical-light-type glossy quadrature, image-source mirrors, sub-bin directional tables, PRT-style low-rank transfer, and neural radiance caches.

---

## 0. Verdict summary

1. **I reject the starting point as literally stated** ("the sender state is a learned field L(x, ω)"). [INF]
   - A learned *outgoing radiance* field is source-dependent baked radiance. It has to be refit after every light change and every edit; this is the neural-radiance-cache class.
   - The canon excludes that as runtime state:
     - `docs/Architecture.md` §4, the paragraph after the ownership table: "Runtime rule/state inputs exclude … baked final radiance";
     - the AGENTS.md invariant "learned transport **response**, not canonical baked radiance".
   - If instead the field is computed from the current incident state through ER, then it *is* ER evaluation and holds no learned information.
   - ER already provides the continuous (x, ω) query exactly, once the incident state is known at continuous x [FACT `experiments/fnto_compiled83/closure83_dr.py:6-15`].
2. **Store the smooth quantity and evaluate the sharp one.** [MEAS-S]
   - Setup: a noise-free model of WL95A's CK2 S1 → box-face exactly-2-event delivery. It reproduces WL95A's binned over-delivery (1.42 / 0.43 vs measured q256 1.422 / 0.433) and X1c (1.02 / 0.23 vs 1.037 / 0.268).
   - Storing only the *incident* light state (direction and irradiance) at g × g control points per 0.2 m patch, and evaluating ER at the exact outgoing (x, ω), gives rel L1 **0.113 / 0.029 / 0.008 / 0.002 for g = 1 / 2 / 4 / 8**.
   - The binned patch × bin table gives 0.43 and X1c gives 0.23.
3. **Two materially distinct candidates.** Neither is recommended; selection is still premature (§6).
   - **A, CIS-ER** (Continuous Incident State + ER emission + deterministic delivery).
     - It removes the WL95A angular *and* spatial projection bias by construction.
     - **It contains no learned transport.** It is the representation-level instance of WL95's G1 class, so it is a comparator under `docs/Architecture.md:13` unless the owner rules otherwise.
     - Its optional learned piece, an integrable lobe compiler, is not supported by my capacity calculation (§3.D), so I drop it.
   - **B, LIR** (Learned Incidence Resolvent), in the ECRO lineage.
     - ECRO's source-independent, geometry-conditioned generator is re-targeted to output **non-negative incidence** for the ≥ 3-event family whose last pre-arrival scatterer is broad.
     - ER owns every local response, at continuous directions.
     - The last flight from sharp senders is A's explicit carrier.
     - B specifies WL95 G2's missing Λ_θ and maps ECRO's event partition onto canonical K/C.
     - Its learned utility over explicit sweeps is unproven and a priori small. Its decisive preconditions can be tested without learning.
4. **Delivery variance is not a carrier property.** [MEAS-S]
   - An exact continuous carrier read through ECRO's receiver-anchored 512-ray relation stays heavy-tailed per draw: single-draw median rel L1 0.56 for point reads, 0.34 when each ray's whole sub-cell is integrated.
   - A deterministic sender-cell × receiver-point quadrature reaches 0.026, with no variance.
   - The carrier fixes the bias; the estimator sets the variance and the ray cost.
5. **Validation lights must be qualified as sharp-regime.** [MEAS-S] With a moderate light at (−0.7, −0.2, 2.4), even the binned table is nearly unbiased (0.994 / 0.051). A non-grazing control cannot test whether the sharp fix generalises.
6. **Indirect incidence on a sharp sender is mainly a spatial-resolution problem, not an angular-basis one.** [MEAS-S]
   - Setup: an extended emitter 0.2–0.3 m from the panel (box A's own face, bright column).
   - Exact incidence held only at the patch centre gives 0.184; at 4 × 4 cells it gives 0.033.
   - At 4 × 4 cells, the angular representation adds little:
     - 64-bin-uniform 0.045;
     - 2 × 2 sub-bins 0.031;
     - 4 × 4 sub-bins 0.020;
     - a per-bin mean-direction point atom is worse, 0.084 with +8 % bias.

---

## 1. Facts the design relies on

| # | Fact | Source |
|---|---|---|
| F1 | ECRO sender: u = c·s, Bq = a·u, rank ≤ 16, non-negative, c tied; Bq constant per (patch, outgoing bin) | [FACT] `experiments/ecro86/model86.py:111-120` |
| F2 | Flight: E1[n,m] = Σ_κ 1[hit≥0]·Bq[hit, dbin]·rw_k, rw_k = z_k ΔΩ/κ; K = 512, κ = 8, one shared offset pattern (seed 86001) | [FACT] `model86.py:40-46, 115-119`; `experiments/ecro86/pt86.py:113-116, 353-376` |
| F3 | Global term y_g = P·Σ_j area_j Q_j u_j (rank 24), driven by u; B = a·(u1 + y_g) | [FACT] `model86.py:121-124` |
| F4 | Source s = patch-mean direct irradiance per incident bin from 16 fixed points | [FACT] `pt86.py:378-402` |
| F5 | ER eval(wi, wo) = f·cosθ_o; Lambert + 3 outer + 4 refracted-inner GGX; exact eval / pdf / sample; NEE = eval(wi = viewer, wo = light)·I/d² | [FACT] `closure83_dr.py:1-15`; WL95A T_COS |
| F6 | S1 state: GGX α 0.026–0.075 (au ≈ av), η_e 2.50 | [FACT] `experiments/ecro86/data/mats86.npz` states[1] |
| F7 | field_src(split) tallies the *receiver outgoing* radiance per (patch, bin), split by event k and by whether vertex 2 is material 1 | [FACT] `experiments/ecro86a/pt86a.py:73-77, 175-199` |
| F8 | X1 / X1c: exact ER + light at the ray's hit point / patch centre, shadowed | [FACT] `experiments/ecro_attr95a/a95.py:90-136` |
| F9 | C3URX1: 1 probe intersection + 1 shadow ray per light per landed relation ray; no cache | [FACT] `experiments/gm1_c3urx1_91/cand91.py:1-15, 86-94` |
| F10 | Canon: ECRO lineage is the learned-global base; EXP/C3UR are comparators; X1 findings do not justify explicit GI; no "exhaustive explicit per-surface full path tracer hidden under a neural name" | [FACT] `docs/Architecture.md:13`; `docs/RESEARCH_ROADMAP.md:128`; `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md:86` |
| F11 | Peaked-material rule → S1 and S2 peaked; floor B not | [FACT] `docs/worklogs/88_…md:115` |

**Measured anchors.**
- WL95A pooled (ratio / abs rel L1) [MEAS `docs/worklogs/95A_…md:72-83`]:
  - M64 1.544 / 0.546;
  - q256 1.422 / 0.433;
  - X1c 1.037 / 0.268;
  - X1 1.022 / 0.111;
  - X1 single draw: median 0.617, p90 1.186, worst 2.212;
  - reference noise n0 = 0.059.
- WL95A attribution [MEAS `results/ecro_attr95a/an.json`]: Ra 0.232 ± 0.047, Rp 0.157 ± 0.060, Rq_incident 0.066 ± 0.005.
- 9,995 deployed relation rays land on S1 [MEAS 95A:111].
- ECRO per edit 76–78 ms (geometry 31, state 21, source 5, operator 11, transfer 1.5) [MEAS `docs/worklogs/86_…md:214`]. Source-only 13–18 ms [MEAS, WL95 §4.1].
- C3URX1 (GM1 scene) [MEAS `docs/worklogs/91_…md:228-257`]:
  - 526,467 probe + 526,467 shadow rays per light;
  - edit to frame 124.7 ms;
  - source update without render 21.4 ms.
- Scene86: N = 1,270 patches; relation N·K = 650,240 rays; S1 has 42 patches, S2 has 32 [FACT].

---

## 2. Information argument and scratch evidence

**2.1 Physics of the first sharp flight.**

```
L1(x, ω) = I · eval_z(ω^loc(x), ω_ℓ^loc(x)) · V(x, ℓ) / |ℓ − x|²
E2(y)    = ∫_S L1(x, ω_{x→y}) cosθ_x cosθ_y |x − y|⁻² V(x, y) dA_x
```

- L1 is sharp in ω. S1's half-energy radius is 0.13–0.165 rad over θ_i 10–88° [MEAS-S, numpy ER; consistent with WL86B 0.146]. The lobe centre moves with x through ω_ℓ(x).
- ω_ℓ(x) and log(I/d²) are smooth in x; only ER's response is sharp. [INF] The minimal exact state is therefore the incident field at continuous x.

**2.2 Noise-free carrier ladder.**
[MEAS-S `ck2_model2_ck2.json`. Setup: the same deterministic quadrature in every row (4 mm panel grid × 4 × 4 receiver points per face patch, 9 patches), with only the carrier substituted for L1. Quadrature convergence, 2 mm vs 4 mm: 5e-5.]

| Carrier | ratio | rel L1 | WL95A counterpart |
|---|---|---|---|
| Binned patch × bin mean | 1.420 | 0.426 | q256 1.422 / 0.433 |
| Exact position, outgoing direction snapped to bin centre | 1.423 | 0.527 | — |
| X1c-type (patch-centre point, exact direction) | 1.016 | 0.234 | X1c 1.037 / 0.268 |
| CP1: incident state at patch centre, emission at exact x and ω | 0.976 | 0.113 | — |
| CP2 / CP4 / CP8 (log- and normalised-bilinear incident state) | 0.971 / 0.992 / 0.998 | 0.029 / 0.008 / 0.002 | — |

[INF] What the ladder shows:
- Outgoing-direction quantisation alone reproduces the 1.42 over-delivery: grazing ring-0 aliasing (WL95 L4/L6), not the incident encoding.
- X1c's residual comes mostly from anchoring the emission point, not from the incident state.
- A 5 cm incident lattice suffices for CK2.

Non-grazing control [MEAS-S `ck2_model2_alt.json`]:

| Carrier | ratio / rel L1 |
|---|---|
| binned | 0.994 / 0.051 |
| X1c | 1.001 / 0.023 |
| CP1 | 1.010 / 0.010 |
| CP4 | — / 0.0008 |

**2.3 Estimator variance.**
[MEAS-S `ck2_model_ck2.json`. Setup: 64 ECRO-like relation draws, 512 rays per receiver patch, 8 per bin.]

| Estimator | pooled ratio / rel L1 | single draw median / p90 / worst |
|---|---|---|
| binned, point read | 1.40 / 0.41 | 0.70 / 0.94 / 1.17 |
| X1 point read | 0.86 / 0.17 (heavy tail; not converged at 64) | 0.56 / 0.96 / 2.44 |
| X1, 2 × 4 stratified per bin | 0.98 / 0.09 | 0.51 / 0.93 / 3.80 |
| ray-cell footprint read, 3 × 3 sub-directions | 1.01 / 0.10 | 0.41 / 0.66 / 1.03 |
| ray-cell footprint read, 6 × 6 | 1.00 / 0.06 | 0.34 / 0.50 / 0.65 |

Deterministic sender-cell × receiver-point quadrature, with exact ER at cell centres and one visibility ray per kept pair [MEAS-S `ck2_model2_ck2.json`; 9 receiver patches]:

| Sender cells per 0.2 m patch | Receiver points per patch | Cull | rel L1 (ratio) | Pair rays |
|---|---|---|---|---|
| 4 × 4 | 4 × 4 | none | 0.026 (1.004) | 96,768 |
| 4 × 4 | 4 × 4 | 1e-3 of the patch max | 0.031 (0.976) | 12,171 |
| 8 × 8 | 2 × 2 | 1e-3 | 0.031 (0.969) | 12,312 |
| 8 × 8 | 4 × 4 | none | 0.0075 | 387,072 |
| 16 × 16 | 4 × 4 | none | 0.0009 | 1,548,288 |
| 4 × 4 | 4 × 4 | 1e-2 | 0.145 (0.855) | 3,920 |

[INF] What this shows:
- Footprint reads remove the angular variance but not the receiver-origin variance: the mirror image moves with the random origin.
- Only an area-to-area deterministic quadrature removes both.
- Culling biases low, because the GGX tails carry energy.

**2.4 Extended (indirect-like) incidence on the sharp panel.**
[MEAS-S `ext_incidence.json`. Setup: the emitter is box A's panel-facing face as a Lambertian emitter (bright +y column 1.0, rest 0.05), standing in for the caustic-lit column of an e3_p path; receivers are points on the same face. Panel 2 cm, emitter 3 cm, 36 receiver points; nearest-cell state.]

| Incident carrier at the panel | g = 1 (patch) | g = 4 (5 cm cells) |
|---|---|---|
| exact incidence held at the cell centre | 0.184 | 0.033 |
| 64-bin Φ_m × bin-averaged ER response (BINU) | 0.115 | 0.045 |
| per-bin mean-direction point atom (MATOM) | 0.199 | 0.084 (ratio 1.083) |
| 2 × 2 sub-directions per bin, non-negative (SUB4, 256 directions) | 0.163 | 0.031 |
| 4 × 4 sub-directions per bin (SUB16) | 0.186 | 0.020 |

[INF] For a close extended source, the spatial resolution of incidence on the sharp sender dominates. At g = 4 the angular basis matters at the 0.01–0.03 level. Moment point-atoms over-sharpen.

**2.5 Lobe-family capacity, for an analytic integrable carrier.**
[MEAS-S `lobe_fit.json`, `lobe_fit2.json`.] I fitted moment-shaped anisotropic Gaussians to S1's ER response, with L1-optimised width and amplitude, one per ER component.

| θ_i | rel L1 | energy | peak |
|---|---|---|---|
| 10° | 0.275 | 0.78 | 0.67 |
| 45° | 0.266 | 0.85 | 0.69 |
| 75° | 0.304 | 0.73 | 0.56 |
| 85° | 0.448 | 0.57 | 0.42 |
| 88° | 0.555 | 0.47 | 0.22 |

[INF] Gaussian-family lobes miss the GGX tails, worst at grazing, exactly where CK2 lives. An analytic lobe compiler would add approximation error to an exact closure. This is consistent with WL95's rejection of MEK.

**2.6 Consequence for where learning can sit.** [INF]
- For point lights the exact sharp first order is closed-form (light, geometry, ER) plus visibility. Learning there can at most save compute.
- Learning has a non-trivial job only where the incident state is itself a transport result: incidence on sharp senders from scattered light, the e3_p family (0.755 of panel-face e3 on CK2 [MEAS WL94B via WL95 §2.3]), or the broad multi-bounce.

---

## 3. Candidate A: CIS-ER (Continuous Incident State + ER emission, deterministic delivery)

**A. State representation**
- *Material-only:* the frozen ER state z_m (69 floats). Nothing learned.
- *Geometry-dependent:*
  - the peaked-class mask (F11 rule, prospective);
  - per sharp patch, a g × g control-point lattice in material coordinates (position, normal, frame: 9 floats each). g is chosen by a geometry-only rule [HYP]: bilinear direction-interpolation error (h/d_ℓ,min)²/8 ≤ α_min/4, i.e. h ≤ d_ℓ,min·√(2α_min). For CK2 (d ≈ 0.3 m, α ≈ 0.03) that gives h ≲ 7 cm, so g ≥ 3. Consistent with CP2 0.029 / CP4 0.008.
  - The delivery cache D2-G: a sender-cell × receiver-point pair-visibility bitmask, geometry-only.
    - Scene86 at g = 4, 4 × 4 receiver points: (672 S1 + ~512 S2 cells) × 20,320 points ≈ 24 M pairs, about 12 M after back-face culling. That is ≈ 1.5–3 MB as bits [INF arithmetic].
- *Source-dependent:*
  - Point atoms per light per control point: (Φ rgb, ω_ℓ) = 6 floats, i.e. 16 × 6 floats per sharp patch per light.
  - Extended incidence, only when present: 64 bins × 2 × 2 sub-directions, non-negative rgb weights, at 4 × 4 cells per sharp patch.

**Mathematical definition.**

```
L_s(x, ω) = Σ_l Φ̃_l(x) · eval_z(ω^loc, ω̃_l^loc(x))  +  Σ_j W̃_j(x) · eval_z(ω^loc, ω_j^loc)
E2(y_r)   = Σ_c L_s(x_c, ω_{c→r}) · cosθ_c cosθ_r |x_c − y_r|⁻² · ΔA_c · V_cr          (D2)
```

- The tilde denotes log-bilinear (Φ), normalised-bilinear (ω) or nearest-cell (W) interpolation.
- The receiver patch value is the mean over its points, passed through the exact M64 row (Lambert) or through ER at continuous ω (sharp receivers).
- The camera readout of sharp surfaces evaluates L_s at the exact (x2, ω_{x2→x1}). This removes L10 for sharp surfaces.

**Information.**
- *Retained:* exact incident directions of point atoms; incidence at control-point resolution; exact outgoing x and ω at every query.
- *Discarded:* within-cell variation of extended incidence; structure finer than a sub-bin; tails below any cull; within-cell visibility variation.

**B. Operator interface.**
- Query(x ∈ sharp surface, ω, z_m, incident state) → rgb radiance.
- Deliver(sender cells, receiver points, visibility bitmask) → receiver irradiance per patch.
- Inputs: geometry (frames, cells, bitmask), the light set, ER. No learned input.

**C. Physical ownership.**
- ER owns all local scattering, exactly (point atoms) or as a 256-direction point-atom quadrature (extended).
- Geometry owns visibility: control-point shadow rays and the pair bitmask.
- Nothing learned. Segments have exclusive owners: event 2 from sharp senders lies in A; everything else lies outside A.

**D. Learning.**
- **None.** The only learnable option considered, an ASG lobe compiler trained from ER, is rejected (§2.5): L1 0.27–0.56, peak 0.22–0.69.
- What A expresses that explicit ER + transport does not: nothing. A *is* explicit ER + transport with a better carrier and estimator.
- A is G1 class: a comparator, oracle, carrier and information floor, not the method, unless the owner rules at Gate A.

**E. Preserving the sharp lobe.**
- On the WL95A ladder, A at g ≥ 4 lies at the X1 end: CP4 0.008 vs X1 0, X1c 0.234, binned 0.426 (§2.2).
- A removes both the Ra (outgoing-angular) and Rp (spatial) projection terms. The D2 delivery removes the per-draw variance that made pooled X1 necessary (WL95A single-draw median 0.62).

**F. Dynamic reuse.**
- *Rigid edit of a non-sharp occluder:* dirty control-point shadow segments and dirty pairs are those intersecting the moved OBB (ECRO's dirty test, `ecro86.py:57-67`). Re-test only those; nothing else changes. ER and the cell lattice survive.
- *Rigid move of a sharp sender:* the lattice moves with it in material coordinates. All its atoms and all its pairs are recomputed: 672 × 20k ≈ 13.6 M pair rays for S1 [INF].
- *Light change:* point atoms are re-traced (1,184 rays for S1 + S2) and ER is re-evaluated over the pairs. No pair rays.
- *Non-rigid edit:* the material-coordinate lattice gives correspondence. Positions, frames and pairs of moved cells are invalidated. The g rule is re-evaluated because curvature changes the planar-cell assumption.
- *Self-contact:* the 1/r² near-field of the cell quadrature needs adaptive subdivision. Clamping is not admissible.

**G. Complexity** [INF arithmetic unless marked].
- *Offline:* none (no training).
- *Per geometry build:* ≈ 12–24 M pair-visibility rays at g4 × r4. That is 20–37× ECRO's 650 k relation rays, and quadratic, O(A_sharp × A_receiver).
- *Per light:*
  - A: ~1.2 k shadow rays + ~12–24 M ER evaluations, with no rays per light.
  - X1: per draw 9,995 probe + 9,995 shadow rays on S1, and ≈ 64 draws for CK2 accuracy ⇒ ≈ 0.64 M + 0.64 M per light, plus 64 stored or retraced relations (64 × 650 k).
  - C3URX1 (GM1): 0.53 M + 0.53 M [MEAS].
  - ECRO: 20,320 source shadow rays [FACT 16 × 1,270].
- *Per edit:* dirty pairs plus atoms; the geometry (31 ms) and relation (21 ms) parts are as ECRO's. Wall time is unmeasured [HYP].
- *Memory:* bitmask ~1.5–3 MB; atoms ~KB per light; sub-bin incidence 74 × 16 × 256 × 3 floats ≈ 3.6 MB.
- *Scaling:*
  - lights: O(L × pairs) ER evaluations (bad for 100+ lights; merging lights into sub-bins approximates);
  - sharp surfaces: O(sharp area × receivers);
  - objects: receivers grow linearly, pairs bilinearly. A hierarchical (cluster) pair refinement is the known mitigation [INF].
- *Work avoided vs X1:* per-light probe and shadow rays (≈1.3 M → ≈1.2 k), and multi-draw pooling (deterministic).
- *Work introduced:* a dense geometry-only pair cache and O(L·pairs) ER evaluations.

**H. Failure modes.**
- *α → 0:* the cell count grows as α⁻² because cells must stay below the lobe footprint α·r (measured: g4 0.026, g8 0.0075, g16 0.0009 at α ≈ 0.03–0.075). A planar mirror needs the image-source special case; a curved mirror (caustic) has no deterministic shortcut.
- *Close contact:* near-field singularity.
- *Many lights:* linear ER cost.
- *Moving sharp objects:* full pair rebuild for that object.
- *Non-rigid:* planar-cell and frame validity.
- *Curved senders (S2):* WL95 L11 frame issues.
- *Unseen materials:* ER is exact for any state; misclassification by the peaked rule falls back to the binned path, which is adequate for broad materials (floor B rank-16 field L1 0.016 [MEAS WL95 §2.1]).
- *e3_p:* A owns it only if extended incidence at sharp cells is supplied. Explicit relation incidence per cell has only 512 / 16 = 32 rays for 64 bins [INF], so it needs D2-style gathers or B.

---

## 4. Candidate B: LIR (Learned Incidence Resolvent), ECRO lineage

**Event and ownership partition** (exclusive; tallied).

```
i_n   = i⁰_n                                 direct point atoms (A, exact)
      + C_rel[ M64 · i⁰ ]_n^{broad senders}  explicit first flight from broad senders (ECRO E1 with exact M64)
      + D2[ L_s(i_sharp) ]_n                 explicit last flight from sharp senders (A's carrier), J_s = 2 rounds
      + Λ_θ(G)[ b¹ ]_n                       LEARNED: ≥3-event incidence whose last pre-arrival scatterer is broad
i_sharp = i at the sharp cells;   b¹ = M64 · (i⁰ + first-flight incidence), the binned once-scattered table
Λ_θ(G)[b¹]_n = P_n · ( Σ_j area_j · Q_j · b¹_j )        non-negative, linear, rank ρ, per channel
b_n = K_n[i_n]: ER at continuous ω for sharp n (and for camera readout); M64 rows for broad n
```

**A. State.**
- *Material-only:* ER z_m; generator weights θ (ECRO trunk, 735,664 params [MEAS WL86]).
- *Geometry-dependent, regenerated per edit by the generator:*
  - encoders Q_j [3, ρ, 64];
  - decoders P_n: sharp cells [74 patches × 16 cells × 64 bins × 3 × ρ]; Lambert patches [3, 1, ρ] (irradiance only); floor-B patches [3, 64, ρ].
  - At ρ = 32: ≈ 7.3 M + small floats ≈ 30–36 MB [INF] (ECRO factors 43 MB).
- *Source-dependent:* A's atoms, s, and b¹.
- §2.4 argues for sub-patch cells over sub-bins at the sharp patches. 64 bins with bin-uniform response cost +0.01–0.025 there [MEAS-S].

**B. Operator interface.**
- Λ_θ: (geometry state G, b¹) → incidence at every patch: per-cell 64-bin non-negative weights at sharp cells, bins or irradiance elsewhere.
- No light, target or identity input. Linear in b¹, so superposition across lights is exact.

**C. Ownership.**
- ER owns all local scattering, exactly.
- Explicit geometry owns visibility and one flight: the relation for broad senders, D2 for sharp senders.
- Λ_θ owns only ≥ 3-event broad-last incidence, including chains that pass through sharp intermediates before a broad last scatter.
- [INF] Mapping onto canonical K/C (`Architecture.md` §3): K = ER; C = one explicit flight; Λ_θ ≈ the truncated resolvent Σ_{j≥1}(C K)^j C restricted to broad last vertices. This gives ECRO's event partition a K/C reading (the WL95 "unresolved mapping") for this candidate.
- Unowned residue: sharp→sharp→sharp chains beyond J_s, declared.

**D. Learning.**
- *Mapping learned:* geometry → non-negative rank-ρ incidence resolvent.
- *Supervised family:* a **new incident tally** [HYP, constructible]. It is a field_src variant: vertex 1 is treated as a cosine collector; incoming direction and pre-BSDF contribution are accumulated per (patch or cell, incident bin); only paths with k ≥ 3 whose vertex 2 is broad are kept. Event-exclusive.
- *Consistency contract:* for Lambert receivers, tally × ρ/π must equal field_src e3 for the broad-last subset.
- *Data scale:* the ECRO dataset (1,024 geometries × 4 lights × 256 spp [MEAS WL86 via WL95 §4.3]).
- *What it expresses beyond explicit ER + transport:* **nothing in expectation.** The claimed utilities are:
  - (i) deterministic higher orders, with no relation-draw noise;
  - (ii) source change with no relation rays (ECRO's only measured learned advantage, 3.5–3.9× source-only [MEAS WL87:315]);
  - (iii) per-edit regeneration ≤ explicit sweeps.
- Prior expectation is weak: sweeps 11.6 ms vs generator ≈ 12.5 ms, and P2plus ≤ 0.0009 in GM1 [MEAS cross-regime, WL95 §3.2].

**E. Sharp lobe.**
- Λ_θ is never asked to represent a sharp outgoing lobe. That was ECRO's failure: learned S1 at zero-operator level, op flux-L1 0.96–1.04 [MEAS WL95 §2.1].
- Every sharp emission is ER-continuous from incidence. Every sharp last flight is explicit with A's carrier.
- The e3_p family (light → A → panel → receiver) becomes: Λ_θ or explicit incidence at panel cells, then ER-continuous emission, then the D2 delivery. This gives the e3_p family an owner (it was WL95 reason 5).

**F. Dynamic reuse.**
- θ is fixed. Under a rigid edit:
  - the relation is selectively updated (bit-equal to full);
  - the generator regenerates Λ factors (dense, ≈ ECRO operator 11 ms);
  - A's dirty pairs and atoms are updated;
  - the first flight is recomputed.
- *Non-rigid:* rule reuse only. The generator's generalisation to deformation is untested, and earlier learned geometry→response maps failed deformation (WL68/71/73 DEF results [MEAS, AGENTS/memory index]).

**G. Complexity.**
- *Training:* ECRO-scale (32 ms/step [MEAS WL86:81]) plus the new tallies.
- *Per edit:* ≈ ECRO (76–78 ms) + A's dirty pairs [INF].
- *Per light:* A's atoms + 20,320 source rays + tensor applications; no relation rays.
- *Memory:* ≈ ECRO + A.
- *Avoided vs explicit H-0:* J sweeps for broad-last orders, replaced by a rank-ρ application (predicted ≈ equal at scene86 scale; possibly favourable asymptotically, O(N·ρ) vs O(N·K·J) [HYP]).
- *Introduced:* training, incident tallies, and the generator per edit.

**H. Failure modes.**
- Rank limits for caustic-lit broad surfaces re-illuminating sharp cells.
- Near-contact locality (WL52's learned-C failure: contact p99 2.3 vs 0.27 [MEAS AGENTS.md WL52]).
- Compensation if trained on totals; prevented only by exclusive tallies.
- Albedo compression (CK2 slopes 0.28–0.63 [MEAS WL93D/94]).
- Passivity of the Λ composition is not guaranteed by non-negativity; it needs a checked bound.
- Seed dependence, as in all WL92–94 learned smokes.
- α → 0, many lights and moving sharp objects are inherited from A.

---

## 5. Comparison

| | ECRO v0 / OWN | X1 / C3URX1 (H-0 lineage) | H-0 EXP / C3UR | A CIS-ER | B LIR |
|---|---|---|---|---|---|
| Sharp first-order carrier | learned rank-16 patch × bin | exact per ray | binned exact q | exact (x, ω) from control-point incidence | as A |
| Expected bias on CK2 | learned 0.52× / 1.63× flux [MEAS] | none (1.02) | +42 % (q) | ≤ 1 % at g4 [MEAS-S] | as A (e2) |
| Per-draw variance | fixed draw | heavy (median 0.62) | moderate | none (D2) | none (D2); Λ deterministic |
| Learned content | everything | none | none | **none** | broad-last ≥ 3 incidence |
| Per-light rays | 20 k | ≈ 1.3 M (64 draws) | 20 k | ≈ 1.2 k + ER evaluations | ≈ 21 k + ER evaluations |
| Per-edit | 76–78 ms [MEAS] | 125 ms GM1 [MEAS] | 89–95 ms [MEAS] | [HYP] ECRO-class + dirty pairs | [HYP] ECRO + A |
| Canon status | method base | comparator | comparator | comparator / oracle (G1) | ECRO-lineage candidate (G2 with a specified Λ) |

---

## 6. Why no recommendation [INF]

- **A** is learning-free, so it is inadmissible as the method under F10 pending Gate A. It is the carrier that B and every future learned design must reproduce.
- **B's learned utility** is predicted near zero at scene86 scale and is unmeasured. Its preconditions (oracle-incidence sufficiency, rank bound, explicit comparator) are untested.
- **No sharp-qualified held-out workload exists** (WL95 §4.5), and §2.3 shows non-grazing lights do not qualify.
- **All numbers in §2 are a simplified CPU model** of one face, one light and no occlusion.

---

## 7. Predeclared falsification conditions

**A**
- **FA1, representability.** At the rule-chosen g ≤ 4, rel L1 ≤ 0.03 and ratio ∈ [0.97, 1.03] against the dense deterministic oracle (DQ). Scope: box A (54 patches), the panel face, and every Lambert receiver with ≥ 1 % of S1 flux, on W-a and on every qualified validation light. Fail → report the g needed; the class is unrepresentative at the rule.
- **FA2, oracle contract.** DQ vs path-traced e2_p: |ratio − 1| ≤ 2 SE and rel L1 ≤ n0 + 2 SE. Fail → STOP (reference or contract defect, not a carrier verdict).
- **FA3, generalisation.** FA1 and FA4 hold on V1–V3 (§8). A failure on S2 only → scope "planar senders".
- **FA4, variance.** The *deployable single deterministic configuration* (chosen by a geometry-only rule before results) reaches rel L1 ≤ n0 + 0.03 vs e2_p. Passing only when pooled → VARIANCE UNRESOLVED (the WL91 pattern).
- **FA5, cost.** At matched delivery error (≤ pooled X1), per-light rays ≤ 0.25 × X1 (64-draw), per-light wall ≤ ECRO source-only (same machine) × 2, and per-edit wall ≤ ECRO per-edit + 25 %. Otherwise ECONOMICS NOT SUPPORTED.
- **FA6, classification.** A has no learned transport. It cannot be reported as a learned method, whatever FA1–FA5 show.

**B** (pre-training gates; no training in the next experiment)
- **FB1, oracle incidence.** Path-traced broad-last ≥ 3 incidence (4 × 4 cells × 64 bins at sharp patches) → ER emission → D2 must reproduce e3_p on box A and the panel face within n0(e3) + 0.05. Fail → the incidence carrier loses required information; B is rejected.
- **FB2, rank bound.** A non-negative rank-ρ ≤ 64 resolvent *fitted* (not learned) on the 96 training items must reach validation incidence rel L1 ≤ 0.15 and e3 delivery ≤ 1.5 × FB1's error on the 24 validation items and on CK2. Fail → the learnable form is infeasible at that capacity.
- **FB3, learned utility.** If explicit J = 3 sweeps (C3UR-type, same relation) reach the FB2 error at ≤ the bound's regenerate + apply cost for both per-edit and per-source → NO_LEARNED_UTILITY. B is not trained.
- **FB4, later batch only.** Held-out geometries, 2 seeds, passivity of the composed operator, and an ownership ablation showing no compensation of sharp-last error.

---

## 8. Smallest next feasibility experiment (no training)

**Batch class.** Owner decides per AGENTS.md:3-11. [INF] Stages 0–4 test a new carrier representation (CIS-ER), which arguably makes it a regular number. If the owner treats them as resolving WL95A's open question "whether any fixed-basis local carrier … removes the outgoing bias … without the exact estimator's variance", it is a suffix of WL95. Stage 5 (B's bounds) introduces a new learned hypothesis's preconditions, so it is regular.

**Stage 0: contracts.**
- Reproduce WL95A pooled q256 / X1c / X1 from the stored `deliv.npz` (read only).
- Dr.Jit ER vs the closure82 reference ≤ 1e-5.
- DQ oracle (exact ER at 2.5 mm sender cells × 8 × 8 receiver points, Mitsuba visibility) vs pooled X1 within 2 SE.
- CP∞ = DQ to ≤ 1e-6.
- The numpy model's predictions from this report are recorded as pre-registered expectations.

**Stage 1: spatio-angular representability (test 1).**
- CP ladder g ∈ {1, 2, 4, 8} on CK2 W-a with real visibility.
- Scope: all Lambert receivers (exact M64 receiver) plus a sender-field metric at continuous (x, ω) on view-relevant directions.
- g* is fixed by the geometry-only rule before results. Rule: FA1.

**Stage 2: physical oracle (test 2).** CIS-ER (g*, D2 no-cull) and DQ vs path-traced e2_p (CKF_none). Rule: FA2, then FA1.

**Stage 3: generalisation (test 3).** The validation set is frozen and prequalified from DQ alone, before any arm runs.
- **V1:** a grazing light on another panel region and azimuth. Qualification: cos θ_i ≤ 0.25 over ≥ 80 % of the panel, ≥ 50 % of S1-delivered flux on ≤ 10 % of Lambert patches, and SNR ≥ 4× against a new field_src 2 × 4,096 spp reference.
- **V2:** a moderate-incidence light satisfying the same concentration rule. Excluded and declared if none qualifies; §2.2 shows non-concentrated lights are uninformative.
- **V3:** S2 sphere as the sharp sender (curved).
- **V4:** two co-binned point lights (superposition contract; exact by linearity for point atoms).
- The existing E2 T1 region C reference is optional, with event tallies only if available.

**Stage 4: variance vs bias (test 4) and cost (test 5).**
- Arms on the same carrier:
  - D1-point (X1);
  - D1-footprint s ∈ {3, 6};
  - D2 g × r ∈ {4, 8} × {2, 4} with no cull and cull 1e-3;
  - D2-G (cached bitmask).
- D1 runs 64 relation draws with common random numbers; D2 is deterministic. Report pooled bias ± SE and per-draw median / p90 / worst. Rule: FA4.
- Cost:
  - a full-scale GPU-utilisation probe of every stage before freezing (standing rule);
  - warm, uncontended timings for: geometry build (bitmask), the WL86 rigid box edit (dirty pairs), a light change, and a frame;
  - ray counts, ER evaluations and MB.
- Comparators re-measured on the same machine: X1 (single draw and 64 draws) and ECRO OWN s8601 (edit and source-only). Rule: FA5.

**Stage 5: learned-utility preconditions for B (test 6). No network training.**
- (a) Implement the incident tally and check its consistency against field_src e3 on Lambert receivers.
- (b) FB1 on CK2.
- (c) FB2 rank ladder ρ ∈ {8, 16, 32, 64}: non-negative factorisation fitted on the 96 training items (new tallies), validated on the 24 validation items + CK2.
- (d) FB3 explicit J = 3 sweep comparator at matched cost.
- For A, test 6 is answered by construction (no learned component). Optionally, measure GPU ER-evaluation throughput to confirm a lobe compiler has no cost case.
- Stage 5(c) needs new path-traced tallies for 96 × 4 lightings. It is the most expensive stage and can be split off.

**Stop rules.** FA2 fails → stop everything. FB1 fails → skip FB2 and FB3.

**Not authorised by this design:** training, adoption, threshold changes after results, and any successor.

---

## 9. Scratch files (CPU only; not evidence of record)

All under `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\`.

**Scripts**
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\er_np.py` — numpy port of closure82 ER. Sanity check: S1 directional albedo ≤ 0.85, r50 0.13–0.165 rad.
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\ck2_model.py` — relation-draw estimators (§2.3).
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\ck2_model2.py` — noise-free carrier ladder and deterministic cells (§2.2, §2.3).
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\ext_incidence.py` — extended incidence (§2.4).
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\lobe_fit.py` and `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\lobe_fit2.py` — lobe capacity (§2.5).

**Results**
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\ck2_model_ck2.json`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\ck2_model_alt.json`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\ck2_model2_ck2.json`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\ck2_model2_alt.json`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\ext_incidence.json`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\lobe_fit.json`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d4\lobe_fit2.json`

**Model limitations (shared by every [MEAS-S] number)**
- one face of box A (9 patches, not 54);
- no occlusion;
- omni light;
- panel frame t = +y;
- ER ported, not the Dr.Jit runtime;
- the X1 pooled value in `ck2_model_ck2.json` is not converged at 64 draws (heavy tail);
- the extended-emitter scenario is a proxy, not a tallied e3_p.