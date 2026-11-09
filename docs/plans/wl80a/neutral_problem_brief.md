# WL80A — Neutral Problem and Evidence Brief (Track B input)

Prepared by the Coordinator on 2026-11-09. This is the only project document that Track B (blind architecture discovery) receives.

**Exclusions.** This brief deliberately contains no proposed architecture, no prior recommendation, and no causal explanation that was not directly measured.

**How to read it.** Statements are labelled:
- **FACT** — physical law or measured result;
- **CONTRACT** — a required semantic of the system;
- **OUTCOME** — a measured result of an approach that was tried.

An OUTCOME says what happened, not why.

## 1. Research objective

ReST-GI targets reusable light transport for dynamic global illumination. The scene's transport is split into two owned parts:

- **K (intrinsic, material-domain transport).** For each object or material domain, K maps light entering the domain through its boundary to the light that first exits the domain through its boundary. K depends on the domain's material and geometry. It should not depend on the rest of the scene, the illumination or the camera.
- **C (free-space transport).** Scene-dependent visibility, coupling between domains, and same-object re-entry through air. C is computed explicitly from the current scene. C is not part of this study.

**Goal.** K should be reusable when external illumination and scene visibility change, without retraining it.

**The open problem.** A learned K must predict accurate responses for **source positions (boundary entry locations and directions) that were not in its supervision**, on a finite body with boundaries, without memorising per-source responses. It must also be computationally practical as an operator that is applied repeatedly inside a global-illumination solve.

**Central question.** What representation should intrinsic light transport use to generalise to novel source positions, while respecting physical correctness and runtime constraints?

## 2. Physical problem (immediate test domain)

### Body (FACT)

| Property | Value |
|---|---|
| Shape | Homogeneous slab, 100 × 100 × 10 mm (finite: top, bottom and four side faces) |
| Extinction σ_t | 1000 m⁻¹ (mean free path ≈ 1 mm), all channels |
| Single-scattering albedo (R / G / B) | 0.999 / 0.99 / 0.9 |
| Phase function | Isotropic |
| Boundary | Index-matched (no refraction or Fresnel); a photon that reaches the boundary leaves |

The physical reference is a validated volume path tracer. It agrees with an analytic H-function to about 0.5 %.

### Discretisation (CONTRACT)

- **Ports.** The boundary is divided into P = 3,840 ports of 2.5 × 2.5 mm: 40 × 40 on top and on bottom, 4 × 40 on each side.
- **Angular bins.**
  - Incoming: 16 per port (a 4 × 4 concentric square-to-disk partition of the inward hemisphere).
  - Outgoing: 64 per port (8 × 8 of the outward hemisphere).
  - Bins are defined in a per-port tangent frame (t, b, n). At rest, that frame is constant on each face.
- **Coarse aggregates used in evaluation.**
  - "fine16": the 64 outgoing bins summed into 16 groups.
  - "coarse": ports summed into 240 cells of 4 × 4 ports on top / bottom and 4 × 1 on the sides, with the fine16 angular grouping.
- **Geometry changes.**
  - A "dent" deforms the top surface: depth up to 6 mm, radius 15–25 mm, at various centres. The bottom is fixed and the material is unchanged.
  - Rigid motion of the whole body is also allowed.

### Physical operator (CONTRACT)

- **Definition.** For a source port i and incoming bin a, the reference response is the outgoing flux S[i, a, j, o, c]: radiance-weighted power leaving receiver port j in outgoing bin o, channel c, per unit incident power. It counts light that has collided at least once inside the medium before its first exit.
- **Ballistic part.** Light that crosses the body without collision is carried by a separate, exactly computed ballistic operator B (geometry only). Its complement d(i, a) = 1 − β(i, a) is the collided budget of an incident bin.
- **Required properties of K.**
  - Linearity in the incident flux.
  - Non-negativity.
  - Passivity: total output ≤ input; per source bin, Σ_{j,o} K ≤ d(i, a) ≤ 1.
  - Zero output for zero input.
  - Correct source / receiver / bin correspondence.
- **Symmetries (FACT, verified in WL79).**
  - The rest slab has exactly 16 symmetry operations: the in-plane square group × top / bottom flip.
  - Under each operation, ports map to ports, and incoming / outgoing bins map by exact permutations derived from the port frames.
  - Reference responses of symmetry-related sources agree within Monte Carlo noise.
  - Dented poses are not exactly symmetric under the current per-port frame convention.
- **Reciprocity.** The physical medium is reciprocal, but the discretised port / bin operator need not be symmetric as stored.

### Supervision data (FACT)

- **References.** Monte Carlo, per (source port, incoming bin), 32M photons per packet, 4–8 independent replicas per source, each a tensor (16, 3,840, 64, 3).
- **Training population "S16-R".** 16 source ports, at rest and at one dented pose: 12 interior / face sources and 4 boundary-near sources (within 1.8–4 mm of a perpendicular face).
  - Fitting uses replicas 0–3; held-out evaluation uses replicas 4–7.
  - Three pairs of the 16 sources are exact symmetry images of each other.
- **Cost.** About 3.5 s of GPU time per packet, i.e. about 4 GPU-minutes per (source, geometry) for K = 4.
- **Population limit.** Supervising many sources is costly; the training population has been held at 16 sources.

### Evaluation conventions (CONTRACT)

**Errors.** Relative errors are unbiased cross-replica estimates, reported as worst-channel upper-95 % bounds (jackknife):
- `abs_native` — full port × 64-bin resolution;
- `abs_fine16` — 16 outgoing groups;
- `abs_coarse` — 240 cells × 16 groups;
- `con_*` — incoming-direction contrast: the deviation of the response from its mean over incoming bins, normalised by the collided budget;
- `con_coarse` — the key directional metric.

**Quality gates (historical).**

| Gate | Threshold |
|---|---|
| `abs_coarse` | ≤ 0.10 |
| `abs_fine16` | ≤ 0.15 |
| `con_coarse` | ≤ 0.25 |
| Deformation: relative error of the signed response change (deformed − rest) | ≤ 0.35 |
| Deformation: correlation | ≥ 0.90 |

**Independent confirmation populations.**
- New source ports chosen by geometry-only rules, excluding training sources and all their exact symmetry images.
- Stratified into interior (INT), boundary-near (BND_NEAR) and boundary-far (BND_FAR, far from any supervised position) contexts: top / bottom edges and corners, side middles, side rims, vertical edges.

**Practical constraints.**
- The operator is applied many times per global-illumination solve.
- Geometry changes require rebuilding any geometry-dependent state.
- Hardware: one 12 GB GPU for experiments.

## 3. Measured observations (FACT unless labelled)

### 3a. Physics of the responses

- **Locality of energy.** For interior sources, 5.5 / 3.2 / 0.1 % (R / G / B) of the collided output energy lands outside the 128 nearest receiver ports (≈ 13.5–17 mm); for boundary sources, 1–2 %.
- **Near-field dominance.** For boundary sources, 99.0–99.9 % of the response norm lies within 5 mm of the source.
- **Equivalence of contexts.** Two sources related by an exact symmetry plus an in-face translation, whose nearest boundary is identical and whose second-nearest boundary changes from 18.75 to 41–49 mm, have responses equal within noise:
  - near field: 0.002 relative (noise 0.002);
  - fine16 contrast: 0.005.
- **Real context effects.** Translations that change the nearest-boundary distance (e.g. 1.8 vs 21 mm) give clearly different responses (detected at ≥ 3 × noise).
- **Geometry change.** Dents change the responses measurably. A learned model must track signed changes against references.

### 3b. Outcomes of approaches tried (OUTCOME; no causal claims)

**Family-level outcome.** Every learned operator tested so far fits its supervised sources to the quality gates, or close to them. None meets the gates on independent boundary sources.

| Approach tried | Outcome |
|---|---|
| Graph-network operator (message passing on port neighbourhoods), absolute-coordinate inputs; spatial weights = local pairwise edge weights + a shared set of global receiver patterns; a fixed, globally shared set of incoming / outgoing angular profiles (8, later 16) | Supervised fit near the per-source oracle. Independent BND_FAR `con_coarse` ≈ 0.79–0.95. INT 0.32–0.55. Gate 0.25 |
| Same operator, absolute position / orientation inputs replaced by geometry-relative features (frame-expressed displacements, nearest-boundary probes) | INT `con_coarse` 0.12 (meets gate). BND_FAR 0.67 / 0.86 (two seeds; one 20 % better than the previous operator, one not). Deformation-response correlation worse (blue 0.77–0.85 vs 0.88–0.94) |
| Same as row 1, prediction consistency for symmetry-equivalent contexts | Predictions differ by 12–33 % native / 42–141 % fine16 contrast where references agree within 0.2 %. Shifting absolute-coordinate inputs by a constant changed boundary predictions by 10–23 % |
| Per-source oracle: spatial coefficients fitted freely per source, angular profiles shared and fixed (diagnostic, not deployable) | With 8 shared profiles, new boundary sources fit only to `abs_coarse` ≈ 0.11, `con_coarse` ≈ 0.27. 16 profiles roughly halve the contrast error and meet the gates on supervised sources |
| Same oracle on exact symmetry images of training sources | Boundary images fit 1.26–1.41 × worse in coarse metrics |
| Same oracle with profiles refitted across all symmetry images | Gap reduced to ≤ 1.15 ×; original-orientation accuracy 2–18 % worse; boundary rotations still slightly outside 1.25 × |
| Increasing the supervised population from S16 to S16-R (4 new boundary sources) | Supervised boundary sources at the oracle floor; nearby transfer improved about 3 ×; far transfer small or mixed |
| A direct coefficient map from geometry features to per-source spatial coefficients | Fails on new ports and on deformation |
| Exact-symmetry consistency penalty (on angle-integrated responses) during training | No improvement in independent transfer |
| Alternative relational spatial kernel | No improvement over the baseline on new ports. Side faces and corners dominate the errors |
| Operator anchored at a computed first-interaction distribution, with a learned angular redistribution | Factorisation itself exact. Learned / parametric angular redistribution showed azimuthal error floors 0.38–0.45 (gate 0.226) |
| (Labelled post hoc) Analysis of the row above | Those floors were confounded by single-scatter quadrature noise (q = 1,024 samples); noise-free estimates ≈ 0.07 |
| Training dynamics | Transfer to unsupervised sources is often best early in training (500–2,000 of 20,000 steps) and degrades later (up to 5 × in coarse error) while the supervised fit keeps improving |

### 3c. Cost observations (FACT)

| Item | Cost |
|---|---|
| Learned operator, application to a lighting state | ms-scale |
| Geometry-state rebuild | ~0.1–0.5 s |
| Parameters | ~270k learned |
| Explicit C rebuild | ~5 ms per geometry update |
| Exact per-source physical oracle recompute (full first-exit transport at 2.5 mm) | Minutes per configuration (190 s for a full update) |
| Direct path-traced re-rendering of a small medium workload, at matched quality | A few seconds |

ReST-GI's value depends on reuse being cheaper than re-rendering while being accurate.

## 4. Questions Track B must answer

1. What representation of K should give a credible, mechanistic reason to generalise to unseen source positions? Consider in particular boundary contexts never supervised.
2. What information must be explicit, and what may be learned?
3. How should it be conditioned on source, receiver and geometry? How should it transform under the exact symmetries and under geometry change?
4. Is it learnable from sparse source supervision (tens of sources), and at what computational cost?
5. What is the closest prior work? What, if anything, is plausibly new?
6. What minimal physical experiment would falsify it?

## 5. Allowed resources for Track B

- **Allowed:** this brief; general physics and graphics knowledge; public literature (web search).
- **Not allowed:** the ReST-GI repository, its documents, plans or worklogs, and any other document produced in this study.
