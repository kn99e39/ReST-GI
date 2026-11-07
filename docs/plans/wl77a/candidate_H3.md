# WL77A Candidate H3 — Learned-Closure Volumetric Resolvent (LCR) (2026-11-07)

**Role.** Candidate developer for family F3 in the WL77A architecture-discovery workflow (ideation: `docs/plans/wl77a/ideation.md` §3 F3).

**Status.** This is a hypothesis specification only. It does not adopt or authorise anything, and nothing here has been implemented, trained or run.
- Every number marked *estimate* is arithmetic or an extrapolation.
- Every number marked *non-evidence* comes from the specialists' independent CPU emulations.
- Every other number is cited to code or results in the repository.

**Rules followed.**
- Read-only on the repository, except for this file.
- No GPU job, training, server access or Monte Carlo.
- The Gate 4 confirmation population (`ports76.json`) and its references were not read.

**Distinctness from the other two candidates.**
- **H1/F1 (CQR)** learns the angular-spatial source→receiver block directly.
- **H2/F2 (RBG)** learns the long-range scalar resolvent Γ (≈ 15 mm) as a direct Gram kernel between exact projections, with no solve.
- **H3 learns no transport range at all.** The long-range scalar resolvent is never represented. It is obtained at apply time by solving a discretised *local generator*: the exact collision-probability kernel between volume elements, plus a learned *local closure* of one stencil range.
- In H3, all boundary, corner, second-edge and dent context enters as explicit geometry inside the solve, never as a learned context feature.

---

## 1. Problem

We want an intrinsic operator K for the qualified slab fixture that stays accurate for **source ports never seen in training**. That matters most near boundaries and far from the 16 supervised S16-R sources. The fixture is: 100 × 100 × 10 mm, σ_t = 1000 m⁻¹, ω = (0.999, 0.99, 0.9), g = 0, index-matched, P = 3840 ports of 2.5 mm, 16 incoming and 64 outgoing bins.

K must also respect K/C ownership and the operator constraints:
- linear, ≥ 0, passive, zero → zero;
- no source-identity lookup;
- trained on the S16-R supervision only.

Every learned family so far (WL59–WL77) has had transfer error that grows with distance from supervision:
- AR / A75 **BND4_FAR con_coarse 0.79–0.82**, against TRAIN_NEW 0.13–0.27 (`docs/worklogs/75_WORKLOG_RANK16_LEARNED_ATO_TRANSFER.md`, Stage 2 and Stage 3 tables);
- per-source G0 / G2 oracles 0.265 / 0.159 on BND4_FAR.

H3 asks whether transfer becomes **supervision-distance independent** when nothing long-range or source-related is learned. In H3, the only learned object is a local correction to a physically exact discretised transport generator.

## 2. Historical failure targeted

The consolidated causes are listed in ideation §1. H3 targets all six by construction. Its distinguishing target is **C5**: boundary context is regressed rather than composed.

| Cause | Historical evidence | How H3 removes it |
|---|---|---|
| **C5** boundary context regressed jointly, not composed: second-edge distance e₂ 38.75–46.25 mm at far ports vs 18.75 mm in training; near-corner 8.75 / 8.75; SV/UNS contrast 1.0–1.5 everywhere | `specialist_neural.md` R3; RSK scalar edge distance (`experiments/fnto_cmp72/rsk72.py:27-28`); FIA sorted free paths (`experiments/fnto_fia76/fia76.py:39-43`) | No learned function receives boundary context at transport range. Leakage at every face, edge, corner and dent appears as missing coupling in the exact kernel matrix (in-medium elements only), and the solve composes it exactly. e₂ is a real geometric distance inside the solve, not an input that can go out of range |
| **C1** identity channels: absolute xyz and face id as node inputs; global softmax U over all P; free per-source mass; late drift ×3–5 | `experiments/fnto_rule57/r57_geometry.py:58`, `r57_model.py:61-63`; WL71A/71B | The learned closure receives only local element-pair geometry. It is evaluated identically on about 10⁵ translation-equivalent elements, so it has no capacity to store a per-source table |
| **C2** symmetry not quotiented | WL74A §3; WL72 (angle-integrated augmentation) | The grid, the cubature rules and the closure features are D4h-symmetric and port-lattice translation-equivariant. Frames are irrelevant to the learned part: angular ends are computed in each port's own frame |
| **C3** incidence only through the global CP pairing with angle-free T (sub-cell first-collision shift 0.3–0.8 mm) | WL74A; `specialist_physics.md` §2 | Exact sub-port first collision in D1, and exact second-collision arrivals in b̄ |
| **C4** noisy deployable quadratures (q = 1024 one-ray SS) | `fia76.py:79`; `experiments/fnto_fia77/gate77.py:45-48` (non-evidence attribution) | Deterministic cubature is required for every geometric term (§4). H3 does not rely on the C4 reading being true |
| **C6** learned angular ends | FIA R_θ ring lobes; ATO A/O atoms | Both angular ends are exact geometric escape projections, and nothing angular is learned |

## 3. Mathematical operator definition

### 3.1 Semantics (unchanged fixture contract)

**Notation.**
- Input cell X = (i, a): port i, incoming bin a ∈ 16. Unit flux enters, spread uniformly over the port area and over projected solid angle inside the bin. The bin labels the outward direction the light comes from (`experiments/phys_ref53/p53_tracer.py:90-101`).
- Étendue g_X = A_i π / 16.
- Output cell Y = (j, o): port j, outgoing bin o ∈ 64, RGB flux.
- K_c(X→Y) = B(X→Y) + K_scat,c(X→Y).

**B** is the existing geometry-only ballistic matrix (`r57_geometry.ballistic_matrix`), unchanged and shared by every family. The metrics evaluate K_scat against collided references, with d = 1 − β_B.

**Exact continuum identity** (fixture: g = 0, homogeneous, index-matched; `specialist_physics.md` §3.2; Stage-1 decomposition of WL76, 36/36):
- Let e_Z(x) = (1/4π) ∫ e^{−σ_t t(x,ω)} 1[ray x → ω leaves the medium first through cell Z] dω. This is the uncollided escape kernel for an output cell, or for an input cell read as an exit cell under the same outward label.
- Let S(x,y) = σ_t e^{−σ_t|x−y|} / (4π|x−y|²) · vis_med(x,y) be the next-collision kernel. vis_med is in-medium line of sight: segments through air belong to C.
- The first-collision density of X is c1_X = (4πσ_t / g_X) e_X.
- With G = (I − ωS)⁻¹ and the identity G = I + ωS + ω² S G S, the étendue-weighted scattered operator is

  𝒦_c(X,Y) := g_X K_scat,c(X→Y) = 4πσ_t [ ω D1(X,Y) + ω² D2(X,Y) + ω³ ⟨S e_Y, G S e_X⟩ ],

  with D1 = ⟨e_Y, e_X⟩ (single scatter) and D2 = ⟨e_Y, S e_X⟩ (double scatter).

All three terms are symmetric in (X, Y), so reciprocity A_i K16(i,a→j,b) = A_j K16(j,b→i,a) holds with no label reversal.

### 3.2 LCR discretisation

**Grid 𝔊** (deploy). A fixed-topology tensor grid in **material coordinates**, mapped to the current pose by the fixture's exact volumetric deformation map (`p53_geom.map_dent`, line 168, linear in z):
- **x and y:** a 2.5 mm boundary band of five 0.5 mm cells at each side, then 76 interior cells of 1.25 mm, aligned to the port lattice (2 × 2 cells per port). That gives 86 cells.
- **z:** four 0.5 mm layers at each face and six 1 mm core layers. That gives 14 cells.
- **Size:** N = 86 · 86 · 14 = **103,544** elements (computed).
- **Symmetry:** the grid is D4h-symmetric and invariant under port-pitch translations along each face. Every face, edge and corner has ≤ 0.5 mm resolution normal to each nearby face, as `specialist_physics.md` §7 item 4 requires.

**Exact geometric quantities**, all deterministic cubature with no learned part:

| Symbol | Definition | Role |
|---|---|---|
| v_m | element volume | mass matrix V = diag(v) |
| τ̂_mn | ∫_{V_m}∫_{V_n} S(x,y) dx dy (symmetric, ≥ 0). Exact for pairs within 4 mm; 4–12 mm through a two-level Galerkin part R^T τ̂_c R (coarse cells 2.5 × 2.5 mm, 4 z-layers; symmetric, ≥ 0); beyond 12 mm dropped (free-flight survival ≤ e⁻¹² ≈ 6 × 10⁻⁶, reported) | discretised generator |
| D1(X,Y) | ⟨e_Y, e_X⟩ by semi-analytic cubature: exponential depth integration along the entry chord, product Gauss over port area × bin, planar-triangle exit integration with in-medium first-crossing tests | exact single scatter, sub-port first collision |
| D2(X,Y) | ⟨e_Y, S e_X⟩ on a fine auxiliary boundary layer 𝔉 (≤ 2 mm deep; 0.25 mm depth × 0.625 mm lateral), as Σ_f [∫_f S e_X]·ē_Y(f), then symmetrised: D2 ← ½(D2 + D2ᵀ) | exact up to fine-layer projection, symmetric by construction |
| b̄_Z[m] | ∫_{V_m} (S e_Z)(y) dy, for input (16-bin) and output (64-bin) cells | exact *second-collision arrival* per element. The first collision keeps its sub-port position |

**Learned closure.** δ_θ: features → ℝ³, one output per channel. It is defined on the closure stencil 𝒩 = {(m,n): m = n or the elements are face/edge/vertex adjacent}, about 27 pairs per element:

  τ̂_θ,c,mn = τ̂_mn · (1 + c_max tanh δ̄_θ,c(m,n))  for (m,n) ∈ 𝒩,   τ̂_θ,c,mn = τ̂_mn otherwise,

where:
- δ̄(m,n) = ½[δ(f_mn) + δ(f_nm)] (exact symmetry) and c_max = 0.5;
- a symmetric cap (rescale every entry by min(γ_m, γ_n)) enforces max_n Σ_m τ̂_θ,mn / v_n ≤ 1 − ε_ρ, with ε_ρ = 10⁻³.

**Operator.**

  L_θ,c = V − ω_c τ̂_θ,c   (symmetric; Z-matrix; strictly diagonally dominant in the V-similarity ⇒ SPD nonsingular M-matrix, L⁻¹ ≥ 0)

  𝒦̃_c(X,Y) = 4πσ_t [ ω_c D1 + ω_c² D2 + ω_c³ b̄_Yᵀ L_θ,c⁻¹ b̄_X ]

  𝒦_c(X,Y) = s_X s_Y 𝒦̃_c(X,Y),   K_scat,c(X→Y) = 𝒦_c(X,Y) / g_X.

- **Passivity certificate.** s_Z = min(1, (1 − β_Z) / colsum_Z) is computed once per geometry and channel. Here colsum_X = Σ_Y 𝒦̃(X,Y)/g_X, which needs only ζ = L_θ⁻¹ Σ_Y b̄_Y, i.e. one solve. An output cell inherits s from its 16-bin parent.
- s is a deterministic function of geometry and θ. It is computed identically for all 61,440 cells, so it is not a per-source table.
- The 4 mm near radius and 12 mm far radius are geometric discretisation choices, fixed before any evidence.

**Discretisation consistency.** At θ = 0, the third term is the Ritz–Galerkin (indicator-basis) approximation of ⟨Se_Y, G Se_X⟩. It converges to the exact value as the grid is refined. **θ = 0 is a learning-free integral-transport solver**, the collision-probability form of the Peierls equation. That solver is an explicit control (§14).

### 3.3 Guarantees and how they are obtained

| Property | Mechanism | Exactness |
|---|---|---|
| Linear in illumination; zero → zero | All matrices are fixed per geometry. The deployed apply uses a **fixed-degree Chebyshev iteration** p(L) with certified spectral bounds, so it is a fixed polynomial in L (CG's Krylov coefficients depend on the right-hand side, which is why it is used only in training) | exact (deployed) |
| ≥ 0 | D1, D2, b̄ ≥ 0 (integrals of nonnegative kernels); τ̂_θ ≥ 0 (1 + c_max tanh > 0); L_θ⁻¹ ≥ 0 (M-matrix); s ≥ 0 | exact for the operator. The deployed polynomial p(L) ≈ L⁻¹ has relative error ε ≤ 10⁻⁵, so negatives are possible only at the ε level. A monotone Neumann "safe mode" (nonnegative, ≤ exact) is the declared exact-positivity fallback at about 7× cost (260 vs 38 iterations, red; *estimate*) |
| Passive (Σ_Y K ≤ 1 − β_X) | Certificate s, computed on the deployed p(L) with a margin of ε | exact by construction |
| Spectral radius < 1, convergent solve | Column cap 1 − ε_ρ on τ̂_θ V⁻¹ | exact |
| Reciprocity (no label reversal) | 𝒦 symmetric: D1 symmetric by definition and cubature; D2 explicitly symmetrised; b̄ is the same map on both ends; L symmetric; s_X s_Y symmetric | exact by construction (B excluded; it is the shared historical MC quadrature) |
| PSD of the étendue-weighted scattered part | D1 and b̄ᵀL⁻¹b̄ are Gram forms; s is a congruence; D2's PSD is inherited from S being PD only up to cubature | exact for the D1 and k ≥ 3 terms; measured for D2 |
| K/C ownership | vis_med on every S, D1, D2 and b̄ integral. Cross-dent air segments are excluded and stay in C, consistent with the walker's first-crossing exit (`experiments/fnto_rule57/r57_fastwalk.py:23-38`). The quadratic-form argument ⟨q, L_tr⁻¹ q⟩ = ½∮_out (Ω·n) ψ² + σ‖ψ‖² > 0 (vacuum inflow, i.e. no re-entry) keeps S PD on non-convex dented media | exact |

## 4. Representation and information ownership

| Information | Owner | Source-dependent? | Status |
|---|---|---|---|
| B (ballistic) | geometry quadrature, unchanged | yes | existing, shared |
| D1 single scatter, sub-port incidence footprint (depth ≈ cos θ mm, lateral shift ≈ sin θ mm) | deterministic geometry | yes (row of X) | explicit |
| b̄_X second-collision arrival histogram | deterministic geometry | yes | explicit |
| D2 double scatter | deterministic geometry on 𝔉 | yes | explicit |
| b̄_Y, ē_Y exit maps (all 64-bin outgoing structure) | deterministic geometry | no (receiver side) | explicit |
| τ̂ collision probabilities between elements, including leakage at faces, edges, corners and dents, and in-medium visibility | deterministic geometry + σ_t | no | explicit |
| Multiple-scatter resolvent G | **not stored**; obtained by solving L_θ y = b̄_X | no | computed |
| Subgrid closure δ_θ (the error of the indicator-basis discretisation in the sub-mfp boundary layer and in steep near-source gradients) | **learned**: one small MLP, about 6–10 k parameters (*estimate*: about 30 invariant inputs, 2 × 64 hidden, 3 outputs) | **no** | learned, shared, local |
| Passivity scales s | derived (geometry + θ) | no | computed |
| σ_t, ω_c, g = 0, homogeneity | **declared physical constants** | — | Assumed known. B and FIA already use σ_t, and FIA uses ω. The known scalar kernel S (g = 0) is an additional declared assumption beyond W+UV |

**Privileged labels.** None beyond S16-R. Reciprocity is built in, so no derived adjoint labels are needed.

## 5. Source-conditioning mechanism

A source enters only through **exact geometric rows**: D1(X,·), D2(X,·) and b̄_X. No learned function receives a source index, a source position, a bin label or an angle.

**Incidence dependence (contrast)** is *computed*:
- **In D1**, at continuous sub-port positions. This carries the 0.3–0.8 mm footprint shift that WL74A identified as the coarse-contrast mechanism.
- **In D2**, at 0.25 × 0.625 mm on 𝔉.
- **In b̄_X**, at element resolution after two exact flights. The arrival count per element is exact. Only the within-element arrival position is discarded before the third flight.

This follows the physics evidence that incidence information "lives mostly in the exact first-collision and single-scatter terms" (`specialist_physics.md` §7 item 3, non-evidence: snapping only the k ≥ 2 collisions to 2.5 mm columns leaves pair-contrast error 0.05–0.08; LCR snaps only k ≥ 3, at 1.25 mm × 0.5 mm).

**Outgoing dependence** is the exact escape projection (b̄_Y, ē_Y, D1). This covers the first-harmonic receiver-relative dipole that `specialist_physics.md` §6.2 finds dominant (order ≤ 1 leaves 0.06–0.07 with exact SS).

## 6. Geometry-conditioning mechanism

1. **Grid.** Fixed topology. Element vertices = material coordinates mapped by the current deformation (`map_dent` is a volumetric map, so the boundary layers stay surface-conforming and are only thinned: layer thickness × (1 − depth·b(r)/h)). Rigid edits change nothing in material coordinates, so K is reused exactly.
2. **Generator.** τ̂ near and far, b̄, D1, D2 and ē are recomputed from current geometry only where element geometry changed. That is the dent footprint plus the 4 mm near / 12 mm far support.
3. **In-medium visibility.** Ray tests against the current refined mesh (`KO.refined(m,V,4)`, as used in `r57_geometry.py:46`), only for pairs whose segment can approach the deformed top surface.
4. **Closure features.** f_mn uses *current* element geometry only. All features are invariant under translations and D4h:
   - per element: layer thicknesses and lateral sizes, volume, the sorted distances to the three nearest boundary planes and the cosines between their normals, local through-thickness along the inward normal, and normal deviation (dent curvature proxy);
   - per pair: centre distance, and the signed offset components along the nearest planes' normals, ordered by element m's plane sort.

   There is no absolute coordinate, face id, pose descriptor or global feature. By contrast, the W+UV node feature set includes the absolute coordinates and the face id (`r57_geometry.py:58`) and a global dent descriptor (lines 60–62).

## 7. Why transfer to unseen sources is expected

**(a) What differs for an unseen source is exact.**
- Two sources differ only in their geometric rows (D1, D2, b̄_X). These are computed exactly, wherever the source sits.
- The operator acting on those rows (L_θ⁻¹, the exit maps) is the same operator for every source. The far-boundary sources therefore see the identical solved operator that the supervised sources were fitted through.
- **BND4_FAR translates.** s101, s105 and s107 are exact rest-pose translates and reflections of supervised s50, s52 and s53 (`specialist_physics.md` §5.3; ideation §0 verified `ports4.json`). Because LCR is exactly translation- and D4h-equivariant, its predictions for them equal the translated predictions for the supervised partners.
  - They differ only through the genuine corner influence, about 1 % (*estimate*), which the solve contains physically.
  - So their error equals the training error up to that physical difference.
- **Far planes.** e₂ = 41–46 mm needs no extrapolation. It is the actual position of the far edge in the solve, and red leakage there is ≲ 0.2 % of exit beyond 25 mm (`specialist_physics.md` §2).

**(b) The learned object is local and densely supervised.**
- δ_θ acts on one-stencil pairs and is applied identically at about 10⁵ elements.
- Every collision site of every training source supervises it: about 22 / 17 / 5 weighted collisions per photon (R/G/B), over footprints holding 97–99 % of exit within 15 mm (`specialist_physics.md` §2).
- The local-geometry classes are: flat-face layers, top-face edge rows (s50 TBE), side faces (s52 SM, s53 SR) and a top corner (s51, port 3823, TBC class; `experiments/fnto_src73/protocol` `new_ports`). All of them occur within the transport range of S16-R sources, on rest and pt0.

**(c) The error is bounded where supervision is weak.**
- Some classes are seen only weakly: vertical side-side edges, bottom corners, and dent slopes steeper than pt0's.
- There, |closure| ≤ c_max = 0.5 of a one-stencil coupling, and the zero point is the θ = 0 solver. The θ = 0 solver's error is source-independent in kind and measurable without learning.
- This is the structural reason the WL71A drift pattern cannot recur as a ×3–5 transfer loss: the null space contains no per-source or per-region degrees of freedom.

**(d) The expected learning-free floor is small** (*non-evidence extrapolation*):
- `specialist_physics.md` §7 measured piecewise-constant 0.5 mm voxels for the k ≥ 2 field, with SS exact, at native 0.016–0.018, coarse 0.008–0.017, energy +2 % and pair-contrast 0.038–0.045. The edge and side variants gave 0.017 native.
- 1 mm voxels gave +6 % and 0.05–0.07.
- 𝔊 is 0.5 mm normal to faces, 1.25 mm laterally and 1 mm in the core, and grids only k ≥ 3.
- The predicted θ = 0 floor on 𝔊 is therefore **native 0.02–0.06, coarse ≤ 0.03, con_coarse ≤ 0.15**. These are below the WL59 gates (abs_coarse 0.10, abs_fine16 0.15, con_coarse 0.25) and below the G2 per-source oracle on BND4_FAR (0.159).
- The closure's job is to remove the remaining few-percent subgrid bias, not to supply transport.

**Signature.** Unseen-source error does not depend on distance to supervision: BND4_FAR ≈ BND4_NEAR ≈ INT4 ≈ TRAIN_NEW within seed and replica spread. Every historical family violates this signature by a factor of 3–6.

## 8. Physical validity argument

- **Exactness of the factorisation.** The identity in §3.1 is exact for this fixture.
  - The WL76 Stage-1 decomposition (first collision + collided response) reproduced the archived references in 36/36 packets, covering interior, edge, corner, side, vertical-edge, rest, pt0 and te0 sources (`docs/worklogs/76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md` lines 82, 86, 216).
  - LCR adds one more exact split (G = I + ωS + ω² SGS) and the Galerkin discretisation of an SPD operator.
- **Conservation.**
  - Each element's emission is either collided (Σ_m τ̂_mn / v_n) or escaped. Escape is not stored. It is implied by the exit maps and certified by s.
  - At θ = 0, energy differs from exact only by discretisation, with ≤ 6 × 10⁻⁶ per flight dropped beyond 12 mm.
  - With the closure, the column cap keeps ρ < 1, and s enforces Σ_Y K ≤ 1 − β exactly.
- **Positivity, reciprocity, PSD and passivity.** By construction; see §3.3.
- **Symmetry.** Exact D4h and port-lattice translation equivariance on rest:
  - the grid and the cubature point sets are symmetric;
  - the angular ends are integrated in each port's own frame, so the per-face frame non-equivariance (WL74A §3) is handled exactly rather than learned.
- **Dents.** Geometry enters the solve exactly:
  - thinning of boundary layers;
  - 32° maximum surface tilt for te3 (slope depth·π/(2R) = 6·π/30 ≈ 0.63; pt0 5.75·π/45 ≈ 0.40; *computed from POSES*);
  - in-medium visibility.

  Only the closure's response to sheared elements is learned, and only from pt0.
- **Scope limit.** The mechanism is exact only for the fixture's scalar collision kernel. With g ≠ 0, Fresnel boundaries, heterogeneity or path-history-dependent BSSRDF references (WL51), the collision density is direction-dependent and LCR needs angular state.

## 9. Capacity and information bottlenecks

1. **Discretisation floor.** The k ≥ 3 field is piecewise constant at 1.25 mm laterally and 0.5 mm in depth. Errors from within-element field shape that the closure cannot express are the irreducible floor of 𝔊. Size: *estimate* 0.02–0.06 native.
2. **The closure has no source input.** It cannot correct errors whose sign depends on the source-relative field shape inside an element. An element adjacent to the first-collision footprint holds a steep sub-element gradient; the same element far from any source holds a smooth one. The exact second-collision arrival moves most of that steepness out of the gridded part, but not all of it. This is the main counterexample (§13).
3. **Supervision of the closure classes.** All 512 S16-R input cells come from 2 poses.
   - Vertical edges, bottom corners and the bottom face are supervised only through light reaching them from top or side sources.
   - Dent slopes steeper than pt0's (te3) are outside the training range of the closure features.
   - The bound in §7(c) applies.
4. **Geometric term accuracy.** D1, D2, b̄ and τ̂ must have cubature error ≪ the target: ≤ 0.01 native, and ≤ 0.3 of the contrast signal of 1–3 % of response energy (`specialist_neural.md` requirements). Cubature errors are invisible to the closure. Training cannot fix them, and it could warp the closure to compensate on training sources.
5. **Solver tolerance.**
   - The Chebyshev degree is set for ε = 10⁻⁵: about 38 / 33 / 17 iterations (R/G/B). κ is about 38 / 29 / 8, using 1 − λ_max ≈ B²/3 with B² = (π/11.42 mm)² from the lowest slab mode (`specialist_physics.md` §2). *Estimate.*
   - CG in training stops at a relative residual of 10⁻⁶. Implicit gradients are exact only up to that tolerance.
6. **Information not used.** Reciprocity adds no information here; it is built in. No far-boundary information is missing for the three translated BND4_FAR sources. Corner-offset contexts such as s103 rely on the θ = 0 physics plus the closure's corner class (s51).
7. **Material generality.** The learned component carries almost no transport information. That is the source of LCR's transfer argument and also its weakness as a *learned K*: for materials whose kernel is unknown, nothing here learns it.

## 10. Approximate computation and state cost

All entries are *estimates* (arithmetic from the counts above and `specialist_systems.md` §8 budgets), unmeasured, for the RTX 3080 Ti fp32.

| Item | LCR estimate | Historical | Budget (pref / cap) |
|---|---|---|---|
| Elements N | 103,544 (computed) | — | — |
| Near generator nnz (≤ 4 mm) | 150–220 neighbours per element ⇒ 1.6–2.3 × 10⁷. Structured stencil on the fixed tensor topology, so no index storage. Translation-unique at rest; explicit fp32 coefficients ≤ 90 MB | W+UV W 11.8 M | — |
| Far two-level (4–12 mm) | 6,400 coarse cells × about 720 ⇒ 4.6 × 10⁶ nnz (≈ 37 MB) + R (N) | — | — |
| Closure deltas | 3 ch × N × 27 ≈ 8.4 × 10⁶ (34 MB) | — | — |
| D1 (single scatter) | 61,440 × about 830 ⇒ 5 × 10⁷ nnz explicit (0.4 GB). Translation-keyed blocks at rest plus explicit blocks only in deformed footprints (*about 10–15 %*) | FIA SS 448 MB (37 M nnz) | — |
| b̄ in/out, 𝔉 maps (D2) | explicit worst case about 1.5–2 × 10⁸ nnz (1.2–1.6 GB); translation-keyed realistic ≤ 0.2 GB | — | — |
| **Persistent state** | **0.3–0.6 GB** translation-keyed; **≤ 2 GB** fully explicit | W+UV 15–30 M floats (0.06–0.12 GB); FIA 1.5 GB | ≤ 0.5 GB / ≤ 2 GiB |
| 1-light apply | about 38 iterations × (2.5 × 10⁷ MAC × 3 ch) ≈ 2.9 × 10⁹ MAC. As a structured stencil: compute-bound, **≈ 2–5 ms**. As generic CSR: memory-bound (0.2 GB per iteration) ≈ 10 ms, **over the cap**. A two-level preconditioner (an implementation choice that leaves the operator unchanged) might cut iterations to 10–15 | AR ≈ 1 ms; FIA 3.9 ms | ≤ 2 / ≤ 5 ms |
| 64-light apply | about 1.9 × 10¹¹ MAC (solve) + about 3 × 10¹⁰ (D1/D2/b̄) ⇒ **0.05–0.25 s** depending on kernel quality | A75 0.086 s; FIA 0.349 s | ≤ 0.26 / ≤ 0.86 s |
| Rigid rebuild | **0** (material coordinates) | 0 | 0 |
| Nonrigid: learned coefficients | closure MLP on about 7 × 10⁵ footprint stencil pairs + certificate solve ⇒ **≈ 10–30 ms** | W+UV 15 ms | ≤ 0.1 / ≤ 1 s |
| Nonrigid: geometry terms (pt0-size footprint, ≈ 2.5 × 10⁴ elements, about 450 ports) | τ̂: about 5 × 10⁶ pair integrals (about 10⁸ kernel evaluations + ≤ 10⁷ visibility rays) ≈ 0.05–0.3 s. D1 / D2 / b̄ for about 7,200 input and about 28,800 output cells ≈ 0.2–2 s. **Total 0.3–2.5 s; may exceed the 1 s cap** | shared geometry + B 0.3–0.8 s; FIA 75 s | ≤ 0.5 / ≤ 1 s (≤ 32 M casts) |
| Training step | 512 cells × 3 ch on per-source 50 × 50 mm windows (about 2.6 × 10⁴ elements). Warm-started CG about 10 iterations + one adjoint solve ⇒ about 2 × 10¹¹ MAC ⇒ **0.2–0.5 s/step**. About 2–5 k steps for a 10⁴-parameter closure | AR 0.2 s; FIA about 1 s | ≤ 0.5 / ≤ 1 s |
| Training memory | targets 3.7 GB/lane (historical layout) + solver vectors about 1.5 GB (windows) + dense native prediction about 1.5 GB ⇒ **one lane** (≤ 10 GB) unless support-restricted | AR peak 3.82 GB | ≤ 5.1 GB per lane, two lanes |

**Selective invalidation.** Footprint + 4 mm near / 12 mm far support. There is no global normalisation; the certificate is one solve. This is consistent with the systems requirement against global softmaxes (`specialist_systems.md` §8).

## 11. Nearest prior art

Agent 4 verification levels (`specialist_priorart.md`) are given in brackets. Items marked **[mine]** come from my own knowledge, were not checked by Agent 4 and are unverified here.

- **First-collision source method** (neutron transport; RAY3D/DONTRAN3D) **[V-sec]**: exact uncollided term → first-collision source → collided S_N solve. LCR follows it, with an integral-transport collided solve in place of S_N.
- **Collision-probability method (CPM) for the Peierls integral equation** **[mine]**: reactor-lattice codes discretise integral transport into regions with collision probabilities P_ij. They obey the reciprocity V_i Σ_i P_ij = V_j Σ_j P_ji and conservation with escape. **This is exactly τ̂.** LCR's generator is a CPM matrix on a boundary-adapted mesh.
- **SPH / superhomogenisation equivalence factors** (Hébert) **[mine]**: correction factors on coarse-region cross sections so that a coarse transport calculation reproduces reference reaction rates. This is the closest precedent for "a closure correcting a coarse transport discretisation". LCR's closure is a learned, geometry-conditioned, stencil-level version.
- **Learned discretisations and solver-in-the-loop closures** (Bar-Sinai et al. PNAS 2019; Um et al. NeurIPS 2020; Kochkov et al. PNAS 2021) **[mine]**: learned local corrections to coarse PDE solvers, trained through the solver. These are a class precedent for H3's learning principle.
- **Implicit layers / DEQ** (Bai et al. 2019) **[mine; listed in ideation]**: implicit differentiation through a fixed point.
- **Element Learning for RTE** (Du & Stechmann, arXiv:2308.02467) **[V, arXiv]**: learned per-element in2out / in2sol maps assembled by a global solver. Closest structural precedent for "learn locally, solve globally"; there, the learned object is the element response, not a closure of an exact generator.
- **Filtered MgNet** (arXiv:2604.23265) **[S, snippet]**: learned multigrid RTE sub-operators, reported to degrade in transport regimes. This is a warning for any learned solver component; LCR keeps the transport kernel exact.
- **DeepRTE** (arXiv:2505.23190) **[V abstract]**: Green's function from delta inflows. In contrast, H3 does not learn the Green's function.
- **Lattice-Boltzmann Lighting** (Geist et al., EGSR 2004) **[V]**: lattice transport for lighting, not learned.
- **H-matrix / two-level far field** (Hackbusch 1999) **[V as a class]**: the 4–12 mm coarse part.
- **Jensen et al. 2001 / Photon Beam Diffusion** **[V-sec]**: exact single scattering plus a multiple-scattering kernel.

## 12. Plausible novel contribution

**Mechanism-level novelty is weak; the contribution is a combination plus an evaluation.**
- **Not novel:** CPM generator, first-collision source, Galerkin solve, SPH-like closure, learned solver closures, implicit differentiation. None should be claimed.
- **Plausibly not covered as a combination:**
  1. the symmetric two-collision-exact Gram form 𝒦 = 4πσ_t[ωD1 + ω²D2 + ω³ b̄ᵀL⁻¹b̄], which gives a **boundary-to-boundary port × bin operator** with exact reciprocity, PSD (D1 and k ≥ 3 terms) and passivity by certificate;
  2. a **learned, geometry-conditioned, D4h- and translation-invariant stencil closure** of the CPM generator, trained through the solve from 16 one-hot boundary sources;
  3. the **held-out sparse-source boundary transfer** evaluation with full 16 × 64 port/bin correspondence. That is open space O1 in `specialist_priorart.md` F10, a weak negative from search depth.
- **The defensible scientific claim** would be empirical: "supervision-distance-independent source transfer arises when the learned part of K is restricted to a local closure of an exact generator". If the learning-free θ = 0 control already achieves it, the claim belongs to the classical solver, not to learning (§13, §14).

## 13. Strongest potential counterexample

**The "unnecessary or source-conditioned" dilemma.**

**Horn A: learning is vacuous.**
- If the θ = 0 solver on 𝔊 is already within the WL59 gates and near replica noise on unseen sources, which the non-evidence extrapolation in §7(d) suggests, then the learned closure contributes nothing to transfer.
- H3 then reduces to a deterministic CPM transport solver with exact geometric projections: "an explicit solver hidden under a neural name".
- The FNTO contract explicitly warns against this:
  - `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` line 17: "do not replace learned complex shading with a limited explicit radiosity solver";
  - line 86: "not an exhaustive explicit per-surface full path tracer hidden under a neural name";
  - `docs/Architecture.md` line 94: "A free-space radiosity solver … is not an interchangeable implementation of this contract" (it names a free-space solver; an intrinsic CPM solver is analogous, not named).
- The result would answer the research question ("what must be explicit") with "all of the transport; nothing needs learning". That is informative, but it is not a learned K.

**Horn B: the needed correction is source-conditioned.**
- If 𝔊 is coarse enough that the closure matters, the dominant subgrid error may sit in elements adjacent to each source's second-collision arrival peak, where the sub-element field is steep.
- There, the correct correction depends on the source-relative position of the steep gradient (and on its sub-element alignment at edges, where the port lattice and the 0.5 mm bands do not coincide). A source-independent stencil closure cannot represent this.
- Fitting it on S16-R would push δ_θ toward a compromise that helps the training footprints and *hurts* unseen ones: the old supervision-local pattern, at small amplitude.

**Scope horn.**
- Even if both horns fail, the mechanism needs the scalar collision kernel to be known.
- For the Lego Yellow Burley medium (path-history-dependent reference, WL51), anisotropic phase or Fresnel boundaries, it has nothing to solve with unless angular state is added. Angular state costs about 2 × 10¹⁰ floats for uniform sub-mfp grids (`specialist_systems.md` §9).
- So H3's transfer property may be specific to this fixture.

## 14. Minimal experiment capable of falsifying it

Fixture, references and metrics are unchanged (eval68 estimators, worst-channel jackknife upper 95 %). Only S16-R (rest, pt0; fitting replicas 0–3; scoring replicas 4–7) supervises. All other populations are evaluation-only.

**Stage 0 — geometry-term validity** (no learning, no new photons).
- (i) Cubature convergence: doubling every rule order changes D1, D2, b̄ and τ̂ by ≤ 10⁻³ relative in native and fine16 and in coarse contrast, on the 32 S16-R input-cell rows.
- (ii) D1 against the archived single-scatter tallies (WL76 Stage-1 style z² tests) and against FIA's q = 1024 SS. This also gives a direct read on C4.
- (iii) Exactness tests:
  - linearity and zero input;
  - min ≥ −ε;
  - Σ_Y K ≤ 1 − β;
  - A_i K16(i,a→j,b) = A_j K16(j,b→i,a) to float precision;
  - PSD of sampled 𝒦 blocks (D1 and k ≥ 3 exact; D2 reported);
  - translation equivariance: LCR(s107) vs translated LCR(s53) on rest, equal up to the solve's own corner difference.
- **Stop:** any failure.

**Stage 1 — learning-free physics floor** (decides whether the premise holds).
- Evaluate θ = 0 on 𝔊 and on a windowed refined control grid 𝔊_ref (0.25 mm depth bands, 0.625 mm lateral; 4 mm near radius on a ±25 mm window per source) for:
  - TRAIN_NEW / TRAIN_KEEP on held-out replicas 4–7;
  - the WL74 Stage-4 groups BND4_FAR / BND4_NEAR / INT4 and DEF4, whose archived references already exist (WL75: 1,408 packets).
- **Falsifier F1 (physical premise):** 𝔊_ref at θ = 0 fails any WL59 gate (abs_coarse ≤ 0.10, abs_fine16 ≤ 0.15, con_coarse ≤ 0.25) on any group, or DEF4 rel > 0.35 / corr < 0.90. The exact-geometry Galerkin solve then does not reproduce the references, and H3 stops.
- **Signature check:** the slope of error against distance to the nearest supervised source (or its point-group/translation image) is reported. H3 predicts ratio BND4_FAR / TRAIN_NEW ≤ 1.5 at θ = 0.

**Stage 2 — learned closure** on 𝔊, trained on S16-R only.
- Two seeds; frozen schedule; no checkpoint selection.
- The unseen-group trajectory is reported descriptively (ideation D3).
- **Falsifier F2 (transfer):** BND4_FAR con_coarse > 0.25 or > 1.5 × TRAIN_NEW con_coarse for either seed.
- **Falsifier F3 (Horn B, harm):** the closure improves TRAIN_NEW over θ = 0 but worsens BND4_FAR or BND4_NEAR by more than the seed spread.
- **Falsifier F4 (Horn A, vacuity):** the closure changes unseen-group errors by less than the seed spread, or closes < 50 % of the 𝔊 → 𝔊_ref θ = 0 gap.
  - The claim "a learned closure is what makes a coarse generator accurate" is then falsified.
  - Any transfer belongs to the solver. That is reported as such, and the FNTO admissibility question goes to the user.
- **Falsifier F5 (cost):** measured uncontended in the A75 torch style, with roofline counts reported, the run fails if any of these holds:
  - 1-light apply > 5 ms;
  - 64-light apply > 0.86 s;
  - persistent state > 2 GiB;
  - training step > 1 s;
  - pt0 nonrigid geometry rebuild > 1 s.

**Stage 3 — confirmation.** Only if Stages 0–2 pass: one scoring on the frozen WL77A Gate 4 population (`ports76.json`, te1), with the WL59 gates and the deformation gate as predeclared.

**Budget.**
- No new reference photons for Stages 0–2. S16-R and the WL74 Stage-4 / DEF4 references are archived; Gate 4 references were generated in Part 0.
- Geometry precomputation for rest, pt0 and the evaluated dents: minutes (*estimate*).
- θ = 0 evaluation: minutes.
- Two closure fits: about 0.5–2 h each on one lane (*estimate*).

**Contract note.** The human decision on whether "solver + closure" is an admissible FNTO K (Horn A) should be recorded *before* Stage 2. Stage 1 is the learning-free, falsifying physics test regardless.

---

## Key claims for adversarial review

- **H3-C1 (operator).** In this fixture, 𝒦 = g_X K_scat = 4πσ_t[ωD1 + ω²D2 + ω³⟨Se_Y, G Se_X⟩] holds exactly. LCR keeps D1 and D2 deterministic-exact, approximates only the k ≥ 3 Gram term by an SPD Galerkin (CPM) solve, and is by construction linear, ≥ 0, reciprocal without label reversal, PSD on its D1 and k ≥ 3 terms, and exactly passive through the s-certificate.
- **H3-C2 (learning-free floor).** θ = 0 on the refined control grid meets the WL59 gates on every S16-R held-out and WL74 Stage-4 group. On 𝔊 it lands near native 0.02–0.06 / coarse ≤ 0.03 / con_coarse ≤ 0.15 (non-evidence extrapolation from `specialist_physics.md` §7).
- **H3-C3 (supervision-distance independence).** BND4_FAR con_coarse ≤ 1.5 × TRAIN_NEW con_coarse for LCR, against 3–6 × for AR/A75 (0.79–0.82 vs 0.13–0.27). For the three translated BND4_FAR sources, LCR's predictions equal translated supervised predictions up to the physical corner influence (about 1 %, estimate).
- **H3-C4 (zero identity capacity, bounded drift).** No learned function sees source identity, absolute position, face id, frames or a global receiver basis. The only null-space directions are closure values on weakly supervised local-geometry classes, bounded by c_max = 0.5 of one-stencil couplings around a learning-free zero point.
- **H3-C5 (learning is useful).** The trained closure closes ≥ 50 % of the 𝔊 → 𝔊_ref θ = 0 gap on *unseen* groups without harming them (it escapes both horns in §13).
- **H3-C6 (deformation).** With dents represented in the solve (volumetric material-coordinate map plus in-medium visibility), the DEF gates (rel ≤ 0.35, corr ≥ 0.90) pass without pose-specific supervision beyond pt0, at θ = 0 and with the closure, including te3, whose slope is outside pt0's range.
- **H3-C7 (cost).** On a structured fixed-topology stencil: 1-light apply ≤ 5 ms, 64-light ≤ 0.86 s, state ≤ 2 GiB (0.3–0.6 GB expected), rigid rebuild 0, learned-coefficient nonrigid rebuild ≤ 0.1 s. The nonrigid geometry-term rebuild (0.3–2.5 s, estimate) is the most likely budget failure.
- **H3-C8 (incidence without learned angles).** Contrast arises entirely from the exact sub-port first collision (D1) and exact second-collision arrivals (b̄, D2). Snapping only k ≥ 3 to 1.25 × 0.5 mm elements keeps coarse-contrast error well below 0.25.
- **H3-C9 (contract and scope).** LCR is admissible as a "learned intrinsic K" only by human decision (FNTO contract lines 17 and 86; `docs/Architecture.md` line 94). Its transfer mechanism requires a known scalar collision kernel (g = 0, homogeneous, index-matched) and does not extend to Burley, anisotropic or Fresnel media without angular state.
