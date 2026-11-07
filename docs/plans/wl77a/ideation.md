# WL77A Ideation: Mechanism Families for a Source-Transferable Intrinsic K (2026-11-07)

**Role.** Ideation coordinator, WL77A multi-agent architecture-discovery workflow.

**Status.** COMPLETED. This file is a discovery record, not a protocol. It does not adopt or authorise a candidate. The workflow's next steps (adversarial review, selection of at most one candidate, frozen contract) decide what happens to it.

**Scope rules followed.**
- Read-only on the repository except this file.
- No GPU job, server access or training. No new Monte Carlo was run for this step.
- WL77 files and every historical worklog and result are untouched.
- The Gate 4 confirmation population (`ports76.json`) and its references were **not read**. Only its *selection rule* in `experiments/fnto_fia76/ports76.py` (code) was read, to know which symmetry classes are excluded.

---

## 0. Inputs and what I verified myself

**Specialist reports (read in full):**
- `docs/plans/wl77a/specialist_physics.md` (Agent 1)
- `docs/plans/wl77a/specialist_neural.md` (Agent 2)
- `docs/plans/wl77a/specialist_systems.md` (Agent 3)
- `docs/plans/wl77a/specialist_priorart.md` (Agent 4; sections 3–9 in full)

**Spot checks in code and results** (claims the selection depends on):

| Claim | Where checked | Result |
|---|---|---|
| W+UV takes absolute rest/current xyz and an ordinal face id as node inputs | `experiments/fnto_rule57/r57_geometry.py:58` (`local=np.column_stack([xyz0/.05,xyz/.05,…,fm['port_face']/5])`) | Confirmed |
| Pair features use world-axis offsets | `r57_geometry.py:65-66` (`dr=(xyz0[dst]-xyz0[src])/.01`) | Confirmed |
| U is a softmax over **all P** receivers; mass is a free per-source sigmoid | `experiments/fnto_rule57/r57_model.py:61-63` | Confirmed |
| WL77 Stage 1 multiple-scatter part is `M = mean(S_ref) − albedo·SS_q`, with SS_q from the deployable q-quadrature; the "noise" is the replica half-split of S only | `experiments/fnto_fia77/gate77.py:45-48` | Confirmed. SS_q is the same in both halves, so its sampling variance is invisible to the reported floor |
| FIA SS deposits one isotropic exit ray per first-collision sample, weight c·e/q | `experiments/fnto_fia76/fia76.py:79` (`wc=c*(1-e)/q;ws=c*e/q`) | Confirmed |
| BND4_FAR positions: s101 (−48.75, −8.75, −5), s105 (−50, 1.25, −1.25), s107 (−8.75, −50, −3.75), s103 (−41.25, −43.75, −5) | `results/fnto_ato74/ports4.json` | Confirmed. `d_to_S16R_images_mm` is 20.0 / 21.6 / 21.3 for the three translates, i.e. they are far from the *point-group* images but are along-edge translates (Agent 1 §5.3) |
| Gate 4 population excludes previously scored ports and their **point-group** images only | `experiments/fnto_fia76/ports76.py:1-4`, `experiments/fnto_src73/ports73.py:36-40` (`perms` = D4 × z-flip ops) | Confirmed. Translates along an edge are **not** excluded, so translation equivariance is relevant to Gate 4 (the far stratum maximises distance to the point-group images, so e₂ differs from training) |
| WL75 A75: TRAIN_NEW con_coarse 0.131 (G2 oracle 0.129); BND4_FAR 0.803 / 0.824 vs AR 0.793 / 0.792 | `docs/worklogs/75_WORKLOG_RANK16_LEARNED_ATO_TRANSFER.md:128,186,195` | Confirmed |

**Not verified by me:** the two independent non-evidence CPU emulations of the SS-quadrature noise (Agent 1 §6, Agent 2 §5). They agree with each other and with WL77's per-basis values to about ±0.004–0.03. Section 1, C4 explains how they are used below.

---

## 1. Consolidated diagnosis: why source transfer has failed so far

The four reports agree on six causes. Everything in Section 3 is built against this list.

| # | Cause | Families hit | Key evidence | Status |
|---|---|---|---|---|
| **C1** | **Identity channels.** Source identity can be stored in absolute-position inputs, a global absolute-receiver basis U (rank 16 ≥ 16 training sources, so it can hold a per-source table) and a free per-source mass | W+UV (AR, A75) | Code above. WL71: transfer is supervision-local (5–18 mm). WL71B: U mass near fit ports 0.06 → 0.56, corner V 0.03 → 23.6. Late drift ×3–5 after 0.5–2k steps (WL71A/73/75/76) | Code-verified; the drift mechanism is inferred (consistent across 4 batches) |
| **C2** | **Symmetry not quotiented.** Frames and face ids are not equivariant; the symmetry loss was angle-integrated augmentation | W+UV, RSK, AR | 3 of 4 BND4_FAR sources are exact rest-pose translates and reflections of supervised S16-R sources, up to an estimated ~1 % corner influence. A75 still scores s107 at con_coarse 1.19 against an oracle of 0.20–0.31. RSK frames are world-aligned on top/bottom and mismatch bin frames on sides (`rsk72.py:16-21`) | Geometry verified; the ~1 % is an estimate |
| **C3** | **Incidence enters only through the global CP pairing** A_r ↔ T_r ↔ O_r with an angle-free T, while the physical incidence effect is a 0.3–0.8 mm first-collision footprint shift at sub-cell position | Every ATO variant | WL74A: G0 0.27 / G2 0.13 structural floors even with a per-source oracle T; \|HZ\|/\|Z\| 0.05 at a centre sub-cell vs 0.17 at a corner | WL74A (read-only, adversarially verified) |
| **C4** | **Noisy deployable geometric quadratures.** q = 1024 one-ray SS (and W) inject per-entry noise comparable to the target error | FIA, FIA-AZ | Two independent CPU emulations: with exact SS the floors are ring 0.22 / receiver-relative az 0.04–0.07 / free order 2 0.02–0.03. With q = 1024 SS they reproduce WL77's 0.453 / 0.385 / 0.352 and FIA's native 0.29–0.30 | **Non-evidence.** WL76/WL77 verdicts stand as recorded. The angular-expressivity reading of WL77 is *not established*, so it should not drive the design. A cheap read-only recompute of `gate77` with high-q or deterministic SS would settle it |
| **C5** | **Boundary context is regressed jointly, not composed.** Unseen plane combinations are extrapolation: second-edge distance e₂ 41–46 mm vs 18.75 mm in training, near-corner 8.75 / 8.75, vertical edges (SV/UNS) | W+UV, RSK, FIA | RSK edge feature is a direction-free scalar (`rsk72.py:27-28`); FIA uses sorted free paths (`fia76.py:39-43`), which lose which side the boundary is on. SV/UNS contrast 1.0–1.5 in every model | Code-verified; effect size inferred |
| **C6** | **Learned angular ends.** FIA's R_θ re-learns the exact exit projection; ATO learns both ends as global atoms with no reciprocity tie | FIA, ATO | Agent 1 §3.2: for g = 0, all outgoing structure is the exact escape projection of a scalar collision density, and all incoming structure is the exact first-collision projection | Exact derivation for this fixture |

**Two structural facts the families exploit** (Agent 1 §3–4, exact for g = 0, homogeneous, index-matched):
- **(P1) Green form.** g_X K_scat(X→Y) = 4π σ_t ω ⟨P_Y, G P_X⟩ with G = (I − ωS)⁻¹. G is scalar, direction-free, source-independent, symmetric, elementwise ≥ 0, positive definite, and equivariant under every isometry of the medium.
- **(P2) Reciprocity with no label reversal.** A_i K16(i,a→j,b) = A_j K16(j,b→i,a). The incoming-bin dependence of an unseen source equals the outgoing distribution observed at that port from supervised sources (within transport range).

**Scales that bound any design** (Agent 1 §2, Agent 3 §3):
- 66–94 % of collided exit lands within 3.6 mm; 97–99 % within 15 mm; ≥ 99.8 % within 25 mm.
- Contrast signal sits within about 5 mm (14 receiver ports).
- Weighted collision count 22 / 17 / 5 (R/G/B), so Neumann truncation is impossible.
- Depth structure is sub-mfp: 1 mm voxels give +6 % red exit bias, 0.5 mm gives +2 %.
- About a third of all source ports have a contrast neighbourhood that crosses an edge.
- **Not covered by any family here:** the deformation gates (DEF). Every architecture has failed them, and two training poses may not identify them. Section 6 records how each selected family would at least *route* deformation.

---

## 2. Design axes

Families are distinguished by three axes. A different encoder on the same axes does not count as a new family.

| Axis | Options seen across families |
|---|---|
| **How the source (port and incoming bin) enters** | (a) as canonical relative geometry into a learned direct kernel; (b) through an exact geometric first-collision projection; (c) as a volumetric injection into a solved field; (d) as image positions relative to each boundary plane; (e) as a learned embedding shared with the receiver role; (f) as a continuous effective point (first-collision moment) |
| **What is explicit vs learned** | From "only symmetry/frames explicit, all transport learned" through "angular ends explicit, scalar resolvent learned" to "everything explicit except a local closure" |
| **How transport is organised** | Direct source→receiver blocks; Gram product through boundary-layer elements; implicit global solve of a local generator; composition of half-space operators; inner product of cell embeddings |

---

## 3. Mechanism families

Seven families follow. Each card lists the central mechanism, targeted failure, transfer argument, nearest prior art, main risk, constraint compliance and a cost estimate. Costs are estimates unless cited.

### F1 — Canonical-Quotient Relative Kernel (CQR)

**Central mechanism.** Each source cell (i, a) is mapped by a deterministic symmetry operation g(i) into a canonical fundamental domain. g(i) is chosen from the port's boundary configuration; the exact local D4 bin permutations of WL74A §3.3 are applied to incoming and outgoing bins. Ties (corners, diagonals) are resolved by averaging over the stabiliser subgroup, which keeps exact equivariance on the tie set.

In that frame a **learned direct kernel** gives the source→receiver block:

  K(i,a→j,o) = d_{i,a} · m_θ(ctx_i, a) · Σ_{k≤4} α_θ,k(a | pair) β_θ,k(o | pair)

- Normalisation is over (j, o) within the support (passive mass × distribution, local, no global softmax).
- The pair-specific rank-4 angular factors come from a network over canonical relative features:
  - the receiver offset in the source frame;
  - depths;
  - normals;
  - a **set of directed boundary-plane descriptors** (signed distance, plane normal in the canonical frame, saturating exp(−e/L_c)) for every plane within about 2 L_c of source or receiver, encoded permutation-invariantly as a sum over planes;
  - the analytic first-collision mean shift of bin a as an input feature.
- Two zones: a pair-expressive near core (R ≤ 5–7.5 mm, 14–31 receivers) and a passive scalar far zone to 128-NN on continuous relative offsets, with no coarse-grid quantisation.
- **Removed:** absolute coordinates, face id, global U, free per-source mass.

**Targeted failure.** C1 (identity channels removed by construction), C2 (exact images become *identical inputs*, a quotient rather than augmentation), C3 (pair-specific incidence blocks replace the global CP pairing), C5 (per-plane set encoding: the e₂ = 41–46 mm plane is learned from interior sources' far planes, since INT sources have e_min ≥ 21.25 mm).

**Why transfer is expected.**
- Within the rest pose, the BND4_FAR translates s101/s105/s107 get bitwise the same inputs as supervised s50/s52/s53, except for e₂ (saturating: red exp(−18.75/18.1) = 0.35 in training vs ≤ 0.10 at 41 mm).
- Interior sources at any distance from supervision are exact translates.
- Pair-specific rank 4 is enough for the oracle: 2 % residual per pair (Agent 3 §3.3).

**Nearest prior art.** Vicini, Koltun & Jakob 2019 (source-centred local geometry, mass × exit sampler); Deep Scattering 2017 (relative descriptors); G-FNO / EGNN (equivariance by construction); H-matrix near/far split.
- **Structural difference to claim:** a quotient over the medium's point group *and* translations, with exact bin permutations; directed multi-plane boundary context beyond a local polynomial; full 16 × 64 output with port correspondence; exact linear multi-source application.

**Main risk.**
- The angular content is still *learned from 16 sources*. Contrast is 1–3 % of response energy and sub-cell sensitive. The near-core blocks for unseen plane combinations (corners at 8.75 / 8.75, vertical edges) are extrapolated by an MLP, with no physics guarantee.
- No reciprocity, so the angular ends are untied.
- Null-space drift can reappear in unsupervised feature regions (WL71A pattern).

**Constraints.** Linear, ≥ 0, passive (mass ≤ 1 × normalised distribution, ≤ d), zero → zero, no identity lookup, correspondence kept. K/C: an in-medium visibility mask on pairs is needed for dents.

**Cost (estimate).**
- State 0.18–0.45 GB (Tucker / rank-4 near core at 31 receivers) + about W+UV for the far zone.
- Rebuild: one network pass over about 4.9 × 10⁵ pairs (W+UV's edge count), tens of ms.
- 64-light apply about 6 × 10⁹ MAC, within 3 × A75 with a fused kernel.

### F2 — Reciprocal Boundary-Layer Green Form (RBG)

**Central mechanism.** Implement (P1) directly, with the learned part limited to the scalar resolvent:

  K_scat(X→Y) = SS_det(X→Y) + g_X⁻¹ · Π_Yᵀ Γ_θ Π_X

- **SS_det** is single scatter by deterministic cubature (not one-ray MC). It is translation-invariant, so the rest-pose blocks are unique per (relative offset, edge context) and reused.
- **Π** is the *same* exact geometric projection on both ends: the attenuated straight-line kernel P_X (or P_Y) integrated onto **boundary-layer elements**.
  - Elements are face-relative to every nearby face: about 4 depth layers at ≤ 0.5 mm (or an exact Milne-type profile) × port columns, with a 2 × 2 sub-port split in the near core.
  - Π is deterministic and keeps in-medium visibility, so cross-dent air paths stay with C.
- **Γ_θ(e, f)** is a **learned, direction-free, source-independent** kernel on element pairs within about 15 mm. It is a function of the element pair's canonical relative geometry: offset, both depths, and the directed plane set as in F1.
  - It is symmetric by construction (swap-invariant features).
  - It is made passive and symmetric together by **max-normalisation**: Γ_ef = g_θ(e,f) · min(m_e, m_f) / max(R_e, R_f), with R_e = Σ_f g_θ(e,f) ω esc(f) and esc(f) the exact escape probability. Then Σ_f Γ_ef ω esc(f) ≤ m_e ≤ 1 and Γ = Γᵀ.
- **Result.** All incoming-angle dependence (contrast) and all outgoing-angle structure are exact geometry. Reciprocity and the A–O tie hold by construction. The only learned object never sees a source id, a bin or an angle.

**Targeted failure.** C3 and C6 (angular ends exact; the incidence footprint shift enters through Π_X at sub-port resolution, with no CP pairing); C4 (deterministic geometric terms required); C1/C2 (Γ uses relative, quotient features only); C5 (Γ's boundary dependence is supervised by every training source's collisions near every plane they reach).

**Why transfer is expected.**
- Γ is source-independent. Its supervision scales with (training sources × element pairs reached), not with source count: every S16-R footprint samples hundreds of (offset, depth, plane-distance) configurations.
- An unseen source differs only through its exact Π_X, so near-boundary and corner incidence effects (side-exit fraction 0.25–0.41 at an edge, 0.37–0.56 at a corner) are *computed*, not extrapolated.
- Agent 1 §7 (non-evidence) bounds the discretisation error: with depth exact and k ≥ 2 collisions snapped to port columns, native 0.09–0.10, coarse 0.04–0.07, pair-contrast 0.05–0.08 at interior sources. A 2 × 2 sub-port split is needed for ≤ 0.03.

**Nearest prior art.**
- First-collision source method (neutron transport); Jensen et al. 2001 (exact SS + kernel); Photon Beam Diffusion 2013.
- DeepRTE (delta inflows + Green's function) and Element Learning (learned in2out boundary operator per element).
- Vicini 2019 for the geometry conditioning.
- **Structural difference to claim:** the learned object is the scalar symmetric resolvent *between* two exact, shared geometric projections (reciprocity and PSD by construction), equivariant under the medium's symmetries, with full 16 × 64 port/bin output. Unlike FIA, the exit projection is not learned.

**Main risk.**
- The exact factorisation may not be a *learnable* approximation: Γ near edges and corners must be learned from 4 boundary training sources, and max-normalisation may leave energy deficits at ω → 1 (red).
- Π cubature cost and noise. Deterministic and translation-reused at rest; dents need a certified local recompute.
- Element resolution: snapping k ≥ 2 to port columns already costs about 0.1 native, and the sub-port split multiplies Γ state about 4×.

**Constraints.** Linear, ≥ 0 (all three factors ≥ 0), passive by construction, zero → zero, no identity, correspondence kept, K/C via in-medium visibility.

**Cost (estimate; Agent 1 §9, Agent 3 §7).**
- Γ about 2 × 10⁷ floats with port-column elements (about 1.5 × 10⁴ elements × about 450 neighbours × RGB). About 10⁸ if the 2 × 2 split is used everywhere, so it is restricted to a ≤ 3.6 mm near core.
- Π_in about 2 × 10⁶ floats. Π_out shares unique translation blocks.
- 64-light apply: three sparse products, 10⁹–10¹⁰ MAC, tens of ms.
- Rebuild: Γ network over about 10⁷ element pairs (about 0.1–0.2 s); deterministic Π for dented regions only (unmeasured; must be ≤ 1 s).

### F3 — Learned-Closure Volumetric Resolvent (LCR)

**Central mechanism.** Do not represent G at all; **represent its local generator and solve**.
- **Grid.** A boundary-adapted volumetric discretisation of the medium in material coordinates: ≤ 0.5 mm depth layers with 1.25 mm lateral cells within about 2 mm of every face, and about 1 mm cells in the interior. That is about 1.2 × 10⁵ elements for the slab (estimate).
- **Source.** Enters as the exact deterministic first-collision injection s_X = Π_X (sub-port, same Π as F2).
- **Solve.** The scalar collision density solves (I − ω S_θ) c = s_X by conjugate gradients.
- **S_θ** = the exact discretised isotropic collision kernel (σ_t e^{−σ_t r}/(4π r²) × in-medium visibility, known for this fixture) × a **learned local closure** (1 + δ_θ(relative geometry of the element pair: sizes, depths, face distances)).
  - It is symmetric (volume-weighted) and nonnegative.
  - Rows are capped at 1 − exact escape, so the spectral radius is < 1 and energy is exact at ω = 1.
- **Readout.** The exact exit projection P_Y (64 bins), plus SS_det and exact B.
- **What is learned.** Only the discretisation closure, the error a coarse grid makes against the sub-mfp boundary layer: +2 % at 0.5 mm, +6 % at 1 mm (Agent 1 §7). Training uses implicit differentiation; S_θ is symmetric, so the adjoint solve is the same CG.

**Targeted failure.** C5 most of all: edges, corners, vertical edges, far planes and **dents** all enter as explicit mesh geometry *inside the solve*, never as a learned context vector. Also C1/C2 (nothing source- or position-specific is learned), C3/C6 (exact projections), C4 (deterministic Π).

**Why transfer is expected.**
- The learned object is a *local* stencil correction. It is applied identically at every element and supervised at every collision site of every training source (about 20 weighted collisions × the whole footprint, red).
- Unseen sources and boundary configurations differ only in where light is injected and which geometry the CG sees. Both are exact.
- Corner and edge stencils are seen locally by the corner (s51) and edge training sources.
- **Limit.** With θ = 0 and a fine enough grid the family *is* a deterministic transport solver. That gives a learning-free control that bounds the representation floor separately from learning.

**Nearest prior art.** First-collision source method + S_N / collided solve; lattice-Boltzmann lighting (Geist 2004); Filtered MgNet (learned multigrid RTE, reported to degrade in transport regimes); Element Learning; DEQ / implicit layers.
- **Structural difference to claim:** a learned closure of a coarse *volumetric* resolvent used as an intrinsic boundary-to-boundary operator with exact port/bin projections, symmetric and passive by construction, evaluated on held-out sparse-source transfer.

**Main risk.**
- **Contract interpretation.** In this fixture the learned part may be nearly unnecessary (S is known), so the result could reduce "learned K" to "numerical solver + small closure". Whether that is an admissible FNTO K is a **human decision**.
- **Systems.** This family carries the highest systems risk:
  - 1-light apply about 30 CG iterations × an SpMV of about 3.6 × 10⁷ nnz (S is channel-independent; ω scales it), memory-bound ≈ 5 ms. That is at the 5 ms hard cap and over the 2 ms preference.
  - Training needs windowed implicit solves for 512 right-hand sides.
- **Generality.** Anisotropic phase or Fresnel boundaries would need angular state (discrete ordinates), which Agent 3 §6 shows breaks the memory budget.

**Constraints.** Linear (solve of a fixed linear system), ≥ 0 (Neumann series of a nonnegative operator with ρ < 1), passive by construction, zero → zero, no identity, correspondence via exact Π, K/C via in-medium visibility in S.

**Cost (estimate, my arithmetic from Agent 3's figures).**
- κ(I − ωS) ≈ 1/(1 − ωλ_max) ≈ 37 for red (slab mode B² ≈ 0.076, 1 − λ ≈ B²/3 + leakage), so about 20–30 CG iterations to 10⁻³.
- 64-light apply ≈ 30 × 7 × 10⁹ MAC sparse, about 0.1 s.
- State: S nnz 3.6 × 10⁷ (0.14 GB) + Π.
- Rebuild after a dent: local re-mesh and S rows in the footprint, about < 1 s if certified (unmeasured).
- Training windows of 50 × 50 × 10 mm: about 0.2 GB per CG vector for 512 RHS, about 1 GB per lane.

### F4 — Half-Space Kernel with Per-Plane Image Composition (HSI)

**Central mechanism.**
- A canonical **half-space kernel** H_θ(offset, depths, a, o) is learned once (interior/top-face data) in a canonical frame.
- The finite medium is handled by an **explicit composition rule over boundary planes**: for each nearby plane p, a multiplicative truncation/image operator T_p(signed distance, incidence, exit) built from the image position of the source across p.
- Corners and vertical edges are products of the plane operators × a small learned interaction term in saturating two-plane features.
- Multiplicative (not subtractive) factors keep positivity.

**Targeted failure.** C5: e₂ extrapolation, near corners and SV/UNS by composition. WL74A §4.5: edge outgoing shapes are 85–90 % inside the cone of interior shapes, and the edge-specific truncation is about ±5 %.

**Why transfer is expected.** Each plane factor is a 1-D function of plane distance, learned from all (source, plane) pairs. Unseen combinations are compositions of seen single-plane effects.

**Nearest prior art.** The dipole / method of images (Jensen 2001); multipole slab model (Donner & Jensen 2005; my addition, not checked by Agent 4); GreenNet / VGF (explicit singular + learned boundary correction, elliptic); Neural SSS half-space sharing.

**Main risk.**
- Images are exact only for diffusion-type boundary conditions. Within 1–2 mfp (where 66–94 % of the signal lies) the RTE is not image-separable.
- Red corners within 18 mm of two planes interact strongly.
- Curved dents need tangent-plane approximations.
- Multiplicative truncation cannot remove the per-incoming-azimuth side-exit fraction (0.25–0.56) without becoming a learned joint function, which collapses into F1.

**Cost.** Cheap: a few factors per pair (≈ W+UV).

### F5 — Dual-Role Reciprocal Cell Embedding (DRE)

**Central mechanism.** One learned nonnegative cell embedding φ_θ(port, bin; canonical local geometry) serves **both** as source and receiver representation:

  g_X K(X→(j,o)) = ⟨φ16(X), Λ_θ(Δ_XY) φ64(j,o)⟩,  with φ16(j,b) = Σ_{o∈b} φ64(j,o)

- Λ_θ is a nonnegative positive-definite offset kernel (Bochner mixture in canonical offset).
- K16 is symmetric by construction (P2).

**Targeted failure.** C6 and data starvation. The incoming dependence of every port is learned from its abundant **receiver** observations (16 sources × about 300 receivers × 64 bins), turning outgoing data into contrast supervision.

**Why transfer is expected.** Every port within about 15–25 mm of any supervised source has its embedding directly supervised. Beyond that, it relies on F1-style canonical features.

**Nearest prior art.** Kernel Nyström (column completion); reciprocity in BSSRDF/PRT; DeepONet branch × trunk (tied).

**Main risk.**
- Far ports gain nothing from reciprocity (Agent 1: the gain is within transport range only). Far transfer reduces to F1's quotient.
- Exact passivity needs local normalisation, which breaks exact symmetry unless the max-normalisation of F2 is borrowed.
- Low-rank inner products may not represent the sub-mm near-field Gram structure.

**Cost.** Embedding m ≈ 16–32 per cell: about 6 × 10⁶ floats; apply similar to F1.

### F6 — Moment-Shift Continuous-Source Field (MSF)

**Central mechanism.**
- Source and incidence enter as **one continuous effective point** x_eff(i,a) = x_i + δ(i,a), where δ is the analytic first-collision mean displacement: depth ≈ cos θ mm, lateral ≈ sin θ mm, truncated at edges.
- A shared continuous spatial field F_θ(x_eff → j, o) is evaluated there, with first-order shift channels (F, ∂_u F, ∂_v F; Agent 3 template 3).
- No learned incoming-angle dependence.

**Targeted failure.** C3 at minimum cost: removes the CP-pairing floor by putting the footprint shift into geometry.

**Why transfer is expected.** Incidence becomes position, and position is handled by translation equivariance.

**Nearest prior art.** SH gradients (Annen 2004); Photon Beam Diffusion (sources along the refracted ray); directional dipole (Frisvad 2014).

**Main risk.**
- First moments miss the *shape* of the truncated first-collision distribution near edges and at grazing incidence, which is where boundary contrast lives.
- Outgoing angular structure is still learned.
- It is a moment truncation of F2's Π_X, so it is subsumed by F2.

**Cost.** ×3 W-type state; negligible geometry.

### F7 — Analytic Transport Prior with Bounded Learned Residual (APR)

**Central mechanism.** K = K_prior ⊙ exp(r_θ), where:
- K_prior = exact B + SS_det + a slab multipole-diffusion collided term with exact exit projection, computed from geometry for every source;
- r_θ is a bounded, magnitude-penalised residual on canonical relative features.

The physics prior is the "zero point" for unsupervised sources, which removes the null-space drift (Agent 2 M8).

**Targeted failure.** C1 and the late drift (unsupervised sources anchored to physics rather than to initialisation), plus C5 via the analytic image series.

**Why transfer is expected.** The prior is exact in position; the residual is small and smooth by construction.

**Nearest prior art.** Jensen 2001, d'Eon & Irving 2011 (quantised diffusion), NLT (physical base + learned residual).

**Main risk.**
- Diffusion is inaccurate exactly where the signal is: within 3.6 mm (66–94 % of exit) and for blue (ω = 0.9).
- The residual must carry a large, sharp correction there, which brings back F1's learnability problem.
- The prior biases the result wherever it is wrong. That is a declared modelling choice, not a fix.

**Cost.** ≈ W+UV plus analytic prior evaluation.

### Ideas rejected outright (not families)

| Idea | Reason |
|---|---|
| Interpolating or warping supervised source columns to unseen sources (Nyström copy, nearest-supervised-source transport) | A per-source table at inference: forbidden by the constraints |
| More rank or width in W+UV; more or redistributed sources with the same family | WL75: training reaches the G2 oracle (0.131) while BND4_FAR stays 0.80–0.82. WL73: far transfer small or mixed |
| Raising q of the FIA quadratures | 1/√q: 27–150× the 75 s rebuild (Agents 1, 2) |
| Higher-order azimuthal lobes per anchor pair (WL77's open option) | Physically unnecessary for g = 0 once SS is exact (order ≤ 3 leaves ≤ 0.017, non-evidence). About 15 GB fp32 for full 64-bin in the FIA layout (Agent 3) |
| Online MC inside the K apply; uniform 0.25 mm volumetric grids with directional state; FNO on the port grid; early stopping | Cost (Agent 3 §9), domain mismatch (Agent 4), or not authorised |

### Cross-cutting ingredients (not families; any selected family may need them)

- **D1 — deterministic geometric terms (Agent 2 M5).** Every deployed geometry-only term that is also in the targets (SS, Π, β) must have variance well below the contrast signal (1–3 % of response energy). Mandatory for F2, F3, F7.
- **D2 — reciprocity-derived training entries (Agent 2 M7).** Exact for this medium; derived from the same S16-R, so they must be declared as derived labels. Built in by construction for F2, F3 and F5; optional for F1.
- **D3 — descriptive unseen-source development trajectory** during training (no checkpoint selection), so that drift is visible.
- **D4 — in-medium visibility masks** on every pair or element interaction (K/C ownership in dents).

---

## 4. Comparison matrix

● = addressed by construction; ○ = partially or conditionally; blank = not addressed.

| | C1 identity | C2 symmetry | C3 incidence/CP | C4 quadrature noise | C5 boundary composition | C6 learned angular ends | Reciprocity | Passive by construction | Cost risk |
|---|---|---|---|---|---|---|---|---|---|
| F1 CQR | ● | ● | ● (learned per pair) | n/a (no MC terms) | ○ (plane set, learned) | | optional | ● | low |
| F2 RBG | ● | ● | ● (exact Π) | ● (requires D1) | ○ (Γ learned near planes; incidence exact) | ● | ● | ● (max-norm.) | medium (Π cubature, element count) |
| F3 LCR | ● | ● (exact geometry) | ● (exact Π) | ● (requires D1) | ● (geometry in the solve) | ● | ● | ● | **high** (CG apply, implicit training) |
| F4 HSI | ● | ○ | ○ | n/a | ● (explicit composition) | | | ○ | low |
| F5 DRE | ● | ○ | ○ | n/a | ○ | ○ (tied) | ● | ○ | low |
| F6 MSF | ● | ○ | ● (moments only) | n/a | | | | ● | low |
| F7 APR | ● | ○ | ○ | ● (requires D1) | ○ (analytic images) | ○ | ○ | ● | low |

---

## 5. Selection: three maximally different families

**Selected: F1 (CQR), F2 (RBG), F3 (LCR).**

### 5.1 Distinctness

The three sit at three separated points on the "what is explicit vs learned" axis, and they differ on the other two axes too.

| | F1 CQR | F2 RBG | F3 LCR |
|---|---|---|---|
| **How the source enters** | Canonical relative coordinates (+ moment feature) into a learned network | Exact geometric first-collision projection Π_X; the learned part never sees the source | Exact volumetric injection into a solved field |
| **Explicit** | Symmetry group, frames, bin permutations, boundary-plane geometry | + both angular ends (Π, SS_det), reciprocity, escape accounting | + the transport equation itself (exact S, geometry in the solve) |
| **Learned** | The whole angular-spatial pair kernel (incidence and exit blocks) | A scalar, direction-free, source-independent element-pair resolvent Γ (long range, ≤ 15 mm) | Only a local closure of the discretised generator (one stencil range) |
| **How transport is organised** | Direct source→receiver blocks (one sparse product) | Gram product Πᵀ Γ Π through boundary-layer elements (three sparse products, no solve) | Iterative global solve (CG) of a local operator |
| **What it bets on** | Symmetry quotient + learned angular blocks suffice | Angular physics must be exact; the long-range scalar Green's function is learnable | Nothing long-range should be learned at all |
| **Characteristic failure if wrong** | Fails at unseen plane combinations (corners, SV) and in contrast | Fails where Γ near edges/corners is under-supervised, or on Π cost/noise | Fails on cost/contract, or if a coarse closure cannot replace the sub-mfp boundary layer |

**Why these are three mechanisms, not three encoders.**
- F1 and F2 both use relative quotient features, but on *different objects*. F1 learns K's angular blocks. F2 learns a scalar Γ with no angular index at all, so its contrast is computed, not learned.
- F2 and F3 share the exact projections (P1), but organise G in the two classical, distinct ways: **learn the Green's function** (F2) vs **learn the generator and solve** (F3). They differ in learned range (15 mm vs one stencil), apply structure (direct vs iterative) and in what carries boundary/dent context (learned Γ features vs explicit mesh).

### 5.2 Credible physical transfer argument

- **F1.** For the rest pose, three of four historical BND4_FAR sources become identical inputs to supervised sources, up to the saturating e₂ feature. The Gate 4 population excludes point-group images but not translates (Section 0), so the translation part of the quotient is exactly what Gate 4 tests at its far strata.
- **F2.** The only learned object is source-independent and is supervised at every element pair any training source's collisions reach. Contrast and outgoing structure are exact geometry. Agent 1's non-evidence bound for port-column snapping with exact depth (coarse 0.04–0.07, pair-contrast 0.05–0.08) sits inside the WL59 gates.
- **F3.** The strongest physics argument: the learned object is local and supervised everywhere, and all boundary/corner/dent context is explicit in the solve. Its learning-free limit (θ = 0, fine grid) is a deterministic solver, and that limit is itself an oracle control for F2 and F3.

### 5.3 Plausible cost

| | F1 | F2 | F3 |
|---|---|---|---|
| Persistent state | 0.2–0.5 GB | 0.1–0.4 GB (port columns; sub-port split only in the near core) | about 0.15 GB S + Π |
| 64-light apply | ≲ 0.1 s | tens of ms (estimate) | about 0.1 s (estimate) |
| 1-light apply | ~1–2 ms | ~1–3 ms | **~5 ms (at the hard cap)** |
| Nonrigid rebuild | tens of ms | 0.1–0.2 s Γ + local Π (unmeasured) | local re-mesh + S rows (unmeasured) |
| Training | ≈ W+UV | Γ on support-restricted element pairs | windowed implicit CG, about 1 GB/lane |

All three are within Agent 3's hard caps on paper. F3 carries the highest systems risk and should be read that way.

### 5.4 Why the others were not selected

- **F4 (HSI).** Its composition idea is kept *inside* F1 (permutation-invariant plane set) and F2 (plane features of Γ). As a standalone mechanism it is physically weakest exactly where the signal is (< 2 mfp; red corners), and making it represent the per-incidence side-exit fraction collapses it into F1.
- **F5 (DRE).** Its far transfer reduces to F1's quotient; its distinctive part (reciprocal tying) is built into F2 and F3 exactly. Its passivity needs F2's construction.
- **F6 (MSF).** A first-moment truncation of F2's Π_X. It misses the truncated first-collision *shape* at edges and grazing incidence, where boundary contrast lives.
- **F7 (APR).** The diffusion prior is wrong in the 3.6 mm near field that carries 66–94 % of the signal and in blue. Its useful idea (physics zero point against drift) is a regulariser any family could declare.

---

## 6. Selected families: falsifiable signatures and controls

These are signatures and controls the workflow could predeclare. They are not gates, and they are not chosen from the Gate 4 population.

**F1 CQR.**
- *Exactness check:* predictions for exact point-group images are identical to numerical precision. For along-edge translates they differ only through e₂.
- *Expected signature:* TRAIN_NEW con_coarse ≤ the G2 oracle (0.129) without a global rank increase; BND_FAR error approaches BND_NEAR error.
- *Falsifier:* far-boundary transfer at A75 level (≈ 0.8) although the inputs are identical up to e₂. That would mean the residual is not identity storage but e₂/plane-set extrapolation or angular learnability.
- *Deformation route:* current-geometry relative descriptors (thickness, normal, displacement in the canonical frame). No new mechanism.

**F2 RBG.**
- *Learning-free controls:* (i) an exact-Γ oracle at the chosen element resolution (the numerical Schur complement of the discretised G, computed once offline as a diagnostic), which gives the representation floor separately from learning; (ii) Π/SS_det noise floors reported separately from replica noise (D1).
- *Exactness check:* reciprocity A_i K16(i,a→j,b) = A_j K16(j,b→i,a) and row-sum ≤ 1 − β hold to float precision.
- *Falsifier:* the learned Γ reaches the exact-Γ floor on training sources but not on far-boundary sources. Γ's plane-feature generalisation is then the limit.
- *Deformation route:* Π recomputed exactly in the dent footprint (certified); Γ conditioned on local thickness and normal.

**F3 LCR.**
- *Learning-free control:* θ = 0 at the chosen grid, and at a finer grid, against the references. This separates discretisation error from the closure's learned contribution.
- *Falsifier (cost):* the measured 1-light apply above 5 ms, or the training step above 1 s on two lanes.
- *Falsifier (learning):* the closure fits training sources but changes unseen-source error by less than the seed spread. Learning is then not doing the work, and the claim reduces to the solver.
- *Deformation route:* explicit re-mesh in the footprint; the closure is local and needs no pose-specific supervision.
- *Contract note:* the human decision on whether a solver + closure is an admissible FNTO K should come before any F3 implementation.

---

## 7. Open items and uncertainty

1. **SS-quadrature noise (C4)** rests on two non-evidence CPU emulations (one interior configuration, no W noise). It is the main reason WL77's "higher-order azimuthal" reading is treated as unestablished here. A read-only recompute of `gate77.expr` with high-q or deterministic SS would confirm or refute it. That run is outside this agent's authority.
2. **The ~1 % corner influence on BND4_FAR translates** is inferred from exit-radius fractions. The existing WL74/75 references of s101/s105/s107 and their S16-R partners would allow a direct read-only comparison.
3. **All cost figures for F2 and F3** are back-of-envelope (element counts, CG iterations, Π cubature time). None was timed.
4. **The max-normalised symmetric Γ** (F2) guarantees passivity and symmetry but can leave energy deficits at ω → 1. Its bias near boundaries is unquantified.
5. **The F1 transfer argument** depends on the canonicalisation rule and stabiliser averaging being correct at frame seams. The bin permutations are exact (WL74A §3.3), but the dented poses (pt0, te*) have no exact symmetry, so only approximate canonicalisation applies there.
6. **Deformation (DEF gates)** is not specifically addressed by any family. All three only route it through current-geometry inputs or exact geometry.
7. **Prior-art negatives** (no held-out sparse-source boundary-operator transfer study; no RTE "explicit boundary + learned correction" instance) are weak, search-depth-limited negatives (Agent 4 §10). The multipole slab reference in F4 is mine and was not checked by Agent 4.
8. **Scope.** The Green-form arguments (F2, F3) are exact only for g = 0, homogeneous, index-matched media. Anisotropic phase, Fresnel boundaries or heterogeneity make G direction-dependent.
