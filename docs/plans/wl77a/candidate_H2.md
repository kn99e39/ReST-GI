# WL77A Candidate H2 — Reciprocal Boundary-Layer Green Form (RBG) (2026-11-07)

**Role.** Candidate developer for H2 (family F2 of `docs/plans/wl77a/ideation.md`), WL77A multi-agent architecture-discovery workflow.

**Status.** Hypothesis specification only. It is not a protocol and does not adopt, authorise or freeze anything. Nothing was trained or run on a GPU or server. No historical file was changed.

**Inputs.** The four specialist reports and the ideation record under `docs/plans/wl77a/`. Spot-checked code:
- `experiments/fnto_fia76/fia76.py` (header and lines 1–60)
- `experiments/fnto_fia76/phys76.py` (header)
- `experiments/fnto_rule57/r57_geometry.py:19-58,80` (poses, dent map, node features, ballistic q = 256)
- `experiments/phys_ref53/p53_geom.py:168-173` (`map_dent`)
- `experiments/fnto_src73/fit73.py:16-35` and `results/fnto_src73/ports.json` (S16-R)
- `experiments/fnto_repr68/eval68.py:14-16,53` (metrics; contrast map H·diag(1/d))
- `experiments/fnto_ato74/eval74.py:56` (BND4 groups)
- `docs/worklogs/76_*.md` (Stage 1: 36/36)

The Gate 4 confirmation population (`ports76.json`) was not opened.

**Labels used below.**
- **[V]** verified in code or results by me.
- **[S]** taken from a specialist report and not re-verified.
- **[E]** my own estimate or derivation, not measured.

---

## 1. Problem

A learned intrinsic operator K must give accurate first-exit responses for **source ports it was never supervised on**. This matters most for boundary ports far from the 16 S16-R training sources. The failure history (WL71–WL77) splits into two kinds:

- **Information-free models.** Every learned family so far has had to learn the angular structure (incoming contrast and outgoing distribution) together with the spatial transport, from 16 sources. These families either:
  - stored source identity in absolute-position or global-basis channels (W+UV, AR, A75), or
  - learned an angular object whose supervision was dominated by quadrature noise (FIA, FIA-AZ).
- **What physics says the learned object should be.** For this fixture (g = 0, homogeneous, index-matched), the angular structure is not something to learn. It is exact geometry. The only nontrivial transport object is a scalar, direction-free, source-independent, symmetric volumetric resolvent G [S: Agent 1 §3.2].

H2 tests one claim. If the learned part of K is restricted to this one object, discretised on a boundary-adapted basis and conditioned only on relative geometry, and both angular ends are exact reciprocal geometric projections, then source transfer becomes a question of **geometric generalisation of a scalar kernel**. It stops being a question of per-source angular extrapolation.

## 2. Specific historical failure targeted

| Historical failure | Evidence | What H2 does to it |
|---|---|---|
| **BND4_FAR transfer** (A75 con_coarse 0.80 / 0.82 group; s107 at 1.19 vs per-source oracle 0.20–0.31). Rank 16 reaches the G2 oracle on training (0.131 vs 0.129) but not far boundary | WL75 [S, V via ideation §0 spot check of WL75 l.128/186/195] | The learned object never sees a source. Three of the four BND4_FAR sources (s101, s105, s107) are port-lattice translates/reflections of S16-R sources s50 / s52 / s53 [V: `ports.json`, `ports4.json` positions; ~1 % corner influence is S/E]. Under H2 they get **identical** K up to a saturating second-plane distance feature (§7) |
| **CP-pairing contrast floor** (G0 0.27 / G2 0.13 even with per-source oracle T). Incidence enters only through the global A_r ↔ T_r pairing with an angle-free T, while the physical effect is a 0.3–0.8 mm first-collision footprint shift | WL74A [S] | Incidence enters only through the exact first-collision projection Π_X, at sub-port depth and lateral position. There is no global angular atom and no pairing |
| **FIA's learned exit / quadrature noise** (training contrast 0.69–0.75, native 0.32; WL77 projection floors 0.453 / 0.385 / 0.352) | WL76, WL77 [S]. Emulation suggests q = 1024 one-ray SS noise reproduces those floors (Agents 1 and 2; non-evidence) | (a) The exit projection is exact geometry, not learned. (b) Every geometric term (Π, SS, esc) is a deterministic cubature with no RNG. Its error is measured separately (D1) |
| **Identity channels and late drift** (WL71B: U mass near fit ports 0.06 → 0.56, corner V 0.03 → 23.6; best transfer at 0.5–2k steps) | WL71A/71B/73/75/76 [S] | The hypothesis class contains no per-source quantity: no absolute coordinates, face id, global receiver basis or free per-source mass. Drift can only happen in unsupervised regions of the relative-geometry feature space |
| **No reciprocity tie** between incoming and outgoing structure in any tested family | Agent 1 §4 [S] | Reciprocity holds by construction (one shared Π, symmetric Γ), off the passivity guard (§3.5) |

Not targeted by construction: the deformation (DEF) gates. They are only routed through exact current-geometry projections and current-geometry features (§6, §13).

## 3. Mathematical operator definition

### 3.1 Semantics (unchanged from the fixture)

- **Ports.** P = 3840 boundary ports of 2.5 mm.
- **Input cells** X = (i, a): incoming bin a ∈ 16, labelled by the outward direction, cosine-weighted within the bin, unit flux uniform on the port.
- **Output cells** Y = (j, o): outgoing bin o ∈ 64, nested in the 16 incoming bins by `bin_parent`.
- **Étendue.** g_X = A_i π/16 per 16-bin cell; π/64 per 64-bin cell.
- **Flux.** RGB flux in and out:

      Φ_out(j,o,c) = Σ_{i,a} K_c(i,a → j,o) Φ_in(i,a,c)

- **Split.**

      K = B + K_scat

  - B is the existing ballistic matrix, unchanged: geometry-only, q = 256 quadrature [V `r57_geometry.py:80`].
  - K_scat is the collided first-exit part. It is what the S16-R targets contain ("collided means" [V `fit73.py:35`]).
- **Ownership.** A photon leaves K at its first boundary crossing. Re-entry belongs to C.

### 3.2 Exact identity that H2 discretises

The identity is [S: Agent 1 §3.2]. I checked the normalisation by change of variables [E]:

    g_X K_scat,c(X→Y) = 4π σ_t ω_c ⟨P_Y, G_c P_X⟩,     G_c = (I − ω_c S)⁻¹ = I + ω_c S G_c

- P_Z(x) = (1/4π) ∫ e^{−σ_t t(x,ω)} 1[ray from x leaves the medium first through port k in bin b] dω is the escape kernel. It includes in-medium visibility.
- S(x,y) = σ_t e^{−σ_t r} / (4π r²) · vis_medium(x,y).
- The first-collision density of a unit-flux cell source is (4πσ_t/g_X) P_X.

Split G = I + (G − I):

    K_scat,c = K_SS,c + K_M,c
    K_SS,c(X→Y) = (4πσ_t ω_c / g_X) ⟨P_X, P_Y⟩                          (single scatter: exact geometry, channel enters only as ω_c)
    K_M,c(X→Y)  = (4πσ_t ω_c / g_X) ⟨P_Y, (ω_c S G_c) P_X⟩             (≥ 2 collisions)

### 3.3 Boundary-layer basis (explicit, geometry-only)

- **Node set.** A boundary-graded tensor grid in **material (rest) coordinates**, pushed forward exactly by the pose map. For this fixture the dent map is a volumetric map, linear in z [V `p53_geom.py:168-173`].
  - Lateral nodes x, y: the 41 port corners at 2.5 mm pitch from −50 to 50 mm, plus graded nodes at ±49.75, ±49.5, ±49.0 mm, giving 47 per axis.
  - Depth nodes z (mm from the bottom face): 0, 0.25, 0.5, 1, 2, 3.5, 5, 6.5, 8, 9, 9.5, 9.75, 10, giving 13.
  - Total N ≈ 47 × 47 × 13 = **28,717 nodes** [E].
  - Every face, edge and corner gets ≤ 0.5 mm resolution in its own normal direction within the first mm. This meets Agent 1's requirement of a boundary layer relative to every nearby face (§7 of that report; port columns of a single face lose 11–23 % of energy at edges [S]).
  - The grid is invariant under D4h. In the flat interior it is invariant under port-pitch translations along each face, except where the side-face grading enters.
- **Basis.** φ_e is a nonnegative trilinear hat on that grid. The φ_e form a partition of unity on the medium (Σ_e φ_e ≡ 1), which survives the push-forward. Volume weight V_e = ∫φ_e.
  - Hats preserve the **first moment** of the first-collision position, so the 0.3–0.8 mm lateral footprint shift appears as barycentric weights. That information is lost by piecewise-constant port columns.
- **Declared refinement (Stage 0 rule only, §14).** Variant L2 refines the lateral pitch to 1.25 mm in the near zone (r ≤ 5 mm pairs) and uses a two-level restriction for the far zone.

### 3.4 Explicit geometric factors (deterministic, shared by both ends)

- **Π.** Π_Z(e) = ∫ φ_e(x) P_Z(x) dx for every 64-bin cell Z. The 16-bin input rows are the sums of their 4 nested children.
  - **Reciprocity:** the same table serves as source projection (Π_X) and exit projection (Π_Y). There is no label reversal, because both bin sets carry outward labels in the same triangle frames [S: Agent 1 §4; the incoming label convention is S, the frames are p53].
  - **Computation (deterministic):** the inward-ray deposit identity. By reciprocity the first-collision density of cell Z, (4πσ_t/g_Z) P_Z, is the attenuated track-length density of rays launched from port k inside bin b. Launch a **fixed stratified set** of 4 × 4 positions × 4 × 4 directions per 64-bin cell, the latter through the concentric map at equal projected solid angle. Integrate σ_t e^{−σ_t s} · φ_e **analytically** along each segment, up to the first boundary exit in the current mesh, which gives in-medium visibility for free. There is no RNG.
- **Escape.** E_e = Σ_Y Π_Y(e) = ∫ φ_e esc. These are the exact row sums of the same table, so they are consistent by construction.
- **Element budget.** m_e = 1 − E_e / V_e: the φ_e-average non-escape probability.
- **SS.** SS(X→Y) = (4πσ_t/g_X) ⟨P_X, P_Y⟩, a deterministic product cubature on a **fine voxel layer**.
  - Fine voxels: 0.3125 mm lateral; depth layers graded from 0.0625 mm to 0.5 mm, down to 4 mm.
  - P_X and P_Y on these voxels come from the same inward-ray deposit primitive.
  - SS is symmetric in g-weighted form, translation-shared at rest (one block per translation/D4h class), and explicit inside the dent footprint.
  - It replaces the q = 1024 one-exit-ray estimator (`fia76.py:79` [V]).

### 3.5 Learned factor Γ_θ and its passive symmetric construction

Per channel c:

    K_M,c^raw(X→Y) = (4πσ_t ω_c / g_X) · Σ_{e,f} Π_X(e) Γ_c(e,f) Π_Y(f)

- Γ_c(e,f) approximates the lumped-mass Galerkin kernel of ω_c S G_c, ⟨φ_e, ω_c S G_c φ_f⟩ / (V_e V_f). It has units of 1/volume and is symmetric.
- It is nonzero only for |x_e − x_f| ≤ ρ_c, with ρ = 15 / 10 / 5 mm for R / G / B.
  - Basis: lowest-mode lateral decay 3.6 / 3.1 / 1.7 mm [S]. The L2 tail beyond 15 mm is ≤ 0.6 % [S: Agent 3 §3].

**Construction (ratio-min normalisation):**

    h_c(e,f) = softplus(NN_θ,c(ψ(e,f)))  ≥ 0,   ψ swap-symmetric (§6)
    R_c(e)   = Σ_f h_c(e,f) E_f
    t_c(e)   = m_e · sigmoid(NN'_θ,c(ψ_node(e)))                      ∈ (0, m_e)
    Γ_c(e,f) = h_c(e,f) · min( t_c(e)/R_c(e), t_c(f)/R_c(f) )

Properties:
- Γ_c(e,f) = Γ_c(f,e) ≥ 0.
- The row response r_c(e) = Σ_f Γ_c(e,f) E_f ≤ t_c(e) ≤ m_e.
- All normalisers are **local**: R_c(e) sums only within ρ_c. There is no softmax over all P.

**Deviation from the ideation card (deliberate).** The ideation formula was Γ = g · min(m_e,m_f) / max(R_e,R_f). It cannot represent the physical kernel Γ* where the budget varies across the boundary layer [E, analytic]:
- Write ρ_e = R_e / r*_e. At the element where ρ is largest, the fixed-point condition forces r*_e ≥ the corresponding m.
- In practice, a deep element whose exit runs through shallower elements with a smaller m loses a fraction of order 1 − m_shallow/m_deep. Near a face, m_e runs from about 0.5 at the surface to about 1 in depth, so the deficit could reach tens of percent in red.

With ratio-min, h = Γ* and t = r* reproduce Γ* **exactly** (both ratios equal 1). This is the representability property the mechanism needs.

### 3.6 Final operator and guarantees

    K_c(X→Y) = B(X→Y) + s_c(X) · [ ω_c SS(X→Y) + K_M,c^raw(X→Y) ]
    s_c(X)   = min(1, (1 − β_B(X)) / E_c^raw(X)),   E_c^raw(X) = Σ_Y [ω_c SS + K_M,c^raw](X→Y)  (one sparse product per rebuild)

- **Linear in illumination; zero in → zero out.** K is a fixed matrix per geometry and θ. s_c(X) depends on geometry and θ, never on illumination.
- **Nonnegative.** B, SS, Π, Γ ≥ 0.
- **Passive.**
  - Raw bound [E, derivation in §8]: E_c^raw(X) ≤ ω_c (1 − β_Π(X)) up to the lumped interpolation error.
  - The guard s_c makes Σ_Y K_c(X→Y) ≤ β_B + (1 − β_B) = 1 **exactly**, for every input cell, seen or unseen.
  - The guard also absorbs the mismatch between the historical q = 256 β_B and the deterministic β_Π.
- **Reciprocal.** g_X K_scat^raw(X→Y) = 4πσ_tω_c [⟨P_X,P_Y⟩ + Π_Xᵀ Γ_c Π_Y] is exactly symmetric. Hence A_i K16(i,a→j,b) = A_j K16(j,b→i,a) holds to float precision wherever s_c = 1.
  - The fraction and magnitude of s_c < 1 are **reported, not assumed**. Expected: only grazing/edge cells, by ≲ 1–3 % [E].
- **Correspondence.** Port/bin indices are untouched. Π is evaluated in the per-triangle bin frames of p53.
- **Not guaranteed.** PSD of the étendue-weighted K_M (physically true, Agent 1 §4) is not enforced. It is a reported property.

## 4. Representation and information ownership

| Object | Owner | Source-dependent? | Learned? | Notes |
|---|---|---|---|---|
| B (ballistic) | K, existing | yes (geometry) | no | unchanged q = 256 quadrature |
| Π (first-collision / escape projection onto the hat basis), E_e, m_e | K, explicit geometry | the X row is source geometry; nothing stored per source | no | deterministic; carries **all** incoming-angle (contrast) and **all** outgoing-angle structure, sub-port depth/lateral position, edge/corner truncation (side-exit fraction 0.25–0.56 [S]) and in-medium visibility |
| SS (single scatter, n = 0) | K, explicit geometry | geometry only | no | 16–41 % of collided exit [S]; deterministic cubature |
| σ_t, ω_c | material constants | no | no | used as in FIA/B (`fia76.py` header [V]) |
| Γ_c(e,f): scalar, direction-free n ≥ 1 resolvent between basis elements | K, learned shared | **no** | **yes** (h, τ networks) | sees only invariant relative geometry of the element pair; no bin, angle, source id, absolute coordinate or face id |
| t, R (normalisers), s_c(X) (guard) | K, derived | s_c is per-cell geometry+θ, not a table | derived | local sums; s_c expected = 1 almost everywhere |
| Air paths between medium points across a dent | **C** | — | — | excluded: Π uses first exit; Γ's training targets are walker first-exit references (no air continuation [S]) |

The S16-R supervision is the **only** training signal. Reciprocity is structural, not an extra label. Γ_phys (§14) is a **declared privileged diagnostic** and is never used in training.

## 5. Source-conditioning mechanism

A source cell X enters K only through two rows of deterministic geometry:
- **Π_X.** About 80 nonzeros over basis nodes [E]. It holds the first-collision mass 1 − β_Π(X), its depth profile (mean ≈ cos θ mm) and its lateral shift (≈ sin θ mm, 0.30 / 0.74 / 0.82 mm for inner / mid-side / corner bins [S]), truncated by every nearby face.
- **SS(X, ·).** The exact single-scatter row.

No learned function takes X, its port index, its position, its bin or its angle. The model therefore **cannot distinguish a supervised source from an unsupervised one** except through geometry: two cells with congruent geometric neighbourhoods get congruent rows of Π and SS, and therefore congruent K. Incoming-angle dependence (contrast) is

    K(X_a) − K(X_b) = ω[SS_a − SS_b] + Π_Yᵀ Γ (Π_{X_a} − Π_{X_b})

The bin difference is an exact geometric dipole/depth difference in the first-collision footprint, propagated by the same Γ. A smooth multiplicative error in Γ produces the same **relative** error in contrast. It does not produce a contrast floor of the CP-pairing type (WL74A).

## 6. Geometry-conditioning mechanism

**Explicit.**
- The node grid is pushed through the current deformation. All coordinates, frames and neighbour lists stay in material coordinates, so rigid motion needs **zero** rebuild.
- Π, E, m, SS and esc are recomputed from the current mesh, only where they change: the edit footprint plus the Π/SS reach of about 5 mm.
- In-medium visibility comes from ray casting in the current mesh.

**Learned, through invariant features ψ(e,f).** All features are scalars, so they are exactly invariant under D4h and port-lattice translations:
- **Per element.** For each boundary face patch within 2ρ_c: the distance from the node to the patch's closest point, d, and exp(−d/L_c) with L = 18.3 / 5.8 / 1.9 mm. Also the local thickness along the normal of the nearest face, the Jacobian of the pose map, and the normal spread.
- **Per pair.**
  - r = |x_e − x_f|.
  - For each face patch p near the pair: (d_e^p, d_f^p), the in-plane separation along p, and the **image distance** |x_e − M_p x_f| (method-of-images feature).
  - The in-medium fraction of the segment e–f, and the minimum thickness along it.
- **Encoding.**
  - Per-plane embeddings are summed: permutation-invariant over planes, as in F1's plane set.
  - A plane-pair interaction term has saturating features (edges, corners, vertical edges).
  - Swap symmetry: ψ(e,f) = ν(e,f) + ν(f,e), so h(e,f) = h(f,e) exactly.
- **Channels.** Three heads, each with its own range ρ_c.

**Dents.** The dent of pt0 (5.75 mm deep) changes G through thickness and lateral confinement over the dent footprint plus about 1.7–3.6 mm [S]. H2 routes this through (i) exact Π/SS on the deformed surface (tilted frames, changed first-collision footprints) and (ii) thickness and image features in ψ. Training sees two poses (rest, pt0), so the learned part of the deformation response rests on two geometries (§13).

## 7. Why transfer to unseen sources is expected

1. **Exact symmetry images are identical inputs.**
   - The learned object depends only on invariant element-pair geometry, and the grid is D4h-invariant and translation-invariant along faces away from grading and dents.
   - So for any port-lattice translate or point-group image X′ of a supervised X, the rows Π_{X′} and SS(X′,·) are permutations of those of X.
   - Γ entries on the corresponding element pairs are equal **up to features that see a different second plane**.
   - For s101 / s105 / s107 vs s50 / s52 / s53, the different second plane is at e₂ = 41–48.75 mm vs 18.75 mm in training [V positions]. The red saturating feature exp(−18.75/18.26) ≈ 0.36 in training vs ≤ 0.10 for the far translates [E]. That is an interpolation toward the interior value (e₂ → ∞), which every interior S16-R source supervises.
   - This is the same quotient argument as F1, but on a **scalar** object with no bins.
2. **Contrast and outgoing structure are not learned.** The contrast floor WL74A attributes to the CP pairing, and the exit-lobe problem of WL76/77, have no counterpart: both ends are Π.
   - What remains is that Γ must be accurate on element pairs of the unseen footprint. A smooth relative error ε in Γ yields contrast error ≈ ε (§5), not ε divided by the 5–17 % contrast signal [S: |HZ|/|Z| 0.054–0.17].
3. **Supervision density of Γ.**
   - Each S16-R source-geometry constrains Γ(e,f) for e in its footprint (about 100 nodes) against all f within ρ_c (about 1,750 per node, all channels) [E], i.e. about 1.75 × 10⁵ entries.
   - The interior part of Γ is a 3-parameter function per channel (Δρ, z_e, z_f) after the quotient, and every interior source oversamples it.
   - Each boundary source supervises pairs at its edge distance against receivers at all distances up to 15 mm from it. By symmetry Γ(e,f) = Γ(f,e), so receiver-side boundary elements are supervised too.
   - Examples: TBE s50 is 1.25 mm from one edge; the corner TBC s51 is at 1.25 / 1.25 mm; SM s52 sits 3.75 / 6.25 mm from top/bottom [V].
4. **No storage path for source identity.** No absolute coordinates, global U, per-source mass or face id. The WL71B drift mechanism has no parameter to act on. Drift is confined to Γ in unsupervised feature regions, such as corner-offset combinations.
5. **Discretisation evidence (non-evidence emulation, Agent 1 §7 [S]).**
   - With exact SS and depth, snapping k ≥ 2 collisions to 2.5 mm port columns costs native 0.09–0.10, coarse 0.04–0.07 and pair-contrast 0.05–0.08. All are inside the WL59 gates.
   - **Caveat:** H2 also projects the *first* collision onto the basis before the first S step, which that emulation did not do. The hat basis keeps the first moment, but this gap is the main representation risk (§13). Stage 0 measures it directly.

**What is not expected.** For s103 (6.25 / 8.75 mm from two edges, bottom face) no S16-R source shares the two-plane configuration of both ends of the near-diagonal Γ block. Transfer there depends on composition over the plane set, which is genuine extrapolation (§13).

## 8. Physical validity argument

- **Factorisation.** It is exact for isotropic phase, homogeneous medium and index-matched boundary. It is independently consistent with WL76 Stage 1: the decomposed estimator vs archived references passes 36/36, and the forward control is rejected 4/4 [V `docs/worklogs/76_*.md:75,82`].
- **Positivity and linearity.** Every factor is ≥ 0. K is a fixed matrix.
- **Passivity, raw bound** [E derivation].
  - Σ_Y K_M^raw(X→Y) = ω (4πσ_t/g_X) Σ_e Π_X(e) r_e, with r_e ≤ t_e ≤ m_e = 1 − ⟨esc⟩_{φ_e}.
  - Hence E^raw(X) ≈ ω ∫ c₁_X (esc + Σ_e φ_e r_e) ≤ ω(1 − β_Π). The approximation is lumped: the φ_e-average replaces the pointwise value.
  - The guard s_c removes the residual approximation and the β_B/β_Π mismatch, so passivity is **exact**.
- **Energy at ω → 1.** Physically r(x) = 1 − esc(x) at ω = 1. Ratio-min lets Γ reach t_e = m_e·τ with τ → 1. The only structural energy loss is the guard and the lumped average.
  - Estimate [E]: the depth derivative of esc is log-singular at a face (E₂′ = −E₁), so linear interpolation of 1 − esc over the first 0.25 mm layer underestimates by up to ~6 %.
  - That layer carries 12–22 % of first collisions, which gives ≲ 1 % red energy deficit. A thinner first layer reduces it.
- **Reciprocity.** Exact physical law here [S: Agent 1 §4; non-evidence MC ratios 0.991–0.995; WL53 at noise]. Exact in H2 off the guard.
- **Equivariance.** Exact under every isometry of the current medium that maps the grid to itself.
- **K/C.** First-exit projection only. Cross-dent air segments are excluded from Π and SS by first-boundary stop, and Γ is trained on first-exit references.
- **Scope limits.** Anisotropic phase (g ≠ 0), Fresnel boundaries, heterogeneity or surface BSDFs make G direction-dependent. H2's "exact angular ends" claim then fails, and angular moments of G would be needed [S: Agent 1 §3.4].

## 9. Capacity and information bottlenecks

1. **Basis floor (representation).** Lumped Galerkin on a 2.5 mm lateral hat basis with graded depth.
   - Expected [E, extrapolated from Agent 1 §7]: native 0.05–0.15, coarse 0.03–0.08, contrast 0.05–0.15.
   - Unknown: the cost of projecting the first collision (not only k ≥ 2) onto the basis. Measured by Γ_phys in Stage 0.
2. **Near-diagonal block.** 32–53 % of collided exit leaves through the source port, and 66–94 % within 3.6 mm [S]. This is carried by Γ's self and nearest-neighbour entries, which average the integrable 1/r² singularity of S. It is translation-invariant in the interior and supervised there by all 12 retained interior S16-R sources. Near edges and corners it is supervised only at the 4 new boundary sources' configurations.
3. **Two-plane configurations.** Corners, vertical edges (SV/UNS) and corner-offset points (s103) need plane-pair interaction terms seen at only a few configurations: TBC s51 at 1.25 / 1.25, and SM/SR with vertical edges at 18.75 mm [V]. This is the binding *information* limit of H2.
4. **Deformation.** Two training poses. G's dependence on thickness and curvature is sampled once (pt0), at radius 22.5 mm and depth 5.75 mm [S].
5. **Normaliser coupling.** If the learned t/R ratio is not smooth between neighbours, the min() wastes energy. This is learnable but adds optimisation coupling.
6. **Range truncation.** ρ = 15 / 10 / 5 mm costs about ≤ 0.6 % red L2 [S].
7. **Capacity is not the issue.** Γ is one scalar per element pair per channel, with a small MLP over about 30 invariant features [E]. The function class is far smaller than ATO's angular atoms, so over-capacity per-source memorisation is not possible in the flat interior.

## 10. Approximate computation and state cost (all [E] unless cited)

**State.** Fp32, per geometry, at 2.5 mm lateral (L1):

| Object | Size |
|---|---|
| Γ (3 channels, symmetric half; about 1,750 neighbours per node summed over channels) | about 2.5 × 10⁷ floats ≈ **0.10 GB** |
| Π (64-bin; ~70 nnz per cell × 245,760 cells) | 1.7 × 10⁷ ≈ **0.07 GB** |
| Π16 (or derived on the fly) | +0.02 GB |
| SS, explicit for all 61,440 input cells × ~1,200 output cells | 7.4 × 10⁷ ≈ **0.29 GB** |
| SS, translation-shared at rest | < 0.01 GB |
| SS, pt0 dent footprint only (about 6.6k cells) | about 0.03 GB |
| E, m, t, R, s | < 0.001 GB |
| B (existing) | 79 MB [S] |
| **Total** | **≈ 0.2 GB typical, ≈ 0.55 GB worst case** (fully explicit). Historical: W+UV 15–30 M floats; FIA 1.5 GB [S]. Within the preferred ≤ 0.5 GB with sharing; within the 2 GiB hard cap in every layout |
| Variant L2 (1.25 mm near zone) | about +0.12 GB |

**Apply.**
- Per light: Π16ᵀx (≈ 1.5 × 10⁷), Γ (≈ 5 × 10⁷), Π_out (≈ 5 × 10⁷) and SS (≈ 2.2 × 10⁸) MAC, about 3.4 × 10⁸ in total.
- 64 lights: about 2.2 × 10¹⁰ MAC. The state is read once per batch, so the roofline is about 1–2 ms.
- At the observed W+UV implementation efficiency (about 100× roofline [S]): **0.05–0.2 s**. Historical A75: 0.054 / 0.086 s on a 3080 Ti; hard cap 0.86 s.
- 1-light: roofline ≈ 0.6 ms; realistic 2–5 ms. That is at risk against the 2 ms preference and under the 5 ms cap.

**Rebuild.**
- Rigid: **0**.
- Learned coefficients (Γ network over unique symmetric pairs, plus R, t and the guard product): about 2.5 × 10⁷ pairs × ~6k MAC ≈ 1.5 × 10¹¹ MAC, giving **≈ 0.02–0.1 s**. Far fewer pairs at rest after feature deduplication.
- Deterministic geometry, full pose: Π deposits of 245,760 × 256 = 6.3 × 10⁷ rays with analytic segment integration, **≈ 1–2 s**. This is from B's ≈ 88 M rays/s [S] plus a 2–3× traversal overhead.
- SS for a dent footprint: about 4 × 10¹⁰ MAC fine-voxel dot products, **≈ 1–3 s**.
- **The deterministic SS rebuild for a nonrigid edit is the most likely budget violation** (hard cap 1 s for extra quadrature). Still about 25–75× cheaper than FIA's 75 s.
- Selective invalidation footprint: the edit plus Π/SS reach (about 5 mm) plus 2ρ_red (30 mm, through the normaliser R). It is local, with no global softmax.

**Training.**
- Per step, for the S16-R source-geometries: Γ rows over footprints, plus R over the union of neighbourhoods. About 1.7 × 10⁸ h-evaluations without deduplication; **about 2 × 10⁷ with feature-class deduplication** (rest: ~10³ classes; pt0 dent: ~10⁴ nodes).
- **≈ 0.1–0.6 s/step** on two lanes with chunked recomputation.
- Lane memory: model plus activations about 1–2 GB, on top of about 3.7 GB of targets [S]. That is tight against 5.1 GB per lane; the fallback is one lane.
- Two seeds × 20k steps ≈ 1–7 h.
- One-off cost of Γ_phys (Stage 0 diagnostic): about 2,400 node rows × 10⁶ photons with Galerkin tallies, **≈ 3–10 GPU-min** (scaled from about 2.1 s per 32 M-photon packet [S]).

## 11. Nearest prior art

Verification status from Agent 4 unless marked; **[mine]** = my addition, not checked by Agent 4 or by me online.

- **First-collision source method** (neutron transport; OSTI 6326046; RAY3D/DONTRAN3D) [V-sec]. Exact uncollided term plus first-collision source plus a collided solve. H2's P_X is this projection.
- **Jensen et al. 2001** (exact single scatter plus a dipole for multiple scattering) and **Photon Beam Diffusion** (Habel, Christensen, Jarosz 2013) [V]. The split "exact SS + kernel for the multiply scattered part, sources along the incident beam" is classical.
- **Wang, Tran, Luebke 2005**, All-frequency relighting of translucent objects with single and multiple scattering [V-sec]. A precomputed surface-to-surface translucent transfer matrix (single scatter separate), wavelet-compressed. A non-learned, per-object transfer relative of Πᵀ Γ Π.
- **Direct-to-Indirect Transfer** (Hašan, Pellacini, Bala 2006) [V]. A source-agnostic linear operator on a first-bounce field.
- **Element Learning** (Du, Stechmann, arXiv 2308.02467) [V]. Learned per-element in2out / in2sol RTE maps from element geometry, assembled by a solver: the closest learned-RTE precedent.
- **DeepRTE** (arXiv 2505.23190) [V abstract]. Green's-function learning with delta inflows.
- **GreenNet / Deep Generalized Green's Functions / VGF** [S, snippet level]. Explicit singular part plus learned correction (elliptic).
- **Boullé–Townsend 2022** [V-sec]. Green's-function recovery from input–output pairs.
- **[mine]** GreenLearning (Boullé, Earls, Townsend 2022): a neural G(x,y) learned from forcing/solution pairs.
- **[mine]** Galerkin and hierarchical radiosity (symmetric kernel between basis functions).
- **[mine]** Arbree, Walter, Bala 2011 (FEM diffusion for heterogeneous SSS).
- **[mine]** Symmetric matrix scaling (Sinkhorn–Knopp / Knight), for symmetric normalisation.
- **Vicini, Koltun, Jakob 2019** and **Deep Scattering 2017** [V-sec]. Geometry-conditioned learned scattering relative to the incident point.
- **8DNA** [V]. Passivity as mass × normalised distribution, which is **not** what H2 uses: H2 uses a symmetric ratio-min with exact escape budgets.
- **Not prior art for this mechanism:** RNA and 8DNA as baselines. Their role terms follow `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`.

## 12. Plausible novel contribution

**Mechanism-level (weak).**
- (a) The **ratio-min symmetric passive normalisation** Γ = h·min(t_e/R_e, t_f/R_f) with geometric escape budgets. It is exactly symmetric, locally normalised, passive (with the column guard) and exactly able to represent the physical kernel. I found no precedent, but this is a weak negative: it is a small algebraic device near the symmetric matrix-scaling literature.
- (b) Learning a **direction-free, source-independent, geometry-conditioned volumetric resolvent between two exact reciprocal first-collision projections** on a face-graded hat basis, with full 16 × 64 port/bin output.

**Combination-level (what can honestly be claimed).** Classical exact angular ends (first-collision source method, Jensen 2001 / PBD), a Galerkin symmetric kernel and Green's-function learning (DeepRTE, GreenLearning, Element Learning), combined so that:
- (i) the learned object has no angular index and no source index;
- (ii) reciprocity and per-cell passivity hold for every unseen source;
- (iii) it is evaluated on **held-out sparse-source transfer** (16 supervised of 3,840 ports, 0.42 % [S]), with port/bin correspondence, near edges and corners.

Agent 4 found no study of (iii) (O1, a weak negative). Without a positive transfer result, the contribution is the combination only.

**Contrast with F1 and F3.**
- F1 learns K's angular blocks. H2 learns no angular content.
- F3 learns a local generator closure and solves it iteratively. H2 learns the long-range resolvent directly: no solve at apply, three sparse products.

## 13. Strongest potential counterexample

**(A) Representation counterexample, strongest: it can kill H2 even with perfect learning.**
- The coarse contrast signal at a cell-centre sub-cell is only |HZ|/|Z| ≈ 0.054 [S: WL74A, Agent 3]. The WL59 contrast gate is relative (≤ 0.25 of that).
- In H2, the multiple-scatter share of contrast flows through Πᵀ Γ (Π_{X_a} − Π_{X_b}). The first-collision footprint difference between bins, a 0.3–0.8 mm shift and a 0.5–0.93 mm depth change, is projected onto 2.5 mm lateral hats before the first S step. The hats keep its first moment but smear its shape.
- Agent 1's favourable 0.05–0.08 pair-contrast number kept the first and second collisions exact and snapped only k ≥ 2. It does **not** cover H2's discretisation.
- If the shape loss removes about half of the M-part contrast at centre sub-cells, the exact Γ_phys floor will show con_coarse > 0.25 on S16-R training sources. The mechanism then fails at L1 and possibly L2, independent of learning.
- A shape-preserving alternative is a second-collision projection Π = element projection of S P_X, plus an exact double-scatter term. It costs about a further 4D cubature per pair class and is outside this specification.

**(B) Learning counterexample.** For s103 (corner offset, 6.25 / 8.75 mm from two edges), and for the SV/UNS vertical-edge contexts that fail in every model (1.0–1.5 [S]):
- The near-diagonal Γ block needs both ends at two-plane distances that no S16-R footprint has.
- The plane-pair interaction term is then extrapolated from TBC s51 (1.25 / 1.25) and from pairs at e₂ = 18.75 mm.
- Red still feels the second plane at 6–9 mm (exp(−6.25/3.6) ≈ 0.18 of the lowest-mode amplitude [E]).
- If H2 fixes the three translates but s103 stays at A75 level (≈ 0.8), the mechanism is shown to be a symmetry quotient plus exact angular ends, **not** a geometric generaliser.

**(C) Contract counterexample (human decision, not falsification).** In this fixture, Γ_phys is computable by geometric MC in minutes per geometry. If learned Γ_θ only matches Γ_phys where supervised and is worse elsewhere, a non-learned per-geometry Γ_phys (≈ 3–10 GPU-min [E]) dominates the learned one in accuracy. The learned part then buys only rebuild time (0.05 s vs minutes). Whether that is the intended "learned K" is for the user to decide.

## 14. Minimal falsifying experiment (this fixture, S16-R budget)

**Fixed:**
- the fixture and the unchanged walker references;
- eval68 estimators and the WL59 gates;
- the S16-R fit targets (replicas 0–3) and held-out replicas 4–7;
- the historical evaluation groups: TRAIN_NEW / TRAIN_KEEP, BND4_NEAR / BND4_FAR / INT4 (`eval74.py:56`), and the WL73 BND groups.

**Not used:** the Gate 4 population (`ports76.json`) stays reserved.

**Privileged-label declaration:** Γ_phys is a diagnostic only, never used for training.

**Stage 0: no learning (falsifies the representation and the geometric terms).** About 15 GPU-min [E].
- **0a Determinism and cubature floor.**
  - Π, E and SS at base vs 2× stratification per dimension.
  - Rule (predeclared): the change in native / fine16 / coarse-contrast space must be ≤ 0.2 × the replica half-split noise. SS_det must agree with the phys76 order-split single-scatter tally (z² within the WL76 Stage 1 tolerance) on 4 S16-R cells.
  - Failure falsifies H2-C3.
- **0b Exactness tests (float precision).**
  - Reciprocity of g_X K_scat^raw.
  - Σ_Y K ≤ 1 for all 61,440 cells.
  - Bitwise-identical ψ features and K rows for exact D4h images.
  - Identical K rows (up to the e₂ feature) for port-lattice translates.
  - Failure is an implementation bug, not a hypothesis result.
- **0c Representation floor.**
  - K_phys = B + SS + Πᵀ Γ_phys Π, with Γ_phys from the unchanged walker physics. Emission density φ_e/V_e; tally Σ_k ω^k φ_f(x_k) / V_f; 10⁶ photons per row, for rows reached by the evaluated sources.
  - Scored with eval68 on S16-R held-out replicas (training sources) and, descriptively, on the independent groups.
  - Predeclared rule on **training sources only**: L1 (2.5 mm) passes if all three WL59 gates hold (worst channel, jackknife hi). Otherwise try L2 (1.25 mm near zone). If L2 also fails, **STOP: H2 falsified as a representation** (H2-C2).

**Stage 1: learning.**
- Train h_θ and τ_θ for the selected L on the S16-R fit targets (rest + pt0) with the historical AR/A75 objective J, imported unchanged. *I did not re-read J's terms; to be checked when the protocol is frozen.*
- 2 seeds, a fixed 20k-step budget, final checkpoint. The descriptive unseen-source trajectory (D3) is logged, with no selection.

**Stage 2: falsifiers (fixed before any Stage 1 result).**
- **F-T (core transfer claim, H2-C6).** On the rest-pose BND4_FAR translates s101 / s105 / s107, learned con_coarse (jackknife hi, worst channel) must be ≤ max(0.25, K_phys floor + 0.10) for **both** seeds, and the group must improve on A75 (0.80 / 0.82).
  - Failure means the source-independent Γ with exact Π does not deliver symmetry transfer, so the mechanism is falsified.
- **F-G (geometric generalisation, H2-C7).** Learned − K_phys error gap on independent groups (BND4_FAR incl. s103, BND4_NEAR, WL73 BND_FAR) vs the same gap on training sources.
  - Ratio ≥ 2 attributes the limit to Γ's feature generalisation, not the representation.
  - It is backed by a direct element-pair comparison of Γ_θ vs Γ_phys, binned by plane-distance features.
- **F-D (descriptive, H2-C9).** DEF rel / corr on the historical deformation set (cf*, te*) for learned and K_phys, separating representation from learning.
- **F-C (cost, H2-C8).** Measured state, apply64 / apply1, learned-coefficient rebuild and deterministic dent rebuild, against the hard caps (2 GiB, 0.86 s / 5 ms, 1 s, 1 s) on the 3080 Ti, including roofline counts.

**Total.** About 15 GPU-min for Stage 0 and about 2–8 GPU-h for Stage 1 on one 12 GB GPU [E]. Stage 0 alone can falsify the representation (counterexample A) before any training.

---

## Key claims for reviewers to attack

- **H2-C1.** For this fixture, every incoming- and outgoing-angle dependence of K_scat is carried by the exact reciprocal escape projection P. The only object that needs learning is a scalar, direction-free, source-independent symmetric resolvent between basis elements.
- **H2-C2.** At the L1 basis (2.5 mm lateral trilinear hats, depth graded to 0.25 mm at every face), the learning-free floor K_phys meets all three WL59 gates on S16-R training sources. Expected [E]: abs_coarse ≤ 0.08, con_coarse ≤ 0.15. Projecting the *first* collision onto the basis does not destroy the 0.3–0.8 mm incidence information.
- **H2-C3.** Deterministic stratified-ray / analytic-segment cubature makes Π and SS errors ≤ 0.2 × replica noise in every metric space. This removes the q = 1024 one-ray noise that dominated FIA and the WL77 projection floors.
- **H2-C4.** The hypothesis class has no source-identity channel. Exact D4h images get bitwise-identical K rows. Port-lattice translates differ only through a saturating second-plane feature.
- **H2-C5.** Ratio-min normalisation plus the column guard gives exact positivity, linearity and per-cell passivity for every unseen source. Reciprocity is exact wherever the guard is inactive (expected ≥ 97 % of cells [E]). Unlike the ideation's min/max form, it can represent the physical Γ exactly.
- **H2-C6.** Learned RBG brings the BND4_FAR translates (s101, s105, s107) to con_coarse ≤ 0.25, against A75 at 0.80–1.19, for both seeds.
- **H2-C7.** A Γ trained on 16 sources × 2 poses generalises across relative boundary-geometry features. Independent-source error stays within 2× the K_phys floor, including the near-boundary group. This is the weakest claim; s103 and SV/UNS are its stress points.
- **H2-C8.** Cost is plausible: state ≤ 0.6 GB, learned rebuild ≤ 0.1 s, 64-light apply ≤ 0.26 s, rigid rebuild 0. The deterministic SS/Π rebuild for a dent edit is 1–3 s [E]. That exceeds the 1 s cap and is a declared risk.
- **H2-C9.** Exact current-geometry Π and SS carry the first-collision and exit part of a deformation response. Whether the learned Γ change from two poses meets DEF (rel ≤ 0.35, corr ≥ 0.90) is open and is reported descriptively, not claimed.
