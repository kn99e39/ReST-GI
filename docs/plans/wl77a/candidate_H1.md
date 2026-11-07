# WL77A Candidate H1: Canonical-Quotient Relative Kernel (CQR) (2026-11-07)

**Status.** Hypothesis specification only. It is not adopted, not authorised and not implemented. No GPU job, training, server access or new Monte Carlo was run for it. The WL77 files and all historical worklogs and results are untouched. The Gate 4 confirmation population (`ports76.json`) and its references were **not read**. Only its selection rule (`experiments/fnto_fia76/ports76.py:1-5`) and the WL77A Part 0 description were read.

**Family.** F1 of `docs/plans/wl77a/ideation.md`. The other two candidates are F2 (RBG: an exact projection Πᵀ Γ Π with a learned scalar resolvent) and F3 (LCR: a learned closure of a volumetric generator, then a solve). H1 is distinct from both:
- H1 learns the **angular-spatial pair kernel itself**, including the incidence and exit blocks.
- It applies no exact projection operator and no solve.
- Its bet is that a **symmetry quotient plus removal of identity channels** turns most unseen sources into seen inputs, and that the rest is a smooth interpolation in canonical geometry.

**Verification legend.** [V] = checked by me in code or results (file:line given). [S] = taken from a specialist report and not rechecked. [E] = my estimate. [U] = unverified.

---

## 1. Problem

The question is how to build a learned intrinsic operator K for this fixture that stays accurate for **source ports never supervised**, especially boundary sources far from the 16 S16-R sources, without per-source tables.

The fixture:
- Homogeneous slab 100 × 100 × 10 mm; σ_t = 1000 m⁻¹; ω = (0.999, 0.99, 0.9); g = 0; index-matched boundary.
- P = 3840 ports of 2.5 mm; 16 incoming × 64 outgoing bins; RGB flux.
- Supervision is S16-R at rest and pt0. pt0 is a dent of depth 5.75 mm, radius 22.5 mm, centred at (5, −5) mm (`experiments/fnto_ato74/protocol/ato74_v1.json:6-11`) [V].

Every learned family so far fits its supervised sources and then degrades with distance from supervision:
- W+UV rank 8 and rank 16, RSK and the symmetry loss all do this.
- FIA does not even fit its supervised sources.

H1 tests one specific explanation. The degradation comes from how the source enters the model (absolute identity and non-equivariant frames), and from the global CP angular pairing. It does not come from missing information or missing capacity.

---

## 2. Historical failure targeted

| Target | Evidence | Status |
|---|---|---|
| **C1. Identity channels in W+UV.** Node inputs include absolute rest and current xyz and an ordinal face id. U is a softmax over **all P** receivers, and the mass is a free sigmoid per source and mixture. | `experiments/fnto_rule57/r57_geometry.py:58` (`local=[xyz0/.05, xyz/.05, …, port_face/5]`); `r57_model.py:59-63` (`U=softmax(…,dim=0)`, `mass=sigmoid(…)`) [V]. WL71B: U mass near fit ports 0.06 → 0.56, corner V 0.03 → 23.6 [S]. Late drift: best transfer at 0.5–2k steps (WL71A/73/75/76) [S]. | Code [V]; drift mechanism inferred [S] |
| **C2. Symmetry not quotiented.** The face frames differ per face. 12 of the 16 slab operations induce face-dependent local maps. The WL72 symmetry term was angle-integrated augmentation. | WL74A §3.1 / §3.3 frame table [V, `docs/worklogs/74A_…:92-119`]; WL74A P11, `fit72.py:39-41` [S]. | [V] |
| **Images and translates fail.** Three of four BND4_FAR sources are rest-pose images or translates of supervised sources, yet A75 scores BND4_FAR con_coarse 0.803 / 0.824 against a G2 oracle of 0.159. | `results/fnto_ato74/ports4.json` [V]: s101 (−48.75, −8.75, −5) f4 TBE; s105 (−50, 1.25, −1.25) f0 SM; s107 (−8.75, −50, −3.75) f2 SR. These are translates of s50 (48.75, −31.25, 5) f5, s52 (−50, 31.25, 1.25) f0 and s53 (−31.25, −50, 3.75) f2 (`results/fnto_src73/ports.json`) [V]. WL75 table `docs/worklogs/75_…:186-195` [V]. | [V] |
| **C3. The global CP pairing floor.** K/d = Aᵀ diag(T_ij) O with angle-free T leaves G0 0.27 / G2 0.13 even with a per-source oracle T. | WL74A P1 and §4 item 3 (\|HZ\|/\|Z\| about 0.05 at a centre sub-cell vs 0.17 at a corner) [V, lines 59, 144-150]. | [V] |
| **Orientation excess.** Aligned shared refits remove it: at M = 8, TBE 0.17–0.19 raw vs 0.064–0.075 aligned. | WL74A §3.4 [V, line 132]. | [V] |

**Not targeted:** the FIA estimator noise (C4; there is no MC geometric term inside H1's operator) and deformation (DEF71/73/4). Deformation is only routed (§6).

---

## 3. Operator definition

### 3.1 Semantics and the ballistic/scattered split

**Inputs and outputs.**
- Input x(i, a, c) is incident flux into source cell (port i, incoming bin a, channel c). Bins are labelled by the outward direction the light comes from (`p53_tracer.cell_source:90-101`) [V].
- Output y(j, o, c) is first-exit outgoing flux at (port j, outgoing bin o, channel c).
- y = Σ_{i,a} K_c(i, a → j, o) x(i, a, c).

**The split.** K_c = B + K^s_c.
- **B** is the unchanged geometry-only zero-collision operator (`r57_geometry.ballistic_matrix`, q = 256 fixed seed) [V].
- β(i, a) is the column sum of B, and d(i, a) = 1 − β(i, a), both unchanged.
- B, β and d are kept bit-identical to the historical arms, so the eval68 normalisation and the comparison with W+UV / A75 stay valid.

### 3.2 Canonical frames and bin permutations

**Native frames.** Each port has its native frame F_i = (t_i, b_i, n_i) from `tri_frames` [V, WL74A §3.1].

**Bin grids.** Both grids are Shirley–Chiu concentric squares, with bin index iu + N·iv (`p53_geom.local_to_bin:51-56`) [V]. Local D4 acts on (iu, iv) by the index-square operations u ↦ N−1−u, v ↦ N−1−v and u ↔ v. These are exact permutations on both the 16-bin and the 64-bin grid, and `bin_parent` nests the 64 bins into the 16 (purity 1.0, WL74A §3.2) [V]. Write Q^{16}_g and Q^{64}_g for the permutations induced by g ∈ D4.

**Source frame candidates.** For source i and g ∈ D4 (acting in the (t_i, b_i) plane with n_i fixed), the candidate canonical frame is C_i(g) = F_i ∘ g.

**Frame weights π_g(i).** These are continuous, equivariant and convex, with π_g ≥ 0 and Σ_g π_g = 1.
- Compute the in-plane boundary moment m₁(i) = Σ_p ω(r_ip) A_p e^{−r_ip/ℓ₁} P_{t,b}(x_p − x_i). The sum runs over boundary ports p ≠ own-face neighbours, with r_ip = |x_p − x_i|, ℓ₁ = 3.6 mm and ω the window of §6.
- π_g(i) is a smooth indicator that g·m₁(i) lies in the canonical sector 0 ≤ arg ≤ 45°, with soft edges of ±3°.
- As |m₁| → 0, π blends smoothly to uniform over D4. Ties and the interior then average over the stabiliser exactly.
- Resulting weights:

| Source class | π |
|---|---|
| Interior | uniform over all 8 |
| TBE, SR, SM | 2 elements (reflection about the edge-normal axis) |
| TBC on the diagonal | 2 elements |
| Generic corner offset (e.g. 6.25 / 8.75 mm) | 1 element |

**Receiver frame given g.** For each receiver j, the canonical receiver frame is the native frame F_j rotated by the D4 element whose t-axis best aligns with the projection of C_i(g)'s t-axis onto j's tangent plane. If that projection is degenerate, b-axis alignment is used instead. This gives Q^{64}_{g,ij}.
- At rest all faces are axis-aligned, so the alignment is exact.
- On curved dent ports a residual in-plane angle ψ_ij ∈ (−45°, 45°] remains. It is passed to the network as a feature, so the snap is approximate there and the model has to learn the residual.

### 3.3 Learned direct kernel

For channel c, source cell (i, a), receiver (j, o) and g ∈ D4, write ã = Q^{16}_g(a) and õ = Q^{64}_{g,ij}(o). Then

  K^s_c(i, a → j, o) = d(i, a) · Σ_g π_g(i) · m_c(g; i, ã) · Π_c(g; j, õ | i, ã)

  Π_c(g; j, õ | i, ã) = 1[j ∈ N(i)] · Σ_{k=1}^{R(i,j)} α_{c,k}(g; ã | i, j) · β_{c,k}(g; õ | i, j) / Z_c(g; i, ã)

  Z_c(g; i, ã) = Σ_{j ∈ N(i)} Σ_k α_{c,k}(g; ã | i, j)   (each β_{c,k}(·) is a softmax over the 64 bins, so it sums to 1)

**Support N(i).** All ports with rest-coordinate Euclidean distance ≤ R_f = 15 mm. This is radius-based, not kNN, so it is exactly congruent under lattice translations. It is fixed topology across poses.
- **Near core, |Δ| ≤ R_n = 6 mm:** about 21 receivers on a flat face (distances 0, 2.5, 3.54, 5, 5.59 mm), more at edges [E]. Rank R = 4, with α_k ∈ ℝ₊^{16}, β_k ∈ Δ^{63} and 64 free logits per k and channel.
- **Far zone, 6 < |Δ| ≤ 15 mm:** R = 1. The outgoing lobe is β(õ) = softmax_õ(λ₀ log Ω̂(õ) + λ·u(õ)). Here u(õ) is the mean exit direction of bin õ in the canonical receiver frame and λ ∈ ℝ³ is predicted. This is the order-1 receiver-relative dipole that Agent 1 lists as the minimum outgoing content for g = 0 [S].
- **Outside N(i):** zero. The target energy there is reported as a fixed truncation error. Agent 3 measured the L2 tail beyond 15 mm at ≤ 0.6 % on fitting means [S].

**Learned heads.** All are MLPs with fp32 weights shared by all sources, receivers and poses.
- m_c(g; i, ã) = sigmoid(f_m(s_i(g), φ_FC(i, ã; g), u_ã(g))) ∈ (0, 1).
- α_{c,k}(g; ã | i, j) = softplus(f_α(s_i(g), φ_FC(i, ã; g), u_ã(g), q_ij(g)))_{c,k}. It is evaluated per (pair, bin).
- β_{c,k}(g; · | i, j) = softmax(f_β(s_i(g), φ̄_FC(i; g), q_ij(g))_{c,k,·}). It is evaluated per pair and is shared by all 16 incoming bins (pair-level CP). Per-pair 16 × 64 blocks are rank ≈ 4 to 1.7 % residual, red interior, oracle SVD (Agent 3 S4) [S].

The features s, φ_FC, u and q are defined in §5–§6. **None of them contains absolute coordinates, a face id, a port id or a source id.**

### 3.4 Guarantees by construction

| Property | Mechanism |
|---|---|
| **Linearity, zero → zero** | The coefficients depend on geometry only and never on x, so y is a fixed matrix times x. |
| **Positivity** | sigmoid, softplus, softmax, d ≥ 0 and π ≥ 0. B ≥ 0. |
| **Passivity** | Σ_{j,o} K^s(i, a → j, o) = d(i,a) Σ_g π_g m_c(g; …) ≤ d(i,a) = 1 − β(i,a). Adding B's column sum β gives ≤ 1 per (i, a, c), the same budget as the WL59/ATO contract (`ato74_v1.json` operator string) [V]. A convex combination over g preserves this. |
| **Correspondence** | Ports are native. Bins return to the native indices through the inverse permutations inside the sum. |
| **Exact equivariance** | Let h be a slab symmetry or lattice translation that maps the support and the ω-windowed context of source i congruently onto those of source h(i). Then every input feature is identical after the frame map, π transforms by π_{g}(h(i)) = π_{g·h̃⁻¹}(i), and K^s(h(i), Q a → h(j), Q o) = K^s(i, a → j, o) holds exactly up to d. The deployed d is fixed-seed MC: WL75 found sorted-β row mismatches between images, so identity holds only up to the β/d noise (WL75 dev note `75_…:115`) [V]. |
| **No per-source table** | There are no free per-source, per-port or absolute-receiver parameters. Two sources with congruent windowed geometry must receive identical predictions. A per-source lookup table is therefore **outside the hypothesis class**. |
| **K/C ownership** | K models only in-medium first-exit transport (as do the S16-R targets). There is no air term. Cross-dent air paths stay with C. In-medium segment clearance is a *feature*, not an ownership decision. |

**Not guaranteed (measured property only):** reciprocity A_i K16(i, a → j, b) = A_j K16(j, b → i, a) [S, Agent 1 F2]. H1 deliberately does not build it in, to stay distinct from F2/F3. The violation is reported.

---

## 4. Representation and information ownership

| Quantity | Owner | Computed how | Source-dependent? |
|---|---|---|---|
| B, β, d | Explicit geometry (historical) | Fixed-seed q = 256 ray quadrature, unchanged | Yes, through geometry only |
| D4 bin permutations Q^{16}, Q^{64} | Explicit, exact | Index-square operations | No (a frame relation) |
| Frame weights π_g(i) | Explicit, deterministic | Boundary moment m₁ of the current geometry | Through the local geometry only |
| Source context s_i(g) | Learned encoder over explicit geometry | DeepSets over windowed boundary ports in C_i(g) (§6) | Through the local geometry only |
| First-collision features φ_FC(i, a; g) | Explicit, deterministic | Port-local QMC chords with analytic exponential integration (§5) | Through the local geometry only |
| Pair features q_ij(g) | Explicit geometry | Offsets, normals, clearance, receiver context moments in C_i(g) | Relative only |
| m, α, β | **Learned shared rule** f_m, f_α, f_β | MLPs | Never by identity |
| Reciprocity, the A–O tie | Not owned | None | Measured only |
| Free-space coupling, re-entry | C (not K) | n/a | n/a |

**Inference-time source dependence.** It enters only through (a) the source's local geometry, (b) the incoming bin's canonical direction u_ã, and (c) exact first-collision features. Nothing is looked up.

---

## 5. Source conditioning

**1. Canonical incoming direction.** u_ã(g) is the bin-mean direction of ã in C_i(g) (3 numbers, plus the bin's projected solid angle, which is constant). Because the frame is canonical, "bin 7" means the same physical direction relative to the nearest boundary for every source in the same symmetry class. That was not true for the per-face native indexing of A/O (WL74A §3.3) [V].

**2. Deterministic first-collision features φ_FC(i, a; g)** (11 numbers, expressed in C_i(g)):
- the mean first-collision offset μ ∈ ℝ³ relative to the port centre;
- its covariance (6 numbers);
- the uncollided escape fraction β̂(i, a);
- the collided-mass depth fraction within 1 mfp.

They are computed per (i, a) with one fixed **port-local** QMC pattern: 16 Sobol positions on the port × 16 Sobol directions in the bin, i.e. 256 chords per cell. The same pattern is used for every port, so exact images receive bit-identical features at rest. Each chord is cast once against the current mesh, and the truncated-exponential moments along the chord, ∫₀ᵗ σ e^{−σs}(x + s·d) ds, are analytic.

**What this carries.** The incidence-dependent footprint shift of about cos θ mm in depth and about sin θ mm laterally (WL74A §4 item 1) [V], including its truncation by edges and the dent. This is the quantity the angle-free T could not carry (C3). Here it enters the **pair-specific** α as a feature, not through a global A_r ↔ T_r pairing.

**What this is not.** It is not FIA's projection: there is no first-collision density operator and no anchors. It is a fixed-size descriptor.

**3. Per-bin mass m(i, ã).** This is a function of the same canonical features. It replaces W+UV's free per-source sigmoid mass.

**Source-side information the model never receives:** xyz, face id, port index, rest coordinates, slot or population membership, and any global field such as W+UV's 11 × 11 dent grid (`r57_geometry.py:60-62`) [V].

---

## 6. Geometry conditioning

**Window.** Boundary context is windowed: ω(r) = 1 for r ≤ 6 mm, smoothstep to 0 at r ≥ 12 mm. The physical reason [E, from Agent 1's lateral decay lengths of 3.6 / 3.1 / 1.7 mm (R/G/B) [S]]:
- A plane whose distance from both endpoints is ≥ 12 mm can only enter through paths at least about 18 mm longer than the direct pair distance (≤ 6 mm in the near core).
- Its influence on near-core blocks is then about e^{−18/3.6} ≈ 0.7 % (red).
- This physical difference is the only one the window ignores.

**Source context s_i(g) ∈ ℝ⁶⁴.** It is computed as follows:

  s_i(g) = Σ_{p ∈ ∂, r_ip ≤ 12 mm} ω(r_ip) (A_p / A₀) · φ_θ( C_i(g)ᵀ(x_p − x_i)/ℓ, C_i(g)ᵀ n_p, {e^{−r_ip/ℓ_k}}_{k} )

with ℓ_k ∈ {1, 3.6, 10} mm.
- It is a permutation-invariant point-set encoder over current boundary ports.
- At rest it encodes planes (own face, the opposite face at the local thickness, side planes within 12 mm), including which side they are on. RSK's direction-free scalar (`rsk72.py:27-28`) and FIA's sorted free paths (`fia76.py:39-43`) dropped that information [S, ideation C5].
- Under a dent it encodes the curved surface directly. No tangent-plane approximation is needed.

**Pair features q_ij(g).** All are in C_i(g) and use current geometry:
- the offset Δ (3 numbers), |Δ| and log|Δ|;
- n_j (3 numbers) and n_i·n_j;
- the residual receiver-frame angle ψ_ij (cos and sin);
- the in-medium straight-segment clearance fraction (a fixed 8-point deterministic check against the current mesh);
- the receiver's windowed boundary moments M0, M1 (3) and N1 (3) at ℓ ∈ {1, 3.6, 10} mm, rotated into C_i(g). That is 21 numbers.

These moments are computed once per port in the world frame. Rotating them per pair costs one 3 × 3 product. They give each receiver the side and distance of its own nearby planes relative to the source frame. This is how the e₂-type plane near a far receiver enters, only where the receiver really is near it.

**Current geometry only.** No rest-pose feature enters f_m, f_α or f_β. The rest coordinates only define the fixed neighbour topology N(i).
- The physical reason: transport in a homogeneous medium depends only on the current shape. A rest-relative displacement field would be an extra pose-memorisation channel.
- Rigid edits change no feature, because every feature is intrinsic. So the rebuild cost of a rigid edit is zero.

**Deformation route.** A dent changes the boundary point set inside the window, the clearance, the normals, ψ and φ_FC. No new mechanism is added. Two training poses (rest, pt0) are the only deformation supervision. This is declared as a bottleneck in §9.

---

## 7. Why transfer to unseen sources is expected

The mechanism has four parts, each tied to evidence.

**(a) Images and translates become seen inputs.** At rest, with the 12 mm window, an along-edge translate whose second-edge distance exceeds about 12 mm receives source context and near-core pair inputs identical to its supervised partner.
- This holds for s101 ↔ s50 (e₂ 41.25 vs 18.75 mm), s105 ↔ s52 and s107 ↔ s53 [V, positions above; the congruence is my geometric reading of the port lattice, consistent with Agent 1 F5 [S]].
- It also holds for every interior source with all side planes beyond 12 mm, which includes every S16-R interior source (edge distance ≥ 21.25 mm, `ports.json` audit) [V].
- **Far-zone receivers differ legitimately.** For s50 the far-zone receivers with y ≤ −38 mm see the −y plane; for s101 they do not. That is a real physical difference.
- **Prediction.** The error at those translates equals the supervised partner's held-out error, plus the ≤ ~1 % physical out-of-window difference [E; Agent 1's corner-influence estimate is ~1 % [S]], plus the β/d noise.
- **Contrast with A75.** A75 scores 0.80–0.82 there [V], while the per-source G2 oracle is 0.159 [V]. The WL74A aligned-refit experiment (0.17–0.19 → 0.064–0.075) is the oracle-level measurement that alignment removes the multi-orientation burden [V].

**(b) Incidence transfers through geometry, not through atoms.** The incidence dependence is a pair-specific α(ã | i, j), driven by the exact footprint moments φ_FC. The supervised loss therefore fits it per pair, without the global CP pairing whose floor is G0 0.27 / G2 0.13 [V]. Sub-cell position is a property of the coarse metric, not of the operator: H1 is translation-equivariant at port level, and the 12 S16-R interior sources cover centre, edge and corner sub-cells. Port-level accuracy therefore transfers to any sub-cell. This follows from ports.json [V]: 2085 (3.75, −6.25) centre, 1025 (−28.75, −6.25) edge, 1123 (−28.75, 28.75) corner.

**(c) Identity storage is removed, which removes the drift channel.** The WL71B drift components all lie in parameters that do not exist in H1 [S]:
- U concentrating on fit ports;
- the corner V blow-up;
- the slot-5 mass.

Any null-space direction left in H1 must move the prediction of *every* source with the same canonical inputs, including supervised ones. Motion along such a direction is therefore penalised.

**(d) Unseen boundary configurations are interpolated in canonical geometry.** Example: a top-face source 3.75–8.75 mm from an edge.
- Receiver-side plane distances are densely supervised. Every training footprint that comes within 12 mm of a plane samples receivers at many plane distances. That covers the four boundary sources plus four interior sources at 21.25 mm, whose 15 mm support reaches within 6.25 mm of a side plane.
- Source-side, the perpendicular-plane distances seen are 1.25 mm (TBE, TBC, SR), 3.75 / 6.25 mm (SM: top and bottom planes) and 8.75 mm (SR: bottom plane), plus "absent" for the interior [V, coordinates in ports.json].
- φ_FC supplies the exact first-collision truncation at any distance.
- This part is interpolation, not identity. §13 gives the case where it is expected to break.

**What H1 explicitly does not expect to transfer:** sources outside the quotient orbit **and** outside the source-side interpolation range. These are vertical-edge (SV) contexts, which have no S16-R analogue (`ports.json` S16R_sources: SV = 0) [V], and strongly dented contexts unlike pt0.

---

## 8. Physical validity argument

- **Linearity, positivity, zero → zero and passivity (≤ 1 − β collided plus β ballistic)** hold by construction (§3.4), for every parameter value, geometry and frame weight. This is the same passive budget as WL59/ATO, so the K∘C composition gets the same spectral bound argument as historical K.
- **Symmetry.** For g = 0, isotropic, homogeneous, index-matched media, the exact K is equivariant under every isometry of the medium [S, Agent 1 P1]. H1 enforces exactly the subgroup that maps the port lattice and bins onto themselves: in-plane D4 at each port (exact bin permutations) composed with lattice translations. It enforces this only as far as the windowed context is congruent. The out-of-window approximation is a declared physical bias of ≤ ~1 % [E].
- **Frame averaging.** At symmetric configurations (interior, the diagonal corner), the stabiliser average makes the predicted block exactly invariant under the configuration's own symmetry. The physical block has this invariance; a single arbitrary frame would break it.
- **The incidence effect** is carried by exact geometry (φ_FC) and learned per pair. This matches the derivation that, for g = 0, the incoming angle acts only through the first-collision distribution (WL74A §4 item 1 [V]; Agent 1 F1 [S]).
- **What H1 gives up physically:**
  - Exact reciprocity: not enforced, measured.
  - The exact exit projection: outgoing structure is learned as rank-4 per pair near and a dipole far. Agent 1 states the order-1 dipole is the minimum [S].
  - Energy beyond 15 mm: about 0.6 % L2 [S].
- **Ballistic part.** It is exact geometry and unchanged. Dent self-shadowing in B stays as historically certified.

---

## 9. Capacity and information bottlenecks

1. **Effective training diversity after the quotient is small.** At rest, the 16 S16-R sources collapse to about **5 canonical source classes** (interior, TBE, TBC, SM, SR) [V, contexts in ports.json; collapse my reading]. The 12 interior sources become replicas of one class at different sub-cell positions. pt0 adds about 16 dented contexts with thicknesses 4.4–10 mm [V, `thickness_mm_pt0`]. The source-side boundary function is therefore identified from **4 boundary source configurations**. Receiver-side coverage is much richer (hundreds of pairs per footprint). Whether that suffices to pin down the source-side α at unseen configurations is the core risk.
2. **Precision demand.** Contrast is 1–3 % of response energy [S, WL74A §6.6 via Agent 2]. At centre sub-cells (\|HZ\|/\|Z\| ≈ 0.05), a smooth 1 % block error already gives con_coarse ≈ 0.2–0.3 [E]. On interpolated inputs, an MLP must be accurate to about 1 %.
3. **Representation floor.**
   - Rank-4 near-core blocks: 1.7 % pair-L2 residual, oracle SVD on supervised sources [S].
   - Far zone: rank-1 plus a dipole. Untested; it carries ≤ 11 % of native L2 (outside 7.5 mm) [S].
   - Truncation at 15 mm: ≤ 0.6 % L2 [S].
   - The resulting **parameterisation floor on con_coarse is unmeasured**. §14 Stage A measures it first.
4. **Approximate canonicalisation on dents.** ψ_ij is learned from two poses. Rest-trained equivariance does not certify dented ports.
5. **β/d noise** breaks exact image identity at the level of WL75's β-row mismatch [V, dev note].
6. **No reciprocity tie.** Contrast at an unseen source cannot borrow its outgoing observations (Agent 1 F2 / Agent 2 F12) [S].
7. **Capacity is not expected to bind.** WL75 reached the G2 oracle on training with 273k parameters [V]. H1's MLPs are of similar size (≈ 0.3–0.6 M parameters [E]).

---

## 10. Computation and state cost

All figures are [E] unless cited. They assume rest geometry, P = 3840, fp32 and TF32 off. Pair counts are radius-based: near ≈ 23 receivers per source (8.8 × 10⁴ pairs), far ≈ 100 per source (3.8 × 10⁵ pairs).

| Item | H1 estimate | Historical (cited) | Agent 3 budget |
|---|---|---|---|
| **Persistent state** | Near: 8.8e4 pairs × (16·4 α + 4·64 β) × 3 ≈ 8.5e7 floats. Far: 3.8e5 × (16 α + 4 lobe) × 3 ≈ 2.3e7. Plus m and Z ≈ 2e5. **Total ≈ 1.1e8 floats ≈ 0.43 GB** (0.22 GB in fp16) | W+UV 14.7 M / 29.5 M floats; FIA 1.47 GB [V/S] | ≤ 0.5 GB preferred, ≤ 2 GiB cap |
| **Extra geometry quadrature** | φ_FC: 61,440 cells × 256 chords = 15.7 M casts, the same as B's quadrature (B 0.18 s for 15.7 M rays) [S]. Clearance: 4.7e5 pairs × 8 points, ~3.8 M point tests | B 15.7 M rays; FIA 126 M casts | ≤ 16 M casts / 0.5 s preferred |
| **Coefficient rebuild** | Near f_α per (pair, bin) × mean \|supp π\| ≈ 5: 8.8e4 × 16 × 5 ≈ 7e6 evaluations of a ~5e4-MAC MLP ≈ 3.5e11 MAC. f_β per pair: 4.4e5 × ~3e5 MAC ≈ 1.3e11. Far: ~1e11. **≈ 6e11 MAC → about 0.05–0.2 s** on an RTX 3080 Ti. At rest, lattice congruence allows caching per unique canonical pair class, which is far cheaper | W+UV 0.014–0.015 s; FIA 75 s [S] | ≤ 0.1 s preferred, ≤ 1 s cap |
| **1-light apply** | ≈ 1.9e8 MAC; reads 0.43 GB of state → ≥ 0.5 ms at the bandwidth roofline; **1–4 ms realistic** with a grouped small-GEMM kernel | AR 1.07–1.37 ms (RTX 5080) [S] | ≤ 2 ms preferred, ≤ 5 ms cap: **at risk** |
| **64-light apply** | 1.2e10 MAC (state reused across lights); **≈ 0.1–0.5 s** with an index_add-style kernel like W+UV's (~100× roofline [S]), less if fused | A75 0.086 s (3080 Ti) [S] | ≤ 0.26 s preferred, ≤ 0.86 s cap |
| **Nonrigid update** | Selective: sources whose 15 mm support plus 12 mm receiver window reach displaced ports, i.e. dent radius + 27 mm. For te1 (r = 17.5 mm) that is about 40–60 % of ports. Local Z, no global normalisation. B and geometry preparation (0.3–0.8 s) charged separately | WL59 dent S/F 0.98–0.99 because of the global U softmax and the B rebuild [S] | Charge B separately |
| **Rigid update** | Zero coefficient work (intrinsic features) | — | Zero |
| **Training step** | 32 (source, pose) × ~123 receivers × ≤ 8 frames: ≈ 3e4 pair and 5e5 (pair, bin) evaluations. Targets cropped to the support (~1.2e7 floats per pose set). The truncated target energy enters J as a constant. **≈ 0.05–0.3 s/step, < 2 GB per lane** | AR/A75 0.20 s/step, peak 4.3 GB [V, WL75 line 118] | ≤ 0.5 s/step, ≤ 5.1 GB per lane |

**Scaling caveat.** The near-core state scales as h⁻⁴ in port pitch [S, Agent 3 S10]. At 1.25 mm pitch it would be about 5.5 GB, so it is fixture-specific. Finer ports would need β generated from a compact per-pair coefficient set evaluated inside the apply.

---

## 11. Nearest prior art

From `specialist_priorart.md`, with Agent 4's verification marks unless stated otherwise.

| Work | Relation to H1 |
|---|---|
| Vicini, Koltun, Jakob, *Learned subsurface scattering*, TOG 38(4) 2019, DOI 10.1145/3306346.3322974 (V). Its polynomial degree and frame canonicalisation are U | Shared learned model conditioned on source-centred local geometry, plus absorption × exit sampler (mass × distribution). It is the closest learned precedent for "shared source-relative conditional response". |
| Kallweit et al., *Deep Scattering*, arXiv 1709.05418 (V) | Hierarchical relative descriptors around the shading point and the light. |
| 8DNA (arXiv 2604.25129, V) | α × p: passivity by mass × normalised distribution. Not novel in H1. |
| Hackbusch 1999, H-matrices (V-sec); PRT / CPCA, Sloan et al. 2002–2003 (V-sec) | Near-dense / far-compressed block split and per-cluster low-rank blocks. H1's rank-4 near / rank-1 far is an instance. |
| Helwig et al., G-FNO, ICML 2023, arXiv 2306.05697 (V); Satorras et al., EGNN, ICML 2021, arXiv 2102.09844 (V) | Equivariance by construction over augmentation. |
| Loos et al., *Modular Radiance Transfer*, SIGGRAPH Asia 2011 (V) | Transport on canonical blocks reused across placements. |
| Zeltner et al. 2024, learned shading frames, arXiv 2305.02678 (V) | Canonical local frames. |
| **Not in Agent 4's list (cited from my own knowledge, links not fetched, [U]):** Puny et al., *Frame Averaging for Invariant and Equivariant Network Design*, ICLR 2022 (arXiv 2110.03336); Kaba et al., *Equivariance with Learned Canonicalization Functions*, ICML 2023 (arXiv 2211.06489) | Direct precedent for the weighted frame/stabiliser averaging and canonicalisation in §3.2. |

---

## 12. Plausible novel contribution

**At the mechanism level H1 is mostly a combination.** Each part has precedent:
- canonicalisation and frame averaging (Puny / Kaba [U]);
- relative geometric conditioning (Vicini, Deep Scattering);
- near/far low-rank blocks (H-matrices, CPCA);
- mass × distribution passivity (8DNA, Vicini).

**What may be new, stated narrowly and as a weak negative** (Agent 4 O1, O3; search-depth limited):
1. **The quotient of a bounded medium's boundary-to-boundary transport operator by its local isometries, with exact bin permutations on both angular ends and continuous frame weights.** Passivity, positivity and linearity are preserved, so the quotient is compatible with a K∘C composition. No equivariant boundary-operator instance was found by Agent 4 [S].
2. **A held-out sparse-source transfer study:** 16 one-hot sources to unseen boundary ports, with port/bin correspondence and independent references. Agent 4 found no such evaluation (O1) [S].
3. **An attribution claim specific to ReST-GI:** that the historical far-boundary failure is a representation-symmetry failure (identity channels plus non-quotiented frames plus CP pairing), not an information limit. H1 is designed so that this claim can fail cleanly (§14).

If H1 succeeds, the claimable contribution is (2) plus (3), with (1) as an engineering enabler. If it fails at translates, it refutes (3).

---

## 13. Strongest potential counterexample

**Source-side boundary configurations outside the quotient orbit, where the incidence block must be extrapolated by an MLP to about 1 % precision.**

- **Concrete case 1: vertical-edge (SV) sources.** A side-face source 1.25–6.25 mm from a vertical slab edge has its own side face, a perpendicular side plane within 12 mm, and the top and bottom planes. No S16-R source has this source context (SV = 0) [V].
- **Concrete case 2: near-corner top/bottom sources** at (6.25, 8.75) mm, like s103.
- **Gate 4 includes these.** The confirmation population includes an SV context and far strata chosen to maximise distance to S16-R *images* (`ports76.py:3-4`) [V].
- **Why it should fail:** the incoming-azimuth-dependent side-exit fraction (0.25–0.41 at an edge port, 0.37–0.56 at a corner, 0.55–0.70 on the top side row; WL74A §4 item 5) [V] is a source-side function. In training it is pinned only at {1.25 mm on the top/bottom face; side-face rows; absent}.
- **What φ_FC does and does not cover:** it provides the exact first-collision truncation, but not the multiply-scattered side exit. Most of the lateral redistribution happens over 3.6 mm decay lengths after the first collision.
- **What a failure would look like:** BND-near-type (3.75–8.75 mm) and SV sources at con_coarse ≈ 0.4–1.0, while translates pass. That would mean the quotient is necessary but not sufficient. The remaining failure would be **source-context learnability** under 4 boundary configurations, the regime in which F2/F3 compute the incidence side rather than learn it.

**A second counterexample is the precision argument (§9 item 2).** Even exactly-seen interior inputs inherit the training-fit error. If the supervised fit settles at about the G2 level (0.13), translates are bounded by that level too. H1 cannot beat the quality of its own supervised fit.

---

## 14. Minimal falsifying experiment

**Fixture and budget.** Unchanged slab, S16-R supervision (rest + pt0; fitting replicas 0–3 → `fit_targets73.pt`), held-out replicas 4–7, eval68 estimators, WL59 gates. Two seeds. The objective is the family68 J (mean of native / fine16 / coarse relative squared error plus coarse contrast; `family68.py:80-92`) [V], Adam with cosine LR and 20k steps as in fit71 (`fit71.py:97`) [V]. No privileged labels and no reciprocity-derived labels.

**Stage A: parameterisation floor (no learning, a few GPU-minutes).**
- Fit the H1 *output class* with free per-pair α, β and m, per supervised source, to the fitting replicas. Score on held-out replicas.
- **STOP (representation falsified)** if TRAIN con_coarse hi > 0.10 or abs_coarse hi > 0.06 on either geometry. A family whose own oracle sits near the gates cannot carry the transfer claim.
- Unit tests at the same time:
  - image identity of the inputs and outputs at rest under all 16 slab operations and lattice translates, to ≤ 1e-6 relative excluding d;
  - linearity, zero → zero and column sums ≤ 1;
  - inverse bin-permutation round trips.

**Stage B: training (2 seeds; plus the control arm H1-noQ, 2 seeds).**
- **H1-noQ** is identical to H1 except for frames: native frames F_i, π = δ_id and no permutations. It isolates the quotient.
- **Fit gate:** TRAIN_NEW must pass the WL59 gates on both geometries. Otherwise the transfer test is uninformative and is reported as INSUFFICIENT FIT.
- Record the unseen-source trajectory on the WL73/WL74 development groups at fixed checkpoints, descriptively, with no checkpoint selection.

**Stage C: predictions (development, descriptive).** These sources were seen while H1 was designed, so they are not blind.
- (P1) s101 / s105 / s107 con_coarse within +0.05 of their supervised partners s50 / s52 / s53, at the partner's held-out value.
- (P2) INT4 ≤ 0.20.
- (P3) final/@2000 unseen-source ratio ≤ 1.2.

**Stage D: falsifier on the frozen, never-scored Gate 4 population** (WL77A Part 0: 26 static slots + 4 te1 footprint slots; references generated under the unchanged contract) [V, `77A_…:26-33`]. H1 is **falsified** if, for both seeds, any of the following holds:
- **F-a:** pooled far-stratum con_coarse hi over the contexts covered by the quotient (TBE, TBC, SM, SR far) is > 0.40. That is less than about two thirds of the gap closed from A75's level (~0.80) toward the gate.
- **F-b:** the far/near ratio of con_coarse is > 1.5. Transfer would then still be supervision-local.
- **F-c:** H1 and H1-noQ differ by less than the seed spread on far strata. The quotient would then not be the operative mechanism, even if F-a passes.

**Reported but not falsifying:** SV and the hi/lo sub-cell pairs (the predicted risk, §13); the te1 footprint (DEF is not targeted); the reciprocity violation; cost against §10.

**Compute.** Stage A takes minutes. Stage B is 4 runs × about 0.5–1.7 h [E]. Stage D needs no new references beyond Part 0.

---

## Key claims for adversarial review

| ID | Claim |
|---|---|
| **H1-C1** | With the 12 mm window, s101, s105 and s107 receive source context and near-core pair inputs identical to s50, s52 and s53 at rest. Their errors therefore equal their partners' held-out errors, up to ≤ ~1 % physical out-of-window effect plus β/d noise. A75's 0.80 at BND4_FAR is a representation failure, not missing information. |
| **H1-C2** | The output class (pair-specific rank-4 near core within 6 mm, rank-1 plus receiver-relative dipole to 15 mm, mass × local softmax) has a supervised floor con_coarse ≤ 0.10. The global CP floor (G0 0.27 / G2 0.13) is an artefact of the shared A/O, not of 16 × 64 transport. |
| **H1-C3** | Linearity, positivity, zero → zero, passivity (Σ ≤ 1 − β collided) and exact equivariance at symmetric ties hold for all parameters under weighted frame averaging, and the frame weights are continuous on dented geometry. |
| **H1-C4** | Removing absolute xyz, face id, the global U and free per-source mass removes the WL71B drift channels. Unseen-source error at the end of training is ≤ 1.2× its value at 2k steps. |
| **H1-C5** | The deterministic first-collision features φ_FC carry the incidence × boundary interaction well enough that unseen top-face sources 3.75–8.75 mm from an edge transfer at near-gate contrast. This is the claim most likely to fail (§13). |
| **H1-C6** | Unseen interior sources at any sub-cell position inherit port-level accuracy, because the S16-R interior set spans centre, edge and corner sub-cells. INT transfer is ≤ 0.20 con_coarse. |
| **H1-C7** | Cost stays within Agent 3's hard caps: state ≈ 0.43 GB, rebuild ≤ 0.3 s, 1-light apply ≤ 5 ms (at risk), 64-light apply ≤ 0.86 s, training ≤ 0.5 s/step at < 5 GB per lane. Rigid edits need no rebuild, and nonrigid edits need no global renormalisation. |
| **H1-C8** | If H1 passes F-a but fails SV, the remaining limit is source-side context learnability from 4 boundary configurations. The failure at that point would then argue for computing the incidence side (F2/F3) rather than learning it. |

---

## Uncertainty and items not verified

- **Image congruence.** That the ω-windowed contexts of s101/105/107 match s50/52/53 bit-for-bit is my reading of the lattice positions. It is not computed. The ~1 % out-of-window physics is an estimate from Agent 1's decay lengths. Agent 1 notes a direct reference comparison is possible read-only.
- **Representation numbers.** The rank-4 near-core sufficiency and the support tails are oracle statistics on fitting means of supervised sources (Agent 3) [S]. The rank-1 plus dipole far zone is untested.
- **Costs.** Every cost in §10 is arithmetic and nothing was timed. The 1-light apply is the most at-risk item.
- **Frame-weight behaviour on pt0/te1 dents.** Not analysed numerically. ψ_ij learnability from two poses is unknown.
- **Prior art.** Puny 2022 and Kaba 2023 are cited from my knowledge and were not fetched by Agent 4.
- **Thresholds.** The F-a / F-b thresholds are proposals for the workflow's frozen contract. They were chosen without access to Gate 4 data.
