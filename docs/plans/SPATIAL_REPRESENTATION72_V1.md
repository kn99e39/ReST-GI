# Spatial Representation Comparison v1 — Candidate Design Record (WL72, Stage 0)

User direction dated 2026-11-03. This record is frozen before any candidate is trained or any candidate output on physical data is inspected. The frozen contract is `experiments/fnto_cmp72/protocol/cmp72_v1.json`. Neither candidate is adopted by this document.

## Common contract (both candidates, unchanged)

```text
K x = B x + O^T · T · A · ((1 − β) ⊙ x)
```

- **B** is the geometry-only first-exit zero-collision operator (unchanged).
- **A, O** are the fixed WL68 `net_all_5902` angular bases (8 mixtures, 16 incoming and 64 outgoing bins).
- **T[i → j, r, c]** is the spatial collided transport from source port i to receiver port j for mixture r and channel c.

**Requirements on T:**

- T ≥ 0;
- Σ_j T[i → j, r, c] = mass_i[r, c] ∈ (0, 1), so K is passive;
- K is linear in the incident flux x and zero for x = 0.

Training data, objective J, optimizer and step budget are the WL71 D1 contract:

| Element | Value |
|---|---|
| Data | S16, rest and pt0, fitting replicas 0–3 |
| Objective | WL68 objective J |
| Optimizer | Adam, cosine 3e-3 → 3e-5, 20,000 steps, clip 1.0 |
| Seeds | two per candidate |

## Candidate A — W+UV with exact domain-symmetry consistency

**Representation.** The WL59 P network, unchanged:

- T = scatter(W·scale) + U(V·scale);
- 157,024 parameters;
- spatial-only training: A/O fixed.
- Init is P_5901 and P_5902 with the fixed A/O, identical to WL71 D1, so A differs from the D1 baseline only by the added constraint.

**Physical relation enforced.** Exact symmetries of the optical domain.

- The material is homogeneous and isotropic and the boundaries are index-matched, so the transport operator commutes with any rigid isometry g that maps the domain onto a domain of the same family.
- For the 10 × 10 × 1 cm slab family, the in-plane dihedral group D4 (four rotations, four reflections) maps a dented pose (depth, radius, centre c) onto the pose with centre g·c.
- The reflection z → −z additionally maps the undented rest slab onto itself.
- Measured on the archived rest geometry, all of these map the 3,840-port grid onto itself exactly: maximum position error 0, bijective.

**Quantity constrained.** Incident angles enter through port-local bins, and the fixed A/O are not symmetric. The constraint is therefore applied to the angle-integrated collided response:

```text
F_i(j, c) = Σ_a Σ_o S[i, a, j, o, c]
```

This is the exit flux map for Lambertian (cosine-weighted, all-bin) incidence at port i. It is exactly equivariant:

```text
F^{g·G}_{g·i}(g·j, c) = F^G_i(j, c)
```

**Implementation.** The label for the transformed configuration is the transformed physical training label. No new photons are used; the identity copy is excluded.

The model's integrated response at the transformed configuration is:

```text
Σ_a d[a] Σ_r A[r, a, c] T^{g·G}[g·i → j, r, c]     (Σ_o O = 1)
```

It is evaluated on the transformed geometry: the pose with centre g·c, built with the unchanged geometry code.

**Loss and strength.**

```text
L = J + λ · L_sym
L_sym = mean over (geometry, training source, non-identity symmetry, channel) of ‖F_model − F_label‖² / ‖F_label‖²    (native receivers)
λ = 1, fixed
```

λ = 1 was chosen because L_sym uses the same relative-squared-error normalisation as J's terms. There is no search. The selection method uses no evaluation data.

**Symmetry sets.**

- rest: 15 non-identity elements of D4 × {1, z-flip}.
- pt0: 7 non-identity elements of D4. They yield 4 distinct poses, centred at (±5, ±5) mm.

**Why this is physically justified and not a heuristic.**

- The related configurations are physically identical up to a rigid motion. No equality between merely "similar-looking" sources is assumed.
- Boundaries, dent and thickness are transformed with the source, so the constraint respects every boundary and geometry-state difference.
- It enforces neither locality, nor a W/UV ratio, nor smoothness.

**What it tests.** Whether supervision-local specialisation (WL71, 71A, 71B) is reduced when the same W+UV map must reproduce physically equivalent configurations at symmetry-image source positions. These include all four slab corners when a corner is supervised.

**Structure.** The constraint acts during training; it is not structural. The W+UV map is not made equivariant by construction.

**Fairness rule.** Because the constraint supervises symmetry images of the training sources, every confirmation port must be ≥ 1 cm (same face) from every symmetry image of every training and previously referenced port. The confirmation geometry must not be a symmetry image of rest or pt0. Otherwise A's advantage would be trivial.

## Candidate B — relational shared kernel (RSK)

**Persistent learned rule (θ).** Three small MLPs:

- φ_loc: local transfer;
- φ_far: far transfer;
- φ_m: source mass.

Nothing per port, per destination or per geometry is learned.

**Geometry-dependent state, rebuilt from the current geometry without training.**

- **Port frames** (n_p, t_p, b_p). t_p is a deterministic tangent of n_p.
- **Port descriptors** δ_p, all geometry-only and position-free:
  - local thickness;
  - distance to the nearest non-coplanar boundary (edge distance);
  - area ratio;
  - normal spread;
  - displacement d_p, as in the WL59 local features;
  - the normal.
- **Coarse cells.** The existing 240-cell parent partition, with area-weighted centroid, normal and descriptors.
- **Neighbour graph.** The existing 128-nearest rest-coordinate graph.

**Transport law.**

```text
w_loc(i → j) = softplus(φ_loc(ρ_ij, δ_i, δ_j)) · prior_c(|x_j − x_i|)     for j ∈ N128(i)
w_far(i → J) = softplus(φ_far(ρ_iJ, δ_i, δ_J))                            for every coarse cell J
T[i → j]     = mass_i · ( w_loc(i → j) + w_far(i → J(j)) · a_j / a_J ) / ( Σ_j w_loc + Σ_J w_far )
mass_i       = sigmoid(φ_m(δ_i, mean_J δ_J weighted by w-independent distance kernel))
```

The pair relation ρ is position-free:

```text
ρ_ij = ( (x_j − x_i)·t_i, (x_j − x_i)·b_i, (x_j − x_i)·n_i ) / 1 cm,  |x_j − x_i| / 1 cm,  n_j expressed in the source frame
```

The same applies to cell centroids. Here a_j / a_J is the area share of port j inside its cell J, and prior_c is the WL59 per-channel distance prior.

**Positivity and passivity hold by construction.** All weights are ≥ 0, and the normalisation gives Σ_j T = mass_i < 1.

**Difference from W+UV.**

- **No absolute coordinates.** The WL59 W+UV conditions node states, and hence W, U, V and mass, on absolute rest and current position features through message passing. RSK sees only relational and local-invariant quantities. Two sources with the same relational context (including boundary distance, thickness and the coarse-cell field of the whole domain) get the same transport law, by construction. This is translation sharing.
- **Long range from one shared function.** In W+UV, long-range transport goes through a destination basis U(j) and source codes V(i), which are free per-port functions of absolute features. WL71B showed U concentrating on the fit-port neighbourhoods and the V of an uncovered source exploding. In RSK, long range is the same shared function φ_far applied to every source-to-cell relation. There is no learned destination basis that can specialise to supervised destinations.
- **Mass is a shared function of descriptors,** not of position.

**Long-range representation.** A dense source × 240-cell relation set (P × 240 = 0.92M pairs per geometry) plus the 128-neighbour local set (P × 128). There is no dense P² term.

- **Application cost** per incident field is O(P · (128 + 240)) per (mixture, channel).
- **Runtime state** is the local weights (P × 128 × 24) plus the far weights (P × 240 × 24) plus the descriptors.

**Directional response** comes through the fixed A/O exactly as in W+UV. Mixture weights may vary with the source-frame direction of the relation.

**Geometry change** rebuilds frames, descriptors, cells and pair relations; θ is unchanged.

**Initialisation confound.** RSK cannot be initialised from P_5902. It is trained from seeds 7201 and 7202 with the same optimizer and budget. This is reported as an architectural confound.

**Parameter count.** About 30k, against 157k for W+UV. The count is reported; capacity is not matched.

**What it tests.** Whether an explicitly shared, position-free relational law transfers to unseen sources and geometry states better than the absolute-feature W+UV map.

## Distinct hypotheses

- **A** keeps the representation and adds exact physical-equivalence supervision. It tests whether the existing map can learn source-transferable transport when its specialisation is constrained by physics.
- **B** changes the representation so that sharing is structural. It tests whether position-free relational sharing produces transferable transport without additional constraints.

A difference between them is attributable to the complete designs, not to a single factor.
