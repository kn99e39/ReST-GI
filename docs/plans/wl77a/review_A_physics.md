# WL77A Review A — Physics / Mathematics (2026-11-07)

**Role.** Reviewer A (physics and mathematics) in the WL77A architecture-discovery workflow. The job is to try to falsify the three candidate specifications: H1 CQR (`candidate_H1.md`), H2 RBG (`candidate_H2.md`) and H3 LCR (`candidate_H3.md`).

**Rules followed.**
- Read-only on the repository, except this file.
- No GPU job, training, server access or new Monte Carlo.
- The Gate 4 population (`ports76.json`) and its references were **not** opened. Only its selection rule was read (`experiments/fnto_fia76/ports76.py:1-5`).
- WL77 files and historical worklogs and results are untouched.

**Labels.**
- **[V]** I checked it in code or archived results (file:line given).
- **[C]** computed by me, read-only on CPU, from archived development references.
- **[D]** my own derivation.
- **[S]** taken from a specialist or candidate report and not rechecked.
- **[U]** unverified.

---

## 0. Read-only checks performed for this review

Scripts are in the scratchpad (`…/scratchpad/revA/`); they are not committed.

**C1. Are the BND4_FAR translates physically identical to their supervised partners? [C]**

Method:
- Archived rest references: `~/rest_rank75/refs/bundles/rest_s10{1,5,7}.pt` (WL75 Stage 3) and `~/rest_src73/refs/bundles/rest_s5{0,2,3}.pt` (S16-R).
- Statistic: angle-integrated collided response per receiver port, F(j,c) = Σ_{a,o} S.
- Receivers mapped by the rigid maps:
  - s101→s50: (x, y, z) ↦ (−x, y − 22.5, −z)
  - s105→s52: (x, y + 30, −z)
  - s107→s53: (x − 22.5, y, −z)

| Pair | Receivers ≤ 6 / 15 / 25 mm (mapped / total) | Relative L2 difference R/G/B | Replica noise | Energy ratio |
|---|---|---|---|---|
| s101→s50 | 21 / 143 / 357 of 384 | ≤ 1e-4 | 1e-4 | 1.0000–1.0001 |
| s105→s52 | 24 / 145 / 352 of 380 | ≤ 1e-4 | 1e-4 | 1.0000–1.0002 |
| s107→s53 | 21 / 136 / 344 of 370 | ≤ 1e-4 | 1e-4 | 1.0000–1.0001 |

The references are identical to within replica noise. The ~1 % "corner influence" estimate used by all three candidates is pessimistic by about two orders of magnitude (angle-integrated; the bin-resolved comparison was not done).

**C2. Size of the fixed-seed β/d noise across exact translates. [C]**

Method:
- β comes from `r57_geometry.ballistic_matrix`: q = 256 random rays per cell, one fixed seed for the whole matrix, so every cell gets different samples (`r57_geometry.py:80-93`) [V].
- I compared β rows of along-edge translates at rest; the frames along a face are identical.

| Population | sd(d)/d per bin (median / max) | Bin-centred rms |
|---|---|---|
| Top edge row x = 48.75, n = 24 | 0.44 % / 2.3 % | 0.73 % |
| Side rim row (SR) z = 3.75, n = 24 | 0.49 % / 2.3 % | 0.68 % |
| Interior | 0 | 0 |
| SM mid row | ≤ 0.08 % | — |

**Code facts verified for the arguments below.**
- **WL77 / WL76 definitions.** The M-part and its "noise floor" are M = ref − SS_q, with SS_q from q = 1024 first collisions and one isotropic exit ray each. The noise is the replica half-split of S only, so it excludes the SS_q variance (`gate77.py:45-52`, `posthoc76.py:28-33`, `fia76.py:65-89`) [V].
- **eval68 contrast map.** It is L = H·D⁻¹ with the same MC d for model and reference. The error is pooled ΣE/ΣS per channel, worst-channel jackknife hi (`eval68.py:3-4, 56-80`) [V].
- **Gate 4 rule.** It excludes previously scored ports and **their 16 point-group images** (D4 × z-flip at rest; `sym72.ops`), not lattice translates. "Far" means maximal distance to the S16-R *images* (`ports76.py:1-5`, `sym72.py:9-11`) [V].
- **Context definitions.** TBE = top/bottom with nearest edge e < 10 mm and farther edge ≥ 10 mm, so e ∈ {1.25, 3.75, 6.25, 8.75}. TBC = both edges < 10 mm. SM / SR = side rows with along-edge distance e ≥ 10 mm. SV = side with e < 10 mm (`ports73.py:30-35`) [V].
- **s51 orbit.** s51 = port 3823 at (48.75, 48.75, −5), a **bottom** corner, orbit size 8 (`results/fnto_src73/ports.json`) [V].
- **WL72 RSK.** RSK had no xyz or face id. Its descriptor contains `edge` = distance to the nearest non-coplanar port, clamped at 50 mm (`rsk72.py:27-28`) [V]. RSK still drifted: confirmation con_coarse @2000 → final was 0.589 → 0.629 and 0.328 → 0.560 (`docs/worklogs/72_…:163-168`) [V].

---

## 1. Summary of objections

Each objection is stated once, with every candidate it applies to.

| ID | Candidates | Severity | One line |
|---|---|---|---|
| A1 | H1, H2, H3 | **major** | The headline translate transfer is a verified symmetry identity, not generalisation. It cannot discriminate the candidates, and Gate 4 is built to avoid it in the TBC/SV strata. |
| A2 | H2 (major), H3 (major), H1 (minor) | **major** | Unsaturated long-range plane features re-create location / identity keys. |
| A3 | H1, H2, H3 | **major** | "No identity channel ⇒ no drift" is contradicted by WL72 RSK. H3's c_max = 0.5 is not a small bound. |
| A4 | H1, H2, H3 | **major** | The representation floors are extrapolated from the wrong metric and the most favourable sub-cell. H2 adds an unemulated first-collision projection. |
| A5 | H2 (major), H3 (minor) | **major** | The "deterministic" SS product cubature on line deposits is aliased and biased, not ≤ 0.2× replica noise. Fixing it costs ≫ 1 s. |
| A6 | H2, H3 (H1 by motivation) | **major** | The SS_q-noise reinterpretation of WL76/77 is mechanism-verified but numerically non-evidence. Its minimal competitor (FIA with a deterministic SS) is missing as a control. |
| A7 | H1 (major), H2 (major) | **major** | Out-of-orbit contexts depend on learned source-side functions identified from about 4 configurations. The "1.75e5 Γ entries per source" overstates the information. |
| A8 | H3 | **major** | The learned part is vacuous (horn A) or cannot be source-independent (horn B). The transfer mechanism is a classical CPM solve. |
| A9 | H3 (major), H2 (minor) | **major** | Exactness requires privileged physical knowledge (scalar isotropic kernel, ω, σ_t, homogeneity, index matching) and does not extend to the project's other media. |
| A10 | H1 | minor | Output ∝ MC d_B creates an unlearnable per-bin contrast floor of about 0.04–0.07 at edge/rim sources. |
| A11 | H1 | minor | The receiver-frame D4 snap is discontinuous in geometry. "Continuous frame weights" covers only the source side. |
| A12 | H2 | minor | Ratio-min exact representability needs r*_e < m_e, which lumping does not guarantee. The source-only guard breaks reciprocity where active. |
| A13 | H3 | minor | Positivity holds only to ε for the deployed polynomial. The exact fallback and the Chebyshev degree scale with the learned spectral radius. |
| A14 | H3, H2 | minor | DEF at θ = 0 is not by construction. H3 also misstates s51 as a top corner. |
| A15 | H2, H3 | minor | WL76 Stage 1 (36/36) validates the source-side decomposition in coarse space, not the full Gram identity with exit projection. |

---

## 2. Objections in detail

### A1 — The translate test is a symmetry identity, not evidence of generalisation (H1, H2, H3; major)

**Claims challenged.**
- H1-C1 and H1 Stage C (P1).
- H2-C6 and falsifier F-T.
- H3-C3 and falsifier F2 (on BND4_FAR).
- The common premise that BND4_FAR success demonstrates the transfer mechanism.

**Counterargument.**
- C1 shows the archived references of s101/s105/s107 equal the rigid images of s50/s52/s53 to ≤ 1e-4 relative L2, at replica noise, with energy ratio 1.0000.
- So any operator that is exactly equivariant under these maps, and that fits s50/s52/s53, passes the translate prediction automatically. That includes an orbit-keyed lookup table.
- The test therefore checks equivariance plus training fit. It cannot rank H1, H2 and H3 (all three are designed to be equivariant there), and it says nothing about learned generalisation.
- A75's 0.80 at these ports is confirmed as a pure representation failure. That is already established; it is not evidence for any candidate.

**Consequence for the decisive test (Gate 4).**
- Gate 4 excludes the point-group images of every S16-R source.
- s51's orbit is all 8 corner ports at (1.25, 1.25), so **every Gate-4 TBC slot is outside every candidate's quotient orbit** by construction.
- SV has no S16-R analogue.
- TBE slots may sit at e = 3.75–8.75 mm, which is also outside the orbit.
- SM/SR slots with 10 ≤ e ≲ 15 mm have a vertical edge inside the transport range.
- H1's falsifier F-a pools "contexts covered by the quotient (TBE, **TBC**, SM, SR far)". TBC is never covered, so F-a mixes quotient transfer with interpolation, and F-c cannot attribute the result.

**Requirement.** Selection evidence must be on out-of-orbit contexts: TBC, SV, TBE with e ≠ 1.25, near-vertical-edge SM/SR, and dents. Translates should be a unit test (equality to the partner's prediction), not a transfer result.

**Evidence.**
- C1 [C].
- `ports76.py:1-5`; `sym72.py:9-11` (16 ops, no translations).
- `ports73.py:30-35` (context definitions).
- `ports.json` s51 `orbit_size: 8` [V].

### A2 — Long-range plane features re-create location / identity keys (H2 major, H3 major, H1 minor)

**Claims challenged.**
- H2-C4: "no source-identity channel; translates differ only through a saturating second-plane feature".
- H2-C6.
- H3-C4: "no learned function sees … absolute position".
- H3 §7(a): "exactly translation- and D4h-equivariant".

**H2.**
- Element features include, for every face patch within 2ρ_c (30 mm red), d and exp(−d/L) with L = 18.3 mm.
- For s50 the second plane is at 18.75 mm, giving exp(−18.75/18.3) ≈ 0.36. For s101 it is at 41.25 mm, which is outside 30 mm, so the feature is absent.
- C1 shows the physical response is identical to 1e-4, so the feature is physically irrelevant at this range. Yet it is the one variable that separates s50's neighbourhood from its translates.
- Among the training data, the tuple (edge at 1.25 mm, second plane at 18.75 mm) occurs only in s50's footprint. The same holds for s52 and s53 (vertical edge at 18.75 mm), and for every interior S16-R source within 30 mm of a side, each with its own plane-distance tuple.
- A Γ correction keyed on these features fits a source's residual **at that source's location only**. It does not transfer to the translate. This is the W+UV absolute-coordinate mechanism by another route.
- The 18.3 mm scale is also physically wrong for plane influence: the slab-confined lateral decay is 3.6 / 3.1 / 1.7 mm (specialist_physics §2 [S]).

**H3.**
- Closure element features include "the sorted distances to the three nearest boundary planes" with no stated saturation.
- For any interior element, the three nearest planes are top, bottom and the nearest side, at 20–50 mm. That third distance is a quasi-absolute coordinate.
- So the learned closure is **not** translation-equivariant for interior translations. It is equivariant only for along-edge translates at fixed edge distance, which is why s101/s105/s107 still map correctly.
- The closure can store location-specific corrections at the 12 interior training footprints, bounded only by c_max.

**H1.**
- The source context is hard-windowed at 12 mm, which is correct.
- The far-zone receiver moments (ℓ = 10 mm, window 12 mm) still co-occur uniquely with s50/s52/s53's far receivers near their second plane, where C1 shows no physical difference.
- This is a small partial channel on about ≤ 11 % of native L2 [S].

**Precedent.** WL72 RSK removed xyz and face id but kept `edge` clamped at 50 mm (`rsk72.py:27-28`). It still showed the supervision-local pattern and late drift (A3).

**Implication.** Every plane / boundary feature needs a hard physical cutoff, or saturation on the 1–3.6 mm decay scale (H3: about one stencil, a few mm). Otherwise the "no identity" claims do not hold.

### A3 — Drift and unbounded extrapolation are not removed by deleting identity parameters (H1, H2, H3; major)

**Claims challenged.**
- H1-C4: final / @2k ≤ 1.2.
- H2 §7(4): "the WL71B drift mechanism has no parameter to act on".
- H3-C4: "bounded drift … cannot recur as a ×3–5 transfer loss".

**Counterevidence.**
- WL72 Candidate B (RSK) had no absolute coordinates, no face id, no global U and descriptor-predicted mass.
- Its confirmation con_coarse still worsened during training: 0.589 → 0.629 and 0.328 → 0.560 (×1.7) (`72_…:163-168`) [V].
- So "drift lives only in identity parameters" is refuted by the closest relational precedent. The WL71B attribution (U, corner V, slot-5 mass) is specific to W+UV.
- In H1 and H2, MLP outputs in unsupervised feature regions (two-plane configurations, SV, corner offsets) are constrained only by positivity and passivity. Nothing anchors them during late training.

**H3's bound is weak.**
- c_max = 0.5 scales the self and adjacent stencil couplings. For elements of about 1 mfp (0.5 × 1.25 × 1.25 mm), these carry most of an element's collision probability.
- Raising them by 50 % near a vertical edge or bottom corner can drive the local non-escape fraction from its physical value (about 0.5–0.7 at an edge) to the cap 1 − 10⁻³. That cuts local escape by O(1) and redistributes exit by O(1) [D].
- "Bounded around a learning-free zero point" therefore bounds the *parameter*, not the *error*. A small c_max, or explicit regularisation toward θ = 0 on unsupervised classes, would be needed for the claim to hold.

### A4 — The representation floors use the wrong metric and the most favourable configuration (H1, H2, H3; major)

**Claims challenged.**
- H1-C2: con_coarse floor ≤ 0.10 from rank-4 blocks.
- H2-C2: K_phys meets WL59, con ≤ 0.15.
- H3-C2 and H3-C8: θ = 0 con ≤ 0.15.

**Mismatch 1: the metric.**
- The cited emulation numbers (specialist_physics §7 "pair-contrast error", 0.045 for 0.5 mm voxels, 0.05–0.08 for port columns) are a **two-bin (b0 − b5) difference** at port (20, 20).
- That port is 1.25 mm from both coarse-cell boundaries: a **corner sub-cell**, the most favourable case. Its signal is |ΔZ/2|/|Z̄| = 0.12–0.16, about WL74A's corner value 0.17.
- eval68 con_coarse is the 16-bin centred H·D⁻¹ error, pooled over sources, worst channel, jackknife hi. At centre sub-cells its signal is |HZ|/|Z| ≈ 0.054 [S, WL74A].
- The same absolute discretisation error gives about a 2–3× larger relative contrast error there [D]:
  - 0.045 → about 0.10–0.13 (H3's 0.5 mm voxels, before its 1.25 mm lateral core);
  - 0.05–0.08 → about 0.13–0.24 (port columns, H2's reference point).

**Mismatch 2 (H2 only): the first-collision projection.**
- The emulation kept k = 1 and k = 2 exact and snapped only k ≥ 2. H2 also projects the *first* collision onto trilinear hats, which have 5 mm support at 2.5 mm pitch, before the first S flight.
- The mean free path (1 mm) is smaller than h (2.5 mm). Galerkin error is then not O(h²/ℓ²)-small near the source, where 66–94 % of exit lands within 3.6 mm [S].
- H2's own counterexample (A) is therefore the expected outcome at L1, not a tail risk [D].

**H1.**
- The 1.7 % rank-4 residual [S] is pair-L2 of the whole 16 × 64 block. The contrast part is only 1–3 % of block energy.
- If the residual is not orthogonal to the bin-centred part, a 1.7 % residual is a 0.3–0.6 relative contrast error.
- So rank-4 sufficiency in energy does not bound con_coarse ≤ 0.10.

**Requirement.** All three floor stages (H1 Stage A, H2 0c, H3 Stage 1) must score eval68 con_coarse (worst channel, hi) on every S16-R sub-cell class, including the centre sub-cell sources (e.g. 2085). They must not use pair contrast.

### A5 — "Deterministic" SS and Π cubature is aliased and biased, not ≤ 0.2× replica noise (H2 major, H3 minor)

**Claim challenged.** H2-C3: "Π and SS errors ≤ 0.2 × replica noise in every metric space". Also, by extension, H2-C8 and H3-C7 rebuild cost.

**H2's SS construction.**
- Per 64-bin cell it uses one fixed lattice of 4 × 4 start positions on the port (0.625 mm spacing) × 4 × 4 directions.
- Each ray's deposit is integrated along a line through 0.3125 mm lateral voxels.
- SS = Σ_v P_X(v) P_Y(v) / V_v.

**Why that fails [D].**
- In shallow layers (z ≲ 1 mm, where most single scatter happens), the rays of one 64-bin cell spread laterally by only about z·Δtanθ ≲ 0.25 mm. So only about ¼ to ½ of the voxel columns under the port receive any deposit.
- X and Y cells on the **same port** use the **same start lattice**. Their deposits land in the same columns, and the voxel product over-weights them by up to the inverse coverage, i.e. a bias of up to ~2–4× on shallow same-port contributions.
- Different bins shift the lattice by z·tanθ. The overlap fraction therefore varies with the bin pair, which produces **spurious bin-dependent structure (fake contrast)** in a term that is 16–41 % of collided exit and dominates at the source port.
- This is structured aliasing, so it does not average away. It is the deterministic analogue of the SS_q problem (A6).
- Reaching about 0.1 % native error, which is 0.2× replica noise at 32M photons, needs roughly 10–100× denser stratification: about 10⁹ rays per pose, about 10 s at the ~88 M rays/s quoted for B [S]. That is beyond the 1 s rebuild cap. The alternative is an analytic D1-type integral as in H3.

**H3 (milder).**
- D1 uses Gauss product rules over port area × bin with an inner exit integral clipped by concentric-bin boundaries. The integrand is discontinuous in position and direction, so convergence is algebraic, not spectral.
- D2 is restricted to a fine layer ≤ 2 mm deep. Second collisions deeper than 2 mm are omitted from the ω² term (about a few % of D2) [D].
- The 0.2–2 s footprint rebuild estimate is therefore optimistic.

**Testability.** H2 Stage 0a (2× stratification) and H3 Stage 0(i) would detect this. The point here is that the cost/accuracy trade-off, not the mathematics, decides feasibility.

### A6 — The WL76/77 reinterpretation is unverified, and its minimal competitor is missing (H2, H3; H1 by motivation; major)

**Claims challenged.**
- H2 §2: "FIA's learned exit / quadrature noise" is targeted.
- H3 C4/C6 targets.
- H1 §1: "FIA does not even fit".

**Counterargument.**
- I verified the *mechanism* in code: M = ref − SS_q with q = 1024 one-ray SS, and the reported floor excludes SS_q variance (`gate77.py:45-52`, `posthoc76.py:28-33`, `fia76.py:65-89`). So the WL77 "higher-order azimuthal structure" reading is not established.
- The *magnitude* (exact-SS ring 0.22, FIA-AZ 0.04–0.08) comes from a one-cell, two-bin, non-evidence CPU emulation [S].
- If it holds, the smallest change addressing WL76/77 is **FIA / FIA-AZ with a deterministic or high-q SS (and W)**: same learned exit lobes, same anchors, no new architecture.
- None of the candidates includes that control. H2 and H3 attribute their expected gain over FIA to the exact exit projection (H2-C1, H3-C8), but that is confounded with removing SS noise.
- A read-only rerun of the gate77 statistic with a high-q SS on archived bundles (specialist_physics §6.4) would separate the two before selection.

### A7 — Out-of-orbit contexts depend on source-side functions identified from about 4 configurations (H1 major, H2 major)

**Claims challenged.** H1-C5, H1 §7(d), H2-C7, H2 §7(3).

**H1.**
- The incidence-dependent side-exit fraction (0.25–0.41 at an edge port, 0.37–0.56 at a corner; WL74A §4.5 [S]) is a **multiply scattered** quantity.
- φ_FC supplies only first-collision moments, so the map from φ_FC to α at 1 % precision (§9 item 2) must be learned.
- The training data cover source-side plane distances {1.25 (TBE / TBC / SR), 3.75 / 6.25 (SM), 8.75 (SR bottom), absent}, on one geometry class each.
- No mechanism constrains α at e = 3.75–8.75 mm on the top face, at two-plane corners, or at SV. WL72 / WL73 relational models failed on exactly these contexts (side face, corners: `72_…` observation 4).

**H2.**
- Each S16-R source supervises Γ only through Π_X of its 16 bins. These 16 rows sit on the same ~80–100 footprint nodes and differ by sub-mm shifts and depth changes, so they are nearly collinear.
- The source-side information per source is therefore about 3–5 effective moments contracted against Γ, not "≈ 1.75 × 10⁵ entries" [D].
- Swap symmetry turns receiver-side observations into constraints on pairs with **one** end near a plane. The near-diagonal block with **both** ends at an intermediate plane distance (e.g. a TBE port at e = 3.75 mm, or a corner offset) is constrained by smoothness alone. This is the block that carries the 32–53 % same-port exit.

**H3** is not subject to this objection, because the boundary physics is in the solve.

### A8 — The learning-free solver dilemma (H3; major for selection)

**Claims challenged.** H3-C5 ("learning is useful") and the claim that LCR is a *learned* intrinsic K.

**Horn A [D].**
- By H3's own estimates (θ = 0 floor coarse ≤ 0.03, con ≤ 0.15), the 𝔊 → 𝔊_ref gap is near replica noise.
- F4 (vacuity) is the predicted outcome, and transfer is credited to a classical collision-probability solve with exact first and second collisions.

**Horn B [D].**
- The Galerkin error of an SPD operator in an indicator basis depends on the within-element shape of the source-driven field.
- The exact coarse correction that makes b̄ᵀ L*⁻¹ b̄ exact for in-span sources is L* = V (Φᵀ G Φ)⁻¹ V, which is a **dense** coarse operator.
- A symmetric one-stencil multiplicative closure can represent only its local part. It cannot represent the source-dependent part. So a "useful and source-independent" closure is not available in general.

**Contract.** Verified lines:
- FNTO contract line 17: "do not replace learned complex shading with a limited explicit radiosity solver".
- FNTO contract line 86: "not an exhaustive explicit per-surface full path tracer hidden under a neural name".
- `docs/Architecture.md:94`: a free-space radiosity solver "is not an interchangeable implementation".

**Effect on the research question.** H3 answers "what must be explicit" with "all of the transport". That is informative physics, but it is not evidence about a learned shared operator. This needs a human decision before selection, not after (H3 already says so).

### A9 — Privileged physical knowledge and scope (H3 major, H2 minor)

**Claims challenged.** H3-C1 and H3-C9; H2-C1 ("all angular dependence is exact geometry").

**Counterargument.**
- Both exact factorisations need the scattering kernel to be **known** and scalar: g = 0 isotropic phase, homogeneous medium, index-matched boundary, plus σ_t and ω_c.
- W+UV used only σ_t, through B. FIA used σ_t and ω.
- H3 declares the kernel as an assumption. H2 does not list S / ω / g = 0 among its privileged inputs; it should.
- For the project's other media these exact ends do not exist, and the learned parts (Γ; the stencil closure) carry no angular state. Examples: Yellow Burley with a path-history-dependent reference (WL51), g ≠ 0, Fresnel.
- This does not falsify either candidate **on this fixture**. It does bear on what the fixture result can claim about "learned K".

### A10 — The fixed-seed d_B noise sets a per-bin contrast floor for an output ∝ d (H1; minor)

**Claim challenged.** H1-C1: "identity holds up to β/d noise" (implied negligible); H1-C2 floor.

**Counterargument.**
- H1's collided output is d_B(i, a) × (learned). d_B is per-cell MC with q = 256 [V].
- C2 measured d noise at edge-row and rim-row ports of 0.7 % rms per bin (max 2.3 %); it is 0 in the interior.
- eval68 divides model and reference by the same d_B. A model ∝ d_B therefore carries δd/d per bin, which training cannot learn because there is no identity.
- con floor ≈ 0.007 / (0.1–0.17) ≈ 0.04–0.07 at edge/rim sources [D].
- H2 and H3 outputs are not ∝ d_B, except through an active guard, so they do not have this floor. Small, but it is H1-specific and adds to A4.

### A11 — The receiver-frame snap is discontinuous (H1; minor)

**Claim challenged.** H1-C3: "frame weights continuous on dented geometry".

**Counterargument.**
- π_g is continuous.
- Q⁶⁴_{g,ij} is an argmax over D4 alignment of the projected source t-axis, with a fallback to the b-axis when the projection is degenerate.
- Both switch discontinuously: at ψ = ±45°, and when the projection is nearly degenerate (receivers on faces perpendicular to the canonical t, slightly tilted by a dent). So K jumps with geometry.
- Exact at rest. On dents this is a discontinuity the network cannot remove, because ψ jumps by 90° at the same point.
- A soft (weighted) receiver alignment would be needed for the stated continuity.

### A12 — Ratio-min representability and reciprocity under the guard (H2; minor)

**Claim challenged.** H2-C5: "can represent the physical Γ exactly" and "reciprocal wherever s = 1, ≥ 97 % of cells".

**Counterargument [D].**
- With t_e = m_e·sigmoid < m_e, exact representation needs r*_e < m_e.
- For the lumped Galerkin Γ*, r*_e − m_e = (1/V_e)⟨φ_e, ωSG(Q esc − esc)⟩ − O(1 − ω), where Q is the hat quasi-interpolant.
- esc is convex near a face (½E₂), so Q esc − esc changes sign across the first layer, and the sign of the smoothed term is not controlled.
- For red the physical margin is only about 2 % (collided exit 0.978 of 1 − β). H2's own estimate of the first-layer lumping error is about 1 %.
- So r*_e ≥ m_e can occur at surface nodes, and then Γ* is not representable.
- The guard s_c(X) acts on the source side only, so wherever it is active it breaks the exact symmetry. H3's symmetric s_X s_Y form would keep reciprocity.

### A13 — Positivity and cost depend on the learned spectral radius (H3; minor)

**Claims challenged.** H3-C1 (positivity) and H3-C7 (1-light apply ≤ 5 ms).

**Counterargument [D].**
- The deployed Chebyshev p(L) is accurate to ε = 10⁻⁵ relative.
- Red far entries decay like e^{−r/3.6 mm}, i.e. ≤ 10⁻⁴ of the near entries beyond about 35 mm. Negative entries at the ε level are possible exactly where the true K is tiny.
- The exact-positivity Neumann fallback needs ln ε / ln ρ iterations: about 440 at the physical ρ ≈ 0.974 (red). If the closure pushes columns toward the cap 1 − 10⁻³, that becomes up to about 1.15 × 10⁴.
- The certified Chebyshev degree grows from about 38 (κ ≈ 38) to about 190 (κ ≈ 10³).
- Cost and positivity therefore depend on θ. The cap should be tied to the physical non-escape, not to 1 − 10⁻³.

### A14 — Deformation claims and a factual slip (H3, H2; minor)

**Claim challenged.** H3-C6: DEF passes at θ = 0, including te3.

**Counterargument.**
- DEF scores the *change* ΔK (te − rest).
- `map_dent` thins the boundary layers inside the footprint (to about 42.5 % at the pt0 centre) and shears the lateral cells. So the discretisation bias differs between poses and does not cancel in ΔK.
- With a 1–3 % bias per pose and an unquantified ΔK magnitude, rel ≤ 0.35 is not guaranteed.
- Every historical model fails DEF (0.26–0.83; WL73 l.232-240, WL75 l.214) [V].
- H2 sensibly reports DEF descriptively only (H2-C9).

**Factual slip.** H3 §7(b) lists s51 as "a top corner" (TBC). It is port 3823 at z = −5, a **bottom** corner [V]. At pt0, top corners are therefore unsupervised. This matters for the closure classes H3 claims are covered.

### A15 — What WL76 Stage 1 actually validates (H2, H3; minor)

**Claims challenged.** H2 §8 and H3 §8: "exactness … consistent with WL76 Stage 1 36/36".

**Counterargument.**
- Stage 1 tested the walker's first-collision decomposition, with the remaining transport still by MC. It did so in energy and **coarse** space: coarse relative L1 0.13–0.23 %, z² tests (`76_…:75-86`) [V].
- It did not test the full reciprocal Gram identity with the exit projection P_Y in per-triangle bin frames on dented ports, or native per-entry accuracy. Its one self-consistency failure was the SS quadrature check (resolved post hoc).
- The identity itself is mathematically correct for this fixture. I re-derived c₁ = (4πσ_t/g_X) P_X and G = I + ωS + ω² SGS, using the fact that S and G commute [D].
- Its *implementation* (bin frames, dent ports, in-medium visibility) is validated only by WL53-type reciprocity checks: two pairs at 2.5 mm [S].
- A direct check of D1 / SS against the walker's single-scatter tally at native resolution belongs in Stage 0. H3 0(ii) has it; H2 0a has it on 4 cells.

---

## 3. What survives (physics view)

All three operator classes are, as specified:
- linear;
- zero → zero;
- non-negative (H3 up to ε for the deployed polynomial);
- passive by construction (H1 via d·m ≤ d; H2 via the source guard; H3 via the symmetric certificate);
- K/C-consistent (first exit, in-medium visibility).

I found no counterexample to these invariants.

The Green-form identities that H2 and H3 discretise are exact for this fixture. Reciprocity without label reversal follows: H2 off the guard, H3 everywhere apart from the shared MC B.

**What the evidence does not support:**
- that the translate result tests generalisation (A1);
- the "no identity channel" property of H2 and H3 as feature sets are currently written (A2);
- the drift-removal claims (A3);
- the quoted representation floors in the eval68 contrast metric (A4);
- H2's deterministic-cubature accuracy at the stated cost (A5).

**Per candidate, the selection-relevant risk is:**
- **H1:** out-of-orbit learnability at 1 % precision (A7), plus an inherited d_B floor (A10).
- **H2:** an L1 basis floor that is likely above the contrast gate (A4), SS cubature aliasing or cost (A5), and long-range plane features acting as location keys (A2).
- **H3:** physically the most robust transfer mechanism, but its learned component is probably vacuous or cannot be source-independent (A8), and its admissibility as a learned K is a human decision (A8, A9).

## 4. Not verified

- The bin-resolved (16 × 64) identity of the translates. C1 is angle-integrated only.
- The emulation numbers of specialist_physics §6–§7, which are non-evidence and were not rerun.
- All cost figures.
- Whether Gate-4 TBE / SM / SR far slots happen to be along-edge translates of S16-R. I did not open `ports76.json`; only the rule was read.
- Rank-4 residual statistics (Agent 3).
- The sign of r*_e − m_e (A12), which is not computed.
