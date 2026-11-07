# WL77A Adjudication Draft: Rulings on Objections, Decision Matrix and Selection (2026-11-07)

**Role.** Adjudication drafter for the WL77A multi-agent architecture-discovery workflow. The coordinator checks this reasoning against the evidence. This draft does not decide by counting votes. Most rulings below agree with the authors' concessions, but each one was checked against code, results or arithmetic where that was material.

**Status.** This is a draft decision record only. It does not adopt, authorise, freeze or start anything.

**Rules followed.**
- Read-only on the repository, except this file.
- No GPU job, training, server access or Monte Carlo.
- WL77 files (`experiments/fnto_fia77`) and all historical worklogs and results are untouched.
- The Gate 4 population (`ports76.json`) and its references were **not** opened. Only the selection code (`experiments/fnto_fia76/ports76.py`) was read.

**Inputs.**
- `docs/plans/wl77a/`: specialist_physics, specialist_neural, specialist_systems, specialist_priorart, ideation, candidate_H1/H2/H3, review_A_physics, review_B_novelty_systems, responses.
- `docs/worklogs/77A_WORKLOG_MULTI_AGENT_ARCHITECTURE_DISCOVERY.md` (Part 0).

**Labels.**
- **[V]** I checked this in code, results or a document, at the path:line given.
- **[A]** my own arithmetic.
- **[S]** taken from a workflow file and not rechecked by me.
- **[U]** unverified.

---

## 0. Independent checks done for this adjudication

| # | Claim the rulings depend on | Where checked | Result |
|---|---|---|---|
| V1 | The WL77 M-part is ref − SS_q (q = 1024 quadrature). Its "noise" is the S replica half-split only. | `experiments/fnto_fia77/gate77.py:45-48` (`half=(S[:K//2].mean(0)-S[K//2:].mean(0))/2`; `ss` from `FI.quad_ports(...,quadrature_q)`; `M=ref-ss`) | [V] SS_q variance is not in the reported noise floor. The magnitude of its effect is still non-evidence (specialist_physics §6, one interior cell, two bins). |
| V2 | A75/W+UV carries identity and world-axis channels. | `experiments/fnto_rule57/r57_geometry.py:58` (`xyz0/.05, xyz/.05, d/.006, …, port_face/5`), `:60-62` (global 11×11 dent grid), `:65-66` (world-axis `dr`); `r57_model.py:61-63` (`U=softmax(…,dim=0)` over all P, `mass=sigmoid`) | [V] |
| V3 | β/d come from one sequentially consumed generator, so symmetry images get different draws. | `r57_geometry.py:80-93` (`seed=57501`, one `torch.Generator`, chunked loop) | [V] |
| V4 | The Gate 4 far stratum maximises distance to point-group images, not to translates. Context definitions. | `experiments/fnto_fia76/ports76.py:1-5, 50-56`; `experiments/fnto_src73/ports73.py:30-40` (TBE: e < 10 and E ≥ 10; TBC: E < 10; SM/SR: side, e ≥ 10, fixed row; SV: side, e < 10; `images` = D4 × z-flip permutations) | [V] |
| V5 | RSK (WL72 B) is not identity-free. | `experiments/fnto_cmp72/rsk72.py:28` (`edge` clamped at 50 mm), `:29` (descriptor includes `nd[:,6:9]` = rest displacement), `:34` (20 mm whole-slab `ctx` kernel), `:55-56` (`phi_far` over all 240 coarse cells), `:62` (`scale = mass/(Σw_loc + Σw_far)`, a global normaliser), `:47` (frozen global A/O logits) | [V] The authors' rebuttal of the RSK precedent holds. |
| V6 | WL72 drift numbers | `docs/worklogs/72_WORKLOG_CONTROLLED_SPATIAL_REPRESENTATION_COMPARISON.md:162-168`: A 0.371→0.871, 0.422→0.643; B (RSK) 0.589→0.629, 0.328→0.560 | [V] |
| V7 | WL75 group and per-source values | `docs/worklogs/75_…:195-200` (BND4_FAR con_coarse AR 0.793/0.792, A75 0.803/0.824, G0 0.265, G2 0.159; INT4 A75 0.151/0.185); `:239-251` (s107 A75 1.19, G2 0.20; sub-cell G2 0.14 / 0.09 / 0.09) | [V] |
| V8 | What WL76 Stage 1 validated | `docs/worklogs/76_…:79-90`: decomposed vs references 36/36 (coarse rel. L1 0.13–0.23 %, z² 0.90–1.13); the SS quadrature vs MC tally failed once (rest_s70, z² 1.703) | [V] It validates the coarse-space source-side decomposition, not the native Gram identity with exit projection. |
| V9 | FNTO contract wording | `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md:17` ("…do not replace learned complex shading with a limited explicit radiosity solver"), `:86` ("…not an exhaustive explicit per-surface full path tracer hidden under a neural name"); `docs/Architecture.md:94` | [V] |
| V10 | Arithmetic in several rulings | L_d = 1/√(3σ_aσ_t): red 18.26 mm, green 5.77 mm. H3 ρ ≈ 0.9738, so Neumann to 1e-5 needs ≈ 434 iterations. B5 memory roofline: 38 × 0.12–0.15 GB / 912 GB/s = 5.0–6.25 ms. B2 dense near state 8.8e4 × 3072 × 4 B = 1.08 GB. | [A] Matches the reviews and responses. |

**Not checked by me:**
- Reviewer A's C1 translate-identity computation (≤ 1e-4 relative L2, angle-integrated, archived bundles on the server).
- Reviewer A's C2 d-noise measurement.
- Reviewer B's web prior-art checks.
- Every cost estimate.

These are taken as [S]. None of the rulings depends on them alone, except A-A1, whose geometric premise is in any case verified (V4 plus the translate maps).

---

## 1. Rulings on the objections

"Decisive reason" names the fact the ruling rests on. Where part of an objection fails, the ruling covers the part that bears on the challenged claim, and the failed part is stated.

### 1.1 Review A (physics and mathematics)

| ID | Ruling | Decisive reason |
|---|---|---|
| **A-A1** (H1, H2, H3; major) | **UPHELD** | The three BND4_FAR translates are exact rigid images of s50/s52/s53. Their references agree to replica noise [S, C1], and the maps are verified geometric translates or reflections [V4, V7]. So any exactly equivariant operator that fits the partners passes "translate transfer". The test cannot rank H1/H2/H3, nor separate them from a de-identified W+UV. Gate 4 excludes all point-group images [V4]. s51's orbit therefore holds every (1.25, 1.25) corner, and every Gate-4 TBC slot is out of orbit, so H1's F-a pooling of TBC as "covered" is wrong. The authors' defence (that H2 F-T and H3 F2 are necessary conditions) is correct but non-discriminating. |
| **A-A2** (H2, H3 major; H1 minor) | **UPHELD** | H2 §6 gives every plane within 2ρ_c = 30 mm the features d and exp(−d/18.3 mm). 18.3 mm is the infinite-medium red diffusion length [A], not the 3.6 mm slab-confined decay. H3 §6 uses "sorted distances to the three nearest boundary planes" with no saturation, so for interior elements the third distance (20–50 mm) is a quasi-absolute coordinate. Both are location keys as written, so H2-C4 and H3 §7(a) are false as written. For H1 the source side is hard-windowed at 12 mm, and the residual far-receiver key is small. The fixes (cutoffs) are non-central and not specified. |
| **A-A3** (H1, H2, H3; major) | **UPHELD** (claims); the precedent is **refuted** | The RSK precedent fails: RSK had a whole-slab far field, a global normaliser, a 50 mm edge key and a displacement channel [V5]. So WL72 does not show that a strictly local, identity-free model drifts. The challenged claims still fail as stated:<br>• H1-C4, H2 §7(4) and H3-C4 are untested predictions. Shared MLP weights still move in unsupervised feature regions.<br>• H2 and H3 keep a key route (A-A2).<br>• H3's c_max = 0.5 bounds the parameter, not the error. The conceded estimate is an O(40 %) local escape change at a weakly supervised edge element. |
| **A-A4** (H1, H2, H3; major) | **UPHELD** | Every quoted floor (H1 ≤ 0.10, H2 ≤ 0.15, H3 ≤ 0.15 con) is extrapolated from specialist_physics §7's two-bin pair contrast at a corner sub-cell (signal 0.12–0.16). That is not eval68's 16-bin centred H·D⁻¹ pooled worst-channel hi (`eval68.py:1-5` [V]), whose centre sub-cell signal is about 0.054. The partial rebuttal (shift-type errors do not rescale) is sound, but it rescues no number: the authors concede that amplitude-type bias dominates for H2/H3, and the H1 bound is indeterminate either way. H2 adds a first-collision projection on 2.5 mm hats (h > mfp). The physics specialist states that snapping the first collision "would remove the 0.3–0.8 mm footprint shift" (§7 item 3), so H2's own counterexample (A) is the expected L1 outcome. **All floors are unmeasured.** |
| **A-A5** (H2 major; H3 minor) | **UPHELD** | H2 uses 16 rays per 64-bin cell at 0.625 mm spacing against 0.3125 mm voxel columns, so shallow deposits cover about ¼ of the columns [A]. Same-port X/Y share columns, so the product over-weights them, and the bin-dependent z·tanθ offset creates bin-pair-dependent overlap, i.e. spurious contrast. H2-C3 ("≤ 0.2× replica noise in every space") is unsupported at the stated cost. The correct alternative (an entry-chord integral) is uncosted. For H3, D2 stops at 2 mm, dropping ≈ 0.2–0.6 % of collided energy (minor, uncaught by Stage 0(i)). |
| **A-A6** (H2, H3; H1 by motivation; major) | **UPHELD** | The mechanism is verified [V1]: WL76/77's M-part is contaminated by q = 1024 one-ray SS noise that the reported noise excludes. Two non-evidence emulations reproduce WL77's 0.453 / 0.385 / 0.352 from that noise alone. The minimal competitor (FIA / FIA-AZ with deterministic or high-q SS) is untested, so H2-C1 and H3-C8 ("gain from exact exit projection") are confounded with SS-noise removal. H1 §1's "FIA does not even fit" is a correct report of WL76 but may be an estimator artefact. The candidates' absolute gates are unaffected; their **motivation and attribution** are not established. |
| **A-A7** (H1 major, H2 major) | **UPHELD** | **H1.** φ_FC carries first-collision moments only. The incidence-dependent side-exit fraction (0.25–0.56) is multiply scattered and must be learned to ≈ 1 % from source-side plane distances {1.25, 3.75/6.25, 8.75, absent} [V via ports.json; S]. H1 itself calls C5 "the claim most likely to fail" (§13). **H2.** The 16 Π_X rows of a source are nearly collinear (≈ 3–5 effective vectors), so "1.75e5 constraints per source" overstates the information by ≈ 10×. The near-diagonal Γ block with both ends at intermediate plane distances carries the 32–53 % same-port exit and is identified by smoothness only. H3 is correctly exempt. |
| **A-A8** (H3; major) | **UPHELD** (Horn A and the contract); Horn B as an impossibility is **refuted** | **Horn B refuted.** The authors show that the exact coarse correction L* = V(ΦᵀGΦ)⁻¹V is source-independent for piecewise-constant second-collision densities, and plausibly quasi-local (exponential decay of the inverse of a well-conditioned SPD matrix). So "useful" and "source-independent" do not contradict each other. Its tail share is unmeasured.<br>**Horn A stands.** H3's own transfer argument places all boundary, corner and dent physics in the exact CPM solve. H3's own θ = 0 estimate would make the closure's contribution small, so F4 (vacuity) is a likely outcome, and the transfer would then belong to a classical solver.<br>**Contract.** Admissibility as a learned K is a human decision (H3-C9, FNTO :17/:86 [V9]) that has not been taken. |
| **A-A9** (H3 major, H2 minor) | **UPHELD** | Both exact factorisations need a known scalar kernel S: g = 0, homogeneous, index-matched, σ_t, ω_c. H2 did not declare this as privileged physics. Neither learned part carries angular state, so neither extends to Burley with a path-history-dependent reference (WL51), g ≠ 0 or Fresnel. A fixture pass would support "learned scalar correction on a known kernel", not "learned K". |
| **A-A10** (H1; minor) | **UPHELD** | H1's collided output is ∝ d_B (fixed-seed MC per cell [V3]), and eval68 divides model and reference by the same d_B. So H1 carries δd/d to first order, while H2/H3 carry it only to second order (off their guards). The d-noise magnitude (≈ 0.7 % rms at edge/rim rows) is [S]. The ≈ 0.04–0.07 con floor at edge/rim sources, including training sources, is a derivation. A deterministic d̂ fix is unspecified. |
| **A-A11** (H1; minor) | **UPHELD** (scoped) | Q⁶⁴ is an argmax with a fallback, so it is discontinuous. Its reach is narrow: it cannot occur at pt0 (the dent is ≥ 22.5 mm from the sides), but it can for side↔rim pairs at te2-type dents ((−17.5, 12.5) mm, R 22.5 mm reaches x = −40 mm). H1-C3's "continuous on dented geometry" covers only the source side. |
| **A-A12** (H2; minor) | **UPHELD** | t_e = m_e·sigmoid < m_e, so exact representation needs r*_e < m_e. For red the absorption margin (≈ 2 %) is comparable to H2's own lumping error (≈ 1 %), and the sign of the error is uncontrolled. H2-C5's "exactly" becomes "approximately". The source-only guard breaks reciprocity where it is active. A symmetric s_X s_Y guard would keep reciprocity, but it is not specified. |
| **A-A13** (H3; minor) | **UPHELD** | Positivity holds to ε only for the deployed Chebyshev polynomial. The exact-positivity Neumann fallback needs ≈ 434 iterations at ρ ≈ 0.974 [A], not 260. Degree and fallback grow with the learned spectral radius (κ → 10³ at the 1 − 10⁻³ cap). Cost and positivity therefore depend on θ. |
| **A-A14** (H3, H2; minor) | **UPHELD** (DEF claim and s51 slip); the drawn consequence is **refuted** | DEF scores ΔK. map_dent thins and shears elements, so discretisation bias is pose-dependent and H3-C6 is an assertion. s51 is a bottom corner (port 3823, z = −5) [V]. The reviewer's consequence ("top corners unsupervised at pt0") fails, because H3's closure features are D4h-invariant scalars and the pt0 dent is ≥ 36 mm from every corner. |
| **A-A15** (H2, H3; minor) | **UPHELD** | WL76 Stage 1 validated the first-collision decomposition in energy and coarse space only [V8]. The native reciprocal Gram identity with exit projection in per-triangle frames on dented ports is mathematically exact but not implementation-validated. |

### 1.2 Review B (novelty and systems)

| ID | Ruling | Decisive reason |
|---|---|---|
| **B-B1** (H1, H2, H3; major) | **UPHELD** | A75's translate failure (0.40–1.30 per source vs TRAIN_NEW 0.131 [V7/S]) is fully explained by its verified identity and world-axis channels and its lack of a point-group quotient [V2]. A de-identified, quotiented W+UV (W+UV-Q) would therefore also pass the translates. None of the candidates specifies this control, so no candidate-specific mechanism is isolated. The ideation §5.2 statement ("the translation part of the quotient is exactly what Gate 4 tests at its far strata") is also false for TBC and SV (V4). |
| **B-B2** (H1; major) | **UPHELD** (state, 1-light); the 64-light part is **unresolved** | Frame averaging makes each pair block a sum of up to \|supp π\| rank-4 blocks with separate normalisers. H1 §10 counted one frame for state but ≈ 5 for rebuild. The corrected state is ≈ 1.5 GB (dense near, [A] 1.08 GB + far) or ≈ 2.1 GB (factors, at the 2 GiB cap). The 1-light apply reads ≥ 1.5 GB (≥ 1.6 ms at roofline) and is likely > 5 ms in the prescribed torch style. The batched-GEMM 64-light rebuttal is plausible but unmeasured. |
| **B-B3** (H1; major) | **UPHELD** | Weighted continuous frames (Dym, Lawrence, Siegel, ICML 2024) and frame averaging (Puny et al., ICLR 2022) are prior art [S, URLs in review B]. With \|D4\| = 8, the Reynolds average is exact and tractable, so the weights add only canonicalisation and ≈ 1.7× cost. H1-noQ confounds equivariance, canonicalisation and 8-fold ensembling, and attribution claim (3) never ablates identity removal or CP pairing. |
| **B-B4** (H2, H3; major) | **UPHELD** | The transfer mechanisms of H2 and H3 are in explicit physics. "Beats A75" is not evidence that the *learned* part transfers, because A75 gets none of this physics. H2 has no deployable learning-free Γ comparator (Γ_phys is a per-geometry MC diagnostic). H3's F4 has no noise-floor "inconclusive" outcome. Only "learned beats learning-free on unseen groups by a paired interval" would assign credit, and neither specification predeclares that. |
| **B-B5** (H3; major) | **UPHELD** (claim and F5 as written); the custom-kernel rebuttal is **unresolved** | With explicit coefficients, 38 iterations × 0.12–0.15 GB give 5.0–6.25 ms at peak [A], so "compute-bound, 2–5 ms" is wrong and torch-style F5 fails. The on-the-fly translation-keyed stencil rebuttal (≈ 0.3 ms at rest, ≈ 1.2 ms at pt0 roofline) follows H3 §10's structured-stencil intent and the systems budget's roofline clause (`specialist_systems.md` §8). But it is unbuilt and unmeasured, and it departs from the prescribed measurement style. The coupled-solve multiplier (×6–20) is unaddressed. |
| **B-B6** (H3, H2; major) | **UPHELD** | **H3.** ≈ 20 points per weakly singular 6-D self/adjacent integral is inconsistent with its own ≤ 1e-3 doubling rule. Visibility per cubature point near the dent surface gives a plausible 1–10 s, against a 1 s cap (conceded).<br>**H2.** The dent SS of 1–3 s exceeds the cap (declared), and the full-pose Π (6.3e7 casts) exceeds 32 M. Its partial rebuttals hold: the dent-footprint Π is ≈ 7e6 casts, and the wide normaliser radius touches only cheap learned coefficients.<br>**Both.** No locality certificates are specified. |
| **B-B7** (H2; major, framing) | **UPHELD** | Neural Green's Functions (NeurIPS 2025) already learns a geometry-only, source-agnostic operator decomposition that generalises to unseen sources and geometries. Element Learning, DeepRTE, Wang–Tran–Luebke 2005 and Garg 2006 cover the remainder. H2's mechanism-level distinction is withdrawn. What remains is an RTE boundary-to-boundary instance, the ratio-min device and the evaluation. |
| **B-B8** (H3; minor) | **UPHELD** | Learned RTE closures (Huang et al., JCP 2022; NeurIPS 2023 workshop) cover item 2 of H3 §12, except the CPM-generator target and the M-matrix-preserving parameterisation. |
| **B-B9** (H1, H2, H3; minor) | **UPHELD** | A 16 × 16 Sobol pattern is not D4-symmetric (H1). d_B differs between images [V3], so every β-dependent guard (H2 s_c, H3 s_Z) and H1's d·m break "bit-identical image" claims where active. |
| **B-B10** (H1, H2, H3; minor) | **UPHELD** | Port-level equivariance equalises predictions, not con_coarse. G2 by sub-cell is 0.14 / 0.09 / 0.09 (corner / edge / centre) [V7], and Gate 4 hi/lo pairs are sub-cell-mismatched by design [V4]. Ratio falsifiers (H1 F-b, H3 F2 "≤ 1.5 × TRAIN_NEW", H2 F-T) mix transfer with sub-cell composition. |
| **B-B11** (H1, H2, H3; minor) | **UPHELD** | A two-seed spread is one difference. WL37 → WL38 is the project's own counterexample. |
| **B-B12** (H1, H2, H3; minor) | **UPHELD** | Scope bounds:<br>• H1 near-core state ∝ h⁻⁴ (× the B2 frame factor).<br>• H2 Γ ∝ ρ³, with ρ set by the red diffusion length in thick media (≈ 47×).<br>• H3 state ∝ volume and iterations ∝ √κ (≈ 5× for thick red).<br>• H2 and H3 need a known scalar kernel (A-A9). |

**Net result of the rulings.**
- No objection shows a violation of the hard invariants. All three operator classes remain linear, zero → zero, non-negative (H3 to ε), passive by construction and K/C-consistent (review A §3, unrefuted).
- What is refuted is narrow: the RSK precedent (A-A3), Horn B as an impossibility (A-A8) and the "top corners unsupervised" consequence (A-A14).
- Every major objection that targets a candidate's **transfer argument**, **floor**, **attribution** or **cost** is upheld.

---

## 2. Decision matrix

Ratings: strong / moderate / weak / fatal. "Fatal" means the upheld evidence prevents the candidate from answering the WL77A question in its stated form, whatever the experimental outcome. It is not a prediction that an experiment would fail.

**No criterion stands in for another.** In particular, H3's strong physical transfer is rated where it belongs (mechanistic relevance, physics). Whether that transfer is *learned* is rated separately (learnability, novelty, interpretability).

| Criterion | H1 CQR | H2 RBG | H3 LCR |
|---|---|---|---|
| **Mechanistic relevance to source transfer** | **weak.** Orbit sources transfer by symmetry identity, which W+UV-Q shares (A-A1, B-B1). For the out-of-orbit contexts Gate 4 discriminates on (TBC offsets, TBE e ≠ 1.25, SV, near-vertical-edge SM/SR), transfer is MLP interpolation from ≈ 4 boundary configurations, which H1 itself predicts fails (§13; A-A7). | **moderate.** Incidence and exit angular structure are computed (exact Π), which is a real transfer mechanism for the angular part. The spatial Γ near boundaries is learned, and the near-diagonal block at unseen plane configurations is identified only by smoothness (A-A7). | **strong (solver), not learning.** All boundary, corner, second-edge and dent context sits in an exact CPM solve, so A-A7 does not apply. The transfer is carried by the classical solver, not the learned closure (A-A8 Horn A, B-B4). |
| **Physical consistency** | **moderate.** Passive, linear and positive by construction. No reciprocity. Output ∝ MC d_B adds an unlearnable δd/d floor (A-A10). The receiver-frame snap is discontinuous on te2-type dents (A-A11). | **moderate.** The exact Green identity is the basis. Ratio-min is only approximately representable at red surface nodes (A-A12). The source-only guard breaks reciprocity where active. The aliased SS term would inject spurious bin structure (A-A5). | **strong.** Exact two-collision Gram form, symmetric M-matrix, reciprocity everywhere, passive by certificate. Positivity holds to ε only (A-A13), and D2 is truncated at 2 mm (≈ 0.2–0.6 %, A-A5). |
| **Novelty plausibility** | **weak.** Frame averaging and weighted frames are prior art (Puny 2022, Dym 2024; B-B3). The rest is Vicini/Deep Scattering conditioning plus H-matrix blocks. Only the evaluation and an attribution claim remain. | **weak.** Covered at mechanism level by Neural Green's Functions, Element Learning, DeepRTE, the first-collision source method and Wang 2005 (B-B7). The ratio-min normaliser is a small device. | **weak.** CPM, the first-collision source method, SPH-type equivalence corrections and learned RTE closures are all prior art (H3 §12; B-B8). It is a combination plus an evaluation. |
| **Information preservation** | **moderate.** No absolute or identity inputs, and a 12 mm window. φ_FC keeps first-collision moments but not the multiply-scattered side-exit (A-A7). The out-of-window loss is physically ≤ 1e-4 for the translates [S, C1]. | **weak.** The first collision is projected onto 2.5 mm hats (h > mfp = 1 mm) before the first flight. This is the step that removes the incidence footprint shape (specialist_physics §7 item 3; A-A4; H2 counterexample A). The SS cubature is aliased (A-A5). | **strong.** D1 and D2 are exact at sub-port resolution, b̄ keeps the element-integrated second collision exactly, and only k ≥ 3 is gridded (1.25 × 0.5 mm). The losses are small and stated (within-element shape, D2 depth truncation). |
| **Representation capacity** | **moderate.** Pair-specific rank-4 near blocks (1.7 % energy residual [S]) remove the global CP floor. The con_coarse floor is unmeasured and could be 0.10–0.6 (A-A4), plus ≈ 0.04–0.07 from d_B (A-A10). | **weak.** By its own counterexample (A) and A-A4, the L1 floor is expected above the contrast gate. The L2 refinement is untested, and a shape-preserving fix is outside the specification. | **moderate.** The θ = 0 floor on 𝔊 is unmeasured in eval68 terms (A-A4). Within-element source-dependent shape cannot be represented by a source-free closure (H3 §9.2). |
| **Learnability** | **weak.** Source-side multiply-scattered functions must be learned to ≈ 1 % from 4 boundary configurations. Out-of-orbit regions are unanchored against late drift (A-A3, A-A7). WL72/73 relational models failed in exactly these contexts. | **weak.** About 10⁴ contracted constraints per source, not 1.75e5. The near-diagonal block at intermediate plane distances is pinned only by smoothness. Location keys exist as written (A-A2, A-A7). | **moderate.** The closure is densely supervised and local, so it is learnable. But its useful signal is likely below the seed spread (Horn A). c_max ±50 % allows O(40 %) local escape excursions in weakly supervised classes (A-A3). |
| **Computational feasibility** | **weak.** State ≈ 1.5–2.1 GB (B-B2; the factor layout sits at the 2 GiB cap). 1-light apply likely > 5 ms torch-style. The 64-light apply is plausible but unmeasured. Rigid rebuild is 0. | **weak.** Dent SS 1–3 s and full-pose Π 63 M casts both exceed hard caps (B-B6). The cost of a non-aliased SS replacement is not estimated (A-A5). Apply is plausible. | **weak.** The nonrigid geometry rebuild is plausibly 1–10 s against a 1 s cap (B-B6). The 1-light apply is memory-bound at 5–6 ms with explicit coefficients, and meets the cap only with an unbuilt custom stencil kernel (B-B5). The coupled-solve multiplier is unaddressed. |
| **Experiment interpretability** | **moderate.** The specification is interpretable only with the H1-Reynolds, H1-noQ and W+UV-Q arms at ≥ 3 seeds and sub-cell-matched ratios (B-B3, B-B10, B-B11), none of which is in it. Translate predictions are unit tests only. | **weak.** There is no deployable learning-free Γ comparator (B-B4). The SS aliasing would confound Stage 0, and the FIA-detSS competitor is absent (A-A6). A pass cannot be credited to the learned Γ. | **moderate.** The θ = 0 learning-free control is built in, and Stage 1 is informative physics. But F4 (vacuity) has no noise-floor outcome and is the likely result, so the experiment would most likely credit the classical solver (A-A8, B-B4). |
| **Extension potential beyond the fixture** | **weak.** No physics assumptions beyond geometry, so in principle it is material-agnostic. State ∝ h⁻⁴, and the frame quotient depends on lattice congruence (B-B12). | **weak.** Exact ends require g = 0, homogeneous, index-matched media with a known kernel. Γ state ∝ ρ³ in thick media (A-A9, B-B12). | **fatal.** The learned part "carries almost no transport information" (H3 §9 item 7). For any medium whose scalar kernel is unknown or direction-dependent (Burley/WL51, g ≠ 0, Fresnel), nothing learned can substitute (A-A9). |

---

## 3. Synthesis

### 3.1 What the evidence does and does not support

**Supported** (verified or unrefuted):
- The historical BND4_FAR translate failure is a representation-symmetry failure of W+UV: the references are identical [S], and A75 carries identity and world-axis channels [V2].
- The exact first-collision / Green structure (P1, P2) holds for this fixture. The operator invariants of all three candidates hold.
- The WL76/77 reported floors are contaminated by SS_q noise that their noise estimate excludes [V1]. How large that contamination is remains non-evidence.

**Not supported:**
- that any learned component in H1, H2 or H3 has a credible mechanism to transfer to the **out-of-orbit** contexts the confirmation population is built on;
- any of the quoted representation floors in the eval68 metric;
- the identity-free claims of H2 and H3 as specified;
- the attribution of an expected gain to exact exit projections, as opposed to SS-noise removal.

### 3.2 Each candidate against the selection rule

The workflow rule: no recommendation for (a) novelty without a credible transfer mechanism, (b) capacity without a transfer mechanism, or (c) reproduction of published work without substantive differentiation.

**H1 falls under (b).**
- Its only credible transfer is the symmetry quotient. That is unit-test transfer for orbit sources, and a de-identified W+UV-Q would deliver it equally.
- For the non-orbit contexts, the learned mechanism is the same class (a shared relative-geometry MLP) that failed in WL72/73. The one additional input, φ_FC, does not carry the multiply-scattered side-exit (A-A7).
- Its mechanism-level novelty is withdrawn (B-B3), its corrected state is 1.5–2.1 GB, and its floor is unmeasured.

**H2 falls under (b) and partly (c).**
- The angular half of its mechanism is credible.
- The central representation projects the first collision onto a basis coarser than the mean free path. By the physics specialist's own measurement logic, that removes the incidence shape the contrast metric depends on, and H2 names this as able to "kill H2 even with perfect learning".
- Its learned Γ has the same smoothness-only identification problem as H1 at the decisive near-diagonal block (A-A7). Its "deterministic exact" SS term is aliased as specified (A-A5).
- Its mechanism is covered by Neural Green's Functions (B-B7).
- The repairs (a second-collision projection, an entry-chord SS, a symmetric guard, plane cutoffs) together move H2 toward H3's structure and are not specified.

**H3 falls under (c), with a pending contract question.**
- It is the only candidate whose source-transfer argument is physically credible for out-of-orbit contexts.
- That credibility comes entirely from a classical collision-probability solve with a first-collision source, which is published method, conceded in H3 §12.
- Its learned component is predicted by its own estimates to be vacuous or below the seed spread (A-A8 Horn A). It has no extension beyond a known scalar kernel (A-A9).
- Its nonrigid rebuild plausibly exceeds the hard cap (B-B6).
- Whether "CPM solver + closure" is an admissible learned FNTO K is a human decision that has not been made (FNTO contract :17, :86). The adjudication cannot presume it.
- Selecting H3 would answer "what must be explicit" with "all of the transport" under a learned-K label. That is the situation the contract's line 86 describes.

**Conclusion: no candidate survives.**

### 3.3 Confound that blocks attribution for any selection (A-A6)

The motivation for exact angular ends (H2, H3), and part of H1's framing, rests on reading WL76/77 as a representational failure of learned exit lobes. The code shows the WL77 floors contain un-noised SS_q variance [V1]. Two independent non-evidence emulations attribute most of 0.453 / 0.385 / 0.352 to it.

Until the minimal competitor (FIA / FIA-AZ with deterministic or high-q SS) is measured, no candidate's expected advantage over FIA can be attributed to its distinctive mechanism. This is not a fatal issue for any one candidate. It is a reason why even a passing candidate would not answer the research question cleanly.

---

## 4. Recommendation

**Selected: NONE.** Stop at selection. No candidate is recommended for implementation.

The decisive reasons, one per candidate:
- **H1:** no credible mechanism beyond the symmetry quotient for the decisive out-of-orbit contexts.
- **H2:** its central first-collision projection likely destroys the incidence information it relies on, and its learned Γ is under-identified at the decisive block.
- **H3:** transfer is credited to a published classical solver rather than to learning; it is admissible as a learned K only by an untaken human decision; it has no extension beyond a known kernel.

---

## 5. Fatal issues

1. **All three (selection-level).** The only mechanism-derived transfer prediction (the BND4_FAR translates) is a symmetry identity: references equal to ≤ 1e-4 [S, C1], geometry verified [V4]. Any de-identified equivariant model passes it, and no candidate specifies the W+UV-Q control (A-A1, B-B1). The confirmation population is built from out-of-orbit contexts [V4], and none of the learned components has a credible mechanism for them (A-A7 for H1/H2; A-A8 for H3).
2. **H1.** Non-orbit transfer depends on MLP interpolation of multiply-scattered source-side functions from about 4 boundary configurations, at about 1 % precision. H1 itself predicts this fails (§13, C5). The quotient mechanism is prior art (Puny 2022, Dym 2024) and is shared with the W+UV-Q control (B-B3).
3. **H2.** The central representation projects the first collision onto 2.5 mm hats (h > mfp) before the first flight, which removes the 0.3–0.8 mm incidence footprint shape (specialist_physics §7 item 3; A-A4; H2 counterexample A). The near-diagonal Γ block at unseen plane configurations is identified by smoothness only (A-A7). The mechanism is covered by Neural Green's Functions / Element Learning (B-B7).
4. **H3.** Source transfer is carried by a classical collision-probability / first-collision-source solve (published), not by the learned closure, which is likely vacuous (A-A8 Horn A, B-B4). Admissibility as a learned FNTO K is an untaken human decision (FNTO contract :17, :86). The mechanism requires a known scalar kernel and does not extend to the project's other media (A-A9; H3 §9 item 7).

**Non-fatal but selection-blocking confound (all three).** The WL76/77 floors include SS_q quadrature variance that their noise estimate excludes [V1]. The minimal competitor (FIA with deterministic SS) is untested, so no expected advantage of exact exit projections can be attributed (A-A6).

---

## 6. Open questions left by this adjudication (not recommendations)

These are recorded as facts about what the evidence leaves undecided. The choice of any next step is the user's.
- Whether the WL77 Stage 1 statistic falls to the emulated exact-SS values (ring ≈ 0.22, FIA-AZ ≈ 0.04–0.08) when SS_q is replaced by high-q or deterministic SS. Specialist_physics §6.4 notes this is computable read-only from archived bundles.
- Whether a de-identified, D4-quotiented W+UV (W+UV-Q) transfers to the BND4_FAR translates. That would test the WL74A / C1–C2 attribution directly, without a new architecture.
- Whether the learning-free θ = 0 CPM operator (H3 Stage 1) meets the WL59 gates on the archived groups. This would be a physics result about "what must be explicit", not a learned-K result. Its contract status is a human decision.
- The bin-resolved (16 × 64) identity of the translates. Reviewer A's C1 is angle-integrated only.

## 7. Not verified

- Reviewer A's C1 (translate identity) and C2 (d noise) computations. These are on server archives; I did not rerun them.
- The specialist emulations (§6–§7 of specialist_physics). Non-evidence.
- Every cost figure (B2, B5, B6, candidate §10s). These are arithmetic only; nothing was timed.
- The web prior-art checks of Review B (NGF, Dym 2024, Huang 2022, Garg 2006).
- The off-diagonal decay of L* (A-A8), the sign of r*_e − m_e (A-A12) and the magnitude of the D2 truncation (A-A5).
- Whether Gate 4's TBE/SM/SR far slots are along-edge translates of S16-R. The population was deliberately not opened.
