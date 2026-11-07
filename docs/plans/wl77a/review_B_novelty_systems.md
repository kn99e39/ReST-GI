# WL77A Review B: Novelty and Systems (2026-11-07)

**Role.** Reviewer B (novelty / systems) in the WL77A multi-agent architecture-discovery workflow.

**Status.** Read-only adversarial review. This file is the only repository change. No GPU job, training, server access or Monte Carlo was run. WL77 files, historical worklogs and results are untouched. The Gate 4 confirmation population (`ports76.json`) and its references were **not** opened; only the selection rule in `experiments/fnto_fia76/ports76.py` was read.

**Scope.** For each material objection: the claim challenged, the counterargument, the evidence, and a severity (fatal / major / minor). Objections that apply to several candidates are recorded once. The review does not select a candidate or recommend a research direction. It states which tests would resolve each objection.

**Labels.**
- **[V]** verified by me in code, results or primary web pages (path:line or URL).
- **[S]** from a specialist or candidate report, not rechecked.
- **[E]** my arithmetic or estimate.
- **[U]** unverified.

---

## 0. What I verified

### Repository

| Item | Finding |
|---|---|
| A75 / AR per-source rest con_coarse at the BND4_FAR translates [V] | `results/fnto_rank75/final/per_matrix.csv`, worst channel, `hi`, max over replica sets:<br>• A75 s101 0.72 / 0.75, s105 0.40 / 0.46, s107 1.07 / 1.30, s103 0.64 / 0.91<br>• AR s101 0.75 / 0.86, s105 0.43 / 0.51, s107 0.96 / 1.01<br>• G2 oracle s101 0.19, s105 0.09, s107 0.20<br>• A75 TRAIN_NEW (the group containing the partners s50–s53) is 0.131 (`summary.json` gate block) |
| Translate geometry [V] | Partners from `results/fnto_ato74/ports4.json` and `results/fnto_src73/ports.json`:<br>• s101 (−48.75, −8.75, −5) ↔ s50 (48.75, −31.25, 5)<br>• s105 (−50, 1.25, −1.25) ↔ s52 (−50, 31.25, 1.25)<br>• s107 (−8.75, −50, −3.75) ↔ s53 (−31.25, −50, 3.75)<br>Each is a point-group image composed with an along-edge translation. Their 1 cm coarse sub-cell positions also match: each is at the cell-edge port in the along-edge axis, from coordinates [E]. `ports4.json` lists `d_to_S16R_images_mm` 20.0 for s101, so "images" in the protocol means point-group images only. |
| Identity and world-axis channels in W+UV [V] | Node features include `xyz0/.05, xyz/.05, …, port_face/5` (`r57_geometry.py:58`) and a global 11 × 11 dent grid (`:60-62`). Pair features `dr` are world-axis rest offsets (`:65-66`). U is a softmax over all P and mass is a free sigmoid per node (`r57_model.py:61-63`). |
| B/β quadrature [V] | One generator seeded once (seed 57501) and consumed sequentially across all cells (`r57_geometry.py:80-93`). Ballistic samples for symmetry images are therefore different random draws. |
| Gate 4 selection rule [V] | The far stratum maximises distance to `sl.images(actual)` (`ports76.py:55`). `Slab.images` is the point-group permutation set only (`experiments/fnto_src73/ports73.py:37,40`). Contexts: TBE = top/bottom with e < 10 mm and E ≥ 10 mm; SM / SR = side face with e ≥ 10 mm and a fixed row; INT; SV (`ports73.py:33-35`). |
| Contract lines [V] | • `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md:17`: "…do not replace learned complex shading with a limited explicit radiosity solver"<br>• `:86`: "…not an exhaustive explicit per-surface full path tracer hidden under a neural name"<br>• `docs/Architecture.md:94`: a free-space radiosity solver "is not an interchangeable implementation of this contract" |

### Web (new kill searches, 2026-11-07)

- **Neural Green's Functions**, Yoo, Yeo, Hwang, Sung, NeurIPS 2025, arXiv 2511.01924 [V abstract]. A network takes per-point features of a volumetric point cloud of the domain and predicts a decomposition of the solution operator of a linear PDE. It is "by design, agnostic to the specific functions used during training", and generalises to unseen source/boundary functions and irregular geometries. It is elliptic and eigendecomposition-based. https://arxiv.org/abs/2511.01924
- **Dym, Lawrence, Siegel**, "Equivariant Frames and the Impossibility of Continuous Canonicalization", ICML 2024, arXiv 2402.16077 [V]. Introduces *weighted* (robust) frames with input-dependent weights so that frame averaging stays continuous. https://arxiv.org/abs/2402.16077
- **Puny et al.**, "Frame Averaging for Invariant and Equivariant Network Design", ICLR 2022, arXiv 2110.03336 [V]. https://arxiv.org/abs/2110.03336
- **Garg, Talvala, Levoy, Lensch**, "Symmetric Photography", EGSR 2006 [V]. Recovers a dense transport matrix from few illumination patterns using Helmholtz-reciprocity symmetry plus hierarchical local low rank. https://graphics.stanford.edu/papers/symmetric_photography/
- **Huang, Cheng, Christlieb, Roberts**, "Machine learning moment closure models for the RTE I–III", JCP 2022, arXiv 2105.05690 [V abstract]. Also "Learning Closure Relations using Differentiable Programming: An Example in Radiation Transport", NeurIPS 2023 workshop [S, listing only]. Learned closures of transport discretisations, trained through or alongside a solver. https://arxiv.org/abs/2105.05690 , https://neurips.cc/virtual/2023/76092
- **Element Learning**, Du and Stechmann, arXiv 2308.02467 [V abstract]: geometry → learned in2out / in2sol (Green's) element operators, assembled by a global solver.
- **Vicini et al. 2019** project page [V]: a low-order implicit polynomial descriptor of local geometry. Frame orientation and degree are not stated on the page, so the canonicalisation detail remains [U].
- A search for learned SPH or collision-probability corrections in reactor physics found nothing direct. That is a weak negative.

---

## 1. Objections

### B1 — The translate argument does not discriminate between mechanisms. A cheap identity-free control is missing (H1, H2, H3) — **major**

**Claims challenged.** H1-C1, H2-C6, H3-C3, and ideation §5.2 ("the translation part of the quotient is exactly what Gate 4 tests at its far strata").

**Counterargument.**
- All three candidates rest their headline transfer prediction on s101 / s105 / s107 becoming *seen inputs* under exact D4h + lattice-translation equivariance. That property is shared by every identity-free equivariant model. It is not specific to:
  - canonical frames (H1);
  - exact reciprocal angular ends (H2);
  - a solved generator (H3).
- The A75 failure at these sources is large: 0.40–1.30 per source against TRAIN_NEW 0.131 and G2 0.09–0.20 [V]. A75 also carries exactly the channels that break this equivalence: absolute xyz0/xyz, face id, the global dent grid, world-axis pair offsets, a global U and a free per-node mass [V].
- The most economical explanation of the translate failure is therefore "identity / world-axis channels plus no point-group equivariance". A pass at the translates would confirm that, but it would not separate H1 from H2 from H3, nor any of them from a de-identified, symmetrised W+UV.
- The translates are also development sources that were used while designing the candidates. H2's F-T and H3's F2 use them as *falsifiers*, so those falsifiers are non-blind and nearly automatic for any correct implementation.
- **Blind data.** Gate 4 far strata maximise distance to point-group images only (`ports76.py:55`, `ports73.py:40`) [V]. Context membership is fixed for SM / SR / INT (e ≥ 10 mm, fixed row), so their far strata are plausibly along-edge translates of S16-R contexts again (inference; the population was not read).
- **Consequence.** Part of Gate 4's far-stratum outcome is predicted by symmetry alone. The discriminating evidence sits in the non-orbit contexts (TBE at untrained edge distances, TBC offsets, SV) and in the sub-cell pairs.

**Evidence.** Section 0 table; `r57_geometry.py:58-66`; `r57_model.py:61-63`; WL75 per-source table (`docs/worklogs/75_…:239-244`).

**What would resolve it.** Add one shared control arm under the same S16-R, objective, seeds and schedule: **W+UV-Q**, i.e. A75 with these changes:
- xyz0 / xyz / face id / global grid removed;
- pair offsets and normals expressed in canonical source frames, with exact D4 bin permutations on A/O indices (H1's quotient applied to A75);
- U localised.

Then report every candidate's far-stratum gain *relative to W+UV-Q*. Predeclare falsifiers separately for orbit-equivalent and non-orbit contexts.

---

### B2 — H1's state and apply are understated about 3.5–5× because frame averaging cannot be stored as one rank-4 factor set (H1) — **major**

**Claim challenged.** H1-C7 ("state ≈ 0.43 GB … 1-light ≤ 5 ms").

**Counterargument.**
- K^s = d Σ_g π_g m(g) Π(g). Each g has its own α(g), β(g), Z(g), evaluated with frame-g inputs and mapped back by inverse permutations. A pair block is therefore a **sum of |supp π| rank-4 blocks**: rank ≤ 4|supp π|, with per-g normalisers. It cannot be stored as one (16·4 + 4·64)·3 factor set.
- The §10 rebuild estimate already uses mean |supp π| ≈ 5, but the state estimate counts one frame.
- Supp sizes: interior top/bottom ports 12 mm or more from a side edge give π uniform over 8. The opposite face is parallel, so the in-plane moment is zero. That is about (76/100)² × 3200 ≈ 1,850 ports [E]; most others have 2. Mean |supp π| ≈ 4.8 [E].
- **Corrected state [E]:**
  - stored factors: 8.45e7 × 4.8 + 2.3e7 × 4.8 ≈ 5.2e8 floats ≈ **2.1 GB** (the 2 GiB hard cap is 2.15 GB);
  - or dense near blocks: 8.8e4 × 3,072 = 2.7e8 floats (1.08 GB) plus a ×4.8 far zone (0.44 GB) ≈ **1.5 GB**.
- **Corrected apply [E].** A 1-light apply reads ≥ 1.5 GB, which is ≥ 1.7 ms at the 912 GB/s vendor peak. In the A75 torch style that the systems budget prescribes (W+UV runs about 100× roofline, `specialist_systems.md` §4.1 [S]), it would very likely exceed the 5 ms cap. The 64-light apply scales by about 5× from the 0.1–0.5 s estimate, to about 0.5–2.5 s, with the 0.86 s cap inside that range.
- **Mitigation.** At rest, interior blocks are identical across lattice translates and can be cached per class, as H1 notes. Dented poses (pt0, te*, cf*) need explicit blocks over the dent support.

**Evidence.** `candidate_H1.md` §3.3 (operator), §10 (state counted for one frame, rebuild for ~5 frames); `specialist_systems.md` §7–8 budgets.

**What would resolve it.** Restate the state, apply and 1-light budget with frame multiplicity. Measure them on pt0 (not rest) in Stage A. Alternatively, collapse frames at rebuild into a dense near block and charge 1.5 GB.

---

### B3 — H1's control cannot isolate its mechanism, and its "possibly new" element is prior art (H1) — **major**

**Claims challenged.** H1-C3, H1 §12 item (1) (the quotient with continuous frame weights as a possibly new mechanism), item (3) (attribution to "identity channels + non-quotiented frames + CP pairing"), and F-c.

**Counterargument.**

1. **Originality.** Continuous, input-dependent convex frame weights are the "robust weighted frames" of Dym, Lawrence and Siegel (ICML 2024) [V]. Frame averaging itself is Puny et al. (ICLR 2022) [V]. D4 has only 8 elements, so the full Reynolds average is already tractable and exactly equivariant. Weighted frames add no exactness. Their only role is canonicalisation, which reduces the network's orientation burden, together with a cost saving: mean |supp π| ≈ 4.8 vs 8, about 1.7× [E]. Element (1) of §12 is therefore an application of known tools, and the claimable content reduces to the evaluation and the attribution.
2. **Identifiability.**
   - H1-noQ (native frames, π = δ_id) differs from H1 in three ways at once: (a) exact point-group equivariance, (b) canonicalisation (learning burden), and (c) an ensemble of up to 8 evaluations, a variance-reduction effect.
   - F-c cannot say which one did the work.
   - Attribution claim (3) bundles three factors, and only frames are ablated. Identity removal and pair-specific angular blocks are never ablated separately.
   - The F-c threshold "less than the seed spread" with 2 seeds is the range of two numbers, not an uncertainty estimate (see B11).

**Evidence.** URLs in §0; `candidate_H1.md` §12, §14.

**What would resolve it.** Add these arms:
- **H1-Reynolds:** uniform π over all 8 elements everywhere, which is exactly equivariant and not canonicalised.
- **W+UV-Q** from B1, to separate identity removal and CP-pairing removal.

Compare H1 vs H1-Reynolds (canonicalisation), H1-Reynolds vs H1-noQ (equivariance plus ensembling) and H1 vs W+UV-Q (pair-specific angular blocks). Cite Dym 2024 and Puny 2022 as the mechanism precedent.

---

### B4 — Credit assignment: H2 and H3 put the transfer mechanism in exact physics, so a pass may belong to a classical solver, not to learning (H2, H3) — **major**

**Claims challenged.** H2-C6 / C7, H3-C3 / C5, and both §12 novelty statements.

**Counterargument.**
- Both candidates require known σ_t, ω_c, g = 0, isotropy and index matching. They compute exactly:
  - single scatter (H2, H3) and double scatter (H3);
  - first-collision / escape projections;
  - in H3, the transport generator itself.
- What is learned is a scalar resolvent (H2) or a ±50 % one-stencil closure (H3, about 6–10k parameters).
- **H3** names Horn A and has a learning-free control (θ = 0) and F4. But F4 ("closes ≥ 50 % of the 𝔊 → 𝔊_ref θ = 0 gap") is undefined or noise-dominated if that gap is at or below replica noise and seed spread. Its own predicted θ = 0 floor is native 0.02–0.06 against 𝔊_ref, which is plausibly within a few hundredths. A pass of F2 with an F4 failure is then a classical collision-probability result.
- **H2** has K_phys at Stage 0, but:
  - Γ_phys is a walker-MC diagnostic, not a deployable learning-free comparator;
  - F-T and F-G compare learned Γ only with A75 (which is given none of this physics) and with Γ_phys;
  - a deployable learning-free Γ exists in principle: the Galerkin Schur complement of H3's θ = 0 CPM operator on H2's nodes, or an infinite-medium Green's function with face images (Jensen 2001 / dipole-image class). Neither is a control.
- **Information fairness.** The comparison with A75 is unfair in information. A75 receives only B / β, while H2 and H3 receive exact SS (16–41 % of collided exit [S]) and exact angular ends. "Beats A75" is therefore not evidence that the learned component transfers.
- **What remains claimable for H2.** If deterministic Γ transfers as well, H2's distinctive learned value is **rebuild amortisation**: about 0.05 s against minutes per dented geometry for a per-geometry Γ, which H2 states in counterexample C. That is a cost claim, not a source-transfer claim, and it moves the result onto the contract question in FNTO lines 17 / 86 [V].

**Evidence.** `candidate_H2.md` §13(C), §14; `candidate_H3.md` §13 Horn A, §14 F4; contract lines in §0.

**What would resolve it.**
- For both candidates, make the **learning-free deployable control** the primary comparator for every transfer falsifier:
  - H2: Γ_CPM on the same nodes, or Γ_phys if the protocol accepts per-geometry MC;
  - H3: θ = 0.
- Predeclare the claim as "learned beats the learning-free control on unseen groups by more than a replica-jackknife interval".
- For H3, add a noise floor to F4: compute the 𝔊 → 𝔊_ref gap with jackknife intervals, and make F4 *inconclusive* if the gap is below about 2× its interval.
- Record the FNTO admissibility decision before Stage 2, as H3 already proposes.

---

### B5 — H3's 1-light apply is memory-bound and exceeds the 5 ms cap even at roofline, as specified (H3) — **major**

**Claims challenged.** H3-C7 ("1-light apply ≤ 5 ms … ≈ 2–5 ms compute-bound") and F5.

**Counterargument.**
- **Compute.** 2.9e9 MAC (5.8 GFLOP) is about 0.17 ms at 34 TFLOP/s. The apply is not compute-bound.
- **Memory.** Each Chebyshev iteration must read:
  - the near generator, 1.6–2.3e7 coefficients, shared over channels;
  - the closure deltas, 8.4e6;
  - the far two-level part, 4.6e6, plus R.

  That is ≈ 3–3.7e7 floats ≈ 0.12–0.15 GB per iteration, plus indices unless a structured stencil kernel is written. At 38 iterations (red, ε = 1e-5) that is about 4.6–5.7 GB, or **5–6.3 ms at the 912 GB/s vendor peak** [E], i.e. at or above the 5 ms cap before any inefficiency.
- **Implementation style.** F5 prescribes measurement "in the A75 torch style". 38 torch sparse matvecs over 2.5e7+ nnz would land at tens to hundreds of ms per light. The A75-style apply ran about 100× roofline (`specialist_systems.md` §4.1 [S]).
- **Mitigation.** At rest, translation-keyed coefficients generated on the fly in a custom kernel could avoid the reads. Dented footprints remain explicit, and a custom kernel breaks the "same implementation style" comparability.
- **Coupled solve.** The K∘C solve applies K 6–20 times per frame, which multiplies the cost.
- **Scalability [E].** For thicker or larger media, red κ approaches 1/(1 − ω) ≈ 1000 instead of the slab-leakage-limited 38. The iteration count grows by roughly √(1000/38) ≈ 5×.

**Evidence.** `candidate_H3.md` §10 (counts), §9 item 5 (κ ≈ 38 / 29 / 8, which I reproduce from B² = (π/11.42 mm)²); `specialist_systems.md` §8 (1-light cap 5 ms; roofline guidance).

**What would resolve it.**
- Restate F5's 1-light apply budget with a bytes-per-iteration roofline.
- Either commit to a structured on-the-fly stencil kernel (declared, with its rest/dent split), or accept that F5 will likely fail.
- Measure on pt0, not on rest.

---

### B6 — Nonrigid geometry-term rebuilds are under-counted and have no locality certificates (H3, H2) — **major**

**Claims challenged.** H3-C7 (τ̂ "about 1e8 kernel evaluations", "≤ 1e7 visibility rays", 0.3–2.5 s) and H2-C8 (Π 1–2 s, SS 1–3 s).

**Counterargument.**

1. **H3 pair integrals.**
   - About 5e6 near pair integrals with about 1e8 kernel evaluations is about 20 points per 6-D integral of the weakly singular kernel e^{−σr}/r² over adjacent or self elements.
   - Stage 0(i) requires ≤ 1e-3 change when rule orders double. That needs singularity-adapted cubature with ≥ 10²–10³ points for self and adjacent pairs [E].
   - Pairs straddling the dented surface need visibility *per cubature point*, not one ray per pair. That is plausibly 1e8–1e9 rays, or 1–10 s at the measured B throughput of about 88 M rays/s [S].
   - Kernel arithmetic is cheap on a GPU; visibility dominates.
2. **H2.**
   - A full-pose Π costs 245,760 × 256 = 6.3e7 casts. That is 2× the systems hard cap of 32 M casts for extra quadrature (`specialist_systems.md` §8).
   - SS for a dent is a declared 1–3 s, over the 1 s cap.
3. **Both.** Neither specifies the per-path / per-segment **locality certificates** that the systems budget requires ("with locality certificates", §8). Without them, "recompute only where geometry changed" defaults to full recomputation, which is the WL59 / FIA lesson.
4. **H2 invalidation radius.** H2's ratio-min normaliser (R over ρ, then min over neighbours) reaches 2ρ_red + Π/SS reach, about 35 mm. A pt0-size dent (r = 22.5 mm) then invalidates a radius of about 57 mm, i.e. most of the 100 × 100 mm top face [E].

**Evidence.** `candidate_H3.md` §10; `candidate_H2.md` §10; `specialist_systems.md` §5 item 3 and §8.

**What would resolve it.**
- Predeclare cubature orders for self, adjacent and far pairs.
- Count visibility rays per cubature point.
- Specify the certificate (the set of triangles each path or segment touches).
- Report the rebuild as measured on pt0 / te1 against the 1 s cap, with B charged separately.

---

### B7 — H2's novelty framing omits the closest learned prior art; its distinction is application-level (H2) — **major** (framing; not fatal to the experiment)

**Claims challenged.** H2 §12(b) (learning a "direction-free, source-independent, geometry-conditioned volumetric resolvent", presented as mechanism-level) and H2-C4 (no source-identity channel as a distinguishing property).

**Counterargument.**
- **Neural Green's Functions** (NeurIPS 2025) [V] already learns a *geometry-only* operator decomposition from a volumetric point cloud. It is agnostic to source functions by design and is evaluated on unseen sources and unseen geometries. That is H2's central design principle, for elliptic PDEs.
- **Element Learning** [V] learns geometry-conditioned RTE in2out / in2sol maps.
- **DeepRTE** [S] learns from delta inflows.
- **Wang–Tran–Luebke 2005** [V-sec] is a surface-to-surface translucent transfer matrix with separate single scatter, the non-learned Πᵀ·Π analogue.
- **Garg et al. 2006** [V] exploits reciprocity symmetry plus hierarchical low rank to infer transport columns that were never directly measured. That bears on H2's and H3's use of reciprocity as a transfer argument.
- **What remains.** The RTE boundary-to-boundary instance with exact reciprocal angular ends on a face-graded hat basis, port/bin correspondence and per-cell passivity, plus the ratio-min normaliser (a small algebraic device next to symmetric matrix scaling).
- **The O1 claim.** "Held-out sparse-source transfer" is an evaluation, not a mechanism. Kernel Nyström 2009 and symmetric photography already treat sparse-column transport completion; the geometry-conditioned part is what differs.

**Evidence.** URLs in §0; `specialist_priorart.md` §3.6–3.7 and §6 (O1).

**What would resolve it.** Cite NGF, Element Learning, DeepRTE, Wang 2005 and Garg 2006 in H2 §11. Restrict §12 to the combination plus the evaluation. Drop "source-independent" as a mechanism-level distinction.

---

### B8 — H3's learned-closure precedent is broader than listed (H3) — **minor**

**Claim challenged.** H3 §12 item 2 ("learned … stencil closure of the CPM generator, trained through the solve" as plausibly not covered).

**Counterargument.**
- Learned closures of RTE discretisations already exist: ML moment closures for the RTE (Huang, Cheng, Christlieb, Roberts, JCP 2022 [V abstract]) and differentiable-programming closure learning in radiation transport (NeurIPS 2023 workshop [S]). Solver-in-the-loop closures (Um 2020; Kochkov 2021) also exist [U, cited by H3].
- Only the CPM-generator target and the symmetric / M-matrix-preserving parameterisation are specific.
- H3 already concedes weak mechanism novelty, so this does not change selection. It only narrows item 2.

**Evidence.** URLs in §0.

---

### B9 — "Bit-identical image" claims depend on symmetric quadratures that are not all symmetric (H1, H2, H3) — **minor**

**Claims challenged.** H1 §5.2 ("exact images receive bit-identical features"), H2-C4 and H3 §3.3 exactness.

**Counterargument.**
- **H1.** φ_FC uses one fixed Sobol pattern of 16 positions × 16 directions. A Sobol point set is not D4-symmetric, so image features differ by quadrature error, unless the pattern is symmetrised as an 8-fold orbit or evaluated per canonical frame (which would multiply the 15.7 M casts by |supp π|). H1 itself says the MLP needs about 1 % precision on a 1–3 % contrast signal.
- **All three.** B / β come from one sequentially consumed generator (`r57_geometry.py:80-93`) [V], so d differs between images. H2's guard s_c = min(1, (1 − β_B)/E^raw) then breaks reciprocity and image identity wherever it is active.
- **H2 / H3.** Their midpoint-stratified concentric grids and product Gauss rules are D4-symmetric if no jitter is used [E].

**What would resolve it.**
- Symmetrise the H1 pattern.
- Report image-pair discrepancies of features and K rows in the Stage 0 / A unit tests, including d-induced differences.
- For H2, report how often the guard is active and its magnitude at image pairs.

---

### B10 — The coarse-metric sub-cell confound limits "error equals partner" and the far/near ratio falsifiers (H1, H2, H3) — **minor**

**Claims challenged.** H1-C1 / C6 and F-b, H3-C3 (BND4_FAR ≤ 1.5 × TRAIN_NEW), H2 F-T.

**Counterargument.**
- Port-level equivariance makes predictions equal, not con_coarse equal.
- The coarse contrast signal depends on the 1 cm sub-cell position: |HZ|/|Z| is about 0.054 at an interior centre and 0.14–0.17 at edge / corner (`specialist_systems.md` §3.2 [S]); G2 by sub-cell is 0.09 / 0.09 / 0.14 (WL75 table, lines 247-251 [V]).
- The three dev translates happen to match their partners' sub-cell [E, coordinates]. Gate 4's hi/lo pairs are, by design, sub-cell-mismatched.
- Ratios of far over near or TRAIN_NEW con_coarse therefore mix transfer with sub-cell composition.

**What would resolve it.** Predeclare sub-cell-matched comparisons: ratios within hi and within lo separately, or ratios against the G2 / K_phys / θ = 0 floor of the same source.

---

### B11 — Two-seed "seed spread" criteria are not uncertainty estimates (H1, H3; H2 F-T "both seeds") — **minor**

**Claims challenged.** H1 F-c, H3 F3 / F4, and H2's both-seed criteria used as an implicit robustness argument.

**Counterargument.**
- The spread of two seeds is a single difference, with no coverage guarantee.
- The project has history of single-seed differences that did not replicate: WL37 → WL38, where the gap was −0.313 dB on one seed and a mean of −0.082 dB over three.
- Mechanism-vs-capacity comparisons (H1 vs H1-noQ; closure vs θ = 0) need an interval on the *difference*.

**What would resolve it.** ≥ 3 seeds for the decisive comparisons, or a predeclared paired-replica jackknife interval on the difference, combined with the seed range.

---

### B12 — Scalability beyond the fixture is weak for all three, and differently so (H1, H2, H3) — **minor**

**Claims challenged.** Implicit generality in the §12 contribution statements.

**Counterargument [E].**
- **H1:** the near-core state scales as h⁻⁴ (H1 admits about 5.5 GB at 1.25 mm, before the B2 ×4.8).
- **H2:** Γ nnz scales as ρ³. If thickness no longer truncates the red tail, ρ grows from 15 mm towards about 5 × 18 mm, a factor of about 47 in Γ state.
- **H3:** element count scales with volume, and iterations with √κ, up to about 5× more for thick red (B5).
- **H2 / H3 material scope:** both need a known scalar collision kernel. They do not transfer to the project's Burley / Fresnel / anisotropic regimes (WL51) without angular state (H3-C9; H2 §8).

None of this blocks a WL77A fixture test. It bounds what a pass can claim.

---

## 2. Re-derived cost summary [E]

| | H1 (as stated) | H1 (B2 corrected) | H2 | H3 |
|---|---|---|---|---|
| Persistent state | 0.43 GB | 1.5 GB (dense near) to 2.1 GB (factors) | 0.2–0.55 GB (plausible) | 0.3–0.6 GB translation-keyed; ≤ 2 GB explicit |
| 1-light apply | 1–4 ms | ≥ 1.7 ms roofline; likely > 5 ms in torch style | 2–5 ms (at risk) | ≥ 5–6.3 ms roofline (memory-bound); torch style ≫ 5 ms |
| Extra quadrature | 15.7 M casts | (×|supp π| if per-frame φ_FC) | 63 M casts full pose (> 32 M cap); dent footprint ≈ 7e6 | visibility per cubature point near dents, plausibly 1e8–1e9 rays |
| Nonrigid rebuild | 0.05–0.2 s learned | same | learned 0.02–0.1 s; geometry 1–3 s (> 1 s cap) | learned 10–30 ms; geometry under-counted (B6) |
| Training step | 0.05–0.3 s | ~5× frame evaluations | 0.1–0.6 s; lane memory tight | 0.2–0.5 s claimed; one lane |

---

## 3. Uncertainty

- Every cost in §2 is arithmetic from the candidates' own counts and vendor peak bandwidth. Nothing was timed.
- Whether Gate 4 far strata are translates in each context is inferred from the selection rule. The population was deliberately not read.
- Vicini 2019's frame canonicalisation remains [U].
- NGF and DeepRTE were checked at abstract level only.
- Frame-support counts for H1 (mean |supp π| ≈ 4.8) assume the ω-windowed in-plane moment vanishes for ports ≥ 12 mm from side edges. That holds by symmetry of the parallel opposite face at rest.
