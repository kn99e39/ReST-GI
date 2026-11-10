# GM1 review — novelty kill search for Track B (path-space / reservoir reuse)

**Status:** design-only review record (2026-11-10), repository `Selective_Recompute` @ `c7b0698`. Target: the sealed report `docs/plans/gm1_architecture_review/11_design_trackB_path_space_reuse.md` ("the report"). Nothing was implemented, run or measured; no existing file was changed. This document authorises nothing and recommends no direction.

**Evidence labels:** [FACT] canonical contract or verified measurement; [HIST] historical evidence in its original scope; [DERIV] derivation; [INFER] inference; [HYP] hypothesis; [SPEC] speculation.

**Citation verification (this session, 2026-11-10):**
- **V** — title, authors, venue confirmed on a publisher / author / arXiv / dblp record returned by search or fetch.
- **V-x** — as V, and the mechanism was read from the arXiv HTML through an automated extraction (not a line-by-line reading by me; numbers quoted from it carry that caveat).
- **V-s** — abstract or snippet level only.
- **U** — not verified this session (background knowledge); no classification rests on a U entry alone.

Three primary PDFs could not be opened (size limit): GRIS, ReSTIR PT Enhanced, the 2020 ReSTIR paper; the SIGGRAPH 2023 ReSTIR course PDF returned unreadable binary. Statements about their internals are marked.

**Repository sources read:** the report; `00_PROBLEM_BRIEF.md`; `docs/plans/wl77a/specialist_priorart.md`; `docs/plans/wl80a/prior_art_matrix.md` (l.1–289); `docs/plans/wl80b/c1_prior_art.md`; `AGENTS.md` baseline-terminology gate; `docs/Architecture.md` l.42, l.105 (GM1 reference levels). The earlier kill searches targeted an intrinsic K and contain only one item that bears on Track B: Drettakis & Sillion 1997 (WL80B P10, "recompute the affected links, then re-solve"), re-used below.

---

## 0. Result in brief

1. **The report's own verdict ("no mechanistic distinction") survives the kill search and is, if anything, understated.** [INFER] Every central mechanism is ALREADY KNOWN. The report claims no novelty, so there is no novelty claim to kill; what the search changes is the *comparator set* and two of the report's secondary claims.
2. **One closer predecessor was missed:** Xu, Li, Georgiev, Hedstrom, Ramamoorthi 2024, *Residual Path Integrals for Re-rendering* (CGF 43(4), EGSR; arXiv 2406.16302) [V-x]. It treats exactly "a small part of the scene moves rigidly, the previous frame is already rendered": previous image as control variate, the difference written as an integral over the *dynamic* path space only, paths sampled **from the moving object and from its vacated ("ghost") placement**, vertex-wise path mapping (object-attached vertices move rigidly, static vertices identity / reconnection, random-seed replay for diverged subpaths), extra acceleration structures for the old placement. This removes the last "GM1-specific setting" the report lists in §13 (single known rigid edit with object identity) as even a setting-level difference, and it is a stronger form of PR2 than the one the report designs.
3. **Two secondary claims in the report are NOT SUPPORTED as stated** (section 4): (a) "the saving is bounded above by 1/ρ" as a bound on *any* path-granular selective scheme; (b) "births are covered at the fresh sampler's rate, no better" as a property of path reuse in general (it is a property of PR1/PR2 as designed).
4. **Nothing in neural GI / learned transport overlaps Track B**, because Track B has no learned transport state. That is not open space; it means Track B is outside the class in which the project seeks a contribution (the report says so itself, §14).

---

## 1. Central mechanisms extracted from the report

| ID | Mechanism (report section) |
|---|---|
| M1 | Edit-revalidated path store, PR1: store sampled paths + seeds, keep clean paths, replay dirty suffixes with the same random numbers; equals the same-seed cold render (§1–3) |
| M2 | Geometric discovery of the invalid set: test stored segments (incl. shadow segments) against the moved object's old and new placement; optional hierarchy over segments; D_T ≠ D_G "by construction" (§3, §8) |
| M3 | PR2 difference re-render: I_T1 = I_T0 + correlated mean(f1 − f0), per-pixel control-variate weight (§1, §7) |
| M4 | PR3 reservoir temporal reuse across the edit: GRIS shift (hybrid replay / reconnection), T0-target MIS against the T0 placement, confidence reduction on the dirty set (§1, §4) |
| M5 | Birth / death / sharp-specular handling: suffix replay, manifold-walk path repair, bidirectional or photon candidates (§4, §10) |
| M6 | ER integration: `sample`/`eval`/`pdf` reuse, read-only lobe index + lobe width for reconnection eligibility (§9) |
| M7 | Learned add-ons: guiding for newborn paths, neural radiance cache, learned global state as control variate with path-space residual (§9) |
| M8 | Role claims: warm path-traced comparator; exact reuse-headroom measurement ρ with a 1/ρ ceiling (§0, §7, §12) |

---

## 2. Per-mechanism kill search

### M1 — stored paths, clean-path retention, same-seed suffix replay

**Closest prior methods.**
- Séquin & Smyrl 1989, *Parameterized Ray Tracing*, SIGGRAPH [U]: stored ray trees re-evaluated after an edit.
- Jevans 1992, *Object Space Temporal Coherence for Ray Tracing*, Graphics Interface '92, pp. 176–183 [V]: static-camera animation; object-space record of which rays pass which voxels, only affected pixels re-traced.
- Murakami & Hirota, *Incremental Ray Tracing* (c. 1990) [U — search did not return a record].
- Brière & Poulin 1996, *Hierarchical View-dependent Structures for Interactive Scene Manipulation*, SIGGRAPH 96 [V]: fixed camera; "detects and recomputes the exact portion of the image that has changed" from stored ray trees and hierarchies.
- Bala, Dorsey, Teller 1999, *Interactive Ray-Traced Scene Editing Using Ray Segment Trees*, EG Workshop on Rendering 1999 [V]: hierarchical 5D ray-segment trees find the stored radiance interpolants an object edit affects (about 0.1 s), re-render 3–4× faster than the base ray tracer [HIST, abstract; Whitted-style, not path tracing].
- Havran, Damez, Myszkowski, Seidel 2003, *An Efficient Spatio-Temporal Architecture for Animation Rendering*, EG Symposium on Rendering 2003 [V]: path samples (incl. bidirectional path tracing) reused across frames with a multi-frame visibility structure; only changed connections recomputed; "usually over one order of magnitude" vs frame-by-frame [HIST, abstract].
- Kelemen et al. 2002; Veach & Guibas 1997 [U]: primary-sample-space replay.
- Lin et al. 2022 (GRIS) [V for identity; internals U]: reservoirs carry a random seed for random replay — the stored-seed path record at reservoir granularity.

**Shared mechanism.** A path is a deterministic function of (scene, random numbers). If the edit does not intersect the path's segments, the function value is unchanged; otherwise re-evaluate from the first changed segment. [DERIV] This is the defining idea of 1989–1999 incremental ray tracing and of primary-sample-space replay.

**Actual architectural difference.** Full Monte Carlo path tracing with next-event estimation and a learned-closure BSDF, on a GPU megakernel, with all N paths per pixel stored.

**Is the difference known?** Yes. Path-traced (not Whitted) reuse across frames is Havran 2003; same-random-number re-evaluation of a changed scene is the "correlated path tracing" baseline that Xu et al. 2024 and temporal gradient-domain rendering (Manzi, Kettunen, Durand, Zwicker, Lehtinen 2016, ACM TOG 35(6) [V]) compare against. The BSDF being ER is irrelevant to the mechanism (ER is deterministic in u).

**Evidence for a credible contribution.** None would make it a mechanism contribution. Measured utility would need: equal-error GPU kernel time against cold REF-ER *and* against correlated path tracing without a store, with discovery and bookkeeping itemised, on a workload where cold REF-ER is not already at the cost floor.

**Classification: ALREADY KNOWN** (mechanism). GPU store with bit-identical replay under Dr.Jit: implementation engineering. Utility: unmeasured.

### M2 — geometric discovery of the invalid set

**Closest prior methods.** Bala et al. 1999 (ray-segment trees: exactly "index stored ray segments so an object edit finds the affected ones", output-sensitive) [V]; Brière & Poulin 1996 [V]; Jevans 1992 [V]; Drettakis & Sillion 1997, *Interactive Update of Global Illumination Using a Line-Space Hierarchy*, SIGGRAPH 97, pp. 57–64 [V]: the link hierarchy is traversed to find light-transfer links cut or changed by a moving object; Chen 1990, *Incremental Radiosity*, SIGGRAPH (Computer Graphics 24(4), pp. 135–144) [V] and George, Sillion, Greenberg 1990, *Radiosity Redistribution for Dynamic Environments*, IEEE CG&A 10(4) [V]: affected transfers found geometrically, corrected by shooting positive and negative energy; Dmitriev, Brabec, Myszkowski, Seidel 2002, *Interactive Global Illumination Using Selective Photon Tracing*, EG Workshop on Rendering 2002 [V]: photons re-traced selectively toward regions needing update, exploiting quasi-random sequence structure (the pilot-photon / corrective-photon detail is from memory [U]); Xu et al. 2024 [V-x]: "ghost" geometry and extra acceleration structures for the old placement.

**Shared mechanism.** Old-placement ∪ new-placement swept test against stored transport carriers (rays, links, photons); stationary receivers at any depth are found because the carrier, not the receiver, is tested. D_T ≠ D_G handling "without a dependency graph" is the standard property of all of these. [DERIV]

**Difference.** None beyond carrier type (Monte Carlo path segments incl. shadow segments).

**Known?** Yes.

**Classification: ALREADY KNOWN.** The report's cost caveats (c_obj ≈ c_scene, hierarchy build amortisation) match Bala's design space; no new element.

### M3 — PR2 difference re-render with control-variate weight

**Closest prior methods.**
- Günther & Grosch 2015, *Consistent Scene Editing by Progressive Difference Images*, CGF 34(4) (EGSR issue), pp. 41–51 [V]: object add / remove, material, camera, light edits rendered as a stochastic difference image, with heuristics for the high-noise removal case.
- Rousselle, Jarosz, Novák 2016, *Image-space Control Variates for Rendering*, ACM TOG 35(6) (SIGGRAPH Asia) [V]: previous render as control variate for re-rendering, per-pixel weighting; states that scene-editing and gradient-domain methods are control-variate estimators.
- Manzi et al. 2016, *Temporal Gradient-Domain Path Tracing* [V]: correlated temporal differences by shift mappings.
- **Xu et al. 2024, *Residual Path Integrals for Re-rendering* [V-x]** — the closest: rigid object motion; the residual integral is supported only on the dynamic path space; six sampling techniques start from dynamic vertices and ghost points (from emitter / sensor / both) plus rejection-filtered path tracing, combined by MIS; path mapping = rigid transform for object-attached vertices, identity / reconnection for static vertices with roughness and distance criteria extending Lin et al., random-seed replay for diverged subpaths, two-way mapping for unbiasedness. Reported: about 2–5× over correlated path tracing for object motion, benefit shrinking as dynamic objects occupy more of path space (about 1.24× with eight objects), large motions break the mapping, deformation unsupported, negative residuals with noise concentrated at the old and new positions [HIST, their scenes, via extraction; not verified line by line]. It uses only the previous *image*, no stored paths.

**Shared mechanism.** I_T1 = I_T0 + E[f1 − f0] with correlated sampling; variance fallback where correlation fails.

**Difference in the report.** PR2 draws the difference from a subset of *stored* T0 paths, found by M2. Xu et al. instead *importance-sample* the dirty path space directly from the moved object.

**Known?** Both sides are known; the report's variant is the weaker one (it is the "correlated path tracing" baseline plus a store). [INFER]

**Classification: ALREADY KNOWN.**

### M4 — PR3 reservoir temporal reuse across the edit

**Closest prior methods.** Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman 2022 (GRIS / ReSTIR PT), ACM TOG [V identity; internals U]; Ouyang, Liu, Kettunen, Pharr, Pantaleoni 2021, *ReSTIR GI*, CGF 40(8) (HPG), pp. 17–29 [V; its sample-validation details not re-read]; Lin, Kettunen, Wyman 2026, *ReSTIR PT Enhanced*, PACM CGIT 9(1), Art. 13 [V; internals V-s]; Kettunen, Lin, Ramamoorthi, Bashford-Rogers, Wyman 2023, *Conditional RIS and ReSTIR*, SIGGRAPH Asia [V]; Liu, Lin, Kettunen, Wyman, Ramamoorthi 2025, *Reservoir Splatting*, SIGGRAPH [V]; Boissé 2021, SIGGRAPH Asia Technical Communications, Art. 22 [V]; Boksansky, Jukarainen, Wyman 2021, RTG II ch. 23 [V]; Zhang & Wang 2023, *World-Space Spatiotemporal Path Resampling for Path Tracing*, CGF 42(7) (Pacific Graphics) [V — fills the report's "authors not verified"]; Sawhney, Lin, Kettunen, Bitterli, Ramamoorthi, Wyman, Pharr 2024, *Decorrelating ReSTIR Samplers via MCMC Mutations*, ACM TOG 43(1) [V].

**Shared mechanism.** Identical: PR3 *is* ReSTIR PT temporal reuse with one frame pair.

**Difference.** Offline quality-matched budget instead of 1 spp real time; one discrete edit instead of continuous animation; exact dirty set from M2 used to reset confidence.

**Known?** Evaluating the previous frame's target against previous-frame scene state for unbiased temporal MIS is, to my knowledge, the standard statement in the ReSTIR literature (2020 paper and the 2023 course) — **not re-verified this session** (PDFs unreadable), so the report's [INFER] label on this point stands. Resetting or clamping history where the scene changed is routine ReSTIR practice [INFER, U].

**Classification: ALREADY KNOWN.** Using the exact M2 dirty set to modulate reservoir confidence: UNCERTAIN as a named technique, and implementation engineering in kind.

### M5 — births, deaths, sharp specular chains

**Closest prior methods.** Hybrid shift degeneracy on glossy chains: GRIS [U internals]; lobe-level postponement in ReSTIR PT Enhanced [V-s]. Path repair: Jakob & Marschner 2012; Kaplanyan et al. 2014; Lehtinen et al. 2013 [all U]; Zeltner, Georgiev, Jakob 2020, *Specular Manifold Sampling*, ACM TOG 39(4) [V]; **Hong, Duan, Wang, Yuksel, Zeltner, Lin 2025, *Sample Space Partitioning and Spatiotemporal Resampling for Specular Manifold Sampling*, SIGGRAPH Asia [V]** — manifold sampling combined with ReSTIR temporal reuse for interactive caustics (not cited by the report; it is the published form of "carry a specular path across frames"). Caustic-type births: Hedstrom, Kettunen, Lin, Wyman, Li 2025, *ReSTIR BDPT*, ACM TOG (DOI 10.1145/3744898) [V]: caustics reservoirs accumulated across frames; Kern, Brüll, Grosch 2024, *ReSTIR FG*, EGSR [V]. Births sampled *from the moved object*: Xu et al. 2024 [V-x].

**Shared mechanism / difference.** The report proposes nothing here; it lists known remedies and predicts failure (its F3).

**Classification: ALREADY KNOWN** (all listed remedies). The report's derivation that reuse "contributes nothing where the edit matters most" is **NOT SUPPORTED as a general statement** — see section 4(b).

### M6 — ER integration, lobe index for reconnection eligibility

**Closest prior.** ReSTIR PT Enhanced supplemental: "lobe-indexed paths", only (near-)specular *lobes* are postponed with random replay rather than classifying a whole vertex [V-s, supplemental snippet]. Per-lobe roughness criteria for reconnection are the GRIS / Enhanced design [U internals].

**Difference.** The lobes are ER's compiled lobes (incl. refraction-conditioned inner lobes whose outside footprint is not their α). That is a property of ER to be handled, not a mechanism.

**Classification: ALREADY KNOWN** (lobe-indexed reconnection). The ER-specific footprint caveat is a correct engineering observation [INFER]. Whether exposing a lobe index counts as "changing ER" is a contract question for a human decision, not a novelty matter.

### M7 — learned add-ons

**Closest prior.** Zeng, Kettunen, Wyman, Wu, Ramamoorthi, Yan, Lin 2025, *ReSTIR PG*, SIGGRAPH Asia (conference track) [V]; Dittebrandt, Hanika, Dachsbacher 2020, EGSR (DL-only track) [V]; Müller, Gross, Novák 2017, CGF 36(4) (EGSR) [V]; Müller, Rousselle, Novák, Keller 2021, *Real-time Neural Radiance Caching*, ACM TOG 40(4) [V, per WL77A]; Müller, Rousselle, Novák, Keller 2020, *Neural Control Variates*, ACM TOG (SIGGRAPH Asia) [V — note author order].

**Shared mechanism.** Identical constructions. In particular "learned global transport state as control variate + unbiased path-space residual" is the neural-control-variates estimator with the control variate supplied by another representation. [DERIV]

**Difference.** The control variate would be an *edit-updated reusable transport representation* rather than an online-fitted scene cache. No published instance was found in this search; that is a weak negative and the report itself labels the idea [SPEC]. Its novelty, if any, would belong to the other track's representation, not to the residual estimator.

**Classification: ALREADY KNOWN** (guiding, NRC, NCV). The hybrid "reusable learned state + residual path correction": **UNCERTAIN, application-level only**; NOT SUPPORTED as a Track B contribution (nothing specified, nothing measured).

### M8 — role claims: warm comparator and reuse-headroom measurement

**(i) Warm path-traced comparator.** A protocol decision, not a mechanism. The search changes its content: the strongest published warm estimators for a rigid edit are Xu et al. 2024 (offline, unbiased, previous image only) and ReSTIR PT Enhanced / ReSTIR BDPT (interactive). A comparator built only from the report's PR1/PR2 would be weaker than published practice, so "any GM1 mechanism must beat warm path reuse" would be under-tested by PR1/PR2 alone. [INFER]

**(ii) Headroom ρ.** Measuring the invalidated fraction is known in substance: Bala 1999 reports affected-interpolant discovery and the resulting speed-up; Xu et al. report benefit as a function of dynamic-path share; Jevans / Brière–Poulin report changed-pixel fractions [HIST, their scopes]. A *radiometrically weighted, per-depth* dirty fraction on a GM1 fixture with ER is a new **number**, not a new mechanism (same status as WL80A's "PR: per-pair rank as a measurement").

**Classification:** (i) not a novelty item; (ii) **ALREADY KNOWN** as a concept, possible **measured utility** as a project-internal diagnostic. The "1/ρ ceiling for any path-granular selective scheme": **NOT SUPPORTED** (section 4(a)).

---

## 3. Coverage of the prioritised literature classes

| Class | Bearing on Track B | Finding |
|---|---|---|
| Neural GI / learned transport (neural radiosity, NRC, neural PRT, RNA, 8DNA, TransGI, generalizable LT embedding — records in `docs/plans/wl77a/specialist_priorart.md` §3.1–3.2) | Track B has no learned transport state; these are other baseline roles (Neural Full Recompute / Native Reuse) | No overlap; no open space implied. A search for neural caches with explicit geometric invalidation after an object move returned only online-adapting caches (NRC; a 2025 Gaussian-splat volume cache adapting to parameter changes [V-s]) — weak negative |
| ReSTIR GI / PT / GRIS, world- and path-space reservoirs | M4, M5, M6 | Identity |
| Directional radiance caches (irradiance / radiance caching with gradients, DDGI, surfel GI) | Not used by Track B (it stores samples, not a function) | Temporal cache invalidation by aging: Tawara, Myszkowski, Seidel 2004 [V-s; venue CGI 2004 per one listing, a conflicting PDF metadata entry exists]; Render Cache (Walter, Drettakis, Parker 1999) [V-s]; Shading Cache (Tole, Pellacini, Walter, Greenberg 2002, SIGGRAPH) [V]. These are the biased, function-space siblings of M1/M2 |
| Incremental rendering / delta transport | M1–M3 | Chen 1990; George et al. 1990; Günther & Grosch 2015; Rousselle et al. 2016; Manzi et al. 2016; Xu et al. 2024 — identity up to carrier |
| Precomputed / reusable transport (PRT family, shadow fields, direct-to-indirect, row-column sampling) | None: Track B precomputes no operator | Not applicable; covered for K/C in WL77A / WL80A records |
| Hierarchical transport, sparse updates | M2 (segment hierarchy) | Drettakis & Sillion 1997; Bala 1999 — identity |
| Dynamic interreflection 2007–2013 (implicit visibility, antiradiance, ISM, voxel cone tracing), Lumen-style caches | Not path-store methods; they trade bias for update speed | No overlap with M1–M4; they are alternative comparators of a different (biased) class, outside Track B |
| Path reuse across frames / animation | M1 | Havran et al. 2003; Bekaert, Sbert, Halton 2002, EGWR pp. 125–134 [V]; Sbert, Castro, Halton 2004, CGI pp. 532–535 [V]; Méndez-Feliu, Sbert, Szirmay-Kalos 2006, J. WSCG [V] — reuse of paths for moving lights / cameras; moving *geometry* is covered by Havran 2003 and Xu 2024 |

---

## 4. Objections to secondary claims in the report

**(a) "The saving is bounded above by 1/ρ" and F2 ("no path-granular selective scheme can pay").** [DERIV + INFER]
- The bound holds for PR1 only: an estimator that must reproduce the same-seed cold render and therefore retraces every dirty stored suffix.
- It does not bound an estimator that *samples the residual* instead of *re-evaluating stored samples*. Xu et al. generate paths from the moved object and its ghost, so work is spent in proportion to residual energy, not to the count of stored paths that happen to touch the object; and the T0 image enters as a control variate with far more samples than the T1 budget (the report notes this asymmetry itself for PR2 in §11, which already contradicts a strict 1/ρ ceiling).
- So ρ measured in Stage 1 is the headroom of **stored-sample revalidation**, not of path-space reuse in general, and not of function-space representations at all. Using F2 to conclude about "any selective scheme" would over-generalise an exact identity into an economic statement (the brief's §6 rule).

**(b) "Births are covered at the fresh sampler's rate, no better" / F3 "expected by derivation".** [INFER]
- True for PR1/PR2 as specified (replay of a unidirectional sampler).
- Not true of the published class: Xu et al. start paths on the object in its new pose (covering new interreflection onto stationary receivers by construction of the sampler), ReSTIR BDPT and PSMS-ReSTIR accumulate caustic-type paths across frames. Whether these help on a sharp ER highlight via a moved glossy object is unmeasured; but "reuse gives no help on births" is a property of the report's arms, not a derived property of path-space methods.
- Consequence for the review: F3, if observed, would not show that the warm path-traced comparator is weak on GM1's headline phenomenon.

**(c) Citation corrections to the report.** Neural Control Variates author order is Müller, Rousselle, Novák, Keller. *World-Space Spatiotemporal Path Resampling* is Zhang & Wang, CGF 42(7), 2023. ReSTIR FG is EGSR 2024 (Kern, Brüll, Grosch). Günther & Grosch 2015 is CGF 34(4), the EGSR 2015 issue. ReSTIR PT Enhanced: PACM CGIT 9(1), Article 13, May 2026, DOI 10.1145/3804494; the "2 × 64 bytes per pixel" figure remains unverified by me as well.

---

## 5. Kinds of novelty, separated

| Kind | Track B status |
|---|---|
| Mechanism novelty | **None** (M1–M7 ALREADY KNOWN) [INFER, high confidence for M1–M4] |
| Application novelty | Rigid-edit re-rendering with stored / residual paths is published (Günther & Grosch 2015; Xu et al. 2024). "With a learned closure as BSDF" changes nothing in the estimator. **None** |
| Implementation engineering | GPU path store with bit-identical replay in Dr.Jit; exact dirty-set-driven reservoir confidence; ER lobe-index read. Real work, not a contribution |
| Measured utility | **Unmeasured.** Could exist only as a project-internal comparator and a headroom number for one fixture. [HIST] WL84: a 64-spp ER image costs about 10 ms of GPU kernel time in its scenes — at that scale no reuse scheme has room (the report says so) |

**What a credible contribution would require (listing is not a recommendation; nothing is authorised).** For Track B as specified, no experiment turns it into a mechanism contribution. A defensible *measured-utility* statement would need, at matched error against REF-ER T1 per receiver class: cold REF-ER; correlated path tracing without a store; a residual-path-integral sampler (Xu et al. class); a ReSTIR PT (Enhanced) temporal-reuse arm, unbiased variant; GPU kernel time itemised (discovery, retrace, shading, bookkeeping); and the baseline-role label "path-traced re-rendering" throughout (never Neural Full Recompute; ER reuse is Native Reuse of a material-frame asset).

---

## 6. Classification summary

| Mechanism | Classification | Closest prior |
|---|---|---|
| M1 path store + suffix replay (PR1) | ALREADY KNOWN | Jevans 1992; Brière & Poulin 1996; Bala et al. 1999; Havran et al. 2003; correlated path tracing |
| M2 geometric discovery, segment hierarchy | ALREADY KNOWN | Bala et al. 1999; Drettakis & Sillion 1997; Chen 1990; Dmitriev et al. 2002 |
| M3 difference re-render (PR2) | ALREADY KNOWN | Günther & Grosch 2015; Rousselle et al. 2016; **Xu et al. 2024** |
| M4 reservoir temporal reuse (PR3) | ALREADY KNOWN (dirty-set confidence: UNCERTAIN, engineering) | GRIS 2022; ReSTIR GI 2021; ReSTIR PT Enhanced 2026 |
| M5 births / specular repair | ALREADY KNOWN | ReSTIR BDPT 2025; PSMS-ReSTIR 2025; SMS 2020; Xu et al. 2024 |
| M6 ER lobe index for reconnection | ALREADY KNOWN | ReSTIR PT Enhanced (lobe-indexed paths) |
| M7 learned add-ons | ALREADY KNOWN; hybrid with a reusable learned state UNCERTAIN (application), NOT SUPPORTED as Track B content | ReSTIR PG 2025; NRC 2021; NCV 2020 |
| M8(i) warm comparator role | not a novelty item; comparator set incomplete without Xu et al. 2024 | — |
| M8(ii) headroom ρ | ALREADY KNOWN concept; measurement only | Bala 1999; Xu et al. 2024 |
| "1/ρ bounds any path-granular selective scheme" | NOT SUPPORTED | counter-class: residual sampling |
| "Reuse gives no help on births" (general form) | NOT SUPPORTED | counter-class: object-origin sampling, caustic reservoirs |

---

## 7. Limits of this search

- About 30 queries, 2 successful page extractions. GRIS, ReSTIR PT Enhanced, ReSTIR (2020) and the ReSTIR course could not be opened; their internals (stored seed, previous-frame target evaluation, reservoir size) are from background knowledge or snippets and marked.
- Xu et al. 2024 was read through an automated extraction of arXiv HTML v1; quoted factors and limitations need a direct reading before any reliance.
- Not searched in depth: production-renderer documentation (RTXDI / RTXGI validation passes), patents, non-English work, 2026 conference papers not yet indexed, MLT-based animation reuse.
- "No published instance found" statements (M7 hybrid; neural cache with geometric invalidation) are weak negatives from a bounded search, not evidence of novelty.
- U-marked entries (Séquin & Smyrl 1989; Murakami & Hirota; Kelemen 2002; Veach & Guibas 1997; Jakob & Marschner 2012; Kaplanyan 2014; Lehtinen 2013) were not verified this session.

## 8. Sources (external, this session)

- Xu et al. 2024: https://arxiv.org/abs/2406.16302 ; https://diglib.eg.org/handle/10.1111/cgf15152
- Günther & Grosch 2015: https://diglib.eg.org/handle/10.1111/cgf12677
- Rousselle, Jarosz, Novák 2016: https://cs.dartmouth.edu/wjarosz/publications/rousselle16image.html
- Manzi et al. 2016: https://www.cs.umd.edu/~zwicker/projectpages/TemporalGradientPathTracing-TOG16.html
- Bala, Dorsey, Teller 1999: https://diglib.eg.org/handle/10.2312/EGWR.EGWR99.031-044
- Brière & Poulin 1996: https://www.cs.princeton.edu/courses/archive/spr01/cs598b/papers/briere96.pdf
- Jevans 1992: https://graphicsinterface.org/wp-content/uploads/gi1992-21.pdf
- Drettakis & Sillion 1997: https://history.siggraph.org/?p=117251
- Chen 1990: https://www.scilit.net/publications/7ff6448daadb2da7a7dafae34964c35e ; George, Sillion, Greenberg 1990: https://www.scilit.net/publications/dbf203113df83d03d834dc6d1c56d192
- Dmitriev et al. 2002: https://resources.mpi-inf.mpg.de/anim/spt/
- Havran et al. 2003: https://diglib.eg.org/items/7eceaea3-e827-44dc-a4c6-4be4fec992a5/full
- Bekaert, Sbert, Halton 2002: https://diglib.eg.org/items/61a41a15-3500-4fc7-b61a-f5ef1b08ec26 ; Sbert, Castro, Halton 2004: https://dugi-doc.udg.edu/handle/10256/3071
- Tawara et al. 2004: https://pure.mpg.de/pubman/item/item_1329667_3/component/file_3583775/CGI2004_tawara.pdf ; Tole et al. 2002: https://history.siggraph.org/?p=118441
- ReSTIR GI: https://diglib.eg.org/handle/10.1111/cgf14378
- ReSTIR PT Enhanced: https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/
- Conditional ReSTIR: https://research.nvidia.com/labs/rtr/publication/kettunen2023conditional/
- Reservoir Splatting: https://research.nvidia.com/labs/rtr/publication/liu2025splatting
- ReSTIR BDPT: https://research.nvidia.com/labs/rtr/publication/hedstrom2025restir
- ReSTIR PG: https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg
- PSMS-ReSTIR (Hong et al. 2025): https://research.nvidia.com/labs/rtr/publication/hong2025partition ; SMS: https://rgl.epfl.ch/publications/Zeltner2020Specular
- ReSTIR FG: https://diglib.eg.org/handle/10.2312/sr20241155 ; Sawhney et al. 2024: https://research.nvidia.com/labs/rtr/publication/sawhney2024decorrelating/
- Boissé 2021: https://gpuopen.com/download/SA2021_WorldSpace_ReSTIR.pdf ; ReGIR: https://research.nvidia.com/labs/rtr/publication/boksansky2021rendering
- Zhang & Wang 2023: https://diglib.eg.org/handle/10.1111/cgf14974
- Dittebrandt et al. 2020: https://diglib.eg.org/handle/10.2312/sr20201135
- Neural Control Variates: https://arxiv.org/abs/2006.01524 ; Practical Path Guiding: https://diglib.eg.org/handle/10.1111/cgf13227
