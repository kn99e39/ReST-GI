# GM1 review — novelty kill search for Track C (LPTG, Lobe-Partitioned Transport Graph)

**Status:** design-review record (2026-11-10), revision `Selective_Recompute` @ `c7b0698`. Document-only. Nothing was implemented, run, rendered or measured, and no existing file was changed.

**Target:** `docs/plans/gm1_architecture_review/12_design_trackC_alternative_operator.md` (sealed). Of the review folder only that report and `00_PROBLEM_BRIEF.md` were read.

**Earlier kill-search records reused:** `docs/plans/wl77a/specialist_priorart.md`, `docs/plans/wl80a/prior_art_matrix.md`, `docs/plans/wl80b/c1_prior_art.md`. They concern a learned or explicit intrinsic medium K, so they overlap LPTG only in the classes "precomputed linear transfer", "incremental link update" and "per-asset neural transport". Entries taken from them are marked V-repo.

**Statement labels:** FACT = ESTABLISHED FACT; HIST = HISTORICAL EVIDENCE; DERIV = DERIVATION; INF = INFERENCE; HYP = HYPOTHESIS; SPEC = SPECULATION.

**Citation verification levels.**

| Mark | Meaning |
|---|---|
| V-s | Authors, title, venue and year confirmed by web search this session; content known at abstract or search-summary level only. No full text was read for any paper in this session. |
| V-d | Verified by the Track C author in the sealed report (its §13); not re-checked here. |
| V-repo | Verified in an earlier project kill-search record; not re-opened. |
| M | From memory. Identity believed correct, not verified this session. Mechanism details may be imprecise. |

**Classification labels:** ALREADY KNOWN / PLAUSIBLY DISTINCT / UNCERTAIN / NOT SUPPORTED. Absence of an identical published system is never treated as evidence of novelty.

---

## 0. Verdict

- INF (high confidence). **LPTG has no new mechanism.** Each of its eight central mechanisms (§2) maps onto published work, and for six of them the match is at the level of the mechanism itself, not only of the class.
- INF. The sealed report's own position ("every ingredient is known … a modernised two-pass directional radiosity with certified link reuse", its §13) is upheld. This search makes it **stronger**, because the report missed the closest single predecessors:
  - Granier and Drettakis 2001: a finite-element solve for the non-sharp part, traced samples for the glossy part, a line-space structure for incremental update after an object move, and selective re-emission of the traced samples. This is the LPTG architecture at 2001 scope.
  - Martín, Pueyo and Tost 2003: frame-to-frame coherent two-pass radiosity that recomputes only what dynamic objects affect.
  - Günther and Grosch 2015: consistent re-rendering after an object edit by correlated sampling of the difference.
  - Lin et al. 2022 (GRIS / ReSTIR PT): a roughness-thresholded hybrid of random-number replay through sharp vertices and reuse at rough vertices.
- The three items the report lists as "specific to ReST-GI" classify as follows:
  - (a) partition taken from the frozen learned closure's lobe list: **application novelty only** (ALREADY KNOWN as a mechanism).
  - (b) F ≡ S on a glossy operator with third-party and same-asset validity: **ALREADY KNOWN in substance**; an exact identity, not a contribution (problem brief §6).
  - (c) quality-matched lifecycle evidence: **measured utility**, currently NOT SUPPORTED because nothing is measured.
- The only possible contribution is empirical: a measured operating point under the GM1 controls. Its required evidence is listed in §5.

---

## 1. What is being tested

FACT (sealed report §1–§9). LPTG's central mechanisms, as I decompose them:

| ID | Mechanism |
|---|---|
| M1 | Split each material's lobe list by a fixed roughness threshold α\* into rough R and sharp S, f = f_R + f_S; label every path vertex R or S; this partitions path space exactly. |
| M2 | Hybrid carrier: R-events go through a finite directional field; S-events are never discretised and are traced at continuous directions. |
| M3 | Persistent, view-independent transport graph: cache points × K fixed material-frame directions, each with a stored ray link; per-material K × K matrix B compiled from the BRDF's rough lobes; unknown = indirect incident radiance; iterative sparse solve to a residual. The claim "stores an operator, not radiance". |
| M4 | Sharp continuation stored inside links as replayable ray chains (frozen random numbers); direct light by NEE at every chain vertex. |
| M5 | Edit invalidation by per-segment geometric certificates against the mover: four origin/hit classes, third-party intrusion test, same-asset links kept in the object frame under a no-overlap certificate. |
| M6 | Selective update: retain a chain up to its first invalid segment, regenerate the suffix with the stored random numbers, re-solve with a warm start. Claim F ≡ S to floating-point tolerance. |
| M7 | Readout: primary ray, exact direct light, exact rough-lobe evaluation against the interpolated field, fresh sharp chains per pixel that terminate into the field. |
| M8 | Exact path-sample store with random-number replay, kept as a control (upper bound on avoided work). |

One further element is not a transport mechanism but is claimed as project-specific:

| ID | Element |
|---|---|
| A1 | The R/S split and B are derived from the frozen learned ER closure (WL82–84), not from an analytic BRDF class. |

---

## 2. Per-mechanism kill search

### M1 — Roughness-thresholded lobe partition of path space

**Closest prior methods.**
- Heckbert 1990, "Adaptive radiosity textures for bidirectional ray tracing", SIGGRAPH (M): regular-expression path classification L(S|D)\*E by vertex type. The standard formal device for an exact path-space partition by scattering class.
- Wallace, Cohen, Greenberg 1987, "A two-pass solution to the rendering equation", SIGGRAPH (M; cited in the sealed report); Sillion and Puech 1989, "A general two-pass method integrating specular and diffuse reflection", SIGGRAPH (V-s, bibliographic only); Sillion, Arvo, Westin, Greenberg 1991 (V-d): split of the reflectance into a part carried by a finite basis and a specular part traced.
- Chen, Rushmeier, Miller, Turner 1991, "A progressive multi-pass method for global illumination", SIGGRAPH, pp. 165–174 (V-s): separate passes for low-frequency and high-frequency transport, with no restriction to ideal Lambertian or mirror surfaces.
- Jensen 1996, "Global illumination using photon maps", Eurographics Workshop on Rendering (M): photons stored at diffuse vertices, specular and glossy continuation traced.
- Křivánek, Gautron, Pattanaik, Bouatouch 2005, "Radiance caching for efficient global illumination computation", IEEE TVCG 11(5) (V-s): directional caching restricted to low-frequency glossy BRDFs. Křivánek's thesis lists "an automatic choice of surfaces on which radiance caching is used" (V-s). The exact criterion and the routing of high-frequency BRDFs to Monte Carlo sampling were **not verified** this session (M: an SH-fit error test per BRDF).
- Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman 2022, "Generalized resampled importance sampling: foundations of ReSTIR", ACM TOG 41(4), SIGGRAPH (V-s): hybrid shift that uses random replay through sharp vertices and reconnects at rough ones. A later ReSTIR PT paper from the same group states that the hybrid shift "relies on simple roughness and distance thresholds" (V-s, secondary; the 2022 paper's own definition not read).
- Kaplanyan and Dachsbacher 2013, "Path space regularization for holistic and robust light transport", Eurographics (M): per-vertex treatment that depends on lobe sharpness.
- Lumen, Wright, Narkowicz, Kelly, SIGGRAPH 2022 Advances course (V-s for the course; threshold from Epic's performance guide, V-s): pixels with roughness below a threshold (default 0.4 per the guide; a forum source says 0.3) trace dedicated reflection rays, rougher ones reuse the diffuse gather's radiance cache.
- Per-lobe ("lobe-selection") sampling of multi-lobe BSDFs in production path tracers, with per-lobe roughness tests for caching, regularisation or denoiser routing (M; common practice, no single citation verified).

**Shared mechanism.** Classify each scattering event by angular bandwidth; send broad events to a cached or finite-basis carrier and narrow events to tracing. With a multi-lobe BSDF the classification is per sampled lobe, so f = f_R + f_S holds trivially (DERIV: linearity of the BSDF in its lobe sum).

**Actual difference.** LPTG fixes the threshold once per material from the lobe parameters of a compiled closure, uses the same split in the solve, in the links and at readout, and states the partition as exact.

**Is the difference known?** Yes. A fixed roughness threshold applied per lobe is how GRIS hybrid shift and Lumen decide; exactness of the split is the Heckbert / two-pass identity. Using one threshold consistently is a design discipline.

**Evidence a credible contribution would need.** None can make M1 a mechanism contribution. Its *utility* needs the Stage 0 lobe census of the frozen closures and the "all lobes through the field" ablation of the sealed report §12.

**Classification: ALREADY KNOWN.**

### M2 — Finite directional field for rough transport plus traced sharp transport

**Closest prior methods.**
- Wallace et al. 1987; Sillion and Puech 1989; Sillion et al. 1991 (as above). Sillion 1991 solves directional-diffuse transport in a spherical-harmonic basis and traces ideal specular transport.
- Immel, Cohen, Greenberg 1986, "A radiosity method for non-diffuse environments", SIGGRAPH (V-repo, WL80B P6): discretised hemisphere of directions per patch. This is LPTG's "K fixed directions per cache point" at 1986 scope.
- Aupperle and Hanrahan 1993, "A hierarchical illumination algorithm for surfaces with glossy reflection", SIGGRAPH (V-s): discrete three-point transport with hierarchical links for glossy reflection.
- Christensen, Stollnitz, Salesin, DeRose 1996, "Global illumination of glossy environments using wavelets and importance", ACM TOG 15(1):37–71 (V-s, bibliographic); Christensen, Lischinski, Stollnitz, Salesin 1997, "Clustering for glossy global illumination", ACM TOG 16(1):3–33 (V-s): finite-element radiance solves for glossy scenes.
- Szirmay-Kalos 1999, "Stochastic iteration for non-diffuse global illumination", CGF 18(3) (V-s, bibliographic only): iteration of a randomly sampled transport operator for non-diffuse scenes.
- Granier, Drettakis, Walter 2000, "Fast global illumination including specular effects", Eurographics Workshop on Rendering (V-s, bibliographic); Granier and Drettakis 2001 (below): hierarchical radiosity with clustering plus particle tracing for the non-diffuse part.
- Gassenbauer, Křivánek, Bouatouch 2009, "Spatial directional radiance caching", CGF 28(4), EGSR, pp. 1189–1198 (V-s): incident radiance cached in space and direction, combined with BRDF importance sampling so that sharper glossy BRDFs are handled.
- Müller, Rousselle, Novák, Keller 2021 (V-repo; V-s): trace until the path's spread is large enough (heuristic of Bekaert et al.), then terminate into a cache.

**Shared mechanism.** i = e + C·b with the band-limited part of b = K·i in a finite angular basis and the sharp part carried by rays. The readout by final gather with traced specular transport is the second pass of the two-pass method.

**Actual difference.** (i) The finite field is restricted to lobes above α\* of a GGX mixture rather than to "diffuse / directional-diffuse"; (ii) sharp vertices occur *inside* field links (R → S → R as one link), not only between the field and the eye.

**Is the difference known?** (i) is a parameter choice. (ii) is known: the two-pass method's extended form factors already carry diffuse → specular → diffuse transfer through specular surfaces (Wallace 1987; Sillion and Puech 1989; M for the mechanism detail), and photon mapping and Granier 2000 trace specular chains between stored vertices.

**Evidence a credible contribution would need.** Not available as mechanism novelty. Utility needs H-rep of the sealed report: a prospective highlight gate met by the split and missed by the all-field ablation, on ER materials. HIST: WL58's finite-interface failure and WL46's conditional result motivate the split inside the project; they do not make it new outside it.

**Classification: ALREADY KNOWN.** The sealed report says so for the 1987 / 1991 pair ("this is the same decomposition"); the wider set above removes the remaining qualification "static scenes" (see M5, M6).

### M3 — Persistent view-independent transport graph; compiled per-material B; iterative solve; "operator, not radiance"

**Closest prior methods.**
- Hierarchical radiosity links: Hanrahan, Salzman, Aupperle 1991 (V-repo); glossy links in Aupperle and Hanrahan 1993 (V-s). The link set is a stored, reusable discretisation of the transport operator; radiosity is re-solved on it.
- Ray-sampled operators with a fixed sample set: Besuievsky and Sbert 1996, "The multi-frame lighting method: a Monte Carlo based solution for radiosity in dynamic environments", Eurographics Rendering Workshop, pp. 185–194 (V-s): one set of global lines intersected once and reused for all frames of an animation with moving objects. Stochastic / well-distributed-ray-set radiosity (Bekaert thesis 1999; Neumann 1995; M).
- Precomputed transport operators: Sloan, Kautz, Snyder 2002 (V-repo); Hašan, Pellacini, Bala 2006, direct-to-indirect transfer (V-repo); Kontkanen, Turquin, Holzschuch, Sillion 2006, "Wavelet radiance transport for interactive indirect lighting", EGSR, pp. 161–171 (V-s): 4D wavelet radiance basis and a precomputed global transport operator for diffuse and glossy scenes, static geometry; Lehtinen et al. 2008, meshless hierarchical basis for PRT (M).
- Per-material BRDF matrix in a fixed angular basis: Kautz, Sloan, Snyder 2002, "Fast, arbitrary BRDF shading for low-frequency lighting using spherical harmonics", Eurographics Workshop on Rendering (M); Lehtinen and Kautz 2003, "Matrix radiance transfer", I3D (M). B is this object in a point-sampled basis.
- Path graphs: Deng, Hašan, Carr, Xu, Marschner 2021 (V-d): a fixed set of traced paths reused through iterative propagation.
- Sparse probe / surfel operators with a local solve: Silvennoinen and Lehtinen 2017, ACM TOG 36(6), SIGGRAPH Asia (V-s); DDGI, Majercik et al. 2019, JCGT (M); surfel GI, Halén et al. 2021, SIGGRAPH Advances course (M); GI-1.0, Boissé et al., AMD technical report 2022 / arXiv 2310.19855 (V-s): two-level directional radiance caches with hash-grid world storage.

**Shared mechanism.** A fixed sample-based discretisation of C (links / lines / paths), a tabulated K in an angular basis, and a Jacobi / Gauss–Seidel / Southwell solve of the discrete rendering equation.

**Actual difference.** Unknown is incident radiance on fixed tangent-frame nodes at meshless cache points; links are single frozen rays with chains; nothing is precomputed offline.

**Is the difference known?** Yes as a class (Immel-type discrete ordinates on surface elements; stochastic radiosity with a frozen ray set; path graphs). "Operator rather than radiance" is the defining property of radiosity links and PRT, not a new distinction. The one-ray-per-node point sampling is a quadrature choice whose spatially correlated error the sealed report already lists (its §10.9).

**Evidence a credible contribution would need.** A measurement, not a search result: B quadrature error at grazing angles and field accuracy at the declared K against REF-ER (Stage 0 (ii), Stage 2).

**Classification: ALREADY KNOWN.**

### M4 — Stored, replayable sharp ray chains inside links

**Closest prior methods.**
- Lin et al. 2022 (V-s): random replay shift, i.e. re-tracing a path from stored random numbers in a changed context, combined with reconnection at rough vertices.
- Dmitriev, Brabec, Myszkowski, Seidel 2002 (V-d): photon groups regenerated from the same quasi-random indices after a scene change.
- Schied, Peters, Dachsbacher 2018, "Gradient estimation for real-time adaptive temporal filtering", Proc. ACM CGIT 1(2), HPG (V-s): previous-frame samples re-shaded "under the same random sequence" to isolate the change.
- Vicini, Speierer, Jakob 2021, "Path replay backpropagation", ACM TOG 40(4) (M): exact replay of a stored random walk.
- Günther and Grosch 2015 and Rousselle, Jarosz, Novák 2016 (below, M6): correlated re-rendering with common random numbers.
- Extended form factors through specular chains (M2) and photon-map specular continuation (M1).

**Shared mechanism.** A path is a deterministic function of (origin, random numbers, scene). Storing the random numbers and the hit record lets a path be reproduced, compared or partially regenerated after a change.

**Actual difference.** The chains hang off view-independent field links rather than off pixels or photons.

**Is the difference known?** INF: yes in substance; common random numbers and replay are generic, and Dmitriev 2002 applies them to a view-independent sample set.

**Evidence needed.** None for novelty. For utility: mean chain length L̄ and the share of segments in chains on the WL84 material family (unknown, sealed report §10.3).

**Classification: ALREADY KNOWN.**

### M5 — Edit invalidation by per-segment geometric certificates

**Closest prior methods.**
- Drettakis and Sillion 1997, "Interactive update of global illumination using a line-space hierarchy", SIGGRAPH (V-repo): finds the links whose line-space shafts an object move affects. The project topic document already ranks it as prior art, not a starting point (FACT, `AGENTS.md` decision 9).
- Bala, Dorsey, Teller 1999, "Interactive ray-traced scene editing using ray segment trees", Eurographics Workshop on Rendering (V-d): a hierarchy over stored ray segments used to find those an object edit affects.
- Brière and Poulin 1996, "Hierarchical view-dependent structures for interactive scene manipulation", SIGGRAPH (M); Jevans 1992, "Object space temporal coherence for ray tracing", Graphics Interface (M); Murakami and Hirota 1990, incremental ray tracing (M): stored ray trees with change detection by voxel or object dependency.
- Baum, Wallace, Cohen, Greenberg 1986, back-buffer extension of radiosity to dynamic environments (M); Chen 1990, incremental radiosity (M); George, Sillion, Greenberg 1990, radiosity redistribution (M); Forsyth, Yang, Teo 1994 and Shaw 1997, dynamic hierarchical radiosity (M): the "newly blocked / newly unblocked" link taxonomy for a moving object, with negative-energy shooting.
- Granier and Drettakis 2001, "Incremental updates for rapid glossy global illumination", CGF 20(3), Eurographics, pp. 268–277 (V-s): line-space hierarchy for the finite-element part plus "a dynamic spatial structure that limits how many particles are re-emitted after a local scene change" for the non-diffuse part.
- Same-asset transport kept in the object frame under rigid motion: PRT self-transfer (Sloan et al. 2002, V-repo); precomputed shadow fields (Zhou et al. 2005, V-repo); Iwasaki, Dobashi, Yoshimoto, Nishita 2007, "Precomputed radiance transfer for dynamic scenes taking into account light interreflection", EGSR, pp. 35–44 (V-s): rigid objects with diffuse and glossy BRDFs treated as secondary sources; Pan, Wang, Liu, Peng, Bao 2007, "Precomputed radiance transfer field for rendering interreflections in dynamic scenes", CGF 26(3), Eurographics, pp. 485–493 (V-s).
- 8DNA's validity condition "no other objects can intersect with the asset's convex hull" (V-repo, `docs/plans/wl77a/specialist_priorart.md` §3.1): the same third-party-intrusion condition on reuse of intrinsic same-asset transport, stated as a scope limit instead of a runtime certificate.
- ReSTIR GI, Ouyang, Liu, Kettunen, Pharr, Pantaleoni 2021, CGF 40(8), HPG (V-s bibliographic): temporal reuse of stored indirect samples; its sample-validation re-tracing was **not verified** this session (M).

**Shared mechanism.** A stored ray or link is valid after an object move iff the mover's old pose was not its hit and the mover's new pose does not intersect it; discovery by index for links incident on the mover and by a swept / bounding-volume test for the rest.

**Actual difference.** LPTG states the four cases as an exact table for single rays (not shafts), and attaches an O(1) bounds certificate to same-asset links.

**Is the difference known?** Yes. The single-ray case is simpler than the shaft case of 1997 and is the case Bala 1999 and the stored-ray-tree literature treat. The same-asset certificate is 8DNA's convex-hull condition made a test; shadow-field and PRT-field work handles the intruding third object by composition. INF: correct handling of the FNTO same-asset counterexample is a contract-compliance property, not a new mechanism.

**Evidence a credible contribution would need.** A discovery cost that is sub-linear in stored segments in closed scenes would be an *engineering* result; the sealed report claims none (its §3, "honest worst case is linear"). The invalidation controls of its §12 (null edit, off-path move) test correctness only.

**Classification: ALREADY KNOWN.**

### M6 — Selective regeneration with stored random numbers, warm-started re-solve, F ≡ S

**Closest prior methods.**
- Granier and Drettakis 2001 (V-s); Martín, Pueyo, Tost 2003, "Frame-to-frame coherent animation with two-pass radiosity", IEEE TVCG 9(1) (V-s, bibliographic plus search summary; abstract not read on the publisher page): a two-pass solution in which only the parts affected by dynamic objects are recomputed.
- Dmitriev et al. 2002 (V-d); Günther and Grosch 2015, "Consistent scene editing by progressive difference images", CGF 34(4), pp. 41–51 (V-s; issue 34(4) is the EGSR issue, INF, venue not confirmed): after object insertion or removal, material or light edits, the difference image is estimated stochastically and added to the converged render; built on stochastic progressive photon mapping with preliminary bidirectional results.
- Rousselle, Jarosz, Novák 2016, "Image-space control variates for rendering", ACM TOG 35(6), SIGGRAPH Asia (V-s): re-rendering after a scene edit with the earlier render as control variate and a low-sample correlated difference.
- Laine, Saransaari, Kontkanen, Lehtinen, Aila 2007, "Incremental instant radiosity for real-time indirect illumination", EGSR, pp. 277–286 (V-s): keep still-valid VPLs, replace invalid ones (light motion; a secondary source says static geometry is assumed).
- Havran, Damez, Myszkowski, Seidel 2003, "An efficient spatio-temporal architecture for animation rendering", EGSR, pp. 106–117 (V-s): path samples reused across frames with motion compensation, including bidirectional path tracing.
- Tole, Pellacini, Walter, Greenberg 2002, "Interactive global illumination in dynamic scenes", SIGGRAPH, pp. 537–546 (V-s): object-space shading cache with selective refresh where object motion is likely to matter.
- Chen 1990 / George et al. 1990 (M): incremental re-solve starting from the previous solution.
- Gautron, Bouatouch, Pattanaik 2007, temporal radiance caching (M; cited in the sealed report).
- ReSTIR PT (Lin et al. 2022, V-s) and world-space reservoir reuse (Boissé 2021, SIGGRAPH Asia technical communication, M): temporal reuse of path samples in dynamic scenes, with replay.

**Shared mechanism.** Keep the sample subset the edit did not touch, regenerate the rest with correlated (identical) random numbers, and restart the iterative solve from the old solution. Identity of the incremental and from-scratch results follows whenever the from-scratch build uses the same sample indices and the retained samples are certified unchanged (DERIV; the sealed report's own induction, its §3).

**Actual difference.** The identity is asserted for a directional (glossy) operator with replayed sharp chains, and is to be reported against a same-architecture full rebuild.

**Is the difference known?** INF: yes. Dmitriev 2002 and Günther–Grosch 2015 rest on the same determinism argument; Granier–Drettakis 2001 and Martín et al. 2003 do it for a glossy two-pass system. F ≡ S is an exact identity; under the brief's rule it establishes neither an efficient approximation nor a runtime advantage. HIST: the project already demonstrated the same kind of identity twice (WL56 same-sample hashes; WL59 same-checkpoint F/S), each time without a decisive cost advantage.

**Evidence a credible contribution would need.** H-econ of the sealed report, at predeclared quality levels, with the split "S cheaper than REF-ER but not than F" reported separately (its §12 table). None exists.

**Classification: ALREADY KNOWN** (mechanism and identity); **NOT SUPPORTED** (any economic claim built on it; the report makes none).

### M7 — Readout: exact rough-lobe evaluation against the field plus fresh sharp chains into the field

**Closest prior methods.** Second pass of Wallace 1987 / Sillion 1991; final gathering on a photon map (Jensen 1996, M) and on radiosity (Reichert 1992 thesis; Lischinski et al. 1993; M); radiance caching with exact BRDF evaluation at the shading point (Křivánek 2005, V-s; Gassenbauer 2009, V-s); NRC path termination into a cache (Müller 2021); Lumen rough / sharp reflection split (V-s).

**Shared mechanism / difference / known?** Identical at mechanism level. "No outgoing basis; evaluate the BRDF exactly against cached incident radiance" is the stated design of radiance caching.

**Classification: ALREADY KNOWN.**

### M8 — Exact path-sample store with replay, as a control

**Closest prior methods.** Stored ray trees (Séquin and Smyrl 1989, parameterized ray tracing, M; Brière and Poulin 1996, M; Bala 1999, V-d); path reuse across frames (Havran 2003, V-s; Bekaert, Sbert, Halton 2002, M); ReSTIR PT random replay (V-s); Günther–Grosch 2015 (V-s); A-SVGF (V-s).

**Assessment.** As a *control* it claims no novelty and needs none. Its role (an avoided-work upper bound) parallels the WL56 exact selective oracle (FACT, sealed report §0). INF: if it were promoted to a mechanism it would be a restatement of difference / correlated re-rendering.

**Classification: ALREADY KNOWN** (appropriate for a control).

### A1 — Partition and B compiled from the frozen learned ER closure

**Closest prior methods.** Neural or fitted BSDFs with explicit analytic lobes used inside classical transport: Zeltner et al. 2024, real-time neural appearance models, with microfacet importance-sampling lobes (V-repo); Fan et al. 2022, neural layered BRDFs (V-repo); lobe-wise treatment of fitted multi-lobe models (Lafortune et al. 1997, M; Ngan et al. 2005, M). BRDF matrices compiled by quadrature from an arbitrary BRDF: Kautz et al. 2002 (M).

**Shared mechanism.** Once a material is an explicit lobe sum, every lobe-wise classical technique applies unchanged; the lobes' provenance (learned compiler or hand fit) does not enter the transport algorithm.

**Actual difference.** The closure is ReST-GI's own qualified baseline; the R/S restriction is a new runtime entry point on it (sealed report §9).

**Is the difference known?** As a mechanism, yes. As a system, I found no published pipeline that feeds a learned compiled layered-material closure into a lobe-partitioned persistent GI structure; that is a weak negative from a bounded search and not evidence of novelty.

**Evidence needed.** The three ER identity tests of the sealed report §9 and the lobe census. A pass would show compatibility, not a contribution. The brief already states that local BSDF compression is not the contribution (FACT, `00_PROBLEM_BRIEF.md` §1).

**Classification: ALREADY KNOWN (mechanism); application novelty only, UNCERTAIN as a system.**

---

## 3. Position against the neural and reservoir families named in the review scope

LPTG learns nothing above ER (FACT, sealed report §2, §14.1). The families below therefore do not anticipate it as a *learned* mechanism; they occupy the niche LPTG leaves empty and are potential comparators. None was measured against LPTG or against the WL84 baseline.

| Family | Representative work (verification) | Relation to LPTG |
|---|---|---|
| Per-asset neural transport | RNA, 8DNA, NeuPreSS (V-repo); Neural PRT, Rainer et al. 2022 (V-repo) | Learned intrinsic or per-asset response. Under the baseline-role contract their T1 lifecycle is Native Reuse or Neural Full Recompute (FACT, `AGENTS.md`). LPTG shares only the object-frame reuse condition (M5). |
| Object-centric neural transfer composed at runtime | TransGI, Deng, Han, Fang 2025 (V-repo; no code, HIST WL67); NeLT, Zheng, Huo, Mo, Zhong, Wu, Hua, Wang, Bao, ACM TOG 2023, DOI 10.1145/3596491 (V-s): per-object neural light transfer composed explicitly, dynamic objects | The published form of "reusable learned per-object transfer + explicit composition" for dynamic scenes. LPTG's K/C mapping is the non-learned analogue. |
| Scene-level neural GI for dynamic scenes | Neural radiosity (V-repo); dynamic neural radiosity, Su et al. 2024 (V-repo); LightFormer, Ren et al., ACM TOG 43(4), 2024 (V-s); Diolatzis et al. 2022, active exploration for neural GI of variable scenes (M); generalizable light-transport embedding, arXiv 2510.18189 (V-repo) | Radiance or solution networks over a scene-parameter space; no validity certificate. Different lifecycle, not a mechanism match. |
| Online radiance caches | NRC (V-s); GI-1.0 (V-s); DDGI, surfel GI (M); Lumen (V-s) | Same rough / sharp split and termination-into-cache (M1, M2, M7); invalidation by ageing or online retraining instead of certificates. In practice the strongest competitors; UNMEASURED and not among the designated controls (the sealed report says this for NRC). |
| Reservoir reuse | ReSTIR GI (V-s); GRIS / ReSTIR PT (V-s); world-space ReSTIR (M) | Covers M1 (roughness-thresholded hybrid), M4 (replay) and temporal sample reuse under dynamics. Differs in being unbiased / consistent per frame and view-dependent (screen-space) or hashed world-space, with no persistent solved field. |
| Many-light and matrix methods | Lightcuts, Walter et al. 2005 (M); VSLs, Hašan et al. 2009 (M); matrix row-column sampling, Hašan et al. 2007 (M) | The sealed report rejects these for sharp receivers; no overlap beyond stored light paths. |
| 2007–2013 dynamic GI | Implicit visibility and antiradiance, Dachsbacher, Stamminger, Drettakis, Durand 2007, ACM TOG 26(3) (V-s; directional antiradiance stored per element; glossy support not confirmed this session); imperfect shadow maps, Ritschel et al. 2008 (M); voxel cone tracing, Crassin et al. 2011 (M); modular radiance transfer (V-repo) | Link-based or cached dynamic interreflection without per-segment certificates. FACT: the topic document names this period the priority historical reference (`AGENTS.md` decision 8). LPTG's link structure is closest to the antiradiance / hierarchical-link line, with explicit visibility restored. |

INF. Relative to these families LPTG's distinguishing property is a *certificate* (exact validity at its own sample resolution) in place of heuristic or learned adaptation. Certified incremental update is itself the 1990–2003 incremental-radiosity and stored-ray line (M5, M6), so the property is known; whether it buys quality-matched time against a path tracer or an online cache is unmeasured.

---

## 4. Summary table

| ID | Mechanism | Closest prior work | Classification | Kind of any residual claim |
|---|---|---|---|---|
| M1 | Roughness-thresholded lobe partition | Heckbert 1990; two-pass 1987–1991; GRIS hybrid shift 2022; Lumen 2022 | ALREADY KNOWN | none |
| M2 | Finite directional field + traced sharp | Sillion 1991; Immel 1986; Aupperle–Hanrahan 1993; Christensen 1997; Granier 2000 | ALREADY KNOWN | measured utility (H-rep) |
| M3 | Persistent transport graph, compiled B, solve | glossy HR links; Besuievsky–Sbert 1996; Kontkanen 2006; Kautz 2002; path graphs 2021 | ALREADY KNOWN | implementation engineering |
| M4 | Replayable sharp chains | GRIS random replay; Dmitriev 2002; A-SVGF 2018 | ALREADY KNOWN | none |
| M5 | Segment validity certificates | Drettakis–Sillion 1997; Bala 1999; Granier–Drettakis 2001; 8DNA hull condition | ALREADY KNOWN | implementation engineering (discovery cost) |
| M6 | Selective regeneration, warm start, F ≡ S | Granier–Drettakis 2001; Martín 2003; Dmitriev 2002; Günther–Grosch 2015 | ALREADY KNOWN; economics NOT SUPPORTED | measured utility (H-edit, H-econ) |
| M7 | Readout by exact rough eval + sharp chains | two-pass second pass; radiance caching; NRC | ALREADY KNOWN | none |
| M8 | Exact path-replay store (control) | stored ray trees; correlated re-rendering | ALREADY KNOWN | none claimed |
| A1 | Split and B from frozen learned ER | lobe-wise use of fitted / neural BSDFs | ALREADY KNOWN (mechanism); UNCERTAIN (system) | application novelty |
| — | Whole combination on ER, GM1 workload | Granier–Drettakis 2001 + GRIS-style replay + radiance-cache readout | UNCERTAIN, combination only | application + measured utility |

No element reaches PLAUSIBLY DISTINCT.

---

## 5. What a credible contribution would require

Listed as evidence requirements only. None is authorised, and listing them is not a recommendation.

1. **Mechanism novelty:** not obtainable from LPTG as specified (INF). It would need an element absent from §2, for example a validity certificate that is sub-linear in closed scenes, or a learned global component with a stated K/C ownership; the sealed design contains neither.
2. **Application novelty (A1):** the ER identity tests and lobe census (sealed report Stage 0). A pass shows the frozen closure can drive a classical hybrid GI structure. It does not address the brief's requirement that the contribution be the global transport mechanism.
3. **Measured utility:** all of the following, with thresholds fixed before each stage.
   - H-rep: split passes and the all-field ablation fails a prospective highlight gate against REF-ER.
   - H-edit: S = F to tolerance; stale fails on stationary receivers; invalidation controls behave.
   - H-econ: quality-matched GPU-kernel time of S below REF-ER re-rendering at predeclared quality levels, reported separately from F.
   - Baseline roles kept apart (FACT, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`): REF-ER re-rendering is the path-tracing axis only; same-architecture F is the internal comparator; Neural Full Recompute and Native Reuse stay UNMEASURED and cannot be claimed against.
4. **Position against the practical competitors:** an online cache (NRC-class) or reservoir reuse (ReSTIR PT-class) under the same edit is the comparison a reviewer outside the project would ask for (INF). Neither is a designated control and neither is available in the project (not checked).
5. **Contract question, prior to any evidence:** LPTG is learning-free above ER. HIST: WL80B judged a learning-free operator "PHYSICAL BASELINE ONLY" and inadmissible as a learned K under FNTO v0.1 (`docs/plans/wl80b/c1_prior_art.md` F0, F4). Whether an explicit unlearned C with learned local K satisfies the research contract is a human decision; the sealed report already reports it (its §14.1) and this search adds no ground for or against.

---

## 6. Corrections and additions to the sealed report's prior-art section

All INF unless marked.

1. **Missed closest predecessor.** Granier and Drettakis 2001 combines the finite-element / traced split, glossy transport, a line-space structure for object moves and selective re-emission of traced samples. The report's table cites Drettakis–Sillion 1997 as "diffuse" and Dmitriev 2002 as "no directional operator"; the 2001 paper closes both gaps in one system (V-s; details beyond the abstract not read).
2. **"Static scenes" is not a valid difference for the two-pass lineage.** Martín, Pueyo, Tost 2003 is a frame-to-frame coherent two-pass method for dynamic objects (V-s).
3. **Replay and difference rendering have a direct modern lineage** not cited: Günther–Grosch 2015, Rousselle et al. 2016, Schied et al. 2018, Lin et al. 2022.
4. **The roughness threshold is a published decision rule** in GRIS hybrid shift and in Lumen, not only an "ideal BRDF class" split as the report's first table row implies.
5. **Glossy finite-element operators** (Immel 1986; Aupperle–Hanrahan 1993; Christensen et al. 1996 / 1997; Kontkanen et al. 2006) are absent from the report; they are the prior art for B and the K-node field.
6. **Rigid same-asset reuse with interreflection** has PRT-era precedents (Iwasaki et al. 2007; Pan et al. 2007) and 8DNA's hull condition.
7. The report's statement that Bala et al. 1999 "reports 3–4× over its base ray tracer" was not checked here.

None of these changes the report's conclusion; they remove qualifiers from it.

---

## 7. Search limits and uncertainty

- Tools: web search only, 24 queries; no publisher full text was opened. Every "V-s" mechanism statement is abstract- or summary-level.
- Explicitly unverified details: Křivánek 2005's automatic BRDF criterion; the roughness threshold in the 2022 GRIS paper itself (confirmed only through a later paper's description); ReSTIR GI's sample-validation step; glossy support in antiradiance; the content of Sillion–Puech 1989, Szirmay-Kalos 1999, Christensen 1996 and Granier 2000 beyond title and venue; the EGSR attribution of Günther–Grosch 2015; the exact title of the 2026 ReSTIR PT follow-up (NVIDIA Research page seen, title not read).
- All "M" entries are from memory and should be re-verified before external use.
- Possible blind spots: 2024–2026 real-time GI papers on certified or validity-tracked caches; production renderer documentation; patents; the 1990s parallel-rendering literature on stored ray trees.
- The "no published pipeline found" statement in A1 and the "combination" row of §4 are weak negatives bounded by this search.
- Repository evidence was taken from the sealed report, the problem brief, `AGENTS.md` and the three earlier kill-search records; worklogs were not re-opened and no number was re-computed.

## 8. Sources (web, this session)

- Granier and Drettakis 2001: https://diglib.eg.org/handle/10.2312/8873 ; https://scilit.net/publications/1b717524f4eaf7a2ed14293591708bb7
- Granier, Drettakis, Walter 2000: https://rd.springer.com/chapter/10.1007/978-3-7091-6303-0_5
- Martín, Pueyo, Tost 2003: https://manuscript.isc.ac/Inventory/49/1131263.htm
- Günther and Grosch 2015: https://diglib.eg.org/handle/10.1111/cgf12677 ; https://cris.fau.de/publications/242946933
- Křivánek et al. 2005: https://dcgi.fel.cvut.cz/en/publications/2005/krivanek-tvcg-rceg/ ; https://cgg.mff.cuni.cz/~jaroslav/projects/rcaching/
- Gassenbauer et al. 2009: https://diglib.eg.org/items/187965be-76a6-4e07-a97a-6942a44c1a74
- Lin et al. 2022 (GRIS): https://research.nvidia.com/labs/rtr/publication/lin2022generalized ; https://graphics.cs.utah.edu/research/projects/gris ; follow-up: https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced.pdf
- Christensen et al. 1997: https://dblp.uni-trier.de/rec/journals/tog/ChristensenLSS97.html ; https://grail.cs.washington.edu/projects/glob-illum
- Aupperle and Hanrahan 1993: https://graphics.stanford.edu/papers/glossy
- Tole et al. 2002: https://history.siggraph.org/?p=118441
- Besuievsky and Sbert 1996: https://rd.springer.com/chapter/10.1007/978-3-7091-7484-5_19 ; https://dblp.dagstuhl.de/rec/conf/rt/BesuievskyS96.html
- Havran et al. 2003: https://dx.doi.org/10.2312/EGWR/EGWR03/106-117
- Rousselle et al. 2016: https://cs.dartmouth.edu/wjarosz/publications/rousselle16image.html
- Laine et al. 2007: https://research.nvidia.com/publication/2007-06_incremental-instant-radiosity-real-time-indirect-illumination
- Dachsbacher et al. 2007: https://dblp.dagstuhl.de/rec/journals/tog/DachsbacherSDD07.html ; https://history.siggraph.org/?p=115370
- Boissé et al., GI-1.0: https://ar5iv.labs.arxiv.org/html/2310.19855 ; https://gpuopen.com/download/GPUOpen2022_GI1_0.pdf
- Kontkanen et al. 2006: https://diglib.eg.org/handle/10.2312/EGWR.EGSR06.161-171 ; https://hal.archives-ouvertes.fr/inria-00379398
- Iwasaki et al. 2007: https://diglib.eg.org/handle/10.2312/EGWR.EGSR07.035-044
- Pan et al. 2007: https://diglib.eg.org/handle/10.2312/CGF.v26i3pp485-493
- Lumen: https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf ; https://dev.epicgames.com/documentation/unreal-engine/lumen-performance-guide-for-unreal-engine
- Müller et al. 2021 (NRC): https://arxiv.org/pdf/2106.12372
- Ouyang et al. 2021 (ReSTIR GI): https://diglib.eg.org/handle/10.1111/cgf14378
- Schied et al. 2018: https://cg.ivd.kit.edu/atf.php
- NeLT: https://history.siggraph.org/?p=164203 ; LightFormer: https://wylighting.github.io/lightformer
- Chen et al. 1991: https://dblp.org/rec/conf/siggraph/ChenRMT91
- Silvennoinen and Lehtinen 2017: https://arisilvennoinen.github.io/Projects/RTGI/index.html
- Szirmay-Kalos 1999: DOI 10.1111/1467-8659.00344 (bibliographic record only)

**Repository files read (read-only):** `docs/plans/gm1_architecture_review/00_PROBLEM_BRIEF.md`, `docs/plans/gm1_architecture_review/12_design_trackC_alternative_operator.md`, `docs/plans/wl77a/specialist_priorart.md`, `docs/plans/wl80a/prior_art_matrix.md` (first 289 lines), `docs/plans/wl80b/c1_prior_art.md`, `AGENTS.md` (as loaded).
