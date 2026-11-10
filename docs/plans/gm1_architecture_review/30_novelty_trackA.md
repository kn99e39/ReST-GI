# GM1 review — novelty kill search — Track A (Directional Delta Transport)

**Status:** design-only review record (2026-11-10). Revision `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. This is the only file written.
**Target:** `docs/plans/gm1_architecture_review/10_design_trackA_directional_delta_transport.md` (sealed; not modified). Tracks B and C were not opened.
**Repository inputs read:** `00_PROBLEM_BRIEF.md`; `docs/Architecture.md` (full); `docs/RESEARCH_CENTRIC_TOPIC.md` (full); `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` (full); `docs/RESEARCH_ROADMAP.md` (milestone table, l.5–31); `AGENTS.md`; earlier kill-search records `docs/plans/wl77a/specialist_priorart.md`, `docs/plans/wl80a/prior_art_matrix.md` (l.1–289), `docs/plans/wl80b/c1_prior_art.md`. Worklogs 46, 56, 58, 59, 70, 82–84 are used only through the `AGENTS.md` / canonical summaries and are cited as such.

**Labels.** FACT = ESTABLISHED FACT; HIST = HISTORICAL EVIDENCE; DERIV = DERIVATION; INF = INFERENCE; HYP = HYPOTHESIS; SPEC = SPECULATION.
**Citation verification.** **V** = abstract or page text read this session. **V-s** = search-result summary or bibliographic record only. **V-repo** = taken from an earlier project record. **M** = from memory, not verified this session; no classification rests on an M item alone.
**Classes.** ALREADY KNOWN / PLAUSIBLY DISTINCT / UNCERTAIN / NOT SUPPORTED. Absence of an exact match is never counted as novelty.

## 0. Result

- **No central mechanism of Track A is new** (INF, high confidence). Every mechanism in the full DDT and in the narrowed DDT-D maps to published work from 1990–2024. The report's own novelty position (§13, last paragraph) is confirmed and tightened.
- **Two close predecessors are missing from the report's §13** and weaken its one remaining candidate ("directionally complete persistent state + discovery-then-overwrite"):
  - Bala, Dorsey and Teller 1999: a view-independent 4D directional radiance state, geometric discovery of the entries an edit invalidates, and recomputation of exactly those entries.
  - Granier and Drettakis 2001: incremental update of *glossy* global illumination after object motion.
- **What remains** is a combination at a new operating point: sharp-glossy bandwidth, multi-bounce, GPU, frozen learned local closure. Its class is UNCERTAIN as application / engineering / measured utility. Its mechanism novelty is none. Under `docs/RESEARCH_CENTRIC_TOPIC.md` l.46 ("novelty cannot rest on … selective updating … alone") it is not a contribution without measured quality-matched economics (FACT for the rule; INF for the mapping).

## 1. Mechanisms tested

Decomposition of the sealed report (FACT: section references are to that report).

| ID | Mechanism | Report § |
|---|---|---|
| M1 | Delta identity `(I − K C1) db = K(de + dC b0)`; directional incremental radiosity / defect correction | §1 |
| M2 | Persistent, directionally complete outgoing field `b0(x, ω)` on all surfaces, including never-consumed directions | §2 |
| M3 | Geometric discovery of the first-order changed-ray set `D` from the mover's old and new pose | §3 |
| M4 | Signed first-order source from two stored-state lookups per changed ray; explicit connection toward the changed set | §1, §4, §9 |
| M5 | Higher-order invalidation by propagating source magnitude through `K C1` to a radiometric tolerance; Southwell ordering | §1, §6 |
| M6 | DDT-D: delta used only to discover and rank the dirty set; values obtained by overwriting through selective regather | §1, §5 |
| M7 | Band-limited stored state plus path continuation through lobes narrower than the stored resolution | §2 |
| M8 | Ghost-residue / signed-cancellation argument against additive deltas | §5, §6 |
| M9 | Cost claim: work scales with the radiometric measure of `D`; advantage for view-independent state | §1, §7, §11 |
| M10 | Integration boundary with frozen ER (read-only lobe parameters steer lookup vs continuation) | §9 |

## 2. Per-mechanism kill search

### M1 — delta identity

- **Closest prior.**
  - S. E. Chen, "Incremental radiosity: an extension of progressive radiosity to an interactive image synthesis system", SIGGRAPH 1990, pp. 135–144 (V-s).
  - D. W. George, F. X. Sillion, D. P. Greenberg, "Radiosity redistribution for dynamic environments", IEEE CG&A 10(4), 1990, pp. 26–34 (V-s; indexed keywords include "negative energy").
  - B. Xu, T.-M. Li, I. Georgiev, T. Hedstrom, R. Ramamoorthi, "Residual Path Integrals for Re-rendering", CGF 43(4) (EGSR 2024), DOI 10.1111/cgf.15152 (V, arXiv HTML read).
  - J. Arvo, K. Torrance, B. Smits, "A framework for the analysis of error in global illumination algorithms", SIGGRAPH 1994 (M).
- **Shared mechanism.** The T1 solution is the T0 solution plus the response to the T1 residual of the T0 solution. Chen and George et al. shoot positive and negative unshot energy; Xu et al. write the same difference as a path integral over paths touching the moved or "ghost" (old-pose) surfaces.
- **Actual difference.** Operator form with directional `b` and a pointwise glossy K.
- **Is the difference known?** Yes. Glossy incremental updates: X. Granier, G. Drettakis, "Incremental updates for rapid glossy global illumination", CGF 20(3) (Eurographics 2001), pp. 268–277 (V-s). Path-space form for arbitrary BSDFs: Xu et al. 2024.
- **Class: ALREADY KNOWN.** The report says so itself (DERIV, §1; FACT that it concedes this).

### M2 — persistent directionally complete surface radiance state

- **Closest prior (classical).**
  - D. Immel, M. Cohen, D. Greenberg, "A radiosity method for non-diffuse environments", SIGGRAPH 1986 (V-s): a discretised hemisphere of outgoing directions per patch.
  - L. Aupperle, P. Hanrahan, "A hierarchical illumination algorithm for surfaces with glossy reflection", SIGGRAPH 1993 (V-s): three-point transport.
  - P. Christensen, D. Lischinski, E. Stollnitz, D. Salesin, "Clustering for glossy global illumination", ACM TOG 16(1), 1997, pp. 3–33 (V-s).
  - K. Bala, J. Dorsey, S. Teller, radiance interpolants: 4D line-space radiance per surface in "linetrees", with a user error bound, reused across viewpoints (V, Cornell project page). They are built lazily, so they are *not* complete in unvisited directions.
  - J. Křivánek, P. Gautron, S. Pattanaik, K. Bouatouch, "Radiance caching for efficient global illumination computation", IEEE TVCG 11(5), 2005 (V-s; low-frequency BRDFs). P. Gassenbauer, J. Křivánek, K. Bouatouch, "Spatial directional radiance caching", CGF 28(4) (EGSR 2009) (V-s; high-frequency BRDFs by combining the cache with BRDF importance sampling).
- **Closest prior (recent, directional and/or learned).**
  - T. Müller, F. Rousselle, J. Novák, A. Keller, "Real-time neural radiance caching for path tracing", SIGGRAPH 2021 (V-repo): a learned `L_o(x, ω)`.
  - S. Hadadan, S. Chen, M. Zwicker, "Neural Radiosity", SIGGRAPH Asia 2021 (V-repo): a network for `L_o(x, ω)` over all surfaces and directions, trained on the rendering-equation residual. This is directionally complete by construction, including directions no camera consumes (INF from the method description in the earlier project record).
  - W. Tatzgern, A. Weinrauch, P. Stadlbauer, J. H. Mueller, M. Winter, M. Steinberger, "Radiance caching with on-surface caches for real-time global illumination", PACMCGIT (HPG 2024), DOI 10.1145/3675382 (V-s; venue wording not confirmed on the DOI page): on-surface caches storing outgoing radiance "for every direction".
  - J. Ren et al., "Neural cone radiosity for interactive global illumination with glossy materials", arXiv 2509.07522, 2025 (V, abstract; no venue listed). J. Ren et al., "OctaOctree neural radiosity for real-time glossy material rendering", arXiv 2606.08469, 2026 (V, abstract; no venue listed): spatial octree with octahedral directional maps. D. Klepikov, V. Frolov, "Neural caching of prefiltered radiance for specular lighting", arXiv 2610.11702, 2026 (V, abstract). None of the three abstracts mentions geometry edits or invalidation.
- **Shared mechanism.** A stored 4D outgoing-radiance field on surfaces, tabulated, hierarchical or learned.
- **Actual difference.** The report *requires* completeness at sharp-glossy bandwidth in never-consumed directions; it proposes no representation that achieves it (FACT: §2 "open (tabulated or learned)", §14).
- **Is the difference known?** The requirement is known (Neural Radiosity's uniform (x, ω) residual sampling; Immel's full hemisphere). The bandwidth problem is an active, unsolved research topic with 2025–2026 entries. The report's angular-count estimate is a derivation, not a mechanism.
- **Class: ALREADY KNOWN** as a state class. **NOT SUPPORTED** as a contribution, because no representation is proposed. The "contract tension" for a scene-fitted learned `b0` (report §2) stands and is a human decision (FACT, `docs/Architecture.md` §4).

### M3 — geometric discovery of the changed-ray set

- **Closest prior.**
  - Xu et al. 2024 (V): "We always start the dynamic path by sampling the initial vertex on a dynamic/ghost surface." The ghost surface is the report's old pose `O0`.
  - K. Bala, J. Dorsey, S. Teller, "Interactive ray-traced scene editing using ray segment trees", Eurographics Workshop on Rendering 1999, pp. 31–44 (V-s abstract; V project page): a box in ray-segment space is a shaft in world space; the trees record which 4D radiance interpolants depend on which world-space region; an edit walks the trees to find affected interpolants (about 0.1 s reported).
  - G. Drettakis, F. Sillion, "Interactive update of global illumination using a line-space hierarchy", SIGGRAPH 1997 (V-repo, `c1_prior_art.md` P10).
  - E. Shaw, "Hierarchical radiosity for dynamic environments", CGF 16(2), 1997, pp. 107–118 (V-s): shadow links, ghost links, motion volumes that cull interactions not needing an update.
  - D. Forsyth, C. Yang, K. Teo, "Efficient radiosity in dynamic environments", EGWR 1994 (V-s, secondary description).
  - N. Brière, P. Poulin, "Hierarchical view-dependent structures for interactive scene manipulation", SIGGRAPH 1996 (V-s; fixed view).
  - K. Dmitriev, S. Brabec, K. Myszkowski, H.-P. Seidel, "Interactive global illumination using selective photon tracing", EGWR 2002 (V-s): pilot photons detect regions needing an update.
- **Shared mechanism.** The set of light-path segments intersecting the union of the old and new pose is enumerated geometrically (shafts, trees) or by sampling from the mover.
- **Actual difference.** None at mechanism level. The report samples points and directions on `O0` and `O1` and traces outward, which is Xu et al.'s initial-vertex strategy.
- **Class: ALREADY KNOWN.** The report's limit "exact only at sample resolution" is also internal history (HIST, WL70 D6 per `AGENTS.md`).

### M4 — signed source from two state lookups; connection toward the changed set

- **Closest prior.** George et al. 1990 and Chen 1990 (old contribution removed, new added, from stored radiosities); Xu et al. 2024 (integrand `f2 − f1` on dynamic paths, six sampling techniques combined by MIS, sub-paths traced from the moving object toward emitter and sensor; V); Dmitriev et al. 2002 (corrective photons; the negative-energy detail is **M**, not confirmed this session); F. Rousselle, W. Jarosz, J. Novák, "Image-space control variates for rendering", SIGGRAPH Asia 2016, and M. Manzi et al., "Temporal gradient-domain path tracing", SIGGRAPH Asia 2016 (both V-s, as cited by Xu et al.).
- **Shared mechanism.** The residual source is non-zero only on changed segments and is the difference of what the segment carried before and after.
- **Actual difference.** Recursion at T0 is replaced by a lookup into a stored directional field. Xu et al. state they are "given the image at the previous frame … without auxiliary information or any cache" (V).
- **Is the difference known?** Each half is. Terminating a path into a stored radiance field is standard (M7). Incremental radiosity already reads stored `B`. I found no paper that pairs residual-path sampling with a persistent directional cache. That is a weak negative from a bounded search and it describes a combination.
- **Class: ALREADY KNOWN** (components); the pairing is **UNCERTAIN**, combination only.

### M5 — magnitude propagation and ordering

- **Closest prior.** Chen 1990 (the positive and negative unshot energy is propagated by progressive shooting; V-s). M. Cohen, S. E. Chen, J. Wallace, D. Greenberg, "A progressive refinement approach to fast radiosity image generation", SIGGRAPH 1988 (M): shoot the largest unshot energy first. S. Gortler, M. Cohen, P. Slusallek, "Radiosity and relaxation methods", IEEE CG&A 1994 (M): the Southwell interpretation. Dmitriev et al. 2002 (V-s): energy- and perception-based criteria to order progressive updates. Granier and Drettakis 2001 (V-s): incremental particle trace for non-diffuse transport.
- **Shared mechanism.** Residual energy is propagated in descending magnitude until below a threshold; the propagation is the transport solve.
- **Actual difference.** The propagated quantity is used for marking rather than for values (that is M6).
- **Class: ALREADY KNOWN.** The report's own statement that an a-priori sparsity certificate would need hierarchical-radiosity link bounds (§7) names the prior art for the missing piece.

### M6 — DDT-D: discover and rank by delta, overwrite by regather

- **Closest prior.**
  - Bala, Dorsey, Teller 1999 (V-s / V): affected directional interpolants are found geometrically and then updated; "the viewpoint can be changed freely without invalidating them"; reported 3–4× over the base ray tracer. This is discovery-then-recompute on a view-independent directional radiance state. Whether updates patch or recompute is not explicit on the pages read; whether diffuse or glossy multi-bounce transport is covered is **not verified** (the pages describe reflections, refractions and shadows).
  - Drettakis and Sillion 1997 (V-repo): find affected links, re-solve only those.
  - P. Gautron, K. Bouatouch, S. Pattanaik, "Temporal radiance caching", IEEE TVCG 13(5), 2007 (V-s): for *glossy* global illumination in animated scenes, refresh indirect lighting only where incoming radiance changes significantly.
  - C. Schied, C. Peters, C. Dachsbacher, "Gradient estimation for real-time adaptive temporal filtering", PACMCGIT (HPG 2018) (V-s): a temporal gradient from re-evaluating shading with the same random sequence detects change; stale history is then discarded rather than corrected. This is exactly "delta for detection, not for values".
  - P. Tole, F. Pellacini, B. Walter, D. Greenberg, "Interactive global illumination in dynamic scenes", SIGGRAPH 2002 (V-s): an object-space cache refined first where object motion is likely to affect it.
  - T. Tawara, K. Myszkowski, H.-P. Seidel, "Exploiting temporal coherence in final gathering for dynamic scenes", CGI 2004 (V-s; venue from the PDF name only). M. Smyk, S. Kinuwaki, R. Ďurikovič, K. Myszkowski, "Temporally coherent irradiance caching for high quality animation rendering", CGF 24(3) (EG 2005) (V-s).
  - S. Laine, H. Saransaari, J. Kontkanen, J. Lehtinen, T. Aila, "Incremental instant radiosity for real-time indirect illumination", EGSR 2007 (V-s): keep valid virtual lights, recompute only invalid ones.
  - ReSTIR-family reservoir validation: Y. Ouyang, S. Liu, M. Kettunen, M. Pharr, J. Pantaleoni, "ReSTIR GI", CGF 40(8) (HPG 2021) (V-s; the sample-validation detail is **M**). D. Lin et al., "Generalized resampled importance sampling", SIGGRAPH 2022 (V-s). M. Kettunen et al., "Conditional resampled importance sampling and ReSTIR", SIGGRAPH Asia 2023 (V-s; suffix reuse with a final gather). G. Boissé, "World-space spatiotemporal reservoir reuse for ray-traced global illumination", SIGGRAPH Asia 2021 Technical Communications (V-s, from a citing listing). These reuse *samples*, not a radiance field, and have no dirty-set propagation.
  - Project-internal (FACT / HIST): the Selective_Recompute definition in `AGENTS.md` (recompute the complete state of every invalidated unit, keep valid units); WL56 / WL59 same-rule F/S; WL70 dependency-contract reuse of first-order sun state. The report concedes this overlap in §14.
- **Shared mechanism.** A change signal selects which cached entries to recompute; valid entries are kept; values are never patched additively.
- **Actual difference.** The change signal is the propagated signed first-order source, evaluated on a directional surface state.
- **Is the difference known?** Yes in each part: geometric plus radiometric dirty marking (M3, M5), directional state (M2), overwrite (above).
- **Class: ALREADY KNOWN.**

### M7 — band-limited state plus path continuation through narrow lobes

- **Closest prior.** Müller et al. 2021: paths are terminated into the cache by an area-spread heuristic after Bekaert et al. 2003 (V-s; confirmed via the search summary of the NRC text and of Kandlbinder et al. 2024). Gassenbauer et al. 2009 (V-s): cache plus BRDF importance sampling where the BRDF is too sharp for the cache. H. W. Jensen, photon mapping with specular continuation and final gathering, 1996 (M). Production roughness-threshold switches between traced reflections and cached radiance (Lumen: D. Wright, K. Narkowicz, P. Kelly, SIGGRAPH 2022 Advances course; the threshold detail is **M**).
- **Class: ALREADY KNOWN.** The report notes that the zero-storage limit is on-demand T0 tracing, i.e. Xu et al. 2024 (INF, confirmed by M4).

### M8 — ghost residue and signed cancellation

- **Closest prior.** C. Dachsbacher, M. Stamminger, G. Drettakis, F. Durand, "Implicit visibility and antiradiance for interactive global illumination", ACM TOG 26(3) (SIGGRAPH 2007) (V-s): directional signed radiance replaces visibility. A follow-up (Mückl and Dachsbacher, WSCG 2011; V-s) names the cost of positive/negative interplay. Xu et al. 2024 (V): "a large motion will break the possibility of any path mapping"; the residual reduces to a finite difference of the whole path space if all paths are affected.
- **Class: ALREADY KNOWN** as a phenomenon. The report's formal statement (consistency between the removal term and the baked value) is a correct derivation and a design argument, not a mechanism.

### M9 — cost claim

- **Closest prior claims.** Chen 1990; Drettakis and Sillion 1997; Bala et al. 1999 (view-independent state, updates of about 1 s); Xu et al. 2024 (speed-ups over correlated differencing and from-scratch rendering, fixed view, rigid motion only). All make the claim "work proportional to the affected set".
- **Actual difference.** The setting: sharp glossy, multi-bounce, view-independent, GPU, frozen ER.
- **Class: ALREADY KNOWN** as a claim type. **NOT SUPPORTED** as a result: nothing is measured (FACT, report header), and the report shows cases where the dirty fraction tends to 1 (§7). An exact identity carries no cost content (FACT, brief §6).

### M10 — integration with frozen ER

- Reading compiled lobe parameters to choose lookup vs continuation is M7 with ER as the BSDF. Material-frame transforms on the mover are standard instancing.
- **Class:** application only. It is trivially "new" because ER is project-internal; that is not mechanism novelty (INF).

## 3. Neural and reusable-transport families: bearing on Track A

All INF unless marked.

| Family | Representative works (verification) | Bearing |
|---|---|---|
| Online neural caches | Müller et al. 2021 (V-repo) | Adapt the whole cache by continuous training. No explicit invalidation. A competitor for "edit response", not a predecessor of the delta mechanism. |
| Residual-trained scene solutions | Hadadan et al. 2021 (V-repo); R. Su et al., "Dynamic neural radiosity with multi-grid decomposition", SIGGRAPH Asia 2024 (V-s); S. Diolatzis, J. Philip, G. Drettakis, "Active exploration for neural global illumination of variable scenes", ACM TOG 2022 (V-s; 5–18 h training) | The Neural Radiosity loss is the rendering-equation residual of the stored solution. Warm-started retraining after an edit is defect correction on a learned `b` (DERIV, same algebra as report §1). Variable-scene versions parameterise the edit instead of updating. I found no localized, dirty-set-restricted retraining paper (weak negative). |
| Object-centric learned transfer | RNA, 8DNA, TransGI (V-repo); C. Zheng et al., "NeLT: object-oriented neural light transfer", ACM TOG 42(5), 2023 (V-s); Z. Ren et al., "LightFormer", ACM TOG 43(4), 2024 (V-s) | They answer rigid edits by a different mechanism (per-object transfer plus runtime composition). Under the baseline gate their rigid-edit case can be Native Reuse (FACT, `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`). Track A claims no comparison with them; any such comparison is UNMEASURED. |
| Dynamic-scene PRT | K. Iwasaki, Y. Dobashi, F. Yoshimoto, T. Nishita, EGSR 2007 (V-s; diffuse and glossy BRDFs, rigid objects as secondary sources); M. Pan et al., "Precomputed radiance transfer field for rendering interreflections in dynamic scenes", CGF 26(3), 2007 (V-s; the indexed author list was garbled); Zhou et al. 2005 shadow fields (V-repo); B. Loos et al., "Delta radiance transfer", I3D 2012 (checked by Track A) | Precomputed operators for moving rigid objects, low to moderate angular frequency. Delta operators for inserted objects are published (Loos 2012). |
| Many-light and reservoir reuse | Laine et al. 2007; ReSTIR GI / PT / conditional ReSTIR (above) | Sample-level temporal reuse with validation. No persistent directional field, no dirty-set certificate. |

## 4. Combination-level assessment

**Candidate named by the report (§13):** "a directionally complete, sharp-glossy-capable persistent state together with a discovery-then-overwrite update whose cost is certified by the dirty measure."

| Element | Status |
|---|---|
| View-independent directional radiance state with geometric invalidation and recompute | Published: Bala, Dorsey, Teller 1999 |
| Incremental update of glossy global illumination after object motion | Published: Granier and Drettakis 2001; Gautron et al. 2007 (low-frequency glossy caches) |
| Changed-set sampling from the old and new pose | Published: Xu et al. 2024 |
| Sharp-glossy-capable persistent state | Open problem with active 2025–2026 work; the report proposes none |
| Cost certificate by dirty measure | Not provided; the report states no a-priori certificate exists without hierarchical-radiosity bounds (§7) |

**Difference that survives.** Angular bandwidth (α ≈ 0.04 class), glossy-to-glossy multi-bounce, GPU execution, a frozen learned local closure, and REF-ER / REF-P discipline. I found no single paper with all of these. That is a weak negative from about 30 queries.

**Class: UNCERTAIN, at combination level only.**

| Kind of novelty | Verdict |
|---|---|
| Mechanism novelty | None found. ALREADY KNOWN for M1–M8. |
| Application novelty | Possible: applying discover-then-overwrite to a sharp-glossy directional state around ER. Not a new principle. |
| Implementation engineering | Most of the remaining content: state layout, cell-level marking in 4D, GPU scheduling. |
| Measured utility | NOT SUPPORTED: nothing measured. This is the only route to a contribution and it depends on a representation the report does not supply. |

## 5. Corrections and additions to the report's §13

1. **Add Bala, Dorsey and Teller 1999** (and Bala's radiance-interpolant work). It is the closest predecessor of DDT-D as narrowed, closer than Xu et al. 2024, which is the closest predecessor of the on-demand form. Xu et al. cite it (V).
2. **Add Granier and Drettakis 2001.** It removes "directional restatement of incremental radiosity" as even an application-level first.
3. **Add Shaw 1997 detail**: shadow links and ghost links are the newly-blocked and newly-visible terms of report §4.
4. **Add Schied et al. 2018** and Gautron et al. 2007's refresh rule as predecessors of "delta for detection, overwrite for values".
5. **Add the NRC termination heuristic** and Gassenbauer et al. 2009 as predecessors of the path-continuation rule.
6. **Confirmed (V):** Xu et al. keep only the previous image, assume a fixed camera, support rigid motion only, and list Chen 1990, Drettakis and Sillion 1997, Dmitriev et al. 2002, Günther and Grosch 2015 and Bala et al. 1999 as related work. The report's "honest difference" statement about Xu et al. is accurate.
7. **Confirmed (V-s):** T. Günther, T. Grosch, "Consistent scene editing by progressive difference images", CGF 34(4), 2015, pp. 41–51.
8. The report's falsification arms A2 and A3 are both members of the Xu et al. family. A fair control would state which of that paper's estimators (correlated differencing vs residual sampling with path mappings) each arm implements (INF).

## 6. Evidence a credible contribution would require

None of this is measured or authorised; listing it is not a recommendation.

1. A specified directional state that holds α ≈ 0.04-class lobes in never-consumed directions within a declared memory cap, with its T0 build cost. Without it the mechanism is Xu et al. 2024 (report F2).
2. Measured dirty fractions on predeclared GM1 edits, including the adversarial dense case, and early detection of the dense case.
3. Quality-matched cost against: same-state full rebuild; a warm-started full solve without sparsity; REF-ER re-render; residual path tracing in the published form; a Bala-style geometric invalidation without radiometric ranking (to show the delta adds anything beyond geometry).
4. Second-order (glossy-to-glossy) receivers handled without over-invalidation, with ghost and negative-radiance mass reported.
5. Any external claim under the baseline gate: Neural Full Recompute or Native Reuse named per owned state; path-traced re-render kept as a separate role. Currently UNMEASURED.
6. A statement of which element is claimed as new after items 1–4. On present evidence the only defensible wording is a measured operating point of known mechanisms.

## 7. Search limits

- Tools: web search (US index) and page fetch. About 30 queries; pages read: Xu et al. arXiv HTML, Bala project page, three arXiv abstracts. The Xu PDF and the MPI selective-photon-tracing page were not readable.
- **Unverified (M) details:** negative-energy corrective photons in Dmitriev et al. 2002; ReSTIR GI sample validation; Lumen's roughness threshold; Cohen et al. 1988; Gortler et al. 1994; Jensen 1996; Arvo et al. 1994; the TOG 1999 radiance-interpolants article title.
- **Partly verified:** Tatzgern et al. 2024 venue wording; Tawara et al. 2004 venue; Boissé 2021 venue; Pan et al. 2007 author list; transport scope of Bala et al. 1999 (diffuse / glossy multi-bounce not confirmed); how Granier and Drettakis 2001 represent glossy radiance on surfaces.
- **Possible blind spots:** game-engine talks and patents on cache invalidation; 2025–2026 papers not yet indexed; non-English literature; full texts behind paywalls. The weak negatives in M4 and §4 are bounded by this.
- Earlier project kill searches (WL77A, WL80A, WL80B) target intrinsic K of a scattering medium. They contribute verified citations for PRT, neural baselines, hierarchical radiosity and Drettakis–Sillion, but they do not cover incremental or delta transport; this record is the first to do so.

## 8. Sources (external, this session)

- https://diglib.eg.org/handle/10.1111/cgf15152 ; https://arxiv.org/html/2406.16302 (Xu et al. 2024)
- https://diglib.eg.org/handle/10.2312/8873 (Granier and Drettakis 2001)
- https://diglib.eg.org/handle/10.2312/EGWR.EGWR99.031-044 ; https://www.cs.cornell.edu/~kb/projects/ips/index.html (Bala, Dorsey, Teller)
- https://doi.org/10.1145/97879.97894 (Chen 1990, DOI as returned by search); https://en-cg-web.coecis.cornell.edu/pubs/1990/GSG90.html (George et al. 1990)
- https://en-cg-web.coecis.cornell.edu/pubs/2002/TPWG02.html (Tole et al. 2002)
- https://history.siggraph.org/?p=116984 (Brière and Poulin 1996)
- https://unpaywall.org/10.2312%2FEGWR%2FEGWR02%2F025-036 (Dmitriev et al. 2002)
- https://diglib.eg.org/handle/10.1111/cgf12677 (Günther and Grosch 2015)
- https://research.nvidia.com/publication/2007-06_incremental-instant-radiosity-real-time-indirect-illumination (Laine et al. 2007)
- https://diglib.eg.org/handle/10.2312/CGF.v24i3pp401-412 (Smyk et al. 2005); https://pure.mpg.de/pubman/item/item_1329667_3/component/file_3583775/CGI2004_tawara.pdf (Tawara et al.)
- https://diglib.eg.org/items/187965be-76a6-4e07-a97a-6942a44c1a74 (Gassenbauer et al. 2009); https://dcgi.fel.cvut.cz/en/publications/2005/krivanek-tvcg-rceg/ (Křivánek et al. 2005)
- https://history.siggraph.org/?p=142617 (Gautron et al. 2007)
- https://cg.ivd.kit.edu/atf.php (Schied et al. 2018)
- https://diglib.eg.org/items/2b3d3883-9412-4d6f-9615-8cb6eabc5e14/full (Shaw 1997); https://rd.springer.com/chapter/10.1007/978-3-7091-6809-7_21 (Forsyth et al. 1994, via citing record)
- https://diglib.eg.org/handle/10.1111/cgf14378 (ReSTIR GI); https://research.nvidia.com/labs/rtr/publication/lin2022generalized (GRIS); https://research.nvidia.com/labs/rtr/publication/kettunen2023conditional/ (conditional ReSTIR)
- https://dblp.dagstuhl.de/rec/journals/tog/DachsbacherSDD07.html (antiradiance)
- https://advances.realtimerendering.com/s2022/ (Lumen course; slides not read)
- https://arxiv.org/pdf/2106.12372 (NRC)
- https://highperformancegraphics.org/slides24/hpg24_oscgi.pdf (Tatzgern et al. 2024, slides)
- https://arxiv.org/abs/2509.07522 ; https://arxiv.org/abs/2606.08469 ; https://arxiv.org/abs/2610.11702
- https://woaixuexisr.github.io/papers/dynamic-neural-radiosity ; https://arxiv.org/abs/2203.08272v1 (Diolatzis et al. 2022)
- https://dblp.dagstuhl.de/rec/journals/tog/ZhengHMZWHWB23.html (NeLT); https://wylighting.github.io/lightformer (LightFormer)
- https://diglib.eg.org/handle/10.2312/EGWR.EGSR07.035-044 (Iwasaki et al. 2007); https://diglib.eg.org/handle/10.2312/CGF.v26i3pp485-493 (Pan et al. 2007)
- https://graphics.stanford.edu/papers/glossy (Aupperle and Hanrahan 1993); https://dblp.uni-trier.de/rec/journals/tog/ChristensenLSS97.html (Christensen et al. 1997); https://history.siggraph.org/?p=116032 (Immel et al. 1986)
