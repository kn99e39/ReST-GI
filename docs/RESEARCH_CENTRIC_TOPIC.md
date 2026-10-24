# Research Centric Topic — Surface-Owned Selective Neural Transport Recomputation

**Last updated:** 2026-10-23 (Batch 6 FNTO v0.1 reported NOT SUPPORTED; worklog 41 latest measured evidence)  
**Status:** Active problem; Batches 1–4 of ≥ 5 reported. Static ownership feasibility is established on native Lego. The physical-edit Batch 4 (worklog 37) is CONDITIONAL: oracle-dirty units are sparse and include stationary receivers, their recompute recovers the changed region, and preserved units are exact and valid, At seed 0 the whole image is 0.05 dB short of the predeclared margin to full recompute. The 3-seed replicate (worklog 38) is within the margin (mean −0.08 dB). Hierarchical relation/learned selection is a future direction, not an implemented or validated method.  
**Target:** SIGGRAPH 2027 (official Technical Papers deadline to be verified separately)  
**Superseded framing:** object-level-only selection (2026-10-12), archived at docs/history/ARCHITECTURE_OBJECT_LEVEL_SELECTION_2026-10-12.md

## Primary Representation Direction (2026-10-23; design, not evidence)

At the user's request, FNTO is now the explicitly designed primary *next architecture hypothesis*, replacing SPT-1 as a candidate (not replacing its preserved negative Worklog). See [FNTO Architecture Contract v0.1](FNTO_ARCHITECTURE_CONTRACT_V0_1.md).

- **What is learned:** a reusable **intrinsic optical-domain** source-to-receiver transport kernel accepting spatially/directionally varying illumination (local sharp scattering, SSS, medium-internal transmission); exposed free-space self-interreflection, even within one asset, belongs to **dynamic coupling**.
- **What changes after a supported rigid edit:** external incidence, spatial-perspective / visibility couplings along **all editable free-space paths (including same-asset paths)**, and compact incoming/outgoing radiance coefficients. Reusable learned operator parameters do not need refitting in the hypothesis.
- **Why this differs from SPT-1:** SPT-1's free-space-only low-bandwidth relation state omitted important object-internal paths and scored −11.33 dB vs RNA; FNTO moves those paths inside the learned reusable *transport response*, not into ad-hoc residual shading.
- **Why this differs from Surface-Owned reconstruction:** the validated historical baseline rebuilds whole owned neural units given post-edit supervision; FNTO seeks update of dynamic boundary state with learned source/receiver kernels and no post-edit training.
- **Scope:** rigid motion of locally invariant domains initially. Internal deformation/self-contact may invalidate K, so do not claim nonrigid generality before evidence.
- **Research novelty remains open:** existing neural BSSRDF, learned neural light transport, radiosity compression and geometry-conditioned interreflection prior art must be compared; the candidate's specific compression/reuse/coupling claim is unproven.

Designing FNTO does not retroactively change Batch 5's `NOT SUPPORTED` verdict or authorise unbounded implementation.

---

## 1. Fundamental Research Problem

High-quality neural shading / learned global illumination fitted to a fixed geometry configuration can retain stale interreflection, indirect-path occlusion, glossy and multi-bounce transport after a geometry edit. Re-evaluating the frozen shader with current local surface inputs is not necessarily sufficient.

**Fundamental question:** after a geometry edit, can learned transport be repaired by recomputing the **complete owned neural state of every invalidated transport unit**, while every still-valid unit is reused unchanged?

A single large object can contain regions whose learned transport has become invalid while other regions remain valid. The representation should therefore be able to divide an object's surface into multiple independently reconstructible ownership domains.

## 2. User Decisions in Force (2026-10-13, recorded in worklog 32)

1. RNA (Relightable Neural Assets, adobe-research @ `66b5b098`) is the **initial implementation baseline** — a research substrate, not a permanent constraint or the required final model.
2. The ownership architecture **may be finer than a scene object**.
3. **Surface-Owned Transport Units** are the active research hypothesis.
4. Each invalidated unit **may have its complete owned neural state recomputed** (every parameter it owns).
5. **Object-level full recomputation remains a valid coarse-grained comparison.**
6. At least **five meaningful architecture batches** come before a broad viability judgment.
7. A negative result for one candidate does **not** falsify the general ownership hypothesis.
8. Dynamic-GI research of roughly **2007–2013** is the priority historical design reference (section 5).
9. The 1997 line-space hierarchy (Drettakis & Sillion) is **not** the primary architectural starting point.
10. Full efficiency and scaling validation come **after representation feasibility**.

**2026-10-23 (Batch 6, user brief "FNTO v0.1 Initial Implementation & Physical-Edit Validation"): IMPLEMENTATION AUTHORISED.** The user authorised one bounded architecture batch that implements `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` (authoritative for the representation; intrinsic optical-domain K with exposed same-asset free-space paths in C, a=b allowed) and evaluates it on the official Lego + red box held-out T0/T1 edit (worklog 41; exports `results/evaluation/41/`). Required: reference-renderer semantics study (Christensen-Burley SSS scope) before main kernel training, isolated-domain spatial/directional probe supervision of K, synthetic contracts A-F on the production code, predeclared screening rules, arms A-F (historical RNA / Surface-Owned / SPT-1 unchanged; FNTO stale / updated / optional oracle diagnostic), full cost accounting and a SUPPORTED / CONDITIONAL / NOT SUPPORTED verdict. Not authorised: a learned dirty-unit selector, a second architecture family after a failure, SPT-1 modification, changes to historical baselines, automatic start of a later batch. No FNTO result exists yet.

**Batch 6 result (worklog 41, 2026-10-23): NOT SUPPORTED** under protocol fnto_v1. The implemented FNTO v0.1 (learned intrinsic kernels K_loc/K_glob/K_band/K_mid trained only on isolated-domain first-intrinsic-event probes; ray-cast free-space coupling incl. same-asset paths; learned coupling rule empty) diverged in the multi-bounce solve on 64/64 held-out view solves, so static fidelity and edit recovery are unmeasurable. Attribution (descriptive): the learned nonlocal kernel parts are non-passive on incident fields outside the span of the 1,400 sun-probe training fields (effective albedo > 1 on 26 % of ports for K_mid, 18 % for K_glob; spectral radius ~2.1 with K_glob, explosive with K_mid; one bounce already 5.5x the reference remainder); the kernel alone reproduces first-event transport at 22 dB vs first-event references (~10 dB above probe noise). Coupling semantics and the exact incremental relation update (0.83 % of rays) behaved as designed; kernel parameters stayed bit-identical. Conclusion: an intrinsic operator must be passive on the fields the coupling produces, not only linear. No later batch is authorised.

**2026-10-23 (user brief "FNTO Validity Restoration — Post-Batch-6 Controlled Implementation & Re-evaluation"): CORRECTIVE IMPLEMENTATION AUTHORISED (worklog 42).** FNTO stays the primary architecture; the batch verifies the six findings of an independent Batch 6 validity audit (buried port origins / biased incident lighting; insufficient independent spatial excitation of K; effective kernel support narrower than declared; incomplete internal reflection / directional transport; reference clamping and reconstruction-filter mismatches; insufficient signed-change, solver-stability and coupling-compression evaluation) against code and evidence, corrects confirmed defects in a distinct candidate and protocol version, and re-evaluates through gates 1-5. Batch 6 artefacts, protocol, verdict (NOT SUPPORTED) and worklog 41 are preserved unchanged; no other architecture is started.

**Worklog 42 result (2026-10-24): NOT SUPPORTED** under protocol `fnto_r1` (static rule: 24.49 dB vs RNA 33.57 in the historical regime). Evidence state for the research question: the audit's six findings were confirmed and corrected; with the corrections the composed intrinsic-kernel × free-space-coupling solve converges on all 64 held-out views **only** when passivity is imposed in training (unconstrained kernel ρ 10.5, diverges), and the reuse contract then behaves as specified — a geometry-only incremental coupling update with bit-identical kernel parameters reproduces a full rebuild and moves the image in the direction of the reference change (corr 0.987, slope 0.948 on the changed region; F 2.44, R_RNA 0.65). The unresolved limit is the intrinsic kernel's own fidelity (25 dB first-event; glass, SSS and the box worst), with its cause not separated (capacity / operator family / aggregation-quadrature / optimisation). Gates 1, 2b, 2c, 2d failed as declared and are carried. Scope: one scene, one edit, one seed. No later batch is authorised; the next step is a user decision.

**2026-10-19 direction decision (not an implementation approval):** Make interreflection, including nonlocal and multi-bounce dependencies, a first-class requirement of future representation/selection contracts. Favor a physically interpretable, algorithmically constructed spatio-directional hierarchy for candidate discovery over asking a network to learn the entire scene hierarchy end-to-end; investigate learned transport-relation scoring / invalidation selection on top. The specific hierarchy, features, and learner remain unvalidated hypotheses. Preserve the current oracle-based physical-edit Batch 4 controls.

Never reinterpret "selective" as a mask over triplane cells, feature channels or MLP layers inside one jointly owned model (worklog 30's tested and negative hypothesis). Selection is over **owned units**, each of which is recomputed whole.

## 3. Transport Unit Contract

A unit is U_i = (S_i, Θ_i):
- S_i — a well-defined owned surface domain; every evaluated surface sample resolves to exactly one owner (or a declared explicit composition of owners);
- Θ_i — the full set of learned parameters owned by that domain; updating U_i never changes Θ_j (j ≠ i).

Unit ownership is **storage and update ownership, not physical locality**: a unit's targets are full-scene GI, including light from other objects. A shared trainable decoder that must change to recover one unit is a representation coupling to report, never to hide with optimizer masks.

Separate properties, always reported separately:
- **parameter preservation** (Θ_j bit-identical),
- **output preservation** (predictions read only from preserved units are unchanged),
- **radiometric validity** (a preserved unit's prediction is still correct for the current scene).
The first two are contracts of the representation; the third depends on the scene edit.

## 4. Geometry-Dirty vs Transport-Dirty

- D_G: units/objects whose geometry was directly edited.
- D_T: units whose **learned transport is invalid under G1**, including stationary receivers whose indirect light changed.

D_G ≠ D_T. Worklog 31 measured this on the canonical teaset: mover-only refit left the stationary receivers as stale as the frozen model; at object granularity every object was transport-dirty, so whole-object selection could save nothing there. Finer ownership is the motivation for the current hypothesis — whether D_T becomes usefully sparse at unit granularity is an open empirical question.

## 4a. Long-Horizon Interreflection and Relation-Selection Direction (2026-10-19; unvalidated)

The central research goal remains *selective reconstruction of invalid learned transport*, not merely predicting an image change or declaring adjacent surfaces dirty. Interreflection is a required target phenomenon for later tests: geometry edits can affect distant stationary receivers, including through glossy/multi-bounce transport and third-party occluders. A unit's learned-parameter validity and its current output radiance must remain distinct.

**Design preference, not a fixed data structure:** construct a physically interpretable candidate-discovery hierarchy algorithmically. A plausible starting point is spatial bounding regions containing normal-direction bounds (e.g., normal cones), surface-unit references, and material/BRDF bounds. Position, relative direction, extent, orientation, visibility/occlusion and directional scattering play different roles; normal-only clustering is insufficient. A spatial-first tree plus directional summaries is a candidate, **not** a proven or mandatory architecture. Joint spatial-directional alternatives remain open if evidence favors them.

- **Hierarchy vs transport graph:** a tree organizes geometry/search; transport relations connect nodes or units across branches and can be directed, nonlocal, cyclic and multi-bounce. Never treat parent/child adjacency as physical transport dependency.
- **Candidate discovery vs learned decision:** algorithmic traversal proposes potentially affected relations/receivers; a learned scorer *may* predict radiometric influence, invalidation probability, or expected error reduction. Learnability and the need for a neural scorer must be established against an algorithm-only control.
- **Coverage risk:** a receiver omitted at candidate discovery cannot be recovered by an accurate downstream network. Include edited-occluder effects on otherwise unchanged sender/receiver pairs; compare candidate recall against physically grounded oracle transport changes before permitting pruning.
- **Separate representations:** surface/parameter ownership, scene transport relation, learned-state validity, output contribution and recomputation schedule are distinct contracts. Hierarchy nodes need not correspond one-to-one to neural ownership units.
- **Selection objectives:** false negatives leave stale GI; false positives waste recomputation. Report radiance-error-weighted misses, oracle coverage, selected-unit count and total selector+recompute cost, not classification accuracy alone.
- **Evidence order:** Batch 4 is complete (worklog 37, CONDITIONAL). For this edit it showed changed-region recoverability (R = 0.98) and a sparse D_T (8/17 units), The seed replicate (worklog 38) resolved the whole-image margin: mean −0.08 dB over 3 seeds, within the margin. The next evidence gap is breadth: interreflection-dominated or other edits, before relation-observability or selector work. Only then consider a separately approved, controlled relation-observability / algorithm-only-vs-learned selection batch. Do not claim learned hierarchy/selection validity or start implementation from this direction record.

Historical 2007–2013 dynamic-GI methods are references for *how they changed the computational and reuse contract*, not algorithms to import unchanged. Prior sparse K=32 relational radiance-residual failures (worklogs 28–29) do not directly test learned dirty-unit selection; they do caution against assuming sparse directional samples suffice for glossy transport.

## 5. Prior Art and Historical Design References

Priority references (advisor-approved direction; design references only, nothing implemented automatically):

| Reference | What it contributes | Category |
|---|---|---|
| Gautron, Bouatouch, Pattanaik, *Temporal Radiance Caching*, IEEE TVCG 13(5), 2007 | reuses cached (ir)radiance records across animation frames with temporal gradients; glossy GI in animated scenes | radiance caching, temporal reuse / invalidation by age-gradient |
| Debattista, Dubla, Banterle, Santos, Chalmers, *Instant Caching for Interactive Global Illumination*, CGF 28(8), 2009 | irradiance cache + instant radiosity; updates only light paths invalidated by geometry changes | radiance caching, transport invalidation |
| Wang, Zhou, Pan, Bao, *An Efficient GPU-Based Approach for Interactive Global Illumination*, SIGGRAPH 2009 | clusters shading points (adaptive seeding + k-means), final gather at cluster centres, interpolation | sampling and scheduling (clustered evaluation) |
| Crassin, Neyret, Sainz, Green, Eisemann, *Interactive Indirect Illumination Using Voxel Cone Tracing*, CGF 30(7) (PG) 2011 | on-the-fly voxel octree of the scene, cone-traced visibility/incoming energy, 2 bounces incl. glossy | surface/volume representation rebuilt per frame |

Keep five concepts distinct: **surface representation**, **radiance caching**, **learned parameter ownership**, **transport invalidation**, **sampling/scheduling**. These papers mainly contribute caching, invalidation and scheduling ideas; none defines learned-parameter ownership, which is this project's representation question.

Also prior art (not the starting point): hierarchical/line-space dependency structures (Drettakis & Sillion 1997), Luksch et al. 2019, Enlighten workflows; neural per-object composition (NeLT 2023, Superposed Deformable Feature Fields 2024). Do **not** claim novelty for "a hierarchy", "update affected parts only", or "per-object/per-region neural representations" alone.

## 6. Established Evidence and Scope

- Worklogs 21–25: a changed relative placement within the canonical teaset produces nonlocal indirect transport changes at stationary surfaces; frozen 8DNA and RNA do not track it; indirect-path occlusion and moved-part reflection dominate.
- Worklogs 26–27: physical and historical whole-asset neural recomputation are expensive (different regimes).
- Worklogs 28–29: two compact relational-state candidates fail held-out near-mirror T3.
- Worklog 30: intra-object triplane-cell masking in one 8DNA model (shared flows frozen) fails — a valid negative result **for that feature-level hypothesis only**.
- Worklog 31: explicit (non-neural) per-part surface radiance units with no shared decoder compose to 8DNA-level T0 quality and, refitted, recover the T3 change (tracking gain ≈ 1); at object granularity D_T = all objects; no wall-clock saving in that implementation.
- Worklog 32 (Batch 1): RNA split into K complete RNA units (P1/P4/P16, matched and full-size). Parameter and output ownership hold exactly (REINIT: 15 frozen units bit-identical, every frozen-unit pixel bit-identical, re-initialised unit recovers its P16 error). Static fidelity of P16 is within ~0.2 dB / 1.03× MAE of original RNA, with a measurable excess discontinuity at intra-part seams. **But the predeclared static-credibility gate (original RNA ≥ 25 dB) failed in both configurations** (C1 17.97 dB; C2 24.92 dB), so the ownership-fidelity verdict is not adjudicated.

- Worklog 33 (baseline attribution):
  - The project's RNA integration is faithful to upstream.
  - The C1 gate failure is an RNA representation limit on near-mirror conductors.
  - The C2 miss is dominated by reference noise in the gate measurement.
  - A credible static reference needs a prospective protocol scored against converged targets.

- Worklog 34 (Batch 2):
  - Seams between owned units come from independent per-unit error fields, exposed by hard selection.
  - An explicit partition-of-unity composition (P16-PU) removes most of the seam excess. It does so with exact output-influence accounting and without changing any learned state.
  - Independent decoders still cost fidelity inside units.

- Worklog 35:
  - Native RNA on the official Lego reproduces the released checkpoints (classification: comparable).
  - The remaining gap to the paper is validation-target noise.
  - RNA is a credible substrate when evaluated against converged targets.

- Worklog 36 (Batch 3, native Lego):
  - 16 independent complete RNA units keep static fidelity within the predeclared screening (Case A): −0.32 dB, logL1 1.05× vs native RNA HQ, interior error unchanged. The comparison is capacity-confounded upward (1.54× parameters).
  - PU composition removes 91% of the seam excess.
  - One-unit REINIT is exact in parameters and output influence and recovers the unit.
  - Radiometric validity after a geometry edit is still untested.

## 7. Immediate Scientific Frontier

Representation feasibility first (decision 10). Open questions, in order:
1. ~~a credible static baseline under a predeclared protocol~~ — met on the native Lego (worklogs 35–36);
2. unit-level reconstruction after a **physically grounded** geometry edit (radiometric validity, not just ownership semantics);
3. whether D_T is sparse at unit granularity on a scene where that is physically plausible;
4. seam/composition quality and capacity cost of ownership;
5. only then dependency observability, interpretable algorithmic hierarchy, optional learned relation scoring/selection, and measured efficiency and scaling (see §4a).

**2026-10-22, Batch 5 (user brief):** the project now also evaluates a **factorized** representation, SPT-1 (docs/Architecture.md §9a).
- **Persistent learned state:** canonical neural surface fields and shared heads, never re-learned after an edit.
- **Dynamic transport state:** incident SH computed from current-geometry relations and persistent sender properties.
- **Reusable operator:** linear in the incident state.

Motivation: complete-unit reconstruction (worklogs 37–38) works but re-learns configuration-specific transport with target supervision. SPT-1 asks whether an edit can be absorbed by state recomputation alone.

The new properties are hypotheses. Glossy and broader generalisation tests are evaluation targets of the batch, not prerequisites for design.

**Result (worklog 39): NOT SUPPORTED** on static representability (−11.3 dB vs RNA). Most of the error is object-internal transport (subsurface, transmission), which the free-space relation state cannot carry. The reuse mechanism worked descriptively for free-space transport (R = 0.82; sand R = 0.92; no re-learning).

## 8. Research Conduct / Stop Conditions

- Preserve historical scripts, worklogs 21–32 and archived architecture snapshots; never reinterpret worklog 30 as a test of unit-level or object-level ownership.
- Predeclare scenes, quality gates and thresholds; never move them after seeing results.
- A stop condition ends the current experimental route, not the research direction; attribute a negative result (baseline limitation, candidate limitation, ownership isolation, optimisation/capacity, partition, supervision/correspondence, compositing) before interpreting it.
- No automatic advance to the next batch; each batch needs a user decision.

## 9. One-Line Research Definition

> **Keep high-quality learned GI valid under geometry edits by owning learned transport in independently reconstructible surface units, recomputing the complete state of only the units whose transport became invalid, and reusing every unit that remains valid.**
