# Architecture — Surface-Owned Baseline and FNTO Candidate

**Updated:** 2026-10-23 (Batch 6 FNTO v0.1 reported NOT SUPPORTED, worklog 41 latest measured evidence)  
**Status:** Batches 1–3 of ≥ 5 reported. Ownership contract exact; on the credible native Lego (worklog 36) independent units under PU composition keep static fidelity within the screening (−0.31 dB, logL1 1.05×; Case A); after a real edit (worklog 37, oracle D*) complete-state recompute of the dirty units recovers the changed-region radiance (R = 0.98) with V* bit-identical and valid, the whole image is −0.31 dB vs full recompute at seed 0 (CONDITIONAL), and a 3-seed replicate puts the mean gap at −0.08 dB, within the margin (worklog 38)  
**Branch:** Selective_Recompute  
**Historical architectures:** docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md, docs/history/ARCHITECTURE_INTRA_OBJECT_FEATURE_REFIT_2026-10-08.md, docs/history/ARCHITECTURE_OBJECT_LEVEL_SELECTION_2026-10-12.md

## 0. Primary Next Representation — FNTO (Design Only, 2026-10-23)

**Design source:** [FNTO Architecture Contract v0.1](FNTO_ARCHITECTURE_CONTRACT_V0_1.md). The user requested that architecture choices be made explicitly by the assistant, rather than delegated as an unconstrained decision to an implementation Agent.

**Current architecture decision:** Do not repair SPT-1 with ad-hoc SSS/Transmission branches. Instead, adopt **Factorized Neural Transport Operator (FNTO)** as the *primary next hypothesis*. The learned **intrinsic optical-domain kernel** K_a maps spatially/directionally varying incident radiance to outgoing radiance, including local BRDF, SSS, internal transmission and protected medium scattering. Current-geometry **free-space coupling** C_ab(R_ab^t) transfers radiance between optical domains **and between surfaces of the same asset** when paths exit into editable free space. Never freeze same-asset exposed interreflection merely because both endpoints have the same Object ID. K and its learned coupling rule remain fixed across a **supported rigid-body geometry edit** while external illumination, coupling coefficients and solved radiance state change.

Conceptual equation: `b_a^t = K_a[e_a^t + sum_b C_ab(R_ab^t) b_b^t]`, including `a=b` for editable exit/re-entry paths.

**Crucial physics boundary:** K_a includes within-domain scattering (not a frozen radiance texture). C_ab is cross-domain free flight; this boundary prevents double counting paths. Local SSS cannot be gated by queried-pixel direct visibility as in SPT-1. A conditional response to spatially varying incident illumination requires new object-local illumination-probe training, not RNA's fixed per-directional-light interface.

**Candidate and baseline distinction:** The Surface-Owned L16 full-model reconstruction contract below remains intact and supported by WL32–38. FNTO is a new learned **operator-state reuse** hypothesis with different update semantics. SPT-1 (WL39) remains a negative historical prototype. This document does not claim FNTO implementation, validation or new batch authorisation.

**2026-10-23 (Batch 6, user brief "FNTO v0.1 Initial Implementation & Physical-Edit Validation"): IMPLEMENTATION AUTHORISED.** The user authorised one bounded architecture batch that implements `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` (authoritative for the representation; intrinsic optical-domain K with exposed same-asset free-space paths in C, a=b allowed) and evaluates it on the official Lego + red box held-out T0/T1 edit (worklog 41; exports `results/evaluation/41/`). Required: reference-renderer semantics study (Christensen-Burley SSS scope) before main kernel training, isolated-domain spatial/directional probe supervision of K, synthetic contracts A-F on the production code, predeclared screening rules, arms A-F (historical RNA / Surface-Owned / SPT-1 unchanged; FNTO stale / updated / optional oracle diagnostic), full cost accounting and a SUPPORTED / CONDITIONAL / NOT SUPPORTED verdict. Not authorised: a learned dirty-unit selector, a second architecture family after a failure, SPT-1 modification, changes to historical baselines, automatic start of a later batch. No FNTO result exists yet.

**Batch 6 result (worklog 41, 2026-10-23): NOT SUPPORTED** under protocol fnto_v1. The implemented FNTO v0.1 (learned intrinsic kernels K_loc/K_glob/K_band/K_mid trained only on isolated-domain first-intrinsic-event probes; ray-cast free-space coupling incl. same-asset paths; learned coupling rule empty) diverged in the multi-bounce solve on 64/64 held-out view solves, so static fidelity and edit recovery are unmeasurable. Attribution (descriptive): the learned nonlocal kernel parts are non-passive on incident fields outside the span of the 1,400 sun-probe training fields (effective albedo > 1 on 26 % of ports for K_mid, 18 % for K_glob; spectral radius ~2.1 with K_glob, explosive with K_mid; one bounce already 5.5x the reference remainder); the kernel alone reproduces first-event transport at 22 dB vs first-event references (~10 dB above probe noise). Coupling semantics and the exact incremental relation update (0.83 % of rays) behaved as designed; kernel parameters stayed bit-identical. Conclusion: an intrinsic operator must be passive on the fields the coupling produces, not only linear. No later batch is authorised.

**Architectural unknowns to test, not excuses to defer design:** boundary-port/source/receiver compression fidelity (especially SSS/glass), angular bandwidth, runtime observable coupling, operator generalisation to held-out incident fields/edits, sparse graph coverage and full cost.

---

## 1. The Architecture Question

After a geometry edit, which **transport units** hold invalid learned transport, and can the **complete owned neural state of only those units** be recomputed while every valid unit is reused unchanged?

- **Selection unit:** a Surface Transport Unit U_i = (S_i, Θ_i). It can be finer than a scene object (user decision 2); a whole object is the coarsest unit.
- **Recompute unit:** all of Θ_i (decision 4). This is never a subset of cells, channels or layers inside a jointly owned model; that was worklog 30's tested and negative hypothesis.
- **Coarse comparison:** object-level full recomputation, where every unit of an object is recomputed together (decision 5).
- **Baseline substrate:** RNA, adobe-research `66b5b098` (decision 1). It is a substrate, not a commitment.

## 2. Unit Contract (mandatory for every candidate)

**Surface ownership:**
- every evaluated surface sample resolves to one owner, or to a declared explicit composition of owners;
- report coverage, identity, boundary behaviour, overlap semantics, mapping stability and coordinate conventions;
- the owner function must be identical across training, rendering and later updates.

**Learned-parameter ownership:**
- each unit has an identifiable trainable set Θ_i;
- updating U_i changes no Θ_j (j ≠ i);
- every parameter in Θ_i may change.

**Shared components.** A shared component is allowed only with defined semantics, either:
- frozen forever (a fixed function), or
- declared as a coupling.

If a shared trainable decoder must change to recover one unit, that is a representation coupling to report, never to hide with optimiser masks.

**Transport semantics:**
- a unit's targets are full-scene GI, including light from other objects, so ownership does not mean physical locality;
- keep RNA's semantics: outgoing radiance under the dataset light distribution, with direct visibility selecting the lit or shadowed output;
- declare any change to these semantics as a confound.

**Report separately:**
1. parameter preservation;
2. output preservation;
3. radiometric validity.

## 3. Update Model

~~~text
T0 geometry G0 -> train all units U_i^0 on full-scene transport
edit G0 -> G1
   D_G: units/objects whose geometry changed
   D_T: units whose learned transport is invalid under G1 (includes stationary receivers)
   for i in D_T:  Θ_i^1 = FULL_RECOMPUTE(Θ_i^0 or init, supervision_i(G1))   # all of Θ_i eligible
   for j not in D_T: Θ_j^1 = Θ_j^0  (bit-identical)
   render: compose all units on G1
~~~

**D_G ≠ D_T** (worklogs 21–25, confirmed in worklog 31). Whether D_T is sparse at unit granularity is an open empirical question. At object granularity on the compact teaset it was all objects (worklog 31).

## 3a. Future Interreflection-Aware Relation/Selector Layer (direction only, unvalidated)

**Reason for the layer.** Under a geometry edit, locally unchanged surfaces can carry stale learned indirect radiance. Multi-bounce interreflection, glossy directional coupling and third-party occluder motion make distance, normal similarity and spatial adjacency unreliable as sufficient invalidation criteria. The architecture must separate learned-state validity from a changed outgoing-radiance value, and separate changed transport coupling from propagation of radiance through an otherwise reusable response. No specific baked-radiance versus transport-response representation is selected yet.

**Long-term design preference (user discussion, 2026-10-19):** form the candidate relation hierarchy through explainable geometric/transport algorithms, *not* by requiring a network to invent the full scene tree; consider a learned scorer for difficult influence/invalidation decisions on the discovered candidates. Neither the need for learning nor a particular hierarchy algorithm has been demonstrated.

Proposed layer decomposition (roles, not an implementation prescription):

1. **Algorithmic candidate index:** organize surface elements spatially, retaining node bounding regions, directional bounds such as normal cones, material/BRDF summaries and references to owning transport units. Normal-only grouping is inadequate; spatial-first and joint spatial-directional partitions are competing designs, not a prematurely fixed choice.
2. **Transport-relation candidates:** cross-node / cross-unit relations informed by relative position and direction, occlusion and scattering. The index tree is *not* itself the transport graph; physical relations can be nonlocal, directed, cyclic and multi-bounce. Edits to a third-party occluder can change the coupling of two untouched surfaces.
3. **Optional learned selection:** infer candidate influence, state invalidation or recoverable radiometric error from observable pre/post-edit features. Compare against an algorithm-only selector; neural inference must justify its errors and cost. Learning the hierarchy end-to-end is not the adopted default.
4. **Owned-state reconstruction:** map selected transport-dirty receivers to their existing full owned neural parameter sets Θ_i. Preserve Θ_j exactly for every unselected j; separately quantify whether the preserved states remain radiometrically valid and any composition-band output influence.

**Candidate discovery is a coverage contract, not an accuracy afterthought.** A missed receiver cannot be restored by downstream learning. No pruning based only on normal cones, distance, or hierarchy membership may be called safe without physical-oracle coverage evidence. Distinguish false-negative stale-GI error from false-positive computation; include selector traversal/inference in end-to-end cost. Training targets, tolerance semantics and fallback policies require a future predeclared batch.

**Phase boundary:** Batch 4 (worklog 37) ran its oracle physical-edit protocol unchanged except for execution-only amendments 2–4. Do not train a selector, alter dirty-unit labels, add a hierarchy to production, or modify baseline ownership in this documentation-only decision. Await the oracle D_T, recovery and sparsity results, then obtain an explicit subsequent-batch decision.

## 4. Batch 1 Candidate (worklog 32): Independent Complete RNA Units

**Representation.** K complete upstream `NeuralSurfaceTriPlane` models. Each unit has:
- its own triplane, framed on the unit's AABB (+2% margin);
- its own MLP.

Nothing trainable is shared.

**Partitions (geometry only, predeclared):**
- P1: whole asset;
- P4: one unit per part;
- P16: k-means per part (tray 9, tea pot 4, milk pot 2, tin 1).

The owner of a point is its exact part (point-to-triangle distance at contact zones), then the nearest dense sample's unit.

**Dispatch:** the owner id is appended as a 4th position channel, and each pixel is evaluated by its owner only (hard selection).

**Training:** upstream Lightning module, loss, optimiser and schedule, unchanged.

**Capacity:**
- matched: A's texel size per unit and MLP width 512/√K. Matched P16 still holds 1.89× A's parameters because unit AABBs overlap.
- P16w: full-size units, 16× A (diagnostic only).

Code is in `experiments/rna_surface_ownership/`; the protocol is in `protocol/rso_v1.json`.

**Evidence** (details in worklog 32; C2 = rough nickel α 0.3; C1 = canonical near-mirror):

| property | result |
|---|---|
| static credibility of original RNA (predeclared ≥ 25 dB) | **FAIL** — C1 17.97 dB, C2 24.92 dB (stop condition; fidelity verdict not adjudicated) |
| P16 vs A, C2, descriptive | −0.19 dB, display MAE 1.02×, boundary 1.05×, HDR logL1 **1.15×**, seam discontinuity **1.16×** (C1: 1.11×, 1.39×) |
| P16w (16× params) vs A, C2 | logL1 1.12×, seams 1.14× — the cost is **not** a capacity shortfall |
| P1 (local frame only) vs A | equal (C2) / better (C1) — local normalisation is harmless |
| REINIT (unit re-initialised + fully retrained, 15 frozen) | parameters of 15 units and **100% of their pixels bit-identical**; u* recovers to 0.95× / 1.00× of its prior error |
| composition | correct; hard selection leaves **visible shading steps at intra-part seams** |

**Inherent limits of this candidate:**
- seam steps under hard selection;
- no cross-unit sharing of angular/material learning (hypothesised cause of the HDR cost);
- capacity overhead from overlapping per-unit grids.

**Unknowns:**
- radiometric recovery after a physical edit;
- D_T sparsity at unit granularity;
- any cost advantage (REINIT training cost is 0.65× of a full P16 run with this implementation, because all pixels still pass through all units).

## 4b. Batch 2 (worklog 34): Boundary Attribution and P16-PU Composition

**Seam attribution** (C2; 4 dev views vs 8× references, existing checkpoints):
- **Independent learned functions (H2), dominant.** Seam-pair error correlation is 0.24 vs 0.71 inside units; A shows no drop. A neighbour evaluated across the cut is about as accurate as the owner, yet disagrees with it by as much as each errs.
- **Hard-selection composition (H3) exposes the mismatch.**
- **Supervision starvation is contradicted.** The band has full light coverage, and error is uncorrelated with per-unit sample counts.
- **A small near-seam error elevation** (≈ 13% of the excess) has an unresolved mechanism.

**P16-PU (composition contract, no learned-state change).**
- **Composition:** output = (1 − w) f_own + w f_nbr, within h = 7.2 mm of same-part seams; outside the band it is exactly f_own.
- **Ownership:** parameter ownership unchanged. A unit's output influence is S_i ∪ B_i (its neighbours' band).
- **Held-out results** (36 C2 views vs 1024-spp references):
  - removes 86% of P16's excess seam discontinuity;
  - lowers seam and band error (−9% / −6%);
  - interior, logL1 and high-frequency error unchanged;
  - +0.025 dB image-wide.
- **REINIT regression:** changed outputs equal S_12 ∪ B_12 exactly (3.2% in neighbour bands); recovery 0.993×.
- **C1 stress:** the jump is reduced without an accuracy gain.
- **What it does not fix:** P16's remaining fidelity loss is interior and per-unit (−0.20 dB, logL1 1.18× vs A). Composition cannot fix it.

## 5. Candidate Space for Later Batches (not authorised)

Each is a different realisation of the same contract, to be compared one factor at a time:
- **B1 (Batch 1):** independent complete units with hard selection.
- **Frozen shared decoder + owned per-unit features.** The decoder is trained once and frozen forever, so it is a fixed function; units own features (and optionally a small head). The semantics must state what happens if the decoder becomes inadequate.
- **Overlapping units with explicit partition-of-unity blending.** Implemented as P16-PU in worklog 34; output influence is accounted as S_i ∪ B_i.
- **Non-RNA substrates** (decision 1 allows replacing RNA later), e.g. worklog 31's explicit surface radiance units or other neural surface representations.

## 6. Historical Design References (decisions 8–9)

The priority references are 2007–2013 interactive/dynamic GI:
- Temporal Radiance Caching (TVCG 2007);
- Instant Caching (CGF 2009);
- An Efficient GPU-Based Approach for Interactive GI (SIGGRAPH 2009);
- Voxel Cone Tracing (CGF/PG 2011).

They inform caching, invalidation and scheduling design. None defines **learned-parameter ownership**, which is this architecture's question. Keep distinct:
- surface representation;
- radiance caching;
- learned parameter ownership;
- transport invalidation;
- sampling/scheduling.

The 1997 line-space hierarchy is prior art, not the starting point. See docs/RESEARCH_CENTRIC_TOPIC.md §5.

## 7. Prior Evidence (preserved, scoped)

- Worklogs 21–27: stale nonlocal transport under a part edit (problem evidence) and recompute-cost context.
- Worklogs 28–29: compact relational-state candidates negative on held-out T3 (separate, on hold).
- Worklog 30: intra-object triplane-cell masking negative. This is a different hypothesis, not a test of owned units.
- Worklog 31: explicit per-part surface radiance units (no learned decoder):
  - compose to 8DNA-level T0 quality;
  - all-local refit recovers T3 (tracking gain ≈ 1);
  - object-level D_T = all four objects, so no saving at object granularity;
  - whole-object refit degrades unaffected regions inside dirty objects (MC noise), which motivates finer ownership;
  - no wall-clock saving in that implementation.
- Worklog 32: section 4.
- Worklog 36: section 8 (Batch 3, native Lego).

## 8. Implementation Frontier and Non-Goals

**Current execution frontier (2026-10-21):** Architecture Batch 4 (worklog 37) is reported as CONDITIONAL. Measured: D_T ≠ D_G (7 of the 8 oracle-dirty units are stationary receivers); D* is sparse for this edit (8/17 units); complete-state recompute of D* recovers R = 0.98 of the changed-region error; V* parameters bit-identical and V* error 1.03× full recompute; composed output changes exactly on the predicted support. Not resolved: the −0.31 dB whole-image gap (69% from weakly dirty unit 3, re-learned worse than its own frozen state; single seed). The seed replicate (worklog 38) is WITHIN_MARGIN: mean B_sel − A_full −0.082 dB over seeds 0–2, unit 3 not systematic, V\* and support exact in every seed. Consistent but unattributed: frozen unit 9 is 1.22–1.32× A_full, and the re-learned box unit is 1.12–1.18×. **Batch 5 (worklog 39): SPT-1 is NOT SUPPORTED** (static representability; §9a). The factorization mechanism moved stationary receivers correctly without re-learning. **No successor batch is authorised.**

**Batch 3 (worklog 36), native Lego.**
- **Partition L16:** a rendered point is identified on its mesh triangle; its unit is the k-means cell (K = 16) of the on-surface point, and PU blending is allowed only within the triangle's connected component.
- **Static fidelity:** B −0.32 dB, C −0.31 dB vs A on 16 held-out 4096-spp views; logL1 1.05×, hf 1.06×, off-band 0.996×; PU removes 91% of the seam excess. B has 1.54× A's parameters (capacity-confounded upward).
- **REINIT:** parameter preservation exact; composed output changes exactly on S_i ∪ B_i; the unit recovers its quality.
- **Teaset contrast:** the Teaset interior loss does not recur; the remaining cost is boundary-local and high-frequency, plus a material dependence.
- **Recommended next (user decision):** physical geometry-edit reconstruction on Lego (oracle D_T units vs all units vs object level), preserving valid units.

**Baseline qualification (worklog 35).** Native RNA reproduces its published Lego quality: released and retrained checkpoints agree within 0.2 dB, and against 4096-spp targets they reach the Table 1 level. RNA is therefore a credible substrate on its material class. Future static gates use converged references (≥ 4096 spp or 8× the training spp) and state the aggregation and checkpoint rule.

**Do not implement before feasibility is evidenced and the user authorises a new batch (decision 10):**
- deployable dependency tracking or dirty-unit detection;
- the proposed algorithmic hierarchy or learned selector (§3a);
- efficiency/scaling claims without measured end-to-end costs;
- automatic advance to the next batch.

## 9a. Batch 5 Candidate (worklog 39): Spatial-Perspective Transport Factorization, SPT-1 (new architecture; hypothesis)

**Status (2026-10-22):** authorised by the user's Batch 5 brief. Contract recorded before any SPT-1 result. This is a **new architecture hypothesis**, separate from the Surface-Owned complete-unit contract above, which stays the validated reconstruction baseline and is not redefined.

**Why now.** Worklogs 32–38 showed that complete owned states can be preserved and selectively rebuilt, but every rebuild re-learns configuration-specific transport with target-scene supervision. Each owned unit (like RNA) compresses *persistent appearance* and *current inter-surface transport* into one learned function. SPT-1 tests whether separating them lets a held-out geometry edit be absorbed by **recomputing state, not re-learning parameters**. This realises the archived 2026-10-07 contract (persistent appearance + current incident-transport state + shared operator; docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md), reinterpreted after worklogs 28–29 and 32–38.

**Representation contract v1:**

| category | SPT-1 realisation | changes after a geometry edit? |
|---|---|---|
| A. Persistent local state | per-object canonical neural feature fields φ_o (triplanes in the object's T0 frame; Lego A, box B) and three shared heads: sender albedo a_θ(φ), direct response d_θ(φ, n, ω_o, ω_L), and local transfer t_θ(φ, n, ω_o) → 9×3 linear weights on incident SH | no: bit-identical; inputs are canonical position, local normal and directions only |
| B. Spatial-perspective dependency | relation table R^t built by ray queries in the **current** geometry: surface anchors (canonical position, normal, object); per anchor K = 128 cosine-distributed gather directions → hit sender anchor or miss; per anchor sun visibility over 512 light-direction bins | yes: recomputed from geometry (no radiance, no reference) |
| C. Dynamic transport state | per anchor and current light: cosine-weighted incident radiance projected to real SH order 2 (9 × RGB). Sender radiance = a_θ(φ(s))/π · (E_dir(s) + E_ind(s)), with E_dir = E_sun·max(0, n_s·ω_L)·vis_bin and E_ind from Jacobi iterations over the same relations (multi-bounce). Per pixel: inverse-distance interpolation of ≤ 4 normal-compatible same-object anchors | yes: recomputed (no optimisation) |
| D. Reusable operator | L_o = v · d_θ(...) + Σ t_θ(...) ⊙ g(x): the indirect term is linear in the current incident state | no |
| E. Update semantics | edit → rebuild R^t (rays) → recompute anchor states per light → render. No parameter update and no target-scene supervision. Learned-parameter ownership: Θ = {φ_A, φ_B, heads}, all persistent; Surface ownership: per-object canonical fields; Transport dependency: R^t; Radiance propagation: Jacobi over R^t; Invalidation: every relation entry whose hit or visibility changed; Reconstruction: state recomputation only; Composition: per-pixel interpolation + operator | — |

**Intentionally cannot represent:**
- incident angular detail beyond SH order 2 (sharp glossy reflections of other objects);
- non-Lambertian senders;
- rotation or non-rigid edits (world-frame SH, translation-only edits);
- indirect variation finer than the anchor spacing (~1.5 cm);
- sender sun visibility finer than the 512 light bins.

**Falsifiers:**
- (i) static fidelity far below RNA at T0, i.e. representability;
- (ii) the updated relation state does not move held-out T1 radiance toward the reference (updated ≈ stale);
- (iii) recovery requires changing persistent parameters.

**Leakage risk.** The direct head sees (x, ω_L) and could absorb canonical indirect light. It is trained on T0 only and measured by the stale-state control.

**Supported already (worklogs 32–38):**
- stale indirect light at stationary receivers (D_T ≠ D_G);
- the physical Lego edit and its references;
- the Surface-Owned baselines.

**New and unvalidated:** every SPT-1 property. Glossy and other generalisation experiments are **evaluation targets of this batch, not prerequisites** for designing it.

**Code:** `experiments/spt_lego/` (new; no historical module is modified).

**Result (worklog 39, 2026-10-22): NOT SUPPORTED** under spt_v1. Static representability fails: held-out T0 is 22.23 dB vs RNA_T0 33.57 dB (−11.3 dB; fail line −2 dB).
- **Attribution:** shadowed (indirect-only) pixels are 17.7 vs 32.4 dB. Per material, the yellow subsurface plastic (41.6% of pixels) is 70× RNA's error and the two transmissive materials 4–4.5×. The free-space relation state has no path for object-internal transport. Reflective materials are 1.4–3.8× worse (unresolved).
- **Factorization mechanism (descriptive):** with bit-identical persistent parameters and no T1 supervision, recomputing only the T1 relation state gives F = 1.67 and R = 0.82 vs the candidate's own retraining on the changed region, and F = 2.9 / R = 0.92 on stationary sand receivers. The update costs about 7 s of relation rebuild plus 0.12 s per frame.
- **Absolute quality:** post-edit quality stays far below the Surface-Owned and RNA baselines.
- **Next open question (not authorised):** a persistent configuration-invariant channel for object-internal transport alongside the relation-dependent channel.


## 9. Completion Condition (per batch)

> Can a learned transport representation be divided into independently owned surface units whose complete state can be reconstructed while all other units are preserved, without unacceptable static-GI fidelity loss — and, in later batches, can recomputing only the transport-invalid units recover current-scene GI?

A negative result must be attributed to one of the following before it is interpreted:
- baseline limitation;
- candidate limitation;
- ownership isolation;
- optimisation/capacity;
- partition;
- supervision/correspondence;
- compositing.
