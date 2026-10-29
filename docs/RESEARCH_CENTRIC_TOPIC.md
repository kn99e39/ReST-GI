# Research Topic — Physically Valid Dynamic Learned Transport Reuse

**Current as of:** 2026-10-29. **Role:** research problem, objective, scope and scientific decision frontier. [Document ownership](README.md) defines the other sources of authority.

## Research problem and motivation

A learned GI model fitted to one geometry can retain stale interreflection, indirect occlusion, glossy response and multi-bounce transport after an edit. Supplying current local surface inputs does not necessarily repair those dependencies. Stationary receivers can change even when their geometry does not: geometry-dirty and transport-dirty sets are different.

**Scientific objective (ADOPTED):** maintain physically credible, high-quality learned GI after geometry edits by reusing transport that remains valid and updating invalid transport at favorable quality-matched end-to-end cost. Compression of an intrinsic operator is a means to this objective.

**Central hypothesis (INCONCLUSIVE; not adopted as a working model):** a source-independent reusable intrinsic transport rule can be separated from geometry-dependent within-domain state, and selective state reconstruction can improve lifecycle cost without losing physical or directional fidelity. This additional separation must earn its benefit beyond exact optical-domain selective recomputation; physical K/C ownership alone does not establish it.

The physical and representation requirements are owned by [Architecture](Architecture.md). The current bounded candidate and stage dependencies are owned by the [roadmap](RESEARCH_ROADMAP.md). This document grants no experimental authorisation.

## Scope and research constraints

- Preserve intrinsic optical-domain K versus editable free-space C ownership, spatial boundary information near contact, directional incidence/response, and scattered plus ballistic readout. Learn transport response to illumination, rather than one configuration's radiance.
- Consider rigid motion, local deformation, global bending/twisting and self-contact/contact changes as mandatory long-term coverage. Test sequentially; every major design decision must state the implications and untested limitations for all four.
- Do not infer non-rigid validity from fixed weights or geometric locality. Intrinsic changes may be nonlocal, dense or high rank. Parameter preservation, output preservation and radiometric validity are separate properties.
- Quality-matched Full Recompute is the primary dynamic baseline. Exact domain-selective and stale-K controls isolate avoided work and invalidation. Oracle recomputation savings do not establish a learned contribution.
- TransGI is an external comparator requiring aligned scenes, edits, materials, references, hardware and complete update/inference/memory accounting before any advantage claim. No such advantage is established.
- Preserve RNA as the initial research substrate and quality control, not a permanent final-model constraint. Surface-Owned complete-unit reconstruction remains a separate baseline: selection rebuilds complete owned models, never arbitrary cells, channels or layers in one jointly owned network.
- A negative candidate does not falsify the general ownership or reuse hypothesis. The historical requirement to investigate multiple meaningful architecture batches before a broad viability judgment remains part of the research rationale, not permission for automatic experiments.

## Current milestone and decision frontier

Exact domain-selective oracle recomputation can reproduce full recomputation on the tested edit, but complete physical GI quality remains unsupported. The fixed direct and Rule–State learned recipes are not eligible for integration; the abstract rule/state hypothesis remains INCONCLUSIVE. These are distinct conclusions, preserved in the [selective-oracle evidence](worklogs/56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md) and [completed learned-candidate evidence](worklogs/57_WORKLOG_LEARNED_INTRINSIC_K_RULE_STATE.md).

The [intrinsic-state reuse candidate](plans/INTRINSIC_STATE_REUSE_V1.md) has now stopped at its prospective angular representation gate: its finite interface cannot carry the declared sharp GGX response. This is a bounded representation admission failure, not a learned-model or homogeneous-volume failure. Persistent-state correctness, F/S fidelity and selective-update cost were not evaluated and remain INCONCLUSIVE. The [completed preflight report](worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md) owns the evidence; the [roadmap](RESEARCH_ROADMAP.md) owns the next decision boundary.

Broader open questions are whether a sufficiently accurate learned K can be composed stably, whether validity certificates permit useful selective work after non-rigid edits, and whether benefits survive global deformation and contact changes. None has a general affirmative answer.

## Research direction and novelty discipline

The priority historical design references remain 2007–2013 dynamic GI: Temporal Radiance Caching, Instant Caching, clustered GPU GI and Voxel Cone Tracing. They inform caching, invalidation and scheduling. Keep surface representation, radiance caching, learned parameter ownership, transport invalidation and sampling/scheduling distinct. The 1997 line-space hierarchy is prior art rather than the primary starting point.

An algorithmically constructed, physically interpretable spatial/directional candidate hierarchy remains a design preference for possible later dependency discovery. A search hierarchy is not the transport graph. Optional learned relation scoring must demonstrate value against algorithm-only controls on the same candidates; coverage failures cannot be repaired downstream. No particular hierarchy or learned selector is adopted.

Novelty cannot rest on a hierarchy, selective updating, per-region neural models, BSSRDF approximation or operator compression alone. A research claim needs correct physical ownership, demonstrated validity under edits and measured quality/cost relative to appropriate prior art. [Evidence strategy](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md) owns comparison discipline.

## High-level stop conditions

Stop a bounded route when its baseline/reference is invalid, its interface loses required information, ownership or correspondence cannot be maintained, composition fundamentally fails, or controls cannot be preserved. Attribute the failure before interpreting it. A route ending does not end the research direction.

Predeclare conditions and thresholds; never move them after results. No automatic successor, rescue architecture or production adoption follows from a gate result. A publication target (SIGGRAPH 2027; deadline not verified here) never overrides a scientific stop.

## Provenance

Historical Surface-Owned and SPT-1 framing, original user decisions, prior-art descriptions and superseded milestone statements remain in the [source archive and classification record](history/living_document_cleanup_2026-10-29/README.md). Complete experimental evidence remains in [worklogs](worklogs/); frozen conditions and raw results remain at their original locations.
