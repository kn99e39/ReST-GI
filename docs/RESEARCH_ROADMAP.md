# Research Roadmap — Dynamic Learned Transport Reuse

**Current as of:** 2026-11-09. **Role:** milestone dependencies, current bounded scope and next decision gate. [Research objective](RESEARCH_CENTRIC_TOPIC.md), [architecture contracts and candidate status](Architecture.md), and [evaluation strategy](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md) have separate owners.

## Current frontier

The latest completed controlled intervention is [WL64](worklogs/64_WORKLOG_DIRECTIONAL_CONTRAST_SUPERVISION.md): NOT SUPPORTED. Fixed-capacity contrast supervision lowered noisy training-pair risk but worsened both original and independently qualified new-field directional fidelity. The [WL63 physical component audit](worklogs/63_WORKLOG_PHYSICAL_BALLISTIC_SCATTERED_ATTRIBUTION.md) localizes qualified Pair7 error predominantly to learned collided transport, not the direct ballistic B output; it does not identify why that learned response fails. [WL60–62](worklogs/60_WORKLOG_FROZEN_K_FIDELITY_ATTRIBUTION.md) record earlier diagnostic and label-precision findings. [WL59](worklogs/59_WORKLOG_PERSISTENT_INTRINSIC_FS.md) remains the F/S correctness baseline without qualified complete GI fidelity; the [WL56](worklogs/56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md) physical GI floor remains open. [WL58](worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md) remains a separate sharp-GGX representation preflight.

[WL67](worklogs/67_WORKLOG_BASELINE_SUPERIORITY_EVIDENCE.md) executed the first quality-matched external comparison: on no workload does ReST-GI show a measured quality-matched advantage. On the WL56 medium fixture, Cycles **path-traced re-rendering** meets the complete 3% contract at 4 spp in 4.7 s for four views, while the exact oracle Selective update takes 190 s and fails it (9/12); the oracle is ahead only in per-port stationary-receiver signed-change precision and in a 0.69 s rigid resident update that fails the contract. On the WL22 teaset, 64-spp path tracing (0.5 s) beats every frozen or refitted RNA/8DNA output; ReST-GI has no K there, and TransGI has no released implementation.

[WL69](worklogs/69_WORKLOG_COMMON_WORKLOAD_EXTERNAL_BASELINE.md) established one common workload with an external neural baseline: the WL55 floor medium fixture under a rigid edit, with released 8DNA. ReST-GI CAK meets the 3 % contract on both states; 8DNA does not (3–7/12, static blue bias). Quality-matched Cycles **path-traced re-rendering** was faster than ReST-GI (16 spp in 2.4 s vs a 24 s update). The frozen WL69 protocol calls this outcome B, but its 8DNA arm reused the T0-trained slab asset without T1 retraining: **Neural Full Recompute was not measured**. RNA, Neural Radiosity and TransGI could not run this workload.

[WL70](worklogs/70_WORKLOG_SUN_STATE_REUSE_C1.md) tested validity-aware reuse of the first-order direct-sun state on that C1 edit. The cached T0 state passed an explicit dependency contract: rest mesh, medium, rigid pose, material-frame sun and cameras, estimator, and a bit-identical replay of source visibility. It was physically identical to the retraced T1 state, and four invalidation controls were rejected. The CAK update fell from 24.6 s to 2.09 s with unchanged quality, narrowly beating quality-matched Cycles **re-rendering** (16 spp, 2.40 s under WL69 timing; equal under Blender's internal render time). This is an **11.8x intra-ReST-GI sun-reuse acceleration**, not a measured advantage over 8DNA T1 neural retraining. Cycles still meets the bare contract at 1 spp in 1.2 s. The diagnostic cache is not adopted.

### Learned intrinsic-K source transfer (WL71–WL78, as of 2026-11-07)

Learned rank-16 W+UV reaches the fixed-basis oracle on its supervised sources (WL75), but does not transfer to independent boundary sources.

- **WL78** removed the operator's absolute-position and orientation inputs, giving the geometry-relative W+UV-Q. Supervised fidelity was preserved and interior and near-boundary transfer improved descriptively (interior independent sources now meet the WL59 gate). The frozen boundary-far endpoint stays **INCONCLUSIVE**: one of two seeds improved. The deformation response also degraded.
- **Remaining boundary failure:** physically equivalent boundary contexts related by a bin-frame rotation or reflection are still not mapped to equivalent responses. This attribution is unmeasured.
- **WL80** (2026-11-09) stopped at its Stage 2 gate with **ORBIT-BALANCED BASIS INADEQUATE**, before any learned training.
  - Refitting the global rank-16 basis on the exact rest D4h orbit halved the orientation gap: the worst transformed/original oracle ratio fell from 1.41 to 1.15.
  - The cost was a 2–18 % loss in original-orientation accuracy.
  - The boundary-rotation images still missed the frozen rule marginally (con_coarse 1.26–1.27× the G2 original level; WL59 abs_coarse 0.101–0.103).
  - The basis remains non-closed.
  - The learned 2×2 was not run, and the WL79/80 confirmation population remains unscored.
  - Whether to consider an equivariant angular factorisation is an open user decision.
- **WL79** (2026-11-08) stopped at its pre-training representability gate with **ANGULAR BASIS LIMIT**.
  - The rest-slab D4h symmetry contract is exact and was verified against references. Dented-pose bin maps are not exact under the first-triangle bin frames.
  - Under the frozen rank-16 G2 A/O, labelled oracles fit the exact symmetry images of the boundary training responses up to 1.26–1.41× worse than the originals (coarse absolute and contrast). Interior sources are unaffected.
  - The basis is not closed under the bin permutations.
  - Caveat: the basis was fitted on the original orientations.
  - No symmetry-trained candidate exists. Whether to change A/O or the operator class is an open user decision.
- **WL77A** selected no architecture candidate. It also found that the WL76/WL77 angular-failure attributions are confounded by q = 1024 single-scatter quadrature noise. That correction is an **unresolved attribution question**; the recorded WL76/WL77 verdicts are unchanged.

See [WL80](worklogs/80_WORKLOG_ORBIT_BALANCED_ANGULAR_BASIS.md), [WL79](worklogs/79_WORKLOG_ANGULAR_FRAME_EQUIVARIANCE.md), [WL78](worklogs/78_WORKLOG_SOURCE_IDENTITY_LEAKAGE.md) and [WL77A](worklogs/77A_WORKLOG_MULTI_AGENT_ARCHITECTURE_DISCOVERY.md).

No learned corrective recipe or successor architecture is adopted. Work on `Selective_Recompute` only; no branch/worktree change without user consent.

## Current decision boundary

The immediate unresolved hypothesis is the cause of inaccurate learned collided directional response: noisy/biased supervision, objective-induced fitting or generalization, and constraints of the spatial/angular/geometry-conditioned family remain confounded. Do not repeat WL63 tagging or re-run the rejected WL64 fixed loss. Select at most one prospectively controlled differentiating intervention, with genuinely independent physical labels and held-out qualification, before changing representation or retuning. Keep F/S update efficiency as a required later objective, not a parallel change. The user plans substantial experiments after the 2026-10-31 14:00 KST agent-token reset, with a 2026-11-02 competitive-evidence target; neither timing nor the target relaxes physical or matched-cost evidence. No successor, sweep, or coupled solve is automatically authorized.

## Longer-term decisions, without implementation permission

- Resolve the complete physical quality floor and the shared angular/deformation response limitations before claiming learned integration readiness.
- Revisit rigid motion, local deformation, global bending/twisting and self-contact at each major design decision. Rigid material-frame evidence is conditional; the tested learned local-deformation recipes fail their complete response contract; global deformation and contact generality remain unestablished.
- Consider interpretable dependency discovery and optional learned scoring only after physical coverage and representation feasibility are established, with algorithm-only and oracle controls.
- Make quality-matched external comparisons, including TransGI, only after reference, workload and accounting alignment. Plan durations from measured costs; venue planning does not relax gates. WL67 supplied aligned physical/reference comparisons, but a path-traced re-render is **not** a Neural Full Recompute. Any learned dynamic-update superiority claim requires an actually invalidated external neural representation and its native T1 rebuild/retrain/refit lifecycle, quality-matched to ReST-GI; if native T0-state reuse is valid, measure that separately. Compare Cycles/Mitsuba spp-dependent full-scene re-rendering under an additional, explicitly labelled cost axis. See [binding baseline-role contract](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md#mandatory-baseline-terminology-and-comparison-gate).

## Updating this roadmap

Replace the current frontier when a user direction decision or completed gate changes it. Link the owning plan, protocol or Worklog; do not append a result diary. A newly proposed mechanism needs its own explicit decision. Preserve failed conditions, amendments and provenance at their original evidence locations.

The former batch program, resolved baseline prerequisites and chronological decision record are preserved in the [source archive](history/living_document_cleanup_2026-10-29/README.md).
