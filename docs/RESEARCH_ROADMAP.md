# Research Roadmap — Dynamic Learned Transport Reuse

**Current as of:** 2026-11-02. **Role:** milestone dependencies, current bounded scope and next decision gate. [Research objective](RESEARCH_CENTRIC_TOPIC.md), [architecture contracts and candidate status](Architecture.md), and [evaluation strategy](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md) have separate owners.

## Current frontier

The latest completed controlled intervention is [WL64](worklogs/64_WORKLOG_DIRECTIONAL_CONTRAST_SUPERVISION.md): NOT SUPPORTED. Fixed-capacity contrast supervision lowered noisy training-pair risk but worsened both original and independently qualified new-field directional fidelity. The [WL63 physical component audit](worklogs/63_WORKLOG_PHYSICAL_BALLISTIC_SCATTERED_ATTRIBUTION.md) localizes qualified Pair7 error predominantly to learned collided transport, not the direct ballistic B output; it does not identify why that learned response fails. [WL60–62](worklogs/60_WORKLOG_FROZEN_K_FIDELITY_ATTRIBUTION.md) record earlier diagnostic and label-precision findings. [WL59](worklogs/59_WORKLOG_PERSISTENT_INTRINSIC_FS.md) remains the F/S correctness baseline without qualified complete GI fidelity; the [WL56](worklogs/56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md) physical GI floor remains open. [WL58](worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md) remains a separate sharp-GGX representation preflight.

[WL67](worklogs/67_WORKLOG_BASELINE_SUPERIORITY_EVIDENCE.md) executed the first quality-matched external comparison: on no workload does ReST-GI show a measured quality-matched advantage. On the WL56 medium fixture, Cycles Full Recompute meets the complete 3% contract at 4 spp in 4.7 s for four views, while the exact oracle Selective update takes 190 s and fails it (9/12); the oracle is ahead only in per-port stationary-receiver signed-change precision and in a 0.69 s rigid resident update that fails the contract. On the WL22 teaset, 64-spp path tracing (0.5 s) beats every frozen or refitted RNA/8DNA output; ReST-GI has no K there, and TransGI has no released implementation.

[WL69](worklogs/69_WORKLOG_COMMON_WORKLOAD_EXTERNAL_BASELINE.md) established one common workload with an external neural baseline: the WL55 floor medium fixture under a rigid edit, with released 8DNA. ReST-GI CAK meets the 3 % contract on both states; 8DNA does not (3–7/12, static blue bias). Quality-matched Cycles Full Recompute still dominates ReST-GI (16 spp in 2.4 s vs a 24 s update), so the outcome is B. RNA, Neural Radiosity and TransGI could not run this workload.

No learned corrective recipe or successor architecture is adopted. Work on `Selective_Recompute` only; no branch/worktree change without user consent.

## Current decision boundary

The immediate unresolved hypothesis is the cause of inaccurate learned collided directional response: noisy/biased supervision, objective-induced fitting or generalization, and constraints of the spatial/angular/geometry-conditioned family remain confounded. Do not repeat WL63 tagging or re-run the rejected WL64 fixed loss. Select at most one prospectively controlled differentiating intervention, with genuinely independent physical labels and held-out qualification, before changing representation or retuning. Keep F/S update efficiency as a required later objective, not a parallel change. The user plans substantial experiments after the 2026-10-31 14:00 KST agent-token reset, with a 2026-11-02 competitive-evidence target; neither timing nor the target relaxes physical or matched-cost evidence. No successor, sweep, or coupled solve is automatically authorized.

## Longer-term decisions, without implementation permission

- Resolve the complete physical quality floor and the shared angular/deformation response limitations before claiming learned integration readiness.
- Revisit rigid motion, local deformation, global bending/twisting and self-contact at each major design decision. Rigid material-frame evidence is conditional; the tested learned local-deformation recipes fail their complete response contract; global deformation and contact generality remain unestablished.
- Consider interpretable dependency discovery and optional learned scoring only after physical coverage and representation feasibility are established, with algorithm-only and oracle controls.
- Make quality-matched external comparisons, including TransGI, only after reference, workload and accounting alignment. Plan durations from measured costs; venue planning does not relax gates. WL67 established the comparison contract and measured it on two aligned workloads; any superiority claim must beat quality-matched Full Recompute measured on the same workload, views and hardware, which on small fixtures is a few-second path-traced update.

## Updating this roadmap

Replace the current frontier when a user direction decision or completed gate changes it. Link the owning plan, protocol or Worklog; do not append a result diary. A newly proposed mechanism needs its own explicit decision. Preserve failed conditions, amendments and provenance at their original evidence locations.

The former batch program, resolved baseline prerequisites and chronological decision record are preserved in the [source archive](history/living_document_cleanup_2026-10-29/README.md).
