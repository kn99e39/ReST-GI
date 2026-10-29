# Baseline Roles and Evidence Strategy

**Current as of:** 2026-10-29. **Role:** baseline roles, comparison controls, evidence limits and evaluation accounting. [Architecture](Architecture.md) owns candidate dispositions; the [roadmap](RESEARCH_ROADMAP.md) owns current bounded scope. This document neither grants implementation permission nor specifies replacement thresholds for frozen protocols.

## Baseline roles

| Baseline / control | Scientific responsibility | Limit |
|---|---|---|
| Quality-matched Full Recompute | Primary dynamic-update comparison; rebuild current intrinsic state and all required C/solve/readout state | Must meet the same physical fidelity contract as selective execution |
| Exact domain-selective oracle | Isolate work avoided by preserving unchanged optical domains | Oracle invalidation and Monte Carlo rebuild savings are not a deployable learned contribution; [evidence](worklogs/56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md) |
| Stale K / stale state | Negative control for intrinsic invalidation and signed receiver response | Parameter identity does not imply physical validity |
| Same-architecture F/S | Isolate selective persistent-state updates with identical rule parameters, inputs and quality requirements | Reference data and solved training-light responses cannot be privileged runtime state |
| TransGI | External dynamic-GI comparison after workload, quality, reference, hardware and lifecycle alignment | No advantage claim before an aligned comparison |
| RNA (`66b5b098`, upstream configuration) | Initial implementation substrate and static full-material quality control; native Lego credibility established in [baseline qualification](worklogs/35_WORKLOG_RNA_REFERENCE_REPRODUCTION_LEGO.md) | Not a permanent final-model constraint; credibility does not transfer automatically to other material regimes |
| Surface-Owned complete-unit models | Separate relearning baseline, including full/selective reconstruction, capacity, composition and REINIT controls | Uses post-edit supervision; does not demonstrate state-only updates. [Physical edit](worklogs/37_WORKLOG_ORACLE_SELECTIVE_RECONSTRUCTION_DIFFUSE_LEGO.md), [replication](worklogs/38_WORKLOG_B4_SEED_REPLICATE_SELECTIVE_VS_FULL.md) |
| SPT-1 and historical FNTO recipes | Immutable architecture-contrast evidence, including their negative outcomes | Never silently merge recipes, change roles or promote partial reuse evidence to full fidelity; [current dispositions](Architecture.md) |
| Cycles volume plus independent analytic controls | Physical intrinsic-transport reference for canonical homogeneous medium fixtures | Not a full-material production reference; [reference contract](worklogs/53_WORKLOG_PHYSICAL_INTRINSIC_TRANSPORT_REFERENCE.md) |

Implementation baselines are not scientific replications. RNA/Rain correspondence confounds, 8DNA/Teaset staleness, sparse relational-state failures, intra-model masking and complete-unit reconstruction answer different questions. Their scoped historical roles and original numerical evidence remain in the [source archive](history/living_document_cleanup_2026-10-29/README.md) and [Worklogs](worklogs/).

## Reference validity and comparison units

A credible baseline and independently validated target semantics precede candidate judgment. Freeze the scene/material regime, aggregation, checkpoint rule and noise/convergence controls prospectively. Report a baseline limitation separately from a candidate limitation; never repair a failed gate by changing its threshold or evaluation set after results. [Baseline attribution](worklogs/33_WORKLOG_RNA_STATIC_BASELINE_ROOT_CAUSE_ATTRIBUTION.md) and [qualification](worklogs/35_WORKLOG_RNA_REFERENCE_REPRODUCTION_LEGO.md) own the historical measurements.

Label every comparison with its renderer, build/device, machine, sample/clamping/filter regime and spatial/angular interface. Historical `hist` and corrected `r1` references are different regimes; original RNA/Surface-Owned/SPT-1 scores cannot silently be relabelled as r1. [Regime provenance](worklogs/42_WORKLOG_FNTO_VALIDITY_RESTORATION_R1.md).

Cycles BSSRDF vertex semantics cannot supervise a single vertex-independent recursively composed K without an explicit contract change. Preserve the homogeneous-volume reference distinction and the recorded renderer precision/film limitations. [Semantics attribution](worklogs/51_WORKLOG_FNTO_TRANSPORT_OWNERSHIP.md), [reference controls](worklogs/53_WORKLOG_PHYSICAL_INTRINSIC_TRANSPORT_REFERENCE.md).

Compare compatible area/port integrals and visibility masks. Point-sampled oblique boundary images require pixel convergence; partially visible ports and cameras starting inside a medium need separate validity handling. Resolve near-contact structure on both K and C. Include scattered and ballistic camera contributions exactly once. Diagnostic continuous oracles and fine-cell solutions are controls, not substitutes for an independent physical reference. [Spatial controls](worklogs/54_WORKLOG_EXPLICIT_C_SPATIAL_FIDELITY.md), [readout attribution](worklogs/55_WORKLOG_SPATIAL_INTRINSIC_K_INTERFACE.md), [amended masks](worklogs/56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md).

## Controlled comparisons and metrics

- Predeclare geometry, materials, illumination, cameras, splits, coordinate/correspondence conventions, target semantics, seeds, precision, training/checkpoint rules and acceptance thresholds in the owning protocol. Keep held-out edits out of fitting and selection.
- Isolate the declared mechanism. For Surface-Owned comparisons change the ownership boundary and necessary composition only; for persistent-state F/S use the same architecture and rule checkpoint. Report capacity confounds rather than hiding them.
- Evaluate unseen spatial and directional incident fields and all-boundary sender outputs. Zero/linearity/nonnegativity/passivity/cache tests are necessary but do not establish angular, deformation or coupled fidelity.
- Report absolute operator/image quality separately from full/selective equivalence, signed change and stationary-receiver recovery. Include HDR error, glossy/high-frequency, shadow/material and boundary/interior regions; declare aggregation and uncertainty.
- Track geometry-dirty versus transport-dirty support. For complete-unit reconstruction report complete learned-state ownership, unchanged parameter identity, declared output-influence/composition support and radiometric validity separately. Oracle dirty sets require declared noise, split, coverage and cost.
- Include selection/dependency maintenance, geometry preparation, normalization, cache build, K/C application, solve, readout, loading/checkpoint traffic, warm/cold lifecycle time and unique memory. Separate offline reference generation/training from runtime update/query costs; report hardware contention and convergence. Unit counts and parameter counts do not prove speedups.
- Preserve reviewer-visible ownership/seam maps, error and changed/stationary/unaffected-region views where applicable, with provenance in `results/evaluation/<worklog>/`. Scripts, gates and detailed metrics remain in their owning protocol/Worklog/results.

The original protocol owns its numerical gates. Use a valid spectral-radius estimator (A² iteration or Arnoldi for sign-paired modes), and distinguish numerical convergence from physical correctness; [estimator audit](worklogs/54_WORKLOG_EXPLICIT_C_SPATIAL_FIDELITY.md).

## Future selection and evidence breadth

A physically interpretable algorithmic spatial/directional hierarchy is a candidate-discovery preference, not an adopted transport solver. Search hierarchy, transport dependency graph, parameter ownership and invalidation scoring are distinct. Normal similarity or local distance alone cannot guarantee coverage, especially with third-party occlusion and nonlocal glossy/multi-bounce transport.

Before any separately authorised learned-selector test, establish physical-oracle candidate coverage. Compare algorithm-only and learned scoring on identical candidates, labels and held-out scenes. Report radiometrically weighted false negatives, false-positive work and total discovery/inference/recompute cost. A missed candidate is irrecoverable downstream. No runtime oracle-label leakage; any fallback and tolerance must be prospective.

One asset, material family, edit or candidate failure cannot establish project-level generality. Revisit rigid, local deformation, global bending/twisting and self-contact at major decisions; test sequentially. Prior art and novelty constraints are owned by the [research topic](RESEARCH_CENTRIC_TOPIC.md).

## Interpretation and stopping

Stop on invalid reference/baseline semantics, lost required information, failed ownership/correspondence, fundamental composition failure or broken controls. Attribute baseline, candidate, ownership, optimisation/capacity, partition, supervision/correspondence and composition causes before interpreting results. No rescue heuristic, threshold revision, automatic successor or adoption follows from a result.

The original failed WL57 reference gate, subsequent recovery evidence, its bounded negative learned result and the WL56 physical quality floor remain preserved. [WL57](worklogs/57_WORKLOG_LEARNED_INTRINSIC_K_RULE_STATE.md) owns these distinctions; abstract Rule–State superiority remains INCONCLUSIVE.

The [intrinsic-state reuse preflight](worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md) stopped on its finite angular representation before training or state-update implementation. Treat that as candidate admission evidence; it supplies no F/S lifecycle comparison and does not falsify abstract persistent-state reuse or the homogeneous-volume model.
