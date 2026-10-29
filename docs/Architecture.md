# Architecture — FNTO Dynamic Transport Reuse

**Current as of:** 2026-10-29. **Role:** current physical/representation contracts, candidate status and unresolved design decisions. See [document ownership](README.md).

## 1. Objective and design boundary

Maintain physically credible global illumination after geometry edits by preserving still-valid transport and rebuilding invalid transport at matched output quality. Compressing K serves this objective; compression alone is not the objective.

FNTO separates **intrinsic optical-domain transport K** from **editable free-space coupling C**. This document owns the current central contracts. [FNTO v0.1](FNTO_ARCHITECTURE_CONTRACT_V0_1.md) is the unchanged **historical design contract dated 2026-10-23**; its original “not implemented” status is a statement at that date, not today’s implementation status. Experimental candidates do not become accepted architecture merely by being implemented.

**Current status:** FNTO K/C ownership remains the selected physical decomposition; no tested learned recipe has established the complete dynamic quality/cost objective. A within-K rule/state split is INCONCLUSIVE, and no particular learned representation is adopted. The [roadmap](RESEARCH_ROADMAP.md) owns bounded execution scope.

## 2. Physical transport ownership

An optical domain is a connected material/medium region or a local surface interface. Authoring objects and optical domains are different partitions.

- **K** maps spatially and directionally varying boundary incidence to outgoing response at the first exit into free space. It owns local surface response, transport inside the medium, internal scattering and transmission.
- **C** maps an outgoing boundary event through editable free space to the next boundary entry. It includes exposed same-asset and same-domain exit/re-entry paths.
- External illumination enters as a current incident boundary field. Direct and C-mediated incidence use the same intrinsic response rule; no source-identity or direct/indirect response branch is required. The rule models material/medium response to the current spatial/angular incident field, not a memorized X/Y reflection or a fixed irradiance image. Sharp specular/glossy response demands sufficient directional information; glossiness controls angular selectivity, not inherently the strength of C coupling.

The current physical reference for the canonical medium fixtures is the path-history-independent homogeneous RTE volume model, not the historical object-scoped Burley target. A single recursively applied K cannot silently absorb the renderer’s different observed/interior-vertex semantics. Evidence: [reference-semantics attribution](worklogs/51_WORKLOG_FNTO_TRANSPORT_OWNERSHIP.md) and [physical reference contract](worklogs/53_WORKLOG_PHYSICAL_INTRINSIC_TRANSPORT_REFERENCE.md). This reference choice is not a claim of full-material or production validation.

Every path segment has exactly one owner. Object identity must not freeze exposed interreflection or merge editable air transport into K. Internal response must not be gated by visibility of the output point to a direct source.

## 3. Boundary interface and coupled solve

For each domain a:

```text
i_a = e_a + sum_b C_ab(G_current) b_b
b_a = K_a(G_intrinsic, material)[i_a]
```

Boundary fields retain material-coordinate position, incoming/outgoing direction and physical flux/radiance normalization. Source and receiver mappings conserve area-weighted flux and respect current normal/tangent frames. Near-contact spatial information must survive both sides of the K/C interface.

At fixed geometry/state, K is linear, nonnegative and passive. Scattered and unscattered (ballistic) contributions are distinct path families. Camera readout accounts for both without adding the ballistic contribution twice. Repeated composition must remain stable under admissible incident fields. These are requirements, not a claim that the existing learned kernels satisfy them. Evidence for the directional, near-contact and readout requirements: [directional interface](worklogs/46_WORKLOG_FNTO_DIRECTIONAL_INTERFACE.md), [angular information limit](worklogs/47_WORKLOG_MINIMAL_DIRECTIONAL_NONLOCAL_SMOKE.md), [spatial interface and ballistic readout](worklogs/55_WORKLOG_SPATIAL_INTRINSIC_K_INTERFACE.md).

## 4. Parameter, state and radiance ownership

| Component | Meaning | Geometry-edit behavior |
|---|---|---|
| Reusable rule parameters | Source-independent intrinsic material/medium response | Fixed across the supported edits; no post-edit retraining |
| Geometry-dependent intrinsic state | Information required to apply K on the current intrinsic geometry | Preserve only with a valid dependency/correspondence certificate |
| Cached transport coefficients | Computed coefficients derived from rule and geometry state | Invalidate with their actual dependencies |
| Free-space state | Current visibility, directions and transport relations for C | Update for affected editable air paths |
| Incident/outgoing radiance | Response to current external and inter-domain illumination | Recompute through the current coupled solve |

A rule/state split inside K is a research hypothesis, distinct from physical K/C ownership. No particular learned state representation or selective-update algorithm is adopted here. Reusing network weights is different from preserving computed transport state. Source-swap response correctness at fixed geometry is necessary input-generalization evidence, not evidence of selective geometry-state reuse or update savings.

Runtime rule/state inputs exclude target K, reference images, source-object identities, pose-specific oracle lookups and baked final radiance. Different sources are represented by different incident fields.

## 5. Geometry changes and validity

Rigid motion preserves intrinsic state only when intrinsic geometry/material are unchanged and directions/correspondence are transformed consistently into the material frame. Free-space coupling and solved illumination can still change.

Non-rigid edits can alter nonlocal intrinsic paths. Geometric locality does not establish transport-state validity. Invalidation must include dependencies through unchanged endpoints and edited intervening geometry; it must not assume sparse, local or low-rank changes. Missing validity information requires conservative invalidation or a full rebuild.

The long-term design must accommodate rigid motion, local deformation, global bending/twisting and self-contact. Self-contact can change interface topology and domain ownership. Supported cases must have explicit correspondence and validity semantics rather than an implicit non-rigid reuse claim.

## 6. Full and selective execution contract

Full reconstruction F and selective update S use identical rule parameters, physical inputs, output interfaces and fidelity requirements. F rebuilds all relevant geometry-dependent state. S preserves certified-valid state and rebuilds every invalidated component. Neither policy receives privileged target data.

Selection, dependency maintenance, reconstruction, normalization, cache preparation, K/C application, coupled solving and readout belong to the relevant lifecycle cost. A state update that is cheaper only because it produces an unacceptable response does not satisfy the architecture objective.

## 7. Separate reconstruction baseline

Surface-Owned complete-unit reconstruction remains a distinct baseline. A unit owns its complete parameter set; rebuilding one unit preserves every other unit's parameters. Shared mutable parameters constitute an explicit coupling. Parameter preservation, output preservation and physical validity are separate properties.

This baseline reconstructs learned owned models; FNTO seeks reusable response rules with current transport state. Their update semantics must not be combined silently. Preserve exact ownership and declared output-influence/composition support, including any partition-of-unity blending. Evidence: [ownership](worklogs/32_WORKLOG_RNA_SURFACE_OWNED_TRANSPORT_UNITS_BATCH1.md), [composition](worklogs/34_WORKLOG_SURFACE_OWNED_BOUNDARY_ATTRIBUTION_AND_PU_COMPOSITION.md), [physical-edit reconstruction](worklogs/37_WORKLOG_ORACLE_SELECTIVE_RECONSTRUCTION_DIFFUSE_LEGO.md) and [seed replication](worklogs/38_WORKLOG_B4_SEED_REPLICATE_SELECTIVE_VS_FULL.md).

## 8. Candidate and architectural direction status

These are current dispositions with brief rationale, not a new adjudication. Worklogs own their complete measured scope and verdicts.

| Direction or candidate | Current disposition and rationale | Evidence / design |
|---|---|---|
| Complete-unit Surface-Owned reconstruction | Separate reconstruction baseline; conditional physical-edit evidence on limited coverage, not general dynamic validity | [physical edit](worklogs/37_WORKLOG_ORACLE_SELECTIVE_RECONSTRUCTION_DIFFUSE_LEGO.md), [replication](worklogs/38_WORKLOG_B4_SEED_REPLICATE_SELECTIVE_VS_FULL.md) |
| Intra-model feature masking | Rejected as the main ownership route for the tested jointly owned model; not a test of complete-unit ownership | [negative control](worklogs/30_WORKLOG_ORACLE_SELECTIVE_REFIT_FEASIBILITY_8DNA_TEASET.md) |
| SPT-1 | NOT SUPPORTED; free-space-only low-bandwidth state lacks required internal transport. No SPT-1-plus-SSS patch is adopted | [evaluation](worklogs/39_WORKLOG_SPATIAL_PERSPECTIVE_TRANSPORT_FACTORIZATION_SPT1.md) |
| FNTO v0.1 / r1 | NOT SUPPORTED under their respective protocols; coupled instability and then intrinsic fidelity failure prevent adoption of those recipes | [v0.1](worklogs/41_WORKLOG_FNTO_V0_1_IMPLEMENTATION_AND_PHYSICAL_EDIT_VALIDATION.md), [r1](worklogs/42_WORKLOG_FNTO_VALIDITY_RESTORATION_R1.md) |
| K_k44P | Preserved corrected historical kernel recipe, not a validated current solution; directional-sender and reference-semantics limits remain | [correction](worklogs/44_WORKLOG_FNTO_K_SIDE_FIDELITY_ATTRIBUTION.md), [directional failure](worklogs/46_WORKLOG_FNTO_DIRECTIONAL_INTERFACE.md), [semantics](worklogs/51_WORKLOG_FNTO_TRANSPORT_OWNERSHIP.md) |
| Directional K/C interface | CONDITIONALLY SUPPORTED; needs accurate/passive queried sender directions and adequate glossy sampling, not a guarantee for frozen K44 | [interface evaluation](worklogs/46_WORKLOG_FNTO_DIRECTIONAL_INTERFACE.md) |
| DASQ / composable L0C and L1C | Not adopted; spatial aggregation improvement does not establish a convergent reference-consistent composition | [aggregation](worklogs/49_WORKLOG_YELLOW_SPATIAL_AGGREGATION.md), [composition](worklogs/50_WORKLOG_DASQ_COMPOSABLE_K.md), [attribution](worklogs/51_WORKLOG_FNTO_TRANSPORT_OWNERSHIP.md) |
| Learned inter-domain C | Tested replacement NOT SUPPORTED on cost and transition fidelity; no learned C is adopted | [evaluation](worklogs/52_WORKLOG_LEARNED_INTERDOMAIN_TRANSPORT.md) |
| Contact-adaptive K/C cells (CAK) | Not adopted; spatial-resolution requirement retained, but the candidate misses its complete accuracy/tractability contract | [evaluation](worklogs/55_WORKLOG_SPATIAL_INTRINSIC_K_INTERFACE.md) |
| Exact optical-domain selective oracle | Baseline for avoided work; full/selective equivalence does not establish physical GI quality or a learned speedup | [evaluation](worklogs/56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md) |
| Fixed learned direct D / Rule–State R | Both recipes NOT SUPPORTED for integration; quality-matched runtime winner and abstract Rule–State superiority INCONCLUSIVE | [completed evaluation](worklogs/57_WORKLOG_LEARNED_INTRINSIC_K_RULE_STATE.md) |
| Dependency-certified persistent graph transport state (`intrinsic_state_reuse_v1`) | Not adopted; angular representation NOT SUPPORTED on the prospectively included sharp GGX probe. Candidate stopped before learned rule/F/S implementation; state correctness, fidelity and cost remain INCONCLUSIVE | [candidate plan](plans/INTRINSIC_STATE_REUSE_V1.md), [preflight closure](worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md) |
| Algorithmic spatial/directional discovery with optional learned scoring | Unvalidated design preference; no specific hierarchy or learned selector adopted. Candidate coverage and algorithm-only comparison required | [research constraints](RESEARCH_CENTRIC_TOPIC.md), [selection evidence contract](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md) |

A free-space radiosity solver, SH2-only Lambertian model, scene-baked monolithic radiance MLP or SPT-1 repair is not an interchangeable implementation of this contract. A genuine contract conflict must be reported before changing scope.

## 9. Unresolved design decisions

- **INCONCLUSIVE — learned K adequacy:** which spatial/angular representation and supervision can meet physical, directional and deformation fidelity while remaining passive under composed incidence? Passing algebraic constraints alone does not answer this.
- **INCONCLUSIVE — persistent within-K state:** can geometry-only dependency certificates preserve enough computed state to beat same-architecture full rebuilding at matched quality, including globally affected normalization and ballistic paths? WL58's sharp-GGX finite-interface stop left F/S unimplemented and unmeasured; it does not adjudicate reuse in the homogeneous-medium regime.
- **INCONCLUSIVE — complete physical GI:** the oracle baseline’s residual physical error remains open. Exact F/S equivalence must not be promoted to reference accuracy.
- **INCONCLUSIVE — general edit coverage:** global bending/twisting and self-contact, changing domain topology and correspondence, and broader refractive/anisotropic or full-material regimes remain unvalidated. Rigid and local fixture evidence does not close these questions.
- **Contract version boundary:** current reference semantics, explicit ballistic readout, near-contact spatial requirements and geometry-state invalidation are later requirements, not retroactive contents or validation of v0.1. A materially different detailed learned implementation needs an explicitly versioned successor design if adopted; the present candidate plan is not that adoption.

No architecture decision or scientific verdict changes through document reorganisation. Superseded design detail and the pre-cleanup source versions remain in the [historical archive](history/living_document_cleanup_2026-10-29/README.md).
