# Architecture — FNTO Dynamic Transport Reuse

## 1. Objective and design boundary

Maintain physically credible global illumination after geometry edits by preserving still-valid transport and rebuilding invalid transport at matched output quality. Compressing K serves this objective; compression alone is not the objective.

FNTO separates **intrinsic optical-domain transport K** from **editable free-space coupling C**. The detailed physical contract is [FNTO Architecture Contract](FNTO_ARCHITECTURE_CONTRACT_V0_1.md). This document defines the central structure. Experimental candidates do not become accepted architecture merely by being implemented.

## 2. Physical transport ownership

An optical domain is a connected material/medium region or a local surface interface. Authoring objects and optical domains are different partitions.

- **K** maps spatially and directionally varying boundary incidence to outgoing response at the first exit into free space. It owns local surface response, transport inside the medium, internal scattering and transmission.
- **C** maps an outgoing boundary event through editable free space to the next boundary entry. It includes exposed same-asset and same-domain exit/re-entry paths.
- External illumination enters as a current incident boundary field. Direct and C-mediated incidence use the same intrinsic response rule.

Every path segment has exactly one owner. Object identity must not freeze exposed interreflection or merge editable air transport into K. Internal response must not be gated by visibility of the output point to a direct source.

## 3. Boundary interface and coupled solve

For each domain a:

```text
i_a = e_a + sum_b C_ab(G_current) b_b
b_a = K_a(G_intrinsic, material)[i_a]
```

Boundary fields retain material-coordinate position, incoming/outgoing direction and physical flux/radiance normalization. Source and receiver mappings conserve area-weighted flux and respect current normal/tangent frames. Near-contact spatial information must survive both sides of the K/C interface.

At fixed geometry/state, K is linear, nonnegative and passive. Scattered and unscattered (ballistic) contributions are distinct path families. Camera readout accounts for both without adding the ballistic contribution twice. Repeated composition must remain stable under admissible incident fields.

## 4. Parameter, state and radiance ownership

| Component | Meaning | Geometry-edit behavior |
|---|---|---|
| Reusable rule parameters | Source-independent intrinsic material/medium response | Fixed across the supported edits; no post-edit retraining |
| Geometry-dependent intrinsic state | Information required to apply K on the current intrinsic geometry | Preserve only with a valid dependency/correspondence certificate |
| Cached transport coefficients | Computed coefficients derived from rule and geometry state | Invalidate with their actual dependencies |
| Free-space state | Current visibility, directions and transport relations for C | Update for affected editable air paths |
| Incident/outgoing radiance | Response to current external and inter-domain illumination | Recompute through the current coupled solve |

A rule/state split inside K is a research hypothesis, distinct from physical K/C ownership. No particular learned state representation or selective-update algorithm is adopted here. Reusing network weights is different from preserving computed transport state.

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

This baseline reconstructs learned owned models; FNTO seeks reusable response rules with current transport state. Their update semantics must not be combined silently.
