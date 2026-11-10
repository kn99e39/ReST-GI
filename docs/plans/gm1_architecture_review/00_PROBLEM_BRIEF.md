# GM1 Architecture Review — Neutral Problem Brief

**Status:** design-review input, written by the Coordinator before any design track ran (2026-11-10). It selects no mechanism.
**Repository revision:** `Selective_Recompute` @ `c7b0698`.
**Batch type:** document-only. No implementation, training, rendering, benchmarking or GPU use.

## 1. The question

> Can complex, directionally dependent multi-bounce interreflection be represented and incrementally updated after geometry edits with sufficient fidelity and lower quality-matched lifecycle cost than appropriate alternatives?

The research objective requires all three at once:

1. **Fidelity.** High-fidelity complex global light transport: glossy / specular, multi-bounce interreflection, and eventually nonlocal material transport.
2. **Edit response.** Correct adaptation to geometry changes, including indirect effects on *stationary* receivers.
3. **Economics.** Quality-matched end-to-end updates without unnecessary reconstruction of still-valid learned transport.

GM1 is the first milestone on that path (see `docs/RESEARCH_ROADMAP.md`, milestone table): with the local material closure frozen, can **sharp glossy / specular, multi-bounce, editable-visibility interreflection** be represented and updated under a **rigid occluder / object edit**, including changed radiance on stationary receivers?

Local BSDF compression is not the contribution. The contribution has to be the global transport mechanism.

## 2. Fixed foundation

- **Local material baseline: frozen learned ER closure** (WL82–84). A compiled reflection closure with R = 8 lobes (Lambert + 7 anisotropic GGX), symmetric incident / outgoing attenuation and refraction-conditioned inner lobes. It is produced per material by a learned descriptor-to-closure compiler and is passive and reciprocal by construction. Dr.Jit `eval` / `pdf` / `sample` exist.
  - Qualified scope: one coated-conductor family, static scenes, limited rendering.
  - Measured: image-error bias floor of about 0.035–0.04 relative L1; GPU time per spp about 1.5× lower than the layered reference.
  - It is pointwise local scattering. It is **not** a nonlocal intrinsic K.
- **Physical K / C ownership** (`docs/Architecture.md`, `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md`). Intrinsic material / medium response (K) is source-independent. Free-space coupling (C) depends on current geometry and includes exposed same-asset interreflection. The two must own disjoint paths.
- **Baseline-role terminology** (`AGENTS.md`, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`). These are four separate roles:
  - Neural Full Recompute;
  - Native Reuse;
  - same-architecture Full Rebuild;
  - path-traced re-rendering.

## 3. What every proposal must specify

- Persistent learned and non-learned state; its spatial and angular information content.
- State lifecycle; what a rigid geometry edit invalidates; how the invalid set is discovered and at what cost.
- How newly created and destroyed light paths (changed visibility, new glossy paths) are handled.
- How directional information survives repeated transport (multi-bounce) without being collapsed.
- Coupling / multi-bounce computation; expected memory and computation scaling.
- Nonlocal visibility and stationary receivers.
- The integration boundary with frozen ER.
- Principal failure conditions.
- A testable advantage over a suitable control, and one decisive falsification experiment.

## 4. Reference levels and controls for any future experiment

- **REF-P:** original layered-material physical path tracing.
- **REF-ER:** frozen ER with full path tracing. It isolates global-mechanism error from ER's local bias.
- **Diagnostic controls:** stale / no-update state; same-architecture full state rebuild; the proposed incremental update; independent T0 / T1 references.
- **Target phenomena:** rigid geometry / occluder edits; sharp glossy / specular interreflection; several bounces; indirect radiance change on stationary receivers; newly visible and newly blocked paths.
- No T1 supervision or oracle information at runtime.
- No numerical speedup targets before workload and baseline costs exist.

## 5. Sources (read the originals; do not rely on this brief for facts)

- **Canonical:** `docs/README.md`, `docs/RESEARCH_CENTRIC_TOPIC.md`, `docs/Architecture.md`, `docs/RESEARCH_ROADMAP.md`, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`, `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md`, `AGENTS.md`.
- **ER foundation:** `docs/worklogs/82_*`, `83_*`, `84_*`.
- **Historical interface / ownership / composition evidence:** `docs/worklogs/46_*` … `56_*`.
- **Historical nonlocal representation / reuse / economics evidence:** `docs/worklogs/57_*` … `80_*`. In particular:
  - 59 — persistent F/S;
  - 67, 69, 70 — external baselines and economics;
  - 52 — learned C;
  - 54, 55 — explicit C resolution, CAK;
  - 58 — sharp-GGX angular preflight.
- **Earlier multi-agent reviews and kill-search records:** `docs/worklogs/77A_*`, `80A_*`, `80B_*`; `docs/plans/wl77a/`, `docs/plans/wl80a/`, `docs/plans/wl80b/`.

## 6. Evidence discipline

Label every statement as one of:

- **ESTABLISHED FACT**: canonical contract or verified measurement, with source.
- **HISTORICAL EVIDENCE**: an older result, with its original scope.
- **DERIVATION**
- **INFERENCE**
- **HYPOTHESIS**
- **SPECULATION**

Rules that follow from this:

- A historical negative result is a possible failure mechanism. It is not a measured defect of the WL84 ER baseline.
- A mathematically exact identity establishes neither an efficient approximation nor a runtime advantage.
- Do not fabricate measurements. Do not infer novelty from the absence of an exact prior match.
