# WL80A — Lab-meeting summary (Coordinator-corrected version)

This version supersedes the chair's draft in `final_architecture_decision.md` §F. The corrections follow `coordinator_disposition.md` (critic HIGH-1/2, MEDIUM-3/4/7/8).

**What this is.** A **design-only thought experiment**. Nothing was implemented, trained or measured, and nothing here is a validated method.

## Research problem

ReST-GI splits light transport into an intrinsic, material-domain operator K and an explicit, scene-dependent free-space part C. K maps light entering a material domain to the light that first exits it, and must be reusable under lighting and scene changes. It must therefore predict responses for boundary source positions it was never supervised on.

**Test medium.**

| Property | Value |
|---|---|
| Geometry | 100 × 100 × 10 mm homogeneous slab |
| Mean free path | ≈ 1 mm |
| Albedo | 0.999 / 0.99 / 0.9 |
| Phase / boundary | Isotropic phase, index-matched boundary |
| Discretisation | 3,840 ports × 16 incoming / 64 outgoing angular bins |

## Existing observations (measured)

- **Supervised fit vs. transfer.** Learned operators fit their 16 supervised sources to the quality gates, or close to them; some variants were classified INSUFFICIENT FIT. None meets the gates on independent boundary-far sources: con_coarse ≈ 0.67–0.99 against a 0.25 gate.
- **Inconsistent predictions.** Physically equivalent source contexts (identical references within 0.2 %) receive predictions that differ by 12–33 %.
- **Training dynamics.** Transfer to unsupervised sources is often best early in training and degrades later.
- **Angular-basis experiments.** These reduced, but did not remove, orientation-dependent errors (WL79 / WL80). Earlier work interpreted the spatial source → transport map as the dominant problem (WL75). That is an interpretation, not a measurement.

## Why current approaches remain insufficient (interpretation)

In the learned families tried, nothing structurally connects unsupervised boundary contexts to the supervised ones. Physical constraints (passivity, linearity, symmetry) do not identify those contexts.

## Hypotheses reviewed

- **Track A — the proposed BGC-Transport** (pair-conditioned learned angular-spatial factors with first-flight features).
  - **Rejected** as a source-generalisation mechanism: nothing identifies out-of-orbit contexts.
  - Its corrections fix physical validity only (a proper ω·d energy bound, no redundant normalisation).
- **Track B — blind discovery** (no access to the project or to BGC) produced four mechanisms:

| Candidate | Description | Outcome |
|---|---|---|
| **C1 CP-K** | Explicit collision-probability operator: exact first flight → scalar collision-density solve → exact last flight; nothing learned | Selected |
| C2 SBG | Learned bounded scalar Green's function between exact flights | Deferred |
| C3 | Context-atlas tabulation | Rejected |
| C4 | Signed modal tail | Rejected |

- **Cross-review.** An independent panel (physics, generalisation, computation, three novelty searches, one critic) reviewed both tracks.

## Outcome and rationale

- **No validated learned K.** No candidate is defensible today as a validated *learned* K for unseen boundary sources, and none is novel as a mechanism. C1 is classical integral-transport numerics; within the project it is the learning-free member of a class proposed and rejected in WL77A. C2 is a revised earlier proposal.
- **C1 as a candidate operator.** C1 is the strongest physically grounded, falsifiable candidate *operator*. A new source changes only an exactly computable input, so its invariants hold by derivation (under stated conditions). Whether it reaches the gates becomes a discretisation-convergence question that can be tested without training.

## Evidence available today

Derivations only:
- the energy identity (passivity ≤ ω·d, given a non-negative, sub-stochastic flight kernel);
- source independence;
- a passivity defect in an earlier split form, found and repaired;
- the finding that both tracks derived the same exact operator.

Earlier project non-evidence (an MC emulation) suggests coarse lateral resolution mainly hurts absolute accuracy at edges and side faces, while preserving contrast.

**No accuracy or cost number exists for any candidate.**

## Claims not yet supported

- That C1 meets the gates on edge, corner and side strata, or on the dent signed change.
- That C1 is cheaper than path-traced re-rendering. The estimates are 18–51× slower per light than the learned operators. Lighting-only reuse at fixed geometry is the plausible regime.
- That any learned K generalises (C1 contains no learning).
- Novelty.
- That a learning-free operator is admissible as an FNTO K. That is a project-contract decision.

## Open question the panel identifies as most decisive

Can a learning-free collision-probability operator, with every discretisation setting frozen by reference-free criteria, reproduce the Monte Carlo reference K at gate accuracy on independent boundary-far sources and on the dent signed change?

| Outcome | Meaning |
|---|---|
| **Pass** | The target K, as defined by the reference, is deterministically computable from geometry on the scored strata (conditional on the reference convention). It would not attribute the earlier learned failures to any single cause. |
| **Miss while still converging** | Affordability-limited. Not a falsification. |
| **Miss at a plateau** | A convention, reference or lateral-bias floor. A fail for C1. |

This is a decision input for the user, not a recommendation to run it.
