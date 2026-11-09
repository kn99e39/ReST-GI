# WL80B — Coordinator disposition of the independent critic's findings

**Author.** The Coordinator, 2026-11-09.

**Preservation.** `c1_final_verdict.md` (the verdict chair's record) and the critic report appended to it are preserved verbatim.

**Precedence.** The corrected statements below **supersede** the chair's wording for any use of WL80B.

**Scope.** No new analysis was performed and no conclusion was re-selected by count.

## Disposition

### HIGH

**HIGH-1 — The conclusion label overreaches.** ACCEPTED: label qualified, not replaced. The brief requires exactly one of four labels. The chair's **C1 PHYSICAL BASELINE ONLY** is kept as the closest category, with these binding qualifications:
- **As a method.** C1 is **not a method candidate**. It is rejected as a learned-K answer, and is learning-free, ALREADY KNOWN and inadmissible under FNTO v0.1 (l.6, 26, 103).
- **"Physical".** The word means *physics-based classical comparator*. It does **not** mean physical reference, oracle or label source. The project's physical reference remains Cycles volume / the MC walker.
- **Baseline role.** Its comparator role is **unqualified and UNRESOLVED**:
  - its bias and cost are unmeasured;
  - it is not executable on the dent within the stated caps;
  - BASELINE_ROLES has no row for it.

  Whether to give it any role is a human decision.

**HIGH-2 — "No admissible W on the dented pose" was stated as fact.** ACCEPTED. Corrected statement:
- No visibility-exact W within the recorded memory and cost caps has been specified.
- How much escape truncation misplaces has not been quantified.
- A matrix-free, ray-telescoped W with visibility meets every invariant and fails only on cost.
- The dent is therefore **UNRESOLVED (affordability / specification)**, not impossible.

**HIGH-3 — The falsification design is predictably non-decisive at 12 GB.** ACCEPTED.
- **Predicted outcome.** By the record's own estimates, explicit near-field storage at lateral L1 needs 13–19 GB. A three-level lateral convergence is therefore unaffordable without a matrix-free near field, which is new design work.
- **Default reading.** The predicted outcome under the design as written is **INCONCLUSIVE / affordability-limited**.
- **Far-field contradiction.** The verdict's "the far field has no feasible form" conflicts with the design's float64 Toeplitz far field at rest; that far field is an untested hypothesis. It must be reconciled before the design could be used.
- `future_falsification_design.md` is to be read with these three statements.

### MEDIUM

| Finding | Disposition | Corrected statement |
|---|---|---|
| **MEDIUM-4.** Privileged-information claim misattributed | ACCEPTED | Only the A75 / W+UV family lacked σ_t, ω, g = 0 and index matching. FIA shares the first-interaction physics (WL76 l.20, l.100–102). |
| **MEDIUM-5.** Cost comparisons at unmatched accuracy, internally inconsistent | ACCEPTED | See the cost note below. |
| **MEDIUM-6.** Role analysis contradicts itself (the "deterministic analogue of the WL56 oracle" vs "not an oracle"; the CD l.17 citation) | ACCEPTED | C1 does not qualify as an independent analytic control because its bias is unmeasured. The CD l.17 citation for the WL56 analogy is wrong; that analogy is the redesign author's own, not the WL80A coordinator's. |
| **MEDIUM-7.** Outcome table gaps; the "no new references" rule is broken; spent held-out population | ACCEPTED | Converging to a wrong limit (order ≥ 0.5 but floor above replica spread) is a **fail**. Mixed strata are classified per stratum. The S1 check against a q ≥ 65,536 MC single-scatter tally **would need new MC**, so it conflicts with the no-new-references rule and is marked so. Scoring the never-used rest strata would **consume** a held-out population a future learned-K test may need, and that cost must be weighed by the human decision. |
| **MEDIUM-8.** Agreement and concession used as evidence | ACCEPTED | Claims rest on their derivations or sources; agreement is recorded separately. |
| **MEDIUM-9.** Mislabelled [HM] | ACCEPTED | `specialist_physics` l.90–93, the WL77A coordinator SS floors and `candidate_H3.md` l.123 are non-evidence or design claims, not historical measurements. |

**Cost note (MEDIUM-5).**
- **Unmatched accuracy.** Every C1 cost comparison is at unmatched accuracy. FIA was INSUFFICIENT FIT and A75 does not transfer, so "N× slower than FIA / A75" is not a like-for-like statement.
- **Units.** MC costs 3.37 s per packet (WL71 l.147), which is one column. C1 figures are per light.
- **Consistent C1 estimate.**
  - Per-light: about 0.9–1.3 s for near field only, in invariant-exact cold mode (525–663 red sweeps; the "175" lower figure is withdrawn).
  - Far-field evaluations: about 1e12 per sweep matrix-free, otherwise unaffordable.
  - This assumes FP32 bandwidth-bound sweeps. A float64 far field on the RTX 3080 Ti runs at about 1/64 of the FP32 rate.
- **Matched accuracy** against MC is undefined until C1's bias is measured.

### LOW

| Finding | Disposition | Corrected statement |
|---|---|---|
| **LOW-10.** Original non-negativity overstated | ACCEPTED | Original C1 guaranteed entrywise non-negativity for option G only, in exact arithmetic. Option U's masked FFT can produce negatives. |
| **LOW-11.** Identity conditions stated loosely | ACCEPTED | The energy identity is algebraic: it needs the complement definitions and a convergent ψ. W ≥ 0 gives passivity through 1ᵀψ ≥ 0. 1ᵀW ≤ 1 gives non-negative escape and convergence. |
| **LOW-12.** Generality overstated | ACCEPTED | g ≠ 0 and Fresnel boundaries are **outside the scalar C1 form** and change state and cost. They are not "incompatible": CP / MOC solvers handle anisotropic scattering and reflective boundaries. |
| **LOW-13.** Citations unverified | ACCEPTED | Citations not re-fetched in WL80B are marked unverified. The critic located Roy, Hébert & Marleau, NSE 101 (1989) 217–225, for 3-D ray-tracing CP. The prior-art conclusion is unaffected. |
| **LOW-14.** H3 ground 1 mis-described | ACCEPTED | For C1, H3 ground 1 reads "transfer is carried by a classical solve". C1 has no closure to be vacuous. |
| **LOW-15.** "No file was written" | ACCEPTED | The verdict author wrote nothing. The `docs/plans/wl80b/` records are design files written by the Coordinator from the agents' returned texts. |
| **LOW-16.** "+31 % lateral D inflation" presented too strongly | ACCEPTED | It is a crude upper-bound hypothesis (new derivation). The nearest non-evidence (WL77A `specialist_physics` l.299–302) suggests smaller gate effects. Whether that emulation bounds the cumulative chain is ambiguous. |

## Final position (after this disposition)

**C1 PHYSICAL BASELINE ONLY — qualified:**
- C1 is **not a method candidate**. It is learning-free, ALREADY KNOWN (flat-source CP / first-collision source / MOC-style far field / successive orders), and inadmissible as a learned K under FNTO v0.1.
- **No revision was proposed.** Every identified defect has one of two fixes:
  - a specification correction inside known classical numerics;
  - a resolution / memory spend.

  Neither adds a method contribution.
- Its use as a physics-based classical comparator is **unqualified and a human decision**:
  - bias and cost unmeasured;
  - dent W unresolved within the caps;
  - the falsification design as written is predicted to be affordability-limited.
- Learned-K transfer remains the open research question. C1 neither answers nor attributes it.
