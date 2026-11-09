# WL80A — Coordinator disposition of the independent critic's findings

**Author.** The Coordinator, 2026-11-09.

**What stays untouched.** The chair's decision record (`final_architecture_decision.md`) and the critic's report appended to it are preserved verbatim. The sealed Track A / B records are untouched.

**What this file does.** It records which critic findings the Coordinator accepts. Where a correction changes a statement, the corrected statement below **supersedes** the chair's wording for any use of this study, including the lab-meeting summary in `lab_meeting_summary.md`.

**What it does not do.** No new analysis was performed and no candidate was re-ranked.

## Disposition

| Critic finding | Disposition | Corrected statement |
|---|---|---|
| **HIGH-1.** The lab summary inverts the predeclared outcome rule | ACCEPTED | A miss that is still converging at the finest affordable level is **affordability-limited** (not a falsification of the factorisation). A miss with observed order < 0.5 (a plateau) indicates a **convention, reference or lateral-bias floor** and counts as a C1 fail (first-experiment design §1, outcomes ii / iii). |
| **HIGH-2.** A C1 pass is overclaimed | ACCEPTED | A pass would show only that **the target K, as defined by the Monte Carlo reference, is computable from geometry at gate accuracy on the scored strata**, conditional on the reference convention (radiance weighting, port / bin conventions). It would **not** attribute the historical learned-family failures to "learning" rather than data volume, invariance, optimisation or the angular basis; no candidate measures those shares (unresolved B10). |
| **MEDIUM-3.** The primary designation is weakly supported against the brief's learned-K question and WL77A's H3 rejection | ACCEPTED (designation qualified) | **No candidate is defensible today as a primary learned-K answer.** C1 CP-K is designated only as the primary *candidate operator / diagnostic instrument* for a future decision. WL77A rejected H3, the same operator class at θ = 0, and those grounds are **not refuted** here: admissibility of a learning-free operator as an FNTO K is an untaken human decision; the apply cost meets the cap only with an unbuilt kernel; scope is g = 0, homogeneous, index-matched. These become explicit questions of any first experiment rather than being treated as resolved. C2 SBG stays a conditional secondary (a revised WL77A H2). |
| **MEDIUM-4.** Existing project non-evidence on C1's lateral bias was omitted | ACCEPTED | `docs/plans/wl77a/specialist_physics.md` §7 (non-evidence MC emulation) already bounds the mechanism. Isotropic voxels: red energy +2.1 / +6.0 / +17 % at 0.5 / 1.0 / 2.5 mm. 2.5 mm lateral snap with exact depth: interior native 0.096 / coarse 0.065; edge 0.20 (energy −11 %); side 0.36 (−23 %); pair contrast 0.05–0.08. Read as: contrast is largely preserved; absolute error and boundary strata are at risk. Identity restated as "C1 is the WL77A H3 θ = 0 operator class with a revised discretisation". |
| **MEDIUM-5.** Disagreement 5 (red context-equivalence length) is decidable from existing facts | ACCEPTED | The slab-mode lengths (≈ 3.6 / 3.1 / 1.6 mm) are supported by the brief's measured out-of-128-NN energy fractions (5.5 / 3.2 / 0.1 %; a K₁ tail estimate gives ≈ 4.5 / 2.7 / < 0.1 %). An 18 mm infinite-medium length would predict ≈ 70 %. Near-boundary diffusion caveat retained. |
| **MEDIUM-6.** The passivity bound is stated unconditionally | ACCEPTED | ΣK ≤ ω_c·d holds at convergence **provided W ≥ 0 entrywise and 1ᵀW ≤ 1**. Linked to chair amendment 2 (positivity-preserving far field) and to the "passivity breach" falsifier. |
| **MEDIUM-7.** Cost anchors are mislabelled | ACCEPTED | A75 single-light ≈ 1.34 ms is **amortised** (0.086 s / 64; DERIVATION), not measured. The measured single-light anchor is FIA 0.0039 s, against which the C1 estimate (0.07–0.2 s) is 18–51×. "3–10× of 0.086 s" reads 2.3–9.3×. The estimates assume an accelerated solve: plain iteration at red ρ = 0.974 needs ≈ 524 sweeps to 1e-6, and acceleration robustness is unverified (B6). "≈ 60 GPU-h full MC tabulation" is a DERIVATION (3,840 × 16 × 3.5 s), not a FACT. |
| **MEDIUM-8.** FACT overstatements | ACCEPTED | (a) Learned operators fit their supervised sources "to the gates or close to them"; WL73 (all arms) and WL76 FIA were classified INSUFFICIENT FIT. (b) The BND_FAR range spans 0.67–0.99 (FIA development 0.96–0.99). (c) "The spatial geometry → transport map is where transfer fails" is WL75's **interpretation** (INFERENCE), qualified by WL79 / WL80. |
| **LOW-9.** Invariants claim too broad | ACCEPTED | C1 is the only candidate with **no fitted quantity** and exact D4h without Reynolds averaging. C2's linearity, positivity and D2 passivity are also structural. |
| **LOW-10.** "Deployable: yes" | ACCEPTED | Relabelled "applicable (inference-time information available); deployability unmeasured". |
| **LOW-11.** Lateral-bias wording; "red averages ~38 collisions" | ACCEPTED | Variance (diffusion-coefficient) inflation h²/12–h²/6 against 2/3 mfp² per flight, i.e. 9–18 % in length. "38" is 1/(1−ρ) of the fundamental mode, an INFERENCE, not a measured mean collision count. |
| **LOW-12.** Labels justified by headcount | ACCEPTED | Agreement between reviewers is recorded separately from epistemic labels. No selection rested on a count (critic's finding). |
| **LOW-13.** Unfair to Track A on R-a | ACCEPTED | The Track A audit already noted the noise-free 0.21–0.22 ring floor; only the revised-architecture record cites 0.45. |
| **LOW-14.** The reciprocity disagreement talks past itself | ACCEPTED | A difference of scope (constraint in C3 / C4 versus aggregated diagnostic), not a contradiction. |
| **LOW-15.** Process and wording | ACCEPTED | The chair read 4 of 10 sealed files in full and relied on panel citations for the rest (disclosed). The "most important next experimental question" is rephrased as "the open question the panel identifies as most decisive". The chair's note about a runtime status question belongs to the workflow log. |

## Final position of the record (after this disposition)

- **Track A.** BGC (original and revised) is not defensible as a source-generalisation mechanism. Its revisions improve physical validity only (ω·d bound, no redundant ρ, explicit measure).
- **Track B.** C1 CP-K is the strongest physically grounded and falsifiable mechanism. It is learning-free: classical integral-transport numerics, internally the WL77A H3 θ = 0 class. C2 SBG is a conditional learned secondary; C3 and C4 are rejected.
- **Comparative.** No candidate is novel as a mechanism. No candidate is defensible today as a validated *learned* K for unseen boundary sources. C1 is the primary candidate *operator* for a future decision, subject to the open contract question and the unrefuted WL77A H3 grounds.
