# GM1 architecture review — 60 — Coordinator synthesis

**Status:** design-only (2026-11-10), `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. No candidate is adopted; no canonical document was changed; nothing is authorised.

**Basis.** I read the three cross-review responses (`40_`–`42_`) and the provenance audit (`50_`) in full, the agents' summaries of the designs and reviews, and the classification tables of the novelty searches. Detailed derivations are cited through the reports, where the audit confirmed them (53 of 62 repository claims CONFIRMED, 8 PARTLY, 1 INACCURATE, none fabricated; arithmetic reproduced).

**Caution.** Agreement among agents is not evidence of physical correctness or novelty. Every cost figure below is an estimate (EST) or a derivation (DERIV), never a measurement.

## 1. Disposition of the three tracks

| Track | Sealed proposal | After review and cross-review |
|---|---|---|
| A — Directional Delta Transport | Rejected as stated by its own designer; a narrowed "discover by delta, overwrite by regather" form (DDT-D) on a tabulated outgoing state | **Withdrawn as a mechanism.** Two independent FATAL findings were accepted: the 4D outgoing state scales as α⁻⁴ (about 38 GB at α = 0.05 for a small fixture, DERIV); the dirty fraction needed for break-even lies below its geometric lower bound (DERIV / EST). What survives is a solver policy (residual-ordered overwrite, HYP-Δ) that needs someone else's state. |
| B — Path-space / reservoir reuse | Declared itself not a mechanism: a retracer with sampling memory; no distinction from GRIS / ReSTIR PT, difference re-rendering, ray-segment invalidation | **Confirmed as a control only.** PR3 and PR2's economic claim are withdrawn. The paired same-seed T0 / T1 path log survives as an exact, learning-free measurement. PR1 is a lower bound on a warm path-traced comparator; the strongest published comparator is residual path integrals (Xu et al. 2024). |
| C — LPTG (lobe-partitioned transport graph) | Persistent band-limited incident field for rough lobes + stored replayable chains for sharp lobes, segment-certified invalidation, F ≡ S | **Withdrawn as a GM1 mechanism.** Three accepted objections: the visible sharp highlight comes from fresh per-frame chains, so the state holds no sharp content (a retracer for the sharp part); the field is point-sampled and aliases sharp-sender → rough-receiver transport; the per-edit re-solve is dense, so S ≈ F in cost. It survives as an admission census, as REF-ER path-class instrumentation, and possibly as a learning-free physical baseline (human decision). |

The identity `(I − K C1) Δb = K(Δe + ΔC b0)` is exact and carries no cost content. It is incremental radiosity restated directionally, and it relocates the problem into storing `b0` directionally.

## 2. What the review established (as inference, not measurement)

1. **No design holds sharp directional transport in persistent state.**
   - At the sharpness of the qualified ER family, the angular requirement scales as about α⁻². A 4D outgoing state scales as about α⁻⁴ (DERIV in `21_`).
   - A practical persistent field is band-limited near α\* ≈ 0.2–0.35 at a few hundred directions. Everything sharper is retraced.
   - The roadmap states that a full retracer does not complete GM1.
2. **The headline GM1 phenomenon is carried by no design.**
   - The phenomenon is a newly created sharp chain onto a rough stationary receiver.
   - The 65536-spp reference itself is estimated to be noise-limited there (EST in `21_`).
3. **Selective update is not where the cost is, on paper.**
   - Dirtiness spreads through the Lambert / broad part after one rough bounce (this rests on an unchecked Lambert weight).
   - The solve is then dense.
   - This is the same pattern as the historical record: WL56 / WL59 common work dominates; WL67 / 69 / 70 path-traced re-rendering is the binding comparator.
4. **No mechanism novelty anywhere.**
   - Every central mechanism is classified ALREADY KNOWN (see `35_prior_art_matrix.md`).
   - What remains is application novelty or unmeasured utility.
5. **Nothing global is learned in any design.**
   - Even a full pass could not be reported as selective recompute of *learned* transport.
6. **Every conclusion depends on two unmeasured numbers:**
   - (i) the albedo-weighted fitted-lobe roughness census of the frozen ER closures, including the Lambert weight;
   - (ii) the cold REF-ER time-to-quality ladder on a GM1-type fixture. At WL84 scale (about 10 ms GPU per 64 spp at 256², 0.4–1.2 s Python tracing per launch) no cost claim is testable.

## 3. Coordinator judgment

**Recommendation class: C with D.** No candidate is sufficiently defensible as the GM1 global mechanism, and specific missing evidence must be resolved before an architecture is selected.

I do not recommend the initial DDT proposal, a hybrid, or LPTG.

- **Why not a hybrid.** The only technically coherent one is C's link graph with A's residual ordering. It inherits C's representational limits and has predicted-negative economics. Combining it would create a third unmeasured estimator.
- **Why not "pick the least bad".** The roadmap requires one falsifiable *global representation / update mechanism*. After cross-review each track's own advocate declines candidacy, on arguments that survived audit. Selecting one anyway would repeat the WL76–80 pattern of testing a mechanism whose admission conditions were never measured.
- **What is defensible to test first, if the owner wants a test.** A mechanism-neutral characterisation. All three advocates and two reviewers reached it independently:
  1. Lobe census of the frozen closures (zero render).
  2. Cold REF-ER time-to-quality ladder on one GM1 fixture, with REF-P and the ER gap `g` reported first.
  3. Paired same-seed T0 / T1 REF-ER path log with R / S lobe-class labels (path-class shares of radiance and of |Δ|, dirty fractions, null-edit control).

  It decides admission for every stored-state design at once, and whether the workload has economic room at all. It is **not** a GM1 mechanism test, and the roadmap text currently has no slot for it (`50_` §5.1).

**My own reservations about this synthesis.**

- The withdrawal arguments use estimated constants. A factor of 3 can open or close a window.
- α\* estimates span a factor of about 3 across reviewers.
- "Every ER material has a Lambert lobe" is a parameterisation fact, not a fitted-weight fact.
- The review searched a design space of three tracks. It does not show that no defensible mechanism exists — for example a *learned* global state whose angular content is conditioned on geometry rather than tabulated. No track proposed one, and that absence is not evidence against it.

## 4. Decisions that belong to the human owner (from `50_` §5, §7)

1. May a mechanism-neutral workload / closure characterisation precede the GM1 mechanism decision? (The roadmap, read literally, says no.)
2. Must the GM1 global mechanism contain a learned component, or is a learning-free global solve around learned ER admissible as candidate, or as baseline only?
3. The sharpness scope of GM1: which lobe-roughness range and reflector distances does the claim cover?
4. Are multi-view / multi-frame-per-edit workloads inside GM1? The only economic windows found for stored states lie there.
5. Workload scale: a discriminating fixture needs a cold REF-ER of several seconds of kernel time.
6. A name and accounting rule for *warm* path-traced re-rendering as a comparator role.
7. One path-termination convention shared by every arm and reference.

## 5. What would falsify this recommendation

- **A census showing most reflected energy in lobes at or above the reachable α\*.** A rough-lobe field would then own most of the multi-bounce edit response, and a C-type state becomes worth specifying (as a baseline, or as a candidate if the owner admits it).
- **A path-class log showing that the stationary-receiver |Δ| lies mainly in broad-lobe classes.** Same consequence.
- **A specified mechanism with learned global state** that carries sharp directional content at practical size, with an admission argument that survives the α⁻² / α⁻⁴ scaling. None exists in this review.
- **Conversely:** if cold REF-ER reaches the ER-gap-anchored quality in about a second or less on the GM1 fixture, no reuse scheme of any track has economic room on that workload. The first question is then the workload and gate, not a mechanism.

## 6. Unresolved disagreements

See `50_provenance_audit.md` §7 (twelve items). The main ones:

- link-granular versus path-granular reuse headroom;
- the value of α\*;
- whether an output-sensitive solve is possible;
- the composition of the warm comparator;
- whether a masked cold re-render degenerates.
