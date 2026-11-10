# GM1 cross-review — Track B (path-space / reservoir reuse)

**Status:** design-only cross-review response (2026-11-10), `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. Only this file was written. It authorises nothing.

**Read for this response (in full unless noted):** `00_PROBLEM_BRIEF.md`; designs `10_`, `11_`, `12_`; reviews `20_`–`23_`; novelty search `31_` in full, `30_` and `32_` in their verdict and mechanism-list sections only. Originals: `docs/worklogs/84_*` (final report section), `docs/RESEARCH_ROADMAP.md` (GM1 row, next-decision paragraph), `AGENTS.md`. WL52–59, 67–70, 82, 83 are used only through the reviews' and `AGENTS.md` summaries and are cited as such. No web search was run; literature is cited as verified in `31_` (its V / V-x marks), not re-verified here.

**Labels:** FACT, HIST (original scope only), DERIV, INF, HYP, SPEC, EST (estimate, not a measurement).
**Objection IDs** are prefixed by the review file: `20/B-1` is defect B-1 of `20_review_physical_correctness.md`.

## 0. Position in brief

- Track B is **not a GM1 mechanism** and does not become one after review. This was the sealed report's own verdict and every review upheld it. [FACT for the roadmap exit clause; INF for the mapping]
- After criticism, three things are withdrawn (PR3 as an unbiased arm, PR2 as an arm with its own economic claim, the `1/ρ` ceiling as a general bound) and one thing survives intact: the **paired same-seed T0/T1 path log** (the former Stage 1), which `23_` calls the only implementation-independent, noise-free quantity proposed by any track.
- Strongest competitor: **Track C (LPTG)**.

## 1. Strongest competitor: Track C

Track C, not Track A. Reasons:

1. **Track A has no window inside GM1 by two independent reviews.** State size ∝ α⁻⁴ (`21/A1`, FATAL); required dirty fraction below the geometric lower bound and no amortisation under motion (`22/A1`, `22/A2`, FATAL); its experiment cannot isolate its claim (`23`, UNSOUND as written). [DERIV in those reviews; nothing measured]
2. **Track A's repairs leave it no content of its own.** Its physically adequate repairs converge on Track C's stored links (`20` §2.4, `21` §2.4), and its on-demand form is residual path tracing (Xu et al. 2024, per `10` §13 and `30`), which is a Track B comparator arm that is stronger than PR2. [INF]
3. **Track C is the only design with an affordable persistent state** (build ≈ 1–3 high-quality frames, `22` §4.6, EST) **and a provable same-architecture F ≡ S** (`20` §4.1). It is the only one that can be called a representation at all. [DERIV / EST]
4. **Track C contains Track B.** Its sharp transport is "stored, replayable ray chains" with per-segment geometric certificates and suffix regeneration from stored random numbers (`12` §1, §3). That is Track B's M1 / M2 at link granularity with a world-anchored origin. Whatever is valid in Track B is already inside Track C. [DERIV]

## 2. Challenge to Track C's central assumption

**Assumption under challenge (C-A).** Splitting the *material's* lobes at α\* makes the persistent field band-limited at α\*, so a K-node field owns a meaningful, GM1-relevant share of the transport, and reuse of that state is where the advantage comes from (`12` §1, §5, §11).

**D1 — the quadrature integrand is a product; band-limiting one factor bounds nothing.** [DERIV]
`Φ_R(x, ω_o) = ∫ f_R(ω_i, ω_o) L_i(x, ω_i) cos θ_i dω_i`. A fixed K-node rule is accurate only if `f_R · L_i` is resolved at node spacing `h = √(2π/K)` (0.22 rad at K = 128). A product in angle is a convolution in angular frequency, so its bandwidth is the *sum* of the two; `L_i` contains sender highlights and occlusion edges and is not limited by α\*.
Counterexample (the GM1 headline): rough stationary receiver, sharp sender α_S = 0.05 (the mover), small light. Highlight solid angle seen from the receiver Ω_h ≈ 0.008–0.031 sr; node cell 2π/K = 0.049 sr (`21/C1`). With hit probability `p = Ω_h / Ω_node` = 0.16–0.64 per cache point, the per-cache-point relative RMS error of the highlight contribution is `√((1 − p)/p)` = **2.3 to 0.75**, and it is spatially coherent because nodes are shared in the material frame. `21/C1` and `20/C-1` give the expectation; the RMS is my arithmetic on their numbers. Removing it needs K ≈ 3·10³–10⁴, the 1/α² wall the partition was meant to avoid.

**D2 — the persistent field is correct only on the complement of the GM1 target.** [DERIV + INF]
Classify one indirect interaction by (receiver lobe class, angular width `w` of the incident feature):

| Receiver lobe | Incident feature | LPTG carrier | Status |
|---|---|---|---|
| R (α ≥ α\*) | broad, `w ≳ h` | persistent field | correct |
| R | sharp, `w < h` | persistent field | aliased (D1) |
| S (α < α\*) | any, inside a link | one frozen chain sample per node | a K-sample path store with no accumulation |
| S | any, at the camera | fresh chains per frame | a path tracer |

"Sharp glossy / specular interreflection" (FACT, roadmap GM1 row) means the receiver lobe is sharp or the incident feature is sharp, i.e. rows 2–4. Row 1 is broad-lobe transport. So for the GM1 target set LPTG is either Track B's mechanism (rows 3–4) or wrong (row 2). The roadmap's exit clause ("implementing a full retracer does not complete GM1") therefore applies to LPTG's sharp part exactly as it applies to Track B; `23/C-1` reaches the same point from the experiment side. With α\* ≈ 0.17–0.34 at K = 128 (`21/C2`, `22` §4.2(b)) and coat roughness drawn log-uniform on [0.05, 0.5] (FACT, descriptor box per `21` §1.1), 70–85 % of materials would have their coat highlight in S (INF in `21`; fitted-lobe census unchecked).

**D3 — where the field is correct, the selectively reused quantity is no larger than Track B's.** [DERIV on EST]
LPTG's selective saving is `(1 − f) · build − discovery`, with build ≈ 1–5·10⁸ rays ≈ one high-quality frame, and the field re-solve common to S and F (`22/C1`). Track B's PR1 reuses `(1 − ρ)` of one frame. Both reuse at most about one frame of rays per edit. LPTG's possible advantage is therefore the *cache* (view independence, several frames per edit), not edit reuse — its own outcome table already calls for dropping the selective claim in that case (`12` §12, last row).

**What would refute this challenge.** Two unmeasured numbers: (i) the fitted-lobe census of the frozen closures showing a large share of *non-Lambert* albedo at α ≥ α\*; (ii) a path-class log (section 6) showing that a large share of stationary-receiver |Δ| lies in row 1. Until then D2 is an inference, not a measured defect of LPTG.

## 3. Objections against Track B: accept or rebut

### 3.1 Physical correctness (`20_`)

| ID | Sev. | Response |
|---|---|---|
| 20/B-1 destroyed-highlight ghost in PR2 | MAJOR | **ACCEPT.** The ghost has standard deviation `σ0 · √(1/n − 1/N)` and vanishes only at n = N, which is PR1. The fallback must not be chosen from sampled covariance; and a decision taken from the same n paths that are then averaged is still a data-dependent estimator choice. Revision: the per-pixel fresh / reuse decision is taken from geometric dirty flags of a pilot subset disjoint from the estimation subset; ghost and negative mass are reported per mask. See 3.4 for PR2's resulting status. |
| 20/B-3 T0-domain shift for mover vertices in PR3 | MAJOR | **ACCEPT; PR3 withdrawn from the unbiased arms.** "One extra instance transform" was wrong: the shift and Jacobian for material-anchored vertices were never specified. Specifying them is new design work outside a bounded revision. |

Minor items 20/B-2, B-4, B-5, B-6 are accepted: unselected subset rule; Russian-roulette depth; null-edit control; `ρ` scope.

### 3.2 Directional fidelity (`21_`)

| ID | Sev. | Response |
|---|---|---|
| 21/B1 reuse radius ≈ 1.3·α·r | MAJOR | **ACCEPT for PR3 and for path repair; REBUT as a description of PR1 / PR2.** PR1 / PR2 never shift a vertex: reuse there is binary by segment validity and independent of α, so no "radius" exists. The reviewer's consequence stands for every arm and is accepted: all reuse is void on chains that touch the mover and on stationary sharp vertices whose next vertex is on the mover. The store carries no transferable directional information. |
| 21/B2 caustic-type births; the mask is reference-limited | MAJOR | **ACCEPT.** At p ≈ α²–4α² a 65536-spp unidirectional reference has ≈ 0.04–0.08 relative noise in that mask at α = 0.05 (EST in `21` §1.5), above the WL84 quarter-rule for a 0.10 gate. F3 "expected by derivation" is **withdrawn**: it could not be told from reference noise (HIST precedent: WL83's sole ER failure was a reference artefact, FACT WL84). The mask is declared INDETERMINATE unless separately qualified. |

Minor items 21/B3–B6 are accepted; B3 and B6 (lobe-width descriptor, Jacobian-dependent sampler evidence below α 0.1) become moot with PR3 withdrawn.

### 3.3 Economics (`22_`)

| ID | Sev. | Response |
|---|---|---|
| 22/B1 saving is a bounded constant | MAJOR | **ACCEPT**, with one precision: `c_touch / c_path` falls as paths get longer or scenes costlier, so the ceiling rises toward, and is bounded by, `1/ρ`. It never becomes asymptotic. This was the track's claim; the reviewer's numbers (about 3–5× PR1, below 1× for a large mover; EST) replace my illustrative f = 0.1. |
| 22/B2 store-read term missing | MAJOR | **ACCEPT.** Ceiling restated as `1 / (ρ + (c_touch + c_test) / c_path)`. Cheap ER lowers it. |
| 22/B3 WL84-scale saving below launch overhead | MAJOR | **ACCEPT.** At about 10 ms per 64-spp image (FACT, WL84) and 0.4–1.2 s per launch (FACT, WL84), F1 would trigger for harness reasons. Any cost stage needs a workload with cold REF-ER of at least seconds of kernel time, and work counts beside kernel time. (The reviewer's 2.4 ns per path assumes 256²; I also did not find the resolution in the WL84 report section.) |
| 22/B4 PR1 memory caps reachable quality | MAJOR | **ACCEPT.** PR1 as a *store* is limited to `N ≤ M / (100 B · P)`. But the headroom measurement needs no store: T0 and T1 can be traced with the same seeds in one paired kernel and compared per path on the fly [DERIV]. PR1 survives as a measurement, and as a cost arm only at modest N. |

Minor 22/B5, B6 accepted (anchor lifecycle; geometric Crofton estimate of f before any render).

### 3.4 Identifiability (`23_`)

| ID | Sev. | Response |
|---|---|---|
| 23/B-1 with X5 — PR2's T0 anchor uncharged | MAJOR | **ACCEPT, and it removes PR2's independent claim.** DERIV (decorrelated dirty paths, equal variance): if the n difference paths are a subset of the N paths that built the anchor, `Var ≈ σ²/N + 2qσ²(1/n − 1/N)`. At a gate-level anchor this equals the cold N-spp variance only at n = N (PR1) and is strictly worse for n < N in every pixel with q > 0. PR2 beats cold only with an over-converged anchor, i.e. uncharged T0 samples. PR2 is therefore reported only under a declared anchor rule with the excess charged; it makes no economic claim of its own. |
| X5(iv) masked cold re-render control | MAJOR | **ACCEPT the control; one prediction to test.** With path-dirty probability about 15 % for a small mover (`22` §3.2, EST), a pixel with N ≥ 64 paths almost surely holds a dirty path, so a *pixel*-granular mask degenerates to a full cold render in an enclosed glossy fixture unless thresholded (a bias). Path-granular replay's attributable margin is exactly the gap between pixel-dirty and path-dirty fractions; both are measured in section 6. [DERIV on EST] |
| 23/B-2 with X6 — F1 implementation-sensitive | MAJOR | **ACCEPT.** Work counts (rays by type, ER calls, segment tests, bytes touched) and uncontended kernel time; three-way verdict including IMPLEMENTATION-LIMITED and "economically non-discriminating workload". |
| 23/B-3 — ρ over-read | MAJOR (scope) | **ACCEPT.** §0's "reuse headroom" for "any selective scheme" is withdrawn (also `20/B-6`, `31` §4(a)). ρ is reported with radiance-weighted and \|Δ\|-weighted shares. |
| X1 matched quality biased vs unbiased | MAJOR (shared) | **ACCEPT** as a fixture rule. PR1 is immune (identical to the same-seed cold render); it matters for any state arm scored on the same fixture. |
| X2 quality-level anchoring | MAJOR (shared) | **ACCEPT.** Levels anchored to the in-scene ER gap g (rule fixed now, numbers after references exist). |
| X3 signed-change noise model | MAJOR (shared) | **ACCEPT.** Split-half instruments; geometric masks; INDETERMINATE regions. |
| X4 replicate seeds | MAJOR (shared) | **ACCEPT.** ≥ 8 seeds for every stochastic quantity. |

Minor 23/B-4, B-5, B-6 accepted (null and move-and-return edits; intervals; "not invalidated" instead of "Native Reuse" for ER; cold REF-ER and cold store build timed separately).

### 3.5 Novelty search (`31_`)

| Finding | Response |
|---|---|
| All mechanisms ALREADY KNOWN | **ACCEPT**; it was the report's own verdict. |
| Xu et al. 2024 missed; removes the "single known rigid edit" setting-level difference | **ACCEPT.** §13's "honest difference" is reduced to: frozen ER as BSDF and GM1 lifecycle accounting. Neither is a contribution. |
| "Saving bounded by 1/ρ" for any path-granular scheme: NOT SUPPORTED | **ACCEPT; withdrawn.** It bounds stored-sample revalidation only. Residual sampling from the mover and its ghost (Xu et al.) spends work in proportion to residual energy. |
| "Births covered at the fresh sampler's rate, no better" (general form): NOT SUPPORTED | **ACCEPT; narrowed** to PR1 / PR2 as designed. Whether object-origin sampling or caustic reservoirs help on a sharp ER highlight is unmeasured. |
| Warm comparator built from PR1 / PR2 is weaker than published practice | **ACCEPT.** PR1 is a *lower bound* on the warm path-traced comparator, not the comparator. |
| Citation corrections (§4(c) of `31_`) | **ACCEPT** all. |

No FATAL objection was raised against Track B. No MAJOR objection is rebutted in substance; the single partial rebuttal (21/B1) concerns which arms the wording applies to.

## 4. Revised claim set

**Survives.**
1. PR1 is an unbiased estimator of REF-ER whose geometric discovery finds transport-dirty stationary receivers at any depth without a dependency graph. [DERIV, verified in `20` §3.1]
2. The paired same-seed T0/T1 path log is an exact, learning-free, implementation-independent measurement of what a rigid edit invalidates at path granularity. [DERIV; `23` §3]
3. Track B contains no transport representation and cannot complete GM1. [FACT for the clause; INF]

**Narrowed.**
4. Cost ceiling: `1 / (ρ + (c_touch + c_test)/c_path)`, for PR1, fixed view, this sampler, this fixture; about 3–5× for a small mover and below 1× for a large one (EST from `22`, unmeasured).
5. Comparator role: PR1 is a lower bound on the warm path-traced comparator. A complete warm comparator set would need a residual-path-integral sampler and an unbiased ReSTIR PT arm (`31` §5); neither is designed here.
6. ρ: a property of path-granular replay with one sampler on one fixture. It bounds neither Track A's φ_τ nor Track C's f.
7. Births: "no better than fresh" holds for PR1 / PR2 only.

**Withdrawn.**
8. PR3 as an unbiased arm (20/B-3).
9. PR2 as an arm with an economic claim (23/B-1 with the variance derivation in 3.4); retained only as a labelled variant with the anchor charged.
10. F2 as a statement about "any selective scheme"; F3 as "expected by derivation".
11. Any setting-level novelty.

## 5. Hybrid with another track

**At mechanism level: not justified.** The natural construction — Track C's field as control variate or path terminator, Track B's path-space residual as the unbiased corrector — is the neural-control-variates estimator with another control variate (`31` M7; ALREADY KNOWN). By D1–D2 the residual is order-one exactly on the sharp-sender → rough-receiver class, so the corrector pays cold-sampling cost where the field fails, and the readout already costs 20–30 spp-equivalents per pixel (`22/C6`, EST). It would add a third estimator before either part has a measured value. [INF]

**At measurement level: justified, with one interface.**
- **Interface:** a read-only per-vertex label (sampled ER lobe index, mapped to R / S by Track C's fixed α\* rule) written into Track B's paired same-seed T0/T1 path log on one common fixture. ER values are untouched; exposing the label is the same read-only runtime extension both tracks already asked for and needs the same explicit approval.
- **Track B contributes:** the exact dirty-path oracle, and dirty work / radiance / |Δ| shares by region and first-dirty depth.
- **Track C contributes:** the lobe partition and the zero-render lobe census.
- **What it yields:** the share of stationary-receiver |Δ| in each row of the D2 table, i.e. a direct test of challenge D2 and of Track C's economic admission, plus the oracle against which Track A's discovery and Track C's link invalidation can be scored (`23` §6). It builds no mechanism.

## 6. Revised decisive experiment (design only; not authorised)

The track survives only as a measurement and a lower-bound comparator; the experiment is revised accordingly.

**Fixture.** One common GM1 fixture shared by all tracks: closed enough for several bounces, stationary sharp and rough ER receivers, one rigid ER mover, small emitter; one edit that newly blocks, unblocks and moves a glossy interreflection; plus a null edit, a move-and-return edit and an off-path move. Depth by Russian roulette, identical in every arm. Sized so that cold REF-ER at the anchored level costs at least several seconds of kernel time (`22` §5).

**Stage 0 (no rendering).** Fitted-lobe census of the frozen closures by roughness; geometric (Crofton-type) prediction of the per-segment dirty probability, written down before Stage 1.

**Stage 1 (primary deliverable).** REF-P and REF-ER at T0 / T1 with split halves; ER gap g reported first, per region and for Δ. Then a paired same-seed REF-ER T0/T1 kernel, ≥ 8 seeds, logging per path: first-dirty depth, geometric dirty flag, vertex R / S labels. Reported per geometric mask: segment-work fraction ρ; share of T1 radiance on dirty paths; share of |Δ| on dirty paths; pixel-dirty vs path-dirty fraction; false negatives of geometric discovery (must be zero); null-edit dirtiness (must be zero); |Δ| share per row of the D2 table.

**Stage 2 (conditional on Stage 1).** PR1 at the largest N that fits, against cold REF-ER, the masked cold re-render and same-seed correlated path tracing without a store. Work counts and uncontended kernel time; both edit-to-valid-output and T0 + T1 lifecycle. No PR3. PR2 only as a labelled variant.

**Falsification (rules fixed now; numbers frozen after Stage 0 and the reference ladder, before Stage 1 is read).**
- **F-a.** The work-count ceiling `1 / (ρ + (c_touch + c_test)/c_path)` from Stage 1 is inside the predeclared timing margin of 1 → path-granular replay is not a competitive comparator on this workload; cold REF-ER is the binding path-traced baseline; Stage 2 is not run.
- **F-b.** Stage 2 work counts favourable but kernel time not → IMPLEMENTATION-LIMITED, no mechanism verdict. Both unfavourable → the lower-bound comparator is falsified at this workload.
- **F-c.** Cold REF-ER reaches the anchored level within a small predeclared multiple of the fixed-overhead floor → "economically non-discriminating workload"; no track's cost claim is testable on it.
- **F-d.** Geometric discovery has a false negative, or the null edit is not bit-clean → the exactness claim (survivor 1) fails.

The caustic-type birth mask is INDETERMINATE unless its reference is separately qualified.

## 7. Frank statement

**As a mechanism: no.** I would not test Track B first, or at all, as a GM1 candidate. It holds no representation, has no novelty at mechanism or application level, and the roadmap excludes a retracer by name. No experiment changes that; only a human contract decision that a learning-free retracer is admissible would, and that is not evidence.

**As a measurement: yes, Stage 0 and Stage 1 before any mechanism is built**, because they are nearly free once the REF-ER T0 / T1 references every track needs exist, they are exact, and they decide three things in advance: whether the workload discriminates, how much of the edit response a band-limited field could own (D2), and what dirty set any discovery scheme must reproduce. This is the advocate's opinion, not an authorisation.

**The single missing piece of evidence that would change the second answer:** the cold REF-ER time-to-quality ladder on the GM1 fixture at the ER-gap-anchored level. If cold REF-ER meets that level in about a second or less of kernel time (as WL67 / 69 / 70 suggest can happen at loose contracts; HIST, other workloads), then no reuse scheme of any track has economic room there, the dirty-set log has no decision value on that workload, and the first question becomes the workload and gate, not any mechanism.

**Strongest risk to this response.** Section 2's challenge and section 5's measurement hybrid rest on α\* ≈ 0.17–0.34 at K = 128 and on an unchecked fitted-lobe census. If the census puts substantial specular energy above α\*, D2 weakens and Track C's field owns more of the target than argued here.
