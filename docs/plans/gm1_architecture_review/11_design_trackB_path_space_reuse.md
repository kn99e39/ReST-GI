# GM1 design — Track B — Path-space / reservoir reuse

**Status:** design-only review document (2026-11-10), repository `Selective_Recompute` @ `c7b0698`. Sealed track: only `00_PROBLEM_BRIEF.md` was read in this folder. Nothing was implemented, run or measured. This document authorises nothing.

**Evidence labels:** [FACT] established fact (canonical contract or verified measurement), [HIST] historical evidence in its original scope, [DERIV] derivation, [INFER] inference, [HYP] hypothesis, [SPEC] speculation.
**Citations:** † = cited from memory, not re-verified in this session. Unmarked literature entries had title / authors / venue checked by web search today; paper *details* beyond that are from my reading and are flagged where they matter.

## 0. Verdict in brief

- **No meaningful mechanistic distinction** exists between "path-space reuse for a GM1 rigid edit" and existing work (ReSTIR PT / GRIS temporal reuse, common-random-number incremental re-rendering, difference / control-variate re-rendering, ray-segment invalidation). [INFER, section 13]
- Path reuse is **not a representation of transport**. It is a variance / work-reduction layer on a path tracer that uses frozen ER. The roadmap states that "implementing a full retracer does not complete GM1" [FACT, `docs/RESEARCH_ROADMAP.md` GM1 row]. Track B therefore **cannot be the GM1 mechanism**.
- It should be **narrowed to two roles**: (i) the strongest *warm* path-traced comparator that any GM1 mechanism must beat, in addition to cold REF-ER; (ii) an exact, learning-free **measurement of reuse headroom** (the radiometrically weighted fraction of sampled transport that a rigid edit actually invalidates).
- Neither frozen ER nor a "reusable learned transport state" adds something to path reuse that the cited methods lack, except as known constructions (path guiding, neural control variates, radiance caching). [INFER, section 9 and 13]

## 1. Central mechanism (explicit, falsifiable)

**Edit-revalidated path store (ERPS).** At T0 the renderer keeps, per receiver domain (pixel, or surface cell in the material frame), the sampled light paths themselves: vertices, the random numbers that generated them, per-vertex ER lobe choice, contribution and sampling weight. After a rigid edit, every stored path segment is tested against the moved object's old and new placement. A path none of whose segments or vertices touch either placement is **bit-identical** under random replay at T1 and is kept without tracing. A dirty path is replayed from its first dirty segment with the same random numbers (suffix replay), or is shifted by a GRIS shift mapping and resampled against fresh T1 candidates. The T1 estimate is the T0 estimate plus the correlated difference over dirty paths.

**Falsifiable claim.** For a GM1 rigid edit, ERPS reaches the same error against REF-ER at T1, in every receiver class (changed, stationary-affected, unaffected), at strictly lower measured GPU work than a cold REF-ER re-render, and the saving is bounded above by 1/ρ, where ρ is the retraced fraction of segment work. It is falsified if ρ is large, or if invalid-set discovery plus bookkeeping costs as much as retracing (section 12).

Three concrete arms share this mechanism:

| Arm | State | Estimator at T1 | Bias vs REF-ER |
|---|---|---|---|
| **PR1** exact CRN store | all N paths per pixel | same-seed re-render, only dirty suffixes retraced | none; equals the same-seed cold render [DERIV] |
| **PR2** difference re-render | converged T0 image + n ≤ N stored paths | I_T0 + mean(f1 − f0) over the n paths, per-pixel control-variate weight | none in expectation [DERIV] |
| **PR3** reservoir (ReSTIR PT) | 1–few reservoirs per pixel | GRIS temporal reuse across the edit + fresh candidates | none only if the T0 target is evaluated against the T0 placement; otherwise biased [INFER] |

PR1's equality with a same-seed render is an **exact identity, not an algorithm or a cost advantage**: its cost depends entirely on discovery and bookkeeping.

## 2. Persistent state and its information content

**Learned state:** only the frozen ER closure states (69 floats per material [FACT, WL83]). They live in the material frame and are not invalidated by a rigid edit. No other learned state exists in this track.

**Non-learned state (per stored path):**
- vertex positions and surface identity (object id, triangle, barycentrics, i.e. material-frame coordinates);
- random-number seed / offset for replay;
- sampled ER lobe index per vertex;
- next-event light samples and their visibility bits;
- contribution f0, sampling pdf or unbiased contribution weight; for reservoirs also target value and confidence M.

**Spatial content:** point samples of the transport, at exact positions, with density proportional to the T0 sampler. **Angular content:** exact directions per sample; no angular basis, so the WL58 histogram-interface obstruction (64-in / 256-out bins lose 52–80 % relative L2 at α 0.04 [HIST, WL58]) does not arise by construction [INFER]. The price is that the state is a *sample set*, not a function: it carries no information about directions or positions it did not sample.

**Ownership.** Each segment is a C event (free flight on current geometry); each vertex is a K event evaluated by ER. Every segment has one owner, so the K/C contract (`docs/Architecture.md` §2) holds trivially. With ER as a pointwise surface closure, K has no nonlocal part here. [DERIV]

## 3. Lifecycle and invalidation under a rigid edit

**What stays valid.** ER states; every path whose segments (including shadow segments) miss both the old and new placement and whose vertices are not on the moved object. For these, replay at T1 yields the identical path and contribution. [DERIV]

**What is invalidated.**
- Segments that ended on the object at T0 (the hit moves or disappears).
- Segments that cross the new placement (newly blocked).
- Everything downstream of the first such segment on a path (the suffix). A dirty *shadow* segment invalidates only that vertex's direct term, not the continuation.
- For reservoirs: additionally the resampling weights, because the target function changed on the dirty set (section 4).

**Discovery.** Test all stored segments against the object's bounds, then against its geometry, in both placements. Stationary geometry needs no test. This discovers transport-dirty paths at stationary receivers at any bounce depth without a dependency graph: D_T ≠ D_G is handled by construction. [DERIV]

**Discovery cost [DERIV, symbols only].** Let S be the stored segment count, c_obj the cost of a segment-vs-moved-object query, c_scene the cost of a full-scene trace, c_sh the ER shading / sampling cost per vertex.
- Brute force: 2·S·c_obj. A cold re-render costs S·(c_scene + c_sh). Reuse cost ≈ 2·S·c_obj + ρ·S·(c_scene + c_sh) + bookkeeping.
- If the moved object is a large share of scene complexity (c_obj ≈ c_scene), discovery alone approaches the cold cost.
- A hierarchy over stored segments makes discovery output-sensitive (O(log S + k)) at an O(S log S) build and extra memory, amortised only over repeated edits. This is the ray-segment-tree idea (Bala, Dorsey, Teller 1999†).

Near-contact and thin geometry need conservative bounds; a missed dirty segment is an uncorrected stale path (bias).

## 4. Created and destroyed light paths

**Death (newly blocked).** Detected exactly by revalidating every segment of a kept path, including the reconnection and shadow segments. Partial validation (a subset of pixels or segments per frame, common in real-time ReSTIR GI practice†) leaves stale energy: light leaking and lag. [INFER]

**Birth.** Three kinds: (a) paths through the vacated volume; (b) paths that now hit the object in its new pose, including new glossy interreflection onto stationary receivers; (c) paths that used to end on the object and now continue.
- In PR1 / PR2, replaying a dirty path with the same random numbers produces exactly the path a fresh tracer would produce; births are covered at the **fresh sampler's rate, no better**. [DERIV]
- In PR3, births are covered only by fresh canonical candidates. GRIS needs the union of candidate supports to cover the target support; the canonical sample provides that, at 1-sample variance in the newly lit region. A T0 reservoir with high confidence M can dominate resampling there and delay convergence unless M is reduced on the dirty set. [INFER, from the GRIS framework: Lin et al. 2022]
- Unbiased reuse across the edit needs MIS weights that evaluate the **T0 target for T1 candidates**, i.e. visibility against the T0 placement. For a rigid edit this is one extra instance transform, not a rebuild. Omitting it biases newly occluded / disoccluded regions. [INFER; not re-verified against the GRIS paper text]

**Sharp specular transport.**
- *Reconnection* through a narrow ER lobe changes the half-vector and collapses the contribution; *random replay* preserves lobe-relative directions but diverges as soon as geometry differs, after which the shifted path is statistically a fresh sample. The hybrid shift (replay until a sufficiently rough, sufficiently distant vertex pair, then reconnect; Lin et al. 2022) therefore **degenerates to fresh sampling on exactly the paths GM1 targets**: glossy chains through the moved object. [DERIV]
- *Path repair* by continuity (keep the vertex in the object's material frame, then solve for the new specular point by a manifold / half-vector-preserving walk: Jakob & Marschner 2012†; Kaplanyan et al. 2014†; Zeltner et al. 2020†) can carry a T0 highlight path to its T1 position. It is iterative, fragile on tessellated geometry, undefined when the path family ceases to exist, and is prior art in gradient-domain MLT shift mappings (Lehtinen et al. 2013†). [INFER]
- A new highlight on a rough stationary receiver from a small light via the moved glossy object is a caustic-type path. Unidirectional sampling finds it rarely with or without reuse; reservoirs only spread found paths. Bidirectional or photon candidates are the known remedy (ReSTIR BDPT, Hedstrom et al. 2025; ReSTIR FG, Kern et al. 2024†). [INFER]

## 5. Directional information under repeated transport

Each path keeps exact incoming and outgoing directions at every bounce and ER is evaluated exactly, so nothing is collapsed to irradiance or to bins. What degrades is not information but **sample density**: a product of k narrow lobes is sampled only as well as ER's own `sample` allows, identical to REF-ER. A stored suffix chosen for one outgoing direction is poorly distributed for another when the receiver lobe is sharp, so view or receiver changes at glossy vertices void reuse. [DERIV]

## 6. Coupling and multi-bounce computation

The coupled solve `b = K(e + C b)` is evaluated as a sampled Neumann series: one path is one term. No linear system, no spectral-radius condition beyond ER passivity (certified [FACT, WL82–83]); truncation depth and Russian roulette must equal REF-ER's (WL84 used max depth 4 [FACT]). Multi-bounce changes are captured because any dirty segment at any depth dirties the path.

## 7. Memory and computation scaling

**Memory [DERIV + ESTIMATE].** PR1 stores P·N paths (P pixels, N spp). With roughly 4 vertices, 4 light samples, contribution and seed, a path needs on the order of 100 bytes (my estimate; no implementation exists). 512² pixels × 64 spp ≈ 1.7·10⁷ paths ≈ 2 GB; × 1024 spp exceeds the 12 / 24 GB cards in use. PR1 is therefore limited to modest N. PR2 needs the converged image plus n paths per pixel. PR3 is constant per pixel; a secondary source reports 2 × 64 bytes per pixel for ReSTIR PT Enhanced (Lin, Kettunen, Wyman 2026), not verified in the paper.

**Dirty fraction [DERIV under an assumed parameter; illustrative arithmetic, not a scene estimate].** If each of s segments of a path is independently dirty with probability f, then P(path dirty) = 1 − (1 − f)^s and the expected retraced suffix is s − (1 − f)(1 − (1 − f)^s)/f segments. For f = 0.1, s = 8 (4 bounces + 4 shadow segments): 57 % of paths are dirty and 36 % of segment work is retraced, a ceiling of about 2.8× before discovery cost. Independence is false in detail; the point is that **dirtiness compounds with depth**.

**Honest worst cases.**
- Enclosed, strongly interreflecting scene with a central moving object: ρ → 1; reuse saves nothing and adds discovery cost.
- Large or complex moving object: c_obj ≈ c_scene.
- Cheap scenes: in WL84 a full 64-spp ER image costs about 10 ms of GPU kernel time [FACT, WL84]. At that scale no reuse scheme has room; ER's low shading cost *reduces* the value of reuse, because it makes cold REF-ER cheaper.
- PR2 on fully changed pixels: Var(f1 − f0) = Var f1 + Var f0 when replay decorrelates, i.e. worse than fresh sampling unless a per-pixel control-variate weight falls back to fresh (Rousselle, Jarosz, Novák 2016).
- PR3: confidence caps bound the benefit of history; correlation between pixels produces structured error that pixel-wise error metrics under-report (motivation of Kettunen et al. 2023 and Sawhney et al. 2024†).

## 8. Nonlocal visibility and stationary receivers

Handled exactly and without a selector in PR1 / PR2: a stationary receiver's estimate changes iff one of its sampled paths has a dirty segment, at any depth, including occlusion by a third object. The limitation is sampling, not dependency: a stationary receiver whose *new* transport is carried by paths the sampler rarely finds converges no faster than cold REF-ER. Screen-space reservoirs additionally lose history at disocclusions; world-space reservoirs (Boissé 2021; "World-Space Spatiotemporal Path Resampling for Path Tracing", CGF 2023, authors not verified) avoid that for rough receivers only.

## 9. Integration boundary with frozen ER

ER is queried only through its existing interface, with identical values:
- `sample(wo, u)` at every non-terminal vertex, for fresh candidates and for random replay (must be deterministic in u);
- `eval` and `pdf` at both ends of a reconnection segment (new wi at the vertex before, new wo at the vertex after), for next-event estimation and its MIS, and for shift Jacobians;
- reverse-direction `pdf` only if bidirectional candidates are used (ER is reciprocal by construction [FACT, WL82]).

One **read-only addition** is needed: the sampled lobe index and a per-lobe angular-width descriptor, to decide reconnection eligibility. This reads the compiled state and does not change closure values, but it is a runtime-interface extension that needs explicit approval under "ER must not be changed". Two cautions: ER's refracted inner lobes have an outside-direction footprint that is not their α; and the float32 `sample`/`eval_pdf` pdf discrepancy (max 1.8e-4 relative near the horizon [FACT, WL83]) enters Jacobians.

ER has no delta lobes, so every path is formally reconnectable; the hybrid criterion is a variance heuristic, not a support requirement. All PR arms are estimators of REF-ER; their error against REF-P inherits ER's bias floor (about 0.035–0.04 image L1 in the WL84 scenes [FACT, that scope]).

**What learned components could add [all prior-art constructions]:**
- learned proposal for newborn paths = path guiding (ReSTIR PG, Zeng et al. 2025; Müller et al. 2017†); scene-fitted, and it relearns after the edit;
- learned terminal radiance to shorten paths = neural radiance caching (Müller et al. 2021†); this is baked scene radiance, in tension with the "learned response, not baked radiance" invariant (`AGENTS.md`);
- a learned global transport state as control variate with a path-space residual (neural control variates, Müller et al. 2020†): path samples would then be the unbiased corrector of another track's representation. [SPEC]

None is specific to ReST-GI.

## 10. Principal failure conditions and most dangerous counterexamples

1. **New sharp highlight on a stationary rough receiver** via the moved glossy object (GM1's headline phenomenon). No stored sample represents it; replay diverges; cost equals cold sampling of a caustic-type path. Reuse contributes nothing where the edit matters most.
2. **High dirty fraction** in enclosed multi-bounce scenes (section 7).
3. **Biased temporal reuse**: skipping T0-target evaluation or partial validation gives lag and leaking that look like "stale state", contaminating a control that is supposed to be unbiased.
4. **Workload too cheap**: cold REF-ER already at the cost floor (WL84 scale); also the Python-traced Dr.Jit harness spends 85–95 % of wall time in tracing overhead [FACT, WL84], so wall-clock comparisons there are uninformative.
5. **It is a retracer.** Passing every image test does not answer GM1's representation question.
6. **Continuous motion**: per-frame dirty sets are small but accumulate; PR2's T0 anchor ages, PR3's history is capped.

## 11. Testable advantage over controls

- **vs stale state:** unbiased at T1 (PR1 / PR2 exactly, PR3 with full T0-target MIS). Certain by construction; not informative.
- **vs same-architecture full rebuild** (cold initialisation of the store or reservoirs at T1, i.e. cold REF-ER plus store build): the incremental update does a fraction ρ of the tracing plus discovery. It *can* win at matched quality only because unchanged sampled transport is not recomputed and, for PR2, because the T0 image carries more samples than the T1 budget in unaffected regions.
- **vs REF-ER re-render:** same statement; REF-ER cold *is* the full rebuild here.

The advantage is purely "do not redo unchanged work", bounded by 1/ρ, and vanishes on the changed sharp paths. [HIST] context: cold path tracing already beat earlier ReST-GI oracles at matched quality (Cycles 4 spp in 4.7 s vs 190 s, WL67; 16 spp in 2.4 s vs 24 s, WL69; narrowly matched only after sun-state reuse, WL70). Those were other workloads and renderers; they are not measurements of this track, but they show the path-traced comparator is already binding, and a warm one is at least as strong.

## 12. One bounded decisive experiment (design only)

**Purpose.** Decide whether warm path reuse beats cold REF-ER on a GM1 workload, and measure the reuse headroom ρ that bounds *any* selective scheme at path granularity.

**Workload.** One GM1 fixture at two geometric complexities: stationary glossy and rough ER receivers, one rigidly moved ER object, small emitter, depth fixed and equal for all arms. One rigid edit that (i) newly blocks a path to a stationary receiver, (ii) unblocks another, (iii) moves a glossy interreflection highlight across a stationary receiver. Materials from the qualified coated-conductor family, including a sharp / hard one.

**References.** REF-P and REF-ER at T0 and T1, independent seeds, split-half uncertainty. Arms are scored against **REF-ER T1**; REF-P is reported to show the ER floor.

**Stages and arms.**
- Stage 0: cold REF-ER spp ladder (error vs GPU kernel time); STALE (T0 image) as negative control.
- Stage 1 (headroom, exact): T0 and T1 rendered with identical seeds and per-path logging. Measure path-dirty and segment-dirty fractions and the dirty share of radiance per mask; verify clean paths are bit-identical and that geometric discovery finds every diverging path (zero false negatives).
- Stage 2: PR2 (difference re-render with per-pixel fallback) and PR1 at the largest N that fits memory; full rebuild = cold store build at T1. PR3 is a non-required extension; if run, both the unbiased and the commonly used biased variant, the latter as a labelled negative control.

**Measurements.** Relative L1, highlight-ROI error, signed-change slope / correlation on stationary receivers, per mask (visibility-affected, unaffected, newly lit, newly blocked); GPU kernel time split into discovery, retrace, shading, bookkeeping; memory. Wall time reported but not used.

**Falsification (thresholds frozen after Stage 0, before Stage 1).**
- F1: at matched error in every mask, best PR arm's GPU time ≥ cold REF-ER beyond timing uncertainty → path reuse is not a competitive comparator at this scale; cold REF-ER is the binding path-traced baseline.
- F2: measured ρ so large that 1/ρ is within the predeclared margin of 1 → no path-granular selective scheme can pay on this workload.
- F3 (expected by derivation): in the newly-lit sharp-highlight mask, PR error at equal traced rays is not below cold → reuse gives no help on births.

If F1 fails to occur, warm path reuse becomes a mandatory comparator for every GM1 mechanism.

**Baseline-role gate.** No learned state is invalidated (ER is material-frame; its reuse is valid Native Reuse). All PR arms are *path-traced re-rendering*, never Neural Full Recompute.

## 13. Closest prior art and honest difference

| Work | Relation |
|---|---|
| Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman 2022, *GRIS: Foundations of ReSTIR*, ACM TOG (SIGGRAPH) | PR3 is this method; hybrid shift, temporal reuse. No difference. |
| Ouyang et al. 2021, *ReSTIR GI*, CGF (HPG) | Reconnection-only reuse with sample validation; subset of PR3. |
| Lin, Kettunen, Wyman 2026, *ReSTIR PT Enhanced*, PACM CGIT 9(1) | Better reconnection criteria and robustness; would be the PR3 implementation of choice. |
| Kettunen, Lin, Ramamoorthi, Bashford-Rogers, Wyman 2023, *Conditional RIS and ReSTIR*, SIGGRAPH Asia | Suffix reuse with final gather = suffix replay / world-anchored reuse. |
| Zhang, Lin, Kettunen, Yuksel, Wyman 2024, *Area ReSTIR*, ACM TOG | Sub-pixel / lens domains; orthogonal. |
| Liu, Lin, Kettunen, Wyman, Ramamoorthi 2025, *Reservoir Splatting*, SIGGRAPH | Forward reprojection of reservoirs under motion; closer temporal correspondence than back-projection. |
| Boissé 2021 (world-space ReSTIR, SIGGRAPH Asia tech. comm.); Boksansky et al. 2021 (grid reservoirs, RTG II) | World-space persistent reservoirs. |
| Zeng et al. 2025, *ReSTIR PG*, SIGGRAPH Asia; Dittebrandt et al. 2020† | Guiding fitted from reused paths, responsive to scene change. |
| Hedstrom, Kettunen, Lin, Wyman, Li 2025, *ReSTIR BDPT*, ACM TOG | Caustic reservoirs across frames in dynamic scenes; covers the birth counterexample better than this design. |
| Günther & Grosch 2015, *Consistent Scene Editing by Progressive Difference Images*, CGF 34(4) | Object add / remove edits rendered as a stochastic difference image: PR2's idea, for edits. |
| Rousselle, Jarosz, Novák 2016, *Image-space Control Variates*, ACM TOG | Re-rendering with the previous render as control variate: PR2's estimator. |
| Kettunen et al. 2015†, Manzi et al. 2016†, Schied et al. 2018† (gradient-domain / temporal gradients) | Correlated T0 / T1 sampling by shift mappings. |
| Bala, Dorsey, Teller 1999†; Brière & Poulin 1996†; Séquin & Smyrl 1989†; Dmitriev et al. 2002† (selective photon tracing); Drettakis & Sillion 1997† | Stored ray / segment structures with geometric invalidation after an object edit: PR1's discovery. |
| Kelemen et al. 2002†; Veach & Guibas 1997† | Primary-sample-space replay and path mutation. |

**Honest difference: none in mechanism.** The only GM1-specific elements are (i) a single known rigid edit with object identity, which permits exact geometric discovery instead of image heuristics, (ii) frozen ER as the BSDF, and (iii) quality-matched lifecycle accounting instead of a real-time 1-spp budget. These are settings, not contributions. Absence of a paper combining exactly these is not evidence of novelty.

## 14. Self-assessed weaknesses

- No measurement supports any cost statement here; the cost model's ratios (c_obj, c_scene, c_sh, ρ) are unknown for a GM1 workload that does not yet exist.
- Details of GRIS temporal MIS under animated geometry and of ReSTIR PT Enhanced were not re-read; †-marked citations are from memory.
- No public ReSTIR PT implementation for Mitsuba / Dr.Jit is known to me; the public ones (Falcor-based, e.g. NVLabs `conditional-restir-prototype`) would require porting ER to another shading language, which is an integration cost and a parity risk.
- The independence assumption in the dirty-fraction arithmetic is crude.
- The track does not address the project's research objective (learned transport reuse); its value is as a control and a headroom bound, and a strong result for it would be evidence *against* the economic case for a learned GM1 mechanism on that workload, not a ReST-GI contribution.
