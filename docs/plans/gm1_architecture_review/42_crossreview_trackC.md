# GM1 cross-review — Track C response (LPTG, Lobe-Partitioned Transport Graph)

**Status:** design-only cross-review record (2026-11-10), revision `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run, rendered or measured. This is the only file written; no existing file was changed. It authorises nothing.

**Role:** fresh agent acting as Track C's advocate-with-integrity after the seal was lifted.

**Read in full for this response:** `00_PROBLEM_BRIEF.md`; the three designs `10_`, `11_`, `12_`; the four reviews `20_`–`23_`; the Track C kill search `32_`. Read in part: `30_` and `31_` (result sections and first mechanisms); `docs/RESEARCH_ROADMAP.md` (GM1–GM3 rows); `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` (the "limited explicit radiosity solver" sentence, l.17); `AGENTS.md`. Worklogs were **not** re-opened in this session: every worklog number below is taken from the reviews, which state that they checked the worklog text, and is cited through them. No web search was run; literature is cited at the verification level the reviews gave it.

**Labels:** FACT = ESTABLISHED FACT; HIST = HISTORICAL EVIDENCE (original scope only); DERIV = DERIVATION; INF = INFERENCE; HYP = HYPOTHESIS; SPEC = SPECULATION; EST = order-of-magnitude estimate, not a measurement. "R20 / R21 / R22 / R23 / N32" = the review files `20_` … `23_`, `32_`.

## 0. Position in brief

- INF. **LPTG does not survive as a GM1 mechanism candidate.** It survives as (i) a zero-render admission census that every track needs, (ii) a path-class instrumentation of REF-ER, and (iii) a possible learning-free physical baseline, whose admissibility is a human decision.
- Three accepted objections decide this, independently of each other:
  1. R23 C-1: the camera-visible sharp highlight is produced by fresh per-frame chains, so the persistent state holds no sharp directional content. For the sharp part LPTG is a retracer, and the roadmap states that a retracer does not complete GM1 (FACT, `docs/RESEARCH_ROADMAP.md` GM1 row).
  2. R20 C-1 / R21 C1: the field is point-sampled, not band-limited. Sharp sender → rough stationary receiver, a GM1 headline phenomenon, is aliased coherently and has no convergence path at fixed state size.
  3. R22 C1: the field re-solve is dense per edit, so the selective part of the update is immaterial. `S ≈ F` in cost is the expected outcome.
- No FATAL or MAJOR objection is rebutted in substance. Two are rebutted in framing only (§3, rows R22-C1 and R21-C2).

## 1. Strongest competitor

**Track B (ERPS), in its PR2 / residual form.** Reasons:

- FACT (R20 §3.4, R21 §3.5). PR1 and PR2 are sound estimators of REF-ER with exact directions. No basis, no operator, no passivity question. LPTG is a biased-consistent estimator (R20 C-2).
- DERIV (R22 §5). At the fixture scale the reviews used, B's edit cost is 0.2–0.6× one cold 1024-spp render; LPTG's is 3–400× (EST constants, R22 §1.3).
- INF (R20 §5 item 4). On the headline phenomenon the ordering by physical correctness is B > C > A.
- FACT (N32 §0; `31_` §0 item 2). The published form of B for exactly this setting exists: Xu, Li, Georgiev, Hedstrom, Ramamoorthi 2024, "Residual Path Integrals for Re-rendering", CGF 43(4), EGSR (verified by the Track A and Track B kill searches; not re-read by me).
- B asks for no GM1 credit, so it cannot lose on the contribution axis.

**Why not Track A.** Its author rejects it as stated; R22 finds no economic window inside GM1 (dirty fraction needed 0.05–0.8 %, geometric lower bound about 3 %); and R20 §2.4 shows that each physically adequate repair of DDT-D (stored receiver gather rays, common random numbers, chain continuation) turns it into LPTG's stored links plus a forward tracer. A-as-repaired is Track C with a different discovery signal, so it inherits Track C's defects.

## 2. Challenge to Track B's central assumption

**The assumption.** Reuse is defined on whole sampled camera paths at a fixed view, and the retraced fraction ρ is "the reuse headroom" that bounds selective schemes (`11_` §0, §1, §12 F2).

**Challenge 1 — dirtiness compounds with path depth only because the store has no state at the new hit point (DERIV).**
With per-segment dirty probability `f` and `s` segments per path, Track B's own formula gives the retraced share ρ = [s − (1 − f)(1 − (1 − f)^s)/f] / s ≈ f·(s + 1)/2 for small `f`.

| `f` | `s` | paths dirty | ρ (path store) | invalid share (link store) |
|---:|---:|---:|---:|---:|
| 0.02 | 8 | 15 % | 0.086 | 0.02 |
| 0.02 | 20 | 33 % | 0.19 | 0.02 |

`f` = 0.02 is R22's Crofton estimate for a 0.2 m cube in a 2 m room; `s` = 20 stands for a closed court with Russian roulette (illustrative). A path store must retrace everything after the first dirty segment, because the replayed suffix lands on new points where it holds nothing. A view-independent link store regenerates one link and terminates in state at the new hit. So ρ is a property of path-granular replay with one sampler, and it worsens exactly where GM1 goes (several bounces, closed scenes). It is not a bound on reuse. R20 B-6, R23 B-3 and `31_` §0 item 3 reach the same conclusion independently.

*Honest limit of this challenge.* The same compounding occurs inside LPTG's own sharp chains (length 1 + L̄, plus shadow segments). The link-granular advantage therefore exists only for R-terminated links and vanishes as the sharp albedo share approaches 1. And R22 C1 shows that LPTG spends the saved retracing on a dense solve. The challenge defeats B's universality claim, not B's lead over LPTG as written.

**Challenge 2 — a geometry-null edit voids the store (DERIV).**
PR1 and PR2 are keyed to pixels. Moving the camera with no geometry change gives ρ = 1. The roadmap requires per-frame accounting (FACT, GM1 row), so B's economics hold for one fixed view only. Track B concedes this (`11_` §5, §8); the reviews do not charge it.

**Challenge 3 — "sound" is vacuous in the headline mask at any storable sample count (DERIV on the reviews' EST values).**
- R21 §1.5: a light → sharp sender (α = 0.05) → rough receiver path is found at rate p ≈ 2.5·10⁻³ – 10⁻².
- R22 §3.3: PR1 holds at most about 2,000 spp at 256² on 12 GB.
- Per-pixel relative noise in that mask is then ≈ 1/√(p·N) ≈ 0.22–0.45.
- R21 B2: REF-ER itself is reference-limited there at 65,536 spp (0.04–0.08).

So B's unbiasedness delivers no usable estimate of the newly created sharp chain on a rough stationary receiver. B does not alias it; it does not resolve it either. This is a statement about all three tracks and about the reference: **no design in this review carries that phenomenon** (R20 §5 item 4, R21 §0).

## 3. Objections against Track C: ACCEPT or REBUT

### 3.1 FATAL / MAJOR objections

| Source | Objection | Response | Effect on the claim set |
|---|---|---|---|
| R20 C-1, R21 C1 | Point-sampled incident field is not band-limited. 0.16–0.64 nodes per highlight at α_S = 0.05, K = 128; coherent shifted copies; node flips under small motion read as signed change; Stage 0(ii) cannot detect it. | **ACCEPT.** | Withdraw "band-limited field" (`12_` §2) and "loss does not compound because every R lobe is band-limited" (§5). The bounded repair (solid-angle average per node cell, or per-cache-point jitter with exact `f_R` evaluation) turns coherent bias into noise but does not raise the hit rate p. The sharp-sender → rough-receiver class is therefore **removed from the track's scope**, not repaired. |
| R20 C-2 | Fixed point of a frozen random operator: biased at finite K, bias grows with loop gain; chain termination unspecified; heavy `f_S cos / p_S` tails plausible. | **ACCEPT.** | Restate the estimator as biased-consistent. Withdraw the reading "unbiased with correlated noise" (§10.9). Termination must be declared (Russian roulette at sharp albedo); realised ρ and a split-realisation bias gate become mandatory. |
| R20 C-3 | Collocated B does not inherit ER's passivity; the proposed audit checks the wrong sum. | **ACCEPT.** | The audit is Σ_m f_R(ω_j, ω_m) cos θ_m w_m ≤ 1 for every j, including grazing nodes. A violation is a stop (no clamping). The §9 "column sums" wording is withdrawn. |
| R20 C-4 | Collocation and interpolation are not flux-conserving; light leaks across a contact line; conflicts with the area-weighted mapping requirement of `docs/Architecture.md` §3 (as quoted by R20; not re-read by me). | **ACCEPT** as a contract deviation. | Reported for human decision. No closure claim is made for the realised C. Edit feature (d), near contact, is withdrawn from the claimed scope. |
| R21 C2, R22 C3 | K = 128 implies α\* ≈ 0.15–0.35, so most specular lobes of the family fall in S. | **ACCEPT the derivation. REBUT only the status** of the "70–85 % of materials" figure: it assumes fitted outer lobes track descriptor α_i, which nobody checked (R21 labels it INF). | Adopt as the planning assumption. `12_` §10.4's mid-roughness band (0.05–0.15) is withdrawn; its upper edge is α\*. The field is a Lambert-plus-broad-lobe carrier. |
| R21 C3, R22 C4 | A mover subtending less than one node cell (0.6 nodes for a 0.2 m face at 1.15 m) is quantised to 0 or 1 node on stationary receivers; F ≡ S hides it. | **ACCEPT.** | Declare a scope limit: the mover's indirect effect on a rough receiver is represented only above Ω_min = 2π·m/K with m ≥ 16 (R21's resolution count); at K = 128 that is about 0.8 sr. Below it the stationary-receiver signed change is not claimed. This excludes the small-box edit of `12_` §12. |
| R21 C4 | R → R loses peak amplitude per bounce near α\* (about 19 % at α = 0.17, 5.5 % at 0.34, Gaussian approximation). | **ACCEPT.** | α\* must sit well above node spacing; the two-bounce attenuation test joins Stage 0. |
| R22 C1 | Field re-solve is dense per edit and dominates; `S ≈ F`. Labelled FATAL for "H-edit as an economic claim". | **REBUT the framing, ACCEPT the prediction.** `12_` §1 states H-edit as a correctness contract, and §7 worst case 2 plus the §12 row "S cheaper than REF-ER but not cheaper than F" already name this outcome. But R22's arithmetic makes that row the expected result, not a risk. | The selective-update cost advantage is **withdrawn in advance**. A residual-driven solve (R22 §4.8) is not adopted: every ER material has a Lambert lobe, so the change spreads to every receiver that sees an affected one after one rough bounce (R21 A3), and sparsity would come only from a tolerance amplified by 1/(1 − ρ) (R20 A-1). |
| R22 C2 | Memory 4–16 GB once L̄ = a_S/(1 − a_S); the 1–3 GB estimate is low by up to 5×. | **ACCEPT.** | Estimate withdrawn. Memory is computed from the census, with a stop rule. |
| R22 C5 | At 256² the per-edit cost exceeds a high-quality re-render for all stated parameters; a window needs megapixel output and several frames per edit. | **ACCEPT**, with R22's own caveats (256² unverified in WL84; a closed court raises `c_path` by perhaps 2–4×; constants are EST and a factor of 3 can open or close the window). | H-econ narrowed to a per-frame cache claim (§4). |
| R23 C-1 | The highlight gate is passed by fresh readout chains; it does not test the persistent state. | **ACCEPT.** The most damaging objection for GM1 status. | H-rep as written is withdrawn. The state's directional content is R-class only, plus frozen one-sample S vertices inside links. |
| R23 C-2 | F ≡ S hides everything F and S share; the box mover never provokes the blind spot. | **ACCEPT.** | F ≡ S is demoted to an implementation test. Independent-seed F′ (≥ 8) is required for any fidelity statement. |
| R23 C-3 | H-rep is tied to an equal-time condition. | **ACCEPT.** | Bias gate has no time condition; equal-time comparisons belong to the economic claim only. |
| R23 C-4, X1 | Structured error is under-reported by pooled L1; "matched quality" between biased and unbiased estimators is undefined. | **ACCEPT.** | Matching is a simultaneous threshold on every gated metric at pixel scale and two pooled scales, with a bias / variance split. |
| R23 C-5 | No retracer test. | **ACCEPT.** | The no-field arm becomes the first gate (§6). |
| R23 X2–X6 | Unanchored quality levels; no noise model for signed change; no arm replicates; free T0 investment; mechanism vs implementation maturity not separated. | **ACCEPT all.** | Levels anchored to the in-scene ER gap; instrumented slope; ≥ 8 seeds; lifecycle and edit-to-output both reported; work counts beside kernel time. |
| N32 §0, §6 | No new mechanism. Missed predecessors: Granier and Drettakis 2001 (finite-element non-sharp part, traced glossy part, line-space update after object motion, selective re-emission); Martín, Pueyo, Tost 2003; Günther and Grosch 2015; Lin et al. 2022. The three "specific to ReST-GI" items are application novelty, a known identity, and unmeasured utility. | **ACCEPT.** (N32 verified these at search-summary level; I did not re-verify.) | Withdraw the "static scenes" difference for the two-pass lineage and item (b) of `12_` §13. No mechanism contribution is claimed. |
| R23 X7, N32 §5.5, `12_` §14.1 | Nothing global is learned. | **ACCEPT.** | No F/S result of this track may be reported as selective recompute of learned transport (FACT, `AGENTS.md` terminology). FNTO v0.1 l.17 warns against replacing learned shading with "a limited explicit radiosity solver"; LPTG's field is one. Admissibility is a human decision. |

### 3.2 MINOR objections

All accepted: R20 C-5 (weight 1 for the `f_R` part of next-event estimation, no emission on node hits), R20 C-6 (exact `eval` at link hits instead of interpolated B rows), R20 C-7, R21 C5 (classify by the narrowest direction-dependent width: grazing, anisotropy, refracted exit), R21 C6 (rotation-equivariance control), R21 C7 (lobe-space sampler test for α < 0.1), R22 C6, R22 C7, R23 C-6, R23 C-7 (REF-ER runs the unrestricted ER).

### 3.3 What the reviews confirmed

FACT (R20 §4.1, R21 §4.2, R22 §4.6, R23 §4). The lobe partition and K/C ownership are exact. The segment-validity table is correct, including the same-asset certificate. F ≡ S is proved. S → S → eye survives at full sharpness. The build is about 1–3 high-quality frames, three to four orders of magnitude below Track A's. The staging and control rules were judged the best of the three. None of these is a contribution: they are identities, known constructions or experimental hygiene.

## 4. Revised claim set

**Survives (unchanged in content, reduced in status).**

- **C-id (DERIV, known).** `f_ER = f_R + f_S` partitions path space exactly and maps onto K/C ownership. Use: instrumentation and bookkeeping.
- **C-val (DERIV, known).** The four-row segment-validity table and F ≡ S. Use: implementation test of any stored-link structure. Not evidence of accuracy, not a cost advantage.

**Narrowed.**

- **H-rep′ (HYP).** A persistent field carries the *R-terminated* multi-bounce interreflection (Lambert and lobes with direction-dependent width ≥ α\*) with seed-mean bias against REF-ER below a gate anchored to the in-scene ER gap, on those path classes only. α\* is fixed by K through α\* = 4/√K (from R21's N_ω ≈ 16/α²: 0.35 at K = 128, 0.25 at K = 256), with no sweep.
- **H-econ′ (HYP).** With a valid state, per-frame readout is cheaper than REF-ER at the anchored quality in a multi-frame-per-edit, megapixel regime. This is a cache claim. Its lifecycle includes the dense per-edit solve.
- **Scope.** Rigid edits; one material per object; mover subtending ≥ Ω_min from the gated receivers; no near contact; materials of the qualified family (α ≥ 0.05, R21 §1.1).

**Withdrawn.**

- H-rep as a claim about sharp glossy / specular interreflection. The persistent state does not represent it.
- The sharp-sender → rough-receiver class and the thin / small mover, as anything the track handles.
- Any cost advantage of S over F (H-edit as economics).
- "Band-limited field"; "R → R loss does not compound".
- The 1–3 GB memory estimate; the 0.05–0.15 mid-roughness band.
- "Static scenes" as a difference from the two-pass lineage; F ≡ S on a glossy operator as project-specific.
- **Candidacy as the GM1 global mechanism.** With the sharp part retraced, the headline class excluded and nothing global learned, a full pass of the revised claims would show a diffuse-to-rough interreflection cache with certified links around frozen ER. That does not answer the GM1 question (INF from the roadmap exit clause).

**Residual roles (human decision whether any is wanted).** A learning-free physical baseline or same-architecture comparator for a later learned mechanism, in the sense HIST WL80B gave its C1 candidate ("PHYSICAL BASELINE ONLY", via `12_` §14.1 and N32 §5.5); and the two measurements of §6.

## 5. Is a narrow hybrid justified?

**One is: a measurement hybrid with Track B. No mechanism hybrid is justified now.**

- **Single interface.** A read-only per-vertex label on REF-ER paths: the sampled ER lobe's class, R or S, under the fixed α\* rule. Nothing else crosses.
- **Track C contributes** the partition, the α\* rule and the lobe census.
- **Track B contributes** its Stage 1: same-seed T0 / T1 REF-ER renders with per-path logging and exact geometric dirtiness (R23 calls this the only implementation-independent, noise-free quantity in the three designs).
- **Product.** Per region and per path class: share of T1 radiance, share of |Δ|, share of dirty segment work. It answers, before any mechanism is built, (a) how much of the stationary-receiver change a rough-lobe field could own at all, (b) how large the caustic-type class is that no design carries, (c) whether the workload discriminates. This is R23's correction C-1.1 and its closing shared-fixture observation, joined.
- It adds no ER numerics beyond reading the sampled lobe index, which Track B already requests (`11_` §9). That read is an interface extension needing approval under "ER must not be changed".

**Considered and not justified.**

- *C links + Track A's residual ordering* (to make the solve output-sensitive). Technically coherent: with common-random-number links an unchanged gather reproduces exactly, which removes A-4 and A-2. But the Lambert lobe makes the residual support dense after one rough bounce, so the saving is a tolerance, i.e. a bias budget (R21 A3, R20 A-1). No evidence supports it.
- *C field as control variate under a path-traced residual* (unbiased by construction; `11_` §9 lists it as SPEC). It is the known neural-control-variate / radiance-cache construction with an unlearned cache, costs a path trace per frame, and is pointless until the field is shown to carry a material share of the transport.

## 6. Revised decisive falsification experiment

The track survives only as an admission question, so the experiment is an admission test. It ends before any LPTG code exists. Design only; not authorised.

**Question.** Does a rough-lobe field own enough of the GM1 edit response on stationary receivers, for the frozen ER family, to be worth building even as a baseline?

**Stage 0 — zero-render census (frozen closures only).**
1. Fitted-lobe roughness census, weighted by albedo, using the narrowest direction-dependent width (grazing cos θ factor, min(α_u, α_v), refracted exit). Outputs: R-share a_R and S-share a_S at α\* = 4/√K for one K fixed beforehand from the memory cap.
2. Corrected passivity audit of B (weighted sum, all nodes including grazing).
3. B quadrature against an **un-band-limited** incident field (small source), plus two-bounce R → R peak attenuation and a rigid-rotation control.
4. Economic admission from census values: L̄ = a_S/(1 − a_S), stored-segment memory, and R22 §4.4's break-even frames per edit for the fixture's image size.

**Stage 1 — path-class log (REF-ER only; the §5 hybrid).** One common GM1 fixture, T0 and T1, same seeds, independent halves, Russian-roulette depth, unrestricted frozen ER, masks from geometry. Per-class images for: camera-visible S chains; R-terminated indirect; R → S → R; light → S … → R onto rough receivers. Signed change read with split-half instrumented slopes (R23 X3); regions below the predeclared reference signal-to-noise are INDETERMINATE.

**Predeclared rules.** Thresholds are fixed by rule before each stage and anchored to g = err(REF-ER, REF-P) per region (R23 X2).

| Outcome | Meaning |
|---|---|
| Stage 0.2 fails, or 0.3 exceeds its gate | Discretisation not passive or not accurate at the fixed K. Closed; no retuning. |
| Stage 0.4: memory above the cap, or break-even frames per edit above what the fixture renders | Closed on economics before any render. |
| Stage 1: removing the field-owned classes from REF-ER changes the stationary changed-ROI by less than g (this is the no-field arm, computed from reference images alone) | The field is immaterial for this family and edit. The mechanism would be a retracer. Closed. |
| Stage 1: the light → S … → R class carries more than g of the stationary changed-ROI | The phenomenon the track excludes is material. The track cannot be reported as addressing GM1 on this fixture, whatever else passes. |
| Stage 1: stale passes the changed stationary ROI | Fixture not discriminating; not a result. |
| All pass | Admission only: a rough-lobe baseline is worth specifying. No fidelity, update or cost claim follows. |

A later F′ / S / cost stage with R23's corrections (≥ 8 seeds, bias gate without time condition, multi-scale matching, lifecycle accounting, work counts) would be a separate decision.

## 7. Frank statement

**As research owner I would not test LPTG first as a mechanism.**

- Its best possible outcome is a correct, learning-free rough-lobe interreflection cache with a narrow multi-frame megapixel window. The GM1 question concerns sharp directional transport and the project's contribution concerns learned transport reuse; that outcome answers neither.
- The reviews' common residue (R21 §6) applies to all three tracks: at ER's sharpness the only persistent directional content any of them can hold at practical size is a field band-limited near α\* ≈ 0.2–0.35.
- HIST pattern (R22 §5, via WL56 / WL59 / WL67–70 as cited there): the selective part has not been the dominant cost in any earlier batch, and path-traced re-rendering has been the binding comparator. LPTG's accounting repeats that pattern on paper.

What I would look at first is not a track: the Stage 0 census and the Stage 1 path-class log. They are cheap, track-neutral and decide admission for A, B and C at once. Whether to run anything is the owner's decision.

**Single piece of missing evidence that would change my answer:** the albedo-weighted fitted-lobe roughness census of the frozen ER closures. If a dominant share of reflected energy lies in lobes at or above the α\* reachable with K of a few hundred, the field owns most of the multi-bounce edit response and LPTG becomes worth testing first — as the explicit baseline any learned GM1 mechanism must beat, still not as the GM1 answer. If the share is small, the track closes at zero rendering cost.

## 8. Limits of this response

- Nothing here is measured. Every number is taken from the reviews (their EST / DERIV labels apply) or is arithmetic on Track B's formula.
- Worklogs and `docs/Architecture.md` were not re-opened; contract quotations are as given by the reviews and by `AGENTS.md`.
- Literature was not re-verified; Granier and Drettakis 2001 and Martín et al. 2003 rest on N32's search-summary verification.
- The α\* = 4/√K rule inherits R21's small-angle, piecewise-constant calibration; a factor of 2 is not significant.
