# WL98 — Independent judge J2: research-objective weighting

**Date.** 2026-11-16. **Role.** Independent judge J2 of the design-only batch WL98, "learned directional transport representation after ECRO". My weighting: the research objective first. The question I asked of each candidate: if it worked, would it actually deliver reusable learned transport with sharp-interreflection fidelity after rigid edits and a quality / cost advantage over explicit transport? And is that outcome plausible enough to deserve the experiment?

**Status.** Design judgement only. Nothing was implemented, trained, rendered or adopted. No existing file was changed and nothing was committed. One numpy scratch calculation was run (§2.3). It is design input, not evidence of record.

**Inputs.** All read in full, from `docs/plans/wl98_learned_directional_transport/`:
- `DESIGN_REVIEW.md` (coordinator draft: FLS + ECRO-FC);
- `D1_physics_measure_first.md` (XQT + LLT);
- `D2_learning_generalization_first.md` (LIEM-S + LIEM-B);
- `D3_systems_cost_first.md` (RB-DTO + LER-Δ);
- critiques `X1` (physics / measure), `X2` (learnability), `X3` (systems / cost) and `X4` (prior art).

**Facts verified this session.**
- `docs/Architecture.md:20,25`: C "includes exposed same-asset and same-domain exit/re-entry paths"; "Object identity must not freeze exposed interreflection or merge editable air transport into K".
- `docs/RESEARCH_ROADMAP.md`:
  - the GM1 row asks for sharp glossy/specular *multi-bounce* interreflection under a rigid edit;
  - GM2 "depends on an interpretable GM1 result";
  - **current human decision (2026-11-13):** ECRO is the central learned-global-transport base for GM1, and the next architecture decision must isolate *one falsifiable ECRO-lineage hypothesis*.
- `experiments/ecro86/model86.py:83-104`: the per-patch factors `c`, `a` take as inputs:
  - the current state-ray form factors `Ft` and openness `1 − ΣFt`;
  - world centre, normal and tangent;
  - dense `Ft @ msg` message passing;
  - a scene-global mean token.
- `results/wl98_calc/X3/cost_check2.json` `K_rbdto`: X3's symmetric RB-DTO recount.

**Labels.** [MEAS] measured, with its source. [AUDIT] a WL97 interpretation. [CODE] read from code. [ARITH] / [SCRATCH] arithmetic or a numpy toy (design input). [PRED] my prediction or argument. [HUM] a human decision.

---

## 0. Summary

1. **The decisive structural finding is a pincer [PRED, merged from D1 Prop. I, D2 §3, D3 §2.5, plus my §2.3 arithmetic].** It applies to the central hypothesis as a whole, not to one design.
   - **(a) Transport that is not rigid-invariant.** This is inter-body transport in the scene86 / GM1 family, where every body is planar or convex. Here learned *state* cannot survive an edit. Only weights are reused, so the state is regenerated per edit. That regeneration competes with cheap explicit evaluation:
     - measured role ceilings are 1.02–1.04 for the resolvent alone [MEAS / AUDIT WL96A, D-04];
     - a per-edit generator costs more kernel time than the explicit work it replaces [ARITH D3 / X3].
   - **(b) Transport that is rigid-invariant.** This means body self-transport (D3) and intrinsic media (D2). Here the learned state *is* reusable, but so is a learning-free **per-asset precomputed table of the same transport**. Its edit-time apply cost is equal or lower.
     - The edit-time lever over per-update explicit sweeps therefore belongs to *precomputed rigid-invariant transfer* (PRT-class prior art), not to learning.
     - What is specific to learning is memory, one-off precompute and generalisation to unseen assets. My scratch puts that at about 35–180 MB vs about 2 MB per 200–1,000-patch asset at α 0.03, and seconds of precompute [SCRATCH §2.3, placeholder constants].
   - **Consequence.** A geometry-edit quality / cost advantage that is *attributable to learning* is structurally implausible in every workload the batch can currently build. It can appear only:
     - at memory-bound scale (many unique, very sharp assets); or
     - where per-edit regenerated learned state beats explicit transport at equal quality, which needs deep iterated sharp transport.

     Neither regime exists in a current workload. The falsification plan must credit utility separately to "precomputation" and to "learning" (§5, H-L split).
2. **The coordinator's pair (FLS / ECRO-FC) must not go forward as primary / alternative.** I agree with all four critics and all three designers on this.
   - In GM1, FLS's learned map is ER × incident deposits compressed into lobes, plus a mixture reducer. Its delivery is explicit lobe cubature.
   - ECRO-FC is a confounded lineage diagnostic whose factors are scene-global.
3. **Ranking (§3).**
   1. RB-DTO in a canon-compatible form (**RB-DTO-E**, §4.1): score 4.5;
   2. LIEM-S (+ LIEM-B as its control): 4.0;
   3. the ECRO lineage (ECRO-FC / ECRO-COV / C1 / G1-G2, merged): 2.5 as a candidate, but **mandatory as a diagnostic**;
   4. XQT: 2.5;
   5. FLS: 2.0;
   6. LLT: 1.5;
   7. LER-Δ: 1.0.
4. **Primary: RB-DTO-E** (D3 RB-DTO, reformulated). It is a learned, continuous-lobe operator for the *sharp multi-bounce self-transport of a rigid non-convex body*, reused unchanged under that body's rigid motion. Under my reformulation it consumes the body's **cached explicit self-relation records**: C_self stays owned by C and is validated by the canonical dirty test, so ownership of visibility does not move. D3's original form, where self-visibility is implicit in the learned compiler and protected by a certificate, needs a canon change [HUM].
   - It is the only candidate whose success would answer GM1's own stated question: sharp glossy multi-bounce interreflection under rigid edits, with learned reusable *state*.
   - Its prior of success is low. If the learning-free gates are not passed, it should not be trained.
5. **Independent alternative: LIEM-S** (D2). Its ownership is canonical (interior collided transport in K, air in C) and its rigid reuse is exact. But:
   - it does not learn sharp interreflection: sharpness comes from an exact interface closure;
   - it is GM2 scope, out of roadmap sequence [HUM];
   - it sits in a prior-art class (Vicini et al. 2019);
   - it needs a medium reference that is not yet qualified.
6. **ECRO lineage.** By the 2026-11-13 human decision it is the central base, so Track G must exist. But it is a *learning-free attribution* (H1 / H2 / H3) in the first instance. Its training arms run only if the learning-free ceilings leave room, or if the owner wants the attribution for its own sake.
7. **Proposed verdict: DESIGN_READY_FOR_HUMAN_REVIEW** with conditions:
   - the human decisions in §7;
   - a predeclared, learning-free first stage that I predict [PRED] is the most likely stopping point.

   This verdict carries a mandatory dissent: under the current canon and the current scene86 / GM1 workload family alone, the correct reading is **NO_DEFENSIBLE_CANDIDATE** (§8). Not adoption, not experiment approval.

---

## 1. How I weighed the designs

| criterion | question | weight |
|---|---|---|
| objective fit | If it worked, would the *learned* part carry sharp, spatially distributed, multi-bounce transport that is reused (state, not only weights) after a rigid edit? | highest |
| attributable advantage | Would the advantage over explicit transport belong to learning, or equally to a learning-free twin (precomputed table, sender-side deposition, flux-conserving or adaptive carrier, pooled draws)? | high |
| plausibility | Learnability (coverage, shape generalisation, discontinuities), the size of the cost regime, and prior evidence (WL52, WL68–80, WL96A) | high |
| contract soundness | Measure, passivity, linearity, ownership, independent reference | gate (a fatal defect stops a candidate) |
| falsifiability | Can a cheap, learning-free gate kill it before training? | gate |

I did not defer to the critics. Where I depart from them:

- **ECRO-FC "implicit visibility".** X1, X2 and X4 call ECRO-FC's dependence on `Ft` a case of "dynamic visibility implicitly coupled into the intrinsic representation". That overstates it.
  - `Ft` is computed from *explicit* traced state rays [CODE], so it is visibility as an input, not learned visibility. ECRO is also not an intrinsic representation.
  - The real defects are two:
    - (i) the "local" factors depend on whole-scene inputs (world pose, global token, dense message passing), so no local validity certificate exists and every edit regenerates every factor;
    - (ii) the sender half re-learns ER.
  - The difference matters. A local-frame, bounded-receptive-field ECRO variant (D1 HD-5) would cure (i) and is a legitimate lineage question. Cure (ii) as well, and the lineage's learned content reduces to the resolvent (§4.3).
- **X3's RB-DTO recount.**
  - It is right that D3's regime map is inflated: D3 compared against a uniform-binned twin and used fixed M_in.
  - But the recount assumes that a sender-side twin at K_int = 512 per order reaches *equal quality* on iterated sharp chains. Nothing has measured that. D3 itself argues that sender-side sampling of a non-analytic incident distribution has to be repeated per order, with compounding variance.
  - Both numbers are placeholders. The regime must be measured at matched quality, not at matched K.
- **The verdicts of X2 and X4.** They propose NO_DEFENSIBLE_CANDIDATE because both survivors need scope or canon decisions. I keep that as the scoped dissent. But the canon-compatible reformulation of RB-DTO (§4.1) removes the need for a canon change, and LIEM-S's ownership is canonical. What remains are scope and lineage decisions, which is precisely what human review is for.

---

## 2. Evidence base the judgement rests on

### 2.1 Facts I lean on (verified in the cited sources or in code)

- E-a [MEAS / AUDIT J2]. The solid-angle-mean 4×16 sender over-delivers to CK2 box A by +40–50 %.
  - About half is the averaging convention: flux-conserving 4×16 gives about +20 %.
  - The sign depends on the receiver.
  - At 8×32 the bias is essentially gone, at box A. Problem (2) is therefore solvable learning-free *for extended receivers*.
- E-b [AUDIT J1 / B-04 / B-06].
  - Grazing ring-0 sharp training coverage is 0.11 %.
  - A passive rank-16 operator exact on CK2 exists, but all-incidence fitting trades fidelity (rank 46–59 for 10 % error).
  - Problem (1) is therefore coverage *and* operator-class capacity, not coverage alone (X2 X-1).
- E-c [AUDIT J3]. Exact direction and hit point remove the bias in expectation, but a single draw at the deployed budget gives no average gain (≥ 16× draws needed).
  - X3's L4: one extra pooled draw costs about 0.76 ms of kernel time, so draws can be pooled over the steady frames after an edit. Single-draw variance is a latency problem, not a steady-state one.
- E-d [MEAS WL96A; AUDIT J4 / J5 / D-04].
  - The resolvent-only ceiling is 1.02 / 1.04 with frame.
  - The floor table has a structured 0.010 rel-L2 readout error from patch-spatial averaging seen through the sharp reflection.
  - Problem (3) is role-specific.
- E-e [MEAS fixture]. Scene86 / GM1 has only planar and convex bodies. Intrinsic transport there is pointwise ER, and all nonlocal transport is inter-body C.
- E-f [MEAS WL52]. Learned inter-domain C was valid and stable but 8–60× slower than GPU explicit C, and it failed at contacts and transitions.
- E-g [MEAS WL68, WL71–80]. Learned-K shape generalisation repeatedly failed on new ports, corners and deformation. This is the prior for any shared compiler over unseen shapes (RB-DTO, LIEM-S).
- E-h [MEAS WL87; AUDIT D-05]. The only measured learned lever in the lineage is the source-only update (3.5–3.9×). It requires exact linearity in the source.

### 2.2 What the three independent designers agree on

D1 (Proposition I), D2 (§3) and D3 (§2.5, §7) each reach the same conclusion: in the GM1 surface-only family, no learned object exists that is rigid-reusable, non-local and not ER compression.

The coordinator's draft places its primary (FLS) exactly there. I find no argument in the draft that defeats this convergence. The draft's own HD-1 concedes the point it would need.

### 2.3 Scratch: does learning own RB-DTO's edit-time lever? [SCRATCH, design input only]

- Script: `experiments/wl98_calc/J2/rb_twin_split.py`. Output: `results/wl98_calc/J2/rb_twin_split.json` (numpy, local CPU, deterministic).
- **TwinP** is a learning-free per-asset precomputed body transfer, defined as follows:
  - path-traced self-transport exits per incident cell, at lobe-width incident resolution (the D3 rule: about 200 cells per patch at α 0.1 and about 2,222 at α 0.03);
  - lobe-fitted with K_s lobes of 10 float16 each;
  - applied per incident sample like RB-DTO's sharp channel.
- Both RB-DTO and TwinP are rigid-invariant and reused unchanged after a rigid edit.

| α | N_b | TwinP memory | TwinP precompute (16 events, 0.27 ns / ray, 256–4,096 exits per cell) | RB-DTO memory (D3 example) |
|---|---|---|---|---|
| 0.1 | 200 / 1,000 | 3.2 / 16 MB | 0.04–0.7 s / 0.2–3.5 s | ≈ 2.1–2.5 MB |
| 0.03 | 200 / 1,000 | 36 / 178 MB | 0.5–7.9 s / 2.5–39 s | ≈ 2.1–2.5 MB |

**Reading [PRED].**
- At edit time, TwinP applies a table lookup plus K_s lobe evaluations per incident sample. That is no more than RB-DTO's network query (D3: 5–20 ns, placeholder).
- So whatever edit-time utility RB-DTO shows against per-update explicit sweeps, TwinP shows too. The learning-specific advantage is memory (10–80× at α 0.03), avoidance of seconds-scale per-asset precompute, and continuity across incident directions.
- That is a genuine but *static* lever. It grows with sharpness (α⁻² incident cells) and with the number of unique assets, not with edits.
- The same split applies to LIEM-S against Twin-T (2.0 GB per configuration at CAK resolution [MEAS WL55]).
- The plan must therefore include TwinP / Twin-T and report the two credits separately: U(TwinP) / U(per-update explicit) and U(learned) / U(TwinP).

**Caveat.** TwinP's quality at matched memory is unknown, especially near the incident-direction boundaries where the bounce count changes. That is exactly what Stage 1 measures.

---

## 3. Ranking (candidates merged across designs)

Scores are my research-objective-weighted judgement on a 0–10 scale. They are not evidence.

| rank | candidate | sources (merged) | score | objective fit | attributable advantage | plausibility | decisive unknown |
|---|---|---|---|---|---|---|---|
| 1 | **RB-DTO-E**: rigid-body sharp self-transport operator over cached explicit self-relations (canon-compatible); RB-DTO original as an [HUM] variant | D3 RB-DTO; D2 R-D; D1 O1 (asset-level); X4 MA-1 / MA-5 twins | 4.5 | high: learned sharp multi-bounce, reused *state*, GM1 physics | edit lever shared with TwinP (§2.3); learning owns memory, precompute and unseen-asset generalisation | low–moderate: shape generalisation (E-g), bounce-count discontinuities, multimodal exits, narrow cost regime (X3) | whether a workload exists where (i) per-update explicit sharp self-transport at *matched quality* is a large share of the lifecycle, and (ii) the shared learned operator beats TwinP / compressed transfer on memory at equal fidelity on held-out bodies |
| 2 | **LIEM-S**, with LIEM-B as its H2 control | D2 LIEM-S / LIEM-B | 4.0 | partial: reusable learned intrinsic transport with canonical ownership; sharpness exact, not learned; GM2 | Twin-W (cost) only for long walks; Twin-T (memory) static; the edit cannot discriminate it (X2 X-9) | moderate learnability (Vicini 2019 precedent); headroom tension (D2 §5.2); reference unqualified | I0.2: whether a configuration has *both* a long-walk share of the lifecycle and exit structure above noise; I0.1 reference qualification |
| 3 | **ECRO lineage**: ECRO-FC / ECRO-COV / C1 / C1f / G1–G2, factorised | DR ECRO-FC; D1 C1 / C1f; D2 G1 / G2; D3 Track G | 2.5 as a candidate (diagnostic mandatory) | human-chosen base, but its learned content in GM1 collapses: sender = ER, flight explicit, global = resolvent | none attributable to geometry edits on current workloads; the source-update lever survives (E-h) | H1 / H2 attribution is plausible and cheap learning-free; utility is not plausible | class floors (carrier ladder × operator rank × incidence weighting) against fine references; role ceilings against a quality-matched engineered frontier |
| 4 | **XQT**: learned flux-conserving redistribution of exact samples | D1 XQT | 2.5 | low: a learned estimator holds no transport state (D1 says so) | twin / oracle 1.0–1.33 [SCRATCH D1]; no pooling gain for glints whose footprint is smaller than a patch (X1 §2.6); pooling over frames (E-c) | trainable, but predicted decorative | S1-c with an oracle restricted to XQT's own information (X2) |
| 5 | **FLS**: flux-allocated spatio-angular lobe state | DR FLS; D1 C1; D2 R-C; D3 C-patch | 2.0 | none in GM1: ER compression plus mixture reduction; explicit cubature delivery | ELT (with image-source shear) shares all of it | lobe count about 10 per element per light at CK2 grazing [CALC X1 §1.1]; generic lobes fail at grazing [SCRATCH D2 §5.1] | optimal-fit-vs-ELT gap at equal K and Q (X2) |
| 6 | **LLT**: learned lobe transfer from probes | D1 LLT | 1.5 | it must infer unprobed visibility to beat its twin | photon twin at about 0.3 ms per 10⁶ rays | WL52 prior | S1-d collision test (generator-circular, X4) |
| 7 | **LER-Δ**: environment resolvent + Δ iteration | D3 LER-Δ | 1.0 | per-environment state (Native Reuse class) | ghosting by cancellation; wrong convergence condition (X1 §1.4) | low | none worth gating |
| — | A (continuous kernel), B (learned sparse table), C (patch lobes), O1, R-E, NRC | all designs | rejected | ER compression, learned C, or a dominated twin | — | — | — |

Learning-free arms that every plan must carry are controls, not candidates: flux-conserving 4×16 / 8×32 / adaptive tables, ELT with image-source shear, sender-side deposition, an MIS combination, pooled C3URX1, TwinP / compressed transfer, and an engineered EXP / C3UR.

---

## 4. Judgement per candidate

### 4.1 Primary: RB-DTO-E (reformulated from D3 RB-DTO)

**Why it ranks first.**
- It is the only candidate whose learned part, if it worked, would be:
  - high-frequency: continuous lobe position and width;
  - non-local and multi-bounce;
  - *reused as state* after a rigid edit;
  - inside GM1's stated scope: sharp glossy multi-bounce interreflection.
- Each other candidate either has no learned transport content (FLS, XQT, ECRO-lineage sender), infers visibility (LLT), is per-scene (LER-Δ), or leaves sharpness to exact physics (LIEM-S).

**Reformulation that removes the canon conflict [PRED].**
- **D3's form.** The shared compiler infers self-visibility from mesh tokens, protected by a self-path-region certificate. This puts exposed same-asset air paths into a learned body operator. Architecture §2 forbids it [HUM: canon refinement].
- **RB-DTO-E (my reformulation).**
  - The body's self-relation records (rays from S_b whose first hit is S_b) are traced explicitly once, in the body frame. They are cached under rigid motion and validated after each edit by the canonical dirty test, the same test ECRO and C3UR use for their relations.
  - The learned operator consumes those records (plus exact ER) and outputs the multi-bounce exit pushforward as passive, normalised lobes.
  - C_self segments remain owned by C. The learned operator owns only the *composition* (amortisation of (K∘C_self)^j for j ≥ 1), exactly as ECRO's learned global term consumes explicit relations. That is admissible under the reinstated ECRO lineage.
  - A partial intrusion dirties specific records, not the whole body, so the learned operator re-reads updated records. X1's "invaded paths cannot be subtracted" objection then becomes a measurable question of generalisation to modified records. Whole-body explicit fallback remains the conservative control.
- **What this costs.**
  - By D1's Proposition I, the learned operator is now an amortiser (lever b) over explicit records, with no information beyond them.
  - Its twins are explicit iteration over the same records (Twin 1, per incident change) and TwinP (precomputed from the same records).
  - That is the honest position. The learned contribution must show up against TwinP on memory or continuity at equal fidelity, and against Twin 1 on per-update cost at equal quality.

**Contract corrections carried from the critics, all accepted.**
- Passivity budget: E_esc is computed exactly from the cached records at compile time, never "bounded from below" (X1 §2.3).
- Per-patch output-cache mixture reduction breaks superposition. Either cache per incident sample (linear) or declare a superposition metric with a tolerance [HUM] (X1 §2.2, X3 L5).
- Sharp exits leaving the body to other bodies need sender-side deposition from the cache, charged in L_rb (X3).
- A convex-body null test (N_b ≡ 0), path-tag disjointness, rigid equivariance to float tolerance, and per-realisation passivity of every iterated explicit factor (X1 §1.3).
- An S1-d-style collision-pair test on any learned inference beyond the records (X2), with collision pairs drawn *outside* the training generator as well (X4 D1-P3).

**Plausibility [PRED].** Low to moderate.
- **Learnability risks:**
  - shape generalisation of sharp self-transport to unseen bodies (E-g);
  - exit-mode discontinuities where the bounce count changes (X2);
  - grazing multimodal exits that generic lobes fit poorly (D2 §5.1).
- **Utility risk:** the cost regime may not exist at matched quality (X3), and the edit lever belongs to TwinP (§2.3).
- **Novelty:** the composition is PRT / OSF / TransGI class (X4). The niche is "sharp + shared".
- **What the experiment is for.** The learning-free Stage 1 (§5) is cheap enough to be worth running on this candidate. Training is warranted only if Stage 1 shows (i) a matched-quality regime where per-update explicit self-transport is a large lifecycle share, and (ii) an oracle-vs-TwinP gap beyond paired noise.

### 4.2 Independent alternative: LIEM-S (D2)

- **Independent of RB-DTO-E** in learned object (intrinsic medium exit measure vs surface self-transport), physics regime, twins and failure modes, and in the human decision it needs (scope vs lineage).
- **Strengths:**
  - canonical ownership;
  - exact rigid equivariance;
  - passive normalised output family;
  - designed impulse coverage (μ_D vs μ_S as a treatment);
  - no exposure bias;
  - three named twins.
- **Weaknesses for my weighting:**
  - Sharpness is exact (interface closure), not learned.
  - The learned part is smooth wherever walks are long enough to pay (D2 §5.2 [SCRATCH]).
  - The edit cannot discriminate it from Twin-T (X2 X-9).
  - It is out of milestone sequence: GM2 depends on GM1 [HUM].
  - The prior art is close (Vicini 2019, public code; X4).
  - The reference for refractive-boundary media is unqualified.
- **Required repairs** (X1–X4, accepted):
  - the kill rule must make "faster than Twin-W at matched quality" *necessary* for a utility claim, with memory reported but not decisive (X1 #14, X3);
  - I0.2 needs a measured Amdahl U0 on the walk's lifecycle share;
  - add a per-shape fitted arm (X2);
  - Vicini 2019 as an external arm (X4);
  - declare the interface model before reference qualification (X1 #16).

### 4.3 ECRO lineage: diagnostic, not candidate

- **Why it has no learned content left in GM1.** If the sender half re-learns ER, that is BRDF compression [MEAS J1: non-passive rows; WL93D: about 80 % wrong]. If ER replaces it, the learned content reduces to the flight weights over explicit rays plus the global resolvent. The resolvent's CK2 ceiling is null [MEAS WL96A].
- **Why it still runs.** It is the human-chosen base, so its failure must be *attributed* before the owner decides on lineage.
- **How to run it.**
  - The H1 / H2 attribution is largely learning-free: carrier floors and operator-class floors against fine references.
  - Training arms must be factorised (X1 / X2 / X3), each changing one thing:
    - **ECRO-R:** a seed pilot;
    - **ECRO-COV:** coverage only;
    - **ECRO-SUP:** physical S1 target only, never M64;
    - **ECRO-FCc:** carrier only, at matched state rays per bin;
    - optionally **ECRO-LOC:** local-frame inputs and bounded receptive field (D1 HD-5);
    - plus a rank axis on the class floor.
  - The coordinator's ECRO-FC (three changes at once) is not readable as H2.

### 4.4 XQT, FLS, LLT, LER-Δ

- **XQT.** I agree it is the best-posed *learning problem* (X2) and the cleanest physics contract (X1). But it carries no transport state.
  - Its edit reuse is that of any pretrained denoiser (X4).
  - Its pooling cannot help glints narrower than a receiver patch (X1 §2.6).
  - Its own toy predicts no learning margin.
  - If HD-1 admits learned estimators, it is a legitimate comparator family. It should not be the batch's primary.
- **FLS.** It is not forwarded.
  - The lobe-count result (about 10 separable lobes per element per light at CK2) and the image-source-shear observation (X1 §1.1) are decisive in GM1. For direct light the sheared lobe is analytic, so nothing is left to learn.
  - The radiance-measure formula is inconsistent near the horizon (X1 §1.2).
  - The NULL sink is unphysical learned absorption.
  - Its learning-free form, **ELT with image-source shear**, is kept as a comparator.
- **LLT.** Rejected: it is either decorative or a learned-visibility inference.
- **LER-Δ.** Rejected:
  - its convergence condition is mis-stated (X1 §1.4);
  - learned-minus-explicit cancellation leaves ghosts;
  - its state is per-environment.

---

## 5. Merged minimal falsification plan

**Design rule.** One plan, learning-free first. Every stage is frozen before its inputs exist. Every stop is final for the batch. No rescue architecture. Every number marked [HUM] is unset, and none is invented here.

### 5.1 Hypotheses

- **H1** (coverage / optimisation). This is the ECRO-lineage hypothesis. It is split three ways (X2 X-1):
  - **H1-opt:** learned arm vs the operator-class floor at the *same* incidence weighting;
  - **H1-alloc:** the floor's movement with the weighting;
  - **H1-cov:** natural vs designed coverage.
- **H2** (angular representation), split two ways:
  - **H2-carrier:** a carrier-floor ladder against fine references;
  - **H2-op:** the operator-class floor over a rank axis, e.g. r ∈ {16, 48, 64} [HUM].
- **H3** (no geometry-edit utility). Measured as a quality-matched U against the engineered explicit frontier at T1 with fixed weights.
  - The credit is split: **U_pre** = U(TwinP) / U(per-update explicit) goes to precomputation; **U_learn** = U(learned) / U(TwinP) goes to learning.
  - An H3 refutation for a *learned* candidate requires U_learn ≥ U_min [HUM] on two evidence workloads.
- **H-L** (learning beats its strongest twin). An oracle of the learned family, *restricted to the learned model's own information and linearity* (X2 X-5), against the strongest learning-free twin at the same information and budget.

### 5.2 Stages

**S0 — Freeze, instruments, workloads, frontier (learning-free, shared).**
- **S0a. Reference-instrument qualification** (X1 §2.1, §4.1).
  - Add to the frozen-ER path tracer: exact-direction outgoing probes; cos-weighted cell tallies; incident flux per receiver cell; path tags (self vs inter-body, event count, last sender); and deeper kmax with a truncation-residual estimate.
  - Validate the additions against the deterministic exact-ER first-flight quadrature, and against the existing `field_src` tallies where the two overlap.
  - Use independent halves throughout. M64 is never truth.
  - Stop: `STOPPED_REFERENCE_UNQUALIFIED`.
- **S0b. Workload admission.** Admission uses geometry-only or SNR rules declared before any arm output exists. Every evidence workload needs path-traced references at T0 and T1, with no T1 data used anywhere in training or selection.
  - **GM1 Track G:**
    - **W-G2** (grazing sharp). A new configuration, admitted on the *delivered* sharp share with glint-core and tail receivers, not on sender irradiance (X2 X-7).
    - **W-N** (non-grazing sharp).
    - **W-H** (independent held-out). A declared distribution shift: another geometry family or partition, radiance uninspected (X2 X-8).
    - **CK2.** Diagnostic only, with a new T1 reference, and excluded from all gate arithmetic (X2 X-11).
  - **Track B** (only if HD-L / HD-C allow RB-DTO-E):
    - **W-graze-B:** glossy groove or bracket lit at cos_i below a [HUM] value;
    - **W-broad-B:** the same family lit near-normal, moderate α;
    - **W-held-B:** an unseen body class plus an unseen light class;
    - **W-adv-B:** an adversarial workload with moderate self-transport and frequent intrusion (X4 D3-P2);
    - edits: one rigid move of a body; one occluder move near the body; one record-dirtying intrusion.
  - Stop: `STOPPED_UNQUALIFIED_WORKLOAD`. Amendments are explicit and human-approved.
- **S0c. Engineered explicit frontier and cost anatomy** (X3 §5, D3 Stage 0).
  - Arms:
    - R0 / EXP and C3UR;
    - C3URX1 at 1–16 pooled draws;
    - sender-side deposition, and MIS of gather + deposition;
    - FC 4×16, FC 8×32 and adaptive FC tables;
    - ELT with image-source shear;
    - in Track B: Twin 1 (sender-side chain tracing per incident change, quality-matched) and TwinP (precomputed per-body lobe or compressed transfer).
  - Measurement conventions:
    - one unified kernel basis plus wall time, with overhead disclosed;
    - compile and cache misses counted per update;
    - a draw-pooling convention: time-to-tier measured from the edit, with progressive explicit draws allowed;
    - tiers away from every arm's floor, with a NOT-REACHED rule;
    - light counts 1 / 4 / 16 and N ∈ {1,270, about 5k, about 11k} [HUM].

**S1 — Learning-free decisions (no training).**
- **S1-G (ECRO lineage).**
  - Carrier floors on W-G2 / W-N / W-H, against the fine references, fitted on half A and scored on half B (X2 X-2): SA 4×16, FC 4×16, FC 8×32, adaptive, exact per-ray.
  - Operator-class floors (light-independent, non-negative) over the rank axis and at the declared incidence weightings.
  - Role ceilings at zero and at measured learned cost, against the quality-matched S0c frontier.
  - **Gate G:** if no ECRO-lineage role reaches U_min [HUM] at matched quality on ≥ 2 admitted workloads, then H3 is supported *at ceiling* for the lineage on GM1. Its training arms then run only by HD-T.
- **S1-B (RB-DTO-E).**
  - U0 = L / (L − A_intra), with A_intra measured from the *quality-matched* Twin 1.
  - The per-body oracle floor of the lobe family, fit on half A and scored on half B.
  - TwinP at matched memory.
  - The gap between the learned-family oracle (restricted to the records) and TwinP.
  - A collision-pair test for any inference beyond the records.
  - **Stop RB-DTO-E** if any of the following holds: U0 < U_min; the floor fails the representation gates on W-graze-B; or the oracle–TwinP gap is within paired noise on both the fidelity and the memory axes (learning decorative).
- **S1-I (LIEM-S, only if HD-S opens GM2).** D2's I0.1 reference, I0.2 headroom with a measured Amdahl U0, I0.3 twin frontier, and I1 floors. The kill rule follows §4.2.

**S2 — Training (survivors only).**
- **S2a. Seed pilot.** Train ≥ 5 seeds of the cheapest learned arm and measure the seed SD on each gate metric. Then compute the seed count n for the declared minimum effect [HUM].
  - X2's power table: 3 seeds detect only effects of ≥ 2.4× at WL93D-sized spread, or ≥ 11.6× at the OWN box-A spread.
- **S2b. ECRO factorised arms** (only if Gate G passes, or by HD-T): ECRO-R, ECRO-COV, ECRO-SUP, ECRO-FCc, and optionally ECRO-LOC. Each is read against the weighting-matched floor of S1-G.
- **S2c. RB-DTO-E arms.**
  - Shared model on natural vs designed (grazing-stratified) incident coverage.
  - A per-body fitted arm (generalisation vs capacity).
  - An exposure-bias measurement: error on reference-delivered vs pipeline-delivered incidence, per sweep (X2 X-6).
  - Contract tests run before any evaluation; a failure is `STOPPED_CONTRACT`.

**S3 — T0 → T1, fixed weights.**
- Arms: all survivors, the S0c frontier at quality-matched points, TwinP, and STALE.
- Edits: rigid, occluder and record-dirtying.
- Measures:
  - changed-region fidelity and signed-change slope / correlation;
  - leakage / ghosting outside the predicted support;
  - extra T1 − T0 error;
  - selective = full for the reuse certificate.

**S4 — Systems.**
- Update critical path, steady frame (including lobe-query overhead), learned inference, relation recompute and dirty test, memory, and compile / training as one-off costs.
- The U_pre / U_learn split.
- External baselines: Native Reuse of an object-level neural asset (an OSF / TransGI / 8DNA type) measured, or marked UNMEASURED with a reason. Neural Full Recompute applies only if that asset is actually invalidated. Path-tracing cost is reported as reference cost only.

### 5.3 What can and cannot share data, budget and reference

| comparison | training data | budget | physical reference |
|---|---|---|---|
| ECRO-R / COV / SUP / FCc / LOC | same generator (COV changes the measure: the treatment) | matched steps, samples, parameters | shared |
| ECRO arms vs S0c frontier / floors | none on the explicit side | quality-matched, not training-matched | shared |
| RB-DTO-E shared / designed / per-body | same body generator and exit-sample budget | matched | shared, plus per-body exit references |
| RB-DTO-E vs TwinP | TwinP is built from the same exit samples | matched memory or quality, stated | shared |
| Track G vs Track B vs Track I | **no** (different workloads; RB-DTO-E ≡ 0 on GM1, which is a contract check) | no | renderer, metrics, statistics and conventions only |
| ReST-GI vs external Native Reuse | no | lifecycle-aligned only | shared if aligned per the baseline gate |

### 5.4 Statistics (binding, from the WL97 lessons)

- Paired evaluation with common random numbers.
- Block-bootstrap SE over pixels and patches.
- Reference noise from independent halves, kept separate from treatment differences.
- Seed treated as a random effect, with n taken from the S2a pilot.
- ≥ 16 relation draws, reporting the deployed-draw distribution.
- Never an L1 noise floor as a paired tolerance.
- Heavy-tail guards (per-geometry tables, median of means).
- Two-workload confirmation for every positive claim.
- CK2 is never a gate input.
- Dev runs use non-evidence inputs only.

### 5.5 Why both synthetic contract tests and real scenes are needed

- **Contract tests** check measure, per-realisation passivity, linearity / superposition, rigid equivariance, the convex null, path-tag disjointness, the furnace and the image-source exactness of planar mirrors. They have exact answers, and they catch compensating errors that image metrics hide (WL86A).
- **Real scenes** are the only place to measure coverage under scene-induced incidence, composition over bounces, readout through the sharp vertex, edit response, ghosting and economics.
- A pass on one kind without the other cannot be attributed.

---

## 6. Grafts (what the final synthesis should take from each source)

- **From D1:**
  - Proposition I (the three levers) as the framing of what learning can add;
  - measure rules M1–M3 and the projected-disk flux-exact carrier for controls;
  - local-frame, bounded-receptive-field learned inputs (HD-5) for any learned arm;
  - the T1–T10 contract tests;
  - the decision-mapping table;
  - admission on the delivered sharp share with glint-core and tail receivers.
- **From D2:**
  - a designed impulse coverage measure (μ_D vs μ_S) as a treatment;
  - event-likelihood supervision against unbiased physical samples;
  - three named twins per candidate;
  - the G0 ceiling stop rule;
  - ECRO-R as the seed-spread pilot;
  - heavy-tail guards;
  - the I0.2 two-property headroom admission.
- **From D3:**
  - a cost-anatomy Stage 0 before any training;
  - the convex-body N_b ≡ 0 test;
  - path-tagged component supervision;
  - the B2 → B5 → B3 / B4 attribution ladder;
  - a certificate / intrusion control edit;
  - an unseen-body-class held-out.
- **From DESIGN_REVIEW:**
  - ELT (made strong by X1's image-source shear) as an explicit comparator;
  - the T-a … T-r list (with T-a corrected to test radiance, X1 §1.2);
  - the WL97 statistics block;
  - dev runs on non-evidence inputs.
- **From X1:**
  - the reference-instrument gate;
  - per-realisation passivity;
  - fit / score split halves;
  - matched event truncation;
  - exact E_esc;
  - whole-body fallback as the conservative control.
- **From X2:**
  - the H1 / H2 three-way split with a rank axis and weighting-matched floors;
  - seed power from a pilot;
  - oracles restricted to the learned model's information;
  - per-instance fitted arms;
  - an exposure-bias metric;
  - distribution-shifted held-out workloads.
- **From X3:**
  - floor-through-frame quality matching with tiers away from floors;
  - the draw-pooling convention;
  - a unified kernel basis and compile accounting;
  - an engineered explicit comparator;
  - a multi-light axis;
  - a symmetric RB-DTO recount.
- **From X4:**
  - the closest prior art as measured comparators (compressed PRT / TransGI / OSF Native Reuse for RB-DTO-E; Vicini 2019 for LIEM-S);
  - an adversarial workload.
- **From J2 (this report):**
  - the U_pre / U_learn credit split with TwinP;
  - the RB-DTO-E canon-compatible reformulation (records as input, C keeps ownership of C_self).

---

## 7. Human decisions required

- **HD-L (lineage).** The 2026-11-13 decision asks for one ECRO-lineage hypothesis. May the next GM1 architecture hypothesis be RB-DTO-E, which is related to ECRO's learned-over-explicit-relations form but is localized to rigid bodies? Or must it stay within ECRO proper? If ECRO proper, my reading is that no defensible learned ECRO-lineage *candidate* exists on current workloads. Only Track G attribution remains.
- **HD-C (canon).** Choose between RB-DTO-E (records as input; canon-compatible by my reading, which the owner must confirm) and D3's original RB-DTO (implicit self-visibility under a certificate; needs a refinement of Architecture §2).
- **HD-S (scope).** Whether LIEM-S may run in GM2-type media before an interpretable GM1 result, against the roadmap dependency. This includes:
  - adding an analytic rough-dielectric closure next to ER;
  - the reference renderer choice.
- **HD-W (workloads).** Extending GM1 with non-convex glossy rigid bodies (in GM1's physics scope, but new). Admission values: s_min, the grazing threshold, α range, body classes and the held-out class.
- **HD-1 (estimators).** Whether learned estimators or guiding (XQT; X1–X4 missed alternatives) are in scope as candidates, or as comparators only. I recommend comparators only, but this is a human decision.
- **HD-E (comparator definition).** Which learning-free arms count as "explicit transport" for H3. In particular, is a per-asset precomputed rigid-body transfer (TwinP / PRT class) an explicit comparator? This decides whether the U_pre / U_learn split is required.
- **HD-N (numbers).**
  - U_min (1.25 is the WL96A precedent, not a derivation);
  - tiers and minimum meaningful effects per ROI;
  - gate tolerances;
  - superposition tolerance;
  - the coverage quota;
  - the rank axis;
  - α_split;
  - the minimum effect that sets the seed count;
  - the draw count.
- **HD-U (utility conventions).**
  - kernel vs wall basis;
  - whether an engineered explicit comparator is required;
  - frame inclusion and spp;
  - pooling horizon;
  - which lifecycle events count;
  - light counts;
  - the scale axis.
- **HD-T.** Whether the ECRO training arms run for attribution if Gate G closes the lineage for architecture purposes.
- **HD-X.** External baselines: Native Reuse measured or UNMEASURED; Neural Full Recompute only if the asset is actually invalidated.
- **HD-B.** Batch classification under the numbering policy, compute budget and server.

---

## 8. Dissent the final synthesis must preserve

1. **Scoped NO_DEFENSIBLE_CANDIDATE** (X2, X4; and each designer's own scoped reading: D1 §8 under HD-1 "no", D2 §3 / §9, D3 §7).
   - Under the current canon and the current scene86 / GM1 workload family, no learned candidate meets the brief.
   - A positive H3 there is predetermined to be impossible on the recorded cost anatomy.
2. **INSUFFICIENT_EVIDENCE** (X3).
   - The only candidates with a possible regime live in workloads that do not exist.
   - Symmetric recounts put that regime at J_int · K_int ≳ 10³–10⁴ per patch, or ≥ 10–30 internal events with a small network.
3. **D2 vs D3 on rigid-cluster self-transport.** D2 (R-D) rejects the idea on canon grounds and weak learned value: its twin is the cached explicit relation. D3 makes it primary. My §2.3 calculation sharpens D2's point: the edit lever belongs to precomputed transfer. My primary keeps the idea only in its canon-compatible form.
4. **The coordinator's position** that FLS's Λ_θ is "not local BRDF compression", because it maps distributed incident states under a downstream loss. Every designer and every critic rejects that reading for GM1. It is preserved as a minority position, and HD-1 of the draft remains the owner's to answer.
5. **D1's position.** In GM1 the honest learned transport is an estimator. If HD-1 admits estimators, XQT should be the GM1 candidate.
6. **D2's position.** Canonical ownership should decide the primary, which would make LIEM-S primary over RB-DTO.
7. **My own reservations, which apply even to the primary.**
   - (i) RB-DTO-E's edit-time advantage over per-update explicit sweeps may be wholly attributable to a learning-free precomputed transfer (§2.3).
   - (ii) Its learned contribution may reduce to memory, precompute and generalisation, which are static levers.
   - (iii) Its most likely outcome is a learning-free stop at S1-B.
8. **Novelty (X4).** Every candidate's core mechanism exists in the literature. The ReST-GI addition is a contract (passivity, measure, ownership, certificate), so novelty requires measured comparison against the closest prior art.
9. **ECRO-FC "implicit visibility" (my disagreement with X1 / X2 / X4).** `Ft` is explicit visibility *input*. The fatal defects are whole-scene conditioning, per-edit full regeneration and re-learning ER.

---

## 9. Verdict and rationale

**Proposed verdict: DESIGN_READY_FOR_HUMAN_REVIEW.** This is not adoption and not experiment approval.

- **What is ready.**
  - A primary (RB-DTO-E) and an independent alternative (LIEM-S). Each has a complete contract once the critics' corrections are carried in.
  - A merged plan with the following properties:
    - (i) it separates H1 / H2 / H3, with factorised arms and a weighting-matched rank axis;
    - (ii) it splits utility credit between precomputation and learning;
    - (iii) its first decisive stage is learning-free and can close each line, including the ECRO lineage, before any training.
- **Why it is a human-review verdict.** Both candidates need a human decision on lineage, scope or workload before they are candidates at all. The candidates are not kept alive by force: each passes the brief's six unsuitability criteria on its own corrected contract.
- **The scoped dissent stands.** Under the current canon and the current GM1 workload family alone, the correct reading is NO_DEFENSIBLE_CANDIDATE. If the owner declines HD-L / HD-W and HD-S, that is the batch's verdict.
- **Plausibility [PRED].** A positive, learning-attributable geometry-edit utility is unlikely in any workload buildable now. The experiment deserves its learning-free stage. Training deserves to run only if that stage finds a matched-quality regime and an oracle-vs-strongest-twin gap.

---

## 10. Provenance

- **Read in full:** `DESIGN_REVIEW.md`, `D1_*`, `D2_*`, `D3_*`, `X1_*`, `X2_*`, `X3_*` and `X4_*` (this directory).
- **Canon and code checked:**
  - `docs/Architecture.md:20,25`;
  - `docs/RESEARCH_ROADMAP.md` (GM0–GM4 table, current decision boundary 2026-11-13);
  - `experiments/ecro86/model86.py:83-104`;
  - `results/wl98_calc/X3/cost_check2.json` (`K_rbdto`).
- **Scratch:** `experiments/wl98_calc/J2/rb_twin_split.py` → `results/wl98_calc/J2/rb_twin_split.json` (numpy, CPU, deterministic, placeholder constants labelled).
- **Not done:** no existing file was modified; nothing was trained, rendered or committed; no GPU or server was used.
