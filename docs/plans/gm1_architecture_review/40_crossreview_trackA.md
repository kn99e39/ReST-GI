# GM1 cross-review — Track A (Directional Delta Transport) response

**Status:** design-only cross-review record (2026-11-10), `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. This is the only file written; no existing file was changed. It authorises nothing.
**Author role:** fresh agent acting as Track A's advocate after the seal was lifted.
**Read in full:** `00_PROBLEM_BRIEF.md`; the three designs `10_`, `11_`, `12_`; the four reviews `20_`–`23_`; the novelty search `30_`. `31_` and `32_` were read in their verdict and mechanism-list sections only. `docs/RESEARCH_ROADMAP.md` (GM1–GM3 rows) and `docs/Architecture.md` (§3, §6 and candidate table) were read in the original. Worklog numbers below are **not** re-read; they are taken from the reviews, which state where each was checked.
**Literature:** no web search was run by me. Every citation is taken from `30_novelty_trackA.md` / `32_novelty_trackC.md` at the verification level recorded there.
**Labels:** FACT, HIST (original scope only), DERIV, INF, HYP, EST (order-of-magnitude estimate, not a measurement).

## 0. Position in brief

- Track A does **not** survive as a standalone GM1 mechanism. The tabulated 4D outgoing state is withdrawn (two independent FATAL findings, both accepted).
- What survives is a **solver policy, not a representation**: residual-driven, overwrite-only update, defined on someone else's state. Its only technically motivated host is Track C's band-limited incident field.
- I would not test Track A first.

## 1. Strongest competitor

**Track C (LPTG).** Reasons (all INF from the reviews):

- It is the only design with a persistent transport state that is affordable to build: about 1–3 image-equivalents against about 7,600 for Track A's table (`22_` §5).
- It is the only one with a proved same-architecture F ≡ S (`20_` §4.1), which `docs/Architecture.md` §6 requires (FACT).
- It stores **incident** radiance and applies exact ER at query time. Outgoing radiance in a never-consumed direction is therefore available for the rough part without an outgoing table. This removes Track A's own most dangerous precondition (`10_` §2, §10.1) for that part. Track A has no answer to this.
- Every physically adequate repair of DDT-D converges on C's stored links (`20_` §2.4, §5.3; `21_` §2.4).

Track B is the strongest **control**, not a competing design; it claims no mechanism (`11_` §0). Track A's on-demand form (arm A3) is Xu et al. 2024 and belongs in Track B as its stronger PR2 arm (`31_` §0.2 says the same from B's side).

## 2. Challenge to Track C's central assumption

**Assumption challenged:** that sparse link invalidation (invalid fraction `f` ≪ 1) makes the edit update selective, with the field "always re-solved, warm-started" as a minor common term (`12_` §3, §11).

**Derivation (DERIV; uniform-landing assumption stated).**

1. *Fan-out.* The `N·K` links land on surfaces interpolated from `N` cache points, so each cache point is referenced by about `K·t` links (`t` = interpolation taps), assuming links land roughly uniformly by area. `f_R` contains a Lambert lobe (FACT, WL82 via `12_` §1), so every row of **B** is dense: a change in **one** incident cell `I(c′, j′)` changes `Φ_R(x, ω)` for **all** `ω` at points interpolating `c′`. One changed cell therefore changes about `K·t` links at the next sweep.
2. *Support saturates in two sweeps.* A cache point has at least one invalid link with probability about `1 − exp(−K·Ω/2π)`, with Ω the solid angle of `O0 ∪ O1`. With the reviewers' own figures (K = 128, Ω ≈ 0.03 sr, i.e. 0.6 nodes; `22_` §4.3) that is about 45 % of cache points after the link update. After one sweep all outgoing directions of those points have changed; after the second, essentially every link has a changed sender. The **unsigned** residual support is dense for `f` of a few per cent.
3. *Magnitude.* A single changed node alters the Lambert response by about `(a_L/π)·(2π/K)·ΔI = 2·a_L·ΔI/K`: about 0.8 % of `ΔI` at `a_L` = 0.5, K = 128 (EST; `a_L` unchecked). The mass contracts by the albedo per sweep while the support grows by `K`.
4. *Consequence.* An output-sensitive solve on C's graph is possible only by **thresholding magnitudes**. A threshold τ leaves an error bounded by `τ/(1 − ρ)` in the sup norm (`20_` A-1, which applies to C's "declared residual" equally; C does not state its norm). At ρ = 0.8 a 1 % threshold certifies only 5 %.

**Counterexample.** Small box (Ω ≈ 0.03 sr from most of the room), K = 128, ρ ≈ 0.8, gate anchored at about 2 % (`23_` X2 with the ER gap about 0.04). `f` ≈ 3 % (`22_` §4.6), yet a certified selective solve needs τ ≤ 0.4 %, which is below the 0.8 % single-node second-order change: every link is marked. The dense solve is not an implementation gap; on this bound it is forced.

**What this establishes.** C's selective-update economics equal Track A's φ_τ question at state size `N·K`. C has not escaped the delta-sparsity problem; it has moved it into a term it does not count as selective. H-edit (F ≡ S) is silent on it, and `22_` C1 (S ≈ F in cost) is the expected outcome unless a thresholded residual is both small and safe. The bound in step 4 is worst-case; the realised accumulation is unmeasured (§6 addresses exactly this).

I add nothing to the reviewers' other objections to C (incident field not band-limited, α\* ≈ 0.17–0.34 at K = 128, frozen-sample bias); they stand as written.

## 3. Objections against Track A: ACCEPT or REBUT

### 3.1 FATAL

| Source | Objection | Response |
|---|---|---|
| `21_` A1 | 4D outgoing state ∝ α⁻⁴; 38 GB at α = 0.05 for a 5 m² court; infeasible below | **ACCEPT.** The tabulated `b0(x, ω_o)` is withdrawn as a GM1 state. `10_` §7 sized only the easiest case. |
| `22_` A1 | Single-view break-even φ_τ < 5·10⁻⁴…8·10⁻³; geometric lower bound ≈ 3 % | **ACCEPT** for the state as specified. **Scoped REBUT of transfer:** the inequality `φ < P/(N_x·N_ω)` is a property of the state size, not of the delta policy. On a state of 2.6·10⁷ band-limited cells the first-order update is `f` × (1–5·10⁸ rays), i.e. a few ms against 0.16 s for 1024 spp at 256² (`22_` §4.1, §1.2, EST). There the binding term is the solve (§2), not the dirty regather. This rebuts only the extension of the FATAL to the surviving policy. |
| `22_` A2 | Continuous motion: full update per frame, no amortisation | **ACCEPT.** No claim is made for sequences. |
| `20_` A-3 | No propagation of dirtiness through lobes sharper than the state | **ACCEPT.** The "band-limited state plus path continuation" form (`10_` §2) is withdrawn as a distinct mechanism: with stored chains it is LPTG, without them it is residual path tracing. |
| `23_` A-1 | F1 infers "no economic content" from an arm that pays two T0 path traces per lookup | **ACCEPT.** F1 is withdrawn. Arm A3 is withdrawn from this track (it is a Track B arm). |

### 3.2 MAJOR

| Source | Objection | Response |
|---|---|---|
| `20_` A-1 | Tolerance norm undeclared; flux norm drops sharp features; `1/(1 − ρ)` amplification | **ACCEPT.** Tolerance is redefined as per-cell **peak** radiance at the consumer's resolution, with the amplification in the bias budget. φ_τ rises accordingly. |
| `20_` A-2 | Mover-side discovery does not certify receiver-cell coverage; S ≠ F; "independent of scene size" false | **ACCEPT.** The scene-independence statement (`10_` §3, §7) is withdrawn; discovery is at best output-sensitive in first-order dirty cells. F ≡ S is obtainable only by testing the state's own stored gather rays, i.e. C's segment certificates. |
| `20_` A-4, `21_` A4 | Stochastic regather: noise and static error read as change; seams | **ACCEPT.** Marking must propagate a deterministic unsigned upper bound; values must be regathered with the build's random numbers. Both require stored links. |
| `20_` A-8, `23_` A-2, A-3 | A3 is not DDT-D; wrong control; free T0 prior | **ACCEPT.** See F1 above. The masked cold re-render (`23_` X5) is adopted as a required control. |
| `21_` A2 | Every bounce passes through the state: ≈ 5 % peak loss per lookup, 10–20 % over two | **ACCEPT.** `10_` §5 "no collapse in transport" is withdrawn for any outgoing table. |
| `21_` A3 | Dirty set is not thin (Lambert lobe, GGX skirts) | **ACCEPT.** `10_` §8 "thin set" is withdrawn. §2 above uses the same fact against C. |
| `21_` A5 | Caustic-type chains through the mover: rate α²–4α² | **ACCEPT.** Track A has no route. No track has one (`20_` §5.4); the path class needs a declared exclusion or its own reference qualification in any fixture. |
| `22_` A3 | T0 build of 10³–10⁴ image-equivalents not charged | **ACCEPT.** Moot after withdrawal of the table. |
| `22_` A4 | View independence is not a GM1 requirement | **ACCEPT the conclusion; narrow REBUT of the premise.** The GM1 row requires "edit-to-valid-output and per-frame accounting" (FACT, `docs/RESEARCH_ROADMAP.md`), so a cost-versus-views curve is admissible evidence. It cannot be the headline, and with the build charged the break-even was thousands of views. The claim "cheaper whenever the state must be view-independent" (`10_` §1) is withdrawn. |
| `22_` A5 | Higher-order discovery cost uncharged | **ACCEPT.** In the surviving form it is the measured touched-link count per sweep. |
| `23_` A-4…A-7 | Argued advantage unmeasured; A4 vs A5 not noise-isolated; discovery unscored against an oracle set; φ_τ estimator-dependent | **ACCEPT all.** Common random numbers between T0 and T1 evaluations give a noise-free oracle dirty set; §6 is built on that. |
| `23_` X1–X6 | Matching, anchoring, signed-change noise, replicates, T0 asymmetry, implementation maturity | **ACCEPT**; adopted where §6 needs them. |
| `30_` | No mechanism is new; Bala, Dorsey and Teller 1999 and Granier and Drettakis 2001 were missing from `10_` §13 | **ACCEPT.** The one remaining novelty candidate of `10_` §13 is withdrawn. DDT-D is discover-then-recompute on a directional state (Bala et al. 1999, as verified in `30_`) and incremental glossy update (Granier and Drettakis 2001, bibliographic level in `30_`). |

MINOR items (`20_` A-5…A-7, `21_` A6–A7, `22_` A6–A7, `23_` A-8…A-10) are accepted without comment.

No objection is rebutted on substance. The two scoped rebuttals limit how far an accepted finding transfers; they do not rescue the design.

## 4. Revised claim set

**Withdrawn.**

- A persistent 4D outgoing-radiance table as the GM1 state.
- "Work scales with the radiometric measure of `D`"; "cheaper than a re-render for view-independent output"; "discovery independent of scene size"; "thin dirty set".
- Band-limited state with path continuation as a mechanism of this track.
- Arms A2/A3 and falsifier F1 as tests of DDT-D.
- Any mechanism novelty.
- The convergence wording "contraction whenever absorption exists" (replaced by `‖K C1‖ ≤ 1` with a scene-dependent rate; `20_` §1).

**Surviving (each was checked as sound by a reviewer).**

- The identity, as defect correction: the T1 residual of the T0 state is supported on changed links only (`20_` §2.1; DERIV).
- Overwrite, never add, for sharp features (`20_` §2.1.5, `21_` §2.2).
- Unchanged rays carry an exactly zero source, so no difference-of-estimators noise arises there (`22_` §2.4).
- State errors enter the source only, never the operator (`20_` §2.1.4).
- Never-consumed directions as an admission test for any stored directional state (`21_` §2.2).

**Narrowed.** DDT-D becomes one hypothesis about a solve policy:

> **HYP-Δ.** On a band-limited incident-field state with stored links, the set of cells whose change exceeds a tolerance anchored to the gate, in the per-cell peak norm and with the `1/(1 − ρ)` budget, is a small fraction of the state on GM1-type edits; a residual-ordered overwrite restricted to that set equals the dense re-solve within that budget.

HYP-Δ is a classical incremental-radiosity policy (Chen 1990; Granier and Drettakis 2001, via `30_`). It contains nothing learned and is not a contribution by itself (`docs/RESEARCH_CENTRIC_TOPIC.md` rule quoted in `30_` §0).

## 5. Hybrid

**With Track C: technically justified in one narrow form, with weak prior prospects.**

- **Single interface:** the per-link residual `r(c, j) = [T1·I0 + s1 − I0](c, j)` on C's link graph, where `T1` and `s1` are C's operator and direct term after link regeneration.
- **C contributes:** the state (incident field at α\*), stored links with common random numbers, the lobe partition, and segment certificates. These give the exact first-order support of `r` at the estimator's own sample resolution, which repairs `20_` A-2 and A-4.
- **A contributes:** the residual-ordered overwrite schedule, the unsigned peak-bound marking rule and its stopping rule, replacing C's dense warm-started re-solve. This is the change `22_` §4.8 asks of C.
- **Limit (DERIV, §2):** unsigned support saturates in two sweeps, so any gain is bought by τ. The worst-case bound predicts no certified sparsity at K = 128, ρ ≈ 0.8. A physical estimate is less pessimistic: first-order rough-lobe change exceeds 1 % only where the mover subtends more than about 0.06 sr (`a_L` = 0.5), i.e. within about 0.8 m of a 0.2 m box face (EST, same model as `22_` §2.2). The two estimates differ by more than the decision margin; only a measurement separates them.
- The hybrid does not touch C's other open defects, its readout cost, or the workload-scale problem (`22_` §5).

**With Track B: no hybrid.** B supplies a measurement Track A needs (the same-seed dirty log, `23_` §6), and receives Track A's on-demand arm as a comparator. Neither is a shared mechanism.

## 6. Revised decisive falsification (design only; not authorised)

**Question.** Is HYP-Δ's thresholded dirty set small and safe, before any state is built?

**Stage D0 — zero render.** From fixture geometry and a lobe census of the frozen closures (shared with C's Stage 0): the invalid-link fraction `f`; per-point solid angle of `O0 ∪ O1`; the rough-lobe albedo share; a bound on ρ. Compute the predicted marked fraction per sweep under the peak-norm rule at τ = anchored gate × `(1 − ρ)`.
*Stop (F-Δ0):* the predicted marked work summed over sweeps is at least a predeclared fraction of the dense warm-started solve on E-small and E-large. HYP-Δ then has no certified economic content.

**Stage D1 — oracle field differences (only if D0 does not stop).** No LPTG implementation. For a predeclared sample of cells `(c, j)` at the declared K, evaluate incident radiance by REF-ER path tracing at T0 and T1 with **identical seeds**, Russian-roulette depth, at least 8 seed sets. Unchanged cells reproduce bit-identically, so the changed set is a noise-free oracle. Edits: E-small, E-large, E-adv, a null edit and a move-and-return edit.
Measurements: fraction of cells with peak change above τ, by first-dirty depth; an offline replay of the marking rule on these differences, scored for radiometrically weighted false negatives and false-positive work; the realised error of the thresholded fixed point against the unthresholded one.

*Falsifiers (thresholds frozen before D1 is read).*

- **F-Δ1:** the oracle marked fraction on E-small or E-large exceeds the predeclared ceiling. No sparsity exists to exploit.
- **F-Δ2:** the realised thresholding error exceeds the anchored gate, or the marking rule misses more than the predeclared weighted share. The policy is unsafe.
- **F-Δ3:** E-adv is not flagged as dense by D0. No early detection.
- *Invalid experiment:* the null edit marks any cell.

**Scope of a pass.** It shows only that C's solve could be made output-sensitive on one fixture. It makes no statement about path-traced re-rendering cost, about C's representation, or about any neural baseline (UNMEASURED, per the baseline-role contract).

## 7. Frank statement

**Would I test Track A first? No.**

- As a standalone mechanism it has two accepted FATAL findings on independent axes and no representation.
- As HYP-Δ it is a correction to one accounting gap of another track. It has value only if that track's own zero-render admission (lobe census, α\*, rough share) passes, and my own bound in §2 predicts it is more likely to fail than pass at K = 128.
- The measurement that all three reviews treat as prior to everything else (lobe census; same-seed dirty log on one common fixture) is not Track A's.

**Single missing evidence that would change the answer:** the D0/D1 result, i.e. a measured oracle marked fraction at the anchored peak-norm tolerance that is well below the dense-solve break-even on a workload where cold REF-ER costs seconds. Even then Track A would enter as the solve policy inside a C-type state, not as a track tested first or alone.

## 8. Limits of this response

- Nothing here is measured. The fan-out and magnitude figures in §2 use a uniform-landing assumption, an unchecked Lambert share and the reviewers' unmeasured unit constants.
- The `τ/(1 − ρ)` argument is a worst-case sup-norm bound, not a prediction.
- Worklogs and literature were not re-opened in this session; they are relied on through the reviews at the verification levels those reviews record. `31_` and `32_` were read only in part.
