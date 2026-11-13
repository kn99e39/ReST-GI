# LSIT adversarial refutation: review of both verifier reports (read-only, commit 4ed4ac3)

**Sources**
- **Definition.** LSIT is defined only by `c:\Projects\ReST-GI\docs\plans\DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md` (346 lines). I read it in full.
- **Re-opened for this check:**
  - `docs/Architecture.md` §1–2;
  - WL86 (lines 45–56, 71, 158, 180);
  - WL86A (lines 145–153);
  - WL87 (lines 61–64, 225–253);
  - WL88 (lines 55–65, 279, 288);
  - WL89 (lines 60–70);
  - WL90 (lines 23, 100–101, 237–238);
  - WL92 (family table 123–126, local-operator table 138, ladder 142–158, free fits 160–180, 192–197);
  - WL92B (counterfactual definitions 18–30, render table 86–120);
  - WL93A (Task A table 133–143);
  - WL93C (lines 96, 120–140);
  - `experiments/fnto_compiled83/closure83_dr.py:88–156`;
  - `results/gm1_char85/results/census/fixture.json` (entry S1).
- **Rules followed.** I edited no files and ran no jobs. Labels: OBSERVED / SUPPORTED INFERENCE / PLAUSIBLE HYPOTHESIS / UNSUPPORTED / CONTRADICTED.
- **D1** (explicit events 1–2 vs the 2026-11-13 owner decision, Architecture.md §1) is treated throughout as an unresolved contract conflict, not as a defect.

---

## 1. Verdict on each claimed critical contradiction

### Verifier 1 (physics)

| # | Claim | Verdict | Justification |
|---|---|---|---|
| V1-1 | Ownership of lobe events at order ≥ 3 conflicts with R_θ's training target t − t2. | **STANDS as an internal specification inconsistency. DOWNGRADE in severity: not structural.** | §4.2 C4 (line 130) and §8 say R_θ "never represents a sharp local lobe". C4 (line 131) and Stage 1 (line 243) still train it on t − t2, which contains those events. The sentence "OWN-style training is intended to prevent compensation" (§4.7) is UNSUPPORTED. However, LSIT discloses the truncation (A2) and makes it a rejection criterion (R2 / P-T2), so this is a false ownership statement, not a hidden design flaw. The closest existing analogue is WL92B **X_L2**: path-traced L₂ plus the OWN learned global term, which was trained on t − t2 and so absorbs every lobe event at order ≥ 3. It gives E2 T1 region C **0.022 / 0.019** and E2 T0 **0.016 / 0.015**, against a floor of 0.013 / 0.010 (OBSERVED, WL92B render table). On the inspected E2 conditions, a learned order ≥ 3 term that absorbs this content is therefore empirically tolerable. Its input is s, not L₂. |
| V1-2 | The readout patch's own lobe at order ≥ 3 has no owner. | **REJECT as "unowned" or "new". DOWNGRADE to an omission.** | When x2 is sharp, its last scattering in L₃₊ is a lobe event at order ≥ 3. It is therefore inside A2, and P-T2's criterion ("a lobe event at order ≥ 3") covers it if x2's vertex is counted. The omission is real: §4.7 lists only chained events. Region C contains this sub-case: WL86 line 51 defines region C to include "panel / sphere receivers whose mirror cone sees a change", for example a sphere whose mirror cone sees the panel. The 0.46 / 0.47 figure is the share of **all** 3–5-event paths (WL92 lines 125–126), not the sharp-x2 share. Using it as the magnitude of this effect is an over-reading, and V1 labelled it only as plausible. A pooled P-T2 share can pass while a small set of sharp-x2 pixels fails completely. |
| V1-3 | P-T2's "tag lobe-sampled vertices" cannot measure the truncation. | **STANDS. DOWNGRADE to a specification defect.** | The ER sampler picks a lobe, but the vertex weight is full f/pdf_mix, and next-event estimation evaluates full f (`closure83_dr.py:121–155` plus the sampling code cited by V1). In addition, f_L is a fitted half-vector function, not ER's lobe index. A technique tag therefore does not split the f_L and f_S contributions. The fix is deterministic f_L/f and f_S/f weights at each vertex. P-T2 also depends on P-T1's f_L, and the protocol does not state that ordering. |
| V1-4 | The Stage-1 "LSIT-oracle" and P-T3 are not oracles of the LSIT contract. | **STANDS** (inferential confound). | Both use path-traced L₃₊, which includes the truncated content. "Oracle passes, learned fails ⇒ learning is the limit" therefore mixes truncation with learnability. See §2 for a further mismatch in P-T3's comparison bar that neither verifier raised. |
| V1-5 | "Matches ER's own microfacet structure" is contradicted. | **STANDS as a factually false sentence. DOWNGRADE as a design contradiction**, because LSIT hedges it as assumption A1 and lists multi-lobe ER as a failure case (§4.7). | OBSERVED in code: inner lobes are evaluated at refracted directions with a 1/η² factor; each lobe has its own Schlick f0; masking is height-correlated Smith G2; the inner sum is multiplied by s(μi)·s(μo) (`closure83_dr.py:91–155`). OBSERVED in the census (`fixture.json`, S1): per-lobe r50 at 45° spans 0.049–0.211 rad, and the outer lobe k0's albedo rises from 0.0014 at 0° to 0.066 at 80°. A single product F(θd)·D(θh) cannot be exact. **Correction to V1:** the "5× width change" (0.078 → 0.41 rad) holds for an isolated inner lobe. The composite S1 closure r50 actually **narrows** from 0.146 at 0° to 0.090 at 80° (census), because weight shifts to the outer lobes. Non-separability stands; the width-growth framing does not describe the composite lobe. |
| V1-6 | §4.4 contradicts itself on binning. | **REJECT as a contradiction.** | The same §4.4 bullet list says the second event is evaluated "around incident-bin centres". The phrase "binned only when deposited" refers to the output. The substantive risk is real: receiver-side incident binning at sharp receivers is unmeasured (V2-A4). That is an omission, not an internal contradiction. |
| V1 storage | 37.6 MB contradicts "10–20 MB". | **DOWNGRADE to underspecified.** | LSIT stores (hit, bin, weight) and no segment. Segments can be regenerated from the ray id if the stream is deterministic; WL89 line 70 shows C3UR keyed by global ray id. 7–11 B per record fits 8.5–13.3 MB (arithmetic checked: 1,212,416 records). |
| V1 R_θ energy | R_θ has no passivity bound. | **STANDS as an omission** (minor). | R_θ is applied once, so it cannot diverge, but gain above 1 is not excluded. ECRO v0 has the same gap. |

### Verifier 2 (representation)

| # | Claim | Verdict | Justification |
|---|---|---|---|
| V2-1 | The target contradicts the class. | Same item as V1-1. **STANDS, downgraded.** | See V1-1, including the X_L2 evidence. |
| V2-2 | "Identity by construction at the camera bounce" vs unsplit-ER LOW. | **DOWNGRADE** to a precision issue. | Identity holds to the f = f_L + f_S tolerance, which P-T1 already gates. |
| V2-3 | Event-order-dependent local response vs Architecture.md §2 ("Direct and C-mediated incidence use the same intrinsic response rule…"). | **DOWNGRADE to a contract-clarification item.** It is not a defect, but it is a second conflict alongside D1. | At order 2, C-mediated incidence is scattered by K_L + K_S. At order ≥ 3 it is scattered by R_θ's learned, geometry-conditioned a·c. That is an order-dependent response branch for C-mediated incidence (OBSERVED in the text, §4.2 C3/C4; Architecture.md §2). ECRO v0 used a single learned a·c at every order, so LSIT introduces this split. Whether a learned higher-order composite approximant counts as a "response rule" is an owner decision. |
| V2-4 | The §4.6 cost basis is misattributed. | **Partly REJECT, partly STANDS.** | LSIT cites WL90. WL90 line 101 says "Exact once-scattered radiance per source: about 10 ms for about 0.6 M rays". Applied to 19 k rays this is a conservative bound for that one step, not a misattribution (V2 compared it with WL91's 526 k + 526 k instead). STANDS: §4.6 omits the per-source second-event scatter over 1.2 M records, the E1^S gather, flight(L₂), R_θ application, and the full per-edit recomputation of E1 / L₂. STANDS as risk: quality at about 19 k lobe rays per light is unmeasured. Every WL90 / WL91 quality figure was obtained with about 30× more rays. |
| V2-5 | "Low rank after reparameterisation" is mathematically false. | **REJECT** (mostly). | Linear-operator rank is invariant under invertible basis change. **Separation rank of the kernel function is not.** A shift kernel g(ωo − R ωi) is rank-1 in the (ωi, offset) variables and full rank in absolute (ωi, ωo). §3 ("low-dimensional in a mirror-aligned parameterisation") and Alt-A are therefore meaningful. Only the wording of P1 ("become low-rank once reparameterised") is loose. DOWNGRADE to imprecise wording. |
| V2-6 | P1 is not load-bearing at runtime. | **STANDS as a justification gap, not a contradiction.** | M^S enters only dense explicit products (73,728 floats). Its rank matters only to whether R_θ's rank-16 local factors can stand in for K_S at order ≥ 3 (C4). More important: **the split itself is unnecessary for events 1–2.** Exact per-ray ER evaluation already exists (WL87 W_ray, WL90 X1), and ER's own lobe sampler could drive the lobe relation. The P-T1 rejection rule therefore targets the least necessary component. |
| V2-7 | The second-event specification is ambiguous (§4.2 C2(b) vs C3). | **STANDS. This is a major specification defect.** | The bin-indexed lobe relation stores *hits at other patches* (C2(b)). L₂[j, ·] is j's *outgoing* table (C3), which needs only the outgoing direction at j, not traced hits. Either the traced rays are unused, or they deliver sharp L₂ sender-side to receivers, which then means R_θ's input is the incident field E2. C4 instead names "flight(L₂)" via ECRO state rays: a receiver-side gather of sharp L₂, which brings back the 86A-8 sampling limit (median 13–17 / 512 rays hit the panel) at order 3. Information, cost and selective-update scope all depend on which reading is meant. |
| V2-8 | E6 "sender-side placement removes it at the same budget" is labelled [M] but overstated. | **STANDS** (minor; not load-bearing). | WL88 line 279 gives C3UR − R0D +0.050 / +0.053 at equal rays. WL88 is STOPPED_WORKLOAD_UNQUALIFIED and this is a single draw. "Removes" is also wrong: C3 reaches 0.024 against an oracle of 0.0137 (WL87), and C3UR reaches 0.040 against an oracle of 0.014 (WL90). |
| V2-9 | Exact per-light-sample emission at P = 16 does not remove g00010. | **DOWNGRADE to an unflagged risk.** | LSIT never claims to fix g00010. V2's 2.67 figure is the realisation-to-realisation disagreement N_b of a random estimator (relative L2 on panel q bins, heavy-tailed; WL93A lines 133–143). The deployed fixed P = 16 points give a deterministic bias of similar order, but its effect on the rendered sharp region is unknown. On sharp E2 the B0 value is 0.029. |

V2 also raises A4: incident binning at sharp receivers. That point is valid as a risk (an omission, not a contradiction):
- WL90 line 23 confirms that the WL90 receiver gather uses the **exact local direction**.
- WL89 lines 60–70 confirm that C3UR deposits into the dense incident cells.
- So WL90's "order ≥ 2 projection negligible" does not cover LSIT's bin-centre second event.
- The 29° polar-cap offset is arithmetically correct: the cap centroid sits at cos θ = 0.875, i.e. 29°.

---

## 2. Checks of cited numbers (misquotes and over-reading)

**Correct as quoted (checked against the source):**
- WL86: oracle C 0.0074 / 0.0137, noise 0.007 / 0.013 (line 158); learned C 0.0717 / 0.0634 (line 180); selective 23 ms vs full 29 ms (line 71).
- WL86A: global term "+0.02 … +0.06" (line 153). LSIT quotes WL86A correctly. WL92's decomposition gives +0.038 to +0.096 (lines 125–126), so the term varies by analysis.
- WL92: family shares 0.18 / 0.36 / 0.46 and 0.12 / 0.41 / 0.47 (lines 125–126); S1 rank-16 residual 0.52 / 0.46 / 0.46 and rank 46–59 (line 138).
- WL90: 0.040 → 0.024 at 0.56 / 0.82× cost (lines 237–238).
- Arithmetic checked: 74·16·64·16 = 1,212,416 records; 74·16·16 = 18,944; panel 688,128; sphere 524,288.

**Misquoted or over-read:**

1. **LSIT E12 and the P-T3 bar pick only E2 T1.** WL93C lines 131–136 give ORA region C as follows (OBSERVED):

   | Condition | ORA | T0 |
   |---|---|---|
   | E2 T1 | 0.030 / 0.033 | 0.076 / 0.042 |
   | **E2 T0** | **0.059 / 0.058** | 0.067 / 0.049 |
   | E5 T0 | 0.015 / 0.014 | 0.034 / 0.024 |
   | **E5 T1** | **0.031 / 0.031** | 0.032 / 0.025 |

   ORA is worse than T0 seed 8602 on E2 T0 and on E5 T1, and on E2 T0 it sits just under the 0.06 rule. "Far better on rendered sharp readers" holds for 2 of 4 inspected conditions. The ledger row 93C-3 repeats the same omission.

2. **The P-T3 comparison bar is mismatched. Neither verifier caught this.**
   - P-T3 scores LSIT-L2 plus **path-traced L₃₊** against ORA, which is the binned M64 composition plus the **learned OWN global term**.
   - The matched learning-free comparator already exists: WL92 ladder **ENCB** (exact M64 at both uses, deployed draw) plus true L₃₊ gives **0.045 / 0.052** (E2 T1 / T0), with a floor ORA2 of 0.013 / 0.010 (WL92 lines 146–151).
   - On E2 T1, ORA with the learned global (0.030) beats the same L₂ with the true L₃₊ (0.045). SUPPORTED INFERENCE: the learned global term partially cancels the L₂ deficit at rendered level on E2 T1, but not on E2 T0 (0.059 vs 0.052). Caveat: the two figures come from separate WL92 / WL92B runs, and common random numbers apply only within a batch.
   - WL93C's "no level-compensation conflict" is a table-level, validation-population statement. It does not establish the rendered region-C behaviour.

3. **LSIT §2 "E5 shows the opposite" (representability of a shared physical operator) over-reads WL92's per-geometry fits.**
   - OWN-f8601 reaches 0.029 on E2 T1 (WL92 line 166). That **beats the exact physical composition ENCB (0.045)** on the same deployed interface.
   - The fits therefore calibrate around the state-ray sampling shortfall for one geometry. They are not the physical operator.
   - LSIT's lower end, 0.017, comes from the **unconverged** OWN-k16 fit (11.6 %) on E2 T0.
   - The material-level rank-16 point R16 fails: 0.084 / 0.073.

4. **Over-reading in V2 A5 (vi).** It reads WL93A's N_b (estimator disagreement between realisations) as LSIT's receiver-level error at P = 16. See V2-9.

5. **Over-reading in V1 contradiction 2.** It attaches the share of all 3–5-event paths (0.46–0.47) to the sharp-x2 case.

6. **Smaller points.** WL92's RS ("exact radiance per deployed ray") is 0.048 / 0.044 and EXP is 0.041 / 0.048 (line 147). So on E2 the deficit is **receiver-side state-ray sampling**, not sender-side angular projection. LSIT's E7 premise comes from W1 / W2, where sender-side relations were already used. On E2, LSIT's expected gain must come from the **sender-side placement** of the event-1 lobe, not from per-ray exactness alone. LSIT does not draw this distinction.

---

## 3. Inconsistent use of evidence

- **Inspected conditions treated as held-out.** LSIT's support (E12, E3, E2) rests on E2 / E5, both inspected (ledger caveat 19). LSIT does disclose this and plans a new admitted edit, which is correct. E4 failed admission at Q4 SNR 3.9, so admission risk is real.
- **Oracle success read as learnability.** LSIT's Stage-1 reading "oracle passes, learned fails ⇒ learning is the limit" is a confound (V1-4). The reverse also applies: X_L2's 0.016–0.022 is an upper-bound-type oracle with an s-input global term. It does not show that R_θ(L₂) is learnable. LSPOT shows the learned higher-order term is the weak point there (X_L3 0.013 / 0.011 vs X_L2 0.020 / 0.029, WL92B lines 113–117).
- **Smoke negatives read as unrepresentability.** LSIT avoids this explicitly (§2, "does not show"). V2's matrix also respects it.
- **Pooled metrics used alone.** LSIT §7 forbids it. Yet P-T2's rule is a pooled share over region C, which hides failures concentrated on sharp-x2 pixels.
- **Cherry-picked conditions.** E12 and the P-T3 bar (point 2.1).

---

## 4. What each report missed that matters for the verdict

**Both reports missed:**
- (a) **WL92B X_L2** (true L₂ plus a learned order ≥ 3 term trained on t − t2) at 0.016–0.022 on E2 region C. This is the most direct existing evidence for LSIT's allocation premise. It is inspected-only, uses s input, and is weaker on LSPOT.
- (b) **The P-T3 comparator is mismatched.** The comparator should be ENCB / RS / EXP plus true L₃₊ (0.041–0.053) with floor ORA2, not ORA with a learned global term.
- (c) ORA's E2 T0 / E5 T1 results are not better than T0.
- (d) On E2 the residual is receiver-side sampling (RS ≈ ENCB). LSIT's value there depends on the **sender-side** event-1 lobe relation.
- (e) E5's per-geometry fits beat the exact physical operator, so they are interface calibration, not evidence of physical representability.

**Verifier 1 also:**
- over-read the inner-lobe width growth; the composite S1 r50 narrows toward grazing (census);
- missed that the split is unnecessary for events 1–2 (V2 caught this);
- did not raise the second-event semantic ambiguity (V2-7), which is the largest specification defect.

**Verifier 2 also:**
- wrongly called reparameterised low rank "false" (separation rank does change with the choice of variables);
- attributed LSIT's 10 ms figure to WL91 instead of WL90 (where it covers about 0.6 M rays);
- did not raise the P-T2 estimator defect (V1-3) or the readout-patch sub-case of A2.

---

## 5. The most decisive Stage-0 (learning-free) check

**Contract-compliant composition ladder on region C (rendered E_H, common random numbers).**

**Conditions.** E2 T0 / T1 and E5 T0 / T1 (inspected), plus one newly admitted held-out edit (Q4 SNR ≥ 4, sharp family present, thresholds frozen before any LSIT output). Report every condition separately and split region C by the material at x2 (sharp vs non-sharp) and by receiver type (panel / sphere receivers vs mirror footprint).

**Arms. Arm (a) uses path-traced L₂; arms (b)–(e) each pair an L₂ estimate with a path-traced L₃₊.**

| Arm | Definition | Role |
|---|---|---|
| (a) ORA2 | path-traced L₂ + path-traced L₃₊ | floor |
| (b) Matched binned | ENCB-type M64 composition L₂ (deployed draw) + path-traced L₃₊ | the comparator LSIT must beat |
| (c) LSIT-L2 | as specified: P = 16, K_r = 16, second-event semantics fixed first (V2-7), sphere frame fixed; ≥ 16 independent realisations of point and lobe offsets | LSIT's explicit part |
| (d) Contract-compliant LSIT oracle | (c) + path-traced L₃₊ with every f_L event at table order ≥ 3 removed by deterministic f_S/f vertex weights, **including x2's own vertex** | what LSIT's contract can deliver with a perfect R_θ |
| (e) No-split control | (c) with exact per-ray ER (ER's own sampler, f/pdf), no M^S for S1 / S2 | tests whether the split is needed |

**Falsifies LSIT (learning-independent) if:**
- (d) > 0.06 in any condition including held-out, or (d)'s realisation pass rate is below a predeclared rate; or
- (d) is not better than (b) by more than max(noise bound, 0.005) on both E2 cases and on held-out; or
- the truncation cost (d) − (c) carries more than a predeclared share of (d)'s error on the sharp-x2 subset (A2 false where it matters); or
- the C-region edit slope of (d) lies outside [0.75, 1.25].

(e) ≤ (c) does not falsify the event allocation, but it does falsify the necessity of C1.

**Inconclusive if:**
- P-T1 cannot produce f_S ≥ 0 within tolerance, so (d) is undefined. This would be an R1 rejection of C1, with the allocation question left open via (e);
- the C3 second-event semantics are not frozen, or both readings are run and pooled;
- the held-out edit is not admitted, or its region C lacks panel / sphere mirror-cone receivers (the E3 case);
- arm differences lie within the reference noise bound (0.013 / 0.010) or within realisation spread;
- the sharp-x2 subset has too few pixels for the predeclared confidence interval;
- the truncation estimate changes materially between two predeclared split fits.

Scope stays limited to one room family, point lights, the two-light interface and rigid edits.

---

## 6. Recommended verdict label

**NEEDS_CONTRACT_CLARIFICATION.** Even after a "yes" on D1, LSIT is **not READY_FOR_FALSIFICATION as written**. It would then be NEEDS_SPECIFICATION.

**Why contract clarification is the gating label:**
- D1 (explicit events 1–2 vs the 2026-11-13 decision) is unresolved. If the answer is no, the design is inadmissible whatever the measurements show.
- A second contract question is added: the order-dependent response rule (K at order ≤ 2, learned a·c at order ≥ 3) against Architecture.md §2's single intrinsic response rule.

**Why not READY_FOR_FALSIFICATION:**
- the second-event semantics are ambiguous (V2-7);
- the P-T2 estimator is ill-defined (V1-3);
- the oracles are not contract-compliant (V1-4);
- the P-T3 comparator is mismatched and its bar picks one condition (§2 points 1–2);
- R_θ's ownership claim conflicts with its target (V1-1);
- the sphere-frame and horizon handling of the bin-indexed relation is unspecified;
- no realisation-count or pass-rate rule is given for P = 16 / K_r = 16.

**Why not STRUCTURALLY_UNJUSTIFIED:**
- The allocation (sender-side exact event 1, explicit L₂, learned order ≥ 3) has direct supporting evidence on inspected conditions:
  - X_L2 0.016–0.022 (WL92B);
  - the receiver-side sampling deficit (86A-8; RS ≈ ENCB in WL92);
  - the first-gather isolation in W1 / W2 (WL90).
- The weakly justified part is C1, the half-vector split. Its exact form is contradicted by the closure (V1-5), and it is not needed for events 1–2 (V2-6). That weakens one component, not the whole structure.
