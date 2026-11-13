# IDTS (Candidate B): adversarial refutation of the two verifier reports

**Scope.** Read-only at commit 4ed4ac3. No files were edited and nothing was run beyond `python -c` reads of stored JSON. I re-opened these sources:
- `experiments/ecro86/model86.py:83-124`, `pt86.py:153-156`, `scene86.py:13-22,90-101`
- `experiments/ecro_attr92/diag92.py:114-130` (metric definitions)
- `results/ecro_attr92/results/local.json` (materials and learned blocks)
- `results/ecro_stage93c/oracle_gate.json`
- `results/gm1_char85/results/census/fixture.json` (S1 per-lobe albedo and r50)
- `docs/worklogs/93C_*.md` §Measurements (l.103-135)
- `docs/worklogs/92_*.md` ladder table (l.143-158)
- `docs/worklogs/86A_*.md` l.72-79, 133-136, 143-147
- `docs/worklogs/86_*.md` l.1-56
- `docs/worklogs/90_*.md` l.48-59, 201
- `docs/worklogs/93A_*.md` l.143-165, 205-214, 282
- `docs/worklogs/92B_*.md` l.39, 126
- `docs/Architecture.md` l.13, 36-44
- `docs/plans/DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md` l.170-176, 219-223, 310-315

Labels: OBSERVED / SUPPORTED INFERENCE (SUP) / PLAUSIBLE HYPOTHESIS (PLAU) / UNSUPPORTED / CONTRADICTED. [ARITH] marks my own arithmetic, never a measurement.

---

## 1. Claimed critical contradictions: refutation attempts

### Verifier 2 (representation)

**C1. B1 vs ECRO's rank-16 `a cᵀ` class: DOWNGRADE to an exclusion constraint on instances.**
- The numbers are correct (OBSERVED `local.json` → `materials/S1_panel/{0,1,2}`):
  - Eckart–Young rank-16 residual 0.520 / 0.458 / 0.461.
  - Rank for 99 % energy 59 / 46 / 47.
  - NMF-16 peak 0.472 / 0.535 / 0.528.
  - Unweighted residual 0.190 / 0.368 / 0.329.
- It is not a contradiction of IDTS. B1 says "the precise operator class is NOT YET FIXED", and nothing in the text adopts rank 16.
- What survives is the obligation that any material-shared K_phi for S1 at 64 bins must not be a rank ≈ 16 factorisation, under the flux norm.
- It is also measured against M64 (the uniform-in-bin projection of ER), not against ER itself.

**C2. B1 + B5 vs a 64-bin z^t: DOWNGRADE. The B5 part is REJECTED as a misreading.**
- B5 requires the same law on *physically equivalent* incident distributions. Two physically different fields that share a binned state are by definition not equivalent. A K that returns the same output for both does not violate B5; the problem is that z^t lost information, which is B2.
- B2 already disowns low-dimensional summaries. So "64 bins are insufficient" is a resolution obligation for B2, not a contradiction.
- Evidence quality varies:
  - The angular beam pairs are valid but post hoc and synthetic: panel azimuthal 0.94 vs noise 0.26, polar 0.62 vs 0.22 (OBSERVED 86A l.136). The "two halves" pair is spatial, not angular.
  - g00010 (M(enc) 3.77 vs noise 0.66) is **over-interpreted as angular-bin insufficiency**. WL93A does not attribute its cause. Spatial within-patch variation (visibility, 1/d², cone) dominates the q noise ("Spatial noise dominates", 93A l.143), and WL93A lists the physical nature of g00010 as open (l.282). It is a per-patch-state insufficiency of unknown angular or spatial split.
  - WL90 is the clean angular evidence, but it comes from the explicit pipeline on the WL89 W1/W2 workloads, not ECRO E2.

**C3. Curved patches vs a patch-level shared K: DOWNGRADE to an obligation.**
- B1 defines K_phi "in a local material frame", which points to the point level. IDTS never adopts the patch-centre frame.
- The analytic point is correct and important. A patch-level operator is ER convolved with the within-patch normal distribution.
  - Sphere patches use one centre frame (OBSERVED `scene86.py:96-98`).
  - [ARITH] The cap band spans θ 0–60° around a 41.4° centre, and the equatorial band 60–90°, so the patch operator is geometry-dependent though rigid-invariant.
- This becomes an obligation: either use point- or sub-patch frames, or let C^t / a geometry term own the normal-distribution convolution.

**C4. Contract conflict with the 2026-11-13 owner decision: DOWNGRADE to a conditional dependency (it drives NEEDS_CONTRACT_CLARIFICATION only once K_phi and the explicit order are chosen).**
- The conflict is not forced by IDTS.
  - ECRO itself already has an explicit current-geometry flight (`E1` gather over state rays, `model86.py:115-119`) and path-traced LOW e1+e2. "Explicit flight" alone is inside the lineage.
  - WL93C's ORA (= M64·flight·M64·s + learned G) was run inside the ECRO lineage.
- The conflict becomes binding only if IDTS chooses non-learned ER / M_D for K_phi with all orders below G_psi computed exactly, which is the LSIT D1 question (LSIT l.173, l.313).
- A second contract hook neither verifier quoted is `Architecture.md:44`: "Future modifications to ER require a new, separately justified hypothesis." This binds if K_phi is a new learned law that departs from ER, rather than an approximation of ER's projection.

**Tension B4 + B6 for sharp chains: STANDS as a risk, with a corrected evidence base (see §3).**

### Verifier 1 (physics)

Verifier 1 correctly states that it found no strict internal contradiction. My verdicts on its items:

| Item | Verdict | Reason |
|---|---|---|
| X1 (B3 vs B4) | DOWNGRADE (already a latent conflict) | B4 explicitly defers the relationship. |
| X2 (sharp higher orders vs a smooth learned tail) | STANDS as an omission, but the example is mis-assigned (below) | See the event-numbering correction. |
| X3 (no common state space across roles) | STANDS as an omission. The role-dependence mechanism is DOWNGRADED to the B1 resolution obligation | Different within-bin statistics are physically different fields, so B5 is not violated. |
| X4a ("not nominally intrinsic") | DOWNGRADE to wording imprecision | WL86 explicitly tested a "geometry-conditioned learned linear operator" (86 l.7). Later batches treat `a cᵀ` as the panel's material response ("panel intrinsically unchanged", 92B l.39/126; "shared intrinsic material operator", 93B l.248). "Nominally intrinsic" fits the later lineage. |
| X4b (ECRO states are 64-bin directional, not an irradiance summary) | STANDS (minor) | The bottleneck is u = c·s at rank 16 (`model86.py:112`). |
| X5 | Same as C4 | |
| X6 | Same as C3 | |

**Over-claim: B-1, "the core hypothesis holds by construction for ECRO". REJECT.**
- Only the presence of geometry conditioning holds by construction (`model86.py:95-107`, OBSERVED).
- "Principal failure" needs a ranking of causes. That ranking does not follow from construction (see §4.1).

**Over-claim: the narrow-beam counterexample, "reflections differ by ≈ 2Δ, so K_phi(z) is wrong". PARTIALLY REJECT.**
- For an *ideal* mirror, the deployed bins map exactly bin-to-bin: φ → φ+π is a shift of exactly 8 sectors and θ is unchanged (`pt86.py:153-156`, OBSERVED). The bin-level output is therefore identical and exact for both beams.
- The loss appears (a) for lobes of about bin width, where spill into neighbouring bins depends on in-bin position (Verifier 2's framing is the correct one), and (b) downstream, where the flight resolves the within-bin direction (the WL90 first-gather locus).

---

## 2. Cited measurements checked

| Claim | Status |
|---|---|
| S1 rank / residual / NMF numbers (both reports) | Correct (`local.json`) |
| Learned `a cᵀ` (V1): walls mat 3 Frobenius 0.131 / 0.134, mass 1.031 / 1.034; box B mat 5 0.257 / 0.315; panel 0.877 / 0.885 | Correct (`local.json` → `learned.ecro_s860x`). Mat 3 covers 832 patches (walls and ceiling). |
| WL93C Task A table: ORA single 0.1033, whole 0.0776 / 0.0774; ORA_SRC single 0.1101 / 0.1104; ORA_RCV 0.0997 / 0.1033, whole 0.0756 / 0.0772; ORA_PHYS 0.0998 | Correct (93C l.107-112) |
| ORA − T0 single / whole CIs include 0 | Correct (`oracle_gate.json`) |
| Not cited by either report (`oracle_gate.json` contrasts): ORA − T0 RECV −0.0070 [−0.0076, −0.0064], ORA − T0 SRC −0.404 [−0.60, −0.17]; vs TQ RECV −0.0098 / −0.0106 | OBSERVED, stated here because it matters for §4.1 |
| S1 closure r50 0.146 → 0.090; sharpest lobe r50 0.049 at 45° | Correct as numbers, **but over-interpreted** (next row) |
| V1 resolution "Δ = sharpest r50 → ≈ 2,600; half → ≈ 10,500"; V2 "≈ 3,350 (4 cells per sharpest Ω50)" | **Energetically irrelevant scale.** In `fixture.json`, S1 lobe index 3 (r50 0.049) has albedo 0.0005 / 0.0006 / 0.0008 / 0.0041 / 0.0409 at 0 / 30 / 45 / 60 / 80°. The energy-carrying lobes are index 5 (albedo 0.357, r50 0.164 at 45°) and index 2 (0.208, r50 0.124). The relevant scale is about 0.12–0.16 rad, which gives [ARITH] ≈ 250–440 directions at Δ = r50 and ≈ 1,000–1,700 at r50/2. That is consistent with WL90's 1,024. The 2,600–10,500 figures matter only near grazing. |
| V2: "M64 keeps only 0.42–0.46 of the flux in the mirror bin, so it must be wrong for at least one of the two deltas" | **Misread metric.** `peak_share_of_truth` = Xt[ii,pk].sum()/Xt.sum() is the share of M64's *own* flux in its argmax bin (`diag92.py:130`). It describes M64's spread, not an error of M64. The "wrong for one of two deltas" argument is analytic, and is supported only by the 86A beam pairs. |
| V2: order-≥ 3 transport is "46–62 % of region-C / panel reads" | **Imprecise.** 0.461 / 0.472 / 0.622 are shares of the *higher-order term h* (86A l.143-147), not of total radiance. LOW supplies about 76 % overall (86-10). In ECRO's convention these are table events ≥ 3, i.e. camera events ≥ 4. |
| WL90: 256 directions leave 9–20 %; 16 sub-bins remove the gap; single-read lower bound | Correct (90 l.59, l.201) |
| 86A beam pairs | Correct; post hoc (l.136). The written counterexample rule did not fire. |
| `Architecture.md:38`, `:42` quotations | Correct |

---

## 3. Inconsistent use of evidence

1. **The ORA render success reflects compensation by the learned global term (missed by both reports).**
   - The WL92 ladder row ENCB uses the same exact-local single flight as ORA, with path-traced L₃₊. It scores **0.045 (E2 T1) / 0.052 (E2 T0)** (92 l.150).
   - ORA (the same single flight with the frozen OWN G) scores **0.030 / 0.033 (E2 T1)** and **0.059 / 0.058 (E2 T0)** (93C l.133-134).
   - So on E2 T1, the learned G beats the true higher orders. ENCB's signed deficit is −0.054 / −0.076 of h, about half of it deployed-draw sampling (92 l.158). The G term apparently offsets it.
   - ORA's 0.030 / 0.033 and slope ≈ 1.0 (inspected E2) therefore cannot support "intrinsic local + learned G is accurate". (SUP: a cross-batch comparison of separate render runs, both 2 × 4,096 spp. ORA = X_M12 to 1e-7.)
2. **Whole-ECRO region-C slopes are used as evidence against a learned G** (V1 B-9, V2 B-6/B6: "learned C slopes 0.36–0.63").
   - Those slopes are of the whole learned hl term. The single flight dominates the region-C deficit: −0.123 against global +0.050 (86A l.145).
   - The G-specific measurements are:
     - "true L₂ + learned global" 0.052 / 0.064 (E2 T1) and 0.028 / 0.041 (E2 T0), against ORA2 0.013 / 0.010 (92 l.154-156). Neither report cited this.
     - LSPOT X_L3 71–82 % (92B-3).
     - Item 1 above.
3. **Pooled metric used alone (V1 B-5).**
   - ORA_RCV whole 0.0756 / 0.0772 vs ORA 0.0776 / 0.0774 is a 0.0020 / 0.0002 difference with no CI. It is pooled with g00010, where ORA's source use is poor (0.464 vs 0.289 / 0.393).
   - A learned source compensating on g00010 is the simpler explanation. Calling it "consistent with role-dependent projection" over-reads it.
4. **"Principal" asserted from component facts.**
   - V2 labels this correctly (PLAUSIBLE HYPOTHESIS).
   - V1's "holds by construction" does not.
5. **Seed vs geometry.**
   - OWN panel `a cᵀ`: the seed-to-seed difference of 0.51 exceeds the box-move change of 0.145 / 0.146 (92B-4).
   - V2 notes this; V1 cites both numbers without drawing the non-identifiability reading.
6. **Inspected conditions.**
   - Both reports label ORA renders "inspected / diagnostic". That is correct.
   - E2 dev aggregates were seen before the freeze, E5 has been inspected since WL93, and no never-inspected admitted sharp held-out exists.
7. **Smoke negatives / oracle as learnability.**
   - No violation found. V2 explicitly labels learned-K learnability UNSUPPORTED.
   - V1's "separation helps" rests on an exact, non-learned oracle and should be read only as "exact separation helps".

---

## 4. What each report missed that matters for the verdict

### 4.1 The closest existing test of the core hypothesis is already in the data, and it is ambiguous

**WL92 ladder on E2 T1 region C (one deployed draw, inspected; the decomposition is order-dependent and not strictly additive).** [ARITH] steps:

| Arm | Region C E_H | Step | Change |
|---|---|---|---|
| Learned single + L₃₊ | 0.124 / 0.128 | | |
| R16 (rank-16 material NMF) | 0.084 | rank-16 class alone | ≈ −0.04 |
| ENCB (exact local) | 0.045 | rank 16 → full rank | ≈ −0.04 |
| pooled ENCB | 0.027 | deployed-draw sampling | ≈ −0.02 |
| ORA2 | 0.013 | projection / encoding | ≈ −0.014 |

- So the local operator is the largest contributor on this inspected case. About half of that is the rank-16 class itself, and about half is learning beyond the class ceiling: learned peak 0.11–0.21 vs NMF-16 0.47–0.54.
- **This supports "the local operator is the principal error source" (SUP, inspected E2).**
- **It does not show *why***: geometry-encoded incident information, versus generic inaccuracy, versus non-identifiability through tied factors.
- Stage-isolated, the geometry-free exact operator beats learned T0 at the receiver (−0.0070, CI excludes 0) and at the source (−0.40). Composition ties on pooled single and whole (g00010).

### 4.2 Other omissions

- **Event numbering.** ECRO's camera LOW is e1+e2 and table events 2..5 = camera events 3..6 (`pt86.py:228-233`). V2's "sharp → sharp → diffuse caustic" (light → panel → sphere → floor) is camera event 3 = table event 2. In ECRO's convention it belongs to the single flight, not to an order-≥ 3 G_psi. V1's X2 uses the same example. IDTS must state whether it inherits LOW and this convention; neither report asked.
- **The WL90 locus is sender-side.** The binned *outgoing* L1 table is what the first gather reads. In IDTS terms that is K_phi's output resolution and the C^t transfer, not the receiver's incident z^t. V1 flagged this; V2 did not, and frames all of B-7 as a z^t problem.
- **ER is itself a learned compiler output** (WL83 `fit_*_L_ER_s8301`, fact sheet §7). "K_phi = ER" is therefore not "non-learned". The real question is what a *new* learned local operator adds over ER, and `Architecture.md:44` governs that.
- **Source linearity.** The only measured learned edge in the lineage is source-only update cost: 13–18 vs 51–64 ms (87-9). IDTS keeps it only if G_psi is linear in emission and cheap. IDTS does not state this; V2 noted the cost side only.
- **S2 rank.** V1 obligation 2 says low rank is "excluded for S1/S2-class". For S2 the rank-16 residual is 0.20–0.33 (`local.json`), so "excluded" is too strong for S2.

---

## 5. The single most decisive Stage-0 check (learning-free)

**Incident-span decomposition of the frozen learned local operators.** This tests IDTS's *core causal claim* directly, and does not depend on its unfixed operator classes.

**Procedure** (frozen T0 / OWN / WL86 checkpoints; panel and sphere patches; E2 T0/T1 plus the 55 validation geometries; per use: source on s16, receiver on M64(flight(q))):

1. Build each patch's realised incident subspace V_g,i, at a predeclared energy threshold, from the incident fields of a predeclared training-distribution light set at geometry g. All fields are computed learning-free.
2. On fields from *different* lights, split the learned operator's error against M64 into an in-span part (A−M)P_V and an out-of-span part (A−M)(I−P_V).
3. Compare the out-of-span operator difference between T0 and T1 geometries (box move) with the difference between seeds.
4. Render check with path-traced L₃₊ in place of any learned G, to avoid the compensation in §3.1. Substitute A' = M·P_V + A(I−P_V) at both uses. Report the fraction of the learned-single excess (0.124 → ENCB 0.045) that it removes.

**Falsification rule (to be predeclared).**
- **The core hypothesis is falsified for this lineage if** the in-span error carries most of the learned excess (A' ≈ ENCB), and/or the out-of-span difference is dominated by seed rather than geometry. The failure would then be generic inaccuracy or non-identifiability, not geometry-encoded incident information. Separation would remain sensible, but IDTS's stated motivation would be unsupported.
- **The core hypothesis is supported if** the in-span error is near M64's own floor, the excess sits out-of-span, and it tracks geometry.

**Inconclusive conditions.**
- dim V_g,i approaches 64, so there is no in/out separation.
- The q / target noise is heavy-tailed (g00010-type). Report per geometry, with a predeclared exclusion rule; never use the pooled value alone.
- Tied-use co-adaptation: the ORA_SRC lesson shows that substituting one use changes the other. Report both-use and per-use results.
- M64 is itself a uniform-in-bin projection and is not truth for within-bin-concentrated sources.
- The deployed-draw sampling deficit (≈ half of ENCB's error). Compare against ENCB, not ORA2.
- Low power: 42 panel patches, 32 sphere patches, 2 seeds.
- All renders are on inspected E2/E5. The check can falsify a causal claim on these configurations but cannot establish generalisation.

**Why this check and not a resolution ladder.** An exact-ER resolution ladder (D = 64 / 256 / 1,024) mostly re-tests WL90, which is already answered for explicit transport, and it tests a design choice rather than IDTS.

---

## 6. Recommended verdict: **NEEDS_SPECIFICATION**

**Reasons:**
- **No claimed contradiction survives as a contradiction of the IDTS text.**
  - C1 and C3 target instances IDTS never fixed.
  - C2's B5 part misreads "physically equivalent".
  - C4 is conditional.
  - X1–X6 are omissions or obligations.
- **Not STRUCTURALLY_UNJUSTIFIED.** Measured evidence makes the core hypothesis plausible:
  - the local operator is the largest error term on inspected E2 (§4.1);
  - the exact geometry-free operator beats learned operators stage by stage (`oracle_gate.json`);
  - the learned panel operator changes with an unchanged material (92B-4).
- **Not READY_FOR_FALSIFICATION as an architecture.** Several things are undefined:
  - K_phi's class, resolution, frame and relation to ER;
  - z^t's carrier, resolution and whether it is per-order;
  - the K∘C vs G_psi path partition, including who owns sharp chains and whether LOW is inherited;
  - G_psi's inputs, source-linearity and feedback;
  - the B5 equivalence definition;
  - B6 acceptance thresholds.
  
  Any Stage-0 beyond §5 would test a design choice, not IDTS. Many of these open choices are themselves risks, and IDTS gets no credit for leaving them open.
- **Not primarily NEEDS_CONTRACT_CLARIFICATION.** The owner-decision conflict (explicit transport as comparator only; `Architecture.md:44` on ER modification) becomes binding only after specification fixes K_phi and the explicit order count.
- **It should be listed as a pre-declared dependency:** once specified, a K_phi = ER / M_D instance with exact lower orders requires the same owner ruling as LSIT D1.
- **Only the core causal claim is falsifiable now,** via the §5 check on frozen checkpoints.

**Key files:**
- c:\Projects\ReST-GI\results\ecro_attr92\results\local.json
- c:\Projects\ReST-GI\experiments\ecro_attr92\diag92.py
- c:\Projects\ReST-GI\results\ecro_stage93c\oracle_gate.json
- c:\Projects\ReST-GI\results\gm1_char85\results\census\fixture.json
- c:\Projects\ReST-GI\docs\worklogs\92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md
- c:\Projects\ReST-GI\docs\worklogs\93C_WORKLOG_ECRO_DUAL_STAGE_PHYSICAL_SUPERVISION_SMOKE.md
- c:\Projects\ReST-GI\docs\worklogs\86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md
- c:\Projects\ReST-GI\docs\worklogs\93A_WORKLOG_ECRO_SOURCE_STAGE_TARGET_RELIABILITY_AND_SUPERVISION_SMOKE.md
- c:\Projects\ReST-GI\docs\Architecture.md
- c:\Projects\ReST-GI\experiments\ecro86\model86.py
- c:\Projects\ReST-GI\experiments\ecro86\pt86.py
- c:\Projects\ReST-GI\experiments\ecro86\scene86.py
