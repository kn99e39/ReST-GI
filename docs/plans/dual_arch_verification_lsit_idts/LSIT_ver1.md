# LSIT adversarial review: representation, interface and runtime lens

**Source.** `c:\Projects\ReST-GI\docs\plans\DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md` (346 lines, read in full). It is the only definition of LSIT used here.

**Method.** Read-only throughout. I edited no files and ran no jobs.
- I re-checked the load-bearing numbers in:
  - WL90 (§A, §C, factor table, verdict);
  - WL91 (cost table and the "Rays and memory" section);
  - WL92 (family-share table, ladder arm definitions);
  - WL93A (N_b table);
  - WL87 (C2-BINQ definition, source-only cost);
  - WL89 (C3UR definition, C_ss);
  - WL86 (oracle and dirty-ray figures);
  - `docs/Architecture.md` §§1–2.
- All worklogs are under `c:\Projects\ReST-GI\docs\worklogs\`.
- Every other number comes from the ledger or the fact sheet.

**Labels.** OBS = measurement verified in the named file. SUP = supported inference. PLAU = plausible hypothesis. UNSUP = unsupported. CONTRA = contradicted.

**Contract conflict, not treated as a defect.** LSIT makes events 1–2 explicit. That conflicts with the 2026-11-13 owner decision (Architecture.md line 13). It stays an unresolved contract conflict (LSIT's own D1).

---

## Findings

### A4. Does explicit second-order transport keep the directional information sharp receivers need?

**Finding.** No, not in general. The second event at sharp receivers (panel S1, sphere S2) brings back a binned *incident-direction* projection. This projection is not the one WL90 measured. No measurement in the repo covers it.

**1. What LSIT specifies.**
- E1 (the first-event field) is deposited into the receiver's 64 incident bins.
- The second event then draws K_r lobe samples "around incident-bin centres" (§4.2 C2(b), C3; §4.4 "Sharp directional effects").
- Sub-bin incident direction is therefore discarded at sharp receivers.

**2. Why WL90 does not cover this (OBS).**
- In every WL90 operator (EE, X(a,s), M_m, X1), the receiver gather G uses "per-ray frozen-ER weights W … **exact local direction**" (WL90 §A table).
- C3UR deposits into the receiver's dense 4,096-cell incident grid (8×8 cells per bin), not 64 bins (WL89 arm table, line 65).
- WL90's finding "order ≥ 2 projections negligible (P2plus +0.0009 / +0.0007 / −0.0004; m* = 1)" is about *sender-side* binning of L_k, with exact incident direction at the receiver.
- So WL90 gives no evidence that LSIT's receiver-side incident binning is harmless. **UNSUP** as a transfer.

**3. Closest measured control (OBS, weak).**
- WL87 C2-BINQ replaces each receiver ray's W by its in-bin mean. It gives E2 T1 C 0.046 against EXP 0.049 (WL87 lines 61, 225).
- E2 region C is dominated by light→panel→*Lambert/floor* receivers, where incident binning barely matters.
- BINQ therefore does not test sharp receivers. WL93C ORA uses M64(E1) at the receiver (the same incident-bin information) and reaches E2 T1 C 0.030 / 0.033 against oracle 0.0137. ORA also contains binned M64(s16), a single relation draw and a learned global term, so the receiver-binning share is **not isolated**.

**4. Analytic magnitude (SUP, arithmetic on the scene86 bin grid).**
- Grid: rings uniform in cosθ (boundaries θ = 90 / 75.5 / 60 / 41.4 / 0°), 16 × 22.5° sectors.
- Distance from a direction to its bin centre:
  - outer rings: up to about 11–13°;
  - polar-cap bins (centroid at cosθ = 0.875, θ ≈ 29°): up to **≈ 29°** for near-normal incidence.
- Lobe sizes (census85 fixture):
  - S1 r50 is 0.146 rad = 8.4° at normal incidence and 5.2° at 80°;
  - S2 r50 is 12–23°.
- Bin-centre misplacement is therefore 1.5–3.5× S1's r50 and about 1–2.4× S2's.

**5. Mitigating structure (SUP).**
- The grid is mirror-symmetric for a flat patch: φ→φ+π maps sector i to i+8, and rings map to themselves.
- So for a *flat* sharp receiver the dominant outgoing bin is still correct. The error is in how the lobe spills into neighbouring bins. S1's half-flux solid angle is 0.067 sr against 0.098 sr per bin, so a beam near a bin edge truly splits about half into the neighbour, and bin-centre placement does not reproduce that.
- This protection disappears for curved patches. Sphere normals vary by about ±22.5° inside one patch, so the reflected direction varies by **±45°** (2δ). LSIT does not say whether the bin-indexed lobe relation reflects about the patch-centre normal or the per-point normal, or how it handles the horizon. WL87's W_ray uses per-point frames, horizon and valid-solid-angle normalisation (fact sheet §6). LSIT's handling of these is unspecified.

**6. Which paths are affected.**
- (a) Camera pixels whose x2 is a sharp patch: for example camera→panel→sphere, or camera→sphere→panel.
- (b) Every L₂ at a sharp patch that R_θ then propagates: for example light→panel→sphere→wall.
- In E2 region C, 2-event paths through *Lambert* x2 make up most of the 0.54 explicit share (WL92 family table: 0.18 via the panel + 0.36 other). The affected subset is not separated anywhere.

**Counterexamples.**
- *Narrow source → sharp → sharp.* Two beams reach a sphere patch inside one incident bin but at different sub-bin directions. They produce the same E1[j,b] and hence the same L₂[j,·]. This is the identical-encoding pair of WL86A (86A-4: patch-level difference 0.94 azimuthal, post hoc) moved from the source to the second event.
- *Two facing mirrors.* The beam light→M1→M2 is quantised at M2. Its third-event footprint is displaced by distance × bin offset, for example 2 m × tan 11° ≈ 0.4 m ≈ 1.6 patches. That is the WL90 loss class, now at order 2.

**Label.** The projection exists: **SUP**, by construction in §4.2. That it is harmless: **UNSUP**, untested. For flat sharp receivers in this room it is plausibly small (**PLAU**); for the curved sphere it is plausibly large (**PLAU**).

### A5. Further information loss after the exact first gather

| Intermediate | Loss | Evidence | Label |
|---|---|---|---|
| (i) E1 deposited into 64 incident bins at **Lambert / floor** receivers | Essentially none if the cosine is applied per ray, as written: Lambert needs only cos-weighted flux. The floor B (α 0.27–0.45) needs rank 8 | BINQ − EXP −0.003 at E2 T1 C (WL87); WL92 floor rank 8 | OBS (control) → SUP |
| (ii) Same deposit at **sharp** receivers, then bin-centre lobe | Sub-bin direction lost (A4). Position–direction correlation inside the patch also lost: E1 is patch × bin | Not measured. ORA's total residual (≈ 0.016–0.019 above oracle at E2 T1 C) mixes this with other approximations and a learned global | UNSUP / PLAU |
| (iii) Output table B read by the camera | Adequate in this workload: oracle table at noise | WL86 C oracle 0.0074 / 0.0137, noise 0.007 / 0.013 (WL86 line 158); WL90 ORACLE 0.014 passes F2 | OBS (this room, point lights) |
| (iv) Binned L₂ as R_θ's input, propagated by flight | Sender-side order-2 projection is small in W1 / W2 (P2plus ≤ 0.0009; compounding −1…+9 %). If R_θ's flight uses the *receiver-side* state rays (ECRO form, §4.2 C4), the single-draw sampling limit returns for sharp L₂ senders | WL90 (pooled relation, OBS); 86A-8: median 13–17 / 512 rays hit the panel, minimum 0, RAY vs truth −20 %; 87-8: 56–62 % of important relations never hit | OBS (projection, W1 / W2) / PLAU (sampling at order 2) |
| (v) Smooth part q^S = M^S·s16 | Binned encoding cost is small at E2 C | WL92 ENCQ 0.045 / 0.053 vs EXP 0.041 / 0.048 | OBS (for full M64; M^S untested) |
| (vi) P = 16 points per sharp patch at the first event | Spatial sampling dominates once-scattered noise. The estimator class LSIT proposes (exact per-point light direction, P = 16) has panel relative L2 0.095 on train-representative data and **2.67 on g00010** (B0 = P 16, Mq 1,024). Raising Mq does almost nothing (×1.1) | WL93A Task A table (lines 137–143) | OBS (q-table level); propagation to receivers PLAU (lobe blur partly averages it) |

**Quantified conclusion.**
- After the exact first gather, the losses measured as negligible are (i), (iii) and the sender-side part of (iv), in W1 / W2 / E2.
- The unmeasured ones are (ii), sampling in (iv), and (vi) at receivers.
- (vi) is the only one with a measured magnitude that bears directly on LSIT: P = 16 with exact directions does **not** resolve g00010.

### A7. Is R_θ (events ≥ 3, input L₂) defined on sufficient inputs, and can its class represent its target?

**1. Sufficiency of a continuous L₂ (SUP, mathematical).** By the Neumann series L_{k+1} = K C L_k, every higher order is determined by L₂ and the geometry. A continuous L₂ is therefore a sufficient input.

**2. Sufficiency of the 1,270 × 64 binned L₂: conditional.**
- Holds when L₂ at sharp senders is not narrow (OBS: WL90 W1 / W2).
- Fails when L₂ leaves a sharp patch as a narrow beam: two facing mirrors, light→panel→sphere (A4 counterexample). **PLAU.**
- Also fails when position–direction correlation inside a patch matters (curved sphere, near lights). **PLAU.**

**3. The class cannot represent the target as LSIT defines it.**
- R_θ is the ECRO generator form with geometry-conditioned rank-16 local factors a cᵀ (§4.2 C4).
- Its training target is t − t2: *all* events 3–5 (WL92A split; fact sheet §2).
- That target includes every path with a sharp-patch scattering at event ≥ 3, such as light→wall→wall→panel→x2. S1's Lambert λ ≤ 0.002, so almost every panel vertex is a lobe vertex.
- A sharp patch's L₃₊ outgoing radiance is the mirror image of its incident field. That is a near-permutation however smooth the incidence is. Smooth input does not make the operator smooth.
- The same family at rank 16 is OBS-limited: NMF-16 mirror retention 0.47–0.54; learned a cᵀ 0.11–0.14; OWN 0.21 (WL92, WL92A).
- R_θ therefore either:
  - re-learns geometry-specialised factors at sharp patches, which is exactly LSIT's H-struct failure mechanism, now inside R_θ; or
  - leaves a bias equal to the unmeasured lobe-at-≥3 share.
- LSIT says the class "does not represent" these events (truncation A2) yet trains on a target that contains them. **CONTRA** (internal; see contradiction 1).

**4. Paths the class can handle (SUP).**
- light→wall→panel→wall: the panel event is order 2 (explicit). The third event is at a Lambert wall (rank-1 local). Only the binned-L₂ flight question applies.
- light→wall→wall→panel→·: lobe event at order 3. Outside the class.

**5. Magnitude (OBS).** Events 3–5 are 0.46 (E2 T1 C), 0.47 (E2 T0 C) and 0.62 (E1 panel) of the higher-order reading h (WL92 family table). The lobe-at-≥3 sub-share is unmeasured; LSIT's own P-T2 would measure it.

**6. Input difference from measured learned global terms (OBS).**
- The existing learned global terms (WL86 / WL92A OWN) take u = c·s (the source encoding) as input, not L₂.
- WL93C's "no level conflict" (93C-3) was measured with *that* global term. It is not evidence for R_θ(L₂).
- LSIT's formula reuses "P·Σ area·Q·u" but never defines u for L₂ input: underspecified.
- The learned global term carries 71–82 % of the LSPOT error (92B-3 X_L3). Extrapolation to unseen source classes is a measured weakness of this class. Whether L₂ input fixes it is PLAU, untested.

**Label.** Continuous input sufficient: **SUP**. Binned input sufficient: OBS in W1 / W2 only, PLAU-fails for chained or curved sharp paths. Class sufficient for its declared target: **CONTRA** (internal), with magnitude **UNSUP** until P-T2.

### A9. Do selective lobe-relation updates after rigid edits keep visibility and correspondence?

**Finding.** It is implementable for rigid edits with the mechanism that already exists. It is untested for the lobe relation, has gaps in the specification, and costs a lot when the mover is sharp.

**1. Implementable (SUP).**
- WL86 GPU segment dirty test: "selective 23 ms, full 29 ms; identical state" (WL86 line 71).
- WL91 C3UR / C3URX1: "selective = full (digest, segments, probe buffer; table 7e-9)", plus null-edit and move-and-return checks.
- The same old/new-volume segment test applies to stored lobe rays.
- The per-source first-event lobe rays are re-emitted for every source (§4.2 C2), so their visibility is always current.

**2. Correspondence (SUP).**
- Bins are patch-local. A ray starting on a moved sharp patch keeps its (point, incident bin, sample) index in the mover frame, but its world direction and hit change, so all of its rays must be retraced. §4.4 says so.
- A ray from a static patch that lands on the mover has a stale "outgoing bin at the hit" (mover frame). It is caught by the old-volume test only if the test includes the hit surface on the volume boundary. WL86 handles this, as verified for state rays.
- Rays that newly hit the mover are caught by the new-volume test.

**3. Gaps.**
- (a) **Misses / escapes:** the stored segment endpoint for rays that leave the semi-enclosed room is not specified (**UNDET**; WL86 state rays store o/e, which I did not verify for misses).
- (b) **Curved patches:** the frame and horizon handling of the bin-indexed relation on sphere patches is unspecified (A4).
- (c) **Sample offsets:** if fixed shared lobe-sample offsets are reused across patches and geometries (the WL86 seed-86001 convention), the result is fixed-pattern bias, not noise (**PLAU**).
- (d) **Validation:** a "selective = full" test for the lobe relation does not exist yet.

**4. Transport-dirty receivers (D_T) (SUP).** These are covered, but only because E1, L₂ and R_θ application are recomputed in full on every edit. The relation is geometry-only, which is correct. The cost is that every edit pays the full per-source explicit stage (A10).

**5. Non-rigid edits.** Not supported. Origins, normals and lobe frames of a deformed sharp patch change, so all of its rays must be rebuilt. Architecture.md §3 (line 64) requires conservative invalidation here.

**6. Sharp-mover cost (ARITH).** If the panel moves (as in W1 / V3, the yawed S1 slab), all 42 × 16 × 64 × 16 = **688,128** lobe rays are retraced (sphere: 524,288). That is about 10× WL86's dirty state set (8.7–10.2 % of 650,240 ≈ 57–66 k rays).

**Label.** Validity for rigid edits: **SUP** (mechanism precedent), untested for the lobe relation. Cost: **PLAU** high for sharp movers.

### A10. Are the complexity and memory estimates complete?

**Finding.** The estimates in §4.6 are incomplete and in two places misattributed. See the runtime table and contradiction 4.
- **Record count.** 74 × 16 × 64 × 16 = 1,212,416. This is correct (N_sharp = 42 + 32).
- **Storage.** "10–20 MB" holds only for compact records:
  - hit int16 + bin uint8 + weight float32 = 7 B → 8.5 MB;
  - storing a hit distance instead of a segment: 11 B → 13.3 MB;
  - storing the WL86-style 24-byte segment for the dirty test: 31 B → **37.6 MB** (WL86 state rays use 27 B/ray).
- **Per-source cost basis.** LSIT cites "C3URX1-type ≈ 10 ms (WL90)" for its 18,944 lobe rays. WL91 measured C3URX1 with **526,467 probe intersections + 526,467 shadow rays per light** (27.8× LSIT's count). The exact-evaluation time is flat at 11–14 ms from 0.26 M to 2.1 M landed rays, so the time does not scale with ray count. The R_mirror quality of 0.024 belongs to that larger ray set; quality at 19 k rays is unmeasured.
- **Missing terms in the per-source / per-edit estimate.** None of these are in ECRO's 76–78 ms:
  - the second-event scatter over 1.2 M records (it is **per source**, because E1 depends on the source; §4.6 does not say so);
  - the E1^S state gather;
  - flight(L₂) inside R_θ;
  - the explicit E1 / L₂ recomputation on every edit.
- **Scaling.** Storage and per-source cost grow linearly with N_sharp. If every patch is glossy: 1270 × 16 × 64 × 16 = 20.8 M records, 32× the state rays. Cost also grows with light samples (area lights) and is capped by the two-light interface.

## Claims that hold only under restrictive assumptions

1. **Point lights.** "Exact per light sample" depends on them. Area lights need many samples, and per-ray exact L₁ is no longer deterministic (WL90 unresolved list).
2. **At most two lights.** `scene86.light_arrays` is hard-limited (fact sheet §1).
3. **Few sharp patches.** N_sharp = 74 / 1,270 = 5.8 %. Memory, per-source and sharp-mover retrace costs all scale with N_sharp.
4. **Sharp surfaces that do not chain.** The room has one flat sharp panel (which cannot see itself) and one moderate sphere. A2 fails for two facing mirrors, panel↔sphere chains, and any sharp event at order ≥ 3.
5. **Isotropic, half-vector-separable, per-channel-consistent lobe.**
   - S1 has α_u ≈ α_v, so it is isotropic enough.
   - S1's green/blue dominant lobe is an *inner* lobe at refracted directions (η_e 2.4955, at the bound). Its exterior width depends on θ (≈ η× near the normal, growing toward grazing).
   - There are s(μ_i)s(μ_o) per-RGB factors.
   - The red and green/blue dominant lobes differ (fact sheet §7).
   - A separable F(θ_d)D(θ_h) is therefore an approximation, and A1 / A4 are at risk.
6. **Flat sharp patches.** The bin-centre second event is benign only because the grid is mirror-symmetric (A4). Curved sphere patches break this.
7. **Patch-scale content.** The table container was qualified at noise in this room only (WL86, WL90).
8. **Rigid edits.**
9. **Smooth higher orders.** R_θ owns 46–62 % of the reading (WL92).
10. **Lights not close or grazing to sharp patches.** g00010-type configurations: P = 16 exact-direction noise is 2.67 (WL93A).
11. **One scene family** for R_θ's training (WL86 population).
12. **Static materials.** The split is fitted offline per material.

## Claim / evidence matrix (A1–A6 as listed by the brief)

### (A1) Sharp intrinsic scattering is unnecessarily high-rank in absolute bins
- **Supporting:** S1 M64 needs rank 46–59 for 99 % energy; rank-16 residual 0.46–0.52; floor B rank 8 (WL92). S1 Ω50 0.067 sr is smaller than one bin (0.098 sr). Per-geometry rank-16 fits reach the rule (92-4).
- **Contradicting / weakening:**
  - Matrix rank is invariant under any invertible linear change of bin basis, so a near-permutation is high-rank in *every* linear basis.
  - A half-vector reparameterisation reduces the parameter dimension (shift / circulant structure), not operator rank. §3 ("low-dimensional … in a mirror-aligned parameterisation") and Alt-A ("low rank in the reparameterised coordinates") conflate the two.
  - Isotropy already makes M64 nearly block-circulant over the 16 sectors (16 frequencies × 4×4 ring blocks), with no lobe split.
  - For explicit use, a dense M64 costs 73,728 floats in total, so rank is irrelevant there.
- **Unestablished:** that "unnecessary" rank can be avoided by a *learned* operator without geometry specialisation.
- **Follows logically?** Only as "an unstructured rank-16 factorisation is the wrong parameterisation". Not as "the operator is low-rank in another basis".
- **Discriminator:** a learning-free check of M64's block-circulant residual, plus P-T1.
- **Label:** high rank OBS. "Unnecessary / low-dim structure" PLAU. "Low rank after reparameterisation" CONTRA (mathematical).

### (A2) K = K_L (per-ray lobe) + K_S (compact non-negative residual)
- **Supporting:** ER is a sum of GGX lobes plus Lambert (closure83, reciprocal and passive by construction).
- **Contradicting / weakening:**
  - Inner refracted lobes and s(μ) factors are not of half-vector separable form, so f^S ≥ 0 forces the lobe fit to undershoot and leaves a narrow, lobe-shaped residual. M^S could then stay high-rank (**PLAU**).
  - **Functional role:** in LSIT's runtime (§4.3), M^S appears only in the explicit products M^S·s and M^S·E1. P1 (low rank of M^S) is load-bearing nowhere in the runtime; it matters only for justifying that R_θ's rank-16 factors suffice, which A7 contradicts.
  - The split adds a fit approximation (A4 positivity, f^L + f^S ≈ f) and no information over evaluating the exact ER per ray, which W_ray and X1 already do (**SUP**).
- **Unestablished:** separability, positivity, M^S rank. Also P-T1's residual normalisation (relative to the full S1 or to M^S?) is ambiguous.
- **Discriminator:** P-T1, plus a per-ray exact-ER control with no split.
- **Label:** PLAU (unmeasured). Necessity of the split for events 1–2: UNSUP.

### (A3) Explicit transport through the second event
- **Supporting:**
  - WL90: first-gather angular projection isolated (97–102 %), m* = 1.
  - WL91: pooled X1 within −0.0025…+0.0035 of the oracle in 6 conditions.
  - WL93C ORA: E2 T1 C 0.030 / 0.033, slope ≈ 1.0.
  - WL87: explicit ≥ ECRO (0 better / 3 worse out of 36).
- **Contradicting / weakening:**
  - WL91: single-draw variance; held-out validity 1 / 3; LSPOT89 C_ss harm +0.0025 [+0.0005, +0.0045].
  - EXP passes 75 % of realisations.
  - LSIT's event 2 is bin-centre and untested (A4).
  - LSIT's first event uses about 28× fewer rays than C3URX1, with P = 16 spatial noise (WL93A).
- **Unestablished:** fidelity at LSIT's budget and discretisation; held-out sharp-reader validity (no clean admitted held-out exists after WL92B).
- **Follows logically?** Only for event 1, in the W1 / W2 regime.
- **Discriminator:** P-T3 with an arm that uses exact per-ray incidence at sharp receivers.
- **Label:** value of an exact event 1 SUP (W1 / W2, point lights). LSIT's event-2 form PLAU. Admissibility: unresolved owner-contract conflict (D1).

### (A4) Learned geometry-conditioned linear operator for events ≥ 3
- **Supporting:**
  - WL86 broad accuracy (stationary E_H 0.004–0.006).
  - WL92A OWN global error ratio 0.42–0.46 against SUM; E4 C reading +0.006 / +0.004 (E4 not admitted).
  - WL93C ORA composes with the frozen global without level conflict.
- **Contradicting / weakening:**
  - Those global terms take s as input, not L₂.
  - LSPOT X_L3 71–82 % (92B-3).
  - The target contains lobe events at order ≥ 3 that the class is said not to represent (A7).
  - The learned operators are geometry- and seed-dependent (92B-4: 14.5 % and 51 %).
  - Receiver-side flight of sharp L₂ hits the sampling limit (86A-8, 87-8).
- **Unestablished:** learnability with L₂ input; size of the lobe-at-≥3 share.
- **Discriminator:** Stage 1 LSIT-learned against LSIT-oracle, plus P-T2. Oracle success must not be read as proof of learnability.
- **Label:** broad part SUP within the training family. As specified: PLAU, with an internal CONTRA (A7).

### (A5) Most sharp-lobe transport falls within the explicit orders
- **Supporting (OBS, WL92 family table):** E2 C h splits into 0.18 via-panel 2-event, 0.36 other 2-event, 0.46 3–5-event (E2 T0: 0.12 / 0.41 / 0.47).
- **Weakening:**
  - E1 panel: 0.62 of h is 3–5-event.
  - The lobe-at-≥3 sub-share is unmeasured.
  - Analytically, every panel or sphere vertex at order ≥ 3 is a lobe event, and the panel is a large reflector.
- **Discriminator:** P-T2.
- **Label:** UNSUP (unmeasured; plausible only for the single-flat-panel room).

### (A6) One shared intrinsic kernel at the source and receiver uses
- **Supporting:**
  - Physically, ER is one closure, so sharing is a physical fact (SUP).
  - In LSIT, events 1–2 have no role-specific parameters, so the learned WL93A–C stage conflict cannot arise there (SUP by construction).
- **Contradicting / weakening:**
  - (i) At events ≥ 3, local scattering of the same material is done by R_θ's learned geometry-conditioned factors. The response operator therefore depends on event order, which conflicts with Architecture.md §2 line 21 ("Direct and C-mediated incidence use the same intrinsic response rule; no … direct/indirect response branch").
  - (ii) The source use (exact per-point direction) and the receiver use (bin-centre) discretise the same kernel differently.
  - (iii) "Identity by construction … at the camera bounce" is false: LOW uses the unsplit ER, so identity holds only within the split tolerance.
  - WL93B (57–68 % divergence) and WL93C (ORA ties T0 in the pooled metric) concern learned or binned operators. They neither prove nor refute that sharing is sufficient.
- **Discriminator:**
  - an identity test at events 1–2;
  - P-T5 / Stage 1 checks for role conflict inside R_θ;
  - an identical-material, different-geometry check applied to R_θ's local factors, not only to the split.
- **Label:** SUP for events 1–2. CONTRA as a whole-path claim.

## Runtime table

All rows are on an RTX 3080 Ti class GPU, warm, uncontended, one room, 1,270 patches.

| Cost | Measured (OBS, source) | LSIT estimate (author's) | My arithmetic / assessment (PLAU unless marked) |
|---|---|---|---|
| **Per frame** | ECRO 0.33 ms/spp GPU vs PT 1.80 (WL86); the same structure in EXP / TF (WL87 §E) | unchanged | SUP: same LOW + table readout, so ECRO-equal |
| **Per edit, wall** | ECRO 76–78 ms (geometry 31, state 21, source 5, operator 11, transfer 1.5; WL86). EXP 89–95, PT 35–36 (WL87). Without render: C3URX1 69.4, C3UR 83.0, R0 89.8, PT 30.9 (WL91). C3UR selective update 14.9 ms, full relation build 52 ms (WL91) | "≤ 76–78 ms + selective lobe re-trace" | Omits per-edit full recomputation of E1 / L₂ / flight(L₂) / R_θ application per source. Lobe relation selective update roughly 15–30 ms; about 50–100 ms when a sharp patch moves (688 k / 524 k rays), scaled from WL91 build rates. Total plausibly about 100–150 ms (unmeasured) |
| **Per source** | ECRO 13–18 ms; EXP 51–64 (q 42–54 + recurrence 10) (WL87). OWN source 4.6 + application 1.1 + transfer 2.3 (WL92A). C3URX1 21.4 vs C3UR 37.4 without render; exact source evaluation 11–14 ms, flat in ray count (WL91). 16 sources: 314 vs 517 ms | "≈ 19 k rays, C3URX1-type ≈ 10 ms" | Lobe first flight about 11–14 ms (flat floor) + second-event scatter over 1.21 M records + E1^S gather (≈ 2.7 ms first-gather class) + flight(L₂) + R_θ application. Plausibly about 20–40 ms, near LSIT's own R4 limit (2× ECRO = 26–36 ms). Unmeasured |
| **Rays** | State 650,240; dirty 8.7–10.2 % per edit (WL86). s16 shadow rays 20,320 per light. C3URX1 526,467 probe + 526,467 shadow per light (WL91) | 1.2 M lobe records; about 19 k rays per light | ARITH: 1,212,416 records. Per light 18,944 lobe + 1,184 shadow (inside s16's 20,320). Sharp mover: 688,128 (panel) / 524,288 (sphere). All-glossy scene: 20.8 M records |
| **Storage** | State 17.6 MB; W 3.9 MB; ECRO factors 42.9 MB; table 1 MB; C3UR relation 20.8 MB + probe buffer 12.6 MB; process 3.3–4.9 GB (WL86, WL87, WL91) | lobe relation 10–20 MB | ARITH: 8.5 MB (7 B/record), 13.3 MB (11 B), 37.6 MB (with WL86-style segments). M^S 73,728 floats. Per-light E1 / L₂ buffers about 1 MB each |
| **Offline** | ECRO about 15 min supervision + 22 min training (WL86). OWN 2,288–2,297 s per arm + 16 min component targets (WL92A) | K-split fit; R_θ OWN-style | The cost of the K-split fit is unmeasured. Explicit L₂ inputs per training item are cheap (PLAU) |
| **Quality at LSIT budget** | C3URX1 R_mirror 0.024 at 526 k rays per light (W1 T1); X1 single draw fails held-out 1 / 3 (WL91) | — | **Unmeasured** at 19 k lobe rays and P = 16. WL93A shows P = 16 exact-direction noise of 0.095, and 2.67 on g00010 |

## Critical contradictions (precise only)

1. **Target contradicts class (§4.2 C4 / §4.7 / A2).** R_θ is said not to represent lobe events at order ≥ 3, but it is trained on t − t2, which contains them. So either H-struct (geometry-specialised rank-16 factors at sharp patches) reappears inside R_θ, or there is a bias of unmeasured size. "OWN-style training prevents compensation" does not hold, because the target itself requires the excluded component.
2. **§4.4 "identity by construction … at the camera bounce" vs §4.2 C5.** LOW path-traces the *unsplit* ER closure. Identity with K_L + K_S holds only within the split-fit tolerance.
3. **Event-order-dependent local response vs Architecture.md §2 (line 21).** The same material is scattered by K_L + K_S at events 1–2 and by R_θ's learned, geometry-conditioned factors at events ≥ 3. This conflicts with the "same intrinsic response rule" contract. It is separate from the D1 / owner-decision conflict.
4. **§4.6 cost basis.**
   - The cited ≈ 10 ms "C3URX1-type" figure measures a path with about 28× more per-light rays (526,467 + 526,467) whose time is flat in ray count (WL91).
   - The per-edit bound "≤ 76–78 ms + re-trace" leaves out the per-source explicit E1 / L₂ / flight costs that ECRO never had.
   - The second-event redistribution over 1.2 M records is per source, which §4.6 does not state.
5. **§3 and Alt-A rank wording.** "High-rank in absolute bins but low-dimensional / low rank after mirror / half-vector reparameterisation" is false as an operator-rank statement. Rank is invariant under invertible linear basis change, and the reparameterisation is a non-linear re-indexing of the kernel domain (fewer parameters, same rank).
6. **P1 is not load-bearing at runtime (§4.3).** M^S is used only in explicit dense products, so its rank affects nothing at runtime. Meanwhile the split adds an approximation (A4) with no information gain over exact per-ray ER, which WL87 W_ray and WL90 X1 already evaluate.
7. **The second-event specification is ambiguous (§4.2 C2(b) vs C3).** The lobe relation stores *hits at other patches* ("hit patch, outgoing bin at the hit"). But L₂[j,·] is written as j's *outgoing* table, which needs no traced rays. One reading makes the relation a sender-side flight of L₂, which belongs to R_θ's input. The other leaves L₂[j] as a bin-centred matrix product. LSIT's two-event information and costs depend on which is meant.
8. **Overstated evidence label.** E6 ("sender-side placement removes it at the same budget", marked [M]) rests on WL88. WL88 is STOPPED_WORKLOAD_UNQUALIFIED with every hypothesis item NOT TESTED. Its sender / receiver attribution is a single draw (88-3), and C3 is not identical-state (a relation per order and per source).
9. **g00010 claim.** "Exact per light sample" at P = 16 does not remove g00010-type error. The same estimator class (B0: P 16, exact per-point directions) measures 2.67 relative L2 on g00010's panel (WL93A Task A). LSIT removes the direction-binning part only.

**Files most relevant.**
- `c:\Projects\ReST-GI\docs\plans\DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md`
- `c:\Projects\ReST-GI\docs\Architecture.md` (lines 13, 19–21, 38, 62–64)
- `c:\Projects\ReST-GI\docs\worklogs\90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md` (§A, §C, verdict)
- `c:\Projects\ReST-GI\docs\worklogs\91_WORKLOG_GM1_C3URX1_HELDOUT_VALIDATION_AND_ECONOMICS.md` (lines 255–312)
- `c:\Projects\ReST-GI\docs\worklogs\92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md` (family table, lines 120–128)
- `c:\Projects\ReST-GI\docs\worklogs\93A_WORKLOG_ECRO_SOURCE_STAGE_TARGET_RELIABILITY_AND_SUPERVISION_SMOKE.md` (Task A table, lines 135–145)
- `c:\Projects\ReST-GI\docs\worklogs\87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md` (lines 61–64, 225, 281)
- `c:\Projects\ReST-GI\docs\worklogs\89_WORKLOG_GM1_SENDER_RELATION_QUALIFICATION_AND_DISCRIMINATION.md` (line 65)
- `c:\Projects\ReST-GI\docs\worklogs\86_WORKLOG_ECRO_GLOBAL_RESPONSE.md` (lines 71, 158)
