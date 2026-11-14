# WL95 critic report: completeness and correctness of the Stage D/E synthesis

**Scope.** Read-only. I read all eight evidence files in full and re-verified the following:
- `experiments/fnto_compiled83/closure83_dr.py:1-14`
- `docs/Architecture.md:13,21`
- `docs/RESEARCH_ROADMAP.md:128`
- `experiments/ecro86/ecro86.py:55-67`
- `experiments/ecro_mat94/mat94.py:25-35,288-299,408`
- `experiments/ecro_attr92/diag92.py:204-210`
- `docs/worklogs/91_*.md:285-290`
- `docs/worklogs/89_*.md:345-356`
- `docs/worklogs/86_*.md:41`

I edited nothing and ran nothing on a GPU.

**Overall.** The PREMATURE verdict is defensible and consistent with canon (Architecture.md:13, roadmap:128). The synthesis still has one structural gap: Stage C fields are missing. It has three errors that would invalidate the proposed smoke: the bound-operator file is missing its arms, the D1 isolation is wrong, and the D6 "impossibility" is false. It also has several overreaches and unmatched-regime uses, listed below with the replacement text.

---

## A. Missing required items

**A1. Stage C per-hypothesis fields are absent (the main gap).**
- The brief requires at least 2 materially distinct hypotheses. Each must state:
  - state representation and operator inputs/outputs;
  - local-scattering ownership and geometry-dependent propagation ownership;
  - sharp-structure treatment and diffuse/nonlocal treatment;
  - reuse after edits;
  - training targets and supervision semantics;
  - inference-time physical computation;
  - memory/compute scaling and cache invalidation/update rules;
  - why it addresses the failure.
- §3 is only a deduplication table that refers to the designer documents. The final report must carry these fields for at least G1 and G2.
- Mark G2's Λ_θ fields as "UNSPECIFIED", not omitted. Fill G1's from the cited facts:
  - **ER eval:** the eval function returns f·cosθ_o [FACT closure83_dr.py:7].
  - **First-gather cost:** C3URX1's first gather costs 526,467 probe intersections plus 526,467 shadow rays per light [MEAS WL91:256].
  - **Invalidation and dirty test:** a light move or edit forces a full first-flight recompute. The dirty test is the rigid-OBB, box-only test [FACT ecro86.py:57-66].

**A2. Evidence section A (physical/mathematical contracts) is missing.**
- No transport equation, state variables or invariants are written. Add:
  - **[FACT]** WL86: T = S_ER C_G, q = S_ER e, LOW = q + Tq, b̂ = q + Tq + T·R_θ(G, z_ER)[e]. ECRO owns events ≥ 3. Events are truncated at D = 6.
  - **[FACT]** The canonical K/C form: i_a = e_a + Σ_b C_ab(G) b_b and b_a = K_a[i_a]. K is linear, non-negative and passive at fixed geometry. Every segment has exactly one owner (Architecture.md:25,31-38).
  - **Invariants:** ER is passive and reciprocal (reciprocity 1.2e-5, furnace 6e-7; WL87). ECRO is linear, non-negative and exact under superposition, with no passivity constraint (WL86:41).
  - **[FACT]** ECRO partitions by event count, not by K/C segment, and no canonical document maps one to the other (read_canon (B)). This is unresolved for G1 and G2 too.

**A3. Evidence section B is not organised as supporting / weakening / inconclusive for WL92–WL94D per hypothesis.** Add a 3-column table per hypothesis. For example, for E-L:
- **Weakening:** zero-operator level (auditB); S_geo below 1 %; WL94D NO_RELIABLE_IMPROVEMENT.
- **Supporting:** the M64 sender is directly supervisable (WL94D); WL93A TQ peak 0.31→0.48 and E2 render 0.076→0.046 (secondary metrics).
- **Inconclusive:** WL93C NO_SIGNAL; WL94A seed dependence; WL94C seed disagreement.

**A4. Mandatory deformation/bending/self-contact implications per hypothesis are missing** (RESEARCH_CENTRIC_TOPIC.md:20, binding for each major decision). §4 row 4 and §7 note only that no such workload exists. Add one line per row:
- **[FACT]** The current dirty test, for ECRO, G1, G2 and H-0 alike, is rigid OBB of boxes A/B only (ecro86.py:57-66). It is not G1-specific, as §4 currently implies. Move it to all rows.
- **[INF]** Non-rigid edits have no invalidation rule in any row.

**A5. Initial construction cost omits offline cost.**
- §4 "Initial vs per-edit" lists only the state build. Add the following, all as [FACT]/[MEAS]:
  - **ECRO and E-L:** target generation of 1,024 geometries × 4 lights × 256 spp plus training (WL86:129).
  - **G2:** M64 per material, 64 × 64 bins × 64 cells × 16,384 MIS (quad87.py:223-242), plus Λ_θ training.
  - **EXP:** W_ray at 65,536 MIS per material.
  - **G1:** no training, but per-light first-flight tracing at every light change.

**A6. Worst case is not stated.** "Worst case and scaling assumptions" lists only assumptions. Add per-row worst cases [INF]:
- **ECRO:** any edit regenerates all factors through dense Ft O(N²) and 3 message rounds; scene-size scaling is unmeasured (86:260).
- **G1:** cost ∝ L × peaked area × density; full recompute on every light change; per-atom continuation as α → 0.
- **H-0:** J sweeps × N·K.
- **G2:** ECRO generator plus Γ_s relation update (GM1 14.3 ms, WL89:350; another scene).
- **All rows:** dirty fraction near 1 when a large mover crosses most rays.

**A7. Criterion 10 violates the AGENTS.md baseline-terminology rule.** "Usefulness vs full recompute" mixes roles. Split it into four lines:
1. Path-traced re-render (PT role): ECRO 8.8–9.6× per frame, descriptive (WL86); C3URX1 7.6–31× on the GM1 sharp tiers (WL91:288; C3URX1 only, not EXP/C3UR).
2. Same-checkpoint F/S: selective 21 ms vs full 26–28 ms state rebuild (WL86).
3. Neural Full Recompute (8DNA/RNA): UNMEASURED.
4. Native Reuse: not applicable.

**A8. The smoke needs an explicit non-comparative declaration and implementation boundary.** Add to §6:
> "This smoke is attribution-only and non-comparative: no external-baseline role (Neural Full Recompute / Native Reuse / F/S / PT) is measured and no superiority claim follows. Implementation boundary: a new diagnostics directory; read-only use of own_s8601/8602, scene86/CK2 fields and S1 operators; no change to `ecro86/`, `model86.py`, checkpoints or production paths."

**A9. Stage B matched/unmatched corrections are not carried into §2.** Add:
> "[FACT] WL94D's statement 'operator ≈1.0 vs rank-16 Eckart–Young floor 0.46–0.52' compares pooled flux-L1 with per-colour flux-Frobenius; it is unmatched (auditB §1c.1). Likewise WL92 operator-peak 0.47–0.54 vs WL94D field-peak 0.22–0.33 (gap 0.08 from the metric alone, auditB §1c.2). WL94D's 'the 64-bin flight floor exists even for the exact sender' is stronger than the evidence (auditA §3, bottom line)."

**A10. Labels are missing on many claims.** Examples:
- the §3 disposition cells;
- §5 reasons 2–6;
- the §6 decision outcomes;
- the rejected-alternatives table.

In particular, §5 reason 2 ("predict no accuracy or cost headroom") must be [INF] and must carry the cross-regime caveat (B5).

**A11. The verdict needs a one-sentence justification against MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION.** §8 lists "what remains to be discriminated", which reads like MULTIPLE. Add:
> "Not MULTIPLE_CANDIDATES: the only fully specified class (G1) is inadmissible as the method under current canon pending Gate A, and G2/E-L are not yet falsifiable; were Gate A to admit G1, the status would become MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION (G1 vs H-0)."

---

## B. Overreaches and unmatched-metric uses

**B1. §2, "Optimisation/objective is currently binding. The learned operator is far from its own rank bound."**
- auditB lists optimisation versus generator capacity as untested.
- The bound arms are unconstrained, material-level, M64-relative factorisations. The learned `a` is also the decoder of the global term: B = a·(u1 + y_g) (model86.py:121-124).
- Replace with:
> "[MEAS] The learned operator is 0.36–0.66 above admissible non-negative rank-16 operators in the same flux-L1 norm (relative to the M64 teacher, not physical q). [INF] Therefore the rank-16 bound is not what currently binds; optimisation, objective/normalisation (WL94D teacher den), generator parameterisation, and the sharing of `a` with the global decoder are not separated."

**B2. §2, "Geometry conditioning adds nothing measurable for S1."**
- S_geo measures how much the learned operator varies across geometries. It does not measure the value of conditioning.
- Replace with:
> "[MEAS] The learned S1 operator is geometry-invariant to 0.2–0.8 % (S_geo, WL93D) and C1 swaps change nothing measurable; [MEAS] a position-conditioned rank-16 bound is worse on held-out fields (0.451 vs 0.362, auditB); a geometry-conditioned bound was not computed."

**B3. §2, "Co-adaptation is the explanation."**
- WL93D's mechanism is labelled an inference and was not isolated.
- Replace with: "Co-adaptation is the recorded interpretation (WL93D COADAPTATION_LIMITED; mechanism [INF], not isolated)."

**B4. §2, "RAY-vs-BIN doubled the patch RMS error."**
- The measured values are 0.50/0.38 on E2 T1 (1.3×) and 0.93/0.47 on E2 T0 (2.0×). Aggregate readings were mixed: RAY −20/−20 %, BIN −16/−26 % (WL86A Amendment 2).
- Replace "doubled" with "raised by 1.3× (E2 T1) to 2.0× (E2 T0)".

**B5. GM1 numbers used to predict ECRO/CK2 behaviour (unmatched regime).**
- **WL90 P2plus ≤ 0.0009:** this is GM1 R_mirror E_H for order ≥ 2 *bin-projection* error. It is used to predict that G2 has no headroom (§3, §5.2) and to reject VSA ("No measured target").
- **It is the wrong quantity.** It does not measure the importance of sharp ≥ 3-event transport on CK2.
- **The synthesis contradicts itself.** It also reports e3_p = 0.755 of panel-face e3 and calls that family unowned (§5.5).
- **Fix:** label both uses [INF, cross-regime, GM1 only]. Delete "No measured target (WL90 P2plus ≤ 0.0009)" from the VSA row; VSA's other reasons stand.
- **Also cross-workload:** "sweeps 11.6 ms vs generator ~12.5 ms" compares C3UR's 4-order application in GM1 (WL89:353) with the ECRO operator plus transfer in scene86 (WL86). Label it [INF, different scenes and sizes].

**B6. §2, "[MEAS] ECRO E2: sender placement dominates (+0.050 / +0.053, WL88)."**
- This is the explicit-relation factor C3UR − R0D on the scene86 E2 workload, from a single draw. It is not about the learned operator.
- Reword accordingly. Also add the pooled residual that is already measured: [MEAS] WL92 pooled EXP/ENCB 0.027 vs ORA2 0.0127 on E2 T1. So about 0.014 above the oracle remains after pooling, unattributed. This bears directly on D3 and D4.

**B7. §4 row 5, "ECRO v0: Not passive [MEAS]."**
- No ECRO passivity measurement exists. WL86:41 states that no passivity is imposed. CK2 energy 1.71–1.75× M64 (W94-biased) is not a passivity test.
- Replace with: "No passivity constraint [FACT WL86:41]; passivity not measured; compensation [MEAS WL86A Amendment 1]."
- For H-0, replace "Passive" with "passive up to W_ray quadrature error (S1 directional-albedo error median 5.3 %, max 18 %, WL87)".

**B8. §1 item 2, ER description.**
- [FACT, closure83_dr.py:3-12] ER is Lambert plus 7 GGX lobes split into outer lobes and refracted inner lobes, with t(w) mapping by η_e and a piecewise-linear s(μ_i)s(μ_o) factor. It has 69 floats and eval = f·cosθ_o.
- The closure *supports* anisotropy (au, av). Whether the S1/B/S2 states actually use au ≠ av was not checked.
- So in the MEK row, change "contradicts ER's anisotropic GGX [FACT]" to "conflicts with ER's anisotropic-capable lobes [FACT closure]; actual anisotropy of S1 unverified".

**B9. §2, "Floor B is exact at rank 8–16."**
- The values are field L1 0.012–0.017 at rank 16 and 0.065 at rank 8.
- Replace with "near-exact at rank 16; 0.065 field L1 at rank 8".

**B10. §2, G_PHYS "0.953, so no generic floor is shown."**
- G_PHYS is a pooled ratio. Its rel L1 of 0.641 is not interpretable because the labels are unreliable (256 vs 64 spp corr 0.525).
- Replace with: "G_PHYS's pooled ratio 0.953 shows no generic over-delivery; it cannot exclude a generic misplacement floor (label reliability 0.525)."

**B11. §5 "Evidence-favoured constraint."**
- As written ("not justified as the owner of local angular response") it leans toward declaring a representation limit. The exact-sender render gains it cites are confounded: textured and pooled metrics move the other way.
- Replace with:
> "[INF] Current evidence does not support a geometry-conditioned learned rank-16 tied bin operator, as trained, as owner of S1 local angular response (matched-norm zero-operator level relative to M64; S_geo < 1 %). It does not establish that the rank-16 class is inadequate for delivery/E_H; that is untested."

**B12. §5 reason 1, "G1 is excluded as the method by canon."**
- Canon names EXP/C3UR and X1-style findings specifically (Architecture.md:13).
- Replace with "[INF] G1 is inadmissible as the method under current canon by close analogy (Architecture.md:13 names EXP/C3UR as comparators and states X1-style findings do not justify explicit GI; roadmap:128 requires an ECRO-lineage hypothesis); formal status = Gate A."

**B13. G1's class rule needs a prospective basis.**
- G1's "directly lit peaked senders" partition resembles WL86B H-P (WEAKENED 2 / REFUTED 1; class boundary post hoc at the panel).
- Add: the peaked rule must be the material-only Π > 8 rule fixed in WL88 and must meet the WL86B admissibility criteria: prospective class rule, tallies summing to the reference, per-term tallied targets. Otherwise it is an unjustified decomposition under the brief's multi-model rule.

**B14. Decision outcomes D1 and D4 overgeneralise from one material and one or two workloads.**
- In D1, "admissible only with ≥ rank ~32" rests on field L1 relative to M64, not on delivery. In D4, "any candidate must carry sub-bin…" is too broad.
- Append to each: "within the tested workload class (S1, compact point lights, planar sender, rigid box edits)".

---

## C. Factual or design errors that would break the smoke

**C1. The `S1_ops.npz` contents are misstated.**
- [FACT, auditB code] `wl95_calc.py` saves only these arms: svd1–svd64 (signed), svd16_clip0, nmf16_wl92, nmf16_l1irls, zero, rrr{4..32}_{W94,WT} and rrr16_W94_clip0.
- `fa_nmf16_l1` and the L1 non-negative rank ladder, including rank 32 (0.146 / 0.758), are written only to JSON by `wl95_calc2.py`. They are **not saved**.
- So "L-bound32 = L1 rank-32 (fa ladder) from S1_ops.npz" is wrong. svd32 is signed and inadmissible.
- Replacement:
> "L-bound16a = nmf16_wl92, L-bound16b = nmf16_l1irls (both in S1_ops.npz); L-bound32 = the rank-32 field-adapted non-negative L1 operator, which must be regenerated deterministically by re-running the `wl95_calc2.py` ladder (seed 9500+k·10+c, CPU, ~1 min) with an added `np.savez` (new code, declared)."

**C2. D1 confounds rank with incident projection.**
- The bound operators are rank-k approximations of M64, applied to the 16-point s. Comparing L-bound16 with L-q256 mixes the rank loss with L1/L2/L12.
- Replacement:
> "D1 (rank): at pooled propagation compare L-bound16 and L-bound32 against **L-M64** (same s, same frame). D2 (incident/spatial projection): L-M64 vs L-q16 vs L-q256. If L-bound16 ≈ L-M64 within noise, rank is not binding for delivery/E_H on this workload, and the learned deficit is attributable to factors not separated here (optimisation, objective, generator parameterisation, shared `a`)."

**C3. D6 is not logically impossible.**
- Q_r goes through the patch × bin table and the nearest-patch readout (L10). That readout has a floor even for an oracle table (ORA2 0.0127 on E2 T1).
- Replacement:
> "Q_r floor = E_H of the path-traced oracle table (LOW + e2 + e3 tables through the same readout), computed on each workload; arms are judged against this floor, not zero. D6 (defect STOP) applies only if pooled X1 ≈ L-DQ on Q_d2 but Q_r with that L2 differs from the oracle-table floor by more than the noise of both."

**C4. The positive control e2_o is not Lambert-sender-only.**
- e2_o covers every non-S1 sender, including glossy floor B (mat 0) and the peaked S2 sphere. `field_src` splits only by `mid == 1` (pt86a.py:175-180).
- Rename it "non-S1-sender control (Lambert + B + S2)". A pure Lambert-sender or floor-B-sender reference needs a new material-tagged split, which is new code plus a new render.
- The "Floor B sender: a = 1 ≈ a = 16" control therefore has no path-tagged reference. Restrict it to operator- and field-level bin adequacy, or declare the new split.

**C5. Q_d3 is a mismatched quantity.**
- e3_p covers events k ≥ 3, up to the field truncation, with the panel as last vertex. One explicit second panel flight gives exactly 3 events.
- Replacement: "compute explicit sweeps j = 2..4 with panel-last-vertex tagging to match e3_p's event range, or report the single sweep against e3_p as a labelled lower-order bound, not as a matched quantity."

**C6. Q_r mixes non-panel delivery errors into the panel attribution.**
- Specify the composition: "arm L₂ = arm's panel-mediated L₂ (e2_p estimate) + path-traced e2_o; + path-traced L₃₊". Otherwise the deployed flight's non-panel errors contaminate every arm.

**C7. Q_s must fix its reference.**
- auditB's operator and field norms are relative to M64 / Bm (the teacher). q is physical and has no operator form.
- Replacement: "Q_s reports (i) operator flux-L1 vs M64 and (ii) field rel-L1/energy/peak vs Bm = M64·s and, separately, vs q; never mixed."
- Add a q-noise control: redrawn MIS seeds. q is an MC histogram (16 points × 1,024 MIS), and its accuracy on grazing CK2 is unmeasured; 0.013 is the GM1 value.

**C8. L-DQ compute is undeclared.**
- [INF] The panel at mm cells is about 1.7 M cells; × 54 patches × e.g. 64 points ≈ 5.8 G visibility pairs.
- Declare ray counts and wall time, add a convergence check (1 mm vs 0.5 mm), and state that L-DQ exists only for exactly-2-event, point-light, planar-sender delivery. No higher-fidelity propagation level exists for e3_p.

**C9. W-b has no constraints stated.**
- Require the light to be inside the training rule `light_ok`, so the learned arms are in-distribution. Otherwise any learned-arm failure is confounded with extrapolation.
- For W-c, verify that a split e2_p/e3_p E2 field reference exists (WL86A Amendment 1 / WL92 t2). If not, it is a new render.

**C10. §7.6 distance disagreement.**
- [INF from auditA §3.2] The 0.15–0.29 m figure is the perpendicular panel–face gap. The footprint is fed by panel patches at y ∈ [−0.3, −0.1], with the receiver column at the face's +y end, so sender–receiver path lengths are much longer than the gap.
- The designer's 0.25 m is probably the gap, not the path length. Keep it as a CPU geometry check before freeze, but note this likely resolution.

---

## D. Hidden ray-tracing and table costs

- **G1 per-light cost is acknowledged but not quantified per light change.** Add [MEAS WL91:256]: C3URX1 uses 526,467 probe + 526,467 shadow rays per light, against 20,320 for the q table. G1 recomputes this on every light move.
- **G2's Γ_s is an explicit ray-traced relation,** about 650k rays, the same class as ECRO's state. Its cost belongs in G2's row explicitly: relation update 14.3 ms in GM1 (WL89:350, another scene).
- **ECRO's own per-frame cost includes path-traced LOW** (≤ 2 events). The "9×" is LOW + table structure, shared by H-0 (WL86B, WL87). This is already stated; keep it.
- **Smoke L-DQ:** see C8.
- **Pooled arms:** ≥ 64 realisations × arms × workloads. Declare the total ray budget at freeze.

---

## E. Progress-driven recommendation check

- **No primary is recommended.** The smoke is required by the brief and is marked NOT authorised. Acceptable.
- **Two fixes are needed.**
  1. The phrase "First discriminating question. Yes:" plus the batch-class suggestion reads as advocacy. Prefix §6 with "Specified for owner decision; not recommended or authorised".
  2. The proposed roadmap text should add "relative to the M64 teacher" after "matched-norm zero-operator level".

---

## F. Correct as stated (verified)

- **Code facts:** eval = f·cosθ_o (closure83_dr.py:7). The dirty test is rigid OBB plus the moved-box origin, box-only (ecro86.py:57-66).
- **Canon text** at Architecture.md:13 and :21 and roadmap:128 is quoted correctly.
- **CK2 reference data:** CKF_<v>.npz stores every `field_src(split=True)` key per half, so e2_p and e3_p are available (mat94.py:288-299). Box A = mask 'T' = rects 6–11, 54 patches, 1130–1183 (mat94.py:31,408).
- **ENCB** = c = I, a = M64, R = 64 (diag92.py:204-210). It has the same prefix as ORA / X_M12, so ENCB 0.0449 (true L₃₊) against ORA 0.030/0.033 (learned G) supports partial cancellation by the learned tail [INF].
- **One misattribution to fix:** "Prefix already measured at 0.0449" (rejected-alternatives table and §5.2) is the **R0 receiver-side** analogue, not G2's Γ_s prefix. Reword to "its R0-relation analogue measured 0.0449 single-draw / 0.027 pooled (E2 T1 C); the M64(s) × Γ_s prefix is unmeasured".
- **W defect:** `an94.W` (8 × 8) and `delivery_box` = 1130:1238 (boxes A + B) are confirmed against auditA/B. The 1/3 e2_p share of panel-mediated transport is arithmetically consistent: 0.39·0.58 / (0.39·0.58 + 0.61·0.755) ≈ 0.33. It is in WL94B loss-normalised target units, not reader-weighted; label it so.
- **"7.6–31× vs PT"** is the C3URX1 figure on the GM1 sharp tiers (WL91:288). Attribute it to C3URX1, in the PT role.
