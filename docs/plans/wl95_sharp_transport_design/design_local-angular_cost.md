# WL95 adversarial critique of the local-angular design (H1 LRS, H2 MEK)

**Lens:** cost, scaling, reuse and falsifiability.

**Scope.** This is a read-only review. I edited nothing and ran nothing on a GPU. I read the 8 evidence files completely. I checked these contracts in code:
- `experiments/ecro86/pt86.py:374-375, 378-402`
- `experiments/ecro86/ecro86.py:57-78`
- `experiments/fnto_compiled83/closure83_dr.py:1-20, 121, 164`
- `experiments/ecro86/scene86.py:1-30`
- `experiments/gm1_c3urx1_91/cand91.py:1-40`
- `docs/Architecture.md:17-38`
- `docs/RESEARCH_ROADMAP.md:15-16, 128`

**Labels:** [FACT] verified, [MEAS] cited measurement, [INF] inference, [HYP] untested.

---

## 0. Code-contract findings that affect the design

These are new relative to the design; each is checkable at the cited line.

**K1. Double cosine hazard.**
- [FACT] `closure83_dr.py:7` documents `eval(wi, wo) = f cos(theta_o)`.
- H1 eq. 2–3 forms L1_j(ω_d) = Σ f_ER·E and then multiplies by cos θ_d again in Φ.
- If `DrClosure83.eval` is used as written, the cosine is applied twice.
- The convention must be fixed in the protocol: which argument carries the cosine, and wi/wo order for reciprocity.

**K2. The memory estimate omits the dirty-test segments.**
- [FACT] The existing state stores `seg` as 6×float32 per ray plus int16 hit and uint8 bin, i.e. 27 B/ray (`pt86.py:374-375`). `dirty_rays` needs `seg` (`ecro86.py:57-66`).
- H1's "≈6 B/ray, 7 MB at D = 1,024" is therefore under-stated [INF]:
  - D = 1,024: 1.21 M × 27 B ≈ 33 MB.
  - Worst case, all 1,270 patches peaked at D = 1,024: ≈ 562 MB, not 125 MB.
  - D = 4,096: ≈ 2.2 GB.
  - α = 0.02 (D ×6, per the design's own α⁻² rule): ≈ 13 GB.
- Storing only the hit distance and regenerating origins from the deterministic grid would bring this to about 10 B/ray. That is an implementation choice that must be declared.

**K3. The current dirty rule only knows the two boxes.**
- [FACT] `ecro86.py:65` hard-codes `SID_BOXA if b == 'A' else SID_BOXB`. A peaked mover is not supported by the existing invalidation code.
- [INF] The design's sharp-mover update rule ("a sharp mover dirties all its own rays") has no code path and no test in the proposed workload.

**K4. The F̄ readout table repeats the L12 defect.**
- [FACT] The design defines F̄_m[b, d′] = (1/ΔΩ)∫_b f_ER dω: an incident-bin average that is uniform in solid angle with no cosine. That is exactly M64's incident averaging (`quad87.py:48-56`; auditA L12).
- The input I_p is cosine-carrying bin irradiance. At peaked x2 this re-introduces the ring-0 mismatch the design claims to remove.
- A cosine-weighted average, ∫_b f cos dω / ∫_b cos dω, is the consistent choice.

**K5. The atoms really exist** [FACT]. `pt86.py:386-398` computes, per (point, light), the direction wl, the shadowed irradiance e and the per-point frame, before the `bincount` binning. So H1's "no new rays beyond the existing 16-point shadow rays" is correct for point and spot lights.

**K6. ER is already a learned local rule** [FACT].
- The 69-float compiled state comes from the WL83 learned compiler (`closure83_dr.py:1-4`). Architecture.md and the roadmap name "learned ER" as the qualified local baseline (roadmap:15).
- H1's "no learned local factor exists" and H2's motivation ("keep a learned local layer in case the owner requires one") both mis-state this. The local layer is already learned. The open question in roadmap:128 concerns learned *global* transport.

---

## H1: Lobe-Resolved Sender relation (LRS)

### Fatal flaws (relative to the stated purpose)

**F1. The scope misses the dominant sharp family on its own primary workload.**
- H1's dense atom path covers only light → peaked → receiver (e2_p). Indirectly lit sharp senders (light → Lambert → panel → receiver, the e3_p family) still go through bins and R.
- [MEAS] WL94B and WL94 post hoc, on the CK2 panel face:
  - e2_p share of box e2: 0.580;
  - e3_p share of box e3: 0.755;
  - higher-order share of B: 61 %.
- [INF] e3_p is therefore about 0.46 of B and e2_p about 0.23. The panel-mediated ≥ 3-event family is about 2× the family H1 fixes.
- That remainder is routed through R_θ: the learned global-term class with documented compensation and LSPOT regression (WL86A Amendment 1; WL92B X_L3 0.71/0.82; WL94A panel-face G −0.44). Otherwise it goes through bin sweeps, where outgoing-bin projection at order ≥ 2 is measured as negligible only in GM1 at non-grazing incidence (WL90: P2plus ≤ 0.001).
- So H1 cannot claim to "address the CK2 floor" unless Q_d and Q_r also measure e3_p.

**F2. H1 is the explicit comparator class, which the standing human decision bars from becoming the method.**
- [FACT] roadmap:128 (human, 2026-11-13): ECRO is the learned-global base; EXP/C3UR are "immutable null/comparator evidence, not a replacement for learned global transport"; "Do not … promote an explicit candidate as the project goal."
- Roadmap:16: "implementing a full retracer does not complete GM1."
- H1's ≤ 2-event transport is entirely physical: an atom-resolved closure evaluation plus a geometry-only dense sender relation, with exact M64/EXP senders elsewhere. Its only learned element is R_θ, for ≥ 3 events, and the design itself labels R_θ's value over explicit sweeps unproven (WL87: explicit ≥ ECRO; WL86B: fill advantage bounded by J ≈ 3–5).
- The design labels this "owner decision" but still ranks H1 as primary. Without an owner reclassification, H1 is admissible as a *diagnostic/comparator* hypothesis, not as the WL95 architecture answer.

### Serious issues

**S1. Not distinguished from the nearest existing arm (C3URX1).**
- [FACT] `cand91.py:1-12`: C3URX1 is already a geometry-only sender-uniform relation (κ_s rays per sender patch × bin, selective = full bitwise) with exact per-ray once-scattered radiance Lx_1(y, v) at each ray's own origin and direction, i.e. sender-side placement plus continuous sender.
- H1 differs only in:
  - (a) peaked senders only;
  - (b) a deterministic per-frame direction grid at the 16 fixed source points, instead of random origins and directions;
  - (c) reuse of the source-pass visibility instead of 1 probe + 1 shadow ray per landed ray.
- [MEAS] WL91: pooled C3URX1 equals the oracle, but its single-draw variance fails held-out validity (1/3) and quality-matched utility (3.1× slower at the 0.06 tier).
- H1's distinctive claim is therefore *determinism and density beat random sender draws at matched rays* [HYP].
- The proposed baselines omit C3URX1 entirely. "Rejected alternative 3" cites the X1/RAYSEND variance failure, but that conflates receiver-side R0X1/RAYSEND with sender-side C3URX1.

**S2. "Deterministic" relocates variance into fixed-pattern aliasing rather than removing it.**
- One fixed grid phase at 16 fixed points is a single realisation. Its error is a fixed bias whose size depends on the phase.
- This is the same pathology as the deployed 512-ray draw in the bad tail [MEAS WL87:245-262] and the 16-point encoding bias [MEAS WL86A, WL93A g00010].
- The smoke has no grid-phase or rotation family, so a lucky or unlucky phase is indistinguishable from the mechanism.

**S3. Cost scaling is a 4-D geometry-only light-field table.**
- Size is N_peaked × P_s × D, with D ∝ α⁻². If P_s must grow to resolve CK2's 307× irradiance range or reflected texture edges (the design's own open Q5), this is the DDT problem class: WL84A withdrew DDT at α⁻⁴, about 38 GB.
- The ladder allows D·P_s up to 4,096 × 64 before "kill". That is about 19 M rays for 74 patches and about 330 M rays (≈ 9 GB at 27 B) for an all-peaked scene [INF].
- The kill threshold should be an absolute memory/time budget tied to the baselines, not a ladder endpoint.
- Mirrors (α → 0) need per-atom per-source tracing, i.e. a per-source retracer. H1 resolves every peaked sender regardless of reader importance [MEAS WL86B: 89 % of region C reads come from Lambert senders].

**S4. The per-source and per-edit economics probably lose ECRO's only measured learned advantage.**
- [MEAS WL87] Source-only update: ECRO 13–18 ms vs EXP 51–64 ms. Per-edit: PT 36 ms, ECRO 69–78, EXP 89–95, C3 430–580 ms.
- [MEAS WL86B/87] The roughly 5–9× gain over PT belongs to the LOW + table structure, which every table arm shares.
- H1 adds the dense build (≈ 50–200 ms incremental, extrapolated), the per-source closure evaluations and deposits, and the "EXP-like" non-peaked senders. Whether "exact bin M64/ER senders" means a cheap M64(s) matmul or EXP's 16 × 1,024-MIS q is not specified; with q, the per-source cost is at least EXP's.
- [INF] H1's practical comparators are therefore C3UR, C3URX1 and EXP at matched E_H, not PT. On cost it has no a-priori edge over C3UR, and its fidelity edge on E2 C is bounded (next point).
- The Neural Full Recompute and four-role baseline gate is not addressed. The design correctly makes no comparative claim, but criterion 10 is left unquantified.

**S5. E2 C has too little power to discriminate.**
- [MEAS] E2 T1 C:
  - C3/C3UR 0.023–0.026;
  - C2_src (M64 sender) 0.018–0.024 (WL93D);
  - oracle 0.0137;
  - E_H noise about 0.010–0.013 (WL88).
- The maximum gain H1 can show over C3UR on E2 C is therefore about 0.01, i.e. at the noise.
- Only the CK2 Q_d patch-level measure has SNR [MEAS box A rel L1 0.528 vs half-noise 0.058]. A power analysis must be predeclared.

**S6. The acceptance criteria are ill-posed.**
- **(d)** asks Q_r to improve on C2_srcS1 by more than the tolerance (≈ 0.017).
  - [MEAS] C2_srcS1 on CK2 untextured s8602 is already 0.019, with a noise bound of 0.0084. A > 0.017 improvement is arithmetically impossible on s8602.
  - Q_r is also confounded by the seed-specific learned G and receiver co-adaptation [MEAS WL93D: C2_src and C2_rcv move metrics in opposite directions]. Use path-traced L3+ ("single + true L₃₊", WL92 ladder) or explicit sweeps for ≥ 3 events.
- **(e)** "cost ≤ path-tracing the e2_p family to the same Q_d error" is not a quantity any renderer pays.
  - The e2_p field_src probe estimator's cost is not the REF-ER render cost.
  - Replace (e) with a WL91 rule (a)-style edit-to-first-frame comparison at a matched E_H tier against C3UR, C3URX1, EXP and REF-ER PT, with all build, dirty-test, deposit and readout costs on the clock.
- **(b)** "≤ 1.5× O3 error" is ill-conditioned when O3's error is near the reference noise. Use an absolute tolerance plus a noise floor.
- **O3 is ill-defined.** "D = 16,384, P_s = 256 plus pooled receiver relation" mixes two estimators.
  - Use the existing validated O3: pooled C3URX1 or R0_INF-style pooled relations with exact reads [MEAS WL91: within −0.0025…+0.0035 of the oracle], ported to the ECRO scene.
  - Then add a target-identity check between the dense sender relation and that pooled relation (as in WL88).
- **Q_d covers Lambert receivers only.** "Lambert exitance" is matched to e2_p only for Lambert receivers (box faces; auditA §3.1). Floor B and sphere S2 receivers need the receiver M64 with the true 4×16 W, or must be excluded.

**S7. The falsification statement mis-assigns outcomes.**
- "If O3q ≈ O3 … the lever is propagation; H1's local premise is falsified" does not kill H1, because H1 also *is* the propagation change.
- The decisive kill conditions are:
  1. Bin-sender sender-side relation (C3UR) at matched rays ≈ H1 on Q_d (including e3_p) and Q_r. Then the continuous sender adds nothing, and H1 collapses to an existing comparator.
  2. Random sender-side exact reads (C3URX1) at matched rays, with variance over draws, ≈ H1 with variance over grid phases. Then determinism adds nothing.
  3. H1's edit-to-first-frame at a matched tier is > C3UR's.
  4. Grid-phase spread exceeds the claimed gain.

**S8. The workload omits sharp movers.**
- The core claim is geometry-edit reuse, but no workload moves a peaked object. The panel is stationary in E2, E5 and CK2.
- The worst-case update path (a mover dirties all its relation rays plus every ray crossing its OBB) is never exercised.
- The WL88 slab-G mover is unqualified (Q5 3.76× vs 4×), so it cannot be used for a fidelity gate. Its cost and timing can still be measured without one.

**S9. Over-labelled claims.**
- "Exact for any ER state [FACT]" (criterion 3) is false. The D grid is α-specific, and a sharper state is under-resolved.
- "Neither can be removed by more training of a cᵀ alone": the optimisation component is by definition potentially removable by different training. Only the rank ceiling is structural [MEAS Stage B: learned ≈ 1.0 vs admissible non-negative rank 16 at 0.38].
- "Rank ≈ 90 … matches measured 46–59" cannot be tested inside a 64-dimensional space; treat it as [INF] only.

**S10. Direct/indirect contract risk.**
- [FACT] Architecture.md:21 forbids a direct/indirect *response* branch. H1 keeps one response rule, which complies.
- But identical panel outgoing radiance is propagated by different C operators depending on whether its incidence was direct or indirect. This is the root of F1, and it is the physically visible case of a mirror reflecting a lit wall.
- Declare it as a C-side partition by event order, with a measured fidelity asymmetry, not as a "quadrature choice".

### Genuinely supported by evidence

- [MEAS WL90] In GM1 non-grazing, the explicit-vs-oracle gap is the first-gather angular projection of L1 (97–102 %). 16 sub-bins close it.
- [MEAS WL87/88] Sender-side placement removes about 90 % of the E2 C deficit at equal rays, and the constructions agree on the target.
- [MEAS WL91] A continuous exact sender with pooled relations equals the oracle.
- [MEAS Stage B] No admissible rank-16 non-negative factorisation gets below about 0.36 field L1 for S1. The learned operator sits at the zero-operator level.
- [MEAS WL93D] S_geo < 1 % for S1, so a material-only local response is adequate.
- These justify a *continuous sender + sender-side relation* direction for the e2_p family. They do not establish the decomposition of the CK2 floor (auditA §3.3: open). They also say nothing about e3_p or grazing chains.

### Changes required for a valid falsifiable hypothesis

1. Re-scope H1 as a learning-free attribution and comparator hypothesis, with an explicit owner gate before it can be considered as the architecture (roadmap:128). Alternatively, specify a learned element with its own per-term target and a stated learning-unique utility.
2. Add C3URX1 (ported to the ECRO scene) and C3UR at matched rays and origins as mandatory arms. Add a grid-phase family (≥ 8 rotations or offsets) for H1 alongside C3URX1's draws, and report the mean, spread and worst case.
3. Measure Q_d and Q_r for the e3_p family, or declare that H1 does not address it, and remove "fixes the CK2 floor" from the hypothesis.
4. Fix K1 (cosine convention), K2 (bytes/ray including segments), K4 (cosine-weighted F̄) and K3 (a mover code path) in the protocol.
5. Replace acceptance (b), (d) and (e) as described in S6.
6. Use pooled C3URX1 / R0_INF + X1 as O3, with a target-identity check.
7. Use the true 4×16 W, with `an94.W` only as a bridge. Restrict Q_d to Lambert receivers or use matched non-Lambert exitance.
8. Predeclare the CK2 Q_d power analysis.
9. Add a sharp-mover timing arm.
10. Set absolute memory and time kill budgets: worst-case all-peaked memory and per-edit ms against the C3UR and EXP baselines.

### Verdict

**VIABLE_WITH_CHANGES**, as a learning-free diagnostic hypothesis that separates sender representation from propagation for the e2_p family.

It is **NOT_VIABLE as the primary WL95 architecture candidate** as written, because:
- it covers about 1/3 of the panel-mediated CK2 transport;
- it is the explicit-comparator class, which a standing human decision bars from promotion;
- its distinguishing claim versus C3URX1 is untested and absent from the baselines;
- its acceptance criteria (d) and (e) are unachievable or mismatched.

---

## H2: Mirror-Equivariant lobe Kernel (MEK)

### Fatal flaws

**F1. It is not a materially distinct hypothesis.**
- State, relation, deposits, resolvent, invalidation and propagation cost are identical to H1. Only the local evaluator changes, from exact ER to a lossy re-fit of ER.
- Within frozen-ER scope the best possible outcome is to equal ER. Any deviation is pure loss. The design itself concedes "dominated".
- It therefore does not satisfy the brief's "at least two materially distinct hypotheses".

**F2. It conflicts with GM0 and with prior negative evidence.**
- [FACT] roadmap:15 (GM0): "No renewed local-closure search without new failure evidence."
- The new failure is in ECRO's binned a cᵀ, not in ER. MEK is a renewed compact-lobe closure fit.
- [MEAS] The compact lobe-mixture fits in that line failed exactly where MEK would be stressed:
  - WL80C/80D: highlight width, and grazing passivity with the supremum at 90°;
  - WL81: R = 8 representation failure on a coated conductor.
- WL94D T (direct oracle supervision moved energy but not peak) is a further warning, as the design notes.

### Serious issues

- **Isotropy assumption.** "Isotropy about n built in" conflicts with ER's anisotropic GGX lobes (au, av: `closure83_dr.py:3-9`). Whether S1/S2 are anisotropic was not checked [HYP]. If they are, the equivariance prior is wrong by construction.
- **Unmatched kill norm.** "Held-out field L1 > 0.10 at peak < 0.8" names no norm or reference: M64-binned (Stage B norm) or continuous against the closure at atoms? Without that, the kill is ambiguous.
- **No independent cost or reuse content.** Its only claimed scope gains, non-analytic materials and patch-aggregated effective lobes, are outside every available reference and workload.

### Genuinely supported

- [MEAS WL93D] The S1 geometry-common local error (S_geo < 1 %) means a material-only kernel is enough in principle.
- [MEAS Stage B] Fixed-basis rank is the structural obstacle for S1.
- Neither result favours MEK over exact ER.

### Changes required

- Drop MEK as a WL95 candidate. Or reformulate it as a distinct hypothesis whose learning has a scope gain that can be measured on existing references: for example, a material class without a closed form, with a physical reference that exists. As specified, no such workload exists.

### Verdict

**NOT_VIABLE.** It is dominated by construction, it is not materially distinct from H1, it conflicts with GM0, and it has prior negative evidence.

---

## Assessment of the design's overall conclusion

- **The conclusion is not supported.** The design concludes MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION, but:
  - H2 is not a materially distinct candidate;
  - H1, as an architecture, falls outside the learned-global decision frame (roadmap:128);
  - the one evidence gap that would justify any local-representation design, the CK2 floor decomposition C1–C10 (auditA §3.3), is still open.
- **The evidence supports ARCHITECTURE_SELECTION_PREMATURE.** H1's O-hierarchy, corrected per the H1 changes above, is a useful **learning-free attribution smoke**. With the C3URX1 and C3UR arms, grid-phase controls, e3_p coverage, true W, a path-traced L3+ for Q_r and a WL91-style matched-tier cost rule, it would decide whether the remaining sharp error is:
  - local sender representation (O3q vs O3);
  - propagation placement or density (O1q → O3q; H1 vs C3URX1 vs C3UR);
  - or higher-order and global (e3_p, R).
- **Hidden cost.** The design is honest that it moves the sharp first-scatter cost into explicit geometry-only ray relations; nothing is hidden under a learned name.
  - The unaccounted costs are the 27 B/ray segment storage, the worst-case α⁻²·P_s growth, per-source EXP-like q (if that reading is intended), and sharp-mover rebuilds.
  - None of these is benchmarked against the comparators that actually share the 5× table advantage: C3UR, C3URX1 and EXP.

## Sources

**Evidence files:**
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\auditA.md`
- `...\scratchpad\wl95\auditB.md`
- `...\scratchpad\wl95\read_code.md`
- `...\scratchpad\wl95\read_canon.md`
- `...\scratchpad\wl95\read_wl86_88.md`
- `...\scratchpad\wl95\read_wl89_92.md`
- `...\scratchpad\wl95\read_wl93.md`
- `...\scratchpad\wl95\read_wl94.md`

All eight are in the same folder, `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\`.

**Code and documents checked:**
- `c:\Projects\ReST-GI\experiments\ecro86\pt86.py:374-375, 386-402`
- `c:\Projects\ReST-GI\experiments\ecro86\ecro86.py:57-78` (line 65 box-only)
- `c:\Projects\ReST-GI\experiments\fnto_compiled83\closure83_dr.py:1-20` (line 7 `eval = f cos θ_o`), `121`, `164`
- `c:\Projects\ReST-GI\experiments\ecro86\scene86.py:1-30`
- `c:\Projects\ReST-GI\experiments\gm1_c3urx1_91\cand91.py:1-40`
- `c:\Projects\ReST-GI\docs\Architecture.md:17-38`
- `c:\Projects\ReST-GI\docs\RESEARCH_ROADMAP.md:15-16, 128`

**Derived arithmetic (bytes, ray counts, family shares) is [INF] from these sources.** The e2_p / e3_p panel-face shares come from WL94B / WL94 as summarised in `read_wl94.md`.