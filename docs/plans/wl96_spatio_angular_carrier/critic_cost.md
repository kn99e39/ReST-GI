# WL96 adversarial critique: cost, scaling, dynamic reuse and the explicit comparator

**Scope.** Read-only; no repository file edited, nothing trained, no GPU or renderer job. One CPU numpy check was added (the dirty-pair fraction, §1.3): `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\critic_cost\dirty.py`. It imports `experiments/ecro86/scene86.py` and the CK2 scene JSON read-only.

**Labels.** [FACT] code or doc · [MEAS] repository measurement · [MEAS-C] my scratch CPU number (not evidence) · [INF] inference · [HYP] untested.

## 0. Bottom line

1. **There are eight named candidates but only two kinds of thing.**
   - Every "learned" candidate that survives (d1 SIC-R Λ_θ, d2 SRR-LB Λ_θ, d3 LSC-R and ESC-R R_θ, d4 LIR Λ_θ) has the same learned component: a non-negative, low-rank, geometry-conditioned resolvent for ≥ 3-event broad transport, fed by an exact 2-event field. That is **one learned hypothesis written four times**. The differences between the candidates are their explicit sharp carriers.
   - The learned sharp candidates (d1 E-L+, d2 LSTO/C1, and the d3 / d4 starting points) are falsified or dominated by learning-free capacity bounds that the designers measured themselves.
2. **The shared learned resolvent is decorative on measured, same-scene evidence.** Two reasons:
   - **The cost ceiling is too low (Amdahl argument, [INF] from [MEAS]).** Assume a perfect learned resolvent that costs nothing and removes the whole explicit sweep stage. The lifecycle speed-up still stays below the project's own 1.25× utility rule (WL87; WL91 rule a) on every end-to-end metric that includes a frame (§2.2).
   - **The accuracy headroom is about zero (cross-workload).** WL90 found that projection at orders ≥ 2 costs +0.0009 / +0.0007 / −0.0004. WL87 found learned fidelity utility 0 better / 3 worse / 36 equal. On E2 the explicit C3UR (0.026) beats exact-prefix + learned G (0.030 / 0.033).
3. **The "cross-scene" caveat that all designers (and WL95) attach to "sweeps 11.6 ms vs generator 12.5 ms" overstates the uncertainty.**
   - [MEAS, WL87:111, 175, 281] On scene86, same machine: explicit 4-order recurrence 10–11 ms vs ECRO operator 11 ms.
   - [MEAS, WL89:353; WL91 table] The receiver-gather R0 sweeps take 5.6–6.1 ms.
   - [FACT] The GM1 workloads use the same room layout (N = 1,270, a 650,240-ray relation; WL91 scaling table) with a slab added. These are near-same-regime numbers, not cross-scene ones.
4. **The only measured learned advantage (source-only update, 3.5–3.9×) is confounded by implementation overhead.**
   - [MEAS WL87:175, 281] EXP's source-only cost of 51–64 ms is dominated by the MIS q table: 42–54 ms wall, but **2 ms GPU**.
   - [FACT WL87:331] WL87 itself flags this as untested.
   - All four designs feed the broad part with M64·s (ECRO's own source encoding), so the explicit source-only path becomes about s (5 ms) + M64 + 6–11 ms of sweeps ≈ ECRO's 13–18 ms [INF from MEAS components].
5. **None of the designers' smokes can discriminate a learned utility with useful power.**
   - Their cost rules are component-level.
   - Their accuracy rules use the 2 × 4,096-spp render tolerance (0.0167), which WL95A showed cannot even resolve pooled M64's +0.013 bias.
   - None uses the existing qualified **moving-sharp** workload (WL89 W1 / W2). That is where the explicit carriers' invalidation costs bite and where WL90 / 91 showed the first-gather projection is visible in images.
6. **Recommended WL96 judgment: NO_VIABLE_LEARNED_MECHANISM** (scoped and provisional; §6), with one bounded learning-free smoke that would confirm or overturn it (§7).
   - The explicit carriers (DQ / CIS-ER, photon relay, ESC-R table) are viable **comparator / oracle** classes that need discrimination. They are not learned methods.

---

## 1. Cross-cutting cost findings (number re-checks)

### 1.1 Arithmetic the designers got right [MEAS-C re-computation]

| claim | check |
|---|---|
| d2 C1 factors 672 × 256 × 1,270 × 3 ≈ 6.6e8 floats, 2.6 GB | ✓ (655 M floats; 2.62 GB at fp32) |
| d2 DQ 672 × 20,320 ≈ 13.7 M pairs | ✓ |
| d4 (672 + 512) × 20,320 ≈ 24 M pairs; 12 M bits ≈ 1.5 MB; 20–37× ECRO's relation | ✓ |
| d3 ESC-R Γ: 168 × 1,024 × 4 = 688,128 rays; α = 0.01 → 11–44 M rays | ✓ |
| d1 photon counts 672 × 64 = 43 k; 2,688 × 64 = 172 k; 530 draws × 650 k ≈ 345 M | ✓ |
| d4 LIR decoders 74 × 16 × 64 × 3 × 32 ≈ 7.3 M floats | ✓ |
| d1 Λ memory 43 MB × 64 / 24 ≈ 115 MB | ✓ |

### 1.2 Numbers that are wrong, inflated or hidden

1. **Per-light comparisons against pooled X1 are against a straw man.**
   - Pooled X1 (64 draws) is a diagnostic, not a deployable comparator. The deployed explicit comparator is single-draw C3URX1: 0.53 M + 0.53 M rays per light, 11.5 ms [MEAS WL91:256].
   - d3 makes it worse: it counts 64 × 650 k = 41.6 M *geometry-only relation rays* as per-light cost, which inflates the photon carrier's advantage about 30×. The per-light part of pooled X1 is about 1.28 M probe + shadow rays. The relations are geometry-only and could be cached (64 × 17.6 MB ≈ 1.1 GB).
   - d1's "7–30× fewer rays" recomputes to 6.6–20× vs pooled X1, and to 5–16× vs single-draw C3URX1.
2. **ESC-R memory is understated.**
   - d3 sizes Γ at 8 B per ray. Selective invalidation needs the segment, and ECRO stores 27 B per ray [MEAS WL86].
   - At r = 4 that gives 18.6 MB per facet; at r = 16, 74 MB per facet. **The r = 16 setting fails d3's own FB3 (≤ 32 MB per facet).**
3. **SIC-R and LSC-R photon counts are scored on box A only.**
   - 20 % (CK2) / 46 % of the photons land there [MEAS-toy d3]. The per-patch error on every other receiver (floor B, walls, S2) is unreported.
   - Fixed per-receiver accuracy needs M ∝ 1 / (ε² · f_land · A_receiver). So d1's claim that relay cost is "independent of the receiver count" [INF d1 §3G] is false for any accuracy target defined per receiver.
4. **d1's node-interpolated light visibility (195–725 nodes) is a hidden bias source.** A 10 cm node grid blurs shadow edges on the sharp sender by about 10 cm. The toy has no occluder on the panel, so this bias was never exercised.
5. **The camera-continuation and readout costs are per frame, which the designers did not cost.**
   - d2 SRR-LB and d3 LSC-R add one ER-sampled bounce plus L shadow rays per pixel path whose second vertex is sharp. That is explicit path tracing inside the LOW + lookup renderer, and it erodes the measured 5.5–9× per-frame table gain (WL86 / WL87) in sharp-dominated views.
   - d4 CIS-ER reads L_s at the exact (x2, ω). With extended incidence that is 256 directions × L atoms per sharp camera vertex. At 512², 64 spp and a 20 % sharp share, that is ~8.6 × 10⁸ ER evaluations per frame [INF arithmetic].
6. **Culling or "pruning" the DQ pairs is not free.** d4 measured cull 1e-3 → 0.031 and cull 1e-2 → 0.145 (biased low). d2's "0.7 M ER evals with lobe-cone pruning" therefore carries an unbudgeted bias.

### 1.3 DQ / CIS-ER / SRR-LB per-edit cost is badly underestimated [MEAS-C, `dirty.py`]

**Setup.** CK2 geometry from `scene86`. S1 at 5 cm cells (672); 4 × 4 receiver points on every rectangular patch except the panel (19,136 points); segment–OBB test against box A.

| quantity | value |
|---|---|
| S1 → receiver pairs | 12.86 M |
| front-facing pairs | 9.09 M |
| front-facing pairs blocked by box A at the CK2 pose | **47 %** |
| dirty pairs (old ∪ new pose) for a modest 0.3 m y-shift of box A | **60 %** (≈ 5.5 M pair re-tests) |
| pairs landing on box A itself | 1.8 % |

**What this means [INF].**
- d2 estimated "≤ about 1–2 M rays" per edit; d4 left the dirty fraction unquantified. The dirty set is dominated by *pairs crossing the swept volume*, not by pairs with moved receivers.
- At WL91's measured throughput (about 1.05 M rays plus ER evaluation in 11.5 ms), 5.5 M re-tests is about 60 ms. That puts the per-edit cost at about 136–140 ms, roughly 1.8× ECRO's 76–78 ms.
- So d2 F2e (≤ 78 ms), d4 FA5 (≤ ECRO + 25 %) and d3 FC (≤ ECRO + 10 ms) are **predicted to fail on the designers' own favourable workload**, where the sharp sender is static.
- On a rotating sharp slab (WL89 W1) every pair of the mover is invalid: about 13.6 M pair rays per edit for an S1-sized facet (d4's own figure).

### 1.4 Scaling worst cases [INF; per light / per edit / per geometry kept separate]

**How each carrier class scales**

| carrier class | per geometry | per light | per edit | α → 0 | many sharp surfaces F_s |
|---|---|---|---|---|---|
| DQ / CIS-ER / SRR-LB (geometry-only pair cache) | O(A_sharp · A_rcv) pairs; **cells ∝ α⁻², pairs ∝ α⁻⁴** | ER evals O(L · pairs) | dirty pairs (60 % CK2 box; 100 % sharp mover) | α = 0.01 → ≈ 81× pairs (≈ 1 G); image-source only for planar | linear in sharp area |
| photon relay (SIC-R, LSC-R) | small (continuations) | O(L · F_s · M) rays, M ∝ 1 / (ε² f_land A_rcv) | re-shoot | robust | linear |
| ESC-R table | O(cells · bins(α⁻²) · r) rays | shadow rays + O(cells · bins · P · D) ER evals | Γ crossing the OBB (likely large near the panel) | hard fail | linear in memory |
| X1 / C3URX1 | relation O(N · K) | 1.05 M per light | relation 14.5 ms | variance grows | capped by N · K |
| ECRO and the learned resolvent | relation O(N · K) + **generator dense O(N²) per edit** | 20 k shadow rays | 76–78 ms | n/a (binned) | — |

**Example:** F_s = 20 S1-sized facets and L = 8 lights.
- Photons: 13–51 M rays per lighting change.
- ESC-R: Γ 14–56 M rays and 0.4–1.5 GB.
- DQ: about 180 M geometry-only pairs.
- ECRO's relation stays at 650 k, but it cannot represent the lobe.

[INF] That regime, many sharp surfaces and many lights over static geometry, is the only place a learned compact sharp operator would pay off. Yet the designers' own rank evidence (d1: rank 148 / 381 / 614 for 90 / 99 / 99.9 %; d2: r ≥ 256 per sub-cell; d3: 492–579 at 99 % with 1,024 bins) says it cannot be compact, and it grows as α⁻². **Learning fails where explicit is expensive and is redundant where explicit is cheap.**

**Learned-resolvent scaling [INF].**
- The per-edit generator is a dense N² form-factor build plus 3 message rounds (`ecro86.py:82-87`). Explicit sweeps are O(J · N · K) (WL91: 21 → 28 ms from N 1,270 → 5,080). d4's "possibly favourable asymptotically" ignores the generator.
- Both Λ and the sweeps act on the *summed* field, so neither gains with more lights.
- At this scale every timing is overhead-dominated (WL86 GPU kernels: state 0.06 ms, source 0.02 ms). A FLOP advantage (Λ apply about 6 M MAC vs 4 sweeps about 70 M MAC [MEAS-C arithmetic]) is invisible in milliseconds.

---

## 2. Learned utility: decorative or credible?

### 2.1 Measured anchors (scoped)

| item | value | source | regime |
|---|---|---|---|
| ECRO per edit | 76–78 ms, of which operator 11 ms | [MEAS WL86:214] | scene86 |
| explicit 4-order recurrence | 10–11 ms | [MEAS WL87:175] | scene86, same machine |
| R0 sweeps | 5.6–6.1 ms | [MEAS WL89 / WL91] | GM1 room, N 1,270 |
| C3UR sweeps | 10.6–11.6 ms | [MEAS WL89 / WL91] | GM1 room, N 1,270 |
| source-only update | ECRO 13–18 ms vs EXP 51–64 ms (q 42–54 ms wall, 2 ms GPU) | [MEAS WL87:175, 281] | scene86 |
| fidelity | learned 0 better / 3 worse / 36 equal | [MEAS WL87:346] | scene86 |
| projection at orders ≥ 2 | P2plus ≤ 0.0009 | [MEAS WL90:176] | GM1 |

### 2.2 Amdahl ceiling: perfect zero-cost resolvent replacing the measured sweeps [INF from MEAS WL91 C3URX1 lifecycle]

| lifecycle metric | measured total | sweep stage | best possible ratio |
|---|---|---|---|
| edit to frame | 124.7 ms | 10.6 ms | **1.09×** |
| edit without render | 69.4 ms | 10.6 ms | 1.18× |
| source to frame | 75.8 ms | 10.6 ms | **1.16×** |
| source without render | 21.4 ms | 10.6 ms | 1.98× (only this metric clears 1.25×; no user-visible update omits the frame) |

- ECRO equivalent: 78 / (78 − 11) ≈ 1.16×.
- With a real generator of about 11 ms per edit (ECRO-class) instead of 5.6–11.6 ms of sweeps, the **per-edit ratio is ≤ 1, a guaranteed loss**.
- Every candidate adds a sharp prefix to the denominator, so all these ratios get *worse*.

### 2.3 Conclusion on utility [INF]

- The designers' "small utility" means the learned part is **decorative**: it is placed to satisfy roadmap:128's "ECRO-lineage" wording, not because it carries information or saves meaningful work. All four designers concede it expresses nothing explicit transport cannot (d1 §D, d2 §D, d3 §D, d4 §D).
- **Canon conflict.** The roadmap requires a hypothesis that "addresses the measured learned-flight sharp deficit … without sacrificing source-independent reuse". The surviving candidates address that deficit by removing learning from it, so the learned part addresses nothing that is measured as failing.

---

## 3. Canon compliance and owner gates

| issue | status |
|---|---|
| `Architecture.md:13` (EXP / C3UR are comparators; X1 findings do not justify explicit GI) | Every surviving sharp carrier is explicit GI: G1 class with better estimators. **Gate A, beyond WL96's authority**, for all of SIC-R, SRR-LB, LSC-R, ESC-R and LIR. CIS-ER honestly declares itself a comparator (FA6) |
| roadmap:128 (one falsifiable ECRO-lineage hypothesis; no sacrifice of source-independent reuse) | Photon carriers (SIC-R, LSC-R) are **source-dependent per-light transport ray work** and fail the clause literally. DQ, CIS-ER and ESC-R keep a geometry-only relation with per-light ER evaluation only, which is the closer fit. The learned resolvent is ECRO-lineage in form only |
| `Architecture.md:21` (no direct / indirect response branch) | All four split the *carrier* by delta vs extended incidence (DQ atoms vs relays; photons vs continuations; point atoms vs sub-bin incidence; relay vs bin-averaged e3_p), with the same ER response. **Gate B (owner)**: arguably compliant, but it is a direct / indirect transport branch, and the peaked-class Π > 8 threshold is a misclassification risk |
| FNTO v0.1:86 ("no exhaustive explicit per-surface path tracer hidden under a neural name") | Camera continuation (d2, d3) and relay chains are explicit path tracing. They are declared, so not "hidden", but presenting the package as a learned method would violate the spirit, because the information-bearing parts are explicit and the learned part is decorative |
| four baseline roles | No designer claims Neural Full Recompute (correctly UNMEASURED). All "economics" are internal (ECRO, C3URX1) or PT re-render and must be labelled so. None of the smokes measures an external role, which is fine for a design smoke |
| research objective (selective recompute) | Every learned resolvent is regenerated in full on each edit (ECRO generator). That is rule reuse with no selective learned-state preservation, which does not advance the dynamic-reuse objective |

---

## 4. Per-candidate assessment

### d1 SIC-R (node incident carrier + sender-directed photon relay + learned Λ_θ)

- **Explicit GI under another name?** Yes, for e2_p and e3_p (photon relay = light-side lobe photons; G1 / ELCF class).
- **Avoided vs X1.** Receiver-anchored heavy-tailed gathering; 5–16× fewer per-light rays than single-draw C3URX1 at far lower per-draw error (CK2 0.068 at 43 k photons vs 0.62).
- **Introduced.** O(L · F_s · M) relay rays per light, which is 3–9× ECRO's source rays and loses ECRO's source-only property for peaked senders, as d1 admits. Also relay segment caches per light (1.2–4.6 MB × L), node-visibility interpolation bias, receiver binning, and a second relay for e3_p.
- **Fatal (as a learned method).** Λ_θ is decorative (§2).
- **Serious.**
  - Ideal-distribution sampling, not ER's VNDF.
  - Box-A-only scoring.
  - No occlusion on the panel.
  - S-F4 is loosened (≤ 3× ECRO source-only).
  - S-F5's target is ambiguous: if Λ is fit to explicit-sweep outputs, its accuracy headroom is ≤ 0 by construction.
- **Supported.** The bias is a carrier property and the variance an estimator property [MEAS-96, consistent with d3 and d4]. The deterministic e2_p oracle matches PT at 1.015 / 0.068 (one light).
- **Required changes.**
  - Real sampler.
  - Per-receiver-family error over all receivers.
  - Occluded node visibility.
  - Lifecycle-level cost rule.
  - Moving-sharp workload.
  - Fix S-F5's target to held-out PT tallies.
- **Verdict.** Learned method: NOT VIABLE (decorative). Comparator: VIABLE; the strongest α-robust explicit carrier for static geometry.

### d1 E-L+ (learned material-level sharp operator, exact-outgoing readout)

- **Explicit?** No; it is ECRO-faithful.
- **Fatal.**
  - The ECRO interface keeps 64-bin incident s, and the BIC64 error is 0.24–0.51 regardless of rank.
  - Signed rank 64 / 128 / 256 gives 0.51 / 0.28 / 0.17.
- **Caveat.** "A non-negative fit cannot be better in that norm" is not a theorem. SVD is Frobenius-optimal, not delivery-L1-optimal. The NMF run is required before writing FALSIFIED.
- **Verdict.** Learned method: NOT VIABLE (pending a cheap NMF confirmation). Comparator: none (dominated by M64 / q).

### d2 LSTO / C1 (lobe state + learned transfer)

- **Explicit?** No; it is learned ray casting (WL52 class) or a low-rank PRT transfer.
- **Fatal.**
  - 2.6 GB of generated factors per geometry at rank 256.
  - Per light, Σ_c E_cl · T_c is about 6.6e8 MAC, more than DQ's 6 M ER evaluations.
  - Dominated in memory, per-light compute and accuracy.
- **Verdict.** NOT VIABLE on both counts.

### d2 SRR-LB / C2 (DQ atoms + relay continuations + camera continuation + learned Λ_θ)

- **Explicit?** Yes, apart from Λ.
- **Avoided.** Per-light transport rays: only about 21 k shadow rays plus ER evaluations. This is genuinely the most source-independent explicit carrier.
- **Introduced.** 6–9 M geometry-only pair rays; **per-edit dirty pairs about 5.5 M on the CK2 box edit [MEAS-C], not the 1–2 M claimed**; α⁻⁴ pairs; S2 about 5e7 rays; per-pixel camera continuation.
- **Fatal (learned).** Λ is decorative.
- **Serious.** F2e is predicted to fail; the moving-sharp invalidation cost is not tested.
- **Supported.** DQ is an excellent deterministic **oracle** (R6: 0.026 at 5 cm; 0.0009 at 1.25 cm).
- **Verdict.** Learned method: NOT VIABLE. Comparator: oracle yes; runtime comparator only for static sharp senders at moderate α.

### d3 LSC-R (light-side photons + lobe continuation + R_θ)

- Same class and verdict as SIC-R; it adds a credible e3_p continuation [HYP].
- **Serious.** The per-light ray comparison is inflated about 30× (§1.2). FL1's tolerance (0.0167) has low power.
- **Verdict.** Learned method: NOT VIABLE. Comparator: VIABLE.

### d3 ESC-R (10 cm × 1,024-bin sender table + sender-anchored Γ + R_θ)

- **Explicit?** Yes (directional radiosity).
- **Avoided.** All per-light transport rays (best per-light profile).
- **Introduced.** 0.7–2.8 M rays per facet; 18.6–74 MB per facet at realistic storage (fails FB3 at r = 16); α⁻² bins (hard fail at α ≈ 0.01); Γ dirty fraction near the panel unmeasured but likely large (§1.3 geometry).
- **Verdict.** Learned method: NOT VIABLE. Comparator: VIABLE only above a measured α_min and for static sharp senders.

### d4 CIS-ER (continuous incident state + exact ER + D2 deterministic delivery)

- Honestly declared learning-free (FA6). Same cost profile as DQ, with the §1.3 dirty-pair problem and α⁻⁴ scaling.
- Its per-pixel exact readout with extended incidence is an unbudgeted per-frame cost (§1.2 item 5).
- **Supported.** It sets the information floor: CP4 0.008. FA4 (deployable single configuration) is the best variance rule of the four.
- **Verdict.** Comparator / oracle: VIABLE. Not a method.

### d4 LIR (learned incidence resolvent)

- The best-specified ownership: it maps ECRO's event partition onto K / C and owns only broad-last ≥ 3-event incidence. It is otherwise the same learned component, so decorative (§2).
- FB2 needs a new incident tally plus 96 × 4 path-traced lightings, which is the most expensive precondition proposed.
- **Verdict.** Learned method: NOT VIABLE on the current evidence (decorative), but it is the right *form* if the §7 smoke unexpectedly finds headroom.

---

## 5. The designers' proposed smokes

| | bounded | matched workloads | independent validation | predeclared failure | can discriminate learned utility |
|---|---|---|---|---|---|
| d1 | moderate: 5 stages, 16 lights, 2 new PT references | CK2 only (box edit, static sharp) | 2 PT lights; S-F3 scored against its **own** DQ oracle (shared ER port and geometry → shared-mode error) | good (S-F0 to S-F6) | weak: S-F5 target ambiguous; 2× rule is component-level |
| d2 | yes (about 1 GPU-h) | CK2 + W-b (+ W-c if verified) | PT on CK2 and W-b ✓ | good; F2e predicted to fail (§1.3) | weak: F2c has no threshold on lifecycle saving |
| d3 | moderate | CK2, W-b, E2 C, CK2 T0 / T1 ✓ | PT ✓ | good (FA / FB / FC / FI / FL1) | weak: the 0.0167 tolerance cannot resolve WL95A's 0.013 bias; "sweeps ≤ generator time" is not lifecycle |
| d4 | heavy: Stage 5(c) new tallies × 384 lightings | V1–V4 including S2 ✓ | PT ✓ | best (FA4 deployable configuration, FA6) | FB3 OK in form, but gated behind an expensive tally |

**Gaps common to all four**
- WL89 W1 / W2 (a rotating S1 slab, qualified, with family-tagged references) is ignored. That is the case where pair and Γ caches invalidate fully and where the first-gather projection is visible in images (WL90 / 91 R_mirror).
- Explicit and learned costs are not compared at a matched implementation: same input s + M64, same framework, GPU-kernel time and wall time reported separately (WL87's overhead confound).
- CK2 box-A delivery against a home-built deterministic oracle risks tuning to the CK2 footprint.

---

## 6. Recommended overall WL96 judgment

**NO_VIABLE_LEARNED_MECHANISM** (scoped: scene86 / GM1 room family, point lights, rigid edits, α ≈ 0.03–0.18; provisional pending §7).

1. **Learned sharp carriers** (E-L+, C1, the d3 / d4 starting points) are excluded by designer-measured capacity bounds (rank ∝ α⁻²; signed rank 256 still 0.17) and dominated by learning-free DQ in memory and per-light compute.
2. **The one remaining learned element** (the shared broad resolvent) is falsifiable but predicted decorative:
   - its lifecycle ceiling is < 1.25× even at zero cost, and per edit it is ≤ 1×;
   - its accuracy headroom is about 0.001;
   - the one measured learned advantage is overhead-confounded.
3. **The explicit carriers fix the sharp bias** and are distinct comparator / oracle choices (MULTIPLE require discrimination). Adopting any of them inside "the method" needs **Gate A / Gate B owner decisions** outside WL96's authorisation.
4. **Why not EVIDENCE_INSUFFICIENT.** The decisive quantities (sweep cost, generator cost, lifecycle totals, ≥ 2-order projection headroom) are already measured on the same room. Only a confirmation of the ceiling on an exact prefix is missing.

---

## 7. The single bounded smoke that most reduces the remaining uncertainty

**Learned-resolvent utility ceiling (learning-free, no training).**
- **Batch class.** [INF] AGENTS.md priority (2), dependent follow-up: e.g. WL96A. The owner decides.
- **Comparisons.** Internal only. Neural Full Recompute is UNMEASURED.

**Workloads**
- CK2 "none", T0 / T1 box edit (existing CKF tallies).
- WL89 W1 T0 / T1 (rotating S1 slab) with R_mirror and E2 region C as sharp-reader regions. Verify before the freeze that its family-tagged references give an e2 / e3 split; if they do not, declare the smoke CK2-only.

**Prefix**
- The **oracle** 2-event field from the path-traced tallies (e2 tables) feeds every arm.
- This decouples the learned question from the choice of carrier.

**Arms on the ≥ 3-event tail, restricted to the broad-last family**

| arm | contents | role |
|---|---|---|
| T-PT | PT e3 | floor |
| T-EX | explicit J = 4 sweeps on the R0 and C3UR relations, from the same oracle e2, with M64 local operators | explicit comparator |
| T-Λ* | per-geometry in-sample non-negative rank-ρ ∈ {24, 64} fit to PT e3, fitted on replica half A and scored on half B | non-deployable upper bound; a pass is weak evidence, a fail is decisive |
| T-ECRO | frozen OWN global term | descriptive |

The broad-last restriction exists so that Λ* cannot win by absorbing sharp-sender e3_p structure, which would be compensation.

**Measurements**
- **Accuracy.** Field-level per-region rel L1 (more sensitive than ROI) and ROI E_H.
- **Cost.**
  - Same GPU (LabServer63), wall time and GPU-kernel time, with a full-scale utilisation probe per the standing rule.
  - Measured stages: sweeps, Λ apply, and ECRO `factors`, at N = 1,270 / 2,802 / 5,080 (WL91 scaling geometry).
  - Embedded in the measured C3URX1 / ECRO lifecycle: U_max = lifecycle / (lifecycle − t_sweeps + t_Λ [+ t_generator per edit]) for edit-to-frame and source-to-frame.

**Decision mapping (frozen before results)**
- **Confirms NO_VIABLE_LEARNED_MECHANISM** when all of the following hold:
  - U_max < 1.25 on edit-to-frame and source-to-frame at every N;
  - H_acc = err(T-EX) − err(T-Λ*) ≤ max(2 SE, n0 / 2) in every region.

  The explicit carriers then go to the comparator role and the next step is owner Gate A.
- **ONE_FALSIFIABLE_CANDIDATE (learned resolvent; training needs separate authorisation)** when H_acc exceeds the bound in a predeclared sharp-reader region on both workloads.
- **COST_ONLY candidate** (a scaling hypothesis needing a training smoke at large N) when U_max ≥ 1.25 at the largest N while H_acc is within the bound.
- **STOP** if the tallies do not sum to the reference (≤ 1e-6), or the explicit sweeps fail to reproduce WL87 / WL89 values.

**Budget [INF].**
- No new PT renders if the W1 split exists; otherwise 2 × 4,096 spp for W1.
- NMF fits: CPU minutes.
- Timing: GPU minutes.
- Renders: about 20 at about 10 s each.
- Under 1 GPU-hour.

**Optional separable Stage B (comparator selection, not architecture).** Compare DQ g4 vs photon relay (M 40 k / 160 k) vs ESC-R r4 on the same two workloads, scored against **PT e2_p** (not the in-house oracle), with dirty-cache counts and milliseconds per edit (the box edit and the slab rotation). This resolves which explicit carrier is the comparator for static vs moving sharp senders.
