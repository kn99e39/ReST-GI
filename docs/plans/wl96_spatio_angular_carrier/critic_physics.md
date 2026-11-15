# WL96 adversarial critique: physics, representation, training and evidence consistency

**What I did.** Read-only. I read all four designer reports in full, plus WL95A, WL95 §1–3 and §5–8, `results/wl96_calc/ring_alloc.json`, `docs/Architecture.md:1-60`, `closure82.py`, `closure83_dr.py`, `pt86a.field_src` and the designers' scratch scripts. I edited no repository file and ran no GPU or renderer job. I ran one CPU re-check of d1's rank calculation; script and output are in `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\critic_phys\` (`rankw.py`, `rankw.log`, `rankw.json`).

**Labels.** [FACT] code or doc · [MEAS] repository measurement · [MEAS-D] a designer's scratch number · [MEAS-C] my scratch re-check · [INF] inference · [HYP] untested.

## 0. Bottom line

1. **The four designers converge on one structure. They are not four architectures.** [INF]
   - An explicit, learning-free sharp first order: node carrier + photon relay (d1), DQ + relays (d2), light-side photons or a fine table (d3), CIS + DQ (d4). These are WL95 G1-class estimator variants.
   - Plus one ECRO-type, geometry-conditioned, non-negative, low-rank **broad ≥3-event resolvent** (d1 Λ_θ, d2 Λ_θ, d3 R_θ, d4 LIR). This is WL95 G2 with Λ now specified.
   - **No candidate carries sharp structure in a learned component.**
2. **The designers' "no learned sharp carrier" conclusion is probably right, but the stated proof is wrong.** [MEAS-C, INF]
   - All rank arguments (d1, d2, d3) bound one form only: a *linear, light-independent, separable, fixed-basis* operator, fitted in weighted Frobenius norms that down-weight grazing rows.
   - My re-check shows d1's CK2 capacity numbers roughly double because of that weighting (§2.2).
   - d1's claim that "a non-negative fit cannot be better in that norm" is false. Repository evidence contradicts it: WL95 §2.1 has signed svd16 at op-L1 0.891 against L1-NMF16 at 0.380.
   - The real reason a learned sharp *local* carrier is pointless is that ER already is the compact (69-float), exact learned local representation. The remaining sharp problem is geometric delivery (C). A learned form of that is learned C, untested for sharp senders; WL52 failed for broad senders on cost and contact.
3. **The e2_p oracles are credible for one configuration only.** CK2, grazing light, box A, Lambert receivers.
   - [MEAS-D] d1 and d3 implemented it independently and both get ratio 1.015 / rel-L1 0.068 against PT, with n0 = 0.059.
   - Read as quadrature-added error, that still allows roughly 0.03 of oracle bias.
   - d2's and d4's surrogates (9 face patches, no occlusion) are compared against WL95A's 54-patch numbers. That is a cross-scope check, not a validation.
4. **Ownership specifications have real gaps in three designs.**
   - d2: relayed e3_p sits inside b2 and is also in Λ's e3 target, so it is double counted.
   - d1: incidence on the panel from broad senders (E2 at the panel) is never relayed, so part of e3_p is unowned.
   - d3: R_θ keeps ECRO's learned `a·cᵀ` local operator, which contradicts its own "ER owns every local event".
   - All except d4: the existing `pt86a` split (`mid == 1`) cannot give "broad-last" targets, because S2 is also peaked under the Π rule.
5. **The learned element has no measured utility, and prior evidence predicts none** (WL87, WL90, sweeps 11.6 ms vs generator 12.5 ms). All four designers say so.
   - Per-light explicit sharp work erodes ECRO's only measured learned advantage, the 3.5–3.9× cheaper source-only update.
6. **Image-level relevance of the sharp e2_p fix is undemonstrated on CK2.** [FACT, MEAS WL95A B3]
   - Pooled M64 E_H is 0.0245 against a floor of 0.0118. The difference, +0.0127, is below the 0.0167 tolerance.
   - Every design premise rests on delivery-level metrics.
7. **Recommended overall judgment: EVIDENCE_INSUFFICIENT**, with the scoped finding that no specified mechanism carries sharp structure in a learned component (§6).

## 1. ER ports and deterministic oracles

**Port fidelity [FACT, checked line by line against `closure82.py:121-146` / `closure83_dr.py`].** All four ports (d1 `er_np.py`, d2 `er_np.py`, d3 `toy96.py`, d4 `er_np.py`) correctly have:
- outer lobes at the unrefracted directions;
- inner lobes at `t(w)` with the 1/η² factor. d1/d2 use the cancellation-free form, which is algebraically equal to `refract_up`;
- Fresnel on `wi·h` using the refracted `wi` for inner lobes;
- `s(μ_i)s(μ_o)` multiplying only the inner + Lambert part;
- validity masks as the runtime;
- `eval = f·cosθ_o`.

**Convention [FACT].** Every oracle uses `eval(wi = viewer, wo = light)·I/d²`, which is the T_COS / `pt86a` NEE convention:
- d1 `deliv96.py:work`;
- d3 `L1`;
- d4 `ck2_model.py:34`;
- d3 photons.

The photon weight `eval·I·cosθ_v / (d²·p_A·p_v·M)` is correct. No cosine is double counted.

**What the CK2 agreement does and does not validate.**
- [MEAS-D] d1 and d3 match WL95A PT e2_p on box A (54 patches) at 1.015 / 0.068 with corr 0.995. These are two independent codes.
- [INF] n0 = 0.059 is roughly the expected error of the reference mean. 0.068 against 0.059 therefore leaves an oracle bias of order √(0.068² − 0.059²) ≈ 0.034 unresolved, plus a 1.5 % flux offset. d1's S-F0 tolerance (≤ n0 + 0.02) accepts this; FA2 in d4 is stricter.
- [FACT] The validation covers one grazing omni light, Lambert receivers, and planar-sender visibility using OBB/sphere segment tests for light→panel plus convexity for panel→box A. Box B and the sphere are not tested as panel→box occluders.
  - The CK2 light is indeed omni (`scene_ck2.json` has no `aim`; `light_arrays` cos_cut = −2), so omni vs spot is not an error on CK2.
  - Spot lights in the training distribution are untested.
- [FACT] Every number for L2, L3, d3 light 1, d2's other lights and d4's alt light is oracle-only, with no PT check.
- d2 and d4 use 9 panel-facing patches and no occlusion. The face carries 70.6 % of e2_p.
  - d2's table placing its 9-patch q 1.430 / 0.435 beside WL95A's 54-patch 1.422 / 0.433 is a cross-scope coincidence, not a reproduction.
  - d4's X1 pooled ratio 0.86 (unconverged) disagrees with WL95A's 1.022.
  - For CK2 specifically, missing light→panel occlusion is harmless [INF: light at x = −1.45, box A at x ≳ −1.35, so the segments stay in x ∈ [−1.5, −1.45]].

**Inconsistency on non-grazing lights.**
- d4 generalises from one moderate light (binned 0.051) to "non-concentrated lights are uninformative".
- d2 ([−1.2, 0.2, 2.6]: binned 0.216), d3 (light 1: 0.162) and d1 (L2/L3, M64: 0.14–0.17) contradict it.
- [INF] The binned bias depends on light configuration, not only on grazing incidence. d4's V2 exclusion rule is unsupported.

**Shadow edges are untested by every carrier ladder.**
- CK2 has no shadow boundary across the panel. Node or CP interpolation (d1 NODE, d4 CP) interpolates Φ across a visibility discontinuity. d4's *log*-bilinear Φ is undefined at shadowed nodes.
- d4's g-rule `h ≤ d·√(2α)` ignores visibility entirely.
- Photon carriers (d3) are robust here, because they sample visibility per photon.

## 2. Rank and capacity claims

### 2.1 What each claim measures

| claim | operator | norm / weighting | what it can show |
|---|---|---|---|
| d1: 148 / 381 / 614 for 90 / 99 / 99.9 % energy; CK2 delivery 0.51 / 0.28 / 0.17 at rank 64 / 128 / 256 | material-level ER kernel, 1,024 incident × 16,384 outgoing cells, light-independent | Frobenius on X = K·z_in·z_out. Grazing incident rows get weight ≈ 0.03 | capacity of a separable, light-independent, linear operator under a grazing-down-weighted norm |
| d2: per-sub-cell transfer needs rank ≥ 256 | one S1 sub-cell, 4,608 incidences × 1,196 receiver patches | unweighted Frobenius on raw T. Grazing rows are small because eval contains cos(light), so ring-0 rows fit worst (1.3–1.8) | the same weighting artifact, applied to a transfer that includes visibility, i.e. learned C |
| d3: about 500 for 99 % at 1,024 bins | local operator, unit irradiance per incident bin, columns weighted by W | Frobenius energy | linear separability of a near-specular kernel |
| WL96 ring_alloc | M64·P_ring0 | exact | rank 16 suffices for ring-0 inputs; the bound operators gave ring 0 only 6.4 % of row weight |

**Conceptual flaw shared by all three.** [INF]
- A near-specular kernel is close to a blurred reflection permutation. It is high-rank in (ω_in × ω_out) yet trivially compact in a lobe-aligned (half-vector) parameterisation; ER itself is 69 floats.
- Rank therefore measures *separability*, not *representational cost*.
- These calculations falsify ECRO's specific u = c·s, Bq = a·u form at fine resolution, which is what E-L+, LSTO and d3's learned basis are. They do **not** show that "no compact learned representation can carry sharp structure".

### 2.2 Re-check of the weighting [MEAS-C]

Setup: d1's code, 1 cm panel grid; the flux-64 row reproduces d1's 0.507.

| SVD weighting | rank | CK2 ratio / rel L1 | L2 | L3 |
|---|---|---|---|---|
| d1 flux (z_in·z_out) | 64 | 0.714 / 0.507 | — | — |
| row-normalised (each incident row at unit outgoing flux) | 32 / 64 / 128 | 1.153 / 0.347; 1.079 / 0.262; 0.991 / 0.118 | 64: 0.065 | 64: 0.089 |
| CK2-incidence-weighted (light-conditioned) | 16 / 64 | 0.964 / **0.062**; 0.971 / 0.055 | 16: **0.850** | 16: **0.688** |
| untruncated | — | 0.971 / 0.055 | — | — |

- CK2 irradiance occupies 32 of the 1,024 incident cells (99 % of irradiance).
- d1's CK2 capacity requirement is inflated about 2× by the norm. A light-independent operator still needs rank about 128 for 0.12 on grazing CK2.
- A light-conditioned rank-16 operator is essentially exact on CK2 and fails on other lights.
- **The binding issue is light-independence over the hemisphere, not intrinsic capacity.** This matches the ring-0 correction.
- d1's E-F1 ("kill condition essentially met learning-free") is **not met**. Its premise compares Eckart–Young (Frobenius-optimal) with an L1 delivery norm. WL95 §2.1 shows L1-targeted NMF beating signed SVD in L1 by more than 2× (0.380 vs 0.891).

### 2.3 Consequence for WL95A's RANK16 verdict

- [INF] WL95A's RANK16_DELIVERY = INADEQUATE holds for the bounds as fitted (training incidence had ring 0 at 4.3 %).
- ring_alloc and my re-check show that this is fit-distribution allocation, not a rank floor, for CK2 alone.
- The finding is moot for design: even an exact rank-16 copy of M64 inherits M64's +42–54 % binned-outgoing delivery bias.

## 3. The "learned outgoing field = baked radiance" argument (d4 §0.1)

- [FACT] `Architecture.md:58` excludes baked final radiance as a runtime rule/state **input**.
- [FACT] §4's table has rule parameters source-independent with no post-edit retraining, and incident/outgoing radiance recomputed through the current solve.
- [INF] d4's conclusion is right for a field *fitted* to radiance (per-light or per-scene refit lifecycle). The cited sentence is the weaker basis; the §4 lifecycle rows are the decisive one.
- The dichotomy "either baked, or it is ER with no learned information" is **false**. A third option, a learned *response* operator that outputs L(x, ω) from the current incident field, is neither. ECRO's B, and d4's own LIR, are exactly that.
- Separately, **source independence of ECRO's generated factors is an economic property of ECRO, not a canon requirement.**
  - Canon requires source-independent rule *parameters* and response to the current incident field (`Architecture.md:21`).
  - d1's rejection of a light-aware generator because it "breaks source independence" overstates the canon.
  - Such a generator is admissible as a response operator, linear in power and nonlinear in incident direction. It would re-learn ER (K side) or become learned C.

## 4. Event-family ownership and double counting

| candidate | issue | severity |
|---|---|---|
| d1 SIC-R | Step 5 relays only E≥3 at peaked patches. E2 at the panel (light→broad→panel→receiver) is excluded from Λ ("last sender peaked") and never relayed, so part of e3_p is **unowned** | serious |
| d1 SIC-R | Λ = Σ_j (T_G S_ER)^j includes panel scattering at intermediate vertices (orders ≥ 4). That is learned sharp response, not acknowledged | moderate |
| d1 SIC-R | Panel incident E≥3 is supervised "through M64_S1": the biased binned operator re-enters the target | serious for e3_p |
| d1 SIC-R | e3_o includes S2-last paths, but S2 is peaked | serious (tally) |
| d1 SIC-R | The moment carrier is nonlinear across lights in one atom (acknowledged). Λ passivity is not mentioned | moderate |
| d2 C2 | b2 = M64∘𝒯[q̂_broad] relays through panel hits, so b2 already contains light→broad→panel→receiver. By `pt86a` counting (vertex adjacent to the receiver = panel, NEE at k = 3) that is **e3_p**. Λ is "supervised by e3_p + e3_o", so the family is **double counted** unless the targets exclude relay-owned paths. No such tally exists | serious |
| d2 C2 | Camera continuation "reads B(x3)". It must state whether B includes q(x3) (ECRO's b̂ does; the table term does not), or add NEE at x3 as d3 does. Otherwise light→x3→panel→x1 is missing or doubled | moderate (spec) |
| d2 C2 | DQ handles delta lights only and relays handle extended emitters, so there is no double count between them [FACT, correct] | — |
| d3 LSC-R | R_θ = "ECRO apply" keeps the learned `a·cᵀ` local operator at broad patches, contradicting "ER owns every local event". Should be fixed M64 as in d2 / d4 | moderate |
| d3 LSC-R | The table read at continuation hits (all orders vs truncated) is left open by d3 itself | moderate |
| d3 ESC-R | The source of T_ind (the incident field at sharp cells for e3_p) is unspecified | serious (e3_p unowned) |
| d4 LIR | b¹ = M64·(i⁰ + first-flight incidence): it is unclear whether D2-delivered sharp incidence at broad patches enters b¹. If not, light→panel→A→B→receiver is unowned | moderate |
| d4 LIR | Learned sharp intermediates and sharp→sharp truncation beyond J_s are declared. Supervision needs a **new** incident tally, which is honestly labelled | adequate |
| all | Λ passivity is not guaranteed by non-negativity. The learned ECRO senders measured 2.3 / 3.1 albedo (WL95A). d2 and d4 require a check; d1 and d3 omit it. Truncation (no iteration) prevents divergence but not energy error | moderate |

**K/C mapping.** d4's "Λ ≈ Σ_{j≥1}(CK)^j C restricted to broad last vertices" is the only explicit mapping of ECRO's event partition onto K/C. It is coherent [INF]. The others inherit WL95's unresolved mapping.

## 5. Falsifiability and supervision

- **Form.** All four Λ forms are concrete enough to falsify: ECRO P/Q generator, rank 24–64, non-negative, a stated input field and a stated output.
- **Targets.**
  - Existing `field_src(split)` tallies are adequate only if peaked = S1 alone.
  - With S2 peaked, all designs need a new class tag. d2 also needs a relay-excluding tally.
  - d4's incident tally is constructible (a `pt86a` variant). It is the most expensive item: 96 × 4 new PT tallies.
- **"Learning-free upper bound"** (d1 S-F5, d2 F2d, d3 S5.3, d4 FB2).
  - A per-geometry NMF or least-squares fit is neither an upper nor a lower bound: NMF is non-convex, and in-sample fit says nothing about cross-geometry learnability.
  - It is useful only as a feasibility probe and should be relabelled as such.
- **Kill conditions.** The utility kills (d1 S-F5, d2 F2c, d3 FL1, d4 FB3) are equivalent and well posed: explicit J-sweeps on the same prefix vs PT tail headroom, plus timing.
  - d3's FI / FL1 adds the needed **image-level** test.
  - d1 E-F1 is mis-premised (§2.2).

## 6. Mechanisms the designers missed or dismissed too quickly

1. **Light-conditioned or amortized learned delivery (learned C for sharp reflected footprints).** Input: incident atoms and geometry. Output: receiver incidence.
   - Admissible as a response operator.
   - Never tested for sharp senders; WL52 tested broad, patch-level senders.
   - Prior is negative on cost and visibility transitions. Not a viable primary, but its rejection should cite "untested, WL52 prior", not "rank".
2. **Learned guiding or proposal for the receiver-anchored gather.**
   - It directly targets the measured X1 heavy tail (0.46 rays per draw in Ω50, d1).
   - WL95's rejection ("cannot recover relations never hit") does not apply to continuous proposals.
   - d3's analytic image-source proposal (0.158 / 0.081 at 64 / 256 samples) bounds what it could deliver, and photons beat it.
   - Whether it is "learned transport" is an owner question. Low value.
3. **Analytic lobe integration (LTC-type) as a non-learned carrier.**
   - d4's capacity test used moment Gaussians (rel L1 0.27–0.56), a weak member of the family. LTC fits GGX far better.
   - It is not learned and is ALP-class (WL95-rejected for pair visibility), so it does not change the learned verdict.
4. **Learned incidence at sharp cells for e3_p.** This is LIR's genuinely non-trivial learned job.
   - d4 §2.4 shows 64-bin incidence at 5 cm cells is nearly adequate for extended sources (0.045, MEAS-D proxy).
   - This is the only place a learned output feeds sharp structure, through ER, at plausibly low rank. Untested.

## 7. Per-candidate verdicts

**d1 SIC-R** (node carrier + photon relay + learned Λ_θ)
- *Fatal:* none intrinsic, but the e3_p relay gap (E2 at the panel unowned) must be fixed.
- *Serious:*
  - M64_S1 in the panel supervision;
  - the S2-last tally;
  - photons measured only with an ideal flux distribution, not ER's VNDF sampler;
  - NODE never tested at shadow edges.
- *Supported:*
  - NODE 100 mm 0.038 / 50 mm 0.011 on CK2 (MEAS-D);
  - photon per-draw 0.068 at 43k (ideal sampler);
  - M64 deterministic reproduction 1.486 / 0.487 against WL95A 1.495 / 0.500, which locates the bias in the carrier.
- *Changes:* relay E2 at the panel; real sampler; new tags; Λ passivity; a moment carrier per light only.
- **Learned method: VIABLE_WITH_CHANGES, conditional on Gate A and Λ utility. Comparator: VIABLE_WITH_CHANGES.**

**d1 E-L+** (learned separable kernel, exact-outgoing readout)
- *Fatal:* duplicates ER, with no information gain. The patch-constant u imposes the X1c-type floor (0.27) regardless of rank.
- *Serious:* its capacity kill is mis-argued (§2.2).
- **Learned: NOT_VIABLE** (redundancy and the spatial floor, not capacity). **Diagnostic arm: VIABLE** as a bound.

**d2 LSTO / C1** (lobe state + learned transfer)
- *Fatal:* it is learned C plus a re-learned ER (WL52 class). The 2.6 GB factor arithmetic is order-of-magnitude right, if conservative (no shared basis across sub-cells).
- *Serious:* its rank evidence carries the grazing-weight artifact.
- **Learned: NOT_VIABLE. Comparator: NOT_VIABLE** (DQ dominates).

**d2 SRR-LB / C2** (DQ + relays + learned broad Λ)
- *Fatal:* e3_p double count (b2 vs the Λ target).
- *Serious:* the B(x3) / q semantics; DQ α⁻² to α⁻⁴ scaling for curved senders.
- *Supported:* DQ 5 cm × 4² at 0.026; the clearest e3_p owner (relays, importance on the lobe, Γ ≤ albedo).
- *Changes:* a relay-exclusive target tally, or targets with Λ fed only after relays; NEE at x3.
- **Learned: VIABLE_WITH_CHANGES (Gate A). Comparator (DQ + relays): VIABLE.**

**d3 LSC-R** (light-side photons + continuation + R_θ)
- *Serious:* the learned `a·cᵀ` in R_θ; the continuation-table semantics.
- *Supported:* photons with a realistic mixture sampler give 0.069 / 0.033 at 40k / 160k, with no heavy tail (worst ≈ 1.5× median). X1 reproduction is 0.603 vs 0.617. d3 is the best-evidenced sharp estimator.
- **Learned: VIABLE_WITH_CHANGES (Gate A; replace a·cᵀ with fixed M64). Comparator: VIABLE.**

**d3 ESC-R** (10 cm × 1,024-bin table + sender relation)
- *Serious:*
  - the T_ind source for e3_p is unspecified;
  - the α⁻² bin count;
  - relation error is a fixed-pattern bias (0.070 at r = 4), not "no variance".
- **Learned: same Λ as LSC-R. Comparator: VIABLE_WITH_CHANGES** (only above a measured α_min).

**d4 CIS-ER** (continuous incident state + ER emission + D2)
- *Serious:*
  - log-bilinear Φ at shadow edges;
  - e3_p needs an incidence supply it does not specify;
  - a quadratic geometry-only pair cache (12–24 M rays, 20–37× ECRO's relation).
- *Supported:* the cleanest finding is that the spatial error is mostly the emission anchor: CP1 0.113 vs X1c 0.234; CP4 0.008. d1 (NODE) and d2 (R3) corroborate it.
- **Learned: NOT_VIABLE by construction** (honestly declared). **Comparator / oracle: VIABLE.**

**d4 LIR** (learned broad-last incidence resolvent)
- *Serious:*
  - the b¹ composition ambiguity;
  - learned sharp intermediates;
  - costly new tallies;
  - predicted null utility.
- *Supported:* the best ownership and K/C specification; gated preconditions FB1–FB3 before any training.
- **Learned: VIABLE_WITH_CHANGES (the strongest Λ specification).**

**Rejected starting points**
- d2's learned transfer, d3's learned basis and d4's learned outgoing field: correctly rejected as *learned* carriers, on weaker stated grounds (rank artifacts; a false baked/ER dichotomy).
- d1's E-L+ rejection: right conclusion, wrong proof.

**Cosmetic variants**
- Λ_θ (d1) = Λ_θ (d2) = R_θ (d3) = LIR Λ (d4): one hypothesis, differing only in input (E2 incidence vs b2 / b¹ outgoing) and in flight edges.
- The sharp carriers are WL95-G1 estimator factors: NODE + photon, DQ, photon, fine table, CIS + DQ.
- E-L+ ≡ LSTO ≡ d3's learned basis ≡ MEK class.

## 8. Recommended WL96 judgment: **EVIDENCE_INSUFFICIENT**

- **Not ONE_FALSIFIABLE_CANDIDATE.**
  - The single falsifiable learned element (the shared broad resolvent) is not a sharp carrier.
  - Three of four specifications have defective supervision or ownership.
  - Its sharp prefix is explicit, so admissibility hinges on owner Gate A (`Architecture.md:13`; roadmap:128).
  - Prior evidence predicts no utility.
- **Not MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION.** The variants are cosmetic.
- **Not NO_VIABLE_LEARNED_MECHANISM.**
  - The designers' proof covers only linear, light-independent, separable forms and is norm-dependent (§2.2).
  - Sharp learned C and the Λ utility are unmeasured.
  - The scoped finding stands: no *specified* learned mechanism carries sharp spatio-angular structure; ER already owns the compact local form.
  - [INF] The most probable post-smoke outcome is NO_VIABLE_LEARNED_MECHANISM / NO_UTILITY on this workload class.

## 9. The single bounded smoke that most reduces uncertainty

**A learning-free headroom and relevance smoke.** The batch class is an owner decision; it is not authorised here.

It kills or keeps all four Λ variants at once and tests whether the sharp carrier matters at image level.

**Workloads**
- CK2 light 0.
- One non-grazing light, prequalified by DQ alone (binned-vs-exact ≥ 0.15 and SNR ≥ 4), with one new 2 × 4,096 spp `field_src` reference. The reference adds a peaked-class tag (S1 + S2) and a relay-exclusive split.
- E2 T1 region C, if its split is verified before freeze.

**Stages**
1. **V0, STOP gate.** A Dr.Jit DQ e2_p oracle with real scene visibility, checked against PT on both lights: |ratio − 1| ≤ 2 SE and rel L1 ≤ n0 + 2 SE.
2. **Prefix relevance.** E_H of {DQ e2_p + PT rest} vs {pooled M64 or q e2_p + PT rest} vs the oracle floor. Render spp must be chosen so the tolerance falls below the predicted 0.0127 difference (WL95A's 2 × 4,096 cannot resolve it); declare the spp at freeze.
3. **Learned-utility headroom.**
   - H_L = E_H{exact prefix + explicit J = 3 sweeps on the same relation, including relays} − E_H{exact prefix + PT tail}.
   - Time the sweeps against ECRO `factors` on the same machine, warm and uncontended, with a full-scale GPU-utilisation probe.
   - Add an in-sample non-negative rank-24 fit of the explicit tail, labelled a feasibility probe, not a bound.

**Frozen mapping**
- H_L ≤ tolerance and sweeps ≤ generator time → every Λ variant is NO_UTILITY, i.e. NO_VIABLE_LEARNED_MECHANISM on the tested class.
- H_L > tolerance → one Λ hypothesis proceeds to supervised-target construction (d4's tally), after Gate A.
- Stage 2 unresolvable → the sharp-carrier choice has no image-level target on CK2.

**Outside this smoke** (separable later if wanted): the photon/DQ estimator comparison, α ladders and sharp learned C.