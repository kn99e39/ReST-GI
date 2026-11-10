# GM1 architecture review — 50 — Provenance and consistency audit

**Status:** design-only audit record (2026-11-10), `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. This is the only file written; no existing file was changed. It authorises nothing and recommends no direction.

**Audited (read in full):** `00_PROBLEM_BRIEF.md`; designs `10_`, `11_`, `12_`; reviews `20_`–`23_`; novelty searches `30_`–`32_`; cross-reviews `40_`–`42_`.

**Originals opened for verification:** `docs/Architecture.md`, `docs/RESEARCH_CENTRIC_TOPIC.md`, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` (all in full); `docs/RESEARCH_ROADMAP.md` (frontier, milestone table, decision boundary); `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` (§1); `AGENTS.md`; WL84 in full; the cited passages of WL30, 46, 52, 53, 58, 59, 66, 67, 69, 70, 74, 77A, 79, 80A, 80B, 81, 82, 83; `experiments/fnto_compiled80c/closure80c.py`, `experiments/fnto_compiled81/protocol/compiled81.json`, `experiments/fnto_compiled83/protocol/compiled83.json`, `experiments/fnto_compiled84/protocol/compiled84.json`; `docs/plans/wl77a/specialist_priorart.md`, `docs/plans/wl80b/c1_prior_art.md` (cited entries). WL54, 55, 56, 38, 47, 44, 45, 34 were checked against their `AGENTS.md` summaries only.

**Literature:** 23 web searches and one page extraction (arXiv HTML of Xu et al. 2024, read through an automated extractor, not line by line).

**Labels:** FACT = ESTABLISHED FACT; HIST = HISTORICAL EVIDENCE; DERIV = DERIVATION; INF = INFERENCE. Verdicts: CONFIRMED / PARTLY / INACCURATE / NOT FOUND for repository claims; VERIFIED / PARTLY / UNVERIFIED / WRONG for literature.

## 0. Findings in brief

1. **Repository provenance is sound.** Of 62 checked quantitative or quoted claims, 53 are CONFIRMED, 8 are PARTLY (paraphrase, scope or label drift), 1 is INACCURATE, none is fabricated. Every number that carries an argument (WL84 costs and bias floor, WL67/69/70 economics, WL52, WL58, WL59, WL46) matches the worklog text.
2. **One open caveat of the economics review is now closed.** The 256 × 256 image size that `22_` and `41_` could not find is stated in the WL84 and WL83 protocols (FACT). The 2.4 ns per path figure stands.
3. **Four statements carry a FACT label they do not earn** (§2.2). The most consequential: "every ER material has a Lambert lobe and rough lobes". The closure has a Lambert *slot*; its fitted weight is unchecked. Two cross-reviews use it to reject an output-sensitive solve.
4. **No report presents a historical negative as a measured defect of the WL84 ER baseline.** Two borderline passages are listed in §3.3.
5. **Contract consistency:** one terminology slip survives uncorrected (`31_` repeats "Native Reuse" for ER), one FACT label silently narrows GM1 to fixed cameras, and one contract tension is reported by nobody: LPTG discretises the response to direct and to C-mediated incidence differently (§3).
6. **Literature:** 20 of 23 spot-checked citations VERIFIED, 2 PARTLY, 1 UNVERIFIED, none WRONG. One "correction" issued by `31_` is itself only half right (§4).
7. **Central documents:** three genuine gaps between what the review needs and what the canonical text says (§5): no slot for a mechanism-neutral admission measurement, no sharpness scope for GM1, no statement on whether the global mechanism must contain a learned component.
8. **Track independence:** no sign of leakage between sealed designs (§6).
9. **After cross-review no track survives as a GM1 mechanism candidate**; twelve disagreements or open human decisions remain (§7).

## 1. Repository claims: verification

"Used by" lists the documents that rely on the claim. Line numbers are in the cited original.

### 1.1 WL84 / WL83 / WL82 / WL81 (ER foundation)

| # | Claim | Used by | Verdict | Source / correct text |
|---|---|---|---|---|
| 1 | GPU kernel time, ER_L, 64 spp: 9.9–10.1 ms, RTX 3080 Ti, uncontended | 11, 22, 23, 41 | CONFIRMED | WL84 economics table |
| 2 | Layered reference / ER_L = 1.45–1.63× per spp ("about 1.5×") | 00, 22 | CONFIRMED | WL84 |
| 3 | Integrator `path`, max depth 4 | 11, 20, 21, 22, 23 | CONFIRMED | WL84 Implementation Fact |
| 4 | Python tracing 0.40 / 0.81 / 1.22 s per call, 85–95 % of wall, identical for every arm | 11, 12, 22, 23 | CONFIRMED | WL84 |
| 5 | ER_L image L1 0.086–0.099 at 64 spp; floor 0.035–0.043 at 1024 spp ("0.035–0.04") | 00, 10, 12, 22, 23 | CONFIRMED | WL84 time-to-quality |
| 6 | Reference passes ER_L near 1024 spp, about 150–250 ms GPU per image | 12, 22, 23 | CONFIRMED | WL84 |
| 7 | ER_L highlight error 0.033–0.093 against gate 0.10; margin 0.007 stable over 8 seeds | 21, 23 | CONFIRMED | WL84 fidelity table (0.0326–0.0930) |
| 8 | Reference 65,536 spp (two halves × 32,768); uncertainty ≤ 1/4 rule; INDETERMINATE class | 21, 23 | CONFIRMED | WL84 design B, C |
| 9 | Timing margin factor 1.10 | 23 | CONFIRMED | WL84 design D |
| 10 | Scope: one coated-conductor family, 3 images, direct + limited indirect, static | 00, 21, 23 | CONFIRMED | WL84 verdict table |
| 11 | Image size 256 × 256 (`22_` and `41_`: "not found in WL84") | 22, 41, 42 | **CONFIRMED, from the protocols** | `experiments/fnto_compiled84/protocol/compiled84.json` l.23; `compiled83.json` l.192. Not in the worklog prose |
| 12 | "WL84's target images use grazing light" | 21 | PARTLY | Light 2 is the grazing light (`compiled83.json` l.191). Two of the three images use it (`test83_iid_1_L2`, `test83_iid_0_L2`); `test83_hard_1_L1` uses the key light |
| 13 | WL83's sole ER_L failure was a reference-qualification artefact | 21, 41 | CONFIRMED | WL84 Observation 1 |
| 14 | 69-float closure state | 11, 12 | CONFIRMED | WL83 l.45, l.183 |
| 15 | `sample` vs `eval_pdf` pdf discrepancy max 1.8e-4 relative near the horizon; "enters Jacobians" | 11 | PARTLY | WL83 l.117–127, l.148: the 1.8e-4 arises only when the pdf is recomputed in a *separate* kernel; the in-kernel value is correct to 2e-7; 41 of 235,526 samples exceed 1e-5; WL83 calls it rounding, not a semantic defect |
| 16 | Hard cluster "defined by high reflectance" (`20_`) / "the lowest α_b tercile" (`21_`) | 20, 21 | PARTLY | WL83 l.26: joint box terciles of thin τ, sharp α_b **and** high r. Each review names one of three conditions |
| 17 | R = 8 lobes: Lambert + 7 anisotropic GGX; INNER = Lambert + GGX 4–7, OUTER = GGX 1–3 | 00, 12, 20 | CONFIRMED | WL82 l.33; `closure80c.py` l.7, l.20–21 |
| 18 | s(μ) piecewise linear on 5 μ knots; softmax Σλ ≤ 1; reciprocity exact by construction | 20 | CONFIRMED | WL82 l.38, l.43, l.48, l.136 |
| 19 | Sampler / pdf tests restricted to lobes with α ≥ 0.1 ("WL82 T5, reused in WL83"), hence "did not run" below 0.1 | 21 (B6, C7), 41, 42 | PARTLY | WL82 l.95–96: the restriction is on *random test closures* under a uniform-quadrature estimator. For WL83 the worklog does not state it; WL83 R2 reports sample–pdf consistency over 235,526 samples and R5 a passivity audit on every DF / learned closure. "No evidence below α 0.1" is too strong; "no quadrature-based χ² evidence" is supported |
| 20 | Lobe roughness range of the closure α ∈ [0.004, 1.0] at `closure80c.py:24` | 21 | CONFIRMED | l.24 |
| 21 | Descriptor box α_i ∈ [0.05, 0.5], α_b ∈ [0.05, 0.6], log-uniform, at `compiled81.json:37`; r up to 0.98 | 21, 41 | CONFIRMED | l.37. P(α_i ≥ 0.25) = 30 %, P(α_i ≥ 0.35) = 15 % follow (DERIV re-computed) |
| 22 | WL81 B6: α 0.004 lobes noise-limited; worst hard material α_b 0.059 | 20, 21 | CONFIRMED | WL81 l.150, l.203 |

### 1.2 Historical economics (WL52, 56, 59, 67, 69, 70)

| # | Claim | Used by | Verdict | Source / correct text |
|---|---|---|---|---|
| 23 | WL67: Cycles 4 spp, 4.67 s, meets the contract; oracle S 189.8 s, 9/12 | 11, 12, 22 | CONFIRMED | WL67 l.9, l.78, l.83 |
| 24 | "Cold path tracing already beat earlier ReST-GI oracles **at matched quality**" (WL67) | 11 §11 | **INACCURATE for WL67** | WL67 l.87: "QM_contract = 4 spp, 4.67 s. QM_FNTO is **not reached** within the predeclared ladder": Cycles matches or beats S on M1–M3 from 64 spp but not on M4 (stationary-receiver signed change) up to 1024 spp. Correct wording: Cycles met the *contract* at 4 spp where S failed it. "Matched quality" is correct for WL69 only |
| 25 | WL69: Cycles 16 spp 2.3–2.4 s vs CAK 24.0 s; contract met at 1 spp, 1.2 s | 11, 12, 22, 23 | CONFIRMED | WL69 l.27–28, l.191 |
| 26 | WL70: 2.09 s vs 2.40 s (13 %); margin disappears under Blender-internal time (2.05–2.09 s) | 11, 22, 23 | CONFIRMED | WL70 l.27–34, l.184 |
| 27 | WL70: 16 M samples replayed in 0.090 s (5.6 ns each), RTX 3090; validation 0.10 s; the reused term was 90 % of the update | 22 | CONFIRMED | WL70 l.134, l.157, l.169–170 |
| 28 | WL70 D6 "exact at the estimator's own sample resolution"; four controls rejected, off-path accepted | 10, 12, 20, 23 | CONFIRMED | WL70 l.74, l.117–124 |
| 29 | WL59: geometry + B ≈ 97.5 % of S; saving 1.41 / 1.71 %; F/S equal to ≈ 1e-7 | 12, 22, 23, 32 | CONFIRMED | WL59 l.182, l.188 |
| 30 | WL56: changed-domain recompute 92.9 % of S | 22 | CONFIRMED | `AGENTS.md` (175.50 s, 92.89 %) |
| 31 | WL52: learned C 8–19× slower per update; explicit C ≈ 5 ms at 10k ports; dense O(N·M); fails contact / transition fidelity | 12, 22 | CONFIRMED | WL52 l.30–31, l.236 |
| 32 | WL52: 109 M rays in 0.105 s (1.04 G rays/s), RTX 3090 | 22 | CONFIRMED | WL52 l.11, l.48 |
| 33 | "Finite-interface mechanisms had coupled errors of 3–10 % (WL53–55: 1 cm ports 3–8 %, refined 2–5 %)" | 22 §4.4 | PARTLY | "3–8 %" is WL53 l.388. "Refined 2–5 %" and "3–10 %" are the reviewer's paraphrase of several WL54 / WL55 numbers (CASP worst 4.7 %; medium blue +3…+5.5 %); no source states those ranges |

### 1.3 Directional and representation evidence (WL46, 47, 58, 54, 66, 74, 77A, 79, 80A, 80B, 30, 38)

| # | Claim | Used by | Verdict | Source / correct text |
|---|---|---|---|---|
| 34 | WL58: optimal 256-bin outgoing projection loses 52–80 % relative L2 at α 0.04; 9.9–17.7 % at α 0.25; capacity-independent | 10, 11, 12, 20, 21, 22 | CONFIRMED | WL58 l.38, l.82 (52.36–80.00 %; 9.90–17.71 %) |
| 35 | WL58: α 0.04 loss 0.60 → 0.80 and α 0.25 loss 9.9 → 17.7 % from 0° to 60° | 21 | CONFIRMED | WL58 table l.42–53 |
| 36 | WL58 candidate had "no continuous-direction/specular bypass" | 12 | CONFIRMED | WL58 l.32, verbatim |
| 37 | WL46: 128 cosine gathers inadequate at α 0.04; lobe importance sampling required; ρ 9.4 / 17.1 with K44 as directional sender | 10, 12, 21, 22 | CONFIRMED | WL46 l.19, l.121, l.178, l.218 |
| 38 | WL46: K44 "non-physical at unsupervised (port, direction) pairs" | 10 | CONFIRMED | `AGENTS.md`: "non-passive at unsupervised (port, direction) pairs" |
| 39 | WL47: visible-only supervision gives wrong, non-passive hidden outputs (albedo 1.77) on a linear surrogate | 21 | CONFIRMED | `AGENTS.md` |
| 40 | "The WL41–51 divergence mechanisms were learned non-passive kernels" | 10 §6 | PARTLY | True for WL41 / 42 / 46. WL50 / 51 attribute the divergence to recursion of a vertex-specific, itself non-passive *reference* operator (`AGENTS.md`, WL51). The conclusion drawn (does not carry over to certified-passive ER with ray-cast C) is unaffected |
| 41 | WL54: errors up to 42 % at 1 cm ports; unknowns 7.35× on the 4 mm gap; ±3.3 % per realisation; 2.3 mm strip aliased | 12, 20, 22 | CONFIRMED | `AGENTS.md` WL54 |
| 42 | WL74: rank-8 restart spread 120 % | 21 | CONFIRMED | WL74 l.186, l.261 |
| 43 | WL79: fixed basis 1.26–1.41× on exact symmetry images | 21 | CONFIRMED | WL79 l.278–284, l.306 |
| 44 | WL77A: no candidate selected; WL76 / 77 floors confounded by q = 1024 single-scatter noise | 21, 23 | CONFIRMED | WL77A l.83, l.112–115; roadmap l.120 |
| 45 | WL80B: C1 "PHYSICAL BASELINE ONLY", inadmissible as a learned K under FNTO v0.1 | 12, 32, 42 | CONFIRMED | WL80B l.116–120; `c1_prior_art.md` F0, F4 |
| 46 | WL30: "shared decoder owns transport" | 12 | CONFIRMED | WL30 l.412 |
| 47 | WL38: full-recompute seed spread 0.23 dB | 23 | CONFIRMED | `AGENTS.md` |
| 48 | 8DNA validity only if "no other objects can intersect with the asset's convex hull" | 32 | CONFIRMED | `wl77a/specialist_priorart.md` l.82 |

### 1.4 Canonical contracts as quoted

| # | Claim | Used by | Verdict | Source / correct text |
|---|---|---|---|---|
| 49 | GM1 row: "Proving only an image match or implementing a full retracer does not complete GM1" | 10, 11, 23, 41, 42 | CONFIRMED | `RESEARCH_ROADMAP.md` l.16 |
| 50 | GM1 row requires "edit-to-valid-output and per-frame accounting" | 22, 40, 42 | CONFIRMED | l.16 |
| 51 | "GM1's valid output is images at given cameras" (labelled FACT, roadmap) | 22 A4 | **PARTLY — label wrong** | The row does not mention cameras or views. This is an INFERENCE (see §3.2) |
| 52 | `Architecture.md` §3: "Source and receiver mappings conserve area-weighted flux … Near-contact spatial information must survive both sides of the K/C interface" | 20 C-4, 42 | CONFIRMED | verbatim |
| 53 | `Architecture.md` §4: radiance row "Recompute through the current coupled solve"; baked final radiance excluded from runtime inputs | 10 | CONFIRMED | §4 table and last paragraph |
| 54 | `Architecture.md` §6: F and S use identical rule parameters, inputs, interfaces | 20, 40 | CONFIRMED | §6 |
| 55 | FNTO v0.1 warns against replacing learned shading with "a limited explicit radiosity solver" | 12 (§1), 42 (l.17) | CONFIRMED | `FNTO_ARCHITECTURE_CONTRACT_V0_1.md` l.17, inside §1 |
| 56 | "Novelty cannot rest on … selective updating … alone" (topic l.46) | 30, 40 | CONFIRMED | `RESEARCH_CENTRIC_TOPIC.md` l.46 |
| 57 | "A missed candidate is irrecoverable"; "Normal similarity or local distance alone cannot guarantee coverage" | 10, 12, 23 | CONFIRMED | strategy document l.66, l.68 |
| 58 | Required report: "radiometrically weighted false negatives, false-positive work and total discovery cost" | 23 A-6 | PARTLY | l.68 reads "… and total discovery/inference/recompute cost", and the sentence governs a *learned-selector* test. Applying it to DDT-D discovery is a reasonable extension, not a quoted requirement |
| 59 | Research invariant "learned transport response, not canonical baked radiance" | 11, 12 | CONFIRMED | `AGENTS.md` |
| 60 | `AGENTS.md` decisions 8 and 9 (2007–2013 priority reference; 1997 hierarchy not the starting point) | 32 | CONFIRMED | The representative works `32_` lists differ from the four the topic document names (Temporal Radiance Caching, Instant Caching, clustered GPU GI, Voxel Cone Tracing) |
| 61 | "Native Reuse" is a role for an *external* baseline | 23 B-6 | CONFIRMED | strategy document, comparison 2 |
| 62 | TransGI: no released code (WL67) | 32 | CONFIRMED | strategy document l.15 |

**Arithmetic.** I recomputed the derived figures that decide verdicts: `22_` (2.4 ns, 7,600 images, φ bounds, Crofton fractions, PR1 / PR2 ceilings, iteration counts, memory), `21_` (N_ω ≈ 16/α², encircled energy, 38 GB, hit rate p, 0.16–0.64 nodes, α\* 0.17–0.34, per-bounce peak loss), `40_` (45 %, 0.8 %, 0.4 %), `41_` (RMS 2.3–0.75), `42_` (ρ table, Ω_min 0.8 sr). All reproduce. One internal inconsistency: `22_` finding 1 says a pass over "about 10⁸ entries" costs a 1024-spp re-render at 256²; with its own constants that is 15–50 ms against 160 ms. Its §2.3 figure (5·10⁸ cells, 0.1–0.25 s) is the consistent one.

## 2. Label discipline

### 2.1 Second-hand chains

The three cross-reviews did not reopen worklogs (each says so) and cite numbers through the reviews. I traced every such number to its original; none was distorted in transit.

### 2.2 FACT labels that are not facts

| Statement | Where | Actual status |
|---|---|---|
| "Every ER material has a Lambert lobe and rough lobes (FACT, WL82 lobe groups)" | `21_` A3; reused in `40_` §2 and `42_` §3.1 (R22-C1 row), §5 | The closure *parameterisation* has a Lambert slot (FACT). Its fitted weight per material is unchecked, as the same reviews note elsewhere. INF. This carries the "residual support is dense after one rough bounce" argument in both cross-reviews |
| "GM1's valid output is images at given cameras" | `22_` A4 | INF (row #51) |
| "FACT (R20 §3.4, R21 §3.5): PR1 and PR2 are sound"; "FACT (R20 §4.1 …): F ≡ S is proved" | `42_` §1, §3.3 | Reviewer derivations. DERIV. The brief reserves FACT for canonical contracts and verified measurements |
| "ER is qualified at α ≥ 0.05" (stated without label in `21_` §5.1, adopted as scope in `42_` §4) | `21_`, `42_` | 0.05 is the lower edge of the *descriptor* box (coat and base roughness). The fitted closure lobes may be sharper or broader; `21_` §1.1 says this itself (INF) but the later sections drop the qualifier |

## 3. Consistency with canonical contracts

### 3.1 K/C ownership

- All three designs keep K pointwise (frozen ER) and every segment in C, and say so. No double count, no ownership gap, no change to ER values. CONSISTENT (FACT for the texts; `20_` verified the derivations).
- **Unreported tension (INF).** `Architecture.md` §2 states that "Direct and C-mediated incidence use the same intrinsic response rule; no source-identity or direct/indirect response branch is required", and `RESEARCH_CENTRIC_TOPIC.md` l.19 that they "must share the same source-independent K response". LPTG applies exact `f_ER` to direct light at every vertex and the K-node matrix **B** to indirect R-events. The continuous rule is the same; the *discretised* response differs by illumination path. No design, review or cross-review raises this. It is a question for the human contract owner, not a demonstrated violation.
- **Uncited current-contract sentence.** `12_` §14.1, `32_` §5.5 and `42_` cite the historical FNTO v0.1 warning (l.17). None cites the current one, `Architecture.md` §8: "A free-space radiosity solver, SH2-only Lambertian model, scene-baked monolithic radiance MLP or SPT-1 repair is not an interchangeable implementation of this contract. A genuine contract conflict must be reported before changing scope." The conflict *is* reported (as a human decision) in all three, so the obligation is met; the citation should be to the living document.
- Read-only ER interface extensions (sampled lobe index, lobe width, R/S-restricted `eval` / `sample`) are flagged as needing approval in `11_`, `12_`, `41_`, `42_`. CONSISTENT with `Architecture.md` §3 ("Future modifications to ER require a new, separately justified hypothesis").
- `10_` §2 reports that a scene-fitted learned `b0` would be baked radiance. CONSISTENT and correctly escalated.

### 3.2 Baseline-role terminology

- No report calls a path-traced re-render "Full Recompute"; all keep Neural Full Recompute and Native Reuse UNMEASURED. CONSISTENT.
- **Slip, flagged and fixed in one place only.** `11_` §12 calls ER's survival "valid Native Reuse". `23_` B-6 objects; `41_` accepts ("not invalidated"). But `31_` §5 repeats "ER reuse is Native Reuse of a material-frame asset" and is not corrected anywhere. Under the strategy document the term is reserved for an external baseline's learned representation.
- **Silent narrowing.** `22_` A4 treats view independence as outside GM1 with a FACT label. The roadmap asks for "edit-to-valid-output and per-frame accounting" and names no camera count. `40_` §3.2 rebuts narrowly; `22_` §0.5 itself calls the workload question a human decision. The FACT label should be INF.
- **Roles the gate has no name for.** The reviews introduce "warm path-traced comparator", "masked cold re-render" and "residual path re-rendering". All are path-traced re-rendering under comparison 3 of the gate, and are labelled so. See §5.3.

### 3.3 Historical negatives and the ER baseline

No statement presents a WL41–80 result as a measured defect of ER. Every use I found carries "possible mechanism", "other operator" or "original scope". Two borderline passages:

- `11_` §9 and `20_` C-2 use the WL83 float32 pdf discrepancy as a caution for shift Jacobians and chain weights. It is a WL83 measurement of ER's own runtime, so it is in scope, but both omit that the in-kernel value is exact to 2e-7 (row #15). `20_` labels the consequence INF and unmeasured.
- `21_` B6 / C7 turn a test-design restriction into "no sampler evidence below α 0.1" (row #19). This overstates an evidence gap; it does not assert a defect.

### 3.4 GM1 scope and the no-adoption rule

- No report adopts a candidate; every file states it authorises nothing. CONSISTENT.
- Three passages read as direction advice although each disclaims it: `21_` §6 ("should precede any other GM1 work on these designs"), `41_` §7 ("yes, Stage 0 and Stage 1 before any mechanism is built"), `42_` §7 ("What I would look at first"). They are opinions requested by the cross-review format, not authorisations, and must not be read as a next-batch decision.
- `23_` X7 is correct and should be kept in view: none of the designs has learned global state, so no F/S outcome here could be reported as selective recompute of *learned* transport.

## 4. Literature spot-check (23 citations)

| # | Citation as used | Cited in | Verdict | Note |
|---|---|---|---|---|
| 1 | Xu, Li, Georgiev, Hedstrom, Ramamoorthi 2024, "Residual Path Integrals for Re-rendering", CGF 43(4), EGSR; arXiv 2406.16302; DOI 10.1111/cgf.15152 | 10, 30, 31, 40–42 | VERIFIED | Mechanism confirmed by extraction: starts paths on dynamic and ghost surfaces; "given the image at the previous frame … (without auxiliary information or any cache)"; rigid motion only; six techniques plus rejection-filtered path tracing; 2–5× over correlated path tracing; "not targeted at a dynamic camera or light" |
| 1a | Same paper: benefit "about 1.24× with eight objects", shrinking with dynamic share | 31 M3 | PARTLY | The extraction reads 1.24× as an equal-time **MSE ratio** in Table 3 and reports 1.22× for one object in the same table, which does not show a shrinking trend. Needs a direct reading |
| 1b | Same paper "cites" Bala, Dorsey, Teller 1999 | 30 §5.1, §5.6 | PARTLY | The extraction finds a Bala et al. 1999 entry titled "Radiance interpolants for accelerated bounded-error ray tracing", not the ray-segment-tree paper |
| 2 | Bala, Dorsey, Teller 1999, "Interactive Ray-Traced Scene Editing Using Ray Segment Trees", EGWR 1999, pp. 31–44; about 0.1 s discovery, 3–4× over the base ray tracer | 11, 12, 30–32 | VERIFIED | Abstract level. Resolves `32_` §6.7 ("3–4× not checked") |
| 3 | Granier and Drettakis 2001, "Incremental Updates for Rapid Glossy Global Illumination", CGF 20(3), pp. 268–277 | 30, 32, 40, 42 | VERIFIED | Abstract: hierarchical radiosity with clustering for diffuse transport, particle tracing for non-diffuse, a dynamic spatial structure limiting re-emitted particles. `32_`'s "LPTG architecture at 2001 scope" is an INFERENCE from the abstract |
| 4 | Lin, Kettunen, Wyman 2026, "ReSTIR PT Enhanced", PACM CGIT 9(1), DOI 10.1145/3804494 | 11, 31, 32 | VERIFIED | The "2 × 64 bytes per pixel" figure remains unverified, as both reports say |
| 5 | Müller, Rousselle, Novák, Keller 2020, "Neural Control Variates" — `31_` §4(c) "author order" correction | 31 | PARTLY | The published ACM TOG 39(6) order is Müller, Rousselle, **Keller, Novák**; the arXiv listing has Novák before Keller. `11_` gave no order, so there was nothing to correct |
| 6 | Günther and Grosch 2015, "Consistent Scene Editing by Progressive Difference Images", CGF 34(4), pp. 41–51 | 10, 11, 30–32 | PARTLY | Title, CGF 2015 and mechanism (stochastic difference image, removal heuristics, SPPM with preliminary BDPT / VCM) confirmed. Volume, pages and second author not confirmed by my search; `30_` and `31_` report them from the Eurographics record |
| 7 | Martín, Pueyo, Tost 2003, "Frame-to-frame coherent animation with two-pass radiosity", IEEE TVCG 9(1) | 32, 42 | VERIFIED | pp. 70–84. The abstract says the method "uses prior knowledge of a scene's dynamics" over a time interval. That is a known-animation setting, weaker support than `32_` implies for an unanticipated edit |
| 8 | Hedstrom, Kettunen, Lin, Wyman, Li 2025, "ReSTIR BDPT", ACM TOG; caustics reservoirs across frames | 11, 31 | VERIFIED | DOI 10.1145/3744898 from a third-party listing only |
| 9 | Zhang and Wang 2023, "World-Space Spatiotemporal Path Resampling for Path Tracing", CGF 42(7) | 31 | VERIFIED | Fills `11_`'s "authors not verified" |
| 10 | Dmitriev, Brabec, Myszkowski, Seidel 2002, "Interactive Global Illumination Using Selective Photon Tracing", EGWR 2002 | 10–12, 30–32 | VERIFIED | Pilot photons confirmed. "Corrective photons" and negative energy not confirmed (`30_`, `31_` already mark this). `10_` §13 states the corrective-photon detail without that caveat |
| 11 | Hong, Duan, Wang, Yuksel, Zeltner, Lin 2025, PSMS-ReSTIR, SIGGRAPH Asia | 31 | VERIFIED | |
| 12 | Deng, Hašan, Carr, Xu, Marschner 2021, "Path graphs", ACM TOG 40(6) | 12, 32 | VERIFIED | |
| 13 | Rousselle, Jarosz, Novák 2016, "Image-space Control Variates for Rendering", ACM TOG 35(6) | 10, 11, 20, 30–32 | VERIFIED | Re-rendering with the previous render as control variate; unifies scene-editing and gradient-domain estimators |
| 14 | Havran, Damez, Myszkowski, Seidel 2003, EGSR pp. 106–117; "over one order of magnitude" | 31, 32 | VERIFIED | |
| 15 | Müller, Rousselle, Novák, Keller 2021, NRC; termination by the spread heuristic of Bekaert et al. | 10, 12, 30, 32 | VERIFIED | Heuristic confirmed at secondary level |
| 16 | Kern, Brüll, Grosch 2024, "ReSTIR FG", EGSR | 11, 31 | VERIFIED | |
| 17 | Sillion, Arvo, Westin, Greenberg 1991, SIGGRAPH; SH directional-diffuse, extends two-pass methods | 10, 12, 32 | VERIFIED | The traced-specular detail was not confirmed beyond keywords |
| 18 | Loos, Nowrouzezahrai, Jarosz, Sloan 2012, "Delta Radiance Transfer", I3D | 10, 30 | VERIFIED | Low-rank delta operators for clutter objects |
| 19 | Lin et al. 2022, GRIS, ACM TOG (SIGGRAPH); hybrid shift with roughness / distance thresholds | 11, 31, 32 | VERIFIED | Threshold detail only through the 2026 follow-up, as `32_` states |
| 20 | Hadadan, Chen, Zwicker 2021, "Neural Radiosity", ACM TOG 40(6); residual-trained 4D radiance | 10, 12, 30 | VERIFIED | |
| 21 | Drettakis and Sillion 1997, SIGGRAPH pp. 57–64; shafts identify modified links | 10–12, 30–32 | VERIFIED | |
| 22 | Ren et al., "Neural cone radiosity", arXiv 2509.07522; "OctaOctree neural radiosity", arXiv 2606.08469 | 30 | VERIFIED | Title and identifier level; no edit handling mentioned in what I saw |
| 23 | Klepikov and Frolov, "Neural caching of prefiltered radiance for specular lighting", arXiv 2610.11702 | 30 | UNVERIFIED | No record returned for the identifier, title or authors. Not load-bearing |

Not re-checked and still carried from memory in the reports: Santaló 1976 (the Crofton estimate in `22_`), Fuller 1987 (attenuation formula in `23_`), Arvo 1995, Veach 1997, Chai et al. 2000, Durand et al. 2005, Keller 1997, and the †- and M-marked items. Both the Crofton relation and the attenuation formula are standard results; the *numbers* built on them are the reviewers' derivations.

## 5. Conflicts between the review's needs and central documents

Quoted text is verbatim. Proposed wording is a suggestion for the human owner; nothing was edited.

### 5.1 A mechanism-neutral admission measurement has no place in the roadmap

- **Need.** All three cross-reviews converge on two zero- or low-cost measurements that involve no global mechanism: a fitted-lobe roughness census of the frozen closures, and a same-seed T0 / T1 REF-ER path log.
- **Text.** `RESEARCH_ROADMAP.md` l.128: "Before execution, the human architecture decision must identify **one falsifiable global transport representation/update mechanism**, its spatial/angular interface and ownership …". GM1 row, evidence item (3): "one explicit proposed global mechanism".
- **Conflict.** Read literally, nothing under GM1 may run before a mechanism is chosen, and the review ends with no mechanism to choose.
- **Possible correction.** Add after l.128: "A separately authorised, mechanism-neutral characterisation of the GM1 workload and of the frozen ER closures (no global representation, no update mechanism, no cost claim) may precede that decision. It neither starts nor completes GM1."

### 5.2 GM1 names no sharpness scope

- **Need.** `21_` §5.1: angular requirements scale as α⁻², 4D state as α⁻⁴; no design states which α its claim is for.
- **Text.** GM1 row: "**sharp glossy/specular, multi-bounce, editable-visibility interreflection**". `Architecture.md` §1: ER is supported "only within a coated-conductor family".
- **Conflict.** "Specular" admits α → 0, where the closure permits 0.004 (`closure80c.py` l.24) but the qualified descriptor box starts at 0.05.
- **Possible correction.** Add to the GM1 row: "The protocol declares the lobe-roughness range and minimum reflector distance for which the claim is made; ranges outside the qualified ER family are out of scope unless ER is separately qualified there."

### 5.3 The baseline gate has no slot for warm path-traced re-rendering

- **Need.** `31_` and `41_`: the strongest published comparator for a rigid edit is a *warm* path-traced estimator (Xu et al. 2024; ReSTIR PT). `23_` X5 adds a masked cold re-render.
- **Text.** Strategy document, comparison 3: "**Path-traced Full-scene Re-render vs ReST-GI:** Cycles or Mitsuba can provide physical ground truth and an optional direct rendering cost comparator."
- **Conflict.** The text describes only a cold full-scene re-render. A T0-informed re-render is neither named nor excluded, and the nearest available word, "Native Reuse", was already misapplied twice (§3.2).
- **Possible correction.** Add to comparison 3: "Path-traced re-rendering that reuses T0 samples or the T0 image (correlated, residual, reservoir or masked re-rendering) belongs to this role, is labelled *warm path-traced re-render*, and has its T0 cost reported. A local rule of ReST-GI that an edit does not invalidate is described as *not invalidated*, not as Native Reuse."

### 5.4 Must the GM1 global mechanism contain a learned component?

- **Need.** `12_` §14.1, `23_` X7, `32_` §5.5, `41_` §7 and `42_` §4 all end on this question.
- **Text.** `RESEARCH_CENTRIC_TOPIC.md` l.9: "maintain physically credible, high-quality **learned GI** after geometry edits by reusing transport that remains valid"; l.11: "The concrete global representation and update mechanism remain open." `Architecture.md` §2 defines C as explicit editable free-space coupling, and §8 records learned C as NOT SUPPORTED, yet states that "A free-space radiosity solver … is not an interchangeable implementation of this contract." The GM1 row does not use the word "learned".
- **Conflict.** The documents permit an explicit C and forbid a free-space radiosity solver as a substitute, without saying where a learning-free global solve around a learned local K falls.
- **Possible correction.** One sentence in `Architecture.md` §9 under GM1, either: "A GM1 mechanism whose only learned component is frozen ER is admissible as a physical baseline or control, not as the candidate global mechanism", or its opposite. The choice is the human owner's.

### 5.5 Smaller points

- **Reference depth.** WL84 fixed `path` depth 4; `Architecture.md` §3 calls frozen-ER path tracing "the local-approximation ceiling" without a termination rule. `20_` §5.5, `21_` §5.6 and `23_` A-8 show that a residual-terminated solve and a depth-4 reference estimate different quantities. This belongs in a protocol, but a line in the GM1 row ("all arms and references share one declared path-termination rule") would prevent the mismatch.
- **Direct versus coupled response** (§3.1): if a discretised mechanism may treat direct and C-mediated incidence with different numerical operators, `Architecture.md` §2 should say that the requirement applies to the continuous rule.

## 6. Design-track independence

No sign that a sealed design used another track's content (INF, from content; file times are only consistent with the declared order: designs 23:15–23:17, reviews 23:23–23:26, cross-reviews 23:30).

- **No cross-naming.** None of `10_`, `11_`, `12_` contains another track's candidate name (DDT, ERPS / PR1–3, LPTG) or refers to another track.
- **Complementary blind spots.** Track A found Xu et al. 2024 and missed Bala et al. 1999. Tracks B and C both cite Bala et al. 1999 and both missed Xu et al. 2024; Track C also missed Günther and Grosch 2015, which A and B cite. Shared reading would have removed these gaps.
- **Incompatible conventions.** Lobe solid angle 4πα² (A) versus πα² (C); depth silent (A), fixed 4 (B), Russian roulette (C); three different invalid-fraction measures (φ_τ, ρ, f).
- **Overlap explained by common sources.** WL58's 52–80 %, WL70's D6 sentence, WL84's harness figures, the REF-P / REF-ER scaffold and the quarter-uncertainty rule appear in several designs; each is in the brief or in a worklog the brief lists.
- **Convergent ideas, different framing.** An exact replay store appears as B's mechanism and as C's rejected-then-kept control; a difference re-render appears as A's control A2 and B's PR2. These follow from the problem, and the wording differs.
- **Reviews.** `20_`–`23_` each state that no other review was read and cite none. `30_`–`32_` each read only their own track. One caution: the four reviews reach the same α\* and node-aliasing conclusions through different arithmetic (`20_` ≈ 0.3, `21_` 0.17–0.34, `22_` 0.125 → 0.15–0.3), which is evidence of independence rather than copying.

## 7. Disagreements and open decisions remaining after cross-review

All three advocates withdrew their track as a GM1 mechanism candidate. What is still unresolved:

1. **Strongest competitor.** `40_` and `41_` name Track C. `42_` names Track B "in its PR2 / residual form", a form `41_` has withdrawn as an economic claim and says is "not designed here".
2. **Output-sensitive solve on a link graph.** `22_` §4.8 requires it for any selective economics. `40_` proposes it (HYP-Δ) while predicting failure at K = 128, ρ ≈ 0.8 on a worst-case bound and a milder result on a physical estimate. `42_` declines to adopt it. All three rest on an unchecked Lambert weight (§2.2).
3. **α\* at K = 128.** 0.125 at four nodes per lobe (`22_`), 0.17–0.34 (`21_`), about 0.3 (`20_`), 0.35 by rule (`42_`), against the design's own 0.05–0.15 band (withdrawn). The spread is a factor of about 3 and decides how much the field can own.
4. **Does link granularity reuse more than path granularity?** `42_` §2: yes, a link store's invalid share stays at f while a path store's ρ grows with depth. `41_` D3: no, both reuse at most about one frame of rays per edit.
5. **Status of `22_` C1.** FATAL for "H-edit as an economic claim" (`22_`); `42_` accepts the prediction and rejects the framing, since H-edit was stated as a correctness contract.
6. **The headline phenomenon** (new sharp chain onto a rough stationary receiver). `20_` orders B > C > A on correctness. `42_` answers that B's unbiasedness is unusable there (noise 0.22–0.45 at storable sample counts) and `21_` that the reference itself is noise-limited. `31_` §4(b) holds that published path-space methods may help births; unmeasured. Agreed by all: no design carries it in persistent state.
7. **Composition of the warm comparator.** `31_` requires a residual-path-integral sampler and an unbiased ReSTIR PT arm; `41_` accepts and designs neither; `30_` §5.8 asks that A2 / A3 be mapped onto Xu et al.'s estimators. No document specifies it.
8. **Masked cold re-render.** `23_` X5 makes it the attribution control; `41_` predicts it degenerates to a full cold render at pixel granularity in an enclosed fixture. Untested.
9. **Are multi-view / multi-frame workloads inside GM1?** `22_` says the windows of A and C exist only there and calls it a human decision; `40_` argues a cost-versus-views curve is admissible; `42_` rests H-econ′ on it. Human decision (§3.2, §5).
10. **Workload scale.** `22_` and `41_`: at WL84 scale (10 ms per 64 spp, 0.4–1.2 s per launch) no cost claim is testable; a discriminating workload needs a cold REF-ER of several seconds. Not fixed anywhere.
11. **Sharpness scope** (§5.2) and **admissibility of a learning-free global mechanism** (§5.4). Human decisions.
12. **The two unmeasured numbers every conclusion depends on:** the albedo-weighted fitted-lobe roughness census of the frozen closures (including the Lambert weight), and the cold REF-ER time-to-quality ladder on a GM1 fixture. Until they exist, `41_` D2, `40_` §2 and `42_`'s withdrawal are inferences, as each says.

## 8. Limits of this audit

- WL54, 55, 56, 38, 47, 44, 45 and 34 were checked against `AGENTS.md` summaries, not their own text.
- Literature verification is at search-result level except for Xu et al. 2024, which was read through an automated extraction; rows 1a and 1b need a direct reading before reliance.
- Roughly 150 literature entries appear across the reports; 23 were checked. The M- and †-marked ones remain as their authors labelled them.
- I verified that derivations reproduce numerically, not that their physical assumptions hold; that was the reviewers' task.
- File modification times cannot prove independence; §6 rests on content.
