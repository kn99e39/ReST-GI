# WL95 judge report: Stages D and E (design-only)

**Scope.** I edited no repository file, trained nothing and ran no GPU job. I read the evidence files `auditA.md`, `auditB.md` and `read_canon.md` in full. For the other five reader digests I relied on the designers' and critics' cited content. I re-verified three things in code and docs:
- `docs/Architecture.md:13`: ECRO is the resumed learned-global base. EXP/C3UR are "comparators/diagnostics rather than the global method to be published". "X1-style first-gather findings … do not justify converting the project to explicit GI".
- `docs/Architecture.md:21`: there is no direct/indirect response branch, and "sharp … response demands sufficient directional information".
- `docs/RESEARCH_ROADMAP.md:128`: "one falsifiable ECRO-lineage representation/training/ownership hypothesis … Do not … promote an explicit candidate as the project goal".
- `experiments/fnto_compiled83/closure83_dr.py:1-12`: ER `eval(wi,wo) = f·cosθ_o`; the GGX lobes are anisotropic (au, av).

**Labels.** [FACT] verified · [MEAS] measured value, with its source · [INF] reasoned inference · [HYP] untested.

**Overall verdict: ARCHITECTURE_SELECTION_PREMATURE.** I recommend no primary hypothesis.

---

## 1 Verified Current Architecture

All of auditA §1 is [FACT]. Ten facts matter for the judgment:

1. **ECRO v0 / OWN pipeline.**
   - Source: s = patch-mean direct irradiance per incident bin, from 16 fixed points shared by every patch (L1, L2).
   - Local operator: a learned M_n = a cᵀ, rank ≤ 16, non-negative, with c tied between sender and receiver (L3, L7).
   - Sender table: Bq is bin- and patch-constant (L4).
   - Flight: 512 receiver-anchored rays per patch, κ = 8 per bin, one fixed shared pattern (seed 86001) (L5). The flight weight `rw` uses each ray's own z (L6).
   - Global term: rank 24, driven by u rather than E1 (L8).
   - Readout: nearest patch (L10). Sphere uses a flat frame (L11).
   - M64 averages over incident cells uniformly, with no cosine (L12).
   - Edit: selective state re-trace (bit-equal to a full rebuild) plus a full generator recompute. [MEAS] 76–78 ms per edit.
2. **Frozen ER is already a learned local rule.** It was compiled by the WL83 learned compiler: Lambert plus 7 anisotropic GGX lobes, 69 floats, exact `eval`/`pdf`/VNDF `sample` [FACT]. ECRO re-learns a bin-projected rank-16 copy of it inside the table.
3. **Canon** [FACT]. These decisions bind any architecture recommendation; selection cannot override them:
   - ECRO lineage is the learned-global base.
   - Explicit EXP/C3UR are immutable comparators.
   - Explicit GI conversion is not authorised by X1-type findings.
   - Mandatory four-role baseline gate.
   - No direct/indirect *response* branch.
   - The open owner decision "is a learning-free global solve around ER a candidate or a baseline?" (`60_coordinator_synthesis.md:94-100`) was *constrained* by the 2026-11-13 decision, which leans to "comparator". It was not formally closed.

## 2 Failure and Evidence Attribution (what the evidence actually supports)

**Local sender** [MEAS, auditB, matched norms]:
- WL94D operator flux-L1: zero operator 1.000; learned S1 0.961–1.044; admissible non-negative rank-16 0.380 (L1-targeted) to 0.603 (NMF16).
- Held-out field (W94): learned rel-L1 0.846–0.892 at peak 0.22–0.33; non-negative rank-16 0.36–0.54 at peak 0.35–0.44.
- Rank ~32 is needed for field L1 ≤ 0.15 with peak ≥ 0.7. Floor B is exact at rank 8–16.

What follows from these numbers:
- **Optimisation/objective is currently binding.** The learned operator is far from its own rank bound.
- The rank ceiling becomes binding only after that gap closes. The ceiling is an empirical floor over the factorisations tried, not a proof.
- S1 S_geo is 0.2–0.8 % [MEAS WL93D]. Geometry conditioning adds nothing measurable for S1.
- **Not measured:** delivery and E_H with the bound operators substituted at S1. This is auditB's stated gap. The operators are saved in `~/wl95_calc/S1_ops.npz` on LabServer63. *This is the only existing-data test that bears directly on whether an ECRO-lineage learned sender can work.*

**Exact sender substitution** [MEAS]:
- It improves untextured sharp renders: C2_src E2 T1 C 0.018–0.024 vs C0 0.040–0.077; C2_srcS1 CK2 0.031 / 0.019 vs 0.047 / 0.042, with noise 0.0084.
- It does *not* improve textured CK2 or the pooled single-flight table (+0.005…+0.008). Co-adaptation is the explanation.
- ENCB (exact M64 sender, path-traced L3+) is 0.0449 on a single draw and 0.027 pooled, against ORA2 0.0127 (E2 T1 C). ORA (exact prefix plus learned G) is 0.030 / 0.033. The learned tail partly *cancels* the prefix deficit, so the "exact sender + learned G is fine" reading is contaminated (ownership-physics critic, X1).

**Propagation:**
- [MEAS] GM1 only: first-gather angular projection explains 97–102 % of the explicit-vs-oracle gap (WL90). Pooled X1 equals the oracle, but single-draw variance fails held-out (1/3) and utility (3.1×) (WL91).
- [MEAS] ECRO E2: sender placement dominates (+0.050 / +0.053, WL88). RAYSEND exact reads on the deployed state are *worse* (0.059 vs EXP 0.049, WL87). RAY-vs-BIN doubled the patch RMS error (WL86A Amendment 2).
- [INF] So the GM1 projection attribution does not transfer unchanged to the ECRO Lambert-receiver regime, where sampling dominates.

**CK2 "exact-sender flight floor"** (1.378 / 1.211 / −0.334): **decomposition OPEN** (auditA §3). The candidate causes C1–C6 (outgoing-bin L4, spatial L4/L2, incident L1+L12, 16-point L2, κ = 8 single draw L5, ring-0/1 cosine L6) are confounded in one light, one realisation and 9–54 patches. G_PHYS on training items gives 0.953, so no generic floor is shown.

**Sharp-mediated ≥ 3-event family** [MEAS WL94B/94]:
- On the CK2 panel face, e3_p is 0.755 of box e3 and 61 % of B is higher-order.
- [INF] Any design that fixes only the panel's *direct-lit* first flight (e2_p) addresses about 1/3 of the panel-mediated transport there.

**Metric defects** [FACT]:
- `an94.W` uses an 8 × 8 layout. It is neutral for Lambert outgoing radiance and biased for non-Lambert outgoing radiance.
- `delivery_box` = boxes A + B.
- The panel-face correlation is computed over 9 patches.
- All WL94-series non-Lambert exitance numbers carry an unquantified W bias. On the bound arms it is ≤ 0.02.

## 3 Independent Architecture Hypotheses: deduplication

| Group | Members (designer · name) | Common mechanism | Disposition |
|---|---|---|---|
| **G1** ER-continuous sender + explicit first gather (learning-free sharp first order) | local-angular H1 LRS (deterministic dense sender grid); propagation H-A LCFG (receiver-point MIS with a stationary-point lobe proposal); ownership H-B ELCF (light-side lobe-sampled photons); state-reuse H1 DSR (sender-anchored, shared origins) | ER evaluated at exact directions for directly lit peaked senders; explicit ray-traced first flight; remainder by explicit sweeps or an optional learned resolvent | **One architecture class, four estimators.** It is the WL86B H-0 / C3URX1 class (`cand91.py`: sender-side, exact per-ray L₁) with a different first-gather quadrature. The estimator choices are *factors of a smoke*, not architectures. Canon: comparator, not method (Architecture.md:13; roadmap:128). |
| **G2** ER-binned local response + binned reusable sender-side relation + learned resolvent on e3 | ownership H-A ELBR | M64 at both ends (removes L3/L7); C3UR-type Γ_s; learned Λ_θ owns only the separately tallied ≥ 3 events | **The only ECRO-lineage, source-independent candidate.** It is materially distinct from G1 because it keeps the reusable binned first flight and a learned global object. Its learned part is unspecified, and existing evidence predicts it has no headroom over explicit sweeps (WL90 P2plus ≤ 0.0009; sweeps 11.6 ms vs generator ~12.5 ms). |
| G3 learned local lobe surrogates | local-angular H2 MEK; state-reuse H2 ALP (LTC lobe integral + scalar pair visibility) | Learned material-level re-fit of ER with an angular prior | **Rejected** (§5) |
| G4 persistent virtual-source atoms | propagation H-B VSA | Image-source / atom state for orders ≥ 2 | **Rejected** (§5) |

Two further reference rows are kept for comparison; neither is a designer candidate:
- **H-0**: the explicit EXP/C3UR/C3URX1 null.
- **E-L**: an ECRO-lineage learned sender with a changed representation or objective (WL86B H-E type). No designer specified E-L. auditB shows the current learned sender is far below its rank-16 bound.

## 4 Physical and Computational Trade-offs

These ten criteria integrate the critics' valid objections. Numbers come from the cited sources; G1 and G2 cost figures are [INF].

| Criterion | ECRO v0 (status quo) | G1 (ER + explicit continuous first gather) | G2 (ELBR) | H-0 null (EXP/C3UR/C3URX1) |
|---|---|---|---|---|
| 1 Angular representability | Learned S1 at the zero-operator level; rank-16 non-negative empirical ceiling ≥ ~0.36 field L1 [MEAS] | Exact ER at directly lit peaked senders [FACT ER]; binned for indirect incidence on peaked senders (e3_p: 0.755 of panel-face e3) | Full-rank M64 at both ends; keeps L1/L2/L4/L6/L12 and the bin floor | As G2 (EXP q), or exact per ray (X1) |
| 2 Propagation fidelity | 8 fixed receiver rays per bin; bad-tail draw [MEAS] | Unbiased in expectation; **single-draw variance is the measured failure of the nearest relative** (WL91). Estimated lobe hits per CK2 footprint receiver: designers ≈ 1, critics 0.16–0.4 per realisation [INF, conflicting; recompute from geometry] | Sender-side binned: E2 C ~C3UR class (0.026 was measured with EXP q, not M64(s): unmatched). The CK2 binned floor is unknown | Pooled ≈ oracle; single draw heavy-tailed [MEAS] |
| 3 Material generalisation | Albedo compression (CK2 slopes 0.28–0.63) [MEAS] | Exact for any ER state; the sampling density is α-specific | Exact final vertex; Λ_θ embedding risk | Exact |
| 4 Dynamic geometry reuse | Weights reused; factors fully regenerated | No learned transport state. Source- and geometry-dependent caches; full recompute on light moves and sharp movers; the dirty test is rigid-OBB box-only (`ecro86.py:65`); deformation / self-contact not derived | Rule reuse; Γ_s selective; global regeneration | Selective = full on the relation; global sweeps |
| 5 Physical correctness | Not passive; compensation [MEAS] | Unbiased first order (ER passive and reciprocal). Watch the cosine convention: `eval` already contains cosθ_o | Exclusive e2/e3 tallies [FACT `pt86a.py:175-199`]; Λ_θ only audited (the ρ/(1−ρ) bound is vacuous) | Passive |
| 6 Training feasibility | Seed-dependent; co-adaptation [MEAS] | Nothing learned for sharp transport | One learned term; WL93–94D-scale smokes repeatedly ended NO_SIGNAL / INCONCLUSIVE; the form is unfrozen | None |
| 7 Inference / update cost | Edit 76–78 ms; source-only 13–18 ms [MEAS] | Edit ≈ 100–150 ms (anchored on EXP 89–95 ms and C3URX1 124.7 ms to frame [MEAS]); **loses the only measured learned advantage**, source-only update 3.5–3.9× (WL87); scales ∝ L·(peaked area)·density | ≈ ECRO [INF]; source-update niche shrinks to a few ms once q̂ = M64·s | EXP edit 89–95 ms; PT 36 ms [MEAS] |
| 8 Memory | ~43 MB factors; dense Ft O(N²) as coded (structurally O(N·K) sparse; the global mean kills locality) | Relation at 27 B/ray including segments (not 6 B); dense variants up to 0.56 GB all-peaked at D = 1,024, ×α⁻² beyond [INF] | ≈ ECRO (the 27 MB figure omits the output basis) | Relation ~17–21 MB |
| 9 Debuggability | Entangled terms | High: every term physically tallied | High for the prefix; Λ_θ opaque | Highest |
| 10 Usefulness vs full recompute | 9× is LOW + table structure, shared by H-0 (WL86B); economics NOT TESTED | Not shown; the nearest relative failed WL91 rule (a) | Not shown; the learned value over H-0 is predicted ≈ 0 | Beats PT 7.6–31× on GM1 sharp tiers (WL91) [MEAS, GM1 only] |

**Cost-shift risk.**
- G1 moves the sharp first flight into explicit ray tracing. Every designer declares this, so it is not hidden.
- G1 becomes a per-source light / path tracer for that family under three conditions: many or moving lights; area or environment lights; α → 0, where per-atom mirror continuation is needed.
- Hidden or understated costs found by the critics:
  - per-photon `fbar` (an MC histogram, `quad87.py:85-110`), which is infeasible per photon;
  - per-ray f̄ tables for deferred orders ≥ 2 (≈ 0.5 GB);
  - 27 B/ray segment storage;
  - VSA's per-frame atom visibility at readout;
  - ALP's quadratic, source-dependent pair table.

**Initial vs per-edit cost.**
- G1 and G2 initial cost is about the state build (26–28 ms) plus relations. G1 adds per-light first-flight tracing.
- Per edit, both are dominated by global recompute (sweeps or generator). Selective state saves only 5–7 ms [MEAS WL86].

**Worst case and scaling assumptions** (true of all four rows):
- compact point lights, planar peaked senders, rigid box edits, N ≈ 1.3 k, α ≈ 0.05;
- nothing verified beyond scene86 / GM1;
- no sharp-mover, deformation or self-contact workload exists.

## 5 Recommended Hypothesis

**None.** Recommending one now would mean selecting on theoretical expressiveness, which the brief forbids. Six reasons:

1. **G1 is excluded as the method by canon.** It is barred unless the owner explicitly rules that a learning-free ER-centred global solve is an admissible candidate (Architecture.md:13; roadmap:128). All 8 critiques independently reach this conclusion, and I verified the canon text. Without that ruling, G1 is a comparator / oracle class.
2. **G2's learned component is unspecified**, so its learned claim cannot be falsified. Existing measurements (WL90 P2plus; WL89 sweep cost; WL87 source-update attribution) predict no accuracy or cost headroom over H-0. Its prefix is already measured at 0.0449 single draw on E2 T1 C with R0. Only the M64(s) × Γ_s combination is unmeasured.
3. **E-L has no specified mechanism.** Its decisive learning-free precondition, delivery and E_H of the auditB bound operators, is unmeasured.
4. **The CK2 floor decomposition is open.** It is the hinge between three things:
   - "binned reusable first flight suffices" (G2, E-L);
   - "continuous first gather needed" (G1);
   - "sampling only" (H-0 placement).
5. **The sharp-mediated e3_p family is unowned** by any well-specified mechanism. G1 leaves it binned; G2 gives it to an unspecified Λ_θ.
6. **No admitted held-out sharp workload exists** (WL93D E6 failed admission; WL88 stopped unqualified).

**Evidence-favoured constraint** (not a demonstrated fix, not a selection) [MEAS-backed]: for S1, a *geometry-conditioned* learned rank-16 tied bin operator is not justified as the owner of local angular response. The supporting measurements are S_geo < 1 %, zero-operator-level performance, the rank-16 empirical ceiling and the exact-sender render gains on untextured ROIs. Whether a *material-level* learned sender at the bound (rank 16 or 32) is delivery- and E_H-adequate is **untested**, and that is decided by the smoke below.

## 6 Falsification Experiment (future bounded smoke; NOT authorised)

**First discriminating question.** Yes: it is the CK2 flight-floor decomposition, combined with the bound-operator substitution. Both are learning-free.

**Batch class** [INF, owner decides per AGENTS.md:3-11]: this is a dependent evidence-gap follow-up (of WL94D's CK2 floor and WL95 Stage B), so a **suffix batch**, not a regular number. No new learned mechanism is introduced.

### Workloads (smallest valid)

- **W-a: CK2 "none", light 0** (grazing, ring 0). Reference: e2_p and e3_p from `CKF_none.npz` (2 × 4,096 spp halves; half-noise box A 0.058, panel face 0.069).
- **W-b: one non-grazing CK2 light variant.**
  - Needs one new 2 × 4,096 spp `field_src(split=True)` reference.
  - Must be prequalified before freeze: SNR ≥ 4× on box A, by a geometry-only rule and a minimum-signal rule (WL92A / WL88 lesson).
  - If it fails qualification, drop it and declare the CK2 result grazing-only.
- **W-c: E2 T1 / T0 region C** (WL86 REF-ER; WL92 "single + true L₃₊" ladder; ORA2 0.0127).
- **GM1 W1 / W2** are **not rerun**. WL89 / WL90 / WL91 already quantify the binned floor there; they are cited as a known-floor reference only, in their own norm, never pooled with CK2 or E2.

### Matched quantities (each regime labelled; never compared across norms)

- **Q_s (sender).** Panel once-scattered table against exact q at patch × bin, using auditB's norms (operator flux-L1, field rel-L1, energy and peak jointly, overlap (1 + mass − L1)/2), weighted with **W_T[d] = ((d//16) + ½)/4 · ΔΩ**. `an94.W` may be reported only as a bridge to old numbers.
- **Q_d2 (delivery, exactly 2 events, S1-mediated).**
  - Receivers: Lambert only. Box A is patches 1130–1183 (not 1130:1238). The panel face (9 patches) is reported separately. The 3 footprint patches are descriptive only.
  - Statistics: per-patch signed ratio with half-split CIs; flux ratio; signed and absolute rel-L1; Pearson only over the 54 box A patches, with a bootstrap CI.
  - Reference: e2_p. It is material 1 only, so S2-mediated delivery is out of scope and declared.
- **Q_d3 (panel-mediated ≥ 3 events).** e3_p against an explicit second panel flight (EXP-style sweep on the pooled relation, driven by the arm's B₂). This only scopes how much of the panel-face failure lies beyond the first gather; it is not a gate.
- **Q_r (integrated sharp interreflection).** Rendered E_H on CK2 ROI_T and E2 region C, with **LOW + arm L₂ + path-traced L₃₊**. Never use the seed-specific learned receiver or G (WL93D co-adaptation). Report the signed error, the T0→T1 edit slope and per-family noise_bound. Not PSNR.

### Oracle hierarchy and arms

Every arm uses existing or clearly constructible capability. New code is declared as new.

| Level | Sender | Propagation | Isolates | Capability |
|---|---|---|---|---|
| L-learned | C0 own s8601 / s8602 Bl | deployed R0 (512 rays, seed 86001, `rw`), exact Lambert receiver | learned sender + existing propagation | existing (`s94.py:152-160` path) |
| L-bound16a / 16b / 32 | NMF16, L1-NMF16, L1 rank-32 (fa ladder) from `S1_ops.npz` | deployed and pooled | **ECRO-lineage capacity in delivery / E_H** (auditB gap) | existing operators; same flight code |
| L-M64 | M64(s) | deployed and pooled | WL94D floor re-measured; must reproduce 1.211 / 0.528 on box A | existing |
| L-q16 / L-q256 | exact q (`quad87.qtable`, P = 16 and P = 256 stratified) | deployed and pooled | M64→q16: L1 + L12; q16→q256: L2 (C3, C4) | existing function; scene86 / CK2 plumbing new |
| L-q16-sub(a) | q at outgoing sub-bins a ∈ {4, 16} | pooled | L4 angular (C1) | new small code (finer binning in qtable plus a matching read) |
| L-X1 | exact per-ray L₁ (`cand91.radiance`, probe + shadow) | pooled R0 and pooled C3UR-type | full L4 / L6 removal at the read | port of cand91 to scene86 / CK2 (new plumbing) |
| **L-DQ** (exact sender + higher-fidelity propagation) | exact L₁ from ER at a point light | **deterministic dense area quadrature** over the panel plane (mm-scale cells) × dense receiver points on each box A patch, with shadow and receiver visibility rays; no relation sampling | the propagation reference below the physical reference | new code: ER eval (`closure83_dr` / `closure82`) + Mitsuba visibility; well defined for a point light and a planar sender |
| REF | path-traced e2_p / e3_p; REF-ER renders | — | physical reference | existing |

**Pooled** means ≥ 64 relation realisations with redrawn origins and directions (a new seed per draw), and redrawn source points for the P = 16 arms. Each arm also reports a **per-draw distribution over ≥ 16 draws** (median, 90th percentile, worst), with paired comparisons on common random numbers.

**Optional estimator factors.** These are G1 variants, run only if the owner wants G1 characterised as a comparator; they never gate the decomposition:
- sender-side dense grid with grid-phase rotations (LRS);
- the SAM-X1 uniform-area proposal vs the stationary-point lobe proposal (LCFG; SAM-X1 is the mandatory control);
- light-side photons with dense-cell receivers (ELCF);
- C3UR / C3URX1 at matched total rays.

### Error separation

1. **Local sender representation.** At *pooled* propagation, compare L-learned, L-bound*, L-M64 and L-q*. Read Q_s and Q_d2. Pooling removes L5 confounding.
2. **Directional propagation / flight.** With an exact sender, step deployed → pooled → sub-bin → X1 → L-DQ, all against REF. Steps are single-factor where possible; otherwise use a factorial of sender × {deployed, pooled}. E_H and rel-L1 are absolute metrics, so do **not** treat step differences as additive shares.
3. **Integrated.** Q_r with each arm's L₂ plus path-traced L₃₊, plus the Q_d3 scoping.

### Controls

- **Lambert-sender positive control:** e2_o on box A. Exact-sender arms at pooled propagation must match within noise. Use pooled relations so the bad-tail draw cannot cause a false invalidity.
- **Floor B sender:** sub-bin a = 1 ≈ a = 16 is expected; the bins should be adequate.
- **Incident distribution:** grazing (W-a) vs non-grazing (W-b) vs E2 lights; report ring shares. Superposition sub-test: two point lights co-binned at the panel (binned vs continuous).
- **Directional discretisation:** sub-bin ladder; half-sector bin-frame rotation (L6 sensitivity).
- **Spatial:** P ∈ {16, 256}; redrawn vs fixed source points (L2 fixed bias vs noise).
- **Capacity:** rank 16 vs 32 bound operators.
- **Compute:** every arm reports relation, probe, shadow and visibility ray counts, ER evaluations, warm uncontended ms (Dr.Jit compile excluded and reported separately) and MB. Per the standing GPU-utilisation requirement, every stage gets a full-scale probe of GPU% and wall time before freeze.
- **Reference noise:** an effect counts only if it exceeds 2 × half-split noise, plus the arm's own split-half MC noise.
- **Not run:** a textured CK2 variant (`field_src` uses SMAT; unverified per auditA C8). Declare it out of scope.

### Validity gates (STOP if failed; frozen before results)

- **V0.** L-DQ vs e2_p on box A: flux ratio ∈ [0.95, 1.05] and rel-L1 ≤ 2 × half-noise + L-DQ's own MC noise. Failure → STOP (contract, reference or implementation defect). Do not attribute the failure to any loss point.
- **V1.** The Lambert positive control passes at pooled propagation.
- **V2.** L-M64 deployed reproduces the WL94D audit values (box A ratio 1.211, rel-L1 0.528; Lambert receivers are W-neutral).
- **V3.** W-b qualification (or it is dropped).

### Decision outcomes (thresholds proposed for owner freeze)

- **D1, local representation dominant.** At pooled propagation, L-bound16 exceeds L-q256 on box A Q_d2 and Q_r by more than 2 × noise, while L-bound32 does not.
  - Decision enabled: an ECRO-lineage learned sender is admissible only with ≥ rank ~32 or a non-fixed-basis local representation. That would be a new regular-batch hypothesis, stated against ER ownership (G2).
  - If L-bound16 ≈ L-q256 within noise, rank is not binding for delivery or E_H. The learned failure is then optimisation / objective, and an H-E-type training hypothesis becomes the admissible ECRO-lineage route.
- **D2, incident projection.** L-M64 vs L-q16 vs L-q256 at pooled propagation give the C3 and C4 shares on CK2. Decision: whether the source encoding (s) must change, i.e. a directional state at the sender input.
- **D3, sampling dominant.** With exact q, pooled closes most of deployed-vs-REF while sub-bin / X1 add nothing beyond noise.
  - Decision: a binned reusable first flight suffices given adequate relation sampling. G2 / E-L stay viable; G1's continuous gather has no CK2 target.
  - The design question becomes relation placement and density (C3UR class), with the WL91-type per-draw rule.
- **D4, angular projection dominant (WL90-type on CK2).** Pooled q stays off but sub-bin a = 16 or X1 closes it.
  - Decision: a fixed 64-bin reusable flight has a CK2 floor. Any candidate must carry sub-bin or continuous sharp sender information through the first flight.
  - For the learned lineage, this decides between source-independent sender-only refinement (per-material operator, peaked senders only) and G1-type per-source evaluation. That choice needs the owner ruling of §7.
- **D5, beyond the first gather.** Even L-DQ-consistent L₂ leaves Q_r beyond noise and Q_d3 shows a large panel-mediated e3 residual.
  - Decision: the binding problem is the second panel flight (e3_p). No first-flight design (all of G1) addresses it, and G2 must specify a Λ_θ with flight geometry before it can be a hypothesis.
- **D6.** Pooled X1 ≈ L-DQ but both ≠ REF while V0 passed. This is logically impossible given V0. Treat it as a defect and STOP.

**Acceptance.** None. This smoke selects no architecture. It enables exactly the decisions D1–D5, together with the §7 owner ruling.

## 7 Unresolved Questions (exact decision gates)

1. **OWNER GATE A.** Is a learning-free sharp first order (G1 class) admissible as part of the method, or only as a comparator? Canon (Architecture.md:13; roadmap:128) currently implies comparator. Until ruled, G1 cannot be ONE_CANDIDATE.
2. **OWNER GATE B.** Would a G1-type estimator branch (dense or explicit flight for directly lit peaked senders; binned for C-mediated incidence) violate the spirit of Architecture.md:21? The response rule is shared, but propagation fidelity differs by event order. Gate B is about the C-side partition. Gate A is about admissibility as the method.
3. **EVIDENCE GATE C.** D1–D5 of the smoke.
4. **EVIDENCE GATE D.** A specified Λ_θ / learned global form must state a learning-unique claim with predicted headroom over H-0, and pass a per-geometry free-fit representability bound before any training. Without it G2 is not a hypothesis.
5. **Other open items:**
   - the e3_p (panel-mediated ≥ 3-event) ownership;
   - sharp movers, deformation, bending and self-contact (no workload; the dirty test is box-only);
   - area and environment lights;
   - the α → 0 limit;
   - an admitted held-out sharp workload;
   - the W-defect impact on learned and render-derived WL94-series numbers;
   - the event-count partition vs K/C placement of ECRO;
   - external Neural Full Recompute: UNMEASURED (mandatory gate not run).
6. **Numerical disagreements to resolve by geometry-only CPU computation before any freeze:**
   - the CK2 receiver distance d_x (designer 0.25 m vs critic 1.1–1.5 m);
   - expected lobe hits per footprint receiver patch (≈ 1 vs 0.16–0.4 vs ≈ 3–4 under different estimators).

**Rejected alternatives, with reasons:**

| Alternative | Reasons |
|---|---|
| **MEK** (learned equivariant lobe kernel) | Duplicates ER, which is already learned and exact; adds error with no information gain. Built-in azimuthal isotropy contradicts ER's anisotropic GGX [FACT]. GM0 ("no renewed local-closure search"). WL80C/80D/81 negatives. Not distinct from H1. |
| **ALP** (LTC lobe-polygon) | Estimator variant of G1. Scalar pair visibility is wrong in the gated visibility-change region. Quadratic, source-dependent pair table (the named large-table risk). Kill threshold transplanted from an unmatched (M64-binned) norm. Kept only as a diagnostic idea, not run in the smallest smoke. |
| **VSA** (virtual-source atoms) | Unit error (Φ where intensity is needed). A single atom cannot represent grazing incidence-dependent reflectance. Hidden per-frame readout tracing. O(L·P^d) growth. No measured target (WL90 P2plus ≤ 0.0009). |
| **G1 as the method** | Canon-barred absent Gate A. Repeats C3URX1's measured failure mode (single-draw variance, utility). Forfeits the source-update advantage. Addresses about 1/3 of panel-mediated CK2 transport. |
| **G2 as specified** | Learned part unspecified and predicted null. Receiver inconsistent with C3UR's dense cells. Prefix already measured at 0.0449 single draw on E2. |
| **Higher rank or more bins for the learned a cᵀ alone** | The learned operator is far below its rank-16 bound, so optimisation binds first. The outgoing projection stays. Must not be assumed to solve the problem. Retained only as the bound-operator diagnostic arm. |
| **Learned beam / C propagation** | WL52: cost, contact and transition locality. |
| **Learned sampling density** | Cannot recover never-hit relations; its status as learned transport is an open owner question. |
| **Per-frame retrace (LPTG) / DDT-type α⁻⁴ tables** | Withdrawn precedents (WL84A). |

## 8 Architecture Verdict

**ARCHITECTURE_SELECTION_PREMATURE.**

- **No candidate justified.** No surviving hypothesis is both admissible under the current canon and supported by evidence that discriminates it from H-0 or E-L.
- **The four "primary" recommendations are all G1.** Their four first-gather estimators are one learning-free comparator class. All eight critiques independently reach PREMATURE, and I concur.
- **What remains to be discriminated:**
  - G2 (the ECRO-lineage, ER-owned binned prefix plus a learned resolvent; not yet a falsifiable hypothesis);
  - E-L (a learned material-level sender; unspecified);
  - the G1 / H-0 comparator class.
- **The discriminating evidence is learning-free.** It is the §6 smoke plus owner Gates A and B.
- No design is adopted. No successor is authorised or started. No novelty is claimed.

**Living-document amendment.** A dated, append-only record is justified only when WL95 closes, because a batch closing is a trigger under AGENTS.md. The Architecture.md contract text stays unchanged, since Architecture.md holds contracts only and is not the place for batch verdicts.

PROPOSED text for `docs/RESEARCH_ROADMAP.md`, after the line-128 decision block. Nothing is adopted; this is for the owner:

> **WL95 (design-only, 2026-11-15): ARCHITECTURE_SELECTION_PREMATURE.** Four independent designs converged on one learning-free class: ER-continuous sender + explicit first gather (H-0 / C3URX1 lineage). Under the 2026-11-13 decision this class is a comparator, not the method. The ECRO-lineage alternative (ER-owned binned prefix + learned ≥ 3-event resolvent) lacks a specified learned form and a predicted advantage over explicit sweeps. Evidence-favoured constraint, not a fix: a geometry-conditioned learned rank-16 tied bin operator is not justified as the owner of S1 local angular response (matched-norm zero-operator level; S_geo < 1 %; rank-16 empirical ceiling). Open: the CK2 flight-floor decomposition, the delivery and E_H of bound (rank 16/32) senders, ownership of the panel-mediated ≥ 3-event family, and owner Gates A (learning-free first gather: candidate vs comparator) and B (event-order propagation asymmetry vs Architecture.md:21). A learning-free suffix-class attribution smoke is specified in the WL95 worklog. It is not authorised.

Separately, as worklog content and not living-document content: record the `an94.W` (8 × 8) and `delivery_box` (A + B) metric defects.

## 9 Reproducibility / Sources

- **Evidence files** (scratchpad):
  - `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\auditA.md`
  - `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\auditB.md`
  - `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_canon.md`
  - The digests `read_code.md`, `read_wl86_88.md`, `read_wl89_92.md`, `read_wl93.md` and `read_wl94.md` in the same folder, as cited by the designers and critics.
- **Verified this pass:**
  - `c:\Projects\ReST-GI\docs\Architecture.md:13, 21`
  - `c:\Projects\ReST-GI\docs\RESEARCH_ROADMAP.md:128`
  - `c:\Projects\ReST-GI\experiments\fnto_compiled83\closure83_dr.py:1-12` (eval = f·cosθ_o; anisotropic au/av)
- **Critic-verified code relied on:**
  - `experiments/gm1_c3urx1_91/cand91.py:1-110` (sender-side, exact per-ray L₁, probe + shadow)
  - `experiments/gm1_relation89/rel89.py:57-75` (fresh origin per ray; dense 8 × 8 receiver cells)
  - `experiments/gm1_explicit87/quad87.py:85-110, 268-338` (fbar is MC; qtable P = 16)
  - `experiments/ecro86a/pt86a.py:73-200` (e2_p / e3_p; material 1 only; patch × bin only)
  - `experiments/ecro86/ecro86.py:57-67` (rigid-OBB, box-only dirty test)
  - `experiments/ecro_mat94/an94.py:19, 32` (W defect)
  - `experiments/ecro_s94d/s94.py:152-162, 309` (CK2 floor; box slice)
- **Bound operators for the smoke:** `~/wl95_calc/S1_ops.npz` (LabServer63).
- **Inference only:** all cost and memory figures marked [INF] are arithmetic by designers or critics, not measurements.
