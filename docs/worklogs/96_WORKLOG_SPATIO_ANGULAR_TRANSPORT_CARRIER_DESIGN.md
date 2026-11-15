# Worklog — WL96: Spatio-Angular Transport Carrier Architecture Design (2026-11-15)

## Origin, scope and process

- **Brief.** "WL96 — Spatio-Angular Transport Carrier Architecture Design" (user).
  - It is a regular numbered, design-only batch.
  - Nothing was implemented or trained, and no renderer or GPU job was run.
  - No production code, checkpoint or canonical document was changed.
- **Process.** Independent design followed by adversarial critique, using the Agent tool. Workflow orchestration was not opted into for this batch.
  1. **Supervisor checks (CPU only).** The supervising session verified the algebraic correction stated in the brief (`results/wl96_calc/ck2_support.py`, `ring_alloc.py`, `ring_alloc.json`).
  2. **Four independent designers ran in parallel.** Their prompts held the same evidence pointers, constraints and verified facts, and differed only in the starting point:
     - designer 1: no starting point. A prompt-template slip delivered an unfilled `{ANGLE}` placeholder, the designer chose its own angle and said so; this is disclosed here;
     - designer 2: a physically conditioned directional-lobe state plus a learned multi-event operator;
     - designer 3: a source-independent reusable basis with source-dependent coefficients;
     - designer 4: a continuous position/direction response carrier.
  3. **Two independent adversarial critics** read all four reports and the scratch code:
     - one with a physics / representation / training lens;
     - one with a cost / scaling / explicit-comparator lens.
  4. **Synthesis and verdict** (this worklog) by the supervising session. It verified the cost anchors (WL91:228-246; WL87:175, 281) and the canon lines.
- **Agent outputs** are kept verbatim in [docs/plans/wl96_spatio_angular_carrier/](../plans/wl96_spatio_angular_carrier/) (`designer1_free.md`, `designer2_lobe_state.md`, `designer3_basis.md`, `designer4_continuous.md`, `critic_physics.md`, `critic_cost.md`). They are not adopted.
- **Scratch calculations** (CPU numpy only) are copied to `results/wl96_calc/scratch/{d1,d2,d3,d4,critic_phys,critic_cost}/`. They are design input, not frozen evidence.
- **Labels.**
  - [FACT]: code or doc.
  - [MEAS]: repository measurement.
  - [MEAS-S]: a designer or critic scratch number (numpy ER port, simplified geometry; not evidence of record).
  - [INF]: inference.
  - [HYP]: untested hypothesis.
- No novelty is claimed. Light tracing, photon splatting, directional radiosity, image sources, PRT bases and neural radiance caches are classical.

## Verdict

**NO_VIABLE_LEARNED_MECHANISM** — scoped and provisional.

- **Scope.** The scene86 / GM1 room family, point lights, rigid edits and α ≈ 0.03–0.18. The judgment covers the learned mechanisms that could be specified in this study.
- **The physics critic's reading is EVIDENCE_INSUFFICIENT.** It is recorded as the dissent in §E, together with the observation that would overturn the verdict.
- **What fixes sharp delivery.** In every design, the mechanism that fixes sharp first-flight delivery is learning-free and explicit: G1 class, a comparator under current canon.
- **The only learned element that survives** is one broad ≥3-event resolvent, written four ways. It carries no sharp spatio-angular information, and measured same-room economics cap its utility below the project's own 1.25× rule.

## A. Evidence reconciliation (WL95A)

### Solid

- **[MEAS] / [MEAS-S] The binned patch × bin sender table has an expected-value delivery bias that relation sampling does not remove.** Two independent deterministic area-form e2_p oracles were built:
  - designers 1 and 3: numpy ports of ER, panel quadrature at 4–5 mm, box A 54 patches;
  - both match the path-traced CK2 e2_p at ratio 1.015 / rel L1 0.068 (n0 0.059, corr 0.995).
  - With **no relation sampling at all**, they reproduce WL95A's binned over-delivery: M64 binned 1.486 / 0.487 vs WL95A pooled M64_spool 1.495 / 0.500; q-binned 1.420 / 0.428 vs q256 1.422 / 0.433.
- **[MEAS] / [MEAS-S] The X1 per-draw variance is a property of receiver-anchored gathering, not of the carrier.**
  - On CK2, the half-energy solid angle of the glint, seen from the brightest box-A patches, is 0.0056 sr. That is 0.46 expected relation rays per draw.
  - The exact, node-interpolated and fine-binned carriers read through ECRO's 512-ray gather all have equal variance within 4 %.
  - The binned table has about 3× lower variance because it smears the lobe.
  - This also settles the WL95 §7 "≈ 1 vs 0.16–0.4 hits" dispute: about 0.46.
- **[MEAS] Pooled X1 reaches the reference (1.022 / 0.111), and the learned S1 operators are non-passive (albedo 2.3 / 3.1).** Unchanged.

### Qualified

- **RANK16_DELIVERY = INADEQUATE.**
  - [FACT, supervisor check] CK2 panel incidence occupies only 4 ring-0 input bins (12–15). M64·P_ring0 has rank 16 per channel, and an exact non-negative rank-16 factorisation reproduces M64·s for any ring-0-supported s (max error 0).
  - [MEAS] WL95A's bound operators gave ring 0 6.4 % of the row weight in the flux norm; NMF16 ring-0 op L1 is 0.97 vs ring 3 at 0.40 (`results/wl96_calc/ring_alloc.json`).
  - [MEAS-S, physics critic] A CK2-incidence-weighted (light-conditioned) rank-16 operator reaches 0.062 on CK2 but 0.85 / 0.69 on two other lights.
  - [INF] So the verdict is about fit-distribution allocation and light-independence, not rank capacity. It is also **moot for design**: an exact rank-16 copy of M64 inherits M64's +42–54 % binned-outgoing bias.
- **ANGULAR_PROJECTION_EVIDENCED.**
  - [MEAS-S, d4] Holding the exact position but snapping the outgoing direction to the bin centre reproduces the over-delivery: 1.423 / 0.527.
  - [MEAS-S, d4] Holding the incident state at the patch centre while emitting with ER at the exact (x, ω) gives 0.113. X1c (the emission point fixed at the centre) gives 0.234.
  - [INF] So the dominant mechanism is outgoing-direction quantisation. WL95A's "spatial" step (X1c → X1) is mostly the emission anchor, not the incident state. A 5 cm incident lattice gives 0.008–0.011 (d1 NODE 50 mm; d4 CP4).
  - [MEAS-S] The binned bias depends on the light configuration, not only on grazing incidence: 0.05 (d4, moderate light) up to 0.14–0.22 (d1, d2, d3 non-grazing lights). d4's claim that "non-concentrated lights are uninformative" does not survive.

### Over-interpreted (corrected here)

- **[MEAS WL95A B3] Image-level relevance on CK2 is not demonstrated.** Pooled M64 E_H is 0.0245 against the oracle-table floor of 0.0118. The +0.0127 is below the 0.0167 tolerance of the 2 × 4,096-spp renders. Every design premise rests on delivery-level metrics.
- **The capacity arguments used against learned sharp operators prove less than stated.**
  - d1's signed continuous-kernel ranks (148 / 381 / 614 for 90 / 99 / 99.9 % energy), d2's per-sub-cell transfer rank (≥ 256) and d3's local rank (~500 at 1,024 bins) all bound a *linear, light-independent, separable, fixed-basis* form, in Frobenius norms that down-weight grazing rows.
  - [MEAS-S, physics critic] Re-weighting rows to unit outgoing flux halves d1's CK2 requirement: rank 128 → 0.118.
  - d1's "a non-negative fit cannot be better in that norm" is false. In WL95 §2.1, L1-NMF16 scores 0.380 against signed svd16 0.891.
  - These calculations falsify ECRO's u = c·s, Bq = a·u form at fine resolution. They do **not** prove that no compact learned representation can carry sharp structure.

## Verified facts used by the study

- [FACT] ECRO: s is the patch-mean direct irradiance per incident bin (`pt86.py:378-402`); u = c·s, Bq = a·u, rank ≤ 16, tied c (`model86.py:111-120`); 512 fixed state rays (`pt86.py:353-376`); global term rank 24 driven by u (`model86.py:121-124`). The generator never sees the light (`model86.py:91-107`).
- [FACT] ER: `eval(wi, wo) = f·cosθ_o`, Lambert + 3 outer + 4 refracted-inner GGX, exact sample/pdf (`closure83_dr.py:1-15`). S1: α_u ≈ 0.026–0.075, η_e 2.5, effectively isotropic (`mats86.npz` row 1).
  - All four designers' numpy ports were checked line by line against `closure82.py:121-146` by the physics critic: inner lobes, Fresnel, s(μ)·s(μ), validity masks and the cosine convention all match.
- [FACT] Canon:
  - ECRO lineage is the learned-global base, EXP/C3UR are comparators, and X1 findings do not justify explicit GI (`docs/Architecture.md:13`).
  - The next step must be one falsifiable ECRO-lineage hypothesis (`docs/RESEARCH_ROADMAP.md:128`).
  - No explicit per-surface path tracer hidden under a neural name (`docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md:86`).
  - Rule parameters are source-independent and baked radiance is excluded as runtime state (`docs/Architecture.md` §4 lifecycle rows).
  - No source-identity / direct–indirect *response* branch (`docs/Architecture.md:21`).
- [MEAS, verified by the supervising session]:
  - C3URX1 lifecycle: edit→frame 124.7 ms; edit without render 69.4 ms; source→frame 75.8 ms; source without render 21.4 ms; transport-order sweeps 10.6 ms (`docs/worklogs/91_…md:228-246`).
  - WL87 per-edit costs: q 26–30 ms (2 ms of it GPU); 4-order recurrence 10–11 ms; ECRO operator 11 ms.
  - Source-only update: ECRO 13–18 ms vs EXP 51–64 ms. EXP's q accounts for 42–54 ms of wall time but only 2 ms of GPU time (`docs/worklogs/87_…md:175, 281`).

## B. Independent candidate designs (complete contracts in the plan files)

**Two kinds of thing.** Eight named candidates reduce to two kinds (both critics agree):
- **K1.** Learned *sharp* carriers, which put the sharp first order inside a learned operator.
- **K2.** Explicit, learning-free sharp carriers, paired with one learned broad ≥3-event resolvent.

### K1 — learned sharp carriers (all rejected as learned methods)

| candidate | state / mechanism | reason |
|---|---|---|
| d1 **E-L+** | material-level learned separable kernel f ≈ Σ c_r(ω) a_r(v), read at the exact outgoing direction; ECRO interface (64-bin s, patch-constant u) | Duplicates ER (no information gain). Patch-constant u imposes the X1c-type emission-anchor floor (0.27) at any rank. 64-bin incident encoding error 0.24–0.51 (BIC64). Its capacity kill was mis-argued (§A) but is not needed |
| d2 **LSTO / C1** | 5 cm lobe atoms (exact light direction + irradiance) → geometry-conditioned learned transfer T_θ(G)_c(ω) to receivers | Learned C plus a re-learned ER. 2.6 GB of factors per geometry at rank 256. Per-light Σ E·T ≈ 6.6e8 MAC, more than the DQ alternative's ER evaluations. Dominated by explicit deterministic quadrature in memory, compute and accuracy |
| d3 starting point | source-independent learned transport basis Φ_k(G) with source coefficients | For Lambert receivers the sharp content is the reflected-visibility map K_j(y, ω). It is either a dense table (α⁻²·h⁻²), a re-learned ER (MEK class) or learned visibility (WL52 class) |
| d4 starting point | learned outgoing field L(x, ω) | If fitted to radiance it is a refit lifecycle (canon §4 rows). If computed from current incidence through ER it holds no learned information. The physics critic notes that a learned *response* operator emitting L(x, ω) is a third option; ECRO's B and d4's LIR are such operators |

### K2 — explicit sharp carrier + learned broad resolvent

Sharp-carrier variants. All are learning-free and G1 class; they are estimator and representation choices, not learned architectures.

| carrier | mechanism | [MEAS-S] CK2 e2_p | main cost and scaling |
|---|---|---|---|
| node / control-point incident state + exact ER emission (d1 NODE, d4 CIS) | incident (Φ, ω_ℓ) at 5–10 cm nodes; ER at the exact (x, ω) | NODE 100 / 50 mm: 0.038 / 0.011; CP4: 0.008 (noise-free carrier error) | needs a delivery estimator (below); shadow-edge interpolation untested |
| deterministic sender-cell × receiver-point quadrature (DQ; d2, d4) | exact ER at cell centres, one visibility per pair | 5 cm × 4²: 0.026; 1.25 cm × 4²: 0.0009 | pairs ∝ A_sharp·A_rcv, cells ∝ α⁻², pairs ∝ α⁻⁴. [MEAS-S, cost critic] 9.1 M front-facing S1 pairs; a 0.3 m box-A shift dirties **60 %** (≈ 5.5 M re-tests ≈ 60 ms) |
| light-side lobe-sampled photons (d1 relay, d3 LSC) | per light, emission from sender cells by lobe importance | 40 k / 160 k photons: per-draw 0.069 / 0.033, worst ≈ 1.5× the median (no heavy tail) | O(L·F_s·M) rays per light; M ∝ 1/(ε²·f_land·A_rcv); erodes ECRO's source-only advantage |
| fine sender table + sender-anchored relation (d3 ESC) | 10 cm × 1,024 bins, exact projection, fixed relation | r = 4 / 16: 0.070 / 0.035 (fixed-pattern bias) | 0.7–2.8 M rays per facet, 18.6–74 MB per facet at 27 B/ray; bins ∝ α⁻² (fails at α ≈ 0.01) |
| relay continuation through sharp vertices for e3_p (d2, d3, d4) | ER-sampled continuation of relation rays at peaked hits | [HYP] untested | source-independent; per-pixel camera continuation at sharp x2 is an explicit per-frame cost |

**Learned element.**
- Common form: Λ_θ (d1) = Λ_θ (d2) = R_θ (d3) = LIR (d4), a non-negative, rank 24–64, geometry-conditioned ECRO P/Q resolvent.
- It takes the exact 2-event field as input and owns ≥ 3-event transport whose last scatterer is broad.
- d4's LIR is the best specified:
  - it maps ECRO's event partition onto K/C (Λ ≈ Σ_{j≥1}(C K)^j C restricted to broad last vertices);
  - it requires a new event-exclusive incident tally;
  - it has gated preconditions (FB1 oracle incidence, FB2 rank feasibility, FB3 explicit comparator) before any training.
- [INF] What the learned element can express beyond explicit ER + transport: **nothing in expectation**. All four designers concede this. Its only claims are amortisation (cost) and a deterministic tail.

## C. Adversarial critique (strongest objections)

**Physics / representation.**
- **Ownership gaps** (physics critic §4):
  - d2: relayed e3_p sits inside b2 and also in Λ's e3 target, so it is double counted.
  - d1: incidence on the panel from broad senders is never relayed, so part of e3_p is unowned. d1 also supervises panel incidence through the biased M64_S1.
  - d3: R_θ keeps ECRO's learned a·cᵀ local operator, contradicting "ER owns every event".
  - All except d4: the existing `mid == 1` split cannot supply broad-last targets once S2 is also peaked under the Π rule. A new class tag is needed.
- **Passivity.** Non-negativity does not guarantee passivity of the resolvent. The WL95A learned senders were non-passive.
- **Oracle validation is one configuration.**
  - The CK2 grazing omni light, Lambert receivers, no shadow edge across the panel.
  - The other lights are oracle-only.
  - Agreement of 0.068 against n0 0.059 leaves up to about 0.03 of unresolved oracle bias.
- **Image-level relevance on CK2 is unresolved** (§A).

**Cost / scaling.**
- **Amdahl ceiling (cost critic §2.2; anchors verified).** Assume a perfect, zero-cost learned resolvent that replaces the whole 10.6 ms sweep stage. The best lifecycle ratios are then:
  - edit→frame 124.7 / 114.1 = **1.09×**;
  - source→frame 75.8 / 65.2 = **1.16×**;
  - edit without render 1.18×;
  - only "source without render" (1.98×) clears 1.25×, and no user-visible update omits the frame.
  - With an ECRO-class generator (about 11 ms per edit), the per-edit ratio is ≤ 1.
- **Accuracy headroom of the tail** (cross-workload): about 0.001 (WL90 P2plus ≤ 0.0009). WL87 learned fidelity was 0 better / 3 worse / 36 equal.
- **The only measured learned advantage is overhead-confounded.** The 3.5–3.9× cheaper source-only update is dominated by EXP's q table: 42–54 ms wall time against 2 ms GPU time.
- **Explicit carriers carry real costs that the designers underestimated:**
  - DQ / CIS per-edit dirty pairs are 60 % on the CK2 box shift (≈ 136–140 ms per edit, about 1.8× ECRO);
  - α⁻⁴ pairs;
  - per-light photon cost ∝ L·F_s;
  - ESC-R memory 18.6–74 MB per facet;
  - per-frame camera continuation;
  - a moving sharp object invalidates every cache of that object.
- **No designer smoke can discriminate learned utility with power.**
  - The cost rules are component-level, not lifecycle.
  - The 0.0167 render tolerance cannot resolve a 0.013 effect.
  - The qualified moving-sharp workload (WL89 W1/W2) is unused.

**Missed mechanisms (physics critic §6).**
- Light-conditioned learned delivery for sharp senders (learned C). Admissible as a response operator, but untested, with a negative WL52 prior.
- Learned proposal / guiding for the receiver gather (low value; photons already beat the analytic proposal).
- An LTC-type analytic lobe carrier (not learned).
- Learned incidence at sharp cells for e3_p: the only place a learned output feeds sharp structure, through ER, at plausibly low rank. Untested.

## D. Explicit-comparator analysis

- **Avoided work vs explicit X1 / C3URX1.**
  - The explicit carriers avoid receiver-anchored, heavy-tailed gathering: single-draw C3URX1 needs 0.53 M + 0.53 M rays per light, and pooled X1 needs about 1.28 M per light plus relation re-draws.
  - Photons reach per-draw 0.03–0.07 with 40–160 k rays per light per facet. DQ is deterministic with no per-light rays, but is geometry-quadratic.
  - **None of this is learned.**
- **Learned methods (K1).** The actual function a learned sharp component must learn is either ER (already exact and compact: 69 floats) or the reflected-visibility map (learned C). The first adds approximation error with no information gain. The second has a negative prior on cost and transitions and an α⁻²-growing content.
- **Learned resolvent (K2).** The function learned is the ≥ 3-event broad tail. Measured same-room economics cap its benefit below 1.25× lifecycle, and its accuracy headroom is about 0.001.
- [INF] **Conclusion.** Learning fails where explicit is expensive (many sharp surfaces and lights, small α: the content is not compact) and is redundant where explicit is cheap. No candidate makes a supportable claim that the learned method is preferable to explicit ER / transport computation.

## E. Architecture judgment

**NO_VIABLE_LEARNED_MECHANISM** — scoped and provisional.

- Learned **spatio-angular** carriers (K1) are excluded or dominated: duplicating ER, learned C, or non-compact.
- The single falsifiable learned element (K2's broad resolvent) is not a sharp carrier and is predicted decorative by verified same-room measurements.
- The explicit carriers do fix sharp delivery and are viable **comparator / oracle** classes. Adopting any of them inside the method is owner **Gate A**; their delta-vs-extended carrier split is **Gate B** (`Architecture.md:13, :21`). Both are outside this study's authority.

**Dissent (physics critic): EVIDENCE_INSUFFICIENT.**
- The incapacity proofs cover only linear, light-independent, separable forms.
- Sharp learned C and the resolvent's utility are unmeasured.
- Image-level relevance on CK2 is unresolved.

**What would overturn the verdict.**
- The §F smoke finds accuracy headroom H_acc above its bound in a sharp-reader region on two workloads, which would make it ONE_FALSIFIABLE_CANDIDATE (the learned resolvent).
- Or it finds a lifecycle ceiling U_max ≥ 1.25 at the largest N, which would make it a COST_ONLY candidate.
- Or the owner commissions a sharp learned-C test that beats DQ on cost at equal fidelity.

## F. Next feasibility experiment (specified, NOT authorised)

**Learned-utility ceiling and sharp-prefix relevance smoke.** This merges both critics' proposals.
- Learning-free, no training. Comparisons are internal only; Neural Full Recompute stays UNMEASURED.
- **Batch class.** A dependent follow-up of WL96 (e.g. WL96A); the owner decides.

**Workloads**
- CK2 "none" with the T0 / T1 box edit (existing CKF split).
- One non-grazing CK2 light. It must be prequalified **before any reference** by a geometry-only DQ rule (binned-vs-exact ≥ 0.15) and by SNR ≥ 4. It needs one new 2 × 4,096-spp `field_src` reference with a peaked-class (S1 + S2) tag.
- The WL89 W1 rotating-S1-slab workload, if its family-tagged split exists. Verify before freeze; otherwise declare the smoke CK2-only.

**Stages**
1. **V0 contract (STOP gate).** A Dr.Jit deterministic e2_p oracle with real scene visibility vs path-traced e2_p on every PT-referenced light: |ratio − 1| ≤ 2 SE and rel L1 ≤ n0 + 2 SE.
2. **Sharp-prefix relevance.** Compare E_H of:
   - {oracle e2_p + PT rest};
   - {pooled binned M64 / q e2_p + PT rest};
   - the oracle-table floor.

   The render spp is fixed at freeze so the tolerance falls below the predicted 0.0127 difference; 2 × 4,096 spp cannot resolve it.
3. **Learned-utility ceiling (broad-last family only, so a learned tail cannot absorb sharp structure).** All arms take the oracle 2-event field as their prefix.
   - Arms:
     - PT e3 floor;
     - explicit J = 4 sweeps (M64 local operators, R0 and sender-side relations);
     - an in-sample non-negative rank-ρ ∈ {24, 64} fit of the tail on replica half A, scored on half B. It is labelled a feasibility probe, not a bound;
     - the frozen OWN global term (descriptive).
   - Accuracy H_acc = err(explicit) − err(fit), at field level per region and in E_H.
   - Cost: sweeps vs Λ apply vs ECRO `factors`, warm and uncontended on the same GPU, with a full-scale utilisation probe, at N = 1,270 / 2,802 / 5,080. Then U_max = lifecycle / (lifecycle − t_sweeps + t_Λ [+ t_generator per edit]) for edit→frame and source→frame.

**Frozen decision mapping**

| condition | result |
|---|---|
| V0 fails | STOP |
| U_max < 1.25 for all N and H_acc ≤ max(2 SE, n0 / 2) in every region | NO_VIABLE_LEARNED_MECHANISM confirmed on the tested class; explicit carriers → comparator role; Gate A to the owner |
| H_acc above the bound in a predeclared sharp-reader region on two workloads | ONE_FALSIFIABLE_CANDIDATE (learned resolvent; d4 LIR specification; training needs separate authorisation) |
| U_max ≥ 1.25 at the largest N, H_acc within the bound | COST_ONLY scaling hypothesis |
| Stage 2 unresolved at the declared spp | the sharp-carrier choice has no image-level target on CK2 (reported) |

**Separable, optional (comparator selection, not architecture).** DQ g4 vs photons 40 k / 160 k vs ESC-R r4 on the same workloads, scored against PT e2_p (not the in-house oracle), with dirty-cache counts and ms per edit for the box edit and the slab rotation.

**Budget [INF].** Under about 1 GPU-hour, plus at most two new PT field references.

## Unresolved owner decisions

- **Gate A.** May an explicit sharp carrier (DQ, photons, fine table, relays) be part of the ReST-GI method, or only a comparator / oracle? Canon currently says comparator.
- **Gate B.** Is a carrier split by sender class and by delta vs extended incidence (same ER response everywhere) compatible with `Architecture.md:21`?
- **Smoke.** Whether to commission the §F smoke, and whether its WL89 W1 arm requires a new split.
- **Sharp learned C.** Whether to commission a sharp learned-C test at all. It is untested, with a negative WL52 prior, and was not proposed as a candidate here.

## Proposed living-document amendment

None. No architecture is adopted or superseded.

A dated roadmap record of this batch's closure is added, per the AGENTS.md batch-closing rule. `Architecture.md` is unchanged.

## G. Reproducibility and provenance

- **Supervisor CPU checks.**
  - `results/wl96_calc/ck2_support.py` gives the CK2 panel support bins 12–15, all ring 0. It uses numpy geometry, unshadowed (an upper bound), with the 16 deployed source points from rng 86001.
  - `results/wl96_calc/ring_alloc.py` writes `ring_alloc.json` (rank 16 per channel; exact non-negative rank-16 factorisation, error 0.0; ring allocation of A1–A3). Its inputs are `results/ecro_attr95a/ops95a.npz` and `ck2_support_E.npy`; random check inputs use rng 96000.
- **Designer and critic scratch** (`results/wl96_calc/scratch/`, numpy only; seeds as stated in each report):
  - d1: `deliv96.py` (oracle and carrier ladder), `var96.py`, `rank96.py`, `photon2.py`;
  - d2: `truth_area.py`, `detquad.py`, `rank_transfer.py`;
  - d3: `toy96.py`, `toy96b.py`, `photon.py`, `fbc.py`, `rankM.py`;
  - d4: `ck2_model*.py`, `ext_incidence.py`, `lobe_fit*.py`;
  - physics critic: `rankw.py`;
  - cost critic: `dirty.py`.
- **Inputs** (read only): `experiments/ecro86/data/mats86.npz`, `results/ecro_attr95a/{ops95a,deliv}.npz`, `experiments/ecro_local93d/addendum_checker/scene_ck2.json`, `experiments/ecro86/scene86.py`.
- **Agent reports, verbatim:** `docs/plans/wl96_spatio_angular_carrier/`.
- **Environment.** Local PC, Python 3.11 / numpy 2.4.6. No server or GPU was used.
