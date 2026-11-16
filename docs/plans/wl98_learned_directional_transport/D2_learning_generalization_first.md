# WL98 — Designer D2 report: learning and generalization first

**Role.** Independent designer D2 of the design-only batch WL98 (learned directional transport representation after ECRO). Starting angle: learning and generalization first. Written 2026-11-16.

**Status.** Design input only. Nothing was implemented, trained, rendered or adopted. Nothing here is an architecture decision, an experiment approval or a change to any frozen result or verdict.

**Independence.** I did not open `DESIGN_REVIEW.md` or any other file in `docs/plans/wl98_learned_directional_transport/` except this report. I read the canonical documents, the WL97 audit (README, COORD, J1, J2, J7, F0) and the WL86 / WL58 / WL57 / WL96 / WL96A worklogs selectively, and the ECRO code (`experiments/ecro86/model86.py`, `scene86.py`, protocol `ecro86.json`).

**Labels.**
- [MEAS] measured, with its source.
- [AUDIT] the WL97 audit's interpretation.
- [PRED] my prediction or argument (not measured).
- [SCRATCH] my CPU numpy calculations in `experiments/wl98_calc/D2/` and `results/wl98_calc/D2/`. These are design input, not evidence of record.
- [HUM] a decision that belongs to the human owner.

Literature references are from memory and were not checked in this batch: [LIT-unverified].

---

## 0. Summary

**Proposed verdict: DESIGN_READY_FOR_HUMAN_REVIEW.** It is conditional on the human decisions in §8. The verdict is not adoption and not experiment approval.

**Main finding from the learning angle.** A learned transport component earns its place only when four conditions hold together:
1. It carries something the runtime cannot compute exactly at the target cost.
2. It is conditioned only on quantities that are invariant under the edit it is meant to survive.
3. Its training measure covers its whole input space by design.
4. It beats its own learning-free twin. The twin has the same output interface, ownership and physics, with the learned inference replaced by explicit computation.

The ECRO lineage violated conditions 1 to 3 at the same time:
- Its learned local factors re-learn the frozen, exactly known ER.
- Its generator conditions on scene-global geometry, which it then largely ignored.
- It is a light-independent linear operator whose input space was sampled only through random point lights. The grazing ring-0 sharp configuration appears in 0.11 % of training lightings.

Condition 4 was then decided against it by the same-state explicit twin (WL87, WL96A).

**Structural consequence for GM1 [PRED].** In the scene86 / GM1 room family every object is planar or convex and every surface is the pointwise ER closure. Intrinsic transport therefore equals ER, which is known, and every spatially distributed transport path runs through editable free space, which is C and depends on geometry.

A learned element in that family can be only one of three things:
- **(a) a re-learned ER.** This is BRDF compression, which the brief excludes, and the WL93D / WL95A failure mode.
- **(b) a learned C.** WL52 measured this to be slower than explicit GPU C for broad senders [MEAS]. For sharp senders I expect it to be dominated by ray casting [PRED].
- **(c) an amortized geometry-conditioned resolvent.** Its zero-cost edit utility ceiling on the WL96A lifecycle is 1.02× for the tail alone. It is 1.27× only if it also owns the prefix, which means re-learning ER. At the measured learned inference cost these fall to 0.92× and 1.12× (§4).

No candidate in this family meets the brief's suitability criteria. This is a scoped design finding and not a class-level result. It is not measured until the learning-free Stage G0 of §7 has run.

**Where the central hypothesis is testable within canon.** It is testable in intrinsic, rigid-invariant, nonlocal transport. That means FNTO's own K in optical domains with internal transport: scattering media bounded by rough dielectric interfaces.

**Primary candidate: LIEM-S, the Learned Intrinsic Exit Measure with structured lobes.**
- It is one shared network. For an entry event (x_i, ω_i) in object-local coordinates, it outputs a normalized sub-probability measure over (exit point on the domain boundary, internal direction at the exit).
- The exact rough-dielectric interface closure turns the internal direction into the sharp exit direction.
- Training is event-likelihood supervision on physically sampled internal random walks, from a declared impulse measure that includes a grazing quota.
- Passivity holds by construction. Rigid reuse is exact by construction. C stays explicit.
- What learning buys over its twins is the falsifiable core:
  - against the explicit random walk: cost;
  - against the analytic shared diffusion rule: accuracy;
  - against the per-asset tabulated K: memory, angular continuity and no per-asset precompute.

**Alternative: LIEM-B.** It has the same ownership and supervision, with a hierarchical sparse multi-resolution categorical exit distribution. It is a *representation-level* alternative. It is not mechanism-independent, and **no mechanism-independent alternative survives**.

**Rejected, with complete contracts (§6).** The rejected designs are:
- the ECRO-lineage continuous directional kernel (A);
- the ECRO-lineage sparse multi-resolution table (B). Its learning-free twin is retained as a control;
- the ECRO-lineage learned lobe (C);
- the rigid-cluster certified self-transport, which conflicts with canon;
- learned sharp delivery (learned C).

**Main risks of the primary candidate, stated up front.**
1. **[SCRATCH] Tension between cost headroom and sharpness.** On a toy slab, regimes with long internal walks (27 to 50 events per exit) have no angular structure above Monte Carlo noise. Regimes with high-frequency exit structure have only about 2 to 12 events per exit (§5.2). Workload admission must find configurations with both properties, or the candidate stops before training.
2. **Not in the human-selected lineage.** It sits outside the ECRO lineage and in GM2-type workloads.
3. **Prior art.** It belongs to the learned-BSSRDF prior-art class [LIT-unverified: Vicini, Koltun, Jakob 2019]. The novelty discipline does not let this alone carry a claim.
4. **Reference.** A reference for refractive-interface media is not yet qualified. WL53 validated only index-matched media.

---

## 1. Why the ECRO lineage failed *as a learned system*

Each item gives the measured fact first and then the learning interpretation.

**L1. It learned what was already known.**
- [MEAS] ECRO's S1 sender is `Bq = a·(c·s)`, with `a` and `c` generated per patch by the network (`model86.py:111-120`). Its target for that stage is ER applied to the direct field, and ER is the frozen, exact WL84 closure.
- [MEAS] The learned S1 operator is about 80 % wrong against frozen-ER M64 in a geometry-common way (WL93D).
- [MEAS] Field rel L1 is 1.04 / 1.12 at CK2 (J1).
- [MEAS] 10 to 16 % of S1 rows have directional albedo above 1, in 14 checkpoints. The maximum is 2.70 / 2.21 for the WL86 deployed checkpoints (J1-c).
- *Learning reading.* Where the target is an exact cheap function, a learned approximation can only add error. No amount of coverage makes it better than the exact function. Learning had negative value on this stage by construction.

**L2. Its input space was never sampled by design.**
- [MEAS] The operator is light-independent and linear over 1,270 patches × 64 incident bins. Its training signal came from random point lights in one room family, 1,024 geometries × 4 lights (WL86).
- [AUDIT, J7 / B-04] Ring-0 grazing panel-to-box sharp transport occurs in 4 of 3,712 OWN lightings (0.11 %). Ring 0 carries 4.3 % of training irradiance (WL95A:120).
- [AUDIT, J1] Under the all-incidence rank budget, fitting the grazing rows trades off against the others (zero-sum). A non-negative rank-16 operator that is exact on CK2 exists. It costs a training row L1 of 0.42 → 0.54.
- *Learning reading.* For a linear operator, "coverage" is the training mass on each operator column (incident position × direction). That mass was an uncontrolled by-product of the light sampler. The parts of the operator that the light distribution does not excite are fixed by inductive bias, not by data.

**L3. Seed-signed errors are the signature of unconstrained extrapolation.**
- [MEAS] Box-A delivery is 0.52× vs 1.63× of the reference for the two seeds (J7).
- [MEAS] Seed drift is 26 to 32 %; geometry drift is 4 to 8 %, or 9 to 14 % on edit pairs (WL93D).
- *Learning reading.* In an under-covered region, different initialisations land on different extrapolations. The fix is to cover that region, or to use an output family whose extrapolation is physically constrained (normalized measures, exact interface physics). More regularisation would not fix it.

**L4. Sum-only supervision of a composed product is not identifiable.**
- [MEAS] The single-flight term delivers 1 to 18 % of the light → panel → patch family, and the global term compensates without geometry (WL86A).
- [MEAS] Component supervision (OWN) recovered ownership but not a consistent held-out sharp advantage (WL92A / 92B).
- *Learning reading.* Each learned owner must have its own physically supervised target, and the targets must be disjoint.

**L5. The geometry conditioning bought little.**
- [MEAS] The geometry-varying share of the squared error is 0.3 to 0.8 %. Swapping in another geometry's operator barely changes transport (WL93D).
- [PRED] The edit response that ECRO did show (passing slope / corr gates on E1, E2, E5) came mainly from the explicit parts: LOW, the source encoding, and the learned flight through current-visibility state rays. It did not come from geometry generalisation of the learned factors. The model was in effect a scene-family-average local operator routed through explicit relations.

**L6. Its twin was never beaten.**
- [MEAS] The explicit same-state table was ≥ ECRO beyond noise everywhere on E1 to E3 (WL87; 33 of 36 within noise, J7).
- [MEAS] The one measured learned advantage was a source-only update, 3.5 to 3.9× cheaper (WL87). That is a relighting property, not a geometry-edit property.
- [MEAS] On the only qualified never-inspected held-out, E5, learned ≤ explicit at T1, within about 1 noise bound on one draw (J7).

**L7. Evaluation could not separate the hypotheses.**
- [AUDIT] Two seeds of one recipe throughout.
- [AUDIT] Single deployed state draws decided labels. FLIGHT_ATTRIBUTION is ANGULAR in 23, MIXED in 31 and SAMPLING_DOMINANT in 10 of 64 draws (J3).
- [AUDIT] CK2 is post hoc and hand-iterated, and was the sole quantitative workload of WL94 to WL96A.
- [AUDIT] M64 was used as "exact" although at grazing it over-emits by +30 % panel flux (J2).
- [AUDIT] No reference-validated edit was evaluated after WL93D (F0-03).
- [AUDIT] Training learnability was tested only by 1,000 to 2,000-step smokes (J7 / B-09).

**What this implies for H1 / H2 / H3** (the brief's split):
- **H1 (coverage / optimisation)** is the leading explanation for the learned sender error at CK2 [AUDIT J1: "identification / optimisation / generation, not rank"].
- **H2 (angular representation)** is independently established for the fixed 4 × 16 solid-angle-mean table at grazing incidence [MEAS J2; WL58 for 256 bins at α 0.04]. Note that **H2 binds the explicit table too**: it is a property of the carrier, not of learning.
- **H3 (no edit utility)** is measured only for the resolvent-only role at N 1,270 on CK2, statically [AUDIT P0-4].

The three are confounded in every learned arm so far, because each arm used the 4 × 16 carrier, the light-sampled coverage and the same explicit relations.

---

## 2. What makes a learned transport rule reusable without retraining (design principles)

These principles are the falsifiable core of every candidate below. A candidate that cannot state its compliance with each one is rejected.

- **P1. Learn only what the runtime cannot compute exactly at the target cost.**
  - Never re-learn ER or any other known closed-form local physics.
  - Never re-learn current visibility. Ray casting is exact and hardware-accelerated.
  - The learned target must be a quantity whose explicit evaluation is expensive (long internal walks, dense multi-bounce quadrature) or noisy at the deployed budget.
- **P2. Condition only on edit-invariant quantities.**
  - For rigid reuse, every input lives in the object-local frame: intrinsic geometry features, material / medium descriptor, and entry position / direction in the local frame.
  - No world coordinates, scene tokens, other objects, lights or source identity.
  - Equivariance must then hold by construction and is tested exactly, to float tolerance. It is never learned.
- **P3. Design the training measure over the operator's input space, explicitly.**
  - For a linear operator, train on impulses (columns): entry position × entry direction × descriptor, drawn from a declared measure with declared strata (grazing quota, edges, thin regions).
  - Coverage then becomes a number that is known before training, and it is a treatment variable.
  - Scene-induced incidence is a *test* distribution, not the training distribution.
- **P4. Train on unbiased physical samples.**
  - No binned teacher tables. M64 is never ground truth, by the brief and per J2.
  - Use event-level likelihood or unbiased estimators of the exact physical quantity, generated by code independent of the evaluated runtime.
- **P5. Make energy a property of the output family, not of a penalty.**
  - Sub-probability measures: mass ≤ 1 times a normalized distribution.
  - Composed stability then follows from sub-stochastic K and C.
- **P6. Give each learned component a disjoint, separately supervised path family.**
  - No sum-only supervision of a product of learned factors.
- **P7. Name the learning-free twin before training and beat it.**
  - The twin has the same output interface, ownership, physics, workload and budget, with learning replaced by explicit computation (or by a learning-free shared analytic rule).
  - If the twin wins at matched cost and quality, the learned component is decorative.
- **P8. Make generalisation testable with held-out factors that the model never conditions on by identity.**
  - Hold out shapes, descriptors, entry strata and workloads.
  - Separate train, validation and test by those factors before results.
  - Never evaluate a learned arm only on a post hoc diagnostic workload such as CK2.

**What the model must be conditioned on (P2 applied to intrinsic transport):**
- the entry point in object-local coordinates;
- the entry direction in the local tangent frame;
- intrinsic local geometry features computed from the object's own mesh: thickness along the normal and along the refracted ray, distances to edges and corners within the support radius, curvature;
- the material / medium descriptor.

**What must be invariant:**
- the object's rigid pose (exact);
- tessellation (query by continuous surface point, never by patch index);
- the source (linearity by construction);
- everything outside the domain (the domain's intrinsic response does not see external geometry; C does).

**What learning buys, in principle, over a learning-free twin:**
- **(i) Amortisation.** One network evaluation replaces an internal walk of many events, or a multi-sweep solve.
- **(ii) Generalisation without precompute.** A shared rule serves unseen assets and descriptors with no per-asset table.
- **(iii) A continuous, normalized output family** in place of a binned table, at far lower memory.
- **(iv) Variance by bias.** A deterministic learned output has no per-query Monte Carlo variance; the twin's variance at the deployed budget is the comparison.

Each benefit is measurable against a named twin (§7).

---

## 3. The GM1 surface family has no rigid-reusable learned object [PRED, scoped]

- [MEAS] Fixture facts, from `scene86.py` and `ecro86.json`:
  - the room is planar walls / floor / ceiling, a planar S1 panel, a convex S2 sphere and two convex Lambert boxes;
  - every scattering event is the pointwise ER closure, or Lambert;
  - a convex or planar surface cannot see itself.
- [AUDIT F0 §7.3] Surface-only scenes in which every K is the pointwise ER: the FNTO learned object (nonlocal intrinsic K) is not exercised.
- [PRED] So under the canonical K / C ownership:
  - intrinsic transport in this family is exactly ER;
  - every ≥ 2-event path is a composition of ER and free-space C.

A learned element can then only be:

| learned element | violates | measured evidence |
|---|---|---|
| a re-learned local response (ECRO's `a c^T`, a learned S1 lobe) | P1; the brief's "not mere BRDF compression" | WL93D / WL95A / J1: large error, non-passive rows |
| a learned flight / delivery (learned C) | P1 (visibility is exact and cheap), P2 (conditions on scene geometry) | WL52 [MEAS]: valid and passive, but slower than an efficient explicit GPU C (C path 0.18× its speed; update 8–19× slower, 674 vs 11 ms at 40k ports), and fails contact / transition locality (broad Lambertian senders). Sharp senders are untested, but a narrow beam's visibility is a ray query [PRED] |
| a geometry-conditioned resolvent (amortised multi-bounce) | P2 (scene-conditioned; weights are reused, state is not) | WL96A: no accuracy headroom on CK2 (table space, oracle prefix) and a cost ceiling of ≤ 1.04 for the resolvent-only role |

This is a design argument. The audit is right that it is not a tested class-level result (F0-04 / COORD §3). Stage G0 (§7) turns it into a learning-free measurement on admitted workloads before any ECRO-lineage training. If G0 refutes it, the ECRO-lineage arms reopen.

---

## 4. Utility ceilings of the ECRO-family roles on the WL96A lifecycle [SCRATCH arithmetic on MEAS inputs]

**Inputs.**
- The WL96A recorded walls: edit 144.1 ms = render 59.5 + geometry 30.2 + q table 25.7 + relation 23.1 + prefix 2.0 + tail 3.3; source update 88.5 ms (WL96A:132-153; F0-06).
- WL86's learned per-edit cost: source encoding 5 ms + operator 11 ms. The 11 ms includes a CPU cast (J5), so it is an upper value.
- Script: `experiments/wl98_calc/D2/amdahl_ceiling.py`. Output: `results/wl98_calc/D2/amdahl_ceiling.json`.

**Ceilings** (U = explicit time / learned-variant time):

| learned role (what it removes) | edit, zero cost | edit, measured learned cost | source, zero cost | source, measured learned cost |
|---|---|---|---|---|
| resolvent only (tail) | 1.023 | 0.919 | 1.039 | 0.979 |
| prefix + tail (q table, first order, tail): **re-learns ER** | 1.274 | 1.116 | 1.508 | 1.384 |
| + relation update: **learned C** | 1.601 | 1.359 | 1.508 | 1.384 |

Without the frame render, the edit-row zero-cost / measured values are 1.04 / 0.87, 1.58 / 1.22 and 2.77 / 1.82.

**Reading [PRED].**
- The only role whose ceiling exceeds 1.25 at measured cost is the one that also replaces the relation update, which is learned C.
- Prefix ownership reaches 1.27 only at zero learned cost. It requires re-learning ER, which is the measured failure.

**Caveats.**
- This bounds *this* prototype: N 1,270, one light, an overhead-dominated lifecycle (GPU kernel time 25.5 of 144 ms, F0-06).
- Scale (N up to 11,430 tested only for resolvent-only by E5), depth J > 4 and multi-light are not covered. That is why G0 re-measures the ceilings.
- **1.25 is a historical WL96A rule, not a WL98 threshold [HUM].**

---

## 5. Scratch evidence on angular representation and on the learned-value tension

### 5.1 Angular capacity on the frozen S1 ER once-scattered lobe [SCRATCH]

**Setup.**
- Script: `experiments/wl98_calc/D2/angular_capacity.py`. Output: `results/wl98_calc/D2/angular_capacity.json`, with a `.log`.
- Target: g(ω_o) = f(ω_i, ω_o) cos θ_o of `mats86` S1, the scene86 panel material, evaluated with the WL97 Track A numpy ER port (read-only import).
- Grid: 512 × 2,048 equal-area directions.
- These are **per-incidence best-fit (capacity) numbers. They say nothing about learnability.**
- They are lobe-level metrics. Delivery-level errors on a receiver differ (J2: about +40 to 50 % at box A for 4 × 16 sa).

**Metrics.**
- rel L1 of the flux density g (delivery-relevant);
- rel L2 of radiance (readout-relevant);
- flux ratio.

**Lobe-level rel L1 of g, with the stored values per incidence in brackets:**

| cos_i (half-max sr) | 4×16 sa [64] (flux) | 4×16 cos [64] | 8×32 cos [256] | 32×128 cos [4,096] | quadtree [256 / 1,024 / 4,096] | Gaussian mix K = 1 / 2 / 4 [12 / 18 / 30] | Student-t (ν 2) mix K = 1 / 2 / 4 |
|---|---|---|---|---|---|---|---|
| 0.025 (4.2e-4) | 1.65 (1.46) | 1.27 | 1.08 | 0.71 | 0.47 / 0.11 / 0.03 | 1.02 / 0.59 / 0.35 | 0.87 / 0.46 / unstable |
| 0.05 (4.7e-4) | 1.65 (1.45) | 1.28 | 1.08 | 0.69 | 0.44 / 0.11 / 0.03 | 1.03 / 0.58 / 0.34 | 0.85 / 0.45 / 0.39 |
| 0.10 (1.2e-3) | 1.40 (1.20) | 1.25 | 1.03 | 0.59 | 0.35 / 0.12 / 0.04 | 0.94 / 0.49 / 0.27 | 0.76 / 0.36 / 0.37 |
| 0.25 (3.3e-3) | 1.10 (1.02) | 1.09 | 0.79 | 0.29 | 0.23 / 0.10 / 0.05 | 0.63 / 0.30 / 0.17 | 0.43 / 0.20 / 0.17 |
| 0.40 (1.1e-2) | 0.98 (1.01) | 0.98 | 0.61 | 0.16 | 0.18 / 0.08 / 0.04 | 0.41 / 0.19 / 0.10 | 0.20 / 0.11 / 0.11 |
| 0.80 (2.8e-2) | 0.81 (1.02) | 0.80 | 0.38 | 0.11 | 0.14 / 0.07 / 0.03 | 0.35 / 0.12 / 0.04 | **0.04** / 0.04 / 0.05 |

At grazing incidence the mixtures also overshoot peak radiance by 2 to 7× (radiance rel L2 0.6 to 2). "Unstable" means a K = 4 t component collapsed below the 512 × 2,048 grid resolution: a fitting failure, not a capacity statement.

Lobe-level L1 measures the outgoing distribution itself, which is what a camera reading the panel or a receiver in the glint core sees. Delivery to an extended receiver integrates over it and is much less sensitive: J2 measured 8 × 32 within 2 to 6 % at box A [MEAS].

**Reading.**
1. **[SCRATCH, consistent with MEAS J2 / WL58] Fixed piecewise-constant bins fail at grazing incidence at any affordable resolution.**
   - The S1 half-max solid angle at cos_i 0.025 is 4.2e-4 sr.
   - 4 × 16 with the solid-angle-mean convention over-emits flux by 1.46× on the lobe, so the "about half convention" share of J2 also appears here.
   - Even the flux-conserving 32 × 128 table (4,096 values) keeps a lobe L1 of about 0.70.
2. **Adaptivity, not learning, removes most of the projection loss.** A flux-conserving quadtree reaches lobe L1 of about 0.11 with 1,024 leaves and 0.03 with 4,096 at grazing. The adaptive table is learning-free (§6.4).
3. **A "learned lobe" representation is only as good as its lobe family.**
   - At non-grazing incidence (cos_i 0.8) a single heavy-tailed (t, ν 2) lobe with 12 values reaches 0.04 lobe L1. That beats the 4,096-value 32 × 128 table (0.11), which is where type C shines.
   - At grazing incidence the same generic Gaussian / Student-t mixtures, fitted by weighted EM in a fixed gnomonic chart with K ≤ 4, stay at about 0.35 to 0.6 lobe L1 and overshoot peak radiance by 2 to 7×. K = 4 t-mixtures were numerically unstable at cos_i 0.025.
   - The cause is the horizon-truncated, multi-lobe coated-conductor shape: outer and inner GGX lobes all pile up at the horizon at grazing.
   - The exact structured family for this lobe is ER itself: 7 GGX lobes + Lambert, zero error, frozen and available. **For the once-scattered sender, a learned lobe is therefore redundant (P1).**
   - For any learned C-type output, the sharp part must come from an exact physical closure, and the learned part must live where no horizon truncation occurs. This is why LIEM-S places the learned direction distribution in the transmittable internal cone and applies the exact interface closure (§6.1).

The fits are mine and could be improved by direct L1 optimisation. The table bounds what an EM-fitted generic family achieves. It is not the family's true optimum.

### 5.2 Learned-value tension for intrinsic media [SCRATCH, toy]

**Setup.**
- Script: `experiments/wl98_calc/D2/slab_tension.py`. Output: `results/wl98_calc/D2/slab_tension.json` / `.log`.
- An infinite homogeneous slab with smooth dielectric faces (η 1.5), HG phase function, grey, float64, 600,000 photons per case.
- Collided exits only. Ballistic is a separate family.
- Angular high-frequency proxy: the best flux-conserving 4 × 16 projection loss of the exit radiance (32 × 128 histogram), next to a **noise-floor control**: the same weights on Lambertian directions.

| albedo, g, τ, cos_i | mean events per collided exit | 4 × 16 loss, entry side (noise floor) | 4 × 16 loss, far side (noise floor) | RMS lateral exit (mfp) |
|---|---|---|---|---|
| 0.999, 0, 10, 0.9 | 46.7 | 0.073 (0.069) | 0.131 (0.131) | 7.8 |
| 0.999, 0.9, 10, 0.1 | 49.9 | 0.096 (0.094) | 0.087 (0.081) | 25.3 |
| 0.99, 0.9, 10, 0.9 | 32.7 | 0.104 (0.103) | 0.092 (0.082) | 18.5 |
| 0.999, 0.9, 1, 0.1 | 11.6 | 0.203 (0.104) | 0.277 (0.084) | 11.7 |
| 0.99, 0.9, 1, 0.1 | 9.7 | 0.207 (0.109) | 0.284 (0.086) | 10.1 |
| 0.8, 0.9, 1, 0.1 | 2.5 | 0.325 (0.150) | 0.396 (0.105) | 2.4 |
| 0.8, 0.9, 10, 0.9 | 7.8 | 0.366 (0.369) | 0.174 (0.150) | 4.3 |

**Reading [PRED from SCRATCH].**
- Where internal walks are long (27 to 50 events per exit), which is where a learned macro-step would save the most explicit work, the exit angular response shows no structure above the noise floor. A smooth representation or a learning-free diffusion rule may already suffice there.
- Where the exit response is angularly high-frequency (forward scattering, optical thickness about 1), walks are short: 2.5 to 12 events.
- The intermediate regime (high albedo, g 0.9, τ about 1: 10 to 12 events, loss 0.20 to 0.28 against a floor of 0.08 to 0.10, about 10 mfp lateral spread) has both properties, but modestly.
- This is a toy: smooth faces, no edges, grey. Rough faces broaden exits, while edges, thin features and through-object TIR add structure.
- It still makes the risk concrete. **LIEM must pass a learning-free workload-admission gate that requires both twin cost and high-frequency content (I0.2), or stop.**

---

## 6. Candidate contracts

Every candidate gives every field the brief requires.

### 6.1 LIEM-S — Learned Intrinsic Exit Measure with structured lobes (family: other, intrinsic K; angular type C) — **PRIMARY (conditional)**

**Exact physical meaning of the learned transport.**
- The domain is D: a rigid optical domain, a homogeneous (or weakly heterogeneous) scattering medium bounded by a rough dielectric interface, with descriptor z = (σ_s, σ_a per channel, phase g, η, interface α).
- The learned quantity is the *collided* intrinsic transport kernel of D:
  - K^col(x_o, ω_o ← x_i, ω_i) is the outgoing radiance at boundary point x_o in air direction ω_o, at the first exit into free space, per unit flux entering at x_i from air direction ω_i;
  - it is restricted to internal paths with ≥ 1 volume-scattering event, with any number of internal interface reflections or TIR.
- It factorises as
  `K^col = m_c(x_i, ω_i) · ∫ p(x_o, ν | x_i, ω_i) · T̂_iface(ν → ω_o; x_o) dν`, where
  - `p` is a learned **probability density** over (exit point on ∂D, internal direction ν of the transmitting boundary arrival);
  - `T̂_iface = BTDF(ν → ω_o) / T(ν)` is the **exact** rough-dielectric transmission closure, normalized conditional on transmission (known physics, not learned);
  - `m_c ∈ [0, 1]^3` is the learned per-channel probability that refracted-in flux leaves through a collided path, scaled so that `m_c ≤ 1 − m_bal(x_i, ω_i)` (see energy).

**Spatial / angular input and output.**
- **Input**, all object-local:
  - the entry point (continuous surface point) and entry direction in the local tangent frame;
  - intrinsic features from D's own mesh: thickness along −n and along the refracted ray, distances and directions to edges / corners within support radius r_s, curvature;
  - the descriptor z through a shared encoder (as WL83's descriptor compiler).
- **Output**, per query:
  - per-channel mass m_c;
  - weights over K_s spatial components anchored at analytic anchors: a0 = x_i (local diffusion); a1 = the internal hit point of the refracted entry ray (through-object exit); a2 = its first TIR continuation;
  - per anchor, an anisotropic 2D heavy-tailed (Student-t) density in the anchor's tangent chart with per-channel width scale. Chromatic SSS radii differ strongly (WL45: 6.1 / 1.2 / 0.39 cm), so the width scale is per channel;
  - per component, a mixture of K_d direction lobes for ν in a bijective chart from R² onto the transmittable internal cone. The density is normalized on that cone with no horizon truncation (unlike the §5.1 generic mixtures).

**Local / non-local ownership.**
- **Local:** the entry-point interface reflection is the exact analytic GGX dielectric closure. It is *not* ER and is not learned.
- **Ballistic family** (zero-collision internal transport): explicit internal tracing, kept as Architecture §3's distinct path family [HUM HD4 on letting LIEM own it].
- **Collided family** (all nonlocal intrinsic transport, including its sharp low-order part and through-object / TIR displacement): **learned**.
- **All free space**, including same-asset exit → re-entry: explicit C on current geometry.

**Object-local coordinates and rigid transforms.**
- Every input and output is in D's object frame. World ↔ object uses the exact rigid transform.
- Weights and cached per-asset intrinsic features (edge / thickness fields) are reused exactly under rigid motion.
- Exact tests:
  - rotate / translate D: outputs transform to float tolerance;
  - re-tessellate: outputs equal within a declared tolerance.
- Non-rigid edits and contact invalidate D's features. That is outside scope, with conservative fallback to Twin-W.

**Interface to dynamic visibility.**
- None inside K. Exit radiance enters C. External objects never enter the input.
- A certificate invalidates D (fallback to the explicit walk) if foreign geometry intersects ∂D within r_s, which is contact or topology change.
- A box moving near D changes C only. This is tested explicitly (§7, I3).

**Energy / passivity contract.**
- p is an exactly normalized density: R² mixture → bijection onto ∂D charts and the cone. Spatial chart mass that falls off the surface (beyond edges) is counted as absorption and reported.
- m_c = sigmoid(·) · (1 − m_bal). m_bal is the explicit expected ballistic exit fraction.
- So the total albedo is ≤ ρ_loc + (1 − ρ_loc)(m_bal + m_c) ≤ 1 per unit incident flux.
- The composed K / C solve is sub-stochastic, so the spectral radius is < 1 whenever any loss exists [PRED, standard].
- Reciprocity is *not* imposed. It is a diagnostic, using paired adjoint impulses.

**Training target and data generation.**
- Impulse supervision from a declared measure μ_D:
  - x_i area-uniform over ∂D, with declared over-sampling of edges / corners / thin regions;
  - ω_i cosine-weighted bulk + uniform + a **grazing quota** (cos_i < 0.15) [HUM];
  - z from a declared descriptor prior;
  - shapes from a procedural family (slabs of varied thickness, boxes, cylinders, spheres, wedges), with whole shape classes held out.
- For each query, N_w float64 random walks with the reference physics. The code is independent of the evaluated runtime and cross-validated against the qualified reference (I0.1).
- Each walk records the exit point, the transmitting internal direction ν, ω_o, per-channel throughput, the collided flag and the event count.
- Loss: throughput-weighted NLL of p, plus a binomial / MSE loss of m_c against the per-query MC collided-exit fraction.
- No binned teacher. M64 is not used.

**Mechanism for sharp features.**
1. The exact interface closure produces the sharp exit lobe for each internal direction. There is no learned error in lobe width or position at α 0.04.
2. Learned internal-direction lobes can be arbitrarily narrow (low-order forward scattering, TIR-guided transport) and live in a chart with no truncation.
3. Analytic anchors place spatially displaced exits exactly from intrinsic geometry.
4. Heavy-tailed spatial components carry the long tails.

**Runtime evaluation and memory.**
- Per entry sample: one shared-network evaluation [PRED: 10^4 to 10^5 parameters, µs-scale batched on GPU] + ≤ 2 internal anchor ray casts against D's own mesh + one exact BTDF sample.
- Density evaluation at a fixed (x_o, ω_o) for MIS / deterministic gather is a low-dimensional integral over ν (importance-sampled through the inverse refraction). Cost: a few BTDF evaluations.
- Memory: weights (MB) + per-asset intrinsic feature cache, O(boundary samples). No per-asset operator table.
- For comparison, Twin-T at CAK resolution is 2.0 GB per configuration [MEAS WL55].
- In FNTO's deterministic coupled solve, applying K means fixed-sample Monte Carlo or component quadrature per entry cell.

**Scenes likely to fail.**
- GM1 surface-only scenes: K = ER, so the candidate is degenerate and not applicable.
- Smooth (delta) interfaces: no density in ω_o (out of scope).
- Media where ballistic transport dominates: little to learn.
- Dense diffusive media with a smooth response: the learning-free diffusion rule may suffice (§5.2).
- Concave media with exits on disconnected regions not covered by the anchors.
- Thin features and sharp edges: chart mass loss.
- Heterogeneous or textured media: descriptor insufficient.
- Non-rigid edits and contact (out of scope).
- Unqualified refractive-media references (I0.1).

**What learning buys over the best learning-free twin (falsifiable).** Three named twins:
- **Twin-W:** the same runtime with the learned macro-step replaced by the explicit internal random walk. Unbiased; cost ∝ events per exit.
- **Twin-D:** a learning-free *shared* analytic rule: a classical dipole / Burley-type profile × the exact interface, with no anchors.
- **Twin-T:** a per-asset tabulated K from the same walk samples, a histogram / kernel estimate at matched memory, with no learning.

Learning must deliver, on held-out shapes and descriptors with fixed weights:
- **(a)** lower time-to-quality than Twin-W;
- **(b)** better accuracy than Twin-D near edges, thin parts, grazing and through-object transport;
- **(c)** better memory and angular / spatial error than Twin-T at matched sample budget, and no per-asset precompute.

If none of (a) to (c) holds beyond paired noise, learning is decorative. The kill rule is in §7.

**Why it is not BRDF compression.** The learned object is a nonlocal boundary-to-boundary measure (exit position distributed over ∂D, through-object and TIR transport) with no pointwise closed form. The local interface response is exact and not learned.

**Disposition: PRIMARY**, conditional on [HUM] HD2 / HD3 / HD7 / HD10.
- It is the only design that satisfies P1 to P8 together.
- Rigid reuse is exact by construction; coverage is a declared treatment; energy holds by construction; the twins are explicit.
- It addresses both the learned-K lineage's fixed-basis angular limit (H2: WL57's 16-in / 64-out basis; WL58's 52 to 80 % loss at 256 bins and α 0.04) and the ECRO lineage's coverage pathology (H1).
- **Risks:** the §5.2 tension; prior art [LIT-unverified: Vicini et al. 2019, a learned shape-adaptive SSS model]; GM2 placement; the long negative history of learned-K shape generalisation (WL68, WL71 to WL80).

### 6.2 LIEM-B — intrinsic sparse multi-resolution categorical exit distribution (family: other, intrinsic K; angular type B) — **ALTERNATIVE (representation-level only)**

**Physical meaning.** The same K^col of D, with the same ownership, the same exact local and ballistic split, and the same impulse data.

**Input / output.**
- Input: the same object-local features.
- Output: a hierarchical categorical distribution over (exit cell on ∂D) × (exit direction cell). The direction cells live in an equal-area hierarchy on the air-side hemisphere (HEALPix-like) with flux-conserving cells (cos convention, J2).
- It is predicted autoregressively, coarse → fine (hierarchical softmax with refinement), plus mass m_c with a NULL / absorption category (π-softmax-with-NULL as in WL52).

**Ownership.** As 6.1.

**Rigid.** As 6.1.

**Visibility.** As 6.1.

**Energy / passivity contract.**
- Softmax-with-NULL gives column sums ≤ 1 by construction. The deterministic linear operator is passive.
- Hemisphere cells are exact, so there is no horizon truncation. This is an advantage over generic lobes at grazing (§5.1).

**Training target.**
- Multinomial likelihood of the same walk exit events, binned hierarchically. Counts are unbiased; no regression on noisy bins.

**Sharp mechanism.**
- Adaptive depth up to level L. [SCRATCH §5.1] A flux-conserving quadtree reaches about 0.1 lobe L1 at about 1,000 leaves on the grazing S1 lobe, where fixed 32 × 128 sits at 0.70.
- Within-cell smearing keeps J2's receiver-dependent sign below the finest level.
- It does *not* use the exact interface closure for the exit direction. The interface sharpness must be resolved by cells. A variant with an internal-cone hierarchy + exact interface is possible [HUM].

**Runtime and memory.**
- Deterministic sparse gather / scatter: k coefficients per entry sample. It is compatible with FNTO's deterministic F / S state solve, unlike sampling-based LIEM-S.
- Inference cost and output size grow with k and L. Tabulating it per asset reproduces Twin-T.

**Scenes likely to fail.** Those of 6.1, plus:
- sub-cell lobes below level L;
- large categorical output spaces, which are harder to learn (an H1 risk).

**Learned vs explicit value.** The twins are the same, and Twin-T is exactly "the same hierarchy filled by explicit counts per asset". Learning buys generalisation without a per-asset table and memory.

**Why it is not BRDF compression.** As 6.1.

**Disposition: ALTERNATIVE**, flagged as **not mechanism-independent**. It shares ownership, data, workload and therefore most failure causes with 6.1. It is kept because:
- it is the deterministic linear-operator form needed if the human keeps FNTO's F / S state-solve contract;
- it is the natural H2 control (adaptive discrete vs continuous structured).

**No mechanism-independent alternative survives this design pass.**

### 6.3 R-A — ECRO-lineage continuous directional learned kernel (family A) — **REJECTED (scoped)**

**Physical meaning.** b_{≥2}(x, ω) = ∫ K_θ(x, ω ← y, ω'; G) s(y, ω') dy dω'. This is a geometry-conditioned linear kernel from the direct incident field to the ≥ 2-event outgoing radiance, queried at continuous surface points and directions. It is ECRO's B table replaced by a neural field linear in s.

**Input / output.**
- Input: scene geometry tokens + ER states + relations, the source encoding, and a query (x, ω).
- Output: RGB radiance.

**Ownership.**
- It owns compositions ER ∘ C ∘ ER ∘ …, so its learned content is an amortised geometry-conditioned resolvent.
- It is nonlocal at the scene level. There is no intrinsic / C split inside it.

**Rigid.** Weights are reused; all state is regenerated after any edit. World-frame conditioning means no exact equivariance without an SE(3)-equivariant scene encoder.

**Visibility.**
- Either its flights use explicit relations, in which case it inherits relation variance (WL96: 0.46 expected rays per draw into the CK2 glint) and re-learns ER at the sender (P1);
- or it learns visibility (learned C; WL52; P1).
- In the second case dynamic visibility is implicitly coupled into the learned representation (unsuitability criterion 3).

**Energy.** Undefined for a free continuous kernel. Passivity requires integrating K_θ over ω, which is not closed form (unsuitability criterion 5).

**Training.** Path-traced ≥ 2-event radiance at random queries, light-sampled as in WL86, so the L2 coverage pathology applies. Impulse supervision of a scene-level operator needs O(N × directions) columns per geometry, which is prohibitive.

**Sharp mechanism.** Continuous ω avoids readout bin bias. Sharp *delivery* (a glint onto diffuse receivers) is still a C property the kernel must learn.

**Runtime.** One network query per camera vertex: per-pixel inference instead of WL86's 0.34 ms / spp table lookup, plus a generator per edit.

**Failure scenes.** Grazing / under-covered incidences; geometry outside the training family (WL93D: it ignores geometry).

**Learned vs twin.**
- The twin is EXP / C3URX1 (exact ER, explicit relations, exact per-ray reads), which pooled equals the oracle [MEAS WL91 / P0-3], or sender-anchored splatting.
- The only learning-specific value is amortised variance and cost. Its ceiling at the measured learned cost is ≤ 1.12 for any role that does not learn C (§4).

**Why rejected.** Unsuitability criteria 1, 3 and 5. Scope: scene86 / GM1, N 1,270 prototype. It may be reopened only if Stage G0 measures a ceiling ≥ U_min [HUM].

### 6.4 R-B — ECRO-lineage sparse multi-resolution angular table (family B) — **REJECTED as a learned candidate; RETAINED as a learning-free control**

**Physical meaning.** ECRO's table B[patch, angular cell] with an adaptive hierarchical, flux-conserving angular basis per patch, and learned coefficients from the ECRO generator.

**Input / output.** As ECRO, with a variable-length sparse output per patch.

**Ownership.** As ECRO (≥ 3-event paths).

**Rigid.** As ECRO: weights only.

**Visibility.** Through state rays (explicit relation, same variance).

**Energy.** Flux-conserving cells give per-bin flux exactness. Operator passivity is not guaranteed by the learned factors (J1: non-passive rows).

**Training.** As ECRO.

**Sharp mechanism.** Adaptive refinement.

**Runtime.** Sparse table lookups; refinement bookkeeping.

**Failure scenes.** As ECRO.

**Learned vs twin.** The angular improvement is entirely available learning-free. The explicit flux-conserving quadtree reaches lobe L1 of about 0.1 at 1,000 leaves [SCRATCH §5.1]. J2 found 8 × 32 already within 2 to 6 % at box-A delivery [MEAS]. Learning adds only the ECRO operator, which is the failing part.

**Why rejected.** The representation fix is not learning-specific (P7). Its learning-free version is the "flux-corrected / high-resolution / adaptive binned" control of §7.

### 6.5 R-C — ECRO-lineage learned directional lobe / distribution at patches (family C) — **REJECTED**

**Physical meaning.** The per-patch once-scattered sender radiance, or the ≥ 2-event outgoing radiance, as learned lobe parameters (direction, width, weight) generated from geometry and source.

**Input / output.** Patch and source encoding → K lobes per patch.

**Ownership.**
- For the once-scattered sender, the exact lobe family is ER itself (7 GGX + Lambert), so learning it is BRDF compression and a P1 violation.
- For ≥ 2-event glossy outgoing radiance = ER ⊗ incident field, the learned part is the incident field, which is C.

**Rigid.** Weights only.

**Visibility.** Through relations or learned (as R-A).

**Energy.** Normalized lobes × mass ≤ 1 is possible (P5). That is the one advantage.

**Training.** Light-sampled (as ECRO).

**Sharp mechanism.** Lobe width. [SCRATCH §5.1] Generic mixtures with ≤ 4 components reach only 0.35 to 0.6 lobe L1 at grazing incidence, with peak overshoot.

**Runtime.** Lobe evaluation per read; small memory.

**Failure scenes.** Grazing, horizon-truncated multi-lobe responses.

**Learned vs twin.** The twin is exact ER evaluation, which is better by construction.

**Why rejected.** Brief exclusion (BRDF compression), P1, and the scratch capacity evidence.

### 6.6 R-D — Rigid-cluster certified self-transport (RCT) (family: other) — **REJECTED under current canon [HUM HD5]**

**Physical meaning.** For a rigid multi-surface asset (a concave glossy bracket, bowl or panel assembly), a learned operator owning every self-interreflection path that stays within the asset's certified region. It is reused under the asset's rigid motion while no foreign geometry intrudes; an intrusion certificate invalidates it.

**Input / output.** The asset's local incident field → its local outgoing field after self-paths.

**Ownership.**
- It owns exposed same-asset free-space paths. **This conflicts with Architecture §2**: "Object identity must not freeze exposed interreflection or merge editable air transport into K"; C owns same-asset paths.

**Rigid.** Exact under the asset's motion.

**Visibility.** A certificate: on intrusion, fall back to explicit C within the region.

**Energy.** Sub-stochastic if built from normalized measures.

**Training.** Per-asset or shape-conditioned impulse data.

**Sharp mechanism.** As LIEM-S, with ER at every bounce.

**Runtime.** Asset-level operator application.

**Failure scenes.** Partial intrusion (blocked self-paths cannot be subtracted); the shape generalisation of surface self-transport (= learned C).

**Learned vs twin.**
- The twin is the cached explicit relations within the asset. It is equally rigid-invariant and equally reusable, so learning buys only sweep amortisation (small, WL96A) and continuity.
- Per-asset training puts it in the external **Native Reuse** class (8DNA-like), not a new method.
- The scene86 family has no concave asset.

**Why rejected.** A canon conflict and weak learned value. Only the user can reopen ownership.

### 6.7 R-E — Learned sharp delivery / learned C for sharp senders (family: other) — **REJECTED**

**Physical meaning.** A learned prediction of the irradiance delivered by a sharp sender lobe onto receivers, conditioned on current geometry.

**Input / output.** Sender lobe and geometry → receiver irradiance.

**Ownership.** C (free space).

**Rigid.** No: depends on relative poses.

**Visibility.** Learned. Unsuitability criterion 3.

**Energy.** Can be normalized (π-softmax-with-NULL, WL52).

**Training.** Path-traced delivery.

**Sharp mechanism.** It must learn beam visibility.

**Runtime.** O(N · M) dense or candidate-limited (WL52: top-k coverage failed).

**Failure scenes.** Contacts and visibility transitions [MEAS WL52].

**Learned vs twin.** Sender-anchored lobe splatting (ER sample + ray cast) resolves the glint that receiver-anchored gathering misses [MEAS WL96: 0.46 rays per draw]. Its cost is unmeasured (P0-3 smallest missing evidence). Learned visibility cannot beat hardware ray casting on accuracy [PRED].

**Why rejected.** Criteria 1 and 3, and P1. The splat twin enters §7 as an explicit frontier arm.

---

## 7. Falsification plan

**Structure.** Two tracks that **cannot share workloads, training data or references** (§7.3).

| track | role | why it exists |
|---|---|---|
| **Track G** (GM1 scene86 family) | attribution of the ECRO-lineage failure into H1 / H2 / H3, plus the learning-free ceiling that tests §3 | the brief requires the H1 / H2 / H3 separation with frozen ECRO, coverage, binned and explicit controls. It is not a candidate-adoption track |
| **Track I** (intrinsic media workloads) | the LIEM-S (and LIEM-B) candidate test | the primary candidate |

Every stage order, gate and stop rule is declared before results. Numerical thresholds marked [HUM] are not invented here.

### 7.1 Track G — ECRO-lineage attribution

**G0 — learning-free, no training.**
- **G0.1 Workload admission.** The admitted workloads are:
  - E2: the WL86 sharp failure.
  - E5: the only qualified never-inspected held-out (J7).
  - **One new independent held-out workload with a grazing sharp component and a T0 → T1 rigid edit with a path-traced T1 reference.** It is geometry-chosen with no radiance inspected, margin-disjoint from every training population, and admitted on a qualification partition (sharp-change share; change SNR ≥ 4 on independent replicas; reference noise bound).
  - **CK2 is diagnostic only and never decides a gate.**
  - Stop: `STOPPED_UNQUALIFIED_WORKLOAD` if the new workload fails admission. No substitution is allowed after inspection.
- **G0.2 Explicit quality–cost frontier.** Arms, all with frozen ER and the same references and state draws:
  - E-sa64: 4 × 16 solid-angle mean (= EXP);
  - E-cos64: 4 × 16 flux-conserving;
  - E-cos256: 8 × 32;
  - E-QT: adaptive flux-conserving quadtree, about 256 / 1,024 leaves;
  - E-X1: C3URX1 exact per-ray reads at 1×, 4× and 16× relation draws;
  - E-SPL: sender-anchored lobe splatting at two budgets.
  - Outputs: sharp-ROI E_H, region C, changed-region fidelity, extra T1 − T0 error, leakage / ghosting maps, and lifecycle cost (edit-to-first-frame, steady frame).
  - This closes P0-2 / P0-3 smallest-missing evidence: a non-grazing light, a second receiver, single-draw renders, and the cost of a low-variance exact estimator.
- **G0.3 Ceilings.** Recompute §4's zero-cost and measured-learned-cost ceilings for each role (resolvent; prefix + tail; + relation) against the **quality-matched** explicit frontier point:
  - at N ∈ {1,270, about 5,000, about 11,000};
  - with lights J ∈ {1, 4};
  - and with the frame both included and excluded.
  - **Stop rule G0:** if no role that does not learn C has a zero-cost ceiling ≥ U_min [HUM] at matched quality on ≥ 2 admitted workloads, then H3 is SUPPORTED at ceiling level for the ECRO lineage on GM1. No ECRO-lineage training arm runs for architecture purposes. G1 / G2 run only if the human wants the attribution for its own sake [HUM HD8].

**G1 — H2 capacity, no learning.** Class floors against the path-traced reference (not M64) on the admitted workloads:
- the delivery-optimal patch × 64-bin operator, sa and cos conventions (J1 §8: never computed);
- the best admissible non-negative rank-16 / 32 operators under a declared incidence weighting (J1 construction);
- the 8 × 32 and quadtree floors.

If the 64-bin class floor already misses the quality target, the existing representation is **representation-limited (H2)** for that workload, whatever training does.

**G2 — H1 learnability.** Only if G0 passes or HD8 says to run it. Arms share the WL86 geometry family, frozen ER, the target generator, references, step and sample budget, and ≥ 3 seeds [HUM]:
- **F-ECRO:** frozen WL86 / WL92A checkpoints. No training; a control.
- **ECRO-R:** the unchanged recipe retrained with fresh seeds. This measures training variance.
- **ECRO-COV:** the same architecture and budget, with a **designed coverage** treatment:
  - training lights stratified so that every learned (patch, incident ring) cell receives at least a declared fraction of the training mass, including ring 0 [HUM];
  - plus direct supervision of the once-scattered stage by the **path-traced** q (not M64).
  - This is "existing representation + improved training coverage".
- **ECRO-COV-FC:** ECRO-COV with the flux-conserving output convention or 8 × 32. This is H2 within the existing learned form.
- Measures: the gap (learned − G1 class floor) on grazing sharp, non-grazing sharp and broad regions, per seed; seed spread at covered vs under-covered incidences (L3 signature).

Predeclared readings:
- **H1 supported** if ECRO-COV closes ≥ X % [HUM] of ECRO-R's gap to the class floor on grazing sharp regions in every seed, without a non-grazing regression beyond tolerance.
- **H2 supported** if the class floor exceeds the target and FC / 8 × 32 lowers the floor by more than the paired tolerance.
- **H3 (utility)** if the best learned arm's utility at its achieved quality is < U_min against the quality-matched explicit frontier.

**G3 — fixed-weight edits.** On the admitted held-out workloads: T0 → T1 with weights fixed, a STALE control, signed-change slope / correlation in the changed region, extra T1 − T0 error, leakage / ghosting, and per-region errors. These are paired against the explicit frontier arms at matched cost.

### 7.2 Track I — LIEM primary candidate

**I0 — learning-free admission and references.**
- **I0.1 Reference qualification for refractive-interface media.**
  - The reference candidates are Cycles 3.5.1 volume + rough glass boundary, and Mitsuba (llvm / cuda on LabServer69 only, since LabServer63's cuda film is wrong; with the float32 boundary-layer bias bounded at the chosen σ_t). The choice is [HUM HD7].
  - Independent checks: a float64 random walk (own code; also the training-data generator, so it is cross-validated before use); analytic controls (index-matched furnace; slab with Fresnel boundaries by adding-doubling / diffusion limits; reciprocity; superposition).
  - Stop: `STOPPED_REFERENCE_UNQUALIFIED` if disagreement exceeds tolerance [HUM].
- **I0.2 Workload admission (the headroom gate).** A (descriptor, shape class, light) configuration is admitted only if, on a qualification partition:
  - (a) the collided-family share of radiance in the evaluated regions ≥ s_min;
  - (b) the mean internal events per collided exit ≥ e_min (Twin-W cost);
  - (c) the angular high-frequency content (best flux-conserving WL57 16-in / 64-out projection loss of the collided exit field) is ≥ h_min **above the matched MC noise floor**, as in §5.2;
  - (d) T0 → T1 change SNR ≥ 4;
  - (e) both grazing (cos_i < 0.15) and non-grazing incidence on the medium are present.
  - The workload template: the scene86 room with its frozen-ER sharp S1 panel, ER floor and S2 sphere, plus ≥ 1 movable medium object and one Lambert occluder. GM1's sharp panel interreflection is kept and now couples to a learned K.
  - At least one **independent held-out workload** is admitted the same way: an unseen shape class and an unseen light set, with no radiance inspected.
  - **Stop `STOP_NO_HEADROOM`** if no configuration satisfies (b) and (c) together. That is the §5.2 tension resolved negatively, and LIEM is not trained.
- **I0.3 Twin frontier.** Twin-W time-to-error curve; Twin-D error; Twin-T error at memory budgets (all vs the reference). This sets LIEM's quality / cost target.

**I1 — H2 capacity, no amortisation.**
- Per-query oracle fits on the synthetic contract set (slabs, edges, corners, wedges; grazing / non-grazing strata). For each sampled entry, the representation is fitted to its own walk samples:
  - LIEM-S family;
  - LIEM-B hierarchy;
  - WL57 16-in / 64-out basis;
  - WL58 64 / 256 basis.
- **Stop `STOP_H2`** if the LIEM-S oracle floor exceeds the target on more than a declared share of strata [HUM]. LIEM-B is then evaluated as the alternative, and if it fails too, the track stops before training.

**I2 — H1 learnability and coverage.** Arms share the impulse data generator, the training budget and ≥ 3 seeds:
- **LIEM-S on μ_D**, the designed measure with the grazing quota;
- **LIEM-S on μ_S**, the scene-induced entry distribution from the training scenes' lights (the ECRO-style uncontrolled coverage);
- **WL57-family decoder on μ_D**: the existing representation with improved coverage;
- **LIEM-B on μ_D**, if alive.

Evaluation sets: held-out entries, held-out shapes and held-out descriptors.

Predeclared readings:
- **H1 (coverage)** if μ_S − floor ≫ μ_D − floor on grazing strata.
- **Learnability failure** if μ_D − floor stays large despite coverage.
- **H2 for the old representation** if the WL57 decoder on μ_D stays near its I1 floor and that floor is above target.

**I3 — H3 utility, fixed weights.**
- Admitted held-out workloads; T0 → T1:
  - **(i)** a rigid move / rotation of the medium object (K reused exactly);
  - **(ii)** a move of the Lambert occluder only (C changes, K must be untouched: a test for hidden visibility coupling).
- Arms, all with identical explicit C and frozen ER:
  - LIEM-S; LIEM-B (if alive);
  - Twin-W, Twin-D, Twin-T;
  - STALE (T0 incident field reused);
  - the ballistic-explicit control and its learned-ballistic variant [HUM HD4].
- Image evidence:
  - error vs reference;
  - a sharp-interreflection ROI (the panel glint delivered onto and through the medium object, and medium exits seen in the panel);
  - changed-region fidelity, signed-change slope / correlation;
  - leakage / ghosting (STALE-like residue after the edit);
  - extra T1 − T0 error (the reuse penalty).
- System evidence:
  - edit-to-valid-frame critical path (geometry, C relation recompute, K invalidation check, solve, readout);
  - steady-state frame cost;
  - learned inference cost;
  - memory and precompute: LIEM shared weights vs Twin-T per-asset tables and their build time;
  - utility against the **quality-matched** best twin.
- **Kill rule (H3 SUPPORTED → candidate not supported):** LIEM-S neither has a lower time-to-quality than Twin-W, nor is more accurate than Twin-D beyond paired tolerance, nor beats Twin-T on memory at matched error without per-asset precompute, on ≥ 2 admitted held-out workloads.
- **External baselines** (binding gate): Native Reuse of an 8DNA slab-type asset is possible on a WL69-type workload [HUM HD9]. **Neural Full Recompute is UNMEASURED** unless an invalidated external representation is aligned. Path tracing is reference / re-render cost only.

### 7.3 What can and cannot share data, budget and reference

| comparison | shares training data | shares budget | shares physical reference |
|---|---|---|---|
| F-ECRO / ECRO-R / ECRO-COV / ECRO-COV-FC | yes (WL86 family; COV changes the sampling measure, which is the treatment) | yes (steps, samples) | yes |
| ECRO arms vs explicit frontier E-* | no training in E-* | matched cost / quality points, not training budget | yes (same renders, state draws) |
| LIEM-S / LIEM-B / WL57 decoder (I2) | yes (same walk generator; μ_D vs μ_S is the treatment) | yes | yes (synthetic: walk + analytic) |
| LIEM vs Twin-T | Twin-T is built from the same walk samples | matched sample budget and memory | yes |
| LIEM vs Twin-W / Twin-D | no (no training in the twins) | matched runtime cost | yes |
| Track G vs Track I | **no**: different physics and workloads; frozen ECRO cannot represent media; EXP / C3UR appear in Track I only as the shared explicit C + ER | no | no (different reference families) |
| ReST-GI vs 8DNA Native Reuse | no | lifecycle-aligned only | yes, if aligned per the baseline gate |

### 7.4 Evidence and statistics plan (both tracks)

- **Representation evidence:**
  - angular response fidelity per stratum (grazing / non-grazing / edges);
  - peak / lobe position and width error (exit-direction lobe for LIEM; delivery footprint for Track G);
  - spatially distributed error (exit-position distribution: Wasserstein / L1 on ∂D; lateral moments);
  - energy (mass error; passivity margin; composed spectral radius on worst-case fields);
  - sampling variance (per-query variance of the estimator at the deployed sample count vs the twin's).
- **Image evidence:** listed under I3 / G3.
- **System evidence:** listed under I3 / G0.2.
- **Statistics:**
  - **Pairing:** every comparison uses the same reference renders, cameras, state / relation draws and sample seeds.
  - **Reference noise:** independent reference halves give a debiased error, and the reference-noise term is reported separately.
  - **Uncertainty:** block-bootstrap SE over pixels / patches. The tolerance of a paired difference is max(2 SE, M_min), with M_min [HUM]. A noise floor is never used as a paired tolerance (J6).
  - **Seeds:** ≥ 3 training seeds per learned arm [HUM]; report per seed.
  - **Draws:** ≥ 16 state draws for any relation-sampled arm. No label may depend on one draw (J3).
  - **Heavy-tail guards** for pooled metrics: per-geometry tables and median-of-means (the WL93A g00010 lesson).
  - **Train / validation / test** are separated by shape class, descriptor range, workload and edit before results. Held-out workloads are geometry-chosen and radiance-uninspected.
  - **Process:** freeze, then run; amendments are append-only.
- **Synthetic contract tests vs real-scene evaluation.** Both are required; neither alone can approve.
  - Synthetic tests use analytic or independent targets: exact equivariance, tessellation invariance, normalization, linearity, analytic slabs, impulse fidelity by stratum. They isolate representation (I1) and learnability (I2) from coupling and readout.
  - Real-scene evaluation is the only place where the scene-induced incidence distribution differs from the training measure (the coverage question). It is also where K / C composition, camera readout through the sharp panel, edit response and lifecycle cost appear.
  - A synthetic pass with a real-scene fail localises the failure to coverage or coupling. A real-scene pass without synthetic contracts cannot be attributed.

### 7.5 Stage order and stop rules (summary)

```text
Track G:  G0.1 admit -> G0.2 explicit frontier -> G0.3 ceilings --(no role >= U_min)--> H3@ceiling, STOP (G1/G2 only by HD8)
                                                                --(pass)--> G1 class floors -> G2 H1 arms -> G3 fixed-weight edits
Track I:  I0.1 reference --(fail)--> STOPPED_REFERENCE_UNQUALIFIED
          I0.2 headroom admission --(no (b)&(c) config)--> STOP_NO_HEADROOM
          I0.3 twin frontier -> I1 capacity --(LIEM-S and LIEM-B floors miss)--> STOP_H2
          I2 learnability (mu_D vs mu_S vs WL57 decoder) -> I3 fixed-weight edits + economics --(kill rule)--> NOT SUPPORTED
```

No stage result authorises a successor or an adoption.

---

## 8. Human decisions needed

- **HD1.** Accept or reject my scoped design finding that no learned candidate in the GM1 scene86 surface-only family meets the brief's suitability criteria. Alternatively, require Track G0 to measure it first.
- **HD2.** Whether the learned-transport hypothesis may be tested on workloads with nonlocal intrinsic transport (rough-dielectric media), which is GM2 territory.
  - LIEM is in FNTO's learned-K lineage, not the ECRO lineage that the 2026-11-13 decision made central.
  - It keeps GM1's sharp ER panel in the workload.
- **HD3.** Whether an analytic rough-dielectric interface closure may be introduced next to the frozen ER. This adds a closure; it does not modify ER.
- **HD4.** Ballistic-family ownership: explicit, which is canonical (Architecture §3), or learned inside LIEM.
- **HD5.** Whether rigid-cluster self-transport ownership (R-D) may ever be considered. It is rejected under the current canon.
- **HD6.** All numerical thresholds:
  - U_min;
  - M_min;
  - reference tolerances;
  - admission thresholds s_min, e_min, h_min;
  - the grazing quota of μ_D;
  - the per-cell coverage floor for ECRO-COV;
  - the H1 gap-closure fraction X;
  - the I1 stop share;
  - seed and draw counts.
- **HD7.** The reference renderer for mixed ER + medium scenes, and the medium density range given Mitsuba's float32 boundary layer.
- **HD8.** Whether Track G training arms run for attribution even if G0 closes the lineage for architecture purposes.
- **HD9.** External baselines on the medium workload: 8DNA Native Reuse setup. Neural Full Recompute is declared UNMEASURED unless aligned.
- **HD10.** Novelty acceptance. LIEM sits in the learned-BSSRDF prior-art class. The novelty discipline (RESEARCH_CENTRIC_TOPIC) needs joint evidence of ownership, edit validity and economics.
- **HD11.** The compute budget and which server is main.

---

## 9. Proposed batch verdict and rationale

**DESIGN_READY_FOR_HUMAN_REVIEW.**
- The design contains one primary candidate with a complete contract and one representation-level alternative.
- Every other candidate has a full contract and an explicit rejection reason.
- There is a staged falsification plan. It puts learning-free kill gates first: G0 ceilings, I0 headroom, I1 capacity. Training follows only after them.
- H1, H2 and H3 are separated by arms that share data, budget and reference.
- Every numerical threshold is left to the human.

**Why not NO_DEFENSIBLE_CANDIDATE.** LIEM-S satisfies P1 to P8 and every brief suitability criterion within its scope. Its value over its twins is measurable.

**Why not INSUFFICIENT_EVIDENCE.** The unknowns that decide the candidate's fate are measured inside the plan, cheaply and before training:
- the headroom tension;
- reference qualification;
- representation capacity.

**Caveats.**
- The verdict is conditional on HD2 / HD3 / HD7.
- For the GM1 surface-only family alone, my reading is that **no defensible learned candidate exists** (§3). That reading is a design argument until G0 measures it.

---

## 10. Provenance

**Scratch scripts** (numpy only, CPU, local):
- `experiments/wl98_calc/D2/angular_capacity.py` → `results/wl98_calc/D2/angular_capacity.json` and `.log`. It imports `experiments/audit97/A/er_np.py` read-only and `experiments/ecro86/data/mats86.npz`.
- `experiments/wl98_calc/D2/slab_tension.py` → `results/wl98_calc/D2/slab_tension.json` and `.log`. Seed 98003, 600,000 photons per case.
- `experiments/wl98_calc/D2/amdahl_ceiling.py` → `results/wl98_calc/D2/amdahl_ceiling.json` and `.log`. Inputs are the recorded WL96A and WL86 times.

**What was not done.** No existing file was modified. Nothing was trained or rendered, and no GPU or server was used. Nothing was committed.

**Sources read.**
- `docs/Architecture.md`;
- the frontier and GM1 entries of `docs/RESEARCH_ROADMAP.md`;
- `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`;
- `docs/RESEARCH_CENTRIC_TOPIC.md`;
- WL97 `README.md`, `COORD_synthesis_and_p0_ledger.md`, `J1`, `J2`, `J7` (parts) and `F0` (parts);
- worklogs 86, 57 (parts), 58, 96 (parts) and 96A (cost lines);
- `experiments/ecro86/model86.py`, `scene86.py` and `protocol/ecro86.json`.
