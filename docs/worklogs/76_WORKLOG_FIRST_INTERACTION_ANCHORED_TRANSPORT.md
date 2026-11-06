# Worklog — First-Interaction Anchored Spatial Transport Feasibility and Transfer (2026-11-06)

## Origin and authorisation

- **Request.** The user pasted the WL76 brief, asked whether it should run on 63, and then said 「어어 진행해」. The batch therefore runs on LabServer63 (the main server). The local exception was for WL75 only.
- **Locations.** Code is in `experiments/fnto_fia76/`, the protocol in `protocol/fia76_v1.json`, and the evidence root is `LabServer63:~/rest_fia76`.
- **Not authorised:**
  - production K/C changes;
  - new materials, glossy transport or coupled GI;
  - source coverage changes;
  - loss tuning or capacity sweeps;
  - per-port tables;
  - held-out selection;
  - adoption or a successor batch.

**Delay note.** About 30 minutes passed between the user's go-ahead and the first visible progress, all on the Stage 0 design and none reported. The user rightly called this out (「왜 30분 넘게 가만히 있었냐」); progress is now reported stage by stage.

## Stage 0: physical factorization contract (Part A of the protocol)

**Medium** (`p53_fixtures`): homogeneous, σ_t = 1000 m⁻¹ (mean free path 1 mm), albedo (0.999, 0.99, 0.9), isotropic phase (g = 0), index-matched boundary.

**Cell sources.** Unit flux, uniform on the 2.5 mm port, cosine-weighted inside the incoming bin (16 bins, concentric mapping). Exits are tallied by port and 64 outgoing bins.

**Decomposition.** Consider an entering photon at x0 travelling along d, with chord t to the boundary.
- It leaves without colliding with probability exp(−σt); this is the existing ballistic term B.
- Otherwise it collides first with probability c = 1 − exp(−σt), at a truncated-exponential depth on [0, t). Hence 1 − β(i,a) = E[c] exactly.
- At the collision the weight is multiplied by the albedo, and the new direction is drawn isotropically, independent of d.

This gives

K_scat(i,a → j,o) = (1 − β) · E_{x∼p1(·|i,a)}[ Q(x → j,o) ],  with  Q = albedo · (E1 + M)

- E1 is single-scatter escape from x: geometry-only and exact.
- M covers scattering orders ≥ 2.
- Q depends only on x and the current geometry. This follows from the Markov property of the walk and the isotropic phase.

**Where the factorization does not carry over:**
- an anisotropic phase (Q would depend on the incident direction);
- surface/BSDF scattering (the redirection is frame- and direction-dependent);
- Fresnel boundaries;
- heterogeneous media.

**Energy.** Σ E1 ≤ 1, Σ Q ≤ albedo, Σ K_scat ≤ 1 − β. This is passive for any learned M bounded by 1 − Σ E1.

## Stage 1: physical correctness gate (frozen before the Stage 2 candidate design)

**Validation estimators** (validation only, never used at inference):
- `decomposed` forces the first collision, restarts isotropically, samples the next segment exactly, then runs the UNCHANGED `p53_tracer.walk`.
- `single_quadrature` is the expected-value E1 composition.
- A forward negative control keeps the restart direction equal to the incident direction.

**Comparison.** Against archived WL73/WL71 references (independent seeds): 9 sources × bins {5, 0, 10, 15}, 36 packets in total.

| Pose | Sources |
|---|---|
| rest | INT s60, TBE s64, TBC s66, SM s68, SR s70, SV s72 |
| te0 | INT s80 |
| pt0 | TBE s50, INT s0 |

**Rules:**
- energy |z| ≤ 4;
- coarse-cell mean z² between 0.5 and 1.5, over cells selected by the INDEPENDENT estimator's expected count ≥ 30;
- E[c] against 1 − β within |z| ≤ 4, using a binomial floor on the reference SE;
- leaks ≤ 1e-5 N;
- the forward control must be rejected in every control packet.

**Pre-freeze smoke** (local, non-evidence, 0.2M then 2M photons). The first version's statistic was wrong in three ways, all fixed before the freeze:
1. 0-hit cells have a zero variance estimate, which inflates z².
2. A zero replica SE when no ballistic exit occurs (expected β ≈ 6e-10) produced z = −27.
3. Selecting cells by the tested estimator's own counts inflated z² to about 2.

A self-comparison of two seeds of the same estimator gave z² = 1.0, which confirmed the variance formula. After the fixes, at 2M photons:
- total vs reference: z² 0.92–1.15;
- single quadrature vs MC: z² 0.58–1.52;
- forward control: rejected 4/4 (z² 2,600–9,100).

The evidence run uses 32M photons per packet.
