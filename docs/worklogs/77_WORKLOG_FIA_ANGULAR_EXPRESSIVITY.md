# Worklog — FIA Post-Collision Angular Expressivity Falsification (2026-11-07)

## Origin and authorisation

- **Brief.** The user pasted the WL77 brief "FIA Post-Collision Angular Expressivity Falsification". It authorises ONE bounded change: replace WL76's ring-uniform outgoing lobes with an azimuth-resolving, D4-equivariant outgoing response.
- **Unchanged.** The first-interaction / second-collision factorization, the anchors, the quadrature and the spatial support stay as they were.
- **Placement and paths.** LabServer63 (main). Code: `experiments/fnto_fia77/`. Protocol: `protocol/fia77_v1.json`. Evidence root: `~/rest_fia77`.
- **Preserved verdicts.** WL76's Stage 1 FAIL (with its post hoc amendment) and its INSUFFICIENT FIT verdict are preserved.
- **Not authorised:**
  - changes to the anchors, the quadrature or SS / W;
  - a second architecture;
  - per-port corrections;
  - threshold or training tuning;
  - checkpoint selection;
  - adoption, or a successor batch.

## Design (frozen before any WL77 candidate evidence)

**Removed degrees of freedom.** WL76 ring lobes are uniform inside each of the 4 polar rings of the 8x8 concentric grid. Every azimuthal variation inside a ring is therefore removed: 60 of the 64 per-receiver directions. By the WL76 post hoc diagnostic, 0.45 of the reference multiple-scatter native signal lies in that variation.

**New local outgoing distribution.**

O(o) = Σ_{m<3} α_m L_m(o) + α_az Λ(o)

Λ(o) = ν(o) [1 + s·a1·cos(φ_o − φ_r) + s·a2·cos 2(φ_o − φ_r)] / Z

- α is a softmax over the four components (the three ring lobes and Λ) per pair and channel.
- ν is a learned ring profile.
- φ_o is the azimuth of bin o's centre in the receiver frame (t, b).
- φ_r is the azimuth of the tangential part of the anchor → receiver vector in the same frame. It is geometric, not learned.
- s = |r_t| / (|r_t| + 1 mm).
- |a| ≤ 1/2.
- Z is the exact per-pair normaliser.

**Properties.**
- Nonnegative and normalised, so positivity, linearity and passivity are unchanged.
- Invariant under in-plane rotation or reflection of the frame. It is therefore exactly D4-equivariant wherever bins permute.
- The runtime uses an exact 8-function basis: 3 ring lobes plus ν × {1, cos, sin, cos2, sin2}. A shared int32 CSR index stores it.

**Limitation.** The coarse tail keeps the ring lobes, and its share is reported.

**Initialisation.** WL76 constructors and seeds 7601 / 7602. The azimuthal-lobe weight starts at about 0.04 with a = 0. There is no warm start.

**Unchanged machinery.** The geometry, SS, W, the quadrature (seeds, q = 1024, rescaling) and the objective are imported from `fnto_fia76/fia76.py`, not copied.

**Gates.**

| Stage | Gate | Stop outcome |
|---|---|---|
| 1 | The FIA-AZ relaxation's projection floor of the reference multiple-scatter part (training replicas) must be ≤ 0.5 × the ring floor | STOP |
| 2 | Full-runtime state ≤ 4 GiB and 64-light apply ≤ 10 × A75 | STOP |
| 3 | WL76 fit gate, unchanged | Attribution (brief Stage 4) and STOP |

**If Stage 3 passes.** Reuse the unused WL76 Stage 4 population, after hash and independence checks. Run the transfer test against AR, A75 and the preserved F76.

**Local tests: 11/11 pass.** They cover:
- the WL76 physics, CSR, source-independence and objective tests;
- bin azimuths;
- per-pair normalisation and positivity;
- a synthetic azimuthal target, fitted with residual < 0.05 by the lobe form against > 0.3 by ring lobes;
- full native D4 × z-flip equivariance with exact bin permutations.
