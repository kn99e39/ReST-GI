# Worklog — Angular Frame Equivariance and Symmetry-Consistent Source Transfer (2026-11-08)

## Origin

- **Brief.** "Worklog 79 — Angular Frame Equivariance and Symmetry-Consistent Source Transfer", 2026-11-08.
- **Question.** Is insufficient angular-frame equivariance a material cause of the remaining W+UV-Q (Q78) boundary-transfer failure, and can symmetry-consistent supervision address it within the existing rank-16 ATO / W+UV-Q representation?
- **Intervention (at most one).** Symmetry-consistent presentation of the existing physical supervision to the unchanged Q78 operator. Before any training, a representability gate checks whether the frozen rank-16 G2 A/O can express the symmetry-transformed responses; if it cannot, the batch stops.
- **Nature.** Bounded mechanism validation. Not a novelty claim and not a production adoption. Standard symmetry augmentation is not claimed as a new method.

| Item | Setting |
|---|---|
| Branch | `Selective_Recompute` |
| Server | LabServer63 (main) |
| Code | `experiments/fnto_sym79/` |
| Server root | `~/rest_sym79` (dev: `~/rest_sym79_dev`) |
| Protocol | `experiments/fnto_sym79/protocol/sym79_v1.json` |

**Controls (unchanged, never retrained).** Primary: Q78_5901 / Q78_5902 (sha256 3bdff7e9… / f40b0d1f…). Secondary: A75_5901 / A75_5902 (c0022742… / 358c3881…), AR_5901 / AR_5902.

**Population status carried in.** The WL77A Part 0 references were scored in WL78. In WL79 they are development data only, never independent confirmation.

## Stage 0 — Physical symmetry contract

### Code (traced, not inferred from worklog prose)

- **Bins.** 16 incoming (4 × 4) and 64 outgoing (8 × 8) cells of the concentric square-to-disk map (`p53_geom.square_to_disk / local_to_bin`), expressed in each port's bin frame.
- **Bin frame.** `(t, b, n)` of the port's first triangle (`dom.port_tris[:, 0]`). `t` is the image of the face's rest tangent under the triangle's rest → current affine map (`p53_geom.tri_frames`). At rest it is constant per face.
- **Operator.** `K[i,j,a,o,c] = d[i,a] Σ_r A[r,a,c] T[i,j,r,c] O[r,o,c]`, with A, O global and frozen (G2) and T per (source, receiver).
- **Existing L_sym (WL72 Candidate A).** It uses the exact D4h port permutations, but only on the **angle-integrated** responses `Σ_a Σ_o S`. The angular content of a symmetry image was never supervised.

### Convention (implemented in `sym79.py`)

| Symbol | Definition |
|---|---|
| `perm[p]` | Port at `op(x0_p)`, rest material coordinates; exact bijection (`sym72.permutation`) |
| `Q[p]` | `F_dst[perm p] · M · F_src[p]^T`: source-frame components → image-frame components |
| `pin[p,a]`, `pout[j,o]` | Bin of the image port containing `Q` applied to the bin-centre direction (N = 4 / 8) |
| Transformed response | `S'[pin[p,a], perm[j], pout[j,o], c] = S[a, j, o, c]`, i.e. `K(gG) = P_out(g) K(G) P_in(g)^-1`; budget `d'[pin[p,a]] = d[a]` |

### Symmetry classes

| Class | Transformation | Status |
|---|---|---|
| 1 | Whole-geometry coordinate transform | Exact invariance by construction (material frame) |
| 2 | Exact material-domain symmetry (rest slab: D4h, 16 operations) | Exact, with the maps above |
| 3 | Source translation inside the fixed domain | Not a symmetry |

### Development run of the geometry contract (dev root, before the freeze; geometry only)

- **Rest (all 16 operations).**
  - Every map is exact: bin-centre residual ≤ 6e-16, every per-port map a bijection.
  - Every tangent map is a D4 element. Normal error 0, tangent error 0.
  - Fine16 and coarse-cell aggregation commute with every map.
  - β agrees within quadrature noise (relative 0.58–0.60 %).
  - The refined triangulation is not D4-symmetric (triangle match 0.08–1.0). The faces are flat, so the physical surface and the port regions are identical.
  - W+UV-Q inputs:
    - Pair inputs are exactly equivariant on every mapped edge. 98.8–99.9 % of the 128-NN edges map; the rest are ties at the neighbourhood boundary.
    - Node inputs are exactly equivariant except at 32–48 probe-tie ports per operation, where index-ordered tie breaking picks a different but symmetric probe.
- **Dented poses (pt0 → pt0 / pt0v images; te0 under D4): NOT exact.** The refined triangulation is not D4-symmetric, so the first-triangle frame of an image port in the dent differs from the transformed frame:

  | Quantity | Value |
  |---|---|
  | Normal tilt | up to 0.073–0.079 |
  | Bin residual | 0.06–0.16 |
  | Outgoing maps | not bijective for 90° rotations and the diagonal reflection |

  **Consequence (frozen):** exact transformed angular labels exist for the **rest geometry only**. Deformed-pose images are not used to build labels.

### Development run of the reference controls (dev root, before the freeze; partial, stopped to bound the cost)

- **Positive pairs.** The first 8 exact pairs, between already-referenced S16 sources related by r90 / r180 (e.g. 944 ↔ 2707, 960 ↔ 2732). The transformed reference matches the image source's reference at 0.95–1.03 × replica noise in the 0–5 mm and 5–10 mm strata, native and fine16 contrast.
- **Negative controls.** Every one detected: identity bin maps and t ↔ b-swapped bin maps.
- **Bound for the frozen run.** The frozen run caps the cost at the first 2 pairs per operation plus every self-fixed source.

## Stage 1 — Angular representability gate (frozen in this commit, before any oracle outcome)

**Sources.** The 16 S16-R rest training sources:
- fitting targets = the means of fitting replicas 0–3;
- scoring = held-out replicas 4–7 of the same bundles.

**Operations (chosen from the frame-map table).** `r90`, `r180`, `mx`, `sw`, `id+z`, `r90+z`. They cover 90° and 180° rotations, axis and diagonal reflections, top ↔ bottom exchange alone and with a rotation, and every side face mapped with a non-identity tangent map.

**Oracle.** Labelled oracle, using the WL74 Stage 2 procedure:
- per-source T, float64 FISTA from 0.5/P, `ato74.objective_m`;
- 3,000 iterations, then the same budget again from the solution as a convergence check;
- the frozen G2 basis (sha asserted);
- budget `d'` = the transformed budget of the original source, which isolates the angular permutation exactly.

**Gate.** For every operation and group (ALL / BND (nearest perpendicular face ≤ 10 mm) / INT), on abs_native, abs_fine16, abs_coarse, con_fine16 and con_coarse:
- the transformed pooled worst-channel upper-95 % bound must be ≤ 1.25 × the original oracle's;
- no WL59 gate passed by the original may be failed by the transformed oracle.

The gate also requires a valid Stage 0 contract.

**On failure.** **ANGULAR BASIS LIMIT**: stop. No A/O change and no rank change; the required architectural change is reported as a human decision.

**Algebraic closure (descriptive).** Rank-assignment closure of A, O and the joint rank-1 patterns under the 8 D4 tangent maps. The dev run of the closure module (basis only, no reference) was seen before the freeze. It is not gated: A, O and the joint patterns are **not closed** (relative assignment error 0.46–0.62), so exact equivariance cannot come from a rank relabelling. Whether per-receiver T can compensate is what the oracle measures.

**Dev smoke (non-evidence).** 5-iteration oracle and the decision path ran on the dev root. No number from it is used.
