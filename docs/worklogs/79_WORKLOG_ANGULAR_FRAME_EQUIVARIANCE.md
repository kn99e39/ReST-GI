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

**Execution of Stages 0–1.** Chain A (`runA79.sh`) runs on LabServer63 from commit d45f354 (`~/rest_sym79/code_commit_A.txt`). The Stage 0 reference controls run on CPU in parallel with the Stage 1 oracle on GPU.

## Stages 2–6 — Training contract and downstream rules (frozen in the next commit, before any candidate fit)

**When this was frozen.** Chain A was still running. The only Stage 1 output seen was the progress line of the first original-oracle chunk: J = 0.0094, no transformed value.

### Stage 2: the intervention (`fit79.SymRest`)

- **Presentation.** At optimizer step t, rest S16-R source k is presented under operation `OPS[(k + t) mod 16]` (`sym72.ops(True)`, identity first). This is a deterministic Latin schedule: every step contains all 16 operations once.
- **What is transformed.**
  - Ports: the source and all receivers.
  - Incoming and outgoing bins.
  - The label: the exact image of the fitting-replica-0–3 mean.
  - The model budget: 1 − β of the rest geometry at the image port.
- **Implementation.** The labels are rebuilt in place each step by exact gathers in the channel-major target layout.
- **pt0.** Identity presentation only, because dented-pose maps are not exact.
- **Unchanged.**
  - Operator: `q78.TransportQ`.
  - Objective: J + 1.0 L_sym, with the WL72 term unchanged.
  - Samples per update: 16 rest + 16 pt0.
  - Optimizer: Adam, cosine 3e-3 → 3e-5, 20,000 steps, clip 1.0, final iterate.
  - Initialisation: the Q78 rule with the same seeds (bitwise equal initial state).
- **Disclosed confound.** The rest sources are seen at the 192 ports of their orbit, each orientation 1/16 of the time. This is exact information without new MC labels. It is not a new physical reference budget.

### Stage 3: fit gate

Each S79 seed must satisfy ≤ 1.25 × max(Q78 seeds):
- groups: TRAIN_NEW / TRAIN_KEEP × rest / pt0;
- metrics: abs_native, abs_fine16, abs_coarse, con_fine16, con_coarse;
- plus validity (passivity, A/O identity).

**Also reported.**
- Q78 reproduction against the recorded WL78 Stage 3.
- Exact images of the held-out replicas under the Stage 1 operations (descriptive).
- Checkpoints, as evidence on late drift.

### Stage 4: mechanism gate

- **Gate (M1).** On the historical pairs 101 ← 50, 105 ← 52 and 107 ← 53, near field 0–5 mm, native **and** fine16 contrast: S79 ≤ 0.5 × same-seed Q78, both seeds.
- **Descriptive.**
  - M2: learned equivariance for all 15 operations, on training and out-of-orbit development sources.
  - M3: WL77A Part 0 groups, including DEF te1 (development only).
- **If the gate fails.** Stage 5 is not run.
- **Orbit check.** The far ports 224 / 260 / 1568 and the WL74 S4 / WL77A Part 0 ports lie outside the 192-port training orbit.

### Stage 5: prospective population (`ports79.py`, geometry only)

**Exclusions.**
- 107 earlier referenced or scored ports and all their images.
- The 192-port training orbit.

**Population.** 19 rest slots:

| Group | Slots |
|---|---|
| BND_NEAR | TBE ×2, TBC ×1, SM ×2, SR ×2 (7) |
| BND_FAR | TBE ×2, SM ×2, SR ×2 (6) |
| INT | 4 |
| UNS | SV near, 2 |

- **No admissible port** (recorded, not replaced): TBC slots 5–7 and SV far.
- **References.** Unchanged probes74 code, K = 4, seeds 79xxxxxx. They are generated only after both gates pass.
- **Decision.** As in WL78, with the stronger of Q78 / A75 as the control:
  - far YES: ≤ 0.8 × both control seeds and below their lower 95 % bounds;
  - regression: ×1.25 on native / fine16 / coarse (and con_coarse for INT).
- **Verdict mapping.** Protocol `stage5.decision.verdict`.

### Stage 6: deformation and cost

- **Deformation.** Existing qualified references: DEF71, CONFDEF72, DEF73 and Part 0 DEF te1 (primary; the WL78 regression population).
  - WORSE: worst-channel error ≥ 1.25 × max Q78, or correlation ≤ min Q78 − 0.03.
  - RECOVERED: the symmetric criterion.
- **Cost.** Q78 vs S79. A runtime ratio that deviates from 1 by more than 25 % is reported as significant.
- **When.** Always run. If an earlier gate stopped, the results are descriptive.

### Execution

**Chain B** (`run79.sh` → `chain79.py`):
1. Wait for chain A.
2. Dev smoke on `~/rest_sym79_dev`: 40-step fits, 2 reference slots at 1M photons, forced gates; no metric used.
3. Evidence chain.

Any failure stops.

### Amendment 1 (2026-11-08; before any Stage 0 / Stage 1 decision)

**What happened.** Chain A stopped in the Stage 0 reference controls, at the first class-2 translation control, with "bins do not permute". The Stage 1 oracle had completed. Its output was not read, and no gate had been evaluated.

**Defect.** `sym79.Map` assessed bin exactness over every port. For a partial translation map, the unmapped receivers carried a placeholder frame, so the assertion failed even though every mapped port permutes exactly.

**Change.**
- Exactness is assessed over the mapped ports only. Exact symmetry maps are total, so they are unaffected.
- The reference controls resume. The rows already written by the identical positive / negative code are kept; only the remaining rows are computed.

**Unchanged.** Every rule, threshold and population, and the oracle output.

**State before the stop (log).** Every positive pair and self-fixed source was within 0.97–1.04 × replica noise, and every ruled negative control was detected.
