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

## Execution (2026-11-07/08 UTC, LabServer63, RTX 3080 Ti, torch 2.3.1)

| Step | Result |
|---|---|
| Chain A, first run (commit d45f354) | Geometry contract completed. Stage 1 closure and oracle completed (5,702 s). The reference controls stopped at the first translation control (amendment 1). |
| Chain A, second run (amendment 1, commit ce9e626) | The reference controls resumed: 50 positive rows kept, 3 translation rows computed. Stage 1 decision: **ANGULAR BASIS LIMIT**. |
| Chain B | Started its dev smoke after chain A ended. The smoke stopped at its own tests (8/9 passed). The failing test is a defect in my synthetic decision-logic test: it compared the stored "stronger control" against a list instead of a name. That failure is non-evidence, and no training ran. The chain was then stopped deliberately. |
| Evidence chain (`chain79.py`, evidence root) | Read the Stage 1 gate, did not train, exported `final/`. |

**Not executed (the brief's Stage 1 stop):** Stages 2–6. That is: the symmetry-presentation training, the fit gate, the mechanism gate, the Stage 5 references and decision, deformation and cost. The frozen Stage 5 population (19 ports) was never referenced and remains **unscored**.

## Final report

### Implementation Fact

- **`sym79.py` — the exact symmetry contract of the rest slab.**
  - Port permutation from rest material coordinates.
  - Frame map `Q[p] = F_dst[perm p] M F_src[p]^T`.
  - Bin maps of the bin-centre directions.
  - Transformation `S'[pin[p,a], perm[j], pout[j,o], c] = S[a, j, o, c]`, i.e. `K(gG) = P_out K(G) P_in^-1`.
  - Budget `d'[pin[p,a]] = d[a]`.
- **Stage 0 (`stage0_79.py`).** Geometry contract over 32 maps (rest 16, pt0 8, te0 8) and reference controls on every archived rest bundle. The WL77A Part 0 bundles are included as development data.
- **Stage 1 (`oracle79.py`).** Algebraic closure of the frozen G2 A/O. Labelled per-source T oracles, using the WL74 Stage 2 procedure (float64 FISTA, 3,000 + 3,000 iterations), for the 16 S16-R rest sources: original and under 6 operations, 112 fits.
- **Stages 2–6.** Code and the frozen contract exist (`fit79.py` SymRest, `eval79.py`, `ports79.py`, `probes79.py`, `cost79.py`, `test79.py`, commit 8a176cc). None of it was executed for evidence.
- **Unchanged.** Production FNTO, Q78 / A75 / AR checkpoints, the G2 basis, every earlier root.

### Measurement

**Stage 0 — geometry contract.**

*Rest: all 16 operations exact.*
- Bin-centre residual ≤ 6e-16, and every per-port map is a bijection.
- The normal is preserved, and every tangent map is a D4 element.
- Fine16 and coarse-cell aggregation commute with every map.
- The refined mesh vertices are identical. The triangulation differs (match 0.08–1.0), but every face is flat, so the physical surface and the port regions are identical.
- β agrees to 0.58–0.60 % relative (quadrature noise).
- W+UV-Q inputs:
  - pair inputs are exactly equivariant on every mapped edge (98.8–99.9 % of the 128-NN edges map);
  - node inputs are exactly equivariant except at 32–48 probe-tie ports per operation.

*Dented poses (pt0 → pt0 / pt0v; te0 under D4): not exact.*

| Quantity | Value |
|---|---|
| Image-port frame normal error | 0.073–0.079 |
| Bin residual | 0.06–0.16 |
| Outgoing maps | not bijective for the 90° rotations and the diagonal reflection |

The cause is the first-triangle bin frame on a non-D4-symmetric refined triangulation in the curved dent. Exact angular labels therefore exist for the rest geometry only.

**Stage 0 — reference controls (rest).**

| Control | Count | Result |
|---|---|---|
| Positive: exact pairs | 30 (first 2 per operation, 15 operations) | All passed |
| Positive: self-fixed sources (replica halves) | 20 | All passed |
| Ruled negative controls (boundary sources) | 8 | Detected |
| Class-2 translation pairs (boundary distance 1.8 vs 21.3 mm; 43.8 vs 4.0 mm) | 3 | Detected |
| Unruled negative controls | 92 | 90 detected |

- **Positive-control size.** The largest near-field (0–5 / 5–10 mm) difference among the positive controls was **1.07 × replica noise**.
- **The two unruled negatives that were not detected.** These were identity-bin maps on sources whose tangent map at the source face is the identity (sw+z on top / bottom).
- **Contract: VALID.**

**Stage 1 — algebraic closure (descriptive).** G2 is not closed under the D4 bin permutations:

| Basis component | Best rank-relabelling error (relative) |
|---|---|
| A | ≤ 0.52 |
| O | ≤ 0.46 |
| Joint (A_r, O_r) | median 0.53, max 0.62 |

**Stage 1 — oracle.**
- **Convergence.** Maximum relative J change on doubling the budget: 1.7e-6.
- **Sources.** 16 S16-R rest sources: BND = slots 50–53 (nearest perpendicular face 1.8–4.0 mm), INT = 12 sources (21–44 mm).
- **Table.** Pooled worst-channel upper-95 % bound, transformed / original:

| Operation | Group | abs_native | abs_fine16 | abs_coarse | con_fine16 | con_coarse |
|---|---|---|---|---|---|---|
| r90 | BND | 1.05 | 1.09 | **1.26** (0.084 → 0.106) | 1.11 | **1.35** (0.129 → 0.174) |
| r180 | BND | 1.05 | 1.07 | **1.27** (→ 0.107) | 1.05 | **1.41** (→ 0.183) |
| mx | BND | 1.03 | 1.04 | 1.10 | 0.98 | **1.34** (→ 0.174) |
| sw | BND | 1.00 | 1.01 | 1.11 | 1.02 | 1.05 |
| id+z | BND | 1.07 | 1.10 | **1.27** (→ 0.107) | 1.11 | 1.21 |
| r90+z | BND | 1.02 | 1.02 | 1.05 | 1.01 | **1.38** (→ 0.178) |
| r180 | ALL | 1.02 | 1.03 | 1.16 | 1.01 | **1.26** |
| every op | INT | 0.97–1.00 | 1.00 | 0.97–1.07 | 0.99–1.01 | 1.00–1.03 |
| every op except r180 | ALL | 1.00–1.03 | 1.00–1.04 | 1.00–1.15 | 0.98–1.04 | 1.04–1.24 |

- **WL59 gate.** BND abs_coarse is newly failed for r90, r180 and id+z (0.106–0.107 vs the 0.10 gate; original 0.084). Every other WL59 gate passed by the original oracle is also passed by every transformed oracle.
- **Gate result.** **11 failures** of the predeclared rule. Classification: **ANGULAR BASIS LIMIT** (stop).
- **Per source** (worst-channel rho ratio > 1.15):
  - the four boundary sources: con_coarse up to 1.62 (s53 id+z), abs_coarse up to 1.82;
  - interior source s4: abs_coarse up to 1.72 at a small absolute value, con_coarse ≤ 1.10;
  - interior sources s0 / s3: con_coarse 1.17–1.18 under r180.
- **Native and fine16 metrics.** Within 1.0–1.24 per source.

### Observation

- The transformed responses that the frozen basis represents less well are boundary-source responses, and the loss appears mainly in the **coarse-cell metrics**: coarse incoming-angle contrast and coarse absolute.
- Native and fine16 errors change little (≤ 1.11 pooled).
- Interior sources are essentially orientation-invariant under the basis (≤ 1.07).
- The diagonal reflection sw is almost free (≤ 1.11). Rotations, the axis reflection and the combined z-flip are not.

### Interpretation (inference, labelled)

1. **The angular basis is not equivariant.** Under the rule fixed before the oracle ran, the frozen rank-16 G2 A/O cannot represent the exact symmetry images of the boundary training responses at the level it represents the originals. The algebraic closure analysis is consistent with this: no rank relabelling maps the basis onto itself (error ≈ 0.5). Per-receiver T compensates almost fully for interior sources, but not for boundary sources.
2. **Confound: in-sample basis.** The G2 basis was fitted (in WL74) on exactly these 16 sources in their original orientation. The original oracle is therefore in-sample for the basis and the transformed oracle is out-of-sample. Part of the 1.26–1.41× gap may be orientation-specific basis fitting rather than an intrinsic limit of any rank-16 basis. This batch does not separate the two.
3. **Training was not attempted.** Within the stated constraints (A/O frozen, rank fixed), presenting these symmetry images to the learned operator would ask it to fit labels whose oracle floor for boundary sources is up to 1.4× worse in coarse contrast. That is why the brief stops here. No measurement of what such training would do exists.
4. **Relation to WL78's unresolved factor.** WL78 left frame-orientation equivariance under the frozen global A/O as an unmeasured factor. WL79 measures one necessary part of it: **the frozen A/O is not orientation-closed for boundary responses.** Whether this limit *causes* the Q78 boundary-transfer failure is not measured; no candidate was trained.

### Unresolved Question

1. **Basis fitting vs. intrinsic limit.** How much of the transformed-oracle gap comes from fitting the basis to the 16 training orientations, and how much from an intrinsic rank-16 limit? Examples of a separating test: a G2-like basis fitted on orbit-augmented targets, or a held-out-orientation basis. Not authorised here.
2. **Dented poses.** Exact angular symmetry labels do not exist there in the current bin convention (first-triangle frames). Whether that matters for deformation transfer is open.
3. **Q78's late drift and its deformation regression** (WL78) remain unexplained. They were not touched.
4. **Human decision on the architecture change.** To obtain exact angular-frame equivariance, the A/O (or the operator class) would have to transform with the bin frame. Examples: a permutation-closed basis, or bins defined in a frame canonicalised by the local boundary context. Both lie outside this batch's frozen A/O and rank, and the choice belongs to the user.

### Architecture Verdict

**ANGULAR BASIS LIMIT** (Stage 1 stop under the predeclared rule).

- **Stage 0 contract: VALID.** Rest maps are exact; dented-pose maps are not exact.
- **Not measured** (no training, by protocol):
  - symmetry-training benefit;
  - learned equivariance;
  - independent transfer;
  - deformation;
  - candidate cost.
- **Adoption.** Nothing is adopted. No successor batch is started.
- **Earlier results.** All earlier verdicts (WL75, WL77A, WL78) are unchanged.

### Answers to the brief

1. **Is the frozen rank-16 global basis expressive enough for the transformed responses?**
   - **No, under the predeclared rule:** 11 rule failures, all in the BND group plus one in ALL.
     - BND abs_coarse 1.26–1.27× under r90 / r180 / id+z, with WL59 abs_coarse newly failed at 0.106–0.107.
     - BND con_coarse 1.34–1.41× under r90 / r180 / mx / r90+z.
     - ALL con_coarse 1.26× under r180.
   - Interior sources and the diagonal reflection are within the rule.
   - Caveat: the basis was fitted on the original orientations (in-sample).
2. **Were the physical and angular symmetry mappings implemented correctly?**
   - **Yes for the rest geometry.** Exact maps, aggregation commutation and group composition (dev test T1). The 50/50 positive controls agree within 1.07 × replica noise. Ruled negative controls 11/11, including 3/3 translation pairs.
   - **Not exact for dented poses** (bin-frame convention).
3. **Can symmetry-consistent training preserve Q78's supervised fidelity?** Not measured; Stage 1 stop.
4. **Does it materially reduce the mismatch between equivalent boundary contexts?** Not measured.
5. **Does the improvement transfer to independent, out-of-orbit sources?** Not measured. The frozen 19-port population remains unscored.
6. **Does it address late-training drift?** Not measured.
7. **Does it preserve or recover deformation fidelity?** Not measured.
8. **Measured computational cost.**
   - No candidate exists, so no candidate cost.
   - Batch cost: Stage 1 oracle 5,702 s GPU (112 float64 fits); Stage 0 geometry contract a few minutes; Stage 0 reference controls about 1.5 h CPU in parallel.
9. **Does the result justify retaining W+UV-Q, or is a different operator class needed for exact angular equivariance?**
   - With A/O frozen and the rank fixed, W+UV-Q **cannot represent** the exact symmetry images of boundary responses at the original accuracy. Exact angular equivariance therefore requires an angular component that transforms with the frame, i.e. a change to A/O or to the operator class.
   - Whether to make that change, and whether the effect is basis fitting or intrinsic, is an open human decision (Unresolved Questions 1 and 4).
   - The W+UV-Q spatial findings of WL78 are unaffected.

### Reproducibility

**Commits.**

| Commit | Content |
|---|---|
| d45f354 | Stage 0/1 freeze |
| 8a176cc | Stages 2–6 freeze |
| e5fa2b8 | Test fix |
| ce9e626 | Amendment 1 (deployed commits recorded in `results/fnto_sym79/code_commit_*.txt`) |

Deployed with `git archive` (`core.autocrlf=false`) to `~/rest_sym79/repo`.

**Inputs (read-only).**

| Input | Identity |
|---|---|
| G2 basis | `G2_r0`, sha256 000fc7aa… |
| S16-R fit targets | `fit_targets73.pt`, sha256 20f631ed… |
| Q78 / A75 | sha256 in the protocol |
| Archived reference bundles | `common79.BUNDLE_DIRS` |

**Hardware and environment.** LabServer63, RTX 3080 Ti 12 GB, torch 2.3.1+cu121. Oracle in float64; estimators eval68 (float64).

**Outputs.** `results/fnto_sym79/final/`, with `manifest.json`:
- Stage 0 geometry and references;
- Stage 1 closure, oracle and gate;
- `summary.json`.

Server root: `~/rest_sym79`.
