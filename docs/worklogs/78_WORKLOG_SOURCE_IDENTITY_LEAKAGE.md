# Worklog — Source-Identity Leakage and Geometry-Relative Transfer Falsification (2026-11-07)

## Origin

- **Brief.** "Worklog 78 — Source-Identity Leakage and Geometry-Relative Transfer Falsification", 2026-11-07.
- **Question.** Is the boundary-transfer failure of the rank-16 W+UV intrinsic K (WL75 A75) caused materially by absolute-position conditioning and implicit source identity?
- **Intervention.** One bounded intervention is allowed: a geometry-relative, source-identity-free variant of the same operator, provisionally called W+UV-Q.
- **Nature of the experiment.** It is a causal architecture experiment. It is not a methodology claim.

| Item | Setting |
|---|---|
| Branch | `Selective_Recompute` |
| Server | LabServer63 (main) |
| Code | `experiments/fnto_srcid78/` |
| Server root | `~/rest_srcid78` |

**Preserved and untouched:** WL73 AR, WL75 A75, WL76/77 FIA, WL77A (including its rejected candidates and its negative discovery verdict).

**Correction carried in from WL77A** (append-only; no historical verdict is changed): the WL76/77 angular-failure attribution is confounded by q = 1024 single-scatter quadrature noise. See `docs/plans/wl77a/coordinator_decision.md` §2. That correction is not used or extended here.

### Inputs

| Input | Location | Provenance |
|---|---|---|
| A75 runs | `LabServer63:~/rest_rank75/runs` | `A75_5901.pt` sha256 c0022742…, `A75_5902.pt` 358c3881…; identical to the local WL75 originals |
| WL75 Stage 3 references (the WL74 Stage 4 population, already scored) | Local WL75 run, mirrored to `LabServer63:~/rest_rank75L` for development diagnostics only | 22 bundles; sha256 equal on both sides; `ports4.json` 94dc0479…; `refs/complete.json` 9f815c45… |

The WL77A Part 0 references (`~/rest_arch77a`) are not opened before the WL78 protocol freeze.

## Stage 0a — Code audit of source-identity pathways (A75 = WL59 'P' network at rank 16)

This audit is traced from code, not from earlier worklog prose.

### Node input

`runtime.Transport.layers` (`fnto_persist59/runtime.py:28-33`) encodes **only `node[:, :16]`**, the local block built in `r57_geometry.py:56-57`:

| Columns | Feature | Class |
|---|---|---|
| 0–2 | `xyz0 / 0.05`, absolute rest position (material frame) | **A: explicit absolute position** |
| 3–5 | `xyz / 0.05`, absolute current position | **A** |
| 6–8 | `d / 0.006`, rest → current displacement | C: deformation context; translation-invariant |
| 9–11 | normal, in material axes | D: boundary orientation; constant per face at rest; breaks the D4 × z-flip quotient |
| 12 | area ratio | C |
| 13 | local thickness | C |
| 14 | normal spread | C |
| 15 | face id / 5 | D/B: ordinal face label; redundant with the normal at rest |

**Correction to WL77A.** WL77A listed the global 11 × 11 dent grid (`node[:, 16:140]`) as an A75 input. That was wrong for the P network: the grid is computed but never read. It entered only the historical D/R arms of WL57.

### Paths from the source to the output

- **Message passing.** Two residual layers with mean aggregation over the 16-NN graph (`runtime.py:30-32`). They aggregate absolute features of the neighbours, so z stays position-keyed. Boundary visibility is short, about 2 hops of 16-NN.
- **W (edge) path.**
  - Formula: `softplus(edge(s(z_i) + d(z_j) + pair(i,j))) · prior(dist)` over the 128 nearest receivers of each source (`rank75.py:33-37`).
  - Pair features: world-axis offsets `dr / 0.01` (translation-invariant, rotation-variant), `|dr|`, current distance and `n_i · n_j`.
  - The source enters through `s(z_i)`, which is absolute-position-keyed.
- **UV (latent) path.**
  - Formula: `T_UV[i,j,r,c] = Σ_k U[j,k,r,c] V[i,k,r,c]`.
  - `U[:,k,r,c] = softmax_j(node_head(z_j))` is a **source-independent global receiver distribution**: 16 per (mixture, channel), defined over absolute receiver features.
  - `V[i]` is the source's loading on these 16 receiver patterns. The source enters only through `z_i`.
  - With 16 training sources, V can address a separate global receiver pattern per source. That is a lookup keyed by absolute source position. This is the identity pathway B, and it is structural.
- **Normalisation.** `mass(i)` and `norm(i)` are per-source scalars from `z_i`, so the source is again absolute-keyed.
- **Geometry caches.** `node`, `pair`, `prior` and the 128-NN edge list are computed from the current geometry in the object's material frame. `geometry(V_world, rigid_R, rigid_t)` maps world vertices into that frame first.
  - Consequence: a **whole-geometry translation or rotation of the fixture together with its sources and receivers is an exact invariance of A75 by construction**, invariance class 1.
  - Absolute-position conditioning therefore cannot break class 1. It can only matter for class 2, a source translated inside the fixed finite domain, and for class 3, exact slab symmetries.
- **A and O.** These are the global, frozen G2 bases and carry no source information.
- **No source-indexed learned table exists.** Identity can enter only through features.

### Physical transformation law

| Class | Transformation | What holds |
|---|---|---|
| 1 | Whole system (geometry + sources + receivers) translated or rotated | Exact invariance. Already exact in A75. |
| 2 | Source translated within the fixed 100 × 100 × 10 mm slab | Not an invariance. The response changes wherever the boundary context inside the transport range changes. |
| 3 | D4 × z-flip at rest (16 operations) | Exact, with the receiver-port permutation and the incoming/outgoing bin permutations taken from the port frames. A75 is not equivariant by architecture; L_sym only supervised integrated responses of the training-source images. |

## Stage 0b — Measured invariance and leakage diagnostics (development data; 2026-11-07)

`stage0_78.py`, run on LabServer63 → `results/fnto_srcid78/stage0/stage0.json`. Inputs are archived, already-scored references only. The WL77A Part 0 references were not opened.

**D1. Physical equivalence of the historical translate comparison.** Three boundary-far WL74 Stage 4 sources each have an S16-R analogue: exact slab operation + in-face translation, exact receiver-port map (no unmapped flux), and exact incoming/outgoing bin maps from the port frames.

| Pair | Operation | Δ (mm) | Second-boundary distance |
|---|---|---|---|
| 101 ← 50 | mx + z-flip | +22.5 | 18.75 → 41.25 |
| 105 ← 52 | z-flip | −30 | 18.75 → 48.75 |
| 107 ← 53 | z-flip | +22.5 | 18.75 → 41.25 |

- **Every stratum agrees within reference noise.** Replica-mean relative difference vs replica noise, both channels-pooled:

  | Stratum | Share of the native signal | Native difference | Native noise | Fine16-contrast difference | Contrast noise |
  |---|---|---|---|---|---|
  | 0–5 mm | 99.0–99.9 % | 0.002 | 0.002 | 0.005 | 0.005 |
  | 5–10 mm | — | 0.010–0.013 | equal | — | — |
  | 10–20 mm | — | 0.03–0.05 | equal | — | — |
  | farther | — | noise-dominated | — | — | — |

- **The second-boundary change has no measurable effect.**
- **β rows** agree to 0.01–1.4 %, which is β quadrature noise.
- **Conclusion.** The comparison is a genuine full-response physical equivalence, to the resolution of the references. It is more than the incident-energy invariants that WL77A's check C3 compared.

**D2. A75 on the same pairs** (both seeds; near field 0–5 mm):

| Quantity | Native | Fine16 contrast |
|---|---|---|
| A75 difference between the far-source prediction and the mapped training prediction | 0.12–0.33 | 0.42–1.41 |
| Physical difference | 0.002 | 0.005 |
| A75 error on the far source | 0.14–0.35 | 0.48–1.45 |
| A75 error on the mapped training source | 0.09–0.12 | 0.23–0.30 |

- **Where the difference sits.** It is carried by the W path: the UV share within 10 mm is about 0.
- **Interpretation (inference).** A75 assigns different near-field responses to physically identical contexts. For 105 ← 52 and 107 ← 53, the inputs that differ are only the absolute coordinates and the sign of a world-axis pair offset. The face id and the material-axis normal are equal.

**D3. Transport outside the W support** (128-NN, radius 12.5–17 mm):

| Sources | Collided energy outside (R / G / B) |
|---|---|
| Interior | 5.5 / 3.2 / 0.1 % |
| Boundary | 1–2 % |

The native and fine16-contrast L2 signal outside the support is about 0.

**D4. Origin offset of the absolute-coordinate inputs alone** (physically identical input; offsets of 2.5, 10 and 22.5 mm):

| Source | Prediction change | Error |
|---|---|---|
| s0, interior training | 0.3–2.4 % | unchanged |
| s50, boundary training | 2.5–23 % | 0.12 → 0.15–0.26 |
| s101 | 1.5–10 % | — |

**Stage 0 answer (measurement + inference).** Absolute-coordinate conditioning is a causally active source-identity pathway in A75. Source-position keys determine the near-field response of boundary sources, and on physically equivalent contexts A75 fails by 2–3× its supervised error. The pathway is separable within the W+UV family: it enters through the 16 node inputs and the 3 pair offsets read by two input layers. Stage 1 is therefore defined.

## Stage 1 — Pre-evidence architecture contract (frozen in this commit)

**Protocol.** `experiments/fnto_srcid78/protocol/srcid78_v1.json`. Code: `q78.py` (the inputs and the wrapper), `fit78.py`, `test78.py`, `eval78.py`, `mech78.py`, `cost78.py`, `chain78.py`, `export78.py`, `run78.sh`.

### W+UV-Q

**What stays the same.** The A75 operator is unchanged: `rank75.TransportM`, rank 16, the frozen G2 A/O, the same layers and the same parameter count (273,040 learned).

**What changes.**
- **Node inputs (16, replaced).**
  - Rest → current displacement, in the port's bin frame.
  - A75 area, thickness and normal-spread columns, unchanged.
  - Two boundary probes. Probe 1 is the nearest port with a perpendicular normal. Probe 2 is the nearest port perpendicular to both the port and probe 1. Each probe contributes its unit offset expressed in the bin frame and weights `max(0, 1 − r/20 mm)²` and `max(0, 1 − r/5 mm)²`.
- **Pair inputs.** Rest offsets in the source bin frame, plus the three A75 scalar pair columns.
- **Removed.** Absolute rest and current position, the material-axis normal, the face id and the material-axis offsets.
- **Pre-freeze geometry check** (`featcheck.json`, geometry only):

  | Context | Input difference |
  |---|---|
  | Interior translates | Identical, including their 16-NN neighbourhoods |
  | Stage 0 pair sources, after the per-port frame map | 0.0037 (the 18.75 mm second-boundary weight) |
  | Their neighbourhoods | ≤ 0.096 (genuine boundaries within 20 mm) |

  176 ports at rest have probe-1 ties.

### Training and gates

**Initialisation.** As A75, except that the three input-layer tensors are re-initialised with seeds 7801 and 7802. This is a recorded confound. Training is otherwise identical to A75.

**Gates (frozen).**

| Stage | Rule |
|---|---|
| 2 | Tests T1–T11 |
| 3 | Fit gate: each seed ≤ 1.25 × the worse A75 seed on TRAIN_NEW and TRAIN_KEEP (rest and pt0), metrics abs_coarse, abs_fine16, abs_native and con_coarse, using S16-R held-out replicas; plus validity |
| Mechanism (development data) | The Stage 0 near-field equivalent-context difference falls to ≤ 0.5 × A75, both seeds, native and contrast |
| 4 | WL77A Part 0 population. Far rule: Q78 ≤ 0.8 × the stronger of AR and A75 on BND_FAR con_coarse, beyond its lower 95 % bound. Regression ×1.25 against the better control. Verdict mapping as in the brief |
| 5 | Cost limit 2 × A75 (apply, rebuild including inputs, geometry-dependent state), measured before training |

**Predictions (frozen):** P1–P5 in the protocol.

### Pre-freeze runs (code checks only, no evidence)

- **Tests.** T1–T11 on LabServer63 passed 11/11, after fixing three test-code defects:
  - a float tolerance;
  - a rank-8-only runtime helper;
  - a qualifier string match.
- **Training probe.** A 12-step full-scale probe ran at 0.149 s/step with a 4.6 GB peak.
- **Not read.** No Part 0 response value was read. T11 reads only `complete.json` and the hashes.

### Amendment 1 (2026-11-07; after the dev smoke, before the evidence chain)

**What the dev smoke found.** It ran with the initial model on the dev root, so it is not evidence. It stopped at the frozen cost gate:

| Ratio | Value |
|---|---|
| `rebuild_incl_features` | 5.78 |
| apply64 | 1.00 |
| apply1 | 1.01 |
| Geometry-state bytes | 1.00 |

**Defect.** The rebuild ratio was asymmetric:
- W+UV-Q side: input construction (0.088 s; frames + probes) + coefficients.
- A75 side: coefficients only (0.017 s). It left out A75's own input construction, the r57 geometry preprocessing, which both models need anyway.

**Change.** Both sides now include the shared geometry preparation (without ballistic B) plus their model-specific rebuild. The original ratio is still reported, ungated, as `rebuild_model_specific_only`. A variant that includes the shared ballistic B is reported too. The limit (2.0) and the gated list are unchanged.

**Confirmations.** No candidate metric and no Part 0 response had been produced.

## Execution (2026-11-07, LabServer63)

**Chain.** `run78.sh` ran as one detached process at commit 694eb93.

- **Dev smoke.** 40-step fits, with Stage 4 forced on the dev slots after the freeze. It completed. No metric from it was used.
- **Evidence chain** (`~/rest_srcid78`), stage by stage:

| Step | Result |
|---|---|
| Tests | 11/11 |
| Cost gate (amended definition) | Passed: apply64 1.00, apply1 1.00, rebuild 1.25, state 1.00 |
| Probe | 2 lanes, 0.31 s/step |
| Fits Q78_5901 / Q78_5902 | 6,270 s each (2 lanes, RTX 3080 Ti). Both converged (last-decile change 0.4 / 0.3 %). Final J 0.01623 / 0.01617; A75 was 0.01553 / 0.01551 |
| Stage 3 | Fit gate passed |
| Mechanism diagnostics | Completed |
| Stage 4 | Completed |
| Final cost | Completed |
| Export | `results/fnto_srcid78/` |

## Final report

### Implementation Fact

- **W+UV-Q is A75 with only its geometry inputs replaced** (`q78.py`).
  - **Replaced:** the 16 node inputs and the 3 pair offsets. They are now geometry-relative and expressed in each port's bin frame. No absolute position, material-axis orientation or face label remains.
  - **Identical to A75:** parameter count (273,040 learned), frozen G2 A/O, rank 16, objective, optimiser, schedule, S16-R population, steps and final iterate.
  - **Only other difference:** the three input-layer tensors use the default re-initialisation (seeds 7801 / 7802).
- **Amendments.** One, before any evidence: the definition of the cost measurement.
- **Not changed:** FIA, quadrature, rank, loss, thresholds, population, production code.

### Measurement

**Stage 0** (development data): see Stage 0b above.

**Stage 3 fit gate.** Data: S16-R held-out replicas 4–7, worst-channel upper 95 %.
- Both seeds sit at 0.95–1.05 × the worse A75 seed on TRAIN_NEW and TRAIN_KEEP, for abs_coarse, abs_fine16, abs_native and con_coarse. The gate was ≤ 1.25.
- Passivity passes on rest, pt0, cf3, cf2, te2 and te0. A/O are bitwise frozen.
- **FIT GATE PASSED.**

**Mechanism** (development data; frozen criterion: Q78 / A75 ≤ 0.5 counts as improved).

| Quantity (near field) | 5901 | 5902 |
|---|---|---|
| Native equivalent-context difference, Q78 / A75 | 1.00 | 1.03 |
| Fine16-contrast difference, Q78 / A75 | 0.91 | 0.93 |

The criterion fails: **not improved**.

Per pair, 0–5 mm, worst-channel contrast difference:

| Pair | A75 | Q78 |
|---|---|---|
| 101 ← 50 | 0.83–0.86 | 0.84–0.99 |
| 105 ← 52 | 0.50–0.53 | 0.64–0.71 |
| 107 ← 53 | 1.26–1.41 | 0.81–0.97 |

**Origin offset** of every coordinate tensor (10 and 22.5 mm):

| Model | Prediction change |
|---|---|
| Q78 | 0 (≤ 1e-4) |
| A75, boundary s50 | 10–23 % |
| A75, interior s0 | 1–2 % |

**WL75 Stage 3 population** (development, already scored; fine16 contrast):
- Interior sources: Q78 0.20 on every one, A75 0.22–0.46.
- Boundary sources: mixed. Source 107: Q78 0.85 / 1.01 vs A75 1.33 / 1.45. Source 105: Q78 0.73 / 0.66 vs A75 0.55 / 0.58.

**Stage 4.** WL77A Part 0 population, never scored before. Worst-channel upper 95 %, replicas 0–3. Each cell shows seed 5901 / 5902.

| Group: metric | AR | A75 | Q78 |
|---|---|---|---|
| BND_FAR con_coarse (primary) | 0.977 / 1.010 | 0.902 / 0.945 | **0.668 / 0.860** |
| BND_FAR con_native | 0.948 / 1.034 | 0.884 / 0.875 | 0.640 / 0.747 |
| BND_FAR con_fine16 | 0.952 / 1.039 | 0.877 / 0.868 | 0.622 / 0.731 |
| BND_FAR abs_coarse | 0.194 / 0.199 | 0.185 / 0.164 | 0.129 / 0.157 |
| BND_FAR abs_native | 0.274 / 0.302 | 0.269 / 0.245 | 0.195 / 0.233 |
| BND_NEAR con_coarse | 0.864 / 0.783 | 0.961 / 0.935 | 0.566 / 0.668 |
| INT con_coarse | 0.456 / 0.608 | 0.319 / 0.545 | 0.121 / 0.121 |
| INT abs_coarse | 0.094 / 0.110 | 0.059 / 0.088 | 0.051 / 0.049 |
| FP_rest con_coarse | 0.251 / 0.253 | 0.137 / 0.145 | 0.102 / 0.102 |
| FP_te1 con_coarse | 0.241 / 0.242 | 0.136 / 0.149 | 0.135 / 0.125 |

Deformation (DEF = te1 − rest), per channel R / G / B, seed 5901 ; 5902:

| Quantity | A75 | Q78 |
|---|---|---|
| abs_coarse | 0.31/0.28/0.36 ; 0.26/0.26/0.48 | 0.36/0.41/0.53 ; 0.31/0.36/0.65 |
| corr | 0.95/0.96/0.94 ; 0.97/0.97/0.88 | 0.93/0.91/0.85 ; 0.95/0.94/0.77 |

**Frozen decision.**
- **Stronger BND_FAR control:** A75.
- **Far: OVERLAP.**
  - Seed 5901 meets the rule: 0.668 ≤ 0.8 × 0.902 = 0.722, and it lies below A75's lower bound of 0.901.
  - Seed 5902 does not: 0.860 > 0.722.
  - The result is not NO either: the minimum, 0.668, is ≤ 0.8 × 0.945.
- **Near:** OVERLAP.
- **Regression:** NO for every listed group and metric.
- **Secondary BND_FAR metrics:** OVERLAP.
- **Physics at te1:** all models pass.
- **Verdict: INCONCLUSIVE.**

**WL59 absolute gates** (≤ 0.10 / 0.15 / 0.25):
- INT: Q78 passes in both seeds; A75 and AR fail.
- FP_rest: Q78 and A75 pass.
- FP_te1: all pass.
- BND_NEAR, BND_FAR and DEF (0.35 / 0.90): all fail.

**Per matrix** (con_coarse, Q78 vs A75):

| Matrices | Q78 | A75 | Q78 vs A75 |
|---|---|---|---|
| TBC far | 0.52–1.18 | 0.75–1.64 | better |
| SM far lo | 0.44–0.47 | 1.31–1.48 | better |
| SR far hi | 0.67–0.83 | 1.09–1.12 | better |
| TBE far | 0.78–0.91 | 0.61–0.85 | **worse** |
| SR far lo, seed 5902 | 1.06 | 0.80–0.97 | **worse** |
| INT far s22 | 0.107–0.108 | 0.33–0.59 | better |
| INT far s23 | 0.28–0.31 | 0.69–1.20 | better |

**Checkpoints** (BND_FAR con_coarse; descriptive only):

| Model | 2k | 6k | 10k | Final |
|---|---|---|---|---|
| Q78_5901 | 0.632 | 0.609 | 0.617 | 0.668 |
| Q78_5902 | 0.632 | 0.741 | 0.739 | 0.860 |
| A75_5901 | 0.839 | 0.867 | 0.886 | 0.902 |
| A75_5902 | 0.752 | 0.832 | 0.947 | 0.945 |

INT over training: Q78 stays at 0.105–0.121. A75 drifts 0.22 → 0.32 (5901) and 0.23 → 0.55 (5902).

**Cost** (final, RTX 3080 Ti, same process, Q78_5901 vs A75_5901):

| Quantity | Ratio |
|---|---|
| apply64 | 1.00 |
| apply1 | 1.01 |
| Runtime state | 1.00 |
| Rebuild, incl. shared geometry preparation | 1.26 |
| Rebuild, incl. ballistic B | 1.06 |
| Model-specific rebuild only (0.102 s vs 0.017 s; inputs 0.085 s) | 6.3 |
| Peak GPU above geometry | 1.44 |

Training took 6,270 s per run on 2 lanes. This is not comparable with A75's 4,084 s, which ran on another GPU.

### Observation

1. **Supervised fidelity is unchanged.** Removing the absolute and orientation inputs did not cost training fit.
2. **Identity dependence is gone.** Origin dependence is zero, and every interior source more than 20 mm from a boundary receives identical inputs.
3. **Independent sources.** Q78 is lower than both controls in every primary group and in every secondary BND_FAR metric, with no regression. The largest gains are interior and near-boundary.
4. **BND_FAR is seed-dependent.** The gain is large for one seed (−26 %) and small for the other (−9 %). The gap opens late in training: Q78_5902 goes 0.63 → 0.86 after step 2k.
5. **TBE-far sources do not improve** per matrix.
6. **Mechanism pairs.** These development pairs are related by a slab operation that rotates or reflects the bin frame. On them, Q78's prediction differences are as large as A75's, so the criterion fails.
7. **Deformation response is worse than A75's**, mainly in blue: corr 0.77–0.85 vs 0.88–0.94; error 0.53–0.65 vs 0.36–0.48.

### Interpretation (inference, labelled)

- **Interior contexts.** Absolute-position conditioning is a material cause of A75's failure.
  - Interior sources with equal context receive identical inputs and transfer at near-supervised fidelity: INT 0.12 vs 0.32–0.55, which passes the WL59 gate.
  - A75's late interior drift does not occur in Q78.
- **Boundary contexts.** Removing identity is not sufficient.
  - Every BND_FAR source differs from its nearest supervised context by a frame rotation or reflection, or by a boundary combination (for example, TBE-far's distant second edge).
  - Q78 sees these as different inputs. The frozen global A/O cannot express the bin permutation that an exact symmetry operation implies.
  - Learning that equivalence across frame orientations from 16 sources is not established, and the near-field mechanism pairs show it was not learned.
  - The remaining boundary failure fits a missing frame (orientation) equivariance of the operator plus late-training specialisation. It does not fit absolute-position identity alone.
- **Deformation.** The degradation suggests the frame-expressed displacement, or the lost absolute dent-relative position, removes information that the pt0-trained deformation response used. Not attributed.

### Unresolved Question

- Is the remaining BND_FAR error caused by frame-orientation non-equivariance under the global A/O? Unmeasured. Testing it needs an angular contract that permutes with the frame, which is outside this batch's invariants.
- Why do the seeds diverge late on BND_FAR, and would an earlier iterate (2k–6k) be systematically better? Checkpoints are descriptive only; no selection was allowed.
- What causes the deformation-response degradation: the displacement frame, the probes, or the lost dent-relative absolute position?
- Do TBE-far sources fail because their second edge lies beyond every supervised boundary combination, or because of orientation?
- The SV (vertical-edge) stratum is unmeasured on this population: too few qualified matrices.
- Carried over: the WL77A SS-quadrature-noise correction of the WL76/77 attributions remains an unresolved attribution question. WL78 neither uses nor tests it.

### Architecture Verdict

**INCONCLUSIVE** by the frozen Stage 4 rule: BND_FAR is OVERLAP because one seed meets the 20 % rule and the other does not.

| Area | Verdict | Basis |
|---|---|---|
| A. Coordinate / invariance correctness | **SUPPORTED** | Inputs are origin- and orientation-independent; class 1 invariant; class-3 frame-map law exact (T6/T7) |
| B. Supervised fitting | **SUPPORTED** | 0.95–1.05 × A75 |
| C. Independent boundary-far transfer | **INCONCLUSIVE** | Seed-dependent: −26 % / −9 % |
| C. Interior and near-boundary transfer | Improved descriptively, no regression | INT ×0.2–0.4; BND_NEAR −30 %. Not a frozen endpoint |
| D. Geometry response | **NOT SUPPORTED** relative to A75 | DEF worse; every model fails the DEF gate |
| E. Computational feasibility | **SUPPORTED** | Apply and state 1.0 ×; rebuild 1.26 × incl. shared preparation |
| F. Mechanism attribution | Split | Identity leakage **SUPPORTED** for interior contexts. Boundary equivalent-context invariance **not improved** (criterion failed). Boundary failure **not** attributable to identity leakage alone |

Nothing is adopted. No successor is started.

### Answers to the brief

1. **Is source identity materially encoded?** Yes. It enters through the absolute and orientation inputs that two input layers read.
   - Boundary training sources are origin-sensitive by 10–23 %.
   - Physically equivalent contexts receive near-field predictions that differ by 0.12–0.33.
   - Removing it changes interior transfer materially.
2. **Was the cited translate comparison a physically valid full-response equivalence?** Yes, within reference noise at every distance stratum. However, each pair includes a slab operation that rotates or reflects the bin frame, so it tests orientation equivariance as well as translation.
3. **Can a geometry-relative W+UV variant preserve supervised transport?** Yes: 0.95–1.05 × A75.
4. **Does reducing identity dependence improve boundary transfer?** Not established under the frozen rule: INCONCLUSIVE, with −26 % and −9 % for the two seeds. Interior and near-boundary transfer improve descriptively.
5. **Does the improvement extend beyond coarse contrast?** On BND_FAR, descriptively yes: con_native, con_fine16, abs_coarse and abs_native are all lower, though each is OVERLAP under the rule. It does not extend to the deformation response, which is worse.
6. **Is sensitivity to physical boundaries preserved?** At the input level, yes (negative control T8). But boundary errors stay large and the deformation response degrades.
7. **Is any gain worth its cost?** Cost is not a constraint here: apply and state are unchanged, and rebuild is 1.26 × including shared preparation.
8. **Does the W+UV family remain viable?**
   - Interior contexts: removing identity is enough to make the family transfer within this fixture.
   - Boundary contexts: the evidence points to a factor W+UV-Q does not change, namely frame-orientation equivalence under the frozen global angular basis, plus late specialisation.
   - This is an open question, not a recommendation.

### Reproducibility

- **Commits:**
  - cd8e0be: Stage 0, protocol and code frozen.
  - 694eb93: amendment 1. This is the execution commit, recorded in `~/rest_srcid78/code_commit.txt`.
  - Results are committed separately.
- **Protocol:** `experiments/fnto_srcid78/protocol/srcid78_v1.json`, including amendment 1.
- **Inputs:**

| Input | Hash or note |
|---|---|
| A75 runs (`~/rest_rank75/runs`) | c0022742… / 358c3881… |
| WL75 references mirrored to `~/rest_rank75L` | 22 bundles, hash-equal to the local originals |
| Part 0 `ports76.json` | 0819fa61… |
| Part 0 `geometry_te1.pt` | a2fbcc06… |
| Part 0 `refs/complete.json` | e754dcda… (1,920 packets, all gates passed) |

- **Seeds.** Q78 initialisations P_5901 / P_5902 with re-init seeds 7801 / 7802. References use the WL76-predeclared seeds 766/767xxxxxx.
- **Precision.** float32 models and training with TF32 off; float64 estimators.
- **Hardware.** LabServer63, RTX 3080 Ti 12 GB, 2 training lanes.
- **Results.** `results/fnto_srcid78/`:
  - `stage0/`
  - `final/`: `summary.json`, `group_metrics.csv`, `per_matrix.csv`, copies of the cost, gate and mechanism records, manifest
  - `stage4_analysis.json`
  - `mech/`, `cost/`, `probe/`
- **Server roots.** `LabServer63:~/rest_srcid78`; the dev smoke is in `~/rest_srcid78_dev`.
