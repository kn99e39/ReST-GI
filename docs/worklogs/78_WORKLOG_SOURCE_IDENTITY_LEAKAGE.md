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
