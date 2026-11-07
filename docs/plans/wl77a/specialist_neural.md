# WL77A specialist report — Agent 2: Neural operator / representation (2026-11-07)

**Assigned role.** Agent 2, neural operator / representation lens of the WL77A multi-agent architecture-discovery workflow.

**Research question.** What information must be represented explicitly, and what can be learned as a shared operator, so that a learned intrinsic light-transport operator K stays accurate when the source position changes? The focus is on independent, previously unseen source ports, especially near boundaries and far from supervised sources.

**Status.** COMPLETED. The work was read-only on the repository except for this file. No GPU, server or training was used. One local CPU Monte-Carlo proxy was written in the session scratchpad. It is labelled **NON-EVIDENCE** throughout (Section 5).

---

## 0. Summary

1. **W+UV (rank 8/16) fails transfer because the source enters its spatial map only as an absolute-position embedding.** It also has a built-in memorisation channel.
   - **Absolute position.** The node encoder sees absolute rest and current xyz, an ordinal face id, and world-axis pair offsets. At rest, no other feature distinguishes two top-face ports (WL71 Stage 1).
   - **Short boundary visibility.** Two 16-NN mean-message layers let a port "see" a boundary only within about 5–10 mm. Red transport reaches about 18 mm (diffusion length) and needs boundary awareness out to about 36 mm.
   - **Memorisation channel.** The low-rank tail U·V uses global, absolute-receiver bases of rank 16 per mixture and channel. S16-R has 16 sources, so U·V can represent a per-source lookup table exactly.
   - **Late drift.** WL71A/71B show this happening late in training: U concentrates on fit-port neighbourhoods, and the corner source's V grows about 800×. The cause is null-space drift under Adam once the 16-source loss is flat. It is not a capacity limit: rank 16 reaches the rank-16 oracle on supervised sources (WL75) and leaves far-boundary transfer unchanged.
2. **RSK removes absolute position but keeps four other obstacles to transfer.**
   - **(a) Orientation.** The relation frame is world-aligned on the top and bottom faces. Under slab symmetries it is reflection- and rotation-variant. It also does not match the bin frames on the side faces (f0: RSK (t, b) = (−y, +z), bin frame (+z, +y)).
   - **(b) Boundary description.** Boundary context is a single scalar "distance to nearest non-coplanar port", with no direction and no second boundary.
   - **(c) Sub-cell dependence.** The far tail is quantised to the 1 cm metric grid, so predictions depend on the source's sub-cell position.
   - **(d) Shared CP pairing.** The fixed shared A/O CP pairing remains.
   - **BND_FAR is not "same context" for RSK.** Every WL73 BND_FAR port differs from its training analogue by a symmetry image (another face or edge) *and* by a second-boundary distance outside the training range: e₂ = 41–46 mm against 18.75 mm. That is input extrapolation for RSK, not interpolation.
3. **FIA's WL76 fit failure and WL77's Stage 1 STOP are, on present evidence, mostly an estimator artefact. The angular-expressivity explanation is a minor part.** **NEW FINDING (non-evidence CPU proxy, Section 5).**
   - **The noise term.** The deployable single-scatter term SS is a fixed-seed Monte-Carlo quadrature with q = 1024 samples per (port, bin) cell. Its native-resolution noise is about 0.45 × ‖M‖ (multiple-scatter part) and about 0.26 × ‖S‖ (total collided response). This noise is *common to all reference replicas*, so the replica half-split "noise floor" (0.002) cannot detect it.
   - **The proxy reproduces WL77's three projection floors almost exactly:**

     | Basis | Proxy | WL77 |
     |---|---|---|
     | Ring | 0.449 | 0.453 |
     | Azimuthal (az) | 0.384 | 0.385 |
     | Free order 2 | 0.352 | 0.352 |

   - **It also reproduces FIA's supervised native error:** 0.29 predicted, 0.29–0.30 observed.
   - **True floors are much lower.** The *noise-free* multiple-scatter residuals are 0.22 (ring), 0.067–0.070 (receiver-relative az) and 0.017–0.022 (free order 2). On a noise-free M, the WL77 gate (az ≤ 0.5 × ring) would have passed: 0.07/0.22 ≈ 0.31.
   - **Contrast.** The same SS noise alone gives a coarse directional-contrast error of 0.49 (source at a cell-corner sub-cell) to 1.24 (cell-centre sub-cell). WL76 observed 0.69–0.75 pooled.
4. **What a transferable shared rule must see.**
   - Relative geometry only: no absolute position, no global receiver bases.
   - Boundary-relative coordinates *with direction*, out to the channel transport length.
   - One canonical frame shared by the spatial relations and the angular bins, so that slab symmetries become identical inputs (a quotient, not an augmentation).
   - An incidence-dependent sub-port footprint shift computed geometrically.
   - Low-variance deterministic geometric terms.
   - A composition over boundary planes, so that unseen boundary combinations (second-edge distance, corners, vertical edges) are composed, not extrapolated.
5. **Eight mechanisms (M1–M8) are proposed in Section 7,** each tied to a specific failure. Two are new relative to WL59–WL77:
   - M5, deterministic geometric terms, which removes the FIA/WL77 artefact;
   - M7, reciprocity-derived supervision. It is exact for this medium and turns every supervised source's boundary receivers into partial supervision of boundary sources.

---

## 1. Input sources read

**Code (read in full or in the cited parts):**

| File | What it defines |
|---|---|
| `experiments/fnto_rule57/r57_model.py` (whole) | The W+UV algebra: `coefficients` and `apply` (l.48–83) |
| `experiments/fnto_rule57/r57_geometry.py` (whole) | Node, pair and prior features (l.40–74); ballistic quadrature q = 256, seed 57501 (l.80–98) |
| `experiments/fnto_persist59/runtime.py` (whole) | WL59 P arm: encoder on `node[:, :16]`, two message layers (l.21–38) |
| `experiments/fnto_repr68/family68.py` (whole) | Exact admissible family and objective J |
| `experiments/fnto_cover71/family71.py` | Batched W+UV restriction |
| `experiments/fnto_cmp72/rsk72.py`, `sym72.py`, `fit72.py` l.25–95 | RSK, and the L_sym construction |
| `experiments/fnto_fia76/fia76.py` (whole) | FIA |
| `experiments/fnto_fia76/diag/posthoc76.py` | WL76 post hoc diagnostic |
| `experiments/fnto_fia77/gate77.py` (whole) | WL77 Stage 1 statistic |
| `experiments/phys_ref53/p53_geom.py` l.1–73, 201–207 | Bins and frames |
| `experiments/phys_ref53/p53_tracer.py` l.90–102 | `cell_source` |

**Results:**
- `results/fnto_fia77/stage1_expressivity.json` (per-source floors);
- `results/fnto_fia76/diag_posthoc76.json`;
- `results/fnto_src73/ports.json` (S16-R and confirmation port geometry).

**Worklogs:**
- 71, 71A, 71B, 72, 73, 74A, 75, 76 and 77 (in full);
- 77A (the Part 0 freeze).

WL53/57/59/66/68 were used through the summaries in the brief, AGENTS.md and 74A; they were not re-read in full.

---

## 2. W+UV (rank 8 / rank 16): what the learned map is, and what it discards or entangles

### 2.1 Exact input → output (verified in code)

**Operator.** For source port i, incoming bin a and channel c:

`K_scat(i,a→j,o,c) = d_{i,a} · Σ_r A[r,a,c] · T[i,j,r,c] · O[r,o,c]`

with d = 1 − β, A = softmax over r and O = softmax over o. Both A and O are global: no port, face or geometry input (`r57_model.py:25-26,67`).

**Spatial map.** `T = W·scale scattered to the 128 rest-coordinate nearest receivers + U·(V·scale)`, with `scale = mass/(ΣW + ΣV)` (`r57_model.py:60-66`).

**Source and receiver features.** In the P arm, `z = 2 × (z + .25·MLP([z, mean_16NN(z)]))` over `silu(Linear(node[:, :16]))` (`runtime.py:28-34`). `node[:, :16]` is:
- rest xyz/5 cm and current xyz/5 cm (**absolute position**);
- displacement;
- normal;
- area ratio;
- thickness;
- normal spread;
- `port_face/5`, an **ordinal** face id (`r57_geometry.py:58`).

**Edges.** `h = src_layer(z_i) + dst_layer(z_j) + pair_layer(pair)`. The pair features are the rest offset in **world axes**/1 cm, |dr|, the current distance and n_i·n_j (`r57_geometry.py:65-66`). After the edge MLP and softplus, a fixed, channel-dependent physical prior multiplies the result: `exp(−dist/L_c)·a²/(dist²+a²)`, L = 18.1 / 5.8 / 1.8 mm (l.67–68).

**Tail and mass.**
- U is a softmax over **all P receivers** of the receiver node head: a global destination basis of rank 16 per mixture and channel.
- V and mass come from the source's node head.

**Training.** The loss is J (`family68.objective`) on 16 one-hot sources (S16-R) × 16 bins × 2 geometries, with A/O frozen (group 'net').

### 2.2 How source identity enters T

- **(E1) Position is the only identity at rest.** At rest every top-face port has identical non-position features (WL71 Stage 1). Two top-face sources are distinguished only by xyz plus neighbourhood means.
- **The edge pre-activation carries the source embedding into every edge.** `s_i` is a learned function of absolute position, added to every edge. In the same way, V_i and mass_i are learned functions of absolute position.
- **Consequence.** The map from source to response is a regression over a 2-D position manifold sampled at 16 points, roughly 20 mm apart.
  - The physical rule is translation-invariant in the interior and changes sharply inside thin strips near the six faces. For red the strip is about 1–2 diffusion lengths, i.e. 18–36 mm wide.
  - A position regression from 16 samples cannot tell where those strips are. It can only interpolate between supervised positions. This is the "supervision-local" behaviour of WL71: within 10–18 mm a ≤ .085 / c .19–.38, while the uncovered corners fail.
- **(E2) Boundary visibility is short-range.** Without absolute coordinates, a port learns about a nearby face only through the 2-hop 16-NN mean messages: 16 ports ≈ 5.6 mm radius, ≲ 11 mm after two hops, diluted by the mean. Beyond that, "near an edge" is recoverable only from absolute xyz. The physics needs boundary awareness out to about 36 mm for red, 12 mm for green and 4 mm for blue.
- **(E3) Orientation is not factored out.**
  - World-axis pair offsets and an ordinal face scalar mean the slab symmetries (D4 × z-flip, with face-dependent local bin maps, WL74A §3.3) are not quotiented out.
  - A response learned at the +x top edge says nothing structurally about the −x bottom edge.
  - The A/O atoms are fixed in *local* bin index space, while the edge MLP learns spatial shifts in *world* coordinates. The world shift that a given A_r needs therefore differs per face (WL74A §3.1 frame table), and must be learned separately per face.

### 2.3 The UV tail is a memorisation channel (an identifiability hole)

`T_UV[i,j,r,c] = Σ_{k<16} U[j,k,r,c] · V[i,k,r,c]`, with U a global, source-independent distribution over absolute receivers.

- **Why it can memorise.** With 16 training sources and rank 16, an exact per-source table is admissible: U[:, k] = the far field of source k, and V[i, ·] = one-hot. `family68.py:12-13` itself notes that "with 4 probe sources per geometry and rank 16 ≥ 4 every such T is approached by W+UV". Nothing in J prefers the translation-covariant member over the table.
- **The training trajectory shows the memoriser being approached** (WL71B):
  - U's column mass within 1 cm of the fit ports grows 0.06 → 0.56;
  - its entropy falls 11.7 → 6.6 bits (about 100 effective ports);
  - the uncovered corner source's V grows 0.03 → 23.6;
  - the UV share of that source's flux rises 3 % → 54 %;
  - its near-source concentration collapses 0.91 → 0.30.
- **A translation family cannot fit this form.** In a homogeneous medium the physical far field is a *translated* kernel. As a matrix over absolute receivers, a translation family has rank that grows with the number of distinct sources, so a global rank-16 basis cannot represent it for 3,840 sources.

### 2.4 Normalisation / mass

`mass_i = sigmoid(node_head(z_i))` is a free per-source scalar at supervised positions. At unsupervised positions it is an extrapolation. WL71B's slot 5 lost 0.78 → 0.62 of its mass through this term alone (mass factor degrade .73 / recover .59).

### 2.5 Angular pairing (WL74A, unchanged here)

T has no incoming index. The only way the incoming bin changes *where* light goes is the per-mixture pairing A_r ↔ T_r ↔ O_r, with global A/O. The physical incidence effect is a lateral first-collision footprint shift of about sin θ·1 mm. On the 1 cm metric grid it becomes contrast that depends on sub-cell position (|HZ|/|Z| 0.05 at a centre sub-cell, 0.17 at a corner). The shared CP pairing turns this into the 0.27 (G0) / 0.13 (G2) structural floors.

### 2.6 Late drift (WL71A / WL71B / WL73 / WL75)

**Observed pattern:**
- Cross-source error is best at 500–2,000 steps and then grows ×3–5, while the training J keeps falling slowly.
- The median raw gradient norm drops to about 0.004.
- Adam's step is scale-free (lr × m/√v), so directions with tiny but consistent gradients still receive full-size steps.

**Mechanism (inference; consistent with all four batches, not isolated):**
1. Absolute-position conditioning and the global U make the response at unsupervised positions a set of near-null directions of the 16-source loss.
2. Once the loss is flat, the optimiser moves along them freely: V at the corner, the mass at slot 5, and W for deformation.
3. Early checkpoints transfer better only because they stay closer to the initialisation (P_5902, trained on full-field data and translation-similar: within-1-cm flux 0.895–0.903 everywhere).

This is not label-noise fitting. The fit targets are qualified at signal/SE ≥ 10, and FIT on held-out replicas stays at 0.134–0.138.

### 2.7 Capacity versus learnability

| Evidence | Reading |
|---|---|
| WL75: rank 16 reaches the G2 oracle on supervised sources (0.131 vs 0.129); BND4_FAR unchanged (0.79 → 0.80–0.82); the learned–oracle gap grows 0.53 → 0.65 | Capacity is sufficient; transfer is not a capacity problem |
| WL71 teachers: TB (smooth P_5902 map) is recoverable from S4; TA (a physically fitted map) is not recoverable even from S16 | The procedure can generalise a rule that is smooth *in the network's coordinates*. Physical fits land on maps that are not smooth in those coordinates. |
| WL73: the three spatial families reach the same G0 floor on the new boundary sources (0.110 / 0.119 / 0.270) | Supervised fit is bounded by the angular class, not by the spatial map |
| WL73/75: BND_FAR is 3–5× the oracle under both bases | About 70–80 % of the far-boundary error is spatial, set by the learned map |

**Interpretation.** The physical rule is smooth in *boundary-relative* coordinates. It is a 1-D function of the edge distance per boundary plane, plus a 2-D function at corners. In *absolute* coordinates it is piecewise: constant in the interior, with steep changes inside the boundary strips. The W+UV inputs present it in the second form. Sixteen samples identify the second form only near the samples.

---

## 3. RSK (WL72/73): position-free, but not a quotient of the symmetries

**Verified in `rsk72.py`.** The model is `T[i→j] = mass_i (w_loc + w_far·share)/Σ`. Its inputs are:
- the relation, in the source frame, to 128 neighbours and to 240 coarse-cell centroids;
- position-free descriptors: normal, thickness, scalar edge distance, area ratio, normal spread and displacement;
- a 2 cm context.

A/O are frozen G0, as in W+UV.

It removes E1 (absolute position), yet BND_FAR stays at 1.14–1.15 and BND_NEAR is equal to D1 (WL73). Four reasons:

### R1. The frame is not symmetry-consistent

`frames(n)` sets t = x − (x·n)n on the top and bottom faces (world-aligned) and t = ẑ × n on the side faces (l.16–21).

- **Top and bottom faces.** A mirror or rotation changes the sign or axis of the relation features. The +x-edge TBE training source (port 3439, top) and the BND_FAR TBE port 232 (bottom, −x edge) are exact physical images, but RSK sees the side receivers at +t in one case and −t in the other. That is input extrapolation for an MLP.
- **Side faces.** The frames differ from the bin frames. On f0, RSK uses (t, b) = (−y, +z) while the bin frame is (+z, +y); computed from the cross products in `rsk72.py:19-21` and WL74A §3.1.
  - The mixture A_r has a fixed incoming azimuth in the *bin* frame, while RSK learns its spatial shift in the *RSK* frame.
  - The rotation linking the two frames is face-dependent, so each side face needs its own learned mapping.
  - S16-R has one SM and one SR source, on f0 and f2. The far SM/SR ports are on f2/f3 and f0 respectively (`results/fnto_src73/ports.json`).

### R2. The boundary descriptor is a scalar without direction

`edge` = the distance to the nearest non-coplanar port centroid, capped at 5 cm (l.27–28).

- It does not say *which direction* the boundary lies in, relative to the source or to a receiver.
- It cannot tell an edge from a corner.
- It carries no second-boundary distance e₂.

### R3. The training range of e₂ does not cover the far ports

| Port | Context | e₁ | e₂ |
|---|---|---|---|
| TBE train 3439 | TBE | 1.25 | 18.75 (y = −31.25) |
| TBE far 232 | TBE | 1.25 | 46.25 |
| TBE far 195 | TBE | 1.25 | 38.75 |
| SM / SR train | SM / SR | — | vertical-edge distance 18.75 |
| SM / SR far | SM / SR | — | vertical-edge distance 41.25–46.25 |
| TBC far 47 | near-corner | 8.75 | 8.75 (no corner training source has e = 8.75) |

At e₂ = 18.75 mm, red responses still feel the second edge: L_red ≈ 18 mm, and 1 cm / 2 cm flux fractions are 0.90 / 0.995. "Same context" is therefore not the same input for any model that sees e₂, and is not the same physics in red.

### R4. The coarse tail breaks translation equivariance

`w_far` is computed against the 240 coarse-cell centroids and spread uniformly within each cell (l.56–57, 65). The far field is therefore quantised to the 1 cm metric grid, and identical sources at different sub-cell positions receive different relation inputs.

### Effect of the data-augmentation symmetry (Candidate A)

`fit72.py:32-46,64-71` builds the L_sym labels as permuted *angle-integrated* references at the exact images of the supervised sources. The result:
- A gets 15× (rest) / 7× (pt0) image supervision of the *angle-integrated* field only.
- Directional contrast at the images is unconstrained (WL74A P11).
- This is why A is the best at BND_FAR (0.87–0.94 vs 1.14–1.30) and still fails.

---

## 4. FIA (WL76/77): the right factorisation, an estimator and representation that hide it

### 4.1 What FIA gets right (verified)

- **The factorisation is exact for this medium:** `K_scat = (1−β)·E_{x~p1}[albedo·(E1 + M)]`, with Q source-independent by the Markov property and g = 0 (WL76 Stage 1: 36/36; the forward control was rejected).
- **Source identity is removed by construction.** The only learned function is `R_θ(anchor k → j, o)`, and its features are symmetry-invariant (`fia76.py:49-53,105-109`).
- **The incidence budget is exact:** energy 1.002, incident profile 0.023 (posthoc76).
- **The error is uniform across all sources** (contrast 0.69–0.75, native about 0.32). WL76 notes this is a different failure from W+UV/RSK.

### 4.2 The estimator problem (new; non-evidence proxy, Section 5)

**How SS is built.** In the deployable runtime and in the fit, SS is an expected-value Monte-Carlo quadrature with **q = 1024 samples per (port, bin)** and a fixed seed (`fia76.py:65-89,142-152`; protocol `quadrature_q: 1024`). Each sample deposits its whole single-scatter weight into **one** (receiver port, outgoing bin) entry, out of 3,840 × 64.

- The true single-scatter field of one cell occupies roughly 10³ native entries. With about 500 effective upward samples, every entry is noise-dominated.
- **WL77 statistic.** WL77 projects `M = mean(S_ref) − albedo·SS_q` (`gate77.py:45-48`). Its "noise floor" is the half-split of reference replicas only (l.45, 52); SS_q is identical in both halves, so its error is invisible to the floor.
- **WL76 fit.** The FIA prediction contains the same SS_q (`fia76.py:127-128`), so the fit can never cancel it.
- **The W term has the same estimator.** W is the second-collision mass, snapped to the nearest of 4P anchors, from the same q = 1024 samples, and it adds noise that is independent per incoming bin. Its share is not quantified here; Section 9 lists it as an open item.

**Proxy result (Section 5 gives the method):**

| Quantity | Proxy (interior top source) | WL76/77 observed |
|---|---|---|
| Ring projection floor of `M_q` | 0.449 | 0.453 (0.437–0.462 interior) |
| Receiver-relative az floor of `M_q` | 0.384 | 0.385 |
| Free order-2 floor of `M_q` | 0.352 | 0.352 |
| Same floors on *noise-free* M | ring 0.220 / az 0.067–0.070 / free 0.017–0.022 | — |
| FIA native error floor = √(SS-noise² + ring-floor²)/‖S‖ | √(0.261² + 0.127²) = 0.290 | err_total native 0.293–0.300 per source |
| Coarse SS noise / ‖S_coarse‖ | 0.087 | err_total coarse 0.077–0.153 |
| Coarse contrast error from SS noise alone | 0.49 (corner sub-cell) / 1.24 (centre sub-cell) | FIA con 0.69–0.75 |

The boundary training sources point the same way. In WL77 the floor rises with the single-scatter share: s51 has ss_share 0.36 and ring 0.476, s52 has 0.247 and 0.454. That is what quadrature noise predicts. A physical in-ring structure would not be expected to follow the SS share so closely.

**Reading.**
- WL76's "native limit = ring lobes" and WL77's "remaining structure is higher-order azimuthal" are largely confounded with the q = 1024 SS quadrature error.
- With noise-free SS, the receiver-relative azimuthal lobe would lower the native multiple-scatter floor from about 0.22 to about 0.07. WL77's predeclared Stage 1 rule (≤ 0.5 × ring) would have passed.
- The real ring restriction does exist (0.22 of ‖M‖, about 0.13 of ‖S‖). It is the smaller term.

### 4.3 Representational restrictions of R_θ (code-verified, not quantified)

- **Invariant features that are too invariant.**
  - The anchor and receiver features are *sorted* free paths (`free_paths`, l.39–43). The pair features are only `[dist, cos(r, n_j), n_j·n_k, exp(−dist/2 mm)]` (l.107–108).
  - The rule cannot tell whether a receiver lies *toward* or *away from* a boundary relative to the anchor. Only the receiver's own sorted free paths, which carry no direction, hint at it.
  - Invariance under symmetries was obtained by discarding relative orientation. Equivariant pair-frame features would keep it.
- **Lateral snapping.** Second collisions are snapped to the nearest port-column anchor (2.5 mm lateral, depths 0.25 / 0.75 / 1.5 / 3 mm; l.27, 80). The incidence-dependent shift is about 1 mm. It survives only through how nearest-anchor binning splits the mass, so it is partly lost. Contrast attribution is still open (WL76).
- **Global ring lobes.** `L_m` are 3 *global* ring profiles mixed per pair (l.101–104). In the proxy, receiver-relative azimuth captures most of the true structure (0.22 → 0.07), so this restriction is a real but secondary cost.
- **Transfer moves to anchor space.** R is supervised only at anchors that the supervised sources' second collisions reach, within a few mm of each source. Because R's features are invariant, it should transfer to unsupervised anchors in the same feature region. That transfer was never measured: there was no Stage 4.

### 4.4 Cost

The rebuild cost is dominated by the per-geometry quadrature: 74.3 s of 75 s (WL76). The SS noise falls as 1/√q:
- SS noise / ‖S‖ ≤ 0.05 needs q ≈ 28 k, about 27× the rebuild time.
- A centre-sub-cell contrast noise ≤ 0.1 needs q ≈ 160 k, about 150× the rebuild time.

More samples is therefore not a viable fix. The geometric terms must become deterministic or smooth (M5).

---

## 5. NON-EVIDENCE CPU proxy: the single-scatter quadrature-noise check (method)

**Scripts (scratchpad, not in the repository):** `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\ssnoise.py`, `ssnoise2.py`, `ssnoise3.py`, `ssnoise4.py`. Plain numpy, seeds 1 / 7 / 11 / 3, about 1–3 min each.

**Geometry.** One interior top-face source port on a slab 10 mm thick and laterally infinite. Exits are tallied over ±51 mm on the top and bottom faces.
- Units: σ_t = 1 mm⁻¹; g = 0; index-matched boundaries.
- Albedo (0.999, 0.99, 0.9).
- Ports: 2.5 mm.
- Bins: `p53_geom` concentric 4×4 incoming / 8×8 outgoing, with the f5 / f4 frames copied.

**Estimators.**
- **SS.** The estimator of `fia76.quadrature`, copied: chord c = 1 − e^{−σt}, truncated-exponential first-collision depth, isotropic w₁, weight c·e^{−σt₂}, deposited into the exit (port, bin).
- **M.** An analog walk with per-channel albedo^k weights, keeping exits with ≥ 2 collisions.
- **Norms without bias.** ‖μ‖² = (‖ΣX‖² − Σ‖X‖²)/(N(N−1)). The quadrature noise² is (E‖X‖² − ‖μ‖²)/q at q = 1024.
- **Residuals.** For each basis the per-receiver projector is applied to the mean (unbiased) and to single spikes (noise).
  - The free basis uses its true rank (SVD). WL77's QR on the rank-deficient free basis removes one extra arbitrary dimension per receiver, a negligible difference.
  - The az basis uses φ_r from the source-port centre, as in `gate77.py:48-49`.

**Stability.** Seeds 1 / 7 / 11 / 3 agree to ±0.001 on the predicted WL77 statistics and to ±0.004 on the true-M residuals.

**Plausibility checks.**
- The coarse |HZ|/|Z| of the proxy is 0.171 at a cell-corner sub-cell and 0.054 at a centre sub-cell. These match WL74A's 0.17 / 0.05, an independent check that the proxy reproduces the physics and the metric.
- The single-scatter share is not reported by the proxy; WL77 gives 0.25–0.26 for interior sources.

**Limits.**
- One interior configuration only. Boundary sources and deformed poses are not modelled.
- No 1 − β_B rescaling (negligible in the interior).
- The noise of W is not modelled.
- These are not eval68 worst-channel jackknife upper bounds. They are pooled-channel point values.

**The cheap confirmation, for a later authorised step and not done here:** recompute `gate77.expr` with SS from an independent high-q estimate (for example q = 65,536 for one or two sources) or a deterministic cubature. Then check whether the floors drop to about 0.22 / 0.07 / 0.02.

---

## 6. What a source-independent shared rule must see to transfer, and why it can still fail

### 6.1 Required information

Derived from the failures above and the physics: g = 0, homogeneous, index-matched.

1. **Relative geometry only (translation equivariance).**
   - In the interior, the response depends only on the receiver offset relative to the source (or the anchor), the incoming bin and the outgoing bin.
   - Any absolute-coordinate input, any global receiver basis (U), and any per-source free scalar (mass) is a channel through which the model can store source identity.
2. **Boundary-relative geometry with direction, for each nearby boundary plane.**
   - The signed distance to the plane, its normal expressed in the source/pair frame, and the free path along the pair direction.
   - This is needed out to the transport length per channel: about 2 L, i.e. ≈ 36 / 12 / 4 mm for R / G / B.
   - A scalar nearest-boundary distance (RSK) or *sorted* free paths (FIA) are not enough.
3. **One canonical frame shared by the spatial relations and the angular bins.**
   - With it, the 16 slab-symmetry images of a source become *identical* inputs, a quotient, rather than having to be learned.
   - This makes the frame convention part of the operator contract. WL74A already found that the side-face tangent convention alone produces half of the SR orientation excess.
4. **The incidence-dependent sub-port footprint shift, computed geometrically.**
   - The first-collision distribution p1(x | i, a) is exact geometry (FIA, Stage 1).
   - A shared spatial kernel evaluated at that shift transports the incidence information.
   - An incidence-free T coupled through a global CP pairing does not (WL74A mechanism 1).
5. **Deterministic, low-variance geometric terms.**
   - Contrast is 1–3 % of the response energy (WL74A §6.6).
   - Every deployed geometric quadrature (SS, W, β) therefore needs a per-(cell, bin) variance well below that. With MC at q ≈ 10³ this is impossible (Section 4.4).
6. **A composition over boundary planes.**
   - Edge, corner and vertical-edge contexts are combinations of a few planar boundaries.
   - Transfer from {interior, single edge at e₂ = 18.75} to {single edge at e₂ = 46}, {corner at 8.75 / 8.75} or {vertical edge} requires the rule to *compose* plane effects, not to regress a joint context vector.

### 6.2 What a learned shared operator can carry

Given items 1–6, the learned part reduces to a few smooth functions:
- an interior post-collision emitter response;
- a per-plane boundary correction depending on (distance, angle to the plane);
- the outgoing angular lobe parameterised receiver-relative (order ≤ 2; proxy: noise-free M az floor 0.07).

Each of these is supervised by *every* training source, because every source's footprint samples many (offset, plane-distance) pairs. Supervision then scales with the number of receivers × anchors, not with the number of sources. This is the opposite of the per-source identifiability in W+UV.

### 6.3 Why such a rule could still fail

| Failure route | Evidence or reasoning |
|---|---|
| **Uncovered boundary configurations** | S16-R has no vertical-edge (SV/UNS) source and no top/bottom corner at e = 8.75. If plane interactions at corners are strong (red within 18 mm of two planes), composition will be biased there. |
| **Near-boundary non-diffusive regime** | Within 1–2 mfp of a plane, image/diffusion-type composition is inaccurate. The learned correction must capture it from a handful of boundary sources (4 in S16-R). |
| **Deformation** | Two training poses (rest, pt0). Every architecture fails the signed deformation gates (DEF71/73/4). Dent geometry is a curved boundary; plane-based composition needs local tangent-plane approximations whose error is unknown. |
| **Metric amplification** | At centre sub-cells |HZ|/|Z| ≈ 0.05, so contrast error is about 3× more sensitive than at corner sub-cells. A rule can be physically consistent and still fail con_coarse ≤ 0.25 there. |
| **Residual null space** | Equivariance removes the source-identity null directions, but not, for example, an unconstrained boundary correction at unseen plane-distance values. Late Adam drift can reappear there (the WL71A pattern). |
| **Cost** | Exact geometric anchoring costs per-geometry work (FIA: 75 s, almost all MC quadrature). Deterministic replacements must stay near the W+UV budget (about 0.015 s rebuild) to be useful for F/S. |
| **Angular class** | Global A/O CP pairing keeps a floor (G0 0.27 / G2 0.13 even on supervised sources). If the shared rule keeps ATO, the angular floor stays. |

---

## 7. Proposed mechanisms

These change generalisation behaviour; none of them adds layers or capacity. Each entry states the target failure, the evidence, the expected signature, the risks and the cost.

### M1 — Canonical boundary-relative frame: quotient, not augmentation

**Mechanism.**
- Per port, define one frame used for *both* the spatial relation features and the 16/64 bin indexing.
- Choose it from the local boundary configuration, deterministically: t points toward the nearest boundary plane; ties are broken by a fixed rule, for example the second plane; on a face's interior, a fixed, slab-symmetry-covariant convention.
- Exact slab symmetries then map a source and all its images to identical inputs. The bin permutations are absorbed by the frame.

**Target.**
- RSK R1 and W+UV E3: BND_FAR ports are images on other faces and edges.
- WL74A orientation diversity: at M = 8, shared refit 0.14–0.19 versus 0.064–0.075 when aligned. Alignment removes the whole multi-orientation excess.
- The SR frame-convention share: 0.142 → 0.101.
- WL72 A's angle-integrated-only symmetry.

**Expected signature.** Identical predictions for exact images, bitwise up to the β quadrature; WL75 already found β asymmetric by up to 0.057 for some pairs. The TRAIN_KEEP/NEW fit unchanged or better. The BND_FAR error splits into an "e₂ extrapolation" part only.

**Risks.**
- Frame choice is discontinuous where two planes are equidistant (corners, diagonal), so equivariance breaks on a measure-zero set. Neighbouring ports may get different frames, which requires bin re-indexing at frame seams.
- A dented face has no exact symmetry, so only approximate canonicalisation is possible there.

**Cost.** Negligible: a geometry preprocessing step.

### M2 — Remove the identity channels: relative-only kernel, no global receiver basis, no free per-source mass

**Mechanism.**
- Delete absolute xyz from the inputs.
- Replace the global U·V tail with a translation-equivariant far kernel: a function of the relative offset and the boundary-relative descriptors, for example an analytic diffusion-type Green's-function prior times a learned correction on relative features.
- Make the mass a function of boundary-relative descriptors only.
- **Principle.** The hypothesis class must not contain the per-source lookup table (rank ≥ number of sources).

**Target.** WL71 supervision-local transfer, the WL71A/71B drift (V × 800 at the corner via fit-concentrated U; slot-5 mass), and the WL73 removal cost at REM_NEAR (×2.5–3).

**Expected signature.**
- Interior sources at any distance from supervision match the supervised interior fit.
- The drift ratio (final / @2000) on unseen sources falls toward 1.
- The UV-type "flux within 1 cm" collapse cannot occur.

**Risks.** The deformation (dent) response is genuinely position-dependent through geometry. It must enter through *current-geometry* relative descriptors (local thickness, curvature, displacement field in the pair frame), and two training poses may not identify it. RSK already did most of this and still failed BND_FAR: M2 is necessary, not sufficient, and needs M1 and M3.

**Cost.** Similar to or lower than W+UV; RSK state was 33.9 M floats with a 36 ms build.

### M3 — Compositional per-boundary-plane structure (image / truncation composition)

**Mechanism.**
- Write the spatial response as an interior kernel composed with per-plane boundary operators. Examples:
  - a method-of-images-style subtraction or reflection term per nearby plane;
  - multiplicative truncation factors.
- Each depends on that plane's distance and its normal in the pair frame, through saturating features exp(−e/L_c).
- Corners and vertical edges are compositions of two planes plus a small learned interaction term.

**Target.**
- The e₂ extrapolation (far TBE/SM/SR ports at e₂ = 41–46 mm against 18.75 in training).
- The near-corner TBC far port (8.75 / 8.75).
- The never-supported UNS/SV contexts (contrast 1.0–1.5 in every model).
- WL74A §4.5: edge shapes lie 85–90 % in the cone of interior shapes. The genuinely boundary-specific part is truncation at about ±5 %, consistent with a composition.

**Expected signature.** BND_FAR error approaching BND_NEAR error, i.e. the boundary context transfers along the edge and across faces. The UNS stratum below 1.0.

**Risks.**
- Strong plane–plane interaction at corners in red.
- The near-plane (< 2 mfp) error of diffusion-type images.
- Curved dents need tangent-plane approximations.
- With only 4 boundary training sources, the per-plane correction is identified mainly by *receivers* near planes, which every source has within the transport range (red), but only weakly for blue.

**Cost.** Cheap: a few extra terms per edge.

### M4 — Geometric incidence anchoring at continuous positions (incidence-dependent shift without CP pairing)

**Mechanism.**
- Compute the first-collision distribution p1(x | i, a) deterministically. Options: its first moments (mean depth and lateral shift, and covariance) analytically from the chord geometry; or a small deterministic cubature.
- Evaluate the shared spatial kernel at the shifted, continuous anchor, with no snapping to port columns.
- Incoming dependence then enters through *geometry* (where the light is), not through a global A_r ↔ T_r pairing.

**Target.**
- The WL74A structural floor: the shared CP pairing with angle-independent T acting on a lateral shift of about 1 mm relative to the 1 cm grid. This gives G0 0.27 / G2 0.13 even with a per-source oracle T.
- The FIA anchor snapping (2.5 mm lateral columns).

**Expected signature.** The supervised contrast falls below the G2 oracle (0.129) without a rank increase. The sub-cell ordering (centre < edge < corner) of the residual flattens.

**Risks.**
- First moments may not be enough near grazing incidence or at boundaries, where p1 is truncated.
- Within 1 mfp the single-scatter outgoing structure depends on the exact p1 shape.

**Cost.** An analytic per-(i, a) moment: O(P·16) per geometry, far below FIA's 75 s.

### M5 — Deterministic, low-variance geometric terms (replaces the q = 1024 MC SS / W)

**Mechanism.** Any geometry-only term that is deployed and also subtracted in the targets must have a native-resolution variance well below the metric's sensitivity. Options:
- **(a)** Single scatter by deterministic cubature. The integrand over (entry point, incidence, collision depth, exit direction) is smooth except near the source; integrate per receiver port and bin, for example with stratified QMC or Gauss–Legendre in depth × exit-solid-angle, using next-event-style evaluation into each (j, o).
- **(b)** A shared, learned, smooth single-scatter kernel: a function of the offset, the incidence bin, the exit bin and boundary-relative features. It is source-independent and translation-equivariant, and is fitted to high-q quadrature offline.
- **(c)** W as a smooth density over continuous anchor positions, rather than nearest-anchor MC counts.

**Target.** The FIA native floor of 0.26 × ‖S‖ (proxy), the FIA coarse contrast of 0.49–1.24 from SS noise alone (proxy), and WL77's Stage 1 floors (0.449 / 0.384 / 0.352 = noise-dominated).

**Expected signature.**
- Re-scored WL77 Stage 1 floors near 0.22 / 0.07 / 0.02.
- An FIA-family supervised fit with native error ≈ the ring floor (about 0.13 × ‖S‖), and less with M6.

**Risks.** Cubature cost near grazing exits, and near-source singular behaviour (exits on the source port itself). (b) adds a learned component that must itself be checked for transfer.

**Cost.** Must stay ≪ 75 s per geometry. Analytic single scatter in a half-space plus a boundary-truncation correction is O(P × local support × bins).

### M6 — Receiver-relative azimuthal outgoing lobe for the multiple-scatter part (conditional on M5)

**Mechanism.** Use the WL77 FIA-AZ lobe ν(1 + a₁cos(φ_o − φ_r) + a₂cos2(φ_o − φ_r)) per (anchor, receiver), on a noise-free target.

**Target.** The true ring restriction, 0.22 of ‖M‖ (about 0.13 of ‖S‖) in the proxy.

**Evidence.** Proxy: noise-free az floor 0.067–0.070, free order-2 0.017–0.022. WL74A §4.2: location-integrated a × o is nearly rank 1 (σ₂/σ₁ ≈ 0.02). Multiple-scatter outgoing structure is low-order once it is conditioned on the source–receiver geometry.

**Risks.** The proxy is interior only. Near boundaries, exits on side faces at grazing angles may need more azimuthal resolution. WL77's statistic at boundary sources (0.47–0.48) is also noise-inflated.

**Cost.** WL77 dev probe: state 1.63 GB, apply 5.75 × A75. The apply cost is dominated by the dense anchor × 512-neighbour local response, not by the lobe.

### M7 — Reciprocity-derived supervision or constraint (exact physics; a declared derived label)

**Mechanism.**
- For g = 0, a homogeneous medium and an index-matched boundary, the kernel is reciprocal.
- For unit-flux cell sources (uniform over the port area, uniform in projected solid angle) and flux tallies:
  A_i Ω_a K(i,a → j,O) = A_j Ω_O K(j,O → i,a), with O a 16-resolution outgoing bin (Ω = π/16 for both).
- Each supervised source i therefore yields, for *every* receiver port j, the response of *source* j in incoming bin O at receiver i. Outgoing is aggregated to fine16, which is exactly the model's incoming resolution.
- Use these entries as additional targets (derived from the same S16-R physical supervision; declared as derived, not privileged), or as a model self-consistency penalty.

**Target.**
- Source-coverage starvation: 16 source positions against 3,840.
- In particular, **boundary sources' responses at interior receivers**, from the abundant boundary receivers of interior training sources. For red, interior S16 sources with e = 21.25 mm put measurable flux on edge and side receivers.
- The far-field / tail mass routing behind the WL71B UV drift.

**Expected signature.** Less drift and better mass at uncovered sources (slot-5 / slot-6 type). Improved abs_coarse at BND_FAR in red and green.

**Limits.**
- Reciprocity gives a source's response only at the supervised ports. For sources ≥ 16 mm from all supervised ports this is the tail: within 1 cm lies 90 % of the flux, within 2 cm 99.5 %. It therefore does little for the near field that dominates contrast, and almost nothing for blue (L ≈ 1.8 mm).
- It does not hold for the rescaled d = 1 − β_model targets unless the ballistic and β conventions are handled consistently (reciprocity holds for the total response, and for the collided part separately as well, since B is itself reciprocal).
- Port areas differ on dented poses (use the actual A_j).

**Cost.** Training only. No runtime change.

### M8 — Shrink the optimiser's null space structurally

This replaces the use of checkpoint timing to control drift.

**Mechanism.** The late drift (WL71A ×5 coarse / ×3 contrast between 2k and 20k steps; WL73/75/76 BND_FAR drift) is null-space motion under scale-free Adam once J is flat. The structural remedy is M1 + M2: tie every unsupervised source to supervised data through equivariance and relative features, so that motion in null directions changes supervised outputs too. Two non-structural complements:
- function-space anchoring of unsupervised sources to a physical prior (an analytic or diffusion prior as the zero point of the learned correction, with the correction penalised in magnitude);
- reporting the trajectory of an unsupervised *development* group during training (descriptive only; checkpoint selection stays forbidden).

**Target.** The WL71A/71B drift; the @2000-better-than-final pattern of WL72/73/75/76.

**Risks.** Prior anchoring biases the result wherever the prior is wrong (near boundaries). It has to be declared as a modelling choice.

**Cost.** None at runtime.

**Mechanism versus failure.**

| Failure | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 |
|---|---|---|---|---|---|---|---|---|
| Source identity via position (W+UV) | | ● | | | | | ○ | ● |
| Memorisation via U·V (W+UV) | | ● | | | | | ○ | ● |
| Orientation / frame (RSK, W+UV, WL74A) | ● | | | ○ | | | | |
| e₂ / corner / vertical-edge extrapolation | ○ | | ● | | | | ○ | |
| Shared CP pairing / sub-cell contrast floor | | | | ● | | | | |
| SS / W quadrature noise (FIA, WL77) | | | | ○ | ● | | | |
| True ring restriction (FIA) | | | | | | ● | | |
| Late drift | ○ | ● | | | | | ○ | ● |

● = primary target; ○ = secondary effect.

---

## 8. Rejected claims and ideas

| Claim / idea | Status | Reason |
|---|---|---|
| More rank or capacity in W+UV fixes transfer | Rejected | WL75: rank 16 = G2 oracle on training (0.131); BND4_FAR 0.79 → 0.80–0.82; the gap widens to 0.65 |
| More source coverage of the same family is enough | Rejected as sufficient | WL71: transfer is supervision-local (10–18 mm); WL73: near ×3, far small or mixed, removal cost ×2.5–3 |
| An angle-integrated symmetry loss gives equivariance | Rejected | `fit72.py:32-46`: labels are integrated over a and o, so directional contrast is unconstrained (WL74A P11); A BND_FAR 0.87–0.94 |
| RSK's failure shows that "relational" models cannot transfer | Rejected as a general claim | RSK is translation-invariant but not a quotient of the symmetries, it has scalar boundary features, its tail is grid-quantised, and the far ports are out of range in e₂ (Section 3). The test did not isolate "relational". |
| WL77: the remaining in-ring structure is higher-order azimuthal (order ≥ 3) | **Likely wrong** (non-evidence proxy) | The proxy reproduces 0.453 / 0.385 / 0.352 to ±0.004 from SS q = 1024 noise alone; noise-free M gives 0.22 / 0.07 / 0.02. Needs the cheap server confirmation (Section 5). |
| WL76: the native limit of FIA is the ring-lobe restriction | **Overstated** (proxy) | Native ≈ √(0.261² SS noise + 0.127² ring) = 0.29 against the observed 0.29–0.30. The ring share is real but secondary. |
| Increase q to remove the SS noise | Rejected | 1/√q scaling: 27× (native ≤ 0.05) to about 150× (contrast ≤ 0.1) the 75 s rebuild |
| Early stopping as the fix for drift | Rejected | Not authorised; the early regime still fails (WL71A: unseen contrast 0.32–0.35, deformation fails) |
| Sorted free-path invariants are good boundary features | Rejected | They discard which direction the boundary lies relative to the pair (FIA `free_paths` / `pair`) |
| Per-face or geometry-conditioned A/O as the main fix | Not proposed | Keeps the CP pairing; WL74A shows the floor is the shared pairing plus angle-free T, not the A/O values alone |
| The coarse metric artificially "amplifies" errors | Rejected (WL74A) | con/abs ≈ 2–3; the floors grow with |HZ|/|Z|. The sub-cell normalisation is real sensitivity, not amplification. |

---

## 9. Remaining uncertainty

1. **The proxy is non-evidence.** It covers one interior top-face configuration on a laterally infinite slab, in numpy, with pooled channels and point estimates. Untested:
   - boundary sources;
   - the 1 − β_B rescaling;
   - the W-term noise.

   The three-basis agreement with WL77 (to ±0.004) is strong but needs the server re-computation described in Section 5.
2. **The share of the WL76 contrast (0.69–0.75) due to SS noise versus W noise, anchor snapping and ring lobes is not attributed.** The proxy gives an SS-only range of 0.49–1.24, depending on sub-cell.
3. **Whether M1–M3 transfer to the BND_FAR context** depends on how strongly the planes interact at e₂ = 18.75 mm in red. That interaction is unmeasured.
4. **The deformation transfer mechanism is not addressed.** Every model fails DEF. Two training poses may be insufficient regardless of representation.
5. **Exact reciprocity of the eval68 d-normalised collided targets** with the per-port quadrature β is not checked: β is MC with q = 256, so the reciprocal pairs inherit β noise.
6. **Whether FIA's R_θ would transfer to unsupervised anchors** (feature-region coverage) is unmeasured, because WL76 had no Stage 4.
7. **The cost of deterministic SS/W cubature (M5) at native resolution is not estimated** beyond the requirement that it stay ≪ 75 s.

---

## 10. Status

**COMPLETED.** This report (`docs/plans/wl77a/specialist_neural.md`) is the only repository file written. No WL77 file, historical worklog or result was modified. No GPU, server or training was used. The CPU proxy scripts are in the session scratchpad (paths in Section 5) and are NON-EVIDENCE.
