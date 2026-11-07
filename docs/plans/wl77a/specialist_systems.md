# WL77A specialist report: graphics and GPU systems (Agent 3)

- **Status:** COMPLETED (2026-11-07)
- **Role:** Agent 3, graphics / GPU systems, in the WL77A multi-agent architecture-discovery workflow.
- **Scope:** read-only analysis. The only file written is this report.
- **Not done:** no GPU job, no training, no server access, no change to WL77 (`experiments/fnto_fia77`) or to any historical worklog or result.
- **New numbers:** three families of new numbers in this report come from small CPU-only numpy/torch computations on local archived data (section 3). They are labelled **[new CPU measurement]**. They are descriptive. They do not select a candidate and they read no held-out scores.

## 0. Research question

What must be represented explicitly, and what can be learned as a shared operator, so that a learned intrinsic operator K stays accurate when the source position changes? The hard case is independent, previously unseen source ports near boundaries far from supervised sources.

This report covers the computational side of that question:
- what the historical operators cost and why;
- where the transport energy and the metric-relevant structure actually sit;
- what a candidate may afford on the 12 GB RTX 3080 Ti that ran the recent batches;
- which classes of idea are ruled out on cost alone, independent of fidelity.

It makes no fidelity claim about any candidate and recommends no direction.

## 1. Input sources read

**Code (read in full or in the cited parts)**
- `experiments/fnto_rule57/r57_model.py` (W+UV algebra: `coefficients` and `apply`, lines 47–84).
- `experiments/fnto_rule57/r57_geometry.py`:
  - `DIN=16, DOUT=64, NEAR=128, RANK=16, MIX=8` (line 17);
  - 128-NN graph in rest coordinates (line 64);
  - prior lengths 18.1 / 5.8 / 1.8 mm (lines 68–69);
  - ballistic B quadrature with q = 256 (lines 80–97).
- `experiments/fnto_persist59/runtime.py`: P arm, `full` and `selective`, a global softmax normalisation in `finish` (lines 54–67), and rigid reuse by signature (lines 92–98).
- `experiments/fnto_repr68/family68.py`: the response-family algebra and the `Target` layout (lines 108–117). It shows the native target plus a channel-major copy on the GPU.
- `experiments/fnto_cover71/family71.py` and `experiments/fnto_rank75/rank75.py` (rank-generic W+UV).
- `experiments/fnto_rank75/cost75.py` (cost harness).
- `experiments/fnto_cmp72/rsk72.py` (RSK header).
- `experiments/fnto_fia76/fia76.py` in full:
  - `quadrature`: two ray casts per path, a host-side `cKDTree` anchor query, a per-port fixed-seed loop in `quad_ports`;
  - `Runtime`: CSR assembly and byte accounting.
- `experiments/fnto_k55/k55_ops.py` (`DomainX`) and `experiments/phys_ref53/p53_mi.py` (`Intersector`, Mitsuba `ray_intersect_preliminary`).
- `experiments/fnto_fia76/protocol/fia76_v1.json`: `quadrature_q` = 1024, the cost gate state ≤ 4 GiB and 64-light apply ≤ 10 × A75, and the lane rule.

**Measured results**
- `results/fnto_rank75/final/cost_cost.json`: AR and A75 state, rebuild, apply and training. RTX 5080.
- `results/fnto_persist59/final/out/cost.json` and Worklog 59 §"actual update and query costs": F/S update, geometry+B share, persistent bytes.
- `results/fnto_fia77/dev_stage2_cost_probe_nonevidence.json`: FIA-AZ build breakdown, state bytes, apply, peak GPU 7.39 GB. RTX 3080 Ti.
- Worklog 76, cost table (lines 248–256) and amendments 2–5 (lines 142–160).
- Worklog 77 (lines 87–94, 126–132).
- Worklog 72 (cost table, lines 170–179).
- Worklog 74 (line 148: FP64 is 1/64 rate on the 3080 Ti).
- Worklog 74A (lines 59–80 and 130–156: CP cone, footprint shift, per-source incoming rank 3–5, edge influence fades within one port).
- Worklog 75 (lines 133–266: BND4_FAR, oracle gaps, drift).
- Worklog 70 (line 31: Cycles 16 spp 2.40 s, 1 spp 1.20 s).
- `docs/worklogs/77A_WORKLOG_MULTI_AGENT_ARCHITECTURE_DISCOVERY.md` (Part 0).

**Local archived data, read only**
- `~/rest_src73/fit_targets73.pt`: S16-R means of fitting replicas 0–3, shape (16 slots, 16 bins, 3840, 64, 3) per geometry, 1.51 GB.
- `~/rest_repr68/inputs/wl59/geometry_{rest,pt0}.pt`: port xyz, normals, `parent`, B, β and the build meta.
- `~/rest_repr68/inputs/wl59/lights.pt`: the 64 test incident fields.
- Scripts and outputs: `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\sparsity.py`, `...\contrast.py`, `...\rank.py`, `...\sparsity.json`.

**Hardware facts used without measurement:** RTX 3080 Ti has 12 GB, about 34 TFLOP/s FP32 and about 912 GB/s. These are vendor specifications, not measured here. The FP64 1/64 rate is from Worklog 74.

## 2. Operator dimensions: the dense wall

| Quantity | Formula | Value (P = 3840) |
|---|---|---|
| Input cells | P·16 | 61,440 |
| Output cells | P·64 | 245,760 |
| Dense K (RGB, fp32) | P²·16·64·3·4 B | **4.53e10 floats = 181 GB** |
| One source port's full column (16 bins) | 16·P·64·3 | 11.8 M floats (47 MB) |
| S16-R native target per geometry | 16·16·P·64·3 | 755 MB (file: 1.51 GB for rest + pt0) |
| One light (input / output field) | P·16·3 / P·64·3 | 0.74 MB / 2.9 MB |
| 64-light output | 64·P·64·3 | 189 MB |

**Inputs are dense.** The test lights (`lights.pt`, `test`, 64 × 3840 × 16 × 3) light 37 % of all cells and about 1,530 ports per light **[new CPU measurement]**. Input sparsity cannot be exploited, so every apply touches the whole operator state.

## 3. Transport sparsity of the reference K (new CPU measurement)

**Data.** S16-R fitting-replica means (`fit_targets73.pt`), collided part only. For each source and channel, statistics are pooled over the 16 incoming bins. Six representative sources are shown:
- interior top 2085 (2085 is inside the pt0 dent);
- interior 1123;
- top edge row 3439;
- bottom corner 3823;
- side face 408;
- side face 527.

The values are means of fitting replicas, which is training supervision, not held-out data.

### 3.1 Radial energy and L2 tails

**Interior top source 2085, rest; R / G / B.** The other interior sources (1123, 1699, 2400, …) agree to ±0.005.

| Radius R | collided energy within R | L2 norm beyond R (√ of squared fraction), native (P,64) |
|---|---|---|
| 2.5 mm (5 ports) | 0.47 / 0.52 / 0.68 | 0.34 / 0.32 / 0.27 |
| 5 mm (≈ 14 ports) | 0.71 / 0.78 / 0.94 | 0.09 / 0.07 / 0.03 |
| 7.5 mm (≈ 31 ports) | 0.81 / 0.87 / 0.98 | 0.045 / 0.031 / 0.005 |
| 10 mm (≈ 54 ports) | 0.85 / 0.90 / 0.99 | 0.037 / 0.025 / 0.003 |
| 15 mm (≈ 172 ports) | 0.97 / 0.98 / 1.00 | 0.006 / 0.003 / 0.000 |
| 20 mm (≈ 310 ports) | 0.99 / 1.00 / 1.00 | 0.001 / 0.001 / 0.000 |

**Boundary sources are more compact than interior ones.** Some energy escapes, and their responses are split across 2–3 faces:

| Source | Energy within 5 mm (R / B) | L2 beyond 10 mm (R) | Face split of the collided energy |
|---|---|---|---|
| 3439 (top, edge row) | 0.86 / 0.96 | 0.014 | top 0.64, side 0.33 |
| 3823 (bottom corner) | 0.93 / 0.98 | 0.006 | bottom 0.52, two sides 0.24 each |
| 408 (side) | 0.75 / 0.92 | 0.019 | own side 0.76, top 0.16, bottom 0.08 |

**Ports needed for 99 % of the energy:**
- interior: 317 / 228 / 45 (R / G / B);
- edge: 136–172 / 105–133 / 30–50;
- corner: 61 / 49 / 20.

**W+UV 128-NN support** (mean radius 13.5 mm):
- it captures 94.5 % / 96.8 % / 99.9 % of the interior collided energy, and 98–99.9 % at boundary sources;
- the L2 left outside it is **0.012 / 0.007 / 0.0003** (interior) and **≤ 0.005** (boundary).

**Physical cross-check.** The prior lengths in `r57_geometry.py:68` (18.1 / 5.8 / 1.8 mm) are the infinite-medium diffusion lengths 1/√(3σ_aσ_t′) for albedo 0.999 / 0.99 / 0.9. With a 10 mm leaky slab, the lowest lateral slab mode decays over about 3.6 / 3.1 / 1.6 mm (R / G / B). This is an analytic estimate, consistent with the table above. Thickness, not the red absorption length, is what truncates the red tail in this fixture.

### 3.2 Locality of the incidence dependence (contrast)

**Method.** The incoming-bin-centred, budget-normalised fine16 response HZ, computed at port level.

| Source | Fraction of the √ HZ-norm beyond 2.5 / 5 / 7.5 / 10 mm (R) | coarse \|HZ\|/\|Z\| |
|---|---|---|
| 2085 interior | 0.68 / 0.14 / 0.05 / 0.04 | 0.054 |
| 3439 edge | 0.54 / 0.09 / 0.03 / 0.02 | 0.166 |
| 3823 corner | 0.35 / 0.07 / 0.02 / 0.01 | 0.150 |
| 408 side | 0.57 / 0.15 / 0.06 / 0.02 | 0.136 |

**Reading.**
- About 98 % of the squared incidence-dependent signal lies within 5 mm, which is about 14 receiver ports.
- At the 1 cm coarse metric, the relative contrast signal is about 3× larger at boundary sources (0.14–0.17) than at the interior source (0.05).
- This agrees with the WL74A observation that sub-cell position turns footprint shift into contrast (`74A`, lines 144–146).

### 3.3 Angular rank

**Per (source, receiver) block.** Each 16 × 64 block is evaluated for the 32 nearest receivers. The table gives the L2-weighted residual after rank k:

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| R, interior 2085 | 0.100 | 0.077 | 0.046 | 0.017 | 0.012 | 0.008 |
| R, corner 3823 | 0.088 | 0.066 | 0.045 | 0.011 | 0.008 | 0.006 |

- **Per source:** incoming-mode residual of the 16 × (P·64) unfolding is 0.19 / 0.15 / 0.07 / 0.05 / 0.03 / 0.018 for k = 1…6. That is rank 6–8 at port resolution, which agrees with WL74A item 4.
- **Per-receiver outgoing shape:** the deviation from a per-receiver bin-uniform (Lambertian) shape is 0.21 of the L2 within 5 mm, for every source class.

**Implication.** Each pair block has a pair-specific angular rank of about 4, enough for about 2 % residual. The shared CP form of W+UV ties every pair to the same M global atoms A_r ⊗ O_r (`74A` P1). The gap between "rank 4 per pair" and "rank 8–16 shared" is a structural fact about the representation, not a cost fact. Section 7 shows it is affordable to lift locally.

### 3.4 Boundary prevalence (geometry only)

Ports within a given distance of a non-parallel face:

| Distance | Ports | Share |
|---|---|---|
| 2.5 mm | 648 | 16.9 % |
| 5 mm | 1,248 | 32.5 % |
| 10 mm | 1,792 | 46.7 % |
| 20 mm | 2,688 | 70.0 % |

With the 5 mm contrast radius, about one third of all sources have a contrast-relevant neighbourhood that crosses a face edge. **Supervision coverage:** S16-R supervises 16 of 3,840 source ports (0.42 %), or 512 of 122,880 input cells over the two geometries.

## 4. Historical operators: complexity and measured cost

Notation:
- E = 128 neighbours;
- M = mixture rank (8 for AR, 16 for A75);
- TR = 16 node-tail rank;
- C = 3 channels;
- b = lights per batch.

### 4.1 W+UV (WL57/59 P arm; AR, A75)

**Algebra** (`r57_model.py:47–84`, `rank75.py:31–60`):

K_scat = d·Σ_r A[r,a]·T[i,j,r]·O[r,o], where T = W (128-NN edges) + U·V (rank-16 tail), mass-normalised, and A and O are global softmaxes.

**State** = W (P·E·M·C) + U (P·TR·M·C) + V (P·TR·M·C) = **1.84 M·M floats**:
- 14.7 M floats (59 MB) at M = 8;
- 29.5 M floats (118 MB) at M = 16.

This is verified in `cost_cost.json` (W 11.80 M, U and V 1.47 M each at M = 8).

**Rebuild from geometry (learned coefficients only):**
- node MLPs over P;
- the edge MLP over 491,520 edges;
- a **global** normalisation: U is softmaxed over all P receivers (`r57_model.py:59`), and mass/norm is reduced over all edges.

| Measurement | AR (M = 8) | A75 (M = 16) |
|---|---|---|
| Rebuild, RTX 5080 | 7.0 ms | 8.2 ms |
| Rebuild, 3080 Ti (WL76 table) | 14 ms | 15 ms |
| 64-light apply, RTX 5080 | 34 ms | 63 ms |
| 64-light apply, 3080 Ti | 54 ms | 86 ms |
| 1-light apply, RTX 5080 | 1.07 ms | 1.37 ms |
| Peak GPU above geometry, apply | 0.29 GB | 0.45 GB |

**Apply arithmetic per light:**
- A projection: P·16·M·C;
- W scatter: P·E·M·C;
- UV: 2·P·TR·M·C;
- O expansion: P·M·64·C;
- ballistic B: nnz 3.93 M × 3.

That is ≈ 34 M MAC at M = 8, or ≈ 4.3 GFLOP for 64 lights. Data is about 0.4 GB per 64-light batch: state 59 MB, B 79 MB, in 47 MB, out 189 MB.

**Roofline gap.** On a 3080 Ti the roofline time is about 0.4–0.5 ms. The measured 54 ms is about **100 × above the roofline**. The cause is the `index_add` of materialised gathers `q[:, src]` in 65,536-edge chunks, plus Python and launch overhead. Apply timings are therefore implementation-bound, not representation-bound.

**Geometry preparation shared by every K** (rest, `geometry_rest.pt` meta):
- CPU features and kNN: 0.39 s;
- ballistic B: 15.7 M rays at q = 256 in **0.18 s** (≈ 88 M rays/s via Mitsuba/OptiX), nnz 3.93 M, 79 MB;
- total 0.80 s.

WL75 measures 0.65 s for the cf1 build. In WL59's actual F/S, geometry+B was 0.32 s, about **97.5 % of a nonrigid update** (Worklog 59, line 182).

**Selective update (WL59).**
- A local dent re-evaluates 568 / 828 / 1,131 of 3,840 nodes over the GNN layers and 166,561 of 491,520 edges (33.9 %).
- The global normalisation and B are rebuilt in full.
- The dent saves S/F = 0.98–0.99.
- Rigid S saves about 28 %, but still costs 0.266 s at one query. Most of that is CPU geometry re-derivation done to verify the signature; this is inferred, not separately timed.

**Training** (20k steps, full batch of the 16 S16-R sources × 2 geometries, columns only):
- AR 4,156 s per run, A75 4,084 s, both two lanes: about **0.2 s/step**.
- Peak allocated 3.82 GB (AR) and 4.50 GB (A75).
- About 3.7 GB of that peak is the targets. `family68.Target` keeps on the GPU, per geometry: native S (755 MB), fine16 (189 MB), coarse (12 MB), and channel-major copies of native and fine16 (755 MB + 189 MB). That is 1.9 GB × 2 geometries. This is computed from the code and is consistent with the measured peak.
- The model's own working set is therefore well under 1 GB, because the objective never materialises a prediction (`family68.sq_error`, an algebraic identity).

### 4.2 RSK (WL72 B)

| Quantity | RSK |
|---|---|
| State | 33.9 M floats |
| Coefficient build | 36 ms |
| 64-light apply | 67 ms |
| Training | 1,091–1,140 s (contended) |
| Peak memory | 3.31 GB |

Same apply algebra as W+UV, with a coarse 240-cell far term. The cost is not limiting.

### 4.3 FIA (WL76) and FIA-AZ (WL77)

**Algebra** (`fia76.py`):

K x = B x + albedo·SS x + albedo²·Rᵀ(Wᵀ x)

- W (cells → 4P anchors) and SS (cells → exits) are fixed-seed quadratures of geometry and σ_t.
- R is learned from each anchor to its 512-NN receivers (ties included) and to the 240-cell tail, with 3 ring lobes.

**State, measured on the 3080 Ti:**

| Component | FIA (WL76) | FIA-AZ (WL77) |
|---|---|---|
| SS CSR | 448 MB (≈ 37 M nnz) | 448 MB |
| W | 37 MB (≈ 3.1 M nnz, ≈ 50 anchors per input cell) | 37 MB |
| Local R | 855 MB (4P·512·3 lobes·3 ch ≈ 71 M nnz × 12 B) | 1,013 MB |
| Tail | 133 MB | 133 MB |
| **Total** | **1.47 GB** | **1.63 GB** |

The protocol had expected about 280 MB of local R (`fia76_v1.json`, `expected_cost`). The 12-byte CSR entries (fp32 value + int64 index) roughly tripled it.

**Rebuild: 75.0 s** (WL76; quadrature 74.3 s, response and assembly 0.7 s) and **71–72 s** (WL77). The build peak is **7.39 GB**, which on its own rules out two-lane co-residence during builds.

**What the quadrature does.** It traces P·16·q = 62.9 M paths at q = 1024. Each path takes 2 OptiX casts (126 M casts) plus one **host** `scipy.cKDTree` anchor query, with a GPU→CPU→GPU round trip. It runs as **3,840 separate per-port calls** (`quad_ports`, which seeds per port for rebuild identity), each of only 16,384 paths.

**Cost inference** (not measured):
- At the B throughput (≈ 88 M rays/s at 2 M-ray chunks), the ray casting alone would take about 1.4 s.
- The 70 s is therefore dominated by per-call overhead (≈ 18 ms per port call) and the host kd-tree.
- A batched implementation with counter-based RNG keyed on (port, sample) would plausibly take a few seconds, not 70.
- Even so, about 126 M casts per nonrigid geometry are close to Cycles' whole-scene 16 spp time (2.40 s, Worklog 70 line 31).

**Apply:**

| Measurement | FIA (WL76) | FIA-AZ (WL77) |
|---|---|---|
| 64-light | 0.349 s (4.0 × A75) | 0.495 s (5.75 × A75) |
| 1-light | 3.9 ms | 4.6 ms |

- **Before CSR pre-assembly (amendment 5):** 3.07 s (35.7 ×).
- **Arithmetic** is about 240 M MAC per light, about 1.5 GB read per batch. The 64-light roofline is about 2 ms, so the 0.35–0.5 s measured is about 200 × above it. Ratios to A75 again reflect implementation.

**Training:** 19,834 s for 20k steps, about **1 s/step** on two lanes, 4.75 GB reserved per lane. Each step evaluates the 6-MLP pair response over every anchor reached by the 16 sources' W rows × 512 neighbours.

**Locality is physically available but not coded.**
- First collisions lie within about 1 mm.
- Second-collision anchors lie within a few mm.
- Exit weights exp(−σt₂) are negligible beyond about 10 mm (e^-10 ≈ 4.5e-5).
- So a dent invalidates only cells whose paths reach the changed triangles within about 10 mm.
- The code rebuilds everything and carries no per-path validity certificates. The same gap exists for B (Worklog 59 line 43: "B has no segment validity certificates").

### 4.4 Summary table (3080 Ti unless noted)

| | AR (rank 8) | A75 (rank 16) | RSK | FIA | FIA-AZ (dev) |
|---|---|---|---|---|---|
| Persistent learned-K state | 59 MB | 118 MB | 136 MB | 1.47 GB | 1.63 GB |
| Learned rebuild | 14 ms | 15 ms | 36 ms | 0.7 s | 0.9 s |
| Geometry-only rebuild beyond the shared B | 0 | 0 | ≈ 0 | 74 s | 70 s |
| 64-light apply | 54 ms | 86 ms | 67 ms | 0.349 s | 0.495 s |
| 1-light apply | ≈ 1–2 ms | ≈ 1.4–2 ms | — | 3.9 ms | 4.6 ms |
| Training per run (20k) | 4,156 s | 4,084 s (5080) | 1.1 ks | 19,834 s | not run |
| Training peak per lane | 3.8 GB | 4.5 GB | 3.3 GB | 4.75 GB reserved | — |

Notes:
- The shared geometry preparation with B (0.3–0.8 s) is excluded from the table and is common to all.
- AR and A75 1-light values on the 3080 Ti are scaled from the RTX 5080 (1.07 / 1.37 ms); the 3080 Ti value was not measured.
- The RSK apply was measured in WL72.

## 5. Construction vs application; rebuild vs reuse; selective recomputation

1. **Construction dominates nonrigid updates; application dominates per-frame cost.**
   - For W+UV, the learned coefficients take 15 ms against 0.3–0.8 s of shared geometry/B preparation. A candidate's learned rebuild only matters once it is ≳ 0.1 s.
   - FIA shows the failure mode: a geometry-only quadrature for the scattered part costs 100 × the shared B, because it uses q = 1024 instead of 256 and samples two vertices.
2. **Rigid edits must cost nothing in K.**
   - K is intrinsic and expressed in the material frame. WL53 found rigid K invariant within MC noise, and WL59 rigid S reused all state.
   - Every quantity of a candidate (anchors, frames, quadratures, neighbour lists) must be computed in material coordinates, so that a rigid move changes only C.
   - WL59's 0.266 s rigid S is CPU geometry re-derivation used to confirm the signature, not intrinsic work.
3. **Intrinsic (nonrigid) edits invalidate a dilated footprint, not the domain.**
   - Invalidation radius ≈ edit footprint + feature receptive field + response support. For W+UV that is two 16-NN hops (≈ 10–11 mm) plus the 128-NN support (13.5 mm).
   - For a 20 mm-radius dent on this 100 mm slab, that is roughly 1,000–2,000 of 3,840 source ports, matching the 1,131 nodes WL59 measured after two hops.
   - The selective saving on coefficient work is therefore at most about 2–3 × here.
   - Two things nullify it unless they are designed out:
     - **global normalisations** (U's softmax over all P; global mass/norm reductions);
     - **certificate-free geometry quadratures**, both the B rebuild and the FIA W/SS rebuild.
4. **Port correspondence.** The 128-NN graph is built in rest coordinates and `selective` asserts identical src/dst/graph/parent (`runtime.py:85–88`), so port topology is fixed across poses. A candidate that rebuilds neighbour lists from current coordinates, as FIA does with the 512-NN set on current xyz, changes sparsity patterns under a dent. CSR structures then cannot be patched in place.

## 6. Iterative, multigrid and hierarchical application

1. **Hierarchical far field (H-matrix / FMM / coarse tail) has nothing to gain in this fixture.**
   - L2 beyond 15 mm is ≤ 0.6 %, and beyond 20 mm ≤ 0.1 % (§3.1). The existing UV tail or a 240-cell coarse tail already covers it.
   - For thicker objects (red diffusion length 18 mm not truncated by a 10 mm slab), support would grow to roughly 5 × 18 mm, and hierarchical far fields would matter. That case is unmeasured in this fixture.
2. **Iterating a learned local operator** (a Neumann series of a short-range propagator, or a learned interior volume solver).
   - **Inference fits.** Example: a 1 mm voxel grid of 1e5 voxels × 4 angular moments × 3 channels × 64 lights is about 7.7e7 floats (0.3 GB). With about 80 multigrid-equivalent sweeps, that is about 4e10 FLOP, or ~10–100 ms.
   - **Accuracy needs sub-mfp resolution at the boundary.** 47–75 % of the energy and about 54–68 % of the contrast norm lie within 2.5 mm of the source (§3.1–3.2), in the sub-mfp boundary layer. Diffusion-like interiors are least accurate there, which needs 0.25 mm voxels near the boundary (≈ 6.4e6 voxels for the slab, uniform), or boundary-adapted grids.
   - **Outgoing readout needs high angular order.** The angular readout per port must carry azimuthal order > 2 (WL77: free-phase order ≤ 2 still leaves 0.352). That requires discrete-ordinates-like interior angular state, and 64 directions per voxel blows the 64-light batch beyond 12 GB.
   - **Training is the binding constraint, not inference.** A step has 512 one-hot right-hand sides (16 sources × 16 bins × 2 geometries).
     - Even with a local 40 × 40 × 10 mm window at 1 mm and 9 angular moments, the state is about 0.9 GB per iteration.
     - Back-propagating through 50 iterations would need about 45 GB.
     - So implicit or adjoint differentiation (DEQ-style) or truncated unrolling is required.
   - **Rebuild.** Voxelising a deformed geometry plus the port↔voxel coupling is a per-geometry rebuild with its own certificates.
   - **Verdict:** feasible on cost only with boundary-adapted resolution, implicit gradients and a high-angular-order boundary readout. It is not rejected, but it carries the highest systems risk of the classes considered.
3. **Repeated application in the coupled solve.** Coupled GI is not authorised here, but the eventual K∘C solve applies K about 6–20 times per frame (WL42: 6–7 iterations; WL55: K+C iteration 1–2 ms). The 1-light apply budget below is set with that multiplication in mind.

## 7. Worst-case scaling and what fits on a 12 GB 3080 Ti

**Per-geometry state for P = 3840.** N_s is the number of receivers per source. Rows are evaluated at N_s = 128 (13.5 mm) and at the R ≤ 5 / 7.5 / 10 mm neighbourhoods (13.7 / 30.6 / 54.1 ports mean) **[new CPU measurement]**.

| Representation of the source→receiver block | floats per pair | N_s = 14 (5 mm) | N_s = 31 (7.5 mm) | N_s = 54 (10 mm) | N_s = 128 |
|---|---|---|---|---|---|
| Dense 16 × 64 × RGB block | 3,072 | 0.64 GB | 1.45 GB | 2.55 GB | **6.0 GB** |
| Pair-specific rank-4 (16·4 + 64·4) × RGB | 960 | 0.20 GB | 0.45 GB | 0.80 GB | 1.9 GB |
| Tucker core on global bases (16 in × 8 out) × RGB | 384 | 0.08 GB | 0.18 GB | 0.32 GB | 0.74 GB |
| Incidence-dependent T (16 bins × M = 8) × RGB | 384 | 0.08 GB | 0.18 GB | 0.32 GB | 0.75 GB |
| W+UV CP (M = 8), for reference | 24 | — | — | — | 0.047 GB (+ UV) |
| FIA-style 4 anchors/port × 512 NN, full 64-bin outgoing | 4·512·64·3 per port | — | — | — | **15 GB** (fp32), more in CSR |

**Scaling with resolution.** For a fixed physical support radius, the number of pairs grows as h⁻⁴ in port pitch h: P ∝ h⁻², N_s ∝ h⁻². Halving the port pitch to 1.25 mm (as contact-adaptive cells would locally) multiplies pair-block state by 16. Any per-pair angular block must therefore be budgeted at the finest planned port resolution, or kept only on a short-radius core.

**GPU budget on the 3080 Ti:**
- 12 GB total; the WL76 lane rule allows two concurrent fits only if the sum of reserved peaks is ≤ 0.85 × 12 GB, which gives **≈ 5.1 GB per lane**.
- S16-R targets alone take about 3.7 GB per lane in the historical layout (§4.1). That leaves about 1.4 GB for the model at two lanes, or about 6.5 GB at one lane, which doubles wall time.
- Dropping the channel-major copies (a layout-only amendment in WL68) would free about 1.9 GB per lane; that is an implementation option, not a protocol change.
- FP64 is 1/64 rate (Worklog 74 line 148). Training and quadratures must be FP32. FP64 belongs only in oracle projections.
- The FIA build peaked at 7.39 GB. Build peaks count against concurrency just as training peaks do.

## 8. Budgets a candidate must meet

These are computational admissibility budgets on the LabServer63 RTX 3080 Ti. Measure them in the same torch-level implementation style as A75, uncontended. Report roofline FLOP/byte counts alongside, so that a representation is never rejected for an implementation artefact; WL76 amendment 5 changed apply cost 9× without changing the operator.

| Budget | Preferred | Hard cap | Basis |
|---|---|---|---|
| Persistent learned-K state per geometry | ≤ 0.5 GB | ≤ 2 GiB | A75 0.12 GB; FIA 1.5 GB; leaves room for C/T state (WL56 ~1.7 GB/domain) and two evaluation lanes |
| Build peak GPU memory | ≤ 3 GB | ≤ 5 GB | FIA 7.4 GB blocked co-residence |
| Learned-coefficient rebuild after a nonrigid edit | ≤ 0.1 s | ≤ 1 s | W+UV 15 ms; shared geometry+B 0.3–0.8 s; Cycles whole-scene 16 spp 2.40 s |
| Extra geometry-only quadrature (beyond the shared B) | ≤ 16 M ray casts, ≤ 0.5 s | ≤ 32 M casts, ≤ 1 s, with locality certificates | B: 15.7 M rays in 0.18 s; FIA 126 M casts / 74 s |
| Rigid-edit K rebuild | 0 (exact reuse in material frame) | 0 | WL53 / WL59 |
| Nonrigid selective invalidation | edit footprint + receptive field + support; no global normalisation over P | — | WL59 S/F 0.98 because of global terms |
| 1-light apply | ≤ 2 ms | ≤ 5 ms | AR ≈ 1 ms; FIA 3.9 ms; coupled solve × 6–20 iterations |
| 64-light apply | ≤ 0.26 s (3 × A75) | ≤ 0.86 s (10 × A75) | A75 0.086 s; WL76 gate 10 × |
| Training step, two lanes | ≤ 0.5 s | ≤ 1 s (20k steps ≈ 5.5 h) | AR 0.2 s; FIA 1 s |
| Training per-lane reserved, targets included | ≤ 5.1 GB | ≤ 10 GB on one lane | lane rule 0.85 × 12 GB |
| Numerical type | FP32, TF32 off | — | FP64 1/64 |
| Extra supervision (if any privileged label) | declared separately; priced at ≈ 2.1 s per 32M-photon (port, bin) packet on the RTX 5080 (WL75: 1,408 packets in 2,989 s) | — | — |

**Structural requirements that follow from cost**, independent of fidelity:
1. Prediction for a source must be computable from a bounded neighbourhood, so that training never needs the whole-domain operator.
   - W+UV evaluates columns only for the 16 supervised sources.
   - A design whose per-source prediction requires a global solve or a global softmax pays whole-domain cost every step and every local edit.
2. The objective must not materialise dense per-source predictions beyond the support. Use support-restricted dense blocks plus algebraic identities for the tail; `family68.sq_error` is the model.
3. Geometry-only precomputations must be keyed and seeded so that they are batchable (counter-based RNG instead of per-port loops) and locally certifiable (per-path triangle/segment certificates).

## 9. Rejected classes of idea (cost reasons only)

| Idea | Reason |
|---|---|
| Dense K or dense per-source columns over all receivers | 181 GB; even one source port's dense column is 47 MB, × 3840 = 181 GB |
| Full 16 × 64 × RGB blocks over the 128-NN support | 6.0 GB per geometry: over the 2 GiB cap, with no room for C, two lanes or finer ports (×16 at 1.25 mm) |
| FIA-type anchors × 512 NN with expressive per-pair outgoing (full 64-bin, or a ≥ 8-coefficient basis) | 15 GB fp32 for full 64-bin; about 2–3 GB in CSR at 8 coefficients. The 4-anchors × 512-NN multiplier (7.9 M pairs, 16 × W+UV's pair count) is the cost driver, and per-pair angular expressivity multiplies it. This closes, on cost, the "per-pair full 64-bin outgoing distribution" left open in WL77 (line 132) for the FIA anchor layout |
| Per-query Monte Carlo inside K apply (online photon tracing per light) | One (port, bin) packet at 32 M photons costs about 2.1 s. A light lights about 22 k cells (37 % of 61,440), so even 1/1000 of the photons per cell is minutes per light. It is also not a learned operator; it is re-rendering against the Cycles baseline (1.2–2.4 s per scene) |
| Per-geometry scattered-part quadrature at q ≳ 1024 with two-vertex paths and host-side neighbour search | 126 M casts plus a CPU kd-tree per nonrigid edit: 70–75 s as built, and plausibly seconds even when batched. Over the rebuild budget, and not locally updatable as coded |
| Global normalisation over all P (softmax over receivers, global mass reductions) in a design meant for selective updates | Every local edit touches all derived coefficients (WL59). Admissible only if charged and if selective savings are not claimed |
| Backprop through unrolled iterative interior solves (≥ 50 iterations, 512 RHS per step) | ≈ 45 GB activation memory. Needs implicit/adjoint gradients or truncation |
| Uniform sub-mfp volumetric grids (0.25 mm over 100 × 100 × 10 mm) with directional state | ≈ 6.4e6 voxels × ≥ 16 directions × 3 × 64 lights ≈ 2e10 floats |
| FP64 training or quadrature on the 3080 Ti | 1/64 throughput (Worklog 74) |
| Per-source tables, source-identity lookup, nearest-supervised-source copying | Forbidden by the brief. Systems note: the WL74/75 per-source oracles (G0/G2) are exactly such tables (free T per source) and are diagnostic only |

## 10. Computationally admissible templates (feasibility only, no recommendation)

These are listed only to show where cost does not bind. Their fidelity and transfer are the other agents' and the gates' business.

1. **Two-zone operator.**
   - A pair-expressive near core on N_s ≈ 14–31 receivers (R ≤ 5–7.5 mm), where §3 places about 98 % of the contrast and ≥ 89 % of the native L2.
   - A cheap passive far zone: CP / UV / coarse-cell tail out to 128-NN or global.
   - Near-core state: 0.08–0.45 GB depending on factorisation (§7). Far zone ≈ W+UV.
   - Apply cost: about N_s·(floats per pair) MAC per source per light. For example 31 × 384 ≈ 12 k MAC per source, or 46 M MAC per light at P = 3840, the same order as W+UV's 34 M.
2. **Incidence-dependent spatial coefficients.** T indexed by incoming bin or incoming mode.
   - Per-source incoming rank 3–5 at coarse resolution, 6–8 at port resolution (WL74A; §3.3).
   - State 3–16 × W's, 0.15–0.75 GB at 128-NN.
   - Apply 3–16 × W's scatter term. At 16 × it lands near the 0.86 s hard cap with today's `index_add` implementation, and well inside it with a fused kernel.
3. **First-order sub-port shift channels.** T, ∂_u T and ∂_v T with geometry-conditioned, incidence-dependent shift coefficients. This represents the ~1 mm footprint shift (WL74A) at ×3 state and apply.
4. **Explicit geometry descriptors** such as sorted free paths to the boundary (FIA's 10-ray features). They cost about 38 k rays per geometry, which is negligible, and are local and certifiable.

## 11. Remaining uncertainty and unverified items

1. **The 3080 Ti numbers for AR and A75 1-light apply** are not measured; values are scaled from the RTX 5080. The 3080 Ti rebuild and 64-light values are from the WL76 table.
2. **FIA rebuild decomposition** into ray casting, host kd-tree, per-call overhead and coalescing is **inferred** from the code structure and B's throughput. It is not timed, so "a few seconds when batched" is an estimate.
3. **Roofline estimates** (≈ 0.5 ms for W+UV, ≈ 2 ms for FIA, at 64 lights) use vendor peak FLOP/s and bandwidth. They ignore atomics contention and sparse irregularity. Real fused kernels would land somewhere between roofline and today's measurements.
4. **Sparsity statistics come from S16-R fitting-replica means**, which are supervision data. They cover 16 sources, 2 geometries, mean of 4 replicas at 32 M photons, collided part only.
   - MC noise inflates the native nonzero fraction (0.09–0.36) and the far tails slightly.
   - Tails beyond 20 mm are at the noise level and are reported as ≤ 0.001.
   - Held-out and te-pose sources were not read.
   - The dented pt0 source 2085 has a slightly less compact response than rest: 5 mm L2 tail 0.149 vs 0.089 (R).
5. **Pair-block and per-source ranks** are oracle (per-block or per-source SVD) statements. They say nothing about whether a shared, geometry-conditioned rule can produce those factors for unseen sources. That gap is the research question itself.
6. **Selective invalidation counts** for candidate templates are geometric estimates, not measurements.
7. **Training memory accounting** (targets ≈ 3.7 GB per lane) is computed from `family68.Target`. It is consistent with the measured 3.82 GB AR peak but was not profiled.
8. **Thick or other-geometry objects** (support not truncated by thickness, contact-refined ports) may change the far-field and resolution conclusions. Untested in this fixture.

## 12. Status

**COMPLETED.** Read-only. This report is the only file written in the repository; scratch scripts live in the session scratchpad. No GPU, training or server activity.
