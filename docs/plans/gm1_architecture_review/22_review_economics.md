# GM1 architecture review — 22 — Runtime, memory and update economics

**Status:** design-only adversarial review (2026-11-10), `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. This file is the only file written.
**Reviewed:** `10_design_trackA_directional_delta_transport.md` (DDT / DDT-D), `11_design_trackB_path_space_reuse.md` (ERPS: PR1 / PR2 / PR3), `12_design_trackC_alternative_operator.md` (LPTG). No other review file in this folder was read.
**Sources read in the original:** `00_PROBLEM_BRIEF.md`; `AGENTS.md`; `docs/RESEARCH_ROADMAP.md` (GM1 row and decision frontier); worklogs 84, 67, 69, 70 (result and cost sections), 59, 58, 52 (cost and verdict sections). WL53–56 are used through the `AGENTS.md` summaries and cited as such.
**Labels:** FACT = ESTABLISHED FACT; HIST = HISTORICAL EVIDENCE (original scope only); DERIV = DERIVATION; EST = order-of-magnitude estimate, not a measurement; INF = INFERENCE; HYP = HYPOTHESIS.
**Axis:** economics only. Fidelity is discussed only where a claimed saving is bought with it.

## 0. Findings in brief

1. **The comparator is extremely cheap, and that sets every break-even.** WL84 measured about 10 ms of GPU kernel time per 64-spp frozen-ER image (FACT). That is about 2.4 ns per pixel-sample for a complete depth-4 path. Any pass that merely *touches* a persistent state larger than about 10⁸ entries already costs as much as a 1024-spp re-render at that image size (DERIV, §1).
2. **Track A (DDT-D): UNSOUND on this axis inside GM1.** The tabulated directional state is worth several thousand image renders to build. A single-view update beats a cold re-render only if the dirty fraction is below about 5·10⁻⁴ (256²) to 8·10⁻³ (1024²). The mover's own cells plus the unavoidable sharp first-order cells already exceed that (DERIV, §2).
3. **Track B (ERPS): CONDITIONALLY SOUND, with a bounded constant.** It is an honest retracer. Its saving is capped at roughly 3–5× in GPU work for PR1 and up to about 10× for PR2 with a small mover, and falls below 1× for a large mover or an enclosure. At WL84 scale the absolute saving is below one harness launch overhead (DERIV + EST, §3).
4. **Track C (LPTG): CONDITIONALLY SOUND as a cache, UNSOUND as a selective-update claim as written.** Link reuse is cheap and sparse, but the field re-solve is not output-sensitive and is the dominant edit cost. One solve iteration costs between a fraction of and several times a complete 1024-spp REF-ER frame at 256². `S ≈ F` in total cost is the expected result, which is the WL59 pattern (DERIV + EST, §4).
5. **No design has an economic window inside a GM1 fixture of WL84 scale** (256², few objects, one or two cameras, one edit). A window can exist only for megapixel output, many frames or views per edit, and a quality target where the path tracer needs several hundred spp. Whether that workload is "inside GM1" is a human decision; the roadmap asks for both edit-to-valid-output and per-frame accounting (FACT, `docs/RESEARCH_ROADMAP.md` GM1 row).

## 1. The comparator: cold frozen-ER path tracing

### 1.1 Measured anchors

| Quantity | Value | Label and source |
|---|---|---|
| GPU kernel time, ER_L, 64 spp | 9.9–10.1 ms (RTX 3080 Ti, uncontended) | FACT, `docs/worklogs/84_WORKLOG_RENDERING_QUALIFICATION_UTILITY_CLOSURE.md` |
| Integrator | Mitsuba `path`, max depth 4 | FACT, WL84 |
| Wall time per launch | 0.40 / 0.81 / 1.22 s of Python tracing, independent of spp and arm; 85–95 % of wall | FACT, WL84 |
| ER error vs layered reference | 0.086–0.099 at 64 spp; floor 0.035–0.043 at 1024 spp | FACT, WL84 |
| Explicit GPU relation build | 109 M rays in 0.105 s, i.e. 1.04 G rays/s (RTX 3090, other tracer and scene) | HIST, `docs/worklogs/52_WORKLOG_LEARNED_INTERDOMAIN_TRANSPORT.md` |
| Replay validation of cached shadow samples | 16 M samples in 0.090 s, i.e. 5.6 ns per sample (RTX 3090, full-scene shadow test) | HIST, `docs/worklogs/70_WORKLOG_SUN_STATE_REUSE_C1.md` |
| Path tracing vs oracle selective | Cycles 4 spp 4.67 s met the contract; oracle S 189.8 s did not (9/12) | HIST, `docs/worklogs/67_WORKLOG_BASELINE_SUPERIORITY_EVIDENCE.md` |
| Path tracing vs CAK | Cycles 16 spp 2.3–2.4 s vs CAK 24 s; contract already met at 1 spp, 1.2 s | HIST, `docs/worklogs/69_WORKLOG_COMMON_WORKLOAD_EXTERNAL_BASELINE.md` |
| Best historical win | 2.09 s vs 2.40 s (13 %), gone under Blender-internal timing | HIST, WL70 |
| Selective saving swallowed by common work | geometry + B = 97.5 % of S; saving 1.4–1.7 % | HIST, `docs/worklogs/59_WORKLOG_PERSISTENT_INTRINSIC_FS.md` |

The 256 × 256 image size is taken from this review's task statement. I did not find the resolution stated in the WL84 text. Every number below that depends on it is marked.

### 1.2 Derived baseline model

- **Cost per pixel-sample (DERIV, resolution-dependent).** 10 ms / (65,536 px × 64 spp) ≈ **2.4 ns** per complete path, including ER `sample` / `eval` and next-event estimation.
- **Cold re-render cost.** `T_PT(P, n) ≈ 2.4 ns · P · n`.

| Image | 64 spp | 1024 spp | 4096 spp |
|---|---|---|---|
| 256² | 10 ms (FACT) | 0.16 s | 0.64 s |
| 1024² | 0.16 s | 2.6 s | 10 s |

  The 1024-spp value at 256² agrees with WL84's "about 150–250 ms" for the layered reference at 1.5× the ER cost (FACT). The 1024² row is a linear extrapolation (EST).
- **Noise against REF-ER (EST).** Against REF-ER the ER bias cancels and only Monte Carlo noise remains. Treating the 64-spp error as mostly noise gives relative L1 `ε(n) ≈ 0.08 · sqrt(64 / n)`: about 0.04 at 256 spp, 0.02 at 1024 spp, 0.01 at 4096 spp. This is crude; L1 bias and noise do not add in quadrature.
- **Consequence 1 (DERIV).** At 256², a REF-ER image with about 1 % noise costs under 1 s of GPU time. A mechanism whose edit cost exceeds about 1 s cannot win a single-view comparison at that size at any quality level above 1 %.
- **Consequence 2 (DERIV).** The path tracer needs no state, no discovery and no build. Its edit-to-valid-output cost is one render. Its only lifecycle cost is the acceleration-structure refit for the mover.
- **Consequence 3 (INF).** In this harness, wall time is launch count times 0.4–1.2 s. A multi-stage update (discover, compact, retrace, solve, read out) pays that several times, while the comparator pays it once. GPU-kernel savings below about 1 s are invisible in wall time. WL70's 13 % margin flipped on the timing definition (HIST); the same fragility applies here.

### 1.3 Unit costs used below

| Symbol | Meaning | Value | Label |
|---|---|---|---|
| `c_path` | one complete depth-4 ER path | 2.4 ns | DERIV from WL84 |
| `c_ray` | one traced ray, small scene | about 1 ns | HIST WL52, other tracer |
| `c_touch` | read and test one stored record of 32–100 bytes | 0.15–0.5 ns | EST, assuming 200 GB/s effective memory throughput |
| `c_gather` | one random RGB gather from a large table | 0.2–1 ns | EST, assuming 1–5 G random gathers/s |

`c_touch` and `c_gather` are my assumptions and were not measured. WL70's 5.6 ns per replayed shadow sample (HIST) is an order of magnitude above `c_touch`, so the estimates here favour the designs.

**Rule of thumb that follows (DERIV from the table):** on this GPU class a ray costs about as much as one to five random memory accesses. "Arithmetic on stored data instead of tracing" is therefore not a cost advantage by itself. It becomes one only when the stored data touched is much smaller than the rays avoided.

## 2. Track A — Directional Delta Transport (DDT-D)

### 2.1 Cost model

Notation follows Track A §7: `N_x` spatial samples, `N_ω` directions, `g` gather samples per cell, μ first-order dirty fraction, φ_τ dirty fraction after propagation at tolerance τ, `N_m` cells on the mover, `P` pixels, `n` spp, `V` views or frames rendered per edit.

| Stage | Cost | Label |
|---|---|---|
| Memory | `3 · N_x · N_ω` floats; 6 GB at 10⁵ × 5·10³ in float32 | DERIV; Track A's own figures |
| T0 state build | `N_x · N_ω · g · c_path` per sweep | DERIV |
| First-order discovery | `2 · N_m · N_ω · c_ray` | DERIV; Track A §3 |
| Higher-order discovery | propagation through `K C1`: at least one ray per dirty sender cell, times a fan-out into up to `N_ω` outgoing cells per receiver hit | DERIV |
| Update (overwrite) | `φ_τ · N_x · N_ω · g · c_path` | DERIV |
| Solve | Gauss–Seidel on the dirty set; repeats the update term per sweep | DERIV |
| Readout, static scene | `P` primary rays plus `P` state lookups, about 1 ms at 256² | EST |
| Sustained per frame, moving object | the full update, every frame | DERIV; Track A §3 "Sequences" |

**Gather count (DERIV).** A camera reads about one cell per pixel through a sharp lobe. Cell noise must then equal the pixel noise target, so `g ≈ n`.

**State build in image units (DERIV).** Build ÷ one image = `N_x · N_ω / P`. With Track A's figures that is 5·10⁸ / 65,536 ≈ **7,600 images** at 256², or about 480 at 1024². At 1024 spp the build is about 20 minutes of GPU time, against 0.16 s for one REF-ER image at 256².

**Spatial resolution is understated near contact (DERIV).** On a near-specular surface a reflected feature at distance `d` moves across directions at rate `1/d`. Resolving a lobe of width α needs spatial spacing of about `d · α`: 4 mm at `d` = 10 cm and α = 0.04. Six square metres of surface at 4 mm is 3.8·10⁵ points, about four times Track A's figure, before any oversampling.

### 2.2 Dominant term and break-even

**Single view (DERIV).** DDT-D beats a cold re-render only if

`φ_τ · N_x · N_ω · g < P · n`, i.e. `φ_τ < P / (N_x · N_ω)` with `g ≈ n`.

That is φ_τ < 1.3·10⁻⁴ at 256² and φ_τ < 2·10⁻³ at 1024². Crediting the state with terminating paths at the first vertex (a factor of the mean depth, at most 4) gives **5·10⁻⁴ and 8·10⁻³**.

**Lower bound on φ_τ (DERIV, geometric).** Two contributions cannot be thresholded away:

- The mover's own cells are all dirty (Track A §3). Their share is the mover's surface share: 1 % for a 0.2 m cube in a 2 m cubic room.
- Sharp cells on stationary receivers whose reflected lobe crosses the old or new pose carry an order-one change. By the Cauchy–Crofton relation their share is about `(S(O0) + S(O1)) / S(room)`: 2 % for the same box with disjoint poses.

So φ_τ ≥ about 3 % for a *small* mover, 4 to 60 times above the single-view break-even. The Crofton relation (the probability that a uniformly random chord of a convex body hits a convex body inside it equals the ratio of surface areas; Santaló 1976, *Integral Geometry and Geometric Probability*) is cited from memory and assumes an isotropic chord distribution.

**Rough lobes make it worse (DERIV).** The ER closure contains a Lambert lobe (FACT, WL82 per the problem brief). With diffuse share `a_d`, every outgoing cell at a point that sees the changed set receives a relative source of about `a_d · Ω_D / π`. At τ = 1 % and `a_d` = 0.3, all cells within about 0.6 m of a 0.2 m box face are dirty. The diffuse weight of the frozen closures was not checked by Track A or by me.

**Many views per edit (DERIV).** DDT-D wins when `V > φ_τ · N_x · N_ω / (P · d̄)`. At φ_τ = 3 % that is about 60 views at 256² and about 4 at 1024², ignoring the build. Including the build, the first edit needs thousands of views at 256².

**Dominant term:** the T0 build over the lifecycle; the dense regather of the mover's field per edit.

**Worst case:** continuous motion. Every frame is an edit and `V` = 1. The mover regather alone is `N_m · N_ω · g`: 10³ points × 5·10³ directions × 256 samples ≈ 1.3·10⁹ paths ≈ 3 s per frame, against 40 ms for a 256-spp re-render at 256² (EST).

### 2.3 Counterexamples and threat patterns

| Threat | Finding | Label |
|---|---|---|
| Hidden full path tracing | Arm A3 (oracle `b0`) traces two T0 paths per changed ray plus the T1 continuation. For a fixed view every camera path must still be traced to learn whether it reaches the changed set. Its cost is at least a path tracer's. Track A states this itself. | DERIV |
| Hidden path tracing in the "band-limited store" | With a store coarser than the lobes, every lookup through a sharp lobe continues a T0 path. For sharp materials that is every lookup. | DERIV |
| Saving bought with fidelity | φ_τ is small only through τ. τ is a bias budget on exactly the stationary-receiver change GM1 gates. | DERIV |
| Discovery cost | First order is cheap and scene-independent. Higher order is transport work with a fan-out of up to `N_ω` per hit; it is not charged in Track A's update term. | DERIV |
| Any full pass over the state | 5·10⁸ cells at `c_touch` is 0.1–0.25 s, about one 1024-spp re-render at 256². The update must never scan the state. | DERIV + EST |
| Non-sparse delta | Uncovered emitter; high-albedo enclosure; large mover. Track A lists all three. No a-priori sparsity certificate exists, so the dense case is paid for before it is detected. | INF |
| Estimator noise mistaken for change | The source on changed rays is `b0(y1) − b0(y0)` from two lookups with error δ each. A tolerance below 2δ marks every changed ray. | DERIV |
| Learned compression of `b0` | A learned field replaces a 1 ns lookup by a network query. WL52's learned operator was 8–19× slower per update than explicit GPU work. | HIST, other operator; possible mechanism only |

### 2.4 Verified-sound points

- First-order discovery cost is independent of scene size (DERIV). This is correct.
- Unchanged rays contribute an analytically zero delta, so no difference-of-estimators noise arises there (DERIV). Correct, and a real advantage over naive correlated re-rendering.
- Static-scene readout is nearly free once the state exists (EST).
- Track A's own verdict ("reject as stated") and its worst-case list are accurate. Its falsification F1 uses rays and GPU time, which is the right metric.

### 2.5 Defects

| # | Severity | Defect | Basis |
|---|---|---|---|
| A1 | **FATAL** (single or few views) | The dirty fraction required to beat a cold re-render is below the geometric lower bound for any mover. | §2.2 |
| A2 | **FATAL** (continuous motion) | Per-frame cost is the full update; no amortisation. | §2.2 worst case |
| A3 | MAJOR | T0 build of about 10³–10⁴ image-equivalents is charged by name only. No lifecycle break-even is stated. | §2.1 |
| A4 | MAJOR | The claim "cheaper than a path-traced re-render whenever the output state must be view-independent" compares against a requirement GM1 does not have. GM1's valid output is images at given cameras. | FACT, roadmap GM1 row; INF |
| A5 | MAJOR | Higher-order discovery and marking cost is absent from the update term. | §2.3 |
| A6 | MINOR | Spatial sample count near contact is understated by about 4×. | §2.1 |
| A7 | MINOR | F1 must count the on-demand T0 rays of A3 on the A3 side. The text says "fewer rays" without saying whose. | Track A §12 |

### 2.6 What would have to change

- State the lifecycle break-even `V*` and the single-view φ bound as explicit admission numbers, and compute φ_τ on the fixture from geometry before building anything.
- Restrict the claim to the regime that can win: static scene after one edit, megapixel output, many views. Or drop the persistent table and accept that the remainder is residual path tracing, i.e. a Track B arm.
- Charge propagation marking and the T0 build in every reported total.

### 2.7 Verdict on this axis

**UNSOUND inside GM1** for a tabulated directional state. **Conditionally sound only** for a many-view, static-after-edit, megapixel workload with a small mover, where it is a precomputed directional radiance field with incremental repair. That regime is not GM1's headline and has unbounded memory growth as lobes sharpen.

## 3. Track B — Edit-revalidated path store (ERPS)

### 3.1 Cost model

Notation: `N` stored spp, `S` stored segments (about 8 per path at depth 4 with next-event estimation), ρ retraced fraction of segment work, `q` per-pixel probability that a path is dirty, `f` per-segment dirty probability.

| Stage | PR1 (exact store) | PR2 (difference re-render) | PR3 (reservoirs) | Label |
|---|---|---|---|---|
| T0 preprocessing | one render at `N` spp plus store write | converged T0 image at `N0 ≫ n`, plus `n` stored paths | reservoir warm-up | DERIV |
| Memory | `P · N · ~100 B`: 0.4 GB at 256² × 64; 6.7 GB at 256² × 1024; 107 GB at 1024² × 1024 | image plus `P · n · ~100 B` | about 128 B per pixel (Track B, unverified) | DERIV on Track B's 100 B EST |
| Discovery | `2 · S · c_obj` plus reading the store, `P · N · c_touch` | same on `n` paths | per-reservoir validation ray | DERIV |
| Update | `ρ · S · (c_scene + c_sh)` plus compaction and merge | `q · n` suffix retraces | fresh candidates plus resampling | DERIV |
| Solve | none (sampled Neumann series) | none | none | DERIV |
| Readout | accumulate stored contributions, `P · N · c_touch` | add image and difference | shade reservoirs | DERIV |
| Sustained per frame | the same update; store stays exact | variance accumulates across edits; re-anchoring needed | history cap | DERIV |
| View change | store void | store void | reprojection | DERIV |

### 3.2 Ceiling of the saving

**PR1 (DERIV + EST).** Speed-up = `c_path / (c_touch + c_test + ρ · c_path)`. Reading 100 B per stored path costs about 0.5 ns, 20 % of `c_path`. Segment tests add to that. Even at ρ = 0 the ceiling is about **3–5×** in GPU work. Track B's bound `1/ρ` is correct but loose: the binding bound is `1 / (ρ + c_touch / c_path)`. Frozen ER makes this worse, not better, because it makes `c_path` small (Track B §7 says this; agreed).

**Dirty fraction from geometry (DERIV, isotropic-chord assumption).** Per-segment `f ≈ (S(O0) + S(O1)) / S(room)`.

| Mover in a 2 m cubic room | `f` (both poses) | Path dirty, 8 segments | ρ (about half the dirty path) | PR1 ceiling |
|---|---|---|---|---|
| 0.2 m cube | 2 % | 15 % | about 0.08 | about 3.5× |
| 0.5 m cube | 12.5 % | 66 % | about 0.4 | about 1.6× |

This replaces Track B's illustrative `f` = 0.1 with a geometric estimate. The "half the dirty path" factor is my assumption.

**PR2 (DERIV under decorrelation).** A clean path contributes exactly zero to the difference. With replay decorrelated on dirty paths, `Var(Δ) ≈ 2q · σ²` per stored path. Matching the noise of a cold `N`-spp render needs `n = 2qN` stored and tested paths, of which `q · n` are retraced:

speed-up `≈ 1 / (2q · (c_touch / c_path + q))`.

At `q` = 0.15 this is about **9×**; at `q` = 0.5 it is about 1.4×; at `q` = 0.66 it is **below 1**. The estimate ignores the per-pixel control-variate fallback, which caps the loss at about 1×.

**Absolute size at WL84 scale (DERIV).** A 3–9× saving on a 0.16 s render saves at most 0.14 s of GPU time. One extra harness launch costs 0.4–1.2 s (FACT, WL84). In wall time ERPS loses in the current harness at 256².

### 3.3 Counterexamples and threat patterns

| Threat | Finding | Label |
|---|---|---|
| Hidden path tracing | None hidden: it is declared a retracer. | FACT (Track B §0) |
| Saving bought with fidelity | PR1 and PR2: no. PR3: yes in common practice (1-spp budgets, skipped T0-target evaluation, correlated error that pixel metrics under-report). Track B flags this. | INF |
| Discovery cost | Accounted in symbols. Understated in one respect: the store read is not in the formula and sets the ceiling. | DERIV |
| Non-sparse delta | Enclosure, large mover, mover near the light (shadow segments then have a much larger `f` than the Crofton value, since they all converge on the emitter). | DERIV |
| New glossy paths | Sampled at the fresh sampler's rate. In the mask where the edit matters the saving is exactly 1×. A matched-quality gate "in every mask" is therefore met only by saving work in unaffected masks. | DERIV (Track B §4, agreed) |
| Quality ceiling from memory | PR1 cannot hold more than `N ≤ M / (100 B · P)` spp: about 2,000 at 256² and about 110 at 1024² on 12 GB. Higher quality targets are unreachable for PR1. | DERIV |
| Edit sequences | PR2's variance grows linearly with the number of edits since the anchor. Re-anchoring is a cold converged render and belongs to the lifecycle. PR1 does not drift but stays at `N`-spp noise. | DERIV |
| View dependence | Any camera change voids PR1 and PR2. "Per-frame accounting" with a moving camera is a cold render per frame. | DERIV |
| Estimator noise mistaken for change | PR1: none, clean paths are bit-identical. PR2: noise lives only on dirty paths. | DERIV |

### 3.4 Verified-sound points

- The cost statement is honest: the exact identity is separated from the cost, and the bound is stated as a bound.
- Discovery by segment tests finds transport-dirty stationary receivers at any depth without a dependency graph (DERIV). No higher-order discovery cost exists, unlike Tracks A and C.
- No solve, no build beyond one render, no state larger than the render itself.
- The observation that cheap ER shrinks the reuse headroom is correct and important.
- Stage 1 (measure ρ and the dirty share of radiance exactly, with identical seeds) is cheap and bounds every path-granular selective scheme. It is the most valuable measurement proposed by any of the three designs on this axis.

### 3.5 Defects

| # | Severity | Defect | Basis |
|---|---|---|---|
| B1 | MAJOR | The ceiling is a constant of about 3–10× at best and under 1× for large movers. It cannot grow with scene size, because both sides scale with `P · N`. | §3.2 |
| B2 | MAJOR | The store-read term is missing from the cost formula. It bounds PR1 at 3–5× even with ρ = 0. | §3.2 |
| B3 | MAJOR | At WL84 scale the absolute saving is below the launch overhead. F1 is likely to trigger for harness reasons, not transport reasons, unless the workload is scaled so that `T_PT` is at least several seconds. | §1.2, §3.2 |
| B4 | MAJOR | PR1's memory caps the reachable quality level at megapixel output. | §3.3 |
| B5 | MINOR | PR2 lifecycle: the converged T0 anchor and its re-anchoring period are not charged. | §3.3 |
| B6 | MINOR | The dirty-fraction arithmetic uses an arbitrary `f`. A geometric estimate is available before any render. | §3.2 |

None is fatal, because Track B claims nothing beyond a bounded comparator.

### 3.6 What would have to change

- Add the store-read and compaction terms to the cost model, and state the ceiling as `1 / (ρ + c_touch / c_path)`.
- Size the workload so that the cold REF-ER render takes several seconds of GPU time; otherwise the experiment cannot resolve a 3× effect.
- Report PR2 with its anchor cost and with a stated number of edits per anchor.

### 3.7 Verdict on this axis

**CONDITIONALLY SOUND.** The economics are real, bounded and modest: a constant-factor reduction for a fixed view and a small mover. The regime (fixed camera, small rigid mover, cold render of at least seconds) is inside GM1 as a comparator. It is not a mechanism and cannot produce an asymptotic advantage.

## 4. Track C — Lobe-Partitioned Transport Graph (LPTG)

### 4.1 Cost model

Notation follows Track C §7: `N` cache points, `K` nodes, `L̄` mean sharp-chain length, `n_ℓ` light samples, `s` readout chains per pixel, `f` invalid link fraction, ρ spectral radius, `a_S` albedo share of the sharp lobe set.

| Stage | Cost | Value at N = 2·10⁵, K = 128 | Label |
|---|---|---|---|
| Build `F` | `N·K·(1 + L̄)·(1 + n_ℓ)` rays | 1–5·10⁸ rays, 0.1–0.5 s | DERIV; value EST |
| Memory | segments × 32 B plus `3·N·K` floats | 3–16 GB, see §4.2 | DERIV |
| Discovery | one pass over all stored segments | 1–5·10⁸ tests, 15–80 ms | DERIV; value EST |
| Link update `S` | `f` × build | 2–10 % of build for a small mover | DERIV |
| One solve iteration | `N·K²` multiply-adds per channel plus `g_i · N·K·(1 + L̄)` gathers, `g_i` = interpolation taps (1–12) | 10¹⁰ MAC plus 5·10⁷–2·10⁹ gathers: 0.03–1 s | DERIV; value EST |
| Iterations | `log(ε / ‖Δ‖) / log ρ` | 16 warm vs 26 cold at ρ = 0.8; 68 warm at ρ = 0.95 | DERIV |
| Readout per frame | `P·s·(1 + L̄)·(1 + n_ℓ)` rays plus `P·K` ER evaluations with interpolation | rough part ≈ 20–30 spp-equivalents per pixel; sharp part ≈ `s` chains | DERIV; value EST |
| Sustained per frame, moving object | discovery + link update + solve + readout | solve-dominated | DERIV |

### 4.2 Three corrections to Track C's own accounting

**(a) Chain length and memory (DERIV).** A link's chain continues with one sharp-lobe sample at every vertex. With Russian roulette at the sharp albedo, `L̄ = a_S / (1 − a_S)`. For a coated conductor with `a_S` = 0.6–0.9 that is `L̄` = 1.5–9. Stored segments are then 2.6·10⁷ × (2.5–10) × 2 = 1.3–5·10⁸, i.e. **4–16 GB** at 32 B, not the 1–3 GB Track C estimates. `a_S` is unknown; Track C's Stage 0 census would fix it.

**(b) The node count fixes α\* high (DERIV).** A node covers 2π/K sr. A reflected GGX lobe covers about 4πα² sr. Requiring `m` nodes inside a lobe gives `α* ≥ sqrt(m / (2K))`: 0.06 for `m` = 1 and **0.125 for `m` = 4** at K = 128. WL58 measured a 10–18 % projection loss at α = 0.25 with 256 outgoing bins (HIST, piecewise-constant basis). Interior rough-to-rough bounces in LPTG pass through an interpolated K-node outgoing set, which is the same kind of object. A realistic α\* at K = 128 is therefore 0.15–0.3, and most GGX lobes of a sharp coated conductor fall in the sharp set (INF). Lowering α\* costs memory ∝ 1/α\*² and solve ∝ 1/α\*⁴, as Track C states.

**(c) The solve is the dominant edit cost and is not output-sensitive (DERIV).** Track C re-solves the whole field after every edit. Warm starting changes only the iteration count, and only logarithmically: 16 instead of 26 iterations at ρ = 0.8 for a 10 % perturbation and a 0.3 % tolerance. Each iteration touches every link.

- Optimistic (nearest-neighbour lookup, `L̄` = 1): about 30 ms per iteration, 0.5 s per edit.
- Pessimistic (12 interpolation taps, `L̄` = 5, ρ = 0.95): about 1 s per iteration, about 60 s per edit.

Against that, a complete 1024-spp REF-ER frame costs 0.16 s at 256² and 2.6 s at 1024². **One solve iteration costs between a fifth of and several times a complete high-quality re-render at 256².** The link update, which is the part that is selective, is 2–10 % of a 0.1–0.5 s build and is immaterial next to the solve.

### 4.3 Consequences

**`S ≈ F` in total cost (DERIV).** `S = discovery + f·build + solve + readout`; `F = build + solve + readout`. With the solve and readout common, the ratio `S / F` is close to 1. This is the WL59 structure, where common work was 97.5 % of the selective update and the saving was 1.4–1.7 % (HIST, other representation). Track C's own outcome table then requires dropping the selective-update claim ("S cheaper than REF-ER but not cheaper than F").

**Sample efficiency is that of a frozen final gather (DERIV).** The field at a cache point is a K-sample quadrature of incident indirect radiance with fixed nodes. A pixel that interpolates three cache points sees about 3K ≈ 400 first-bounce samples. For the rough part, LPTG at K = 128 is therefore comparable to a path tracer at a few hundred spp, with spatially correlated instead of white error. Reaching 1024-spp-equivalent quality needs K of about 512, i.e. 4× memory and 16× solve arithmetic. HIST consistent with this: 128 gathers gave ±3.3 % per realization on gap faces (WL54) and were inadequate at α = 0.04 (WL46), both per `AGENTS.md`.

**Small movers alias (DERIV).** A mover subtending Ω is seen by `K · Ω / 2π` nodes per cache point. At K = 128, a 0.2 m box face at 1.15 m (Ω ≈ 0.03 sr) is seen by **0.6 nodes**. Its whole indirect influence on that receiver, both shadowing and reflected light, rests on zero or one node. Resolving it with 16 nodes needs K ≈ 3,300. The path tracer has the same sampling problem but averages it over pixels and spp without bias; LPTG freezes it into blotches that `F` and `S` share. A cost comparison at "matched pooled L1" would then credit LPTG for structured error (Track C §10.9 flags the metric issue; the node-count derivation is mine).

**Readout hides a partial path tracer (DERIV).** The sharp part of every camera pixel is traced with `s` fresh chains per frame. Chains terminate into the field at the first rough event, which shortens them, but the first sharp bounce carries the same lobe-sampling variance as in a path tracer. For sharp-dominated materials, which is GM1's subject, readout cost is a path trace with early termination. View independence applies only to the rough share.

### 4.4 Where LPTG can win

Per frame, with a valid state (EST): rough readout ≈ 20–30 spp-equivalents, plus `s` sharp chains. If the quality target is one where the path tracer needs `n` ≥ 256 spp, the per-frame ratio is roughly `n / (25 + s·(1 + L̄)/d̄)`: about 4–10× when `s` ≪ `n`.

Per edit: 0.5–60 s, independent of image size.

| Workload | Path tracer per frame (256–600 spp) | LPTG per frame | LPTG per edit | Frames per edit to break even |
|---|---|---|---|---|
| 256² | 40–100 ms | about 10 ms | 0.5–60 s | 10 to several hundred |
| 1024² | 0.6–1.6 s | about 0.16 s | 0.5–60 s | 1 to about 40 |

All entries are EST built from §1.3 constants. The window requires all of: megapixel output; several frames or views per edit; a quality target in the band where the path tracer needs a few hundred spp; LPTG's own bias floor below that target; a rough-lobe share large enough for the field to matter.

**The bias floor is the unknown that decides it (INF).** Finite-interface mechanisms in this project have had coupled errors of 3–10 % (HIST: WL53–55 via `AGENTS.md`, 1 cm ports 3–8 %, refined 2–5 %; other physics and interface). A REF-ER path tracer reaches 4 % noise at about 256 spp, i.e. 40 ms at 256² (EST, §1.2). If LPTG's floor is in the historical range, the window is empty at 256² and narrow at 1024². This is a possible mechanism, not a measurement of LPTG.

### 4.5 Counterexamples and threat patterns

| Threat | Finding | Label |
|---|---|---|
| Hidden path tracing | Partial: per-frame sharp chains at readout; chain regeneration inside links. | DERIV |
| Saving bought with fidelity | Possible in three places: node aliasing of small movers, cache spacing near contact, correlated error under a pooled metric. | DERIV / INF |
| Discovery cost | Accounted. Linear in stored segments in the honest worst case; 15–30 % of a full build. | DERIV; Track C §3 agrees |
| Non-sparse invalid set | Large, enclosing or light-carrying mover. Also: the *solve* is dense for every edit, whatever `f` is. | DERIV |
| Changed occlusion, unchanged endpoints | Handled by segment tests, at node resolution only. | DERIV |
| Narrow lobes through angular compression | Avoided by tracing them, at the price of (a) and of readout cost. | DERIV |
| Broad multi-bounce propagation after a local edit | Handled by the re-solve, which is why the re-solve cannot be made cheap without a residual-driven scheme. | DERIV |
| High-albedo recurrence | Iterations grow as `1 / (1 − ρ)`; Monte Carlo link weights can push the realised ρ above 1, which is a stop by Track C's own rule. | DERIV |
| Launch count | Each solve iteration is a launch unless the loop is compiled: 16–68 launches at 0.4–1.2 s each in the current harness. | INF from FACT WL84 |

### 4.6 Verified-sound points

- The segment-validity table and the output-sensitive index lookups are correct, and link invalidation is sparse for a small mover (`f` ≈ 3 % by the Crofton estimate) (DERIV).
- The build is affordable: 1–5·10⁸ rays is about one high-quality frame, not thousands. This is three to four orders of magnitude below Track A's state build, because the field is band-limited at α\* and sharp transport is not stored.
- The scaling table in Track C §7 is correct as counts. The worst-case list includes "readout dominates" and "S and F indistinguishable".
- No speed-up is claimed, and the "loose quality gate" is named as the most dangerous condition. Both are consistent with WL67–70.
- Stage 0 (lobe census and quadrature error, no rendering) is a correct and cheap admission gate. From this review, `a_S` and the realised α\* are exactly the two numbers that decide memory, chain length and readout cost.

### 4.7 Defects

| # | Severity | Defect | Basis |
|---|---|---|---|
| C1 | **FATAL for the selective-update claim (H-edit as an economic claim); MAJOR for H-econ** | The field re-solve is dense per edit and dominates. `S ≈ F` in total cost. | §4.2(c), §4.3 |
| C2 | MAJOR | Memory is underestimated by up to 5× once chain length follows the sharp albedo. 12 GB cards may not hold N = 2·10⁵, K = 128. | §4.2(a) |
| C3 | MAJOR | K = 128 implies α\* of about 0.15–0.3. For sharp coated conductors the field then carries little, and the design degenerates to its own worst case 3 ("a path tracer with overhead"). | §4.2(b) |
| C4 | MAJOR | A small mover is seen by fewer than one node per receiver. The stationary-receiver indirect change, GM1's gated quantity, is aliased identically in `F` and `S`, so `F ≡ S` cannot detect it. | §4.3 |
| C5 | MAJOR | At 256² the per-edit cost exceeds a high-quality re-render for any parameter choice in the stated ranges. A window needs megapixel output and several frames per edit. | §4.4 |
| C6 | MINOR | Readout's `P·K` evaluations cost 20–30 spp-equivalents per pixel. At contracts met by a path tracer at 1–16 spp (HIST WL67, WL69) the readout alone loses. | §4.4 |
| C7 | MINOR | Solve iterations as separate launches make wall time unusable in the current harness. | §4.5 |

### 4.8 What would have to change

- Make the solve output-sensitive: residual-driven (Southwell-type) relaxation that touches only links whose sender changed above tolerance, with the touched-link count reported. Without it there is no selective-update economics to test. With it, the scheme inherits Track A's question of how fast the residual support spreads, at the much smaller state size `N·K`.
- Add importance toward the changed set to the node set (extra links aimed at the mover's bounds), or state the minimum subtended solid angle below which the mover's indirect effect is not represented.
- Recompute memory with `L̄ = a_S / (1 − a_S)` from the Stage 0 census, and add that figure to the Stage 0 stop rule.
- Add a Stage 0 economic admission: with census values, evaluate §4.4's break-even frames per edit for the intended image size. Stop if it exceeds the frames the GM1 fixture will actually render per edit.
- Compile the solve loop into one launch, or report GPU kernel time only.

### 4.9 Verdict on this axis

**CONDITIONALLY SOUND as a view-independent rough-transport cache** in a megapixel, multi-frame-per-edit, few-hundred-spp-quality regime with a meaningful rough share. **UNSOUND as a selective-update cost claim as written**, because the dominant edit cost is common to `S` and `F`.

## 5. Cross-design comparison (economics only)

| Item | A: DDT-D | B: ERPS | C: LPTG |
|---|---|---|---|
| Persistent state size | `N_x·N_ω` ≈ 5·10⁸ cells, 6 GB or more | `P·N` paths; 0.4–7 GB at 256² | `N·K·(1 + L̄)` segments; 4–16 GB |
| Build, in 1024-spp 256² images | about 7,600 | 1 | about 1–3 |
| Discovery | cheap at first order; transport work beyond | 20 % or more of a cold render | 15–30 % of a build |
| Is the invalid set sparse for a small mover? | no: 3 % or more against a required 0.05–0.8 % | yes: ρ ≈ 0.08 | links yes (3 %); solve no |
| Dominant edit term | mover regather and dirty overwrite | store read plus suffix retrace | field re-solve |
| Edit cost vs one cold 1024-spp render at 256² | 10²–10³× more | 0.2–0.6× | 3–400× more |
| Per-frame cost with a valid state | about 1 ms (lookup) | not applicable: fixed view | 4–10× below a few-hundred-spp path trace |
| Hidden path tracing | yes if `b0` is on demand or band-limited | declared | partial (sharp readout) |
| Saving purchasable with fidelity | τ | PR3 only | node aliasing, cache spacing |
| Regime that can win | static after edit, megapixel, hundreds of views | fixed view, small mover, cold render ≥ seconds | megapixel, several frames per edit, few-hundred-spp target |
| Inside a WL84-scale GM1 fixture? | no | as a comparator; absolute saving below launch overhead | no |
| Verdict | UNSOUND in GM1 | CONDITIONALLY SOUND, bounded constant | CONDITIONALLY SOUND as cache; UNSOUND as selective update |

**Common structural finding (DERIV).** In all three, the selective part is not the dominant part. A's dominant term is a dense regather, B's ceiling is set by reading its own store, and C's is a dense solve. This repeats WL56 (changed-domain recompute 92.9 % of S; HIST via `AGENTS.md`) and WL59 (97.5 % common work; HIST). The one historical case that won, WL70, reused the single term that was 90 % of the cost with a 0.10 s validation, and still won by only 13 % (HIST).

**Common experimental risk (INF).** At WL84 scale every outcome is predetermined by the comparator's 10 ms per 64 spp and the harness's 0.4–1.2 s per launch. A GM1 cost experiment that can discriminate needs a workload whose cold REF-ER render costs at least several seconds of GPU kernel time at the gated quality, i.e. roughly `P · n ≥ 2·10⁹`, fixed before any arm runs. It also needs the gate to sit where the path tracer needs hundreds of spp; WL67 and WL69 contracts were met at 1–16 spp (HIST).

**Ranking on this axis only.** B has the only cost model that survives its own arithmetic, and it is a comparator. C has a narrow, checkable window and one fatal accounting gap (the solve). A has no window inside GM1.

## 6. Limits of this review

- Nothing here is measured. `c_touch`, `c_gather`, the "half the dirty path" factor and all 1024² values are estimates. The derived conclusions are robust to a factor of 3 in those constants for A (margin 10² or more) and for B's ceiling, but not for C's window, which a factor of 3 can open or close.
- The 2.4 ns per pixel-sample assumes 256 × 256 images, which I took from the task statement and did not verify in WL84.
- WL84's scenes have limited indirect light and depth 4. A closed glossy court with Russian roulette will have longer paths and a higher `c_path`, which moves every break-even in the designs' favour by the ratio of mean path lengths (perhaps 2–4×; EST).
- The Crofton estimates assume isotropic chords. Glossy transport concentrates chords and can raise or lower the dirty fraction locally.
- The diffuse weight and lobe roughness census of the frozen closures were not checked. They control A's rough-lobe dirty region and all of C's parameters.
- WL53–56 were used through `AGENTS.md` summaries. `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md`, WL82, WL83 and the WL77A / 80A / 80B records were not re-read for this axis.
- Santaló 1976 is cited from memory. No literature was web-verified in this review.
- Path tracing with a denoiser, and neural radiance caching, are cheaper comparators than cold REF-ER at low spp. Neither is a designated GM1 control, and neither is assessed here.
