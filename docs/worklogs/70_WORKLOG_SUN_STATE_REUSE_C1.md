# Worklog 70 — Validity-Aware First-Order Sun-State Reuse on the WL69 C1 Rigid Edit (2026-11-02)

**Authorisation:** the user's 2026-11-02 brief, "one bounded validity-aware reuse experiment on the WL69 C1 rigid-edit workload". It asks whether the first-order direct-sun transport state can be reused after the edit instead of retracing 16 M photons, and, only if that is physically valid, for the full quality and runtime comparison in the same batch.
- Branch `Selective_Recompute`; no branch or worktree.
- **Not touched:** the K/C ownership, the WL55 2.5 mm oracle K, CAK levels, C, the solver, the readout, thresholds, 8DNA, and every WL55/WL67/WL69 output.

**Coordination.** The concurrent session `rest-gi-0f` confirmed that its WL68 follow-up takes Worklog 71 on LabServer63. This batch is therefore Worklog 70 on LabServer69, the WL69 hardware the brief asks for.
- Paths: `experiments/sunreuse70`, `results/sunreuse70`, `results/evaluation/70`, server root `~/rest_sunreuse70`.

**Protocol:** `experiments/sunreuse70/protocol/sr70.json` (`sunreuse70_v1`).
- It was committed at `01acfa8` together with code and tests, before any evidence run.
- Development disclosure: the focused tests had been run once (11/11) with 1 M photons on development seeds 9001–9003.
- No amendment was needed.

## Completion answer

> Can the material-frame direct-sun state be safely reused across C1's rigid geometry edit, and does that reuse materially improve the measured quality-matched update cost?

**Yes on both counts for C1, with a narrow margin against Full Recompute. Outcome A under the predeclared map.**

**Validity.**
- Checked against the dependency contract, the cached T0 sun state is valid at T1: rest mesh, medium, rigid pose, material-frame sun and cameras, estimator, and a replay of all 16 M source samples through the T1 shadow test with 0 mismatches.
- With independent replicas, T1's retraced first-order state is physically identical to T0's under the noise-aware test. This holds for the incident source, the fine-port and port × direction first-exit flux, and L0.
- All four invalidation controls are rejected, each by the expected dependency. The off-path control is accepted and is physically identical.

**Update time.**
- With reuse, the measured T1 update falls from **24.6 s to 2.09 s**. Validation takes 0.10 s of that.
- Quality is unchanged: 12/12 on both states, M1–M4 equal to the historical CAK.

**Against Full Recompute.**
- Quality-matched Cycles needs 16 spp (M4 binds). Under WL69's original timing (manifest wall time) that is **2.40 s**, so B is beaten by 13 %; the predeclared rule needs < 2.16 s.
- **The margin is fragile:**
  - Blender's own render time for the same 16-spp renders is 2.09 / 2.05 s, which would read as "matched".
  - Cycles meets the bare 3 % contract at 1 spp in 1.20 s, below B's time. B's higher change-tracking quality is what forces 16 spp.

## Implementation Fact

### Stage 1 — dependency contract (from code)

The historical computation is `k55_compose.sun_term`, which calls `p53_tracer.run_sun` on `k55_ops.DomainX` at the level-4 refinement:
- n_sun 16 M, seed 5591 + ci, 4 M chunks, Dout 64;
- the cameras are the requested FX direction keys;
- shadow = `classify_sceneset(full scene).isect`.

It outputs:
- `phi_f`: fine port × out-bin flux, in port-local frames;
- `L0`: per camera direction and coarse port, the projected-area mean.

What `compose` recomputes on the current geometry, and so is never cached:
- `aggregate_phi` onto the CAK cells;
- the external Lambert sun `Ep`;
- C, the solve, and the KC/KB readout.

The walk uses only the domain's own intersector and stops at the first air exit. So other geometry enters the sun state only through the source visibility test.

| Dependency | Check (`sr70_state.validate`) | Tolerance |
|---|---|---|
| D1 rest mesh | sha256 of V, F, tri_port, port_tangent, port_area, port_face | exact |
| D2 medium | (σt, albedo, g) | exact |
| D3 pose | Kabsch rest → current | residual ≤ 1e-9 m |
| D4 light | full light list in the material frame (Rᵀs, E) | 1e-9 |
| D5 cameras | every requested key cached, material-frame direction equal | 1e-9 |
| D6 visibility | replay every cached source sample (fine triangle + two uniforms) at the current pose through the current full-scene shadow test (same 4 eps offset) | bit-identical mask |
| D7 estimator | n_sun, seed, chunk, Dout, level | identical |

**Variables checked for C1:**
- **Translation versus rotation.** T1 is a pure translation: Kabsch R = I, residual 5.6e-17 m.
- **Local frames.** Frames are rigid in material coordinates; the refined meshes have identical rest corners per triangle.
- **Sun direction and E.** Unchanged.
- **Material.** Unchanged.
- **Camera directions.** Unchanged.
- **Upstream occlusion.** None in either state; this is what D6 checks.

**Limit.** D6 is exact at the estimator's own sample resolution. An occluder that misses all 16 M cached samples would also be missed by a retrace with that seed.

### Implementation (isolated, all-or-nothing)

`sr70_state.run_sun_rec` / `sun_source_rec` is a recording replica of the historical source and `run_sun`.
- It consumes the RNG identically; the test checks the bitwise outputs and the generator state.
- At T0 it also stores the material-coordinate source record and its visibility mask.

At T1, `validate` either accepts, in which case `reuse` returns the cached `phi_f`/`L0` in the historical format, or rejects, in which case the unchanged `sun_term` runs.
- No scene name is consulted; a test enforces this.
- Everything downstream is the frozen WL69 path: `SB.cak_levels`, `KP.arm_operator`, `KP.compose`, including its spectral-radius diagnostic.

### Execution

One detached chain on LabServer69: tests → identity → gate → e2e → analyze, 2026-11-02 12:08–12:22 KST.
- GPU-idle gating ran before every stage.
- The 15 s sampler saw only this chain's three processes, never more than one at a time.

## Measurement

### Stage 2 — physical identity (independent replicas; B = T0 seeds 7001/7002, A = T1 seeds 7011/7012)

| Element set | R per channel (limit) | Groups | max \|Δ\| | max SE (precision ≤ 0.3 %) |
|---|---|---|---|---|
| Fine-port first-exit flux | 0.986 / 0.974 / 0.943 (≈ 1.20–1.21) | 15 face × channel, all pass | 0.175 % | 0.215 % |
| Coarse port × out-bin flux | 0.989 / 0.981 / 0.962 (1.10–1.14) | — | — | — |
| Incident source power per fine port | 1.051 (1.228) | 9, all pass | 0.094 % | 0.155 % |
| L0 (top, bot, otop keys) | 0.55–1.07 (1.95–2.94) | 12 camera × face, all pass | 0.270 % | 0.257 % |

**Noise-model calibration.** The T0-vs-T0 null comparison ({7003, 5591} against {7001, 7002}) gives R from 0.977 to 1.455, all inside the limits. Every tested group meets the precision rule.

**Implementation checks:**
- The replica equals the historical `run_sun` at T1 seed 5591 bit for bit (max relative difference 0.0).
- The cache and the historical T1 sun term reproduce WL55's stored `floor` / `floor_c4` sun terms bit for bit.

**Same seed, T1 historical against the cache.** The realizations decorrelate:
- max element relative difference 6.3 % on fine ports, 3.0 % on port × bin, 1.2 % on L0;
- this is Monte Carlo variation; the identity statistics above show that the distributions are equal.

### Stage 3 — invalidation controls (cache = T0 seed 5591; physical truth from seeds 7021/7022)

| Control | Decision (failing check) | Physically changed? | Evidence |
|---|---|---|---|
| T1 (edit) | accept (all; 0 / 16 M visibility mismatches) | no | above |
| N1 sun zenith 45° | reject (D4) | yes | R 16–645; face flux up to 7.7 %, source 10.0 %, L0 9.6 % |
| N2 occluder on the sun path | reject (D6: 640,073 of 16 M samples newly shadowed, 4.0 %) | yes | R 44–4,636; face flux 4.6 % |
| N3 albedo blue 0.8 | reject (D2) | yes, blue only | blue R 130–19,424; flux 29.5 %, L0 52.8 %; red/green R ≈ 1 |
| N4 slab rotated 90° (same world volume) | reject (D4, D5) | yes | R 103–36,996; face flux up to 45 %, source 100 % on faces turned away |
| P1 box off the sun path | accept (0 mismatches) | no | R ≤ 1.0, all groups pass |

- Every expected-reject control was rejected, so no stale light was returned. P1 was not over-rejected.
- The controls share seeds 7021/7022 by design. As a result, P1's red/green replicas equal N3's, since N3 changes only blue. Each control is still independent of the cached replicas.

### Stage 5 — end to end (T1, both rest K seeds, interleaved A B B A B B A B)

**Latency** (median over repetitions; K seeds s1 / s2):

| Arm | Total | Breakdown |
|---|---|---|
| A, retraced sun | **24.60 / 24.62 s** | sun 22.61–22.70; levels 0.011; aggregation 0.73; compose 1.23–1.24 |
| B, validated reuse | **2.09 / 2.08 s** | validation 0.099–0.100 (identity 0.0008, rigid 0.0005, directions 0.0001, shadow-scene build 0.007, 16 M visibility replay 0.090); reuse < 0.001; levels 0.011; aggregation 0.72–0.74; compose 1.24–1.25 (C 0.040, solve 0.085 / 51 iterations, plus external sun, readout and the spectral-radius diagnostic) |

- Every B repetition accepted the cache with 0 mismatches.
- Speed-up from A to B: 11.8×.

**Memory and one-time costs:**
- Peak GPU per update: A 6.9–7.0 GB, B 7.2–7.4 GB. The difference is the resident 208 MB source record.
- One-time costs:
  - K load: 0.5–0.9 s (warm file cache).
  - Cache construction: 21.5–21.7 s. This is the T0 sun term itself, through the recording replica; peak 1.9–2.0 GB.
  - Cache size: 211.4 MB (source record 208.0 MB, outputs 2.96 MB, rest geometry 0.46 MB).

**Quality** (unchanged WL69 machinery: mean of the two K seeds, SE with the seed half-difference):

| Arm | T0 M1 | T1 M1 | max \|bias\| T0 / T1 | T1 M3 | M4 change error | slope | corr |
|---|---|---|---|---|---|---|---|
| A | 12/12 | 12/12 | 1.37 / 0.76 % | 0.023 / 0.026 / 0.033 | 0.2055 / 0.1999 / 0.2045 | 1.025 / 1.047 / 1.058 | 0.980 / 0.982 / 0.983 |
| B | 12/12 | 12/12 | 1.37 / 0.74 % | 0.022 / 0.026 / 0.032 | 0.2049 / 0.1996 / 0.2044 | 1.025 / 1.048 / 1.058 | 0.980 / 0.983 / 0.983 |

- A reproduces WL69's stored T1 predictions to 1.9e-7, and T0 to 1.7e-7.
- B differs from A per port by at most 1.0 % (otop), 0.85 % (top) and 0.34 % (bot). These are different sun realizations; group biases differ by ≤ 0.1 points.
- The STALE control is unchanged from WL69: T1 6/12, up to 70 %.

**Full Recompute** (unchanged WL69 Cycles ladder, same RTX 3090):

| Quantity | Value |
|---|---|
| QM_A = QM_B | 16 spp (M4 binds) |
| QM_contract | 1 spp |
| WL69 manifest time, T1 ruled views | 1 spp 1.20 / 1.20 s; 16 spp 2.40 / 2.30 s |
| Blender-internal render time of the same renders (from WL69 `cy.log`; `results/sunreuse70/cy_blender_internal_times.json`) | 1 spp 0.88 / 0.87 s; 16 spp 2.09 / 2.05 s |
| **Predeclared relation** | B 2.09 s vs CY(QM_B) 2.40 s → **beaten** (threshold 0.9 × 2.40 = 2.16 s) |

## Observation

- The direct-sun first-exit state of the C1 slab carries no measurable pose dependence under the translation: about 0.2 % group-level precision, R ≈ 1. In this workload it is exactly the 90 % of the update that WL69 measured as `sun_term`.
- **Contract checks are cheap; only visibility costs.** Every check except visibility is sub-millisecond. The 16 M-sample visibility replay is 0.09 s, less than one-hundredth of the retrace.
- **What remains is K/C work.** It splits into aggregation (0.73 s) and compose (1.25 s). Inside compose only C (0.04 s) and the solve (0.085 s) are timed separately.
- **Only some controls are invisible to world-space geometry.** The rotation control leaves the world boundary unchanged but changes the material-frame state by up to 100 % on faces. The albedo control changes only blue. Neither shows in world-space geometry; the occluder and sun controls change the incident field directly.
- **The repetition-hash expectation was not met.** The protocol expected identical prediction hashes across repetitions within an arm; in both arms they differ. The run-to-run difference is the GPU accumulation nondeterminism already visible against WL69: 1.7–1.9e-7 relative. It is not part of the gate or the verdict and is recorded as an unmet expectation.

## Interpretation

**Why reuse is valid for C1.** Three things hold together:
- the edit is a translation that preserves every material-frame direction;
- the incident field is unoccluded in both states;
- the intrinsic walk is domain-isolated.

The contract tests these explicitly, rather than inferring validity from "rigid motion". The rotation and occluder controls show that rigid motion alone does not guarantee it.

**What the advantage depends on.** Against quality-matched Full Recompute it is measured but small (0.31 s, 13 %), and it depends on the timing definition. WL69's manifest seconds include the Python readback around each Cycles render. On Blender's internal render time alone, CY at 16 spp (2.05–2.09 s) equals B. Cycles also reaches the bare contract at 1 spp (1.2 s); the 16-spp match is forced by B's per-port change-tracking fidelity (M4), not by group bias. In the other direction, the CY timing excludes Blender start-up and scene construction, while B's update excludes nothing; K and the cache are resident.

**Scope.** C1 only: one rigid translation, one collimated sun, one homogeneous slab, three ruled views, one machine. No claim about rotations that preserve the material-frame sun, multiple lights, occluded configurations, larger scenes or view counts.

## Unresolved Question

- **Where compose's ~1.1 s goes.** Compose's remaining time beyond C and the solve (external sun, ballistic KB sampling, readout, spectral-radius diagnostic) is not broken down. Whether any of it is required work was not examined (no unrelated optimisation was authorised).
- **Sample resolution of D6.** Whether an occluder thinner than the cached sample spacing matters physically. D6 is only as fine as the estimator.
- **Scaling of the advantage.** How it scales with view count, which multiplies Cycles' cost but not the K/C update; with scene size; and with edits that change the incident field of some domains, where those domains would fall back to the retrace.
- **Stability of the margin.** Whether the 13 % margin over Cycles survives a common timing definition (e.g. including Blender scene sync and start-up, or excluding readback on both sides).

## Architecture Verdict

**Outcome A (predeclared map): physically valid, with a measured end-to-end advantage that is fragile.**
- The material-frame first-order sun state is reusable across the C1 rigid edit under the dependency contract D1–D7.
- With it, the CAK update (2.09 s) beats quality-matched Full Recompute (2.40 s, WL69 timing) at unchanged physical quality.
- The margin disappears under Blender's internal render timing, and Cycles remains cheaper at the bare 3 % contract (1 spp).

**What changes and what does not.**
- No K/C ownership, production or Architecture change.
- The diagnostic cache is not promoted.
- The original WL69 24-s measurement stays the historical baseline.
- No successor is authorised.

## Reproducibility

- **Code and protocol:** `experiments/sunreuse70/` at `01acfa8`: protocol, `sr70_state`, `sr70_scenes`, `sr70_stats`, `sr70_identity`, `sr70_gate`, `sr70_e2e`, `sr70_analyze`, tests, `chain.sh`. `sr70_figures.py` was added afterwards and is report-only.
- **Server:** LabServer69 `~/rest_sunreuse70`, a `git archive` of the repo with `core.autocrlf=false`. `env_8dna.sh` is copied from WL69 (8dna26 venv, torch 2.3.1, Mitsuba 3.5.1). Run `bash repo/experiments/sunreuse70/chain.sh`; it is resumable.
- **Read-only inputs:**
  - WL55 operators `~/rest_wl55/ops/slab_L4_s{1,2}.pt`;
  - WL55 Cycles references `~/rest_wl55/out/cycles_cuda`;
  - WL55 stored sun terms;
  - WL69 `~/rest_xbench69/{rg,cy,analysis.json}`.
- **Seeds:**
  - cache: 5591;
  - replicas: T0 7001–7003, T1 7011–7012;
  - controls: 7021–7022;
  - development tests: 9001–9003;
  - compose uses its frozen seeds.
- **Hashes:**
  - cache output: `c27c6309…`;
  - rest mesh: `8de00580…`;
  - `results/sunreuse70/identity.json`: `30ae3c23…`; `e2e.json`: `263d2282…`; `analysis.json`: `90784548…`.
- **Outputs:**
  - `results/sunreuse70/`: JSON results and logs (gate, steps, GPU sampler, stage logs) and the Blender-internal CY times.
  - `results/evaluation/70/01_latency_and_quality.png` with its manifest.
- **Left on the server:** prediction npz files and the identity summaries.
