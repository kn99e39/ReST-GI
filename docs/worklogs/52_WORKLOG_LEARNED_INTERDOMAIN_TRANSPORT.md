# Worklog — Geometry-Conditioned Learned Inter-domain Transport: Bounded Architecture Falsification (2026-10-28)

## Protocol, authorisation and isolation

- **Protocol:** `fnto_lc52` (`experiments/fnto_lc52/protocol/lc52.json`).
- **Code commit:** 47246f1, made before any primary evidence.
- **Amendments:**
  - 1 (4ea3c68) and 2 (fce5e57) are training-loop changes, made before any trained model was evaluated.
  - 3 (7cb1d90) is labelled **post hoc**, made after the primary evaluation.
- **Authorisation:** the user brief "ReST-GI — Worklog 52: Geometry-Conditioned Learned Inter-domain Transport, Bounded Architecture Falsification" (2026-10-28).
- **Environment:** every run was on LabServer69 (RTX 3090 24 GB), venv `8dna26` (torch 2.3.1+cu121, Mitsuba 3.5.1 `cuda_ad_rgb`).
- **Frozen:** production K44, DASQ, C, solver, references and worklogs 1–51. The candidate lives in `experiments/fnto_lc52` and imports no production module; Stage A only *times* production code.

## Answer to the completion question

> "Can a learned inter-domain transport rule replace a meaningful fraction of explicit C computation while preserving accurate and stable GI under unseen dynamic geometry configurations?"

**No, under the measured conditions.**

The learned rule itself works as a transport operator:
- it is linear, non-negative and passive;
- it is stable under recurrent composition (280 / 280 coupled solves converged, ρ 0.51);
- it shows no memorization: training and unseen configurations have equal error;
- it generalizes to unseen arm/sheet combinations and to 2–3× more objects, in each case to within a small margin of the ray-traced transfer of the same patch representation.

It nevertheless fails on two of the four predeclared criteria:

1. **Computational overhead (primary limiting factor).**
   - In the target FNTO pipeline, explicit C is not a meaningful per-iteration cost (2.9 % of a step).
   - Its geometry-update path is cheap on the GPU: about 5 ms at 10k ports, uncontended.
   - The learned C costs 8–19× more per update at the tested scales, and its dense receiver × patch evaluation scales as O(N·M): at 4× ports it is 674 ms vs 11 ms.
2. **Transport information at visibility / contact transitions.**
   - Localized errors at near-contact and visibility-transition receivers are about 8–9× those of explicit C.
   - They are mostly inherited from the patch-level source representation that the learned operator needs: the ray-traced patch transfer C0p is almost as bad.
   - The learning itself adds a smaller visibility-inference error.

Physical consistency and model generalization are not the limiting factors within the tested range.

---

## STAGE A — cost of the production explicit C (internal gate)

Measured on the Lego, `results/fnto_lc52/stageA/`.

| Quantity | Value |
|---|---|
| Per-iteration C read (aggregates) + write (gather) | 14.6 + 8.8 ms of an 831 ms step (2.9 %); K (outgoing) 787 ms (95 %) |
| Full relation build, on-device ray tracing (109 M rays) | 0.105 s (1.04 G rays / s) |
| Full relation build, production Python path | 277 s (setup 41 s, host trace 8.5 s, hit mapping 227 s) |
| Incremental T0 → T1 update | 81 s (CPU slab selection 67 s); 6.9 M rays retraced (≈ 7 ms of tracing) |
| Table memory | hit + t 875 MB (host); GPU Hc int64 875 MB + valid 109 MB |

- **Gate decision (recorded in the protocol before training):**
  - The only replaceable costs are the geometry-change update path (BVH, N × K tracing, hit mapping) and the table memory.
  - The batch proceeded because a plausible replaceable cost exists.
  - The production Python path's cost is implementation overhead, not transport cost.
- **Comparison baseline:** every speed comparison therefore uses an efficient GPU explicit C0 (on-device tracing and mapping). Comparing against the Python path would have credited the candidate with that overhead.

## STAGE B — learned C hypothesis (recorded before training)

**Controlled K.** Lambertian domains, `K = ρ/π E_tot` per port. This K is exact under rigid motion, articulation and deformation. There is no learned K and no Burley.

**C1: GLT, geometry-conditioned latent transport** (`lc_model.py`, 923,010 parameters).
- **Operator:** `E_ind(r) = Σ_s F_rs(G) · L̄_s` with `F_r· = π · softmax([ℓ_null, ℓ_rs])` and `ℓ_rs = log(FF0_rs/π) + g_φ`.
  - It is linear in radiance; F ≥ 0; Σ F ≤ π (passive); zero input gives zero output.
  - Boundary convention: receiver = port, source = patch area mean.
- **Geometry encoding:** patch tokens (centroid, normal, log area, Fourier position) feed a 6-layer transformer (d 128). This is the only place occlusion can be inferred; no rays are used.
  - Receiver query: port position, normal and Fourier features, plus the latent of the receiver's own patch.
  - Pair head: 3 × 128 over the query, the patch latent and 10 relative-geometry features.
- **Candidates:** every patch with positive analytic unoccluded FF0, i.e. a dense receiver × patch evaluation.
  - The predeclared top-k rule (smallest k ∈ {64, …, 256} with ≥ 99 % hit-mass coverage) **failed** on 100 training configurations: coverage was 71 / 84 / 91 / 95 %.
  - With positive-FF0 candidates, coverage is 99.5 % (minimum 98.8 %) at 56 % of M per receiver.
- **Scaling, as stated in the protocol:**
  - per geometry, O(M² d + N M d) — dense, so the asymptotic scaling is **not improved** over explicit gathers O(N K log T);
  - per iteration, a dense O(N M) product.
- **Objective:** cross-entropy to the explicit 1,024-gather hit distribution over [NULL, patches]. This supervises the operator, never an illumination.
- **Exclusions:** diffuse (irradiance) transport only, so no directional radiance; ~12.5 cm source patches; an open room in which the sky is NULL.

## STAGE C — data and reference construction

**Scene family** (`lc_scene.py`).
- Static room (2 × 2 m floor and two walls).
- Boxes, including contact gaps of 0.5 mm–2 cm, wall gaps of 1–10 mm, and elevated or tilted boxes.
- Articulated arms (base and two hinged links).
- Deforming sheets (a 3 mm closed slab with a height-field top).
- Ports are ~3.1 cm grid cells; patches are 4 × 4 ports. A configuration has ~9–15k ports and ~650–1100 patches.

**Splits** (by complete configuration or trajectory; seed and configuration-hash disjointness is tested).

| Split | Content |
|---|---|
| train | 3,000 configurations, 1,024 random receivers each |
| val | 200 configurations |
| test_iid | 120 configurations |
| combo | 30 configurations with 2 arms + 2 sheets (training has at most one of each) |
| scale_obj | 30 configurations with 8–12 boxes (training has 1–4) |
| traj | 6 trajectories × 30 frames: slide to a wall (to 1 mm), arm swing, sheet morph, box passing under a sheet, two boxes meeting (to 1 mm), lift-off |
| train_cfg | 30 training configurations evaluated on all ports (memorization control) |

**Supervision validation** (all done before training).
- Point-to-parallel-square analytic form factor: 1.1238 vs 1.1237.
- Constant-radiance normalization: exact.
- Object-pair reciprocity: 0.957–1.045.
- Room only, path-traced vs port solve: 1.000–1.003.
- **A defect was found and fixed:** bilinear sample points on curved sheet cells lay below the triangles, which buried ray origins and produced false shadows. Path-traced / port ratios were up to 1.07–1.17 before the fix and 0.98–1.00 per object after it. Sample points now sit on the triangles and carry the triangle normals.

**Tests.** 10 / 10 pass (`results/fnto_lc52/tests/test_lc52_pre.json`): analytic form factor, normalization, reciprocity, the learned contract (zero / superposition 2e-6 / positivity / passivity), deterministic evaluation and gathers, FF0 on the own face, K/C composition (Neumann series 3e-6), a visibility transition, and train/test separation.

## STAGE D — training

- **Run 3 (the evaluated run):**
  - Amendment-2 setting: batch 8 configurations × 1,024 receivers, AdamW 6e-4, warm-up 2k + cosine, 40,000 steps, bf16, seed 5201.
  - 3,412 s of training (57 min), GPU 100 %, 13.2 GB, plus 3 min of data generation.
  - Validation CE plateaued: the last 5 evaluations span a relative range of 1.6e-5.
  - Train CE 2.508 ≈ validation CE 2.502.
  - Validation E error on smooth fields is 0.049, against 0.730 for the unoccluded analytic prior.
- **Amendment 1:**
  - The user flagged ~38 % GPU utilization. The cause was host-side: per-step CPU→GPU copies and CPU target densification, sequential small kernels, and a per-step sync.
  - Fix: GPU-resident data and one padded batched forward. The batched loss equals the unbatched loss to 2e-8; a step takes 22 ms instead of 75 ms.
  - Run 1 was stopped at step ~13k and preserved (`train_run1_superseded`).
- **Amendment 2:**
  - Measured throughput was compute-saturated: 85k receiver samples / s at 4 × 512 (6.6 GB), 92k at 8 × 1024 (13.2 GB); 16 × 1024 runs out of memory.
  - The VRAM headroom was used for a 4× batch and 2.7× more samples (328 M); the learning rate was scaled by the square root of the batch ratio.
  - Run 2 was stopped after ~2 min and preserved.

## STAGE E — single flight

Mean relative L2 of E_ind vs the 2,048-gather oracle. ORA2 = an independent oracle (noise floor); C0 = explicit 128 gathers; C0p = ray-traced patch transfer from 1,024 gathers (same representation as C1); C1 = learned.

| Set / field | ORA2 | C0 | C0p | C1 |
|---|---|---|---|---|
| test_iid smooth | 0.008 | 0.055 | 0.042 | 0.065 |
| test_iid hot (5 emitting patches) | 0.104 | 0.461 | 0.142 | 0.126 |
| test_iid first event (ρ/π E_sun, with shadows) | 0.017 | 0.093 | 0.118 | 0.127 |
| test_iid i.i.d. port noise | 0.020 | 0.070 | 0.098 | 0.106 |
| train_cfg first event | 0.015 | 0.083 | 0.149 | 0.160 |
| combo first event | 0.014 | 0.076 | 0.122 | 0.142 |
| scale_obj first event | 0.018 | 0.098 | 0.191 | 0.203 |
| traj first event | 0.010 | 0.057 | 0.094 | 0.101 |

**Memorization.** None: on train_cfg, C1's error is not lower than on test_iid.

**Learned error vs representation error.** C1 − C0p is small in every set (0.01–0.02 on the first-event field). Most of C1's error comes from the patch representation, not from learning.

**Uniform-radiance closure vs ORA.** C1 0.990–0.997; C0 1.000; C0p 1.000.

**Near-contact (< 1 cm), first-event field (test_iid).** C1 0.385, C0p 0.363, C0 0.084.

**Metric defect.** The predeclared *relative* error on the "strongly occluded" subset divides by ~0 oracle irradiance (values up to 1e11). Those numbers are meaningless and are not interpreted.

## STAGE F — coupled GI

Lambertian K, sun of each configuration. REF = independent path-traced GI at the ports (2 × 512 paths, continuous geometry). Images come through 3 fixed cameras, display `clip(x · e)^(1/2.2)` with `e` = 1 / p99 of REF.

| Set (configurations) | Converged ORA / C0 / C0p / C1 | Full-image PSNR vs REF: ORA / C0 / C0p / C1 | ΔPSNR C1 − C0 [95 % CI] | E_ind rel L2 vs REF: C0 / C1 |
|---|---|---|---|---|
| test_iid (40) | all | 43.75 / 39.58 / 41.81 / 41.34 | **+1.76 [0.92, 2.60]** | 0.112 / 0.158 |
| combo (30) | all | 44.79 / 40.72 / 41.92 / 40.45 | −0.26 [−1.31, 0.83] | 0.101 / 0.176 |
| scale_obj (30) | all | 43.69 / 39.10 / 41.05 / 40.02 | +0.93 [−0.13, 2.02] | 0.129 / 0.239 |
| traj (180 frames) | all | 45.43 / 42.57 / 43.31 / 42.64 | +0.06 [−0.23, 0.34] | 0.097 / 0.147 |

- **Stability.** Spectral radius C1 0.51, C0p 0.52, C0 0.58. Iterations 12–14 for every arm.
- **Why C1's PSNR beats C0's.** C0's 128-gather Monte Carlo noise is spread over every port; C1 and C0p are noise-free but biased. The per-port indirect accuracy of C1 vs REF is *worse* (0.158 vs 0.112), and REF itself has a split noise of 0.09–0.11.
- **Per-configuration minimum ΔPSNR:** −2.8 dB (test_iid) to −4.9 dB (combo).
- **Temporal tracking** (trajectories; change of E_ind between consecutive frames vs the oracle):

  | Arm | Correlation | Relative L2 at transition ports |
  |---|---|---|
  | C1 | 0.54 | 0.90 |
  | C0p | 0.59 | 0.85 |
  | C0 | 0.75 | 0.64 |

  On average 492 ports per frame change visibility.

**Localized error (amendment 3, POST HOC).** `|E_a − E_ORA| / mean |E_ORA|` of the coupled solutions.

| | C0 | C0p | C1 | C1 − C0p (learned only) |
|---|---|---|---|---|
| test_iid all p99 / top-1 % squared-error share | 0.33 / 0.12 | 0.55 / 0.77 | 0.63 / 0.77 | 0.19 / 0.69 |
| test_iid near-contact p99 / max | 0.27 / 0.37 | 2.30 / 3.60 | **2.32 / 3.63** | 0.79 / 1.52 |
| traj transition ports p99 / max | 0.18 / 0.24 | 1.32 / 1.92 | **1.48 / 2.14** | 0.67 / 0.92 |
| traj near-contact p99 | 0.15 | 1.86 | 1.96 | 0.73 |

- C1's error is **concentrated**: 77 % of the squared error sits on 1 % of the ports, at contact edges.
- C0's error is spread Monte Carlo noise.
- See `results/evaluation/52/02_coupled_error_maps.png`; one shared absolute scale.

## STAGE G — cost and scaling (matched hardware; both arms measured in the same process)

Three benchmark runs exist; they are all recorded:
- **(a) Uncontended:** 00:59, before any other job, same code. The model weights do not change the cost: this run used the 400-step smoke checkpoint, and the iteration counts differ by ≤ 8 applications of 0.2 ms.
- **(b) GPU-contended:** chain 2, with an unrelated `rest_wl53` job holding the GPU.
- **(c) GPU idle but CPU-saturated:** chain 3, with an unrelated Cycles render using 14 of 16 cores (load ~17). Explicit C's Mitsuba path is CPU-sensitive: BVH build 7–18 ms vs 2 ms; Lego BVH 1.46 s vs 0.82 s.

- **(d) Fully idle (chain 4, primary from now on):** run at 07:01 with load average 0.9 and no GPU compute process. It reproduces (a) within ~5 %:
  - base scale: C path 0.18×, frame 0.31× (5.0 vs 44.0 ms update);
  - 16 boxes: 0.08× / 0.13×;
  - 40k ports: 11.1 vs 672.2 ms;
  - coarse port × 2: 0.79× / 0.90×;
  - Lego BVH 0.82 s.

**Primary evidence = (d); the table below is (a), which (d) matches within ~5 %.**

| Scale | N / M | C0 update (BVH + 128 gathers) | C1 update (encoder + dense F) | C path speedup C0 / C1 | Frame (BVH + sun + C + solve + lookup) C0 / C1 |
|---|---|---|---|---|---|
| 2 boxes | 9,780 / 664 | 5.2 ms | 39.4 ms | 0.16× | 12.6 / 47.1 ms |
| 4 boxes (base) | 10,086 / 706 | 5.0 ms | 43.9 ms | **0.18×** | 17.3 / 51.9 ms |
| 8 boxes | 12,556 / 882 | 5.4 ms | 68.6 ms | 0.12× | 15.1 / 76.6 ms |
| 16 boxes | 14,712 / 1,108 | 5.7 ms | 109.2 ms | 0.08× | 16.3 / 117.4 ms |
| port × 2 (coarse) | 2,522 / 206 | 4.6 ms | 7.1 ms | 0.85× | 14.2 / 14.4 ms |
| port × 1 | 9,868 / 698 | 5.1 ms | 42.6 ms | 0.16× | 13.8 / 50.4 ms |
| port × 0.5 (fine) | 39,576 / 2,580 | 11.2 ms | **673.9 ms** | 0.03× | 26.5 / 685.3 ms |

- **Per-iteration application:** 0.1–0.2 ms for both arms. For C1 that is the dense F @ L̄.
- **Encoder only:** 4.1–4.4 ms.
- **Memory at the finest scale:** C1's F is 408 MB vs C0's int64 table 41 MB (10×).
- **Context:** the Lego BVH (2.1 M triangles) builds in 0.82 s.
- **Under (c):** the only C1 advantage appears at the coarsest resolution (2.52× on the C path), where explicit C0's CPU-bound fixed overhead dominates. At the base scale, (c) still gives 0.48×.

## Predeclared criteria

| Criterion | Result |
|---|---|
| P1 coupled fidelity (test_iid ΔPSNR ≥ −0.5 dB) | **PASS**: +1.76 [0.92, 2.60] dB. The cause is C0's Monte Carlo noise, not better transport (per-port E_ind error is worse) |
| P2 localized (p99 near-contact / occluded / transition ≤ 2 × C0) | **FAIL**: predeclared relative metric 101 vs 0.79 (ill-conditioned). Post-hoc well-conditioned metric: near-contact 2.32 vs 0.27, transition 1.48 vs 0.18. The transition-delta sub-item passes (0.90 ≤ 2 × 0.64) |
| P3 operator validity | **PASS**: closure 0.996, ρ 0.51, 280 / 280 converged |
| P4 speed (C path ≥ 2×, frame faster) | **FAIL**: 0.18× and 0.31× at the base scale, fully idle (d); the uncontended smoke (a) gives 0.18× / 0.33×. (b) and (c) give 1.13× / 0.82× and 0.48× / 0.55× |

## OBSERVATION

- **What succeeds.**
  - The learned operator reproduces the action of ray-traced patch-level transfer on unseen geometry, including object counts and combinations never trained on, and on unseen radiance fields; the operator is linear, so radiance generalization holds by construction.
  - It does so to within 0.01–0.02 relative L2 of C0p in single flight and ~0.5 dB in coupled images.
  - It is passive and stable under recurrence.
- **What fails.**
  - Cost: update 8–19× slower than efficient explicit C at 10–15k ports, and 60× at 40k.
  - Sharp transport changes: near contact, at visibility transitions, and over consecutive frames (temporal change correlation 0.54 vs 0.75).

## INTERPRETATION

- **Approximation failure (representation).**
  - With ~12.5 cm source patches, transfer at contacts and occlusion edges is under-resolved (C0p ≈ C1).
  - Making the patches finer raises M, and the dense O(N·M) cost grows with it. The coarse-patch regime that keeps C1 tractable is exactly the one that loses localized transport.
- **Generalization failure:** not observed within the tested range. The learned-only error is small on average but concentrated at contacts (p99 0.79 × mean E) and in occluded regions.
- **Physical-contract failure:** none. The constraints hold by construction and are verified.
- **Computational failure: decisive.**
  - Explicit C on the GPU is already a few milliseconds per geometry update, and its per-iteration application is negligible.
  - In FNTO the dominant cost is K (95 % of a Lego step), not C.
  - The learned C replaces a ray cast with a dense neural pair evaluation that is more expensive and scales worse.
- **The P1 pass is not evidence of better transport:** it reflects variance vs bias.

## UNRESOLVED QUESTION (material to the decision)

- Whether a non-dense learned C — e.g. hierarchical or sparse candidates with a learned visibility test — could beat ~5 ms per update at all. The top-k attempt failed coverage; any such design must also carry the transition fidelity that coarse patches lose. Not tested; out of scope.
- The fully idle benchmark (chain 4) confirmed the uncontended numbers within ~5 %; the cost result is not an artefact of contention.

## ARCHITECTURE VERDICT (measured conditions only: Lambertian K, one synthetic scene family, ~10–40k ports)

| Item | Verdict |
|---|---|
| Learned-C operator validity | **SUPPORTED** (linear, ≥ 0, passive, closure 0.99–1.00, stable) |
| Generalization across geometry | **CONDITIONALLY SUPPORTED** (no memorization; combos and 2–3× object counts hold at the representation's level; learned error concentrated at contacts) |
| Generalization across illumination | **SUPPORTED** (linear operator; 4 field types track C0p) |
| Dynamic temporal fidelity | **NOT SUPPORTED** (transition p99 8× C0; change correlation 0.54 vs 0.75) |
| Repeated composition | **SUPPORTED** (280 / 280 converged, ρ 0.51; coupled images within the P1 margin) |
| Computational advantage | **NOT SUPPORTED** (0.18× C path, 0.33× frame at the base scale) |
| Scalability | **NOT SUPPORTED** (dense O(N·M); 674 vs 11 ms at 40k ports; 10× memory) |
| Full FNTO integration readiness | **NOT SUPPORTED** |

The candidate is not adopted, and no direction is selected.

## REPRODUCIBILITY

**Commits.**

| Commit | Content |
|---|---|
| 47246f1 | protocol + code |
| 4ea3c68 | amendment 1 |
| fce5e57 | amendment 2 |
| 7cb1d90 | amendment 3 |
| — | figures / chain 4 commit |
| this commit | results |

Chains: `experiments/fnto_lc52/lc_chain{,2,3,4}.sh` (`setsid`; status files `~/rest_fnto_r1/launcher/lc_chain*.status`).

**Seeds.**
- Configurations: train 100000–102999, val 200000–200199, test 300000–300119, combo 500000–500029, scale 600000–600029, traj 400000–400005, bench 700000+.
- Training: 5201.
- Gathers: target 10,000,019 + seed; oracles 1000 / 2000 + seed; C0 3000 + seed; C0p 4000 + seed; REF 11 / 12 + seed.

**Checkpoint.** `results/fnto_lc52/train/ckpt_best.pt` (step 40,000; server only).

**Wall time.**

| Stage | Time |
|---|---|
| Data | 183 s |
| Training | 3,420 s |
| Evaluation | 685 s |
| Local diagnostic | 120 s |
| Benchmark | 23–27 s per run |

**Outputs.**
- `results/fnto_lc52/`: stageA, data meta / kselect, train / train.json, eval / eval.json and images.npz, local / local.json and errmaps.npz, bench, bench_idle, bench_clean (fully idle, primary), smoke/bench, report.json, tests.
- `results/evaluation/52/`:
  - 01_coupled_renders.png;
  - 02_coupled_error_maps.png (one shared absolute sqrt scale);
  - 03_single_flight_errors.png;
  - manifest.json.

**Operational notes.**
- `pkill -f` / `pgrep -f` patterns matched the SSH shell's own command line twice; processes were then stopped by PID.
- A Windows-written chain script had CRLF line endings and was fixed.

No later batch is started or recommended.
