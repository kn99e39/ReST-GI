# Worklog — First-Interaction Anchored Spatial Transport Feasibility and Transfer (2026-11-06)

## Origin and authorisation

- **Request.** The user pasted the WL76 brief, asked whether it should run on 63, and then said 「어어 진행해」. The batch therefore runs on LabServer63 (the main server). The local exception was for WL75 only.
- **Locations.** Code is in `experiments/fnto_fia76/`, the protocol in `protocol/fia76_v1.json`, and the evidence root is `LabServer63:~/rest_fia76`.
- **Not authorised:**
  - production K/C changes;
  - new materials, glossy transport or coupled GI;
  - source coverage changes;
  - loss tuning or capacity sweeps;
  - per-port tables;
  - held-out selection;
  - adoption or a successor batch.

**Delay note.** About 30 minutes passed between the user's go-ahead and the first visible progress, all on the Stage 0 design and none reported. The user rightly called this out (「왜 30분 넘게 가만히 있었냐」); progress is now reported stage by stage.

## Stage 0: physical factorization contract (Part A of the protocol)

**Medium** (`p53_fixtures`): homogeneous, σ_t = 1000 m⁻¹ (mean free path 1 mm), albedo (0.999, 0.99, 0.9), isotropic phase (g = 0), index-matched boundary.

**Cell sources.** Unit flux, uniform on the 2.5 mm port, cosine-weighted inside the incoming bin (16 bins, concentric mapping). Exits are tallied by port and 64 outgoing bins.

**Decomposition.** Consider an entering photon at x0 travelling along d, with chord t to the boundary.
- It leaves without colliding with probability exp(−σt); this is the existing ballistic term B.
- Otherwise it collides first with probability c = 1 − exp(−σt), at a truncated-exponential depth on [0, t). Hence 1 − β(i,a) = E[c] exactly.
- At the collision the weight is multiplied by the albedo, and the new direction is drawn isotropically, independent of d.

This gives

K_scat(i,a → j,o) = (1 − β) · E_{x∼p1(·|i,a)}[ Q(x → j,o) ],  with  Q = albedo · (E1 + M)

- E1 is single-scatter escape from x: geometry-only and exact.
- M covers scattering orders ≥ 2.
- Q depends only on x and the current geometry. This follows from the Markov property of the walk and the isotropic phase.

**Where the factorization does not carry over:**
- an anisotropic phase (Q would depend on the incident direction);
- surface/BSDF scattering (the redirection is frame- and direction-dependent);
- Fresnel boundaries;
- heterogeneous media.

**Energy.** Σ E1 ≤ 1, Σ Q ≤ albedo, Σ K_scat ≤ 1 − β. This is passive for any learned M bounded by 1 − Σ E1.

## Stage 1: physical correctness gate (frozen before the Stage 2 candidate design)

**Validation estimators** (validation only, never used at inference):
- `decomposed` forces the first collision, restarts isotropically, samples the next segment exactly, then runs the UNCHANGED `p53_tracer.walk`.
- `single_quadrature` is the expected-value E1 composition.
- A forward negative control keeps the restart direction equal to the incident direction.

**Comparison.** Against archived WL73/WL71 references (independent seeds): 9 sources × bins {5, 0, 10, 15}, 36 packets in total.

| Pose | Sources |
|---|---|
| rest | INT s60, TBE s64, TBC s66, SM s68, SR s70, SV s72 |
| te0 | INT s80 |
| pt0 | TBE s50, INT s0 |

**Rules:**
- energy |z| ≤ 4;
- coarse-cell mean z² between 0.5 and 1.5, over cells selected by the INDEPENDENT estimator's expected count ≥ 30;
- E[c] against 1 − β within |z| ≤ 4, using a binomial floor on the reference SE;
- leaks ≤ 1e-5 N;
- the forward control must be rejected in every control packet.

**Pre-freeze smoke** (local, non-evidence, 0.2M then 2M photons). The first version's statistic was wrong in three ways, all fixed before the freeze:
1. 0-hit cells have a zero variance estimate, which inflates z².
2. A zero replica SE when no ballistic exit occurs (expected β ≈ 6e-10) produced z = −27.
3. Selecting cells by the tested estimator's own counts inflated z² to about 2.

A self-comparison of two seeds of the same estimator gave z² = 1.0, which confirmed the variance formula. After the fixes, at 2M photons:
- total vs reference: z² 0.92–1.15;
- single quadrature vs MC: z² 0.58–1.52;
- forward control: rejected 4/4 (z² 2,600–9,100).

The evidence run uses 32M photons per packet.

## Stage 1 result (LabServer63, commit 680d4df, 32M photons per packet, 2026-11-06/07)

**Gate (frozen rule): FAIL**, because of one failure out of 108 checks plus 4 controls.
- **Decomposed estimator vs archived references: 36/36 pass.** Energy |z| ≤ 1.16, coarse z² 0.90–1.13, coarse relative L1 0.13–0.23 %.
  - rest: INT s60, TBE s64, TBC s66 (corner), SM s68, SR s70, SV s72.
  - te0 INT s80; pt0 TBE s50 and INT s0.
  - Bins 5 / 0 / 10 / 15.
- **First-collision probability E[c] vs 1 − β (reference B16): 36/36 pass** (|z| ≤ 3.3).
- **Forward negative control: rejected 4/4** (z² 17,800–75,400), so the test has power.
- **Single-scatter quadrature vs the MC single tally:** 35/36 pass. The failure is **rest_s70 (SR) bin 0, z² 1.703** (> 1.5; energy z 0.09). Across all 36 packets this statistic ranged 0.60–1.70.

**Diagnostic (non-evidence).** The failing packet was re-run with three fresh seed pairs (`diag/diag_s70.py`) and gave z² 0.936 / 1.106 / 0.762 over 126 cells. The 1.70 looks like a statistical tail of a statistic with about 126 cells, but that is a post hoc reading.

**User decision.** The user answered the stop question with 「사후 amendment로 진행」.

**Amendment 1 (post hoc, labelled; protocol `amendments[0]`).** The single failing self-consistency check is replaced by the fresh-seed re-test. The original Stage 1 FAIL verdict is preserved and will be reported.

## Part B frozen (after Stage 1 and amendment 1, before any candidate evidence)

**Candidate FIA** (`fia76.py`):

K_scat = albedo · SS + albedo² · Σ_k W(i,a→k) · R_θ(k→j,o)

- SS and W are fixed-seed quadratures of geometry and σ_t only (deployable, like B). They are rescaled per cell so that ΣW + ΣSS = 1 − β_B.
  - W is anchored at the **second** collision point.
  - Anchors sit at 4 depths below every port.
- **R_θ** is the response of an isotropic interior emitter, and it is source-independent. It has:
  - escape mass ρ;
  - tail share τ;
  - a softmax over the ports within the 512-th-neighbour distance (ties included), plus a coarse 240-cell tail, with an analytic exp(−r/L)/(r² + 1 mm²) prior and MLP corrections;
  - 3 ring lobes.
- **Features** are invariant under the slab symmetries: free paths, distances, cosines, |n_z|. The model is therefore equivariant by construction, so there is no L_sym.
- **Size:** 42,565 parameters.

**Material difference from the baselines.**
- There is no learned source-port → T map (W+UV) and no shared incident A (RSK).
- Incident dependence is entirely geometric.

**Training.**
- Data: S16-R fit targets, rest and pt0.
- Objective: WL68 objective J, computed directly on dense predictions.
- Optimiser: Adam, cosine 3e-3 → 3e-5, clip 1.0.
- Budget: 20,000 full-batch steps in 4-source chunks.
- Two seeds (7601 / 7602); final iterate used.

**Stage 3 gates.**
- Fit: every TRAIN_NEW / TRAIN_KEEP group × {con_coarse, abs_coarse, abs_fine16} ≤ 1.25 × the worse AR seed.
- Runtime physics: rest and pt0.
- Computation: state ≤ 4 GiB, and 64-light apply ≤ 10 × A75.

**Stage 4 population** (`ports76.py`, geometry only, 26 slots + te1 footprint).
- Exclusions: 81 previously scored ports and all their exact images.
- Strata: TBE / TBC / SM / SR / SV / INT × near / far × a hi / lo coarse-cell sub-position pair.
- SV far has no admissible port.

**Primary decision.** BND_FAR con_coarse against the stronger preserved baseline, three-valued (0.8 and beyond the lower 95 % bound).

**Pre-freeze fixes found by the local tests** (non-evidence):
- Budget rescaling: independent quadrature noise of β and E[c] could break passivity after training.
- Neighbour ties included.
- Receiver face id replaced by |n_z|. The face id broke r90 equivariance: I1 was 0.029 against a 0.005 noise level.

The local tests now pass 7/7.

**Baselines.** A75_5901/5902 and their checkpoints were copied to `LabServer63:~/rest_rank75/runs` with matching sha256.

**2026-11-07, amendment 2 (before any candidate evidence; execution only).** The dev smoke on LabServer63 ran out of memory in Stage 3 while building the full-geometry FIA runtime at q = 1024, because the per-port quadrature rows were kept uncoalesced. The fix coalesces each quadrature immediately; the operator is identical. A local full rest build now peaks at 3.36 GB with a 1.3 GB state and takes 49 s, and the tests pass 7/7. The dev probe on 63 measured about 1.0 s per step on two lanes, which puts the 20k-step fits at roughly 5.7 h.

**Execution notes (2026-11-07).**
- **5-hour stall.** After Stage 1, I put the stop decision to the user as a blocking question and ran nothing else until it was answered. The session ended and LabServer63 sat idle for about 5 hours. The Part B dev smoke did not depend on the answer and could have run in that time.
- **User feedback.** The user raised this directly: 「네가 내 의견 묻는답시고 5시간동안 아무 것도 안하고 있었던」. The lesson is recorded in memory.
- **Watchdog.** `~/rest_fia76/watchdog76.sh` (execution only) relaunches `run76.sh` after a CUDA out-of-memory failure, such as another project taking the GPU. It waits for a free GPU first, makes at most 3 attempts, and leaves any other failure alone.

**2026-11-07, amendment 3 (before any candidate evidence; execution only).** The dev-smoke Stage 3 ran out of memory again, this time because the physics rebuild-identity check held two full runtimes at once. The check now keeps one runtime resident at a time, and its tolerance is the relative 1e-6 of test76 rather than exact equality, since GPU index_add order is not bitwise reproducible. The watchdog was stopped during the fix.

**2026-11-07, amendment 4 (before any candidate evidence; execution only).** The dev-smoke Stage 3 ran out of memory a third time, in `eval76.score`, because every prediction (47 MB each, about 100 of them) stayed on the GPU. Predictions are now kept on the CPU.
