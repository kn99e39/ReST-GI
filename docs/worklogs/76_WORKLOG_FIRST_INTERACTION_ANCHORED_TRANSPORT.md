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

**2026-11-07, amendment 5 (before any candidate evidence; execution only; made after seeing a dev cost measurement).**
- **Trigger.** The dev smoke measured the FIA 64-light apply at 3.07 s against 0.086 s for A75, which is 35.7x and over the 10x computational gate. That figure reflects the implementation, not the representation: the code built a dense anchor x neighbour x lobe scatter on every apply and transposed the sparse matrices on every call.
- **Change.** The runtime now pre-assembles CSR operators and applies sparse products. The result is identical to the old path (rel 5.7e-7; test76 asserts this against the kept reference implementation). On the 5080 the apply drops from 2.90 s to 0.25 s, against 0.063 s for the rank-16 baseline.
- **Unchanged.** The gate threshold.
- **Disclosure.** The decision followed the dev measurement and is labelled as such.

**2026-11-07, dev smoke complete and evidence start.**
- **Dev smoke.** After amendments 2-5 and an export path fix (dev only: the export read the dev root's missing Stage 1 record), the dev smoke ran every stage on LabServer63: tests 7/7, probe, 30-step fits, Stage 3, 640 dev packets, Stage 4 (INCONCLUSIVE on the dev subset, as expected), export. No metric was used.
- **Evidence chain.** Started at 17:40 UTC, code commit at HEAD on the server (`code_commit.txt`): Stage 4 population frozen, tests 7/7, two-lane probe 0.99 s/step and 4.75 GB reserved per run (fits about 5.5 h). The two fits started at 17:43 UTC.

**2026-11-07, evidence chain complete.**
- Fits: 2 × 20,000 steps on two lanes, 19,834 s each, both converged by the WL68 rule. J went 0.993 → 0.670 / 0.659.
- Stage 3 took 1,936 s and gave **INSUFFICIENT FIT**. By protocol, no Stage 4 reference was generated.
- Export completed. The OOM watchdog was never triggered during the evidence chain.

## Final report (2026-11-07)

**Protocol.** `fia76_v1` Part A (680d4df) and Part B (edc7ab4), plus five amendments:
- **Amendment 1** is post hoc and was a user decision.
- **Amendment 5** was made before any evidence, but only after a dev cost measurement had been seen.
- **Amendments 2–4** are pre-evidence execution fixes. Each was found by the dev smoke.

**Server code commit:** 7c6950d.

**What did not happen:**
- No candidate evidence was rerun, reselected or tuned.
- No checkpoint was selected.
- No loss or schedule change was made after evidence.

**Values** below are worst-channel jackknife upper 95 % bounds of pooled errors (eval68 estimators) unless stated otherwise.

### Implementation Fact

**First-interaction factorization** (Stage 0, isotropic homogeneous medium):

K_scat(i,a→j,o) = (1 − β(i,a)) · E_{x∼p1}[Q(x→j,o)],  with  Q = albedo · (E1 + M)

- p1 is the first-collision distribution.
- E1 is the exact single-scatter escape.
- Q is source-independent. This follows from the Markov property of the walk and the isotropic phase function (g = 0).
- The factorization does **not** carry over to anisotropic phase functions, surface/BSDF scattering, Fresnel boundaries or heterogeneous media.

**Candidate FIA** (`fia76.py`):

K_scat = albedo · SS + albedo² · Σ_k W(i,a→k) · R_θ(k→j,o)

- **SS** is the exact single-scatter quadrature.
- **W** is the second-collision mass on anchors placed at 4 depths below every port.
- **SS and W** are fixed-seed geometry-only quadratures with q = 1024. They are rescaled per cell to 1 − β_B, so they are deployable in the same way as B.
- **R_θ** is the learned, source-independent response of an isotropic emitter at the anchor. It has:
  - escape mass and tail share;
  - a local softmax over the 512-nearest-distance neighbourhood plus a coarse tail, each with an analytic falloff prior;
  - **3 ring lobes**, i.e. azimuthally uniform outgoing distributions.
- **Features** are invariant under the slab symmetries, so the model is equivariant by construction.
- **Size:** 42,565 parameters (AR 157,024; A75 276,880).
- **Physics:** passive, positive, linear, with rebuild identity within 2e-7 relative, on rest and pt0.

### Measurement

**Stage 1 (physical gate; 36 packets, 32M photons each).**
- Decomposition vs the archived first-exit references: 36/36 pass (coarse z² 0.90–1.13, relative L1 ≤ 0.23 %).
- First-collision probability: 36/36 pass.
- Forward negative control: rejected 4/4.
- Single-scatter quadrature vs MC: 35/36 pass. The failure is rest SR s70 bin 0 at z² 1.70.
- **Verdict by the frozen rule: FAIL.** Amendment 1 (post hoc, user decision) replaced the failed check with a fresh-seed re-test, which gave 0.94 / 1.11 / 0.76.

**Stage 3 (S16-R held-out replicas 4–7).** Pairs are written as two seeds.

| Group / metric | AR (rank 8) | A75 (rank 16) | FIA F76_7601 / 7602 | G0 oracle | G2 oracle |
|---|---|---|---|---|---|
| TRAIN_NEW_rest con_coarse | 0.270 | 0.131 | 0.688 / 0.687 | 0.270 | 0.129 |
| TRAIN_NEW_rest abs_coarse | 0.110 | 0.083 | 0.142 / 0.140 | 0.111 | 0.084 |
| TRAIN_NEW_rest abs_native | 0.132 | 0.117 | 0.332 / 0.333 | 0.130 | 0.115 |
| TRAIN_KEEP_rest con_coarse | 0.19 | 0.093 | 0.745 / 0.746 | 0.188 | 0.088 |
| TRAIN_KEEP_rest abs_coarse | 0.061 | 0.044 | 0.131 / 0.128 | 0.064 | 0.044 |
| TRAIN_KEEP_rest abs_native | 0.095 | 0.087 | 0.325 / 0.324 | 0.093 | 0.084 |

**WL73 development groups** (FIA vs the AR / A75 range):

| Group | FIA con_coarse | FIA abs_coarse | Baseline con_coarse (AR / A75) |
|---|---|---|---|
| BND_FAR | 0.96–0.99 | 0.39–0.42 | 0.87–0.95 |
| BND_NEAR | 1.00–1.06 | — | 0.40–0.43 |
| INT_ALL | 0.68 | — | 0.15–0.37 |

DEF71 and DEF73 fail (abs_coarse ≥ 1.3).

**Gate classification: INSUFFICIENT FIT.**
- Every listed group, metric and seed exceeds 1.25 × AR.
- Validity passes.
- The computational sub-gate passes: state 1.47 GB, apply at 4.0 × A75.

**Cost** (LabServer63 RTX 3080 Ti, same process):

| Quantity | AR | A75 | FIA |
|---|---|---|---|
| Full geometry rebuild | 0.014 s | 0.015 s | 75.0 s (quadrature 74.3 s, response and assembly 0.7 s) |
| State | — | 29.5 M floats | 1.47 GB (W 37 MB, SS 448 MB, local R 855 MB, tail 133 MB) |
| 64-light apply | 0.054 s | 0.086 s | 0.349 s |
| 1-light apply | — | — | 0.0039 s |
| Training | 4,156 s / run | — | 19,834 s / run (both two lanes) |

Stage 1 references cost 1,153 s in total. No Stage 4 references were generated.

### Observation

- **Plateau.** Training J plateaued by step 2,000. The 2k / 6k / 10k / final checkpoints give TRAIN_NEW contrast 0.691 / 0.690 / 0.689 / 0.688.
- **Development drift.** BND_FAR contrast still drifts up during training, 0.76 → 0.96, as in WL71A / WL75.
- **Uniform error.** Errors are nearly uniform across every training, development, interior and boundary source: contrast 0.69–0.75, native about 0.32. This is not the source-position-dependent failure of W+UV / RSK.
- **Post hoc diagnostic** (labelled, non-verdict; `diag/posthoc76.py` on F76_7601, 16 S16-R rest sources, archived replica means):
  - The total collided energy is right: ratio 1.002.
  - The per-incoming-bin energy profile is right (relative L2 0.023), so the geometric first-collision budget works.
  - The exact single scatter carries 25–36 % of the reference collided energy.
  - The error of the learned multiple-scatter part is native 0.47, fine16 0.31, coarse 0.19.
  - Projecting the reference multiple-scatter part onto ring-uniform outgoing distributions per receiver leaves a residual of **0.45** of its native norm. The replica-noise floor of the same statistic is 0.002.
  - So **almost all of the native error (0.47) is the ring-lobe restriction**: multiply scattered light leaves each receiver with strong azimuthal structure, and the predeclared ring lobes, chosen for exact equivariance, cannot represent it.
- **Contrast attribution not isolated.** The contrast error (coarse incoming-angle dependence, 0.7) was not separated by this diagnostic. Candidate contributors:
  - lateral snapping of the second collision to port-column anchors (2.5 mm), which drops the incidence-azimuth shift;
  - the same angular restriction.

### Interpretation

1. **Is first-interaction factorization physically and numerically valid?**
   - The decomposition reproduces the established first-exit references in all 36 packets, over interior, edge, corner, side, vertical-edge, rest, pt0 and te0 sources, and the forward control shows the test has power.
   - The single frozen-rule failure was a single-scatter self-consistency statistic. Fresh seeds put it inside the band, but proceeding past it was a post hoc decision.
   - Numerically, the deployable SS / W quadratures conserve the β budget exactly and are equivariant up to quadrature noise.
2. **Can the source-independent response fit the training data?** No. With 42.6k parameters, ring lobes and port-column anchors it does not even reach the rank-8 learned baseline on supervised sources. Training contrast is about 2.5–4 × AR; native error is about 3.5 × AR.
3. **Does it generalize to independent boundary sources?** Not measured. The fit gate stopped the batch before Stage 4. Development groups are worse than both baselines.
4. **Does it outperform the preserved rank-8 / rank-16 learned K?** No, on every training and development group measured.
5. **Are there improvements in native / fine16 space?** No improvement in any space. Native is the worst (about 0.32), which matches the ring-lobe floor.
6. **What are the actual compute and supervision costs?**
   - No privileged labels were used.
   - The cost is dominated by the geometry rebuild: 75 s against 0.015 s for W+UV, because every geometry change needs a full SS / W quadrature.
   - Apply is 4 × A75 and state is about 1.5 GB. Training is 4.8 × slower.
   - Stage 1 validation references cost 19 min.
8. **What remains unresolved?**
   - Whether a first-interaction factorization with an azimuth-resolving outgoing model and sub-port anchors would fit and transfer. That would be a different representation, not tested here.
   - Whether the exact-incidence mechanism (correct here: incident profile error 0.023) helps transfer once the post-collision response is expressive enough.
   - The cost of the per-geometry quadrature.

Question 7 is answered separately because it is the mechanism question.

7. **Is the failure mechanism fundamentally changed compared with W+UV and RSK?** Yes.
   - **W+UV / RSK** fit supervised sources well but fail to transfer. Their error depends on source context.
   - **FIA** gets incidence-dependent energy right everywhere, but its learned post-collision response is too restricted in outgoing angle. Its error is uniform and representational, and its native error is set by the ring-lobe floor.
   - **Where the bottleneck sits.** The factorization moves the bottleneck from source transfer to post-collision angular expressivity. That expressivity was limited by my design choice (ring lobes for exact equivariance), not by the physics.

### Unresolved Question

- **Design confound.** The candidate bundled several choices into one predeclared design: ring lobes, 4-depth port-column anchors, q = 1024, and 42.6k parameters. The batch cannot say which besides the ring-lobe floor limits the fit, and contrast was not attributed. Per the brief, no second representation or repair was attempted.
- **Stage 1 rests on a post hoc decision.** The physical-gate pass depends on amendment 1, which was decided after seeing Stage 1.

### Architecture Verdict

**INSUFFICIENT FIT** at the predeclared Stage 3 gate.
- **What holds.** The first-interaction factorization is physically valid for this isotropic homogeneous medium (Stage 1, with the post hoc amendment noted). The deployable geometric first-collision terms conserve the incident budget exactly.
- **What fails.** The tested learned source-independent emitter response cannot fit the supervised physical responses. Its dominant native limit is the azimuthally uniform outgoing representation (residual floor 0.45 vs noise 0.002).
- **What remains unmeasured.** Independent boundary transfer, by protocol.
- **Status.** Not adopted, and no successor started. The preserved WL73 / WL75 baselines and all earlier evidence are unchanged. The living documents are not changed because there is no architecture or contract decision.

### Reproducibility

- **Code:** `experiments/fnto_fia76/`.
  - Part A at 680d4df; Part B at edc7ab4.
  - Amendments 2–5: 812a9b5, 67c8d21, dffe449, 74805d8.
  - Server code 7c6950d.
- **Protocol:** `protocol/fia76_v1.json` (Parts A and B, amendments 1–5).
- **Evidence root:** `LabServer63:~/rest_fia76`.
  - Stage 1: `stage1/physics.json` (sha256 e74fc360…).
  - Diagnostics: `diag_s70_nonevidence.json` (5e11ab0f…) and `diag_posthoc76.json`.
  - Also: `ports76.json`, `runs/`, `stage3/`, `probe/`, `logs/` (including the GPU sampler and `watchdog76.log`).
- **Dev root:** `~/rest_fia76_dev` (no metric used).
- **Repository copies:** `results/fnto_fia76/` (final/, Stage 1 / Stage 3 analyses, diagnostics, ports) and `results/evaluation/76/` (figure + manifest).
- **Baselines:**
  - AR_5901 / 5902 in `~/rest_src73/runs` (sha256 aefabe33… / a5dee943…).
  - A75_5901 / 5902 copied to `~/rest_rank75/runs` (c0022742… / 358c3881…).
  - G0_r0 / G2_r0 in `~/rest_ato74/bases`.
- **Environment:** LabServer63, RTX 3080 Ti 12 GB, torch 2.3.1, float32 with TF32 off; oracle FISTA in float64.
- **Seeds:**
  - Fits: 7601 / 7602.
  - Stage 1: 761 / 762xxxxxx.
  - Quadrature: 762000000 + 1e7 × geometry index + 16 × port.
  - Stage 4 (unused): 766 / 767xxxxxx.
