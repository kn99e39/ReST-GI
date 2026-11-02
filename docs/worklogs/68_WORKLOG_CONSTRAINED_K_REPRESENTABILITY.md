# Worklog — Constrained K Representability, Track A (2026-11-01)

## Authorisation and scope

On 2026-11-01 the user issued a pasted brief, "TRACK A — CONSTRAINED K REPRESENTABILITY, Independent Architecture Experiment".

**Question.** Can the existing positive/passive WL59 P intrinsic-K family constructively realize qualified collided directional transport on controlled physical probes? And what evidence separates representation limits from fitting or generalization failure?

**Allowed:**

- isolated fitting code;
- controlled optimization of experimental copies of the same family;
- bounds;
- independent held-out references;
- tests and reporting.

**Not allowed:**

- rank, width or parameter changes;
- a new module or a change of meaning of A, O, B or K;
- relaxed passivity;
- a production objective;
- retuning WL59, WL64 or WL65;
- checkpoint modification;
- F/S changes;
- coupled GI;
- a successor.

The brief mentions a parallel Track B. This work uses its own roots: `experiments/fnto_repr68/`, `~/rest_repr68*` on the servers, and `results/fnto_repr68/`.

- **Starting HEAD:** 418b1f8. WL68 was free.
- **Execution server:** LabServer63, the main server under the user's 2026-11-01 rule.
- **Inputs:** the WL59/WL66 inputs live on LabServer69. They were exported there, hash-verified and relayed through the PC (no 63↔69 SSH).
- **Design:** [`docs/plans/CONSTRAINED_K_REPRESENTABILITY68_V1.md`](../plans/CONSTRAINED_K_REPRESENTABILITY68_V1.md).
- **Protocol:** `experiments/fnto_repr68/protocol/repr68_v1.json`.

## Prospective design (before evidence)

### Implemented family, read from code

I derived the family from `r57_model.Transport.coefficients/apply` and `fnto_persist59/runtime.py`, not from prose. A one-hot probe response is `d Σ_r A_r ⊗ T_r ⊗ O_r`. Its components:

- **d = 1−β:** geometry-only.
- **A:** column-stochastic over 8 mixtures, global.
- **O:** a probability distribution over 64 outgoing bins per mixture, global.
- **T ≥ 0:** 128-nearest-neighbour W plus rank-16 U·V, with `Σ_j T = sigmoid(·) < 1`. All of T comes from geometry features through the network.

Each probe row is therefore a convex combination of 8 nonnegative space×direction rank-1 fields. They share global outgoing lobes, and their spatial masses are below 1.

The WL66 floors relax this in three ways:

- signed responses (relaxed rank-8);
- free outgoing structure (frozen A);
- signed spatial coefficients (frozen A+O).

None of them includes the geometry-conditioned network.

With 4 probe sources per geometry and rank 16, the network's response closure on the probe population lies in a coefficient class, FCoef: global simplex A/O and per-source capped nonnegative T. Comparing FCoef with network fits isolates geometry conditioning.

### Controls (brief Stage 3 → implementation)

| Brief | WL68 |
|---|---|
| A. Frozen WL59 model | P_5901 and P_5902, scored on identical inputs |
| B. WL66 relaxed rank bound | Relaxed L7 per population (low-biased), plus WL66 values reproduced |
| C. Frozen incoming/outgoing bounds | WL66 signed F_A and PAOc for every A/O. **New admissible counterparts** at frozen A,O, all exact convex FISTA: nonnegative capped coefficients under contrast, free nonnegative outgoing blocks, and J |
| D. Fully constrained family fitted to physical probes | Network fits by parameter group: A, O, AO, net, all. Inits: the two WL59 seeds, plus one scratch init for all |
| E. Same-family synthetic teacher recovery | T1, T1r (other seed), T2 (other seed with physically fitted FCoef angular bases) and Tcoef |
| Intermediate | FCoef with 4 restarts. Labelled held-out angular-transfer oracle |

**Held-out populations.** The ports were chosen by a geometry-only rule (see the protocol). The groups are:

- H1: new ports on rest and pt0;
- H2: cf3 at the fitting ports;
- H3: cf3 at the new ports.

The held-out geometry is cf3, the WL59 diagnostic pose farthest from pt0. The references were traced on LabServer63 with independent seeds.

The fitting stage can read only `fit_targets.pt` (the mean of WL66 replicas 0–3). A guarded loader and a test that spies on `torch.load` enforce this. Fit-population scores use replicas 4–7 only.

### Pre-evidence engineering facts (disclosed)

- **Throughput rewrite.** The native-space objective term was rewritten with an exact Gram identity, so the 47M-element prediction is never materialised. A float64 test certifies equality with the direct objective in value and gradient. Time per step fell from 0.13 to 0.09 s. A `bmm` layout variant was slower and was reverted. GPU utilisation during fitting was 98% median.
- **Non-evidence timings.** Dev timings came from **synthetic teacher targets only**.
- **Dev pipeline smoke.** The pipeline was smoke-run end to end in a separate dev root (`~/rest_repr68_dev`) with these settings:
  - 30-step fits and 20-iteration convex solves on the physical fitting targets;
  - 1M-photon held-out packets with K=4;
  - 5-iteration oracle.

  Its metric values were not inspected. Only completion and file structure were checked.
- **Dev bugs fixed before the protocol commit:**
  - FISTA grad mode;
  - a test tensor shape;
  - a dtype in the contrast term;
  - flattening of native Grams in the evaluator. The first dev WL66-reproduction attempt crashed on this.
- **Geometry identity on LabServer63.** Node, pair, prior, xyz, area, parent and graph features are bitwise equal to the archived WL59/WL57 files for rest, pt0 and cf3. The freshly built q=256 ballistic quadrature β differs from the archived one by a relative L2 of about 0.10 (max |Δβ| 0.075). This is a Monte Carlo re-sampling of the geometry-only quadrature. As in WL59/WL66, the model uses the archived geometry, and the difference is reported, not corrected.
- **Rest signature.** The fresh rest signature string differs from the archived one even though every feature is bitwise equal. pt0 and cf3 signatures match.
- **Dev identity gates (repeated in the evidence chain):**
  - WL66 pooled coarse statistics reproduced to ≤1e-6 relative;
  - cross-machine packets statistically identical (|energy z| ≤ 0.6, coarse mean z² 0.83–1.41), not bitwise;
  - preflight: 1 worker × 32M (1074 packets/h) beat 2 workers × 16M (917/h).
- The dev smoke's evaluation log printed the decision label of the 30-step smoke (D). It carries no information about the 20,000-step fits, and nothing was changed in response. The fixes made during the dev smoke were limited to the robustness items listed above (plus the convergence window for very short runs, the oracle replica split when K < 6, and a missing t(2) quantile).

## Execution

**Commits:**

- 7f9c28a: input staging;
- 0ffc700: protocol, code and tests frozen before evidence;
- 9b437b2: execution-order fix. The pre phase failed its first attempt because two tests need `fit_targets.pt`, which the chain prepared only in the main phase. Preparation now runs first. No evidence existed at that point, and the failed log is preserved as `logs_failed_tests_attempt1`;
- a2a802c: budget fixed before generation.

**Pre-phase gates on LabServer63** (`results/fnto_repr68/pre/`):

- tests 10/10;
- fit targets prepared (sha 3c21b619…);
- geometry features bitwise equal for rest, pt0 and cf3;
- held-out ports 2148, 1025, 2707 and 2422;
- WL66 pooled statistics reproduced (max relative difference ~1e-6);
- cross-machine packets statistically identical to LabServer69 (|energy z| ≤ 0.6), not bitwise.

**Preflight.** Packets per hour: 1057 for 1 worker × 32M, 993 for 2 workers × 16M. By the rule this gives K=8, 1 worker, 2048 packets, predicted 1.94 h.

**Procedural slip, corrected before any evidence.** A first main-phase launch started before the budget commit existed: `results/` is git-ignored, so `git add` failed and the chained launch still ran. I killed it during its GPU-idle wait, with zero packets generated. It is recorded in `aborted_main_launch.txt` and `logs/gpu_sampler_aborted_main_launch.log`. The budget was then committed (`git add -f`) and main was relaunched with `budget_commit.txt` = a2a802c.

**Amendment 1 (55c36f5, execution only).** Held-out generation stopped on a CUDA OOM after 73 of 2048 packets. No foreign GPU process was present.

- **Cause.** On the 12 GB RTX 3080 Ti, 32M-chunk packets at slot 4 reached 10.9–11.0 GB peak allocation. The preflight had timed slot 0 only and measured 9.48 GB.
- **Change.** Generation resumed with 1 worker × 16M action chunk (`heldout/execution_amendment.json`). The 2 × 16M configuration was not used: each worker peaked at 4.8 GB in the preflight, so two workers would approach the limit under the same tail growth.
- **Unchanged.** The estimator, per-packet seeds, K=8, populations and all rules. The 73 completed 32M-chunk packets passed every gate and were kept; each packet records its chunk.
- **Cost.** Packets now take about 5.4 s each, against about 3.4 s at 32M, so the remaining generation is about 3 h instead of the predicted 1.94 h.
- **Logs.** The failed log is `logs/generate_w0_attempt1_oom.log`.

### Amendment 2 — fit throughput (user's separate instruction, execution only)

**This change was made on the user's separate instruction**, given during execution on 2026-11-02 after a server status check: analyse the low GPU utilisation of the fits and fix it. It is not part of the brief's design and not a scientific choice. Protocol entry: `amendments[2]`, code commit 968f77a.

**Diagnosis.** Profiling ran in a non-evidence root on the idle LabServer69 (`~/rest_repr68_prof`, identical fit-target tensors; scripts in `experiments/fnto_repr68/amend2/`). Two different causes were found:

- **Net-group runs (A, O frozen): CPU kernel-launch bound.** These issue about 1,287 small kernels per step. On LabServer63 they took 0.054 s/step at about 48% median GPU utilisation; on LabServer69 they took 0.031 s/step. This was the 45–58% utilisation seen at the status check. No net-group run remained when the fix was ready, so this cause was not addressed.
- **Runs that train A or O (AO, all, coef, teachers): wasted GPU work, not idle GPU.** These ran at about 98% utilisation. However, the O-gradient of the `S ×_o O` contraction in `sq_error` was executed by cuBLAS as a ~245k-long reduction (12.7 ms per call), plus layout copies. Together these took about 60% of a step.

**Change.** `Target` keeps channel-major copies of the native and fine16 targets. `so_contract` computes the same contraction as a wide batched GEMM. The objective, data, optimizer, schedules, steps, inits, runs and all rules are unchanged.

**Equivalence, measured before deployment** (`results/fnto_repr68/amend2/val68*.json`):

- The objective J is bitwise equal.
- Gradient error against a float64 direct reference: 5.7e-7 / 6.5e-7 for net/all / coef with the new path, against 5.3e-7 / 6.7e-7 for the original. The new and original gradients differ by 4e-7.
- Final J after 1,000 compressed-schedule steps:
  - net/all: original 0.038461 / 0.038408 / 0.038415 (three runs), new 0.038401;
  - coef: equal to 6 significant digits.

  The original code itself is not bitwise reproducible, because CUDA `index_add` is nondeterministic.
- Tests: 10/10.

**Deployment.** The two files were replaced atomically on LabServer63 at 2026-11-01 16:56:38 UTC (`amendment2_deploy.txt`); the pre-amendment files are kept in `code_before_amendment2/`.

- **Original code:** net_A, net_O, net_AO, net_net (both seeds) and net_all_5901, which was already running.
- **New code:** net_all_5902, net_all_scratch6801, all coef runs and all teachers. This includes the **primary candidate net_all_5902**.
- **Unaffected:** the convex floors, evaluation and export, which do not call `sq_error`.

Per-run `seconds` are therefore labelled by implementation and are not compared across this boundary.

**Measured improvement on LabServer63** (20,000 steps; GPU utilisation is the median of 15 s `nvidia-smi` samples):

| Runs | Implementation | Wall time per run | GPU utilisation (median) |
|---|---|---|---|
| net_all_5901 | original | 2,080 s | 98% |
| net_all_5902, net_all_scratch6801 | amended | 1,134 s, 1,151 s (**1.82×** faster) | 80%, 76% |
| coef ×4 | amended | 787–901 s | 65% |
| teachers T1/T1r/T2 (all group), Tcoef | amended | 1,163–1,172 s, 884 s | 72% |

- **Measured speed-up:** 1.82× for the same run type on the same GPU. Before deployment, LabServer69 measured 2.2× (all), 2.6× (AO) and 2.9× (coef) per step.
- **Time saved:** about 2.9 h over the remaining runs. This is an estimate, extrapolated from the per-step ratios; no original-code coef or teacher run was timed on LabServer63. The chain finished at 20:58 UTC.
- **Reported utilisation fell.** In the amended runs it dropped from 98% to 65–80%. The GPU work per step shrank more than the CPU launch time, so launch overhead again leaves gaps. Throughput, not the utilisation figure, is what improved. Removing the remaining launch overhead (e.g. CUDA graphs) was not attempted.

**Housekeeping.** The chain's GPU sampler loop survived `terminate()` and kept writing after `done`. It was stopped at 2026-11-02 01:38 UTC. Samples after 20:58 UTC carry no information. A copy of the log is in `results/fnto_repr68/amend2/`.

## Final report (2026-11-02)

The chain on LabServer63 completed on 2026-11-01 at 20:58 UTC with every stage passed. The evaluation was fixed by the protocol before results, and nothing was rerun or reselected afterwards.

- **Exports:** `results/fnto_repr68/final/`.
- **Reviewer copies:** `results/evaluation/68/`, with a provenance manifest.
- **Metrics:** each metric is the upper 95% bound of the jackknifed relative L2 error, worst over the R, G and B channels.
- **Triples:** "a / f / c" means coarse collided / fine16 / coarse contrast. The WL59 criteria are 0.10 / 0.15 / 0.25.
- **Population labels:** "fit" scores fitting replicas 4–7 only. "Held-out" uses independent seeds and new references.

### Implementation fact

- The evaluated family is the code-level WL59 P family. It is described in "Implemented family, read from code" above; rank, width, the 157,024 parameters, and the meaning of A, O, B and K are unchanged.
- Every fitted model is an experimental copy. Production checkpoints were not touched.
- Passivity held for every fitted network:
  - zero/linearity/cache tests pass at all three geometries;
  - the maximum column sum is ≤ 0.9997 for the all-group fits;
  - for net_net it is 1.0000002, float32 round-off of the sigmoid normalisation, inside the WL59 physics tolerance.
- Teacher runs did not meet the last-decile convergence rule, which the adequacy rule does not require. All other runs did.
- **Primary candidate** (predeclared: lowest final training J among the physical all-group fits): net_all_5902, J = 0.01667. It ran on the amendment-2 implementation; see Amendment 2.
- **Reference precision:** all 24 held-out and fit matrices are qualified. The minimum per-matrix signal/SE is 988.

### Measurement

**Fit population and held-out groups** (upper 95%, worst channel, a / f / c):

| Candidate | fit rest | fit pt0 | H1 rest new ports | H1 pt0 new ports | H2 cf3 fit ports | H3 cf3 new ports |
|---|---|---|---|---|---|---|
| frozen P_5901 | .104/.262/.930 | .105/.260/.899 | .153/.259/.972 | .153/.259/.969 | .104/.262/.930 | .154/.263/.971 |
| frozen P_5902 | .104/.255/.899 | .105/.252/.872 | .154/.254/.951 | .154/.253/.948 | .104/.255/.899 | .155/.258/.950 |
| net A, 5901 | .097/.247/.790 | .101/.242/.772 | .148/.247/.870 | .148/.247/.866 | .097/.247/.790 | .151/.251/.872 |
| net A, 5902 | .097/.249/.782 | .102/.241/.768 | .149/.249/.883 | .150/.248/.875 | .097/.249/.782 | .151/.253/.883 |
| net O, 5901 | .109/.267/.906 | .110/.265/.878 | .158/.264/.959 | .158/.264/.956 | .109/.267/.906 | .159/.268/.958 |
| net O, 5902 | .106/.257/.878 | .107/.253/.852 | .156/.256/.938 | .155/.255/.934 | .106/.257/.878 | .156/.259/.937 |
| net AO, 5901 | .094/.249/.610 | .098/.249/.604 | .142/.246/.705 | .141/.247/.710 | .094/.249/.611 | .144/.249/.709 |
| net AO, 5902 | .090/.251/.583 | .111/.254/.573 | .142/.251/.738 | .143/.254/.737 | .090/.251/.583 | .143/.254/.737 |
| net net, 5901 | .218/.245/.737 | .197/.235/.714 | .524/.697/.987 | .523/.701/.982 | .216/.244/.737 | .536/.689/.983 |
| net net, 5902 | .206/.246/.758 | .189/.238/.733 | .566/.628/1.048 | .565/.629/1.056 | .200/.245/.759 | .568/.632/1.046 |
| net all, 5901 | .041/.088/.155 | .052/.090/.153 | .337/.411/1.215 | .333/.408/1.239 | .042/.088/.156 | .340/.409/1.191 |
| **net all, 5902 (primary)** | **.039/.087/.133** | **.049/.090/.135** | .559/.609/.942 | .546/.600/.963 | **.040/.087/.134** | .562/.606/.947 |
| net all, scratch6801 | .043/.085/.156 | .052/.091/.154 | .423/.505/1.355 | .428/.508/1.441 | .043/.086/.157 | .423/.499/1.347 |
| FCoef, 4 restarts | .040–.044/.086–.115/.123–.142 | .050–.053/.088–.116/.127–.141 | — | — | — | — |
| convex J at frozen A,O | .207–.220/.244–.245/.736–.757 | .189–.197/.235–.237/.713–.731 | — | — | — | — |

The coefficient class and the convex floors have no geometry map, so their held-out values are undefined (—).

**Decision inputs:**

- **S_fit:** passes (rest and pt0).
- **S_gen:** fails. H1 rest, H1 pt0 and H3 fail; H2 passes.
- **Teacher controls:** all four are informative and all pass. The final upper errors over rest and pt0 (a / f / c) are:
  - T1: ≤ .0005 / .0006 / .0012;
  - T1r: ≤ .0012 / .0011 / .0030;
  - T2: ≤ .020 / .020 / .026, with initial con .93;
  - Tcoef: ≤ .0006 / .0006 / .0010, with initial con 1.05.

  T1 and T1r start at con .31–.34.
- **Optimizer adequate:** yes. **Primary converged:** yes.

**Admissible convex floors at the frozen WL59 A and O** (exact FISTA, float64; fit population, coarse contrast):

| Floor | Upper 95%, worst channel |
|---|---|
| Nonnegative capped T under J | .736 / .757 (rest, seeds 5901 / 5902); matches the net-group fits to ≤ .02 |
| conAO (contrast-only, nonnegative) | .59–.74 per channel |
| conA_freeout (frozen A, free nonnegative outgoing blocks) | .54–.67 per channel |

The WL66 signed floors for the same A and O are F_A .52–.56 and PAOc .60–.65; WL66's values were reproduced to ~1e-6. The relaxed L7 bound is .023–.027 for every population.

**Basis floors for fitted A and O.** The signed floors (F_A / PAOc, upper, worst channel) are:

| A, O from | fit | new ports |
|---|---|---|
| net all 5902 | .052 / .125 | .058–.059 / .141–.145 |
| coef_5901 | .051 / .120 | .056–.057 / .134–.137 |

**Labelled angular-transfer oracle.** It uses held-out references: the best admissible spatial T is fitted for fixed A and O on held-out replica half 0 and scored on half 1. Results (a / f / c):

| A, O from | Held-out groups |
|---|---|
| net all 5902 | .056–.058 / .086–.088 / .146–.150; H2 .039 / .086 / .131 |
| coef_5901 | .056–.058 / .087–.088 / .140–.143 |
| frozen 5901 / 5902 | .21–.25 / .24–.25 / .74–.78 |

**Signed deformation.** These are medians over matrices and channels of the coarse change error, with its correlation and slope:

| Candidate | pt0−rest at fitting ports (fitted data) | cf3−rest at fit ports | cf3−rest at new ports | pt0−rest at new ports |
|---|---|---|---|---|
| frozen | ρ .37–.38, corr .94, slope .73–.74 | ρ .80–.88, corr .41–.49, slope .12–.18 | corr .21–.29 | corr .39–.40, slope .45 |
| all-group fits | ρ .25–.29, corr .97, slope .93–.94 | ρ 1.63–1.70, corr −.22 to −.04 | corr ≈ 0 | corr ≈ 0, ρ 3.0–4.1 |

**Original WL59 extended fields** (historical; inspected in WL59–65, so not fresh). Coarse mean error and eligibility:

| Model | Coarse mean | Dynamic | Eligible |
|---|---|---|---|
| frozen WL59 | .062–.067 (static useful) | fails in B (error .41, corr .91) | no |
| net AO | .072–.075 | fails in B | no |
| all-group fits | .40–.85 | error 1.4–6.0, corr .16–.55 | no |

Directional pairs fail for every model.

**Cost (LabServer63):**

| Item | Measured |
|---|---|
| Held-out generation | ≈ 3 h for 2,048 packets × 32M photons (amendment 1) |
| Fits | 675–2,080 s each (amendment 2 labels in `runs.json`) |
| Convex floors | ≈ 7 min |
| Evaluation | 2,155 s; peak 2.99 GB |

### Observation

1. The family as fitted with all parameters free meets every WL59 quality criterion on the fitting population, at the 4 fitting ports per geometry. Two independent inits and one from-scratch init all do.
   - The coefficient class without geometry conditioning reaches the same level (FCoef .12–.14 contrast).
   - Same-family teachers starting at contrast .31–1.05 are recovered to ≤ .026.
2. With WL59's own angular bases frozen, no admissible spatial coefficient reaches the contrast criterion. The exact convex floor is .74–.76, and the net-group fits land on it. Freeing O alone does not change this (.85–.91); freeing A alone reduces it to .77–.79, and freeing A and O together to .57–.61.
3. At new source ports the fitted networks are far worse than at the fitting ports. Contrast is .94–1.44 and collided error is .33–.57. Collided error is worse than the frozen WL59 model (.15), while contrast is about the same (.95–.97). The three all-group fits differ substantially from one another at new ports (contrast .94 / 1.22 / 1.36–1.44) despite near-identical fit-population scores.
4. The fitted angular bases are not what fails at new ports. Their admissible oracle on the held-out references reaches .056–.058 / .086–.088 / .14–.15, inside every criterion, in every held-out group, at new ports and at cf3.
5. H2 (cf3 at the fitting ports) passes with values identical to fit-rest to three decimals. Yet the signed cf3−rest change at those ports is not predicted: correlation −.22 to −.04, against .41–.49 for the frozen model. The cf3 change at the fitting ports is therefore small relative to the response.
6. The all-parameter fits degrade the historical full-field predictions of the network from .062 to .40–.85 coarse mean error.

### Interpretation

Each statement is limited to this population: one homogeneous-medium geometry family, 3 poses, 8 ports and 16 incoming bins.

- **Constructive representability holds on the fitting population.** Members of the actual positive/passive family exist that realise the qualified collided directional transport at the WL59 criteria for 4 sources and 2 geometries. A constraint-violating fit is excluded by the passivity tests.
- **WL59's shortfall on these probes is on the angular side of its learned parameters.** Its global A/O bases admit no admissible fit (Observation 2), and the WL66 relaxed floors over-stated nothing here: the admissible floors at frozen A/O are worse than WL66's signed ones.
- **The generalisation failure is located in the geometry-to-spatial-coefficient map at unseen sources.** It is not in the angular bases (oracle), not in the optimizer (teachers) and not in passivity. Fitting 4 sources per geometry yields spatial maps that do not transfer to other sources or to deformation (corr ≈ 0), and that overwrite the network's previous whole-field behaviour.
- **The H2 pass is not evidence of deformation generalisation** (Observation 5).
- **Not shown:** whether any family member generalises across sources. The fit data constrain only 8 source rows, the teacher controls test recovery on the same rows, and no same-family generalisation teacher was run. A fitting failure, an identifiability limit of the 8-source data and a representational limit of the geometry map are therefore not separated for the held-out populations.

### Unresolved question

1. Does a member of the WL59 family with the fitted angular bases exist whose geometry-conditioned spatial map reproduces held-out sources and deformation? The oracle shows admissible coefficients exist per held-out source, but not whether the network can produce them from geometry.
2. Is the cross-source failure a consequence of supervising 8 source rows (identifiability), or of the map's capacity or conditioning?
3. Why do the three all-group fits disagree at new ports while agreeing on the fitting population?
4. How does the fitted family behave on WL59's F/S execution? This is unmeasured and not authorised.

### Architecture verdict

- **Outcome B** by the predeclared mapping: S_fit passes, S_gen fails, the optimizer is adequate and the primary candidate converged.
- **Family status:** the WL59 P intrinsic-K family **needs a specific additional falsification: independent generalisation**, i.e. cross-source and deformation prediction of the geometry-conditioned spatial map.
  - It is not "viable": generalisation failed.
  - It is not "unsupported": constructive representability on the qualified probes was demonstrated.
- **Not claimed:** adoption, a successor, a coupled-GI or F/S claim, or impossibility.

**Completion question: can the existing positive/passive WL59 family constructively represent qualified collided directional transport?**

- **On the controlled fitting probes: yes**, with all parameters free and the family unchanged.
- **With WL59's learned angular bases: no.**
- **From geometry at unseen sources or poses: not shown.** The tested fits fail there, and the cause lies in the spatial geometry map, not the angular bases.

### Reproducibility

- **Commits:** 0ffc700 (protocol and code frozen), 9b437b2, a2a802c (budget), 55c36f5 (amendment 1), 968f77a (amendment 2), and this report's commit.
- **Inputs:** manifest sha256 2abf3ba6…; fit targets 3c21b619….
- **Server:** LabServer63, RTX 3080 Ti, torch 2.3.1+cu121, Mitsuba `cuda_ad_rgb` with the user-space OptiX library path.
- **Held-out references:** seeds `683000000 + g·100000 + slot·10000 + a·100 + rep`, K = 8.
- **Execution:**
  - pre phase: `chain68.py pre`;
  - budget commit;
  - main phase: `chain68.py main`, under the 3-sample GPU-idle wait.
- **Gates passed:** tests 10/10; WL66 reproduction ~1e-6; cross-machine identity |z| ≤ 0.6; geometry features bitwise.
- **Determinism:** fits are not bitwise reproducible (CUDA `index_add`); run-to-run spread is quantified in Amendment 2. Two implementations of the objective are present, labelled per run.
