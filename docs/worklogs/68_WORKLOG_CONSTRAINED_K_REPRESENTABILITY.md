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
