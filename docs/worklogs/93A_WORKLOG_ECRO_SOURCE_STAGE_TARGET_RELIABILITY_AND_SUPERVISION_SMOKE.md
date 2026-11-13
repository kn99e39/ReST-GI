# Worklog — ECRO Source-Stage Target Reliability and Supervision Smoke Test (WL93A) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93A Source-Stage Target Reliability and Supervision Smoke Test" (user, 2026-11-13).
- **Batch class.** Suffix batch of WL93 under the AGENTS.md numbering policy. It resolves the target-reliability uncertainty that stopped WL93 before training; there is no new mechanism.
- **Question.** Can a reliably estimated physical source-stage target produce a measurable, independently validated improvement in ECRO's learned local response, under a small, controlled tied-factor training experiment?
- **Not in scope.** No generalization claim, no decision on untied factors, no adoption, no full-scale run, no successor.
- **Where.**
  - Branch `Selective_Recompute`, LabServer63.
  - Code `experiments/ecro_stage93a/` (reuses `ecro_stage93/model93.py`, `train93.py`, `prep93.py` unchanged).
  - Results `results/ecro_stage93a/`, `results/evaluation/93A/`.
  - Server root `~/rest_x93a`, reading `~/rest_ecro86`, `~/rest_x92`, `~/rest_x92a`, `~/rest_x92b`, `~/rest_x93`.

## Frozen WL93 history (preserved)

- WL93 ended **STOPPED-INCONCLUSIVE / STOP_TARGET_NOT_IDENTIFIABLE** at its identifiability preflight (frozen commit `c30c797`, closing commit `56be753`).
  - Validation: encoding-through-ER vs q 0.508 against the bound 0.485 / 0.483.
  - E2: pass (0.362).
- WL93 Amendment 1 (stop-path analysis crash fix) keeps its provenance and its **pending-approval** status.
- No WL93 file is modified. WL93's verdict and gate are not re-judged here. H_TARGET below is an assessment only.

## WL93A protocol (`experiments/ecro_stage93a/protocol/ecro_stage93a.json`)

**Invariants.**
- Unchanged: ECRO generator; frozen ER materials and path semantics; the deployed 16-sample source encoding; local rank 16; widths; the geometry interface; tied source / receiver factors; the global branch frozen to WL92A own_s<seed>; WL92A checkpoints and split; WL86 thresholds.
- No runtime oracle.
- **What changes:** only offline target estimation / uncertainty accounting, and source-stage supervision in one arm.

**Target q.** `quad87.qtable` with an unchanged estimator: P stratified points per patch × Mq one-sample MIS directions per point on ER patches; Lambert exact per point. Only the budget varies, never the quantity.

| Budget | P | Mq | Note | Probe s / geometry (4 lights) |
|---|---|---|---|---|
| B0 | 16 | 1,024 | WL93 / WL87 budget | 0.39 |
| B1 | 16 | 4,096 | directional × 4 | 0.41 |
| B2 | 64 | 1,024 | spatial × 4 | 0.81 |
| B3 | 64 | 4,096 | | 0.88 |
| B4 | 256 | 1,024 | spatial × 16 | 2.42 |
| B5 | 256 | 4,096 | evaluation reference | 2.74 |

- **Axes.** The spatial axis (P) and the directional axis (Mq) are varied separately.
- **Realizations.** Four independent realizations per budget and configuration.
- **Seeds.** Evaluation reference 93,100,000 + 1,000 cfg + 40 budget + 4 r + lighting. Training targets 93,500,000 + 4 idx + l. Training-validation targets 93,600,000 + 4 idx + l. The ranges are pairwise disjoint and disjoint from WL93 and WL87 (tested).

**Configurations (deterministic, by position only).**
- **Training subset:** the WL92A training population sorted by name → every 7th → first 128.
- **Train-representative:** every 8th of the subset (16 × 4 lights).
- **Validation:** all 55 × 4, including WL93's difficult g00010 / g00039; nothing excluded.
- **Sharp:** E2 T0 / T1.

**Task A metrics.**
- **M(X):** pooled debiased relative L2 on panel patches, sqrt(Σ(X − qa)(X − qb) / Σ qa qb), with qa / qb the means of two disjoint pairs of the four B5 realizations. It is unbiased for ‖X − q‖² / ‖q‖² when X is independent of the reference.
- **Uncertainty:** pairing uncertainty U (max − min over the 3 pairings) and a 2,000-resample geometry bootstrap.
- **Secondary:** relative L1 vs the B5 mean (with its floor), the flux / directional split, panel peak-bin retention, per-geometry values, top-5 shares and the predeclared high-confidence subgroup (B5 per-geometry noise ≤ 0.10, always next to the full population).
- **Three quantities kept distinct:**
  1. independent target disagreement N_b (single-realization relative L2 from the unbiased across-realization variance);
  2. the frozen-ER-through-encoding discrepancy M(enc_q(s16)). This is an operator / encoding discrepancy and is **not** called an information-theoretic irreducible error;
  3. WL92A OWN's learned discrepancy M(Bq_own).

**Rules (fixed before evidence).**
- **H_TARGET.** SUPPORTED if, on the pooled validation panel, |relL1(enc vs WL93's stored q) − relL1(enc vs B5 mean)| ≥ 0.05 while the B5 mean's floor ≤ 0.02. NOT SUPPORTED if < 0.02. Otherwise INCONCLUSIVE.
- **Training budget.** The cheapest of B0–B4 with N_b ≤ 0.10 on the train-representative panel. Else B5 if N_B5 ≤ 0.20. Else STOP TARGET_UNRELIABLE.
- **TARGET_RELIABLE.** δ = 0.10 × mean_s M(Bq_own init) on the validation panel (WL93's STAGE effect size). RELIABLE iff a training budget exists AND U(Bq_own) ≤ δ/3 for both seeds. Otherwise STOP before training.
- **The 0.5 ratio.** WL93's 0.5-ratio gate is neither reused nor replaced by a more permissive threshold.

**Task B (smoke).**
- **Arms.** T0 (WL93 OWN objective) and TQ (+ the WL93 source-stage loss, w = 1, den_q unchanged). Both use `model93.Local(own, 'tied')`.
- **Shared between arms.** Same start (WL92A own_s<seed>), same frozen global term, the 128-geometry subset, data order rng(seed + 93), 2,000 steps, lr 3e-4 → 1e-5 cosine (warmup 200), clip 10, identical parameter count.
- **Targets.** q is one realization at the training budget. Validation is monitored with independent training-validation targets. Evaluation is against the independent B5 reference.
- **Seeds.** 8601 / 8602.

**Criteria.**
- dM_s = M(TQ_s) − M(T0_s) on the validation panel; SE from a paired geometry bootstrap; S = pairing spread.
- **LEARNING_SIGNAL:** RELIABLE and, for both seeds, dM_s ≤ −max(δ/2, 2 SE, S); TQ validation whole and single-vs-L2 relative L1 ≤ 1.05 × T0; no non-finite step.
- **NO_SIGNAL:** RELIABLE, stable, and the threshold is met for neither seed.
- **INCONCLUSIVE:** anything else.
- **Verdict mapping:** FEASIBLE_FOR_FULL_SCALE_EVALUATION / NO_SIGNAL_AT_SMOKE_SCALE / TARGET_UNRELIABLE / INCONCLUSIVE. These are feasibility assessments only.

**Renders (diagnostic only).**
- E2 T0 / T1, E1 T1 LSPOT, E5k995 T0 / T1.
- T0 / TQ / own92a × seeds; 2 × 4,096 spp, seeds 86600 / 86601.
- E_H against the existing references.
- No superiority claim.

**Tests.**
- Arms: T_INIT (T0 / TQ state dicts identical, = WL92A own ≤ 1e-5), T_PARAMS, T_GLOBAL.
- Target: T_QGEN (B0 + WL87 seed reproduces the stored E2_T1 q; every budget runs), T_SEEDS.
- Data: T_SPLIT (train subset vs validation compared by **geometry content hash**; see below).
- T_DET.

## Dev chain and full-scale probes (before the freeze; not evidence)

**Dev chain** (`~/rest_x93adev`, `--dev`: Mq 64, 24 / 3 / 6 configurations, 100 steps, 64 spp). Every stage ran: configurations, references, stage tables, Task A, targets, tests, training, evaluation, render tables, renders, analysis, visuals.

One defect was found and fixed:
- **Problem.** T_SPLIT compared train and validation by file name. The `train/` and `val/` directories reuse the names g00000…, so different geometries were reported as overlapping and the test failed.
- **Fix.** The test now compares geometry content hashes. Afterwards every test passes.
- **Dev numbers.** Not inspected as evidence: Mq 64 on 6 validation geometries is not the protocol.

**Full-scale probes.**
- **qtable cost per budget.** On one training geometry (timing only), see the table above.
- **Chain probe** (6 configurations, timing / GPU only):

| Stage | Time | GPU | Peak memory |
|---|---|---|---|
| B0–B3 concurrently, plus 2 stage-table jobs | 33 s | 30 % | 8.1 GB |
| B4 alone | 55 s (9 s / configuration) | 17 % | 2.7 GB |
| B5 alone | 64 s | 27 % | 9.4 GB |

- **Why P = 256 is CPU-bound.** Each P = 256 call is dominated by CPU-side estimator work: per-call valid-solid-angle evaluation of sphere points, which depends on the sampled points. That part cannot be cached without changing the estimator.
- **Chosen concurrency.**
  - B0–B3 run concurrently with the stage tables.
  - B4 runs in 3 shards (2.7 GB each).
  - B5 runs alone (≈ 13 min). Two B5 processes would exceed the 12 GB card.
  - Target generation runs 2 concurrent jobs, or 1 if the training budget is B5.
- **Training.** Taken from the WL93 probe: 4 concurrent processes give GPU 82 %, so 2,000 steps take ≈ 4 min per run. With 128 geometries the RAM is small.
- **Renders.** 60 renders at ≈ 8 s each, pool 5.
