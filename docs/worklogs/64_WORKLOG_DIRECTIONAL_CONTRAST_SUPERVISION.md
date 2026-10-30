# Worklog64 — Fixed-family directional-contrast supervision

## 2026-10-30 — Authorisation and prospective execution record

The user's explicit bounded brief authorises one objective intervention after completed WL63. Starting HEAD `bad10d5`, current branch `Selective_Recompute`; no worktree or architecture replacement. Read Architecture/research topic/roadmap, WL59 protocol/training/evaluation/provenance, WL60–62 attribution and WL63 tagged component evidence. [Frozen design](../plans/DIRECTIONAL_CONTRAST64_V1.md) and [protocol](../../experiments/fnto_direction64/protocol/direction_v1.json) own the prospective rules.

Implementation scope: original P model, parameters, B, input/output basis, geometry, F/S, labels, original objectives, seeds and optimizer budget preserved. A/B isolated under `~/rest_direction64/arms`; only B adds explicit directional-contrast supervision. Original sources and checkpoints are read-only. Calibration uses training values only and will be committed numerically before training. A fresh eight-pair source population is predeclared for evaluation-only on known pt0; Pair7 remains a historically inspected diagnostic.

Execution pending calibration/tests, final numeric freeze, independent-reference qualification, four controlled training runs, unchanged original evaluation/cost, and component/comparison reporting. No scientific outcome exists at this entry. No coupled solve or automatic successor is authorised by recording this plan.

### Pre-training implementation-test correction

Initial setup stopped before producing calibration or training: a focused identical-target test required bit-exact zero from two separate GPU atomic aggregations. Direct diagnostic measured 9.95e-14 GPU normalized residual versus exact0 on CPU float64. This is aggregation roundoff, not a changed objective. Repair the test to require CPU-double exact0 and GPU residual<=1e-10; retain the original failed setup log. No scientific loss, weight rule, threshold or budget changed.

### Numeric objective frozen before training

Training-only calibration and focused tests pass. Fixed coarse/fine16 eta = **0.0020228057947454018 / 0.02429031329894211**; lambda = **1.4493619017035815**, selected once by the prospectively committed rule. No candidate trained and no held-out model outcomes inspected. Exact compatible original rest r0 train tensor equality is verified; all original source/data/checkpoint identities recorded in [calibration.json](../../results/fnto_direction64/pretraining/calibration.json). Calibration implementation commit `7594e35`.

All8 train pairs have equal spatial flux and distinct angular fields. Pair-swap and normalization controls pass; baseline weight0 returns the original loss tensor, real-model gradient comparison max absolute difference9.47e-10 (GPU atomic roundoff). CPU-double exact-target loss0; GPU residual9.13e-14. Original physical controls pass: zero0, linearity5.41e-8, max column sum0.927172, cache error6.39e-8; parameter count157,024. No runtime source or architecture change. Initial failed zero-equality assertion remains in `setup_attempt0.log`. Numeric freeze and comparison source are committed before launching any training.
