# Worklog — WL94C: ECRO Panel-Mediated Query-Level Supervision Smoke (2026-11-14)

## Origin and scope

- **Brief.** "WL94C — Panel-Mediated Query-Level Supervision Smoke" (user, 2026-11-14). It is a bounded continuation of WL94B.
- **Batch class.** This is a suffix of WL94 under the preserved ECRO contract; only the per-query supervision weights change. The brief names it WL94C and the owner decides the name.
- **Hypothesis H-Q.** The existing ECRO better preserves sharp-panel transport if physically identified panel-mediated receiver/patch queries receive enough training emphasis, beyond the effect of the same emphasis on matched non-panel queries.
- **Preserved:**
  - the architecture (735,664 parameters) and the material buffer;
  - the OWN objective terms and targets;
  - the optimizer, schedule and 2,000 steps;
  - the WL94A C item list, its 50/50 assignment, cached targets, item order and init;
  - the frozen own checkpoints and all WL94 / WL94A / WL94B artifacts (read-only links);
  - the CK2 scene, references and masks;
  - the WL94 / WL94A / WL94B code, imported unchanged.
- **Historical outcomes unchanged.** WL94: H-M / H-A INCONCLUSIVE. WL94A: INCONCLUSIVE. WL94B: STOP_INVALID_INTERVENTION.
- **Code.** `q94.py` and `run94c.sh` in [experiments/ecro_q94c/](../../experiments/ecro_q94c/).
- **Protocol.** [protocol/ecro_q94c.json](../../experiments/ecro_q94c/protocol/ecro_q94c.json). Freeze commit `6d8b254`. The protocol sha256 prefix of the committed (LF) file is 006941ac6cce4741; the Windows CRLF working copy hashes differently.
- **Where.**
  - Server: LabServer63 `~/rest_x94c`, one detached chain, 14:54–15:06 UTC (11 min).
  - Local results: `results/ecro_q94c/`.
  - Figures: `results/evaluation/94C/`.

## Result in one line

The outcome is **INCONCLUSIVE** by the frozen mapping, because the seeds disagree.

| seed | finding |
|---|---|
| s8601 | D is better than C and E. Untextured sharp E_H is 0.044 vs C 0.084 / E 0.086, which is the frozen-own level 0.047. The panel-face global term is restored and F2 / MR are kept. |
| s8602 | D is worse than C and E (0.089 vs 0.067 / 0.066). The learned panel sender overshoots more. |

The matched control E equals C on both seeds, so there is no generic supervision effect. D's effect is panel-specific in both seeds, but its sign differs between them. With the exact panel sender substituted, D equals frozen own on both seeds.

## Pre-freeze feasibility evidence (no training; non-training instrumentation only)

**Query and labels.**
- A query is one (item, lighting, receiver patch): the 64 bins × 3 channels of that patch's table. There are 5,080 queries per item.
- The WL94B audit aggregates give each query's loss-normalised e2_p / e2_o / e3_p / e3_o mass.
- The four parts sum to the per-query total to 9.8e-8. WL94B showed that they sum to the stored f to 4.7e-8, and that e2_p + e2_o equals f2 to 5.8e-8.
- The model does not expose e2_p or e3_p. The weights multiply the **existing** sum, e2 and e3 terms of a query; no isolated panel component is predicted or supervised.

**Rules.**
- **D:** box receiver queries with pm_q = (e2_p + e3_p) / total ≥ 0.5 get weight κ = 8. The other queries of the same item get (Q − 8k)/(Q − k), so the item mean weight is exactly 1.
- **E:** the same number of box queries with pm_q < 0.10, chosen per seed from the same material half by optimal one-to-one matching (Hungarian). The matching covariates are:
  - the standardised log initial own error, which controls the hard-example effect;
  - the log target exitance;
  - box A vs B.
- E gets the identical κ rule.
- κ = 8 came from analytic accounting: κ = 4 gives 1.8× box panel-mediated exposure (below the 2× contract) and κ = 8 gives 2.9×. One value was used; there was no sweep.

**Feasibility refinements before the freeze (disclosed).**
1. Run 1 used greedy same-item matching and failed G_MATCH: SMD log err0 was 0.12 / 0.17 and log exitance 0.21 / 0.23.
2. Run 2 used Hungarian same-item matching with a box penalty and failed exitance balance (0.49 / 0.54). Zero-target box queries, such as faces resting on the floor, entered as log(1e-30) and inflated the exitance SD to 19.5.
3. A matching probe printed balance values only:
   - same-item pools with positive-target candidates stay at 0.13–0.19;
   - the same-half pool with a box penalty gives 0.06 / 0.05.
4. That variant was adopted. TAU, TAU_E, κ, the D rule and every gate threshold are unchanged.

**Feasibility result (re-checked identically in the evidence chain: MANIFEST PASS).**

| gate / test | value |
|---|---|
| G_POP | 974 D queries in 59 items (2.3 % of 41,472 box queries) |
| G_REL | selection agreement with the independent realization 0.914 |
| G_EXPO_D | box panel-mediated loss-mass exposure 2.91× C; D-query weighted share 8.0×; all panel-mediated mass 1.25× |
| G_EXPO_E | box panel-mediated 1.03×; E-query weighted share 8.0× |
| G_MATCH | SMD log err0 0.059 / 0.066 and log exitance 0.054 / 0.054 (s8601 / s8602); mean pm D 0.646 vs E 0.016; box-A share 0.235 vs 0.239 / 0.238 |
| T_BUDGET | item mean weight 1 (4e-16); emphasised query-visits D / E 20,323 / 20,264 (s8601) and 20,225 / 20,291 (s8602); min weight 0.923 |
| T_OBJ_ID | weighted objective with w = 1 equals c94.objective (8.3e-8) |
| T_REPRO | own / C tables recomputed = stored; checkpoints, metric replays — PASS |

**Dev chain.** `~/rest_x94cdev`, 60 steps. It ran end to end. Its T_BUDGET was false only because 60 steps is too small a sample for the visit comparison. It printed a dev outcome label, which is not evidence; its outputs were not opened.

## Candidate training evidence

- Training: D and E × seeds 8601 / 8602. Each run is 2,000 steps, takes 190 s, peaks at 657 MiB, and has 0 non-finite steps.
- Initial states equal C's: `a85ddb42…` and `7a0f1580…`.
- The training log records the unweighted terms (figQ3).

### Sharp ROI, untextured, arm C0

Primary anchor: frozen own. tol = max(2 × noise bound, 0.005) ≈ 0.017.

| | own | C | D | E |
|---|---|---|---|---|
| s8601 E_H (signed) | 0.047 (−0.040) | 0.084 (−0.082) | **0.044 (−0.030)** | 0.086 (−0.083) |
| s8602 E_H (signed) | 0.042 (+0.028) | 0.067 (+0.058) | **0.089 (+0.086)** | 0.066 (+0.057) |
| s8601 with exact panel sender (C2_srcS1) | 0.031 | 0.076 | **0.033** | 0.077 |
| s8602 with exact panel sender (C2_srcS1) | 0.019 | 0.023 | **0.019** | 0.022 |

**Pairwise judgments.**
- s8601: D is better than C by 0.040 (> tol 0.017) and better than E by 0.042. E is the same as C.
- s8602: D is worse than C by 0.022 (> tol) and worse than E by 0.023. E is the same as C.
- D is at the own level (SHARP_OK) on s8601 only.

### Components on the panel-facing box face (rect 6)

**Panel-sender single flight vs e2_p.** The reference half noise is 0.32.

| | own | C | D | E |
|---|---|---|---|---|
| s8601 rel L1 (signed) | 0.61 (−0.58) | 0.53 (−0.48) | 0.50 (−0.31) | 0.53 (−0.48) |
| s8602 rel L1 (signed) | 0.76 (+0.32) | 1.02 (+0.67) | 1.25 (+0.96) | 1.02 (+0.67) |

- The sender does not improve vs C on either seed by the rule.
- D's signed sender bias moves upward on both seeds, by +0.17 and +0.30.

**Global term G vs the e3 reference.** The reference half noise is 0.027.

| | own | C | D | E |
|---|---|---|---|---|
| s8601 rel L1 (signed) | 0.20 (−0.19) | 0.42 (−0.42) | **0.22 (−0.19)** | 0.43 (−0.43) |
| s8602 rel L1 (signed) | 0.15 (−0.13) | 0.092 (−0.071) | 0.067 (−0.011) | 0.090 (−0.066) |

- s8601: global is improved vs C and better than E.
- s8602: the ratio is 0.73, but the difference of 0.025 is below 2 × noise, so it is not improved by the rule.
- Other faces of box A change little: G rel L1 0.053–0.095.

### Textured sharp ROI and checkerboard response

**Textured ROI_T E_H (C0) per variant.**

| | own | C | D | E |
|---|---|---|---|---|
| s8601 | 0.062–0.106 | 0.059–0.086 | **0.034–0.051** | 0.061–0.089 |
| s8602 | 0.030–0.122 | 0.051–0.078 | 0.055–0.114 | 0.051–0.076 |

- On s8601, D is lowest on every variant.
- On s8602, D is worse than C on every variant.

**Image texture-response slope / corr in ROI_T.**
- s8601: D 0.79–0.98 / 0.88–0.95; C 0.61–0.82; own 0.17–0.41.
- s8602: D 0.81–1.13; C 0.93–1.21.
- Contrast alone is not counted as better GI.

**CK2 box-A amplitude (MR).**
- C, D and E are all within [0.85, 1.15] with corr ≥ 0.95.
- For D the amplitude is 0.98–1.03 on s8601 and 0.95–1.02 on s8602.

### Material retention

| | s8601 | s8602 | limit |
|---|---|---|---|
| F2 V_orig, D | 0.0810 | 0.0843 | 0.0856 / 0.0883 |
| F2 V_orig, E | 0.0809 | 0.0843 | 0.0856 / 0.0883 |
| MR, D and E (table and V_ck population; population amplitude 1.05 / 1.08) | pass | pass | — |

The V_ck held-out checker-layout rel L1 of D and E equals C's (B 0.109 / 0.118).

## Observation

- The emphasis is panel-specific in effect. On both seeds E is indistinguishable from C in every sharp, component and material metric, while D differs strongly from both. So the D change is not a generic box-receiver or hard-example weighting effect.
- The direction of D's effect follows each seed's prior state:
  - On s8601, C under-predicts the panel-face global term (−0.42) and the sharp reflection is too dark. D restores the global term to the own level and the image to own-level error.
  - On s8602, C's global term is already near correct (−0.07) and the learned sender already overshoots (+0.67). D moves the global bias to −0.01 but raises the sender overshoot to +0.96, so the image becomes brighter and worse.
- With the exact panel sender substituted, D reaches the frozen-own error on **both** seeds (0.033 vs 0.031; 0.019 vs 0.019), while C and E stay at 0.076 / 0.077 on s8601. The non-sender part of D (global + receiver) therefore reaches own fidelity on both seeds. The seed-dependent residual lies in the learned panel sender.
- The sender metric has a large reference half noise (0.32 per entry), so sender changes are not resolved by the predeclared rule, although the signed shifts are consistent in direction (upward on both seeds).

## Interpretation (separating the levels)

- **Physical correctness.** The path labels exactly partition the targets (≤ 1e-7). The weights change only query emphasis (T_OBJ_ID 8e-8 at w = 1; item budgets exact). The references and baselines are reproduced (T_REPRO). No invalid comparison was found.
- **Observed model behaviour.**
  - Emphasising panel-dominated box queries changes the panel-mediated transport the model learns, and does so more than equivalent non-panel emphasis does.
  - It restores the panel-face global term where the global term had collapsed (s8601), and brings the receiver / global part to the own level on both seeds (shown by the exact-sender control).
  - It does not correct the learned panel sender, and it can increase the sender's overshoot (s8602).
  - Integrated sharp rendering improves on one seed and worsens on the other.
- **Architecture inference.**
  - None about representability: this is a 2,000-step smoke.
  - What the evidence supports is narrower: receiver-query emphasis acts on the global / receiver side, and the remaining sharp failure behaves like an unresolved sender problem. The panel's once-scattered response is supervised only through the joint e2 target at receivers.

## Unresolved questions

- Why the learned panel sender moves upward under D on both seeds is not attributed. Whether the global and single terms trade off on the panel face (the S + G sum is supervised) is not isolated.
- The sender metric's resolution: the 0.32 per-entry half noise of the e2_p reference on the panel face limits component-level conclusions.
- Whether the seed-dependent sign reflects the initialization's sender state or optimisation noise is open: there are two seeds and a short training.

## Architecture Verdict

**INCONCLUSIVE (seed disagreement).**
- s8601 meets every PANEL_SPECIFIC_IMPROVEMENT clause.
- s8602 has D worse than C and E.
- GENERIC_SUPERVISION_EFFECT is excluded on both seeds (E = C).
- Nothing is adopted. Historical verdicts are unchanged. No successor is started; the decision returns to the owner.

**Answers to the brief's questions.**

| question | answer |
|---|---|
| Was panel-mediated query supervision increased meaningfully? | **Yes.** Box panel-mediated loss-mass exposure 2.9× C and panel-dominated query weight 8×, with the budget unchanged (pre-freeze verified, re-checked in the chain). |
| Was the result panel-specific relative to the negative control? | **In effect yes on both seeds** (E = C everywhere, D ≠ E beyond tolerance). In direction: better than E on s8601 and worse than E on s8602, so MIXED by the mapping. |
| Did panel-sender fidelity improve? | **No** on either seed by the rule. The signed bias moves upward on both: s8601 −0.48 → −0.31, s8602 +0.67 → +0.96. |
| Did panel-mediated global transport improve? | **MIXED.** On s8601 it improved strongly (rel L1 0.42 → 0.22, the own level 0.20). On s8602 the signed bias improved −0.071 → −0.011, but the rel L1 change is not beyond noise. |
| Did integrated sharp rendering improve? | **MIXED.** s8601: improved to the own level (untextured 0.044 vs own 0.047; textured best on all variants). s8602: worse (0.089 vs C 0.067, own 0.042). |
| Were original-material fidelity and MR retained? | **Yes**, for D and E on both seeds. |
| Were findings consistent across seeds? | **No** for the integrated outcome. **Yes** for the facts that E = C, that D's receiver / global part reaches the own level under the exact sender, and that D's sender bias shifts upward. |
| What remains unexplained by training exposure? | The learned panel sender, i.e. the panel's once-scattered response delivered to the box. It is not corrected by receiver-side emphasis, its error sign depends on the seed, and with it replaced by the exact sender D matches frozen own on both seeds. |

## Reproducibility

- **Commits and protocol.**
  - Freeze `6d8b254`; protocol sha256 prefix 006941ac6cce4741 (committed LF file).
  - Results are committed with this worklog.
- **Server roots.**
  - Feasibility runs: `~/rest_x94cfeas` (3 manifest runs + 1 matching probe; no training).
  - Dev chain: `~/rest_x94cdev`.
  - Evidence chain: `~/rest_x94c`.
- **Checkpoints.**
  - own_s8601 `739e7eb5…`, own_s8602 `cee9a09a…`.
  - Init state sha256: `a85ddb42…` (s8601), `7a0f1580…` (s8602).
  - Final state sha256: D `141ebff2…` / `e68df71b…`, E `c965152e…` / `81ca4bc3…`.
- **Query manifests.** On the server: `results/manifest/W_s860x.npz` (per-item weight maps and D / E selections). Locally: `results/ecro_q94c/manifest.json` (counts, balance, accounting, gates).
- **Seeds and order.** Training seeds 8601 / 8602; item order rng(seed + 94), the same as C. Audit labels from WL94B (target seeds 940000000 + 16·idx + light at 256 spp; independent 943000000 + … at 64 spp).
- **Budgets.** 2,000 steps per run; 975,360 queries per step; item mean weight 1.
- **References.** CK2 fields, images and masks from WL94 (unchanged), with arm renders at 86600 + r, 2 × 4,096 spp.
- **Cost.**
  - 11 min wall in total.
  - Training 190 s per run, 4 runs in parallel.
  - GPU mean 63 % (74 % of samples > 50 %).
- **Outputs.**
  - `results/ecro_q94c/`: q_s860x.json, final_q.json, manifest.json, repro.json, models/*.json, pop/*.json, chain.status, gpu.txt.
  - `results/evaluation/94C/`, with a sha256 manifest:
    - figQ1 summary;
    - figQ2 textured / checkerboard;
    - figQ3 training;
    - figQ4 images per variant (REF / own / C / D / E / D with exact sender, both seeds, shared absolute scales).
