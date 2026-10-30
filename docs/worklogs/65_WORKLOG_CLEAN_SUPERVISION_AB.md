# Worklog65 — Partial clean-supervision A/B (2026-10-30)

## Phase A — Authorisation and prospective reference preparation

The user's explicit brief authorises **only pre-training reference generation/qualification** before2026-10-31 14:00KST. This Worklog owns the whole eventual Clean Supervision experiment; completing Phase A does not complete that research experiment. Starting HEAD `002c625`, branch `Selective_Recompute`; no branch/worktree change. Read current living documents, WL59 model/training/reference protocol, WL60–61 supervision findings, WL63 physical component result and WL64 negative objective intervention. Do not repeat those completed diagnostics.

[Plan](../plans/CLEAN_SUPERVISION65_V1.md) and [fixed protocol](../../experiments/fnto_clean65/protocol/clean_v1.json) define208 original training actions,832 independent32M packets /26.624B photons, exact source/label identities, covariance and precision gates. All13 original training geometries and all16 original directional fields80–95 are retained. Phase B is **not authorised**: model157,024 parameters, original96-field sampling/partitions, action+.25delta objective, optimizer/6,000steps/checkpoint, B and F/S must remain unchanged in any separately authorised continuation.

Feasibility before generation: historical32M timings1.980–3.096s project32–43min packet time, conservative40–90min total. Prior GPU peak10.38GB; budget16GiB/24GB. Raw packets49.073GB plus labels/variance4.907GB and small reduced outputs; reserve60GiB, require100GiB free. Server read-only inventory shows idle GPU and~1.2TB available. Unlike WL61's2M chunks, this experiment fixes32M chunks prospectively. No physical photon or model execution has occurred at this entry. New samples use one field per action, so no field-mixture dilution of the proposed photon budget.

Preflight/tests and actual identities/resource estimates will be committed before generation. Sufficient precision cannot be guaranteed for weak differences: the unchanged original delta loss floor provides a separate uncertainty scale, while pure delta SNR remains explicitly reported. No field filtering, count extension, changed loss or automatic training is authorised.

### Phase-A preflight passed before any reference generation

Preflight/test source **c8a3648**. All7 focused CPU checks pass: exclusive-bin covariance/grouping, four-half variance/16, shared-rest cancellation, original weak-delta loss floor, observational accumulator preservation/restoration, all8 original pairs retained and832 distinct seeds. All96 train-field arrays exactly match original WL57 lineage. All13 raw legacy/current canonical geometry identities, derived features and parent correspondence pass; original label/physical source/checkpoint hashes match.

Available disk **1,319.25GB**; deadline about **20.10h** away at server preflight, versus conservative40–90min execution estimate. GPU had no external compute peer. Minimum RGB source-proposal ESS fraction across fixed fields is **0.971386**, or≥31.084M source-weight effective samples per32M half. This is before transport/absorption and is **not** exit-action ESS; physical first/second moments determine actual uncertainty. No cross-field mixture is used. The exact proposed208-action/26.624B-photon contract is feasible; no costed alternative or scope/budget substitution is needed.

[preflight.json](../../results/fnto_clean65/preflight.json), tests and preflight log are committed before generation. No physical reference photon or model execution preceded this evidence. The subsequent generation commit changes evidence/Worklog only; protocol/source bytes remain the preflight-verified version. Proceed only with fixed Phase A; Phase B remains unauthorised.

### User steering — event-only waiting

The user requested stopping conversational progress polling to avoid token waste. Leave the existing server generate→analyze→report chain untouched. Install a token-free server stage observer and local SSH→`codex queue` bridge for this existing thread. Notify only on stage completion or failure; no periodic agent turns/progress reports. Stage callbacks must return idle until a real completion/error needs action. No new agent or Phase B is created. The frozen scientific source directory remains unchanged; watcher helpers live outside it.

### Terminal Stage events received

Event-only callbacks confirm generation832/832 packets /26.624B photons, physical pass; analysis and reviewer report completed. Frozen readiness is **STOP_INSUFFICIENT_PRECISION**: all624 coarse/fine16 absolute cells and all312 coarse directional cells pass, but107/576 original-loss-scaled delta cells fail. No photons, fields, gates or labels are changed after this outcome. Final immutable provenance collection/download runs once in the background and queues this same thread only on terminal completion/error; no conversational polling. The complete Clean Supervision research experiment remains open and Phase B unauthorised.


## Phase A terminal report — STOP_INSUFFICIENT_PRECISION

Phase A is finished; **the full Clean Supervision A/B experiment is not complete**. No model has been trained/evaluated and no Phase B is authorised or queued. This status is a fixed-budget precision stop, not an invalid-reference, resource or model-performance verdict.

### Implementation Fact

Completed the exact proposed population: original train fields80–95, all8 pairs, rest plus all12 original training dents =**208 actions**;2 independent64M-photon replicas/action, each2 independent32M halves =**832 packets /26,624,000,000 photons**. Every packet passed the original physical checks and observational response/second-moment equality checks. No failed packet was replaced; no field, geometry, count, chunk, estimator, loss or model was changed.

All96 original train-field bytes and all13 original geometry/correspondence contracts match. Physical K/C, first-exit semantics, B/remaining budget,157,024-parameter P representation, original action+.25deformation loss, optimizer/6,000-step/final-checkpoint rule,96-field distribution/partitions and F/S remain unchanged. The full16-field mean/variance bank is stored separately on the server; original training files were never patched. Other80 fields remain original in the prospective A/B plan. No held-out Pair7, WL64 fresh fields, test label or model prediction selected or weighted these references. Historical timing/memory metadata were used for feasibility only.

### Measurement

These are Monte Carlo uncertainty norms perRGB at the declared spatial/angular aggregation, not model prediction errors or confidence intervals.

| Frozen readiness check | Observed | Gate | Result |
|---|---:|---:|---|
|Coarse absolute final-mean SE/action norm, maximum624 cells|0.003700 (0.370%)|≤0.01|624/624 pass|
|Fine16 absolute final-mean SE/action norm, maximum624 cells|0.014628 (1.463%)|≤0.05|624/624 pass|
|Coarse directional conservative replica SNR, minimum312 cells|10.9806|≥3|312/312 pass|
|Coarse directional final-mean SE/contrast norm, maximum|0.061926 (6.193%)|≤0.10|312/312 pass|
|Delta final-mean SE/original loss scale, maximum576 cells|**0.457260**|≤0.25|**469/576 pass;107 fail**|
|Pooled rest directional new mean SE/old single-label noise RGB|0.026059 /0.026047 /0.025058|≤0.5 each|RGB pass|

The compatible rest comparison gives about**38.4–39.9× lower pooled directional noise norm** for the new final mean than the original single-label estimate. It is a descriptive precision comparison from original r0/r1, not a bias proof or an independent-sample confidence interval.

High absolute precision and qualified directional contrasts do not imply resolved deformation differences. **516/576 pure delta cells have SNR<3** (R166/192, G166/192, B184/192). The predeclared loss-scale test allows weak/zero signals when uncertainty is sufficiently small relative to the original `sqrt(||delta||²+1e-4||rest||²)` denominator. It still fails107 cells (R32, G32, B43), all among those weak signals, involving original fields84–87. No failing field is removed. Fine16 directional diagnostics have minimum SNR3.4226 but maximum mean-SE/contrast0.20638; they are not claimed to meet the10% coarse contrast criterion.

All geometry/pair/RGB SNRs are in [directional_all.csv](../../results/evaluation/65/directional_all.csv); all field/geometry/RGB delta SNRs and original-loss-scaled uncertainty are in [deformation_all.csv](../../results/evaluation/65/deformation_all.csv). The table below retains every training dent; weak/failing counts have48 field/channel cells per dent. Pooled delta SNR does not mask unqualified individual cells.

| Geometry | Min coarse pair SNR | Pooled delta SNR R/G/B | Weak delta cells /48 | Loss-scale failures /48 | Max delta loss-scale SE |
|---|---:|---|---:|---:|---:|
|tr00|11.358|0.882 /0.858 /0.762|48|12|0.394133|
|tr01|11.658|1.847 /1.750 /0.961|44|4|0.358922|
|tr02|11.651|3.213 /3.092 /1.225|42|8|0.371294|
|tr03|11.473|1.245 /1.153 /0.737|46|12|0.457260|
|tr04|11.814|2.124 /1.982 /1.066|42|7|0.365072|
|tr05|11.373|3.892 /3.282 /1.144|43|12|0.400948|
|tr06|11.214|1.181 /1.128 /0.782|48|12|0.408696|
|tr07|12.127|2.996 /2.554 /0.980|44|12|0.419799|
|tr08|11.952|3.273 /3.098 /1.649|34|2|0.289937|
|tr09|10.981|1.370 /1.249 /0.767|46|12|0.438521|
|tr10|11.524|6.580 /6.018 /2.033|42|8|0.370693|
|tr11|11.603|21.020 /21.113 /10.585|37|6|0.366932|

Original-versus-new label discrepancies are retained without attributing unknown old deformed-label variance to systematic bias. Relative L2 min/median/max across the complete respective population:

| Aggregation/quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
|coarse/absolute|0.023117|0.030925|0.065350|
|fine16/absolute|0.063873|0.108182|0.146175|
|coarse/directional|0.045971|0.134120|0.517857|
|fine16/directional|0.096539|0.287400|1.698974|
|coarse/deformation|0.378923|28.440350|107.216119|

The very large pure-delta discrepancy ratios use a tiny, frequently unqualified physical delta denominator. They are descriptive, not z-scores, causal percentages, or evidence that a model would improve. Every weak row remains in [historical_discrepancy_all.csv](../../results/evaluation/65/historical_discrepancy_all.csv).

### Observation

The fixed budget yields precise absolute actions and a fully qualified coarse directional-training population, but does **not** satisfy the entire predeclared precision contract under the unchanged deformation objective. Every dent has at least one failing loss-scaled delta cell. Valid physical references and a lower-noise partial bank exist; the complete proposed unchanged-objective intervention is nevertheless **not READY** by its frozen rule.

There was no resource/identity/physical stop, no adaptive sampling, no post-result threshold change and no outcome-selected subset. Seven focused correctness tests pass. All source/label/checkpoint identities verified before generation were rehashed unchanged at final collection. The event-only watcher delivered completion events and exited; no conversational progress polling continued after the user's steering.

### Interpretation

**STOP_INSUFFICIENT_PRECISION** is justified by107 failed delta-loss precision cells, despite passing absolute/directional/rest-noise-gain gates. This is neither STOP_INVALID_REFERENCE nor STOP_RESOURCE_OR_SCOPE. READY_FOR_CONTROLLED_TRAINING would require every frozen readiness gate; it cannot be inferred from the312 passing directional cells alone.

This phase establishes that substantially less noisy directional action labels can be constructed within the original training distribution. It does not answer whether replacing them improves learned K. It does not falsify clean supervision or establish representation adequacy. The remaining80 fields are deliberately untouched and are not newly certified clean.

### Unresolved Question

How much the partial precision intervention improves independent learned directional fidelity remains entirely unmeasured. Weak deformation targets remain insufficient under the declared loss-scale uncertainty bound. Original deformed-label variance/bias, residual noise in the other80 fields, optimization and representation/generalization constraints remain confounded. Independent photon replicas share the inherited validated reference implementation; this is not an independent-renderer bias check.

Variance is formed from aggregated float64 first/second moments, before spatial/angular grouping variance. Four-half mean variance is the sum of half variances/16. Different fields/current-rest references use independent streams; delta variances add. The same rest label is reused across dents, so Cov(delta_g,delta_h)=Var(rest), and rest cancels in their difference. RGB are correlated and no iid-cell significance claim is made. Source-weight ESS is not exit-action ESS. No extra photons or automatic alternative are selected.

### Architecture Verdict

No architecture/physical contract, B, F/S, training objective, distribution or production change is adopted. **No training, frozen-model performance evaluation, checkpoint selection, coupled GI or successor has run.** All labels—including unqualified/weak differences—are retained; the bank manifest sets `qualified_for_future_controlled_intervention=false` and `training_authorized=false`.

Phase A is complete with a precision stop. The complete Clean Supervision A/B hypothesis remains **OPEN / NOT EXECUTED**, awaiting a later explicit user decision. There is no automatic Phase B, reset-time trigger, photon increase or field filtering.

### Reproducibility

- Prospective source/protocol **c8a3648**, measured identity/resource freeze and generation **866ce7d**. Provenance collector **574e8a1**, event-only watcher **c77c03f**, terminal collection callback **d5a98d5**. Frozen source hashes and31 original input/lineage/checkpoint identities all match at closure; [provenance.json](../../results/fnto_clean65/provenance.json). WL59–64 and living architecture documents are unchanged.
- Source seeds are `650000000 + pose_index*100000 + field_id*100 + replica*10 + half`:832 distinct streams,32M photons each, no cross-field mixture/budget dilution. Minimum perRGB source-proposal ESS fraction0.971386. All source-field and geometry hashes, legacy raw/canonical signatures and environment are in [preflight.json](../../results/fnto_clean65/preflight.json); all seeds/counts/stats/RNG and packet hashes are in [reference_complete.json](../../results/fnto_clean65/reference_complete.json).
- LabServer69 RTX3090 24GB, original Torch2.3.1+cu121/NumPy1.26.4 venv, TF32 off,8 Torch threads,32M action chunk. Generator stage2073.853s (34.56min), inner generation2051.120s, analysis stage65.768s, reviewer export3.005s:35.71min for these three stages. Packet wall times1.886–3.251s. Peak GPU allocated11,083,164,160B; analysis CPU high-water1,965,524KiB. Stage before/after GPU-peer lists are empty, and final GPU has no compute process. These are reference preparation costs, not learned inference or update performance.
- Actual raw packets12,270,810,240B; full mean-label plus variance bank1,226,969,432B; complete recorded artifacts13,583,941,402B before the archive. **Planning correction:** the initial storage arithmetic assumed15,360 ports instead of the verified original3,840 and overestimated raw/bank bytes by4×. Execution used the original3,840-port geometries and exact original tensor shapes throughout; no geometry/basis/sample reduction occurred. This was a conservative resource-estimate error, not a scientific substitution. Initial free space1,319GB remained ample.
- Raw packets, all13 full16-field mean banks and diagonal variance banks remain under `~/rest_clean65/{halves,labels,label_variance}` on LabServer69 with complete hashes. Local [reduced](../../results/fnto_clean65/reduced) arrays preserve all coarse packet moments; CSVs preserve every coarse/fine16 qualification and discrepancy row. No original target was overwritten.
- Terminal provenance collection completed2026-10-30 **18:34:04 KST**, before the2026-10-31 14:00KST reset. Final bundle SHA `d7c6595e897bc135d72a087c7de8ca140ab04e5b7343e28237229fb4c2f0d4d4`,82,458,046B. Bundle/path checks,42 downloaded provenance entries and reviewer manifest checks pass locally; visual QA passes. [Machine status](../../results/fnto_clean65/phaseA_status.json), [label bank manifest](../../results/fnto_clean65/labels_manifest.json), [review manifest](../../results/evaluation/65/manifest.json).

![Phase-A reference precision, all fixed geometries and fields](../../results/evaluation/65/reference_precision.png)

**Phase-A completion answer:** independently precise coarse directional labels were constructed, but the complete unchanged-objective partial-label intervention does not meet its frozen delta precision prerequisite. Stop here with all evidence preserved. No model-performance or architecture-success claim, and no Phase B authorisation, follows.
