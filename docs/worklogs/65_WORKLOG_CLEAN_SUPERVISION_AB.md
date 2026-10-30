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
