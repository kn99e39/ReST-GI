# Living-document correction — source preservation and classification

Date: 2026-10-29. Documentation maintenance only; no research batch, new hypothesis, experimental execution or scientific verdict change.

## Provenance and preservation

[manifest.json](manifest.json) records exact SHA-256 hashes of every source snapshot and the Git HEAD at entry. These are byte-for-byte copies, not amended evidence. Relative paths and section references inside snapshots retain the meaning they had at their **original source locations**. Stale status statements inside them are historical text, not current instructions.

| Snapshot | Origin and reason retained |
|---|---|
| [Architecture.md](Architecture.md) | Working-tree architecture at entry; preserves the user's existing structure/contract cleanup before further edits |
| [Architecture_at_HEAD.md](Architecture_at_HEAD.md) | Committed architecture at entry; preserves unique earlier Surface-Owned, partition/composition and SPT-1 design detail already removed by the in-progress cleanup |
| [RESEARCH_CENTRIC_TOPIC.md](RESEARCH_CENTRIC_TOPIC.md) | Original research framing, full historical user decision list, prior-art table and duplicated evidence |
| [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) | Former batch program, resolved prerequisites, recommendation alternatives and dated direction/approval record |
| [BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md) | Historical baseline tables, reference-regime measurements, arm definitions and duplicated result/authorisation passages |
| [document roles at entry](README_at_entry.md) | Existing untracked document-role draft preserved before clarification |

Original Worklogs, protocols, results, candidate plan and FNTO v0.1 remain at their original paths and are not edited by this correction. Concurrent research-session changes are recorded separately below. No Worklog was created for maintenance. No commit, branch change or worktree operation was performed; the Git working diff plus these snapshots distinguish this correction from the pre-existing edits.

## Classification and disposition

| Class | Passages / examples | Disposition and current owner |
|---|---|---|
| A — current authoritative decision | Dynamic learned-GI objective; K/C physical ownership; spatial/directional boundary fields; separate ballistic/scattered paths; complete-unit baseline; validity distinctions; full/selective quality/cost comparison; four edit families | Retained and clarified in [research topic](../../RESEARCH_CENTRIC_TOPIC.md), [Architecture](../../Architecture.md) and [evidence strategy](../../BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md). No learned candidate promoted to adopted status |
| B — historical experimental detail | Copied WL41/42/56/57 paragraphs; earlier batch tables; per-run metrics, seam attribution and reference-renderer measurements | Replaced with short evidence links. Owning sources include [WL32](../../worklogs/32_WORKLOG_RNA_SURFACE_OWNED_TRANSPORT_UNITS_BATCH1.md), [WL34](../../worklogs/34_WORKLOG_SURFACE_OWNED_BOUNDARY_ATTRIBUTION_AND_PU_COMPOSITION.md), [WL35](../../worklogs/35_WORKLOG_RNA_REFERENCE_REPRODUCTION_LEGO.md), [WL36](../../worklogs/36_WORKLOG_NATIVE_LEGO_SURFACE_OWNERSHIP_GENERALIZATION.md), [WL37](../../worklogs/37_WORKLOG_ORACLE_SELECTIVE_RECONSTRUCTION_DIFFUSE_LEGO.md), [WL38](../../worklogs/38_WORKLOG_B4_SEED_REPLICATE_SELECTIVE_VS_FULL.md), [WL41](../../worklogs/41_WORKLOG_FNTO_V0_1_IMPLEMENTATION_AND_PHYSICAL_EDIT_VALIDATION.md), [WL42](../../worklogs/42_WORKLOG_FNTO_VALIDITY_RESTORATION_R1.md), [WL56](../../worklogs/56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md) and [WL57](../../worklogs/57_WORKLOG_LEARNED_INTRINSIC_K_RULE_STATE.md). Exact removed source versions are also retained above |
| C — superseded decision/status | Surface-Owned or SPT-1 called the active next hypothesis; FNTO called unimplemented/untested; WL56 called designed but not executed; completed Batch 2/4/6 described as future/in progress; blanket no-successor statements preceding newer bounded approval | Replaced by present candidate dispositions and [roadmap](../../RESEARCH_ROADMAP.md). The intervening intrinsic_state_reuse_v1 authorisation and its now-completed angular preflight stop are distinguished from adoption and from earlier batch-close limits. Unperformed F/S stages remain INCONCLUSIVE |
| D — unresolved architecture question | General Rule–State superiority; shared learned angular/deformation error; complete physical quality floor; persistent-state cost; broad deformation/contact validity | Explicit INCONCLUSIVE questions in [Architecture](../../Architecture.md); stopped/unevaluated stages remain unresolved |
| E — unique historical information | Original ten user decisions; old batch recommendations and planning alternatives; complete prior-art descriptions; earlier detailed representation/partition and SPT-1 contracts | Preserved in the exact source snapshots above rather than assumed duplicated in Worklogs. Durable still-current constraints are retained in living documents; superseded prescriptions stay historical |

## Interpretation boundaries

FNTO v0.1 is not silently upgraded to include later reference-semantics, ballistic, near-contact or within-K state requirements. Its original bytes remain unchanged. Architecture identifies the need for an explicit successor version if a materially different detailed design is adopted.

The tested direct/Rule–State learned recipes remain unsupported for integration. The abstract hypothesis remains INCONCLUSIVE. Exact selective-oracle equivalence remains separate from the failed complete physical accuracy target. Reference-semantics attribution does not retroactively change earlier Worklog verdicts. The angular preflight in the existing candidate is a prospective scope choice, not a new requirement imposed retroactively on WL57.

No genuinely conflicting current architecture decision was resolved by invention. Apparent authorisation conflicts are dated batch-close statements superseded only by the newer explicit bounded brief, which has itself now closed at its angular stop. AGENTS.md remains untouched, including its historical records; current navigation explains that temporal boundary.

## Reproducibility and verification

Verification covers the edited living documents and this archive: Markdown heading/fence structure, local relative links and anchors, status/chronology searches, source snapshot hashes, protected-file byte identity and manual contract/diff review. Archived snapshots are exempt from current-link/current-status checks because their original paths and statements are preserved verbatim.

The protected-file comparison includes every existing file under `docs/worklogs/`, `experiments/`, `results/` and `docs/plans/`, plus `AGENTS.md` and `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md`. No GPU, training, inference, rendering or scientific regression is needed or run. Verification outcomes are recorded below.

Concurrent work advanced HEAD from the entry commit through `73542fa` (document-role draft) to `af600ec` (reuse candidate/protocol) and added the completed [WL58 report](../../worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md) while this correction was in progress. Those commits, report additions and scientific artifacts were made outside this correction. During verification that session also updated AGENTS.md with the closure, refined the WL58 compatibility status to INCONCLUSIVE and updated its result provenance and case CSV artifact. These observed concurrent changes are identified with before/after hashes in verification.json; they are not counted as byte-identical across the entire correction. The living conclusions were refreshed from the newly available report; the report itself was not edited. Entry snapshots deliberately preserve the earlier state, not a fabricated final-state snapshot.


## Per-document correction record

| Edited document | Responsibility and decisions retained | Removed duplication / stale status corrected | Historical access |
|---|---|---|---|
| Research topic | Objective, motivation, hypothesis, constraints, four edit families, novelty discipline and scientific stops | Removed batch chronology and per-run results; replaced Surface-Owned/SPT-1 active-next framing, unimplemented FNTO and unexecuted WL56/frontier claims with the current unresolved research question | Exact topic snapshot above; owning Worklogs linked from the current frontier |
| Architecture | Current K/C ownership, boundary/state interfaces, physical validity, candidate dispositions and explicit unresolved decisions | Retained the entry cleanup; added missing status/evidence navigation instead of restoring its old batch diary. Clarified historical v0.1 status, K44's limited role and the stopped reuse candidate; no numerical result tables added | Entry and HEAD architecture snapshots above; candidate rows link their original evidence |
| Roadmap | Dependencies, next scientific decision and bounded execution limits | Removed batch program/results/approval chronology and resolved prerequisites; corrected obsolete pending-batch and blanket no-successor claims, then reflected the concurrent candidate closure | Exact roadmap snapshot above; WL58 owns current stop evidence |
| Evidence strategy | Baseline roles, reference validity, regime separation, controls, metrics and full lifecycle accounting | Removed duplicated approval/result paragraphs, historical measurements and stale present-next-question; separated complete-unit reconstruction controls from persistent-state F/S, without changing frozen criteria | Exact strategy snapshot above; current reference caveats link original attribution Worklogs |
| Document-role README | Ownership, navigation and update discipline | Replaced ambiguous “detailed physical contract” label for v0.1 with explicitly historical version ownership; permits brief current candidate dispositions while keeping execution and measurements in Worklogs | Exact role draft snapshot above |

## Verification outcome

- Markdown headings/fences and **110 local relative links** in the five living documents plus this record pass.
- All **six source snapshot hashes** match the entry manifest.
- **2,153 protected files** retain their earlier observed bytes; **four concurrent-session updates** were inspected separately and then verified against their latest observed hashes. No protected file was written by this correction. FNTO v0.1 and pre-existing historical Worklogs/protocols/results retain their original bytes.
- Current-status searches and manual contract/diff review pass: no stale pending-batch statements remain in edited living documents; unresolved hypotheses remain unadopted; the latest stop leaves F/S cost/fidelity unmeasured.
- `git diff --check` passes. No scientific test or new Worklog was run/created by this correction.

[Verification details and concurrent-change hashes](verification.json) separate maintenance checks from scientific measurements. Recheck source preservation by hashing each snapshot named in manifest.json; recheck Markdown targets relative to each live document, excluding verbatim historical snapshots. The operating working diff retains the correction for review; no commit was created by this task.
