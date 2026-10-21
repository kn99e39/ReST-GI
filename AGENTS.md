# Working On This Repo

## Selective_Recompute research gate — Surface-Owned Transport Units (2026-10-13; updated 2026-10-19, future selector direction)

**User-confirmed definition:** Selective recomputation means **recompute the COMPLETE owned learned state of every transport unit whose learned transport was invalidated by a geometry edit**, while keeping the complete state of every still-valid unit unchanged. A unit U_i = (S_i, Θ_i) owns a surface domain S_i and all of its trainable parameters Θ_i. Units **may be finer than a scene object** (the user's decision of 2026-10-13); a whole object is the coarsest unit, and **object-level full recomputation remains a valid coarse comparison**.

Never reinterpret this as selecting only triplane cells, local features, a subset of MLP layers, or another parameter mask inside a jointly owned model. Every parameter of a selected unit is eligible to update. A shared trainable component that must change to recover one unit is a coupling to report, never to hide with optimiser masks.

**User decisions in force (2026-10-13):**
1. RNA (`66b5b098`) is the initial implementation baseline, not a permanent constraint.
2. Ownership finer than objects is allowed.
3. Surface-Owned Transport Units are the active hypothesis.
4. An invalidated unit's complete state may be recomputed.
5. Object-level recompute is the coarse comparison.
6. At least five architecture batches come before a broad viability judgment.
7. One negative candidate does not falsify the hypothesis.
8. 2007–2013 dynamic-GI work is the priority historical reference.
9. The 1997 line-space hierarchy is not the starting point.
10. Efficiency and scaling come after representation feasibility.

Full list: docs/RESEARCH_CENTRIC_TOPIC.md §2.

**Geometry-dirty D_G and transport-dirty D_T differ.** Stationary receivers can hold invalid transport when another part moves; a mover-only rebuild does not necessarily suffice.

**Unit-level preservation:** report parameter preservation, output preservation and radiometric validity separately. They are different properties.

**Current frontier:** Batch 1 (worklog 32) showed exact ownership and full-state unit reconstruction on independent RNA units, but stopped because original RNA failed the predeclared static-credibility gate (17.97 / 24.92 dB vs 25 dB). Worklog 33 audited the RNA baseline:
- no integration defect;
- C1 is an RNA representation limit on near-mirror conductors;
- the C2 gate miss is dominated by 128-spp target noise.

Batch 2 (worklog 34) attributed the P16 seams to independent per-unit error fields exposed by hard selection. P16-PU composition removes most of the seam excess; the interior per-unit fidelity loss remains. Worklog 35 qualified native RNA on the official Lego: it reproduces the released checkpoints, and the gap to the paper is validation-target noise (classification: comparable). Static gates must use converged references.

Batch 3 (worklog 36, native Lego, user-authorised 2026-10-18) passed the predeclared static screening (Case A). 16 independent complete RNA units score −0.32 dB vs native RNA HQ on held-out 4096-spp views (logL1 1.05×, interior unchanged; 1.54× parameters, capacity-confounded). PU removes 91% of the seam excess, and one-unit REINIT is exact in parameters and output influence (S_i ∪ B_i). Batch 4 (worklog 37, physical Lego edit, oracle D_T; reported 2026-10-21) is **CONDITIONAL**:
- D* = 8 of 17 units: 7 stationary Lego receivers plus the moved box (D_T ≠ D_G).
- Complete-state recompute of D* recovers R = 0.98 of the changed-region error.
- V* units are bit-identical and remain valid (1.03× full recompute), and output changes lie exactly on the predicted support.
- The whole image is −0.313 dB vs full recompute, missing the −0.26 dB criterion. 69% of that gap comes from weakly dirty unit 3 (single seed; cause not isolated).

**Batch 5 is not authorised.** The recommended next step is a seed replicate of B_sel vs A_full, run on LabServer63 as a long job. Do not proceed to another batch automatically.

**Long-horizon research direction adopted 2026-10-19 (not implementation approval):** interreflection and nonlocal/multi-bounce influence must be considered in future representation/selector contracts. Favor an interpretable algorithmically constructed spatio-directional hierarchy for transport-relation candidates (position/bounds, normal-direction ranges, material/BRDF support) over expecting a neural model to learn the whole scene hierarchy; investigate learned relation scoring/dirty-unit selection only after physical-oracle feasibility and coverage evidence. Hierarchy/tree nodes are search indices, not the physical cross-node transport graph. Normal-only clustering is insufficient, and a missed candidate cannot be recovered by a downstream selector. No specific hierarchy, feature set, model or threshold is yet validated or authorised. See docs/RESEARCH_CENTRIC_TOPIC.md §4a and docs/Architecture.md §3a.

**Historical scope:**
- Worklog 30 validly tested triplane-cell masking inside one 8DNA asset with shared layers frozen; its negative result is not a test of owned units.
- Worklog 31 tested explicit (non-neural) surface units and object-level oracle refit.
- Keep worklogs 21–32 unchanged.
- Archived architectures: docs/history/ARCHITECTURE_LIFECYCLE_SEPARATION_2026-10-07.md, docs/history/ARCHITECTURE_INTRA_OBJECT_FEATURE_REFIT_2026-10-08.md, docs/history/ARCHITECTURE_OBJECT_LEVEL_SELECTION_2026-10-12.md.

**Do NOT:**
- resume intra-model feature masking as the main experiment;
- treat a single jointly owned asset model as several owned units;
- ignore stationary transport-dirty receivers;
- implement a dependency hierarchy or learned dirty detector before physical-edit feasibility is reviewed and the user separately authorises that experiment;
- move scenes, gates or thresholds after seeing results;
- automatically advance to the next batch after a result.

Keep code changes and future experiments subject to user research decisions. A doc correction is not approval to implement.

---

## Research context bootstrap

Before substantial research implementation, experiment design, or architecture interpretation, read:

1. `docs/RESEARCH_CENTRIC_TOPIC.md`
2. `docs/RESEARCH_ROADMAP.md`
3. `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`
4. `docs/Architecture.md`

In particular, do not confuse implementation/development baselines with scientific replication or architecture-contrast baselines, and do not generalize a negative result from one model/asset pair into project-level falsification without the evidence breadth defined in the strategy document.

## Living-document synchronization

Worklogs are historical evidence; the central documents above are the current research state.

When the user and assistant reach a **substantive long-term research-direction decision**—for example adopting a future architecture hypothesis, changing the representation/selection contract, elevating an evaluation requirement, or changing the next scientific question—**update the relevant living documents immediately in that conversation**; do not wait for the next Worklog, experiment completion, or Agent prompt. Also update them when a batch closes, a gate closes, a baseline role changes, a kill condition is triggered or implementation is authorized. Interpret the evidence and preserve the difference between *user decision*, *candidate hypothesis*, *measured evidence*, and *implementation authorisation*.

At minimum consider:

- `docs/Architecture.md` for representation ownership, selected working architecture, update contract, or implementation frontier;
- `docs/RESEARCH_ROADMAP.md` for milestone status, current decision frontier, next task, or stop condition;
- `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` for baseline roles, evidence hierarchy, or generality scope.

Do not rewrite prior worklogs to make them agree with the new interpretation. Preserve them as append-only history and record the new current interpretation in the living documents.

Do not update living documents for trivial implementation details or every diagnostic result. Do not use an immediate direction-document update to alter an experiment already in progress, claim validation of a proposed architecture, or advance implementation without approval. Avoid starting a new numbered Worklog solely to record a documentation update when that number may be reserved for an ongoing batch; the dated decision record in central documents is sufficient until the owning batch reports.

## Worklog conventions

`docs/worklogs/` is the base location for human-readable worklogs — Korean or English
narrative explaining a JSON result's purpose, verdict and limits.
Machine-reproducible verdicts and pipeline input/output stay in JSON; a
worklog is not a place to paste one wholesale (an agent once filed
`docs/worklogs/44` as raw JSON — recognize that as a mistake, not a pattern to repeat).

Create a numbered worklog for a substantive session: an experiment, analysis,
implementation, structural change, or decision whose rationale and limits need
to be traceable. Do not create one for minor housekeeping such as a small path,
formatting, or configuration adjustment unless it materially changes an
experiment's execution or interpretation.

Worklog history is one numbered worklog per qualifying session, never a single mutable
handoff doc that gets overwritten (the old `docs/worklogs/11_SESSION_HANDOFF_*.md`
pattern was retired 2026-08-24 for exactly that reason):

- Find the highest-numbered file already in `docs/worklogs/` and use the next
  integer, even if that file doesn't otherwise look like a normal worklog
  (wrong extension, missing `_WORKLOG_` infix, JSON instead of Markdown) —
  fix that file's own format if asked to, but don't reuse its number.
- Name the file for what the session actually did, not its date —
  `docs/worklogs/12_WORKLOG_YAW_TAIL_ATTRIBUTION_AND_A11.md`, not
  `docs/worklogs/12_WORKLOG_2026-08-25.md`. Every worklog filename carries
  `_WORKLOG_` as an infix. The date belongs inside the document, on its
  title line (`# Worklog — <Descriptive Name> (<date>)`), not primarily in
  the filename.
- Never edit or delete a previous worklog to "update" it. A worklog is
  append-only; a correction to an earlier one's claims is a new worklog
  that says so explicitly (e.g. docs/worklogs/38 correcting docs/worklogs/37's wording,
  docs/worklogs/40 correcting docs/worklogs/39's), not a silent rewrite.
- Scope a worklog to session narrative: what was done, why, which commits,
  what's still open. Durable facts that keep being updated over many
  sessions — experiment tables, data-contract fingerprints, gate
  definitions, environment/server setup — belong in their own living
  document instead of being duplicated into every worklog that touches
  them (`docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md` for the lifter
  ablation table, `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md` for server
  setup) — reference it, don't repeat it.
- A worklog is bound by the same discipline as everything else here: label
  every result by evaluation regime and never compare across regimes
  without the label, and never state a gate/case verdict stronger than what
  was actually measured that session.

## Evaluation artifact conventions

Human-reviewable intermediate and final artifacts (images, GIFs, videos, and
their evaluation manifest entries) belong under a folder keyed by the session
worklog number: `results/evaluation/<number>/`. For example, artifacts for
`docs/worklogs/17_WORKLOG_*.md` go in `results/evaluation/17/`.

- Create the matching evaluation folder for every substantive session that
  produces reviewable output, even if earlier numbers have no evaluation
  folder. Number gaps are intentional; never reuse another worklog's folder.
- Keep source experiment outputs in their experiment directories. Copy only
  reviewer-facing artifacts into the matching evaluation folder and record
  their provenance/hash in that folder's manifest.
- Do not move, rename, overwrite, or retrospectively reorganize artifacts in
  a prior numbered evaluation folder unless the user explicitly asks. A new
  worklog/session gets a new numbered folder.

## Running long jobs in local WSL

When the `wsl.exe` client that started a job is killed, the job dies with it.
It gets SIGHUP, and WSL then idles the distro down about 15 s later. That is
what the old "WSL DXG restart" reports were
(`docs/worklogs/16_WORKLOG_WSL_DXG_ROOT_CAUSE_DIAGNOSIS.md`). An agent's
shell-tool timeout is exactly such a kill.

- Never run a WSL render, training, dataset generation or loader job that
  may outlast the tool timeout as a foreground tool call. Treat anything
  over about a minute as at risk.
- Run such a job either as a background task, so the client is not subject
  to the tool timeout, or through
  `experiments/dynamic_transport_failure/scripts/wsl_run_detached.sh`.
  Poll the launcher's `<job>.status` file from short, separate `wsl.exe`
  calls.
- Do not rely on `systemd-run`, a bare `nohup ... &`, or a `setsid` job
  whose launching client is later killed. WSL's idle shutdown kills all of
  them.
- A job that ends without its completion marker is not evidence of a GPU,
  DXG or RNA fault until the launching client is shown to have survived.
  To check, rerun it under
  `experiments/dynamic_transport_failure/scripts/capture_wsl_gpu_pv_failure.ps1`
  and read `termination` in its `summary.json`. Only `job_failed` or
  `killed_without_exit` points at the job or the runtime;
  `client_teardown` means the client was lost again
  (`docs/worklogs/20_WORKLOG_WSL_CAPTURE_CLIENT_TEARDOWN_CLASSIFICATION.md`).
