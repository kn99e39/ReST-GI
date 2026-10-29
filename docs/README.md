# Document ownership and navigation

Living documents are mutable representations of current decisions, not cumulative journals. Start with the research topic, architecture and roadmap to recover the current objective, contracts and decision frontier without reading every Worklog.

| Document / location | Exact responsibility |
|---|---|
| [Research topic](RESEARCH_CENTRIC_TOPIC.md) | Current problem, motivation, scientific objective/hypothesis, scope, research constraints, frontier and high-level stops |
| [Architecture](Architecture.md) | Current physical/representation contracts, interfaces, invariants, adopted decisions, candidate dispositions and unresolved design questions |
| [Roadmap](RESEARCH_ROADMAP.md) | Current bounded scope, milestone dependencies, next decision gate and untested coverage; links authorisation rather than accumulating approval history |
| [Evidence strategy](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md) | Baseline roles, comparison discipline, reference caveats, evidence breadth and cost/quality accounting |
| [FNTO contract v0.1](FNTO_ARCHITECTURE_CONTRACT_V0_1.md) | Unchanged versioned historical design, dated 2026-10-23. Original status and future-tense statements apply at that date; consult Architecture for current status |
| [Plans](plans/) | Explicitly unadopted candidate designs and implementation plans; existence is not adoption or completed evidence |
| [Worklogs](worklogs/) | Complete append-only chronological evidence: authorisation, hypotheses, implementation, metrics, failures, interpretation, verdicts, provenance and amendments |
| Experiment protocols (`experiments/.../protocol/`) | Frozen experimental conditions, stage dependencies and acceptance criteria, with labelled amendments |
| Results (`results/.../`) | Raw measurements, machine verdicts and execution artifacts; reviewer exports retain their numbered provenance |
| [History](history/) | Superseded designs and source snapshots, retained with their original context; no current authorisation inferred from historical text |
| [Repository instructions](../AGENTS.md) | Operating constraints and user instructions; read newer explicit bounded authorisation in context rather than applying an older batch's stop to all future work |

## Maintaining the boundary

Record complete experimental results once in the owning Worklog and link to them. Update a living conclusion when evidence or a user decision changes it; remove superseded current-state text instead of appending another dated paragraph. A brief candidate disposition with rationale and evidence belongs in Architecture; measurements and execution histories do not.

Classify changed passages as current decision, historical detail, superseded decision, unresolved question or unique historical information. Preserve unique material in history before removal. Do not alter scientific verdicts to reconcile wording, and report genuine unresolved decision conflicts. Frozen protocols and append-only Worklogs cannot be revised by a living-document edit.

Keep physical K/C ownership separate from any proposed learned rule/state representation. Candidate implementation and gate success do not imply adoption. Material changes to a versioned architecture require an explicit successor contract; never rewrite v0.1 to imply later requirements were present originally.

Human-facing new documents belong under `docs/`. Protocol/configuration stays alongside isolated experiment code, and primary data stays at its original results location. The [2026-10-29 preservation and classification record](history/living_document_cleanup_2026-10-29/README.md) explains this correction and links every preserved source snapshot. No scientific verdict or experiment scope changes through this maintenance.
