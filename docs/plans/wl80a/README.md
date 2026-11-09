# WL80A — Independent multi-agent architecture thought experiment (index)

**Scope.** A design-only study, 2026-11-09. No implementation, training, GPU work, Monte Carlo generation, confirmation-label inspection, production change, adoption or successor batch.

**Worklog.** `docs/worklogs/80A_WORKLOG_MULTI_AGENT_ARCHITECTURE_THOUGHT_EXPERIMENT.md`.

## Order of creation (isolation and sealing)

| # | Artifact | Role | Written by |
|---|---|---|---|
| 1 | `neutral_problem_brief.md` | Track B's only project input (no BGC, no prior recommendations, no unmeasured causal claims) | Coordinator, before launch |
| 2 | `trackB_blind_brief.md` | Track B isolation rules + contamination audit (clean: only the brief was read) | Coordinator, at sealing |
| 3 | `trackB_specialist_reports.md` | Five independent blind specialist reports | Track B agents (verbatim) |
| 4 | `trackB_candidate_records.md` | Research synthesis, four distinct candidates C1–C4, merged/dropped ideas | Track B candidate architect |
| 5 | `trackB_adversarial_review.md` | Two adversarial reviews per candidate + developer responses | Track B agents |
| 6 | `trackB_selected_candidate.md` | Comparison matrix, rejected mechanisms, selection (primary C1, secondary none), verdict | Track B synthesis chair |
| 7 | `trackA_original_bgc.md` | The BGC proposal, verbatim (on disk only after sealing) | Coordinator |
| 8 | `trackA_critical_audit.md` | Audit of questions 1–15, counterexamples, objection/response matrix, specialist and adversarial reports | Track A agents |
| 9 | `trackA_decision_log.md` | RETAIN / REVISE / REMOVE / UNRESOLVED per element, with cause → effect, plus an independent check | Track A agents |
| 10 | `trackA_revised_architecture.md` | Original vs revised, revised specification, novelty, Track A verdict (BGC REJECTED) | Track A agents |
| 11 | `seal.json` | sha256 of items 1–10, written before the cross-review was launched | Coordinator |
| 12 | `cross_review.md` | Chair synthesis + three independent panel reviews (physics/representation, generalisation/learnability incl. Part E, computation/practical) | Cross-review agents |
| 13 | `prior_art_matrix.md` | Consolidated prior-art matrix + three novelty kill searches (BGC, C1, C2) | Cross-review agents |
| 14 | `final_architecture_decision.md` | Chair decision record (Track A/B answers, comparative verdict, recommendation, first-experiment design, lab summary draft, unresolved) + independent critic, verbatim | Cross-review agents |
| 15 | `coordinator_disposition.md` | Coordinator acceptance of the critic's findings; corrected statements supersede the chair's wording | Coordinator |
| 16 | `lab_meeting_summary.md` | Corrected professor-facing summary | Coordinator |
| 17 | `workflow_journal.jsonl` | Agent start / result events of the three workflows (results removed; full texts are in the files above) | Coordinator |

## Workflows

| Workflow | Run | Agents | Notes |
|---|---|---|---|
| Track B | `wf_1aeb63f6-0ff` | 19 | Resumed once after a session interruption; cached results reused |
| Track A | `wf_ffa3a3a1-8c8` | 9 | Resumed once |
| Cross-review | `wf_9ac78ed1-61a` | 8 | Resumed once at the chair step |

## Outcome in one paragraph

**Track A** rejected BGC-Transport as a source-generalisation mechanism. Its revisions fix physical validity only.

**Track B** independently proposed four mechanisms and selected **C1 CP-K**, a learning-free explicit collision-probability operator (exact first flight → scalar collision-density solve → exact last flight). It deferred C2 SBG (a learned bounded scalar Green's function) and rejected C3 and C4.

**The cross-review** found:
- no candidate is novel as a mechanism;
- none is defensible today as a validated learned K for unseen boundary sources.

C1 is the strongest physically grounded and falsifiable candidate *operator*, designated only for a future decision. Its admissibility as an FNTO K and the WL77A H3 rejection grounds remain open.

Nothing was measured, and nothing is adopted.
