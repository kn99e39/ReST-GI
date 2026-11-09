# Worklog — WL80A Independent Multi-Agent Architecture Thought Experiment for Source-Transferable Intrinsic K (2026-11-09)

## Origin

- **Brief.** "WL80A — Independent Multi-Agent Architecture Thought Experiment for Source-Transferable Intrinsic Light Transport", 2026-11-09.
- **Nature.** A design-only, adversarial, multi-agent study. Two information-isolated tracks:
  - **Track A** — critical review and justified refinement of the user-proposed BGC-Transport hypothesis;
  - **Track B** — blind architecture discovery from a neutral evidence brief only.
- **Afterwards.** A post-sealing cross-review, a novelty kill search, a decision record, a bounded first-experiment design and a lab-meeting summary.
- **Excluded.** No implementation, training, GPU work, Monte Carlo generation, confirmation-label inspection, production change, adoption or successor batch.
- **Lettered numbering.** This is a lettered, unnumbered batch after WL80 (the user rule of 2026-11-03).

**Artifacts.** In `docs/plans/wl80a/`; the index is `docs/plans/wl80a/README.md`.

## Coordinator actions (chronological)

1. **Contamination check.** Searched the repository for any BGC material: none exists. The only matches are unrelated third-party package files.
2. **Neutral brief.** Wrote `docs/plans/wl80a/neutral_problem_brief.md`, Track B's only project input.
   - It contains: the objective, the physical problem and operator contract, data and evaluation conventions, measured observations, outcomes of tried approaches (no causal explanations), and costs.
   - It excludes: BGC, earlier assistant recommendations (e.g. the WL77A candidates), and unmeasured interpretations.
3. **Isolation design.** Workflow subagents share a filesystem, so isolation is procedural:
   - **Track B rules.** Agents may read only the brief file, plus their own knowledge and web literature. They must not open the repository or `~/.claude`, and must not write files.
   - **BGC handling.** The BGC text was passed to Track A only as an in-memory workflow argument, and is not written to the repository before both tracks are sealed.
   - **No shared files.** Both workflows return text only; the Coordinator writes the files after completion.
   - **Audit.** After completion, the Track B agent transcripts are checked for any local file access other than the brief.
4. **Launch.** Both tracks were launched concurrently:
   - Track B: workflow `wf_1aeb63f6-0ff`;
   - Track A: workflow `wf_ffa3a3a1-8c8`.
5. **Interruption and resumption.** The session ended while both workflows were running. Both were resumed from their run ids; completed agents replayed from cache. The dead agents of the interrupted run stay in the transcripts. Track B's candidate architect therefore ran twice; only the resumed run's output is used.
6. **Completion.**
   - Track B: 19 agents, about 19 min, 1.31M subagent tokens.
   - Track A: 9 agents, about 19 min, 0.68M subagent tokens.
7. **Contamination audit (from the agent transcripts).**
   - **Track B.** 20 local tool calls, all `Read` of `neutral_problem_brief.md`. 17 web searches. No Grep / Glob / Bash / Write. No "BGC" in any Track B input or tool result. **The blind track is valid.**
   - **Track A.** It excluded `docs/plans/wl80a/` explicitly in its searches. As allowed, it read the canonical documents, the worklogs and `docs/plans/wl77a`.
8. **Sealing.** Both tracks completed before any track file was written. The Coordinator then wrote the records verbatim, with headings only, and recorded their sha256 in `docs/plans/wl80a/seal.json`. Committed before the cross-review was launched.

   **Sealed outcomes (as recorded by the tracks; not yet cross-reviewed):**
   - **Track A.** BGC REJECTED as a source-generalisation mechanism; only its passive parametrisation is repairable.
   - **Track B.** Four distinct candidates:
     - C1 CP-K: explicit collision-probability operator;
     - C2 SBG: learned scalar bounded-medium Green's function;
     - C3 CCA: gauge-canonical context atlas;
     - C4 NSM: collision-order split.
     Primary: C1; secondary: none.
