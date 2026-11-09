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
9. **Seal commit 861066e**, then the cross-review workflow `wf_9ac78ed1-61a` (8 agents, resumed once at the chair step):
   - panel: physics / representation, generalisation / learnability (including Part E), computation / practical value;
   - novelty kill searches for BGC, C1 and C2, with web sources;
   - a chair, then an independent critic.
   The panel had read-only access to the sealed records, the repository and the web, and wrote no files.
10. **Coordinator disposition** (`coordinator_disposition.md`). All 15 critic findings were accepted.
    - **Two HIGH findings:** the draft lab summary inverted the predeclared outcome rule, and it overclaimed what a C1 pass would show.
    - **MEDIUM-3 (primary designation):** C1 was re-qualified as a candidate *operator*, not a learned-K answer. WL77A H3's rejection grounds are not refuted.
    - A corrected `lab_meeting_summary.md` supersedes the chair's draft. The chair's record and the critic are preserved verbatim.

## Final report

### Implementation Fact

- **No architecture was implemented.** No code was written for any candidate, nothing was trained, and no GPU or Monte Carlo job ran. No production or canonical document changed.
- **Artifacts.** Design records only, in `docs/plans/wl80a/` (index: `README.md`).
- **Workflows.** Three Claude Code workflows with 36 agents in total: Track B 19, Track A 9, cross-review 8.
- **Utility.** A separate convenience tool, the workflow status viewer `C:\Users\dna10\restgi_relay\wf_status.py`, was made on the user's request. It is outside the repository.

### Measurement

**No new physical or model measurement was performed.** Every number in the records is one of:
- a historical FACT, cited to a worklog line;
- a DERIVATION or ESTIMATE, labelled as such.

### Observation

- **Isolation held.** Track B's audit found only reads of the neutral brief, plus web searches. Track A excluded the other track's directory.
- **Convergent derivation.**
  - Track A's physics specialist derived the operator as a symmetric bilinear form of two first-collision densities around the collision resolvent.
  - Track B's physics specialist, blind, derived the same operator as a flight–resolvent–flight sandwich.
  - The cross-review panel confirmed that the two are the same operator.
- **Defects found by agents in their own track's proposals:**
  - Track A: BGC's redundant ρ, its loose bound d instead of ω·d, and a receiver-uniform edit coupling.
  - Track B: a passivity breach in C1's original split form.
- **Prior art.** Every candidate's central mechanism is ALREADY KNOWN, or a revision of a mechanism already proposed internally (WL77A H2 / H3).

### Interpretation (INFERENCE)

- **Where the structural argument comes from.** The only structural reason found for transfer to unseen sources is physical rather than learned. For an isotropic, homogeneous, index-matched medium, all source dependence enters through an exactly computable first flight. That makes "transfer" a question of discretisation convergence (C1).
- **The learned designs.** Their transfer argument stays inductive-bias-only (BGC) or moves the problem to coverage of an intermediate domain (C2).
- **What this means for a learned K.** In this fixture, the learned-K question may be the wrong framing. This is an inference; it bears on the FNTO contract, which is a human decision.

### Unresolved Question

1. **Admissibility.** Is a learning-free operator admissible as an FNTO intrinsic K? The contract currently says learned. The WL77A H3 rejection grounds are not refuted.
2. **C1 accuracy.** Does C1 reach the gates on edge, corner and side strata and on the dent signed change? The open issues are lateral flat-source bias, far-field positivity and convention alignment.
3. **C1 cost.** Can it be made competitive? Under deformation it probably cannot; lighting-only reuse is plausible.
4. **C2.** Are dent-region compressibility and privileged interior tallies affordable?
5. **Extension.** How does any of this extend beyond g = 0, homogeneous, index-matched media?

### Architecture Verdict

| Item | Verdict |
|---|---|
| Original BGC | Not defensible as a source-generalisation mechanism |
| Revised BGC | Physically valid, but not materially better for transfer |
| Track B C1 CP-K | Strongest physically grounded and falsifiable candidate *operator*; learning-free; not novel; deployability and admissibility open |
| Track B C2 SBG | Conditional learned secondary |
| Track B C3, C4 | Rejected |

**No validated or adopted architecture. Canonical K/C ownership unchanged. No successor batch started.** The first-experiment design in `final_architecture_decision.md` §E (amended by `coordinator_disposition.md`) is a design, not an authorised batch.

### Reproducibility

| Item | Reference |
|---|---|
| Workflow runs | `wf_1aeb63f6-0ff` (B), `wf_ffa3a3a1-8c8` (A), `wf_9ac78ed1-61a` (cross-review); scripts and transcripts under `~/.claude/projects/c--Projects-ReST-GI/<session>/` |
| Agent roles and isolation rules | Recorded in the track records |
| Sealed hashes | `seal.json` |
| Event log | `workflow_journal.jsonl` |
| Seal commit | 861066e; the final commit follows |
