# Worklog — WL80B C1 Intrinsic Transport Architecture Reassessment and Method-Oriented Redesign (2026-11-09)

## Origin

- **Brief.** "WL80B — C1 Intrinsic Transport Architecture Reassessment and Method-Oriented Redesign", 2026-11-09.
- **Nature.** A design-only reassessment of the WL80A Track B candidate C1 / CP-K. The batch asks whether C1 can become a credible, computationally useful and methodologically distinct ReST-GI intrinsic-K architecture, or must remain a physical reference or baseline, or be rejected.
- **Excluded.** No implementation, training, GPU work, reference generation, confirmation-label reading, production change, adoption or successor.
- **Lettered numbering.** This is a lettered batch after WL80 / WL80A.

**Records.** `docs/plans/wl80b/`.

## Coordinator actions

1. **Resume attempt.** I tried to resume the original C1 architect, as the brief prefers:
   - target: Track B C1 developer `a5be8ef2b3725efdf` (candidate architect `a2a81ccded9c38cc8`) of workflow `wf_1aeb63f6-0ff`;
   - method: SendMessage;
   - result: **not resumable** ("No transcript found for agent ID"). Completed workflow subagents are not addressable.
   - **Fallback, as the brief allows:** the sealed WL80A C1 design and its reviews are given as immutable inputs, and a stand-in architect reconstructs C1 from them.
2. **Input identities.** `docs/plans/wl80b/inputs.json` records the sha256 of all 49 read-only inputs at batch start:
   - the WL80A records;
   - the WL77A H2 / H3 records;
   - the canonical documents;
   - the WL57 / 59 / 68 / 71–80 / 80A worklogs.
3. **Independence check.** No other independent architecture proposal exists in the repository. `docs/plans` holds only the earlier V1 plans, `wl77a` and `wl80a`.
   - Agents were instructed to read nothing outside the listed inputs, read-only code (for conventions) and public literature.
   - They were not to read `~/.claude`.
4. **Launch.** Workflow `wf_9bbd762a-ae2`, phases:
   1. Part A reconstruction;
   2. five parallel analyses (Parts B, C, D, E, F);
   3. Part G redesign (at most one revision, or none);
   4. two adversarial counterfactual reviews (Part H; physics / generality, and novelty / computation / learnability);
   5. architect response;
   6. independent verdict chair;
   7. independent critic.

   Agents return text only; the Coordinator writes the records.
5. **Completion.** Workflow `wf_9bbd762a-ae2`: 12 agent runs, about 40 min, 2.1M subagent tokens, 191 tool uses. The Coordinator wrote the records verbatim (headings only) to `docs/plans/wl80b/`:

   | Record | Content |
   |---|---|
   | `c1_original_reconstruction.md` | Part A |
   | `c1_physical_audit.md` | Part B |
   | `c1_source_generalisation.md` | Part C |
   | `c1_geometry_update_contract.md` | Part D |
   | `c1_cost_model.md` | Part E |
   | `c1_prior_art.md` | Part F |
   | `c1_revision_decisions.md` and `c1_revised_architecture.md` | Part G |
   | `c1_adversarial_review.md` | Part H + architect response |
   | `c1_final_verdict.md` | Chair, plus the critic verbatim |
   | `future_falsification_design.md` | Falsification design |

   `seal.json` holds their sha256.

6. **Critic disposition** (`coordinator_disposition.md`). All 16 findings were accepted. The three HIGH findings:
   - **HIGH-1:** the label is qualified, not replaced. C1 is not a method; "physical" means physics-based comparator, not reference; its baseline role is unqualified and a human decision.
   - **HIGH-2:** the dent W is a specification / affordability gap, not an impossibility.
   - **HIGH-3:** the falsification design as written is predicted to be affordability-limited at 12 GB.

## Final report

### Implementation Fact

- **Nothing was implemented.** No code was written or executed for C1 or any successor. There was no training, GPU work, reference generation, confirmation-label reading or production change.
- **What exists.** Design records in `docs/plans/wl80b/`.
- **Resumption.** The original C1 architect could not be resumed; a stand-in reconstructed C1 from the sealed WL80A records. Input sha256 values are in `inputs.json`.

### Measurement

**No new measurement was performed.** Every number is one of:
- a cited historical measurement (verified file and line);
- a new derivation;
- an estimate.

Each is labelled. The disposition corrects the labels the critic found wrong.

### Observation

**The original C1 had a passivity defect.** Its split double-counted a fraction Δ ≥ 0 of first collisions on lit single-face cells: about 0.6–2.4 % of q at 0.5 mm, and about 5 % at grazing. The developer-amended split closes it, through:
- point-exact order-1 escape and order-2 deposit;
- cell resolvent from order 2 onwards;
- complement-defined cell escape.

With that fix the algebraic identity holds: output = ω·d − (1−ω)·1ᵀψ. It is passive given W ≥ 0 and 1ᵀW ≤ 1.

**Further WL80A claims were too strong:**
- "partition by construction";
- "passivity for any W";
- "ΣF_in = d to fp" combined with "exact D4h" (B is sampled, q = 256, seed 57501);
- "reference-free transfer check";
- the cost and dent-rebuild figures.

**Source generalisation is structural only at the continuum level.** In the discretisation it reduces to:
- lateral flat-source bias (untested; crude upper-bound estimate);
- quadrature;
- solver convergence (red ρ ≈ 0.974, about 525–663 cold sweeps).

**Geometry change.**
- Rigid motion reuses every K factor in the body frame.
- A dent invalidates 26–42 % of beams and needs a global re-solve. An exact Woodbury update costs more than one solve, and truncated low-rank updates break W ≥ 0.

**Nothing is learned.** The optional ΔW closure and the learned preconditioner were removed.

### Interpretation (INFERENCE)

C1's merit is the exactness of a classical continuum factorisation in a narrow physical scope. Every defect found is fixed either within known numerics or by spending resolution. Neither yields a methodological contribution. The intrinsic-K research question (learned transfer to unseen boundary sources) is not addressed by C1.

### Unresolved Question

1. **Comparator role.** Should C1 have any role as a classical comparator? That requires measured bias against the MC reference and a cost model at matched accuracy; it is a human decision.
2. **Dent operator.** Can an affordable visibility-exact dent W exist, and what does truncation misplace?
3. **Lateral bias.** How large is it in practice?
4. **Contract.** Does the FNTO learned-K contract stand? Changing it is a human decision.

### Architecture Verdict

**C1 PHYSICAL BASELINE ONLY**, qualified by the disposition:
- not a method candidate;
- no justified revision;
- learning-free and ALREADY KNOWN;
- inadmissible as a learned K under FNTO v0.1;
- any comparator role unqualified and pending a human decision.

**Strongest evidence against even this label** (chair and critic): a baseline whose bias and cost are unmeasured, which cannot run on the dent within the caps, and which has no BASELINE_ROLES row, may be neither useful nor competent. **C1 ARCHITECTURALLY INADEQUATE** is the arguable alternative.

No adoption. Canonical FNTO unchanged. No successor.

### Reproducibility

| Item | Reference |
|---|---|
| Workflow | `wf_9bbd762a-ae2`; script and transcripts under `~/.claude/projects/c--Projects-ReST-GI/<session>/` |
| Inputs | `docs/plans/wl80b/inputs.json` (49 sha256) |
| Records | `docs/plans/wl80b/*.md` with `seal.json` |
