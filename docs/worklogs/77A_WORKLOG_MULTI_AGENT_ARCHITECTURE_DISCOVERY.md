# Worklog — Multi-Agent Architecture Discovery for Source-Transferable Intrinsic Light Transport (2026-11-07)

## Origin and authorisation

- **Brief.** The user pasted the WL77A brief.
- **Workflow.** The brief explicitly asks for a Claude Code multi-agent research workflow with:
  - four specialist perspectives;
  - three distinct hypotheses;
  - two adversarial reviewers;
  - selection of at most one candidate;
  - a frozen contract;
  - one isolated prototype;
  - physical-reference gates.
- **Paths.** Code: `experiments/fnto_arch77a/`. Discovery record: `docs/plans/wl77a/`. Evidence root: `LabServer63:~/rest_arch77a`.
- **Not authorised:**
  - C absorbed into K;
  - new materials;
  - coupled GI;
  - production change;
  - sweeps;
  - reactive supervision;
  - held-out-based selection;
  - adoption, or a successor batch.
- **Isolation.** WL77 is complete (STOP at Stage 1, d2d8eae), and its code and roots are not touched.

## Part 0 (frozen before any candidate exists)

**Confirmation population for Gate 4.** The never-scored WL76 Stage 4 population is reused:
- `ports76.json`: 26 static slots plus 4 footprint slots, with the te1 deformation pose.
- It was selected geometry-only, excluding 81 previously scored ports and their images.
- Neither WL76 nor WL77 generated its references. The WL76 dev smoke's 0.5M-photon dev packets are not read.

**Reference contract.** Unchanged `probes74` code with the seeds WL76 predeclared for this population: 766/767xxxxxx.

**Early generation.** The references are generated now on the idle GPU, while the CPU research workflow runs. Nobody reads or scores them before the candidate protocol is frozen.

## Workflow (2026-11-07)

- **Run.** Claude Code Workflow `wf_78ec0541-090`.
  - 12 agents, all completed, 0 errors.
  - 88 minutes, about 2.9 M subagent tokens.
- **Durable records.** The workflow `journal.jsonl` is in the session directory. Each agent also wrote its own report under `docs/plans/wl77a/`.

| Stage | Agents | Output |
|---|---|---|
| Specialists | 4: transport physics; neural operator / representation; graphics / GPU systems; prior art / novelty (web search, verified citations) | `specialist_{physics,neural,systems,priorart}.md` |
| Ideation | 1 | `ideation.md` |
| Candidate developers | 3 | `candidate_H1/2/3.md` |
| Adversarial reviewers | 2: A physics / mathematics; B novelty / systems (with its own searches) | `review_A_physics.md`, `review_B_novelty_systems.md` |
| Responses | 1 advocate for all three candidates | `responses.md` |
| Adjudication | 1 drafter | `adjudication_draft.md` |

- **Ideation.** It proposed 7 mechanism families and selected 3 with distinct central mechanisms:
  - **H1 CQR:** canonical-quotient relative pair kernel.
  - **H2 RBG:** exact first-collision projection with a learned scalar resolvent.
  - **H3 LCR:** learned closure followed by a collision-probability solve.
- **Deduplication.** Reviewers recorded each objection once, even when it applied to several candidates, so the objection count is 27 (A 15, B 12).

The coordinator (main session) checked the adjudication draft against the code and ran three non-evidence numerical checks (`experiments/fnto_arch77a/coord_checks*.py`, `results/fnto_arch77a/`). The decision is in `docs/plans/wl77a/coordinator_decision.md`. The record index, with the 16 required sections, is `docs/plans/wl77a/README.md`.

**Part 0 references.** These were generated on LabServer63 in parallel with the workflow: 1,920 packets, no gate failure at the last check, status file `~/rest_arch77a/refs.status`. No candidate was selected, so the references remain **unscored**.

## Final report (2026-11-07)

### Implementation Fact

- The discovery workflow ran in full: brief Parts A–D.
- Part 0 was frozen and committed (e2c7dd3).
- Three coordinator checks ran as non-evidence local scripts on the RTX 5080. They only read historical data.
- No prototype, frozen candidate contract, training run or Gate 1–4 run exists, because the batch stopped at selection.
- WL77 code and roots are untouched.

### Measurement (coordinator checks; non-evidence, point statistics)

**C1. Projection floors of the reference multiple-scatter part, by SS quadrature size** (two training sources, s0 / s50):

| Basis | q = 1024 | q = 16384 | q = 65536 | Extrapolated noise-free |
|---|---|---|---|---|
| Ring | 0.45–0.47 | 0.24 | 0.22–0.23 | **0.21–0.22** |
| FIA-AZ | 0.39–0.41 | 0.12–0.14 | 0.08–0.10 | **0.07–0.08** |
| Free phase, order ≤ 2 | 0.35–0.37 | 0.09–0.11 | 0.05 | **0.02** |

The pure q = 1024 SS noise is 0.45–0.48 of the M norm.

**C2. Perfect learned multiple-scatter part plus the deployable q = 1024 SS:**

| Prediction | native | fine16 | coarse | contrast |
|---|---|---|---|---|
| Perfect multiple scatter + q = 1024 SS | 0.27 | 0.13–0.14 | 0.07–0.10 | **0.51–1.24** |
| WL76 FIA, measured | 0.32 | 0.19 | 0.13 | 0.69–0.75 |

**C3. Translate identity, WL74-S4 s101 vs training s50:**
- The energy invariants are equal within replica noise: difference 8.6e-5 vs SE 8.8e-5.
- Yet A75 contrast was 0.131 on the training sources and 0.73 on s101.

### Observation

- **Specialists agreed on four points:**
  - The W+UV source enters T only through absolute-position keys, with an identifiable per-source lookup through the global U (neural F1–F3).
  - The contrast signal is confined to about 5 mm, and the pair-specific angular blocks have rank about 4 (systems S3–S4).
  - The exact factorisation g·K = 4πσω⟨P, G P⟩ holds (physics F1).
  - Most candidate ingredients are prior art: first-collision factorisation, CP dictionaries, Green's-function learning (DeepRTE / NGF), frame averaging and learned RTE closures (prior-art F1–F5).
- **Disagreement.** The neural specialist (F7–F8) flagged the SS-noise confound. The physics and candidate documents had treated it as a true floor. Reviewer A upheld the objection (A6), and C1–C2 confirm it numerically.
- **Survival.** Every candidate's central transfer argument was challenged, and the challenge was upheld:
  - H1: shared translate identity, and out-of-orbit interpolation.
  - H2: sub-mfp information loss, and under-identification.
  - H3: its transfer comes from a classical solver.
  - No hard physical invariant failed.

### Interpretation

- **No candidate selected.** None of the three hypotheses offers a credible, novel, learned mechanism for the out-of-orbit boundary contexts where the historical models fail. The brief's rule therefore stops the batch at selection, as a negative discovery result.
- **Corrections to earlier attributions.** These are append-only; the historical verdicts stand as recorded.
  1. **WL77.** The STOP and its reading "order ≥ 3 azimuthal structure" came from q = 1024 single-scatter quadrature noise inside the diagnostic. With noise-free SS, the FIA-AZ floor is about 1/3 of the ring floor, and order ≤ 2 harmonics leave about 0.02.
  2. **WL76.** A large, plausibly dominant, part of FIA's INSUFFICIENT FIT is the deployable q = 1024 SS noise, which creates spurious per-bin contrast. It is not the learned post-collision rule, and not the first-interaction factorisation.
  3. **WL75.** The boundary-far failure includes physically identical translates of training sources. This points to W+UV's absolute-position conditioning (an invariance and identity problem), not to missing capacity.

### Unresolved Question

These are factual open questions for the user's decision (`coordinator_decision.md` §4):
1. How much of the FIA fit failure remains with low-noise or deterministic SS / W terms, and at what rebuild cost?
2. Does a W+UV without absolute-position, face-id, global-grid and world-axis inputs (W+UV-Q) transfer to the translate and orbit sources? What does it do on the out-of-orbit contexts?
3. Is solver + closure admissible as a learned K? This is a contract decision.
4. Should the WL76 / WL77 frozen verdicts be re-run with corrected statistics? The original records stand either way.

### Architecture Verdict

**NOT SUPPORTED: no candidate survives the review, and the batch stops at selection.**

| Property | Verdict |
|---|---|
| H1, H2, H3 as novel, transferable learned K hypotheses | NOT SUPPORTED (rejected on transfer mechanism, information or novelty) |
| Mathematical validity (linearity, positivity, passivity, K/C), on paper | SUPPORTED for all three |
| Implemented correctness, supervised fit, independent transfer, measured cost | INCONCLUSIVE (not reached) |
| Novelty | NOT SUPPORTED (the closest work covers each central mechanism) |

- **Re-attribution (C1–C3).** Non-evidence, but numerically clear. It changes how WL75–WL77 should be read, not their recorded verdicts.
- **Status.** Nothing is adopted, and no successor is started.

### Reproducibility

- **Commits:** Part 0 is e2c7dd3; this record is the commit that adds this section.
- **Workflow:** run `wf_78ec0541-090`. The script and journal are in the session directory; the reports are in `docs/plans/wl77a/`.
- **Coordinator checks:** `experiments/fnto_arch77a/coord_checks.py`, `coord_checks2.py` and `coord_checks3.py`, with outputs in `results/fnto_arch77a/coord_checks*.json`.
  - Data: the archived training bundles (`~/rest_repr68`, `~/rest_cover71`, `~/rest_src73`, local copies) and the local WL75 `rest_s101` bundle.
  - SS: the WL76 quadrature code, with fixed seed arguments per check.
- **Part 0 references:** `LabServer63:~/rest_arch77a/refs`, unscored.

## Part 0 references: completed (2026-11-07 05:44)

- **Generation:** all 1,920 packets on LabServer63 passed their gates. Counts: 6.144e10 photons, 6,814 s of summed packet time.
- **Walker:** the WL57 fast walker, unchanged.
- **Bundles:** built. Their hashes are in `refs/complete.json`, which is copied to `results/fnto_arch77a/refs/complete.json`.
- **Status:** these references have not been read or scored. Because no candidate was selected, they are an unused confirmation set for the WL76 population.
