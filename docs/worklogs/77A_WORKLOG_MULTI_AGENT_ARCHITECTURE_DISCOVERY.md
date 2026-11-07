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
