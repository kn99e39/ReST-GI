# Working On This Repo

## Current architecture design owner and candidate (2026-10-23)

The user explicitly requested that the assistant take responsibility for choosing and specifying the research **Architecture**, rather than asking an implementation Agent to invent the core abstraction. The assistant-authored design is `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` (FNTO: learned **intrinsic optical-domain** scattering kernels + current geometry-dependent **editable free-space** compressed coupling (including same-asset exposed interreflection)).

**FNTO v0.1 was implemented and evaluated in Batch 6 (worklog 41): NOT SUPPORTED** (see the Batch 6 result below); the design contract itself is unchanged. SPT-1 (worklog 39) remains NOT SUPPORTED and must not be patched into FNTO. Preserve complete-unit Surface-Owned reconstruction (worklogs 32–38) as a separate baseline. A future implementation Agent must implement the user-approved specified architecture; it must not silently replace this with a free-space radiosity solver, SH2-only Lambertian model, a scene-baked monolithic MLP or an SPT-1+SSS patch. If FNTO's contract is infeasible, report the conflict before proceeding outside scope.

Research invariants: learned transport **response**, not canonical baked radiance; SSS, transmission, local sharp scattering and exposed self-interreflection must have viable **non-overlapping** information paths; spatially varying incident illumination is part of the operator input; parameter reuse and dynamic coupling are separate ownership contracts; strict no-claim of nonrigid validity for a kernel whose internal geometry has changed. No new batch is authorised by recording the FNTO design.

**2026-10-23 (Batch 6, user brief "FNTO v0.1 Initial Implementation & Physical-Edit Validation"): IMPLEMENTATION AUTHORISED.** The user authorised one bounded architecture batch that implements `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` (authoritative for the representation; intrinsic optical-domain K with exposed same-asset free-space paths in C, a=b allowed) and evaluates it on the official Lego + red box held-out T0/T1 edit (worklog 41; exports `results/evaluation/41/`). Required: reference-renderer semantics study (Christensen-Burley SSS scope) before main kernel training, isolated-domain spatial/directional probe supervision of K, synthetic contracts A-F on the production code, predeclared screening rules, arms A-F (historical RNA / Surface-Owned / SPT-1 unchanged; FNTO stale / updated / optional oracle diagnostic), full cost accounting and a SUPPORTED / CONDITIONAL / NOT SUPPORTED verdict. Not authorised: a learned dirty-unit selector, a second architecture family after a failure, SPT-1 modification, changes to historical baselines, automatic start of a later batch. No FNTO result exists yet.

**Batch 6 result (worklog 41, 2026-10-23): NOT SUPPORTED** under protocol fnto_v1. The implemented FNTO v0.1 (learned intrinsic kernels K_loc/K_glob/K_band/K_mid trained only on isolated-domain first-intrinsic-event probes; ray-cast free-space coupling incl. same-asset paths; learned coupling rule empty) diverged in the multi-bounce solve on 64/64 held-out view solves, so static fidelity and edit recovery are unmeasurable. Attribution (descriptive): the learned nonlocal kernel parts are non-passive on incident fields outside the span of the 1,400 sun-probe training fields (effective albedo > 1 on 26 % of ports for K_mid, 18 % for K_glob; spectral radius ~2.1 with K_glob, explosive with K_mid; one bounce already 5.5x the reference remainder); the kernel alone reproduces first-event transport at 22 dB vs first-event references (~10 dB above probe noise). Coupling semantics and the exact incremental relation update (0.83 % of rays) behaved as designed; kernel parameters stayed bit-identical. Conclusion: an intrinsic operator must be passive on the fields the coupling produces, not only linear. No later batch is authorised.

**2026-10-23 (user brief "FNTO Validity Restoration — Post-Batch-6 Controlled Implementation & Re-evaluation"): CORRECTIVE IMPLEMENTATION AUTHORISED (worklog 42).** FNTO stays the primary architecture; the batch verifies the six findings of an independent Batch 6 validity audit (buried port origins / biased incident lighting; insufficient independent spatial excitation of K; effective kernel support narrower than declared; incomplete internal reflection / directional transport; reference clamping and reconstruction-filter mismatches; insufficient signed-change, solver-stability and coupling-compression evaluation) against code and evidence, corrects confirmed defects in a distinct candidate and protocol version, and re-evaluates through gates 1-5. Batch 6 artefacts, protocol, verdict (NOT SUPPORTED) and worklog 41 are preserved unchanged; no other architecture is started.

**Worklog 42 result (2026-10-24, protocol `fnto_r1` + amendments 1–3, verdict fixed before results): NOT SUPPORTED** by the static-representability rule (held-out T0, historical regime: FNTO r1 24.49 dB vs RNA_T0 33.57 → S = −9.08 dB). All six audit findings were CONFIRMED on the Batch 6 code and corrected in the r1 candidate (on-surface multi-sample ports with triangle-identity attachment; full-radius aggregated support; intrinsic-all probes of four light families with calibrated radiometry; unclamped point-sampled reference regime r1 — in this Cycles build `BOX` ignores `filter_width`, `GAUSSIAN 0.01` point-samples; growth-based divergence detection and a spectral-radius gate; signed-change evaluation). Measured this time, unlike Batch 6: the unconstrained kernel K_r1 is non-passive (ρ_T0 = 10.47, 64/64 solves diverge) even with corrected boundary state, support and 25× richer excitation; the declared conditional kernel K_r1P (uniform-field passivity hinge, otherwise identical) has ρ = 0.68, converges on 64/64 views in 6–7 iterations at no loss of probe fidelity (24.36 vs 23.98 dB); with bit-identical parameters the state-only incremental update (0.63 % of rays) equals a full rebuild to 8e-6 dB, F = 2.44, R_RNA = 0.65, signed change on A_changed corr 0.987 / slope 0.948. The −9 dB gap is the intrinsic kernel's own first-event fidelity (K alone vs first-event references 25.05 dB; the composition adds no error): glass, SSS yellow and the box are the failing materials. Gates carried as FAIL: G1 (inside-face 4.93 % = enclosed geometry; agreement 0.962), G2b (quadrature 0.27 / 0.20), G2c (coverage 0.24), G2d (< 28 dB). Properties: C side (current-geometry coupling, exact incremental update, correct signed change) supported descriptively on one scene / one edit; K side passive only under an explicit constraint and not supported for fidelity. Attribution per cause and the open questions are in the worklog; no replacement is recommended and **no later batch is authorised**.

**2026-10-25 (user brief "K-side fidelity attribution and controlled refinement"): DIAGNOSTIC + ONE BOUNDED CORRECTION AUTHORISED (worklog 44).** FNTO and its K/C ownership contract are preserved; the batch (1) attributes the r1 intrinsic-kernel fidelity limit with frozen K_r1 / K_r1P (mandatory red-box control, source-quadrature isolation, signed indirect-component error, excitation span, passivity vs fidelity), (2) only if one cause is supported, predeclares and implements one within-FNTO correction, trains only that candidate and compares it with the unchanged r1 baseline. Not authorised: capacity increases without causal control, tuning on held-out T1, threshold changes, another architecture family, an automatic later batch.

**Worklog 44 result (2026-10-26, protocol `fnto_k44`, rules fixed before results): IMPROVED.** Phase 1 (frozen K_r1 / K_r1P) ruled out source quadrature (+0.01 dB dense), excitation coverage (train ≈ held-out fit; no out-of-span correlation), the passivity hinge and component weighting; target noise costs ~1 dB of the probe metric only. It found two incident-field defects in the r1 supervision: pixel / port light visibility with false shadows at near-coincident Lego geometry (2.7 % of pixel-light pairs vs the renderer; a 0.3 mm near-hit rule leaves 0.47 %) and box point / spot light geometry evaluated in box-local coordinates. The single predeclared correction (renderer-consistent incident field) raised held-out Lego K-only fidelity by +1.56 dB (T1 +1.53; seed spread 0.10 dB), removed 93 % of the light leaked into shadowed opaque pixels and lifted the box from 16.1 to 31.9 dB, with passivity kept (ρ 0.55), all solves converged and parameters bit-identical. Yellow SSS (Burley, red radius 1 m) is unchanged and now carries 78 % of the K-only error; the composition's indirect component over-predicts Yellow (1.74×) and misses Red_Glass, and limits the full image (S_T0 | hist 25.15 dB vs RNA 33.57). The K/C contract is unchanged; the K_k44P recipe (r1 + incident-field correction) is the current FNTO kernel recipe. **No later batch is authorised.**

**2026-10-26 (user brief "FNTO Independent Fidelity Attribution — Parallel Diagnostic Batch"): DIAGNOSTIC BATCH AUTHORISED (worklog 45).** Two tracks on the frozen K_k44P baseline (commit 63e2741): Track A intrinsic K / Yellow SSS, Track B free-space C / K-C composition. Evidence only: no K / C / solver / passivity / visibility / reference change, no training, no correction implemented; oracles are non-deployable controls. Protocol `experiments/fnto_diag45/protocol/diag45.json`.

**Worklog 45 result (2026-10-26, protocol `fnto_diag45` + amendments 1–6, diagnostic only — no verdict gate, nothing trained or corrected).** (i) **C free flight verified** against an independent Blender single-flight oracle on a known emitted field: pooled 1.05×, corr 0.93 (box 1.005, oracle noise 1.3 %); contract limits are enclosed receivers (inside faces carry 0 → 0.65), near-coincident hits (1.16), no transmission path through glass and an isotropic sender evaluated at the port normal. (ii) **Yellow first-event error = K intrinsic**: on the identical verified input a no-learning object-scoped Burley oracle (3 scalars) ties K overall (23.9 [21.4, 26.3] vs 24.7 [24.1, 25.6] dB) and is +1.9 dB better in shadow, where 78 % of the Yellow error lies (3–30 cm from lit ports, red / green); Cycles Burley d = 6.1 / 1.2 / 0.39 cm (worklog 44's "red radius 1 m" overstated the range; 15 % of red mass lies beyond 30 cm). (iii) **The composed Yellow indirect over-prediction (1.74×) is a K–C interface effect**: K's receiver response to verified fields is 0.56–1.37×, but K's sender emission into C at Yellow ports is 0.35–0.41× of the reference at camera-visible ports while its total first-bounce power is 1.8× the whole Lego's sun power in red (beyond any physical bound; Burley energy conservation verified on a plane only); 82 % of the Yellow indirect prediction comes from irradiance held at ports no camera of the view sees (44 % at ports never observed in training, 89 % of whose area was never lit). Opaque receivers respond exactly (counterfactual 0.99); their gathered field is 0.76–0.89×. The solver adds no error (converged, closures 1.000). Edit: direct change corr 0.99 / slope 0.99, opaque indirect change corr 0.92 / slope 0.88, Yellow indirect change slope 0.17. Smallest evidence-supported intervention target (decision input only): the radiance K sends into C at Yellow ports outside supervision. **No later batch is authorised.**

**2026-10-26 (user brief "Worklog 46 — Directional K/C Interface: Component Attribution, Controlled Composition, and Real-scene Validation"): BOUNDED IMPLEMENTATION AUTHORISED (worklog 46).** Investigate and, where supported, implement a directional boundary-transport interface between intrinsic K and free-space C (outgoing sender radiance toward the receiver, directional incoming, directional free-space transfer; sampling only as a separate conditional variable), with component-isolated controls (R0 frozen production; R1 / R2 known-field original vs directional transfer; R3 frozen-K outgoing-aware; R4 incoming-aware controlled response with an analytic control where K44 lacks the capability; R5 combined; R6 conditional sampling), synthetic directional contracts, one Blender/Cycles glossy / mirror interreflection scene with a T0/T1 edit, and the existing Lego / box regression suite. Frozen: K_k44P parameters, the four-component K, ranks / supports / mixture weights / GGX parameters, training data and loss, passivity and evaluation thresholds, materials, reference illumination, historical baselines and worklogs. Not authorised: K redesign or retraining, learned / compressed C, glass transmission, Yellow-specific heuristics, scene-specific calibration, parameter sweeps, a downstream architecture or worklog 47. Results in `results/fnto_diag46/`. No result exists yet.

---

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

**Seed replicate (worklog 38, protocol b4_seedrep_v1, LabServer63; reported 2026-10-22): WITHIN_MARGIN.**
- Mean B_sel − A_full over seeds 0–2 is −0.082 dB (per seed −0.313 / +0.113 / −0.047).
- The unit-3 degradation is not systematic (1 of 3 seeds).
- The A_full seed spread alone is 0.23 dB.
- Every seed keeps V\* bit-identical, output change on the predicted support, and R ≥ 0.98.

Worklog 37's CONDITIONAL verdict is unchanged; its miss does not replicate. Two observations are consistent across seeds and unattributed: frozen V\* unit 9 is 1.22–1.32× A_full, and the re-learned box unit is 1.12–1.18×. Scope: one scene, one diffuse-dominant edit.

**Batch 5 (authorised by the user's brief, 2026-10-22; worklog 39): Spatial-Perspective Transport Factorization, SPT-1.** This is a new architecture hypothesis. It has persistent learned surface fields and heads, a current relation-dependent incident-transport state recomputed from geometry, and a reusable operator that is linear in that state. Edits are absorbed by state recomputation, not parameter updates. Contract: docs/Architecture.md §9a; code: `experiments/spt_lego/`.

**Reported (worklog 39): NOT SUPPORTED.** Static representability is −11.3 dB vs RNA. The dominant cause is object-internal transport (subsurface yellow 70×, transmission 4×), which the free-space relation state cannot express. Descriptively, the state-only edit update moved stationary receivers correctly with bit-identical parameters and no T1 supervision (F = 1.67, R = 0.82; sand R = 0.92). **Batch 6 is not authorised**; the recommended question is a persistent object-internal transport channel alongside the relation channel.
- The Surface-Owned complete-unit contract and its results remain the historical baseline. Do not redefine or retrain them for convenience.
- Do not switch architecture families or rescue a negative result.
- Do not proceed to Batch 6 without a new user decision.

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
