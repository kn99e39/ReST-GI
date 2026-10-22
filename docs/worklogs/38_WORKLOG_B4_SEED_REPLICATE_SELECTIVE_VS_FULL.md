# Worklog — Batch 4 Seed Replicate: Oracle-Selective vs Full Recompute (2026-10-21 – 2026-10-22)

## SESSION QUESTION

Worklog 37 reported Batch 4 as **CONDITIONAL**. Every positive criterion passed except the whole-image one:
B_sel − A_full = −0.313 dB view-mean against a ≥ −0.26 dB rule. 69% of that gap came from one weakly dirty
unit (unit 3), and each arm had a single seed. The user authorised (2026-10-21) the recommended replicate,
which was predeclared as `experiments/rna_selective_reconstruction_lego/protocol/b4_seedrep_v1.json` before any
run:

> Is the whole-image shortfall of oracle-selective recompute vs full recompute systematic, or within run-to-run
> variance? Is the unit-3 degradation systematic?

**Short answer: WITHIN_MARGIN (predeclared rule).**
- The mean gap over seeds 0–2 is −0.082 dB (rule ≥ −0.26); per seed −0.313 / +0.113 / −0.047 dB.
- **Unit 3: not systematic.** B_sel is worse than A_full in 1 of 3 seeds (1.54×, 0.88×, 0.74×).
- **The seed-to-seed spread of full recompute alone is 0.23 dB**, the size of worklog 37's gap.
- Every seed keeps the ownership properties: V\* bit-identical, output change exactly on the predicted
  support, changed-region recovery R ≥ 0.979.

This **does not change worklog 37's predeclared verdict** (CONDITIONAL). It shows that the miss does not
replicate.

## RELATION TO EARLIER WORK

- **Worklog 37 (Batch 4)** is unchanged and is reused as-is:
  - scene, edit, PC-rendered training data and held-out references;
  - oracle D\* = {0, 2, 3, 6, 7, 8, 13, B};
  - SO_T0 (the B_sel init and the frozen arm);
  - trainer, configuration values, PU composer, evaluation code.
- **Seed 0** is worklog 37's A_full and B_sel (trained on the RTX 5080 PC).
- This is a replication of Batch 4's decisive comparison, **not Batch 5**.

## EXECUTION

- **Placement:** LabServer63 (RTX 3080 Ti 12 GB, 125 GB RAM), under the user's 2026-10-21 rule that long jobs
  run on the server. Nothing ran on the PC at the same time.
- **Inputs:**
  - Code: `git archive` of `2630b48`, the commit that declared the protocol.
  - Data: 457 PC-rendered input files (21 GB), sha256-verified on the server before training. No
    server-rendered image was used.
  - First checksum attempt: the list was malformed, because Git Bash `sha256sum` writes `hash *path`. The list
    was rewritten to the server layout and all files passed.
- **Runs:**
  - A_full_s1, A_full_s2, B_sel_s1 and B_sel_s2 were started concurrently (staggered by 2 min) at 14:39 UTC and
    finished between 21:22 and 22:28 UTC. All returned rc = 0 with no traceback.
  - Wall-clock: A_full 7.9 h, B_sel 6.7–6.9 h. These are concurrent runs on one GPU, so **they are not a cost
    measurement**; worklog 37's PC measurement stands alone.
  - Peak torch memory: 1.92 / 1.07 GiB.
- **Evaluation:** one `b4_eval` call on the server at 22:34 UTC, covering the owned arms of seeds 0–2 and the
  frozen SO_T0 on held-out T1.
- **Records:**
  - Copied to `results/rna_selective_reconstruction_lego/seedrep_server/`: eval.json, train/specs records and
    launcher logs.
  - Verdict: `results/evaluation/38/seedrep_verdict.json`, from `b4_seedrep_verdict.py`.

## EVIDENCE (held-out T1, 16 views, 4096-spp PC references, PU, server evaluation)

**Cross-machine evaluation check.** Server minus PC view-mean for the seed-0 arms: SO_T0 +0.0001, A_full
−0.0001, B_sel −0.0001 dB (tolerance 0.01). Pass. The seed-0 numbers below are therefore the same as in
worklog 37.

| seed | trained on | A_full | B_sel | g = B − A | R | V\*-owned B/A | V\* bit-identical | output inside support | unit 3 B/A | unit 3 B/frozen |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | PC | 33.379 | 33.066 | −0.313 | 0.979 | 1.032 | yes | yes | 1.54 | 1.40 |
| 1 | server | 33.146 | 33.260 | +0.113 | 0.985 | 0.982 | yes | yes | 0.88 | 0.94 |
| 2 | server | 33.251 | 33.204 | −0.047 | 0.986 | 1.046 | yes | yes | 0.74 | 0.95 |

- **Mean g:** −0.082 dB; range [−0.313, +0.113]. → **WITHIN_MARGIN.**
- **Seed spread:** A_full 0.233 dB, B_sel 0.194 dB.
- **Frozen transfer (SO_T0 on T1):** 30.72 dB.
- **Flags:** none. No seed has g < −1.0, R < 0.5, V\* not bit-identical, or output change outside the
  support.

## OBSERVATION (descriptive, outside the predeclared rules)

Per-unit B_sel/A_full MSE ratios in `seedrep_verdict.json` (`per_unit_B_over_A`). Two units deviate in the same
direction in all three seeds:

- **Unit 9 (V\*, frozen in B_sel): 1.22 / 1.25 / 1.32.**
  - The frozen T0-trained unit 9 is consistently worse on T1 than a unit 9 freshly trained on T1.
  - Its oracle change RMS 0.0027 is below τ (dirty at τ/2).
  - Its frozen T1 error is 1.01× its own T0 error (worklog 37).
  - Two explanations are possible and are not separated here: sub-threshold stale transport, or a T0-vs-T1
    training-data effect. The pooled V\*-owned ratio stays 0.98–1.05.
- **Box unit B (D\*, re-learned in both arms): 1.14 / 1.18 / 1.12.**
  - B_sel's re-learned box unit is consistently about 15% worse than A_full's, although both re-initialise it on
    the same data.
  - B owns its own connected component (no PU blending with A).
  - Not attributed. Candidate causes are optimisation differences when 9 units are frozen, or initialisation
    order.
  - The box region stays above 40 dB in worklog 37's seed-0 numbers.

## INTERPRETATION

- **Within the evidence of one scene, one edit and three seeds**, the result supports this reading:
  oracle-selective complete-state recompute of the transport-dirty units recovers current-geometry radiance as
  well as recomputing every unit. The whole-image difference is within the variance between full-recompute
  runs. Preservation of the 9 valid units is exact.
- **Worklog 37's "weakly dirty unit 3 re-learned worse" was a single-run outcome**, not a property of selective
  recompute.
- **Not shown:**
  - Generality to other edits, scenes or materials. The scene's transport here is diffuse-dominant; glossy or
    nonlocal multi-bounce interreflection is untested.
  - Any scene-level selector: D\* remains an oracle.
  - A recompute-cost advantage beyond worklog 37's 0.785× wall-clock.
  - The cause of the consistent unit-9 and box-unit deviations.

## NEXT DECISION (recommendation; not implemented)

Batch 5 needs a user decision. The candidates, in the order I would rank them:

1. **Breadth of the physical-edit result.** Run a second edit on a different transport regime: glossy or
   multi-bounce interreflection, or an edit whose dirty set is dominated by stationary receivers far from the
   mover. Use the same oracle protocol and seeds ≥ 2 per arm from the start.
   - This tests whether D_T stays sparse and recoverable before investing in selection.
   - It matches the 2026-10-19 direction: interreflection is a first-class requirement, and a selector only
     follows oracle feasibility and coverage evidence.
2. **Attribution of the unit-9 / box-unit consistency.**
   - Unit 9: is it sub-τ invalidity (which would matter for τ semantics) or a data effect?
   - Box unit: is it an optimisation coupling under frozen units?
   - This is cheap, because the trained states exist, but it is secondary.
3. **Relation-observability / candidate-coverage study** (§3a of Architecture). This should follow 1, so that
   the coverage oracle spans more than one transport regime.

Neither is authorised by this worklog. The server workspace `~/rest_b4rep` is kept for possible reuse.

## REPOSITORY

| purpose | commit |
|---|---|
| protocol b4_seedrep_v1, server chain | `2630b48` |
| living docs: replicate authorised and running | `065526e` |
| verdict script, this worklog, living docs | this session's commit |
