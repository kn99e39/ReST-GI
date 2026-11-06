# Worklog — Rank-16 Learned ATO Transfer Falsification (2026-11-06)

## Origin and authorisation

The user pasted the WL75 brief "Rank-16 Learned ATO Transfer Falsification" (2026-11-06). The batch is a bounded falsification and not a rank-expansion adoption. It asks whether the WL74 rank-16 global ATO basis, used inside the existing W+UV learned intrinsic K, improves accuracy at independent source positions, and at what measured cost.

- Branch: `Selective_Recompute`; no new branch or worktree.
- Server: LabServer63 (main).
- Code: `experiments/fnto_rank75/`.
- Protocol: `protocol/rank75_v1.json`.
- Design record: `docs/plans/RANK16_LEARNED_ATO75_V1.md`.
- Evidence root: `LabServer63:~/rest_rank75`.
- Reviewer copies: `results/fnto_rank75/` and `results/evaluation/75/`.

Not authorised:
- rank above 16 or a rank sweep;
- a coverage change, or loss / schedule tuning;
- geometry-conditioned A/O, angular canonicalisation, or a new spatial kernel;
- per-port or grid corrections;
- K/C or runtime changes;
- held-out selection;
- coupled GI;
- adoption or a successor batch.

## Design (frozen before any WL75 candidate metric or Stage 3 reference)

**Intervention.** The one change is rank 8 → 16 with the frozen WL74 G2 basis.
- **Basis.** G2_r0 was selected by the lowest WL74 fitting objective J_final (0.0146; r1 0.0203, r2 0.0212), read from the basis files only. Its sha256 is `000fc7aa…`. The stored float64 logits are rounded once to float32.
- **Rank-dependent parameters.** Only `edge[-1]` (M·C outputs), `node_head[-1]` (U, V, mass for M mixtures) and the in/out logits change.
- **Capacity.** Learned parameters rise from 155,104 to 273,040. Angular logits rise from 1,920 to 3,840. The W/U/V state doubles.
- **Unchanged.** Everything else is the WL73 AR recipe:
  - S16-R fitting replicas 0–3;
  - J + 1.0·L_sym;
  - 'net' group (A/O frozen);
  - Adam, cosine 3e-3 → 3e-5, 20,000 steps, clip 1;
  - checkpoints 2k / 6k / 10k;
  - the final iterate is the result.
- **Initialisation (by construction).** The trunk is P_590x, bitwise. The rows of each rank-dependent mixture r copy P_590x mixture r mod 8, which is the duplication G2_r0 itself was started from. No channel is initialised randomly.
- **Controls.** Frozen WL73 AR_5901 / AR_5902, plus labelled source-specific oracles G0_r0 (rank 8) and G2_r0 (rank 16).

**Stage 2 training-fit gate.** Applied to each rank-16 seed, on S16-R held-out replicas 4–7, using worst-channel upper 95 % bounds.
- **Capacity.** con_coarse on TRAIN_NEW_rest and TRAIN_NEW_pt0 must be ≤ 0.8 × the better AR seed. AR sits at the G0 oracle floor of 0.270 there, so meeting this requires going below the rank-8 fixed-basis floor.
- **No regression.** Absolute native / fine16 / coarse error on TRAIN_NEW and TRAIN_KEEP, and TRAIN_KEEP contrast, must be ≤ 1.25 × the worse AR seed.
- **Validity.** Passivity must hold everywhere, and A/O must be bitwise equal to G2.
- **Provenance.** The recomputed AR and oracle values must equal the archived WL73 / WL74 values within 1e-3.
- **Stop rule.** If the gate fails, the outcome is INSUFFICIENT FIT and the batch stops before any reference is generated.

**Stage 3 independent confirmation.**
- **Population.** The WL74 Stage 4 population, frozen geometry-only before any WL74 evidence:
  - TBE / TBC / SM / SR near and far;
  - INT near and far;
  - 6 cf1 footprint sources;
  - 1,408 packets in total.
- **Reference code.** WL74's own probe code, unchanged.
- **Primary test.** BND4_FAR con_coarse. The result is YES only if both rank-16 seeds are ≤ 0.8 × the better AR seed and below the AR lower 95 % bound. It is NO if both are above 0.8 × the worse AR seed. Anything else is OVERLAP.
- **Regression checks.** Three-valued checks at ×1.25.
- **Reported separately.** The WL59 absolute gates and DEF4.
- **Outcome classes.** Defined in the protocol.

**Stage 4 (descriptive) and Stage 5 (cost).** See the protocol.

**Independence notes.**
- WL74 never generated Stage 4 references.
- The WL74 dev smoke (`~/rest_ato74_dev`) did create 0.5M-photon dev packets for this population, used only by its forced 40-step dev path. Nobody read a metric from them, and WL75 does not read them.
- The frozen population contains no previously unrepresented (SV) source. That stratum exists only as the WL73 development group UNS.

## Chronology

**2026-11-06, Stage 0 (pre-freeze, non-evidence).** `test75.py` ran on LabServer63 from the working tree, and 8/8 tests passed:
- rank-8 reproduction (columns, coefficients, apply, column sums, `runtime.full`; rel ≤ 1e-6);
- rank-16 embedding equivalence;
- physics contract at rank 16;
- objective identity with fit73;
- basis provenance and selection;
- initialisation and frozen A/O;
- Stage 3 isolation;
- decision logic on synthetic records.

A dev-smoke chain was then started on the working tree, before the freeze, to exercise every stage path.

**2026-11-06, first dev smoke attempt (pre-freeze, non-evidence).**
- **What failed.** The test stage stopped: `test_objective_identity_rank8` raised a CUDA OOM. The other 7 tests passed.
- **Cause.** Three TransGI training jobs (another project; `diag_bal_*`, 30k iterations, about 8.3 GB of 12 GB) had started on LabServer63 a minute before the attempt. The same test had passed when run alone.
- **Decision.** The user was asked and chose 「63 비면 자동 시작」: other projects' jobs are not touched. `run75.sh` now waits until GPU used memory is below 1 GB on two checks 60 s apart, both before the dev smoke and before the evidence chain.
- **Freeze.** Because of the wait, the code and protocol are committed now, before the dev smoke, as in WL74. A dev-smoke failure will be handled by a labelled pre-evidence amendment.

**2026-11-06, amendment 1: local placement (before any evidence).**
- **Status before the change.** The 63 wrapper was still waiting for a free GPU when the user asked whether the batch could run locally.
- **Decision.** User: 「기본 머신은 63번이되, 내 별도 요청이 있을 땐 예외 처리가 가능하다고 했잖아. 이번 75번 Batch에 한해서 로컬에서 진행시키도록 해」. The 63 wrapper was stopped while it was still waiting; nothing had run on 63 beyond the failed dev-smoke test stage.
- **Inputs.** About 41 GB of read-only inputs are copied from LabServer63 into the same directory names under the local home:
  - the WL68 inputs and held-out bundles;
  - the WL71/72/73 reference bundles;
  - S16-R fit targets, the AR runs and the WL73 analysis;
  - the WL74 bases, ports4 and cf1.
- **Execution-only code changes:**
  - Windows stubs for `resource`/`fcntl`;
  - a Python GPU sampler;
  - `run75_local.sh`, a Git Bash wrapper with no GPU wait.
- **Consequences** (protocol `amendments[0]`):
  - The Stage 3 references come from the local machine. The training/development references and the AR controls come from 63, so the reference set spans two machines; this is labelled.
  - The AR controls and the oracles must reproduce the archived 63 values within 1e-3.
  - Stage 5 costs are measured on the RTX 5080 for both ranks.

**2026-11-06, local dev smoke and evidence start.**
- **Copy check.** The inputs (185 files, about 41 GB) match the LabServer63 listing in path and size. The protocol-listed hashes match (G2_r0, ports4, cf1, AR_5901/5902).
- **Launch.** The wrapper started at 10:50 KST (detached Start-Process) with commit c96349a.
- **Dev smoke** (local root `~/rest_rank75_dev`, 40-step fits, 0.5M photons on the dev slots, Stage 3 forced; no metric used). It completed every stage:
  - tests 8/8 on torch 2.8 / RTX 5080;
  - probe;
  - fits;
  - Stage 2: provenance reproduced the 63 archive within ≤ 2.3e-6 (AR_5901/5902, G0_r0, G2_r0). The gate said INSUFFICIENT FIT, as expected for 40 steps;
  - generate and bundle;
  - Stage 3: INCONCLUSIVE, because the subset leaves the primary groups unmeasured;
  - analysis, cost and export.
- **Observation from the dev path (descriptive analysis only).** The orientation analysis's bin-permutation check passes for only part of the source/operation pairs. The check compares sorted β rows with tolerance 1e-6. Pairs passing: 73/240 at the Stage 4 sources, 150/240 at the training sources. The per-pair differences are stored, and the cause will be examined in the final report.
- **Evidence chain** (root `~/rest_rank75`):
  - tests 8/8;
  - probe at two lanes: 0.20 s/step, peak 4.3 GB reserved per run, extrapolated 1.1 h per fit;
  - fits started at 11:07 KST.

**2026-11-06, fits and Stage 2 (evidence, local RTX 5080).**

**Fits.** Both fits converged by the WL68 rule. Each took 4,084 s on two lanes; at step 12,000 the training objective J was about 0.0158.

**Provenance.** The local re-evaluation reproduces the LabServer63 archive within ≤ 3.0e-6 (AR_5901/5902, G0_r0, G2_r0; tolerance 1e-3).

**Stage 2 gate: FIT GATE PASSED for both seeds.**
- **Capacity.** TRAIN_NEW con_coarse fell from 0.270 (AR, equal to the G0 oracle floor 0.2696) to 0.131 for both seeds and both geometries. That is about 99 % of the AR → G2-oracle gap (G2 0.129).
- **No regression.** The worst candidate/control ratio over the regression list is 0.92 (TRAIN_KEEP_pt0 abs_native), so every regression bound improves rather than worsens.
- **Validity.** Passivity holds on rest/pt0/cf3/cf2/te2/te0, and A/O are identical to G2.

**Gap closed on the other training groups (descriptive).**
- TRAIN_KEEP con_coarse moved from 0.19 to 0.093 (G2 0.088). abs_coarse reached the G2 oracle level. abs_fine16 closed about 0.8–0.9 of its gap and abs_native about 0.7.

**WL73 development groups (descriptive; these sources were already used in WL73/74 and are not independent).** Values are worst-channel hi for AR_5901 / AR_5902 → A75_5901 / A75_5902.

| Group | Metric | AR_5901 / AR_5902 | A75_5901 / A75_5902 |
|---|---|---|---|
| BND_FAR | con_coarse | 0.873 / 0.936 | 0.939 / 0.947 |
| BND_FAR | abs_coarse | 0.166 / 0.211 | 0.172 / 0.171 |
| BND_NEAR | con_coarse | 0.415 / 0.431 | 0.396 / 0.396 |
| BND_NEAR | abs_coarse | 0.109 / 0.111 | 0.086 / 0.082 |
| UNS | con_coarse | 1.05 / 1.02 | 1.01 / 1.17 |
| INT_ALL | con_coarse | 0.268 / 0.373 | 0.151 / 0.170 |
| FP_rest | con_coarse | 0.239 / 0.238 | 0.135 / 0.135 |
| REM_NEAR | con_coarse | 0.491 / 0.546 | 0.316 / 0.500 |

DEF71 and DEF73 fail for both ranks.

**Next.** The chain continued to the Stage 3 references (1,408 packets) as the protocol requires. The Stage 3 decision is unaffected by the development values above.
