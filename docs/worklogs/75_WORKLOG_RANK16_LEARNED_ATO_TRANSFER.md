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
