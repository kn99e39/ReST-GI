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

**2026-11-06, Stage 3 to export.**
- **References.** 1,408 packets, all at 32M photons, all gates passed, unchanged walker. They took 2,989 s on the local RTX 5080.
- **Qualification.** Every Stage 3 matrix qualified (22/22), as did every DEF4 matrix (6/6).
- **Oracle convergence.** Doubling the budget changed J by at most 2.9e-11 (G0_r0) and 5.6e-7 (G2_r0).
- **Completion.** The chain finished without intervention at about 14:25 KST.

## Final report (2026-11-06)

The batch follows protocol `rank75_v1` (commit d0ded8c) plus amendment 1 (c96349a, local placement, made before any evidence). Nothing was rerun, reselected or tuned.

Unless stated otherwise, values are the worst-channel jackknife upper 95 % bound of the pooled error (eval68 estimators). Model pairs are written AR_5901 / AR_5902 → A75_5901 / A75_5902.

### Implementation Fact

- **Rank-generic class.** `rank75.TransportM` reproduces the historical rank-8 runtime within rel ≤ 1e-6. That covers columns, coefficients, apply, column sums, resident rebuild and the fit73 training objective. A synthetic rank-16 embedding reproduces the rank-8 physical response.
- **Rank-16 candidates.**
  - Basis: frozen G2_r0, chosen by fitting objective only.
  - Initialisation: P_590x with duplication r mod 8.
  - Physics: positivity, zero input, linearity, column sums ≤ 1 and cache identity hold on rest / pt0 / cf3 / cf2 / te2 / te0 / cf1 for both seeds.
  - A/O: bitwise identical to G2.
- **Capacity.**
  - Learned parameters: 155,104 → 273,040.
  - Angular logits: 1,920 → 3,840.
  - W/U/V state floats on rest: 14.7 M → 29.5 M.
- **Unchanged.** Every other contract is the WL73 AR recipe.
- **Fits.** Both converged; each took 4,084 s on two lanes on the local RTX 5080.
- **Placement.** Local PC (amendment 1). The local re-evaluation reproduces the LabServer63 archive within ≤ 3.0e-6.

### Measurement

**Stage 2: supervised sources, held-out replicas 4–7.** FIT GATE PASSED for both seeds. There is no regression; the worst candidate/control ratio is 0.92.

| Group / metric | AR (rank 8) | A75 (rank 16) | G0 oracle | G2 oracle |
|---|---|---|---|---|
| TRAIN_NEW con_coarse | 0.270 | 0.131 | 0.270 | 0.129 |
| TRAIN_NEW abs_coarse | 0.110 | 0.083 | 0.111 | 0.084 |
| TRAIN_NEW abs_native | 0.132 | 0.117 | 0.130 | 0.115 |
| TRAIN_KEEP con_coarse | 0.19 | 0.093 | 0.19 | 0.088 |

**Stage 3: independent WL74 Stage 4 population.** "—" means the oracle value is not quoted here.

| Group / metric | AR_5901 / AR_5902 | A75_5901 / A75_5902 | G0 oracle | G2 oracle |
|---|---|---|---|---|
| **BND4_FAR con_coarse (primary)** | 0.793 / 0.792 | 0.803 / 0.824 | 0.265 | 0.159 |
| BND4_FAR con_native | 0.915 / 0.983 | 0.851 / 0.902 | — | — |
| BND4_FAR abs_coarse | 0.181 / 0.209 | 0.180 / 0.183 | 0.125 | 0.097 |
| BND4_NEAR con_coarse | 0.486 / 0.580 | 0.549 / 0.552 | 0.253 | 0.147 |
| BND4_NEAR abs_coarse | 0.109 / 0.125 | 0.126 / 0.096 | — | — |
| INT4 con_coarse | 0.274 / 0.366 | 0.151 / 0.185 | 0.161 | 0.082 |
| INT4 abs_coarse | 0.064 / 0.068 | 0.043 / 0.044 | — | — |
| INT4 con_native | 0.397 / 0.442 | 0.302 / 0.312 | — | — |
| INT4 abs_native | 0.106 / 0.111 | 0.090 / 0.090 | — | — |
| FP4_cf1 con_coarse | 0.230 / 0.205 | 0.140 / 0.155 | — | — |
| ALL4_rest con_coarse | 0.542 / 0.569 | 0.500 / 0.528 | 0.218 | 0.122 |

**Decision (predeclared): NO INDEPENDENT GENERALIZATION BENEFIT (training-only benefit).**
- far = NO; near = NO.
- Secondary BND4_FAR: con_native, con_fine16 and abs_coarse are NO; abs_fine16 and abs_native are OVERLAP.
- Every regression check is NO, and validity holds.

**WL59 gates and deformation.**
- A75 meets the WL59 gates on INT4 and FP4_cf1. AR meets them on FP4_cf1 only. No model meets them on BND4_FAR or BND4_NEAR.
- DEF4 fails for every model on abs_coarse (≥ 0.26). Correlation is 0.90–0.97 for both ranks. A75 is slightly better in blue (0.47 / 0.40 against AR's 0.56 / 0.41).

**Stage 5 cost.** Measured on the local RTX 5080 in the same process for both ranks: median of 10 runs after warm-up, with desktop display processes present.

| Quantity | Rank 8 | Rank 16 | Ratio |
|---|---|---|---|
| Full coefficient rebuild | 7.0 ms | 8.2 ms | 1.16× |
| Apply, 64 lights | 34.0 ms | 63.0 ms | 1.85× |
| Apply, 1 light | 1.07 ms | 1.37 ms | 1.28× |
| Peak memory above geometry | 290 MB | 449 MB | 1.55× |
| W/U/V state | — | — | 2.0× |
| Parameters | — | — | 1.76× |

- Training: 4,084 s per run on two lanes.
- The rank-independent cf1 geometry build takes 0.7 s.
- G2 angular pretraining took 2,178 s on LabServer63 (WL74). Rank 8 with the WL68 basis needs none.
- No ratio exceeds 2 and there is no transferable benefit, so no cost class is attached.

### Observation

- **Supervised sources.** Rank-16 learned K reaches the G2 source-specific oracle without a predeclared regression. About 99 % of the contrast gap and 100 % of the coarse absolute gap are closed.
- **Independent far-boundary sources.** The error is unchanged. The learned error stays about 3× (AR vs G0) and about 5× (A75 vs G2) the corresponding oracle. The learned-vs-oracle gap at BND4_FAR therefore widens from 0.53 (AR − G0) to 0.65 (A75 − G2).
- **Interior sources (INT4).** Coarse contrast improves about 1.5–2×, and the gain also shows in native and fine16 contrast and in absolute error. INT4 includes footprint sources 2.5–3.5 mm from training-source images. The INT far source s109, 10 mm from any image, improves from 0.60 to 0.29.
- **Per-source context** (con_coarse, mean of the two seeds).

  | Source | AR | A75 | G0 | G2 |
  |---|---|---|---|---|
  | TBC near s102 | 1.17 | 1.35 | 0.14 | 0.11 |
  | SR far s107 | 0.99 | 1.19 | 0.31 | 0.20 |

  These two are the worst sources, and both are corners. Rank 16 improves the TBE, TBC-far, SM, SR-near and INT sources, and worsens s102 and s107.
- **Sub-cell stratification** (descriptive; n = 5 corner / 11 edge / 6 center), con_coarse:

  | Sub-cell | AR | A75 | G0 | G2 |
  |---|---|---|---|---|
  | corner | 0.56 | 0.50 | 0.23 | 0.14 |
  | edge | 0.34 | 0.24 | 0.18 | 0.09 |
  | center | 0.40 | 0.35 | 0.12 | 0.09 |

  The oracle ordering center < edge < corner reproduces WL74A. For the learned models, sub-cell position is confounded with boundary context: s102, the worst learned source, sits at a sub-cell centre.
- **Orientation.** Rank 16 lowers the learned operator's angle-integrated symmetry discrepancy I1, which stays an order of magnitude below the boundary errors.
  - Stage 4 sources, median: 0.028 / 0.027 → 0.015 / 0.017.
  - Stage 4 sources, max: 0.19 / 0.24 → 0.12 / 0.14.
  - Training sources: 0.017 → 0.010.
- **Orientation check (post hoc finding).** The predeclared bin-permutation check holds for only part of the source/operation pairs: 73/240 at Stage 4 and 150/240 at training.
  - The check requires sorted β rows to agree within 1e-6.
  - The β differences have a median of 2e-5 / 8e-7, with a source-dependent tail up to 0.057.
  - So the geometry-only ballistic budget is not exactly permutation-symmetric for some sources. I2/I3 are reported only on the passing pairs.
- **Drift** (descriptive; no checkpoint was selected). This is the WL71A pattern: early transfer, later specialisation.
  - BND4_NEAR contrast worsens during training for both ranks: A75 goes from 0.30–0.33 at 2k to 0.55 final; AR from 0.36–0.38 to 0.49–0.58.
  - A75 BND4_FAR is 0.73–0.77 at 2k and 0.80–0.82 final.
  - Supervised contrast is already 0.137 at 2k.
- **WL73 development groups.** These are not independent, and they behave the same way: BND_FAR 0.87 / 0.94 → 0.94 / 0.95; INT_ALL 0.27 / 0.37 → 0.15 / 0.17.

### Interpretation

1. **Can rank-16 learned K fit the physical responses rank-8 learned K misses?** Yes, at supervised sources. The boundary-source floor of 0.270 (equal to the G0 oracle) drops to 0.131, the G2 oracle level, in both seeds with no regression.
2. **Does the improvement transfer to independent source positions?** Not at the boundary.
   - Far-boundary contrast is unchanged: NO by the predeclared rule.
   - Near-boundary contrast does not improve: NO.
   - Interior and near-pose footprint sources do improve without regression, but they were not the predeclared primary target. Part of INT4 lies 2.5–3.5 mm from training images. The single 10 mm interior source also improves.
3. **Native / fine16 or mostly coarse?** Where the gain exists (interior, footprint), it also shows in native and fine16 contrast and in absolute error. At the far boundary nothing improves materially in any space: the native contrast change of about −0.07 fails the 0.8 rule.
4. **Do orientation and sub-cell position affect transfer?**
   - The oracle error shows the WL74A sub-cell ordering.
   - The learned operator's orientation inconsistency is small (median ≤ 0.03) and decreases at rank 16.
   - Neither explains boundary errors of 0.8–1.3. Boundary context (top/bottom corners, side rim) dominates the learned error, and sub-cell position is confounded with it.
5. **Does rank 16 reduce the learned-vs-oracle gap?**
   - Interior: in absolute terms yes (INT4 con_coarse gap 0.16 → 0.09). The ratio stays about 2×.
   - Far boundary: no. The gap grows from 0.53 to 0.65, because the oracle improves while the learned model does not.
6. **Is the benefit worth the measured runtime and state cost?**
   - The cost is modest and at most proportional: apply 1.85×, rebuild 1.16×, state 2×, parameters 1.76×, memory 1.55×, plus 36 min of G2 pretraining.
   - The benefit that would justify it, independent far-boundary transfer, is absent.
   - What remains is an interior / near-supervised gain at about 1.85× apply cost. Whether that trade is worthwhile is a decision outside this batch's predeclared question.
7. **What does the result establish about global ATO factorisation?** In this regime the global rank-8 basis really did limit supervised and interior fidelity, and rank 16 removes that limit for the learned model, not only for the oracle. It is not the limit on independent boundary transfer. That error is set by the learned geometry → T map, which stays 3–5× the oracle under either basis.
8. **What remains unresolved about the geometry-to-spatial-transport mapping?**
   - Why the W+UV map from intrinsic geometry descriptors to T does not extrapolate to boundary contexts and corners absent from S16-R. TBC near and SR far get worse with more capacity.
   - Whether the interior gain holds far from every supervised image; only one 10 mm source exists.
   - Why near-boundary transfer is lost during training (drift).
   - Whether the ballistic budget's small asymmetry under exact slab symmetries matters.

### Unresolved Question

- The interior benefit rests on 8 sources, 6 of them footprint sources near training images.
- The frozen population has no previously unrepresented (SV) source.
- The Stage 3 references are local while the training references come from LabServer63. Provenance agrees to 3e-6, but the cross-machine reference set is labelled, not tested.
- Rank 16 adds both angular and spatial capacity, and the batch does not separate them (by design).
- The oracle's per-source T is latent data, not a transferable model.

### Architecture Verdict

**NO INDEPENDENT GENERALIZATION BENEFIT (training-only benefit)** for the far boundary, by the predeclared rule.

- **What rank 16 does.** Rank-16 global ATO with the frozen G2 basis is physically valid and the learned W+UV operator can use it at supervised sources. It improves interior and near-pose responses without regression at about 1.85× apply cost.
- **What it does not do.** It does not make learned K more accurate at independent boundary source positions.
- **Where the bottleneck is.** Global angular rank is not the bottleneck for the independent-source transfer gap; the spatial geometry → T mapping is.
- **Adoption.** Rank 16 is not adopted, and no successor is started.
- **Living documents.** Not changed: there is no change to the architecture contract and no adoption, consistent with WL71–74.

### Reproducibility

- **Code.** `experiments/fnto_rank75/`, commit d0ded8c (freeze) + c96349a (amendment 1).
- **Protocol.** `protocol/rank75_v1.json`.
- **Hashes.**
  - G2_r0 sha256: `000fc7aa…`.
  - AR_5901 / AR_5902: `aefabe33…` / `a5dee943…`.
  - ports4: `94dc0479…`.
  - cf1: `dcf780ae…`.
- **Environment.** Local Windows, RTX 5080 16 GB, torch 2.8 cu128 (`external/8dna26/.venv-cu128`). Training and evaluation in float32 with TF32 off; oracle FISTA in float64.
- **Seeds.** Runs initialised from P_5901 / P_5902; references 743/744xxxxxx.
- **Evidence root.** `C:/Users/dna10/rest_rank75`:
  - `runs/`, `stage2/`, `stage3/`, `analysis/`, `cost/`, `probe/`;
  - `refs/` with `complete.json` and `manifest.json` (walker sha);
  - `logs/`, including the GPU sampler.
- **Reviewer copies.** `results/fnto_rank75/` (final/, Stage 2 / 3 / 4 analyses, refs manifest) and `results/evaluation/75/` (figure + manifest).
- **Other roots.** The dev smoke is in `C:/Users/dna10/rest_rank75_dev`; no metric from it was used. The aborted LabServer63 attempt is preserved in `LabServer63:~/rest_rank75{,_dev}`.
