# Worklog — Orbit-Balanced Angular Basis and Source Transfer Validation (2026-11-08)

## Origin

- **Brief.** "Worklog 80 — Orbit-Balanced Angular Basis and Source Transfer Validation", 2026-11-08.
- **Question.** Is the WL79 failure primarily a consequence of orientation-biased fitting of the frozen G2 angular basis? Can correcting that bias improve genuine source transfer within the existing rank-16 W+UV-Q architecture?
- **Hypothesis.** One central architecture hypothesis is tested: a global rank-16 basis fitted across the exact rest symmetry orbits.
- **Learned 2 × 2 (conditional).** It runs only if the representability gate passes:
  - basis G2 vs G1;
  - presentation identity vs exact rest-symmetry.
- **Nature.** Bounded architecture validation. No new angular operator class, no rank increase, no production change, no adoption.

| Item | Setting |
|---|---|
| Branch | `Selective_Recompute` |
| Server | LabServer63 (main) |
| Code | `experiments/fnto_orbit80/` |
| Server root | `~/rest_orbit80` (dev: `~/rest_orbit80_dev`) |
| Protocol | `experiments/fnto_orbit80/protocol/orbit80_v1.json` |

### WL79 carried in (interpretation for this batch)

- **What WL79 rejected.** WL79 rejected the specific frozen G2 basis under its preregistered rule. It did not reject the global rank-16 family: G2 was fitted on the original S16-R orientations, so its original-orientation oracle is in-sample and its transformed oracle is out-of-sample.
- **Scale of the effect.** The oracle orientation gap (1.26–1.41× in coarse metrics on boundary sources) is much smaller than the remaining learned boundary-transfer error of Q78. Closing it is therefore not, by itself, evidence of better learned transfer.
- **Population status.** The frozen WL79 Stage 5 population (19 ports) was never referenced or scored. Its independence is re-verified before any use.

## Stage 0 — Contract reconciliation (`stage0_80.py`)

**Checks.**
- **C1.** The WL79 records are present and unchanged (sha256); the contract is VALID; Stage 1 is ANGULAR BASIS LIMIT; no WL79 Stage 5 references or frozen ports file exist.
- **C2.** Group composition over all 16 × 16 operation pairs, for ports and for incoming and outgoing bins.
- **C3.** Feature-construction non-equivariance, classified (documentation only).
- **C4.** The symmetry classes:

| Class | Transformation | Use |
|---|---|---|
| A | Exact whole-domain D4h | The only class that authorises label permutation |
| B | Approximate local-context equivalence (op + in-face translation; historical pairs) | Diagnostic comparison only |
| C | Translation inside the slab | Not a symmetry |

**Dented poses.** Identity presentation only. The bin-frame convention is unchanged.

### Development run (dev root, before the freeze; geometry only)

- **C2.** 256 / 256 composition pairs exact.
- **C3, probe ties.** Every non-equivariant node input is a probe tie: 32–48 ports per operation, none of class OTHER.
- **C3, neighbourhood ties.** Every unmapped 128-NN edge is a tie at the 128th-neighbour distance, within 1e-7 m (float32 rest coordinates): 1,114–5,852 edges per operation, none of class OTHER.
- **Reading.** Feature-construction non-equivariance is purely index-ordered tie breaking. It is not treated as physical.

## Stage 1 — Orbit-balanced basis (`basis80.py`; frozen in this commit)

### Population

- **Rest.** The S16-R fitting-replica-0–3 means presented under all 16 exact rest operations, using the WL79 maps (`d'` = transformed budget). Images are deduplicated per source by image port.
- **pt0.** The 16 originals only.
- **Size.** 256 targets: 240 rest images + 16 pt0.
- **Shared images.** Three S16-R pairs lie in one orbit: slots (0, 3), (1, 7), (10, 13). Their 48 shared image ports stay separate per source, as the sources are in S16-R. No new physical label is added.

### Weighting (predeclared)

J = 0.5 × rest + 0.5 × pt0.
- Rest term: the mean over the 16 sources of the mean over each source's distinct images. Orbit multiplicity never changes a source's weight.
- Boundary / interior balance stays at S16-R's 4 / 12.
- Per-image objective: the WL68 objective of `ato74.objective_m`.

### Operator and constraints (unchanged)

- Rank 16.
- A = softmax over r; O = softmax over o.
- T ≥ 0 on the capped simplex.
- 16 / 64 bins.

### Algorithm (the WL74 G-arm schedule)

- 30 rounds, each consisting of 30 FISTA steps of T followed by 30 Adam steps of the shared logits (lr 0.01).
- Then a 3,000-step T polish; float64.
- The A/O step uses exact sufficient statistics of the fixed T. The identity was checked on real targets: 3.3e-16 relative. T is streamed in chunks of 16.

### Restarts (fixed before any result)

| Restart | Initialisation |
|---|---|
| r0 | The frozen G2 basis |
| r1 | N(0, 1), seed 8001 |
| r2 | N(0, 1), seed 8002 |

### Selection and alignment

- **Selection.** The lowest final fitting J (fitting replicas only).
- **Alignment.** Mixtures are re-indexed onto G2 by the optimal joint-pattern assignment. This is an exact operator symmetry; it only fixes the latent-coordinate identity used by the Q78 initialisation rule.

### Development probe (non-evidence)

- Sufficient-statistics identity: 3.3e-16.
- FISTA: 8.9 ms per target-iteration (float64, chunk 16, 4.66 GB).
- Estimate: about 2.5 h per restart.
- A 2-iteration dev fit and the selection path ran. No metric is used.

## Stage 2 — Oracle representability gate (`oracle80.py`; frozen in this commit)

**Method.** The WL79 Stage 1 oracle code, unchanged, with only the basis swapped.
- **G0.** The WL79 G2 record is reused (sha256 873649c6…).
- **G1.** The full oracle: 16 S16-R rest sources × (identity + r90, r180, mx, sw, id+z, r90+z); T fitted on replicas 0–3, scored on replicas 4–7.
- **Non-selected restarts.** Identity, r90 and r180 only (descriptive).

**Gate.** Groups ALL / BND / INT; metrics abs_native, abs_fine16, abs_coarse, con_fine16, con_coarse; factor 1.25.

| Condition | Requirement |
|---|---|
| (a) WL79 rule on G1 | G1 transformed / G1 original ≤ 1.25 |
| (b) Original-orientation preservation | G1 original ≤ 1.25 × G0 original |
| (c) Orbit level | G1 transformed ≤ 1.25 × G0 original |
| WL59 gates | Each gate passed by the reference oracle is not newly failed |
| Convergence | ≤ 1e-3 |

- **Pass.** RANK-16 ORBIT REPRESENTABILITY SUPPORTED. Since G0 failed the same rule, this also gives ORIENTATION-BIASED BASIS SUPPORTED. Then Stage 3.
- **Fail.** Stop before learned training. The cause is attributed as far as the evidence allows.

**Execution of chain A.** Started on LabServer63 from commit 468c960 (`~/rest_orbit80/code_commit_A.txt`). Sequence: Stage 0 → population → restarts r0 ∥ r1 → r2 → selection → Stage 2 oracle → gate.

## Stages 3–6 — Learned 2 × 2 and downstream rules (frozen in the next commit, while the Stage 1 restarts run; no basis or candidate result exists)

### Stage 3 (`fit80.py`)

**Design.** Unchanged W+UV-Q (`q78.TransportQ`); two factors:

| Factor | Level 0 | Level 1 |
|---|---|---|
| A: frozen basis | G2 | G1 (G2-aligned) |
| B: training presentation | Historical | Exact rest-symmetry: `fit79.SymRest`, the WL79 Latin schedule; pt0 identity |

- **A0B0** is the frozen Q78 control and is never retrained.
- **New arms.** A0B1, A1B0 and A1B1 × seeds 5901 / 5902: six fits.

**Initialisation.** The Q78 rule (the trunk is bitwise Q78's initial trunk), then the arm's frozen logits.

**Unchanged.** Objective, optimizer, schedule (20,000 steps), final iterate, and the supervision budget.

**Disclosed confounds.**
- B1 changes the effective training distribution.
- A1 changes only the basis values.

### Stage 4: supervised fit gate (per arm; both seeds)

≤ 1.25 × max(Q78 seeds):
- groups: TRAIN_NEW / KEEP × rest / pt0;
- metrics: abs_native, abs_fine16, abs_coarse, con_fine16, con_coarse;
- plus validity.

### Stage 4: mechanism gate (per arm; both seeds)

**Primary endpoint.** Exact-symmetry response consistency on out-of-orbit development sources (the WL74 S4 rest ports and the WL77A Part 0 rest ports):
- for each of the 15 operations, the near-field (0–5 mm) relative discrepancy between the prediction at g(p) and the exact image of the prediction at p;
- native and fine16 contrast, worst channel, mean over (source, operation).

**Threshold.** ≤ 0.5 × same-seed Q78.

**Descriptive.**
- The same quantity on the training sources (in-orbit), and coarse contrast.
- Orientation-specific error on the exact images of held-out training replicas.
- Class-B historical pairs, kept separate from the exact orbits.
- Part 0 groups / DEF.

**If no arm passes both gates.** Stop.

### Stage 5: independent source transfer

**Population.** The WL79 frozen population, re-derived and checked by `ports80.py`:
- identical list;
- never referenced;
- outside the training orbit and outside earlier scored ports and their images.

**References.** Generated only if at least one arm passes both gates. WL79 predeclared seeds; K = 4.

**Who is scored.** Every arm that passed the fit gate, against Q78, A75 and AR.

**Per-arm far rule.** ≤ 0.8 × both seeds of the stronger control and below its lower 95 % bounds, with no regression YES.

**Factorial classification (predeclared, no winner selection).**

| Label | Condition |
|---|---|
| BASIS CHANGE | A1B0 benefits |
| SYMMETRY PRESENTATION | A0B1 benefits |
| INTERACTION | A1B1 benefits and (no single-factor arm benefits, or A1B1 is ≥ 20 % better than each) |
| INDEPENDENT SOURCE TRANSFER SUPPORTED | Any arm benefits |
| ORBIT-ONLY TRANSFER | Every scored arm is far NO after passing the mechanism gate |
| INCONCLUSIVE | Otherwise |

### Stage 6: deformation and cost

- **Deformation.** DEF71, CONFDEF72, DEF73 and Part 0 DEF te1 (primary).
  - WORSE: error ≥ 1.25 × max Q78, or correlation ≤ min Q78 − 0.03.
  - RECOVERED: the symmetric criterion.
- **Cost.** Q78 vs each arm on the same GPU, including angular pretraining and the symmetry presentation.
- **When.** Always run. Descriptive after a stop.

**Chain B.** Launched from commit 189038e. It waits for chain A.

### Execution deviation 1 (2026-11-08 08:52 UTC; no rule changed)

**What happened.** Restart r0 failed with CUDA OOM in round 1.

**Cause.**
- r0 and r1 ran concurrently and together occupied 11.8 / 12.3 GB.
- I started a CPU-intended pre-check of the decision-logic tests on the dev root. It loaded a geometry onto the GPU. Its own allocation failed, but it took the last headroom.
- This was my error. No GPU work is started while the fits run.

**Effect on r1.** None; it continued unaffected.

**Recovery.**
- The failed r0 log is preserved as `logs/basis_r0_failed_oom.log`.
- r0 is rerun from its initial state, alongside r1. The fit is deterministic: initialisation from G2, FISTA with no randomness, Adam with no randomness.
- `resumeA.sh` reruns r0, waits for r1, then reruns `runA80.sh`, which skips the completed steps.
- `resumeB.sh` relaunches chain B.

### Execution deviation 2 (2026-11-08 ~09:30 UTC; execution only, no rule changed)

**What happened.** The original r1 also failed with CUDA OOM, in round 4.

**Reading.** r0 and r1 running concurrently (11.8 / 12.3 GB) do not fit on the 12 GB GPU, with or without the foreign allocation of deviation 1.

**Change.** The predeclared "r0 ∥ r1" lane arrangement is replaced by serial execution: r0 (rerun, running alone at about 87 s per round), then r1 (rerun), then r2 (`resumeA2.sh`, then `runA80.sh`).

**Why results are unaffected.** Each restart is a deterministic computation from its predeclared initialisation. Only the wall-clock changes, to about 2.6 h per restart.

**Preserved.** The failed r1 log, as `logs/basis_r1_failed_oom.log`.

## Execution (2026-11-08 UTC, LabServer63, RTX 3080 Ti 12 GB, torch 2.3.1)

**Chain A.** Ran from commit 468c960, with the two execution deviations above. Steps:

| Step | Result |
|---|---|
| Stage 0 | VALID |
| Population | 256 targets |
| Restarts (serial) | r0 8,322 s, r1 8,118 s, r2 8,166 s |
| Selection | r0 |
| Stage 2 oracle | G1 11,400 s; r1 / r2 descriptive 5,185 / 5,188 s (concurrent with G1) |
| Gate | **ORBIT-BALANCED BASIS INADEQUATE** at 18:54 UTC |

**Chain B** (commit 189038e). It read the Stage 2 gate, did not train, and exported `final/` at 18:55 UTC.

**Not executed.** By the gate, Stages 3–6 did not run: no learned arm, no mechanism evaluation, no Stage 5 references, no deformation or cost measurement. The WL79/80 population (19 ports) **remains unreferenced and unscored**.

## Final report

### Implementation Fact

**Implemented and run.**
- `stage0_80.py`: contract reconciliation.
- `basis80.py`: the orbit-balanced global rank-16 basis.
  - Population: 240 exact rest D4h images of the 16 S16-R fitting means, plus 16 pt0 originals.
  - Weighting: per source.
  - Fit: the WL74 block-coordinate schedule. The A/O step uses exact sufficient statistics (identity 3.3e-16). Float64.
  - Restarts: G2 initialisation / seed 8001 / seed 8002.
  - Selection: by fitting J, then aligned to G2.
- `oracle80.py`: the unchanged WL79 oracle with the basis swapped, plus the predeclared gate.

**Written, tested where possible, never run for evidence.** The learned 2 × 2 (`fit80.py`), `eval80.py`, `ports80.py`, `probes80.py`, `cost80.py`.

**Unchanged.** Q78 / A75 / AR, the G2 basis, every WL79 record, production FNTO, and the bin-frame convention.

### Measurement

**Stage 0 — VALID.**
- WL79 records are unchanged and its population is unreferenced.
- Group composition: 256 / 256 pairs exact.
- Every non-equivariant W+UV-Q input is an index-ordered tie:
  - probe ties: 0–48 ports per operation;
  - 128-NN boundary ties: 1,114–5,852 edges per operation;
  - nothing in class OTHER.

**Stage 1 — basis restarts.**

| Restart | Final J (orbit population) | rest / pt0 |
|---|---|---|
| r0 (from G2) | **0.01683** | 0.01728 / 0.01637 |
| r1 (seed 8001) | 0.02011 | 0.02021 / 0.02002 |
| r2 (seed 8002) | 0.02187 | 0.02196 / 0.02179 |

- **Restart spread:** 0.30. The random restarts converge to clearly worse solutions.
- **Selected basis:** G1 = r0.
  - A and O normalised to 4e-16; min A / O 5e-7 / 2e-7 (positive).
  - Change vs G2: A 0.36, O 0.22 (relative).
  - Algebraic closure still absent: A 0.60, O 0.55, joint median 0.56 / max 0.67 (G2: 0.52 / 0.46 / 0.53 / 0.62).

**Stage 2 — oracle (G1 vs the WL79 G0 record).** The convergence change was 1.5e-4.

*Orientation gap of G1 itself.* Transformed / own original is **≤ 1.15** for every operation, group and metric:
- BND con_coarse 1.11–1.15 (G0: 1.05–1.41);
- BND abs_coarse 1.01–1.14 (G0: 1.05–1.27);
- INT 0.98–1.01.

*Original-orientation accuracy (G1 vs G0).* Lower, but within the 1.25 rule:

| Group | Range |
|---|---|
| ALL | 1.06–1.15 |
| BND | 1.02–1.10 |
| INT | 1.03–1.18 |

The worst case is INT abs_coarse, 0.044 → 0.052.

*Transformed level vs the G0 original:* 1.08–1.27.

| Quantity | G0 transformed (WL79) | G1 transformed |
|---|---|---|
| BND con_coarse (G0 original 0.129) | 0.136–0.183 | 0.147–0.164 |
| BND abs_coarse (G0 original 0.084) | 0.089–0.107 | 0.091–0.103 |

*Gate failures (5).*

| Failure | Values |
|---|---|
| (c) BND con_coarse, r90 | 0.163 / 0.129 = 1.258 × G0 original (rule 1.25) |
| (c) BND con_coarse, r90+z | 0.164 / 0.129 = 1.266 × |
| (a) WL59 BND abs_coarse, r90 / r180 / id+z | transformed 0.101 / 0.103 / 0.103 > 0.10, while the G1 original (0.090) passes |

Every other check passed:
- original-orientation preservation (b), including the WL59 gates;
- the own-orientation ratio (a);
- INT, ALL, sw and mx.

**Classification: ORBIT-BALANCED BASIS INADEQUATE** (stop before learned training).

*Non-selected restarts (descriptive; identity / r90 / r180).*

| Restart | BND con_coarse | ALL abs_coarse |
|---|---|---|
| r1 | 0.150 / 0.154 / 0.158 | 0.071–0.072 |
| r2 | 0.147 / 0.158 / 0.161 | 0.082–0.092 |

Both have small orientation gaps but worse absolute levels than G1.

### Observation

- **The gap shrank but did not close.** Orbit-balanced fitting cut the orientation gap of the rank-16 basis by more than half: the worst own-orientation ratio fell from 1.41 to 1.15. The original orientations got slightly worse at the same time (2–18 %, largest for interior coarse metrics).
- **Where it falls short.** Boundary rotations (r90, r90+z) and the WL59 coarse-absolute gate (r90, r180, id+z). The margins are small: 1.258–1.266 vs 1.25, and 0.101–0.103 vs 0.10.
- **Better G1 restarts were not found.** The two random restarts converged to bases with lower orientation gaps but worse absolute accuracy (J 0.020 / 0.022 vs 0.017).

### Interpretation (inference, labelled)

1. **Part of WL79's failure was orientation-biased fitting.** Fitting the same rank-16 family on the exact orbit halves the measured gap, so part of the WL79 failure came from fitting G2 on the original orientations only. The bias is not the whole story: with balanced fitting, the global rank-16 basis spreads its fixed capacity over 15× more angular patterns. Original-orientation accuracy then drops, and the boundary-rotation images still sit just outside the rule.
2. **No single global basis passed.** Among the three restarts, none represented the orbit at the G0-original level without regression. This is consistent with a trade-off inside the global rank-16 factorisation (capacity / global sharing) rather than a pure optimisation failure:
   - the fit converged (oracle 1.5e-4);
   - restarts from random starts were worse, not better.
   Because the problem is nonconvex and only three restarts were run, an optimisation contribution cannot be excluded.
3. **The margins are thin.** The failures are marginal. The batch's conclusion is negative under the frozen rule, not by a wide margin.
4. **What is not tested.** No learned model was trained, so nothing is known about whether a better-balanced basis would change learned transfer. The brief also notes that the oracle gap is much smaller than Q78's learned boundary error, so even a passing basis would not by itself have shown better transfer.

### Unresolved Question

1. Is the residual (boundary-rotation coarse contrast; coarse-absolute at about 0.10) a capacity limit of global rank 16, a consequence of sharing one basis across all orientations, or an optimisation effect? More restarts or other balance weights would partly separate these. Not authorised here.
2. Would an explicitly equivariant angular factorisation (a basis that transforms with the bin frame) avoid the trade-off? That is a new operator class and a user decision.
3. The learned 2 × 2 questions (basis effect, presentation effect, interaction, independent transfer, deformation, cost) remain unmeasured.
4. Dented-pose angular symmetry labels are still unavailable under the current bin convention.

### Architecture Verdict

**ORBIT-BALANCED BASIS INADEQUATE.** Stage 2 stop, under the predeclared rule.

- **Evidence that orientation bias contributed:** the orientation gap fell from 1.41 to 1.15. By the predeclared mapping, "ORIENTATION-BIASED BASIS SUPPORTED" required the gate to pass, so it is not claimed as a verdict. The halving is reported as a measurement.
- **RANK-16 ORBIT REPRESENTABILITY: NOT SUPPORTED** at the frozen thresholds; the failures are marginal.
- **Not measured** (by protocol): basis-change benefit, presentation benefit, interaction, transfer, deformation, cost.
- **Adoption.** Nothing is adopted. No successor batch is started. WL79 and earlier verdicts are unchanged.

### Answers to the brief

1. **Does orientation-balanced rank-16 fitting remove WL79's transformed-oracle gap?** It reduces it from at most 1.41 to at most 1.15 (own-orientation ratio). It does not remove it: BND con_coarse under r90 / r90+z is 1.26–1.27 × the G0 original level, and BND abs_coarse is 0.101–0.103 > 0.10 under r90 / r180 / id+z.
2. **Does it preserve original-orientation fidelity?** Within the 1.25 rule and all WL59 gates, yes. But there is a measured regression of 2–18 %; the worst case is INT abs_coarse, 1.18 ×.
3. **Is global rank 16 empirically sufficient for the tested orbit?** Not at the predeclared thresholds (5 marginal failures, all boundary). Interior and the diagonal reflection are adequate.
4. **Does changing A/O improve learned transport, independent of symmetry exposure?** Not measured (no training, by protocol).
5. **Does symmetry exposure add benefit?** Not measured.
6. **Are the benefits confined to the training orbit?** Not measured.
7. **Does independent boundary-far source generalisation improve?** Not measured. The population remains unscored.
8. **Does deformation fidelity regress or recover?** Not measured.
9. **Is the benefit worth the optimisation and runtime cost?**
   - Measured cost: angular pretraining about 2.3 h per restart (float64, 256 targets, about 6.8 h for three); Stage 2 oracle 3.2 h for G1.
   - The runtime is unchanged by construction (same operator). No benefit was established to weigh against it.
10. **Global ATO or an explicitly equivariant angular factorisation?** Global rank-16 ATO reduces but does not remove the orientation gap without regression. The evidence is consistent with a trade-off from sharing one global basis across orientations. Whether to consider an equivariant angular factorisation is an open user decision; the evidence does not decide it, and the failures are marginal.

### Reproducibility

**Commits.**

| Commit | Content |
|---|---|
| 468c960 | Stages 0–2 freeze |
| 189038e | Stages 3–6 freeze |
| 27faf01, 0f10a4b | Execution deviations |

The deployed commits are in `results/fnto_orbit80/code_commit_*.txt`.

**Inputs.**

| Input | Identity |
|---|---|
| G2 basis | `G2_r0`, sha256 000fc7aa… |
| WL79 G0 oracle record | sha256 873649c6… |
| `fit_targets73.pt` | sha256 20f631ed… |

**G1 provenance.** sha256 and logit hashes in `basis/select.json`, with the per-restart histories.

**Environment.** Float64 for the basis fit and oracle; eval68 estimators. LabServer63, RTX 3080 Ti 12 GB, torch 2.3.1+cu121.

**Outputs.**
- `results/fnto_orbit80/final/` (manifest), including the Stage 0 record, basis selection, the G1 oracle and closure, the descriptive restart oracles, the gate and the summary.
- Server root `~/rest_orbit80`. The failed OOM logs are preserved.
