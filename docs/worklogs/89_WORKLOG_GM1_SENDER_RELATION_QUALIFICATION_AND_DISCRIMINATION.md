# Worklog — GM1 Source-independent Directional Relation Qualification and Architecture Discrimination (2026-11-12)

## Origin and scope

- **Brief.** "ReST-GI — GM1 Source-independent Directional Relation Qualification and Architecture Discrimination" (user, 2026-11-12). One bounded batch.
- **Aim.** Preserve the WL88 negative result. On a fresh, independently qualified moving-sharp workload, test whether a source-independent sender-side relation (C3UR) gives physically accurate and economically justified dynamic sharp multi-bounce transport, relative to R0 (WL87 EXP) and R1 (WL88 SAM).
- **Hypotheses (prospective).**
  - H0: the E2 sender-side advantage does not survive.
  - H1: C3UR improves sharp fidelity or time-to-quality without source-specific reconstruction.
  - H2: relation construction / update cost outweighs the benefit.
- **Preserved and unchanged.**
  - WL86 / WL87 / WL88 code, protocols, results and verdicts. They are imported, never edited.
  - WL88 stays STOPPED_WORKLOAD_UNQUALIFIED. Wc / Wh / Wi are not used; Wi is not promoted.
  - Frozen ER. R0 = WL87 EXP.
  - R1 / R1M = WL88 SAM with the WL88 relation parameters, bit-for-bit (no tuning).
- **Not done.** No learned global model, no H-P / subspace, no GM2, no new branch or worktree.
- **Where.**
  - Branch `Selective_Recompute`, LabServer63.
  - `experiments/gm1_relation89/`, `results/gm1_relation89/`, `results/evaluation/89/`.
  - Server `~/rest_rel89`; it reads `~/rest_gm187` (WL87 tables, integration test).
- **Terminology.** All comparators are physical recomputations with the same frozen ER (path tracing, table fill, explicit relations). No external Neural Full Recompute claim is made.

## Design evidence from WL88 and the qualification contract

**WL88 Q5 inconsistency (reviewed).**
- WL88 measured the Q5 SNR floor on its 4-replica qualification partition.
- Q7 and Q8 projected the noise to the 8-replica final references, and WL86 qualified with 8 replicas.
- WL89 removes the inconsistency differently: the qualification partition has the **same budget as the final evaluation references** (8 × 8,192 spp per T, family-tagged REF-ER, seeds 89140+).
- Every floor (half-difference of the replica halves) is therefore measured at the evaluation budget, and "change ≥ 4 × floor" directly means "resolvable in the evaluation".

**Thresholds.**
- The thresholds are the WL86 / WL88 values, unchanged.
- C_min 0.02 is one third of the F2 threshold 0.06: the smallest sharp-region change whose mis-tracking can move E_H across F2.
- The family-share minimum 0.25 requires the targeted S1 → S1 (C_ss) and moving-S1 (R_mirror) families to carry a material part of the change of the higher-order radiance.
- Q8 requires the reference noise in the sharp regions to be ≤ F2 / 3.

**Fresh workload.**
- WL88's candidates showed that the sharp families carried most of the change but the absolute changes were small.
- WL89 uses a larger glossy S1 slab (half 0.05 × 0.55 × 0.50 m) with fresh poses and lights.

**Screening partition (disclosed).**
- Family-tagged REF-ER, 2 × 1,024 spp per T, seeds 89900–89901, on seven configurations (V1–V6, X1). Masks are geometry-only.
- Results:
  - V4 is geometry-inadmissible.
  - V5 and V6 have R_mirror < 1,000 pixels.
  - X1 (held-out edit) fails `geom_ok` at T1 (too close to box B). It is replaced by **X2**, checked by geometry only, with no radiance.
- Projected final SNRs (this partition, crude two-replica floors):

  | Candidate | C_ss change | C_ss snr | R_mirror | D higher-order snr | H2 share |
  |---|---|---|---|---|---|
  | V3 | 0.027 | 11 | 3,244 px | 8.7 | 0.81 |
  | V2 | 0.029 | — | 3,229 px | 8.5 | — |
  | V1 | 0.022 | — | — | 8.4 | — |

- The screening partition was used only to choose and order the candidates (V3, V2, V1; at most three) and is never reused.
- No relation or method output was computed on any candidate before the freeze.

## Candidates and controls (frozen in `protocol/gm1_relation89.json`)

**Candidates.**

| Arm | Construction | Unique traced rays per build | Role |
|---|---|---|---|
| R0 | WL87 EXP: receiver gather on the deployed 512-ray state | 650,240 | preserved baseline |
| **C3UR** | 8 rays per (sender patch, outgoing bin) cell, cos-weighted in the bin, deposited at the receiver's WL87 dense incident cell; one relation reused for all orders | 650,240 | primary new candidate; geometry-only |
| R1 | WL88 SAM | — | frozen comparator |
| R1M | WL88 SAM, matched-ray variant | — | matched-budget comparator |

C3UR details:
- Per-ray stream keyed by the global ray id, so a selective update (rays starting on the mover, or whose segment meets the old / new mover volume) is bit-identical to a rebuild.
- Application aggregates incident radiance per (receiver, cell) and scatters per material by one contraction. This is an implementation choice; test T_C3UR_FORM checks it equals the per-ray form.
- Sphere receivers use the flat patch-centre frame of the dense table, as in WL87 C3 (declared).

**Budget points.**
- 1× unique rays: R0 / C3UR / R1M.
- 2×: POOL2 (R0) and C3UR16 (C3UR with 16 rays per cell).
- R1 is reported at its own budget.

**Controls.**
- C3: WL87 power-proportional, fresh per order, source-dependent diagnostic.
- C3U: sender-uniform, fresh per order.
- Converged relations for target identity: R0_INF (64 realisations), R1_INF (16 × m 16), C3UR_INF (4 × 64 rays per cell).
- RS0 / RS1: **repaired** RAYSEND. Exact radiance is traced from offset origins and validated before use: table-radiance mode must reproduce the first gather; exact traces must hit the recorded sender ≥ 99 %.
- ORACLE: path-traced table, 4,096 spp per slot.
- PT, TF256 / TF1024: physical full-path and table-fill comparators.
- REF-P: ER approximation, reported separately.

## Prospective rules (summary)

- **Stage 0.**
  - Geometry screening, then the qualification partition, with candidates in order. None admitted: WORKLOAD_UNQUALIFIED; no candidate is evaluated and there is no rescue.
  - tests89 = the WL88 tests on the WL89 W1, plus the C3UR and RAYSEND tests. Any failure: STOPPED_INVALID_COMPARISON.
- **Stage 1 (W1 T0 / T1).**
  - E_H and signed error per region; WL86 F1–F3 thresholds, plus F2 / F3 on the sharp regions.
  - Field and per-order errors against path-traced order fields.
  - Validity gate: tests, finite / non-negative tables, target identity (signed ≤ 0.01 + noise, flux within 3 %), RAYSEND repair. A failure stops the batch before economics.
  - Error decomposition per sharp region: sampling (deployed − converged), relation discretisation (converged − oracle), output representation (oracle − reference), first-gather sender projection (arm − RS).
  - Energy-weighted coverage and missing important relations for R0 / C3UR / R1.
- **Stage 2.**
  - 16 realisations each of R0, C3UR and R1: paired differences with 95 % intervals and pass fractions.
  - Held-out edit W2 = X2 and held-out sources LMIX89 / LSPOT89, with relations reused (F4, F2 on sharp regions ≥ 1,000 px, F3 on W2).
  - Edit and source-reuse validity.
- **Stage 3 (alone on the GPU).**
  - Per-edit lifecycle stages A–G, warm, compilation excluded.
  - Source-only update and multi-source totals; edit-to-first-frame and sustained-frame cost at matched quality (ladders 4–16,384 spp × 3; PT, R0, C3UR, R1, R1M, C3, TF).
  - Memory.
  - Scaling in N (s = 1 / 1.5 / 2), peaked coverage (second glossy box), order (1 / 2 / 4 / 8) and source count (1 / 4 / 16).
- **Verdict mapping.**
  - H0 / H1 / H2 and the item verdicts are computed by `analyze89.verdict`, which also attributes failures.
  - Sharp improvement means the paired C3UR − R0 interval is < 0 in a sharp region with no region > 0.
  - Quality-matched advantage means ≥ 1.25× faster edit-to-first-frame at a sharp level, or reaching a level R0 does not.

## Pre-freeze development (non-evidence; `~/rest_rel89dev`)

1. **Screening.** As above. Seven parallel screening processes first exhausted the GPU and were restarted as two processes; no data was lost or reused.
2. **Dev chain end to end** (`WL89_DEV=1`, W1 = W2 = the dev geometry Wk, 64 spp): every stage completed, from geometry screening through qualification, tests, tables, realisations, renders, stage-1 validity gate, economics, scaling, analysis, verdict and visuals.
   - Dev tests: all WL88 tests plus the C3UR tests and the repaired RAYSEND pass.
     - Aggregated contraction vs per-ray form 3e-7.
     - Known-field flux ratio 0.993.
     - Selective = full, null edit and move-and-return identical.
     - RAYSEND table mode 4e-7; exact traces hit the recorded sender 99.9996 %.
   - Dev numbers are noise-dominated and are not evidence.
3. **Implementation optimisation of the C3UR update** (same operator; digests unchanged, selective = full re-verified). Ray records stay GPU-resident and the coefficients (landed rays, slots, weights) are computed in torch.
   - Warm C3UR update on Wk: 53 → 14 ms.
   - R0 state + coefficients: 28 ms. R1 update: 124 ms.

## Evidence run, part 1 (freeze `7515a29`; LabServer63 `~/rest_rel89`; 2026-11-12 03:55–04:35 UTC)

| Step | Outcome |
|---|---|
| Geometry screening | V3, V2, V1 admissible |
| Qualification of V3 (16 renders, 8 × 8,192 spp per T, seeds 89140+) | **ADMITTED**, all of Q1–Q8 |
| W1 = V3; evaluation references, oracle fields, order fields | complete: REF-ER W1 2 × 8, W2 2 × 8, held-out sources 2 × 4, REF-P 4; sha256 in `results/amend1/reference_manifest.json` |
| tests | **GATE TESTS FAIL → `STOPPED_INVALID_COMPARISON`** (04:35:33) |
| verdict in the stop branch | crashed (analysis bug: it expected stage-1 output) → `CHAIN_FAILED` |

**V3 qualification values.**

| Criterion | Value |
|---|---|
| Q1 A | 0.426 |
| Q2 Bset | 0.0145 |
| Q3 C_ss | 0.0234 (13,466 px; H2 share 1.08) |
| Q4 R_mirror | 0.134 (3,244 px; G share 1.43) |
| Q5 D | 0.0153, higher-order ratio 0.45, higher-order SNR 5.5 |
| Q6 H share | 0.25 |
| Q7 uncertainty | 0.0025 |
| Q8 sharp reference noise | 0.0103 |

- The family "shares" exceed 1 because the predeclared share is Σ|Δfamily| / Σ|Δh|, and family changes partly cancel inside h.
- Candidates V2 and V1 were never rendered.

**The single failing gate.** WL88's frozen `tests88.T_EDIT`, `selective_equals_full`, for R1 (SAM) on V3. Every other WL88 test passed, as did every WL89 test (C3UR and repaired RAYSEND).

**No candidate-quality evidence existed or was inspected at the stop.** `results/tables` was empty; no relation table or candidate render had been computed. The original failure artifacts are preserved in `results/orig_stop/`: chain status, both test JSONs, the logs and the freeze commit.

## Failure attribution and the human decision

**First diagnosis (this session, before the human decision).**
- Exactly 2 R1 area samples differ, in `dbin` only, and both are invalid.
- I attributed this to undefined data in inactive lanes. **That attribution was wrong**; see below.
- I asked for a decision, and the user approved option 2 conditionally (brief "Controlled Post-freeze Correctness Amendment and Conditional Evidence Chain Resumption"): amend only if code- and execution-level evidence proves a false positive from non-semantic invalid-lane data, otherwise stop.

**Code-level audit.** These are the read paths of the area-sample arrays:

| Read path | What it reads |
|---|---|
| `rel88.SAM._finish` (`rel88.py:301-310`) | builds every operator tensor (`a_recv`, `a_idx`, `a_w`, `a_cell`, groups) from **valid** samples only |
| `SAM.update` | copies stored arrays by index without reading values; re-traces dirty samples |
| `SAM.nbytes` | sizes only |
| `SAM.digest` | **the only read of invalid entries** |
| RAYSEND (`run88.raysend`, `rel89.raysend_fixed`) | masks by validity |
| coverage (`run88` / `run89.real`) | masks by validity |
| `run88.pair_energy` | a separate diagnostic trace; invalid entries are multiplied by an exact 0 |

The u-ray part (state, MIS factors) does not read area samples at all.

**Execution-level audit** (`audit89.py`; `results/amend1/audit89.json`; frozen evidence parameters).
1. **The two entries.** Pairs (206 floor ← 1260 sphere) and (1053 box B ← 1268 sphere), sample 2.
   - Both are invalid with weight 0 in the selective and the full relation; `cell` and segment are identical; only `dbin` differs (37 vs 51, 54 vs 53).
   - Both are **traced** grazing rays (sender cosine 0.014 / 0.0007) that miss the sender patch.
   - The stored `dbin` is the bin at the first surface hit **beyond the sample segment**:
     - segment lengths 1.02 / 1.35 m (`segment_check.txt`);
     - at T0 the first hit is the back wall (t 4.94 / 4.80 m);
     - at T1 it is the moved slab (t 2.68 / 2.66 m).
   - Validity ("first hit is the sender at y") depends only on geometry inside [x, y]: a flat sender plane is crossed once, at y, and a convex sphere is met only near y. So changes beyond y cannot change validity or contribution, and the segment-based dirty rule correctly did not flag these samples.
2. **Operational state.** The operational digest, every operator tensor, the u state, all segments and the validity mask are bitwise equal between selective and full.
3. **Transport outputs.** Deterministic CPU evaluation of the unchanged operator, selective vs full, for a random field, the W1 q and the 4-order recurrence: bitwise equal, finite, and self-repeat equal.
   - The GPU recurrence differs by 1.6e-7 relative, within its own repeat floor of 3.2e-7: GPU `index_add_` is atomic.
4. **Null edit and move-and-return.** Equal under both digests.
5. **Invalid-lane perturbation** of all 157,538 invalid entries, three ways:
   - `dbin` random;
   - `dbin` 255 and `cell` int32 max;
   - `w` NaN and `cell` random.

   In all three, the operator tensors, the CPU outputs (finite) and a subsequent selective edit T1 → T0 are bitwise unchanged against an unperturbed twin.
   - The first audit run had a test bug: it compared against a relation built on a different scene object, and a fresh `Ctx(T1)` differs from a `set_geom(T0 → T1)` scene in mesh-vertex bits. It is preserved as `audit89_run1_testbug.json`.

Conclusion: the STOP was caused solely by a test that hashed non-semantic stored values of invalid samples. **AUDIT PASS.**

## POST-FREEZE AMENDMENT 1 (protocol `amendments[0]`)

| | Predicate |
|---|---|
| **Old** | `tests88.T_EDIT`: equality of `rel88.SAM.digest()`, a sha256 over the u state, MIS factors, pair list and the **full** arrays valid / dbin / cell / w of all area samples |
| **New** | `tests89.T_R1_EDIT_OPERATIONAL`: equality of `audit89.op_digest`, a sha256 over the u state (hit, dbin, segments), MIS factors, pair list, validity mask, dbin / cell / w of **valid** samples and all area segments; plus bitwise equality of every operator tensor, and null edit / move-and-return under the same digest. Bitwise, no tolerance |

- This changes the test implementation, not transport semantics. The excluded values are never read by the operator, and perturbing them changes nothing.
- `tests88.T_EDIT` still runs and its FAIL is reported. Every other gate is unchanged.
- Also committed: the stop-branch verdict crash fix (`analyze89.verdict`, analysis only).
- Unchanged: candidates, sampling, MIS, material rule, workloads, qualification, references, seeds, thresholds and evaluation.
- **Resume point.** The chain resumes with RESUME at the Stage 0 test gate. Qualification, references and truth fields are reused (sha256 manifest).
