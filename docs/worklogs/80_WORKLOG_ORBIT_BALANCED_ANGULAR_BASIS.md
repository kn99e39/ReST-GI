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
