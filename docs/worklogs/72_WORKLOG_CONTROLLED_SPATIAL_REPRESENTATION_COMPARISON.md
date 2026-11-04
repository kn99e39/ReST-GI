# Worklog — Controlled Spatial Representation Comparison (2026-11-03)

## Authorisation and scope

On 2026-11-03 the user issued the brief "Worklog 72 — Controlled Spatial Representation Comparison". It authorises one bounded comparative architecture batch on the WL59 intrinsic-K family:

- **Candidate A** keeps W+UV and adds a physically justified generalisation constraint.
- **Candidate B** is a relational, geometry-conditioned shared transport representation.

Both are compared with the preserved WL59/WL71 baselines under one physical contract. The batch includes independent confirmation references on new ports and a previously unscored geometry.

**Preserved:**

- K/C ownership;
- first-exit and ballistic semantics;
- the fixed A/O for the primary comparison;
- 16 incoming / 64 outgoing bins;
- positivity and passivity;
- the thresholds;
- historical checkpoints.

**Not authorised:**

- automatic adoption;
- coupled GI;
- new angular mechanisms or materials;
- coverage increases after results;
- per-port corrections;
- another research hypothesis.

**Parallel work.** The user chose to run the WL57 walker GPU-efficiency rework (WL71C) alongside this batch. WL72 uses a reworked walker only if the rework passes its equivalence gates before WL72 reference generation starts; otherwise it uses the unchanged walker (see the protocol).

**Population status.** WL66/68/71 populations, WL71 fits and WL71A/B results are development evidence here.

**Server note.** At start, LabServer63's GPU was in use by another project's job (`~/TransGI-Reproduction`, `gen_data.py`, about 0.6 GB). This batch does not touch it.

## Execution log

**2026-11-03.** The protocol and code were frozen at 80c9b26 and deployed to LabServer63. `run72.sh` was launched detached. It waited for the TransGI job; the dev smoke started once the GPU had been idle for 10 minutes.

**2026-11-03 11:18 UTC — dev smoke failed at the `ports` stage.**
- The te2 identity gate (bitwise) failed on `node`.
- Every other key of the fresh LabServer63 build is bitwise equal to the archived LabServer69 WL57 geometry.
- In `node`, exactly one of 537,600 elements differs: port 1066, feature 6, archived −5.78e-16 against fresh 0.0. That feature is exactly 0 for 3,578 of 3,840 ports, so this is a cross-machine float32 rounding residue.
- No fit, reference or evaluation had started; the evidence root contains only inputs.
- The failure went unnoticed until 2026-11-04 05:56 UTC, because the wrapper stops on any failure and nothing watched it. The GPU was idle for that interval.

**Amendment 1 (pre-evidence, disclosed, `protocol/cmp72_v1.json` `amendments`).** `node` must now match to |Δ| ≤ 1e-12 absolute; all other keys remain bitwise. The non-bitwise elements and the maximum delta are recorded in `ports.json`. WL72 uses the archived geometry throughout, as before.

The same entry records two more things:
- **Walker decision.** The WL72A engine is not a drop-in, so `~/rest_walker71c/validated.json` is not created and WL72 uses the unchanged WL57 walker.
- **Incident.** During diagnosis, an accidental interactive re-source of `run72.sh` ran idle checks only. It was killed at 05:59 UTC, before any stage started.

**2026-11-04 — first rerun of the dev smoke.**
- Started 06:14 UTC.
- Passed: tests, the four 40-step fits, generation and bundling of all 1,024 dev packets.
- Failed at `eval` at about 06:50 UTC. The dev fits never reach the 2,000- and 6,000-step checkpoints, so the evaluation could not find `A_5901@2000.pt`. This is a dev-only defect.

**Amendment 2 (dev only, pre-evidence).** Checkpoints are now saved at ⌊c·steps/20000⌋ under their nominal names, which is the identity mapping in the evidence root. The dev runs and the partial dev evaluation are deleted and redone; the dev references are kept.

**2026-11-04 — second dev rerun and evidence chain.**

- **Dev smoke.** The second dev rerun completed at 07:15 UTC. Its decision label is not used.
- **Evidence chain, start.** The chain started immediately after the dev smoke. `ports.json` reproduced the dev selection: confirmation ports [751, 1504, 735, 576, 3263, 3150, 3489, 2350]; te2 identity non-bitwise only at [1066, 6]. The 4 tests passed.
- **Fits.** Two lanes; 07:15–08:44 UTC.
- **References.** 08:44–09:37 UTC.
  - 1,024 packets, all gates passed.
  - 1,017 packets used 32M chunks and 7 used the 16M OOM fallback; 3,091.6 s of packet time.
  - All packets used the unchanged WL57 fast walker.
- **Evaluation.** 1,141 s, peak 5.35 GB. The chain reported `done` at 09:58:37 UTC.
- **GPU.** Mean GPU utilisation over the whole evidence chain was 72.6% (sampler, 651 samples). The reference phase was lower, because of the known walker duty cycle.

## Final report (2026-11-04)

The evaluation follows the frozen protocol with amendments 1–2. Both amendments were made before any evidence existed and neither changes evidence behaviour. Nothing was rerun or reselected.

### Implementation Fact

**Candidate A.** WL59 P W+UV with fixed WL68 net_all_5902 A/O. The loss is J + 1·L_sym, an exact domain-symmetry consistency term on the angle-integrated collided response.
- Exact symmetries: rest D4×z-flip (15) and pt0 D4 (7, built by the unchanged geometry code).
- Initialised from P_5901 / P_5902, exactly like WL71 D1.
- 157,024 trainable parameters.

**Candidate B (RSK).** A relational shared kernel with the same fixed A/O and loss J.
- Position-free descriptors; 128-neighbour local plus 240-cell far relations.
- Seeds 7201 / 7202.
- 31,656 parameters.

**Common to both.** Training population (WL71 S16, rest + pt0, replicas 0–3), Adam, a cosine schedule over 20,000 steps, and clipping at 1.0.

**Confirmation population.**
- Geometry te2: the archived WL57 geometry, identity-gated against a fresh LabServer63 build.
- Eight new ports, selected by a geometry-only rule that excludes every exact-symmetry image of a previously referenced port.
- References for rest and te2: 1,024 packets of 32M photons, with replicas 0–3 per (slot, bin).

**Fixed reports.** Validity (passivity, linearity, positivity) and cost come from the frozen evaluator. The labelled fixed-A/O oracle is fitted on confirmation replica 0 and scored on replicas 1–3.

### Measurement

**Precision qualification.**
- Absolute confirmation matrices: all 16 qualify (minimum coarse signal/SE 16,254 ≥ 10).
- Deformation matrices (te2 − rest), minimum signal/SE over channels:

  | Slot | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 |
  |---|---|---|---|---|---|---|---|---|
  | Min signal/SE | 185 | 284 | 1,655 | −1.5 | −0.41 | 0.03 | 0.20 | 1.38 |

  Three of eight reach the predeclared ≥ 3, against the required ≥ 4, so **CONFDEF is UNMEASURED**.

**Confirmation groups.** Worst-channel upper 95% bounds, pooled over the 8 ports. CONF_te2 differs from CONF_rest by ≤ 0.01 in every row and is omitted.

| Run | con_coarse CONF_rest | abs_coarse | abs_fine16 |
|---|---|---|---|
| D1_5901 (WL71) | 0.673 | 0.247 | 0.318 |
| D1_5902 (WL71) | 0.552 | 0.237 | 0.408 |
| A_5901 | 0.871 | 0.199 | 0.223 |
| A_5902 | 0.643 | 0.241 | 0.239 |
| B_7201 | 0.629 | 0.221 | 0.251 |
| B_7202 | 0.560 | 0.319 | 0.349 |
| init_P5902 (diagnostic) | 1.106 | 0.218 | 0.330 |
| R71A_2000 (diagnostic) | 0.672 | 0.209 | 0.365 |
| Oracle, fixed A/O (labelled) | 0.211 | 0.086 | 0.093 |

**Decision metrics.** Maximum over CONF_rest and CONF_te2:
- D1 baselines: c_D = 0.552 and a_D = 0.238.
- The improvement thresholds are 0.8·c_D = 0.441 and 0.8·a_D = 0.190.
- A: c = 0.871 / 0.643, a = 0.199 / 0.241.
- B: c = 0.629 / 0.570, a = 0.222 / 0.319.

**Per-port con_coarse on rest** (worst channel). Abs_coarse follows the same pattern.

| Run | s40 | s41 | s42 | s43 | s44 | s45 | s46 | s47 |
|---|---|---|---|---|---|---|---|---|
| D1_5902 | 0.29 | 0.27 | 0.38 | 0.93 | 0.61 | 0.62 | 0.73 | 0.87 |
| A_5901 | 0.27 | 0.25 | 0.28 | 0.42 | 0.58 | 0.23 | 2.44 | 0.36 |
| A_5902 | 0.28 | 0.26 | 0.31 | 0.43 | 1.30 | 0.25 | 1.28 | 0.29 |
| B_7201 | 0.29 | 0.28 | 0.42 | 0.80 | 0.70 | 0.76 | 1.06 | 0.49 |
| B_7202 | 0.29 | 0.32 | 0.44 | 0.74 | 0.65 | 0.79 | 1.01 | 0.36 |

**Development groups** (con / abs coarse).

| Group | D1 | A | B | WL59 gates |
|---|---|---|---|---|
| S16 (training ports, replicas 4–7) | 0.199 / 0.068 | 0.199 / 0.067–0.070 | 0.200–0.201 / 0.068–0.070 | All pass |
| FIN71_rest | 0.444–0.473 / 0.156–0.167 | 0.226–0.233 / 0.056 | 0.201–0.205 / 0.063–0.064 | A and B pass, D1 fails |
| H3_cf3 | — | — | — | All pass |

DEF71 (cf2, development) fails the deformation gates for every run (error, correlation):

| Run | Error | Correlation |
|---|---|---|
| D1_5901 | 1.37 | 0.81 |
| D1_5902 | 0.91 | 0.88 |
| A_5901 | 0.57 | 0.88 |
| A_5902 | 0.56 | 0.91 |
| B_7201 | 0.65 | 0.79 |
| B_7202 | 0.53 | 0.86 |

**Validity.** Passivity, positivity, zero-input behaviour and finiteness pass on rest, pt0, cf3, cf2 and te2 for every run. All four fits converged by the WL68 last-decile rule. Final J: A 0.0368 for both seeds; B 0.0374 / 0.0373. Final L_sym (A) is 1.7e-4.

**Checkpoints (descriptive only; selection is prohibited).** Confirmation con_coarse at 2,000 steps:

| Run | @2000 | Final |
|---|---|---|
| A_5901 | 0.371 | 0.871 |
| A_5902 | 0.422 | 0.643 |
| B_7201 | 0.589 | 0.629 |
| B_7202 | 0.328 | 0.560 |

**Cost.**

| | A | B |
|---|---|---|
| Parameters | 157,024 | 31,656 |
| Runtime state floats per geometry | 14.7M | 33.9M |
| Coefficient build | 13 ms | 36 ms |
| 64-light application | 63 ms | 67 ms |
| Training (contended, 2 lanes) | 4,177 s | 1,091–1,140 s |
| Peak GPU memory | 3.82 GB | 3.31 GB |

A's runtime figures are identical to D1's.

### Observation

1. **Verdict basis.** The confirmation deformation group is unmeasured because five of the eight confirmation ports (s43–s47) lie 57–75 mm from the te2 dent centre, outside its 22.5 mm radius. Their te2 − rest response is below 3 SE. CONF_rest and CONF_te2 are almost identical for the same reason.
2. **Deduction from the predeclared rule, not the verdict.** "Improves" uses only c and a. On the measured groups, no seed of either candidate meets c ≤ 0.441 and a ≤ 0.190. A measured CONFDEF could therefore only have produced NEITHER IMPROVES.
3. **Near-dent and top-face ports.** On s40–s42, A's errors are at or below D1's and approach the oracle (con 0.25–0.31 vs oracle pooled 0.21). B is close to D1 there.
4. **Side face and corners.** The confirmation error is dominated by side-face s46 (A up to 2.44, B ≈ 1.0) and the corners s43/s44. A_5902's s44 reaches 1.30.
5. **Development vs confirmation.** Both candidates roughly halve D1's error on the WL71 final ports (development) and pass the WL59 gates there. That improvement does not carry over to the eight confirmation ports chosen away from every referenced port and its symmetry images.
6. **Representability.** The fixed A/O oracle reaches 0.21 / 0.086 / 0.093 on the confirmation ports, passing all three WL59 thresholds. The fixed angular interface does not bind here; the spatial maps of every run are 2.6–4× worse in contrast.
7. **Early checkpoints.** Every arm's 2,000-step checkpoint is better on confirmation than its final iterate. This is consistent with the WL71A training drift and is not used for any decision.

### Interpretation

Under this protocol, neither the exact-symmetry constraint (A) nor the relational shared kernel (B) produced a measured improvement over WL71 D1 on new ports.

The gains seen on development ports, including WL71's final ports, do not generalise to ports chosen away from every previously referenced location. That is the same supervision-local pattern WL71 reported for D1.

The formal verdict is INCONCLUSIVE. This comes from the confirmation-port design, not from reference noise or a candidate defect: the eight confirmation ports were chosen for spatial novelty without a rule tying them to te2's deformation footprint. That design omission is mine (Stage 0); the precision rule caught it as intended.

Which mechanism limits transfer to the side face and corners is not attributed.

### Unresolved Question

- Why do both candidates improve FIN71 but not the confirmation ports? Possibilities include FIN71's proximity to the S16 ports or their symmetry images, and similarity of local configuration. This was not measured.
- What drives the side-face (s46) and corner (s43/s44) failures? A's symmetry term and B's relations do not cover them.
- Does training drift (final worse than @2000 for every arm) dominate the confirmation error? Selecting checkpoints by held-out performance is prohibited, so this remains open.
- Te2's deformation-dependent fidelity is unmeasured on these ports.

### Architecture Verdict

**INCONCLUSIVE**, by the predeclared rule: confirmation deformation group UNMEASURED, 3 of 8 qualified matrices.

- Both candidates are valid: passivity, positivity, zero input and finiteness pass on all five geometries.
- Neither meets the improvement criterion on the measured confirmation groups.
- No adoption and no successor.

### Reproducibility

**Code.** Commit 80c9b26 (frozen), plus amendments 88f362b (te2 identity tolerance) and fc04413 (dev checkpoints); evidence ran at fc04413.

**Data and runtime.**
- Protocol: `experiments/fnto_cmp72/protocol/cmp72_v1.json`.
- LabServer63, root `~/rest_cmp72`: references 15 GB, with bundle hashes in `refs_complete.json`.
- Seeds 723000000 + g·1e6 + slot·1e4 + bin·100 + rep.
- Fit seeds: A from P_5901 / P_5902; B 7201 / 7202.
- Python: `~/NeuralShaderDemo/external/8dna26/.venv` (torch 2.3.1).

**Outputs and evaluation.**
- Local copies: `results/fnto_cmp72/final/` (summary, CSVs, figure, manifest with hashes) and `results/fnto_cmp72/eval/analysis.json`.
- Reviewer artefacts: `results/evaluation/72/`.
- Evaluation: 1,141 s, peak 5.35 GB.
