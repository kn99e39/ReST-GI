# Worklog — Fixed-Budget Source Support Intervention (2026-11-04)

## Authorisation and scope

On 2026-11-04 the user issued the brief "Worklog 73 — Fixed-Budget Source Support Intervention". It authorises one bounded, controlled source-distribution intervention on the WL59 intrinsic-K family.

**The question.** At a fixed physical-supervision budget, does better training coverage of optical boundary conditions produce transferable intrinsic-K responses at independent source locations? And how do D1, A and B differ in that transfer?

**Two explanations to separate:**
- **H_support.** The representations transfer within boundary or source conditions that were represented in training, and fail only when those conditions are absent.
- **H_mapping.** The learned geometry-to-transport mapping does not transfer, even when the support is appropriate.

**Architectures, all preserved unchanged.** All use the fixed WL68 A/O.
- D1: WL59 W+UV, objective J.
- A: W+UV, objective J + L_sym, with exact symmetry.
- B: RSK, objective J.

**Intervention, the only change.** The training-source composition. S16-R keeps 16 source ports, replaces exactly 4 of the WL71 S16 sources, and allocates the replacements to underrepresented boundary conditions (side face, edge, corner) by geometry-only rules.

**Supporting work:**
- the new physical references the intervention needs;
- isolated training and evaluation adapters;
- a new independent confirmation population with boundary strata;
- a deformation configuration whose footprint is covered by enough source pairs;
- diagnostic and cost instrumentation.

**Preserved:**
- K/C ownership, first-exit semantics and ballistic B;
- the scattered path A → T → O;
- 16 incoming / 64 outgoing bins;
- positivity, linearity and passivity;
- materials, metrics and thresholds;
- parameter counts, objectives, optimiser, schedule, clipping and the 20,000-step budget;
- every historical checkpoint and result;
- the WL72 INCONCLUSIVE verdict.

WL72's confirmation population is now development evidence.

**Not authorised:**
- more than 16 training sources;
- capacity, feature or loss changes;
- tuning;
- held-out-based source or checkpoint selection;
- a new architecture;
- production K/C changes;
- new physics or coupled GI;
- automatic adoption or a successor.

**Walker.** The primary references use the unchanged WL57 walker. The WL72A streaming engine is not substituted.

**Placement.** Isolated `experiments/fnto_src73` paths, branch `Selective_Recompute`, no new branch or worktree, LabServer63 (main). User instruction (2026-11-04): 「원래 TransGI 작업을 기다리는 감시 프로세스가 있었을텐데 그건 신경쓰지 말고 네가 우선적으로 서버 선점해」 — the wrapper does not wait for other jobs; timings may therefore be contended, and `logs/gpu_sampler.log` records every GPU process.

## Stage 0 design (2026-11-04)

The frozen design record is `docs/plans/SOURCE_SUPPORT73_V1.md`; the protocol is `experiments/fnto_src73/protocol/src73_v1.json`.

**Inputs.** All rules are geometry only and ran on the archived rest/pt0 geometries on LabServer63 (`ports73.py propose`).
- The te0 geometry is the archived WL57 file from LabServer69, sha256 `79216410…`, copied through the PC with matching hashes.

**S16 audit.** Every S16 source is interior with edge distance ≥ 21.25 mm. No side, edge or corner condition is supported, even through exact images.
- Slots 2, 6, 8, 9 and 12 are one exact rest-symmetry orbit.

**S16-R.**
- Removed: slots 6, 8, 9, 12 (rule: rest duplicates among flat-interior sources).
- Added: 50 (TBE, port 3439), 51 (TBC vertex, 3823), 52 (SM, 408) and 53 (SR, 527).

**Confirmation population.**
- 23 static rest ports: INT near/far, BND_NEAR, BND_FAR, UNS and REM_NEAR.
- 6 te0 footprint ports.
- Deformation pose: te0, selected by a predeclared rule.

**References.**
- 1,024 new training packets (the S16 replica structure).
- 2,240 confirmation packets.
- 32M photons each, unchanged WL57 walker.

**Pre-freeze review.** Before freezing, an independent review workflow ran: three lenses, two adversarial verifiers per finding, 15 findings. It led to one revision of the design. No physical outcome, model prediction or WL72 error was used:
- BND_NEAR and BND_FAR became separate primary groups, and BND_FAR gained second far ports 90–92. Pooled BND let 5 mm interpolation decide.
- REM_NEAR (slots 76–79) was added, together with the removal cost in the regression test. INT_ALL by construction avoided the removed sites.
- Comparisons became three-valued (YES / NO / OVERLAP), so overlapping seeds are not read as "no improvement".
- A relative fit criterion now sits next to the absolute WL59 gates.
- The removal rationale was corrected: the removed sources are duplicates at rest only, not at pt0, including for A.
- DEF73 is declared a near-training-pose check, because te0 overlaps pt0.
- Crashed fits now quarantine their orphaned checkpoints.
- Stage 1 asserts that the qualified matrix sets equal WL72's.

**Dev smoke during design (not evidence).** A pre-freeze dev chain (`~/rest_src73_dev`) found one test defect:
- The adapter-identity test asserted bitwise target equality, but the GPU `index_add` aggregation is not bitwise deterministic. The test now uses relative 1e-6.
- After the fix, Stage 1 passed in dev:
  - evaluation reproduces WL72's archived bounds within 1.05e-6 (tolerance 1e-4);
  - the replays match the archived initial objective within 1e-7.
- The dev chain was then stopped for the design revision. Its training packets, which are unchanged by the revision, are reused by the resumed dev smoke.

## Execution log

**2026-11-04, freeze.** Commit ce5d73e. `run73.sh` was launched on LabServer63 at 12:26 UTC.
- **Dev smoke.** Completed at 14:14 UTC: Stage 1 passed, 3,264 dev packets generated, six 40-step fits, evaluation in 1,112 s and export. Its labels carry no information.
- **Evidence chain.**
  - The ports check and the te0 identity passed. The te0 identity has 4 non-bitwise node elements, max 2.9e-16, and the β rel L2 is 0.105, as for cf2 and te2.
  - tests_pre: 5 of 5 passed.
  - Evaluation identity against WL72: worst relative difference 1.45e-6, with the qualified matrix sets equal.
  - **The 500-step replay gate failed.** The initial objective matched (D1 9.2e-8, A 0, B 0), and the logged J agreed with the archives to ≤ 1e-4 up to steps 150–200. After that, individual logged steps deviated with alternating sign: maximum relative difference D1 2.98 %, A 1.94 % and B 2.52 %, against the predeclared 1 %. The trend was unchanged. The chain stopped at 14:51 UTC as designed. No new reference, fit or evaluation existed.

**Amendment 1** (pre-evidence, disclosed; protocol `amendments`).
- **Why.** The 1 % per-step tolerance had been set without measuring the code's own run-to-run variability. The 40-step dev smoke could not reveal it, because the replays only diverge after step 150. The observed pattern is float32 GPU reduction nondeterminism amplified by Adam.
- **New rule.** The gate now compares the archive mismatch with the run-to-run envelope of the same code:
  - a second independent replay (`replay2_`) is made of each run;
  - each replay's maximum deviation from the archive must be ≤ 2 × the replay-vs-replay maximum deviation;
  - the initial-objective identity rule is unchanged.
- **Records.** The failed original record `stage1/summary.json` is preserved. The amended record is `stage1/summary_amendment1.json`.
- **Chain defect fixed in the same amendment.** The frozen chain would have proceeded past an existing *failed* Stage 1 summary on restart. It now stops unless the original or the amended gate passed.
- **Dev check.** The amendment path was smoke-tested in the dev root: replay2 of B_7201 at 40 steps, then the comparison. The test output was then removed.
- **Amendment 1 result** (`stage1/summary_amendment1.json`, 2026-11-04 ~14:40 UTC). Passed. Initial-objective identity held for both replays (≤ 9.2e-8).

  Trajectory deviations (max relative difference over logged steps):

  | Run | Replay vs archive | Replay2 vs archive | Replay vs replay2 | First step beyond 1e-3 | Mean signed deviation (replay vs archive) |
  |---|---|---|---|---|---|
  | D1_5901 | 2.98 % | 2.07 % | 3.70 % | 225 | −0.10 % |
  | A_5901 | 1.94 % | 5.25 % | 4.90 % | 175 | +0.11 % |
  | B_7201 | 2.52 % | 1.50 % | 2.08 % | 150 | +0.53 % |

  Two runs of the same code differ from each other by as much as either differs from the archive. The original 1 % failure is therefore run-to-run nondeterminism, not a path difference. Reference generation started.

**2026-11-04, evidence chain after Stage 1.** The chain finished at 19:51 UTC without intervention.
- **References.**
  - Training: 1,024 packets, all gates passed, all at 32M, 2,446 s.
  - Confirmation: 2,240 packets, all gates passed, 8 OOM fallbacks to 16M, 6,318 s.
  - The WL57 walker was unchanged.
- **Fits** (two lanes, contended), all converged by the WL68 rule:
  - A: 4,156 s per run.
  - D1: 1,620 s per run.
  - B: 1,098 s per run.
- **Evaluation:** 3,053 s, peak 8.1 GB.
- **Qualification:** every absolute matrix qualified (minimum signal/SE 1.18e4), and all 6 DEF73 matrices qualified. No group was UNMEASURED.
- **GPU:** idle from 19:51 UTC until the session read the result (2026-11-05 03:09 UTC). No job was waiting.

## Final report (2026-11-05)

The evaluation follows the frozen protocol (ce5d73e) plus amendment 1. Amendment 1 changed only the Stage 1 replay tolerance, before any evidence existed. Nothing was rerun, reselected or tuned.

All values are worst-channel upper 95 % bounds (eval68 pooled, jackknife). Each "old" entry lists the S16 seeds (WL71 D1 / WL72 A, B finals); each "new" entry lists the S16-R seeds (WL73 finals).

### Implementation Fact

- **Training data.** S16-R keeps 12 S16 sources bitwise and replaces slots 6, 8, 9 and 12 with:
  - TBE: port 3439, edge row;
  - TBC: port 3823, slab vertex;
  - SM: port 408, side mid row;
  - SR: port 527, side rim row.

  Per-matrix budget, replica split (fit 0–3, evaluation 4–7) and number of source matrices are identical to S16.
- **Unchanged by the intervention:**
  - constructors, initialisation and seeds;
  - objective (J; A adds L_sym with λ = 1);
  - optimiser, schedule, 20,000 steps and clipping;
  - A/O, angular bases and parameter counts (W+UV 155,104 learned; RSK 31,656);
  - K/C, ballistic B and physics.

  A/O stayed bit-identical through every fit.
- **Validity.** All 12 final models pass zero-input, positivity, linearity, cache and passivity (column sums ≤ 1 + 1e-6) on rest, pt0, cf3, cf2, te2 and te0.
- **Reproduction (Stage 1).**
  - Evaluation: identical to WL72 within 1.45e-6, with equal qualified-matrix sets.
  - Training path: initial objective within 9.2e-8; J trajectory within the code's own run-to-run envelope (amendment 1).

### Measurement

**1. Training fit of the new support** (replicas 4–7, rest; pt0 is identical to 3 decimals). Values are abs_coarse / abs_fine16 / con_coarse.

| | D1 | A | B |
|---|---|---|---|
| S16 models (sources unseen) | 0.43–0.47 / 0.49–0.50 / 1.14 | 0.41–0.43 / 0.47–0.49 / 1.09–1.21 | 0.47–0.49 / 0.44–0.56 / 1.08–1.18 |
| S16-R models (sources fitted) | 0.110 / 0.119 / 0.270 | 0.110 / 0.119 / 0.270 | 0.110 / 0.121 / 0.270 |
| Fixed A/O oracle (labelled) | | 0.111 / 0.117 / 0.270 | |
| WL59 gate | | 0.10 / 0.15 / 0.25 | |

On the 12 retained sources, every model, old and new, scores 0.060–0.066 / 0.089–0.093 / 0.188–0.191.

**2. Confirmation groups** (rest; old → new, con_coarse and abs_coarse):

| Group | D1 | A | B |
|---|---|---|---|
| BND_NEAR (5 mm from a new source) | con 1.18, 1.30 → 0.50, 0.51; abs 0.44, 0.39 → 0.12, 0.13 | con 1.16, 1.49 → 0.42, 0.43; abs 0.37, 0.39 → 0.11, 0.11 | con 1.53, 1.28 → 0.46, 0.43; abs 0.46, 0.44 → 0.14, 0.12 |
| BND_FAR (16–33 mm, farthest from all images) | con 1.50, 1.00 → 1.30, 1.30; abs 0.50, 0.44 → 0.36, 0.40 | con 1.01, 0.95 → 0.87, 0.94; abs 0.44, 0.39 → 0.17, 0.21 | con 1.23, 1.43 → 1.14, 1.15; abs 0.45, 0.43 → 0.29, 0.30 |
| REM_NEAR (5 mm from a removed source) | con 0.32, 0.30 → 0.99, 1.21; abs 0.14, 0.10 → 0.30, 0.29 | con 0.21, 0.20 → 0.49, 0.55; abs 0.07, 0.07 → 0.09, 0.10 | con 0.28, 0.51 → 0.48, 0.28; abs 0.09, 0.12 → 0.19, 0.13 |
| INT_ALL (interior) | con 0.35, 0.28 → 0.34, 0.49; abs 0.12, 0.11 → 0.11, 0.12 | con 0.25, 0.24 → 0.27, 0.37; abs 0.07, 0.07 → 0.07, 0.08 | con 0.23, 0.22 → 0.24, 0.24; abs 0.08, 0.08 → 0.11, 0.09 |
| UNS (side vertical edge) | con 1.27, 1.35 → 1.27, 1.53 | con 1.08, 1.38 → 1.05, 1.02 | con 1.67, 1.67 → 1.24, 1.09 |

Labelled fixed-A/O oracle (abs_coarse / fine16 / con_coarse):

| Group | Oracle |
|---|---|
| BND_NEAR | 0.087 / 0.115 / 0.268 |
| BND_FAR | 0.104 / 0.114 / 0.267 |
| UNS | 0.087 / 0.108 / 0.237 |
| INT_DED | 0.072 / 0.088 / 0.178 |
| REM_NEAR | 0.062 / 0.086 / 0.121 |

**3. Per port (con_coarse, worst channel; old seeds → new seeds).**

| Stratum | D1 | A | B |
|---|---|---|---|
| BND_NEAR | 1.2–1.7 → 0.26–0.75 | 0.9–2.1 → 0.22–0.61 | 1.2–2.0 → 0.29–0.69 |
| BND_FAR | 0.8–3.0 → 0.9–2.5 | 0.4–2.2 → 0.65–1.5 | 0.8–2.4 → 0.8–2.0 |
| REM_NEAR (actual S16-R sources 18–23 mm away, images 5 mm) | 0.16–0.54 → 0.49–2.06 | 0.16–0.25 → 0.19–0.84 | 0.14–0.70 → 0.16–0.71 |

At BND_FAR every architecture has ports that get worse and ports that improve. A's TBC-far port 67 stays the best of these, at 0.37–0.73.

**4. Classification inputs** (three-valued, predeclared):

| | fit_ok (Q / relative) | far | near | regression INT_ALL / REM_NEAR | Class |
|---|---|---|---|---|---|
| D1 | no / no | NO | YES | OVERLAP / YES | INSUFFICIENT FIT |
| A | no / no | NO | YES | OVERLAP / YES | INSUFFICIENT FIT |
| B | no / no | NO | YES | OVERLAP / OVERLAP | INSUFFICIENT FIT + possible regression (seed overlap) |

Same-budget ranking on BND_FAR: A > D1 YES; A > B OVERLAP; every other pair NO.

**5. Late-fit drift (descriptive; checkpoints never selected).** BND_FAR con_coarse at @2000 / @6000 / @10000 / final:

| Run | Values |
|---|---|
| D1R_5901 | 0.91 / 1.09 / 1.17 / 1.30 |
| D1R_5902 | 0.84 / 0.90 / 1.16 / 1.30 |
| AR_5901 | 0.79 / 0.90 / 0.90 / 0.87 |
| AR_5902 | 0.88 / 0.95 / 0.99 / 0.94 |
| BR_7201 | 0.95 / 1.10 / 1.09 / 1.14 |
| BR_7202 | 0.94 / 0.99 / 1.13 / 1.15 |

REM_NEAR also rises over training: D1R 0.36–0.42 → 0.99–1.21; BR_7201 0.20 → 0.48. BND_NEAR is flat (0.38–0.45 → 0.42–0.51).

**6. Deformation (DEF73, te0 − rest, 6/6 qualified)** — abs_coarse worst channel (blue, error) and correlation range:

| | Old | New |
|---|---|---|
| D1 | 0.58–0.73 / 0.82–0.93 | 0.80–0.83 / 0.82–0.94 |
| A | 0.56–0.61 / 0.82–0.96 | 0.48–0.54 / 0.85–0.96 |
| B | 0.52–0.66 / 0.79–0.94 | 0.61–0.65 / 0.80–0.91 |

All fail Q_def (≤ 0.35 and ≥ 0.90 in every channel). Red/green errors are 0.29–0.55.

**7. Descriptive.**
- **Static te0 footprint (FP_te0)**, abs_coarse / con_coarse:

  | | Old | New |
  |---|---|---|
  | D1 | 0.09–0.11 / 0.26–0.32 | 0.08–0.09 / 0.25–0.27 |
  | A | 0.07 / 0.23 | 0.07 / 0.23 |
  | B | 0.07–0.08 / 0.23 | 0.08 / 0.24 |

- **Development groups**, con_coarse old → new:

  | Group | D1 | A | B |
  |---|---|---|---|
  | FIN71_rest | 0.44–0.47 → 0.30–0.43 | 0.23 → 0.26–0.35 | 0.20 → 0.20–0.23 |
  | CONF72_rest | 0.55–0.67 → 0.72–0.74 | 0.64–0.87 → 0.51–0.53 | 0.56–0.63 → 0.57–0.58 |

- **WL59 historical fields** (worst-channel mean coarse): D1 0.24 → 0.18–0.19; A 0.20 → 0.09–0.11; B 0.23–0.26 → 0.18–0.19.
- **Flux concentration at the new sources** (5 / 10 / 20 / 40 mm): model and reference agree within 0.01–0.03 for every architecture (reference 0.80–0.94 / 0.96–0.99 / ≥ 0.998).
- **Cost:** unchanged by the intervention. W+UV: 14.7M state floats, 13 ms build. RSK: 33.9M floats, 36 ms build. Applying 64 lights takes 63–68 ms.

### Observation

1. **The predeclared classification is INSUFFICIENT FIT for all three architectures.** The four new boundary sources are fitted to 0.110 / 0.119–0.121 / 0.270, identical across D1, A and B, and equal to the labelled fixed-A/O oracle on the same matrices (0.111 / 0.117 / 0.270). That misses the WL59 gate (0.10 / 0.25) and the relative criterion (1.25 × the retained-source fit ≈ 0.077 / 0.24).
2. **The fit limit is the fixed angular interface.** The spatial maps reach the representability floor of the fixed A/O on the new sources, and three different spatial parameterisations converge to the same value. The bound is the A/O-constrained response class at boundary sources, not the spatial representation and not the optimiser.
3. **Near-source transfer improves strongly for every architecture.** BND_NEAR con_coarse falls by about 2.5–3.3× (1.2–1.5 → 0.42–0.51) and abs_coarse by about 3.5×. The new values are near the oracle level (abs 0.11–0.14 vs 0.087), so 5 mm from a supervised boundary source all three architectures transfer to about the representability floor.
4. **Far boundary transfer is small, mixed and outside the gates.**
   - BND_FAR contrast changes by at most −14 % (1.00–1.50 → 0.87–1.30). D1's best old seed (1.00) is better than both new seeds (1.30).
   - Absolute error falls for A (0.39–0.44 → 0.17–0.21) and B (0.43–0.45 → 0.29–0.30), less for D1 (0.44–0.50 → 0.36–0.40).
   - Per port, every architecture has far ports that get worse.
5. **The swap has a measurable cost at the removed sites.** REM_NEAR contrast rises about 3× for D1 (0.30–0.32 → 0.99–1.21) and about 2.5× for A (0.20–0.21 → 0.49–0.55). B is mixed.
   - The A result shows that exact rest images through slot 2 do not preserve these responses after retraining, even though A's symmetry term still supervises those positions at rest.
   - Interior (INT_ALL) shows only seed-overlap changes.
6. **Late-fit drift again moves transfer away from the new supervision.** BND_FAR and REM_NEAR worsen from @2000 to the final iterate for D1 and B, while BND_NEAR and the training fit stay flat. A's BND_FAR changes least (0.79–0.88 → 0.87–0.94).
7. **Deformation is measured and fails for every model.** DEF73 does not improve with the boundary support (D1 worsens in blue). A has the lowest blue error before and after.
8. **The unsupported vertical-edge context (UNS) stays at contrast 1.0–1.5 in every model.**

### Interpretation

- **Within this batch's decision rule, the result is unresolved.** Every architecture is INSUFFICIENT FIT, so the predeclared question "does improved support produce transferable responses?" cannot be answered with a clean support-vs-mapping classification.
- **The source of the unresolved fit is identified.** The fixed WL68 A/O interface cannot represent the boundary-source responses below 0.11 / 0.27. That is an angular-interface limit at boundary sources (grazing ballistic exits, β up to 0.39). The spatial representations reach it.
- **Taken as evidence, not as a classification, the pattern is mixed and leans towards H_mapping for transfer distance:**
  - Supervision at a boundary condition yields near-floor accuracy within about 5 mm, for all three architectures.
  - At 16–33 mm from any image of the new support, the same boundary context is not reproduced (contrast ~0.9–1.3, about 3.5× the oracle).
  - The swap degrades responses near the removed sources even where exact images remain supervised.

  That is consistent with transfer that is local in source position, as in WL71 and WL72. It is not consistent with a missing-support-only account, because support present in the same context did not transfer to far ports.
- **A vs B vs D1 at the same budget.** A is better than D1 on far boundary transfer, and its far-transfer drift is smallest. A vs B is within seed overlap. Development-port gains are not used for ranking.
- **Confound.** The A/O fit floor applies to every confirmation stratum: BND_FAR's oracle is 0.104 / 0.267. Part of every boundary error above the oracle is spatial and part is angular. The batch does not separate them per port beyond the oracle comparison.

### Unresolved Question

- **Support vs mapping.** Whether boundary support transfers to far same-context sources once the angular interface can represent boundary responses. This is not tested here, and it would need an A/O change, which this batch did not authorise.
- **Local transfer mechanism.** Why transfer stays local (~5 mm) in all three architectures, including the position-free RSK.
- **Removal cost under symmetry.** Why A loses accuracy near the removed sources although their exact rest images stay supervised. Candidates are the pt0 supervision loss, interaction with the new sources, or the late-fit drift. Not attributed.
- **Far-transfer drift.** Whether the late-fit drift is the main cause of the BND_FAR and REM_NEAR degradation. Checkpoint selection is prohibited, so this stays open.
- **Deformation transfer.** DEF73 is a near-training-pose check, so deformation transfer to genuinely new poses remains unmeasured.

### Architecture Verdict

**INSUFFICIENT FIT for D1, A and B** (predeclared rule). No architecture is adopted, and no successor is authorised. The boundary-source fit sits at the fixed-A/O representability floor, so the batch does not deliver a clean SUPPORT-SENSITIVE / SUPPORT INSUFFICIENT classification.

Descriptively:
- near-source transfer is support-sensitive for all three;
- far same-context transfer is small, mixed and outside the gates;
- the swap carries a measurable cost near the removed sources (D1, A);
- deformation fails for all.

At equal budget A ranks above D1 on far boundary transfer, and A vs B is within seed overlap.

### Reproducibility

- **Code and design:** `experiments/fnto_src73/`, protocol `protocol/src73_v1.json` (frozen ce5d73e; amendment 1 at 103a6ae, which changes only the Stage 1 replay gate), design record `docs/plans/SOURCE_SUPPORT73_V1.md`.
- **Server:** LabServer63, root `~/rest_src73` (code_commit.txt `103a6ae`); dev root `~/rest_src73_dev`. Python `~/NeuralShaderDemo/external/8dna26/.venv` (torch 2.3.1).
- **Seeds:**
  - training references: 733000000 + g·1e6 + slot·1e4 + bin·100 + rep;
  - confirmation references: 736000000 + g·1e6 + …;
  - fits: P_5901/P_5902 (D1, A) and 7201/7202 (B).
- **Outputs:**
  - `results/fnto_src73/eval/analysis.json`;
  - `results/fnto_src73/final/`: summary, group_metrics.csv, per_source.csv with distances, src73.png/pdf, reference manifests with bundle hashes, the Stage 1 record;
  - `results/fnto_src73/ports.json` (Stage 0 audit and selections);
  - `results/fnto_src73/stage1/`: original failed record, amendment-1 record, evaluation identity.
- **Reviewer copies:** `results/evaluation/73/`.
