# Worklog — Full-Depth HIGH Transport Learning and Truncation Error Resolution (2026-11-25)

**Batch class:** regular numbered batch (WL114). It tests a new HIGH target-depth hypothesis: full-depth physical suffix supervision of the unchanged ECRO response architecture. Numbered after WL113, the highest worklog on `ecro_lr3_adapter`; by owner instruction, numbers on other branches are ignored.

**Owner brief:** "ReST-GI — Full-Depth HIGH Transport Learning and Truncation Error Resolution".

**Code:** `experiments/ecro_fd114/` (protocol `protocol/fd114.json`).
**Raw results:** `results/ecro_fd114/primary/`.
**Reviewer artifacts:** `evaluation/114/`.

**Machine:** local RTX 5080, with two runtimes:
- SSS: the ECRO runtime `external/8dna26/.venv-cu128` (Mitsuba 3.5.1 / Dr.Jit 0.4.6).
- LR3 Stage A: `C:/Projects/TransGI_Reproduction/.venv` (Mitsuba 3.6.4 / Dr.Jit 1.0.5). This is the stack that rendered the authenticated Ref-65 / Ref-7.

## 1. Append-only clarification of WL113's descriptive summary (no historical rewrite)

WL113's final report (`docs/worklogs/113_...`, section "Reuse is mechanically exact") says *"Depth-matched dynamic fidelity holds in 2 of 3 sequences."* That sentence is **wrong**.

I re-read the frozen JSON evidence (`results/ecro_sss113/primary/analysis/stageE_metrics.json`, `gate.json`). Per sequence, `pass_depth_matched` / `historical_all_states` are:

| Sequence | Historical depth-matched contract | Failing state-views (learned CAND) | R7 tracking |
|---|---|---|---|
| SEQ_A_lift | **FAIL** | gap 50 mm V0, both visits (states 3 and 6): innermost band +0.044 / +0.045 vs band 0.026 | V0 slope 1.110 FAIL; H2 1.077 pass |
| SEQ_B_slide | **PASS** | none (10 / 10 state-views) | not applicable (no gap series) |
| SEQ_C_independent | **FAIL** | W0 at all 5 states (inner bands +0.029 vs bands 0.016); W2 at 30 mm and 80 mm (near +0.025 to +0.038 vs bands 0.022 to 0.028) | W0 slope 1.124 FAIL; W2 1.182 FAIL |

Correct reading:
- **The historical depth-matched contract holds in 1 of 3 sequences (SEQ_B only).**
- SEQ_C also fails at its *contact* states in view W0, not only at the lifted states.
- R7 tracking fails in SEQ_A V0 and in **both** SEQ_C views; it is not one narrow miss.

What stays unchanged:
- **WL113's official verdict NOT_RESOLVED is preserved.**
- Its rebuild / revisit exactness and discriminating transitions are unaffected.
- The dense 4×4 uniform table is supported as an experimental representation for the **historical oracle contract only**. It is not supported for the complete approved 3 % localisation contract: oracle R1L fails at every view, and R4 fails at 9 / 14.
- The compact VHS ∪ MEDIUM representation remains **unadopted**.
- WL113's file is not edited.

## 2. Question and design (frozen in `protocol/fd114.json` before any evidence stream)

**Question.** Can ECRO learn and reuse an effectively full-depth HIGH response (camera events ≥ 3) without changing its architecture, capacity or spatial representation, and so remove the D6 truncation limit identified in WL102 and WL113? Once truncation is removed or isolated, which SSS errors genuinely remain spatial or learned?

**Only change:** the experiment-local HIGH target depth, plus the physical suffix estimator needed to produce it.

**Event convention:**
- LOW = camera events 1–2, physical, unchanged.
- HIGH begins at camera event 3; the table is read at event 2.
- D6 target = construction probe f2 (camera event 3) + fh (camera events 4–6).
- Full target = f2 + fh + deterministic deep buckets up to K\*.

**Deep estimator** (`tracer114.py`, new; no shared file edited):
- Event-bucketed accumulation, by camera event: e1 | e2 | h(3–6) | 7–12 | 13–24 | 25–48 | 49–96 | 97–192 | roulette.
- Exact stream compaction.
- Verification-only Russian roulette.
- Events 1–24 are statement- and RNG-identical to WL113's deep pass. The D6 part of the deep construction equals WL113's `construct4` (tests T1, T2).

**Stages:**
- **A — target qualification.**
  - SSS: deterministic passes to event 192, plus roulette passes from event 6, on 12 views:
    - 5 WL113 static views;
    - 3 lifted / contact-changing states;
    - 4 new independent fixtures N1–N4.
  - SSS constructions on 4 states.
  - K\* rule: smallest K in {24, 48, 96} with omitted tail ≤ 1e-3 of the full-depth HIGH in every region / patch class. Roulette consistency within 4 SE (C2 / C4).
  - LR3: authentication of Ref-65 by re-rendering with the identical stack, then a max_depth ladder up to ∞ (Mitsuba roulette) with common random numbers.
- **B — oracle feasibility.**
  - Deep 4×4 constructions per evaluation state give D6 and full-depth tables from the **same samples**, under the WL113 coarse and dense contracts.
  - Paired 2 × 4,096-spp streams against a deep physical reference to K\*.
  - Stop rule: truncation ≥ 2× the full-depth oracle near error at ≥ 75 % of contact-valid static views.
- **C — matched learning.**
  - D114 = the D113 configurations, construction seeds and launch partition, extended to depth K\*.
  - The D6 part is authenticated against D113.
  - Arms, all with the unchanged WL113 recipe:
    - A1r: dense, D6 targets, seed 11330001;
    - B1: dense, full targets, seed 11330001;
    - A2 / B2: the same pair with seed 11330002;
    - OB1: coarse, full targets.
  - The only difference within a pair is the global target key.
- **D — SSS validation.**
  - Views: WL113 static views (regression seeds), N1–N4, sequences SEQ_A / B / C and the new SEQ_D.
  - Approved 3 % quantities at full depth (primary) and depth-matched (secondary).
  - Historical contract for the D6 arms.
  - Decomposition into truncation / spatial / learned components; stale, rebuild and revisit controls.
- **F — cost.**

**LR3 Stages B–E** need the ECRO Mitsuba 3.5.1 stack (see §3). They are **pending an owner decision** on LabServer69; otherwise they are reported NOT_RUN.

**Not run:** R3 (equal-time frontier) and R9 (radius / medium). The SSS gate therefore cannot reach RESOLVED in this batch.

## 3. Execution facts (before the freeze)

- **ECRO stack cannot load LR3 on this GPU.** Under Mitsuba 3.5.1 / Dr.Jit 0.4.6, `mi.load_file(scene_transgi.xml)` never returns on the RTX 5080.
  - A native stack dump shows the stall inside `Mesh::build_pmf` → kernel load (`cuModuleLoadData`).
  - A minimal reproduction: `dr.compress` on n = 5,000 never returns, while n ≤ 1,000 and `dr.sum` / `dr.count` work.
  - Dr.Jit 0.4.6's large compress / scan path is therefore unusable on this GPU. WL102 ran this stack on LabServer69 (RTX 3090).
  - Consequences:
    - (a) `tracer114` compacts with a host-side index list, never `dr.compress`;
    - (b) LR3 Stages B–E cannot run locally. The owner was asked whether they may run on LabServer69.
- **LR3 reference authentication.**
  - The local Ref-65 / Ref-7 copies hash-match WL102's manifest (`evaluation/102/manifest.json`).
  - The local `scene_transgi.xml` differs only in CRLF line endings. Its LF-normalised sha256 equals Ref-65's recorded `modified_sha256` (62583fb9…).
  - The scene file is loaded read-only; nothing is written to the TransGI data tree.
- **Development runs** (throwaway seeds, never evidence):
  - Tracer identity and compaction checks.
  - Throughput probes:
    - deep camera pass to event 24 ≈ WL113's per-event pass;
    - deep construction n = 32: 1.4 s at depth 24;
    - b = 16 peaks at 14.1–14.8 GB, so D114 generation runs alone on the GPU.
  - A tiny-spp LR3 ladder.
  - A complete dev chain.
  - Disclosure: the dev runs showed qualitative tail magnitudes (SSS whole-image tail beyond event 24 ~2e-6 of HIGH; LR3 dev ladder GI change beyond 64 events ~1e-4) **after** the 1e-3 rule and the K candidate set were written into the protocol draft. No threshold was changed afterwards.
- **Verification roulette changed before the freeze.** In the dev chain the construction ring-0 roulette replicas both sat ≈ 0.9 % below the deterministic deep sum (2 replicas, n = 8). This matches the right-skew of roulette driven by absolute throughput: the estimator is unbiased but heavy-tailed, so with few replicas most fall below the mean.
  - The survival probability was changed to q = min(max_c thr_c / max_c thr_c(at roulette start), 0.95). It is still unbiased and has lower variance.
  - Construction roulette replicas were raised from 4 to 8.
  - The C2 / C4 criteria themselves are unchanged.
- **LR3 authentication criterion changed before the freeze.** Re-rendering on a different GPU with the same seeds cannot be required to be bit-identical. L0 now requires the GI mean per channel within 1e-3 relative and RMSE(ours − Ref-65) ≤ 2.0 × Ref-65's GI noise RMSE; an independent same-budget render is expected at √2 ×.
- **Test tolerances.** Two test tolerances were relaxed during development, before the freeze, from bit-identity to atomic-summation-order level:
  - T3: rr_from beyond kmax vs deterministic, float64 rel ≤ 1e-12;
  - T6: LOW across tables, rel ≤ 1e-6.

## 4. Primary execution record

**Freeze and launch.**
- Freeze commit `edf53f8`. The freeze manifest (`protocol/freeze_manifest.json`) holds the hashes of the WL114 sources, the unchanged WL112 / WL113 / ECRO dependencies, the WL113 CAND / O_U checkpoints, and the authenticated Ref-65 / Ref-7.
- The primary chain started at 13:41 KST, detached (`chain114.sh`, root `results/ecro_fd114/primary`).
- Focused tests T1–T8 all passed (`tests114.json`).

## 5. Stage A — full-depth target qualification (measured, frozen rules)

**SSS: QUALIFIED, K\* = 24** (`analysis/stageA_qualification.json`).

Camera views (12 views; 2 × 2,048-spp deterministic passes to event 192 and 6 × 2,048-spp roulette passes per view; about 250 s per view):

| View | Near: share of full-depth HIGH beyond event 6 | Near: share beyond event 24 | Whole frame: share beyond event 6 |
|---|---|---|---|
| F0 V0 | 15.3 % | 4.3e-6 | 8.7 % |
| F1 W0 | 15.4 % | 4.5e-6 | 9.0 % |
| F2 W0 (1 cm gap) | 12.8 % | 3.4e-6 | 8.7 % |
| F3 W0 | 16.9 % | 4.9e-6 | 9.2 % |
| F4 W0 | 16.9 % | 4.8e-6 | 9.2 % |
| SEQ_A gap 50 mm V0 | 11.4 % | 2.9e-6 | 8.7 % |
| SEQ_C gap 30 mm W0 | 13.8 % | 3.7e-6 | 9.0 % |
| SEQ_C gap 80 mm W0 | 11.6 % | 3.0e-6 | 9.1 % |
| N1 W0 (new contact) | 15.3 % | 4.2e-6 | 9.0 % |
| N2 W0 (new contact) | 15.8 % | 4.5e-6 | 8.8 % |
| N3 W0 (new 2 cm lift) | 12.8 % | 3.3e-6 | 8.5 % |
| N4 W0 (new 6 cm lift) | 12.3 % | 3.2e-6 | 9.2 % |

- Every region of every view meets C1 at K = 24. No path is alive at event 192.
- Roulette consistency C2 holds everywhere.

Constructions (F0, F1, SEQ_C 80 mm, N3; n = 64; 2 deterministic and 8 roulette replicas):
- Camera events ≥ 7 carry 9.6–9.8 % of the full-depth HIGH over all patches and 10.0–12.8 % on the ring-0 contact patches.
- The share beyond event 24 is 2.4–3.4e-6 (C3).
- Roulette minus deterministic lies within −1.8 to +1.0 SE (C4).

Reading:
- In this SSS family the omitted energy beyond D6 is large (11–17 % of the near-region HIGH).
- The energy beyond event 24 is negligible (≈ 4e-6). The decay is geometric and fast because paths escape the open room.

**LR3: QUALIFIED.**
- L0: the local Mitsuba 3.6.4 re-render (65,536 spp, max_depth 65, seeds 11 / 12) reproduces the authenticated Ref-65.
  - GI mean within +1.2 to +1.3e-4 per channel.
  - RMSE 0.0012 = 0.31 × Ref-65's GI noise RMSE. Not bit-identical: different GPU, 112 s.
- L1: the GI beyond 64 events (∞-roulette minus max_depth 65, common random numbers, 16,384 spp) is 1.2–1.3e-4 of the whole-frame GI. The largest material group is paint, at 5.4e-4. Every region passes 1e-3.
- **Ref-65 is therefore the qualified effectively full-depth LR3 reference.** Ref-7 is the 6-event reference.

LR3 depth convergence (missing GI share, whole frame, R / G / B):

| Events | Missing GI share |
|---|---|
| 6 | 33.3 / 23.2 / 18.7 % (= WL102's truncation) |
| 24 | 0.66 / 0.26 / 0.20 % (paint 1.2 / 0.66 / 0.57 %) |
| 64 | ≈ 1.2e-4 |

- Unlike SSS, **24 events would not be effectively full depth for LR3**. A LR3 full-depth target would need K ≥ 48–64 (informational; the LR3 learned stages are not run here).

## 6. Stage B — oracle feasibility (measured, frozen rules)

**Inputs.**
- Deep 4×4 constructions: n = 1,024, probe events to 23, b = 8. Static fixtures F0–F4 and N1–N4 use 2 replicas; each sequence state uses 1. Construction took 22 min in 2 processes.
- 148 paired streams of 2 × 4,096 spp at 512² (deep physical pass to K\* = 24 + oracle readouts), about 60–90 s each in 2 processes, about 2 h in total.
- Source: `analysis/stageB_metrics.json`.

**Decision: proceed.** The depth effect is attributable at **25 / 25** contact-valid static views: |truncation| ≥ 2 × |S4_F near error|.

Near-region relative bias of the complete frame against the full-depth reference (LOW + HIGH), contact-valid views:

| Class | Truncation (D6 phys − full) | S4_D6 vs full | **S4_F vs full** | S4_D6 vs D6 | U_D6 vs full | **U_F vs full** |
|---|---|---|---|---|---|---|
| Contact states (WL113 static, N1 / N2, contact sequence states) | −10 to −16 % | −8 to −15 % | **+0.2 to +2.6 %** | +0.1 to +2.3 % | +3 to +18 % | **+13 to +32 %** |
| Small gaps (F2 1 cm, N3 2 cm, SEQ_D 15 mm) | −9 to −13 % | −8 to −10 % | +1.0 to +2.6 % | +0.9 to +2.8 % | +3 to +13 % | +13 to +25 % |
| Lifted 30–100 mm (SEQ_A 50 mm, SEQ_C 30 / 80 mm, SEQ_D 40 / 100 mm, N4 6 cm) | −7 to −11 % | −4 to −8 % | **+2.5 to +4.9 %** | +2.6 to +5.0 % | +5 to +15 % | +13 to +27 % |
| F0 H3 / H4 (grazing views) | −7 % | −8 % | −0.8 to −0.9 % | −0.8 to −0.9 % | +16 to +17 % | +23 to +25 % |

Observations:
- **The dense 4×4 spatial residual is unchanged by depth.** S4_F − FULL equals S4_D6 − D6 to within ±0.2 % at every view. The full-depth oracle removes the truncation completely and keeps the same localised spatial error.
- **The full-depth dense oracle fails the approved 3 % contract at every view.**
  - R1L fails everywhere.
  - R4 fails at most views.
  - R1 (and R2 at some views) fails at every lifted 30–100 mm state and at N4.
- At the lifted states the dense oracle also fails the **historical** depth-matched bands (S4_D6 vs D6). WL113's lifted-state SEQ_A / SEQ_C failures therefore contain an oracle (spatial) component and are not purely learned.
- **The coarse table is strongly over-bright once truncation is removed** (+13 to +32 %, against +3 to +18 % at D6). This quantifies WL113's "signed cancellation": the D6 truncation hid most of the coarse spatial over-brightness.

Target distribution change (oracle tables, static fixtures):
- F / D6 HIGH ratio:

  | Patch class | F / D6 |
  |---|---|
  | Ring-0 sphere | 1.10–1.16 |
  | Ring-0 floor (where present) | 1.15–1.29 |
  | Other sphere | 1.08–1.09 |
  | Other floor | 1.11–1.13 |
  | Walls / boxes | 1.105–1.109 |

- Angular-profile correlation between the D6 and F ring-0 tables: median 0.999.
- Within-cell luminance CV on ring-0 patches: median 0.31 (F) vs 0.34 (D6).
- The deep suffix is spatially and angularly smoother than events 3–6 and mostly scales the D6 response, more strongly at the contact.

## 7. Stage C — D114 and authentication

**Generation.** D114 covers 192 training and 16 validation configurations.
- Construction used the D113 seeds and the b = 16 launch partition, deep to probe event 23, with the GPU otherwise idle (peak ≈ 14.8 GB).
- Wall time ≈ 40 min (16:45–17:26), inside the 3.0 h budget.

**Authentication of the D6 part against D113: FAIL by the frozen tolerance.**

| Check | Result | Frozen tolerance |
|---|---|---|
| Files compared | 208 | — |
| Configuration hash mismatches | 0 | — |
| Coarse targets (float32), max per-file rel L1 | 9.8e-6 | ≤ 1e-5 (met) |
| Dense targets (float16 storage), max per-file rel L1 | 1.6e-5 | ≤ 1e-5 (exceeded) |

- The size of the dense excess matches float16 rounding flips caused by float32 kernel-fusion differences of order 1e-6 (tests T2: 6–9e-7). No semantic difference is indicated.
- By the frozen rule, WL113 CAND / O_U are therefore **reported but not used as matched controls**.
- The matched D6 controls are the planned reruns A1r (seed 11330001) and A2 (seed 11330002), trained on the D114 D6 targets themselves.

## 8. Stage C — matched training (measured)

All arms use the unchanged WL113 recipe: 24,000 steps, final step. A1r and B1 share an identical initial-parameter hash (seed 11330001).

| Arm | Representation / target / seed | Val refined rel L1 | Refined half-noise | Ratio | Val coarse rel L1 | Train wall (contended) |
|---|---|---|---|---|---|---|
| A1r | dense S4 / D6 / 11330001 | 0.0859 | 0.0823 | 1.044 | 0.0246 | 3,707 s |
| **B1** | dense S4 / **full** / 11330001 | 0.0832 | 0.0798 | **1.043** | 0.0238 | 3,673 s |
| A2 | dense S4 / D6 / 11330002 | 0.0859 | 0.0823 | 1.044 | 0.0247 | 3,492 s |
| **B2** | dense S4 / **full** / 11330002 | 0.0831 | 0.0798 | **1.041** | 0.0237 | 3,490 s |
| OB1 | coarse v0 / full / 11330001 | — | — | — | 0.0237 (half-noise 0.0192) | 1,133 s |

- A1r reproduces WL113 CAND's validation numbers (0.0859 / 0.0247).
- At table level, full-depth supervision is learned **as accurately relative to its target noise as D6** in both seeds.
- Peak memory is 5.3 GB per dense arm. Total training wall is 2 h 3 min (two rounds), inside the 3.5 h budget.

## 9. Stage D — SSS validation (measured; `analysis/stageD_metrics.json`, `stageE_metrics.json`, `verdicts.json`)

**Coverage.**
- 74 views; 73 are contact-valid.
- Static: WL113 regression F0–F4 (18 views) and new N1–N4 (8 views).
- Sequences: SEQ_A / B / C (34 state-views) and the new SEQ_D (14).
- The same weights are used throughout. LOW pairing between the physical pass and the readout holds to ≤ 1e-7 (float32 summation).

Near-region relative bias of the complete frame against the full-depth reference: mean over views [min, max].

| Component | Contact / small-gap views (59) | Lifted 30–100 mm views (14) |
|---|---|---|
| Truncation (D6 − FULL) | −12.2 % [−16.0, −6.8] | −8.8 % [−10.7, −7.0] |
| Spatial, full-depth dense oracle (S4_F − FULL) | +1.4 % [−0.9, +2.6] | +3.5 % [+2.5, +4.9] |
| Spatial, D6 oracle vs D6 | +1.4 % [−0.9, +2.8] | +3.7 % [+2.6, +5.0] |
| Learned B1 (B1 − S4_F) | −0.9 % [−2.7, +1.4] | +0.6 % [−1.6, +2.2] |
| Learned B2 | −0.1 % [−2.4, +2.2] | +0.7 % [−1.5, +2.0] |
| Learned A1r (A1r − S4_D6, vs D6) | −0.6 % [−3.7, +2.4] | +0.9 % [−1.9, +2.8] |
| **Total B1 vs FULL** | **+0.4 % [−1.8, +3.9]** | **+4.2 % [+1.5, +7.0]** |
| Total B2 vs FULL | +1.2 % [−2.3, +4.2] | +4.3 % [+1.7, +6.9] |
| Total A1r vs FULL | −11.5 % [−17.4, −8.3] | −4.7 % [−7.1, −2.5] |
| Total OB1 (coarse, full) vs FULL | +24.4 % [+15.8, +30.9] | +17.6 % [+12.4, +25.5] |

**Frozen H1 rule — verdict INCONCLUSIVE.**

| Criterion | B1 | B2 | Rule |
|---|---|---|---|
| Views where \|bias\| is reduced vs the matched A | 0.890 (65 / 73) | 0.890 | ≥ 0.90 |
| Learned excess within 1.25 × A's + 0.005 | 0.959 | 0.904 | ≥ 0.90 (met) |
| Table / noise ratio | 1.043 vs A1r 1.044 | 1.041 vs A2 1.044 | met |

- The reduction criterion misses by one view of 73.
- The 8 "no reduction" views are all lifted 30–100 mm states: SEQ_C 30 / 80 mm (3 state-views), SEQ_D 40 / 100 mm (4), N4 6 cm W0 (1).
- At these views the D6 arm's smaller \|bias\| is **signed cancellation**, not accuracy: truncation (−7 to −11 %) offsets the dense-oracle spatial over-brightness (+2.5 to +4.9 %) plus a learned excess.
- The rule was frozen; it is not reinterpreted. The verdict stands as INCONCLUSIVE.

**Approved 3 % gate at full depth (primary).** Each cell shows views failing a requirement, out of 74; "pass" is the number of views passing all requirements.

| Arm | Pass | R1L | R4 | R1 | R2 | R3 bias | R5 |
|---|---|---|---|---|---|---|---|
| B1 | 1 | 73 | 36 | 17 | 13 | 2 | 0 |
| B2 | 1 | 73 | 44 | 20 | 13 | 4 | 0 |
| Full-depth dense oracle S4_F | 1 | 73 | 49 | 10 | 5 | 0 | 0 |
| A1r (D6 arm vs FULL) | 0 | — | — | — | — | — | — |

- The single passing view is the non-contact-valid F0 H5, where only R5 applies.
- A1r fails R1 71, R3 73, R4 73 and R5a / b bias and RMSE at nearly all views.
- R6b (camera-spread of the near bias / contrast within a fixture): passes for B1 and S4_F at every fixture (spreads 0.002–0.026).
- Attribution of B1's failures by the frozen rule (spatial if S4_F also fails, otherwise learned):
  - R1L: 73 spatial / 0 learned;
  - R4: 33 / 3;
  - R1: 8 / 9;
  - R2: 5 / 8;
  - R3: 0 / 2.

**Dynamic reuse (same weights).**
- Fresh-process rebuild and every revisit table: **bit-identical** (max rel 0.0) for A1r and B1, in all four sequences.
- Discriminating transitions (meaningful full-depth HIGH change ≥ 5 SE and stale table worse than current), B1: SEQ_A 12 / 12, SEQ_B 6 / 8, SEQ_C 6 / 8, SEQ_D 6 / 12. Far-region RMSE is unchanged in every sequence.
- The stale previous-state full-depth table errs by up to +25 % when the sphere returns from 100 mm to 15 mm (figure `SEQ_D_independent_dynamic_near_bias_full_depth.png`).
- R7 tracking (full depth):
  - B1 slope 1.09–1.22 (corr ≥ 0.999); it passes only SEQ_A H2.
  - Post hoc, descriptive: the full-depth dense **oracle** S4_F has R7 slopes 1.07–1.13. The tracking overshoot is mainly a spatial-representation effect at the lifted states.
- The historical depth-matched contract for the D6 arm A1r holds in SEQ_B only, as in WL113.

## 10. Stage F — cost (measured; `analysis/cost.json`; unoptimised Mitsuba 3.5.1 / Dr.Jit 0.4.6 runtime)

**Offline cost.**
- Physical target generation: deep construction to K\* = 24 takes 1.25 s vs 0.61 s for the WL113 depth-6 `construct4` per light at n = 32 (**2.04×**).
- D114 size: 17.6 GB vs D113 10.1 GB (extra full-depth targets).
- D114 generation wall: about 40 min, single process (D113: about 32 min in 2 processes).

**Inference cost** (A1r vs B1, same architecture):

| | A1r | B1 |
|---|---|---|
| Generator | 20.5 ms | 18.8 ms |
| Table | 15.8 MiB | 15.8 MiB |
| 512² 1-spp frame | 39.7 ms | 40.7 ms |
| Geometry update | 151 ms | 153 ms |

- These are equal within measurement noise. The coarse OB1 generator takes 2.8 ms.
- **Increasing the target depth adds offline cost only; inference work is unchanged.** It changes what the same work produces: −11.5 % → +0.4 % mean near bias at contact.

## 11. Living Room 3

- Stage A qualified Ref-65 as the effectively full-depth reference (§5).
- **LR3 Stages B–E: NOT_RUN (execution-stack constraint).** The ECRO stack cannot load LR3 on the local GPU (§3). The owner question about LabServer69 was not answered during the batch, so no cross-machine split was made.
- This is not a negative result. What stays open for LR3:
  - oracle representability at full depth;
  - learned full-depth fidelity;
  - comparison with TransGI 0.0775 / PT + denoiser on Ref-65.

## 12. Final report

### Implementation Fact
- Experiment-local only. Added:
  - a deep, event-bucketed physical suffix estimator with exact compaction and verification roulette;
  - full-depth oracle tables under the unchanged WL113 coarse and dense contracts;
  - dataset D114;
  - five matched arms;
  - the evaluation / analysis pipeline.
- ECRO architectures, capacities, spatial representations, ER / material / medium, LOW ownership and thresholds are unchanged. No shared production file was edited.
- The training objective is unchanged; within a pair only the global target key differs.

### Measurement
- **Physical target sufficiency.**
  - SSS: K\* = 24. The omitted share beyond event 24 is ≈ 4e-6; the share beyond event 6 is 11–17 % of the near HIGH. Roulette consistency passes.
  - LR3: Ref-65 is authenticated and effectively full depth (beyond 64 events ≤ 5.4e-4). At 24 events LR3 still misses 0.2–0.7 %.
- **Existing representation adequacy.**
  - The full-depth dense 4×4 oracle removes truncation completely.
  - Its spatial residual equals the D6 one (contact +1.4 %, lifted +3.5 %). It fails R1L at every view and R4 at 49 / 74.
  - The coarse table is +13 to +32 % over-bright at full depth.
- **Learned full-depth approximation.**
  - Table-level accuracy relative to noise is identical to D6 (1.04, both seeds).
  - Image-level learned excess over the oracle stays at the D6 level (−0.9 to +0.7 % mean).
  - Mean near bias at contact goes from −11.5 % (D6 arm) to +0.4 % / +1.2 % (B1 / B2).
- **SSS spatial correctness.** The approved 3 % contract fails at full depth for the oracle and for both learned seeds. R1L fails everywhere (spatial). R4, R1 and R2 fail mostly at lifted states, with mixed spatial / learned attribution.
- **SSS full-depth correctness.** Truncation is removed at contact. At lifted states the remaining error is spatial over-brightness (+2.5 to +4.9 %) plus a learned excess (≤ +2.2 %).
- **Dynamic reuse.** Rebuild and revisit are exact, transitions are discriminating, and stale tables are clearly wrong. R7 overshoots (1.09–1.22), and the oracle also overshoots (1.07–1.13).
- **LR3 quality improvement.** Not measured: Stages B–E NOT_RUN.
- **Offline vs runtime cost.** Target generation ≈ 2×, data 1.7×; inference unchanged.

### Observation
- With truncation removed, the remaining SSS near-contact error is **visible and localised**: within about 10 cm of the support (R1L, innermost R4 rings), and as a whole-near over-brightness at lifted 30–100 mm states.
- The D6 truncation previously masked part of both the dense lifted-state over-brightness and the coarse over-brightness.

### Interpretation (bounded to this SSS family, one radius / medium, one light per configuration)
- The existing ECRO response architecture learns the effectively full-depth HIGH target at the same table-level accuracy as D6 and reuses it across geometry changes with exact rebuild / revisit. No capacity or representation change was needed for depth.
- The frozen H1 verdict is nevertheless **INCONCLUSIVE**: the reduction criterion misses by one view, and the shortfall is entirely at lifted states, where the D6 arm benefits from signed cancellation.
- **After depth is removed, the remaining SSS errors that require improvement are:**
  - (i) the spatial resolution of the HIGH output near the support (R1L everywhere, most R4; also present in the full-depth oracle);
  - (ii) the spatial representation at lifted 30–100 mm states (oracle +2.5 to +4.9 %);
  - (iii) a smaller learned-generalisation excess at new / lifted positions (B − oracle up to about +2 %; R1 / R2 failures split about evenly between spatial and learned).

### Unresolved Question
- Is the lifted-state over-brightness of the dense 4×4 oracle a projection effect of the uniform 4×4 measure at non-contact geometry, or a resolution limit?
- Would LR3 (albedo ≈ 1, K ≥ 48–64 needed) be learned equally well, given that the energy shares there are much larger?
- How much of R4 is learned vs spatial at individual rings?
- R3 (equal-time) and R9 (radius / medium) are NOT_RUN.

### Architecture Verdict
- **H1 (full-depth learnable without architecture change): INCONCLUSIVE** by the frozen rule. Descriptively, the depth component is learned and reused.
- **Mandatory SSS gate: NOT_RESOLVED.**
  - The approved 3 % contract fails for B1, B2 and the full-depth oracle.
  - RESOLVED was unreachable anyway (R3 and R9 NOT_RUN).
- Nothing is adopted. Dense 4×4 remains an experimental representation; compact VHS ∪ MEDIUM remains unadopted.
- LR3: reference qualified; learned stages NOT_RUN.

### Reproducibility
- Freeze commit `edf53f8`; hashes in `protocol/freeze_manifest.json`.
- Primary chain: 13:41–20:10 KST on 2026-11-25, local RTX 5080, `chain114.sh`.
- Raw outputs: `results/ecro_fd114/primary/` (Stage A sums / images, oracle tables, 148 Stage B and 148 Stage D streams, D114, checkpoints with sha256 in `training.json`, analysis JSONs).
- Seeds per `protocol/fd114.json`. The dev chain (`results/ecro_fd114/dev`) is non-evidence.
- Reviewer figures: `evaluation/114/` (66 figures; `manifest.json` with hashes; one absolute residual scale of 0.806 linear HDR, sqrt display).
