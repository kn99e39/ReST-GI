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
