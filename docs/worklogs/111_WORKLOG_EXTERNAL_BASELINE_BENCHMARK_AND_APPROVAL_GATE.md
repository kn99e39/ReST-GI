# Worklog — External Baseline Benchmark and Contribution Approval Gate (2026-11-24)

> Append-only execution record. Batch class: **regular WL111** (owner brief "ReST-GI — External Baseline Benchmark and
> Contribution Approval Gate", 2026-11-24). Branch `Denoise_Reconstructor` (current branch; no new branch or worktree).
> Execution host: **Main** (RTX 5080), by owner directive during the session ("넌 ssh Main에서 작업해").
> Protocol: `experiments/bench111/protocol/bench111.json`. Code: `experiments/bench111/`.

## 2026-11-24 — Brief, numbering and scope

- **Question.** Exactly what quality, performance, dynamic-reuse and correctness requirements must ReST-GI satisfy to
  advance to its next milestone? Answer from executed, qualified external baselines where feasible.
- **Deliverables required by the brief.** Baseline registry; workload compatibility matrix; reproduction / comparability
  report; executed baseline manifests; raw measurements; quality–performance frontiers; shared-scale images; dynamic and
  lifecycle comparisons; claim-to-evidence matrix; numerical approval-gate proposal; ReST-GI gap analysis; separate
  mandatory SSS near-contact gate; reviewer index.
- **Not authorised.** Redesigning ER or ECRO; changing LOW/HIGH; adopting ER-v1-SDM; tuning ReST-GI to beat a baseline;
  solving SSS near-contact; changing reference outputs or another batch's files; automatic start of the next milestone.
- **Number.** WL111 is the next integer free on every branch: WL110 / WL110A exist on `ecro_lr3_adapter`; HEAD's highest
  is WL109B. WL111 is regular because it opens an independent benchmark / decision-contract question, not a follow-up of
  one completed batch.
- **Active work left untouched.**
  - WL109B is running on LabServer69 under its guard, and the owner's TransGI reproduction batch `b28` runs there in tmux.
    WL111 uses LabServer69 only for read-only file copies.
  - Main's checkout `C:\Projects\ReST-GI` is on `ecro_lr3_adapter` and holds another session's untracked
    `experiments/ecro_live101j/`. WL111 reads frozen sources, checkpoints and the Mitsuba 3.9.1 runtime from it and
    writes nothing there. WL111 code and outputs live in the isolated `C:\Projects\bench111`.

## 2026-11-24 — Sources read (before design)

- Canonical documents: Research topic, Architecture (HEAD and `ecro_lr3_adapter`), Roadmap (both branches),
  Evidence strategy, AGENTS.md.
- SSS chain: WL100, WL101A–G (HEAD); WL101H, WL101I, the construction-qualification plan and the WL101G forensic audit
  (`ecro_lr3_adapter`); denoising track WL103, WL105–WL109B.
- Baselines: WL21A/B, WL22, WL27, WL35, WL43 (`baseline/8dna_dynamic_reuse`), WL67, WL69, WL70, WL102
  (`ecro_lr3_adapter`); the TransGI reproduction's `REPRODUCTION_STATUS.md` (LabServer69, read-only).
- Current architecture and candidates: WL86–WL99, WL104–WL104K-C, WL105 fabric, WL110 / WL110A (`ecro_lr3_adapter`).
- Facts that shape the design:
  - Canonical implemented ReST-GI = learned ER (local) + ECRO v0 (LOW E1 + E2 physical, learned HIGH E3–E6 table,
    D = 6). ER-v1-SDM is a closed candidate (WL104K-C: not adopted). EXP / C3UR are explicit comparators.
  - No positive contribution claim exists in any canonical document.
  - TransGI is listed as "blocked (no released implementation)" in the evidence-strategy document, but an independent
    reproduction exists (WL102 and the owner's TransGI_Reproduction). That line is stale.
  - No NRC implementation exists on any branch.

## 2026-11-24 — Host and isolation

- Main: RTX 5080 16 GB, driver 610.88, Windows 11 (WDDM; the GPU also drives the desktop). Stack = WL101 arm B runtime:
  `external/8dna26/.venv-cu128` (Python 3.11.9, torch 2.8.0+cu128) + `external/mi39_dyn101` (Mitsuba 3.9.1 /
  Dr.Jit 1.5.0), cuda_ad_rgb, FREEZE=1. `mi.OptixDenoiser` is available.
- Long jobs are started detached through WMI (`win/launch.ps1`), so they survive the ssh session; verified with a test job.
- Frozen sources on Main are byte-identical to HEAD (sha256 of rt101, pt100, pt86, model86, common100 checked against
  both `Denoise_Reconstructor` and `ecro_lr3_adapter`).

## 2026-11-24 — Benchmark adapter and development smoke (not evidence)

- **New code (benchmark adapter only).** `pt111.py` subclasses the frozen WL101 runtime and adds one frame program: the
  pt100 per-vertex statements (scene, ER closure, SSS first-exit walk, direct and ballistic-through-sphere NEE, event
  bookkeeping) inside a symbolic `dr.while_loop`, so the physical path tracer can run at full depth (kmax 256) with
  Russian roulette from event 4. Guide buffers for the denoiser come from the same samples. ECRO frames are untouched
  (`rt101.RT.frame`).
- **Development smoke** (`dev111.py`, S0 V0 only; no method-vs-reference comparison):
  - At kmax 6 the adapter equals the archived WL100 reference (independent pt100 code path): image means within
    3.5e-4, RMSE / expected noise 1.00–1.07 (e1 / e2 / h). Roulette at fixed depth is unbiased (≤ 1e-4).
  - Energy beyond six events on S0 V0: 1.6–2.0 % of the total (the room is lossy; unlike LR3, where WL102 measured
    −19…−33 % GI truncation).
  - Raw frame times (not evidence): ECRO 512² × 1 spp 0.80 ms, × 4 spp 3.19 ms; PT full depth 2.29 / 10.36 ms;
    OptiX denoiser call ≈ 3.2 ms including a host upload.
  - Full-depth reference cost ≈ 5 ms per spp at 512² → 2 × 32,768 spp ≈ 5.5 min per state / view.

## 2026-11-24 — Code-path smoke findings, fixed before any evidence

All three were found by `run111.py all --smoke` (256-spp smoke references; no evidence produced, smoke outputs
discarded). Values below are smoke values.

1. **N0 camera.** The harness gave the near-mirror fixture the SSS V0 camera. The WL100 near fixture has its own
   camera (origin (0.3, −1.6, 1.3), target (−1.3, 0.45, 1.0)). G_ADAPT failed (e1 −64 %). Fixed: V0 = the WL100 dataset
   camera of each case.
2. **G_ADAPT rule.** With the right camera N0 still failed the first written rule (|relative mean difference| ≤ 3e-3
   without a noise term). e1 matched to 2e-4, but the heavy-tailed near-mirror e2 / h differed by 0.1–0.4 %. Two
   distributionally identical estimators (e2 is not affected by roulette) differed by 0.6 % with these sample counts.
   Rule changed to a noise-scaled form: |rel| ≤ 3e-3 or |z| ≤ 4, with RMSE / noise in [0.75, 1.25]. S0 passed both forms.
3. **One runtime at a time.** rt101 keeps the closure table in a module-level dict that belongs to the most recently
   constructed runtime. With the SSS and near runtimes alive together, `dr.freeze` failed to record. Worse, it could
   have read the wrong material table. Fixed: the harness keeps one case runtime alive and rebuilds on a case switch.

The complete smoke chain then ran all stages (gates, references, quality, timing, dynamic).

## 2026-11-24 — Focused tests (Main, before evidence): 20 / 20 PASS

`tests111.py`:
- metric arithmetic (relative RMSE, debiasing, floor, LDR, masks);
- seed determinism, distinctness (2,816 / 2,816) and range;
- denoiser device path == host path (max |Δ| 0.0); denoiser energy ratio 1.004;
- S0 V0 masks reproduce WL101C near 1,318 / flank 6,042 and WL101A sphere 49,660 pixels exactly;
- ECRO frame deterministic and finite;
- selective relation update == full rebuild at T1 (89,926 dirty rays, both boxes moved);
- T0 → T1 → T0 restores relation state and learned table bit-exactly;
- light change moves no geometry;
- the exclusivity probe works (no other python process, idle GPU 0 %).

## 2026-11-24 — Freeze

Protocol and code are committed before the evidence chain. Frozen decisions (details in the protocol):
- **Workloads.** Fixtures run fresh on Main:
  - S0 = W1 SSS cfg0, views V0 / H1 / H2;
  - S1 = W3 rigid box edit target;
  - SL = W5 light change target;
  - N0 = W2 near-mirror.
  
  LR3 (W4) follows with a separate harness under the same rules.
- **Arms.** ECRO (canonical, frozen), ECRO_CONV (diagnostic), ECRO_STALE (dynamic control), PT (full depth), PTDN
  (PT + OptiX). ECRO + denoiser is NOT_RUN (it would be a new ReST-GI pipeline variant).
- **References.**
  - Primary: REF_FULL, kmax 256, 2 × 32,768 spp.
  - Depth-matched diagnostic: REF_D6, 2 × 8,192.
  - Truncation is never removed from a primary score.
- **Primary metric.** Full-frame relative debiased HDR RMSE against REF_FULL, as the expected per-frame error over
  replicates. Secondary metrics and SSS near-contact quantities are as listed in the protocol.
- **Timing.** Median / p95 / p99 frame time on Main in exclusive windows. Static frame and edit → first frame are
  reported separately. Display is excluded.
- **Frontier and gate derivation rule.** Margins of 5 % / 20 % (initial owner values) are assessed against measured
  precision. Gates are PENDING_OWNER_APPROVAL.
