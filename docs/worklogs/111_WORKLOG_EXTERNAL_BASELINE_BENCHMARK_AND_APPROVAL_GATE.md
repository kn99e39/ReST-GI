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

## 2026-11-24 — Execution record

**Fixture chain.** `chain.cmd 7d64755` on Main, 16:31–17:02, all stages completed, no preemption.
- Tests 20 / 20 PASS. G_ADAPT PASS on S0 and N0 (noise-scaled rule).
- References: FULL 2 × 32,768 spp at ≈ 81 s per half (S0 / S1 / SL views), 22 s (N0); D6 2 × 8,192.
- Quality: 299 rows per state / view.
- Timing: every block exclusive on its first attempt (no other python, idle GPU before and after).
- Dynamic: both transitions exclusive.

**LR3 chain.**
- `lr3_chain.cmd` (code `a9bcfd7`; the status file's commit label shows the literal `$(git` because of a shell-quoting slip in the launch command) waited for the fixture chain, then ran the PT ladder: 17:02–17:03, timing exclusive.
- **Execution-staging defect (disclosed).** The three TransGI evaluations and two benchmarks failed at once. The reproduction's `Session` also reads `datasets/living_room_3/{scene_manifest,material_tables}.json` from its data root, which the isolated data copy lacked.
  - Both files were copied from LabServer69 (read-only).
  - Only the TransGI steps were re-run (`lr3_tg.cmd`, 17:05–17:06).
  - No protocol, model or option changed; no TransGI output had existed.

**Analysis-code correction (disclosed; analysis written before results, corrected against the frozen text).**
- The first `analyze111.t_to()` returned a comparator's cheapest executed time whenever even that point was better than the ReST-GI error.
- The frozen rule says "beyond executed range = not evaluable". The code now follows the text, and the mechanical verdicts below use it.
- A descriptive dominance record (best comparator point within 1.5× time; cheapest comparator point with lower error) is added next to them. It is not a gate rule.

## 2026-11-24 — Final report

### Implementation fact
- **Benchmark adapter.** pt111 (full-depth, same-semantics physical PT with RR) validated against the independent pt100 path. The OptiX denoiser arm runs on device. LR3: Mitsuba stock `path` + OptiX; the TransGI reproduction runs unchanged on its own stacks.
- **Canonical ReST-GI as tested.** ECRO v0 with the WL100 checkpoints and the WL101 optimised runtime (FREEZE = 1), D = 6, no change of any kind.
- **Isolation.** Nothing written to Main's `C:\Projects\ReST-GI` or the TransGI data root. LabServer69 used for read-only copies only. WL109B and the owner's TransGI `b28` were not touched.

### Measurement (RTX 5080; E = relative debiased HDR RMSE vs full-depth reference; 95 % CIs in `evaluation/111/summary111.json`)

**References.**
- S0 / S1 / SL: half-vs-half relative RMSE ≈ 0.0035 (mean noise 0.18 % per pixel).
- N0: 0.32 (16 % per pixel, heavy-tailed caustics) → **N0 unqualified** for pixel metrics.
- Energy beyond six events: −1.7…−2.1 % full frame; **−13 % in the SSS near-contact ROI**. D6 vs full E = 0.019.

**W1 S0 V0 (H1 / H2 within ±1 % of these):**

| Arm | spp | ms (median) | E |
|---|---|---|---|
| ECRO | 1 / 4 / 16 / 64 | 0.81 / 3.06 / 11.2 / 43.8 | 0.320 / 0.161 / 0.082 / 0.044 |
| ECRO converged | 4,096 | — | 0.0206 (vs D6 0.0057) |
| PT | 1 / 4 / 16 / 64 | 2.38 / 10.3 / 39.4 / 157 | 0.510 / 0.255 / 0.128 / 0.063 |
| PT + OptiX | 1 / 4 / 16 / 64 | 3.88 / 11.7 / 41.0 / 158 | 0.031 / 0.018 / 0.011 / 0.0075 |

- **W3 (after box edit) / W5 (after light change):** same ordering. ECRO 16 spp: 0.086 / 0.155; PT + OptiX 4 spp: 0.016 / 0.047.
- **Dynamic (edit → first 512² × 1 spp frame):**

  | Transition | ECRO selective | ECRO full rebuild | PT | PT + OptiX |
  |---|---|---|---|---|
  | Box edit | 6.71 ms | 5.19 ms | 3.87 ms | 5.24 ms |
  | Light change | 3.97 ms | 4.35 ms | 2.83 ms | 4.26 ms |

  Stale-table E is within 2 % of the updated table (box edit) and within 1–4 % (light change).
- **W2 N0 (unqualified reference, descriptive):** PT + OptiX panel-ROI error 0.017–0.11; ECRO 0.058–0.32.
- **W4 LR3 (GI LDR RMSE vs Ref-65, 640 × 360):**

  | Arm | Time | GI LDR RMSE |
  |---|---|---|
  | PT 1 / 16 / 64 spp | 5.1 / 15.1 / 47.1 ms | 0.413 / 0.150 / 0.085 |
  | PT + OptiX 1 / 16 spp | 7.7 / 17.7 ms | 0.039 / 0.023 |
  | TransGI L40, fast-frame settings | 5.16 ms (1080p 22.5 ms) | 0.061 |
  | TransGI L40, adopted options | — | 0.053 |
  | TransGI WL102 model | — | 0.077 |
  | ECRO 4 spp (historical, RTX 3090) | 27.7 ms | 0.170 |

  ECRO converged (historical) 0.102; Ref-7 truncation alone 0.0963.
- **SSS near contact (ECRO, converged = sampled):** relative near bias +10.1 % (V0), −6.6 % (H1), +1.2 % (H2), +8.3 % (S1), +20 % (SL). Near-minus-flank excess relative to near reference luminance: +23 %, +6 %, +14 %, +21 %, +29 %. PT + OptiX 16 spp: within ±0.6 % everywhere; at 1 spp +5…+8 % near and −5.5 % silhouette bias at V0.

### Observation
- At equal time, ECRO's LOW + lookup estimator has ≈ 3× lower error than raw full-depth path tracing on every fixture state.
- A 1.5 ms OptiX denoiser on 1-spp path tracing beats ECRO at every ECRO operating point, including converged ECRO: 0.031 at 3.9 ms vs 0.021 floor reached only near 4 s.
- On LR3, PT + OptiX at 7.7 ms has lower GI error than TransGI's adopted-quality evaluation and than every published TransGI LR3 number. TransGI remains on the frontier below 7.7 ms.
- On the 5080, ECRO's selective relation update is slower than its full rebuild (6.71 vs 5.19 ms) at N = 1,270.
- The WL111 box edit changes ECRO's frame error by < 2 %, so it does not test reuse.

### Interpretation
- The canonical ReST-GI is not on the quality–performance frontier of any qualified workload once a practical denoiser is admitted.
- Its gap to the proposed 10 / 15 / 20 ms tiers is ≈ 5× error on W1 and ≥ 5–7× on LR3.
- **The fixed D = 6 truncation alone exceeds every tier target on W1 and LR3.**
- The near-contact artifact is a representation bias. It is present against the full-depth reference, in held-out views and after edits.
- These are statements about the current canonical configuration against this benchmark. They are not a falsification of learned transport reuse as a research direction, and they recommend no architecture.

### Unresolved question
- Owner approval of the proposed gate: budget anchoring, monotone envelope, margins 5 / 20 %, SSS bands 3 %.
- A converged caustic-capable near-mirror reference.
- A reuse-qualifying edit workload.
- NRC (REPRODUCTION_REQUIRED).
- Shared-scene edits against TransGI.
- 1080p tiers.
- SSS dynamic contact classes R7–R9 (fixtures not yet built).

### Architecture verdict
**No architecture decision.** Gate statuses for the canonical ReST-GI:
- W1 / W3 / W5 / LR3 tiers: FAIL;
- edit → first frame: FAIL;
- W3 as a reuse test: UNQUALIFIED;
- W2: UNQUALIFIED;
- SSS near-contact gate: FAIL (dynamic classes NOT_RUN);
- novelty condition: NOT_RUN in its gate form;
- numerical thresholds: PENDING_OWNER_APPROVAL.

**Mechanical frozen-rule verdicts (reported unchanged).** PATH 1 / PATH 2 "PASS" at ECRO 1–4 spp against raw PT only, with PT + OptiX not evaluable there. Every other ECRO operating point is FAIL on PATH 2. The gate document explains why this rule was replaced by the budget-anchored proposal.

### Reproducibility
- **Code and protocol.** `experiments/bench111/` at `7d64755` (fixtures, frozen before evidence) and `a9bcfd7` (LR3 protocol / chain, before any LR3 result). Analysis and figures are in the results commit.
- **Run manifests.** `C:\Projects\bench111\evidence\manifest.json` holds protocol sha, code sha, frozen-source sha, checkpoint sha and versions. Local copy: `results/bench111/`.
- **Stack.** Python 3.11.9, torch 2.8.0+cu128, Mitsuba 3.9.1 / Dr.Jit 1.5.0 (ECRO, PT, LR3 PT, TransGI fast frame); Mitsuba 3.6.4 / Dr.Jit 1.0.5 (TransGI evaluator). RTX 5080, driver 610.88.
- **Hashes.**
  - TransGI checkpoints: L40 `1b32a633…`, WL102 `1b75c8a6…`.
  - Ref-65 `62625251…`.
  - ECRO SSS checkpoint `2d9cb580…`.
- **Seeds.** sha256 → splitmix64 (protocol). Raw-result hashes in `evaluation/111/manifest111.json`.

## 2026-11-24 — Owner decision on the gate proposal (recorded; WL111 measurements and verdicts unchanged)

- **Approved as the canonical decision contract:**
  - quality non-inferiority 5 %; performance non-inferiority 5 %;
  - quality superiority 20 %; performance superiority 20 %;
  - runtime tiers 10 / 15 / 20 ms;
  - budget-anchored quality–performance frontier with a monotone envelope;
  - measurement uncertainty must be reflected in every equivalence and superiority judgment.
- **WL111 workload absolute targets:** approved as **Provisional Development Targets**. When further baselines are qualified, the qualified frontier is updated. The standard is never relaxed because ReST-GI does not meet it.
- **SSS near contact.**
  - The gate is mandatory and cannot be compensated by other results.
  - The 3 % band is **conditionally approved**. Conditions: the error normalisation is explicit, and the gate also requires localized artifacts, near-minus-flank contrast, cross-view consistency, dynamic contact and off-target non-regression alongside regional bias.
- **Dynamic reuse:** final judgment deferred, because no edit workload has yet passed qualification.
- **Architecture novelty and causal ablation:** remain mandatory conditions separate from numerical superiority.
- **Not approved:** adoption of the current ReST-GI architecture, and entry into the next milestone.
- **Living documents updated accordingly.**
  - Contribution Approval Gate: status table; approved contract vs provisional targets; frontier update rule; uncertainty rule; dynamic judgment deferred.
  - SSS Near-Contact Gate: §3.1 normalisation; R1L localized artifact (window detection test, Bonferroni); R5b off-target non-regression; R6b cross-view consistency; dynamic contact R7–R8 explicit; R3 contact-disk bias aligned to 3 %.
  - Pointers in Roadmap, Research topic, Evidence strategy, Architecture, README and the claim matrix.
