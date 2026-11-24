# Contribution Approval Gate and Milestone Exit Criteria

**Status: PENDING_OWNER_APPROVAL** (proposed by WL111, 2026-11-24). Nothing here is canonical until the research owner approves it. Once approved it becomes a fixed research contract for subsequent development: future batches may tighten it prospectively, but may not loosen it after results.

**Role:** the quality, performance, dynamic-reuse and correctness requirements ReST-GI must meet to advance to its next milestone. Supporting documents:
- [External baseline registry](EXTERNAL_BASELINE_REGISTRY.md) — baselines;
- [benchmark protocol](BENCHMARK_PROTOCOL.md) — how to measure;
- [claim-to-evidence matrix](CLAIM_EVIDENCE_MATRIX.md) — novelty evidence;
- [SSS near-contact gate](SSS_NEAR_CONTACT_GATE.md) — mandatory and separate.

Measurements: [WL111](worklogs/111_WORKLOG_EXTERNAL_BASELINE_BENCHMARK_AND_APPROVAL_GATE.md); reviewer package `evaluation/111/`.

## 1. Decision rule

ReST-GI passes a workload tier when **either** path holds against the **strongest valid comparator frontier** F of that workload, measured on the benchmark host. The frontier is the lower envelope of every executed comparator; currently that is path tracing + OptiX denoiser on every qualified workload.

The rule is anchored at the owner's frame budgets b ∈ {10 ms ideal, 15 ms desired, 20 ms acceptable}. The **tier error** is T_b = E_F(b), the frontier's error at time b. For a comparator that cannot spend more time, the envelope is monotone: a point (t_c, E_c) with t_c ≤ b counts at b.

| Path | Requirement at budget b | Meaning |
|---|---|---|
| **PATH 1** — quality non-inferiority + performance superiority | ReST-GI reaches E ≤ (1 + m_q) · T_b at t ≤ (1 − s_t) · b | Same quality as the best practical method at that budget, ≥ 20 % faster |
| **PATH 2** — performance non-inferiority + quality superiority | at t ≤ (1 + m_t) · b, ReST-GI has E ≤ (1 − s_q) · T_b | Same time budget, ≥ 20 % lower error |

Recommended margins: m_q = m_t = 0.05, s_q = s_t = 0.20. These are the owner's initial values, **kept unchanged**; justification in §3. PASS requires the **whole 95 % interval** of the ReST-GI quantity to satisfy the inequality. An interval straddling the threshold is UNRESOLVED.

**Additional conditions, all mandatory:**
1. **Independence.** Every mandatory workload tier and every qualified view passes on its own. No averaging across workloads, views or materials.
2. **Dynamic.** The first frame after a rigid edit or a light change satisfies the same tier. Its time is measured edit → first frame and includes every update (relation, source, generator, BVH, history reset). On a qualified edit workload the stale-state control must fail the tier (§5).
3. **SSS near contact.** The [SSS gate](SSS_NEAR_CONTACT_GATE.md) passes. It cannot be compensated.
4. **Novelty evidence.** The passing configuration fails the same tier when its learned component is replaced by the learning-free explicit counterpart at equal time, or that counterpart is shown to cost more than the budget (claim K1 / K2 in the [matrix](CLAIM_EVIDENCE_MATRIX.md)).
5. **Lifecycle accounting** is reported (§6). Per-edit retraining is not allowed inside a real-time claim. If a mode needs it, the retraining time counts as edit latency.

### Why the rule is anchored at budgets (correction of the WL111 frozen evaluation rule)

The WL111 protocol evaluated PATH 1 / 2 at ReST-GI's own operating points. A comparator whose executed error range did not contain ReST-GI's error was "not evaluable". On every qualified workload, PT + OptiX at its cheapest point (1 spp) already has 5–10× lower error than ECRO at 1–4 spp. That point is ruled "not evaluable", which leaves only raw path tracing. The mechanical result is therefore "PASS" at ECRO 1–4 spp against raw PT only, at errors of 0.16–0.61. Those are not usable quality levels: PSNR 13–20 dB LDR, against 37–41 dB for the dominating PT + OptiX point at 1.3–4.8× the time. WL111 reports these mechanical verdicts unchanged. This document proposes budget anchoring and the monotone envelope so that a dominated, unusable operating point cannot pass.

## 2. Hardware, resolution and cost accounting for the budgets

- **Host.** RTX 5080 16 GB (Main, Windows 11, driver 610.88), exclusive windows per the [benchmark protocol](BENCHMARK_PROTOCOL.md) §4. A claim on other hardware needs the comparators re-measured on that hardware.
- **Resolution.** 512² for the ECRO fixtures; 640 × 360 for LR3 (WL102 / reproduction contract). A 1920 × 1080 tier is recommended before any publication claim: TransGI's own frame takes 22.5 ms at 1080p on the 5080.
- **Static frame cost.** From resident state to the image in GPU memory: every learned or physical pass, the denoiser, readout. Presentation excluded.
- **Edit → first frame.** As above, plus every update.
- **One-time costs.** Data generation, training, precomputation. Reported separately and amortised (§6), never hidden.

## 3. Assessment of the proposed margins against measured precision

| Quantity | Measured 95 % interval half-width | 5 % margin resolvable? | 20 % margin resolvable? |
|---|---|---|---|
| ECRO E (fixtures, 8–16 replicates) | ≤ 0.3 % relative | yes | yes |
| PT + OptiX E (fixtures, 16 replicates at ≤ 16 spp) | 1.0–3.0 % relative | yes (tier values carry ≤ 3 %) | yes |
| PT + OptiX GI LDR RMSE (LR3, 8 replicates) | 3.5–4.5 % relative | marginal → use ≥ 16 replicates for LR3 | yes |
| Median frame time | ≤ 0.3 % | yes | yes |
| Reference noise (fixture REF_FULL, per pixel) | 0.18 % relative (S0) | yes | yes |
| Reference noise (near-mirror N0) | 16 % per pixel, heavy-tailed | **no** → N0 unqualified until a converged caustic reference exists | no |

5 % non-inferiority lies above the measured precision on every qualified workload, so it is meaningful and not noise-limited. 20 % is a practically visible effect: about 1 dB PSNR, or one budget step (15 → 12 ms). **Recommendation: keep 5 % / 20 %.** Require ≥ 16 replicates where the comparator interval exceeds 2.5 %.

## 4. Proposed gate table (absolute targets) and current ReST-GI gap

Comparator for every row: path tracing + OptiX AI denoiser (strongest valid; TransGI is on the LR3 frontier only below 7.7 ms). E = relative debiased HDR RMSE against the full-depth reference (fixtures). LR3 uses GI LDR RMSE against Ref-65. ReST-GI = canonical ECRO v0 (WL100 checkpoints, WL101 runtime). Values are from WL111 on the RTX 5080.

| Workload / view | Budget b | Tier T_b (comparator error at b) | PATH 1 target | PATH 2 target | ReST-GI now | Gap | Status |
|---|---|---|---|---|---|---|---|
| W1 SSS cfg0 V0 | 10 ms | 0.0191 | E ≤ 0.0201 at ≤ 8.0 ms | E ≤ 0.0153 at ≤ 10.5 ms | 0.097 at 8 ms; 0.085 at 10.5 ms | 4.8× / 5.5× error | FAIL |
| | 15 ms | 0.0155 | E ≤ 0.0163 at ≤ 12.0 ms | E ≤ 0.0124 at ≤ 15.75 ms | 0.080 / 0.070 | 4.9× / 5.6× | FAIL |
| | 20 ms | 0.0134 | E ≤ 0.0140 at ≤ 16.0 ms | E ≤ 0.0107 at ≤ 21.0 ms | 0.070 / 0.061 | 5.0× / 5.7× | FAIL |
| W1 held-out H1 / H2 | 15 ms | 0.0152 / 0.0155 | E ≤ 0.0159 / 0.0163 at ≤ 12 ms | E ≤ 0.0121 / 0.0124 at ≤ 15.75 ms | 0.080 / 0.070 (both views) | ≈ 5× | FAIL |
| W3 after rigid edit (static at T1) | 15 ms | 0.0148 | E ≤ 0.0155 at ≤ 12 ms | E ≤ 0.0118 at ≤ 15.75 ms | 0.083 / 0.073 | 5.4× / 6.2× | FAIL |
| W5 after light change (static) | 15 ms | 0.0398 | E ≤ 0.0418 at ≤ 12 ms | E ≤ 0.0318 at ≤ 15.75 ms | 0.149 / 0.131 | 3.6× / 4.1× | FAIL |
| W3 edit → first frame (1 spp) | 15 ms | comparator first frame: E 0.027 at 5.24 ms | first frame E ≤ 1.05 · T_15 within 12 ms | — | E 0.334 at 6.71 ms (selective) | 12× error | FAIL |
| W2 near-mirror | — | — | — | — | — | — | **UNQUALIFIED** (reference 16 % per-pixel noise, caustic fireflies) |
| W4 LR3 static V0 (GI LDR RMSE) | 10 ms | 0.0329 | ≤ 0.0346 at ≤ 8.0 ms | ≤ 0.0264 at ≤ 10.5 ms | 0.170 at 27.7 ms (WL102, RTX 3090, 4 spp); converged 0.102 | ≥ 5× | FAIL |
| | 15 ms | 0.0252 | ≤ 0.0264 at ≤ 12 ms | ≤ 0.0201 at ≤ 15.75 ms | as above | ≥ 6.7× | FAIL |
| | 20 ms | 0.0226 (envelope: 16 spp at 17.7 ms) | ≤ 0.0237 at ≤ 16 ms | ≤ 0.0181 at ≤ 21 ms | as above | ≥ 7.5× | FAIL |

**Evidence required for PASS** (each row): ≥ 8 (fixtures) / ≥ 16 (LR3) independent replicate frames at the claimed operating point, scored against the qualified full-depth reference; exclusive-window timing on the benchmark host; the whole 95 % interval inside the target; the frozen frontier re-measured in the same batch if the stack changes.

### Structural requirement exposed by the targets (measured, not a design recommendation)

The depth-6 truncation alone exceeds the targets:
- **W1.** The D6 reference against the full-depth reference has E = 0.0189. ECRO's converged floor is 0.0206 (its bias vs the D6 reference is only 0.0057). The 15 ms PATH 1 target is 0.0163.
- **LR3.** Ref-7 against Ref-65 is 0.0963 GI LDR RMSE. The targets are 0.023–0.035.

So **any configuration that keeps D = 6 fails every tier on W1 and LR3, whatever the accuracy of its learned HIGH or its speed.** Passing requires transport beyond six events, in addition to a 5× error reduction at the budget on W1. How to provide it is an open architecture question. It is not answered here, and no architecture is recommended because of this gap.

## 5. Dynamic-reuse requirements

1. **Qualified edit workload.** The stale-state control must be measurably worse than the updated state, with a pre-registered size and the whole interval beyond noise.
   - WL111's W3 box edit does not qualify: stale vs updated ECRO E differ by < 2 % (16 spp: 0.0853 vs 0.0858; 64 spp: 0.0465 vs 0.0457). WL101 found the same for its SSS tour (0 / 8 material transitions).
   - A qualified edit (contact change, occluder near a receiver, LR3 rigid scenarios M1–M3) is required before any reuse claim.
2. **Edit → first valid frame** within the tier at the budget, against PT + OptiX's edit → first frame. That comparator measured 5.24 ms (BVH update + 1 spp + denoiser) at E 0.027.
   - ECRO now: selective update 6.71 ms (p99 6.79), full relation rebuild 5.19 ms (selective is slower than full on the 5080), stale table 0.80 ms frame. Light change: 3.97 ms vs PT + OptiX 4.26 ms.
3. **Selective = full rebuild** bit-exactly (passes: WL101, WL111 tests). Report stationary-receiver signed change against the reference.
4. **External Neural Full Recompute.** Where a neural baseline's representation is actually invalidated, use its native rebuild lifecycle (8DNA re-bake 1.76 h; RNA 3.24 h; TransGI per-scene data + training ≈ 74 min). Where it is not invalidated, use Native Reuse (TransGI rigid motion, 8DNA asset motion). Path-traced re-render is a separate axis, never a stand-in.

## 6. Lifecycle accounting (WL111 values; one-time vs per-edit vs per-frame)

| Method | One-time | Per edit | Per frame (fixture 512² / LR3 640 × 360) |
|---|---|---|---|
| ReST-GI ECRO (SSS family) | data 108.5 s + training 534 s (RTX 5080, WL100), reused across the family's configurations and lights | 6.71 ms selective (5.19 ms full); light 3.97 ms | 0.81 ms at 1 spp; 3.06 ms at 4 spp; 11.2 ms at 16 spp |
| ReST-GI ECRO (LR3) | data 371 s + training 3,135 s (RTX 3090, WL102), per scene | not run (stage 4 not authorised) | 27.7 ms at 4 spp (RTX 3090, unoptimised) |
| PT + OptiX | none (vendor network) | BVH update: edit → frame 5.24 ms | 3.88 ms (1 spp) / 7.72 ms (LR3, 1 spp incl. guide pass) |
| TransGI (reproduction) | ≈ 40 min data + 34 min training (LR3, RTX 3090) | rigid: no retraining; probes settle in ≈ 8–16 frames (reproduction §4.3) | 5.16 ms at 640 × 360, 22.5 ms at 1080p (RTX 5080, fast frame) |
| 8DNA | per-asset bake ≈ 1.8–2.6 h | native reuse of separate assets; internal edit → re-bake | 0.164 s (teaset 512² × 1 spp, WL27) |

## 7. Milestone exit criteria (proposed)

The next ReST-GI milestone requires all of:
1. **§4 table.** PASS at the **20 ms acceptable** tier on every qualified workload (W1 V0 / H1 / H2, W3, W5, W4 LR3). PASS at 15 ms on at least W1 and LR3. 10 ms is aspirational.
2. **§5.** A qualified edit workload with PASS for edit → first frame.
3. **Mandatory SSS near-contact gate:** PASS.
4. **Novelty condition 4 (§1):** satisfied for at least one learned component (K1 or K2 in the matrix).
5. **Near-mirror qualification.** A converged caustic-capable reference for W2, then the W2 tier. Until then W2 is reported UNQUALIFIED, and no sharp-specular claim is made.
6. **External comparisons on the shared scene.** TransGI on LR3 is QUALIFIED_WITH_LIMITATIONS. NRC is REPRODUCTION_REQUIRED and either gets run or the claim is scoped to exclude it explicitly.

### Current gate statuses (canonical ReST-GI, WL111)

| Gate | Status |
|---|---|
| Numerical thresholds (this document) | PENDING_OWNER_APPROVAL |
| W1 SSS tiers (V0, H1, H2) | FAIL (≈ 5× error at every budget; D6 truncation floor above target) |
| W3 post-edit static tier | FAIL |
| W3 edit → first frame | FAIL |
| W3 as a reuse-qualifying edit | UNQUALIFIED (stale ≈ updated) |
| W5 light-change tier | FAIL |
| W2 near-mirror | UNQUALIFIED (reference) |
| W4 LR3 tiers | FAIL (quality; ECRO timing cross-hardware, historical) |
| SSS near-contact gate | FAIL (see the gate) |
| Novelty condition (learned component necessary) | NOT_RUN in this form; existing evidence (WL87) favours the explicit counterpart |
| NRC comparison | NOT_RUN (REPRODUCTION_REQUIRED) |
