# Claim-to-Evidence Matrix — Architecture Novelty and Contribution

**Current as of:** 2026-11-24 (WL111). **Role:** for every contribution ReST-GI might claim, the competing architecture, the precise distinction, the evidence that would establish it, the evidence that exists, the gap and the failure condition. Novelty is kept separate from numerical superiority: a better PSNR, a lower frame time or an unavailable competitor's code is not by itself evidence of architectural novelty. The numerical decision rules are in the [Contribution Approval Gate](CONTRIBUTION_APPROVAL_GATE.md); baseline availability is in the [registry](EXTERNAL_BASELINE_REGISTRY.md).

**Owner decision (2026-11-24):** architecture novelty and causal ablation are mandatory conditions, kept separate from numerical superiority. A numerical PASS under the [approval gate](CONTRIBUTION_APPROVAL_GATE.md) does not satisfy them, and they do not substitute for it.

**Status of every claim below: NOT ESTABLISHED.** No canonical document states a positive contribution claim ([research topic](RESEARCH_CENTRIC_TOPIC.md): "intended contribution, not yet a demonstrated claim").

## Claims

### K1 — Learned, source-independent higher-order transport operator reused across lights and edits without retraining

| Field | Content |
|---|---|
| Mechanism (canonical ECRO v0) | A geometry-conditioned generator (model86) maps the current relation state and a per-patch incident-source encoding to a patch × 64-bin table of ≥ 3-event radiance. It is linear in the source; the weights are never retrained after an edit or light change. |
| Closest competing architectures | **NRC** (Müller 2021): path-traced prefix + a radiance cache trained online every frame. **TransGI**: per-object neural transfer + light probes, rigid reuse without retraining. **8DNA**: per-asset baked neural transport, natively reusable under rigid motion of the asset. **PRT / precomputed transfer**: fixed-geometry transfer. **Explicit transport on the same state** (EXP / C3UR): the learning-free alternative. |
| Precise distinction claimed | One network serves arbitrary geometry configurations of a family and arbitrary point / area sources without per-configuration training (unlike 8DNA / TransGI per-scene training), and without online per-frame training (unlike NRC). |
| Predicted utility | Lower memory / precompute than explicit per-configuration tables; unseen-configuration generalisation; cheaper light updates. |
| Required causal ablation | Learned operator vs learning-free explicit recurrence on the identical state, ER and readout (EXP / C3UR), at equal quality, on held-out geometry and sources. |
| Required external comparison | NRC (closest; REPRODUCTION_REQUIRED), TransGI on a shared scene (LR3), 8DNA native reuse where its assets apply. |
| Required dynamic evidence | Held-out edits and sources with an independent reference; stale-table control. |
| Existing support | WL86: learned table meets broad gates but fails sharp held-out regions (NOT SUPPORTED). WL87: explicit recurrence is ≥ learned in every region class (0 better / 3 worse of 36); the learned operator's only measured utility is a 3.5–3.9× cheaper source-only update (≈ 40–50 ms). WL96A: no accuracy / lifecycle headroom for a learned ≥ 3-event resolvent behind a physical prefix on CK2. WL102: unseen area light on LR3 reproduced at the depth-matched level (learned 0.015 vs physical 0.014 GI LDR RMSE vs Ref-7). WL111: light change → first frame 3.97 ms (ECRO) vs 4.26 ms (PT + OptiX), but at 1 spp the first frame's error is 0.61 vs 0.08; a stale (old-light) table changes the full-frame error by only 1–4 %, so the learned source update's measurable value on this workload is small. |
| Remaining gap | Causal evidence that learning adds value over explicit transport on a workload where explicit transport is costly; a quality-matched comparison with NRC. |
| Fails if | An explicit table / recurrence on the same state matches quality at ≤ the learned cost (already true on the GM1 room: WL87), or NRC with a short path prefix matches quality at equal time. |

### K2 — Physical LOW + learned HIGH ownership split (no double counting, physically computed first two events)

| Field | Content |
|---|---|
| Mechanism | E1 and E2 are path-traced with the current ER closure; ≥ 3 events come from the table looked up at the second hit; the table excludes x2's once-scattered radiance. |
| Competing | **NRC is structurally the closest prior art** (path prefix, then terminate into a learned cache). TransGI (DI separate, GI from transfer × probes); 8DNA (exterior paths traced, interior baked). |
| Distinction | The terminal query is a geometry-generated, source-linear table rather than an online-trained cache or a per-object transfer. |
| Required evidence | Equal-time comparison with NRC; ablation of the LOW depth (1 vs 2 physical events) at equal time. |
| Existing support | WL87: the 9× per-frame gain over path tracing comes from LOW + lookup **structure**, not from learning. **WL111 (RTX 5080, full-depth reference):** at equal time ECRO has ≈ 3× lower error than raw full-depth PT (W1: 3.06 ms → 0.161 vs PT 2.38 ms → 0.510), but PT + OptiX at 1 spp (3.88 ms, 0.031) is 5× lower than ECRO at 4 spp and better than ECRO at any spp. ECRO's converged floor is 0.021, of which the D = 6 truncation alone is 0.019. **The split as implemented loses to a practical denoised path tracer on every qualified workload.** |
| Gap | NRC comparison; any evidence that the split is better than a path-traced prefix + denoiser at equal time. |
| Fails if | PT + a practical denoiser reaches ECRO's error at ≤ ECRO's time (measured in WL111), or NRC does. |

### K3 — Dynamic edit response through selective relation updates (stationary receivers included)

| Field | Content |
|---|---|
| Mechanism | On a rigid edit the dirty relation rays are re-traced, then the source and a full generator pass are recomputed. Selective state equals full rebuild bit-exactly (WL101, WL111 test). |
| Competing | TransGI native rigid reuse (no retraining; probes converge in 8–16 frames); 8DNA native reuse (separate assets) or re-bake; NRC online adaptation; path-traced re-render; Neural Full Recompute when a neural representation is invalidated. |
| Distinction | Edit-to-valid-GI in one frame with physically updated stationary-receiver transport, without retraining. |
| Required ablation | Selective vs full rebuild vs stale (WL101: stale worse in 14 / 16 states). |
| Required external comparison | Edit → first valid frame and quality against PT / PT + denoiser (WL111) and TransGI's response frames on a shared edit (not yet: LR3 edits not run for ECRO). |
| Existing support | WL101: 11.0–15.7 ms edit → 512² × 1 spp on an RTX 3090; the generator pass scales ~linearly in N (52–69 ms at N 11,430). **WL111 (RTX 5080):** box edit → first frame 6.71 ms selective / 5.19 ms full rebuild vs PT 3.87 ms / PT + OptiX 5.24 ms. Stale table ≈ updated table on this edit (< 2 % E difference), so the edit does not qualify as a reuse test. |
| Gap | A shared-scene edit against TransGI; scaling beyond N ≈ 5,000; non-rigid edits (GM3). |
| Fails if | Path tracing (+ denoiser) produces an equally valid post-edit frame faster, or the update cost grows past the frame budget at practical scene sizes. |

### K4 — Complex-material coverage under one transport mechanism (SSS, near-mirror, layered ER, fabric)

| Field | Content |
|---|---|
| Competing | 8DNA (baked media / surfaces per asset), TransGI (no media, no glass), NRC (any material, online). |
| Existing support | WL100: SSS ROI error 1.47–2.08 % vs the finite-depth reference; near-mirror noise-limited. WL104 / WL104A: learned-ER scene NOT SUPPORTED (HIGH error significant on one object). WL105: fabric learned HIGH insufficient (44.9 % / 59.7 % vs explicit 12–14 %). CK2: system-level robustness under a material shift (descriptive). WL111: the near-mirror fixture's full-depth reference is caustic-noise-limited (16 % per pixel), so near-mirror fidelity stays unqualified; the SSS fixture fails the near-contact gate. |
| Gap | Fidelity on fabric / layered scenes; SSS near contact (mandatory gate); transmission (glass excluded everywhere). |
| Fails if | Any mandatory material class misses the quality gate at the claimed operating point. |

### K5 — Real-time budget (10 / 15 / 20 ms) at practical quality

| Field | Content |
|---|---|
| Existing support | WL101 / WL110: 11–17 ms compute on an RTX 3090 at 512² × 1 spp. WL111 (RTX 5080): 0.81 ms at 1 spp … 11.2 ms at 16 spp on W1, so the budget is met. But at 10 / 15 / 20 ms its error is ≈ 5× the PT + OptiX tier (§4 of the gate). On LR3 at 640 × 360: PT + OptiX 7.7 ms → 0.039, TransGI 5.2 ms → 0.061 (fast-frame settings), ECRO 27.7 ms (RTX 3090) → 0.170. |
| Gap | Quality at the budget against the frontier; resolution scaling (1080p); scene scaling. |
| Fails if | At the budget, a practical comparator is better by the gate margins, or the budget is met only at an operating point whose error fails the quality tier. |

### K6 — SSS near-contact correctness (mandatory, cannot be compensated)

Owned by the [SSS near-contact gate](SSS_NEAR_CONTACT_GATE.md). Existing support: none. The canonical HIGH carries a persistent contact overbrightness (+32 % HIGH, WL101C; reproduced in WL103). WL111 measured it against a full-depth reference and in held-out views: near +10 % / contrast +23 % at V0; contrast +6 % / +14 % at H1 / H2; +20 % / +28 % after a light change. Unchanged from 1 to 4,096 spp. The only fixes measured so far are camera-conditioned (WL101F) and are not deployable.

## What novelty cannot rest on

Hierarchies, selective updating, per-region neural models, BSSRDF approximation or operator compression alone ([research topic](RESEARCH_CENTRIC_TOPIC.md)). It also cannot rest on a frame-time advantage that comes from the LOW + lookup structure (WL87) rather than from the learned component, or on the absence of a runnable competitor (NRC, official TransGI).
