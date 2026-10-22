# Worklog — Architecture Batch 5: Spatial-Perspective Transport Factorization, SPT-1 (2026-10-22)

## SESSION QUESTION

User brief "Architecture Batch 5 — Spatial-Perspective Neural Transport Factorization" (2026-10-22):

> Does an explicitly factorized Spatial-Perspective Neural Transport Representation provide a viable alternative to
> monolithic scene-conditioned learned transport, by preserving reusable learned information while updating
> geometry-dependent interreflection?

**Verdict: NOT SUPPORTED** (predeclared protocol `experiments/spt_lego/protocol/spt_v1.json`). The candidate fails
static representability: on held-out T0 it is −11.3 dB vs original RNA, against a −2 dB fail line.

**What the batch nevertheless shows:**
- **The factorization itself works as designed for the transport it can represent.** The persistent parameters stay
  bit-identical. Recomputing only the relation state from the held-out T1 geometry moves the stationary receivers
  toward the reference, with no T1 supervision:
  - F = 1.67 (stale ÷ updated error) and R = 0.82 on the predeclared changed region;
  - F = 2.9 and R = 0.92 on its sand-plate part.
- **The static failure is attributable.**
  - 41.6% of the image is subsurface-scattering yellow plastic, where SPT-1's error is 70× RNA's.
  - The two transmissive materials are 4–4.5× worse.
  - Plain reflective materials are 1.4–3.8× worse.
  - SPT-1's relation contract carries only free-space surface-to-surface transport. It has no path for light that
    travels through an object.

## RELATION TO EARLIER WORK

**Design foundation:** the archived 2026-10-07 lifecycle-separation contract (persistent appearance + current
incident-transport state + shared operator), reinterpreted.

**Differences from worklogs 28–29:**
- dense cosine gathers (K = 128) per surface anchor, instead of 32 uniform probes at query points;
- radiometric sender values from the candidate's own persistent albedo field plus multi-bounce iteration;
- an SH2 state;
- persistent fields trained jointly from scratch, instead of a sidecar over frozen RNA;
- the diffuse-interaction Lego, not the near-mirror teaset.

**Historical baselines, unchanged and reused** because the evaluation regime matches (held-out T1/T0, the same
4096-spp PC references, inputs and regions as worklog 37):
- original RNA (RNA_T0 / RNA_T1);
- Surface-Owned full and selective reconstruction (worklogs 37–38);
- the Surface-Owned frozen state.

The Surface-Owned contract is not redefined.

**Code, outputs, exports:** code in `experiments/spt_lego/` (new). Outputs in `results/spt_lego/`. Exports in
`results/evaluation/39/`.

## A. ARCHITECTURE DECISION

The contract was recorded before any result, in commit `8a1b403` (docs/Architecture.md §9a).

- **Persistent** (bit-identical after an edit; 6.80 M parameters, all persistent):
  - per-object canonical triplane fields φ_A, φ_B in the T0 frame;
  - shared heads:
    - sender albedo a(φ);
    - direct response d(φ, n, ω_o, ω_L), multiplied by the dataset's direct visibility;
    - local transfer t(φ, n, ω_o) → 9 × 3 weights;
  - global E_sun.
- **Spatial-perspective relations** (recomputed from the current geometry, no radiance):
  - 184,883 surface anchors: observed T0 surface cells of 1.2 cm split by signed dominant normal axis, plus an
    analytic grid on the box faces;
  - per anchor, K = 128 cosine gather rays → sender anchor or miss;
  - per anchor, sun visibility over 512 light bins.
- **Dynamic state:** per anchor and light, the cosine-weighted incident SH9 × RGB. Its inputs:
  - sender radiance = a(s)/π · (E_sun·max(0, n_s·ω_L)·vis + E_ind);
  - E_ind from 4 Jacobi iterations over the same relations (multi-bounce);
  - each pixel interpolates ≤ 4 normal-compatible anchors.
- **Operator:** L_o = v·d + relu(Σ t ⊙ g). The indirect term is linear in the current incident state.
- **Update after an edit:**
  - rebuild the relation table (Mitsuba OptiX);
  - recompute the anchor states for the view's light;
  - render.
  - No parameter update and no target-scene supervision.
- **Why this candidate:** it is the most explicit realisation of the persistent/dynamic separation. Learned
  parameters see only canonical position, local normal and directions. The training lights (random sun per pixel)
  excite the incident state, so its role is supervised from one configuration. The held-out edit is therefore a
  strict reuse test.
- **Intentionally not representable:**
  - incident angular detail beyond SH2;
  - non-Lambertian senders;
  - rotations;
  - sub-anchor variation;
  - light visibility finer than 512 bins;
  - transport outside the free-space relation path. That last item is what failed here.

**Execution.**
- Relations and pixel maps were built on the PC.
- Training ran on LabServer63 (RTX 3080 Ti): 150 epochs, 131,072 pixels per step, about 4.9 h per run.
- SPT_T1 hit an out-of-memory error next to SPT_T0 and ran after it (amendment 1, execution only).
- Before declaration, a 60-step smoke run exposed a slow advanced-index backward (11 s/step). It was replaced by
  `index_select`, with identical math.

## B. EVIDENCE

**Synthetic contracts** (`tests/test_spt_contracts.py`, analytic geometry, the production state code): 5/5 PASS.
- The incident state is linear in sender radiance.
- Moving a sender changes stationary receivers' colour bleeding, while anchors with unchanged relation rows keep a
  **bit-identical** state.
- An inserted third-party occluder removes red incident light behind it.
- SH2 diffuse response is exact (relative error ~1e-16); a Phong-200 glossy response to a 4° source is 97% wrong.
- Multi-bounce Jacobi converges.

These establish contract semantics only.

**Static representability** (held-out T0, 16 views, view-mean display PSNR):

| arm | PSNR | vs RNA_T0 |
|---|---|---|
| original RNA_T0 | 33.57 | — |
| Surface-Owned SO_T0 (PU) | 33.24 | −0.32 |
| **SPT-1 S_T0** | **22.23** | **−11.33 → fail** |

**Held-out edit** (T1: the box moved +0.44 m; same views and references as worklog 37):

| arm | T1 supervision | learned params changed | view-mean | changed-A MSE | sand part of changed-A |
|---|---|---|---|---|---|
| RNA_T1 | yes | — | 33.49 | 0.00060 | 0.00056 |
| Surface-Owned A_full | yes | all | 33.38 | 0.00043 | 0.00035 |
| Surface-Owned B_sel (seed 0) | yes | D\* units | 33.07 | 0.00053 | — |
| Surface-Owned frozen | no | none | 30.72 | 0.00494 | 0.00538 |
| SPT-1 D_stale (T0 relations) | no | none | 22.06 | 0.00676 | 0.00638 |
| **SPT-1 C_upd (T1 relations)** | **no** | **none (bit-identical)** | 21.98 | **0.00404** | **0.00217** |
| SPT-1 C_full (retrained on T1) | yes | all | 22.17 | 0.00343 | 0.00179 |
| SPT-1 D_zero (state = 0) | no | none | 12.19 | 0.02236 | — |

- **Rule values:** S = −11.33 dB (fail); F = 1.67; R = 0.82; W = −0.19 dB; parameters bit-identical.
- **Case:** NOT SUPPORTED, by the static rule.
- **Where the updated state helps most:** the box region (MSE 0.00258 → 0.00138, against 0.00078 own-full) and the
  visibility-changed receivers (0.00516 → 0.00286).

**Where the static error is** (post-result diagnostics, descriptive):

- **Lit vs shadowed pixels** (held-out T0):

  | pixels | SPT-1 | RNA |
  |---|---|---|
  | lit | 32.9 dB | 35.2 dB |
  | shadowed (indirect only) | **17.7 dB** | 32.4 dB |

- **Per material** (held-out T0 display MSE, SPT/RNA):

  | material | ratio | pixel share |
  |---|---|---|
  | Yellow (subsurface) | 0.0206 / 0.00029 = **70×** | 41.6% |
  | transparent | 3.9× | |
  | Red_Glass | 4.5× | |
  | Gray | 2.7× | |
  | Sand | 3.2× | |
  | box | 3.8× | |
  | Black | 1.7× | |
  | Light_brown | 1.7× | |
  | Brown | 1.4× | |

- **`01_static_T0_view6`:** the shadowed yellow bucket glows in the reference, because sunlight passes through the
  plastic. SPT-1 renders it as dark blotches, because its direct term is gated off there and the free-space
  incident state cannot carry the light.

**State accounting:**
- The edit changed 30.7% of anchors: 56,700 rows, 2.5% of hit entries.
- Learned parameters changed: 0.
- Output influence is not localised to changed rows. The multi-bounce cache and anchor interpolation spread a
  changed row's effect. The contract test shows exactness only for rows whose own and sender relations are
  unchanged.

**Cost** (prototype; RTX 5080 PC for update and render, RTX 3080 Ti for training):

| item | cost |
|---|---|
| relation rebuild per edit | 118 M rays: gather 0.6–0.9 s warm (9.0 s in one cold run), visibility 1.2 s |
| hit → anchor mapping | 4.7 s, CPU KD-tree |
| state solve per view | 0.076 s |
| decode per view | 0.040 s |
| training (one-off) | 4.9 h per state |
| target-scene supervision after the edit | none |

So the end-to-end edit update is about 7 s, dominated by unoptimised mapping, plus 0.12 s per frame. This compares
with Surface-Owned selective re-learning at 1.9 h with T1 supervision (worklog 37) and physical path tracing at
≈0.2–3 s per frame (worklog 26). None of this is a real-time claim.

**Directional / glossy stress:** measured synthetically only. SH2 fails a Phong-200 lobe by 97%. On the Lego, SPT-1's
high-frequency error is 0.031 vs RNA's 0.017. A near-mirror real scene was not attempted, because the substrate
already fails static representability.

## C. ATTRIBUTION

**What succeeded:**
- **The separation is operational.** With identical persistent parameters, changing only the current-geometry
  relation state produces the physically correct direction of change on stationary receivers (`02_edit_T1_view6`,
  `03_crops_T1_view6`). The plate darkens under the moved box, and the box region and visibility-changed receivers
  improve.
- **The recovery is nearly what the candidate's own retraining achieves** (R = 0.82 overall, 0.92 on the sand), with
  zero target supervision.
- **No hidden re-learning:** zero parameters change, and the update costs seconds instead of hours.

**What failed:**
- **Static representability, dominated by subsurface transport** (yellow, 41.6% of pixels) and secondarily by
  transmission (two glass-like materials). Neither is a free-space surface-to-surface relation, so SPT-1's state
  cannot carry it.
- **A residual 1.4–3.8× gap on reflective materials.** It is unresolved. Candidate causes are SH2 / Lambertian-sender
  coarseness, the 1.2 cm anchor interpolation, binned sender visibility, and optimisation.
- **Absolute quality after the edit.** Even on the sand, the updated SPT-1 state (0.00217) is 6× worse than
  Surface-Owned full re-learning (0.00035). It is still 2.5× better than the Surface-Owned frozen state, without any
  re-learning.

**Supported failure explanations:**
- missing transport modes in the relation contract (subsurface, transmission): **supported** by the lit/shadow split,
  the per-material ratios and the images;
- configuration memorisation: **contradicted**, since the operator responds correctly to an unseen relation state;
- persistent/dynamic separation failure: **contradicted** for free-space transport. D_zero collapses to 12.2 dB, and
  the state change moves outputs in the right direction.

**Unresolved:**
- the reflective-material gap;
- direct-head leakage of canonical indirect light in lit regions (not isolated);
- glossy and near-mirror behaviour on a real scene.

**For the original RNA-compression hypothesis.** RNA's per-configuration function absorbs every transport mode,
including subsurface and transmission, as a function of (x, ω_L). A factorization that routes indirect light only
through surface-to-surface relations gives that up. The evidence here suggests reuse is achievable for the transport
the relation state can express. A viable factorization needs a persistent, configuration-invariant channel for
**object-internal transport** (subsurface and transmission are local to the object, not to the scene arrangement),
alongside the relation-dependent channel. This is a hypothesis, not a measured result.

## D. COMPARISON WITH HISTORICAL BASELINES

- **Original RNA:** best static fidelity. It cannot reuse anything after an edit: it is per-state.
- **Surface-Owned complete-unit reconstruction (worklogs 36–38):**
  - static fidelity near RNA;
  - after the edit, re-learning the oracle-dirty units with T1 supervision recovers fully (R = 0.98; mean gap
    −0.08 dB over 3 seeds);
  - cost: hours of training and target-scene renders.
- **SPT-1 (this batch):**
  - no re-learning and no target supervision; the edit is absorbed in seconds by state recomputation;
  - static representability fails on this scene's material mix, so its post-edit absolute quality is far below both
    baselines.

None of the earlier results is reinterpreted.

## E. ARCHITECTURE VERDICT

**NOT SUPPORTED** under the predeclared rule (static representability fail).

The factorization mechanism itself behaved as hypothesised for free-space transport: F = 1.67 / R = 0.82 overall,
and F = 2.9 / R = 0.92 on the stationary sand receivers. This is reported as a descriptive result, not a pass.

## F. NEXT ARCHITECTURE DECISION

The single most important open question:

> **Can a factorized representation keep a persistent, configuration-invariant channel for object-internal transport
> (subsurface, transmission, and possibly near-field self-interreflection) alongside the relation-dependent
> incident-state channel, and recover RNA-level static fidelity, so that the reuse mechanism observed here can be
> judged at credible quality?**

Not implemented. It needs a new user decision. Alternatives also need a decision:
- testing SPT-1 on a scene of reflective materials only. This would isolate the reflective-material gap, but it
  sidesteps the material mix that RNA targets.

## REPOSITORY

| purpose | commit |
|---|---|
| living docs: SPT-1 contract before results | `8a1b403` |
| SPT-1 code, contract tests, protocol spt_v1 | `2d8f781` |
| server chain | `9cb48a9` |
| amendment 1 (SPT_T1 after SPT_T0) | `7c04d36` |
| diagnostics, verdict, exports, this worklog, living docs | this session's commit |

Server workspace `~/rest_spt` (repo = archive of `9cb48a9`, inputs sha256-verified). Trained states, launcher logs and
the evaluation are in `results/spt_lego/`. Exports and `spt_verdict.json` are in `results/evaluation/39/` (manifest
hashes).
