# Worklog — FNTO K-Side Fidelity Attribution and Controlled Refinement (2026-10-25)

**Decision under protocol `fnto_k44` (rules fixed before any candidate result): IMPROVED.** The predeclared correction
(renderer-consistent incident field: 0.3 mm visibility rule + box world-frame light geometry) raises held-out Lego
K-only fidelity vs first-event references by +1.56 dB (T0; T1 +1.53 dB) against a training-seed spread of 0.10 dB
(threshold max(0.3, 2 × 0.10) = 0.3 dB), and cuts the predicted light in shadowed opaque pixels by 93 %. The box goes
from 16.1 to 31.9 dB (joint effect of both parts). All K-only error now left is dominated by Yellow SSS (78 % of the
squared error), which the correction does not touch. Phases 1–5 complete (2026-10-25 / 26).
Authorisation: user brief "K-side fidelity attribution and controlled refinement" (2026-10-25). Worklogs 41 and 42,
protocol `fnto_r1`, the r1 checkpoints (K_r1, K_r1P) and every historical baseline are unchanged. Worklog 43 is
another session's batch (8DNA dynamic baseline, branch `baseline/8dna_dynamic_reuse`); this batch is 44.

Regimes: **first_r1** = intrinsic-all first-event references (unclamped, point-sampled, 1,024 spp, held-out T0/T1);
**full_r1** = full-GI r1 references (4,096 spp); **probes** = r1 Stage A held-out probes (512 spp); **clean** =
4,096-spp re-renders of 16 held-out probe configurations (this worklog). Display PSNR as worklogs 37–42.

---

## A. AUDIT VERIFICATION — worklog 42 claims checked against code and data

| Worklog 42 claim | Check | Result |
|---|---|---|
| K-only target is the first-event response of K to the direct sun | `r1_solve` K_only arm: H = −1 everywhere → gathered term 0; loc(sun) + glob + band + mid from port sun fractions; reference = intrinsic-all render (free-space hits absorbed) | Correct and matched (same cameras / suns / pixels / film / no clamp) |
| "The composition adds no measurable error on top of the kernel's" (S_T0 \| r1 25.00 ≈ K_only \| first_r1 25.05) | Indirect component evaluated directly: pred = S_T0 − K_only vs ref = full_r1 − first_r1 (`k44_decomp`) | **Wrong.** The two PSNRs are against different targets. Indirect component: pred/ref luminance 1.51, corr 0.66, explained fraction −0.02 overall; Yellow 1.59× over-predicted (corr 0.25); Red_Glass 96 % missing (0.004 vs 0.105); opaque materials fine (ratio 0.86–0.96, corr 0.73–0.89). Oracle swaps on held-out T0: full prediction 24.84 dB vs full_r1; replacing K by the reference → 26.96; replacing the indirect component by the reference → 27.85 — both sides carry error |
| Signed change corr 0.987 / slope 0.948 shows the coupling update | Edit split (T1 vs T0, Lego pixels in both views): direct/intrinsic change vs coupling-update change | The headline is dominated by the direct/intrinsic change (corr 0.985, slope 0.98, explained 0.97). The **coupling-update change** alone: corr 0.67, slope 0.40, explained 0.44 (stale counterpart: corr −0.51). The state update moves the indirect light in the right direction with ~40 % of the reference amplitude |
| Gate 2b quadrature failure limits K | Frozen K_r1P, dense integration on 40 held-out probes (16,000 Lego pixels) | Not a fidelity limit: dense band+mid +0.01 dB, fine-port band +0.00 dB, bf16 numerics 0.00 dB; uniform-field dense vs representative sums agree within ~10 % |
| Box domain low fidelity "unexplained" | Light-geometry frame audit (`k44_box_frame`) | Implementation defect: box point / spot irradiance evaluated from box-local positions (§B D2) |
| Passivity vs fidelity | K_r1 and K_r1P under identical first-event conditions (no coupled solve of K_r1) | K_r1 24.95 dB vs K_r1P 24.88 dB (held-out T0), same per-material structure; probe fidelity 23.98 vs 24.36 dB; the hinge costs no measurable fidelity |

---

## B. ROOT-CAUSE DIAGNOSIS (Phase 1; frozen K_r1 / K_r1P; files in `results/fnto_lego_k44/`)

**Where the K-only error is (held-out T0, K_r1P vs first_r1, share of squared display error):** Yellow shadowed 32 %,
box shadowed 21 %, Gray shadowed 12 %, Sand shadowed 9 %, Yellow lit 9 %, box lit 5 %, others < 3 % each. Shadowed
opaque materials have reference ≈ 0 but receive predicted non-local light (pred/ref: box 422, Gray 15.6, Light_brown
7.0, Brown 4.2, Sand 3.3).

| Candidate cause | Test | Result | Verdict |
|---|---|---|---|
| Source quadrature / aggregation / precision | B: dense, fine-port, bf16 with frozen K | +0.01 / +0.00 / 0.00 dB | not a cause |
| Probe coverage / identifiability | train vs held-out fit (120 + 120 probes); per-probe out-of-span energy vs PSNR | 24.68 vs 24.36 dB; Spearman −0.07 (p 0.42); spot fields most out of span (0.35) yet best (31.2 dB) | not a cause of the probe-level limit |
| Component mis-weighting | per-material least-squares re-weighting of (loc, glob, band+mid), split by views | Yellow 24.90 → 24.86 dB | not a cause |
| Passivity hinge | K_r1 vs K_r1P | 24.95 vs 24.88 dB | not a cause |
| Target Monte-Carlo noise | re-render 8 probes with another seed; 16 probes at 4,096 spp | single-render floor: sun 37–42, spot ~28, broad ~26, multisun ~30 dB; K vs clean 24.96 vs 23.99 dB vs the 512-spp targets; held-out first_r1 floor 33.3–33.8 dB (Yellow 29.3–29.7) | partial: ~1 dB of the probe metric; does not limit the held-out K-only metric |
| **D1 pixel / port light visibility** | Blender BVH vs r1 bits on random pixels (`k44_vischeck`), scalar Mitsuba re-test, rule validation on 5,317 pixel-light pairs (`k44_visrule`) | r1 rule (origin 0.1 mm along the shading normal, any hit occludes) agrees with Mitsuba itself but disagrees with the renderer's geometry on 2.7 % of pairs (broad 1.9–3.3 %, multisun 2.6–6.2 %, sun / spot 0 %), always as false shadow; 34–58 % of blocking hits lie within 1 mm of the origin (q10 20 µm): near-coincident / interpenetrating Lego parts. The 4,096-spp renders confirm these "shadowed" opaque pixels are really lit (Gray / Sand mean 0.18–0.21 under broad). Ignoring hits closer than 0.3 mm: 0.47 % disagreement (false shadow 0.2 %, false light 0.25 %) | **supported defect** (inconsistent supervision → learned non-local leakage into shadow) |
| **D2 box light-geometry frame** | world vs canonical irradiance on box probe pixels (`k44_box_frame`) | sun / multisun identical; broad: E_canonical / E_world q10 / q50 / q90 = 0 / 0.5–0.6 / 1.9–3.9; with world positions the box albedo ratio is 0.715 in every family | **supported defect** (box only) |
| Yellow SSS range | material audit: Burley, radius (1.0, 0.2, 0.1) m on a 0.89 × 1.60 × 0.94 m object (beyond K_band 5 cm / K_mid 30 cm); per-channel error | R 23.2, G 23.8, B 31.6 dB (B trivially high: albedo 0.03); red and green similar | **unresolved** — range alone does not explain it |

Material facts (blend audit): opaque materials are diffuse + 2–5 % GGX(0.3) with bump-only LegoLogo / scratch
textures; the box is diffuse 0.75/0.03/0.02 + 5 % GGX; `transparent` and `Red_Glass` are rough (0.8) glass;
Yellow is Burley SSS + 2 % GGX.

---

## C. CHOSEN CORRECTION (Phase 2; `experiments/fnto_lego_k44/protocol/fnto_k44.json`, committed 53f41e4 before any
candidate step)

**Renderer-consistent incident field** (one cause class: the incident field the kernel is given must be the one the
renderer applied): (a) visibility ignores occluder hits closer than 0.3 mm for pixel bits, port surface samples and
held-out port sun fractions; (b) box-domain point / spot light geometry in world coordinates. Part (b) is inert in the
Lego domain, so the primary Lego endpoint isolates (a). Everything else is the K_r1P recipe. Seed control K_r1P_s1
(r1 data, seed 1) measures training-seed spread; decision rule: IMPROVED iff ΔT0 (Lego K-only vs first_r1) ≥
max(0.3 dB, 2 × seed spread), ΔT1 > 0 and the mean predicted luminance in shadowed opaque pixels drops by ≥ 30 %.
The coupling C (relation tables) is not changed — the same near-hit question exists there and is out of scope.

## D. IMPLEMENTATION (what actually ran)

- `k44_prep.py`: r1_prep with `visible_t` (origin advanced 0.3 mm along the ray, max distance shortened accordingly)
  for pixel bits and port surface samples; per-image box offset stored. PC, Mitsuba cuda. Smoke gate on 3 held-out
  images: 169 flips 0→1, 0 flips 1→0. Full run: held-out 120 images 163 s; training 1,000 images 1,456 s; 1.95 % of
  the 62.5 M training pixels changed bits; 2,459,781 pixel-light pairs 0→1 (0.83 %), 12 pairs 1→0; 3.40 M port-light
  counts changed. `k44_eval_vis.py`: 0.15 % of held-out port sun fractions changed (a single sun).
- `k44_train.py`: `r1_train` unchanged except four monkey-patched entry points (prep2 split + box offset, box world-
  frame point / spot geometry in K_loc and in box port fields). Same recipe as K_r1P (24,000 steps, passivity 1.0,
  seed 0). LabServer63: K_k44P 4.98 h, peak 8.28 GiB; seed control K_r1P_s1 (r1_train, r1 data, seed 1) 4.94 h.
- Evaluation: `r1_eval run` (K_k44P with `R1_EVAL_VIS=eval_vis_k44.npz`), `k44_decomp`, `k44_verdict`; all run by
  `k44_chain.sh` without a session. Process notes: the first chain start hit a CUDA OOM because a foreign root
  container (HMR2 inference, 3 GB) occupied the GPU; the chain now waits for a free GPU before each GPU step; one
  relaunch failed on CRLF line endings introduced by an edit. No run was influenced by any evaluation number.

## E. MEASUREMENT (held-out views, regime r1; K-only vs first_r1, full vs full_r1; PSNR in dB)

| endpoint | K_r1P (baseline) | K_r1P_s1 (seed control) | **K_k44P** | Δ k44P − r1P |
|---|---|---|---|---|
| **Lego K-only T0 (primary)** | 26.05 | 25.95 | **27.61** | **+1.56** |
| Lego K-only T1 | 26.11 | 26.02 | 27.63 | +1.53 |
| box K-only T0 | 16.07 | 16.28 | 31.90 | +15.8 |
| all K-only T0 | 24.88 | 24.86 | 27.70 | +2.82 |
| shadowed opaque K-only T0 | 25.79 | 25.51 | 32.97 | +7.2 |
| shadowed opaque mean predicted luminance | 1.93e-3 | 2.05e-3 | 1.3e-4 | −93 % |
| Yellow shadow / lit K-only T0 | 23.43 / 27.94 | 23.35 / 27.88 | 23.45 / 28.07 | +0.02 / +0.13 |
| full prediction vs full_r1, T0 | 24.84 | 24.94 | 25.64 | +0.80 |
| S_T0 \| hist (view mean; RNA_T0 33.57) | 24.49 | 24.64 | 25.15 | +0.66 |
| E_upd vs D_stale on A_changed (hist): F / R_RNA | 2.44 / 0.65 | — | 3.13 / 0.75 | |
| spectral radius ρ_T0; diverged solves | 0.682; 0 | 0.645; 0 | 0.549; 0 | |
| held-out probe monitor (own inputs) | 24.36 | 24.25 | 24.79 | +0.43 |

Seed spread on the primary endpoint: 0.096 dB → threshold 0.3 dB. Decision rule: ΔT0 1.56 ≥ 0.3, ΔT1 1.53 > 0,
luminance reduction 93 % ≥ 30 % → **IMPROVED**.

Per material (T0, K-only, baseline → candidate): Gray shadow 21.7 → 28.7 (pred/ref 15.6 → 1.8), Sand shadow
26.1 → 33.8 (3.3 → 0.02), Light_brown shadow 26.4 → 35.4, Brown shadow 30.7 → 37.8, Gray lit 29.8 → 37.7, Light_brown
lit 33.7 → 38.9, box lit 21.0 → 38.1, box shadow 13.3 → 28.9 (422 → 7.3); Red_Glass 23.2 → 22.4 (lit) / 21.8 → 21.0
(shadow), transparent shadow 15.5 → 18.4. Remaining K-only squared error: Yellow shadow 61 %, Yellow lit 17 %,
Black lit 5 %, Gray shadow 4.6 %.

Indirect component (S_T0 − K_only vs full_r1 − first_r1, T0): all pred/ref 1.51 → 1.65, corr 0.66 → 0.66,
explained −0.02 → −0.29 (Yellow 1.59 → 1.74× over-predicted; Red_Glass still 96–98 % missing; Gray / Sand / box
fine: corr 0.77 / 0.90 / 0.94). Oracle swaps (full vs full_r1): replacing K by the reference 26.96 → 26.61;
replacing the indirect component by the reference 27.85 → 30.32 — after the correction the composition's indirect
error, not K, limits the full image. Edit split (Lego, T1 − T0): direct / intrinsic change corr 0.988, slope 0.986;
coupling-update change corr 0.69, slope 0.51 (baseline 0.67 / 0.40).

Probe monitor by family (own inputs; not a protocol endpoint): sun 24.3 → 25.9, broad 23.0 → 24.9, multisun 26.5 →
28.0, spot 24.8 → 23.5, Yellow 21.1 → 20.3, box 18.4 → 31.7, reflective 29.8 → 33.6.

Costs: prep (PC) 27 min; eval_vis 10 s; training 4.98 h (vs 5.35 h baseline); evaluation 12 min; the rule adds no
runtime cost (one ray offset).

Reviewer artefacts: `results/evaluation/44/` — K-only, indirect component and full-scene panels for views 6 / 15
and the probe-12 visibility panel; every error / difference map shares one absolute scale M = 0.597 display units
(the largest value among all maps of this worklog, square-root mapping, real-unit colorbars); all diagnostic JSONs
and `manifest.json` (sha16 per file).

## F. INTERPRETATION

- **Dominant supported cause of the r1 K-side limit outside Yellow:** the kernel was supervised with an incident field
  that disagreed with what the renderer applied — false shadows at near-coincident Lego geometry (0.1 mm offset, any
  hit occludes) and, for the box, point / spot light geometry in the wrong frame. A linear kernel fitted to such data
  learns to place light where its input says there is none, which appeared as non-local leakage into shadowed opaque
  pixels (Gray 15.6×) and a box whose response was carried by band / mid terms. Restoring the renderer-consistent field
  removes that leakage (−93 %) and lifts the Lego K-only fidelity by 1.56 dB, 16× the seed spread; the box recovers
  to 31.9 dB. This is attribution by intervention, not by correlation.
- **What it does not explain:** Yellow (Burley SSS, red radius 1 m) is unchanged (23.45 / 28.07 dB) and now carries
  78 % of the K-only error; glass is unchanged or slightly worse. Nothing in Phase 1 isolated a single cause there.
- **Composition:** with a better K the full image improves less (+0.80 dB) because the indirect component over-
  predicts Yellow more (1.74×) — the oracle swaps show the indirect side is now the larger limiter of the full image.
  This is a measurement on one scene, one edit, one seed per arm; it does not establish why (candidate factors: K's
  SSS response to the indirect fields that C delivers, which lie partly outside the training span (Gate 2c), and C's
  own near-hit behaviour, which this batch did not change).
- **Static rendering vs dynamic correctness:** the improvement is a static-fidelity and a signed-change result on the
  same edit; the coupling-update slope 0.51 shows the indirect part of the edit is still under-recovered.

## G. UNRESOLVED

- Yellow SSS fidelity (78 % of the remaining K-only error): capacity, operator family, aggregation or optimisation —
  not separated. Range alone is not supported (red and green errors similar).
- The indirect over-prediction on Yellow (1.59 → 1.74×) and the missing Red_Glass indirect light (96–98 %).
- Whether the coupling C's relation tables suffer from the same near-hit false occlusion (C was not changed here;
  r1 relations report 4.95 % "enclosed" hits).
- Probe-level spot (−1.3 dB) and Yellow (−0.7 dB) monitor changes under the corrected inputs; not part of the
  decision; not attributed.
- One scene, one edit, one seed for the candidate (seed spread measured on the baseline recipe only).

## H. ARCHITECTURE VERDICT (what the evidence establishes)

The batch question — "What is the dominant experimentally supported cause of FNTO r1's intrinsic-kernel fidelity
limitation, and does a controlled within-FNTO correction improve it?" — is answered for the non-SSS part: an
**incident-field inconsistency between the supervision and the renderer** (visibility false shadows, box light frame)
was the dominant supported cause, and correcting it inside FNTO improves held-out Lego K-only fidelity by 1.56 dB
(IMPROVED under `fnto_k44`), with the K/C contract unchanged, passivity kept (ρ 0.55), all solves converged and
parameters bit-identical across the edit. FNTO's intrinsic operator is therefore not shown to be limited at ~25 dB
by its structure on opaque materials and the box; the remaining limit is concentrated in the long-range SSS material
and in the indirect composition. Static fidelity remains far below RNA (S_T0 | hist 25.15 vs 33.57). This batch
does not re-run the `fnto_r1` verdict, does not authorise a later batch and does not select a next direction.

## I. REPRODUCIBILITY

- Code: `experiments/fnto_lego_k44/` (Phase 1 diagnostics f4871b0; protocol + correction 53f41e4; chain fixes
  f876fd2; exporter ada6721; this worklog's closing commit); r1 code unchanged except `r1_eval` reading an optional
  `R1_EVAL_VIS` (default unchanged).
- Protocol: `experiments/fnto_lego_k44/protocol/fnto_k44.json`.
- Checkpoints (sha16 of kernels.pt): K_k44P `30d55da373a01ab8`, K_r1P_s1 `019166b9b60ff16d`, K_r1P unchanged.
- Data: `results/fnto_lego_r1/stagea/prep2_{holdout,train}` (PC = server hashes verified), `eval_vis_k44.npz`;
  anchors a1 `32aa4032280225b7`; references r1 (a2) unchanged; renderer Blender 3.5.1 Cycles, GAUSSIAN 0.01, no clamp.
- Diagnostics and verdict: `results/fnto_lego_k44/*.json`; exports `results/evaluation/44/` (`manifest.json`).
- Server: LabServer63 `~/rest_fnto_r1/` (`launcher/k44_*`), RNA venv (torch 2.8); PC: 8dna26 `.venv-cu128` (Mitsuba).
