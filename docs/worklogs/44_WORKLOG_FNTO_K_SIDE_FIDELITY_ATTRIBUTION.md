# Worklog — FNTO K-Side Fidelity Attribution and Controlled Refinement (2026-10-25)

**Status:** Phase 1 (attribution on frozen checkpoints) and Phase 2 (predeclaration) complete; Phases 3–5 (candidate
preparation, training, evaluation) run unattended after the WL43 server chain ends — sections marked **[pending]**.
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

## D. IMPLEMENTATION **[pending]**
## E. MEASUREMENT **[pending]**
## F. INTERPRETATION **[pending]**
## G. UNRESOLVED **[pending]**
## H. ARCHITECTURE VERDICT **[pending]**
## I. REPRODUCIBILITY **[pending]**
