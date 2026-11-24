# Mandatory SSS Near-Contact Gate

**Current as of:** 2026-11-24. **Status (owner decision, 2026-11-24):** the gate is **MANDATORY and non-compensable**. The 3 % band is **CONDITIONALLY APPROVED**, on condition that the error normalisation is explicit (§3.1) and that regional bias, localized artifacts, near-minus-flank contrast, cross-view consistency, dynamic contact and off-target non-regression are all required together (§3.2). The gate cannot PASS while any required class is NOT_RUN. **Role:** an unconditional milestone requirement, separate from the [Contribution Approval Gate](CONTRIBUTION_APPROVAL_GATE.md). Better average quality, speed or results on other materials cannot compensate a failure here. This gate does not solve the problem and authorises no implementation.

## 1. Why a separate gate (evidence summary)

- The canonical ECRO renders a persistent overbrightness on the floor near the SSS sphere's support point.
  - Learned HIGH near the contact: +32.3 % (signed +0.0699 luminance; near-minus-flank +0.0654). Positive in 8 / 8 seeds (WL101C). Independently reproduced on another stack (WL103).
- A physical Oracle table on the same patch × bin interface leaves the same excess (+0.0702), so the cause is representation, not learning (WL101E: 95.5 % of the contrast excess is on the Oracle side).
- The only measured reductions are **camera-conditioned**:
  - V0 projection P: −74 % near / −42 % contrast;
  - 2 × 2 spatial subdivision: inside the WL101F bands (+0.0041 / +0.0082);
  - but silhouette RMSE worsens from 4.35 % to 7.0–7.7 % (WL101F).
- The camera-independent variant G was measured only descriptively at V0 under an unqualified contract (+0.0415 near; S_G +0.0106).
- Cross-view evidence does not exist: WL101G stopped before any evaluation stream (verdict D). The construction-qualification contracts v1 / v2 are NOT QUALIFIED (WL101H, WL101I), with four owner decisions pending.
- Dynamic contact changes (sphere translation, separation, contact formation) have never been measured. The fixture family keeps the sphere fixed.
- Denoising-track candidates excluded the contact region by design (WL105–WL109B). Their silhouette / material-boundary biases are separate findings.

These negative and inconclusive results are preserved as recorded. Nothing here re-labels them.

## 2. Reference and uncertainty contract (precondition)

1. **Full-depth physical reference.** The same-semantics path tracer at full depth, two independent halves, per-pixel variance from the halves (the [benchmark protocol](BENCHMARK_PROTOCOL.md) §2). A depth-matched (D = 6) reference is diagnostic only.
2. **Declared contact semantics.** Native (WL100) or contact-valid (WL101H). Never mixed within a comparison. Adopting contact-valid semantics is a pending owner decision.
3. **Qualified construction.** Any per-region or projection quantity used as evidence must pass a construction-qualification contract approved by the owner (successor of WL101I v2, resolving its four pending decisions) before its primary evidence stream.
4. **Precision.** Each signed quantity is reported with a 95 % interval over ≥ 8 independent replicate seeds (splitmix seeds) plus reference noise. A requirement counts as PASS only when the entire interval lies inside the band (an equivalence test). An interval straddling the band edge is UNRESOLVED, not PASS.

## 3. Requirements (all mandatory)

Definitions reuse WL101C / WL101G ROIs and are coded in `experiments/bench111/run111.py` (masks reproduce the WL101C 1,318 / 6,042 and WL101A 49,660 pixel counts exactly):
- **near** = visible floor within 0.18 m of the support point (V0: front half y ≤ 0.4; held-out views: all azimuths);
- **flank** = 0.26–0.40 m;
- **excess** = (method near − flank) − (reference near − flank) in Rec.709 luminance;
- **rings** = 0 / 0.10 / 0.14 / 0.18 / 0.22 / 0.26 / 0.30 / 0.35 / 0.40 / 0.50 m;
- **contact disk** = floor within 0.25 m;
- **silhouette** = sphere sub-pixel coverage in (0, 1) on an 8 × 8 grid;
- **off-target** = valid pixels outside the contact disk, silhouette and sphere.

### 3.1 Normalisation (explicit, required by the conditional approval)

- **Luminance.** Y = 0.2126 R + 0.7152 G + 0.0722 B (Rec.709) of linear HDR radiance, exposure 1. Every band is a ratio, so it does not depend on exposure.
- **Reference.** Ŷ_ref = mean of the two independent full-depth halves of the **same state, view and contact semantics**. Never a depth-matched or different-semantics reference.
- **Method value.** The expected frame at the operating point (mean over ≥ 8 independent replicates) **and** the converged estimate. Both must pass.
- **Regional relative bias** (R1, rings in R4, silhouette and off-target bias in R5): b_ROI = (mean_ROI Y_method − mean_ROI Ŷ_ref) / mean_ROI Ŷ_ref, each pixel of the ROI weighted equally.
- **Near-minus-flank contrast** (R2): c = [(Ȳ_m,near − Ȳ_m,flank) − (Ȳ_ref,near − Ȳ_ref,flank)] / Ȳ_ref,near. The near-ROI reference mean is the denominator, so a uniform scaling of the whole floor by x gives c = x·(1 − Ȳ_ref,flank / Ȳ_ref,near). This form isolates the contact-shaped component.
- **Localized error** (R1L): for every square window of 5 × 5 pixels (at 512²; scaled by resolution so the footprint is equal) centred in the floor within 0.50 m of the support point, or in the sphere's lower hemisphere band, or in the silhouette: e_w = (mean_w Y_method − mean_w Ŷ_ref) / mean_w Ŷ_ref.
- **RMSE-type quantities** (R3, R5): relative debiased HDR RMSE of RGB over the ROI ([benchmark protocol](BENCHMARK_PROTOCOL.md) §3).
- **Uncertainty.** As §2.4. Equivalence-type requirements (R1, R2, R4, R5 bias, R6b) PASS only if the whole 95 % interval lies inside the band. The localized test (R1L) is a detection test with a Bonferroni correction over windows.

### 3.2 Requirements (all mandatory; τ = 3 % conditionally approved)

| # | Requirement | Measured quantity | Band |
|---|---|---|---|
| R1 | Regional near signed bias | b_near, at the operating point **and** at convergence | \|·\| ≤ τ (3 %) |
| R1L | No localized artifact | e_w for every window (§3.1) | no window whose Bonferroni-corrected 95 % interval of \|e_w\| lies entirely above τ (3 %). Window count and correction are recorded |
| R2 | Near-minus-flank contrast | c (§3.1) | \|·\| ≤ τ (3 %) |
| R3 | Local contact-region HDR error | relative debiased HDR RMSE in the contact disk | ≤ (1 + m_q) × the frontier comparator at equal time (Approval-Gate rule), and converged \|b_disk\| ≤ τ (3 %) |
| R4 | Radial / spatial profile | signed relative luminance error per ring, 0–0.50 m | every ring \|·\| ≤ τ (3 %) at convergence; no systematic sign run (≥ 3 adjacent rings of one sign beyond τ / 2) |
| R5a | Silhouette fidelity | silhouette relative RMSE; silhouette bias b_sil | RMSE ≤ 1.10 × the pre-change canonical value at equal spp (WL101G form); \|b_sil\| ≤ τ (3 %) |
| R5b | **Off-target non-regression** | off-target relative RMSE; b_off; b over each WL101A region outside the contact disk (sphere interior, contact shadow, control wall) | RMSE ≤ 1.10 × the pre-change canonical value at equal spp; every regional \|b\| ≤ τ (3 %); no region worse than before the change beyond its interval |
| R6 | Independent cameras | R1–R5 on every contact-valid view (V0, H1–H4; ≥ 200 near / ≥ 1,000 flank px); H5 silhouette / off-target only | all pass |
| R6b | **Cross-view consistency** | spread (max − min over contact-valid views) of b_near and of c | each spread ≤ τ (3 %), no sign change of a resolved bias between views |
| R7 | **Dynamic contact:** geometry translation and separation | sphere translated parallel to the floor; lifted by gaps {0, 2 mm, 1 cm, 5 cm} | each state passes R1–R2; predicted signed change in near tracks the reference change (slope 1 ± 0.1, corr ≥ 0.98) |
| R8 | **Dynamic contact:** formation and removal | sequence lifted → contact → lifted with edit → first frame | the first valid frame after each change passes R1–R2; no hysteresis (state after return equals the initial state within noise) |
| R9 | Repeated near-contact configurations | ≥ 3 distinct contact configurations (other radius / medium / support position), none used for development | all pass R1–R5 |
| R10 | Camera-independent transport reuse | one transport state per scene state serves every view; changing only the camera triggers no transport recomputation | verified by construction (state hash identical across views) and by R6 |
| R11 | No oracle dependency | no camera-conditioned projection, no scene-specific oracle table, no reference data at runtime, no per-configuration tuning | code audit + R6 / R9 |

## 4. Current status of the canonical ReST-GI (WL111, against the full-depth reference)

This is the first measurement of the contact artifact against a full-depth reference and the first in held-out views. WL101G produced none.
- Values are relative to the reference luminance of the ROI. Each is the mean over 8–16 replicate frames; intervals are 95 % bootstrap.
- Native contact semantics, RTX 5080. Converged = 4,096 spp, one replicate.
- Source: `evaluation/111/sss_gate111.json`, `sss_radial_profile.png`, `contact_crop_S0_V0.png`.

| View / state | R1 near bias (ECRO 16 spp) | R1 converged | R2 excess / near ref (16 spp) | R5 silhouette bias | Off-target bias | PT + OptiX 16 spp: near / excess |
|---|---|---|---|---|---|---|
| S0 V0 | +10.1 % [+9.9, +10.4] | +10.1 % | +23.5 % [+23.1, +23.9] | −2.7 % | −1.7 % | −0.2 % / −0.2 % |
| S0 H1 (held-out) | −6.2 % [−6.6, −5.9] | −6.6 % | +6.5 % [+6.1, +6.9] | −1.7 % | −1.9 % | −0.0 % / +0.4 % |
| S0 H2 (held-out) | +1.3 % [+1.1, +1.6] | +1.2 % | +13.9 % [+13.6, +14.1] | −1.5 % | −2.0 % | +0.4 % / +0.6 % |
| S1 V0 (after box edit) | +8.5 % [+8.2, +8.8] | +8.3 % | +21.7 % [+21.2, +22.1] | −2.8 % | −1.7 % | −0.5 % / −0.4 % |
| SL V0 (after light change) | +19.6 % [+19.3, +20.0] | +20.0 % | +28.4 % [+27.8, +29.1] | −2.7 % | −2.1 % | −0.2 % / −0.6 % |

- **The bias is the representation, not sampling.** ECRO's values do not change between 1 spp and 4,096 spp. The radial profile has the same shape in every view: +10…+15 % within 0.18 m, then −4…−7 % at 0.2–0.45 m. The outer deficit matches the D = 6 truncation; the D6 reference alone is −13 % in the near ROI.
- **The artifact transfers.** Every view and both dynamic states fail R2, and V0, H1, S1 and SL fail R1. H2's near mean happens to fall inside 3 %, but its contrast fails. This is expected from a camera-independent table carrying the error to every view.
- **The practical comparator meets the proposed bands at ≥ 16 spp on every view.**
  - At 4 spp it meets them except SL: near +2.4 % [+0.8, +3.8], UNRESOLVED.
  - At 1 spp it has its own contact bias (+5 … +8 % near) and a silhouette bias (−5.5 % at V0). So denoised path tracing also fails this gate at its cheapest point; the gate binds any method, not only ReST-GI.

| Requirement | Canonical ReST-GI status |
|---|---|
| R1 regional near bias | **FAIL** (V0, H1, S1, SL) |
| R1L localized artifact | not evaluated with the window test in WL111. The radial profile shows a resolved +12…+15 % excess within 0.18 m at V0 (likely FAIL) |
| R2 near-minus-flank contrast | **FAIL** (all five) |
| R3 local contact-region error | **FAIL** (contact-disk E at ≈ 11–12 ms: ECRO 16 spp 0.315 vs PT + OptiX 4 spp 0.073; converged bias +1.4 % V0, −5.4 % H1, −4.2 % H2, +6.1 % SL) |
| R4 radial profile | **FAIL** (systematic +/− pattern in every view) |
| R5a / R5b silhouette / off-target | silhouette bias −1.5 … −2.8 %, off-target bias −1.7 … −2.1 % (truncation); both inside 3 %. The non-regression parts apply to a changed method and have nothing to compare yet. These values are the pre-change baseline |
| R6 independent cameras | **FAIL** (V0, H1, H2 measured; H3, H4 NOT_RUN) |
| R6b cross-view consistency | **FAIL** (b_near spans −6.6 … +10.1 %, a resolved sign change between V0 and H1; c spans +6 … +23 %) |
| R7 translation / separation | **NOT_RUN** (sphere fixed in the fixture family) |
| R8 contact formation / removal | **NOT_RUN** |
| R9 repeated configurations | **NOT_RUN** |
| R10 camera-independent reuse | the canonical table is camera-independent by construction (one table for all views, verified by the WL111 harness); fidelity fails R6 |
| R11 no oracle dependency | canonical: satisfied. WL101F P / S (the only measured partial fixes) violate it |

**Overall: FAIL** (static and cross-view); dynamic contact classes NOT_RUN.

## 5. Numerical bands (τ = 3 % CONDITIONALLY APPROVED, 2026-11-24; basis below)

| Band | Proposed value | Basis |
|---|---|---|
| τ for R1 / R1L | 3 % of the reference near luminance | The physical 3 % rule used since WL55. The practical comparator meets it at ≥ 4 spp (except SL at 4 spp). Reference near noise is 1.2 % per pixel, ≪ 1 % for the ROI mean. Tighter than WL101F's ±0.01403 absolute (≈ 4.6 % of V0's near luminance), which was set relative to the Oracle defect, not to physics |
| τ for R2 / R6b | 3 % of the reference near luminance | Same basis; comparator 16 spp ≤ 0.6 % everywhere |
| τ for R4 | 3 % per ring, at convergence | Rings are wider than the near ROI and have lower noise |
| τ for R5a / R5b | 3 % silhouette signed bias, plus RMSE ≤ 1.10 × the pre-change value | Retains WL101G's non-degradation form; catches the denoiser-type silhouette darkening (−5.5 % at 1 spp) |
| R3 contact-disk bias | 3 % (the WL111 proposal of 2 % is replaced by the conditionally approved τ) | The comparator reaches ≤ 0.6 % at 16 spp |
| R7 tracking | slope 1 ± 0.1, corr ≥ 0.98 over ≥ 4 gaps | Standard WL55 / WL101 change-tracking form |

The WL101F equivalence margins (near ±0.014030, contrast ±0.012520, defined against the D6 Oracle excess) are retained as historical orientation; they are not adopted as the gate.
