# WL95 Stage A: error-contract audit of the current ECRO pipeline (read-only)

**Scope.** I edited no files and ran nothing on a GPU. The only computation was a small CPU numpy calculation of CK2 geometry from scene coordinates (§3.2). Paths are relative to `c:\Projects\ReST-GI`. I re-verified the reader summaries against code wherever this report relies on them. Three new code-level findings are in §4.

## 1. Current pipeline (verified)

Common facts:
- [FACT] Bins: `bin = iz·16 + iφ`, with `iz = clamp(⌊4z⌋)` and `iφ = clamp(⌊(atan2+π)·16/2π⌋)` (`experiments/ecro86/pt86.py:153-156`; `experiments/ecro86/protocol/ecro86.json:47`, NZ 4 × NPHI 16). Bin solid angle is ΔΩ = 2π/64.
- [FACT] Patch order: floor, ceiling, 3 walls (0–1087), panel 1088–1129, box A 1130–1183, box B 1184–1237, sphere 1238–1269 (`experiments/ecro86/scene86.py:29-39, 47-56, 64-71`).

**(1) Incident angular field s[N, 64, 3]**
- [FACT] `pt86.py:378-402`: s = (1/16) Σ_j Σ_l I·cos_j/d_j²·V_j·1[bin(w_j) = m]. The 16 points per patch are one fixed draw shared by every patch (`pt86.py:114-117, 127`).
- [FACT] Units: patch-mean direct irradiance per incident bin. The direction inside the bin and the position inside the patch are discarded.

**(2) Learned local sender**
- [FACT] u = c·s and Bq = a·u, so M_n = a cᵀ (rank ≤ 16, non-negative, per patch and channel). Code: `experiments/ecro86/model86.py:111-115`.
- [FACT] The learned output Bq is a *bin-averaged* once-scattered radiance table (patch × outgoing bin).

**(3) Sender → receiver transport (state)**
- [FACT] `pt86.py:353-376`. K = 512 rays per patch, κ = 8 per incident bin. Each ray has one origin (`patch_point(us0, us1)`) and one direction (`dir_in_bin(k//8, us2, us3)`).
- [FACT] The offsets come from `default_rng(86001)` and are identical for all patches (`pt86.py:114-116`).
- [FACT] Each ray stores `hit` (−1 if invalid or back-facing) and `dbin` = bin of −w in the hit patch's frame.

**(4) Bin projection / flight**
- [FACT] E1[n, m] = Σ_κ 1[hit ≥ 0]·Bq[hit, dbin]·rw_k, with rw_k = z_k·ΔΩ/κ and z_k = (iz + us2_k)/4. Code: `model86.py:40-46, 116-119`; the same flight is in `experiments/ecro_stage93b/eval93b.py:149-156`.
- [FACT] Each ray reads the sender's patch-constant, bin-constant value. The cosine is the ray's own. Invalid rays are not renormalised.
- [FACT] Receiver: u1 = c·E1, using the same tied c as the sender (`model86.py:120`).

**(5) Learned global term**
- [FACT] y_g = P·Σ_n area_n·Q_n·u_n (rank 24 per channel). It is driven by the direct-light activations u, not by E1.
- [FACT] Output B = a·(u1 + y_g) (`model86.py:121-124`).
- [FACT] Under OWN, the single term is supervised to t2 and the global term to t − t2 (`experiments/ecro_own92a/own92a.py:172-176`).

**(6) Readout and edit path**
- [FACT] Readout: `hl = thr(x1)·B[patch(x2), bin(x2→x1)]`, nearest patch, no interpolation (`pt86.py:228-233`).
- [FACT] Edit: `set_geom` (`pt86.py:130-139`) → selective re-trace of dirty rays, which is bit-equal to a full rebuild (`experiments/ecro86/ecro86.py:57-78`) → full source recompute → full `factors(gs)` recompute (Ft is N², message passing is global; `ecro86.py:82-87`, `model86.py:83-107`).
- [FACT] Nothing in the learned factors is updated selectively.
- [MEAS] 76–78 ms per edit (`docs/worklogs/86_…md:214-215`).

## 2. Where directional information can be lost

| # | Point | Mechanism | Class |
|---|---|---|---|
| L1 | Source s, angular | [FACT] Point-light direction collapsed to a 0.098 sr bin; only the bin-sum irradiance is kept | discretisation / angular aliasing |
| L2 | Source s, spatial | [FACT] 16 fixed points, patch mean; the within-patch irradiance distribution is lost | discretisation (a fixed-pattern bias, not noise) [INF] |
| L3 | Local operator a cᵀ | [FACT] Rank ≤ 16 non-negative, and c is tied between sender and receiver | representational rank/capacity, and/or training/optimisation (not separated) |
| L4 | Sender output table | [FACT] Bq is constant over each outgoing bin and each patch. A sharp lobe narrower than a bin is spread over the whole 0.098 sr bin and the whole patch | discretisation (angular and spatial projection) |
| L5 | Flight sampling | [FACT] 8 rays per incident bin, one fixed (origin, direction) pattern shared by all patches, a single realisation | propagation/projection (deterministic sampling error) |
| L6 | Flight cosine | [FACT] rw uses the ray's own z, but the radiance it reads is bin-constant. If the true incoming radiance sits at a z different from the rays' z, the cos-weighted estimate is aliased. This is worst in ring 0, where z ∈ [0, 0.25] and cosθ varies by an unbounded factor within the bin [INF] | angular aliasing |
| L7 | Receiver | [FACT] The same c acts on bin irradiance E1, so the direction inside an incident bin is lost a second time | rank + discretisation |
| L8 | Global term | [FACT] A scene-wide, area-summed, rank-24 latent driven by u only. It has no flight geometry, so it cannot be directionally faithful for a specific sender→receiver relation | representational / interface semantics. [MEAS] It compensates for single-flight errors (WL86A Amendment 1; WL93D COADAPTATION) |
| L9 | Generator inputs | [FACT] Geometry enters as patch-pair form factors plus a global mean. Material enters as a 69→16-d embedding | capacity / interface |
| L10 | Readout | [FACT] Nearest patch and bin of x2→x1, no interpolation | discretisation |
| L11 | Curved patches | [FACT] Sphere bins use the patch-centre frame, with below-horizon directions clamped to ring 0 (`pt86.py:154, 180`). M64 uses a flat frame (`experiments/gm1_explicit87/quad87.py:15-16`) | interface semantics |
| L12 | M64 "exact" operator | [FACT] fbar is averaged uniformly in solid angle over 8×8 incident cells per bin (`quad87.py:48-56, 240-242`). s carries the true cosine; fbar has none | discretisation. Correct only if f is constant within the incident bin [INF] |

## 3. WL94D exact-M64 flight floor on CK2 (ratio 1.38 / box A 1.21 / panel face corr −0.33)

### 3.1 What the diagnostic computes (verified)

- [FACT] `experiments/ecro_s94d/s94.py:152-160` computes Sx = M64[rcv]·flight(rw, M64[1]·s on panel patches only), using light 0.
- [FACT] It compares Sx with er = ½(e2_p_a + e2_p_b) from `CKF_none.npz`. That reference comes from `field_src(…, split=True)` (`experiments/ecro_mat94/mat94.py:287-298`).
- [FACT] e2_p means: exactly 2 scattering events, with vertex 2 (the bounce reached from the receiver probe) on shape-material 1. NEE at that vertex uses the true light, ER and visibility (`experiments/ecro86a/pt86a.py:100, 175-180`).
- [INF] The reference therefore has the same path family and units (bin- and patch-averaged outgoing radiance at the receiver), with no projection at the sender. The quantity matches. The comparison is analogous to WL90's ORACLE.
- [FACT] **The "exact sender" is not the exact once-scattered q.** It is M64(s): it carries L1, L2 and L12 (incident-bin and 16-point projection) as well as L4 (outgoing bin and patch projection). Only the learned rank and the tie (L3) are removed. The diagnostic therefore cannot attribute its error to the flight or to the bin count alone.
- [FACT] For the box A and panel-face masks, the receivers are Lambert (box material 4). The M64 Lambert rows are ρ/π for every bin, so receiver bin lookup and receiver directional response are neutral there. The comparison reduces to the patch irradiance delivered from the panel.
- [FACT] Sample sizes:
  - The panel-face mask has 9 patches; box A has 54 (`mat94.py:30-31, 365, 408`).
  - corr −0.33 is a Pearson correlation over 9 numbers for one light. corr_halves is 0.867.
  - [MEAS] Values from `results/ecro_s94d/audit.json`: ratio 1.378 / 1.211 / 1.206; corr 0.713 / 0.716 / −0.334; rel L1 0.940 / 0.528 / 0.598; half-noise rel L1 0.127 / 0.058 / 0.069 (all non-panel receivers / box A / panel face).
  - [INF] Reference noise is 5–9× smaller than the rel L1 error, so reference noise does not explain the error.

### 3.2 CK2 geometry (my CPU computation from `experiments/ecro_local93d/addendum_checker/scene_ck2.json` and `ecro86.json:17`; [INF], not a measurement of radiance)

- **Light position.** The light at (−1.45, −0.6, 1.7) is 5 cm in front of the panel plane (x = −1.5) and 0.3 m beyond the panel's y-edge (panel y ∈ [−0.3, 0.9], z ∈ [0.3, 1.7]).
- **Panel incidence.** Incident cosθ on the panel is 0.024–0.164 everywhere, so all light arrives in ring 0, i.e. extreme grazing incidence. This is consistent with [MEAS] "CK2 excites 4 incident bins".
- **Irradiance range.** Panel irradiance varies 307× across the panel.
- **Ideal-mirror footprint on box A's panel-facing face.** It covers 23 % of the face, in one column of 3 of the 9 patches (the +y end). It is fed by only 2 panel patches (the corner pair at y ∈ [−0.3, −0.1], z ∈ [1.3, 1.7]).
  - Panel incident cosθ on those paths: 0.10–0.16.
  - Receiver incident cosθ: 0.22–0.31. This straddles the ring-0/ring-1 boundary at z = 0.25.
- **Where most panel light goes.** Light reflected from most of the panel leaves at z_out ≈ 0.02–0.10. It travels nearly parallel to the panel through the 0.15–0.29 m gap and largely misses the face [INF].

### 3.3 Candidate causes, and what existing evidence can or cannot discriminate

**C1. Outgoing-bin averaging of the sharp panel lobe (L4, angular).**
- [INF] The ring-0 bin spreads the lobe over z_out ∈ [0, 0.25]. That sends flux from all lit panel patches into steeper directions that do hit box A. This mechanism is consistent with over-delivery (> 1) and with misplacement (negative correlation).
- Evidence for the mechanism class: WL90 (`docs/worklogs/90_…md:162-184`) [MEAS] shows that angular first-gather projection explains 97–102 % of the explicit gap in GM1 W1/W2, with spatial ≈ 0.
- What WL90 cannot settle for CK2:
  - Its sender was exact q, not M64(s).
  - It used pooled 64-realisation relations, not the deployed single draw.
  - Its incidence regime was not grazing.
  - Its metric was rendered E_H on R_mirror, not Lambert-face exitance.
- **Open for CK2.**

**C2. Spatial patch averaging at the sender (L4, spatial; also L2).**
- [INF] The true footprint comes from about 10 cm strips of 2 panel patches. The table spreads the radiance over the whole 20 cm patches.
- WL90 found S ≈ 0 (−0.001 to −0.003) in its workload [MEAS], but there the irradiance did not vary 307× across the panel.
- **Open.**

**C3. Incident-side projection inside M64(s) (L1 + L12).**
- [INF] fbar is averaged over incident cells with z ∈ [0.016, 0.234], uniform in solid angle. The true z is 0.10–0.16 on the footprint paths. A sharp ER at grazing incidence (Fresnel, masking) can differ strongly over that range. The sign is unknown.
- [MEAS] encq vs q on panel patches is 0.36 rel L1 (WL92B, E2 T1; `results/ecro_cf92b/results/tables/CF_E2_T1.json`).
- [MEAS] X_M12 vs X_Q accounted fraction: 0.74 / 0.25 vs 0.87 / 0.63 (WL92B:93-110). So M64(s) ≠ q is measured, but in another geometry.
- **Not measured on CK2.**

**C4. 16-point fixed source pattern (L2).**
- [MEAS] 16 vs 1,024 points changed the frozen operator's error by 1.2–4.6 % (WL86A:112-131), but that was under the E1/E2 lights.
- [INF] With a 307× irradiance range, CK2 is more sensitive. **Open.**

**C5. Receiver state-ray sampling: κ = 8, fixed pattern, single realisation (L5).**
- [INF] Which panel patch and dbin the 8 rays hit is fixed and shared across patches. The 2 corner panel patches subtend small solid angles from some receivers.
- [MEAS] Elsewhere: the deployed draw is in the bad tail (−0.049 vs pooled −0.024; WL87:245-262). Sampling-limited hits were 0 / 13 / 249 of 512 (WL86A:181-189). In WL88, R0_INF converges (WL88:252-299).
- **No CK2 pooled-realisation run exists. Open.**

**C6. Ring-0/1 cosine aliasing in rw (L6).**
- [INF] The receiver incident z (0.22–0.31) sits on the bin boundary. Bin-constant radiance times the rays' fixed z_k implicitly assumes radiance spread uniformly over the bin. The sign depends on the rays' z values against the true z.
- **Untested anywhere.**

**C7. Receiver bin lookup / receiver M64 (L7).**
- [INF] Neutral for the box A and panel-face masks, because the receivers are Lambert.
- It can contribute to the all-receiver 1.38, which includes floor B (glossy) and sphere S2 (flat-frame M64, L11). **Open.**

**C8. Reference definition.**
- [FACT] e2_p uses shape-level `mid == 1`, vertex 2 = the bounce sampled from the receiver, and NEE with the same light, visibility and ER. The probe is averaged over valid samples, and flat patches are all valid.
- [INF] I found no labelling or family mismatch for the "none" variant. The reference noise is small (§3.1).
- [FACT] `field_src` takes the material from shape-level SMAT, not PMAT (`pt86a.py:106-107, 177`). This is irrelevant for "none". I did not verify it for the textured variants.

**C9. Mismatched quantity (exitance weighting).**
- [FACT] The exitance weight `AN.W` uses `NB, NPHI, NZ = 64, 8, 8` (`experiments/ecro_mat94/an94.py:19, 32`), whereas the bins are 4 rings × 16 sectors. So W[d] = ((d//8) + ½)/8·ΔΩ instead of ((d//16) + ½)/4·ΔΩ.
  - Ring 0 gets 0.0625 for sectors 0–7 and 0.1875 for sectors 8–15, instead of 0.125. Ring 3 gets 0.8125 / 0.9375 instead of 0.875.
  - ΣW = π is preserved.
- [INF] For isotropic (Lambert) outgoing radiance the effect cancels exactly, so box A and the panel face are unaffected. It biases the all-receiver 1.378 through the glossy receivers.
- **Not quantified.**

**C10. Bin count alone.**
- [INF] Bin count alone is not identifiable from this diagnostic: C1–C6 are confounded in one number from one light, 9 to 54 patches, and one realisation.
- The positive control does not support a generic floor: [MEAS] G_PHYS on 96 training items gives a pooled ratio of 0.953 (`results/ecro_s94d/audit.json`). The 1.38 / 1.21 over-delivery is CK2-specific [INF].

**Bottom line [INF].** "The 64-bin flight floor exists even for the exact sender" (WL94D) is stronger than the evidence. What is measured is a CK2-specific over-delivery and misplacement by a sender projected on both its incident and outgoing sides, read through the deployed fixed 512-ray state. Its decomposition is **open**. No CK2 run exists with any of: q in place of M64(s), per-ray exact sender radiance (X1-style), pooled state realisations, or sub-bin or sub-patch refinement.

## 4. Code and metric discrepancies found (not reported in the worklogs)

1. **`AN.W` ring/sector mismatch** (`experiments/ecro_mat94/an94.py:19, 32`). [FACT] It is used in:
   - the WL94D CK2 flight floor (`experiments/ecro_s94d/s94.py:157`);
   - G_PHYS (`s94.py:129`);
   - all sender metrics `rel_l1_flux`, `energy_ratio` and `delivery_*` (`s94.py:287, 297-311`);
   - WL94 `ex()` response metrics (`an94.py:37, 104`);
   - `experiments/ecro_exp94b/b94.py:47`.

   [INF] It is neutral for Lambert outgoing radiance. It is non-neutral for the S1 panel sender metrics: the CK2 lobe lies in ring 0, where the weights are ×0.5 or ×1.5 depending on the azimuth half. So the WL94D CK2 energy_ratio of 1.71–1.75 carries an unquantified weighting bias. `peak_ratio` has no W and is unaffected. Note that `s94.py:179, 288` uses the correct `zb`.
2. [FACT] `delivery_box_rel_l1` uses patches 1130:1238 (`s94.py:309`), which is boxes A **and** B (108 patches), not box A.
3. [FACT] The panel-face corr of −0.33 is computed over 9 patches. [INF] The ideal-mirror footprint covers 3 of them.

## 5. Summary table

| Loss point | Quantity | Error class | Evidence | Status |
|---|---|---|---|---|
| L1 source angular bin | s per incident bin | discretisation / angular aliasing | WL86A Stage C: ENC explains ≤ 5 % (point light); post hoc beam pairs 0.94 / 0.62 vs noise 0.26 / 0.22 | measured (not causal for E1/E2 point light); open for grazing CK2 |
| L2 source 16 fixed points | patch-mean irradiance | discretisation (fixed bias) | WL86A 1.2–4.6 %; WL93A enc rel L1 0.495 (g00010) | measured elsewhere; open on CK2 |
| L12 M64 incident-cell average | fbar vs f(w_true) | discretisation | WL92B encq vs q 0.36 (E2 T1) | measured elsewhere; open on CK2 |
| L3 rank-16 tied a cᵀ | local operator | rank/capacity vs optimisation | WL92 EY16 0.46–0.52 (flux norm); learned 0.80–0.89; WL94D operator rel L1 ≈ 1.0; WL93C/93D LOCAL_OPERATOR | measured; cause (capacity vs optimisation) open |
| L4 outgoing bin + patch projection | Bq / Bm table | discretisation | WL90 angular J 97–102 %, spatial ≈ 0 (GM1, exact q, pooled) | established for GM1 W1/W2; inferred for CK2 |
| L5 κ = 8 fixed state pattern | E1 | propagation/projection (sampling) | WL86A Amendment 2; WL87 deployed vs pooled; WL88 R0_INF | measured (E2); open on CK2 |
| L6 ring-0/1 cos aliasing in rw | E1 | angular aliasing | none | inferred (CK2 receiver z 0.22–0.31 straddles the boundary) |
| L7 receiver tied c | u1 | rank / discretisation | WL93D C2_rcv; WL94 C2_rcv amplitude 0.97–1.00 | measured |
| L8 global term | y_g | interface semantics / compensation | WL86A Amendment 1 (+0.02…+0.06); WL93D COADAPTATION; WL94A/94D panel-face G bias | measured; mechanism inferred |
| L10 readout | hl | discretisation | WL86A: amplifier, not defect | measured |
| L11 sphere frame | sphere bins / M64 | interface semantics | WL92B:30 (declared) | established (by construction) |
| CK2 flight floor (exact M64(s) sender + Lambert receiver vs e2_p) | patch exitance | confounded C1–C6 | WL94D audit 1.378 / 1.211 / −0.334; G_PHYS training 0.953 | measured; decomposition **open** |
| Exitance weight `AN.W` | metric W | evaluation mismatch | code `experiments/ecro_mat94/an94.py:19, 32` | established (code); impact on non-Lambert metrics open |
| `delivery_box` slice | metric mask | evaluation mismatch | `experiments/ecro_s94d/s94.py:309` | established (boxes A+B) |
| Reference e2_p definition | path family / units | physical-reference | `experiments/ecro86a/pt86a.py:73-200`; half-noise 0.058–0.127 | inferred consistent ("none"); textured variants unverified |
