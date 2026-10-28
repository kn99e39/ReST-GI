# Worklog — Explicit C Spatial Fidelity and Resolution Contract (2026-10-28)

**Protocol.** `fnto_c54` (`experiments/fnto_c54/protocol/c54.json`), committed with its code (7d7a348) before the Stage B primary
evidence and extended by four labelled amendments (A1 coverage rule and a reference pixel-size diagnostic; A2 a spectral-radius
estimator audit finding; A3 pixel-converged references and a gap8 oblique test; A4 an infrastructure record). Each interpretation rule
was fixed before the run it governs. No rule, threshold or verdict mapping was changed after a result.

**Authorisation.** The user brief "ReST-GI — Worklog 54: Explicit C Spatial Fidelity and Resolution Contract — Bounded Architecture
Evidence & Conditional Correction" (2026-10-28): audit WL53's coupled-composition misses, isolate their spatial part, implement at most
one bounded correction of the explicit free-space coupling C, validate it physically, evaluate the coupled composition and cost. Not a
Learned-C experiment, not an FNTO or K redesign.

**Isolation.** Branch `wl54_explicit_c` (worktree `.claude/worktrees/wl54`), starting commit `f0d3f2287574a66e663aafa5157938321544e53f`
(`Selective_Recompute`). All code is in `experiments/fnto_c54`. `experiments/phys_ref53` is imported unchanged (its 9 tests pass
unchanged after this batch). No production FNTO module, K44 / DASQ checkpoint or WL52 file was read or written. WL53 results were read only.

**Hardware / software.** LabServer69, RTX 3090 24 GB (driver 550.54), 16 cores; torch 2.3.1+cu121, Mitsuba 3.5.1 `cuda_ad_rgb` as a
ray intersector only, Cycles 3.5.1 in `restgi/bpy-runtime:22.04`. The GPU and CPU were idle for the timing run
(`results/fnto_c54/final/idle_check_before_cost.txt`).

---

## Answer to the completion question

> Is the inaccurate near-contact K/C composition primarily a spatial boundary-transport representation problem, and can it be corrected
> with a physically consistent explicit C interface at an acceptable accuracy-versus-update-cost trade-off?

**For domains whose K is local (opaque Lambert), yes on the first half, and largely on the second.** With exact local K, the whole
near-contact error of the 1 cm interface is spatial: the exact-transfer coarse operator (A) reproduces the production C0 error
(floor-control box underside +6.2 % against A +6.2 %; gap4 upper face +42.2 % against A +44.3 %), while C sampling noise, ray count and
directional bins contribute ≤ 0.4 % (±3.3 % per production realization on the gap faces). The error is a **convex, sub-port structure of the
source radiance** (a field decaying by ~7× per port near a gap opening), not a gradient: a linear ramp inside a port is transferred
exactly, an exponential decay is not. One bounded correction, **contact-adaptive sub-port coupling (CASP)**, removes most of it.
Against the converged references, 35 of 36 ruled Lambert groups pass the historical 3 % rule and the worst group error falls from 42.2 % to
4.7 %. The predeclared restoration rule needs every group, so this is **PARTIAL evidence, not a restoration**.

**For domains whose K is port-resolved (the WL53 medium operator), no: C alone cannot fix them.** A medium-to-medium link is unchanged by
CASP (within 0.52 points of C0 on every group, ≤ 0.1 on the large ones), because the K interface carries flux summed over a port and emits uniformly. The only lever is the
resolution of K and C together. A labelled control with the oracle K at 5 mm ports and C at 5 mm halves the red-channel gap-face errors
(gap4 gapdown −4.2 → −1.3 %, gapup −2.8 → −0.6 %) but leaves the blue channel at +3 to +5 % and worsens the oblique side groups.

**Cost.** At an equal ray budget the GPU work (relation build, sun term, 20 applications) of CASP is 1.00–1.04× that of C0. Its predeclared
unknown budget (≤ 4× the coarse unknowns) holds on the floor and L-block scenes (1.2–2.8×) and fails on the 4 mm gap (7.35×). The
wall-clock total is 1.6–2.0× C0 because of unoptimised CPU mesh refinement. This is a high-accuracy diagnostic C and a dynamic-scene
candidate for local-K domains; it is **not** production readiness.

The WL53 headline "the coupled error is a 5 % floor from the 1 cm C" is therefore decomposed into three different things (below): a real
spatial error (box undersides, gap faces), a comparison artifact (floor under the box, oblique 10-port side groups) and a K-side resolution
dependency (medium gap faces).

---

## Verification of the prior claims (Stage A)

| Claim (brief) | Result | Evidence |
|---|---|---|
| **A.** The WL53 composition misses are reproducible | **Confirmed.** The stored G3 tables replay from the stored predictions and Cycles port averages with max |Δbias| = 0.0 over 29 groups. A fresh run of the production composition (same code, same seeds) for gap8 (all arms) and floor_lam is bit-identical (floor_lam M-variants 6e-8 relative, float atomic order) | `audit/replay.json`, `audit/rerun_diff.json` |
| **B.** A ≈ 5–6 % Lambert-control error exists with an exact local K | **Confirmed for the box underside** (new Cycles seeds 5401–5404: +6.2 %; rot_lam +4.6 %; lblock +3.4 %). **Not for the floor under the box** | `final/amend1_tables.json` |
| **C.** More directional bins and rays do not resolve it | **Confirmed, and explained**: the exact-transfer coarse operator has the same error (A +6.2 %), so no sampler or bin count can remove it. **Caveat:** the WL53 ray-count control (128 / 512 / 2048) reused one fixed sun-visibility realization (K_ext 256, seed 5341), so it could not bound that estimator's noise; replicates measure it (below) | Stage B |
| **D.** The gap8 oblique lower-side group (−24.3 / −22.5 / −13.9 %) is real, evaluated, not a summary artifact | **Real and evaluated in the stored data** (replay exact; ruled; 10 ports; 6 % of the signal). **Not in WL53's prose tables.** **Not a reliable measurement:** the comparison unit moves it by 8 points (centre-ray unit −16.2 / −14.3 / −6.4 %); a C refinement does not change it (CASP −16.4 %); a matched 5 mm K and C worsens it (−23.5 %). Its cause is **unresolved** here (the finer-pixel volume test could not run, A4) | `audit/unit_table.txt`, `kfine/kfine_control.json` |
| **E.** Physical K/C ownership is preserved by the implementation | **Supported by audit and tests.** `cast_C` starts rays only on the outward hemisphere of boundary cells and records only front-face hits (zero back-face hits in every medium scene); a medium sub-path never enters C; K is the only in-domain operator; CASP keeps it. The solver's dominant eigenvalue is a ± pair for bipartite couplings, and WL53's power-iteration estimator misreports it for refined operators (A2); WL53's gap ρ values are valid (match Arnoldi to 3e-8) | `final/tests_final.json`, `final/rho_check.json` |

**Further audit findings (all in `protocol` amendments).**
1. **Coverage rule.** WL53's rule "at least 30 pixels per port" meant ~30 % of a port at 1 mm pixels; at 0.5 mm it is 7.5 %. Ports cut by the view border or seen only partly through a gap entered the group averages. Under the amended rule (≥ 90 % of the port's expected pixel count) the floor-under-box group is C0 +1.2 % / F16 −0.8 % (n = 144); the remaining 52 partial ports read +8.4 % for **every** arm including the fine reference, i.e. they are a comparison artifact. WL53's +3.9–5.2 % for "floor under the slab" is largely this artifact.
2. **Reference point-sampling.** A 2.3 mm sunlit strip sampled by 0.5 mm point-sampled pixels aliases: the gap4 gapdown Cycles port signal is 0.8413 (0.5 mm), 0.7861 (0.25 mm), 0.7859 (0.125 mm); the fine solution moves from −5.3 % to +1.4 % against it. Against the converged reference C0 is **+20.0 %** off on that group (not +12.1 %).
3. **Oblique 10-port groups** are seen through the gap (a 50° line of sight enters a 4 mm gap only 4.8 mm deep), so the reference covers part of a port while every prediction is a full-port average. They are reported as partial-coverage groups, not hidden.
4. **Spectral-radius estimator.** WL53's `spectral_radius` is a last-ratio power iteration; with a ± eigenvalue pair (any bipartite coupling) it is not |λ₁|. It is within 1 % on the coarse floor scene and 2–4× low on refined operators (floor s=2: 0.382 vs Arnoldi 0.7318). Validated replacement: power iteration on A², matching Arnoldi to ≤ 5.5e-8 in four cases. WL53's gap4 / gap8 ρ (0.816 / 0.767) are correct; its floor-scene blue channel is 3.6 % low (0.521 vs 0.541). Its convergence statements do not depend on ρ.

---

## IMPLEMENTATION FACT

**Audited and kept frozen.** `p53_compose.cast_C` (unbiased ray estimator of the flux fraction between cells; uniform position, projected-solid-angle bins), `sun_external`, `SceneSet`, `solve`, the TR oracle K, WL53 geometry / fixtures / Cycles volume references. Production C0 is run through its own caster.

**New code (`experiments/fnto_c54`).** Lambert fixtures at any sub-port refinement (`c54_fixtures.py`); fine solves, Galerkin aggregation and projection (`c54_lam.py`); CASP (`c54_casp.py`); Stage B / C / E / F drivers; tests (`tests/test_c54.py`); Cycles drivers; known-field, change-tracking, coverage and K-control diagnostics; the report.

**The candidate (CASP), one mechanism.** Conservative adaptive refinement of the explicit-C unknowns (hierarchical-radiosity-style link refinement), geometry only:
- per coarse port: 256 cosine-weighted rays; ν = fraction hitting within 4h (h = 1 cm); if ν < 0.05 the port stays one unknown; else d = median near hit distance and the port is cut into ℓ × ℓ cells, ℓ = clamp(2^⌈log₂(c h / d)⌉, 1, 8), **c = 2** (cell ≤ d / 2, from the Galerkin error scaling (w/d)²);
- cells are built **exactly on the triangulated surface** by triangle-wise affine subdivision of the port's grid (deformation-aware; a first bilinear version changed port areas by 1.3 % on the dented slab and was rejected by its own test, see Tests);
- the caster is `cast_C` generalised to per-port ray counts (weight 1 / count): same unbiased estimator, exact closure; equal ray budget to production (8192 rays per coarse port; cells of an ℓ × ℓ port get 8192 / ℓ²);
- **conservative mapping to K:** cell flux → coarse port flux by sum (receivers), coarse flux → cells by area weights (sources). A port-resolved K (WL53 medium) keeps coarse unknowns, so CASP refines Lambert domains only. K's coefficients and meaning are unchanged.
- Sensitivity arms c = 1, c = 4 and 4× rays are reported next to the primary; none is selected.

**Tests (10, all pass; WL53 regression 9/9 unchanged).** Analytic coaxial-square form factors (gap 4 / 8 / 30 mm; fine s=8 / 16, coarse M→∞ and CASP all within 0.24 % of Howell's closed form, max at the 30 mm CASP arm with one seed); exact flux closure 1.0 (min = max) with zero back hits in all arms; non-negativity; same-seed determinism (bit-identical); area-weighted reciprocity within the cross-seed noise rule (uniform and CASP mixed levels); zero field exactly 0, linearity 5e-13, sparse apply vs dense 5e-16, coupled solve vs dense inverse 3e-12; blocker plate removes the face-to-face flux (< 1e-9) in coarse, uniform and CASP meshes; rigid-motion consistency within the cross-seed noise (CASP levels 100 % rotation-invariant); concave re-entry 3.09 % (coarse) vs 3.09 % (s=8), convex 0; level rule contained in [1, 8] and top faces never refined; deterministic comparison masks. **One failed run is preserved** (`tests/tests_v1.json`): the first refinement interpolated bilinearly and changed port areas by 0.39 % (bend 90°) and 1.28 % (dent), which broke the exact-surface requirement; the implementation was corrected (area error 3e-15) and the tolerance was not relaxed. One threshold was loosened afterwards: the minimum cell-normal dot on curved ports (0.99 → 0.97), because it measures port curvature, not mesh error (mesh error is checked by the exact area).

---

## MEASUREMENT

### Stage B — where the 1 cm interface loses information (Lambert, exact K)

Fixtures: gap4 / gap8 / gap30 (two closed 10 × 10 × 1 cm boxes, sun 30°), floor_lam (WL53 amendment-4 scene), lblock_lam (concave), rot_lam (box rotated 17° and shifted 3.7 / −2.1 mm: shadow and box edges off the floor-port grid). Fine solves s = 2, 4, 8, 16 (+ 32 for gap4); reference level s = 16. **Convergence:** the pooled gap-face radiance changes by 7.1, 1.3, 0.45, 0.15 % for s = 2→4→8→16→32; the last step equals the replicate noise (0.1 %). Cycles reference: seeds 5401–5404, 2048 spp, 0.5 mm pixels (1 mm for the overview cameras), centre-ray comparison unit.

| fixture / camera / group (reference pixel) | n | C0 vs ref | exact-Galerkin coarse (A) vs ref | fine F16 vs ref | C0 replicate sd | CASP C1 vs ref | C1 replicate sd | C1 c=1 / c=4 / 4× rays vs ref |
|---|---|---|---|---|---|---|---|---|
| floor_lam / underup / dom:bot (0.5 mm) | 100 | +6.2 % | +6.2 % | −0.0 % | 0.4 % | +1.6 % | 0.4 % | +6.2 / −0.3 / +2.0 % |
| rot_lam / underup / dom:bot (0.5 mm) | 100 | +4.6 % | +5.1 % | −0.0 % | 0.3 % | +1.3 % | 0.3 % | +5.7 / +0.2 / +1.6 % |
| lblock_lam / underup / dom:bot (0.5 mm) | 100 | +3.4 % | +3.5 % | −0.1 % | 0.3 % | +0.6 % | 0.2 % | +3.7 / −0.3 / +1.0 % |
| floor_lam / underdown / floor (0.5 mm, full coverage) | 144 | +1.2 % | +1.3 % | −0.8 % | 0.2 % | −0.2 % | 0.1 % | +1.6 / −0.9 / −0.1 % |
| gap4 / gapup / slab_hi:bot (0.25 mm) | 100 | +42.2 % | +44.3 % | −0.1 % | 3.3 % | +1.1 % | 0.6 % | +20.7 / −0.3 / +2.1 % |
| gap4 / gapdown / slab_lo:top (0.25 mm) | 100 | +20.0 % | +21.6 % | +1.4 % | 3.3 % | +2.2 % | 0.4 % | +10.1 / +1.5 / +2.6 % |
| gap8 / gapup / slab_hi:bot (0.5 mm) | 100 | +17.6 % | +20.5 % | +0.0 % | 1.7 % | +2.4 % | 0.3 % | +18.8 / +0.8 / +1.6 % |
| gap8 / gapdown / slab_lo:top (0.5 mm) | 100 | +7.5 % | +9.9 % | +1.7 % | 1.7 % | +2.9 % | 0.3 % | +9.4 / +2.0 / +2.6 % |
| gap8 / obot / slab_hi:bot (0.5 mm) | 10 | +16.1 % | +21.4 % | +1.3 % | 3.1 % | **+4.7 %** | 2.0 % | +10.4 / +2.5 / +2.5 % |
| gap30 / gapup / slab_hi:bot (0.5 mm) | 100 | +0.3 % | +1.0 % | −0.1 % | 0.6 % | +2.2 % | 0.7 % | +2.2 / +0.6 / +1.7 % |
| gap30 / gapdown / slab_lo:top (0.5 mm) | 100 | −1.1 % | −0.6 % | −0.9 % | 0.4 % | +0.1 % | 0.5 % | +0.1 / −0.7 / −0.0 % |

*Spatial attribution rule (predeclared): attribution fraction a = A / (C0 vs F16) in [0.5, 1.5] with replicate-mean bias ≤ 1 %.* It holds for the floor-control box underside (a = 1.00), rot_lam and lblock box undersides (1.1 / 1.0) and the gap4 / gap8 gap faces (a = 1.09–1.42, computed as the ratio of the exact-Galerkin to the C0 deviation from F16); the 30 mm control is within ±2.2 % (A ≈ 1 %). The production C0 realization is noisy on gap faces (sd 1.7–3.3 % per realization; the WL53 realization is one draw).

**Known-field transfer (the protocol's source / receiver separation).** With the converged fine field as the prescribed source, averaging the source per 1 cm port changes the received flux on the receiver group by:

| fixture | solution field | pairs < 1 cm | 1–2 cm | 2–4 cm | exp. edge decay (5 mm) | linear ramp | step in a port | uniform |
|---|---|---|---|---|---|---|---|---|
| gap4 (lower top → upper bottom) | +17.6 % | +16.5 | +1.0 | +0.1 | +7.5 % | −0.0 % | −0.0 % | +0.0 % |
| gap8 | +9.6 % | +4.2 | +5.1 | +0.3 | +7.8 % | −0.0 % | −0.1 % | +0.0 % |
| gap30 | +0.7 % | +0.0 | +0.0 | +0.5 | +4.3 % | +0.0 % | −0.2 % | +0.0 % |
| floor_lam (floor → box bottom) | +2.9 % | +0.0 | +2.5 | +0.3 | +7.1 % | −0.0 % | −0.0 % | +0.0 % |
| lblock_lam | +1.5 % | +0.0 | +1.4 | +0.1 | +3.6 % | −0.0 % | −0.0 % | +0.0 % |
| rot_lam | +2.2 % | +0.0 | +2.0 | +0.2 | +6.3 % | −0.0 % | −0.0 % | +0.0 % |

- **Receiver averaging** is a conservative sum: lossless for the port flux (identity checked to 1e-16). It acts only through the uniform re-emission assumption at the next bounce, which is the same source-averaging assumption; a Galerkin scheme with identical trial and test spaces cannot separate "source" from "receiver" error beyond this.
- **Source averaging** matters only for **convex sub-port structure** (an exponential decay), not for a uniform field, a linear ramp or a step: the effect is 0 for those to 0.2 %.
- **By distance:** 94 % of the gap4 effect comes from source–receiver pairs closer than 1 cm; the floor control's comes from 1–2 cm pairs (the 1 cm clearance). Beyond 2 h the effect is ≤ 0.5 %.
- **By bounce (Neumann terms):** the coarse-vs-fine difference on the gap receiver appears at the first interreflection (+31 % gap4, +17 % gap8, +5.9 % floor) and then stays roughly constant. The first-order (sun) emission carries the sub-port structure (a 2.3 mm sunlit strip in a 10 mm edge port), i.e. sub-port visibility variation enters at bounce 0 and is then transported.
- **Directional binning** cannot matter for a Lambert result: the production 64 / 16-bin solver and the bin-free reduced solve of the same C agree to 2.5e-7 (`prod_vs_reduced_solve_max_rel`).

### Stage C — CASP on the Lambert fixtures (amended coverage, validated references)

36 ruled groups (≥ 5 ports, ≥ 2 % of the camera signal, ≥ 90 % coverage, 6 fixtures, all cameras). One group (gap4 obot slab_hi:−y) has a zero-signal reference and is flagged, not counted.

- **C0 fails 8 groups** (worst +42.2 %); **CASP fails 1**: gap8 obot slab_hi:bot (10 ports) +4.7 % (C1 vs F16 +3.4 %). Worst |B| 42.2 → 4.7 % (−89 %). **Predeclared rule: PARTIAL, not RESTORED.**
- The sensitivity arm c = 4 (not selected) has |B| ≤ 2.5 % on every table row above and uses 3.7× more cells than c = 2 on gap4; c = 1 is too coarse and fails 8 of the 11 rows; 4× rays changes the residuals by less than 1 point, so the level choice, not the ray count, is the limit.
- **Unknowns (CASP primary vs uniform):** gap4 3,528 cells (uniform s = 4: 7,680 at 3.1 %; s = 8: 30,720 at 1.2 %); gap8 1,188; floor_lam 1,866 (uniform s = 2: 4,560 at 1.7 %); gap30 480 (unrefined, by the rule).
- **Residual structure:** CASP's remaining +1–2 % on box undersides is the c = 2 level choice (cells 5 mm at 1 cm clearance) plus 0.3–0.4 % replicate noise; c = 4 removes it.
- **Per-port distribution (vs F16, gap4 faces):** p90 |relative error| C0 176–193 % → CASP 37 %; p90 on the floor box underside 16 % → 7 %. Port-level errors are far larger than the pooled group means.
- **Near / far and visibility-edge ports** are reported in `stageC/*.json` (`*_near`, `*_vis`).
- **Temporal change tracking, rigid box move (floor_lam → rot_lam):** the true change is +2.5 % of the box-underside signal and +8.6 % of the floor-under-box signal. Per-port slope / correlation of the predicted change against the Cycles change: box underside C0 0.936 / 0.984 → CASP 1.008 / 0.991 (fine 0.999 / 0.998); floor C0 0.958 / 0.997 → CASP 0.993 / 0.999.

### Stage E — coupled evaluation with the stored WL53 medium K (1 cm ports, coarse K interface), amended coverage

- **gap4, gap8 (medium–medium): CASP = C0** within 0.52 points on every group (≤ 0.1 on the large groups; the difference is the MC noise of a different ray realization), since all unknowns stay coarse (480 cells both; ρ equal to 4 digits; C0-rebuilt reproduces WL53's stored ρ to 16 digits). WL53's gap-face errors (−4.3 / −0.4 / +5.7 % gap4 gapdown R/G/B, −2.8 / +0.6 / +9.4 % gapup) are unchanged.
- **Slab over a Lambert floor (floor, lblock, dent):** the only C0 miss left under full coverage is the slab underside (`underup dom:bot`: floor +2.5 / +3.5 / +4.4 %, dent +2.7 / +3.6 / +4.6 %). CASP: +1.4 / +2.0 / +1.7 % and +1.4 / +1.9 / +1.6 % (uniform s = 4 floor: +0.9 / +1.3 / +0.7 %). The floor group is 0–2 % in every arm.
- **bend90:** slab underside +2.3 / +2.8 / +4.2 % with CASP (uniform fine floor +2.1 / +2.5 / +3.7 %): the blue channel misses 3 % and does not respond to C refinement, so this is not a C-resolution effect.
- **Small side groups** (otop dom:+y −5…−9 %, dom:−x −4…−10 %, 10–15 ports, 2–3 % of the signal) are unchanged by CASP and are not C effects.
- **Spectral radius** (validated estimator): floor 0.7292 / 0.7317 / 0.7322 (C0 / CASP / uniform s = 4), identical solve iteration counts (38) and converged totals within 0.6 %: refinement leaves the coupled operator stable and unchanged.

**Labelled control — matched-resolution K (not a candidate).** TR oracle slab at 1 cm (240 ports) and 5 mm (960 ports), identical 4 × 4 / 4 × 4 bins, composed with uniform C at the same resolution against the stored Cycles volume images:

| gap4 group | R / G / B at 1 cm | R / G / B at 5 mm |
|---|---|---|
| gapdown slab_lo:top | −4.2 / −0.3 / +5.6 % | −1.3 / +0.1 / +3.2 % |
| gapup slab_hi:bot | −2.8 / +0.7 / +9.5 % | −0.6 / +0.7 / +5.5 % |
| bot slab_lo:bot | −4.6 / −0.5 / +3.5 % | −1.5 / +0.1 / +3.2 % |
| obot slab_lo:−y (oblique side, 10 ports) | −31.1 / −29.9 / −13.8 % | −39.3 / −38.1 / −17.3 % |

gap8 behaves the same (slab_lo:−y −15.6 → −23.5 %; gapup B +7.7 → +2.9 %). ρ values of this control were computed with the WL53 estimator (the run preceded A2) and are not used.

**Paired stale-vs-current K differential (bend90, dent; bin readout, same C).** D = pred(stale K) − pred(current K) as a fraction of the group signal. D is **insensitive to C**: C0 and CASP agree to ≤ 0.1 points on every group. It is small (|D| ≤ 0.8 % except the bend90 slab underside, −2.3 / −2.3 / −1.8 %) and limited by the oracle's own noise: two stale-K seeds differ by 0.77 / 0.80 / 0.57 points on the bend90 slab underside and by 0.7 / 0.7 / 1.7 points on the dent slab underside, where |D| (0.4–1.0 %) sits inside that noise. The predeclared ranking rule |D| > 2(|B_current| + 3 SE) holds on a handful of groups only (e.g. bend90 top dom:top R), because the absolute bias floor (0.5–5.5 %) exceeds |D| / 2 on the slab undersides. The 1 % differential target is not met on the slab undersides (K noise alone is 0.6–1.7 points).

### Stage F — cost and scaling (GPU uncontended, 5 repeats, CUDA-synchronised medians, bins 64 / 16, equal ray budget)

| config | coarse ports | unknowns C0 / CASP / uniform s=4 | GPU work (C + sun + 20 applications) CASP / C0 | total CASP / C0 | total uniform s=4 / C0 |
|---|---|---|---|---|---|
| floor 10 / 20 / 30 / 60 cm | 340 / 640 / 1,140 / 3,840 | 340 / 940 / 5,440 … 3,840 / 4,569 / 61,440 | 1.04 / 1.01 / 1.03 / 1.03 | 1.83 / 1.78 / 1.72 / 1.60 | 12.1 / 12.0 / 12.3 / 13.2 |
| gap 4 / 8 / 30 mm | 480 | 480 / 3,528 / 7,680; 480 / 1,188 / 7,680; 480 / 480 / 7,680 | 1.04 / 1.00 / 1.01 | 2.04 / 1.96 / 1.91 | 10.8 / 10.6 / 10.8 |
| rot_lam / lblock | 1,140 / 1,170 | 1,140 / 1,863 / 18,240; 1,170 / 2,202 / 18,720 | 1.04 / 1.03 | 1.74 / 1.75 | 12.3 / 12.7 |

- C relation build (the GPU part) for C0 grows from 14 ms (340 ports) to 118 ms (3,840 ports), i.e. close to linear over this range (×11 ports, ×8 time); 20 applications cost ~15 ms for C0 and CASP (40 ms for uniform s = 4 at 3,840 ports); GPU peak 0.9–3.0 GB; C bytes 1.3–30 MB. **No asymptotic claim is made.**
- **Wall-clock totals are CPU-dominated** and not an algorithmic result: CASP's extra ≈ 65–145 ms is unoptimised numpy mesh refinement (50–125 ms) plus the fine SceneSet (8–14 ms) and the 4 ms level prepass; the uniform s = 4 total is 0.8–3.1 s because WL53's `SceneSet` loops over ports in Python. The comparison that matters is the GPU column.
- **Predeclared efficiency rule** (unknowns ≤ 4× coarse **and** C build + 20 iterations ≤ 10× C0): time holds everywhere; unknowns hold on floor / lblock / rot (1.2–2.8×) and gap8 (2.5×), and fail on gap4 (**7.35×**). Update cost after a rigid box move is a full rebuild for both arms; no incremental update was built.

---

## OBSERVATION
- Near contact, the radiance of a face decays by several-fold across a 1 cm port (gap4 lower-slab top: 0.0702, 0.0063, 0.0020 … at port centres 0.5, 1.5, 2.5 cm from the open edge in the converged solution; 0.0740, 0.0134, 0.0050 in the 1 cm interface). The coarse interface diffuses numerically: second and third ports are ~2× too bright.
- The error vanishes at a 30 mm gap and for fields that are uniform, linear or stepped inside a port.
- The same refinement that cuts the Lambert error leaves the coupled spectral radius, solve iteration count and totals unchanged.
- A medium K at port resolution cannot use sub-port information from C: its response to a refined input is the sum over the port.
- Several large "errors" in WL53 are properties of the comparison (partial coverage, reference pixel size, partial visibility through a gap), not of the transport.

## INTERPRETATION

| WL53 failure | Origin identified here |
|---|---|
| Box underside +5–6 % (Lambert control) | **Source / boundary representation capacity of the 1 cm interface** (A explains 100 %); not implementation, not estimator noise, not directional binning, not readout |
| Floor under the box +3.9–5.2 % | **Comparison / readout** (partial-coverage ports at the view border); +1.2 % under full coverage, within the spatial effect |
| Lambert gap faces (+7…+44 %, not measured by WL53) | **Boundary representation capacity**, strongest for pairs closer than 1 cm; estimator noise adds ±1.7–3.3 % per realization |
| Medium gap faces (WL53 −4…+9 %) | **Interface resolution of K and C together**: C alone cannot change it; a matched 5 mm K/C halves the red and green errors; the blue-channel remainder (+3…+5.5 %) is unresolved |
| gap8 oblique lower-side group (−24 %) | **Not C-spatial.** Comparison unit (−24.5 → −16.2 %) plus an unresolved remainder; worsened by finer K |
| 10-port oblique upper-face groups of the gap scenes | **Comparison:** partially visible ports (reported separately) |
| Spectral radius of refined operators | **Our metric** (estimator), not the operator |

Multiple factors are present together, as the brief anticipated; the dominant one for local-K domains is the first row.

## UNRESOLVED QUESTIONS (consequential)
1. **The gap8 oblique lower-side group and the blue-channel medium remainder.** The finer-pixel volume render that would separate comparison from physics did not run (A4: LabServer69's root filesystem is full; Docker cannot start). The protocol's rule for it is unevaluated. **The user's decision is needed to free space on that machine** (or provide another one) before this can be closed; I did not delete anything outside this batch.
2. **K with sub-port structure.** The evidence says that, for medium K, accuracy near contact is capped by the K port resolution. Whether to give K sub-port inputs / outputs (a spatially structured intrinsic operator) is an architecture decision, not a C decision.
3. **Pixel-converged references** exist only for gap4 gapdown / gapup; every other group uses 0.5 mm references with the amended coverage rule.
4. **bend90 slab underside** (+4.2 % blue with CASP) is not a C-resolution effect; its K-side or geometry cause is not isolated.

## ARCHITECTURE VERDICT (measured conditions only: protocol `fnto_c54` + A1–A4, fixtures above, LabServer69)

| Item | Verdict |
|---|---|
| Baseline audit reliability | **CONDITIONALLY SUPPORTED.** Tables replay exactly and a fresh run is bit-identical; four caveats found (coverage rule, reference pixel size, oblique partial visibility, ρ estimator) and an omission in WL53's prose |
| Spatial-error attribution | **SUPPORTED** for local-K (Lambert) domains and near-contact transport; **CONDITIONALLY SUPPORTED** for medium K (interface resolution of K and C together; remainder unresolved) |
| Candidate physical correctness | **SUPPORTED.** 10 / 10 tests (one failed refinement run recorded and corrected), exact closure, analytic form factors, ownership checks, stable coupled operator |
| Absolute composition accuracy | **NOT SUPPORTED at the predeclared contract; PARTIAL evidence.** 35 / 36 ruled Lambert groups pass 3 % (worst 42.2 → 4.7 %); medium-to-medium links unchanged; bend90 slab underside misses 3 % in blue |
| Near-contact and oblique-view fidelity | **CONDITIONALLY SUPPORTED** for face-on near-contact groups (floor / L-block / gaps ≥ 8 mm within 3 %); **INCONCLUSIVE** for the oblique 10-port groups (comparison-limited) |
| Differential K-reuse evaluation readiness | The paired differential is C-insensitive (≤ 0.1 points): **CONDITIONALLY SUPPORTED** as a measurement, limited by K-oracle noise (0.6–1.7 points). Stale-vs-current **ranking** by absolute error: **NOT SUPPORTED** (bias floor > |D| / 2 on the slab undersides) |
| Dynamic-update cost | **CONDITIONALLY SUPPORTED.** GPU work 1.00–1.04× C0 at equal rays; unknown budget fails on the 4 mm gap (7.35×); CPU refinement unoptimised |
| Scalability | **INCONCLUSIVE.** Close to linear in ports over 340–3,840 (GPU build); unknowns grow with near-contact density (1.0–7.35×); no asymptotic claim |
| Production integration readiness | **NOT SUPPORTED** |

**Contract for subsequent K-reuse research (what the data support).** For domains with a local (Lambert-like) K: refine C until cells are ≤ d / 2 of the near-field distance (c = 2) for a 3 % contract at an equal ray budget; for a 1 % target use c = 4 (post hoc observation, 3.7× more cells). For port-resolved K: do not claim better than the K port resolution; the measured matched pair (5 mm K and C) halves the red/green near-contact error and leaves blue at +3…+5.5 %. Never evaluate a K-update ranking by absolute error where the bias floor exceeds half the paired differential.

---

## Deviations, post hoc items and failures (all listed in the protocol)
- A1 changes how groups are selected (coverage ≥ 90 %); every table is reported under the declared and the amended rule.
- Exploratory Lambert runs on gap4 / floor_lam (`explore1–3`) were made before the protocol was committed and are disclosed in it.
- A2: my own spectral-radius metric was wrong for refined operators; Stage E was re-run (same seeds); the first run is kept on the server (`out/stageE_v1_wl53rho`, not copied).
- The K control ran before A2's correction; its ρ is not used.
- A3 renders and the final-chain mask test (wrong directory) did not run as planned; see A4. The cost benchmark failed once on ENOSPC and was re-run (`final/final_cost_rerun.log`).
- The first run of `refine_mesh_conservation` failed (above) and the implementation was corrected.
- Stage B's pixelwise fine-vs-Cycles comparison (`*_pixelwise.json`) was computed but is not used for verdicts.

## REPRODUCIBILITY

**Commits (branch `wl54_explicit_c`).** 7d7a348 protocol + Stage A / B code · 7712c32 CASP + tests · d98cdf0, f02ac6d Stage C / E / F drivers and RGB fix · f5873a2 A1 · f9cd6de, a10c68c A2 · 4f61bdf K control · 2d4db30 resolution script · 9d54b66 A3 · afb80d1 launch scripts · 0957838 A4 · this worklog's commit (analysis scripts, results and figures). Every evidence run used code from a commit that precedes it, except the analysis-only scripts listed in the last commit.

**Scripts.** `c54_audit.py` (replay / unit / rerun), `c54_stageB.py`, `c54_stageC.py`, `c54_stageE.py`, `c54_kfine.py`, `c54_cost.py`, `c54_known_field.py`, `c54_change.py`, `c54_dnoise.py`, `c54_amend1.py` (tables / res), `c54_amendE.py`, `c54_coverage.py`, `c54_rho_check.py`, `c54_profiles.py`, `c54_report.py`, `tests/test_c54.py`; `run.sh`, `sync.sh`, `launch_*.sh`.

**Seeds.** Cycles 5401–5404 (resolution diagnostic 5499); C0prod 5340 / 5341, replicates 5600+r / 5700+r; C0inf 5400 / 5401; fine 5410+s / 5420+s (replicate 5510+s / 5520+s); CASP prepass 5801, C 5810 / 5811, replicates 5820+r / 5830+r; TR K control 5851 / 5852; known-field fine 5900 / 5901.

**Sample counts.** C0 128 rays per out-cell, K_ext 256; fine {2: 1024, 4: 512, 8: 256, 16: 128, 32: 64} rays per cell; CASP 8192 rays per coarse port (4× arm 32768); TR 2048 photons per input cell; sun 16 M photons per medium domain.

**Outputs (`results/fnto_c54/`).** `audit/` (replay, unit, rerun diff), `stageB/` `stageC/` `stageE/` `kfine/` (JSON + coarse arrays), `final/` (amended tables, pixel-size diagnostic, known field, cost, change, dnoise, profiles, ρ check, tests, `lambert_table.json`, `summary.json`, `worklog_tables.md`), `tests/` (including the failed v1). Figures with a manifest: `results/evaluation/54/`. Images and operators (Cycles images, T matrices) stay on LabServer69 (`~/rest_wl54/out`, `~/rest_wl53/results`).

**Verified by source and recomputed data (this batch):** claims A, B, C, E; the replay; the decomposition A / B / C / T; the coverage, pixel-size and ρ findings; all of Stages C–F. **Relying on WL53's description only:** the TR oracle's physics and K properties (WL53's tests re-run unchanged), the Cycles volume validation (H-function, furnace, etc.), and the origin of the gap8 oblique group beyond what is stated in D.

No later batch is started or recommended.
