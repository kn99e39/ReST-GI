# Worklog — Spatially Resolved Intrinsic K Boundary Interface: Bounded Architecture Falsification (2026-10-28)

**Protocol.** `fnto_k55` (`experiments/fnto_k55/protocol/k55.json`), committed with the Stage A / B code (0b98812) before any WL55
evidence run, then extended by three labelled post hoc amendments, each fixed before the run it governs:
- **A1:** record of the failed COR check, a COR-free gate, the pixel-size diagnostic, the readout decision, entry to Stage C and the dent operators.
- **A2:** Stage D closure accounting and a rigid replicate diagnostic.
- **A3:** the pixel rule extended to the held-out cameras.

The candidate (CAK), its semantics, its falsifiers and the held-out geometry (gap6s) were declared in the original commit, before Stage B.

**Authorisation.** The user brief "ReST-GI — Worklog 55: Spatially Resolved Intrinsic K Boundary Interface — Bounded Architecture
Falsification" (2026-10-28) with its branch policy:
- work directly on `Selective_Recompute`;
- no new branch or worktree;
- isolate through directories.

Mid-session user decision (2026-10-28): Cycles references are rendered on the GPU with the official Blender 3.5.1 build (CUDA). The user had
validated that build as statistically equivalent to the CPU fork, so no smoke re-check was run and none is kept as a future target.

**Isolation.**
- Code: `experiments/fnto_k55`. `experiments/phys_ref53` and `experiments/fnto_c54` are imported unchanged.
- Small results: `results/fnto_k55`. Figures: `results/evaluation/55`.
- Operators (2–2.5 GB each) and images stay on LabServer69 under `~/rest_wl55`.
- No production FNTO module, K44 / DASQ checkpoint or WL52 file was read or written.
- Commits by a parallel session during WL55 are not part of this batch: 1a9970c (`bpy_native.sh`), 4269869 (`p53_mi` temporary-file
  handling, results unchanged), 5eef575 / f27e73e / 70e4f10 (GPU smoke). WL55 re-ran the WL53 tests itself (9/9, identical values).

**Hardware / software.**
- LabServer69: RTX 3090 24 GB (driver 550.54.15), 16 cores, 125 GB RAM. The root filesystem is 99 % full (lab-managed, untouched); all
  scratch is on `/home`.
- Oracle side: torch 2.3.1+cu121, Mitsuba 3.5.1 `cuda_ad_rgb` as the intersector only.
- References: official Blender 3.5.1 (build e1ccd9d4a1d3) on CUDA, with WL53's `reset()` settings, 1024 spp. Each render's build and
  device are in `render_manifest.jsonl`.
  - A CPU fork chain was stopped after 6 renders, which are kept unused in `out/cycles`. No comparison mixes devices.
  - The units calibration ran on the fork CPU.

---

## Answer to the completion question

> Does FNTO require a spatially resolved intrinsic-K boundary representation to preserve physically important near-contact transport under
> dynamic geometry, and can that representation be constructed and composed accurately at a tractable cost?

**Yes, near contact.** The 1 cm boundary port loses physically important information on both sides of the K/C interface:
- **Outgoing:** uniform emission over a 1 cm port delivers +10 to +22 % too much first-flight flux across a 4–8 mm gap.
- **Incident:** with exact incident fields, single-flight K errors reach +2 to +5 % on the receiving face and the transmitted face.
- **Coupled:** the composed face-on errors reach +8.5 % (gap4 upper gap face, blue).
- **Geometry change:** a rigid move to 4 mm clearance makes the 1 cm interface miss by up to +19 %.

All of this converges with matched K/C refinement. At 2.5 mm, every face-on near-contact group is within ±1.8 % of the reference.

**A spatially resolved interface can be built and composed correctly**, and it handles a near-contact geometry change.
- The predeclared candidate **CAK** (contact-adaptive co-refined K/C cells; K on the cells = exact conservative aggregation of a 2.5 mm
  material-coordinate oracle) passes every face-on group on the four design scenes and on the held-out gap6s.
- It reuses the rest operator for the rigid floor → floor_c4 move with only a re-aggregation (18/18 groups, worst 0.76 %; the reused 1 cm
  interface fails 6/18).

**It is not a complete success under the predeclared contract:**
- Oblique groups still fail. Most of them are **comparison-limited**: their point-sampled references are not pixel-converged even at
  0.125 mm. One held-out oblique group has a real residual.
- The predeclared tractability rule fails. CAK recovers only 63–64 % of the 2.5 mm improvement on the 1 cm-clearance scenes, although every
  group there is within 3 %.

**WL54's "real, unattributed" gap8 oblique lower-side error is not a K boundary effect.** It is the camera readout:
- WL53's next-event readout KC omits unscattered (ballistic) light.
- Seen obliquely near an edge, a face shows light that crossed a thin medium wedge from a bright surface behind it.
- Adding the ballistic term moves the continuous oracle from −33.5 % to −1.7 % (red) against the finest reference.

---

## Verification of WL54 (Stage A)

| Item | Result | Evidence |
|---|---|---|
| Source and provenance | **Verified.** All WL54 commits (7d7a348 … c69900a) are ancestors of `Selective_Recompute` and visible on GitHub (`gh api commits/c69900a`). The reviewer could not find `wl54_explicit_c` because that branch was fast-forwarded and deleted at the user's request; nothing is missing. Stored arrays are committed in `results/fnto_c54`; images and T matrices are present on LabServer69. Provenance note: WL54's A3 renders used `bpy_native.sh`, which was committed only afterwards (1a9970c) | `stageA/replay*.json` |
| Replay with independent code | **Reproduced exactly.** New group statistics (`k55_common.group_bias`, written from the definitions) on the stored arrays and images. 332 values compared, maximum difference 0.0 pp: 147 Lambert values at 0.5 mm, 44 at 0.25 mm, 132 values of the matched-K control and 9 of the gap8 lower-side group (1 / 0.5 / 0.25 mm). CASP = C0 on the medium gap scenes within 0.52 / 0.45 pp | `stageA/replay_v2.json` |
| "35 / 36 Lambert groups pass" | **Reproduced, with an interpretation note.** At the 0.5 mm references CASP fails 2 of 36 genuine groups (gap4 gapdown −4.5 %, gap8 obot +4.7 %) plus the zero-signal flagged group. The 35/36 count substitutes the 0.25 mm reference for gap4 gapdown, which is aliased at 0.5 mm, as WL54 A1 / A3 disclose | same |
| Code audit | **Clean.** `refine_mesh` area conservation 1e-15 (bend90, dent). CASP C column closure exactly 1, 0 back hits. The A² spectral-radius estimator matches scipy Arnoldi to ≤ 3.2e-8 (gap4 medium, floor_lam CASP); WL53's estimator is within 0.02 % on these two (WL54 A2's 2–4× failures were on other refined operators). The padded-port sampling edge case in `SceneSet` never occurred (0 in 8×10⁷ draws). The matched-K control used 4×4 / 4×4 bins and its own ray density, unlike WL53 production (disclosed prior P2) | `stageA/code_audit.json` |
| Regression tests | WL53 9/9 (identical values), WL54 10/10, WL55 10/10 | `tests/` |

**Two WL54 interpretations are revised by this batch.**
1. The gap8 oblique lower-side group is attributed: it is the readout's missing ballistic term (Stage B).
2. WL54 A3 called that group's reference "not a pixel-size effect (0.5 → 0.25 mm +0.6 %)". The reference is **not converged**: it moves
   +1.2 % from 0.5 to 0.25 mm and then +5.0 % from 0.25 to 0.125 mm, because a sub-millimetre ballistic band is under-sampled by
   point-sampled pixels.

WL54's measured numbers are unaffected.

---

## IMPLEMENTATION FACT

**Instruments (Stage B, all new, `experiments/fnto_k55`).**
- **COR**, a continuous oracle (`k55_cor.py`). A multi-domain path tracer with WL53's TR medium physics: free flight, phase function,
  albedo weighting, Russian roulette, float64 accumulation.
  - It uses exact positions and directions everywhere, traces air segments with the scene intersector, and handles Lambert reflection,
    black absorbers and profiled Lambertian emitters.
  - It has no ports and no bins. Readout is the same next-event estimator as KC, plus a ballistic term.
  - It is a diagnostic, never the reference.
- **KB**, a ballistic camera-readout term at the interface's own resolution (`k55_compose.ballistic_KB`). For each output port it is the
  projected-area mean of exp(−σ_t s) / (A_j π / D_in) for the backward camera ray leaving the medium through cell j in bin b. This assumes,
  as the interface itself does, that in-cell radiance is uniform over the cell and the bin.
- **Sparse multi-resolution TR operators** (`k55_ops.py`).
  - WL53's `p53_tracer.walk` is driven unchanged through a duck-typed accumulator.
  - T is stored sparse (one entry per photon exit, coalesced in float64). KC is accumulated directly on the 1 cm output ports.
  - Cells come from `c54_casp.refine_mesh` (exact surface), with welded vertices.
  - Coarser cells are obtained by exact R_out K P_in aggregation of the 2.5 mm oracle. This is tested to be flux-preserving to 6e-9 and
    equal in expectation to a direct TR run (port-pair Frobenius 0.43 %).
- **Matched composition** (`k55_compose.compose`).
  - K and C are on the same cells, with C at an equal ray density per area.
  - One first-order sun term per domain on the 2.5 mm mesh is shared by every arm, so arms differ only by their interface.
  - Readouts: `nee` (WL53), `nee + KB`, and `bin` (for Stage E).
- **Known-field fixtures** (`k55_fixtures.py`). In the gap geometry, the upper slab is replaced by a black box whose bottom is a one-sided
  Lambertian emitter: uniform (U) or exp(−(x + 5 cm)/5 mm) (X). A third variant (S) has no emitter: the sun only, with the upper slab black.
  These are single flight by construction.
  - Emitter units were calibrated: the camera-seen emitter is 1.000. A 0.8 % low reading came from Cycles' sampling of two huge triangles
    at 1 cm; with the 2.5 mm tessellation the fixtures use, the reading is exact (0.79973 vs 0.79974).
- **Outgoing test.** The arrival flux of slab_lo's first-flight exit field on the upper box bottom is compared at 2.5 mm cells, COR vs the
  out-cells at level s emitted uniformly through C.

**Candidate CAK (Stage C, predeclared, one mechanism, nothing tuned).**
- **Cells:** per-port levels {1, 2, 4} by the CASP geometry-only rule on the current geometry of all components (c = 2, ℓ_max = 4).
- **K on cells:** R_out K_fine P_in, where K_fine is the 2.5 mm TR oracle in material coordinates.
- **C:** CASP caster on the same cells (identity correspondence), equal ray budget.
- **Readout:** nee + KB.
- **Rigid moves:** reuse K_fine and re-level / re-aggregate, with no new TR.
- **Non-rigid moves:** K_fine is configuration dependent; Stage E evaluates stale vs current.

**Arms.**

| Arm | Description |
|---|---|
| P10 / P05 / P025 | Uniform matched K/C at 10 / 5 / 2.5 mm, aggregated from K_fine. Two K seeds each |
| P10d / P05d | Direct TR operators at 1 cm / 5 mm (aggregation-equivalence check) |
| P10_din64 | 8×8 incoming bins |
| P10_c2 | Second C seed |
| P10_wl53 | WL53's stored prediction |
| CAK | The candidate, 2 K seeds |
| COR | 2 seeds |
| Stale arms (Stage E) | P10st / CAKst / P025st: the rest K_fine on the deformed geometry |

**Tests.** WL55 10/10 (correspondence, nesting, sparse = dense TR bit-equal in T, aggregation, zero / linearity / dense fixed point 6e-8,
analytic ballistic side face, COR Lambert plane and isolated slab, ownership, rigid frames, deterministic statistics). Stage D operator tests:
5 / 7 as coded, plus the A2 diagnostics (below).

---

## MEASUREMENT

References are Cycles at 0.5 mm with 2 seeds, under the amended comparison unit (WL54 A1) unless stated. Biases are in % against Cycles;
the 3 % rule is |B| ≤ 3 % + 3 SE, with SE = reference ⊕ K-seed.

### Stage B — attribution

**The continuous oracle is accurate** on every face-on group (|R| ≤ 0.3 %) and on lit top faces (≤ 0.1 %). The floor_lam Lambert check
passes.

The predeclared check (iii) **failed** on the lit oblique −x faces (+2.0 … +2.6 % / −0.7 … −1.6 %). Per the rule, COR was therefore demoted
to a labelled diagnostic, and the gate was re-expressed COR-free (A1). Both versions gave the same outcome. The A1.2 pixel diagnostic later
showed that those −x references move by +1.9 % / −1.2 … −2.1 % at 0.125 mm, where COR is within +0.2 / +0.4 %.

**Readout — the WL54 gap8 oblique group (P1, rule B-E):**

| gap8 obot slab_lo:−y vs the 0.125 mm reference, R / G / B | Bias |
|---|---|
| WL53 stored prediction (nee) | −26.0 / −23.5 / −11.0 % |
| COR, nee | −33.5 / −30.6 / −12.0 % |
| COR, nee + ballistic | **−1.7 / −1.5 / +0.1 %** |
| 1 cm, nee + KB | +5.9 / +5.3 / +1.0 % |
| CAK, nee + KB | −0.3 / −1.2 / −3.4 % |

- B-E holds for COR in all 3 channels: at 0.5 mm the reduction is 0.85 / 0.86 / 0.95.
- The interface-consistent KB at 1 cm overshoots. It converges with refinement: I = +7.7 → +1.9 → 0.0 %.
- The same omission explains WL54's "small side groups not C effects": bend90 otop dom:−x goes from −5.8 / −6.1 / −8.4 % to
  −0.5 / −0.6 / −0.9 % with KB.

**Face-on near-contact groups (nee + KB, amended unit):**

| Group | 1 cm | 5 mm | 2.5 mm | CAK | COR |
|---|---|---|---|---|---|
| gap4 gapdown slab_lo:top | −4.2 / −0.3 / +5.6 | −1.0 / +0.3 / +3.1 | 0.0 / +0.5 / +1.8 | 0.0 / +0.5 / +1.8 | +0.2 / +0.2 / 0.0 |
| gap4 gapup slab_hi:bot | −2.8 / +0.6 / +8.5 | −0.5 / +0.7 / +4.4 | +0.2 / +0.6 / +1.8 | +0.2 / +0.6 / +1.8 | +0.2 / +0.2 / +0.2 |
| gap8 gapup slab_hi:bot | −1.2 / +1.1 / +6.7 | −0.3 / +0.5 / +1.9 | +0.3 / +0.6 / +1.4 | −0.2 / +0.5 / +1.8 | +0.2 / +0.2 / +0.2 |
| gap4 bot slab_lo:bot (transmitted) | −5.0 / −1.3 / +0.6 | −1.8 / −0.6 / +0.5 | −0.8 / −0.4 / +0.4 | −0.8 / −0.4 / +0.4 | +0.1 / +0.2 / +0.7 |
| floor underup dom:bot | +2.6 / +3.6 / +4.3 | +0.8 / +1.0 / +0.2 | +0.2 / +0.2 / −1.0 | +1.1 / +1.4 / +0.8 | +0.2 / +0.1 / +0.2 |
| bend90 underup dom:bot | +1.6 / +1.8 / +2.3 | +0.8 / +0.9 / +1.1 | +0.4 / +0.4 / +0.4 | +0.8 / +0.9 / +1.2 | 0.0 / 0.0 / 0.0 |
| *held out:* gap6s gapup slab_hi:bot | −1.2 / +1.3 / +4.7 | +0.2 / +0.9 / +0.8 | +0.4 / +0.6 / −0.8 | +0.4 / +0.5 / −1.0 | 0.0 / −0.2 / −2.4 |
| *held out:* gap6s gapdown slab_lo:top | −1.7 / +1.2 / +4.0 | +0.2 / +1.3 / +2.6 | +0.6 / +1.0 / +1.9 | +0.6 / +1.0 / +2.0 | +0.2 / +0.2 / +0.2 |

Interface sub-attribution (rule B-I), applied to these groups:
- **Spatial:** monotone convergence, with |I(2.5)| ≤ ½ |I(10)|, on every near-contact face-on group.
- **Not oracle noise:** the s1 − s2 difference stays below |I| / 2.
- **Not C noise:** the second C seed changes I by < |I| / 2.
- **Directional:** a minor component in a few channels. 8×8 incoming bins move the transmitted bottom by +0.6 points (gap8, G).

The blue channel converges slowest (diffusion length 1.8 mm). This explains WL54's "blue remainder at 5 mm".

**Single flight (known fields, K_s vs COR):**
- Face-on receiving face: X field +2.4 / +2.2 / +1.8 % (gap4) at 1 cm → ≤ 0.3 % at 2.5 mm. U field ≤ 0.9 %.
- Transmitted bottom face: X field +5.1 / +4.7 / +3.2 % at 1 cm → −0.8 / −0.9 / −1.7 % at 2.5 mm. U field −2.0 … −3.0 % at 1 cm →
  −1.0 … −2.2 % at 2.5 mm, a partly directional remainder.
- Binning the collimated sun into 4×4 incoming bins at 1 cm cells would cost +5 … +20 %. The composition uses the exact first-order
  term (L0), which is within 0.1–2 % of Cycles; L0 = COR on every group.
- Oblique side faces are reference-limited (below).

**Outgoing test (first-flight exit field across the gap, vs COR):**

| Fixture | Pooled flux error, 1 / 5 / 2.5 mm | Per-port p90, 1 / 5 / 2.5 mm |
|---|---|---|
| gap4 | +15.6 / +6.0 / −0.5 % (R); +22.3 / +10.0 / 0.0 % (B) | 97 / 16 / 12 % |
| gap8 | +9.7 / +0.4 / −0.2 % | 34 / 13 / 13 % |

**Gate C: demonstrated**, both declared and COR-free. Spatial interface loss appears on 7 near-contact face-on groups in ≥ 2 channels, with
single-flight A loss and outgoing B loss.

### Pixel-size diagnostic (A1.2 / A3)

The oblique side and through-gap groups carry a sub-millimetre ballistic band, about 0.5 mm high in the image. Changes of the group signal
from 0.5 to 0.125 mm:

| Group | Change, 0.5 → 0.125 mm |
|---|---|
| gap8 obot slab_lo:−y | +6.3 / +5.4 / +0.6 %, non-monotonic (+1.2 % to 0.25 mm, then +5.0 %) |
| slab_hi:−y | +3.3 … +4.3 % |
| obotx bottom faces, blue | +11.8 / −12.7 % |
| gap6s obot slab_hi:bot, blue | −19.6 % |
| Known-field side faces | +8 … +16 % |

These groups are **not pixel-converged at 0.5 mm, and their convergence at 0.125 mm is not demonstrated**. Point sampling under-resolves
the band.

### Stage D — the candidate

**Operator tests (2×2×1 cm block, mixed levels):**

| Test | Result |
|---|---|
| Flux across resolutions | 6e-9 |
| Passivity (max column sum) | 0.998 / 0.979 / 0.843 |
| Reciprocity | 0.76× cross-seed noise |
| Zero / linearity | 0 / 1.4e-7 |
| Correspondence | 7e-16 |
| Ownership (C back hits) | 0 in every composition |
| Composition | Converged in every arm; ρ (A²) 0.43–0.83 |
| Closure at albedo 1 (A2) | **Failed as coded** (min column sum 1 − 1/2048). The column deficit is exactly the 4 leaked photons, leak rate 4.8e-7. A test-specification error |
| Rigid pose (A2) | **Failed as coded** (1.11 / 1.12 / **1.18**, rule [0.85, 1.15]). A2 replicates (4 + 4 seeds): r = 0.82 / 0.85 / 1.05, a formal miss on the low side. No excess over noise, so no evidence of pose dependence |

**Coupled (nee + KB, amended unit, 0.5 mm references): ruled groups passing the 3 % rule.**

| Scene | 1 cm | CAK | 2.5 mm | CAK cells / 2.5 mm cells | CAK failures |
|---|---|---|---|---|---|
| gap8 | 27/40 | 33/40 | 34/40 | 1,188 / 7,680 | all oblique: obot slab_hi:−y, slab_lo:−y; obotx bottom blue |
| gap4 | 20/34 | 28/34 | 28/34 | 3,480 / 7,680 | all oblique |
| floor | 16/18 | **18/18** | 18/18 | 1,866 / 18,240 | — |
| bend90 | 21/21 | **21/21** | 21/21 | 1,515 / 18,240 | — |
| gap6s (held out) | 23/33 | 30/33 | 30/33 | 3,483 / 7,680 | 3 blue oblique groups |

- Every face-on group of every scene passes with CAK, including the held-out scene.
- Against the 0.125 mm references, the oblique failures that remain at CAK are:
  - obotx bottom-face blue: +5.8 % (gap8), −6.4 % (gap4), +4.5 % (gap6s). COR shows the same values, so the reference is the limit.
  - **held-out gap6s obot slab_hi:bot:** CAK +3.8 / +4.0 / +6.1 % against COR +2.6 / +2.0 / +1.6 % (identical at uniform 2.5 mm). This is
    a real interface residual on the shifted upper slab's gap face seen through the gap opening.
- **Tractability rule** (declared): CAK must recover ≥ 80 % of the 2.5 mm improvement at ≤ 50 % of its cells. **Failed.**

  | Scene | Median recovery | CAK cells / 2.5 mm cells |
  |---|---|---|
  | gap8 | 0.88 | 0.15 |
  | gap4 | 1.00 | 0.45 |
  | floor | **0.64** | 0.10 |
  | bend90 | **0.63** | 0.08 |
  | gap6s | 1.04 | 0.45 |

### Stage E — dynamic geometry

**Near-contact rigid move, floor → floor_c4 (4 mm clearance).** The rest K is reused in all arms.

| Arm | Groups passing | Slab underside | Floor under slab |
|---|---|---|---|
| 1 cm | 12/18 | +6.7 / +10.0 / +19.4 % | +3.9 / +4.7 / +3.5 % |
| CAK (re-levelled to 4,704 cells, re-aggregated in 0.72 s, no TR) | **18/18** (worst 0.76 %) | — | — |
| COR | 18/18 (worst 0.40 %) | — | — |

**Operator change, ‖K_cur − K_rest‖ / ‖seed noise‖:**

| Aggregation | bend90 | dent |
|---|---|---|
| 2.5 mm cell pairs | 1.6 / 1.7 / 1.7 | 1.2 / 1.2 / 1.1 |
| CAK cells | 4.3 / 4.5 / 3.9 | 3.2 / 3.2 / 1.9 |
| 1 cm ports | 7.9 / 8.1 / 6.5 | 6.7 / 6.8 / 3.6 |

- The 1 cm values reproduce WL53's 16×-photon results (bend 7.95 / 8.18 / 6.86, dent 6.29 / 6.42 / 3.53).
- Structure:
  - dent: 93–100 % of the change is within the dent support; rank above noise 20–26.
  - bend: dense, rank above noise ~200; 79–86 % of the energy is within the same port.
- Column-sum changes, mean / absmax (blue): bend −0.04 % / 1.3 %; dent +0.18 % / 3.9 %.
- At the 2.5 mm level, out-bin structure is indistinguishable from noise with 2048 photons per cell.

**Criteria A / B / C:**
- **A, detect:** met at 1 cm and at CAK-cell aggregation (except dent blue at CAK, 1.9). Not met at 2.5 mm cell pairs.
- **B, predict:** temporal change floor → deformed, per port. Met for every camera and every arm (slope 0.92–1.08, corr ≥ 0.98).
  Refinement improves the bend90 underside slope from 0.916 (1 cm) to 0.976 (CAK).
- **C, rank** (bin readout, WL54 rule extended): **not met** on most groups. The limit is the **bias floor**, not oracle variance (K-seed
  SE of D ≤ 0.24 points).
  - With a refined interface it is met on the bend90 underside (CAK: R; 2.5 mm: R and G), where |B_cur| falls to 0.3–0.7 % against
    |D| ≈ 1.6 %.
  - Dent (|D| ≤ 0.46 %) cannot be ranked anywhere.
  - Stale-K **nee** readouts are invalid on rotated faces: the rest camera operator does not transfer to bent frames (bend90 otop dom:−x
    D = −15.9 %). Bin is the primary readout.

### Stage F — cost (LabServer69, GPU idle, CUDA-synchronised medians)

**Oracle K_fine (2.5 mm, slab):**
- 126 M photons, 380 s TR.
- 57.6 M nonzeros (bounded by photons; not intrinsically sparse), 2.0 GB including the 0.9 GB camera operator.
- Peak GPU 4.6 GB.
- Direct 1 cm: 7.9 M photons, 22 s. Direct 5 mm: 31 M photons, 78 s.

**Per arm (cells; K nonzeros):**

| Scene | 1 cm | CAK | 2.5 mm |
|---|---|---|---|
| floor | 1,140; 5.3 M | 1,866; 12.6 M | 18,240; 57.6 M |
| floor_c4 | 1,140 | 4,704; 31.3 M | 18,240 |
| gap4 / gap6s | 480 | 3,480; 2 × 28.8 M | 7,680; 2 × 57.6 M |
| gap8 | 480 | 1,188; 2 × 13.2 M | 7,680 |

**Per-iteration K + C application:** 0.9–1.0 ms (1 cm), 1.0–2.2 ms (CAK), 2.3–3.8 ms (2.5 mm). C application alone is ~0.42 ms.

**Geometry update:**
- C build: 14–40 ms.
- CAK re-levelling: 12 ms.
- Re-aggregation from K_fine: 0.5–1.4 s.
- Full composition including the ballistic operator: 0.24–0.43 s.
- Peak GPU 1.5–4.6 GB.

**Scaling:**
- Cells grow ×4 per halving (uniform) and with near-contact area (CAK 1.3–7.3× coarse).
- Directional input 8×8: ×4 oracle photons, 97 s.
- Domains: 2 slabs = 2 aggregated operators.

**No inference-speed claim.** The 2.5 mm oracle is an expensive reference, not a deployable representation.

---

## OBSERVATION

- Near contact, the information that 1 cm ports discard is the **position of incident and exitant flux within the port**, on both sides of
  the interface. The loss vanishes for uniform fields and grows with the field's concentration (the X field) and with proximity (gap4 > gap8).
- The outgoing side dominates the coupled error: uniform emission puts too much flux into the gap.
- The blue channel, with the shortest diffusion length, converges last. Red converges by 5 mm.
- Refinement leaves the coupled operator stable: ρ changes by ≤ 0.012.
- A contact-creating rigid move needs no new oracle when K is held at a fine material resolution and aggregated to the current contact
  structure.
- Oblique views near edges see through thin medium wedges. The resulting unscattered term is large (up to 30 points) and lives in a
  sub-millimetre band that point-sampled references do not resolve.

## INTERPRETATION

| Phenomenon | Origin identified here |
|---|---|
| gap8 / gap4 oblique lower-side under-prediction (WL54 "real, unattributed") | **Readout:** KC omits the ballistic term. Interface-consistent KB at 1 cm overshoots and converges with refinement |
| Face-on gap faces, red and blue (WL53 / WL54 medium remainder) | **Boundary information loss on both sides of K/C** (spatial; minor directional part on transmitted faces) |
| Slab underside (floor, dent, bend90) | Same, at 1 cm clearance; within 3 % at 5 mm |
| bend90 slab underside blue (WL54 +4.2 %, "not C") | **Interface spatial** (K side); +2.3 % → +0.5 % with matched refinement. WL54's value used the WL53 1 mm reference |
| Oblique side and through-gap groups at the few-% level | **Reference / comparison:** the point-sampled pixel reference is not converged; COR, which shows the same residual, confirms this |
| Held-out gap6s obot slab_hi:bot blue (+6.1 % at CAK vs COR +1.6 %) | **Residual interface error at 2.5 mm**, not resolved within this batch (misaligned near-opening geometry) |
| Transmitted bottom faces, −1 to −2 % at 2.5 mm | Partly **directional** input aggregation (single-flight U field; din64 shifts it) |
| Oracle estimator noise | Not a limit for coupled radiance (K-seed SE ≤ 0.24 points). It **is** the limit for resolving ΔK at 2.5 mm cell level with 2048 photons per cell |
| Stale-vs-current ranking | Limited by the **interface bias floor**, which refinement lowers enough to rank only the largest change (bend90 underside) |

## UNRESOLVED QUESTIONS (material to the next architecture decision)

1. **Reference for oblique sub-millimetre structure.** Point-sampled pixels do not converge (0.125 mm still moves by up to 5 %).
   Area-integrated pixels or an exact port-integral reference would be needed to rule on oblique groups. In this Cycles build `BOX` ignores
   `filter_width` (WL42).
2. **Held-out through-gap blue residual** (gap6s): does it need sub-2.5 mm cells, or directional resolution, at misaligned openings?
3. **Level rule strength:** c = 2 gives 63 % of the 2.5 mm gain on 1 cm clearances. That is within 3 %, but not within the 1 % that ranking
   stale K would need. A stricter rule was not tested (one candidate).
4. **Learnability and compression of K_fine.** The oracle is MC-sparse and noise-dominated at its own resolution, so any learned or
   compressed K needs a target defined at an aggregation level where ΔK is resolvable (CAK cells / ports), and an aggregation-consistent
   hierarchy.

## ARCHITECTURE VERDICT (measured conditions only: protocol `fnto_k55` + A1–A3, fixtures above, LabServer69)

| Item | Verdict |
|---|---|
| WL54 evidence reliability | **CONDITIONALLY SUPPORTED.** 332/332 replayed values exact and the code audit clean, but two WL54 interpretations are revised: the gap8 oblique group is a readout omission, and its reference is not converged at 0.25 mm |
| K boundary-information sufficiency (1 cm ports) | **NOT SUPPORTED** near contact: outgoing +10 … +22 % flux, single-flight incident up to +5 %, coupled up to +8.5 %, contact change up to +19 %; all spatial and convergent with matched refinement |
| Spatial-K physical correctness (CAK) | **CONDITIONALLY SUPPORTED.** Flux across resolutions, passivity, reciprocity, linearity, correspondence, ownership and stable composition pass. Closure holds up to the leaked photons (the coded test failed on its own specification). Rigid invariance shows no excess over noise, with a formal low-side miss |
| Coupled GI accuracy | **NOT SUPPORTED at the full predeclared contract (PARTIAL).** CAK passes every face-on ruled group on 4 design scenes and the held-out scene (floor 18/18, bend90 21/21). Failures are oblique groups, mostly reference-limited, with one held-out interface residual |
| Near-contact and oblique fidelity | Near-contact face-on: **SUPPORTED** (CAK, design + held-out). Oblique: **INCONCLUSIVE** (reference not pixel-converged); the 1 cm readout omission is corrected by the ballistic term |
| Rigid reuse | **CONDITIONALLY SUPPORTED.** A contact-creating rigid move with the reused rest K and re-aggregation passes 18/18; the 1 cm reuse fails 6/18. Conditional on the formal rigid-test miss |
| Non-rigid K variation | **SUPPORTED (measured)** at 1 cm and CAK-cell aggregation; not resolvable at 2.5 mm cell level with 2048 photons per cell |
| Differential evaluation readiness | Detect **SUPPORTED** (aggregated); predict **SUPPORTED** (slope 0.92–1.08, corr ≥ 0.98); rank **NOT SUPPORTED** in general (bias floor). The refined interface makes the largest change (bend90 underside) rankable |
| Computational tractability | **NOT SUPPORTED under the predeclared rule** (recovery 0.63–0.64 < 0.8 on floor / bend90 at 8–10 % of the 2.5 mm cells). Per-iteration cost 1–2 ms, re-aggregation ≤ 1.4 s. The oracle (2 GB, 380 s per configuration) is not a runtime representation. Scaling **INCONCLUSIVE** |
| Learned-K integration readiness | **INCONCLUSIVE.** The evidence defines a target (sub-port K near contact, aggregation-consistent, plus a ballistic readout term) but nothing learned was trained or tested |

**Living-document consequence (no adoption).** The data justify recording the following as **evidence-backed contract requirements** for
any future FNTO K interface:
1. Near-contact (< ~1 cm) K/C boundary exchange must be resolved below the port at the scale of the gap and the diffusion length.
2. The camera readout must include the unscattered term.

CAK is **not adopted**. A representation-class choice (learned / compressed sub-port K) remains a user decision.

---

## Deviations, post hoc items and failures (all in the protocol)

- A1 (post hoc): COR check (iii) failed and was recorded. A COR-free gate was added; the outcome was unchanged. The pixel diagnostic, the
  nee + KB readout and entry to Stage C followed.
- A1.5: the dent operators were omitted from the first GPU chain by oversight and built afterwards.
- A2 (post hoc): the coded Stage D closure and rigid tests failed. The closure test's tolerance was mis-specified (a single leaked photon
  in a column); the diagnostic shows closure up to leaks. Rigid replicates show a low-side formal miss. The coded results are kept.
- A3 (post hoc): the pixel rule was extended to the held-out cameras before they were rendered.
- Implementation bugs found and fixed after the protocol commit:
  - the outgoing test passed a scalar level (crash, re-run);
  - the Stage A 0.25 mm replay lookup was keyed wrongly (0 comparisons, fixed, `replay_v2`);
  - bend90's first current-K arms predate the bin readout. They were re-composed with the same seeds (nee identical to 9e-9) and the old
    files kept (`*_v1nobin`).
- Before the protocol commit, a COR source-weight bug (each pool refill carried the full source flux) was found in a debug test pass and
  fixed (disclosed P3).
- The Cycles device changed mid-batch by user decision. 6 CPU renders are unused. Units calibration ran on the fork CPU. The emitter
  calibration extension (units2) is labelled.
- The oblique groups are reported in full, not hidden. Their failures dominate the coupled counts.

## REPRODUCIBILITY

**Commits.**

| Commit | Content |
|---|---|
| 0b98812 | Protocol and Stage A / B code, before evidence |
| ee0e66d | Launchers |
| 8b2d23f | Analysis, calibration extension |
| bbb8c4d | CUDA references, CAK arm, replay fix |
| e4971e6 | Outgoing-test fix |
| ccd95d2 | A1 |
| e818bfc | A2, Stage E |
| d5c2075 | A3 |
| 5e32a05 | Report and cost |
| This worklog's commit | Figures, results, living documents |

Each evidence run used committed code, except the figure script (committed here). The server copy records the commit in
`~/rest_wl55/repo_commit.txt`.

**Commands** (`experiments/fnto_k55/run.sh` on LabServer69):
- `k55_stageB.py {ops,sun,cor,compose,kf,btest} …`
- `k55_stageA.py {code,replay}`
- `k55_stageD.py {ops,diag}`
- `k55_stageE.py`
- `k55_analyze.py stageB`
- `k55_pixdiag.py`
- `k55_report.py {tables,cost}`
- `k55_figures.py`
- `tests/test_k55.py`
- Cycles: `blender -b --factory-startup --python-use-system-env --python k55_cycles.py -- <out> render <scene> <cams> <seeds> device=CUDA [px=]`
- Launchers: `launch_gpu_B.sh`, `launch_gpu_CD.sh`, `launch_cycles_cuda.sh`, `launch_cycles_A1.sh`.

**Seeds.**

| Use | Seeds |
|---|---|
| Cycles, full scenes | 5501 / 5502 |
| Cycles, known fields | 5511 / 5512 |
| Cycles, held out | 5521 / 5522 |
| Cycles, Stage E | 5531 / 5532 |
| Cycles, pixel diagnostic | 5541 / 5542 / 5543 |
| Units | 5509 / 5508 |
| TR 2.5 mm, slab | 5561 / 5562 |
| TR 2.5 mm, bend90 | 5563 / 5564 |
| TR 2.5 mm, dent | 5565 / 5566 |
| TR, direct 1 cm / Din 64 / 5 mm | 5541 / 5542 / 5543 / 5551 |
| COR | 5571 / 5572 (scenes), 5573 / 5574 (known fields) |
| C | 5581 / 5582 |
| Sun | 5591 + domain |
| Sun visibility | 5585 |
| CAK levels | 5801 |
| Known-field rays | 5601 |
| Stage D | 9101–9109 |

**Sample counts.**
- TR: 2048 photons per in-cell at every level.
- Sun: 16 M photons per domain.
- COR: 64 M photons.
- Known-field incident fields: 64 M rays.
- C: 128 rays per out-cell at 1 cm, scaled by area.
- KB: 4096 samples per port.

**Outputs.**
- `results/fnto_k55/`: `stageA/`, `tests/`, `stageD/`, `stageB/` (predictions npz + meta json per scene and arm), `analysis/`
  (`stageB.json`, `stageBD.json`, `stageE.json`, `pixdiag*.json`, `report_tables.json`, `cost.json`), `cycles/` (camera geometry, render
  manifests, logs), launcher step and failure logs, `sha256_manifest.txt`.
- Figures with a manifest: `results/evaluation/55/`.
- On LabServer69 (`~/rest_wl55`): `ops/*.pt` (operators), `out/cycles_*` (images), `out/stageB/*/sun.pt`.

**Independently reproduced here vs inherited.**
- Reproduced here: all WL54 numbers in Stage A; WL53's non-rigid ΔK magnitudes at 1 cm; WL53's composition ρ (gap8 0.7676 vs 0.767);
  WL53 tests.
- Inherited: the Cycles volume physical-validity results of WL53 (H-function, furnace, reciprocity).

No later batch is started or recommended.
