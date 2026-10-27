# Worklog — Physically Consistent Intrinsic-Transport Reference and Geometry-Dependent K Ownership (2026-10-28)

**Protocol.** `phys_ref53` (`experiments/phys_ref53/protocol/phys53.json`), committed with the code in ce9e46c before any
primary evidence; design revisions made during the smoke are disclosed there. Four amendments followed:
- **A1** (7a8258f): sample size only — the non-rigid operator differences at 16× photons. Rules unchanged.
- **A2** (2ae89de): post hoc fix of an implementation defect in the oracle numerics. Rules unchanged; the defective
  run is preserved.
- **A3** (20a3ddb) and **A4** (c1f541c): post hoc attribution diagnostics. Each interpretation rule was fixed before
  its run.

**Authorisation.** The user brief "ReST-GI — Worklog 53: Physically Consistent Intrinsic Transport Reference and
Geometry-Dependent K Ownership Validation" (2026-10-28). It is bounded, independent of WL52, and runs in parallel
with it.

**Isolation.**
- Branch `wl53_physical_reference`, in worktree `.claude/worktrees/wl53`.
- All code is in `experiments/phys_ref53`. No production FNTO module, K44 / DASQ checkpoint or WL52 file was imported,
  read or written.
- Historical Cycles references, worklogs 1–52 and all baselines are unchanged.

**Hardware.**
- **Smoke and renderer investigation:** LabServer63.
- **Primary runs:** LabServer69, on the user's instruction during the smoke.
  - RTX 3090 24 GB (driver 550.54) and 16 cores.
  - Mitsuba 3.5.1 / Dr.Jit 0.4.6 and torch 2.3.1+cu121 (venv `8dna26`).
  - Cycles 3.5.1 in docker `restgi/bpy-runtime:22.04`.
- **Sharing with WL52.** WL52 was training on the same GPU until 03:20 KST.
  - WL53 capped its torch memory at 0.35 of the GPU while WL52 ran, and at 0.8 after it finished.
  - All renders ran on the CPU.
  - The WL52 log records that one of its benchmark runs (chain 3) was contended by "an unrelated CPU-saturating
    Cycles render". That was this batch's reference render. WL52 re-ran that benchmark on an idle server; its protocol
    and files were not touched.

## Answer to the completion question

> Do we have a physically interpretable and independently validated K/C reference that can measure which intrinsic
> transport information remains reusable when geometry changes?

**Yes, at the operator level, for the tested fixtures** (index-matched isotropic homogeneous media; slab / L-block /
air-gap pair / bent / dented slab; Lambert floor).

**At the image level the composed K/C prediction is accurate only to about 5 %** on faces that receive most of their
light through free-space coupling. That floor comes from the explicit C and its 1 cm port discretization, not from K
(A4). It is larger than the image-level effect of a stale K under the tested deformations (≤ 2.3 %), so image
comparisons cannot rank stale against current K in these scenes. The operator comparison can.

What holds, in order:
1. **The reference is valid.** Cycles 3.5.1 volume path tracing (homogeneous RTE medium in an index-matched interface)
   passes every predeclared physical-validity rule:
   - exact H-function (24 cases, 0.9950–1.0005);
   - furnace (1.0000);
   - zero field and superposition (|bias| ≤ 1e-4);
   - camera-vertex vs diffuse- and mirror-ancestor consistency;
   - grouping invariance (identical);
   - reciprocity (0.998–1.000);
   - bounce-cap truncation ≤ 0.5 %.

   An analytic-geometry random walk (IW) with no renderer code independently matches the H-function (0.9992–1.0007) and
   the slab (≤ 0.9 %).
2. **Two renderer-implementation findings**, recorded as findings and not used as references:
   - **Mitsuba 3.5.1 (float32)** skips an ~89 µm boundary layer whenever a path crosses a null interface. At σ_t =
     1000 m⁻¹ this biases near-surface transport by up to −12 % (single scattering matches the prediction
     exp(−σ_t δ / μ0)).
   - **LabServer63's Mitsuba cuda film accumulation is wrong** for deterministic cases.
3. **The intrinsic K is a well-defined, path-history-independent operator.** It is the medium's response from entry to
   the first exit into air, with ports in material coordinates.
   - Its oracle (TR) matches the reference within 0.4 % on all lit / transmitted faces, after one numerical defect of
     ours was fixed (A2).
   - It conserves energy exactly at albedo 1, is passive and reciprocal within noise, and is unchanged by authoring-object
     grouping.
4. **K and C are disjoint by construction.** A control in which K and C both claim the air-gap paths diverges
   (ρ 4.75 / 2.92 / 0.83). Every partition composition converges (ρ 0.45–0.82).
5. **Rigid motion leaves K invariant within MC noise** (‖ΔK‖ / ‖seed noise‖ 1.00–1.05). Composition with the reused
   pose-0 K equals composition with a recomputed K to ≤ 0.05 %.
6. **Non-rigid deformation changes K measurably but by little** (port-pair Frobenius 0.6–1.9 %, at 16× photons).
   - The change is structured: a dent's change is local (91–100 % of its energy within the dent support) and low-rank
     (rank90 8–18); a bend's change is global and not low-rank (rank90 125–151).
   - Relative to their size, nonlocal entries change more than local ones (bend 90°: 1.6 % / 4.3 % / 7.5 % for the same
     port / adjacent ports / 1.5–3 cm, red, noise-subtracted).

---

## IMPLEMENTATION FACT

### Reference model decisions (protocol, before primary evidence)

| Model | Role | Reason |
|---|---|---|
| Cycles 3.5.1 volume (Volume Scatter + Volume Absorption inside a white Transparent BSDF) | **Primary reference** | RTE semantics; volume closures are not subject to Cycles' PATH_RAY_DIFFUSE_ANCESTOR rule (svm/closure.h 903–926 covers BSSRDF closures only); self-intersection by primitive id (no spawn offset); matched the exact H-function within 0.13 % before the commit |
| Mitsuba 3.5.1 volpath (`llvm_ad_rgb`) | Diagnostic second renderer | Decision R1: float32 surface spawn offset RayEpsilon·(1+\|p\|) = 8.9e-5 m skips a boundary layer at every null-interface crossing; bias ∝ σ_t (5.6 % / 0.57 % / 0.04 % at σ_t 1000 / 100 / 10 m⁻¹, albedo 0.5); identical in scalar / llvm / volpathmis / without RR; not reducible by scene scaling below ~2 % for these fixtures |
| Mitsuba `cuda_ad_rgb` | Ray intersector only | LabServer63 film accumulation defect (below). LabServer69 cuda renders are exact, but renders used llvm everywhere for uniformity |
| IW (analytic box random walk) | Independent estimator | No renderer, no mesh; float64 |
| Chandrasekhar H-function | Exact control | Semi-infinite isotropic medium; H(1) = 2.90781 at ω = 1 |
| TR (oracle K) | The measured object, never the reference | Single-domain RTE random walk using Mitsuba only for ray intersection |
| Cycles random-walk SSS (BSSRDF) | Diagnostic only | Path-history dependent by source |
| Refractive interfaces, Lego-scale assets, custom renderer, Mitsuba double build | Rejected | Uncontrolled extra hypothesis; not needed once Cycles volume passed the analytic control |

### Environment facts found in this batch (before the primary run)
- **LabServer63 `cuda_ad_rgb`**, run with the ~/nvlibs OptiX-595 workaround of WL43, renders a Lambert plane under the
  sun as follows:
  - the exact value is 0.12732 with zero variance;
  - it gives 0.1268 / 0.122 / 0.115 at 64 / 1024 / 8192 spp, with std 0.011–0.017 and unequal channels for a grey
    material;
  - an irradiancemeter under a constant sky gives π → 6.7.
- **The same scripts are exact** with llvm on LabServer63 and with cuda on LabServer69.
- **Intersection on LabServer63 cuda is correct:** 9e-8 m from an analytic box.
- **Not re-examined here:** earlier LabServer63 Mitsuba cuda renders made with that workaround (e.g. WL43).

### Physical contract (protocol "conventions")
- **Domain.** A connected region of one homogeneous medium inside a closed interface. Opaque Lambert surfaces are
  boundary-only domains.
- **Ports and correspondence.**
  - A port is one 1 cm voxel face of the rest configuration, tessellated as 4 × 4 sub-cells × 2 triangles. It keeps its
    id, triangles and rest tangent under every map, so correspondence is material.
  - The local frame is the current outward normal plus the affine image of the rest tangent.
- **Bins.** Shirley–Chiu concentric map plus Malley projection, so every bin has projected solid angle π / N²: 8 × 8
  outgoing, 4 × 4 incoming. An incoming direction is labelled by the direction toward its source.
- **K.** T_a[(i,o) ← (j,b)] is the fraction of the flux entering through (j,b) that leaves through (i,o) at its *first
  exit into air*. TR computes it on the isolated domain, so nothing outside the domain exists for TR.
- **KC.** The port-average radiance toward a camera direction, by next-event estimation at every collision.
- **C.** A ray cast through air from each out-cell to the in-cell it first enters. This includes exit and re-entry of
  the same domain; escapes leave the scene.
- **Ownership.** A light path is split at every interface crossing. Air segments belong to C; each in-medium sub-path
  from entry to first exit belongs to K. Neither depends on authoring-object grouping.
- **Composition.**
  - φ_in = φ_ext + C φ_out and φ_out = K φ_in, solved by fixed-point iteration to a relative change of 1e-7. The
    spectral radius comes from per-channel power iteration.
  - φ_ext is the unoccluded external light: light that first hits another boundary is that domain's input.
  - The external term uses the exact light direction (TR sun runs; exact Lambert irradiance); the coupled terms use
    bins.
- **Readout.** `nee` (primary) = L0 + KC φ_in_ind; `bin` = φ_out in the camera's 8 × 8 bin; Lambert = ρ/π (E_ext +
  E_ind).

### Fixtures
- **Medium:** σ_t 1000 m⁻¹ (mean free path 1 mm), albedo 0.999 / 0.99 / 0.9, isotropic phase function. Diffusion
  lengths 18.1 / 5.8 / 1.8 mm.
- **Geometry:**
  - slab 10 × 10 × 1 cm (240 ports);
  - L-block (270 ports, concave);
  - two slabs with a 4 mm or 8 mm air gap;
  - Lambert floor 30 × 30 cm (ρ 0.8);
  - rigid pose 1: 37° about (1, 2, 3); a far variant adds a translation of (10, −7, 3) m (stress, T only);
  - bend 30° / 90°: cylindrical, mid-surface length kept, the top face becomes concave;
  - dent: top surface pushed down by 5 mm · cos²(πr / 4 cm) for r < 2 cm.
- **Cameras:** orthographic, including cameras inside the gaps and under the slab.
- **Seeds and sample counts:** in the protocol. Cycles: 1024 spp at 1 mm pixels. TR: 2048 photons per input cell
  (32768 in A1).

### Code (`experiments/phys_ref53`)
- `p53_geom.py` — bins, voxel boundaries, maps, frames, H-function.
- `p53_mi.py` — Mitsuba IO and the intersector.
- `p53_tracer.py` — TR and IW.
- `p53_compose.py` — C, solve and readout.
- `p53_fixtures.py`.
- `p53_cycles.py` — the primary reference.
- `p53_render.py` — Mitsuba diagnostic, and Cycles port averages (`cports`).
- `p53_oracle.py` — oracle stages.
- `p53_report.py` — the predeclared rules.
- `p53_attrib.py` — post hoc.
- `p53_figures.py`.
- `tests/test_p53.py`.
- `p53_chain.sh` / `p53_chain2.sh`.

### Tests — 9 / 9 pass on LabServer69 (`results/phys_ref53/tests.json`)

| Test | Result |
|---|---|
| Radiometric units | Mitsuba Lambert 0.2205316 vs 0.2205316 |
| Coordinate transforms | Frames rotate with the body to 1e-12 |
| Boundary correspondence | Bent port areas 0.92140 / 1.07847 vs analytic 0.92146 / 1.07854 |
| Zero / superposition | — |
| Conservation | Albedo-1 L-block column sums exactly 1 |
| Reciprocity (C) | Form factor 0.69008 vs 0.69024; port-level asymmetry 0.070 vs cross-seed noise 0.069 |
| Ownership | Concave L-block same-domain air re-entry 3.1 %; convex 0 |
| Air gap | Partition K has no transfer to the other slab; the object-scoped control carries 15 % |
| Path-history independence | Solve = Neumann series with one operator, 1e-7 |
| Oracle composition | First bounce = form factor 0.69046 vs 0.69024 |

---

## MEASUREMENT

### Gate 1 — reference validity (Cycles volume; all predeclared rules PASS)

| Test | Result |
|---|---|
| Units chain | Lambert 1.0000003; fisheye irradiance of a uniform sky / π = 1.000005 |
| Semi-infinite vs exact H-function, 24 cases | Cycles 0.9950–1.0005 (max sem 0.09 %; the lowest values are albedo 0.999, the deep-path bounce cap); **IW 0.9992–1.0007**; Mitsuba (diagnostic) 0.878–1.033 |
| Furnace, albedo 1, uniform sky | 1.0000 (top, oblique); albedo W: 0.979 / 0.840 / 0.439 |
| Zero / superposition | Zero emitters → exactly 0; AB vs A + B bias ≤ 1e-4 (SE 3e-4) |
| Path history, slab lit at 45°, patch 5 cm above | E_diffuse-ancestor / E_camera-vertex (fisheye) = 0.9946 / 0.9946 / 0.9943 (SE 2e-4) = the Lambert control 0.9943 (patch-footprint geometry); mirror / direct 0.9999 |
| Path history, WL51 J_below analogue (gap lit from below) | 0.991 / 0.992 / 1.010 (SE 0.001–0.008); mirror 1.000 / 1.000 / 1.009 |
| Grouping (gap4, gap4_below; one object vs two) | Identical (bias 0.0000 on every face) |
| Reciprocity, total intensity I(c\|s) vs I(−s\|−c) | Reflection 0.9985 / 0.9984 / 0.9980 (SE 1e-4); transmission 1.0002 / 1.0001 / 1.0003 |
| Bounce cap, 512 vs 1024 (L-block) | ≤ 0.47 % |

**Diagnostics (not rules).**
- **Cycles random-walk SSS:**
  - diffuse-ancestor / camera-vertex = 1.082 / 1.074 / 0.991, against the control 0.994, so it is path-history
    dependent;
  - across the 4 mm gap it carries 1e-9–2e-5 against the volume's 0.08–12, i.e. it does not cross air;
  - grouping changes it by −27 % to −100 %, on a near-zero signal.
- **Mitsuba vs Cycles (isolated slab):**
  - top faces +1.2 % / +0.9 % / −1.5 %; transmitted faces +6.2 % / +6.3 % / +6.7 %, consistent with the skipped boundary
    layer;
  - Mitsuba's own furnace is 0.9997, its path history 1.0017, and its grouping identical;
  - its reciprocity point estimator is too noisy to inform (SE 0.02–0.13).

### Gate 2 — oracle K

**TR vs reference.** Values after A2; v1 (float32 accumulation) in brackets.

| Scene | top | bottom | oblique top | oblique bottom |
|---|---|---|---|---|
| iso_L1 (sun normal) | −0.04 / −0.06 / −0.11 % | +0.05 / +0.07 / 0.00 % | −0.04 / −0.05 / −0.13 % | +0.17 / +0.19 / +0.28 % |
| iso_L2 (sun 50°) | −0.06 / −0.06 / −0.09 % | −0.01 / −0.01 / +0.48 % | −0.06 / −0.07 / −0.14 % | −0.11 / −0.24 / −1.45 % |
| iso_L3 (2 × 2 cm beam) | +0.08 / +0.09 / +0.04 % [v1 −12.1 / −13.2 / −10.1 %] | +0.36 / +0.36 / +0.15 % | +0.92 / +0.86 / +0.59 % | +0.74 / +0.84 / +0.96 % |

- **IW vs reference** (iso_L1, iso_L3; top / bottom faces): ≤ 0.92 %. All pass.
- **The rule miss:** the one ruled group that fails is the 10-port −y side face in the oblique-bottom view, at
  −8.8 / −9.4 / −12.0 %. Post hoc, re-averaging the reference with centre-ray instead of interior-pixel classification
  gives +1.6 / +1.5 / +0.2 %. Interior pixels drop the port's edge rows, which matters under a steep within-port
  gradient. This is a limit of the comparison unit, not of K.
- **Defect A2 (our code).**
  - TR accumulated with float32 `index_add_`. In single-source sun runs about 1.9e8 next-event contributions land in a
    few port bins; increments of ~2^−22 of the running sum are lost.
  - Symptoms: lit ports read −9 % against IW, and the result depended on batch size.
  - The face-level exit fluxes of TR and IW still agreed to 4 decimals, i.e. T itself was unaffected.
  - Fixed with float64 accumulators (lit port 0.22584 vs IW 0.22562) and the oracle was re-run; v1 is preserved in
    `report_v1_float32acc.json`.
- **Discretisation floor (predeclared, no rule): the sun binned into 4 × 4 incoming bins.**
  - Transmitted faces: −3.3 to −3.9 % (R/G) and −1.6 to −8.0 % (B).
  - Lit faces: −2.4 to +2.1 %.
  - The 8 × 8 out-bin readout is worse (up to −15 % on oblique transmitted views, −32 % on the beam's bottom face).
- **Operator properties.**
  - Albedo 1: every column sums to 1 exactly.
  - Maximum column sum 0.9935 (passive).
  - Leaks plus wrong-side exits: 5.1e-7 to 8.9e-7 of photons.
  - Port-pair reciprocity: pooled asymmetry 2.7 / 2.6 / 2.9 %, with 0 % of entries at |z| > 4.
  - Direction-resolved reciprocity on adjacent pairs: 0.437 against cross-seed noise 0.436.
- **K under a uniform sky (post hoc, no C):** top +0.12 / +0.11 / +0.14 %; oblique top −0.7 / −0.8 / −1.5 %.

### Gate 3 — composition (nee readout; bias vs Cycles)

Every partition solve converges:

| Scene | ρ R / G / B | Iterations |
|---|---|---|
| gap4 | 0.816 / 0.738 / 0.452 | 63 |
| gap8 | 0.767 / 0.693 / 0.424 | — |
| floor | 0.724 / 0.693 / 0.521 | — |
| L-block | 0.732 / 0.696 / 0.521 | — |
| bends / dent | 0.63–0.72 | — |

| Group | Bias (R / G / B) | Coupling share (R) |
|---|---|---|
| Directly lit faces (all scenes) | ≤ 0.5 % (pass) | 0.03–0.08 |
| gap4 gap face, lower slab | −4.2 / −0.3 / +5.7 % | 0.93 |
| gap4 gap face, upper slab | −2.7 / +0.6 / +8.3 % | 0.61 |
| gap4 transmitted bottom | −4.5 / −0.5 / +3.5 % | 0.79 |
| gap8 gap faces | −2.5 / +0.1 / +3.4 % and −1.1 / +1.2 / +6.8 % | 0.56–0.85 |
| floor: floor under the slab | +3.9 / +4.5 / +5.2 % | 0.40 |
| floor: slab bottom | +2.6 / +3.5 / +4.3 % | 0.73 |
| L-block: concave-corner +x face | −4.2 / −3.9 / −2.8 % | — |
| L-block: bottom | +1.5 / +2.5 / +2.6 % (pass) | — |

The predeclared 3 % rule therefore fails on coupling-dominated groups in every scene. The v1 numbers are essentially
the same.

**Attribution (post hoc, each rule fixed before its run).**
- **A3 — 64 incoming bins.** The misses do not shrink: gap face B +5.7 → +5.0 %, floor under the slab +3.9 → +3.9 %.
  So they are not caused by incident-direction binning.
- **A4 — the floor scene with the slab replaced by an opaque Lambert box** (K exact). It reproduces the misses: box
  bottom +6.2 %, floor under the box +5.0 %, unchanged at 128 / 512 / 2048 C rays per cell.
- **Conclusion:** the coupled-face error belongs to the explicit C / 1 cm port spatial discretization at 4 mm–1 cm
  gaps, not to the medium K and not to the ray count.

**Disjointness power.** The 'dup' control (K continues the gap exit *and* records it) diverges: ρ 4.75 / 2.92 / 0.83,
not converged after 2000 iterations.

**Object-scoped consistent arm** (pair K absorbs all gap re-entries; full C):
- converges (ρ 0.14);
- matches the top face (+0.7 %);
- its gap faces carry no exitance (−99.8 %) by construction, while a camera inside the gap sees 0.08–12 W m⁻² sr⁻¹;
- the stale pair K at gap 8 mm is in `pred/gap8__pair4stale_full`.

### Gate 4 — rigid reuse
- **K at pose 1 vs pose 0**, ‖ΔK‖ / ‖ΔK_seed‖: 1.001 / 1.017 / 1.053 at the port-pair level, 0.9995 for the full
  direction-resolved T. Inside [0.85, 1.15], PASS.
- **Composition at pose 1**, reused vs recomputed K: |Δbias| ≤ 0.05 % on every group (PASS).
- **The concave-corner +x face** misses the 3 % accuracy rule in both arms (−3.7 / −3.5 / −2.4 % and
  −3.8 / −3.5 / −2.4 %), so the G4 rule as a whole is not met; the miss is G3's composition limitation.
- **Far translation, 10 m (stress):** ‖ΔK‖ / seed = 1.39 / 1.53 / 2.11, B mean −1.1 %, leaks + wrong-side 1.3e-4.
  This is the float32 limit at |p| = 12 m.

### Gate 5 — non-rigid K variation

At 2048 photons per cell, ‖ΔK‖ / ‖noise‖ is 1.20 / 1.22 / 1.15 (bend 30°), 2.21 / 2.26 / 1.95 (bend 90°) and
1.91 / 1.94 / 1.37 (dent). That is significant but under the magnitude rule of 3. Results at 32768 photons per cell
(A1):

| | Bend 30° | Bend 90° | Dent |
|---|---|---|---|
| ‖ΔK‖ / ‖noise‖ | 2.81 / 2.89 / 2.49 | **7.95 / 8.18 / 6.86** | **6.29 / 6.42 / 3.53** |
| Port-pair ‖ΔK‖ / ‖K‖ | 0.65 / 0.62 / 0.54 % | 1.85 / 1.76 / 1.49 % | 1.46 / 1.38 / 0.77 % |
| Noise-subtracted relative change, same port / adjacent / 1.5–3 cm (red) | 0.5 / 1.4 / 2.6 % | 1.6 / 4.3 / 7.5 % | 0.5 / 6.0 / 9.0 % |
| Share of ‖ΔK‖² on the same face | 0.76–0.80 | 0.76–0.78 | 0.17–0.41 |
| Share within the dent support (r < 2 cm + 3 L_d) | — | — | 1.00 / 0.98 / 0.91 |
| rank90 (rank above the noise top singular value) | 125–135 (71–124) | 133–151 (193–200) | **8 / 8 / 18** (21–29) |
| Maximum column-sum (albedo) change | 0.03 / 0.24 / 0.69 % | 0.06 / 0.47 / 1.35 % | 0.54 / 3.66 / 3.87 % |
| Area-only correspondence model, share explained | < 0 | < 0 | < 0 |
| Port area change | ±2.6 % | ±7.9 % | +0–5.5 % |
| Same-domain air re-entry under uniform exitance (the C side; rest 0) | 0.34 % | 3.1 % | 0.19 % |

- **Image-level stale effect** (stale rest K vs current K, bin readout, vs Cycles):
  - bend 90° slab bottom: −1.2 % vs +1.1 %; bend 90° top: +1.2 % vs +0.3 %;
  - dent slab bottom: +2.7 % vs +2.3 %.
- **Separation vs ranking:** every difference exceeds 3 SE, but all are inside the ~5 % C / port floor of G3. So stale
  and current differ measurably, but at image level neither can be ranked as the more accurate.

### Costs (LabServer69)

| Item | Cost |
|---|---|
| TR, 7.9 M photons (≈ 1.3e8 collisions, 4 camera directions) | 70 s |
| TR, L-block (8 camera directions) | 330 s |
| TR, 1.26e8 photons (no camera operators) | 106 s |
| Peak GPU memory | 8.7 GB |
| Composition | ~1 min per arm |
| Cycles reference set | 3.3 h on 16 CPU cores |
| Mitsuba diagnostic | 67 min CPU |

---

## OBSERVATION
- **Path-history independence holds** for the volume reference and for the K/C composition. It fails for Cycles BSSRDF
  closures (Burley in WL51, random walk here), which is a renderer semantics choice and not physics.
- **Authoring-object grouping is irrelevant** for the volume reference (bit-identical). Random-walk SSS does not cross
  air either, but its ancestor rule changes its exitance.
- **K is not sparse or low-rank under deformation in general.** A local dent gives a local, low-rank change; a global
  bend gives a dense, high-rank change that is small per entry. In both, nonlocal entries change more, relative to
  their size, than local ones. Most of the change cannot be explained by port-area correspondence.
- **The free-space side (C) changes more under bending than K does.** Same-domain re-entry goes from 0 to 3.1 % at 90°,
  while K changes by 1.8 %.
- **The dominant composition error is C's discretisation**, at 1 cm ports and gaps of the same size. It is not
  ownership: the Lambert control shows the same miss with no medium involved.

## INTERPRETATION
- **Physical-contract findings.**
  - A path-history-independent intrinsic K exists and is measurable when the reference implements volumetric transport
    (RTE).
  - The WL51 inconsistency belongs to Cycles' BSSRDF ancestor rule, not to the K/C factorization.
  - Disjoint ownership (air → C; in-medium → K) composes stably. Double-claiming a path family produces a ρ > 1
    recurrence, the same failure mode WL50/51 showed for a different cause.
  - Rigid reuse of K in material coordinates is exact up to noise.
  - Deformation changes K by small, structured, measurable amounts; locality depends on the deformation.
- **Renderer-implementation findings:**
  - the Mitsuba float32 boundary-layer bias, which is large for dense media;
  - Cycles' BSSRDF ancestor rule;
  - LabServer63's cuda film defect.
- **Numerical limitations:**
  - our float32 accumulation (fixed);
  - the 4 × 4 incoming-bin floor for single-sun input (−4 to −8 % on transmitted faces);
  - the 1 cm-port C discretisation (≈ 5 % on coupled faces);
  - the interior-pixel comparison unit on steep side faces;
  - float32 at 10 m translation (leaks 1.3e-4).
- **Scope.** Index-matched isotropic media, voxel fixtures and one optical density. Refractive interfaces, anisotropic
  phase functions and Lego-scale geometry were not tested. Nothing here validates a learned K, real-scene fidelity or
  real-time GI.

## UNRESOLVED QUESTION (consequential)
1. **C discretisation.** What port size or position resolution makes explicit C accurate to the 1 % level at
   near-contact gaps? Coupled faces are about 5 % off at 1 cm ports. This currently caps image-level evaluation of
   stale K.
2. **Learning ΔK.** Is the non-rigid ΔK learnable or predictable from local geometry? A dent's change is low-rank and
   local; a bend's is dense.
3. **Untested directional media.** How large is the direction-selective share for refractive / anisotropic media,
   where WL47 showed that collapsing incoming light to irradiance loses information? This was not tested.
4. **Self-contact.** Not tested: the medium-boundary semantics at contact were not made unambiguous within this batch.

## ARCHITECTURE VERDICT (measured conditions only: protocol phys_ref53 + A1–A4, fixtures above, LabServer69)

| Item | Verdict |
|---|---|
| Reference physical validity | **SUPPORTED.** Cycles volume passes every predeclared rule; IW independent; Mitsuba float32 bias attributed and excluded |
| Intrinsic K ownership | **CONDITIONALLY SUPPORTED.** TR matches the reference within 0.4 % on all main faces, with exact conservation, passivity, reciprocity and grouping invariance; the remaining ruled miss (one side-face group) is a comparison-unit effect (post hoc); conditional on the A2 defect fix |
| K/C path disjointness | **INCONCLUSIVE under the predeclared mapping.** The partition arms miss the 3 % rule; the post hoc A4 control shows the miss is C / port discretisation, also present with no medium. Disjointness is supported by construction and by the double-claim control (ρ 4.75, diverges); this is post hoc evidence, not the predeclared test |
| Coupled transport correctness | **NOT SUPPORTED at the 3 % tolerance** with 1 cm ports. Every solve converges (ρ < 1); coupling-dominated faces are 3–8 % off, attributed post hoc (A3 / A4) to explicit C / port discretisation, not to K |
| Rigid-domain reuse | **CONDITIONALLY SUPPORTED.** K invariance (1.00–1.05 × noise) and reused = recomputed composition (≤ 0.05 %) pass; the G4 rule as a whole is not met only through the inherited composition miss on one group; float32 limit at 10 m |
| Non-rigid K variation | **SUPPORTED (measured)** at the A1 photon count (bend 90° 7.9×, dent 6.3× noise; bend 30° 2.8× just under the rule); the structure is reported; the image-level stale effect is measurable but below the composition floor |
| Readiness for future learned-K training | **CONDITIONALLY SUPPORTED.** An operator-level K / ΔK reference and oracle with validated physics, material correspondence and noise estimates exist for these fixtures; image-level supervision through composed C at 1 cm ports would carry ≈ 5 % coupled error; no refractive / anisotropic / Lego-scale evidence |

## REPRODUCIBILITY

**Commits.**

| Commit | Content |
|---|---|
| ce9e46c | Protocol and code, before evidence |
| f0fa24e | LLVM path on LabServer69 |
| 7a8258f | A1 |
| 2ae89de | A2 |
| 20a3ddb | A3 |
| c1f541c | A4 |
| This worklog commit | Results and figures |

Server copies were made by tar; `sha256_*.txt` in `results/phys_ref53` matches the local files.

**Chains.**
- `p53_chain.sh`: tests → GPU oracle ∥ CPU branch (Cycles → Mitsuba diagnostic → cports) → report.
- `p53_chain2.sh`: the A2 oracle rerun, then the report.
- A1 / A3 launchers live in `~/rest_wl53/launcher`.
- Status files: `~/rest_wl53/launcher/p53_*.status`.

**Commands.**
- `P53_MI_VARIANT=cuda_ad_rgb python p53_oracle.py {tr,g1a_iw,g2,compose,ops,tr_hi,ops_hi,lamfloor} <out>`
- `P53_MI_VARIANT=llvm_ad_rgb python p53_render.py {all,cports} <out>`
- `docker run … restgi/bpy-runtime:22.04 <relightable venv python> p53_cycles.py -- <out> [stages]`
- `python p53_report.py <results>` and `p53_attrib.py`.

**Wall time.**
- Oracle v1: tr 1171 s, g2 213 s, compose 1461 s.
- Oracle v2: tr 1434 s, g2 508 s, compose 1456 s, tr_hi 354 s.
- Cycles: 11891 s. Mitsuba: 4005 s.

**Outputs.**
- `results/phys_ref53/`:
  - `report.json` (v2) and `report_v1_float32acc.json`;
  - `attrib_posthoc.json`, `amend4_lamfloor.txt`, `posthoc_K_diffuse_sky.txt`;
  - `oracle/` — g2 / ops / ops_hi / preds / meta;
  - `oracle_v1/`;
  - `oracle_din64/compare_din16_din64.json`;
  - `cycles/` — c_*.json, camera definitions, log;
  - `render/` — port averages;
  - `tests.json`, `p53_times.log`.
- The T matrices (≈ 25 GB) and the images stay on LabServer69 (`~/rest_wl53/results`).
- `results/evaluation/53/` (manifest with sha256):
  - `01_semi_infinite_vs_analytic.png`;
  - `02_slab_beam_profile.png`;
  - `03_composition_error_maps.png` — one absolute sqrt scale;
  - `04_nonrigid_operator_change.png`.

**Seeds.** Listed in the protocol. Renders 5301 / 5302 (point measurements 5301–5308); TR 5311–5323; C 5340 / 5350.

No later batch is started or recommended.
