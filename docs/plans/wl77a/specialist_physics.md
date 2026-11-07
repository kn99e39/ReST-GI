# WL77A Specialist Report — Agent 1: Transport Physics (2026-11-07)

**Status: COMPLETED**

**Scope.**
- Read-only on the repository except this file.
- No GPU job, no training, no server access.
- Every new number below comes from an independent, **non-evidence** CPU Monte Carlo (MC) or an analytic calculation. Each is labelled.
- Historical worklogs, protocols and results are cited, never changed.

## 0. Assigned role and research question

- **Role.** Agent 1 (transport physics) in the WL77A multi-agent architecture-discovery workflow.
- **Research question.** What must be represented explicitly, and what can be learned as a shared operator, so that a learned intrinsic first-exit operator K stays accurate when the **source port changes**? The emphasis is on independent, previously unseen source ports near boundaries and far from supervised sources.
- **This report's angle.** Physical invariants of the fixture's first-exit transport:
  - operator decompositions;
  - reciprocity under this flux/bin convention;
  - symmetry;
  - boundary context;
  - angular/spatial coupling;
  - the resolution any transferable intermediate must keep.

## 1. Input sources read

**Code** (verified at HEAD e2c7dd3):
- `experiments/phys_ref53/p53_geom.py`: bins and frames. The incoming bin is labelled by the outward direction −d; the concentric map gives each bin the same projected solid angle; `bin_parent` nests 64 bins into 16. Also the analytic H-function.
- `experiments/phys_ref53/p53_tracer.py`: `cell_source`, `phase_sample`.
- `experiments/phys_ref53/p53_oracle.py:154-184`: the WL53 reciprocity test.
- `experiments/fnto_rule57/r57_fastwalk.py` (the walker) and `r57_geometry.py` (port features, line 58: absolute rest coordinates `xyz0/.05` and `port_face` are node inputs; ballistic quadrature q = 256).
- `experiments/fnto_repr68/eval68.py` and `family68.py` (metric spaces and ATO algebra).
- `experiments/fnto_fia76/fia76.py:65-89`: `quadrature`, q samples per input cell with **one** isotropic exit ray each, giving SS weight c·e/q (line 79).
- `experiments/fnto_fia76/fia76.py:143-152`: `quad_ports`.
- `experiments/fnto_fia76/diag/posthoc76.py:28-33` and `experiments/fnto_fia77/gate77.py:45-52`: how the multiple-scatter part M and its "noise floor" are defined.
- `experiments/fnto_cmp72/rsk72.py` (RSK features).
- `experiments/fnto_src73/fit73.py` (S16-R population).
- `experiments/fnto_ato74/eval74.py:56` (BND4_FAR definition).

**Results:**
- `results/fnto_fia76/diag_posthoc76.json`, `results/fnto_fia77/stage1_expressivity.json`.
- `results/fnto_src73/ports.json` (S16-R new boundary sources).
- `results/fnto_ato74/ports4.json` (WL74/WL75 Stage-4 population).
- The WL77A Gate-4 confirmation population (`ports76.json`) is **not used** in any argument. Disclosure: its geometry metadata for the first five slots (positions and contexts only) was printed once while I was locating port lists. No reference or score of that population was read.

**Documents:**
- Worklogs 53, 74A, 75, 76, 77 and 77A in full or in the relevant sections; 73 for group definitions.
- `docs/Architecture.md`, `docs/RESEARCH_CENTRIC_TOPIC.md`.

**Non-evidence MC (this report).**
- **Implementation.** An independent numpy re-implementation of the rest slab. Its geometry, frames (the WL74A table), cell source and concentric bins are copied from p53. It is an analog albedo-1 walk with per-channel weight albedo^k, plus a collision ("expected-exit") estimator.
- **Runs.** 2,000,000 photons for each of five source cells:

  | Source | Face, port | Incoming bin |
  |---|---|---|
  | Interior top | (20,20) | 5 (near-normal) |
  | Interior top | (20,20) | 0 (grazing corner) |
  | Top-edge row | (39,20) | 5 |
  | +x side face | (20,2) | 5 |
  | Neighbour | (21,20) | 5 |

- **Scripts** (scratchpad, not repository): `mc.py`, `ana.py`, `quadss.py`, `contrast2.py`, `azgate.py`, `recip.py` under `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\phys77a\`.
- **Sanity checks:**
  - Analog and collision estimators agree: collided exit 0.9777 vs 0.9790 (red); bias-corrected native M difference 0.000.
  - The single-scatter exit at bin 5 is 0.1598 against the analytic semi-infinite (ω/2)[1 − μ₀ ln((1+μ₀)/μ₀)] = 0.159 at μ₀ = cos 19.6°.
  - The half-split native noise of the mean is 2.1–2.7 %.

## 2. Physical scales of the fixture

Analytic quantities, isotropic scattering with σ_t = 1 mm⁻¹. k solves (ω/2k) ln((1+k)/(1−k)) = 1. The lowest slab mode uses extrapolation length 0.7104 mfp, so d_eff = 11.42 mm.

| | Red ω = 0.999 | Green 0.99 | Blue 0.9 |
|---|---|---|---|
| Asymptotic transport length 1/k (exact; diffusion) | 18.26 mm (18.26) | 5.80 mm (5.77) | 1.90 mm (1.83) |
| Lateral decay length of the lowest slab mode, 1/√(k² + (π/d_eff)²) | **3.57 mm** | **3.08 mm** | **1.69 mm** |
| Semi-infinite SS albedo at normal incidence, (ω/2)(1 − ln 2) | 0.153 | 0.152 | 0.138 |

**Non-evidence MC, rest slab, 2M photons per cell.** Values are R / G / B; "wcoll" is the albedo-weighted mean collision count.

| Quantity | Interior bin 5 | Interior bin 0 | Top-edge row bin 5 | Side face bin 5 |
|---|---|---|---|---|
| Collided exit (= 1 − β − absorbed) | .978 / .830 / .430 | .983 / .866 / .516 | .907 / .841 / .545 | .985 / .874 / .471 |
| Single-scatter share of collided exit | .163 / .191 / .335 | .239 / .269 / .410 | .251 / .268 / .376 | .166 / .185 / .313 |
| Mean weighted collisions | 22.3 / 16.8 / 5.1 | 17.4 / 13.3 / 4.4 | 8.6 / 7.3 / 3.3 | 15.2 / 12.4 / 4.8 |
| Exit through the source port | .32 / .36 / .53 | .36 / .39 / .53 | .31 / .32 / .40 | .31 / .34 / .49 |
| Exit within 3.6 mm (port + 8 neighbours) | .66 / .73 / .93 | .73 / .79 / .94 | .77 / .80 / .89 | .64 / .68 / .84 |
| Exit within 15 mm | .965 / .982 / 1.00 | .973 / .986 / 1.00 | .991 / .995 / 1.00 | .982 / .990 / 1.00 |
| Exit within 25 mm | .998 / .999 / 1.00 | .998 / .999 / 1.00 | .9995 / … | .999 / … |
| Exit through the opposite face | .136 / .098 / .012 | .097 / .066 / .006 | .032 / .023 / .003 | .085 / .067 / .016 (bottom) |
| Exit through the adjacent side face | — | — | **.41 / .41 / .40** | (top .20 / .18 / .11) |

**Incidence footprint per incoming bin** (analytic MC of `bin_sample`). The first collision sits at a mean depth of about cos θ · 1 mm and is shifted laterally by about sin θ · 1 mm:
- inner bins 5/6/9/10: θ̄ 19.6°, depth 0.93 mm, shift 0.30 mm;
- outer mid-side bins: θ̄ 50.8°, depth 0.61 mm, shift 0.74 mm;
- outer corner bins 0/3/12/15: θ̄ 58.9°, depth 0.50 mm, shift 0.82 mm.

All of these are sub-port (2.5 mm) quantities.

**Consequences:**
- The red channel's natural 18 mm diffusion length is cut to an effective lateral reach of about 3.6 mm by leakage through both faces of the 10 mm slab.
- 97–99 % of every source's collided exit lands within 15 mm, and ≥ 99.8 % within 25 mm.
- Transport is local on the 1–2 cm scale, and **source-specific boundary information decays within about 15–25 mm** (red), less for green and blue.
- Order truncation is impossible: red needs O(20) collisions on average, with a heavy tail. Any representation must be a **resolvent** (collision density or diffusion-like), not a truncated Neumann series.

## 3. Exact operator structure (derivation; valid for this fixture)

### 3.1 First-exit semantics

- The boundary is index-matched, so no Fresnel internal reflection occurs. A photon leaves K's ownership at its first boundary crossing, and C owns any re-entry (`r57_fastwalk.py` has no air continuation).
- The rest slab is convex, so no exit/re-entry path exists. **Dented poses are not convex**: the valley-shaped top surface means the region below it is not convex. Straight segments between medium points on opposite sides of a dent can pass through air; those paths belong to C.
- Any intra-medium kernel must respect **in-medium visibility**, which is geometry-only.

### 3.2 Collision-density (Green's-function) factorization

Notation:
- X = (i, a) is an input cell: port i, incoming bin a, étendue g_X = A_i π/16.
- Y = (j, o) is an output cell.
- P_Y(x) = (1/4π) ∫ e^{−σ_t t(x,ω)} 1[ray from x exits through j in bin o] dω is the exact escape kernel.

The unit-flux cell source produces the first-collision density c₁ = (4π σ_t / g_X) · P_X. This holds because the incoming and outgoing bins carry the same outward label (§4) and both are attenuated straight-line kernels. A collision at x re-emits isotropically with weight ω. The next collision follows the symmetric kernel S(x,y) = σ_t e^{−σ_t|x−y|}/(4π|x−y|²) · vis_medium(x,y). Therefore

    K_scat(X → Y) = (4π σ_t ω / g_X) · ⟨ P_Y , G P_X ⟩,     G = Σ_{n≥0} (ωS)^n = (I − ωS)^{-1}

- The n = 0 term is exactly the single-scatter term SS. FIA's E1 is equivalent.
- The n ≥ 1 terms are M.
- B (ballistic) is separate, exact and geometric.

**What this shows exactly (isotropic phase, homogeneous medium, index-matched):**
1. **All incoming-angle dependence** of K_scat enters only through P_X: where, at sub-port resolution, the first collisions land.
2. **All outgoing-angle dependence** enters only through P_Y, the exact attenuated exit projection of a **scalar** collision-density field over about 1–3 mfp beneath the receiver.
3. **The only non-trivial transport object is the scalar volumetric resolvent G(x, y).** It has no direction, no source identity and no bins. It is symmetric, elementwise ≥ 0 and positive-definite, because S is a PD kernel and ωS has spectrum in (0, ω). Its only geometry dependence is the medium's shape: leakage at faces, edges and dents, and in-medium visibility.
4. **G inherits every isometry of the medium.** In the flat interior of the rest slab it depends only on (|ρ_x − ρ_y|, z_x, z_y). Near an edge it also depends on the two face distances. In a dent it depends on local thickness and shape.

### 3.3 Relation to the tested families

**FIA (WL76).** FIA kept P_X exact: SS and W are geometric quadratures. But its learned R_θ(anchor → j, o) learned **P_out ∘ G together**, so the exact exit projection's angular structure had to be learned through ring lobes. The factorization above puts the learned part on the direction-free G and keeps both angular ends exact.

**ATO / W+UV (WL59–WL75).** These learn the angular structure through global softmax atoms A (incoming) and O (outgoing) with a direction-free T. Physics allows a direction-free T, but:
- it ties angle to space only through the shared index r (WL74A P5);
- it does not represent the incidence-dependent footprint shift, which by §3.2 is exact geometry;
- it does not tie A to O (§4).

### 3.4 What breaks the factorization (scope limits)

Each of the following makes G depend on direction, so angular moments of G are then needed. None occurs in this fixture.
- An anisotropic phase function (g ≠ 0).
- Fresnel or refractive boundaries: internal reflection couples the boundary layer back into the medium. Reciprocity survives, but P is no longer pure attenuation.
- Heterogeneous media.
- Surface or BSDF scattering.

## 4. Reciprocity for this flux/bin convention

**Exact statement.**
- The incoming bin is labelled by the outward direction the light comes from (`p53_geom.py` header line 11–12; `cell_source`, `p53_tracer.py:90-101`).
- The outgoing bin is labelled by the outward exit direction.
- Both use the same frame per triangle and the same concentric map. Each 16-bin cell has projected solid angle π/16, and the 64-bin outgoing grid nests into it (`bin_parent`).

So, with **no label reversal**:

    A_i · K16_scat(i, a → j, b) = A_j · K16_scat(j, b → i, a)      (K16 = 64 → 16 outgoing aggregation; same for B)

- **WL53 check.** WL53 tested exactly this with `T5[J,I].transpose(2,3)` (`p53_oracle.py:168-181`). Port-pair asymmetry was 2.7 / 2.6 / 2.9 % with 0 % |z| > 4. The direction-resolved asymmetry on adjacent pairs was at noise (0.437 vs 0.436), on 1 cm ports.
- **This report's check** (non-evidence MC, 2.5 mm adjacent top ports, scattered part):
  - K(p,5→q,5)/K(q,5→p,5) = 0.995 / 0.994 / 0.993 (half-split SE ≈ 0.002–0.003; that SE estimate has 1 degree of freedom and is itself crude).
  - K(p,0→q,5)/K(q,5→p,0) = 0.991 / 0.992 / 0.989 (SE ≈ 0.009).
  - **Control.** Swapping only the bins, K(p,0→q,5)/K(p,5→q,0), gives 2.5 / 2.6 / 3.5. The constraint is far from trivial.

**Consequences for K (exact).**
1. The étendue-weighted scattered operator g_X K16_scat(X→Y) = 4πσ_t ω ⟨P_Y, G P_X⟩ is **symmetric and positive semi-definite**, because it is a Gram matrix of the P_X under the PD form G. So is its multiple-scatter part, since G − I = ωSG is PD.
   - **Corollary (Cauchy–Schwarz):** g_X K(X→Y) ≤ √(g_X K(X→X) · g_Y K(Y→Y)). A transfer is bounded by the local self-responses of its two ends.
   - B is reciprocal but not PSD.
2. **Contrast and outgoing structure are the same physical object.** The incoming-bin dependence of an unseen source j, seen at receiver i, equals (up to A_i/A_j) the 16-bin outgoing distribution at receiver j produced by source i. Outgoing angular data are observed at every receiver of every supervised source. A reciprocal parametrization therefore turns abundant outgoing observations into incoming-dependence (contrast) supervision.
   - This only adds information within transport range (about 15–25 mm) of supervised sources. It does not by itself supply far-boundary information.
3. **No tested family enforces it.** ATO's global A (16 × M simplex) and O (M × 64 simplex) are independent, and T[i,j,r] is not symmetrized. FIA's anchor W (second collision) and R_θ (exit) are not adjoint. A reciprocal family (K = D_g⁻¹ Πᵀ Γ Π with one geometric Π) would satisfy (1) and (2) by construction.

## 5. Symmetry and boundary context

### 5.1 Point group

- **Rest slab:** D4h, order 16. WL74A §3.3 shows that 12 of the 16 operations induce face-dependent local D4 bin permutations, all exact.
- **Central dents** (te0, te3; rest-like): C4v. The dent breaks z-flip.
- **Off-centre dents:** pt0 (5, −5) mm lies on the anti-diagonal and keeps one mirror. tr06 keeps one mirror. te1 and te2 keep none.

### 5.2 Translation (the larger, unexploited group)

- Faces are flat and ports form an exact 2.5 mm lattice with identical frames along a face. So K is **exactly translation-equivariant** along each face, and along edges, wherever both the source and its transport range (§2: 15–25 mm red) avoid corners and dents.
- An edge-row source 20 mm or more from a corner is the same physical problem anywhere along that edge, up to the tail fraction calculated in §2.
- The 1 cm coarse metric grid is **not** translation-invariant at sub-cell level. That is a property of the metric, not the physics. A native-resolution equivariant operator reproduces every coarse aggregate automatically.

### 5.3 Exact symmetry relation of BND4_FAR to supervised sources

Sources:
- `results/fnto_ato74/ports4.json`, `results/fnto_src73/ports.json`;
- the group definition `eval74.py:56`, which takes the far strata of TBE / TBC / SM / SR.

| BND4_FAR slot | Position (mm) | Symmetry image | Supervised S16-R source (rest and pt0) |
|---|---|---|---|
| s101 TBE | (−48.75, −8.75, −5) | z-flip + x-mirror → (48.75, −8.75, 5) | s50 = port 3439 at (48.75, −31.25, 5): same edge row, translated 22.5 mm along the edge |
| s105 SM | (−50, 1.25, −1.25) | z-flip → (−50, 1.25, 1.25) | s52 = port 408 at (−50, 31.25, 1.25): translated 30 mm along the side face |
| s107 SR | (−8.75, −50, −3.75) | z-flip → (−8.75, −50, 3.75) | s53 = port 527 at (−31.25, −50, 3.75): translated 22.5 mm |
| s103 TBC | (−41.25, −43.75, −5), 6.25 / 8.75 mm from the two edges | — | **not** a translate of the supervised corner port s51 |

- The translations are multiples of the port pitch, and the frames are identical.
- The supervised partners sit 18.75 mm from their nearest corner. The pt0 dent (radius 22.5 mm at (5, −5)) is about 58 mm away.
- From §2, the corner can affect at most the ~1–2 % (red) of exit flux beyond 15–19 mm, and less than 0.5 % for green and blue.
- **Inference (estimate, not measured):** the reference responses of s101, s105 and s107 equal translated and reflected copies of supervised references to within about 1 % native.

**What the learned models did** (WL75 per-source table):
- SR-far s107 scores con_coarse 0.99 (AR) and 1.19 (A75), against 0.20–0.31 for the per-source oracles. On the supervised side, the TRAIN_NEW group (which contains s53) fits at con_coarse 0.270 (AR) and 0.131 (A75); per-source training values were not looked up.
- AR / W+UV take **absolute rest coordinates and the face id** as node inputs (`r57_geometry.py:58`).
- AR's symmetry loss uses angle-integrated point-group images only (WL74A P11). No model enforces translation equivariance.

**Physics conclusion.** For three of the four BND4_FAR sources in the rest pose, the far-boundary transfer failure is **not** missing physical information. It is a failure to represent an exact symmetry that maps them onto supervised sources. Only the corner-offset slot s103 needs genuine extrapolation in corner-relative coordinates.

Caveat: WL72's RSK (position-free descriptors) was trained on the pre-WL73 S16 population. I did not verify whether that population contained edge-row sources of each class. WL72's negative result therefore does not contradict this inference.

### 5.4 Boundary context

- **Edge row (top, bin 5):**
  - 41 % of the collided exit leaves through the side face;
  - β ≈ 0.09 from near-normal rays leaving through the side;
  - weighted collisions fall from 22 to 8.6 (red);
  - more compact: 99.1 % within 15 mm.
- **Edge influence on an interior source** is confined to the edge-distance range where the source's exit flux would cross the edge (§2 radii).
- **WL74A §4.5:** edge outgoing shapes are 85–90 % inside the cone of interior shapes. The genuinely edge-specific content is the per-incoming-azimuth side-exit fraction (0.25–0.41 at an edge port, 0.37–0.56 at a corner). That is a spatial × incidence effect, which in §3.2 is the geometric P_X near two faces.
- **Dents (pt0):** depth 5.75 mm leaves 4.25 mm of thickness at the centre. That changes G (transmission, lateral confinement) across the dent footprint plus about one lateral decay length (1.7–3.6 mm). G must be conditioned on local thickness and shape: the `thick` and normal features exist in `r57_geometry.py`.

## 6. Angular structure re-examined: the WL76 / WL77 "floors" are dominated by single-scatter quadrature noise (non-evidence emulation)

### 6.1 Code facts (verified)

- **The WL77 Stage-1 statistic** (`gate77.py:45-52`) and **the WL76 post hoc ring floor** (`posthoc76.py:28-33`) both define the multiple-scatter part as M = mean(S) − ω · SS_q. SS_q is the deployable q = 1024 quadrature (`fia76.py:65-89`; `quadrature_q: 1024`, `fia76_v1.json:180`, `fia77_v1.json:30`).
  - Each input cell gets **1024 first-collision samples with one isotropic exit ray each**, spread over roughly 10² port × 64-bin entries.
  - SS_q is therefore an unbiased but **very noisy** estimate per entry.
- **The reported "noise floor"** (0.0016–0.0019) is the half-split of S only: `half=(S[:K//2].mean(0)-S[K//2:].mean(0))/2`. It **excludes the SS_q sampling variance.**
- **WL76's Stage-1 check** of SS_q against the MC single-scatter tally passes z² tests. Those test unbiasedness, not variance.

### 6.2 Emulation

Non-evidence. `quadss.py` and `azgate.py` re-implement the fia76 SS quadrature exactly (truncated-exponential first collision on the entry chord, one isotropic ray, weight c·e/q) on the MC slab. The exact SS is the 2M-photon collision estimator, with relative noise 1.4 %. Values are channel-pooled like gate77; ranges are over 3 quadrature seeds.

| Interior cell | Statistic (residual / ‖M‖) | Exact SS | q = 1024 SS (as WL76/77) | WL77 reported (16 S16-R sources × 16 bins) |
|---|---|---|---|---|
| bin 5 | ring | **0.220** | 0.39–0.42 | **0.453** (0.437–0.479) |
| bin 5 | FIA-AZ relaxation (ring × {1, cos, cos2}(φ_o − φ_r)) | **0.039** | 0.32–0.35 | **0.385** |
| bin 5 | free phase, order ≤ 2 | **0.027** | 0.28–0.33 | **0.352** |
| bin 0 | ring | **0.221** | 0.48–0.53 | |
| bin 0 | FIA-AZ | **0.084** | 0.43–0.46 | |
| bin 0 | free phase, order ≤ 2 | **0.021** | 0.39–0.43 | |

- **Gate ratio FIA-AZ / ring:** exact SS gives **0.18 (bin 5) and 0.38 (bin 0)**, which would pass the WL77 ≤ 0.5 gate. The q = 1024 SS gives **0.80–0.88**, against WL77's reported 0.85.
- **Scaling with q** (bin 5, per channel; red / green / blue):

  | q | SS_q relative noise | ring | free h2 |
  |---|---|---|---|
  | 1024 | 0.62 | 0.40 / 0.42 / 0.57 | 0.30 / 0.32 / 0.46 |
  | 4096 | 0.31 | 0.28 / 0.29 / 0.35 | 0.15 / 0.15 / 0.22 |
  | 16384 | 0.17 | 0.24 / 0.24 / 0.27 | 0.08 / 0.09 / 0.13 |
  | 65536 | 0.076 | 0.22 / 0.23 / 0.24 | 0.05 / 0.05 / 0.07 |

  This is the 1/√q signature of sampling noise.

**Physical in-ring structure (exact SS)**, interior / edge / side respectively:
- ring 0.22 / 0.20 / 0.23;
- order ≤ 1: 0.06–0.07;
- order ≤ 2: 0.02–0.03;
- order ≤ 3: 0.008–0.017;
- order ≤ 6: ≤ 0.008.

It is **dominated by the first azimuthal harmonic**, the diffusion-like dipole of light flowing away from the source. 81–93 % of the ring-residual energy sits at the eight neighbours of the source port (0.1–3.6 mm), where the scalar collision density has its steepest lateral gradient. Higher-order azimuthal content is physically small.

### 6.3 Consistency with FIA's measured errors

FIA deploys the same SS_q. In this emulation, SS_q noise alone is a fraction of the total collided response:

| Space | SS_q noise / total (R/G/B, both bins) | WL76 FIA per-source error |
|---|---|---|
| native | 0.22–0.29 | 0.29–0.30 |
| fine16 | 0.13–0.16 | 0.18–0.19 |
| coarse | 0.07–0.10 | 0.077–0.15 |

`diag_posthoc76.json`, err_total.

- **Coarse incoming-bin differences.** For the most-contrasting pair, bin 0 vs bin 5 at an interior cell-corner sub-cell, the pair-contrast signal is |ΔZ/2|/|Z̄| = 0.12–0.16, within WL74A's range. SS_q noise alone produces a **0.41–0.45** relative error on that difference. Per-bin-independent SS and W quadrature noise is therefore a plausible major contributor to FIA's uniform con_coarse of 0.69–0.75.
- **Not quantified:** the W (second-collision anchor) quadrature noise, also q = 1024.

### 6.4 Interpretation

**This emulation is non-evidence.** It covers one interior cell and two bins, with an independent implementation. Within it:
- WL77's "remaining order ≥ 3 in-ring structure (0.35)" and WL76's "ring floor 0.45" are reproduced quantitatively as **the residual of q = 1024 single-scatter quadrature noise**, not as transport physics.
- With an exact SS, the receiver-relative low-order lobe WL77 tested would have relaxed the ring restriction well past its gate.
- **Status of the WL76 and WL77 verdicts.** Both remain as recorded under their protocols. Attributing their failures to angular expressivity is, physically, unsupported by this analysis.
- **Verification route.** Read-only: rerun the `gate77 expr` statistic with a high-q or analytic SS on the archived bundles. It is cheap, but it is outside this agent's authority.

## 7. Required resolution of a transferable intermediate (non-evidence MC)

**Test.** Represent the k ≥ 2 collision density (SS kept exact) at a reduced resolution, re-emit from the represented density, and measure the error against the exact collision estimator. The errors are bias-corrected relative L2 per channel (red / green / blue).

| Representation of the scalar k ≥ 2 field | Energy bias (int b5) | int b5 total native / coarse | int b0 total native / coarse | Edge b5 native | Side b5 native | Pair-contrast error (int b0−b5, coarse) |
|---|---|---|---|---|---|---|
| Piecewise-constant 0.5 mm voxels | +2.1 / +2.0 / +1.2 % | .016 / .017 | .018 / .008 | .017 | .017 | .045 / .045 / .038 |
| Piecewise-constant 1.0 mm voxels | +6.0 / +5.3 / +2.3 % | .051 / .043 | .065 / .019 | — | — | .131 / .126 / .110 |
| Piecewise-constant 2.5 mm voxels | +17 / +13 / −2 % | .130 / .096 | .190 / .094 | — | — | .445 / .424 / .343 |
| Lateral snap to top-face 2.5 mm port columns, exact depth | 0.0 % (interior) | .096 / .065 | .092 / .038 | .20 (energy −11 %) | .36 (energy −23 %) | .081 / .076 / .052 |

**Physical reading:**
1. **Depth must be resolved at sub-mfp scale near every face**, or carried by an exact boundary-layer profile. The collision density has a steep boundary layer (extrapolation length 0.71 mfp). Smearing it over 1–2.5 mm biases the exitance by +6 to +17 % (red).
2. **Lateral resolution at the port pitch (2.5 mm) costs about 0.09–0.10 native and 0.04–0.07 coarse at interior sources.** That already sits inside the WL59 abs gates as a discretization bound. Sub-port lateral moments (for example a 2 × 2 split or a dipole) would be needed for ≤ 0.03.
3. **The incidence information survives lateral snapping of k ≥ 2 collisions** (pair-contrast error 0.05–0.08), because it lives mostly in the exact first-collision and single-scatter terms. Snapping the first collision instead would remove the 0.3–0.8 mm footprint shift that WL74A identified as the coarse-contrast mechanism.
4. **Near edges, "port columns" of one face mis-place depth relative to the adjacent face**, giving −11 to −23 % energy. Any anchor or boundary-layer basis must be **face-relative to every nearby face**: a two-face boundary layer at edges and three at corners.

## 8. What must be explicit vs what can be shared/learned

| Information | Source-conditioned? | Physics status | Representation requirement |
|---|---|---|---|
| Ballistic B | yes | exact geometry | exact (present) |
| First-collision density P_X: depth ~cos θ mm, lateral shift ~sin θ mm, edge truncation | yes | exact geometry, sub-port | explicit, deterministic; must keep sub-port position and the edge side-exit truncation |
| Single scatter ω⟨P_Y, P_X⟩ | yes | exact geometry | explicit, **with low noise**: q = 1024 MC is inadequate (§6); use deterministic / stratified / analytic integration and translation-invariant reuse |
| Exit projection P_Y (64 bins) | no (receiver-side) | exact geometry | explicit; generates all outgoing angular structure, so no learned outgoing lobes are needed for g = 0 |
| Scalar resolvent G(x,y) (multiple scattering) | **no** | direction-free, symmetric PD, ≥ 0, translation/D4h equivariant, geometry-dependent only through boundaries, dents and visibility | the shareable / learnable object; inputs are relative geometry only (offsets, face/edge distances, local thickness, normals), never absolute position, face id or source id |
| Boundary-layer depth profile | no | analytic near flat faces (Milne-like, extrapolation length 0.71) | explicit basis or ≤ 0.5 mm depth resolution |
| Medium constants σ_t, ω_c | no | parameters | shared per material |
| Cross-dent air paths | — | owned by C | excluded from K via in-medium visibility |

## 9. Physical arguments for and against candidate representation classes

**ATO / W+UV (rank 8 / 16).**
- *For:* a direction-free T is physically allowed (§3.2).
- *Against:*
  - A and O are not tied to each other (no reciprocity);
  - the incidence → position coupling only works through the shared index r;
  - absolute-coordinate and face-id inputs break exact translation equivariance, which is the symmetry that maps three of four BND4_FAR sources onto supervised ones (§5.3);
  - the learned-vs-oracle gap at BND4_FAR (0.53 → 0.65) is consistent with this.

**RSK (WL72).**
- *For:* position-free relational features move toward translation equivariance.
- *Against:* it is still coupled to global non-equivariant A/O frames (WL74A §3), and it was not tested with boundary sources in supervision.

**FIA (WL76) / FIA-AZ (WL77).**
- *For:*
  - its exact incident budget is physically right (energy 1.002, profile 0.023);
  - its equivariant features are right;
  - a receiver-relative low-order lobe captures the physical in-ring structure (FIA-AZ floor 0.04–0.08 with exact SS).
- *Against:*
  - deployable q = 1024 quadratures inject per-cell noise comparable to the target error (§6);
  - the learned R_θ must re-learn the exact exit projection;
  - second-collision anchoring snaps the k ≥ 2 field. With face-relative depth this costs ~0.1 native (§7), and more at edges if anchors are not face-relative;
  - the per-geometry quadrature costs 75 s.

**Higher-order or per-pair full 64-bin outgoing models.**
- *Against:* physically unnecessary for g = 0. Order ≤ 3 already leaves ≤ 0.017 with exact SS. The large "higher-order" residual WL77 measured is quadrature noise in this emulation.

**Collision-density / boundary-layer Green form** (K_scat = D_g⁻¹ Πᵀ Γ_θ Π, with Π exact geometric and Γ_θ = learned or analytic-prior symmetric resolvent on face-relative boundary-layer elements, i.e. the Schur complement of G).
- *For:*
  - exact angular ends, so no learned angular basis;
  - reciprocity and PSD by construction;
  - passivity by construction if Γ_θ is built as ω(I − ωS_θ)⁻¹ with S_θ ≥ 0, symmetric, and rows ≤ 1 − exact escape;
  - source-independent;
  - translation/D4h equivariant if Γ_θ uses relative geometry only.
- *Against / risks:*
  - an exact factorization is not automatically an efficient or learnable approximation;
  - Γ on a finite basis is an approximation;
  - Π requires low-noise geometric integration (rebuild cost);
  - dents need a geometry-conditioned Γ;
  - the generalization of the corrections to new dents is untested.

**Analytic diffusion/dipole slab prior for G.** Only as a prior: the near field (< 2–3 mfp) and blue (ω = 0.9) are outside diffusion accuracy, and that is where 66–93 % of the exit signal lies.

**Cost plausibility (estimate, unverified).**
- A face-relative boundary layer of 3840 ports × about 4 depth layers gives about 15k elements; a 2 × 2 sub-port split gives 61k.
- Γ, local to about 15 mm: about 15k × ~450 neighbours × 3 channels ≈ 2 × 10⁷ floats. That is the W+UV state scale of 15–30 M.
- Π_out: about 15k × ~25 ports × 64 bins ≈ 2.4 × 10⁷. In the flat rest regions it is identical for every port, so unique blocks can be stored and reused, and rebuild touches only edited regions.
- A 64-light apply is three sparse products, of order 10⁹–10¹⁰ MAC: plausibly tens of ms on GPU. Not measured.

## 10. Requirements for candidates (physics-derived)

1. The incoming dependence must enter through an exact or deterministic first-collision (sub-port) term. It must not be learned as a global angular mixture.
2. The outgoing dependence (g = 0 fixture) should come from an exact exit projection of a scalar field. Any learned outgoing angular basis must at least represent the order-1 receiver-relative dipole; orders ≥ 3 are physically ≤ 0.02.
3. The learned shared object should be direction-free, source-independent, and use relative geometry only:
   - no absolute coordinates;
   - no face id;
   - exact equivariance under the slab's translations (port lattice, away from corners and dents) and D4h, with the exact local bin permutations.
4. Reciprocity: A_i K16(i,a→j,b) = A_j K16(j,b→i,a), with no label reversal. Enforce it by construction (shared Π, symmetric Γ) or report it as a measured property. The scattered part must also be PSD in étendue-weighted form.
5. Passivity and energy: exact escape accounting (Σ ≤ 1 − β, and = 1 − β at ω = 1). The boundary-layer depth must be resolved at ≤ 0.5 mm or by an exact profile, to avoid the +6–17 % exitance bias (§7).
6. Near edges and corners, the boundary-layer basis must be relative to every nearby face.
7. Every deployable geometric quadrature must have **sampling noise well below the target error** in the metric spaces (native, fine16, coarse contrast). For q = 1024 one-ray SS it does not (0.22–0.29 native, 0.07–0.10 coarse). Report the quadrature-noise floor separately from replica noise.
8. Dented poses: in-medium visibility (cross-dent paths belong to C); G conditioned on local thickness and normal; no claim of reuse across an intrinsic edit.
9. Any extra privileged label must be declared separately. Reciprocal "adjoint" use of the same S16-R data, and exact geometric quadratures, are not privileged labels.

## 11. Rejected claims and ideas (with reasons)

- **"The remaining order ≥ 3 in-ring structure (0.35) is physical" (WL77) / "almost all of FIA's native error is the ring-lobe restriction" (WL76 post hoc).** Not supported physically in this emulation. With an exact SS the physical ring residual is 0.22 and order ≤ 2 leaves 0.02–0.03. The 0.35–0.45 values are reproduced only when the q = 1024 SS quadrature is used, and the reported noise floor excluded that variance (§6). (Non-evidence; needs read-only verification on archived data.)
- **"Far-boundary failure means the information is missing from 16-source supervision."** Rejected for three of four BND4_FAR sources in the rest pose: they are translate/reflection images of supervised sources up to about 1 % corner influence (§5.3, estimate). It stands only for corner-offset contexts (s103).
- **"Reciprocity would solve far-boundary transfer."** Rejected. Reciprocity only adds information within transport range (15–25 mm) of supervised ports. Its value is structural (PSD, A–O tie, contrast ↔ outgoing data), not far coverage.
- **"Neumann/order-truncated learned operators."** Rejected: 9–22 weighted collisions (red) on average, with a heavy tail.
- **"Diffusion theory alone for G."** Rejected as the sole model: the near field and blue are not diffusive, and 66–93 % of the exit lies within 3.6 mm of the source.
- **"Coarse lateral voxels (≥ 1 mm) without a depth boundary-layer model."** Rejected: +6–17 % red energy bias (§7).
- **"Anchors on one face's port columns near edges."** Rejected: −11 to −23 % energy at edge and side sources (§7).
- **"Per-source or per-port angular correction tables."** Excluded by the constraints; physically unnecessary, given exact P_X and P_Y.
- **"The location-integrated angular structure is rich."** Rejected; WL74A agrees: σ₂/σ₁ ≈ 0.02. The rich structure is per receiver and order-1 dominated.

## 12. Remaining uncertainty

1. **Verification on the real references.** §6 is an independent non-evidence emulation (one interior cell, two bins). The WL77 per-source values (0.437–0.479) are consistent with it but were not recomputed from the archived bundles. That needs a read-only GPU run with a high-q or analytic SS.
2. **W-quadrature noise** (q = 1024 second-collision anchors) and its share of FIA's contrast error were not quantified.
3. **The ~1 % corner-influence estimate** for BND4_FAR translates comes from MC exit-radius fractions. It was not measured as a direct difference of the two references, and the pt0-pose versions were not checked.
4. **Reciprocity at 2.5 mm** was tested on two pairs only. The SE estimates are crude, with 1 dof per half-split.
5. **Cost estimates** for a boundary-layer Green form are back-of-envelope.
6. **Whether a learned Γ_θ generalizes across dents** (te0–te2, cf*) is untested. Physics only says G depends on local geometry.
7. **Grazing sources at corners and the bottom face** were not simulated (only interior top, top-edge row and side mid-row).
8. **The MC uses the rest geometry only.** Dented poses, in-medium visibility and the deformation gates were not simulated.

## 13. Reproduction notes (non-evidence)

- **Engine.** `mc.py <seed> <photons> <face> <pu> <pv> <bin> <out>`: 5 estimator variants (exact, pc0.5, pc1.0, pc2.5, port-column).
- **Driver.** `drive.py`: 30 processes × 66,666 photons per cell, seeds 77000 + 1000·r + k.
- **Analysis scripts.** `ana.py`, `quadss.py` (SS quadrature seeds 9100–9200, 9300–9400), `azgate.py` (9500–9502), `contrast2.py`, `recip.py`.
- **Platform.** numpy 2.4.6, CPU, about 25 min in total.
- **Location.** All files sit in the scratchpad path listed in §1. None were committed.
