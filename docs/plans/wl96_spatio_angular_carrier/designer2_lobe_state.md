# WL96 d2 — Spatio-Angular Transport Carrier: independent design report (design-only)

**Scope.** I edited no repository file, trained nothing, and ran no GPU or renderer job.

- **Read:**
  - WL95 worklog §1–8;
  - WL95A worklog;
  - `judge.md`, plus the parts of auditA/auditB that WL95 cites;
  - `Architecture.md` §1–9 and the roadmap tail;
  - `model86.py`, `pt86.py` (`state_rays`, `source`, `trace 'ecro'`);
  - `a95.py` (`x1_radiance`, `flight`, `main`);
  - `closure83_dr.py`, `cand91.py`, `pt86a.field_src`, `quad87.qtable` (header);
  - `results/wl96_calc/ring_alloc.*`, `ck2_support.py`.
- **Scratch calculations:** CPU numpy only, all in `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d2\`:
  - `er_np.py` (numpy port of the ER eval);
  - `ck2_surrogate.py`, `truth_area.py`, `estimators.py`, `variance.py`, `detquad.py`, `rank_transfer.py` (output in `rank_transfer.json`).
- **Labels:** [FACT] code/doc with path:line · [MEAS] repository measurement with source · **[MEAS-S] my scratch numpy surrogate**, not repository evidence · [INF] inference · [HYP] untested hypothesis. I make no novelty claims.

---

## 0. Verdict (summary)

1. **The physically conditioned lobe state is the right sender state, but it is not compressible information.**
   - [MEAS-S] For a directly lit sharp sender, the exact lobe state is {light position and intensity, light visibility, geometry, ER}.
   - Its required spatial resolution is about 5 cm cells on CK2 (4×4 per 0.2 m patch). At that resolution the expectation bias is ≤ 1.5 % rel L1. A patch-centre state (= X1c) is at 0.23.
   - Its natural consumer is an *explicit deterministic integral*. Receiver-side sampling (X1) is unbiased, but 50 % of its estimate comes from about 0.09 % of rays.
2. **C1: lobe state + learned multi-event transfer operator** (the literal starting point). **NOT VIABLE as a learned representation.**
   - [MEAS-S] One S1 sub-cell's continuous-incidence → receiver transfer needs rank ≥ 256 for about 0.09–0.16 rel L1 (floor B: rank 16 → 0.026).
   - A geometry-conditioned generator would need about 0.2–0.7 G floats per geometry. An explicit deterministic quadrature needs a visibility cache of about 1 MB plus on-the-fly ER.
   - A non-linear learned transfer is learned ray casting, the WL52 class.
   - So C1 either reduces to explicit quadrature or is dominated by it by orders of magnitude.
3. **C2 (SRR-LB): sharp vertices as relays + learned broad resolvent.** Specifiable and falsifiable, but its learned-unique claim is predicted to be marginal.
   - Sharp vertices are never tabulated. Point-light incidence goes through lobe atoms and deterministic quadrature (DQ). Extended incidence goes through source-independent relay continuation in the relation and at readout.
   - A learned resolvent owns only ≥ 3-event transport on *broad* tables, where rank 16 is already adequate.
   - The learned part owns no sharp information. Its only possible utility is the source-independent tail cost, the ECRO niche of about 10–50 ms. **If test 6 shows no utility, C2 is explicit GI with an optional learned tail. That is an explicit-GI architecture; under Architecture.md:13 it stays a comparator unless the owner rules otherwise (Gate A).**
4. Overall classification: **NO_VIABLE_LEARNED_SHARP_CARRIER; ONE_HYBRID_SPECIFIED (C2, owner Gates A/B).** Nothing is selected or authorised.

---

## 1. Physical analysis of the sharp first flight (CK2, surrogate)

**Surrogate setup.**
- Geometry is from the repo scene: panel plane x = −1.5, y ∈ [−0.3, 0.9], z ∈ [0.3, 1.7], 6×7 patches of 0.2 m (`scene86.py:30-38`, `ecro86.json` fixture).
- Box A's panel-facing face is rect 6, 3×3 patches of 0.3 m.
- The CK2 light is [−1.45, −0.6, 1.7], rgb 30 (`scene_ck2.json`).
- ER material 1 is from `mats86.npz` (GGX α ≈ 0.026–0.075, η 2.50).
- No occlusion is modelled. Receivers are the 9 face patches, not WL95A's 54.

**Reproduction check of the surrogate against WL95A** (pooled / expectation).

| quantity | WL95A [MEAS] (54 box-A patches, `95A:83-85`) | surrogate [MEAS-S] (9 face patches) |
|---|---|---|
| binned physical q: flux ratio / rel L1 | 1.422 / 0.433 (q256 pooled) | 1.430 / 0.435 (expectation) |
| X1c (patch centre, exact direction) | 1.037 / 0.268 | 1.016 / 0.234 |
| X1 per-draw rel L1, median / worst | 0.617 / 2.2 | 0.568 / 5.5 |
| peak patch | patch 5 | patch 5 |

[INF] The surrogate reproduces the WL95A pattern, so the numbers below are credible *for sizing*. They are not evidence.

**Results**

| ID | Result [MEAS-S] |
|---|---|
| R1. Truth | Dense area integration on the panel converges: total 21.63 at h = 1 cm vs 21.63 at 5 mm, and m = 4 vs 8 receiver points agree to 0.07 % |
| R2. Lobe-axis spread | The image source is [−1.55, −0.6, 1.7]. Panel incidence cos ranges 0.026–0.120. The mirror-axis half-spread across one 0.2 m patch is 0.074–0.34 rad (median 0.11). That is ≥ the outgoing lobe width (about 2α ≈ 0.05–0.15). Hence the WL95A spatial term Rp |
| R3. Sub-cell lobe state (exact direction read at the sub-cell centre; expectation) | rel L1 by cells per patch side s: s = 1: 0.234; s = 2: 0.075; **s = 4 (5 cm): 0.015**; s = 8: 0.005. Flux ratio 0.95–1.02 |
| R4. Incidence dependence (two other lights) | Binned q bias 0.052 (light [−0.7, −0.3, 1.6]) and 0.216 ([−1.2, 0.2, 2.6]); CK2 is 0.435. s = 4 stays ≤ 0.009 in both. [INF] The binned floor is strongly incidence-dependent (grazing is the worst case); a 5 cm exact-direction state is stable |
| R5. Heavy tail of receiver-side gathering | 512 rays/patch, about 289 hit the panel. 50 % / 90 % of the pooled X1 estimate is carried by 0.09 % / 2.0 % of rays (median over patches). That is about 0.5 ray per draw per patch for half the energy. X1c is as heavy-tailed (per-draw median 0.55). [INF] Variance is intrinsic to receiver-side gathering of a sharp lobe, whatever the spatial resolution of the sender read |
| R6. Deterministic point-to-point quadrature (DQ: sender cell centres × receiver points, exact ER, deterministic) | rel L1 by sender cell × receiver points: 10 cm × 4² 0.082; **5 cm × 4² 0.026** (97 k ER evals for the 9 face patches); 5 cm × 8² 0.013; 2.5 cm × 2² 0.015; 1.25 cm × 4² 0.0009 (L-DQ oracle level) |
| R7. Transfer rank (one sub-cell; 4,608 incidence directions × 1,196 rect receiver patches; Frobenius SVD; clip-0 rel L1) | S1 corner: r16 0.99, r64 0.61, r128 0.37, **r256 0.16**. S1 centre: r64 0.36, r256 0.087. S2 material: r64 0.092. Floor B: **r16 0.026**. Ring-0 rows are worse (S1 r64 1.3–1.8). A row's 90 % flux spans a median of 46–76 receiver patches for S1. [INF] The rank grows roughly as lobe solid angle⁻¹ ∝ α⁻² |

---

## 2. Candidate C1 — LSTO: Lobe-State-conditioned learned Transfer Operator (the literal starting point)

**Mathematical definition**
- Sharp-sender sub-cells c (S1: 42 × 16 = 672 at 5 cm). For each light l and sub-cell c, the atom is a_cl = (ω_cl, E_cl), with ω the exact light direction and E = I cosθ/d²·V.
- Learned delivery: b_≥2[r, m, ch] = Σ_c Σ_l E_cl · T_θ(G)_c(ω_cl)[r, m, ch].
- T_c is factorised as Σ_{j ≤ r} κ_{c,j}(ω) R_{c,j}[r, m]. Both factors are generated by a geometry-conditioned ECRO-type network. ER enters as a generator input.

**A. State**
- Material-only: ER states (69 floats each).
- Source-dependent: atoms, 6 floats per (c, l). For CK2 that is 672 × 6.
- Geometry-dependent: generated factors κ, R, about r × (N × 64 × 3) per sub-cell.

**B. Interface**
- Inputs: sub-cell position and frame, continuous incident direction, irradiance, geometry tokens.
- Output: incident-bin irradiance (or outgoing table) at every patch. The outgoing query direction is the receiver bin.

**C. Ownership**
- ER: local response, used only as generator conditioning, so it is relearned implicitly.
- Learned: sharp-sender scattering plus propagation plus multi-event transport.
- Visibility: implicit inside T_θ. This violates the "ER owns every evaluated event" pattern of ECRO (WL95 §1.4) unless T_θ is restricted to propagation.

**D. Learning**
- Mapping (G, c, ω) → delivered field. Families: e2_p and e3_p, both tallied by `pt86a.field_src(split=True)` (`pt86a.py:73-200`).
- [INF] Information it could express beyond explicit ER + transport: none. Its only conceivable advantage is amortising the integral.

**E. Sharp-lobe preservation**
- It keeps position (sub-cell) and continuous direction, so it is X1-class information (R3).
- Fidelity is limited by the rank r (R7): at r ≤ 64 the per-sub-cell transfer error is 0.36–0.61, which is the WL95A rank-16 failure moved from incident bins to continuous incidence.

**F. Dynamic reuse**
- Rigid edit: weights are reused; *all* generated factors are regenerated (as ECRO).
- Light change: atoms only (672 shadow rays). This is its one attractive property.

**G. Complexity** [INF]
- Factors: 672 × 256 × 1,270 patches × 3 ch ≈ 6.6e8 floats (2.6 GB). With incident bins for non-Lambert receivers it is ×64.
- Explicit DQ needs about 6–7 M visibility bits (≈ 1 MB) plus on-the-fly ER.
- Generator output per edit far exceeds ECRO's 43 MB of factors [MEAS WL86 via WL95:83].
- Avoided work vs X1: per-light probe and shadow rays (0.53 M + 0.53 M, single draw [MEAS WL91 via WL95:382]). DQ avoids the same work without learning.

**H. Failure modes**
- α → 0: rank ∝ α⁻² is unbounded.
- Close contact: the transfer is discontinuous at visibility edges, which is WL52's transition failure.
- Many lights: fine (linear).
- Moving sharp objects: full regeneration.
- Non-rigid edits: no validity rule.
- Unseen materials: generator extrapolation over ER.

**Representability / minimum capacity**
- [MEAS-S] r ≥ 256 per sub-cell for ≤ 0.16 at patch level.
- [INF] A non-linear head (attention over receivers keyed by direction) can move the bump, but it is then learned ray casting: WL52 measured 8–19× slower updates and contact/transition failure (Architecture.md §8 row).

**Comparison**
- Strictly dominated by explicit DQ on memory and accuracy at equal information.
- Versus ECRO: it removes the binned bias but inherits a worse capacity problem.
- Versus H-0: no advantage.

**Predeclared falsification**
- F1a: on the real occluded CK2 pipeline (and the W-b light), the best non-negative rank-r (r ≤ 64) per-sub-cell transfer, fit on a dense incidence grid and evaluated at held-out incidences, has box-A e2_p delivery rel L1 > max(0.10, 2 × DQ-5cm) → falsified at practical capacity. The surrogate predicts this.
- F1b: the minimal adequate rank's memory exceeds 100× the DQ cache → dominated.

**Disposition.** NOT VIABLE (reduces to explicit DQ, or to learned ray casting). The F1a rank bound can be run cheaply as an arm of the smoke.

---

## 3. Candidate C2 — SRR-LB: Sharp-Relay Relation with lobe atoms + learned broad resolvent

**Class rule** [FACT for the rule]
- "Peaked" is a prospective material-only class: the WL88 median Π > 8 rule (S1 37.95, S2 16.13 → peaked; B 4.93 → broad; WL95:352).
- The response rule is ER everywhere. Only the *carrier* differs (Gate B, §6).

**Mathematical definition**
- Broad tables b̂ exist only on broad patches. Peaked patches have no table.
- Sharp-transparent relation 𝒯: for each ECRO state ray r (`pt86.py:355-376`) whose hit is peaked, store κ_c ER-VNDF continuation samples. Each sample stores the next hit, its outgoing bin, and weight Γ = eval/pdf per channel (≤ albedo). The sampling uses incoming = −ray direction (`closure83_dr.sample`); it is source-independent and uses reciprocity. Chains are kept up to depth D_r = 2.
- 𝒯[L](x, bin) = Σ_r w_r Γ_r L(end_r, bin_r).
- e2 field:
  - b2 = M64 ∘ 𝒯[q̂_broad] + **DQ**(atoms) + 𝒯-relay-to-emitter (area or environment lights only);
  - DQ(atoms)[x] = Σ_c Σ_l A_c E_cl f_ER(c→x, ω_cl) cosθ_c cosθ_x / d² · V(c, x) at receiver points;
  - DQ handles delta lights only, so there is no double count with relays (a relay ray cannot hit a point light).
- Higher orders:
  - exact: b_{k+1} = M64 𝒯 b_k;
  - learned: B_≥3 = Λ_θ(G)[b2], where Λ_θ is a non-negative low-rank (ρ ≈ 24) geometry-conditioned resolvent on broad tables (ECRO P/Q form, `model86.py:121-123`). Local operators are fixed to physical M64, not learned (WL95 §5 constraint), and its flight geometry is 𝒯.
- Readout: when a camera x2 is peaked, continue (ER sample) to x3 and read B(x3), instead of reading `B[peaked]` (`pt86.py:228-233`).

**A. State**
- Material-only: ER; M64 (broad materials); class flags.
- Geometry-only:
  - relation (650,240 rays, 17.6 MB [MEAS WL86]);
  - relay continuations (about 10 k S1 hits [MEAS 95A:111] plus S2; κ_c = 8 gives about 0.16 M rays, about 1.6 MB [INF]);
  - DQ visibility bits (672 sub-cells × 20,320 receiver points ≈ 13.7 M pairs, about 6–7 M front-facing [INF]).
- Source-dependent: broad s (`pt86.py:378-402`); atoms (672 × 6 per light).
- Learned: Λ_θ factors.

**B. Interface**
- DQ: (atom, sub-cell frame, receiver point) → irradiance.
- 𝒯: (table, relation) → gathered field.
- Λ_θ: b2 → B_≥3.
- Readout: camera vertex chain → first broad vertex → table bin.

**C. Ownership**
- ER: every scattering event, including relay vertices (sampled) and DQ (evaluated).
- Geometry: all visibility (relation, relays, DQ pairs).
- Learned: only the ≥ 3-event broad resolvent.
- Each segment is owned once: x→y (C), the ER response at y (K), y→z (C).
- Relays are passive (Γ ≤ albedo) and DQ is passive (ER passive). Λ_θ needs an added passivity/spectral check; ECRO had none (WL86:41 via WL95 §1.4).

**D. Learning**
- Λ_θ maps (G, b2) → B_≥3, supervised by exclusive e3 tallies (`pt86a`: e3_p + e3_o) or by explicit tail sweeps (noise-free targets; this makes the learned part an amortisation of explicit computation).
- [INF] It expresses nothing explicit sweeps cannot. Its sole claim is source-independent tail amortisation (WL87: source-only update 3.5–3.9× cheaper, about 40–50 ms [MEAS via roadmap]).

**E. Sharp-lobe preservation**
- e2_p: 5 cm atoms + DQ. Expectation 0.015–0.026 and deterministic (R3, R6), versus binned 0.435 and X1 per-draw median 0.57.
- e3_p: panel → receiver flights become relays that read broad radiance at the true reflected direction. There is no 64-bin projection at the peaked vertex. Variance is low because importance follows the lobe and Γ ≈ albedo [INF].
- This makes the e3_p family (0.755 of panel-face e3 [MEAS WL94B via WL95 §2.3]) *owned*, closing WL95's open item 5.

**F. Dynamic reuse** [INF]
- ER, M64 and θ survive any edit.
- Rigid move of a broad object:
  - selective relation re-trace (21 vs 26–28 ms [MEAS WL86]);
  - re-sample of relays whose segments intersect the old or new OBB;
  - DQ pairs involving moved receivers (≤ 54 × 16 × 672 ≈ 0.58 M for box A) or crossing swept volumes;
  - Λ_θ regenerated (ECRO generator).
- Light change: atoms and broad s only; no geometry work.
- Sharp mover: that object's relays, DQ pairs and atoms are all invalid.
- Non-rigid edit: sub-cell normals change, so DQ, relays and atoms are rebuilt over the region. There is no learned sharp state to certify.

**G. Complexity** [INF unless labelled]

| Stage | Cost |
|---|---|
| Offline | none for the sharp part; Λ_θ training as ECRO (1,024 geometries × 4 lights × 256 spp targets [MEAS WL86 via WL95 §4.3]) |
| Per light | 20,320 broad-source shadow rays + 672 atom shadow rays (S1); ER evals: about 6 M brute force, about 0.7 M with lobe-cone pruning (≈ 1,000 receiver points per sub-cell from R7's 46–76 patches × 16); estimated ≪ 1 ms GPU |
| X1 per light, single draw [MEAS GM1] | 0.53 M probe + 0.53 M shadow rays, with heavy tails |
| X1 at WL95A fidelity | pooled 64 draws, i.e. 32,768 rays per receiver patch [MEAS 95A §C] |
| Per edit | relation 14–28 ms [MEAS] + DQ re-trace ≤ about 1–2 M rays + Λ_θ generator, ECRO edit 76–78 ms [MEAS WL86]; explicit-tail alternative 11.6 ms sweeps [MEAS GM1, WL89:353] |
| Scaling | DQ pairs ∝ (sharp area / (α d)²) × receiver points, i.e. O(α⁻²) sub-cells and up to O(α⁻⁴) with receiver refinement; linear in lights; relays ∝ relation hits on peaked × κ_c |
| Curved sharp (S2, r = 0.4 m, α 0.07–0.18) | about 2 cm cells → about 5,000 sub-cells → about 5e7 one-time pair rays, about 77× the relation |

- **Avoided vs X1:** per-light probe and shadow rays, and pooling.
- **Introduced:** the one-time DQ visibility cache and relay samples.

**H. Failure modes**
- α → 0: for planar senders, switch to the image source (exact at α = 0, with an O(receiver points) cost). Curved mirrors give caustics, which DQ cannot resolve without α⁻² cells.
- Close contact: DQ pair visibility near contact inherits the WL54/55 near-contact spatial requirement.
- Many lights: ER evals ∝ L; at about 100 lights this is tens of ms [INF].
- Area or environment lights: atoms do not apply. Relays to emitters do, at low variance for env lights, and need MIS for area lights.
- Moving sharp objects: full invalidation of that object's caches.
- Non-rigid edits: rebuild of the region; no correspondence semantics are needed.
- Unseen materials: ER exact. The class threshold (Π) is the risk: a misclassified moderate surface gets binned or relayed.

**Comparison**
- vs ECRO: removes the learned and binned sharp sender (WL95A A0 0.52×/1.63×).
- vs X1: deterministic, and no per-light rays.
- vs H-0 (EXP/C3UR binned): removes the 64-bin first-gather floor (WL90/95A).
- **vs H-0 + DQ + relays with explicit sweeps: identical except Λ_θ.** That is the decisive comparison.

**Predeclared falsification**
- F2a: DQ-5cm (occluded) must reach box-A e2_p flux in [0.95, 1.05] and rel L1 ≤ max(2 × half-noise (0.059), 0.05) on CK2 and on ≥ 1 independent prequalified condition. Otherwise the carrier is insufficient.
- F2b: relay e3_p, fed exact broad fields, must match path-traced e3_p within noise, pooled over 16 relay seeds. If binned M64 e3_p is already within noise, relays are unnecessary; this is a simplification, not falsification.
- F2c: learned utility.
  - If the explicit 3-sweep tail on 𝒯 is within noise of the path-traced tail *and* costs ≤ 15 ms per source change, Λ_θ's maximum accuracy gain is 0 and its maximum saving is ≤ 15 ms.
  - Then C2's learned claim fails on this workload and it is reported as explicit.
- F2d: a per-geometry non-negative rank-24 free fit of the explicit tail operator (Gate D, WL95 §7.4) gives tail error > 2 × noise → the ECRO-lineage tail form is inadequate.
- F2e: per-light cost above C3URX1 single-draw cost, or per-edit cost above 78 ms → the economics fail.

---

## 4. Comparison

| | ECRO v0/OWN | X1 (C3URX1-type) | H-0 binned | DQ + relays (explicit) | C1 | C2 |
|---|---|---|---|---|---|---|
| Sharp e2_p, expectation | learned 0.52×/1.63× [MEAS] | unbiased | 1.42–1.54× [MEAS] | 0.015–0.026 [MEAS-S] | rank-limited | as DQ |
| Per-draw variance | fixed draw | heavy-tailed (median 0.62) [MEAS] | moderate | none (deterministic) | none | none (e2); low (relays) |
| e3_p owner | global term (compensates) | binned sweeps | binned | relays | learned | relays |
| Per-light rays | 20 k | about 1.06 M (×64 for fidelity) | 20 k + q | about 21 k | 672 | about 21 k |
| Learned role | sender, flight, global | none | none | none | all sharp | broad tail only |

---

## 5. Smallest feasibility experiment

**Status.** Not authorised. The batch class is an owner decision:
- the WL95 → 95A precedent suggests a suffix;
- AGENTS.md priority (1) suggests a regular number, because relays and DQ are new mechanisms.

It is non-comparative with respect to the four external baseline roles: Neural Full Recompute is UNMEASURED. It is learning-free except for per-geometry free fits; there is no generator training.

**Workloads**
- **W-a:** CK2 "none" light 0. CKF_none e2_p/e3_p exist (half-noise 0.059 box A).
- **W-b (independent validation):** one CK2 light chosen *before references* by a geometry-only rule that requires a learning-free predicted binned-vs-exact difference ≥ 0.15. Example: [−1.2, 0.2, 2.6], surrogate 0.216. It is prequalified by SNR ≥ 4 on box A and needs one new 2 × 4,096 spp `field_src(split=True)` reference. If it fails qualification, drop it and declare the result CK2-only.
- **W-c:** E2 T1 region C (held-out geometry and lights). Only if the panel-tagged split file is verified before freeze.
- **Superposition:** W-a + W-b lights together (linearity and many-light cost).

**Arms and tests**

| Test | Arms / measurement |
|---|---|
| 1 Representability | DQ ladder: sender cell {20, 10, 5, 2.5} cm × receiver points {2², 4², 8²}, occluded. C1 rank bound: per-S1-sub-cell transfer NMF r ∈ {16, 64, 256} on a 48×96 incidence grid, held-out incidences plus W-b |
| 2 Oracle | L-DQ-dense (1.25 cm × 8², occluded) vs REF e2_p. Gate V0: flux ∈ [0.95, 1.05], rel L1 ≤ 2 × half-noise + own convergence; failure → STOP. Relay e3_p (fed reference broad fields) vs REF e3_p |
| 3 Generalisation | All thresholds frozen on W-a, applied unchanged to W-b, W-c and superposition |
| 4 Bias vs variance | DQ: deterministic, bias from the ladder. X1/X1c: 64 pooled draws plus per-draw median/p90/worst (reuse `a95.py`). Relays: κ_c ∈ {1, 8, 32} × 16 seeds. Report bias and SD separately |
| 5 Cost | One-time / per-light / per-edit (CK2 box-A rigid move) ray counts, ER evals, warm uncontended GPU ms (Dr.Jit compile reported separately), MB, against X1 (C3URX1 accounting) and ECRO (76–78 ms edit, 13–18 ms source). Full-scale GPU-utilisation probe per stage before freeze |
| 6 Learned utility | Explicit 3-sweep tail on 𝒯 vs path-traced tail given exact e2 (accuracy headroom), and its ms (cost ceiling for any learned tail). Per-geometry non-negative rank-24 free fit of that tail (Gate D) on CK2 and E2 T1 |

**Budget** [INF]
- L-DQ-dense on box A: about 37 M pairs plus occlusion rays (GPU seconds).
- One new reference render.
- NMF fits: CPU minutes.
- Total about 1 GPU-hour.

**Decision outcomes (frozen)**
- **O1:** F2a passes on W-a and W-b, relays pass, and F2c shows no headroom → sharp carrier solved explicitly. C2's learned part is cost-only (quantified) → owner Gate A.
- **O2:** binned M64 e3_p is within noise → relays dropped; C2 = DQ + binned + tail.
- **O3:** DQ fails W-b or W-c (occlusion, contact) → carrier insufficient; no learned fix is implied.
- **O4:** F2d fails → the ECRO tail form is inadequate even sharp-free.
- **O5:** F1a passes at r ≤ 64 (not predicted) → C1 reopens.

---

## 6. Owner gates and open questions

- **Gate A:** is an explicit sharp carrier (DQ, relays, readout continuation) admissible inside the learned method, or only as a comparator? C2 depends on it, and C2's learned share is predicted small.
- **Gate B:** carrier partition by sender material class and by delta vs extended incidence. The response rule (ER) is identical, so it does not obviously conflict with Architecture.md:21, but that is for the owner to rule.
- The surrogate omits occlusion, box-A side faces, S2 and textures, and uses an unvalidated numpy port of ER (agreement with WL95A only to about 0.03–0.04).
- Curved sharp senders (S2) and the α → 0 cost for non-planar senders are the main scaling risks.
- Canon: C2's explicit parts must be declared as such (`FNTO_ARCHITECTURE_CONTRACT_V0_1.md:86`, "not an exhaustive explicit per-surface full path tracer hidden under a neural name").
