# C1 geometry-change contract (Part D)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

# WL80B Part D: geometry-change contract for C1 / CP-K (critical review)

**Basis (read-only).** Nothing was written, run or measured.

| Source | Lines read |
|---|---|
| `docs/plans/wl80a/trackB_candidate_records.md` (CR) | l.131–285 |
| `trackB_selected_candidate.md` (SC) | whole file |
| `trackB_adversarial_review.md` (AR) | l.40–70, 320–345, 400–450 |
| `cross_review.md` (XR) | l.115, 161–163, 636–652, 950–976, 1008–1010 |
| `coordinator_disposition.md` (CD) | whole file |
| `neutral_problem_brief.md` | geometry and locality lines |
| `docs/plans/wl77a/coordinator_decision.md` | l.10–15 |
| `docs/plans/wl77a/specialist_physics.md` | l.295–305 |
| `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` | l.44, 161 |
| `docs/Architecture.md` | l.56, 58 |
| Worklog 59 | l.47, 182, 219 |
| Worklog 76 | l.252 |
| Worklog 79 | l.313 |
| `experiments/fnto_rank66/ports66.py` | l.47 (bin semantics) |

WL53, WL55 and WL42 figures come from the project summary in `AGENTS.md`, which was already in my context. Their worklogs are not on the allowed list, so I did not open them.

**Labels.**

| Tag | Meaning |
|---|---|
| [EP] | Established physics |
| [PA] | Prior art |
| [HM] | Historical measurement |
| [WD] | WL80A design claim |
| [ND] | New derivation (mine) |
| [UH] | Untested hypothesis |

## D0. State inventory used below

I split C1's state into layers. The cases below refer to them by letter. The split is my own [ND], built from SC l.88–115 and CR l.149–158.

| Layer | Contents | Channel-dependent? |
|---|---|---|
| **G** (geometric factors) | Body-fitted mesh, ports, port areas and frames, B, d = 1 − β, F_in, the fine first-collision points, Φ_pt, W1, the W near field and pFFT precorrection, Φout (shape + complement) | No [WD, XR l.932]. All depend on σ_t × length and in-medium visibility |
| **S** (solver state) | ψ_c for each lighting state u and channel, including any warm start (AR l.407) | Yes |
| **T** (optional tabulated K) | Cached response columns (AR l.409; XR l.972) | Yes |
| **C** | Free-space coupling, including same-object re-entry through air (CR l.151) | — |
| **U** | Incident field u in port frames | — |
| **GI** | Coupled K∘C solution | — |

**Ownership rule.** A path belongs to K while it stays in the medium until its first exit. Any segment through air belongs to C [WD, CR l.151; contract l.44]. B and d are geometry-only and channel-free [WD, brief l.58].

## D1. Propagation ranges: local operator edit versus nonlocal effect

### (a) Range of an operator edit (channel-free) [ND]

- An edit changes G-factor entries only where a flight touches moved geometry.
- The free flight kernel σ_t·e^{−σ_t s}/(4πs²) leaves a mass fraction e^{−σ_t s} beyond distance s [EP].
- So ΔF_in, ΔΦ_pt, ΔW1, ΔW and ΔΦout fall below 1e-3 (or 1e-6) of the flight mass beyond r_f ≈ 7 mm (or 14 mm) from moved surfaces or moved cells. The same holds for ΔB and Δd.
- This range is the same for R, G and B because the factors are channel-free.
- The kernel has infinite support. A "local factor rebuild" is therefore exact only up to a truncation tolerance, never exactly. The pFFT far field is recomputed at every application anyway, so only the near-field and precorrection entries of moved cells need rebuilding.

### (b) Nonlocal propagation through the resolvent (I − ω_c W)^{-1} [ND; one-group diffusion with extrapolated boundary]

**Inputs.**
- σ_t = 1 mm⁻¹, D = 1/3 mm, σ_a = (1 − ω).
- z_e = 0.71 mfp, so T_eff = T + 1.42 mm.
- The lateral Green's function decays as K₀(κr), with κ² = σ_a/D + (π/T_eff)².

| Quantity | Red (ω .999) | Green (.99) | Blue (.9) |
|---|---|---|---|
| Infinite-medium length √(D/σ_a) | 18.3 mm | 5.8 mm | 1.8 mm |
| Slab-mode length ℓ = 1/κ, T = 10 mm | **3.56 mm** | **3.08 mm** | **1.63 mm** |
| ℓ in the 4 mm thinned region under a 6 mm dent | 1.72 mm | 1.65 mm | 1.25 mm |
| Halo distance for e^{−r/ℓ} = 1e-2 / 1e-3 | 16.4 / 24.6 mm | 14.2 / 21.3 mm | 7.5 / 11.3 mm |
| Fundamental collision eigenvalue λ = ω/(1 + B_z²/3) | 0.974 | 0.966 | 0.878 |
| Collision gain 1/(1 − λ) | 39 | 29 | 8.2 |

**Cross-checks.**
- The ℓ values reproduce CD l.19 (3.6 / 3.1 / 1.6 mm) [WD/CD].
- For red, the K₁ tail beyond ≈ 15 mm is (κr)K₁(κr) ≈ 4.3 %. That is consistent with the brief's measured 5.5 % of energy outside the 128 nearest ports (brief l.113) [HM].
- λ and the gain reproduce CD l.21 (ρ = 0.974) and CR l.225 (38 / 28 / 8) [WD].

**Caveats.**
- Diffusion is poor within a few mfp of a boundary [EP].
- The thinned 4 mfp region is exactly such a place, so its row of the table is indicative only [UH].

**What the resolvent does to a local edit.**
- A local change ε in the per-collision leak or gain becomes a change in total output of up to about 1/(1 − λ)·ε (AR l.41) [WD]. The magnitude is amplified (red ≈ 39×, blue ≈ 8×).
- The spatial reach is set by ℓ_c, per channel.
- So D_T extends roughly ℓ_c·ln(1/tol) beyond the G-edit region, and red reaches about 2.2× further than blue.

### (c) Coupled propagation through K∘C [ND]

- C has no exponential decay. Its range is line of sight, with 1/r² form factors [EP].
- Each round trip is damped by ρ(KC). Measured values on other fixtures: 0.45–0.82 (WL53) and 0.51 (WL52) [HM, AGENTS.md].
- Reaching 1e-3 therefore takes about 9–34 bounces.
- A local K edit therefore changes the GI solution at every port with a line of sight to the changed exitance, scene-wide but small.
- On an isolated convex slab, C has no self-relations, so there is no outer loop (XR l.966) [WD].

### (d) Consequence for "local update" claims [ND]

There are three distinct kinds of update:

1. **G rebuild.** Local, to r_f.
2. **Solve.** Global. A fresh lighting state has no ψ to update.
3. **Effect.** Ranges ℓ_c in the medium and line of sight in C.

## D2. Per-case contract

### 1. Rigid transformation of the whole body (body separated from everything by air)

**Remains valid.**
- All of G, and all of S and T for lighting fixed in the body frame.
- The intrinsic K is exactly invariant in the body frame [EP]. The port frames are body-attached, and the incoming-bin semantics are per-triangle (t, b, n) frames (`ports66.py` l.47) [HM, code].
- If G is stored in body coordinates, the reuse is bitwise. Recomputing it in world coordinates gives only round-off [ND].
- Measured analogue: rigid frame/state reuse was supported for the learned model (WL59 l.219) [HM].

**Becomes invalid.**
- U: world lights map into different bins and different shadowing.
- C: inter-object relations, and same-asset re-entry if a new occluder appears (contract l.44).
- S for world-fixed lights, and GI.

**Rebuild.** C relations, U, then the GI solve. C1's factors need none. CR l.181 and SC l.118 are correct for K but incomplete as an edit contract [ND].

**Local update.**
- C: incremental relation update. WL42: 0.63 % of rays reproduced a full rebuild to 8e-6 dB [HM, AGENTS.md].
- K: none needed.

**Global change.** Yes, through U and C. The response is not intrinsic.

**K or C.** C only, while the body stays separated from others by air.
- If the move creates index-matched optical contact, the domain merges and K changes (case 7).
- WL55 reports a contact-creating rigid move in which the re-aggregated rest K passed 18/18 and the C resolution was binding [HM, AGENTS.md].

### 2. Local non-rigid deformation: the dent

The dent is up to 6 mm deep with radius 15–25 mm; the bottom is fixed and the material unchanged (brief l.52). One measured instance is depth 5.75 mm, radius 22.5 mm (WL59 l.47) [HM].

**Remains valid.**
- ω_c and σ_t.
- G entries farther than r_f from every moved surface and moved cell.
- The rest translation-invariant templates outside that region.

**Becomes invalid (G).**
- **Under the amended vertical stretch map (SC l.113; AR l.324) every cell in each footprint column moves, through the full depth** [ND].
  - This includes the bottom-face boundary layers. They compress to about 40 % (XR l.648–651), although the bottom surface did not move.
  - The representational change is therefore volumetric and wider than the physical one.
- Moved top ports change area and position. Their per-port frames change too, because they are first-triangle frames (WL79 l.313) [HM]. So bin semantics change for moved ports, independently of the physics.
- Bottom sources under the dent are G-dirty physically, not only through the representation [ND]:
  - a normal-incidence bottom beam now crosses only 4 mm before a ballistic exit;
  - β rises from e^{−10} ≈ 4.5e-5 to e^{−4} ≈ 0.018;
  - so B, d, F_in, Φ_pt and W1 change for bottom beams.
- This confirms SC l.120 (D_T ≠ D_G must include bottom, side and rim sources). It refutes the sufficiency of CR l.184 (about 450 top ports only).

**Becomes invalid (S, T, C, GI).**
- S: all of it, per lighting state.
- T: entries ΔK[j,i] ≲ G(d_{iD})·G(d_{Dj}) under a first-order Born bound [ND].
  - The source-column support is the ports within about R_dent + ℓ_c·ln(1/tol) on top and bottom, plus side ports for off-centre dents.
  - For R = 25 mm in red, that is about 860–1,240 per face at 1e-2 / 1e-3 [ND, assuming a centred dent], i.e. about 45–65 % of the 3,840 ports.
  - This is consistent with XR l.972–975 (1,000–1,600 ports for its assumptions) [WD].
- C: the dent makes the top surface non-convex.
  - Exits from the dent walls and floor can re-enter the cavity, so C gains same-object relations and an outer loop appears even for the isolated slab (XR l.966 [WD]; [ND]). The magnitude for a curvature radius of about 36 mm and slopes up to about 30° (AR l.49–55) is unmeasured [UH].
  - Chords that pass through the cavity become exits plus C re-entry (CR l.151).

**Rebuild.**
- G within r_f of moved geometry, by ray integration on the true mesh (AR l.326).
- The pFFT precorrection for moved cells.
- C self-relations of the cavity.
- A global re-solve of S for every lighting state.
- Unless the tabulated variant is used, T is not rebuilt.

**Is a local update mathematically valid?**
- **Woodbury, exact [EP + ND].** With A = I − ωW and a topology-preserving map, ΔW = E_M X + Y E_Mᵀ has rank r ≤ 2m, where m is the number of moved cells.
  - m: about 23 layers per column (inferred from AR l.338: 1.5e5 cells over 6,400 columns) × 452–1,257 footprint columns ≈ 1.0e4–2.9e4.
  - Hence r ≈ 2e4–6e4, and that needs 2m resolvent solves per channel plus a dense r × r capacitance matrix (fp32, about 1.6–13 GB per channel).
  - A single global solve per lighting state is far cheaper, unless more than about 1e4 lighting states share the edit.
  - Exact but not useful here.
- **Truncated low-rank ΔW.** It breaks W ≥ 0 and 1ᵀW ≤ 1, which is the condition for C1's passivity bound (CD l.20). It is not admissible under C1's own invariants unless a positivity-preserving compression exists [UH].
  - WL53 measured dent ΔK local (≥ 91 %) with a 90 %-energy rank of 8–18 [HM, AGENTS.md; different fixture and 1 cm ports]. That is energy compressibility, not exactness.
  - Architecture.md l.56 forbids assuming local, sparse or low-rank changes [HM, contract].
- **Local re-solve on a halo r_h, with a stored ψ, for the same lighting state.**
  - It is an approximation with error of about e^{−r_h/ℓ_c} [ND].
  - At 1e-3 around a 25 mm dent, the red halo has radius ≈ 50 mm, about 79 % of the top area; blue ≈ 36 mm, about 41 %.
  - It offers little saving on this fixture, and none for a new lighting state.

**Does the response change globally?**
- Yes, through the resolvent: red tail to about 25 mm beyond the dent, blue to about 11 mm, and top to bottom across the full thickness.
- In a multi-object scene, scene-wide through C.

**K or C.** Both. Intrinsic K changes physically, and C gains cavity relations. Measured analogue: WL59 l.182 found geometry + B at about 97.5 % of the learned S update cost [HM]. So even with perfect locality in the solver, the geometric work dominates.

### 3. Large non-rigid deformation (global bend or twist)

**Remains valid.** ω and σ_t only. Nothing in G survives.

**Becomes invalid.**
- All of G. Translation invariance is lost, so no per-face convolution templates remain.
- The body becomes non-convex. Masked-convolution exactness is lost (CR l.170) and in-medium visibility enters the far field; there is no visibility-aware pFFT in the records [UH].
- All of S, T, C and GI.

**Measured analogue.** WL53: a 90° bend changes ΔK globally and at high rank, and C's same-domain re-entry goes from 0 to 3.1 % [HM, AGENTS.md].

**Rebuild.** Full G rebuild, new C self-relations, global solve. An explicit 10 mfp W does not fit in memory (about 13 GB, XR l.957) [WD].

**Local update.** None is valid (the rank is the whole mesh).
- For a gentle bend, the leading K change is plausibly of order T/R_curv per unit. Standard curvature corrections to boundary layers are order 1/(σ_t R_curv) [ND/UH].
- That is a perturbation estimate, not an update scheme.
- If the mesh topology is unchanged, the bias-correlation argument (SC l.121) is again only an inference.

**Global change.** Yes. **K or C:** both. FNTO contract l.161 places this outside the supported edits.

### 4. Changed thickness

**Uniform T → T′.**
- Every K entry changes. The transverse mode B_z = π/T_eff changes, so ℓ_c, λ and the gain all change globally [ND].
- For example, at T′ = 8 mm red ℓ ≈ 2.9 mm.
- The side-port count is proportional to T, so **the port index set changes**. Any K or C indexed by port has no correspondence for the side rows and columns.
- Translation invariance and D4h survive, so the rest-type templates are rebuilt once.
- C changes, because the top face moves relative to other bodies.
- Local update: none (global ΔW).

**Non-uniform thickness.** This is the dent class (case 2). Thinning shortens ℓ locally (table D1).

**K or C.** Both.

### 5. Changed material properties

**Uniform albedo ω_c.**
- All of G and C remain valid. ω enters only the solve [WD, XR l.932, 1010; ND confirmed].
- All of S, T and GI are invalid. K changes globally, as a power series in ω.
- Multi-shift Krylov solves share one Krylov space across ω [PA, standard shift invariance; not re-verified]. They are linear only to tolerance (SC l.57).
- "Free" means no geometric rebuild. It does not mean no verification: the discretisation error is weighted by the collision gain, which depends on ω [ND]. A refinement study done at one ω does not certify another [UH].

**Local albedo region (σ_t fixed).**
- W stays valid; the solve becomes ψ = W1Ωq + WΩψ with a diagonal Ω.
- Woodbury is exact, with rank equal to the number of cells in the region.
- The effect reaches ℓ_c beyond the region [ND].
- This is outside the brief's homogeneous scope, but inside the CP formulation.

**σ_t, uniform.**
- All of G is invalid: B, d, F_in and every flight kernel.
- The grading is defined in mm, not mfp, so the refinement evidence lapses; the port size in mfp changes too [ND].
- Full rebuild.
- RGB-dependent σ_t triples G [ND].

**σ_t, local.**
- The medium becomes heterogeneous. CP still works with optical-depth chords [PA, classical CP lattice methods; not re-verified], but there is no FFT (SC l.86).
- Explicit near and far fields are needed, with a full G rebuild in r_f, plus a global solve.

**g ≠ 0 or an index mismatch.**
- This is a different operator class, not an update. The interior state needs angular moments; Fresnel/TIR breaks the first-exit/escape-only structure (SC l.86) [WD].
- Who owns interface reflection between K and C is an undecided contract question [UH].

**K or C.** K only. C is unchanged unless the change brings a Fresnel interface.

### 6. Changed connectivity or topology (cut, hole, split, weld)

**Remains valid.** G farther than r_f from new or removed surfaces, but only if those cells survive unchanged.

**Becomes invalid.**
- Mesh and port set: new surfaces create ports, and welded patches lose them. The K dimension changes, so there is no fixed-size operator to perturb. **Woodbury on the old W is undefined** wherever cells are split [ND].
- In-medium chords crossing a cut become exits plus C re-entry. A weld turns former C paths, or contact paths, into intrinsic ones.
- K changes within ℓ_c of new boundaries, and a new free surface shortens ℓ locally.
- S, T and GI are invalid.
- C gains self-relations across holes or gaps.
- The union of two welded media is not K_A ⊕ K_B.

**Rebuild.** Remesh, new ports and correspondence, G near the changed surfaces, C, then a global solve.
- The bias-correlation basis for signed change (same topology) is lost [ND].
- Architecture.md l.58 and FNTO contract l.44 require explicit correspondence and ownership semantics. None exist in the C1 records [HM, contract; WD absent].

**K or C.** Both, and ownership moves between them.

### 7. Self-contact

**Approach with an air gap (not touching).**
- K physics changes only through the deformation that caused the approach (case 3).
- C acquires near-field same-object coupling with transfer close to 1 across a small gap. That raises ρ(KC) and slows the outer loop [ND].
- Port-level C resolution near contact was binding in WL54/WL55 (1 cm insufficient; 2.5 mm within ±1.8 % face-on) [HM, AGENTS.md].

**Contact with index matching** (the brief's boundary: a photon reaching the boundary simply leaves) [ND].
- Physically, the transport is **continuous** as the gap closes to zero: exit followed by immediate re-entry in air is the same path measure as in-medium transport across a fused interface.
- The ownership switch from C to K at contact is therefore a **representational discontinuity**.
- Under the port/bin contract (2.5 mm ports, 64 out-bins re-binned into 16 in-bins), the discretised K∘C at zero gap does **not** converge to the fused K. Position within the port and direction within the bin are lost at the interface.
- Near contact, the error is set by the contract, not by C1's discretisation.
- After contact the domain topology changes (case 6). The contact patch's ports vanish and the K dimension changes.

**Opaque or non-matched contact.** K is structurally valid. C has zero-distance, nearly singular coupling.

**Local update.** None is exact (the dimension changes). The effect is global through the resolvent and C.

**K or C.** Before contact, C (plus K for the deformation itself). At contact, K and ownership.

### 8. Multiple objects and external visibility changes

**Remains valid.**
- Each body's K (all of G) while that body is unchanged and separated by air [EP; contract l.44].
- B and d.

**Becomes invalid.**
- C relations between bodies, and same-asset re-entry blocked or unblocked by new occluders. This belongs to C, never K (contract l.44).
- U (shadowing), S and GI.

**Rebuild.**
- The changed relations (exact incremental update; see the WL42 analogue in case 1).
- A global coupled re-solve.
- This is the "lighting-only at fixed geometry" regime that the records call C1's only plausibly favourable one (SC l.132; FD l.163) [WD; cost unmeasured].
- The isolated-slab claim "GI = one K application" (XR l.966) no longer holds here [ND].

**Local update.** For a fixed-dimension coupled system, Woodbury on the changed C relations is exact, with rank ≤ the number of ports whose relations changed [EP]. The effect is global through the coupled resolvent: line of sight, 9–34 bounces to 1e-3 at ρ 0.45–0.82 [ND from HM ρ].

**K or C.** C only, unless the bodies are in index-matched contact (merged domain, case 6/7). Contact with a different medium makes the union heterogeneous, outside C1's scope.

## D3. Summary matrix

| Case | Intrinsic K changes? | C changes? | G rebuild | S re-solve | Exact local update? | Effect range |
|---|---|---|---|---|---|---|
| Rigid, separated | No | Yes | None | Yes, if U changes | C: incremental, exact | Through C, scene-wide |
| Dent | Yes (top, bottom, rim) | Yes (cavity) | Within r_f (≈ 7–14 mm), volumetric under the stretch map | Global | Woodbury exact but rank 2e4–6e4; low-rank truncation breaks the passivity conditions | ℓ_c·ln(1/tol): red ≈ 16–25 mm, blue ≈ 8–11 mm, full thickness |
| Large bend | Yes, global | Yes (self re-entry) | Full | Global | No | Global |
| Thickness | Yes, global; port set changes | Yes | Full (templates rebuilt once) | Global | No | Global |
| Albedo | Yes, global | No | None | Global | Region case: Woodbury exact | ℓ_c (new ω) |
| σ_t | Yes, global | No | Full; refinement evidence lapses | Global | No | Global |
| Topology | Yes; port set changes | Yes | Remesh | Global | No (dimension changes) | ℓ_c plus C |
| Self-contact | Yes at contact (ownership change) | Yes | Remesh at contact | Global | No | Global |
| Multiple objects | No (if separated) | Yes | None | Global coupled | Woodbury on C exact | Line of sight |

## D4. Corrections to the WL80A records

1. **CR l.186, "FFT spectrum unchanged".** The free kernel's spectrum is unchanged, but the dented body is non-convex, so the masked-convolution exactness of CR l.170 lapses. The amended C1 drops local FFT (AR l.326), and the pFFT precorrection of moved cells must be rebuilt [ND].
2. **CR l.184 (about 450 ports) is not sufficient.** Bottom beams change physically: β under a 6 mm dent rises by about 400× [ND]. This supports SC l.120 and XR l.974.
3. **XR l.646, "D_T ≠ D_G creates no invalidation-set problem for the resolvent".** True only for a per-lighting global solve from scratch. Any warm start (AR l.407), stored ψ, or tabulated K (XR l.972) brings the dirty set back. The G rebuild set is itself dilated by the stretch map beyond the physical change [ND].
4. **AR l.329, "signed-change error ∝ O(h)·|change|".** This is an inference contradicted in form by XR l.648–651: the stretch compresses the layers by about 40 %, including at the fixed bottom face. The bias difference scales with the change in h/mfp inside the footprint, not with |change| [UH]. A stretch map confined to the upper layers would avoid spurious changes in the bottom layers. That map is not in the records [ND].
5. **"Albedo edits free" (XR l.1010).** Free only for G. K changes globally, and the convergence certification depends on ω [ND].
6. **"Rigid: nothing to rebuild" (CR l.181; SC l.118).** True for K. As an edit contract it must also state that C, U and GI are rebuilt, and that optical contact moves the edit into the topology class [ND].
7. **Woodbury or low-rank updates.** No WL80A record proposes them. On this fixture they are exact but dominated by one global solve, and truncation violates CD l.20. C1's geometry-edit cost is not reduced by any exact local resolvent update [ND].

## D5. Open questions (untested)

1. Magnitude of the cavity self-coupling in C for the evaluated dent [UH].
2. Whether rest and dented discretisation biases cancel in the signed change under the stretch map [UH] (SC l.121).
3. Whether a positivity-preserving compression of ΔW or ΔK exists [UH].
4. Accuracy of diffusion-based ℓ in the 4 mfp thinned region [UH].
5. Ownership of Fresnel interface reflection [UH].
6. Correspondence semantics for port-set changes in thickness, topology and contact [UH]. None are specified in the C1 records, and Architecture.md l.58 requires them.

Nothing about C1 has been measured. Every cost, range and rank above is a derivation or an estimate.
