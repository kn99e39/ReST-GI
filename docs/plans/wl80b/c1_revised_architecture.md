# C1 revised architecture (or no-revision statement) (Part G)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

## No revision proposed

### 1. Reasons

**1. Every defect is fixed by numerics, not by architecture.**
- Parts B to E found six BLOCKER / BLOCKER-S items (BL-1 to BL-6) plus tiling and watertightness gaps. Each comes from a recorded implementation choice: the pFFT far field, an unspecified clamp, warm-started or accelerated solves, the archived random B rays, and float32 FFT arithmetic.
- None comes from the CP factorisation [ND, Part B0].
- The fixes are standard: ray-telescoped W with visibility, P2* or no clamp, cold-start fixed-count Neumann in float64, a shared deterministic orbit-symmetric B/F_in ray set, offset stratification, and per-beam point layers composed into G1 and S1 at build time.
- They are listed as REVISE / REMOVE in the decision records. They are specification corrections inside a known method family, so the stop rule "no architectural value beyond existing numerical techniques" applies.

**2. The main mechanism is already known.**
- First-collision source, then a scalar CP resolvent, then last-flight escape: Marleau-Hebert; DRAGON EXCELT; Alcouffe 1985; Fan-An-Ying 2019; Ren-Zhang-Zhong.
- The port/bin response-operator role is covered by response-matrix and interface-current methods with directional interface fluxes, and by zonal / exchange-factor methods with geometry-only exchange shared across channels [PA, Part F2].
- No element reaches PLAUSIBLY DISTINCT.

**3. The defects that remain are resolution and affordability limits.**
- Lateral flat-source D inflation is +31 % at 1.25 mm, and the error is second order [ND, Part C4.1].
- Explicit L1 needs 13-19 GB [ND, Part E1].
- The red solve needs about 525-663 cold sweeps in the invariant-exact mode [ND, Parts B8 and B11].
- Removing them needs more state or matrix-free compute: the "unreasonable state or rebuilding" stop rule. Under CD HIGH-1, a converging miss here is affordability-limited, not an architectural defect.

**4. Any learned addition would replace C1, not refine it.**
- A learned closure on W is WL77A H3, rejected on four grounds (coordinator_decision.md l.15 [HM]); the Horn-A "vacuous closure" ground is unrefuted.
- A learned scalar resolvent between exact flights is WL80A C2 / WL77A H2, covered by Neural Green's Functions, Element Learning and DeepRTE.
- In this physical scope (g = 0, homogeneous, index-matched), exact physics already supplies the whole operator. A network would add no information, only a failure mode. The rules forbid adding one for preference.

**5. The research question is untouched.**
- The question is whether a *learned* intrinsic K transfers to unseen sources and geometry edits.
- C1 at any discretisation answers "the MC-defined target K is computable from geometry on the scored strata" (CD HIGH-2). It does not attribute the learned-family failures (CD l.16), and it has privileged physical information the learned arms lacked (WL77A B4).

### 2. C1's best classification

- **What it is.** A learning-free deterministic integral-transport operator: flat-source collision probability with an exact first-collision source and complement-defined escape, at port/bin first-exit resolution. It is the WL77A H3 θ = 0 operator class with a revised discretisation (CD l.18) [HM].
- **Physics.** Exact in the continuum under g = 0, a homogeneous medium and an index-matched boundary [EP]. All accuracy statements are untested [UH].
- **Admissible role today.** Only as a diagnostic instrument / deterministic analogue of the WL56 MC oracle K (56_* l.30; Architecture.md l.87 oracle role; CD l.17).
- **As an FNTO K.**
  - Not admissible as the FNTO *learned* K under contract v0.1 (l.6, 26, 103, 180).
  - Ambiguous under the current Architecture.md (l.17 does not say "learned").
  - Admissibility, teacher use and value comparators are human decisions (Part F5). None is taken or recommended here.
- **Novelty.** None as a method. Any value would be an empirical operating point, and nothing is measured.

### 3. Corrected specification for the qualified diagnostic role

These are not a revised architecture. They are the minimum conditions under which C1's own invariants can be established (Part B13).

**Operator.**
```
out = B u + ω_c S1 u + ω_c Φout ψ
ψ   = Σ_{k=0}^{N-1} (ω_c W)^k ω_c G1 u      (cold start, fixed N, float64 accumulation)
```
- G1 = W1·F_in and S1 = Φ_pt·F_in are composed at build time from per-beam point quadrature with per-point rotated direction sets.
- W is full-chord ray-telescoped with in-medium visibility: explicit near field, matrix-free beyond it, no pFFT.
- Φout escape totals are the exact complement 1 − 1ᵀW, so no clamp exists. The P2* guard aborts if it binds.
- B and F_in are rebuilt on one shared deterministic orbit-symmetric ray set with no nodes on bin edges. The new d is declared and its difference from the reference d is reported.
- Cells are an exact tiling (offset layer surfaces); the intersector is watertight.
- Tallies follow the code convention: flux per flux, exit-triangle frame, no cosine factor.

**Reused-state contract (illumination change).**
- All of G (mesh, B, d, G1, S1, W, Φout) is reused bitwise.
- Only the right-hand side G1 u and the cold solve repeat. ψ is never warm-started in contract tests.

**Reused-state contract (geometry change).**

| Edit | What is reused | What is rebuilt |
|---|---|---|
| Rigid, separated by air | K factors bitwise, in the body frame | C, U, GI |
| Dent | Cells unchanged under a stretch map confined to the upper layers, which leaves unmoved bottom/side layers bitwise identical [ND suggestion; signed-change effect UH] | G within r_f ≈ 7-14 mm of moved geometry, plus all beams whose chords meet the thinned region (26-42 % of beams [ND]); then a global cold re-solve |
| Albedo | All factors | Global re-solve; refinement certification per ω |
| σ_t, thickness, topology, contact | — | Full rebuild; port correspondence unspecified (Architecture.md l.58) |

**Source transfer.**
- Structural. The source enters only through F_in u. Unseen-source accuracy is discretisation accuracy, not generalisation (Part C).

**Geometry-change cost model.**
- Counted only, none measured [ND, Parts E5-E6].
- About a third of full construction per dent, plus a 525-663-sweep red solve. That is about 0.7-1.0 s per single-light RGB application in contract mode, against the measured FIA anchor of 3.9 ms (76_* l.255 [HM]).
- An accelerated mode is permitted only for separately labelled speed studies.

**Learned component.** None.

**Falsification criterion (instrument role).** Predeclared before any MC score is read, in this order:
1. **Convention gate.** Half-space H-function albedo within 1 %, and single-scatter S1 against an MC single-scatter tally.
2. **Hard invariant tests.**
   - Per-column closure and partitions to fp precision.
   - Minimum entry ≥ −1e-9·max.
   - Superposition and zero input within 1e-6.
   - D4h within the stated roundoff tolerance.
   - Cavity-crossing mass = 0 on the dented pose.
3. **Strata gates** (abs_coarse 0.10, abs_fine16 0.15, con_coarse 0.25; dent signed change rel. ≤ 0.35 and corr ≥ 0.90) at the finest affordable level.
   - A miss with observed order < 0.5 is a C1 fail (convention / reference / lateral-bias floor).
   - A converging miss is affordability-limited (CD HIGH-1).
   - A pass shows only computability of the MC-defined K (CD HIGH-2).

Running this is not authorised by this record.
