# Dual-Architecture Verification — LSIT vs IDTS (read-only, adversarial)

**Status.**
- Non-numbered architecture-verification report, 2026-11-14, written in response to the owner's brief "Independent Dual-Architecture Verification".
- **Read-only.** No implementation, training, rendering, GPU job or experiment was run. No code, test, result, canonical document or historical worklog was changed.
- Neither candidate is adopted. This report does not choose a winner and does not merge the candidates.

**Method.**
- A multi-agent workflow (`wf_3a43aa54-890`, 8 read-only agents) ran on commit `4ed4ac30151497668e6cfca3db3ad8d0cbee3281`, branch `Selective_Recompute`:
  1. two evidence agents: a verified measurement ledger, and a code / physics fact sheet;
  2. four independent verifiers: for each candidate, one physics & contract lens and one representation / interface / runtime lens;
  3. one adversarial refuter per candidate, which re-checked every cited number against its source and tried to refute every claimed contradiction.
- The verbatim agent reports are in `docs/plans/dual_arch_verification_lsit_idts/`. This document is the synthesis.
- Where a refuter overturned or downgraded a verifier, the refuter's checked reading is used, and the disagreement is noted.

**Independence note.**
- LSIT was written by the same assistant lineage (2026-11-14, `docs/plans/DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md`). All agents were instructed to verify it adversarially and not to defer to its self-assessment. Several errors in the LSIT document were found and are recorded below. The LSIT document itself is unchanged (no silent rewrite).
- IDTS was verified only against its own definitional text (§1.2). Agents were forbidden to invent an implementation.

**Labels.**
- OBSERVED: a measurement verified in a named source.
- SUPPORTED INFERENCE (SUP).
- PLAUSIBLE HYPOTHESIS (PLAU).
- UNSUPPORTED.
- CONTRADICTED.
- [ARITH] marks arithmetic, never a measurement.

---

## 1. Candidate definitions

### 1.1 Candidate A — LSIT (Lobe-Split Intrinsic Transport)

The authoritative definition is `docs/plans/DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md` (found by title search). Its principal claims:
- **A1.** Sharp scattering is unnecessarily high-rank in absolute direction bins.
- **A2.** K = K_L (a reciprocal half-vector lobe, evaluated per ray) + K_S (a non-negative, low-rank 64-bin remainder).
- **A3.** Transport through the second scattering event (the table's L₂) is explicit.
- **A4.** A learned, geometry-conditioned, linear R_θ owns events ≥ 3. Its input is the explicit L₂ and it is trained on t − t2.
- **A5.** Sharp-lobe transport lies mostly within the explicit orders.
- **A6.** Source and receiver uses share one intrinsic kernel.

**Contract conflict D1.** Events 1–2 are explicit, against the 2026-11-13 owner decision that keeps explicit transport as a comparator only (`docs/Architecture.md` §1). This is treated as unresolved.

### 1.2 Candidate B — IDTS (Intrinsic–Directional Transport Separation), verbatim definition

> CORE HYPOTHESIS: The principal ECRO failure may arise because geometry-dependent incident directional information is encoded inside a nominally intrinsic learned local transport operator. Instead, separate:
> B1. Intrinsic material-local response K_phi — in a local material frame; shared across source and receiver uses; parameters and physical meaning unchanged under supported rigid edits; must represent sharp directional response without geometry-specific factor specialization; operator class NOT YET FIXED.
> B2. Geometry-dependent directional transport state z^t — incident directional information from the current sources, geometry, visibility and free-space propagation; must retain the angular information needed for sharp interreflection; representation NOT YET FIXED.
> B3. Dynamic propagation C^t — owns visibility and free-space transport affected by edits; recomputing / selectively updating C^t and the directional state must not require retraining K_phi.
> B4. Learned nonlocal / higher-order transport G_psi — retains a meaningful learned higher-order responsibility; not all second-order transport must become explicit; its relation to K_phi, C^t, z^t must be formalized before implementation.
> B5. Role-specific incident states allowed; role-specific intrinsic material laws not; the same intrinsic response must operate on physically equivalent incident distributions.
> B6. After a supported rigid edit: intrinsic parameters reused; learned weights fixed; geometry-dependent state updated; GI must respond correctly. Elastic deformation not included.
> CRITICAL LIMITATION: an architectural hypothesis, not a completed operator specification.

---

## 2. Verified evidence (selected; full ledger in `dual_arch_verification_lsit_idts/ledger.md`)

**Scope.** Every row is one room family, rigid box edits, three frozen ER materials (B, S1, S2), point lights (plus cone / two-light held-out sources), 512², 1,270 patches × 64 bins. E2 and E5 are **inspected** conditions. No admitted, never-inspected sharp-reader held-out exists: E3's region C has no panel receivers, E4 was not admitted, and E6 was never generated.

| id | Measurement (OBSERVED) | Source |
|---|---|---|
| V-1 | The patch × 64-bin table oracle reaches E2 T1 region C 0.0137 (noise 0.013). Learned ECRO v0 reaches 0.0717 / 0.0634 (rule 0.06). Broad stationary error is 0.004–0.006. | WL86 |
| V-2 | The single learned flight delivers 1.6–18 % of the light → panel → receiver family. The global term adds +0.02…+0.06 of h; WL92's decomposition gives +0.038…+0.096. | WL86A, WL92 |
| V-3 | S1's 64 × 64 frozen-ER operator needs rank 46–59 for 99 % flux energy. Its rank-16 residual is 0.52 / 0.46 / 0.46. NMF-16 keeps 0.47–0.54 of the mirror bin; the learned `a cᵀ` keeps 0.11–0.14. S2's rank-16 residual is 0.20–0.33. The floor needs rank 8 and Lambert rank 1. | WL92, `results/ecro_attr92/results/local.json` |
| V-4 | WL92 ladder, E2 region C (T1 / T0): learned single + true L₃₊ 0.124–0.128 / 0.083–0.098; R16 0.084 / 0.073; ENCB (exact M64, deployed draw) + true L₃₊ **0.045 / 0.052**; pooled ENCB 0.027 / 0.023; RS 0.048 / 0.044; EXP 0.041 / 0.048; ORA2 floor 0.013 / 0.010; true L₂ + learned global 0.052 / 0.064 and 0.028 / 0.041. | WL92 ladder |
| V-5 | Per-geometry free rank-16 fits reach E2 T1 0.029 (OWN-f8601), better than the exact composition ENCB (0.045). This is calibration around the deployed sampling shortfall, not physical representability. | WL92 free fits |
| V-6 | The explicit-vs-oracle gap is the 64-bin projection of the once-scattered sender radiance in the first gather (97–102 %), measured on the W1 / W2 workloads with sender-side relations. Per-ray exact first gather: C3UR 0.040 → 0.024 at 0.56–0.82× cost. The single draw fails held-out 1 / 3. | WL90, WL91 |
| V-7 | WL92B X_L2 (path-traced L₂ + OWN learned global trained on t − t2), E2 region C: T1 0.022 / 0.019, T0 0.016 / 0.015 (floor 0.013 / 0.010). On LSPOT the learned higher-order term is the weak point (X_L3 0.013 / 0.011 vs X_L2 0.020 / 0.029). | WL92B |
| V-8 | The learned panel `a cᵀ` changes 14.5 % for an intrinsically unchanged panel under a box move. The seed-to-seed difference (0.51) exceeds that geometry change. | WL92B |
| V-9 | WL93C ORA = M64·flight·M64·s + frozen OWN global (= WL92B X_M12 to 1e-7). Region C (ORA vs T0): E2 T1 0.030 / 0.033 vs 0.076 / 0.042; **E2 T0 0.059 / 0.058 vs 0.067 / 0.049**; E5 T0 0.015 / 0.014 vs 0.034 / 0.024; **E5 T1 0.031 / 0.031 vs 0.032 / 0.025**. Stage-isolated ORA − T0: receiver −0.0070 [−0.0076, −0.0064], source −0.40. Pooled single / whole: tie. | WL93C, `oracle_gate.json` |
| V-10 | WL93A–C: source supervision improves the source stage but worsens whole by 4.6–7.3 %. Untying returns whole to T0 parity; the capacity control was inert. Joint supervision loses the receiver gain. g00010's encoding discrepancy is genuine; its angular vs spatial cause is not attributed (spatial within-patch variation dominates q noise). | WL93A / B / C |
| V-11 | Frozen ER closure: 8 lobes (Lambert; 3 outer anisotropic GGX; 4 inner GGX evaluated at **refracted** directions with a 1/η² factor); height-correlated Smith G2; per-lobe Schlick; s(μ_i)s(μ_o) factors; passivity certified (softmax λ); reciprocity exact. S1: α 0.026–0.075, η_e 2.4955. Energy-carrying S1 lobes have r50 ≈ 0.12–0.16 rad. The sharpest lobe (r50 0.049) carries ≤ 0.004 albedo except near grazing. | `fnto_compiled83/closure83_dr.py`, `results/gm1_char85/results/census/fixture.json`, `mats86.npz` |
| V-12 | Costs: ECRO per edit 76–78 ms (state 21, operator 11), source-only 13–18 ms (the only measured learned advantage, WL87), per-frame 0.33 ms/spp. C3URX1 per edit 68–69 ms, source-only 20.6 ms, relation 20.8 MB + 12.6 MB buffer. | WL86, WL87, WL91 |

**Evidence corrections found by this verification** (they apply to earlier assistant summaries, not to the measurements):
1. **WL93C ORA renders.** ORA is better than T0 on 2 of 4 inspected conditions, not uniformly. On E2 T0 (s8602) and E5 T1 (s8602) it is worse. The LSIT document's E12 and its P-T3 bar cite only E2 T1.
2. **Learned-global compensation.** ORA's E2 T1 success (0.030) is partly compensation by the learned global term: the same exact-local single flight with the true L₃₊ (ENCB) scores 0.045 / 0.052. "Exact local + learned global is accurate" is therefore not supported at render level.
3. **WL92 per-geometry fits.** They beat the exact physical operator (V-5). The LSIT document's phrase "E5 shows the opposite (a shared physical operator is representable)" over-reads them.
4. **Locus of the E2 deficit.** On E2 the deficit is receiver-side state-ray sampling (RS ≈ ENCB). WL90's sender-side projection finding comes from W1 / W2, where sender-side relations were already used. On E2, any gain from an exact event 1 must come from sender-side placement, not from per-ray exactness alone.

---

## 3. LSIT verification (Task A)

| Item | Finding | Label |
|---|---|---|
| A-1 half-vector lobe vs frozen ER | The proposal says f_L = F(θd)D(θh)/(4 cos cos) "matches ER's own microfacet structure". **This is false for this closure.** Inner lobes are refracted (exterior width depends on θd). Each colour mixes lobes with different α and per-lobe Fresnel. G2 and s(μ) depend on θi and θo separately (on φd). Anisotropy reaches up to ≈ 12 %. A single product of one D and one F cannot be exact. An approximate fit (Σ_k F_k D_k) is untested. The refuter corrects the verifier's "5× width growth": that holds for an isolated inner lobe, while the composite S1 r50 narrows toward grazing (0.146 → 0.090). | CONTRADICTED (structural sentence); approximate adequacy PLAU |
| A-2 f = f_L + f_S, both ≥ 0, f_S low-rank | There is an obstruction. Positivity of f_S near the lobe peak requires f_L ≤ f pointwise, so the fit residual is pushed into the tails or into f_S. Operator rank is basis-invariant, but the **separation rank of the kernel function changes with the choice of variables**. The reparameterisation idea is mathematically meaningful; only LSIT's P1 wording ("become low-rank once reparameterised") is loose. | PLAU (untested) |
| A-3 energy / masking / layering | The proposal omits G2, the refracted inner lobes, s(μ) and any per-part albedo bound. The sum stays passive only if f_L ≤ f. Per-part passivity is not stated. | UNSUPPORTED (as specified) |
| A-4 explicit L₂ keeps sharp directional information | Event 1 is exact per light sample (WL90 mechanism). The **second event uses K_r lobe samples around incident-bin centres**. That is a binned projection of interreflected incident light at sharp receivers, the same class of loss WL90 measured for order 1. Its size at order 2 is unmeasured: WL90's "order ≥ 2 negligible" used the exact local direction, so it does not cover LSIT's bin-centre second event. | PLAU risk; not covered by evidence |
| A-5 further 64-bin losses | Further binning occurs at three places: (i) the receiver incident deposit, (ii) the bin-indexed second-event lobe relation, (iii) the output table. The output table is adequate (V-1). (i) and (ii) are unmeasured. | PLAU risk |
| A-6 neglect of lobe events at order ≥ 3 | This changes the target problem unless those events are owned somewhere. They are not dropped: R_θ is trained on t − t2, which contains them, so R_θ absorbs them. That contradicts LSIT's own statement that R_θ "never represents a sharp local lobe". The readout patch x2's own lobe at order ≥ 3 (sharp x2) is an unlisted sub-case. V-7 (X_L2, inspected E2) shows a learned order-≥ 3 term absorbing this content is empirically tolerable on E2. | Internal specification inconsistency (stands, downgraded: disclosed as A2 / R2) |
| A-7 R_θ inputs / class | Input L₂ (outgoing table) is physically sufficient for a linear map to L₃₊ if R_θ performs a flight. C4 names "flight(L₂)" via the receiver-side state rays, which reintroduces WL86A's receiver-sampling limit at order 3. Representability of sharp-x2 order-≥ 3 content by rank-16 factors is the WL92 problem again. | PLAU; learnability UNSUPPORTED |
| A-8 event ownership / readout | The LOW / table / R_θ partition has no double counting if the table excludes x2's once-scattered radiance (inherited WL86 contract). Identity "by construction" at the camera bounce holds only to the f = f_L + f_S tolerance. | SUP (with tolerance) |
| A-9 selective lobe-relation update | Feasible in principle (C3UR-style keyed rays). It needs segment regeneration or storage, rays starting on a sharp mover (688 k panel / 524 k sphere records), and horizon / sphere-frame rules for the bin-indexed relation, which are unspecified. | PLAU; underspecified |
| A-10 cost / memory | Storage (1,212,416 records [ARITH], 8.5–13 MB, or 37.6 MB with segments) is plausible. **Omitted:** per-source second-event scatter over 1.2 M records, E1^S gather, flight(L₂), R_θ application, per-edit recomputation of E1 / L₂. Estimated per edit ≈ 100–150 ms and per source ≈ 20–40 ms [PLAU; unmeasured]. That is near or over LSIT's own R4 limit (2× ECRO). Quality at about 19 k lobe rays per light is unmeasured; WL90 / WL91 used about 30× more. | PLAU; incomplete |

**Specification defects** (from the representation lens and the refuter):
- **Major: second-event semantics are ambiguous.** The bin-indexed lobe relation stores hits at *other* patches, but L₂[j,·] is j's *outgoing* table. Either the traced rays are unused, or sharp L₂ is delivered sender-side, which would make R_θ's input an incident field.
- The P-T2 "tag lobe-sampled vertices" estimator cannot measure the f_L / f_S split. It needs deterministic f_L/f weights at each vertex, and the measurement depends on P-T1.
- P-T3 and the Stage-1 "LSIT-oracle" use path-traced L₃₊, so they are not oracles of the LSIT contract (truncation is confounded with learnability).
- The P-T3 bar compares with ORA (which includes a learned global term). The matched learning-free comparator is ENCB-type + true L₃₊ (0.045 / 0.052).
- The C1 split is **not necessary for events 1–2**. Exact per-ray ER evaluation with ER's own sampler already exists (WL87 W_ray, WL90 X1). P-T1 therefore gates the least necessary component.

**Restrictive assumptions.** Single sharp panel plus one glossy sphere; isotropic-enough lobes; point lights; the two-light interface; patch-scale content; one room family.

## 4. IDTS verification (Task B)

| Item | Finding | Label |
|---|---|---|
| B-1 separability | Local scattering is a linear operator on the incident angular field in the local (point) frame. Physically, an intrinsic K exists and is geometry-invariant. ECRO is geometry-conditioned by construction (`model86.py:95-107`). Whether that conditioning is the *principal* failure does not follow from construction; ranking requires evidence. | SUP that separation is physically possible; "principal" is PLAU |
| B-2 required z^t information | Sharp transport needs incident angular resolution of about the energy-carrying lobe width (S1 r50 0.12–0.16 rad). That gives ≈ 250–440 directions at Δ = r50 and ≈ 1,000–1,700 at r50/2 [ARITH], consistent with WL90's 1,024. The verifiers' 2,600–10,500 estimates were based on an energetically negligible lobe and are rejected by the refuter. Point sources and curved receivers also need spatio-angular resolution. | PLAU (quantified obligation) |
| B-3 identifiability | Not identifiable as stated. A learned K_phi and a learned z^t or G_psi can trade content. V-8 (seed difference > geometry difference) and WL92B co-adaptation show the risk. Identifiability needs material-only or stage targets for K_phi, a physically defined z^t (computed by C^t from physical sender quantities, invariant to the receiver material), and G_psi trained on its own path set. | Obligation; UNDEFINED in text |
| B-4 near-specular without geometry specialisation | Any material-shared K_phi for S1 at 64 bins must not be a rank ≈ 16 factorisation (V-3). That excludes instances, not IDTS. The class must be high-rank or structured (shift / convolution-like in mirror-relative variables). S2 is milder (rank-16 residual 0.20–0.33). | Obligation |
| B-5 source / receiver consistency | It follows from ownership only if K_phi's signature is (material state, incident field in the material frame) → outgoing field, with no position, geometry or role inputs, and both roles share one state space, basis and unit. Two different fields with the same binned state are *not* "physically equivalent". The verifier's claim that B5 is violated was a misreading (refuter); the issue is z^t resolution (B2). | Obligation |
| B-6 geometry-conditioned G_psi | It can reintroduce scene-context entanglement (WL86A blind global compensation; V-9 correction 2). The direct G-specific evidence is V-4 "true L₂ + learned global" (0.052 / 0.064 and 0.028 / 0.041), V-7, and LSPOT X_L3. The whole-ECRO region-C slopes are not G-specific evidence. | PLAU risk |
| B-7 interface sufficiency for higher orders | Undefined. Which of z^t or K outputs G_psi consumes, whether G_psi routes free space through C^t, and who owns sharp chains are all unstated. Event-numbering correction: light → panel → sphere → floor is table event 2 in the ECRO convention (it belongs to the single flight, not to an order-≥ 3 G_psi). IDTS must state whether it inherits LOW. | UNDEFINED |
| B-8 cost | Per-patch angular state at 1,024 directions is 15.6 MB for all patches (0.9 MB sharp-only). Relation rays N·D·κ at D = 1,024, κ = 8 is 10.4 M. A full rebuild takes ≈ 0.42 s [PLAU from WL86 throughput]. A dense K_phi at D = 1,024 is 12.6 MB per material; at D ≈ 3,350, 135 MB per material, which needs a structured class. A dense-N² G_psi must be excluded. | PLAU bounds |
| B-9 rigid edits without retraining | Consistent as stated (B3 / B6). The recomputation scope is undefined: which C^t elements, source states, propagated z^t orders, and the sampling rules that avoid aliasing in visibility updates. | Obligation |
| B-10 missing definitions | K_phi class / resolution / frame / relation to ER; z^t carrier, resolution, per-order indexing and construction from lights; the ownership partition (incl. sharp chains, LOW); G_psi inputs, outputs, source linearity, feedback, scaling and exclusive target; the B5 equivalence definition; B6 acceptance thresholds. | Blocks implementation-level verification |

**No claimed contradiction survives as a contradiction of the IDTS text.** The refuter downgraded all of them to obligations, exclusions of instances, or conditional dependencies:
- the rank-16 class;
- 64-bin z^t vs B5;
- curved patches;
- the owner-decision conflict, which binds only once K_phi is fixed to ER or M_D with exact lower orders;
- `Architecture.md` §3 "future modifications to ER require a new, separately justified hypothesis", which binds if K_phi is a new learned law.

**Fourteen specification obligations** (from the physics lens) are listed in `dual_arch_verification_lsit_idts/IDTS_ver0.md` §4. They include the K_phi signature, quantified resolution at every sharp interface, one state space for both roles, a physical definition of z^t, a written path partition, identifying supervision, invariance tests, curved-surface frames, the rigid-edit procedure, the relation to ER and the owner decision, and an admitted, never-inspected held-out.

## 5. Physical and mathematical contract audit (Task C)

| Contract | LSIT | IDTS |
|---|---|---|
| Reciprocity | f_L reciprocal by form; f_S reciprocal if f is; R_θ none (composite) | UNDEFINED (K_phi class open; obligation) |
| Non-negativity / energy | f_L ≥ 0 by form; f_S ≥ 0 only if f_L ≤ f pointwise (a fit constraint); per-part albedo bound not stated; R_θ gain > 1 not excluded (as in ECRO v0) | UNDEFINED (obligation: linear, ≥ 0, passive K; ρ < 1 of the composed operator incl. G_psi) |
| Source linearity / zero | yes (explicit linear + linear R_θ) | UNDEFINED for G_psi |
| Event-order correctness | defined (LOW / L₂ / L₃₊), but lobe events ≥ 3 are owned by R_θ contrary to its stated contract | UNDEFINED (partition not written) |
| No double counting | holds given the WL86 table contract | UNDEFINED |
| Material vs world frame | patch frame for M^S; sphere / curved handling of the bin-indexed relation unspecified | obligation (point or sub-patch frames on curved sharp materials) |
| Source / receiver identity | identical K at orders 1–2; **order-dependent response** (K at ≤ 2, learned a·c at ≥ 3) vs `Architecture.md` §2 single response rule — contract item | holds by B5 if one state space is defined; else UNDEFINED |
| Sharp information preservation | exact at event 1; binned at event 2 incident (unmeasured) | quantified-resolution obligation |
| Interface sufficiency | L₂ → R_θ sufficient if R_θ flies; receiver-side flight reintroduces sampling limit | UNDEFINED |
| Edit invariance of intrinsic parameters | yes (material-only split) | yes by B1 / B6 (stated) |
| Dynamic visibility correctness | selective re-trace plausible; sharp-mover and segment handling unspecified | undefined procedure |

**Counterexamples (analytical).**
- **Two facing mirrors.** LSIT: chained lobe events at order ≥ 3 are absorbed by R_θ (bias); explicit only to order 2. IDTS: undefined ownership.
- **Sharp → sharp → diffuse.** This is table event 2 in the ECRO convention, so LSIT's explicit L₂ covers it if the second-event semantics are fixed. IDTS must state its convention.
- **Narrow directional source.** For an ideal mirror, deployed bins map exactly bin-to-bin (φ → φ + π is an 8-sector shift); bin-level output is identical and exact. Loss appears for bin-width lobes (in-bin spill) and downstream when the flight resolves the in-bin direction (the WL90 locus). Both candidates need sub-bin information at event 1. LSIT has it via per-light directions; IDTS via a z^t resolution obligation.
- **Curved specular surface.** One patch frame convolves ER with the in-patch normal distribution [ARITH: cap band 0–60°], giving a geometry-dependent, rigid-invariant patch operator. LSIT needs per-point emission, which is stated for event 1 only. IDTS needs per-point frames (obligation).
- **Identical material, different geometry.** LSIT's split is material-only (passes). IDTS passes by B1 if the invariance tests are added.
- **Different fields, same binned state; g00010-type ambiguity.** LSIT's event 1 is unaffected (exact directions), its event 2 is affected. IDTS is unaffected only if z^t is sub-bin.

## 6. Claim / evidence matrix (Task D)

| Claim | Supporting | Weakening / contradicting | Unestablished | Label |
|---|---|---|---|---|
| LSIT A1 (high rank in absolute bins) | V-3 | S2 milder; separation rank only meaningful in reparameterised variables | low rank after reparameterisation for this closure | SUP (rank observed); reparameterised low-rank PLAU |
| LSIT A2 (K_L + K_S) | — | V-11 closure structure contradicts the stated form | approximate split with f_S ≥ 0 | CONTRADICTED (exact form) / PLAU (approx.) |
| LSIT A3 (explicit L₂) | V-6 (W1 / W2), V-7 (X_L2 tolerable), V-4 (ENCB) | event 2 binned; E2 deficit is receiver-side sampling | gain at 19 k rays / P = 16 | SUP (allocation evidence, inspected) / PLAU (quality at budget) |
| LSIT A4 (learned ≥ 3) | V-1 broad success, V-7 | LSPOT X_L3 weak; target contains lobe events | learnability from L₂ input | PLAU |
| LSIT A5 (sharp transport within orders ≤ 2) | WL92: 3–5-event share 0.46–0.47 of region-C h (all paths, not the lobe share) | sharp-x2 sub-case | the lobe share at order ≥ 3 | UNSUPPORTED (unmeasured) |
| LSIT A6 (shared kernel) | by construction at orders ≤ 2 | order-dependent rule at ≥ 3 | — | SUP (≤ 2); contract item (≥ 3) |
| IDTS core (principal failure = geometry-encoded incident info in local operator) | V-4 (local operator is the largest step on inspected E2: rank class ≈ −0.04, learning ≈ −0.04); V-9 stage-isolated exact beats learned; V-8 | pooled ties (g00010); seed difference > geometry difference (non-identifiability) | *why* the learned operator errs: geometry encoding vs generic inaccuracy vs non-identifiability | PLAU (component facts SUP; causal ranking UNSUPPORTED) |
| IDTS B1 (intrinsic sharp K without specialisation) | physics | V-3 excludes rank-16; V-5 (per-geometry fits work via specialisation) | an expressive, intrinsic, learnable class | PLAU |
| IDTS B2 (directional state) | V-6 | storage / resolution cost (B-8) | carrier and resolution | PLAU |
| IDTS B3 / B6 (reuse without retraining) | WL86 selective-update contract | — | recomputation scope | SUP in principle |
| IDTS B4 (meaningful learned higher-order) | V-1, V-7 | V-9 correction 2 (compensation) | interface, ownership | PLAU |
| IDTS B5 | physics | — | state-space definition | SUP in principle, UNDEFINED |

WL93C's oracle success is **not** proof of learnability for either candidate, and smoke negatives are **not** proof of unrepresentability. Both were explicitly checked in the agents' evidence-use audits.

## 7. Symmetric comparison (Task E)

Architectural merit and design maturity are separated in the last two rows.

| Criterion | LSIT | IDTS |
|---|---|---|
| Physical correctness | Explicit orders 1–2 correct to the split tolerance; f_L form contradicted for this ER; per-part energy unstated; R_θ absorbs truncated lobe events | Correct in principle (B1–B5 physically sound); constraints not specified |
| Information preservation | Exact event 1 (per light sample); binned event-2 incidence (unmeasured loss) | Obligation only (z^t resolution) |
| Sharp interreflection expressiveness | Orders ≤ 2 yes; ≥ 3 via learned absorption | Depends on unfixed K_phi / z^t |
| Higher-order capability | R_θ (≥ 3), learned, inspected support (X_L2) | G_psi, undefined |
| Ownership (material / geometry) | Clear for K / C; order-dependent response rule (contract item) | Clear in intent; partition undefined |
| Reuse after edits | Defined (K, R_θ reused; state + lobe relation updated) | Stated, procedure undefined |
| Computational complexity | Per edit ≈ 100–150 ms, per source ≈ 20–40 ms [PLAU]; near R4 | ≈ 0.4 s full relation rebuild at 1,024 dirs [PLAU]; unknown overall |
| Memory | ≈ 8.5–38 MB lobe relation [ARITH] | 1–16 MB state; 12.6 MB+ per material K at 1,024 [ARITH] |
| Learning difficulty | Moderate (R_θ only) | High (intrinsic sharp K + G_psi, identifiability) |
| Implementation complexity | Moderate; specification defects fixable | Undetermined |
| Debuggability / testability | High (component oracles), after fixing P-T2 / P-T3 | Core causal claim testable now (§8.2); architecture not |
| Risk of repeating ECRO mechanisms | Lower at orders ≤ 2; same risk inside R_θ at ≥ 3 sharp-x2 | Same risks unless obligations met (rank-16 class, co-adaptation, blind G) |
| Alignment with current research decisions | **Conflicts** (D1; order-dependent response rule) | Aligned in intent; conflict deferred (binds once K_phi = ER / M_D with exact lower orders) |
| Design maturity | Concrete but with major specification defects | Hypothesis-level |

**Neither candidate is clearly better supported.**
- LSIT's allocation has more direct (inspected) evidence: X_L2, ENCB, WL90.
- IDTS's core diagnosis is consistent with the measured component facts, but its causal ranking is unestablished.
- They answer different questions: LSIT reallocates *where* learning is used; IDTS asks *what* the learned local operator wrongly owns.

## 8. Critical contradictions (standing after adversarial refutation)

**LSIT** (all are specification-level, not structural):
1. R_θ is declared not to represent lobe events at order ≥ 3, but its target t − t2 contains them.
2. "f_L matches ER's own microfacet structure" is factually false for this closure (V-11).
3. Second-event semantics are ambiguous (traced hits vs the outgoing table).
4. The P-T2 estimator is ill-defined; P-T3 and the Stage-1 oracle are not contract-compliant; the P-T3 bar is mismatched and cherry-picks E2 T1.
5. Contract items: D1 (explicit orders 1–2), and the order-dependent response rule vs `Architecture.md` §2.

**IDTS:** none against its text. Fourteen open obligations (§4).

## 9. Minimal falsification designs (future; nothing authorised)

### 9.1 LSIT

**Stage 0 (learning-free; after D1 and after fixing the specification defects).** A contract-compliant composition ladder on region C (rendered E_H, common random numbers):
- conditions: E2 T0 / T1 and E5 T0 / T1 (inspected), plus one newly admitted held-out edit;
- every condition reported separately;
- region C split by x2 material (sharp vs non-sharp) and by receiver type.

| Arm | L₂ | L₃₊ |
|---|---|---|
| (a) ORA2 (floor) | path-traced | path-traced |
| (b) matched binned | ENCB-type M64 composition, deployed draw | path-traced |
| (c) LSIT-L2 | as specified (P = 16, K_r = 16), second-event semantics frozen, sphere frame fixed, ≥ 16 independent realisations | path-traced |
| (d) contract-compliant LSIT oracle | (c) | path-traced with every f_L event at table order ≥ 3 removed by deterministic f_S/f vertex weights, incl. x2's vertex |
| (e) no-split control | (c) with exact per-ray ER (ER's sampler, f/pdf), no M^S | path-traced |

**Falsified if:**
- (d) > 0.06 in any condition incl. held-out (or below a predeclared realisation pass rate);
- (d) is not better than (b) by > max(noise, 0.005) on both E2 cases and held-out;
- the truncation cost (d) − (c) carries more than a predeclared share of (d)'s error on the sharp-x2 subset;
- (d)'s C-region edit slope lies outside [0.75, 1.25].

(e) ≤ (c) falsifies the necessity of the C1 split, not the allocation.

**Inconclusive if:**
- f_S ≥ 0 is infeasible;
- the second-event semantics are not frozen;
- the held-out is not admitted or lacks panel / sphere receivers;
- differences fall within the noise bound (0.013 / 0.010) or the realisation spread;
- the sharp-x2 subset is too small;
- the truncation estimate changes across two predeclared split fits.

**Stage 1 (smoke, owner approval).**
- Learned R_θ (input L₂, target t − t2) vs arm (d) and vs WL93A T0, at WL93A scale.
- "R_θ worse than (d) beyond margin" is attributed to learnability only because (d) is contract-compliant.

**Stage 2.** A separate owner decision.

### 9.2 IDTS

**Stage 0 now (tests only the core causal claim; independent of the unfixed classes): incident-span decomposition of frozen learned local operators.**
- Inputs: T0 / OWN / WL86 checkpoints; panel and sphere patches; E2 T0 / T1 plus the 55 validation geometries; source use on s16 and receiver use on M64(flight(q)).
- Steps:
  1. Build each patch's realised incident subspace V_g,i from a predeclared training-distribution light set.
  2. On fields from other lights, split the learned-operator error against M64 into an in-span part (A − M)P_V and an out-of-span part (A − M)(I − P_V).
  3. Compare the out-of-span difference between T0 and T1 geometries with the difference between seeds.
  4. Render check with path-traced L₃₊ (avoids learned-global compensation): substitute A' = M P_V + A(I − P_V) at both uses, and report the share of the learned-single excess (0.124 → ENCB 0.045) removed.
- **Falsified for this lineage if** the in-span error carries most of the excess (A' ≈ ENCB), and/or out-of-span differences are seed-dominated rather than geometry-dominated. That would indicate generic inaccuracy or non-identifiability, not geometry-encoded incident information.
- **Supported if** the in-span error is near M64's floor and the excess is out-of-span and tracks geometry.
- **Inconclusive if:**
  - dim V → 64;
  - heavy-tailed q noise (report per geometry with a predeclared exclusion rule);
  - tied-use co-adaptation (report per use and both uses);
  - M64 is not truth for sub-bin sources;
  - deployed-draw sampling (compare against ENCB, not ORA2);
  - low power (42 + 32 patches, 2 seeds);
  - inspected-only renders.

**Stage 0 of the architecture (after specification).**
- The chosen K_phi class must fit M_D (material-only) at the chosen D for S1 / S2 / B within predeclared residual, mirror-retention and passivity limits.
- Invariance tests: same z → same output across geometries and positions; receiver-material swap leaves z^t unchanged; source / receiver agreement on equal z.
- Exact composition at the chosen D against ENCB and the WL86 oracle, per geometry.
- Share of order-≥ 3 sharp chains.
- Storage and update cost against WL86 / 87 / 91.

**Stage 1 / 2.** Only after specification and owner decisions.

### 9.3 Patterned qualitative evaluation (both candidates)

**Primary pattern.**
- A patch-aligned checker on a Lambert receiver (25 cm cells on the back wall or a box face; albedo 0.3 / 0.8), read through the near-mirror panel.
- It exposes reflected-pattern misplacement, directional blur and sender-side placement errors.
- Patch alignment keeps the piecewise-constant table exact for the texture.

**Additional conditions.**
- A sub-patch checker tests the container, not the operator; keep it only as a labelled separate condition.
- The 2 × 2 cone checker (WL93B) is limited by the two-light interface.

**Controls.**
- The untextured room remains the quantitative control.
- Texture changes reflected energy and higher-order transport, so textured results are compared only with their own path-traced references.

## 10. Unresolved questions / owner decisions

- **LSIT D1.** Are explicit events 1–2 admissible inside the published method?
- **LSIT order-dependent response.** Does a learned higher-order composite at order ≥ 3 count as a second "response rule" under `Architecture.md` §2?
- **LSIT second-event semantics.** Traced sender-side delivery vs outgoing table; the C1 split's necessity given exact per-ray ER.
- **IDTS specification.** The fourteen obligations, especially the K_phi class and resolution, z^t carrier, path partition, G_psi interface, and the relation to ER (`Architecture.md` §3 on ER modifications).
- **Admitted held-out.** Both candidates need an admitted, never-inspected sharp-reader held-out (none exists).
- **Cause of the learned local error.** Geometry encoding, generic inaccuracy or non-identifiability: IDTS Stage 0 is the only proposed check that discriminates these without training.
- **g00010.** Its spatial vs angular cause is unattributed.

## 11. Verdicts

| Candidate | Verdict | Reason |
|---|---|---|
| **LSIT** | **NEEDS_CONTRACT_CLARIFICATION** | D1 and the order-dependent response rule conflict with current decisions. Even if they are cleared, LSIT would be NEEDS_SPECIFICATION as written (second-event semantics, P-T2 / P-T3 / oracle defects, an inconsistent R_θ ownership claim, a false C1 structure claim). Not STRUCTURALLY_UNJUSTIFIED: its allocation has direct inspected support (X_L2, ENCB, WL90). |
| **IDTS** | **NEEDS_SPECIFICATION** | No contradiction of its text; physically sound intent. Core causal claim plausible but unranked. The architecture is not verifiable at implementation level until K_phi, z^t, the path partition and G_psi are fixed. Its core causal claim is falsifiable now (§9.2) on frozen checkpoints without training. The owner-decision conflict is deferred, not absent. |

- **Is one meaningfully better supported?** No.
- **Do both remain viable?** Yes.
- **Is either specified enough for implementation?** **Neither.**

## 12. Reproducibility

- **Commit inspected:** `4ed4ac30151497668e6cfca3db3ad8d0cbee3281` (Selective_Recompute). The working tree was clean before this report.
- **Workflow:** `wf_3a43aa54-890` (8 agents, read-only; 155 tool uses).
- **Verbatim agent outputs:** `docs/plans/dual_arch_verification_lsit_idts/` (ledger, facts, LSIT / IDTS verifier and refuter reports).
- **Sources read directly by the agents:**
  - `docs/Architecture.md` §1–3, §8–9.
  - Worklogs: WL86, 86A, 87, 88, 89 (lines cited by the refuter), 90, 91, 92, 92A, 92B, 93, 93A, 93B, 93C.
  - The LSIT proposal.
  - Code: `experiments/ecro86/{model86,scene86,pt86,mats86}.py`, `experiments/gm1_explicit87/{quad87,xport87}.py`, `experiments/fnto_compiled83/closure83_dr.py` (+ `fnto_compiled82/closure82.py`, `fnto_compiled80c/closure80c.py`), `experiments/ecro_stage93*/`, `experiments/ecro_attr92/diag92.py`.
  - Results: `results/ecro_attr92/results/local.json`, `results/ecro_stage93c/oracle_gate.json`, `results/gm1_char85/results/census/fixture.json`, `experiments/ecro86/data/mats86.npz`, and WL86–93C result JSONs cited in the ledger.
- **Limitations:**
  - WL86B and the earlier design panels were not read (independence).
  - WL89 was read only at cited lines.
  - All runtime figures marked [PLAU] are extrapolations from measured throughputs.
  - Every render-level comparison across batches (WL92 vs WL93C) uses separate runs: common random numbers apply only within a batch.
