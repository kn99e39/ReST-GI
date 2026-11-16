# WL98 designer report D3: systems and cost first

**Role.** Independent designer D3, batch WL98 (design-only). The starting angle is systems and cost. Nothing was implemented, trained, rendered or adopted.

**Independence.** I did not open `DESIGN_REVIEW.md` or any other file in this directory except this report.

**Date.** 2026-11-16. Branch `Selective_Recompute`, HEAD `3cf6973`.

**Labels.**
- [MEAS] a measured value, with its source.
- [AUDIT] a WL97 interpretation or adjudication.
- [CODE] read from frozen code.
- [ARITH] my arithmetic on recorded values. Script: `experiments/wl98_calc/D3/cost_model.py`; output: `results/wl98_calc/D3/cost_model.json`.
- [PRED] my new prediction.
- [HUM] a decision that belongs to the human research owner.

---

## 0. Summary

1. **Where the time goes [MEAS, WL97 E5 / D].** On the GM1 / CK2 prototype (N = 1,270, one light, J = 4, 512² × 64 spp):
   - **69–81 %** of every lifecycle is CPU-side implementation overhead.
   - The parts that are transport *work* are, in GPU-kernel ms:
     - frame: 22.7;
     - selective relation update: about 4.5;
     - q table (source projection, 20.8 M ER evaluations): 1.84;
     - one explicit order: 0.53.
   - The explicit ≥ 3-event tail is 3.3 ms wall of a 144 ms edit.
   - Per-frame cost is identical for ECRO and EXP, because both use a LOW path-traced prefix plus a table lookup [MEAS WL87].
2. **The ceiling depends on which role the learned part plays [ARITH].** The WL96A resolvent-only bound (about 1.02–1.04) is role-specific, as WL97 D-04 already stated. I recomputed the ceilings for the three roles a learned operator can take on GM1, under the predeclared and engineered conventions (§2.3):
   - **Resolvent-only:** 1.02–1.16.
   - **Full ECRO-type operator** (q table and all orders replaced; factors resident):
     - edit → frame: 1.11 (kernel basis) to 1.20 (wall);
     - source → frame without a frame: up to 4.2–4.7.
   - **Full operator with a per-edit generator:** edit → frame **0.91–0.97 on the kernel basis**. It reaches 1.15–1.29 only on the overhead-dominated wall basis.
3. **Reading for geometry edits on GM1 [PRED].** At this scale, any learned role that must regenerate geometry-conditioned factors per edit (rule reuse, ECRO-style) has no geometry-edit cost advantage on an engineered implementation. Its only lever is the source-only update, which is not the brief's question.
   - The ECRO generator also scales as O(N²) [CODE `model86.py:83-89`]: it builds a **dense N × N form-factor matrix** for message passing. That is 6 MB at N = 1,270, 0.5 GB at 11,430 and 38 GB at 100 k.
4. **Ownership fact behind (3) [PRED, from the FNTO contract plus WL96 K1 / WL52].** In the scene86 / GM1 family every object is planar or convex. A learned representation reused across a rigid edit can therefore own only two things there:
   - (a) local surface response. That is ER, which is already exact and compact, and learning it again would be BRDF compression.
   - (b) free-space transport between bodies whose relative pose changes. That is C, which is dynamic. Learning it is the WL52 class.

   The amortisable, rigid-invariant, non-local transport that the central hypothesis needs **does not exist in this workload family**. CK2 can diagnose ECRO's training and binning failures (H1 / H2). It cannot carry a positive H3 result for any learned candidate.
5. **Where learned reuse can pay [PRED].** Transport whose geometry is rigid-invariant:
   - the **self-transport of non-convex rigid bodies**: glossy cavities, brackets, grooves, cups, corner reflectors, multi-part rigid assemblies;
   - secondarily, the static environment between edits.

   There the explicit comparator must redo many intra-body bounces at high angular resolution every time the incident field at the body changes, which includes every edit anywhere in the scene. A learned operator compiled once per body and reused under rigid motion replaces that work. Its cost scales with incident samples, not with N_b · K_int · J_int.
6. **Candidates.**
   - **Primary: RB-DTO, Rigid-Body Directional Transfer Operator.** One shared learned model compiles, per rigid body, a passive measure-valued operator for the body's non-local self-transport. It has a learned sharp-lobe pushforward plus a low-rank broad channel, physically split by path tags. Inter-body C stays explicit and current; a self-path exclusivity certificate controls validity.
   - **Independent alternative: LER-Δ, Learned Environment Resolvent with explicit edit-delta iteration.** A frozen learned all-order directional resolvent of the static environment is corrected after each edit by sparse explicit Δ-transport through the dirty and new paths.
   - **Rejected:**
     - A as an ECRO-lineage continuous kernel;
     - B as a learned sparse multi-resolution angular table (its learning-free twin dominates);
     - C at patch level (equals ER compression);
     - online neural radiance caching (retrains);
     - learned inter-body C (WL52).
7. **Falsification plan (§5).** A cost-anatomy gate (Stage 0) runs **before any training**. It needs only the explicit comparator and can stop the batch on H3 grounds. Separate tracks follow:
   - **Track G** on GM1 / CK2 for H1 / H2 attribution of the existing ECRO failure: shared data, budget and reference;
   - **Track B** on a new admitted rigid-body workload family for the candidate and H3: shared reference and conventions only.
8. **Proposed verdict: DESIGN_READY_FOR_HUMAN_REVIEW**, conditional on two [HUM] decisions (§6, D1–D2). The first is the ownership refinement "certified rigid self-coupling". If the owner rejects it, the D3 verdict for the current workload family becomes **NO_DEFENSIBLE_CANDIDATE** (§7).

---

## 1. What WL97 separated, and how D3 uses it

WL97 separated three problems [AUDIT, COORD §2]. I keep them apart:

| # | problem | measured scope | what it means for a design |
|---|---|---|---|
| P1 | learned-sender training / fit allocation | CK2 learned S1 sender field rel L1 1.04–1.19; box-A delivery 0.52× / 1.63× (seed-signed); grazing ring-0 coverage 0.11 % of training lightings [MEAS E1, B-04]; an admissible passive rank-16 operator is exact on CK2's incident support [AUDIT J1] | a coverage / allocation hypothesis (H1) must be tested with the same architecture and coverage-balanced data before blaming the representation |
| P2 | fixed-angular-bin grazing delivery bias | solid-angle-mean 4×16 sender table over-delivers +40–50 % to CK2 box A; about half is the averaging convention (flux-conserving ≈ +20 %); receiver-dependent sign; essentially gone at 8×32 [MEAS E2 / E3, AUDIT J2] | a representation hypothesis (H2) needs flux-conserving and high-resolution binned **controls**, not only a learned alternative |
| P3 | no accuracy / cost headroom for the restricted resolvent-only configuration | CK2 static, oracle prefix, table space (floor table has ≈ 0.010 rel-L2 readout error); cost ceiling 1.02–1.04 under the frozen convention; overhead-dominated prototype [MEAS WL96A, E4, E5; AUDIT J4, J5] | H3 must be posed per *role* and per *convention*; the resolvent-only bound does not bind a full operator or a different amortisation target |

**Further facts used.**
- Exact direction plus hit point removes the binning bias in expectation. Single draws at the deployed budget give no average gain, and at least 16× more draws are needed [AUDIT J3].
- The learned ECRO local factors have rows with directional albedo > 1 [MEAS E1].
- No reference-validated dynamic edit exists after WL93D [AUDIT J7].
- M64 is biased at grazing incidence and is not physical ground truth [AUDIT J2 / J6].

---

## 2. Cost anatomy: where time and rays go in a dynamic-GI update

### 2.1 Component table (recorded, CK2, N = 1,270, RTX 3080 Ti)

Values are wall ms and GPU-kernel ms, as isolated medians from WL97 E5 run1, cross-checked against WL96A [MEAS].

| component | wall | kernel | rays / work | owner | needed by explicit? | needed by a learned operator? |
|---|---|---|---|---|---|---|
| scene / BVH update (`set_geom`) | 30.1 | 0.04 | numpy table rebuild 23.9 ms; BVH refresh inside 2.7 ms | geometry | yes | yes (any current-visibility method) |
| selective relation update | 18.8 | ≈ 4.5 | 49,936 dirty of 650,240 rays (7.7 %) re-traced; 29,881 change | C | yes | yes if C stays explicit (it must, §2.5) |
| relation coefficients (XOp) | 5.5 | 2.0 | constant W re-upload | C | yes (engineering) | — |
| q table (source projection) | 24.1 | 1.84 | 20,320 shadow rays + 20.8 M ER evaluations per light | sources × K_local | yes | a learned operator can replace the ER-evaluation half; shadow rays stay |
| one binned order | 1.10–1.9 | 0.53 | N·K = 650 k gathers | K∘C | yes | replaceable |
| three tail orders | 3.17 | 1.58 | | K∘C | yes | replaceable |
| frame 512² × 64 spp | 58.4 | 22.7 | LOW path-traced prefix + table lookup; ≈ 35.7 ms fixed per call | readout | yes | yes (identical, WL87) |
| ECRO factors (generator) | 4.35 | 4.20 | dense N × N form-factor message passing | learned | — | per edit (rule reuse) |
| ECRO apply | 0.76 | 0.74 | | learned | — | per source / edit |
| Λ rank-24 apply | 0.146 | 0.105 | | learned | — | per source / edit |

**Shares** [MEAS E5 §3.3]:
- **69–81 % of a lifecycle is CPU-side overhead**: single-threaded Python, Dr.Jit re-tracing and numpy.
- A linear fit across N puts **72–77 % of each component in an N-independent intercept** [AUDIT D-09].
- Every component grows about 1.7× for 4× N (scale91) [MEAS]. Only a full relation build, which no lifecycle event performs, grows faster (4.4×).

### 2.2 What determines a learned role's ceiling

The utility of a learned replacement is an Amdahl ratio U = L / (L − A + c):
- L is the lifecycle under a declared convention;
- A is the explicit work the learned role removes;
- c is the learned replacement's own cost.

For U ≥ u the net saving A − c must be at least L · (1 − 1/u). That is 9 % of the lifecycle for u = 1.1, 20 % for 1.25, 33 % for 1.5 and 50 % for 2 [ARITH].

### 2.3 Role-specific ceilings on GM1 (recorded components; `cost_model.json` `ceilings`) [ARITH]

| role (what learning replaces) | wall, edit → frame 64 | wall, edit, no frame | kernel, edit → frame 64 | kernel, edit, no frame | kernel, source → frame 4 spp | kernel, source, no frame |
|---|---|---|---|---|---|---|
| resolvent-only (WL96A) | 1.02 | 1.04 | 1.05 | 1.16 | 1.35 | 1.55 |
| full operator, factors resident (q ER-half + all orders) | 1.20 | 1.38 | 1.11 | 1.42 | 2.27 | 4.19 |
| full operator + per-edit generator (rule reuse) | 1.15 | 1.29 | **0.97** | **0.91** | n/a | n/a |

**Reading.**

1. **Overhead inflates both wall rows.** The wall rows are inflated by the explicit q table's 22 ms of Python tracing, which an engineered explicit comparator would not pay. On the kernel basis, a per-edit generator costs more (4.2 ms) than everything it replaces (3.95 ms).
2. **Geometry edits on GM1 have no learned cost lever.** No role that respects rigid-edit reuse *by rule regeneration* reaches 1.25 on an engineered basis. The source update does have one (2–4×): the WL87 finding, measured to survive overhead removal [AUDIT D-05].
3. **Rule reuse alone does not support the brief's hypothesis.** The brief asks for reuse after a **geometry edit**. "Reuse of weights with regenerated factors" (ECRO) pays the generator on every edit. At N = 1,270 that already costs more than the explicit transport it replaces. The generator's dense N × N message passing scales quadratically (6 MB / 0.4 GFLOP per round at N = 1,270; 0.5 GB / 33 GFLOP at 11,430; 38 GB at 100 k) [CODE + ARITH]. The explicit order is linear in N · K.

### 2.4 Sharp first-flight delivery: rays, variance and angular resolution

There are three explicit strategies for sharp delivery. Each has a known cost:
- **Receiver-uniform binned gather** (R0 / EXP):
  - cheap (0.53 ms kernel per order at N · K = 650 k);
  - biased at grazing under the solid-angle-mean convention (P2);
  - variance across relation draws of 0.024–0.077 E_H [MEAS WL87].
- **Exact direction per relation ray** (X1): unbiased in expectation. A single draw is heavy-tailed, and at least 16× the draws are needed [AUDIT J3]. That is about 16× the 650 k gathers, which is still only ≈ 8 ms of kernel at this N [ARITH, linear scaling].
- **Sender-side or lobe-directed deposition** (C3 / C3UR / light-side photons): removes about 90 % of the receiver-placement deficit at the same budget [MEAS WL87]. Its cost is about that of R0 [MEAS WL89].

Uniform angular binning that resolves a GGX lobe needs about α⁻² bins. My rule (four bins across a lobe of half-width 2α) gives [ARITH, `uniform_binned_scaling`]:

| α | bins | N = 1,270 | N = 10⁴ | N = 10⁵ |
|---|---|---|---|---|
| 0.1 | 199 | 1.7 ms / order, 0.02 GB | 13 ms, 0.13 GB | 130 ms, 1.3 GB |
| 0.03 | 2,222 | 18 ms, 0.18 GB | 144 ms, 1.4 GB | 1.4 s, 14 GB |
| 0.01 | 20,000 | 165 ms, 1.6 GB | 1.3 s, 13 GB | 13 s, 128 GB |

Times are per order on the kernel basis; GB is relation memory at 8 bytes per entry.

**Reading [PRED].**
- Uniform high-resolution binning is not how an engineered explicit comparator should treat sharp **single** flights. Sender-side deposition keeps their cost close to R0, so the single sharp flight is cheap explicitly at every scale.
- The α⁻² growth bites only when **sharp transport is iterated**: sharp-to-sharp chains, multi-bounce glossy cavities. There every order needs the full angular resolution, sender-side sampling must be repeated per order on a non-analytic incident distribution, and the variance compounds.
- That is the regime a converged, zero-variance learned response could amortise.

### 2.5 Which part of the transport a learned representation could amortise after a rigid edit with fixed weights

This classification uses FNTO physical ownership (Architecture §2–5):

| transport part | rigid-invariant across the edit? | explicit cost | learned + reused value | verdict |
|---|---|---|---|---|
| local surface response (one event at a point) | yes, in the material frame | ER exact, compact (69 floats) [MEAS WL96 line 79/176] | re-learning it is BRDF compression; adds error, no information [PRED, = WL96 K1] | not a target (brief forbids it) |
| free-space coupling between independently moving bodies (inter-body C) | **no**: visibility and relative pose change | relation update 7.7–22 % dirty pairs [MEAS D-07] | learning it = learned C: WL52 failed on cost (8–60×) and transitions | stays explicit (ownership, not a "send sharp to explicit" default) |
| source projection (light → first incident field) | no (lights and occluders move) | shadow rays + ER evaluations | can absorb the ER half (ECRO `c`); shadow rays stay | source-update lever only |
| broad ≥ 3-event tail | no, as a geometry-conditioned resolvent | 1.1 ms / order at N = 1,270, linear in N·K | WL96A: no accuracy / cost headroom on CK2 | small lever; grows only with J (D-03: J = 16 / 32 → 1.20 / 1.41 wall) |
| **self-transport of a rigid body** (paths that leave and re-enter the same rigid body through its own pockets of air) | **yes**, under rigid motion of that body, **if no foreign geometry enters its self-path region** | per update J_int × N_b × K_int, with K_int ∝ α⁻² for glossy cavities; must be redone whenever the body's incident field changes (every edit anywhere) | one compile per body; apply cost ∝ incident samples; converged (no per-update variance) | **primary lever** |
| static-environment transport between edits | yes, except paths invaded by the moved object | full re-sweep after every edit (radiance changes everywhere) | a frozen learned resolvent plus sparse explicit Δ-correction for invaded and new paths | **secondary lever** (alternative) |
| camera readout | no | identical for all table methods | none | shared |

In scene86 / GM1 every body is planar or convex [MEAS: `scene86.py` rectangles, boxes, sphere]. The rigid-body self-transport row is therefore **identically zero** there. The static-environment row is the room, which every box edit invades (the critic union counts 48–59 % of S1-cell pairs; actual dirty relation rays 7.7–8.1 %) [AUDIT D-07].

### 2.6 Regimes where no learned advantage is possible (explicit, before any training)

1. **Planar / convex scenes**, including scene86, GM1, CK2 and V3. Nothing rigid-invariant and non-local exists to amortise.
2. **Edits whose cost is dominated by geometry update, relation update and frame.** If L − A ≥ L / u for the declared u, no learned role reaches u even at zero cost. On GM1 the non-amortisable part is ≥ 88 % of an edit lifecycle under every frame-inclusive convention I computed.
3. **Learned roles that regenerate geometry-conditioned factors per edit** at N where generator cost ≥ the explicit orders it replaces. That covers GM1 at N = 1,270 on the kernel basis. With dense N² message passing it covers every larger N.
4. **Single sharp flights between separately moving bodies.** Explicit sender-side deposition is cheap, and the transport is dynamic C.
5. **Self-path regions frequently invaded.** Contact, objects moving into cavities, and the environment-as-body case under large occluders. The certificate fails, and the candidate falls back to explicit.
6. **Few intra-body bounces with broad materials** (J_int ≤ 3, α ≥ 0.2). The explicit intra-body sweep is already cheap; see the `body_regimes` row (2 bodies, 200 patches, K 512, J 3): U ≈ 1.0.
7. **Mirror-like chains (α ≲ 0.02).** Deterministic explicit chain tracing costs a few rays per incident sample. A learned map cannot beat that, and its lobes blur.

---

## 3. Candidates: complete contracts

The contract fields follow the brief. Every candidate assumes the frozen WL84 ER for local scattering, explicit current-geometry inter-body C, physical LOW (≤ 2 events) camera readout as in WL86, and no T1 data at runtime.

### 3.1 PRIMARY: RB-DTO, Rigid-Body Directional Transfer Operator (family C at body level, with an A-type broad companion)

**Learned physical meaning.** For a rigid body b, with surfaces S_b, ER materials and body frame F_b, let:
- K_b be its local ER scattering;
- C_self,b be the free-space transport from S_b back to S_b through air not occupied by any foreign geometry.

The exact body response is R_b = (I − K_b C_self,b)⁻¹ K_b.

RB-DTO learns only the **non-local part** N_b = R_b − K_b: radiance that leaves the body after at least one self-hit. The first local event stays exact ER, which is what keeps this from being BRDF compression.

N_b is a linear, non-negative, passive map from incident radiance on S_b (position in body coordinates, incoming direction in the body frame) to a measure on S_b × hemisphere: the exit position and outgoing direction of the first exit toward non-self space. It is source-independent and scene-independent. It is intrinsic to the body's geometry and materials under the certificate in "visibility interface" below.

**IO.**
- *Compile (once per rigid asset; repeated only on non-rigid edits).*
  - Input: the body mesh in F_b (surface samples or patch tokens with position, normal, tangent, area, ER material descriptor), plus a cached self-visibility structure.
  - Output: a per-body latent field (patch tokens z_p, about 64–128 floats each).
- *Apply (per coupled-solve iteration).*
  - Input: incident samples (x_i, ω_i, Φ_i) on S_b in F_b, delivered by explicit C_ext or the source projection.
  - Output:
    - (a) **sharp channel**: for each incident sample, K_s lobes (exit point and direction mean, anisotropic width, weight w_k), i.e. a pushforward measure;
    - (b) **broad channel**: a per-patch low-rank outgoing code (R_b channels × angular basis) accumulated from per-sample encodings.
  - Both are reduced into a per-patch output cache: a mixture-reduced lobe set (≤ M_p lobes) plus the broad code.
- *Query.* C_ext sender rays and camera LOW second vertices evaluate the cache at the exact continuous (x, ω) of the ray.

**Ownership.**
- ER owns every first local event.
- N_b owns all paths whose segments stay inside the body's certified self-path region, i.e. every C_self,b segment.
- Explicit C_ext owns every other free-space segment: a ray from S_b whose first hit is foreign geometry, or that escapes.
- External incidence and C_ext-delivered incidence use the same N_b, with no source branch.

N_b is therefore **non-local but body-local** (spatially displaced exits inside one body). It never spans two bodies. Disjointness: C_ext is defined as C − C_self,b for certified bodies. This is verified by a path-tag identity test: tracer self-hit paths versus N_b-tagged paths, with exactly one owner per segment (Architecture §2).

[HUM] This places free-space segments (C_self) inside a body operator under a certificate. It refines, and does not replace, the FNTO rule "object identity must not freeze exposed interreflection" (D1).

**Rigid reuse.**
- Under rigid motion T of body b, N_b is unchanged. Incident samples are mapped by T⁻¹, and outputs (positions, directions, lobe frames) are mapped by T. Equivariance is exact by construction, and is checked as a contract test to float tolerance.
- Stationary bodies whose surroundings change keep N_b unchanged. Only their incident field changes.
- Non-rigid edits, material changes and certificate failures invalidate N_b for that body.
- Weights are never retrained after an edit, and compiled latents are reused. The per-edit cost is the C_ext update plus the coupled solve. **No per-edit inference of the compiler.**

**Visibility interface.**
- Inter-body visibility is explicit and current: the relation update with dirty rays, as in R0 / C3UR, applied only to C_ext.
- Self-visibility is static in F_b. It is cached once per body (or implicit in the learned N_b) and is **not** dynamic visibility, because only the body's own geometry determines it.
- **Certificate.** Each body carries a conservative self-path region Ω_b: the union of AABBs or a BVH of the self-path segments, bounded by the convex hull minus the body. After an edit, every moved foreign primitive is tested against every Ω_b, and the moved body's Ω against the others. The cost is O(#moved primitives × #bodies) bounding-volume tests.
- **On violation.** That body's C_self becomes explicit for the invaded part. The fallback is ER plus explicit selective self-relation sweeps; the alternative is the Δ-correction of §3.2. The fallback is measured and reported, never hidden.
- Dynamic visibility is therefore **not** coupled into the intrinsic representation, which meets the brief's unsuitability criterion.

**Energy contract (measure defined).**
- Unit: radiant flux per incident sample.
- Each incident sample (x_i, ω_i) with flux Φ_i has an exact budget B(x_i, ω_i) = Φ_i · [ρ_ER(ω_i) − E_esc(x_i, ω_i)]:
  - ρ_ER is the directional albedo, computed exactly from ER;
  - E_esc is the flux of the first ER scattering that escapes without a self-hit. It is computed exactly from the cached self-visibility (ray casting at compile time), or bounded by it.
- The network outputs a simplex over {K_s sharp lobes, broad channel, absorbed / null}. Sharp lobes are probability densities over (area × projected solid angle) on S_b; the broad channel distributes its flux by a normalised code.
- Hence Σ outputs ≤ B ≤ Φ_i for every incident sample. Passivity holds by construction, row-wise in flux. Composition with sub-stochastic C_ext gives a spectral radius below 1.
- Output radiance comes from dividing cached flux densities by the projected area–solid-angle measure. This is the convention WL97 found undefined for ECRO's composed operator [AUDIT P0-1c].
- Reciprocity is **not** built in; it is a measured contract.

**Training target and data generation.**
- Procedurally generated non-convex rigid bodies: grooves, brackets, cups, open boxes, corner reflectors, troughs and multi-part rigid assemblies, with ER materials drawn from the frozen WL84 family.
- Incident samples are stratified over (x_i, ω_i), with **explicit grazing strata**: the cos θ_i ring-0 share is fixed by protocol, which is the direct remedy for P1's 0.11 %.
- The frozen-ER path tracer follows each incident ray inside the body and records every first exit after ≥ 1 self-hit: exit position, direction, throughput, event count and per-event lobe class.
- No scene, light, camera or other object enters the target. It is **intrinsic, source-independent and noise-controlled**: exit samples, not M64 tables.
- **Supervision.**
  - Sharp channel: weighted maximum likelihood (or a sliced-Wasserstein / energy distance) of the lobe mixture against exits tagged "all self-events sampled from lobes with α < α_split".
  - Broad channel: projected exit flux of the complementary tag.
  - Total flux: regressed against the exact budget minus tracer-measured absorption.
- Component targets are path-tagged, so each channel is supervised against its own physical component. This avoids the WL92 / 92B tied-factor compensation.
- [HUM] α_split, the body-generator distribution and the dataset size.

**Sharp mechanism.**
- Sharp glossy self-transport (α roughly 0.03–0.15) is carried by **continuous lobes whose position and width are predicted per incident sample**. Peak position and width are therefore not limited by a bin grid (P2 does not arise).
- No receiver-side sampling of the lobe is needed, because the exit measure is predicted directly. So the X1 single-draw variance (P3) does not arise inside the body.
- Exact Dirac transport (ideal mirrors, ballistic pass-through) is outside N_b by definition. The ER family has no Dirac lobe (S1 α_u ≈ 0.026–0.075 [MEAS WL96 l.79]); C_ext traces pass-through.
- This is **not** "sharp to explicit by default". Every sharp self-transport chain of the body is learned. Only inter-body free-space delivery is explicit, by ownership.

**Runtime and memory complexity.**
- *Compile.* One forward pass of the shared model over N_b tokens with **sparse** (k-NN or self-visibility-graph) attention: O(N_b · k · d²). A dense N_b × N_b form-factor matrix is forbidden by contract (§2.3). This is a one-off per asset, outside the edit critical path.
- *Apply* per outer iteration:
  - sharp channel: O(M_in · (t_query + K_s · t_reduce));
  - broad channel: O(M_in · R + N_b · R · n_basis).
  - M_in is the number of incident samples (about N_b × K_ext from the relation rays, or fewer with per-patch incident aggregation).
- *Query.* O(M_p) lobe evaluations per C_ext or camera query, versus one table fetch for a binned table. The frame cost rises by a bounded factor, which must be measured.
- *Memory.* Weights (shared), plus per body N_b · d latents, plus an output cache of N_b · (M_p · lobe parameters + R · n_basis). For N_b = 400, d = 128, M_p = 16 and about 10 floats per lobe that is ≈ 0.3 MB per body, against the explicit precomputed sparse transfer below.
- *Learning-free twins (§4).*
  - **Twin 1, explicit per-update intra-body sweeps:** J_int · N_b · K_int per incident-field change.
  - **Twin 2, explicit precomputed sparse binned per-body transfer:** nnz = N_b · NB(α) · spread. At α = 0.03, N_b = 200 that is 0.17–2.7 GB per asset; at N_b = 1,000 it is 0.85–13.7 GB [ARITH `explicit_precomputed_body_transfer`]. Its apply is O(nnz) per outer iteration.

**Learned versus explicit value** (what learning buys over the best learning-free twin):
- Over Twin 1: a converged, zero-variance response, at apply cost independent of J_int and K_int. This matters only where J_int · K_int is large (§4 regime map).
- Over Twin 2:
  - continuous peak position and width without α⁻² bins;
  - memory (MB rather than GB per asset);
  - **one shared model** that compiles unseen assets without per-asset path tracing.
- Learning buys nothing for convex bodies (N_b = 0), mirror chains (explicit is a few rays) or rarely-updated scenes.
- The value is **prediction, not established**. Stage 0 and Stage 1 (§5) are designed to measure it before training.

**Why not BRDF compression.**
- The first local event stays exact ER.
- N_b predicts transport that is spatially displaced (exit point ≠ entry point) and multi-event, determined by the body's global self-visibility.
- Its information content (a self-reflected-visibility map of a rigid body) is not present in any BRDF.
- Convex-body contract test: N_b must be identically zero.

**Failure conditions** (stop or unsuitable):
- the explicit intra-body share of the lifecycle is too small for the declared utility (Stage 0);
- the representation floor (lobe count K_s and broad rank) fails the angular-fidelity gates on grazing internal incidence (Stage 1, H2);
- the shared model does not generalise to held-out bodies (per-asset fit passes, shared fails);
- the certificate fails often in the admitted workloads;
- the measured frame-query overhead cancels the gain;
- reciprocity or composed energy fails;
- the self-shadow and visibility transitions inside the body are blurred, which is the WL52 P2 analogue (Stage 1 transition metrics);
- the novelty review finds the design equivalent to the PRT-family prior art [HUM D9].

**Disposition.** PRIMARY, conditional on [HUM] D1 / D2 and the Stage 0 gate.

**Rationale.** This is the only learned role I found that is (a) non-local, (b) rigid-invariant with an explicit validity certificate and (c) replaces explicit work that scales with angular sharpness and bounce count without being repeated per edit. It keeps C dynamic and explicit, keeps ER exact, and is one shared model. It does **not** address the GM1 / CK2 inter-body panel → box failure. That failure is C plus a learned sender, and under ownership it is not a reuse target.

### 3.2 ALTERNATIVE (independent): LER-Δ, Learned Environment Resolvent with explicit edit-delta iteration

**Learned physical meaning.** For the static environment E (all stationary surfaces), let T_E = K_E C_EE be its transport at the reference configuration with movable objects removed. LER-Δ learns its all-order directional resolvent, restricted to ≥ 2 events: H_E = Σ_{j≥1} T_E^j. It is linear, non-negative, passive and source-independent, and specific to that environment.

After an edit, T = T_E − B + M:
- B is the environment paths blocked by movable objects;
- M is the paths involving movable objects.

The solution satisfies b = (I + H_E)[e + (M − B) b]. It is computed by a fixed-point Δ-iteration: one sparse explicit (M − B) application over dirty and new paths only, then one learned H_E application per iteration.

**IO.**
- Input: the incident field at environment patches (from sources and from Δ-transport), as patch × continuous-direction samples.
- Output: environment outgoing radiance of ≥ 2 events at (x, ω), with a continuous angular decoder (lobes plus a broad code, as in §3.1).
- (M − B) is evaluated explicitly from the dirty relation rays and the movable bodies' own relations.

**Ownership.**
- H_E owns environment-only paths at the reference configuration.
- B is an explicit **negative** correction for exactly the paths the movable objects block, built from the dirty relation rays the explicit comparator already traces.
- M is explicit for paths touching movable objects; those objects' self-transport can be RB-DTO.
- No path is owned twice. B removes precisely what H_E would wrongly include. This is checked by a path-tag test.

**Rigid reuse.**
- H_E is fixed for all edits of movable objects. Weights are frozen. The compiled latents are per environment: one compile, or a per-environment fit [HUM D4].
- Edits only change B and M, which are explicit and current.

**Visibility interface.** All dynamic visibility enters through B and M, which are explicit. H_E contains only static environment visibility.

**Energy contract.**
- H_E is passive by construction (the simplex budget as in §3.1, per incident sample).
- The Δ-iteration converges if ρ(H_E (M − B)) < 1. This holds for small occluders, but it is not guaranteed for large ones: −B carries negative flux.
- Contract: report the iteration count and residual. Clamping is not allowed (AGENTS forbids energy clamps). If the iteration does not converge within the declared count, fall back to explicit.

**Training target.** Path-traced frozen-ER environment transport, from stratified incident samples at environment patches including grazing strata. These are exit / landing samples of ≥ 2-event paths, as in §3.1.

**Sharp mechanism.** As in §3.1: continuous lobes for the environment's sharp self-reflections (for example, the panel's view of the room). The sharp content of B (for example, a box blocking the panel → receiver beam) is explicit, through the dirty rays.

**Complexity.**
- Per edit: I_Δ iterations × (sparse explicit over dirty and new rays + learned H_E apply).
- H_E apply is about O(M_in · t_query) for the environment.
- The explicit twin re-sweeps J × N · K over the full relation after every edit.

**Learned versus explicit value.** It removes J full-relation sweeps per edit and replaces them with I_Δ learned applies plus sparse Δ-work. Learning buys a compact all-order sharp environment resolvent. The learning-free twin is an explicitly precomputed environment resolvent (dense or low-rank). That twin is infeasible for sharp content at N · NB ≈ 81 k × 81 k on GM1, and low rank fails for sharp transport [AUDIT J1: rank-16 trades fidelity across all incidences].

**Why not BRDF compression.** H_E is all-order, multi-event and non-local over the environment.

**Failure conditions.**
- Large occluders: ρ(H_E(M − B)) near 1 means many iterations, possibly more than explicit.
- WL41 / WL44 out-of-span failure: H_E is applied to Δ-fields unlike its training span. Passivity is built in, but **fidelity** on those fields is untested.
- Per-environment fitting may be out of scope (scene-specific, close to the 8DNA / RNA native-reuse class).
- Small N, where explicit sweeps are cheap (GM1: 4 × 0.53 ms kernel).
- The sharp B-content still has explicit dirty-ray variance.

**Disposition.** ALTERNATIVE: independent of RB-DTO in what is learned (environment versus body) and in how edits enter (Δ-iteration versus certificate). It is only worth a gate if the owner admits per-environment learned state [HUM D4], and Stage 0 must show that full-relation re-sweeps are a large share of the edit lifecycle at the target N and α.

**Rationale.** It is the only design I found that amortises explicit multi-bounce work *across edits* in scenes where the environment, not the moving props, carries the transport, with the edit entering exactly and explicitly.

### 3.3 REJECTED: A, continuous directional learned kernel (ECRO lineage, patch level)

**Contract.**
- *Learned physical meaning:* a geometry-conditioned patch operator mapping direct incident irradiance (continuous incident direction) to ≥ 2-event outgoing radiance (continuous outgoing direction). It replaces the c / a 64-bin factors with per-patch latents plus an angular MLP decoder, and keeps the single learned flight through exact-direction state rays plus the low-rank global term.
- *IO:* source encoding by exact incident direction; table queries at exact outgoing direction.
- *Ownership:* ≥ 2-event paths including free-space segments. That is a learned K∘C composite regenerated per edit.
- *Energy:* softplus factors with no passivity; ECRO's local factor rows are already non-passive [MEAS E1]. A simplex budget could be added.
- *Sharp mechanism:* a continuous decoder at the sender. Delivery still uses receiver-uniform state rays, so single-draw variance remains (P3 / J3).
- *Training target:* path-traced 2–5-event fields (not M64), with grazing-balanced coverage.
- *Complexity:* the generator is O(N²) dense message passing (§2.3) plus one decoder evaluation per state ray per flight (650 k MLP evaluations versus 650 k table fetches).
- *Rigid reuse:* weights only; factors are regenerated per edit, and **the c / a "local" factors depend on current visibility** through the form-factor message passing and openness input [CODE `model86.py:93-104`].
- *Visibility interface:* dynamic visibility is **implicitly coupled into the per-patch response factors**.
- *Learned versus explicit:* the sender half equals ER (exact, compact). The flight is explicit. The global half is the WL96A resolvent role. Per-edit ceilings with the generator are 0.91–0.97 (kernel basis) and ≤ 1.29 (overhead-dominated wall).

**Why not BRDF compression:** it is not purely BRDF compression, but its sender half is.

**Failure conditions:** it meets three brief unsuitability criteria:
- dynamic visibility coupled into the per-patch response;
- the learned operator does not substantially replace explicit transport on edits (§2.3);
- the sharp response requires the explicit flight anyway.

**Disposition:** REJECTED as a candidate. The *existing representation + improved coverage* version stays as an **H1 diagnostic arm** in Track G. It has no H3 role.

### 3.4 REJECTED: B, learned sparse multi-resolution angular representation

**Contract.**
- *Learned physical meaning:* a patch × adaptive-angular-cell table whose refinement pattern and coefficients a network predicts from geometry and material.
- *IO:* as the binned table, with variable cells.
- *Ownership:* as EXP / ECRO.
- *Energy:* flux-conserving cell integrals can be exact.
- *Sharp mechanism:* refinement where lobes are. But the lobe location is known analytically from ER and the incident direction (the specular direction), so refinement needs no learning.
- *Training target:* the refinement pattern.
- *Complexity:* about that of explicit adaptive tables.
- *Rigid reuse:* none beyond explicit.
- *Visibility:* explicit.
- *Learned versus explicit:* the learning-free twin (flux-conserving adaptive or 8×32 binning) already removes the P2 bias [AUDIT J2]. Learning only predicts an index that is cheap to compute.

**Why not BRDF compression:** it is not; it is learned scoring.

**Failure conditions:** dominated by its twin. Learned relation scoring requires an algorithm-only control first (RESEARCH_CENTRIC_TOPIC).

**Disposition:** REJECTED as a learned candidate. Its **learning-free version is a mandatory H2 control arm** (flux-conserving 4×16, 8×32 and adaptive).

### 3.5 REJECTED: C at patch level, learned directional lobe / distribution of patch outgoing radiance

**Contract.**
- *Learned physical meaning:* per-patch lobe mixtures of outgoing radiance given incident radiance. For a planar or convex patch this is ER applied to the incident field.
- *IO:* incident lobes → outgoing lobes.
- *Ownership:* K-local.
- *Energy:* a simplex budget is possible.
- *Sharp mechanism:* continuous lobes.
- *Training target:* ER response.
- *Complexity:* about that of evaluating ER.
- *Rigid reuse:* trivial (material frame).
- *Visibility:* explicit.
- *Learned versus explicit:* none. This equals ER compression [PRED = WL96 K1], which the brief forbids.

**Disposition:** REJECTED. Family C survives only at **body level** (§3.1), where the lobes describe non-local self-transport exits.

### 3.6 Other rejected roles (short)

- **Online neural radiance cache (NRC-style).** It retrains online after every edit, so it violates "without neural retraining".
- **Learned inter-body C.** WL52 measured it as 8–60× slower, with contact and transition failures. It also couples dynamic visibility into learning.
- **Learned sender-side sampling or relation scoring.** It is not a transport representation, and it requires algorithm-only controls first.
- **Resolvent-only at larger J.** The best learning-free twin is accelerated iteration (Krylov or coarse-grid) on a cheap sparse operator. The ceiling grows only slowly with J [AUDIT D-03].

---

## 4. Cost model for the kept candidates

**Explicit comparator per edit** (engineered kernel basis; any wall-basis result must be reported next to it):

```
L_exp = G + R_upd(f_dirty) + Q(L_lights) + J_ext · N K_ext · τ + Σ_b J_int,b · N_b K_int,b · τ_b + F
```

- τ ≈ 0.81 ns per relation entry, as recorded including launch overhead [ARITH]. An engineered implementation would be lower.
- For sharp self-transport, K_int must resolve iterated glossy lobes. Sender-side sampling must be repeated per order.

**RB-DTO per edit:**

```
L_rb = G + R_upd(f_dirty on C_ext only) + Cert + Q + J_ext · (N K_ext τ + Σ_b M_in,b t_q + reduce) + F · (1 + φ_query)
```

There is no compile term on rigid edits.

**Utility:** U = L_exp / L_rb at matched quality (Stage 3). The illustrative regime map (`cost_model.json` `body_regimes`) [PRED]:
- uses placeholder constants (t_q 5–20 ns per sample);
- uses uniform-binned K_int. A better explicit twin lowers these numbers.

The point is the shape, not the values:

| bodies × N_b | K_int | J_int | explicit intra (kernel ms) | learned apply (ms) | U edit (kernel, frame 64) |
|---|---|---|---|---|---|
| 2 × 200 | 512 | 3 | 0.5 | 0.2–0.8 | ≈ 1.0 |
| 2 × 200 | 4,096 | 8 | 10.6 | 0.2–0.8 | ≈ 1.3 |
| 20 × 400 | 4,096 | 8 | 212 | 5–20 | 4.8–6.7 |
| 20 × 400 | 16,384 | 16 | 1,700 | 5–20 | 34–48 |

**Reading.**
- The lever exists only when Σ_b J_int · N_b · K_int is large compared with G + R_upd + F, the costs every method shares. That requires several to many non-convex glossy bodies, iterated sharp self-transport, and edits that change their incident fields.
- Instanced assets (one compiled N_b reused by many instances) amortise compile time and memory further.
- Below that regime the candidate is at parity or worse.
- **Stage 0 must measure Σ_b explicit intra-body work with the best explicit twin**, not with uniform binning. Uniform binning would inflate the lever artificially.

**LER-Δ:**

```
L_Δ = G + R_upd + Q + I_Δ · (sparse(M − B) + H_E apply) + F
```

The lever is J · N · K_E · τ_E minus I_Δ · (H_E apply). Its no-advantage regime: small N, broad materials, large occluders (I_Δ large).

---

## 5. Falsification plan

### 5.1 Hypotheses and how they are separated

- **H1:** the failure is training coverage or optimisation.
  - *Test:* same representation, coverage-balanced data (grazing strata at a declared share), at least 3 seeds.
  - *Supported if:* learned error moves to within a declared fraction of the representation's own oracle floor on the stratum where it failed.
- **H2:** the failure is the angular representation.
  - *Test:* the oracle floor of the representation (fitted directly to path-traced targets, no learning) against learning-free binned controls at matched memory.
  - *Supported if:* the oracle floor itself fails the gate, and a different representation's floor (flux-conserving / 8×32 / continuous lobes) passes.
- **H3:** an improved learned representation still gives no geometry-edit utility.
  - *Test:* the Stage 0 Amdahl ceiling (cost anatomy of the quality-matched explicit comparator, before training), then measured U at matched quality at T1 with fixed weights.
  - *Supported if:* the ceiling is below U_min [HUM], or measured U is below U_min at matched quality.

### 5.2 Tracks and what they can share

| | Track G (diagnostic, GM1 / CK2 + E2 + E5 + one new admitted GM1 edit) | Track B (candidate, new rigid-body family) |
|---|---|---|
| purpose | H1 / H2 attribution of the existing ECRO failure; quantify the role ceilings at N up to 11 k | RB-DTO (and optionally LER-Δ) H1 / H2 / H3 |
| arms | G0 frozen ECRO (two WL92A OWN checkpoints + WL86); G1 ECRO with coverage-balanced retraining (same architecture, ≥ 3 seeds); G2 learning-free binned controls (solid-angle-mean 4×16, flux-conserving 4×16, 8×32); G3 EXP / C3UR (R0 and sender-uniform), ≥ 16 state draws; G4 C3URX1 exact first gather (pooled and deployed-draw) | B0 explicit comparator (C3UR-type with cached intra-body self-relation, sender-side sampling, per-update sweeps); B1 explicit precomputed per-body sparse transfer (Twin 2); B2 RB-DTO oracle floor (per-body direct fit to path-traced exits, no shared learning); B3 RB-DTO shared model, natural coverage; B4 RB-DTO shared model, grazing-balanced coverage; B5 per-asset fitted RB-DTO (capacity / generalisation control); B6 stale (T0 solution at T1) and fallback-forced controls; [HUM] B7 Native Reuse of a per-object external neural asset, else UNMEASURED |
| shared training data / budget | G0, G1 share the scene86 generator and training budget (matched steps / samples); G2–G4 are learning-free | B3, B4, B5 share the body generator, exit-sample budget and steps; B2 uses the same exit samples per body |
| shared physical reference | all arms: frozen-ER path tracing (REF-ER) at T0 / T1, independent replica halves | all arms: frozen-ER path tracing at T0 / T1 + per-body exit-sample references |
| cannot share | Track G and Track B cannot share training data or a learned budget, since the workloads differ. They share only the reference renderer, metric definitions, statistics and lifecycle conventions. ECRO cannot be run meaningfully on Track B, and RB-DTO is identically zero on Track G (a contract check, not a comparison) |

M64 is never used as ground truth. It may appear only as a labelled historical diagnostic. CK2 remains a diagnostic workload: no Track G result can approve or reject RB-DTO.

### 5.3 Workloads (admission declared before results)

**Track B family.**
- A semi-enclosed room, as in scene86, so that conventions remain comparable, with 1–3 non-convex rigid ER bodies plus Lambert movers.

**Admission clauses** (prospective; values [HUM D7]):
- the self-transport share of radiance in the body ROI is at least s_min;
- the changed-region higher-order SNR is at least the WL86 value (4×), measured on an independent qualification partition;
- the grazing internal-incidence share is at least g_min;
- the certificate holds on the admitted edits;
- at least one edit changes the body's incident field strongly (light enters or leaves the cavity) without touching Ω_b.

**Required conditions** (each is its own admitted workload, with separate references):
1. **W-graze.** Grazing sharp transport: a glossy groove or bracket (α ≈ 0.03–0.08) lit at cos θ_i < 0.15 into the cavity. This is the internal analogue of CK2.
2. **W-broad.** Non-grazing transport: the same family, lit near-normal, with moderate α.
3. **W-held.** An independent held-out workload: a body class never generated in training (for example a curved trough or corner reflector), with a held-out light class (cone) and a held-out edit. It is admitted before any learned model is evaluated on it and never inspected during development (the E5 precedent).
4. **T0 → T1 rigid edits.** A body translates and rotates in each workload; a second edit moves another object near Ω_b; plus one **certificate-violation control edit** (an object enters the cavity) to measure fallback correctness and cost.
5. **Fixed weights.** No retraining or refit after training freeze; compiled latents are reused across T0 / T1.

**Track G** uses existing admitted workloads (E2, E5) plus one new admitted GM1 edit with a T1 reference. CK2 is diagnostic only.

**Splits.**
- Train and validation bodies come from one generator seed set.
- Test bodies and W-held come from disjoint seeds and classes, with margins on pose and light (WL86 style: 0.15 m, 0.3 m).
- Checkpoint selection uses validation only.

### 5.4 Stages, gates and stop rules

**Stage 0. Cost anatomy and workload admission (learning-free; mandatory first).**
- Build B0 and B1 on the admitted Track B workloads.
- Measure each lifecycle component on the kernel basis **and** on wall, with overhead disclosed, under the declared convention [HUM D3]: edit → frame and edit → valid table, at the declared resolution and spp.
- Compute A_intra, the explicit intra-body work RB-DTO would remove, and the zero-cost ceiling U0 = L / (L − A_intra).
- **Stop if U0 < U_min**: verdict NO_HEADROOM, H3 supported by anatomy. No training occurs.
- For Track G, recompute the role ceilings of §2.3 at N = 1,270 / 5,080 / 11,430 with the engineered q table. This decides whether Track G has any H3 content at all; my prediction is none for geometry edits.

**Stage 1. Representability (no learning; H2).**
- B2 oracle floors per body and B1 at matched memory.
- Track G: G2 binned controls against REF-ER on the CK2 box-A receivers and on one non-grazing light and one second receiver (WL97 §8 smallest missing evidence for P0-2).
- Metrics, per incidence stratum:
  - exit-measure distance (sliced Wasserstein on position × direction);
  - flux-weighted fine-histogram L1;
  - peak position error (degrees) and width ratio;
  - energy closure and passivity supremum;
  - transition sharpness at self-shadow boundaries.
- **Stop if** the B2 floor fails the representation gates on W-graze. Then H2 is supported for this representation, and no shared model is trained.

**Stage 2. Learning (H1 versus capacity).**
- B3, B4 and B5 with at least 3 seeds each, at matched budget.
- **Attribution:**
  - B4 closes to the floor while B3 does not: H1.
  - B5 reaches the floor and B4 does not: shared-model generalisation or capacity limit.
  - Neither does: optimisation or capacity, unresolved.
- **Stop if** B4 fails the representation gates on test bodies at a level that would make Stage 3 meaningless. That level is declared in advance [HUM D5].

**Stage 3. Coupled image-level T0 → T1 (fixed weights).**
- Arms B0 (at several state-draw budgets and K_int, to trace its quality–cost curve), B4 (or the best predeclared learned arm), B5, B6 and fallback.
- Compared against REF-ER at T0 and T1 on W-graze, W-broad and W-held.
- Report:
  - whole image;
  - sharp self-interreflection ROI;
  - changed-region fidelity and signed change;
  - leakage / ghosting (signed error outside the predicted change support, and the stale-pattern correlation);
  - extra T1 error over T0;
  - composed spectral radius;
  - path-tag ownership identity;
  - rigid-equivariance check.

**Stage 4. Systems.**
- Report:
  - update critical path;
  - steady-state frame (including φ_query);
  - inference cost;
  - relation / visibility recompute on C_ext;
  - certificate cost;
  - memory (weights, latents, caches against B1's GB);
  - one-off compile and training cost (separately);
  - utility U against **quality-matched B0**, interpolated on B0's quality–cost curve at the learned arm's measured T1 error.
- Scaling over body count, instancing, J_int (cavity depth / albedo) and α, at least three points each.
- **Stop or verdict rules** are predeclared: U ≥ U_min on W-held at T1 and in at least one other workload, with the quality gates met.

**Order and gating.**
- Stage 0 → 1 → 2 → 3 → 4, and each stop is final for the batch. There is no rescue architecture.
- Track G Stage 1 may run in parallel with Track B Stage 0, because it is learning-free.
- Track G Stage 2 (G1 retraining) runs only if the owner wants the H1 attribution for ECRO's own record [HUM D8].

### 5.5 Evidence types (both synthetic and real scenes are required)

**Synthetic contract tests (production code, before Stage 1).**
- Zero input, superposition, non-negativity.
- Row-wise passivity in flux against the exact budget.
- Composed ρ < 1 with C_ext.
- Exact rigid equivariance (rotate body plus incidence, compare outputs).
- Convex body: N_b ≡ 0.
- Analytic V-groove of two GGX planes at small α: exit positions and directions against analytically traced chains.
- Lambertian cavity against a converged radiosity solution.
- Furnace test of a closed body.
- Certificate invalidation when a primitive enters Ω_b.
- Path-tag disjointness (each tracer path segment counted once across ER / N_b / C_ext / LOW).

**Why both are needed.**
- Synthetic tests isolate the mechanism (representation, energy, ownership, equivariance) from coupling and readout. Image metrics can hide compensation between components, as in WL86 → WL92B.
- Real-scene evaluation is the only place where utility, composition error, ghosting, edit response and the systems cost exist.
- Neither substitutes for the other.

### 5.6 Statistics

- Paired evaluation: identical reference draws, camera, state draws and lights across arms.
- At least 3 training seeds. WL97 found that two seeds flip sign [AUDIT C1-03].
- At least 16 state / relation draws for explicit arms, reporting the deployed-draw distribution, not only pooled means [AUDIT J3].
- Reference noise from independent replica halves. Treatment differences use the paired SE (block bootstrap over patches / pixels), **never an L1 noise floor as a paired tolerance** [AUDIT J6].
- Minimum meaningful effects [HUM D5] are declared in the protocol before any result.
- Two-workload confirmation for every positive claim (WL96 §F rule).

---

## 6. Human decisions required

- **D1. Ownership refinement.** May the free-space self-paths of a rigid body (C_self) be represented inside a learned body operator, under a self-path-region exclusivity certificate with explicit fallback? Inter-body C would stay explicit and current. Without this decision, RB-DTO is inadmissible under Architecture §2.
- **D2. Scope of the learned component in GM1.** Is it acceptable that the learned component targets rigid-invariant transport (intra-body and environment), while sharp delivery between separately moving bodies stays explicit C by ownership? This is an ownership argument, not the "sharp to explicit, learned residual" default the brief excludes. The alternative reading keeps the CK2-type inter-body sharp delivery as the learned target. In that case D3 finds no defensible candidate (§7).
- **D3. Utility threshold and convention.**
  - U_min (WL96A used 1.25);
  - basis: wall versus an engineered kernel basis, and whether the explicit comparator must first be de-overheaded;
  - frame inclusion, resolution and spp;
  - which lifecycle events count (edit → frame, edit → valid table, steady frame).
- **D4. Shared versus per-asset or per-environment learned state.** Is per-asset fitting (B5) acceptable as a method, or only as a control? Is per-environment learned state (LER-Δ) in scope? This decision also fixes the external-baseline role: a per-object neural asset moved rigidly is **Native Reuse** under the baseline gate.
- **D5. Numerical gates.** No current evidence determines these:
  - peak-position error and width-ratio limits;
  - exit-measure distance limits per stratum;
  - energy-closure and reciprocity tolerances;
  - E_H thresholds per region;
  - ROI minimum meaningful effect;
  - leakage limit;
  - extra T1 error;
  - the Stage 2 stop level.
- **D6. Seeds and draw budgets.** At least 3 seeds and at least 16 draws are proposed; the compute budget is the owner's.
- **D7. Track B workload admission values.** s_min, g_min, body classes, the held-out class and edits.
- **D8. Track G Stage 2.** Should the ECRO H1 retraining run, given that its H3 content is predicted to be nil for geometry edits at GM1 scale?
- **D9. Novelty and prior art.** Recalled from memory, not verified in this batch:
  - precomputed radiance transfer (Sloan 2002);
  - all-frequency wavelet PRT (Ng 2003);
  - shadow fields (Zhou 2005);
  - precomputed radiance transfer fields for rigidly moving objects (Pan 2007);
  - modular radiance transfer (Loos 2011);
  - neural PRT variants.

  A literature review must establish what is new. Candidates for novelty are: the certified ownership composition with explicit current C, the passive measure-valued sharp lobes, and a shared compiler. Compression alone is not novelty (RESEARCH_CENTRIC_TOPIC).
- **D10. External comparators.** Measure Native Reuse of an object-level 8DNA / RNA-type asset, or report it UNMEASURED. Neural Full Recompute applies only if that asset is actually invalidated.
- **D11. Batch classification** under the numbering policy. Track B introduces a new mechanism, and its class follows the owner's rule.

---

## 7. Proposed verdict

**DESIGN_READY_FOR_HUMAN_REVIEW.**

1. The primary candidate (RB-DTO) and the independent alternative (LER-Δ) have complete contracts:
   - physical meaning, IO, ownership, an energy contract with a defined measure, sharp mechanism, training target, complexity, rigid reuse and visibility interface;
   - learned-versus-twin value and failure conditions.
2. The falsification plan separates H1, H2 and H3. It gates H3 by a learning-free cost anatomy before any training, keeps diagnostic Track G apart from candidate Track B, and leaves every unsupported number to the owner.

**Scope limit, stated as strongly as the evidence allows.**
- For the **current scene86 / GM1 / CK2 workload family** (planar / convex objects, N about 10³, one or two point lights, J ≤ 5), my systems analysis finds **no defensible learned candidate with geometry-edit utility**. Every learned role either:
  - re-learns ER (BRDF compression);
  - learns dynamic C (WL52 class);
  - or replaces at most 3–4 ms of explicit work while paying a per-edit generator ≥ that work on the kernel basis.
- This is a [PRED] / [ARITH] result from recorded components, not a measured utility test. Stage 0 of Track G is the cheap test.
- If the owner rejects D1 or D2, my verdict for the batch becomes **NO_DEFENSIBLE_CANDIDATE** (scoped to the current workload family and the FNTO ownership contract as written).
- No verdict here is adoption or experiment approval.

**Stop conditions triggered.** None formally. Two scope findings (§2.3, §2.5) mean that a positive H3 test is impossible on the current workload family under engineered conventions.

---

## 8. Files and reproducibility

- `experiments/wl98_calc/D3/cost_model.py` is the numpy scratch model. All inputs are recorded medians cited in its header, from WL97 E5 / D and WL96A.
- `results/wl98_calc/D3/cost_model.json` holds the role ceilings, the required-share table, uniform-binning scaling, ECRO generator N² scaling, the explicit precomputed body-transfer memory and the body regime map. Everything is [ARITH] or [PRED].
- **Sources read.**
  - Architecture.md §1–9.
  - Roadmap (current decision; WL86–WL97 entries).
  - BASELINE_ROLES (mandatory gate; controlled comparisons).
  - RESEARCH_CENTRIC_TOPIC (novelty discipline).
  - WL97 README, COORD §1–9, D (complete), E5 §1–3.4.
  - WL86 worklog (contract and representation).
  - WL96 worklog (scope, K1, carrier table).
  - `experiments/ecro86/model86.py`, `scene86.py`, `protocol/ecro86.json`.
- Not opened: any other WL98 file.
