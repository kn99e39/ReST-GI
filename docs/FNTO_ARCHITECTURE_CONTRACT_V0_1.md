# FNTO — Factorized Neural Transport Operator (Architecture Contract v0.1)

**Decision date:** 2026-10-23
**Status:** Assistant-authored primary architectural hypothesis at user request. **DESIGNED, NOT IMPLEMENTED OR VALIDATED.**
**Branch:** Selective_Recompute
**Scientific aim:** reusable learned *source-to-receiver transport response*, not a baked one-configuration radiance function or a geometry-only dirty-state sidecar.

## 1. Decision and evidence

**Adopt FNTO as the primary *next representation candidate*, not SPT-1 plus repair patches.** Preserve original RNA, Surface-Owned reconstruction (WL32–38) and SPT-1 (WL39) as immutable historical baselines.

Evidence from the repo:
- WL28–29: K=32 sparse geometric/radiometric sidecars failed held-out near-mirror; limited angular support, aggregation and training coverage are entangled, not a rejection of relational transport in general.
- WL32–38: complete owned learned-unit reconstruction and parameter preservation work. On diffuse-interaction Lego edit, oracle D* was 8/17 (7 stationary receivers), radiometric recovery ~0.98 and three-seed mean PSNR gap −0.082 dB versus full. Existing update requires target-state supervision and training.
- WL39: SPT-1 explicitly rebuilt free-space relations and changed radiance without updating 6.80M learned parameters. Its static T0 quality was 22.23 dB versus RNA 33.57 dB; a missing object-internal scattering/transmission channel, very-low-order angular basis and Lambertian sender assumption compromised transport completeness.

Architecture-level interpretation: the correct factorization is **intrinsic material/medium-local reusable transport response + current-geometry free-space coupling, including exposed self-interreflection**. Do not compress all physical effects into a single fixed-scene radiance MLP, but also do not replace learned complex shading with a limited explicit radiosity solver.

## 2. Exact conceptual transport boundary

Partition the current scene into **intrinsic optical domains** a: material surface interfaces and connected optical media (plus protected microgeometry where no external edit can penetrate). In particular, the official Lego asset must **not** automatically be treated as one transport-closed domain: the open air between its bucket and baseplate is accessible to the moving box and is therefore a **dynamic free-space transport path**, even though both endpoints belong to the same asset. Domains are not identical to Object IDs or the historical L16 Surface-Owned units. The red box and the relevant local Lego media/surfaces have their own optical domains. If a medium deforms and changes its internal optical paths, its learned domain operator is not assumed invariant.

For each domain, define:
- i_a^t(s,ω): incoming radiance at its boundary under current scene configuration;
- b_a^t(x,ω_o): outgoing boundary radiance;
- K_{θa}: reusable learned **intrinsic optical-domain scattering operator** mapping spatially/directionally varying incident radiance to outgoing radiance; it includes local surface response, nonlocal transport *within a material/medium* and protected subgeometry, but excludes editable open-air loops;
- e_a^t: current external light incident on this domain, including actual external occlusion;
- C_{φ,ab}(R_ab^t): free-flight transport operator conditioned on current spatial/directional/occlusion relations R_ab^t, including **same-domain exit-and-reentry through editable free space** (a=b).

**Primary equation:**

    i_a^t = e_a^t + Σ_b C_{φ,ab}(R_ab^t) b_b^t
    b_a^t = K_{θa}[i_a^t]

The equation is understood as an operator equation over spatial/directional boundary fields, approximated in compressed coefficient space. It is NOT merely a scalar formula or a single feature-vector concatenation.

**Path semantics:**
- K_a includes the learned local BRDF/BSDF (a surface-interface interaction), SSS/volume scattering and refraction/transmission through unchanged **material interior**, and any truly protected subgeometry microtransport. K_a must NOT bake free-space interreflection paths that an external edit can occlude.
- C_ab transfers outgoing radiance through **editable free space** to the next scattering/medium boundary, with visibility and direction dependence. This includes reflection paths between two surfaces of the **same asset**, and may include a=b for an exit-and-reentry path. C cannot consume scattering inside a medium; K cannot absorb editable free-flight loops.
- A→B→A and A→free space→A are represented by repeated K/C composition, with path-family disjointness and no double counting. Place the moving box in a Lego cavity to test this exact distinction.
- Current external illumination and shadows enter through e_a^t. Do **not** gate the entire object-internal response at the queried receiving pixel by that pixel's direct-sun visibility: this would erase shadow-side SSS (SPT-1's major failure mode).
- Reflective/refractive events are not assumed Lambertian. Transparent domains must allow entering one boundary port and exiting a different one.

**Intrinsic-domain invariance claim:** K_a can be reused under a rigid transform of an unchanged *optical domain*, with port directions converted to its canonical frame. Even under rigid motion of an entire asset, new occluders may change free-space coupling between two surfaces of that **same asset**; that change belongs in C, never in K. Internal deformation, fracture, topology changes or changed material invalidate the affected K and are future scope.

### Critical domain-validity counterexample (must be tested)

The official Lego asset has physically exposed space between the bucket and the baseplate. The moving red box can obstruct a light path from one Lego surface to another. **Never store that exposed self-interreflection path in a fixed whole-Lego K merely because it has the same asset ID at both endpoints.** If a learned RNA-style whole-asset operator contains it, that operator is configuration dependent and cannot serve as an invariant K in this scene. K stores only local interface, volume/internal-medium and genuinely edit-inaccessible transport; all editable free-flight segments are in C regardless of asset identity.

Partitioning the Lego into optical domains and boundary ports is therefore an architectural requirement, not an optional optimisation. The path boundary is *transport physics* (internal scattering versus editable free flight), not the artist's object hierarchy.

## 3. Concrete learned representation

### 3.1 Boundary ports and coefficients

An object domain maintains spatially indexed boundary ports and an orientation-aware **multiresolution directional dictionary**. Incoming and outgoing fields are compressed into coefficients, but high-frequency/specular directions must not be forced through a fixed SH2 bottleneck.

- Source port encoder U_{θa}(s,ω_i) represents where and how light *enters* a domain.
- Receiver decoder V_{θa}(x,ω_o) represents where and how light *leaves* the domain.
- Fields are in object-canonical coordinates; corresponding world-space transforms rotate directions consistently.
- Spatial ports may overlap and use declared partition-of-unity weights; they are an operator interface, not independent learned-model ownership.
- The number/density of ports and directional support must be reported against angular bandwidth and transport complexity; a small fixed K or a uniform low-rank cap is not a physics theorem.

### 3.2 Two learned response families inside K

**K^sharp — learned local interface scattering.** Handles directionally concentrated reflection and other same-interface scattering; high-angular-frequency transport must have sufficient directional support. Learned angular lobes or equivalent sparse directional support are conditioned on material/local geometry and entry/exit directions. It must have the representational *possibility* of sharp specular variation and cannot be routed only through SH2.

**K^internal — learned nonlocal *material-internal* response.** Covers SSS, transmission/refraction through a medium, and paths remaining inside protected intrinsic microgeometry. Crucially, *exposed* self-interreflection via editable free space is NOT in this kernel; it belongs to C even when both endpoints are on one asset. Its working factorization is:

    K^internal_a(x,ω_o ; s,ω_i) ≈ Σ_m V_{a,m}(x,ω_o) U_{a,m}(s,ω_i)

with possible spatial/directional multiresolution sparse residuals where low rank is inadequate. U and V are trainable functions of source and receiver variables, not a precomputed fixed-configuration radiance value.

The sum K^sharp + K^internal describes **disjoint or explicitly weighted transport path families**, not an arbitrary double-counted residual decomposition. The implementation must document the path partition and any overlap correction.

**Linearity contract:** at fixed geometry/material, transport is a linear operator on incident radiance: K[αi1+βi2] = αK[i1]+βK[i2] for nonnegative α,β in the evaluated radiance domain. The functions defining the kernel may be neural and nonlinear in geometry/direction, but application to incident radiance must preserve this linearity. Zero input produces zero output absent emission; positivity/passivity/energy bounds require tests (do not declare them guaranteed merely from network architecture).

### 3.3 Learned inter-domain coupling

C_{φ,ab} maps compressed outgoing port radiance from domain b to incoming port radiance of a, conditioned on:
- relative transforms, bounds and orientation;
- directional overlap and source/receiver support;
- conservative visibility/occlusion evidence, including moved third-party occluders;
- material-independent free-flight geometry and coordinate/frame transforms.

The geometry search index is **algorithmic** and physically interpretable (spatial bounds plus normal/directional bounds), but **is not the transport graph**. Directed nonlocal relations cross hierarchy branches. The learned part compresses/approximates relation-to-coefficient transfer; this is not an exhaustive explicit per-surface full path tracer hidden under a neural name.

Use a sparse candidate set, maintain coverage evidence and measure all visibility/ray/query costs. The future learned recomputation-target selector is **not part of FNTO v0.1**: oracle or conservative candidate coverage is allowed for the first factorization test. Do not confuse selecting updated couplings with deciding which old neural parameter sets to retrain.

## 4. Current-state solution and rendering

Represent outgoing boundary coefficient vectors β_a^t and incoming vectors γ_a^t. One conceptual iteration:

    γ_a^(k) = ε_a^t + Σ_b C_ab^t β_b^(k)
    β_a^(k+1) = K_a γ_a^(k)

K_a and the learned parameters of C are fixed across a supported geometry edit. e, relation geometry/visibility C_ab^t, and solution states β,γ change. Rendering queries the learned receiver decoder at (x,ω_o) with the updated outgoing state.

The number of iterations is **not hard-coded to SPT-1's 4**: use a declared residual / energy-based termination, report convergence, cycle handling and any stability failure. Recompute changed relations and propagate changed coefficients until bounded error criteria are met; unaffected states may be reused only when both local and upstream dependency influence are valid. Numerical support propagation is not automatically physical invalidation.

### Explicit change ownership

**Persistent:** trainable K_a parameters, domain-local source/receiver features and the learned rule of C_φ (unchanged under supported rigid edits).

**Configuration-dependent:** inter-domain relation geometry/visibility, coupling coefficients, external incoming light, incoming/outgoing boundary radiance state, and affected-state solution.

**Updated learned parameters after a rigid edit:** 0 is the target, not a measured accomplishment for FNTO.

**Not claimed invariant:** K under internal geometry/material changes, reconstructed source coefficients under drastic angular aliasing, or transport relationships missed by candidate discovery.

The Surface-Owned complete-unit full-relearning baseline keeps its historical U_i=(S_i,Θ_i) update contract unchanged. FNTO uses a separate transport-domain/kernel contract, explicitly **not** a retroactive renaming of U_i.

## 5. Learning contract

A single canonical full-scene lighting configuration cannot identify the proposed operator. FNTO requires offline training to excite the source-to-receiver map, not merely render canonical RGB.

**Stage A: object-local operator supervision.**
- Generate physically grounded isolated-domain reference responses under spatially *localized*, angularly varied incident illumination, including probe patterns illuminating different locations independently.
- Include shadow-side SSS, entrance-to-exit glass/transmission, smooth and sharp local reflection, and physically valid isolated-domain response. Separate any exposed self-interreflection through free space and use C for it. Preserve full reference semantics and known material graphs.
- Train K_a under these probes with a linear-response/superposition check and hold out unseen *spatial incidence patterns*, not just camera/light directions.
- These data are expensive offline training supervision and must be accounted for. A released RNA per-directional-light model cannot be reused unchanged as a spatially varying boundary operator.

**Stage B: inter-domain coupling and composition.**
- Train or calibrate C_φ from current geometry features with physically grounded first-flight coupling supervision, not held-out target-scene radiance.
- Use multiple *training* relative configurations to make geometry-dependent coupling learnable; keep reusable K fixed after Stage A qualification.
- For the decisive test, hold out an edit T1 entirely from training the reusable learned operators and any tuned coupling coefficients. At T1 update e, C and β/γ from observable geometry/lighting only. No T1 target RGB or incident-radiance oracle at runtime.
- Where oracle boundary incident radiance is useful for attribution, run it as a separately labeled diagnostic input, never a deployable result.

A fixed source port that never receives independent excitation is an **identifiability failure**, not evidence of operator generalization.

## 6. Controlled architecture evaluation

**First real-scene contract:** official Lego + moved red box (existing Worklogs 37–38 T0/T1), same physical reference regime when reused legitimately. This is a particularly demanding asset because 41.6% of the reported image is SSS yellow; transparent/glass and reflective materials are also present.

Compare:
1. upstream RNA under T0/T1 static training (quality ceiling/control, not zero-retraining);
2. Surface-Owned complete-unit reconstruction and frozen baselines (historical, fixed);
3. SPT-1 historical negative prototype (diagnostic only, NOT starting implementation);
4. FNTO with stale inter-domain relations but same learned parameters;
5. FNTO with current relations and fixed learned parameters;
6. FNTO with **oracle incident boundary fields** only as diagnostic upper bound, if needed.

**Upfront architecture admissibility:** Before spending on the full multi-configuration training, show that K has an information path for SSS, glass entrance/exit and glossy angular peaks. Synthetic source/receiver tests verify the actual implementation and show meaningful response under new source locations, not only stored tensors. Do not approve a known transport-incomplete candidate as the primary baseline replacement.

**Predeclare numerical outcome rules** before training. Suggested *targets*, not evidence: full-scene/static fidelity close to RNA (within approximately 1 dB on a qualified regime, with explicit SSS/glass/reflective per-material checks); strong changed-receiver recovery and preservation of K after the held-out edit; no target-scene re-training; full cost reported. An individually favorable changed-region recovery is insufficient if full-frame representability collapses.

**Core falsification questions:**
- Can the learned source-to-receiver K reproduce isolated-domain SSS/transmission and gloss under unseen spatial incident patterns?
- Can fixed K under new C recover stationary receiver changes on a held-out physical edit?
- Does learned coupling capture the relevant nonlocal influence without an implicit dense ray/path tracing rebuild?
- Is the compressed kernel/coupling actually more efficient than the appropriately quality-matched full recompute, once offline training costs and runtime queries are disclosed?

**Required accounting:** quality by material and shadow/lit regime; sharp direction/angular error; held-out edit signed radiance-change tracking; port/directional coverage; learned bytes; boundary-state bytes; number of affected edges/ports; candidate miss/error attribution; end-to-end rebuild / update / inference timings; cross-source linearity; parameter hashes. Qualitative crops must show yellow shadow-side SSS, glass/transmission, box contact/bleeding, glossy regions and unchanged receivers.

**If internal K quality is inadequate, stop the candidate at that gate.** Do not proceed to learned scene selection, rank sweep or SPT-1 patching. Report which mode is unsupported and whether the failure is insufficient training excitation, angular bandwidth, low-rank compression or incorrect path partition. If K is viable but C fails, attribute separately.

## 7. Near-mirror and nonrigid scope

Near-mirror is a **directional bandwidth stress test**, not an instruction to drop SSS or restore a monolithic baked MLP. The candidate must have a structurally capable directional branch, but an actual near-mirror success claim requires a reliable oracle and adequate training support. Worklog-33 RNA near-mirror static limits cannot be attributed to exactly one cause without further evidence.

Supported immediate edit: rigid relative motion of unchanged transport-closed domains. Non-rigid bending, self-contact, local shape changes or changing volume geometry can invalidate K; they require a future refinement into persistent intrinsic scattering rules and configuration-dependent within-object relations. Do not claim broad nonrigid invariance in v0.1.

## 8. Prior art and novelty discipline

Do not claim novelty merely for neural operator compression, BSSRDF approximation, factorized light transport, sparse radiosity, or hierarchical candidate search:
- RNA (Mullia et al., TOG 2024; DOI 10.1145/3695866): full within-asset learned transport under directional illumination.
- Neural Light Transport (Zhang et al., TOG 2021): physically based diffuse baseline plus learned global/non-diffuse residual.
- Neural SSS: Lightweight Object Appearance Representation (CGF 2024; DOI 10.1111/cgf.15158): compact neural representation of 8D BSSRDF.
- Neural Local Inter-reflection Modeling for Garment Fold Rendering (CGF 2026; DOI 10.1111/cgf.70348): geometry-conditioned factorized fold interreflection and directional SH response.
- Differentiable Light Transport with Gaussian Surfels via Adapted Radiosity (TOG / SIGGRAPH Asia 2025; arXiv 2509.18497): radiosity-inspired coefficient-space transport, including specular capability.

Potential **research claim**, conditional on comparison: an intrinsic optical-domain learned scattering kernel with full material/path semantics, combined with dynamic compressed free-space coupling (including same-asset visible paths) that permits geometry-edit updates without refitting reusable parameters at quality comparable to a scene-baked neural asset. This claim is NOT established by this document.

## 9. Decision boundaries and next work

**Architecture decision NOW:** FNTO is the primary candidate to implement and falsify; SPT-1 remains historical negative evidence. This is a representation **design**, not a measured success or silent implementation authorization for subsequent batch work.

**First bounded implementation unit:** demonstrate optical-domain K on the official Lego material semantics under spatially varied boundary-light probes; verify linearity and the SSS/glass/angular contracts. Only if that gate is credible, wire current-geometry compressed coupling C and a held-out rigid box edit in the same major architecture investigation, keeping source/receiver and training tests separate. Avoid subdividing the research into trivial diagnostic-only Worklogs; stop at the declared architecture decision gate.

**COMPLETION CONDITION:** Can a learned intrinsic-domain source-to-receiver transport kernel plus current free-space coupling retain RNA-class complex material transport while responding correctly to a held-out geometry edit with no reusable-parameter retraining? Report evidence even if negative.
