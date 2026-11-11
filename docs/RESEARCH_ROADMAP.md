# Research Roadmap — Dynamic Learned Transport Reuse

**Current as of:** 2026-11-10. **Role:** milestone dependencies, current bounded scope and next decision gate. [Research objective](RESEARCH_CENTRIC_TOPIC.md), [architecture contracts and candidate status](Architecture.md), and [evaluation strategy](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md) have separate owners.

## Current frontier

**Current direction (human decision, 2026-11-10):** The learned ER closure qualified through WL82–84 is the **primary development baseline for local material scattering**, not the selected final ReST-GI architecture. Its demonstrated scope is a coated-conductor family, learned descriptor-to-closure generalisation, local physical/runtime contracts and limited static rendering/economics. Dynamic interreflection, learned nonlocal intrinsic response and quality-matched end-to-end dynamic GI remain **UNTESTED with this baseline**. [Architecture](Architecture.md) owns the baseline contract; [WL84](worklogs/84_WORKLOG_RENDERING_QUALIFICATION_UTILITY_CLOSURE.md) owns the evidence. The immediate approved *research direction* is GM1; **no implementation batch or later milestone is automatically authorised**.

### Global development milestones — approved direction, not approved implementation batches

The sequence below defines **questions, dependencies, controls and exit decisions**, not a predetermined solver, learned C, spatial data structure or final architecture. A milestone can close negatively if its evidence is valid. Numerical gates, scenes, representation hypothesis, hardware accounting and stop conditions are frozen in each separately authorised experiment *before* its primary results.

| Milestone | Central question and bounded scope | Evidence required for a decision | Exit / dependency |
|---|---|---|---|
| **GM0 — local foundation** | Can one compact learned, physically admissible local scattering rule serve as the development starting point? | WL82 direct-fit representation, WL83 learned generalisation and runtime contracts, WL84 qualified local rendering and measured costs. | **QUALIFIED for development**: learned **ER** primary; learned E lightweight control; neither is a nonlocal K or final architecture. No renewed local-closure search without new failure evidence. |
| **GM1 — dynamic directional interreflection** (**next**) | With frozen ER, can **sharp glossy/specular, multi-bounce, editable-visibility interreflection** be represented and updated under a **rigid occluder/object edit**, including changed radiance on stationary receivers? | At T0/T1, (1) original layered-BSDF physical path-traced reference; (2) full path tracing using the *same frozen ER* to isolate its local bias; (3) one explicit proposed global mechanism; (4) stale/no-update and same-mechanism full rebuild; (5) directional/highlight, signed-change, multi-bounce and visibility-affected/unaffected evidence; (6) all update/solve/readout/memory costs and a predeclared quality-matched edit-to-valid-output and per-frame accounting. No T1 supervision/oracle at runtime. | **Go** only if a valid directional global representation and edit response meet prospective quality and cost requirements; otherwise attribute representation / composition / update / economic failure before changing the hypothesis. Proving only an image match or implementing a full retracer does not complete GM1. |
| **GM2 — complex transport breadth** | Can the mechanism extend from the GM1 glossy case to **nonlocal material/interior response, transmission/SSS and heterogeneous multi-bounce paths** without double counting or losing spatial/angular support? | Independent physically qualified mixed-material scenes, first-exit and exposed-free-space path ownership, boundary-port and ballistic/scattered controls, local-vs-global error decomposition, static and edited fidelity plus cost on the same method. | Decide whether the same representation extends, needs a bounded compatible augmentation, or has an identified capacity/ownership limit. Do not equate WL84 local ER with complete intrinsic K. Depends on an interpretable GM1 result. |
| **GM3 — edit generality and state validity** | Which parts of the transport state remain valid under **local deformation, global bending/twisting and self-contact/topology changes**, in addition to rigid motion? | Correspondence and dependency/invalidation evidence, edited-path coverage, negative stale-state controls, accuracy of unchanged and changed receivers, conservative/full fallback, total update and solver cost. | Define supported edit classes and explicit invalidate/rebuild boundaries; unsupported motion cannot be claimed. Depends on an eligible transport mechanism from GM1–GM2. |
| **GM4 — scalability and contribution qualification** | Does the supported method offer **quality-matched lifecycle advantage** on more complex scenes and repeated edits, including sufficiently accurate interreflection, compared with applicable prior art? | Scaling of relation discovery, angular state, memory, solve iterations and update locality; edit-to-valid-GI and sustained-frame cost including preparation/training amortisation; within-ReST-GI full rebuild and stale controls; physically aligned external **Neural Full Recompute** when actually invalidated, **Native Reuse** when valid, and separate path-traced re-render cost. Novelty claims must survive targeted prior-art comparison. | Decide final supported contribution and architecture viability (or a scoped negative). This is **not** permission to claim superiority before the comparisons run. |

**Across every milestone:** (i) fidelity **and** edit response **and** economics are evaluated together, not as isolated deferred optimisations; (ii) local ER approximation error must be distinguished from global transport/composition error; (iii) do not freeze a representation or selective-update algorithm prior to its mechanism hypothesis; (iv) preserve evidence of negative baselines and genuine full-vs-selective cost; (v) continue only after human interpretation/authorisation at each architecture decision. Rigid-first does **not** remove non-rigid or self-contact coverage from the final research objective.

### Earlier global-transport evidence — historical controls, not demonstrated failures of learned ER

In the earlier intrinsic-K fidelity line, WL64 [WL64](worklogs/64_WORKLOG_DIRECTIONAL_CONTRAST_SUPERVISION.md): NOT SUPPORTED. Fixed-capacity contrast supervision lowered noisy training-pair risk but worsened both original and independently qualified new-field directional fidelity. The [WL63 physical component audit](worklogs/63_WORKLOG_PHYSICAL_BALLISTIC_SCATTERED_ATTRIBUTION.md) localizes qualified Pair7 error predominantly to learned collided transport, not the direct ballistic B output; it does not identify why that learned response fails. [WL60–62](worklogs/60_WORKLOG_FROZEN_K_FIDELITY_ATTRIBUTION.md) record earlier diagnostic and label-precision findings. [WL59](worklogs/59_WORKLOG_PERSISTENT_INTRINSIC_FS.md) remains the F/S correctness baseline without qualified complete GI fidelity; the [WL56](worklogs/56_WORKLOG_EXACT_SELECTIVE_K_RECOMPUTE.md) physical GI floor remains open. [WL58](worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md) remains a separate sharp-GGX representation preflight.

[WL67](worklogs/67_WORKLOG_BASELINE_SUPERIORITY_EVIDENCE.md) executed the first quality-matched external comparison: on no workload does ReST-GI show a measured quality-matched advantage. On the WL56 medium fixture, Cycles **path-traced re-rendering** meets the complete 3% contract at 4 spp in 4.7 s for four views, while the exact oracle Selective update takes 190 s and fails it (9/12); the oracle is ahead only in per-port stationary-receiver signed-change precision and in a 0.69 s rigid resident update that fails the contract. On the WL22 teaset, 64-spp path tracing (0.5 s) beats every frozen or refitted RNA/8DNA output; ReST-GI has no K there, and TransGI has no released implementation.

[WL69](worklogs/69_WORKLOG_COMMON_WORKLOAD_EXTERNAL_BASELINE.md) established one common workload with an external neural baseline: the WL55 floor medium fixture under a rigid edit, with released 8DNA. ReST-GI CAK meets the 3 % contract on both states; 8DNA does not (3–7/12, static blue bias). Quality-matched Cycles **path-traced re-rendering** was faster than ReST-GI (16 spp in 2.4 s vs a 24 s update). The frozen WL69 protocol calls this outcome B, but its 8DNA arm reused the T0-trained slab asset without T1 retraining: **Neural Full Recompute was not measured**. RNA, Neural Radiosity and TransGI could not run this workload.

[WL70](worklogs/70_WORKLOG_SUN_STATE_REUSE_C1.md) tested validity-aware reuse of the first-order direct-sun state on that C1 edit. The cached T0 state passed an explicit dependency contract: rest mesh, medium, rigid pose, material-frame sun and cameras, estimator, and a bit-identical replay of source visibility. It was physically identical to the retraced T1 state, and four invalidation controls were rejected. The CAK update fell from 24.6 s to 2.09 s with unchanged quality, narrowly beating quality-matched Cycles **re-rendering** (16 spp, 2.40 s under WL69 timing; equal under Blender's internal render time). This is an **11.8x intra-ReST-GI sun-reuse acceleration**, not a measured advantage over 8DNA T1 neural retraining. Cycles still meets the bare contract at 1 spp in 1.2 s. The diagnostic cache is not adopted.

### Compiled neural material rules (WL80C–WL84, as of 2026-11-10)

[WL84](worklogs/84_WORKLOG_RENDERING_QUALIFICATION_UTILITY_CLOSURE.md) re-qualified WL83's rendering evidence for the unchanged learned closures. **Outcome A (run 2).** WL83's OUTCOME D stays the historical verdict.
- **Reference.** An independent 65536-spp layered reference (16× WL83) cuts the uncertainty to 0.004–0.007.
- **Learned ER passes the WL81 Stage F rule** on the former failing image test83_iid_1_L2 and on both controls.
  - On the target image the margin is small: highlight error 0.093 vs the 0.10 threshold, stable over 8 seeds (sd 0.001).
  - WL83's failure was a reference-qualification artefact.
- **Learned E's hard-material highlight failure is genuine** (0.136, 0 / 8 replicates pass).
- **Economics, uncontended:**
  - GPU time per spp is 1.45–1.63× lower than the layered reference;
  - reaching the 64-spp quality level takes 2.8–6.3× less GPU time;
  - the advantage is limited by a bias floor of about 0.035–0.04 (the reference wins beyond about 1024 spp);
  - there is no wall-clock advantage in this Python-traced Mitsuba harness (tracing is 85–95 % of wall).
- **Failed attempt.** Run 1 was invalid: a GPU accumulation defect summed every sample twice. It was caught by its 2× anomaly and fixed by amendment 1.
- **Scope.** 3 images, one family, static. FNTO integration not tested; nothing adopted.

**Earlier (WL83):**

[WL83](worklogs/83_WORKLOG_LEARNED_COMPILER_RUNTIME_UTILITY.md) tested learned compilation, the GPU runtime and rendering of the WL82 E / ER closures. **Outcome D by the frozen mapping: learning succeeds, but the rendering rule fails.**
- **Learned compiler: PASS** for E and ER on a new independent population (both seeds). It is a response-trained MLP, 150k parameters; every output is physically valid by construction.
  - test83_iid E_L1: E learned 0.073 vs direct fit 0.072; ER learned 0.057 vs 0.053.
  - Transfer error ≈ 0.025–0.03; small generalisation gap; the held-out block region stays within its thresholds.
- **Hard thin / sharp / high-reflectance region:** ER learned 0.10 vs E 0.19.
- **Physical validity and Dr.Jit runtime correctness: PASS.** A float32 refraction defect at η → 1 was fixed.
- **GPU economics, operator level: PASS** for every arm, orders of magnitude over the matched-quality layered reference.
  - ER costs about the same as E when memory-bound and 1.6× E when compute-bound.
  - Inside a path tracer the BSDF saving is only 1.6–1.9× per spp.
- **Rendering: FAIL** under the frozen all-images rule.
  - ER_L 9 / 10; its single failure is reference-limited.
  - E_L 8 / 10; one genuine hard-material highlight failure.
  - Every closure beat its time-matched reference on every image.
- **Not tested:** FNTO K/C integration and dynamic GI. Nothing is adopted. Whether to keep the line, and E vs ER, are user decisions.

**Earlier (WL82):**

[WL82](worklogs/82_WORKLOG_INCIDENT_ENERGY_VS_REFRACTION_ANGULAR.md) attributed WL81's representation failure. **Outcome A by the frozen mapping: incident-dependent energy modulation is sufficient for the WL81 representation target in the tested scope.** Scope: one coated-conductor family, direct per-material fits, untouched test_iid. A refraction-aware extension is not demonstrated to be necessary for that target.

- **Result on test_iid (E_L1 / E_alb medians, WL81 gate).**
  - Physically admissible energy arm E (symmetric s(μ_i)s(μ_o) attenuation, softmax certificate): 0.073 / 0.007, passes R1–R3.
  - C0, the WL81 basis under the same new fitting protocol: 0.120 / 0.037, fails.
  - ER (E + coat-refracted inner lobes): 0.052 / 0.006, passes.
  - test_block and 8 real-material finishes agree.
- **Qualifiers.**
  - Most of the WL81 → WL82 gap of C0 itself (0.18 → 0.12) comes from the per-incidence metric objective and the 64 × 64 fit data, not from a mechanism.
  - Rescaling the energy of the C0 fit afterwards (oracle) recovers only 13 %. Refitting with the energy free per incidence (non-reciprocal Einc, or a shape-only R8 fit plus the oracle) reaches 0.084.
  - In the predeclared failure cluster (thin slab, sharp and high-reflectance base; n = 10, not gated), the R8 angular ceiling stays at E_shape 0.18:
    - E's outgoing attenuation lowers it to 0.084;
    - ER's refraction to 0.069 (shape-focused fits);
    - under the certificate, E 0.18 vs ER 0.11.
  - ER's fitted η_e does not track the true coat IOR; it is a useful shape parameter, not an identified refraction.
  - The softmax certificate costs E 0.073 vs 0.043 unbounded. The unbounded fits are empirically passive, so the cost comes from the sufficient condition, not from physics.
- **Not tested.** No learned compiler, no GPU / rendering utility. Nothing is adopted into canonical FNTO; no successor is authorised.

**Earlier (WL81, 2026-11-09):**

[WL81](worklogs/81_WORKLOG_COMPILED_NEURAL_MATERIAL_RULES_UTILITY.md) stopped at its Stage C gate. **REPRESENTATION_CAPACITY failed (clear fail)** for every energy arm, so the learned compiler was **not trained** (LEARNED_RULE_GENERALIZATION NOT TESTED; GPU, compression and rendering utility NOT REACHED). FNTO_COMPONENT_VIABILITY: NOT SUPPORTED, with one-family scope.
- **Setup.** The reference is a qualified expensive family: a layered coated conductor (random walk; independent volpath check |z| ≤ 2.15). The fixed R = 8 compiled closure was fitted directly per material.
- **Result.** E_L1 median 0.18–0.21 and E_alb median 0.08–0.11, against 0.10 and 0.03.
  - The energy parameterisation is not the binding factor: the post-WL80D candidate equals the unbounded diagnostic.
  - A post hoc metric-matched objective still leaves 0.129 / 0.065–0.072.
  - Errors concentrate where the reference albedo falls with incidence (thin slabs over sharp, reflective conductors).
- **Physical validity (corrected closure).** Physically valid; passivity is certified for the control and empirical for the candidate.
- **Also found and fixed.**
  - The WL80C/80D runtime is non-passive above 89.98° because of a cos² clamp.
  - A Dr.Jit PCG32 seeding pitfall.

No adoption; no successor. Whether this architecture line continues is a user decision.

### Learned intrinsic-K source transfer (WL71–WL78, as of 2026-11-07)

Learned rank-16 W+UV reaches the fixed-basis oracle on its supervised sources (WL75), but does not transfer to independent boundary sources.

- **WL78** removed the operator's absolute-position and orientation inputs, giving the geometry-relative W+UV-Q. Supervised fidelity was preserved and interior and near-boundary transfer improved descriptively (interior independent sources now meet the WL59 gate). The frozen boundary-far endpoint stays **INCONCLUSIVE**: one of two seeds improved. The deformation response also degraded.
- **Remaining boundary failure:** physically equivalent boundary contexts related by a bin-frame rotation or reflection are still not mapped to equivalent responses. This attribution is unmeasured.
- **WL80** (2026-11-09) stopped at its Stage 2 gate with **ORBIT-BALANCED BASIS INADEQUATE**, before any learned training.
  - Refitting the global rank-16 basis on the exact rest D4h orbit halved the orientation gap: the worst transformed/original oracle ratio fell from 1.41 to 1.15.
  - The cost was a 2–18 % loss in original-orientation accuracy.
  - The boundary-rotation images still missed the frozen rule marginally (con_coarse 1.26–1.27× the G2 original level; WL59 abs_coarse 0.101–0.103).
  - The basis remains non-closed.
  - The learned 2×2 was not run, and the WL79/80 confirmation population remains unscored.
  - Whether to consider an equivariant angular factorisation is an open user decision.
- **WL79** (2026-11-08) stopped at its pre-training representability gate with **ANGULAR BASIS LIMIT**.
  - The rest-slab D4h symmetry contract is exact and was verified against references. Dented-pose bin maps are not exact under the first-triangle bin frames.
  - Under the frozen rank-16 G2 A/O, labelled oracles fit the exact symmetry images of the boundary training responses up to 1.26–1.41× worse than the originals (coarse absolute and contrast). Interior sources are unaffected.
  - The basis is not closed under the bin permutations.
  - Caveat: the basis was fitted on the original orientations.
  - No symmetry-trained candidate exists. Whether to change A/O or the operator class is an open user decision.
- **WL77A** selected no architecture candidate. It also found that the WL76/WL77 angular-failure attributions are confounded by q = 1024 single-scatter quadrature noise. That correction is an **unresolved attribution question**; the recorded WL76/WL77 verdicts are unchanged.

See [WL80](worklogs/80_WORKLOG_ORBIT_BALANCED_ANGULAR_BASIS.md), [WL79](worklogs/79_WORKLOG_ANGULAR_FRAME_EQUIVARIANCE.md), [WL78](worklogs/78_WORKLOG_SOURCE_IDENTITY_LEAKAGE.md) and [WL77A](worklogs/77A_WORKLOG_MULTI_AGENT_ARCHITECTURE_DISCOVERY.md).

No learned corrective recipe or successor architecture is adopted. Work on `Selective_Recompute` only; no branch/worktree change without user consent.

## Current decision boundary

**Next decision is GM1 design and its first bounded experiment.** The **direction** (rigid-edit dynamic glossy/specular interreflection with frozen ER as local baseline) is approved. Before execution, the human architecture decision must identify **one falsifiable global transport representation/update mechanism**, its spatial/angular interface and ownership, representative T0/T1 edits, independent references and prospective quality/cost gates. The Agent may determine its implementation details within those contracts; it must not invent and sequentially switch among architecture hypotheses after seeing results.

**GM1 evidence record (measured; nothing adopted; decisions remain human).**

- [WL85](worklogs/85_WORKLOG_GM1_MECHANISM_NEUTRAL_TRANSPORT_CHARACTERIZATION.md) (2026-11-11): mechanism-neutral characterisation on an open fixture. Its frozen workload qualification is UNSUITABLE by one clause, so its conclusions are formally INDETERMINATE.
- [WL86](worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md) (2026-11-11): ECRO v0, one explicit learned global mechanism (physical ≤ 2-event transport + a geometry-conditioned, source-independent linear operator that fills a patch × direction-bin table of ≥ 3-event radiance), on a qualified semi-enclosed room workload with a held-out edit. Representability SUPPORTED (oracle table at reference noise). The learned operator meets the stationary-pixel, broad-region, unseen-source and edit-response gates for both seeds but misses the region rule on the held-out edit's sharp / moderate receivers (E_H 0.063 / 0.072 against 0.06), so generalisation, physical fidelity and GM1 viability are **NOT SUPPORTED under the frozen gates**; economics is NOT TESTED (descriptively 3–6× edit-to-first-frame at 0.05 / 0.03 against frozen-ER path tracing). ECRO is not adopted and no successor is authorised.
- [WL86A](worklogs/86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md) (2026-11-11): attribution of that failure on the frozen operators (diagnostic; nothing retrained). The source encoding is not the cause in the failing cases (an incident field with the same encoding leaves the truth unchanged; rendered E_H 0.025). The deficit is the operator's single learned flight from the sharp panel, which delivers 1–18 % of the light → panel → patch family while the global term compensates without geometry; the 512-ray state relation that would carry that flight is sampling-limited (one case marginal). Supervision noise and the 16-point source estimator are weakened; coverage and operator-family capacity are indeterminate. No admissible single intervention exists inside the frozen family, so none was run. WL86's verdict is unchanged.

Do not treat historical WL41–80 failures as observed defects of the WL84 ER baseline. They are possible failure mechanisms and diagnostic controls. Do not convert the local ER success into a full-K, global-interreflection, dynamic-reuse or end-to-end superiority claim. GM2–GM4 are directional commitments, not automatically approved work.

## Longer-term decisions, without implementation permission

- Resolve the complete physical quality floor and the shared angular/deformation response limitations before claiming learned integration readiness.
- Revisit rigid motion, local deformation, global bending/twisting and self-contact at each major design decision. Rigid material-frame evidence is conditional; the tested learned local-deformation recipes fail their complete response contract; global deformation and contact generality remain unestablished.
- Consider interpretable dependency discovery and optional learned scoring only after physical coverage and representation feasibility are established, with algorithm-only and oracle controls.
- Make quality-matched external comparisons, including TransGI, only after reference, workload and accounting alignment. Plan durations from measured costs; venue planning does not relax gates. WL67 supplied aligned physical/reference comparisons, but a path-traced re-render is **not** a Neural Full Recompute. Any learned dynamic-update superiority claim requires an actually invalidated external neural representation and its native T1 rebuild/retrain/refit lifecycle, quality-matched to ReST-GI; if native T0-state reuse is valid, measure that separately. Compare Cycles/Mitsuba spp-dependent full-scene re-rendering under an additional, explicitly labelled cost axis. See [binding baseline-role contract](BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md#mandatory-baseline-terminology-and-comparison-gate).

## Updating this roadmap

Replace the current frontier when a user direction decision or completed gate changes it. Link the owning plan, protocol or Worklog; do not append a result diary. A newly proposed mechanism needs its own explicit decision. Preserve failed conditions, amendments and provenance at their original evidence locations.

The former batch program, resolved baseline prerequisites and chronological decision record are preserved in the [source archive](history/living_document_cleanup_2026-10-29/README.md).
