# Baseline Roles and Evidence Strategy — Surface-Owned Selective Neural Transport Recomputation

**Updated:** 2026-10-23 (FNTO v0.1 Batch 6 NOT SUPPORTED, worklog 41; worklogs 37–41 reviewed)  
**Status:** Active evidence/control contract. Surface-Owned oracle edit reconstruction demonstrated on one diffuse Lego edit, WL37 CONDITIONAL, WL38 3-seed WITHIN_MARGIN; SPT-1 WL39 NOT SUPPORTED; FNTO v0.1 is designed but unimplemented.  
**Project:** ReST-GI (formerly NeuralShaderDemo) / Selective_Recompute

## FNTO Evidence and Baseline Separation (2026-10-23)

**2026-10-23 (Batch 6, user brief "FNTO v0.1 Initial Implementation & Physical-Edit Validation"): IMPLEMENTATION AUTHORISED.** The user authorised one bounded architecture batch that implements `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` (authoritative for the representation; intrinsic optical-domain K with exposed same-asset free-space paths in C, a=b allowed) and evaluates it on the official Lego + red box held-out T0/T1 edit (worklog 41; exports `results/evaluation/41/`). Required: reference-renderer semantics study (Christensen-Burley SSS scope) before main kernel training, isolated-domain spatial/directional probe supervision of K, synthetic contracts A-F on the production code, predeclared screening rules, arms A-F (historical RNA / Surface-Owned / SPT-1 unchanged; FNTO stale / updated / optional oracle diagnostic), full cost accounting and a SUPPORTED / CONDITIONAL / NOT SUPPORTED verdict. Not authorised: a learned dirty-unit selector, a second architecture family after a failure, SPT-1 modification, changes to historical baselines, automatic start of a later batch. No FNTO result exists yet.

**Batch 6 result (worklog 41, 2026-10-23): NOT SUPPORTED** under protocol fnto_v1. The implemented FNTO v0.1 (learned intrinsic kernels K_loc/K_glob/K_band/K_mid trained only on isolated-domain first-intrinsic-event probes; ray-cast free-space coupling incl. same-asset paths; learned coupling rule empty) diverged in the multi-bounce solve on 64/64 held-out view solves, so static fidelity and edit recovery are unmeasurable. Attribution (descriptive): the learned nonlocal kernel parts are non-passive on incident fields outside the span of the 1,400 sun-probe training fields (effective albedo > 1 on 26 % of ports for K_mid, 18 % for K_glob; spectral radius ~2.1 with K_glob, explosive with K_mid; one bounce already 5.5x the reference remainder); the kernel alone reproduces first-event transport at 22 dB vs first-event references (~10 dB above probe noise). Coupling semantics and the exact incremental relation update (0.83 % of rays) behaved as designed; kernel parameters stayed bit-identical. Conclusion: an intrinsic operator must be passive on the fields the coupling produces, not only linear. No later batch is authorised.

**2026-10-23 (user brief "FNTO Validity Restoration — Post-Batch-6 Controlled Implementation & Re-evaluation"): CORRECTIVE IMPLEMENTATION AUTHORISED (worklog 42).** FNTO stays the primary architecture; the batch verifies the six findings of an independent Batch 6 validity audit (buried port origins / biased incident lighting; insufficient independent spatial excitation of K; effective kernel support narrower than declared; incomplete internal reflection / directional transport; reference clamping and reconstruction-filter mismatches; insufficient signed-change, solver-stability and coupling-compression evaluation) against code and evidence, corrects confirmed defects in a distinct candidate and protocol version, and re-evaluates through gates 1-5. Batch 6 artefacts, protocol, verdict (NOT SUPPORTED) and worklog 41 are preserved unchanged; no other architecture is started. Historical RNA, Surface-Owned and SPT-1 arms are reused as recorded; their roles do not change.

**Worklog 42 result (2026-10-24): NOT SUPPORTED** under `fnto_r1`. Evidence-regime note for every future comparison: worklog 42 introduces a second held-out reference regime, **r1** (no sample clamping, point-sampled, 4096 spp, LabServer63), next to the historical regime **hist** (clamp_direct 20 / clamp_indirect 5, point-sampled, PC); the two differ by ≈0.5 dB for the same prediction (FNTO r1 S_T0: 24.49 hist / 25.00 r1) and must always be labelled; RNA, Surface-Owned and SPT-1 numbers exist only in the hist regime and were not re-rendered. FNTO r1 (architecture-contrast role) recorded values, hist regime, held-out Lego: S_T0 24.49 dB (RNA_T0 33.57, SO A_full-PU 33.38, SPT-1 22.23); edit A_changed: D_stale 21.97 → E_upd 25.83 dB (RNA_T1 32.23, SO A_full-PU 33.66, SO_T0-PU stale 23.06) → F 2.44, R_RNA 0.65; signed change corr 0.987 / slope 0.948. Generality scope unchanged: one scene, one diffuse-dominant edit, one seed. Baseline roles unchanged.

Read [FNTO Architecture Contract v0.1](FNTO_ARCHITECTURE_CONTRACT_V0_1.md) before designing new representation comparisons.

- **RNA:** static full-material fidelity control including shadow-side SSS/glass.
- **Surface-Owned:** existing complete-unit relearning baseline, with post-edit supervision, held-out quality and exact parameter-preservation evidence (WL37–38).
- **SPT-1:** negative free-space-only low-bandwidth operator prototype (WL39), retained to demonstrate the danger of dropping internal transport semantics.
- **FNTO:** unvalidated primary next representation candidate. Separate learned **intrinsic optical-domain** scattering K from geometry-dependent **editable free-space** coupling C, including exposed same-asset self-interreflection; do not equate Object ID with K ownership. Do not claim it retains RNA fidelity until SSS/transmission and per-material response are shown on actual held-out data.

Required prospective comparisons: isolated-domain probes under unseen *spatial* incidence, not only unseen directional lights; fixed learned operators with stale/current relations on a held-out edit; exact transport path ownership avoiding double counts; per-material shadow-side fidelity, angular/gloss stress, parameter hashes, updated-state fraction and all runtime/offline compute. Use a predeclared quality and selection rule before training. A failure of K expressivity must stop before scene-level selector work. Do not silently re-label SPT-1's negative outcome.

## 1. Scope

- "Selective recomputation" means recomputing the **complete owned learned state of each transport-invalid Surface Transport Unit**. A unit may be finer than an object; a whole object is the coarsest unit and serves as the **coarse comparison**.
- Never select a subset of cells, channels or layers inside a jointly owned model. That is worklog 30's separate, negative hypothesis.
- D_G ≠ D_T: stationary receivers can hold invalid transport.
- RNA is the **initial implementation baseline** (decision 1), not the required final model.
- At least five architecture batches come before a broad viability judgment.
- One negative candidate does not falsify the general ownership hypothesis.

## 2. Historical Baselines and Evidence Roles

| Lineage | Scientific role | What it does NOT show |
|---|---|---|
| RNA/Rain | correspondence, nonrigid failures with known local confounds | generalized failure attribution |
| 8DNA/Teaset, WL21–22 | stationary receiver's GI changes as another part moves; stable whole-rigid-motion control | that owned-unit recompute works |
| RNA/Teaset (WL22, WL32 C1) | independent-backbone repetition of frozen staleness; **static ceiling below the 25 dB credibility bar on the near-mirror teaset** | that RNA is credible on this asset |
| WL24–25 | physical attribution: indirect-path occlusion plus moved-part reflection | online detector for invalid owners |
| WL26–27 | historical full-asset neural and physical recompute cost context | same-method unit-level quality-matched speedup |
| WL28–29 | sparse K=32 dynamic-state candidates negative | owned-unit recompute negative |
| WL30 | **intra-object** triplane-cell-only refit negative (8DNA, flows frozen) | complete-unit or whole-object refit negative |
| WL31 | explicit per-part surface units: O0 pass, T3 recovered by refit, object-level D_T = all objects | neural units; unit-level D_T sparsity; cost advantage |
| WL36 | Batch 3 on the native Lego (A = WL35 reproduced HQ): independent L16 units within the static screening (−0.32 dB, logL1 1.05×, off-band 0.996×; 1.54× parameters); PU removes 91% of the seam excess; REINIT exact | edit-time radiometric validity; D_T sparsity; capacity-equal comparison; seed variance; other assets/materials |
| WL35 | native RNA on the official Lego reproduces the released checkpoints. Released checkpoints and native retraining score 32.16/31.99 (HQ) and 27.96/27.97 (small) on a native 1024-spp val set. Against 4096-spp targets the released models reach the Table 1 level (34.39/29.22 vs 34.09/28.77, 8 views); classification COMPARABLE (small training REPRODUCED) | rendering-performance claims (renderer unreleased); exact paper val set |
| WL34 | P16 seams = decorrelated independent unit errors exposed by hard selection; P16-PU composition (no learned-state change) removes most of the excess with exact output-influence accounting | interior per-unit fidelity recovery; a shared-decoder ownership variant |
| WL33 | integrated RNA pipeline = faithful upstream RNA (eval identical, contract audited, estimator clamping not explanatory); teaset C1 = RNA representation-limited; C2 gate measurement noise-limited | that RNA cannot be credible on rough conductors; a positive proof of the C1 representation limit |
| WL32 | RNA independent complete units: exact parameter/output preservation and unit reconstruction; HDR/seam fidelity cost; **baseline-credibility gate failed** | static fidelity under a credible baseline; recovery after a physical edit |

Do not delete, overwrite, reinterpret, or silently merge these questions.

## 3. Baseline Roles in the Current Program

| Role | Current instance | Use |
|---|---|---|
| **Implementation / quality control (A)** | original RNA (`66b5b098`, upstream config, seed 0) on the same data. **Qualified by worklog 35**: the native pipeline reproduces the released Lego checkpoints, and the gap to the paper is attributed to validation-target noise | the static-fidelity reference that every owned candidate is compared with; must first pass a predeclared credibility bar |
| **Owned candidate (B)** | worklog 32: K complete RNA units (P1/P4/P16 matched, P16w full-size); worklog 36: L16 on the native Lego (triangle-identified k-means cells) | change only the ownership boundary |
| **Capacity control** | matched-texel units (P16) and full-size units (P16w); P1 isolates local framing | separate capacity from ownership effects |
| **Composition control** | the same learned units under hard selection vs P16-PU (worklog 34) | isolates composition effects from learned-function effects |
| **Ownership-semantics control** | REINIT: re-initialise one unit, retrain it fully with all others frozen | parameter and output preservation; reconstruction to prior quality |
| **Coarse comparison** | whole-object full recompute (worklog 31 on an explicit substrate) | the granularity that finer units must beat or match |
| **Physical reference** | Mitsuba full-scene path-traced radiance (worklog-22 producer), T0 and T3 | supervision and evaluation; never simplified to local transport |
| **Physical intrinsic-transport reference (media; worklog 53)** | Cycles 3.5.1 **volume** path tracing (Volume Scatter + Absorption inside a white Transparent BSDF), validated against the exact H-function, furnace, superposition, path-history, grouping and reciprocity controls; independent analytic random walk (IW) | experimental reference for intrinsic K / ΔK on canonical fixtures; **not** promoted to the production training environment |

**Reference-renderer cautions (worklog 53).** Cycles BSSRDF closures (Burley, random walk) are path-history dependent (diffuse-ancestor rule), so they cannot supervise a vertex-independent K (WL51). Mitsuba 3.5.1 float32 volpath skips an ~89 µm boundary layer at every null-interface crossing (−12 % at σ_t 1000 m⁻¹, bias ∝ σ_t) and must not be an absolute reference for dense media. Mitsuba `cuda_ad_rgb` film accumulation on LabServer63 (driver-595 OptiX workaround) is wrong for deterministic cases; LabServer69 cuda and llvm are exact. Label every number with its renderer and machine.

**Batch 5 (SPT-1, worklog 39) roles:**
- **A:** original RNA (RNA_T1 as the per-state reference; RNA_T0 for static credibility).
- **B:** Surface-Owned full and selective reconstruction plus the frozen control, reused from worklogs 37–38 because the evaluation regime matches: held-out T1/T0, the same 4096-spp PC references and regions.
- **C:** SPT-1 with the updated relation state.
- **D:** SPT-1 with the stale (T0) relation state, every learned parameter identical. This is the factorization control.
- **C_full:** SPT-1 retrained on T1 data. It is the candidate's own full-reconstruction reference and separates representability from reuse.

The SPT-1 operator never sees T1 data; T1 is the held-out edit.

**Outcome (worklog 39):** NOT SUPPORTED, because A's static level was not approached (−11.3 dB). This is a candidate limitation of one realisation on one scene and material mix. It is not a project-level falsification of factorized representations.

Implementation baselines are not scientific replications. A credibility failure of A is a **baseline limitation**, reported as such, and never read as a candidate failure.

## 4. Static-Credibility Precondition

Before a candidate's fidelity is judged, A must clear a predeclared static bar on the chosen scene. Worklog 32 used 25 dB RNA val PSNR on final weights; A reached 17.97 dB (C1) and 24.92 dB (C2). The bar and the scene are chosen before training. A failure is reported, not repaired by changing the threshold, the metric, the checkpoint selection or the scene after the fact. Any revision is a new, prospective protocol.

Lessons from worklog 33 for every future credibility gate:
- **Score against converged targets.** At 128 spp the GT noise floor was 29.4 dB, which caps any model's measured PSNR.
- **State the aggregation.** Pooled vs view-mean differs by up to 0.6 dB.
- **State the checkpoint rule.** Best-by-val selects on the evaluation set.
- **Use converged validation targets.** On Lego, 1024- vs 4096-spp targets moved RNA HQ by +1.7 dB (worklog 35).
- **Check the substrate's material regime first.** RNA on near-mirror conductors under a delta light is representation-limited (worklog 33), so a gate failure there is a baseline limitation, not a candidate failure.

## 5. Controlled Comparison Rules

- **Preserve:** scene geometry, materials, lighting, camera distribution, full-scene targets, splits, target semantics, evaluation regions, coordinate conventions, seeds, and hardware/runtime accounting.
- **Change only:** the ownership boundary and what it strictly requires.
- Capacity is accounted for (total, per unit, memory, effective support). If strict matching conflicts with ownership semantics, report the comparison as capacity-confounded.
- No tuning of model size or thresholds after seeing final results.

## 6. Unit Update and Validity

Track:
- D_G and D_T at unit granularity, and at object granularity as the coarse comparison;
- for each recomputed unit: complete-state recompute (scratch or warm start, measured separately);
- for each preserved unit: **parameter preservation**, **output preservation** and **radiometric validity**, as three separate properties.

Bitwise preservation never implies physical validity under a changed scene.

Oracle dirty sets are scientific upper bounds. Predeclare their noise floor, criterion, coverage and split; report their cost separately. Dense dirty closure is a valid negative result.

## 6a. Future Interreflection / Hierarchical Selector Evidence (2026-10-19 direction; not authorised)

Long-horizon design preference: generate a candidate transport relation hierarchy with interpretable geometric/spatial/directional algorithms, optionally followed by a learned relation scorer / invalidation selector. Normal-vector clustering alone cannot capture position, distance, occlusion or nonlocal glossy interreflection. Node search hierarchy, cross-node transport dependency, Surface Unit parameter ownership and learned selection are distinct contracts. This is not an established architecture result.

**Required comparisons when a future user-approved experiment exists:**

- Preserve the same physically grounded pre/post-edit scenes, ownership units, supervision, oracle D_T labels, error thresholds and test split across selectors. Distinguish direct geometry dirty D_G from transport dirty D_T; include stationary receivers, indirect occlusion and multi-bounce effects.
- Report **candidate discovery oracle coverage/recall** before testing a learned scorer. Missed candidates are irrecoverable downstream; explicitly include edits where a third-party blocker changes transport between two unchanged surfaces.
- Compare an algorithm-only hierarchy/selector with learned scoring on identical candidates. Do not infer a learned predictor is needed merely because one can be trained; compare with simple geometric baselines and oracle selection.
- Report radiometrically weighted false negatives (stale-GI error), false positives (unnecessary update), quality of the post-refit render, per-unit and cross-unit dependency errors, and totals for hierarchy build/update, traversal, model inference, supervision generation and reconstruction. Classification accuracy alone is insufficient.
- Prevent reference/selection label leakage, including oracle radiance at runtime; physically expensive oracle labels are for offline supervision/evaluation only unless separately justified. Predeclare tolerance, recall target and fallback rules.
- If Batch 4 shows dense D_T, failed recovery, or uninformative oracle labels, stop and revisit the representation/assumptions rather than tuning a learned selector to rescue feasibility.

Do not change the in-progress Batch 4 oracle protocol, add selector training, or claim speedups based on this direction record.

## 7. Metrics

- **Quality:**
  - full-image metrics plus HDR-appropriate error (log-space and relative error, not PSNR alone);
  - per-unit, boundary/interior and intra-part seam discontinuity;
  - high-frequency/glossy regions;
  - signed change tracking on stationary receivers (edit batches).
- **Accounting:**
  - total and per-unit parameters and memory;
  - convergence;
  - load and inference time;
  - supervision generation, optimisation, checkpoint traffic and oracle cost, each separately;
  - wall-clock to matched quality.

  Never claim a speedup from parameter or unit counts.
- **Visual:**
  - ownership map with seams;
  - seam crops;
  - error maps;
  - a changed mover, a stationary changed receiver and an unaffected region (edit batches).

## 8. Prior Art and Novelty

Priority historical design references (decision 8):
- Temporal Radiance Caching (2007);
- Instant Caching (2009);
- An Efficient GPU-Based Approach for Interactive GI (2009);
- Voxel Cone Tracing (2011).

These, together with hierarchical/selective GI (Drettakis & Sillion 1997 — not the starting point, decision 9; Luksch et al. 2019; Enlighten) and neural per-object composition (NeLT 2023; Superposed Deformable Feature Fields 2024), are prior art. Keep distinct:
- surface representation;
- radiance caching;
- learned parameter ownership;
- transport invalidation;
- sampling/scheduling.

A defensible contribution must concern the validity and reconstruction of learned owned transport, correct nonlocal dependency closure, and measured quality/cost, not "per-region neural models" or "a hierarchy" alone.

## 9. Interpretation and Stop Conditions

- Stop and report if:
  - the baseline is not credible;
  - target semantics cannot be preserved;
  - ownership is impossible without changing scope;
  - surface correspondence fails;
  - composition fundamentally fails;
  - controls cannot be maintained.
- A stop ends the route, not the direction.
- Attribute before interpreting:
  - baseline limitation;
  - candidate limitation;
  - ownership isolation;
  - optimisation/capacity;
  - partition;
  - supervision/correspondence;
  - compositing.
- Do not add rescue heuristics; do not declare the hypothesis dead or successful from one batch.

## 10. Present Next Question

> On a scene where the original substrate is credible, can recomputing the complete state of only the transport-invalid owned units recover the physical GI change of a geometry edit, while preserved units stay valid, and is the dirty-unit set meaningfully smaller than the whole asset?

Architecture Batch 4 (worklog 37, 2026-10-21) partly answers it: CONDITIONAL on one scene, one edit and one seed. Recovery of the changed region and preservation of V* hold; the whole-image gap to full recompute (−0.31 dB) misses the predeclared −0.26 dB margin. A 3-seed replicate (worklog 38) puts the mean gap at −0.08 dB, within the margin. The evidence covers one scene and one diffuse-dominant edit, so it is not a project-level verdict; breadth across transport regimes is the next evidence gap. Future selector work requires a separate user decision.
