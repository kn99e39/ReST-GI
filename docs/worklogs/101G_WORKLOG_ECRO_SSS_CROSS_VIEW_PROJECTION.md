# Worklog 101G — Camera-independent SSS HIGH projection across views and regions (2026-11-18)

## Scope and prospective contract

WL101G is a suffix follow-up of [WL101F](101F_WORKLOG_ECRO_SSS_HIGH_PROJECTION.md) on `ecro_lr3_adapter`. It closes WL101F's evidence gap and introduces no learned mechanism. WL101F's P and S tables were fitted to queries from the near and flank masks of the single V0 camera. They reduced the near-floor excess but darkened the sphere silhouette. This batch asks whether a table built **without any camera** can keep the contact gain in several views and image regions, and whether that needs spatial resolution beyond the 1,270 × 64 layout.

The trained checkpoint, B_theta, LOW, the event contract, source encoding, camera readout, scene, material, references and all WL101C–F evidence are unchanged. Only diagnostic physical tables, viewpoints, regions and reports are new. Nothing is deployed or retrained.

**Views.** V0 is the original camera and the construction view of WL101F's diagnostics. Five held-out cameras are set from geometry alone, before any held-out radiance: four low views (front-left, front-right, back-right and back-left, at 18–22° elevation) and one 50° front view. The sphere overhang hides the floor within about 0.18 m of the support point above roughly 40° elevation, so the high view tests silhouette and the other regions only. Regions are WL101A's view-independent world definitions on each view's own gbuffer. V0 keeps WL101C's front-half near/flank masks. Held-out views use the same radii over all visible floor. Contact endpoints count only at views with at least 200 near and 1,000 flank pixels.

**Camera-independent candidate G.** Each cell takes the mean outgoing E2–E5 radiance under the étendue measure cos θ dA dω, restricted to segments that reach the front face of another surface. This is the density at which a uniform-area Lambertian receiver population over all exposed surfaces queries the cell. It excludes directions that leave through the open room face, since no receiver can ever query those. Samples reuse WL101E's membership-qualified sampler and pinned suffix. No camera, pixel, ROI, reference image or learned table enters the table. Two controls built from the same samples split the U → G change into its parts: Un (WL101E's uniform estimator at four times the samples) and V (uniform restricted to visible segments).

**Resolution under the same measure.** S_G is WL101F's fixed 2 × 2 spatial subdivision under G. A_G and SA_G are secondary. S_U is spatial refinement without the measure change. S_Gad refines only the 10% of slots with the largest cross-validated, étendue-weighted within-cell energy. This gives a table 1.3 times the coarse size. The selection uses construction samples only and no contact heuristic.

**Camera-conditioned controls.** WL101F's P/Q/S/A/SA tables are read by hash. P_v and S_v are WL101F's projection with full-frame equal pixel weight on each view, from independent seeds. Read at its own view, P_v is that view's fixed-layout upper bound. Read at the other views, it measures transfer. These controls never decide the verdict.

**Evaluation and decision.** Every construction table is hashed into the manifest before any evaluation stream is traced. Each view has two 4,096-spp streams. V0 reuses the archived seeds and references. At held-out views the physical HIGH is the exact-continuation suffix of the same stream, the instrument validated by WL101F Gate A. A candidate is *sufficient* if all of the following hold:

- At every contact-valid view, both contact endpoint intervals lie inside WL101F's bands (±0.01403 near, ±0.01252 near−flank).
- At every view, the noise-corrected converged RMSE in silhouette, sphere, contact/shadow, floor and full frame is at most 1.10 × U_E + 0.0025.
- Native support is ≥ 0.995 and the WL101F precision limits are met.

The verdict is A if G is sufficient, otherwise B if S_G or S_Gad is, otherwise D if a gate limits the evidence, otherwise C. The [protocol](../../experiments/ecro_sss101g/protocol/generalization.json) freezes seeds, views, masks, thresholds and the analysis code (`analyze101g.py`).

### Development before freeze

Self-tests check the weighted estimators, fallbacks, the nested reconstruction of G from S_G, the adaptive compact index, the refinement energy and the noise-corrected RMSE. They pass locally and on LabServer63. A reduced smoke run (construction at 64 proposals per slot, one V0 32-spp pass, no endpoint computed) went through every stage. It found five execution issues, each fixed before freezing:

1. **Direction reconstruction.** H3's error was 1.56e-5 against WL101F's 8e-6 tolerance, from float32 conditioning near the pole. The direction tolerance became 5e-5 and the coordinate excursion 3e-5. The position tolerance is unchanged.
2. **Camera-independence test.** The bitwise test failed because float32 atomic scatter order varies between runs. Integer sums and counts must now be identical, and real sums must agree to 1e-5. The same-reader repeat and the V0-versus-H1 reader both differ by 7.7e-9.
3. **Containment classification.** Sixteen visibility rays missed. Fifteen started on the sphere within 0.16 mm of the floor tangency, and one grazed a room corner 2e-4 m from the edge. These became explicit tangency and seam categories, each gated at 2e-5. Unexplained misses must be zero.
4. **Parent-broadcast lookup.** It differs from the coarse readout by 4.7e-8 through float32 kernel fusion. The tolerance is 1e-6. Subcell and adaptive indices agree exactly with an independent float64 recomputation.
5. **Memory.** A shape-order bug in table upload was fixed, and the arm group size was lowered for GPU memory.

In the development launch, 1.6% of accepted segments end on a back face, almost all of them the panel's back seen from the left wall. These are excluded from V and G by definition.

## Amendment 1 — construction membership numerics (owner-approved, 2026-11-18)

The frozen primary run (commit `2f9f5754e5c9e9a770cbf35dafee5f419e241a78`, output `results/ecro_sss101g/primary/`) stopped at Stage 1 in construction replica 0 (seed 10203101). Membership counted 328,376,813 valid, 2,814,099 buried, 1,731,204 back, 763 tie and **1 invalid**. The stop came after Stage 0 consistency and before any evaluation stream, so no held-out radiance existed. A read-only geometry replay located the sample at launch 57, slot 77555 (Box B patch 1211, bin 51). Its area coordinate is u1 = 5.96e-7, 5 float32 epsilons from the patch edge. The native attachment query hit no surface at all. WL101E's inherited rule counts planar edge misses as ties only within 4 epsilons.

The research owner approved Amendment 1 under explicit restrictions ([amendment](../../experiments/ecro_sss101g/protocol/amendment1.json)):

- **Reclassification.** A WL101G subclass reclassifies a sample that the unchanged WL101E sampler marks invalid. Three conditions must all hold: the native query returned no hit (so there is no wrong-surface hit), the shape is planar, and the area coordinate lies within WL101E's existing 8e-6 tie window.
- **Accounting.** Such samples are ties. They are excluded from numerator and denominator and receive no radiance. The tie-rate gate is unchanged.
- **Stop rules.** Any other invalid sample still stops the run. I read "the same kind of miss recurs" as a systematic recurrence and set that, before the rerun, as more than 1e-6 amended ties per eligible proposal in any replica (the incident rate was 3e-9). Either case stops the run with no further tolerance.

On the failing launch, the amended classifier moves exactly that one sample from invalid to tie and leaves the valid, buried and back counts unchanged (`amendment1_verification.log`). The complete primary reruns from Stage 0 in `results/ecro_sss101g/primary_a1/` with identical seeds, spp, views, thresholds and analysis. The failed output, its log and the replay (`primary_membership_replay.log`) are preserved. The code before the amendment is `2f9f575`; the code after is the Amendment 1 commit.
