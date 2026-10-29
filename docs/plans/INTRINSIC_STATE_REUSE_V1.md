# Intrinsic State Reuse v1 — bounded candidate contract

Date: 2026-10-29. User brief: “Validate a source-independent reusable intrinsic transport rule with persistent, selectively updated geometry-dependent transport state.” This newly authorises one bounded hypothesis after WL57, on Selective_Recompute. Worklog58 is the chronological record, not the batch identifier. No adoption or successor is authorised.

## Candidate and ownership

The candidate is **dependency-certified persistent graph transport state**. It changes the old globally conditioned state construction to bounded local geometric feature dependencies, while retaining a nonlocal spatial response. This is one candidate, not a portfolio. It is not a selective weight update and it does not reuse old solved illumination.

K maps boundary entry to first exit. C owns every subsequent editable air segment, including same-domain re-entry. Direct and C-mediated incident fields enter exactly the same K API. Material cells remain2.5mm, with material triangle correspondence and current normal/tangent frames. The medium is the unchanged WL53 isotropic index-matched homogeneous medium. One additional compact static surface-only GGX probe (alpha0.04 and smooth control alpha0.25, unit Fresnel conductor) is included as a mandatory angular-interface preflight; it is not a new full scene family or a change to the volume reference.

The input is RGB flux over **64 incoming bins (8×8)**; the output is RGB flux over **256 outgoing bins (16×16)**. Both use the existing equal-projected-angle concentric-square map. This is4× the WL57 angular coefficient count on each side, selected before any new evidence. There is no continuous-direction or analytic specular side channel. This finite representation must earn adequacy; a failed gate will not trigger a resolution sweep or an analytic bypass within this batch.

Reusable parameters theta consist of a32-channel geometry encoder (local16 geometric features, two16-neighbor message steps, no broadcast global dent descriptor/global pooled latent) and the positive kernel decoder. The decoder has128 fixed material-neighbor edges, dense rank16 tail,64 angular mixture components, pair hidden128 and node hidden144. Nonnegative angular distributions are column normalized; spatial columns share sigmoid survival mass. Unscattered B is geometry-only, disjoint from learned collided response S: Kx=Bx+S[(1-beta)x]. No clamp/damping, target-derived normalization, object/source ID or pose code is allowed. Network weights remain fixed across runtime edits.

## Persistent state and exact invalidation

- State identity includes material/rest topology, vertex correspondence, units, frame convention, material coefficients, rule checkpoint and quadrature seed/version. A mismatch invalidates the affected complete cache, never silently reuses it.
- Geometry state stores per-cell local metrics, dependency indices, intermediate encoder layers, unnormalised local-edge/tail coefficients, normalization sufficient statistics, and zero-collision geometry quadrature records. It contains no oracle K, target images or per-pose optimised latent.
- F rebuilds all these arrays from current geometry. S begins with the actual resident previous arrays. Changed vertex/triangle support invalidates affected area/frame/thickness features; exact graph dependencies propagate through both encoder layers and incident coefficient edges. This can be nonlocal; it is not a distance-based radiance selector.
- Dense-tail normalization can change globally. Preserve unnormalised coefficients where valid; update denominators by reduction of actual changed rows. Normalised output/query work is charged even when its effect is global. If an implementation materialises globally normalised arrays, all affected entries count as updated, not retained.
- For B, source-frame/area changes invalidate all samples launched from the affected triangles. Retain a sample only if its traversed segment and exit certificate exclude every old/new swept changed-triangle AABB. Intersections with the conservative bounds invalidate it, including stationary source/receiver paths blocked by edited geometry. Full/S use identical counter-keyed source/sample seeds. These certificates require geometry only, not target K. Broad false positives are counted honestly.
- Rigid motion pulls geometry/directions into the supplied material frame and retains intrinsic state; current C and radiance are recomputed. Topology/material changes use full invalidation. Unsupported correspondence fails closed.

Rule = fixed reusable response weights. Persistent state = resident reusable computed geometry/coefficient data. Reconstructed state = work performed by F/S after an edit. Cached coefficients are not current incident/output radiance. All query fields are freshly applied; no cached training-light response is a runtime state input.

## Ordered preflight and stop contract

The machine-readable protocol is `experiments/fnto_reuse58/protocol/reuse_v1.json`. Commit the gate implementation and protocol before primary evidence. First test the angular interface without fitting any weights. Giving the candidate the *optimal* output projection isolates an information limit from capacity or optimisation.

Use an independent float64 analytic GGX evaluator with height-correlated Smith masking, F=1. Compare projected-solid-angle midpoint quadratures512²,1024²,2048². Require convergence, nonnegative finite response, energy<=1.001 and reciprocity relative error<=1e-10. These are deterministic quadrature controls, not Monte Carlo confidence intervals or a new Cycles validation.

Test normal/30°/60° sources at17° and61° azimuth, alpha0.04 and0.25. Require the best piecewise-constant256-bin output relative-L2 projection error<=0.10 for every declared case. Also test two independent collimated sources within the same64-bin input cell: input vectors are exactly equal but physical outgoing responses may differ. For every pair, require the triangle-inequality lower bound ||yX-yY||/(||yX||+||yY||)<=0.25 at the actual256-bin output interface. A bound above0.25 proves no deterministic rule can meet25% relative error for both fields, irrespective of training. Report every case, including smooth controls.

If either representation condition fails with validated quadrature, stop before reference expansion, neural training or F/S engineering. Report angular representation NOT SUPPORTED; state fidelity/cost and the abstract reuse hypothesis INCONCLUSIVE. Do not mislabel an optional glossy-probe failure as a measured failure of the homogeneous-volume model. Its inclusion is a prospective scope choice of this candidate, not a requirement inherited from WL57.

Only after angular GO implement and commit the complete F/S path, resource and dependency/physical tests before expensive training. Use source-independent fields and all-boundary supervision; acquire new64-in/256-out references rather than pretending that lifting old16-bin fields supplies unseen angular information. Preserve the WL57 failure and WL56 blue floor. Full/S use identical checkpoints and targets. Require both to meet coarse/fine action, directional and signed-delta criteria; faster inaccurate S cannot win. This document does not authorise a second architecture if the first gate fails.

## Evidence boundaries

Main reference, training, state invalidation, rigid/dent update costs and coupled lifecycle measurements are **not performed evidence** until their stages execute. A preflight stop leaves these unresolved, rather than substituting old WL57 claims. A new major representation class beyond this candidate requires a separate user/Assistant architecture decision.

Primary analytic definition: [PBRT4 microfacet theory](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory). No external source is used as evidence for candidate performance.
