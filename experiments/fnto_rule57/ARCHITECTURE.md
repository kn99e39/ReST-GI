# WL57 bounded hypothesis: learned geometry state versus direct conditioning

This is an isolated experiment, not a production replacement or an adopted Rule–State architecture.

Both D and R use the **same positive, column-substochastic spatial/directional kernel family**. A material boundary cell is 2.5 mm, with 16 incoming and 64 outgoing equal-projected-angle bins. Input and output are RGB **flux** in those cells/bins. No irradiance collapse, baked illumination, camera-only output, or hidden per-pose trainable code is used.

The input geometry is the current intrinsic mesh in its supplied rigid material frame, rest correspondence, current areas/normals/thickness, and an observable height-field summary. Local dent is the sole deformable family. Geometry features are identical for both models. Runtime does not read target K, reference images, stored per-pose target response, or a learned pose-ID lookup.

## Common transport representation

`K x = B_G x + S_theta(G) [(1-beta_G) x]`.

`B_G` is the **zero-collision** entry-to-first-exit term: current mesh intersections and analytic exp(-sigma_t * chord). Fixed-seed within-cell/bin quadrature is used, with beta the corresponding column sum. This geometry-derived term is common to D/R, its construction and storage are charged, and it is not a full scattering oracle. Its information family is disjoint from the learned scattered part. C owns all subsequent air segments, including same-domain re-entry.

The scattered part uses 8 nonnegative angular mixture components and a spatial kernel for each component/channel:

- Angular input mixture A has columns summing to one over components; output distributions O sum to one over 64 outgoing bins.
- Spatial exchange is the sum of learned weights on 128 fixed material-nearest receiver cells and a **dense** positive rank-16 tail. The tail is an approximation basis, not an assertion that geometry changes are low rank or local. Every cell can affect every other cell through the tail.
- Normalize the total spatial columns and multiply by a learned sigmoid survival mass. Thus each column is nonnegative with sum <=1, for every current geometry and all incident fields, rather than only on training probes.
- The local and tail terms are jointly fitted basis components of one normalized kernel, not two independently supervised physical path families.

No energy clamp, solver damping, oracle-mass normalization, current-pose target calibration or post-fit stabilization is used. At a fixed geometry all learned coefficients are independent of light, so zero-field and superposition follow directly. Ballistic plus scattered mass cannot exceed incoming mass. Directional approximation capacity is tested, not presumed sufficient.

## R: explicit learned transport state

The shared geometry encoder maps each boundary cell's 140 observable features to 32 latent channels, applies two material-neighbor message-passing residual blocks plus a global pooled summary, and produces `z_phi(G)` (3840x32 floats, about 0.49 MB before common geometry/ballistic data). A reusable decoder theta constructs the spatial coefficients from source/receiver z and shared metric features. Every pose uses the same learned phi/theta; no per-pose SGD is allowed.

The possible benefit is learning reusable geometry summaries once, then reusing them across new incident fields. Neither changed-state locality nor a small delta is assumed. State construction still costs work and may lose to D.

## D: direct geometry-conditioned operator

D receives the same raw current-geometry feature arrays and shared pair metrics directly, without R's explicit nonlinear transport-state encoder. Its shared conditional decoder predicts the same kernel representation. D does **not** retrain per pose. D is allowed exactly the same partial affine caching and full coefficient caching as R. Decomposing the first affine source/receiver projections is an exact implementation optimization, not an R-only privilege.

Budgets fixed before training: D pair hidden=96, node hidden=128; R pair hidden=128, node hidden=144, state=32. Trainable parameters D=161,328, R=162,048 (<0.5% difference). All angular/spatial basis ranks, data, losses, steps, seeds, precision and field order are matched. R's encoder counts toward its budget.

## Cache lifecycle and cost comparison

For **both** models measure:

1. geometry processing, ballistic build and geometry-only features;
2. compact state plus cached source/receiver affine features;
3. complete learned coefficient cache (still a structured representation, not a full fine K);
4. first and repeated illumination queries under both modes.

Both policies are mathematically identical within each model and have a 512 MB per-geometry cache ceiling. Choose the fastest *permitted* policy separately for each query count {1,4,16,64}, report both, and count all preparation/transfer/memory. A policy that makes R look better solely by disabling equivalent D precomputation is invalid evidence.

## Scope and falsification

Shared material-frame rigid reuse and local dents are evaluated. Global bending/twisting could require global state changes (no locality/rank assumption); self-contact can change optical-domain topology/ownership. Neither is tested or claimed here.

A positive R result requires useful held-out absolute and differential accuracy, physical validity, and a >=25% quality-matched dynamic-cost advantage over D's best legitimate cache policy, including low-query cases. If both are accurate and their cached costs are similar, explicit state separation has no demonstrated architecture-specific value in this realization. If both miss fidelity, no fast inaccurate winner is declared. Only the predeclared family is trained; no architecture sweep or successor batch follows automatically.
