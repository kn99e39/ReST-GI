# Persistent Intrinsic F/S v1

One bounded successor authorised by the2026-10-29 user brief. WL58 remains a completed sharp-GGX angular preflight negative; it is not an F/S result. No central architecture adoption is implied.

## Primary representation and inputs

Use the established WL53 homogeneous, index-matched isotropic medium: sigma_t1000m^-1, RGB albedo0.999/0.99/0.9. Boundary cells2.5mm,16incoming/64outgoing projected-angle flux bins. Inputs are piecewise uniform within each material cell/bin; outgoing reference quantities are integrated cell/bin fluxes. These are the same physical quantities as the validated first-exit oracle, not point-sampled glossy shading. Sharp GGX, sub-bin beam detail, global bend and self-contact are outside scope. This does not repair or relax WL58's historical failed tests.

Retain the established passive positive local-edge+dense-tail kernel family, mixture8/rank16. Use one32-channel local geometric encoder with two16-neighbor message steps and no global dent descriptor/pooling. Keeping the old kernel capacity avoids increasing it in reaction to WL57 held-out errors. This may fail learned fidelity; its actual F/S implementation and cost must nevertheless be measured separately. The local-state mechanism is the bounded persistent-graph hypothesis proposed in WL58, scoped here to the canonical volume interface.

## Rule/state ownership and update

The learned rule is fixed across held-out edits. Source identity, reference K, image targets and incident fields are absent from state construction. Runtime only receives current geometry/material and correspondence. Direct and inter-domain incident fields use the same K. K stops at first exit; all air transport belongs to unchanged C.

Persistent computed state contains encoder intermediate layers, source/destination affine features, raw local-edge coefficients and node-head/tail coefficients. Full F constructs all of them. Selective S mutates the actual resident arrays for the exact changed-feature dependency closure, preserving other entries bitwise. Changes to pair metrics are checked independently of node masks. Both use the same checkpoint.

Global column/tail normalization is recomputed and charged; it can affect every normalized coefficient. The inherited B quadrature has no path certificates, so S **conservatively invalidates/rebuilds all B for every nonrigid edit**. It does not pretend that local geometry implies local ballistic changes. Geometry preparation is also currently full and charged. The candidate tests whether savings in persistent learned computation survive these costs; a negative cost result is valid. Rigid material-frame identity reuses the complete intrinsic state, including B. Weight-version, physical-interface and graph/correspondence mismatches fail closed.

## Prospective experiment

Protocol: `experiments/fnto_persist59/protocol/persist_v1.json`. Runtime API: `full(model,g) -> cache`, `selective(model,resident_cache,g) -> cache,audit`, `apply(model,cache,incident_flux) -> outgoing_flux`. S updates its supplied cache in place. No post-edit optimization.

New primary held-out dent: depth5.75mm/radius22.5mm/centre(5,-5)mm, plus rest and a37°rotation about(1,2,3) with translation(30,-20,11)mm as rigid control. WL57cf0–cf3 are explicitly historical diagnostic geometry controls. Use all64held-out A2light fields, disjoint from96train/32validation. These lights and old diagnostic cases have been studied historically; new-neural-test results are not used to choose the candidate. New primary geometry is not in train/validation or old tested poses.

Train once for6000steps per seed5901/5902 with unchanged paired rest/current objective, AdamW/cosine/clipping/FP32 and final checkpoint. Existing training-only supervision is inherited with checksums and its known Monte Carlo limitations; no claim that those targets are newly independent or noise-free. Obtain new physical actions for the primary dent with2independent64M replicas, each2x32M halves. Rest references are immutable independently seeded A2bundles. Qualification uses reference noise, not candidate errors. No rank/capacity/resolution/training sweep after held-out results.

Internal tests execute actual F/S on real geometry, retained-state identity, dependency coverage, zero/linearity/passivity, checkpoint/correspondence rejection, rigid pullback and explicit pair-metric changes. Run WL53–56 regression suites with isolated outputs. These protect the primary comparison. Trained absolute/directional/dynamic fidelity is assessed after the fits; a failed F cannot be upgraded to physical success because S matches it.

## Evaluation and costs

Use the explicit historical action criteria: coarse mean relL2<=0.10,p90<=0.20,fine16mean<=0.15; detectable signed delta relL2<=0.35,corr>=0.90; qualified directional-pair error<=0.25. Report each RGB, field and geometry for F and S. F/S relativeL1<=1e-5; retained raw/head entries bit-identical. Learned positivity/linearity/passivity thresholds remain unchanged. Independent seed difference quantifies reference uncertainty; no false statistical CI claims for two oracle replicas.

Run fresh uncontended cost workers for F/S x2seeds xdent/rigid,7repetitions. Time actual geometry+B, raw state/update, global normalization, first and1/4/16/64current-field queries; report initial rest state separately. Count actual persistent unique GPU storages, total allocated/reserved peaks and CPU RSS. Preserve every execution attempt. Only a fidelity-qualified advantage whose upper descriptive bootstrap ratio bound is below1for the same low-query count in both seeds supports a useful speedup. Report raw timings even if quality fails.

Only eligible checkpoints proceed to the unchanged canonical WL56 coupled scene/C/solver/reference transfer, separately for F/S. Otherwise report physical-GI eligibility failure, without substituting an inferred complete K/C lifecycle claim. Preserve the historical blue floor. All artifacts and autonomous completion reports remain under `~/rest_persist_v1`; human-facing report under its `docs/` and the repository Worklog. No successor or production change follows automatically.
