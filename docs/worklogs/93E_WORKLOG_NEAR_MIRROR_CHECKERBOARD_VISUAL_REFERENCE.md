# Worklog — Near-Mirror Checkerboard Visual Reference (2026-11-16)

## Authorisation and classification

The user's 2026-11-16 near-mirror visual-benchmark brief authorises a separate scene/reference and minimal instrumentation, with no architecture development, training, tuning, branch/worktree, or successor experiment. This is a **WL93D-dependent evidence-gap follow-up**, hence suffix **WL93E** under the repository's scientific-independence numbering policy. It does not consume or renumber the unrelated ongoing WL98 design work.

Preserved: both WL93D Worklogs, the revision-2 addendum, scene_ck2.json, all frozen checkpoints/tables, renderer source, old references, and canonical Architecture.md. New source: `experiments/ecro_nearmirror93e/`. Reviewer output: `evaluation/93E/`. Raw execution: LabServer63 `~/rest_nm93e/results/`, exported evidence under `results/ecro_nearmirror93e/`.

## Implementation Fact

The geometry, lights, camera origin/target/FOV, checker placement and 3×3 patch checker on each box-A face are **exactly CK2 revision 2**. Only camera resolution increases to 1024×1024. The original framing already shows the reflection at large scale; no scene was chosen using model error. A five-condition, 512×512/128-spp **physical-only** preview was visually inspected before the main rendering. Alpha .01 visibly resolves the cells, and the delta mirror gives hard cell boundaries. No model output was inspected for framing selection.

The material lookup is the original addendum's run-time per-patch PMAT lookup. The preserved `pt86.py` and `closure83_dr.py` files are never edited. The new process overrides only the panel state; the other material rows remain bit-identical. An isolated run-time copy handles true delta reflection without inserting alpha=0 into a finite GGX evaluation. The output records original and applied state-table hashes, trace-patch hashes, scene/config hashes, seeds, settings and timings.

### Material and roughness definition

| Condition | Definition | Comparison scope |
|---|---|---|
| S1 | Original frozen ER 69-float compiled state: 7 anisotropic GGX lobes plus attenuated Lambert/inner terms, effective refraction for the 4 inner lobes | Historical REF-ER control; **not** a single-GGX BSDF or original layered P ground truth |
| G075 | Single isotropic outer GGX, alpha_u=alpha_v=.075; F0=1, neutral coefficient .85 | Roughness-only group |
| G040 | Identical, alpha=.04 | Roughness-only group |
| G010 | Identical, alpha=.01 | Roughness-only group |
| MIRROR | Exact delta reflection, neutral reflectance .85 | Qualified **e2** edge control; full/h have the scope restriction below |

Alpha is the **direct GGX NDF slope parameter**, used as au/av in the denominator and sampling. No perceptual remapping/squaring is applied. Under a UI convention alpha=r², the equivalent r values would be .273861 / .2 / .1; these are explanatory conversions, not renderer inputs. F0=1 fixes Fresnel to unity and .85 fixes the reflection coefficient; neither is a roughness parameter. The GGX implementation uses height-correlated Smith masking and VNDF sampling. This is an explicitly defined passive idealised reflector, not an asserted identification with a specific real conductor.

Original S1 au = [.0440111, .0746922, .0283659, .0298800, .0308658, .0261259, .0406239]; av = [.0433040, .0737488, .0279330, .0266314, .0305056, .0256718, .0423344]. Effective eta=2.4955008. The complete state, spectral weights, F0, sampling probabilities, attenuation knots and original layered descriptor are in `preflight.json`. Its broadening must not be attributed solely to a single roughness number: its lobe composition, refraction and spectral weights differ from the neutral GGX group.

### Render settings and reference scope

Main references: 1024×1024; **2 independent replicas × 2048 spp = 4096 spp** for each material and bx38/uniform variant; seeds 931500/931501, common random numbers across variants/materials; 6 scattering events; batch 8 spp; uniform subpixel jitter/box pixel integration; no denoising, clamping, roulette or per-image normalisation. Preview: seed 931490, 128 spp, 512×512, batch 2.

Each bx38 reference has a uniform-.5 counterpart; the checker is the existing Lambert .3/.8 patch map. The final exposure is fixed from the untextured S1 reference median, and the Reinhard-like map `(key*x/(1+key*x))**(1/2.2)` is common to every material, model and component. All quantitative measurements use linear HDR arrays, not tone-mapped images.

**Path ownership:** e1 is next-event contribution at camera vertex 1; e2 is exactly 2 scattering events; h is 3–6. On primary panel pixels, reflection contribution is e2+h. A reflected checker requires the camera ray to hit the panel and the next path vertex to see a textured surface; the directly visible box has a separate geometry mask.

**Ideal-mirror scope restriction, identified during code review:** the inherited unidirectional point-light NEE estimator cannot sample a delta reflection at the **light-side** end of a path (the light→mirror→receiver caustic connector). Therefore MIRROR's full/reflection/h images are **restricted camera-side diagnostics**, not complete physical ground truth. Its primary-panel **e2** term (camera→mirror→Lambert checker→point light) is valid and independently checked analytically. Only e2 is used for ideal edge/contrast comparisons. No complete ideal-mirror higher-order claim is made. This narrows the evidence rather than silently changing the frozen renderer or adding a new transport algorithm. Finite-alpha GGX lobes do not exclude that delta path family; remaining Monte Carlo noise is measured separately.

### Frozen model evaluation scope

Frozen ECRO has six fixed trained material states; it is **not** a generally trained compiler accepting arbitrary new GGX/delta states. No near-mirror/new-GGX fidelity comparison is run. Unsupported material does not stop physical-reference generation.

Only original-S1 C0 readout is rendered, for own seeds 8601/8602, using the original CK2 bx38/none tables unchanged. Camera, geometry, lighting, PMAT, original S1 and the renderer match the new S1 reference. This separates the existing model's higher-order error from the existing reference softness. **The checker distribution itself remains post-hoc/OOD** (.3/.8 outside the training Lambert range and spatial redistribution): this is a matched-material diagnostic, not an IID fidelity qualification. ECRO full images are physical e1+e2 plus its learned hl, explicitly labelled. M64 is never used as physical ground truth.

## Verification before primary results

- Five implementation preflight checks pass: inherited untextured renderer regression (relative maximum **0** in this run), independent NumPy mirror/OBB checker mapping (**0 mismatches / 55,897 safely interior pixels**, 512 resolution), sufficient reflection ROI (**58,035 pixels** at 512), GGX peak changes, and sampled-lobe narrowing.
- Actual peak evaluations at 40° incidence: 15.6666 / 55.1557 / 882.9576 for alpha .075/.04/.01. Median sampled angular deviation: 7.5472° / 4.0485° / 1.01375°. Thus the parameter is applied in both evaluation and sampling.
- Independent NumPy analytic GGX evaluation on 100,000 random direction pairs agrees with runtime at relative RMS **1.08e-6 / 1.52e-6 / 4.66e-6**. Reciprocity maximum relative error <8.4e-8; sample weights nonnegative and at most .8500003. The validator initially indexed a Mitsuba colour array in the wrong orientation; correcting that analysis-only indexing changed no renderer/config/seed/output.
- Noise and analytic-delta limits are specified prospectively in `protocol.json`; no threshold changes are allowed after primary measurements.

## Reproducibility checkpoint

Working branch: `Selective_Recompute`. Starting local commit: `3cf69733d91e0defa4a4648bb64565e45dcd9c47`. The isolated server dependency bundle is the preserved revision-2 code snapshot `d8d092fd9634921190e7fcc438cb57f2adb79e36`; the used frozen files are compared by content hashes with the current local tree. New code is identified by per-file SHA-256, not attributed to the historical commit. The pre-render freeze is `~/rest_nm93e/freeze.txt`; the new source/config manifest is exported with the results.

The sandbox shell processes initially failed with Windows 0xC0000142; explicitly approved process execution outside the sandbox worked. No project failure is inferred from that tooling failure. Local Python lacks Pillow, so image analysis/rendering runs in the existing server environment. Original checkpoint hashes were recorded before execution and checked again at export. Other user jobs/files are preserved. Reference work overlaps an unrelated GPU job and independent rendering processes; wall times are recorded for reproducibility **without an uncontended cost/speedup claim**.

Commands: `. ~/rest_nm93e/env.sh`; `$PY code/experiments/ecro_nearmirror93e/render93e.py pre ~/rest_nm93e`; `bash code/experiments/ecro_nearmirror93e/run93e.sh ~/rest_nm93e full`; optional `run_high93e.sh` overlaps G010/MIRROR jobs with identical seeds/settings before the primary chain reaches them; completed JSON-marked frames are reused. `validate93e.py` performs the independent analytic tests; `viz93e.py ROOT OUTPUT` generates figures and measurements. All long jobs run detached and leave status/completion markers.

## Execution and final evidence (append below)

The sections above describe the frozen settings, preview inspection, preflight and reference/model limits before final measurements. Final measured results, visual review, interpretation and unresolved questions are appended after the main run; they do not rewrite WL93D or change architecture contracts.

## Final execution record

The main chain completed successfully at **2026-11-16 05:14:33 UTC / 14:14:33 KST**. The high-material worker completed at 05:07:12 UTC. There are **20 main physical-reference frames + 8 original-S1 ECRO readout frames**, in addition to five preview frames. Each physical material/texture condition averages two 2048-spp replicas; ECRO hl averages two 1024-spp readouts per checkpoint/texture condition. The original checkpoint and table hashes are unchanged at export. All five used frozen dependency files match the current local files by SHA-256; local original Worklog/Architecture preservation checks also pass.

**Execution-only analysis issues, no protocol/render changes:** local JSON export initially rejected a NumPy bool; casting that one qualification flag to a Python bool fixed serialisation, without changing numbers. The final captions were expanded to identify the ideal-mirror missing path family and measured higher-order noise. The final analysis code hash therefore differs from the pre-render visualisation hash; both are recorded. Render code, settings, seeds and primary raw outputs stayed fixed.

Automatic approval review **rejected a later upload of additional analysis scripts to LabServer63**, citing private code egress to a destination without sufficiently explicit authorisation. No further code was sent. The safe alternative completed the task: the already generated raw renders/logs were downloaded, and all final analysis, rendering of sheets and provenance export ran locally in an isolated workspace Python venv. No user approval is pending and no requested artifact remains blocked. The complete raw export is local under `results/ecro_nearmirror93e/results/`; the initial archive is also retained.

## Physical Reference Observation

**YES, a reliable near-mirror checker reflection was obtained, for the exactly-two-scattering-event panel reflection.** At direct GGX alpha=.01 the cell interiors and narrow boundaries are visibly resolved in the actual rendered image. The delta control is sharper still. At unchanged CK2 placement, the original compiled S1 reference itself is visibly softer. The independent ideal-mirror/OBB/point-light control agrees with the delta e2 render at **0.03354% relative RMS** over 193,549 safely interior pixels.

The qualified scope is **e2** (camera→panel→checker→point-light connection), with LOW=e1+e2 included for full-frame context. It is not complete multi-bounce GI qualification. The four prospectively specified checks all pass: reflection ROI size, near-mirror e2 noise, edge-profile noise, and independent delta e2 agreement.

## Quantitative Measurement

All edge/contrast measurements use **linear-HDR e2**, divided by the same material's uniform-.5 counterpart, on identical centre-ray-corresponded reflection pixels. The ideal ratio is .6 on dark cells and 1.6 on light cells. Contrast is Michelson contrast of the interior dark/light median ratios; retention divides it by the ideal-mirror value. Slope/correlation are OLS-with-intercept texture-ratio response relative to the rendered delta control. Width is the .7→1.5 transition of a shared geometry-aligned signed-distance profile, i.e. 10–90% of the **ideal** .6→1.6 step, in native 1024-resolution pixels. Width is scene/camera dependent, not a material-independent angular width.

| Material | Dark / light ratios | Contrast retention | Edge width (px) | Response slope / correlation | e2 RMS SEM |
|---|---|---|---|---|---|
| Original S1 | .7700 / 1.3913 | **63.24%** | **106.40** | .5202 / .8500 | .985% |
| GGX alpha .075 | .7456 / 1.4208 | **68.57%** | **88.05** | .5590 / .8633 | .902% |
| GGX alpha .04 | .6622 / 1.5222 | **86.62%** | **43.54** | .7147 / .9239 | .712% |
| GGX alpha .01 | .6048 / 1.5938 | **98.97%** | **10.44** | .9293 / .9892 | **.387%** |
| Ideal mirror | .6000 / 1.6000 | 100% | **1.28** | 1 / 1 | .0707% |

ROI: **232,175 pixels**; reflection crop **[x0,y0,x1,y1)=[0,195,421,811)**. Directly visible box: 350,702 pixels, independently masked. Interior contrast/response mask: 198,898 pixels. Edge profile uses the largest reflected box face (face 0), 93,766 eligible pixels and signed-distance bins −60…+60 px. Near-mirror maximum paired profile SEM is **0.0004571 ratio units** (ideal step amplitude=1), far below its boundary signal. Pixel centre geometry defines the masks; subpixel box filtering broadens even the ideal edge to about 1.28 px. No ROI/cell-boundary matching was selected by model error.

Noise is spatial RMS of `(replica0−replica1)/2`, normalised by RMS of their mean: an estimate of SEM for the two-replica average, not a pixelwise confidence interval or proof against all rare paths. Independent seeds and common random numbers for texture controls are retained.

**Full/h noise limit:** h ROI relative RMS SEM is **12.7 / 19.9 / 29.4 / 48.7%** for S1/G075/G040/G010. Corresponding full-radiance values are 4.09 / 6.97 / 9.24 / 14.13%. Rare bright paths dominate this statistic. These outputs are **NOT QUALIFIED as converged full/h references**. They remain labelled diagnostic images and are not used to infer sharpness or representation limits. No denoising, sample removal, clamping, acceptance-rule change or unapproved compute rescue was used. MIRROR additionally has the stated light-side delta path-support exclusion, irrespective of its lower measured noise.

## Model Evaluation Scope and Qualitative Comparison

The original-S1 diagnostic sheet uses the identical new S1 reference LOW for both ECRO full images and adds only each checkpoint's original C0 hl. It visually separates physical S1 softness (already present in reference LOW) from the different model higher-order response. Descriptively, the higher-order texture-response slopes/correlations are **.2373/.7479** (s8601) and **.5070/.9236** (s8602); higher-order RMS errors are .3235/.2237 on the ROI. These are **OOD, noise-limited descriptive numbers**, not a new architecture or fidelity gate. They do not attribute representation versus optimisation versus global-term errors.

No ECRO alpha=.01 prediction exists in this batch. The new GGX/delta material states are outside the frozen material contract; feeding them through the network would not establish fair near-mirror fidelity. Existing S1 support does not make the .3/.8 redistributed checker an IID training condition. WL93D's historical higher-spp evidence and frozen labels are preserved, not replaced by this diagnostic.

Actual final visual inspection covered the full/ROI sheet, the qualified LOW/e2 sheet, separated path components, ideal edge/ratio profiles, and both-checkpoint S1 comparison. The primary panel reflection is cyan-outlined and spatially separate from the directly viewed yellow-outlined box in the correspondence sheet. The alpha=.01 panel reflection contains distinct checker cells; alpha=.04 is intermediate; original S1 is much softer. These observations agree with the linear-HDR edge profiles rather than being inferred from numbers alone.

## Interpretation

The original WL93D reference blur cannot by itself be called an ECRO near-mirror representation failure: this batch reproduces softness in the **physical REF-ER S1 e2 term before any model readout**, and obtains sharp reflection by replacing that material with a separately defined low-alpha GGX. Within the neutral GGX group, scene, light, camera, reflectance, Fresnel and sampling convention are controlled, so the observed e2 edge narrowing is attributable to roughness. The S1-versus-neutral comparison also changes the composite material definition and is not a single-parameter causal attribution.

The distinction sought by the brief is therefore measurable: **reference-lobe blur is measured by the physical e2 roughness/delta series; model higher-order discrepancy is a separate original-S1 diagnostic**. This does not establish the frozen model's ability or inability to reproduce the new near-mirror material, nor does it validate a proposed replacement representation.

## Unresolved Question

- Complete full-radiance/h convergence for finite-alpha materials, especially the sharp material's rare high-energy paths, is not established by the fixed two-replica budget. The e2 conclusion does not depend on that missing convergence.
- A complete ideal-mirror higher-order reference needs a valid light-side delta/caustic connector or another appropriate estimator; this batch intentionally reports that missing family rather than modifying the transport algorithm.
- Frozen ECRO generalisation/training for new near-mirror material states remains unmeasured. No retraining or architecture extension is authorised or performed.
- The spectral/composite reasons for original S1 softness beyond its documented 69-float BSDF are not isolated here; the original layered P material is not rerendered in this follow-up.

## Architecture Verdict

**NOT APPLICABLE.** No new architecture, training, tuning, production adoption, global fidelity verdict, dynamic-reuse claim or downstream experiment. No living research contract or prior Worklog was changed. The visual-reference request is complete **within the explicitly qualified e2 scope**; full/h limitations are part of the result.

## Reproducibility and reviewer artifacts

Final analysis: Windows Python 3.11 workspace venv; NumPy 1.26.4, SciPy 1.14.1, Matplotlib 3.10.9, Pillow 12.3.0. Physical renderer: LabServer63 NVIDIA GeForce RTX 3080 Ti (12 GiB), driver 595.71.05; exact Python/Mitsuba/Dr.Jit versions and variant are in each render JSON and `execution_manifest.json`. Full physical render settings and seeds are unchanged from the pre-render protocol. Absolute wall times are recorded, not claimed as uncontended performance.

Reviewer folder: [evaluation/93E](../../evaluation/93E/). Standalone entry: [index.html](../../evaluation/93E/index.html).

| Artifact | Purpose |
|---|---|
| [figNM5_qualified_low_and_e2.png](../../evaluation/93E/figNM5_qualified_low_and_e2.png) | **Primary qualified visual**, 4500×1980: common LOW full frames and e2 reflection crops |
| [figNM0_full_and_roi.png](../../evaluation/93E/figNM0_full_and_roi.png) | Full-radiance and ROI diagnostic comparison, with convergence/delta limits in the caption |
| [figNM2_edges_and_contrast.png](../../evaluation/93E/figNM2_edges_and_contrast.png) | Ideal-mirror ratio maps, native-pixel edge profiles, paired profile uncertainty and contrast/width table |
| [figNM1_path_contributions.png](../../evaluation/93E/figNM1_path_contributions.png) | Reflection=e2+h / exactly e2 / higher-order h, same exposure |
| [figNM3_correspondence.png](../../evaluation/93E/figNM3_correspondence.png) | Direct box versus reflected checker correspondence |
| [figNM4_frozen_ecro_S1_only.png](../../evaluation/93E/figNM4_frozen_ecro_S1_only.png) | Original-S1 matched-material, checker-OOD diagnostic only |
| `*_full_1024.png`, `*_qualified_e2_roi.png`, `*_reflection_roi.png`, `*_higher_order_roi.png` | Native standalone image assets; scope as above |
| `metrics_nearmirror93e.json`, `edge_profiles.npz` | Machine-readable measurements and exact plotted curves |
| `protocol.json`, `scene.json`, `preflight.json`, `analytic_material_checks.json` | Configuration and implementation/analytic controls |
| `execution_manifest.json`, `manifest_nearmirror93e.json` | Commit/content provenance, immutable input checks, raw-file and reviewer-artifact SHA-256 |

Local regeneration of final images: `results/ecro_nearmirror93e/analysis_venv/Scripts/python.exe experiments/ecro_nearmirror93e/viz93e.py results/ecro_nearmirror93e evaluation/93E`, then `export93e.py` with the same two arguments. Downloaded raw render arrays and JSONs are complete under `results/ecro_nearmirror93e/results/renders/`; no remote script upload is needed for analysis. Analysis dependency versions are recorded alongside those results. The new source is uncommitted and is identified by hashes; the historical base commit is not misrepresented as containing it.

## Commit note (appended 2026-11-16)

- At the user's request, another session committed this worklog, `experiments/ecro_nearmirror93e/` and `evaluation/93E/` exactly as they were on disk. Nothing was edited apart from this note.
- The raw outputs under `results/ecro_nearmirror93e/` (about 3 GB, including the analysis venv) remain untracked, following the gitignored `results/` convention.
