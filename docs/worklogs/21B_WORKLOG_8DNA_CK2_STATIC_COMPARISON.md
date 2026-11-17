# Worklog — Native 8DNA on CK2 checkerboard interreflection (2026-11-16)

Status: bridge preflight; no comparison image or quality verdict yet.

The user requested one comparison on the ReST-GI checkerboard interreflection
dataset, in the layout of WL84 `references_and_candidates.png`. This is a
dependent static baseline qualification follow-up (21B), with no new ReST-GI
mechanism, dynamic edit, automatic successor, adoption or historical rewrite.
Protocol: `experiments/8dna_ck21b/protocol.json`, declared before new renders.

The workload is WL93D checkerboard addendum revision-2 CK2, variant bx38,
original camera/light/geometry and exact frozen ER states. Reference and
ReST-GI C0 s8601 images already exist on LabServer63 under
`/home/nd/rest_x93d_ck2/results/`. Their geometry and material contracts must
be bridged and measured before any 8DNA image is treated as comparable.

8DNA requires a newly trained asset for this scene. Its owned intrinsic
asset is the sharp S1 panel plus checker box A. The light, rest of the room,
sphere and box B must lie outside the asset AABB, which is sufficient for
the native convex-hull exclusion rule here. Direct scattering remains the
original analytic frozen ER; internal panel/box interreflection is learned
by the unchanged native distribution/albedo model. This is source-independent
per-asset baking, not scene-radiance training or reuse of Seal/Teaset weights.

All geometry/camera/light coordinates are normalized together and point-light
intensity is adjusted to preserve irradiance. The asset instance is identity:
the released query code does not invert a non-unit instance scale. Frozen
ER eval/sample/pdf are called unchanged from `closure83_dr.py` by a Mitsuba
BSDF adapter; no generic GGX/principled approximation replaces them.

The native fixed training recipe is 30 epochs, default network/loss/Adam/
learning rate/batch and 128^4 online path buffer. Path-generation chunking and
validation spp are reduced to fit 12GB; exact deviations are in the protocol.
A preview, if exported, must identify its epoch count and incomplete training.

The stored ReST-GI C0 image is reference LOW plus predicted higher-order
radiance and was trained for six scattering events. A neural asset internally
collapses multiple physical events, so counting native neural bounces is not
the same six-event limit. This discrepancy is explicit and requires a
physical depth diagnostic; the requested static image is not a dynamic
recovery-cost or broad method-superiority result. WL84 supplies the panel
layout and display/error convention only, not CK2 measurements or verdicts.

For uncontended execution, the running public-asset dispatcher PID 3579312
was verified by its exact command and temporarily held with SIGSTOP. Its
foreground Milk render was allowed to finish normally. The CK2 launcher
must restore that dispatcher on every exit, including failed preflight.
No unrelated GPU workload is stopped.

## Bridge implementation preflight (before any 8DNA fitting)

Two adapter defects were caught and corrected before training, with old code,
logs and preflight JSON preserved in server `attempt01/` and `attempt02/`.
First, a preloaded shapegroup not declared as a root scene object had a valid
bounding box but was not intersected by this Mitsuba setup. Using the same
explicit root group plus reference instance layout as upstream restored all
geometry. Second, DrJit 0.4.6 array `==` collapsed the checker parity test to
one Boolean; `dr.eq` and the SpatiallyVarying flag restore the per-hit texture.
No historical source, frozen material state, gate or scene was changed.

After those fixes: front, panel and direct box masks all match the saved CK2
512-square center rays exactly. Every directly visible box patch and material
id agrees exactly. Plugin eval/pdf/sample agrees with unchanged DrClosure83
at max absolute difference zero on 4096 independent directions per material.
The asset AABB is disjoint from all external scene geometry and the light;
the normalized asset's maximum radius is 0.990099.

A new native six-event, 512-spp ER render has whole-image signed bias
-0.00051245 relative to the stored four-by-8192-spp CK2 reference. Whole-image
relative L1 is 0.04450 and reflection-ROI relative L1 is 0.05342, including
the new low-spp render's Monte Carlo noise. This supports the bridge's bulk
radiometry; these figures are not 8DNA model errors. The paired 6/16-event
depth diagnostic and native-recorded-JIT compatibility probe precede fitting.
Native physical training uses upstream default recorded JIT flags, after
checking against the qualified wavefront runtime; neural inference switches
to the upstream notebook's unrecorded PyTorch-interoperability mode.

The paired 128-spp 6/16-event diagnostic estimates an additional positive tail
of 2.15% of full-frame D6 radiance and 3.02% in the reflection ROI. It is not
a converged infinite-depth reference. Error maps against the stored D6 target
therefore carry this disclosed regime difference; no fidelity winner follows.
The recorded/wavefront training-runtime check agreed at relative L1 9.79e-8.

Before any optimizer update, the first native data-buffer generation showed
about 0.88 seconds per chunk of paths from repeated Python BSDF construction.
That execution was terminated during chunk 241/512, with no fitted checkpoint;
its code/logs and successful preflight are kept in `attempt03_preoptimizer/`.
Immutable ER closures and identical material objects are now cached, and the
Lambert-only rows use their exact cosine/albedo algebra. This changes no
geometry, BRDF, training buffer size, optimizer/loss, checkpoint rule or budget.
The same preflight is rerun; uniform and checker eval/pdf/sample agree with
the unchanged DrClosure83 to max absolute 1.1921e-7. Original scientific and
historical results remain untouched.

One resumed public-job CPU startup briefly overlapped the beginning of the
new preflight before it created its GPU context. No preflight runtime is used
as a comparative timing claim. The public dispatcher is held again, the
public worker has exited, and only CK2 remains on the GPU before fitting.
The launcher now also waits for a pending public Python worker, rather than
checking only existing CUDA contexts, to close that scheduling race.

An execution-script replacement while an older shell was still reading that
file caused a shell parse error after the successful preflight. This was an
agent execution error, not a method/renderer result. The failing launcher log
is preserved in `attempt04_launcher/`. Recovery uses a separate immutable
`scripts/launch_stable21b.sh`, checks both pending public workers and live GPU
contexts, and reuses the passed preflight only when newer than the bridge
source. The first-epoch and final figure exports independently check actual
checkpoint global steps (8192 and 245760 respectively) to prevent an early
stop being mislabelled as a completed 30-epoch fit.

## Pre-optimizer memory execution fix

The optimized bridge generated the full first path buffer, then native CUDA
`randperm(128^4)` failed on its additional 4.07GiB sorting-workspace allocation
alongside the resident index array on the 12GB RTX 3080 Ti. The log reports
epoch 0, 0/8192 optimizer steps; no trained model or comparison was produced.
This attempt is preserved under `attempt05_cuda_randperm/` on the server.

The execution wrapper now generates the same-size *uniform global permutation*
using CPU `torch.randperm`, then copies it into the unchanged GPU index array
in 1M-index chunks. Permutation length, SHA256 and elapsed time are logged.
The data buffer remains 128^4, with the same 32768 training batch, model,
objective, optimizer, epoch count and fixed checkpoint rule. The CPU RNG
stream and permutation order differ from a hypothetical successful CUDA run;
the device change is disclosed and reproducible, not a bit-identical run.
No upstream source or previously frozen result is edited.

The CPU accommodation succeeded: all 268,435,456 indices were permuted in
35.93 seconds (SHA256 `ba7722b65bc60fffac7bc73975fb6e10ee71b6b0acc95c19af289dd392e22300`).
The native 669K-parameter model has passed 6000 real optimizer updates in
epoch 0 with finite logged loss. The first-epoch picture is generated only
after the 8192-step checkpoint guard passes; the final image requires 245760
updates. The public batch remains held during the fit and is resumed by the
launcher exit trap. A hidden Windows collector waits for completed, hashed
figure manifests, verifies checkpoint steps and PNG hashes, and writes only
this batch's files to `evaluation/21B/` and `eval_comparison/8DNA/`. It never
overwrites an existing comparison with different bytes.

## First learned comparison image — preview, not final qualification

The native checkpoint contains epoch 0 and global_step 8192: exactly one of
the 30 fixed epochs completed. Two independent 128-spp 8DNA renders of the
complete CK2 scene were produced with seeds 21201/21202, then combined with
the stored four-by-8192-spp REF-ER and two-by-4096-spp ECRO C0 s8601 results.
The C0 full image retains its original reference LOW plus learned H contract.

The WL84-style PNG has two rows (full frame and a fixed geometry-defined
reflection crop) and five columns (REF-ER, ReST-GI C0, 8DNA, and both absolute
HDR-error maps). All images use exposure 1, Reinhard x/(1+x), gamma 2.2.
Errors use sqrt(mean-RGB absolute linear difference) on one shared viridis
scale. The generated image was opened and visually checked: the intended
checker scene, captions, crop, error maps and colorbar are visible without
clipping. The current 8DNA reflection is noisy and substantially inaccurate;
this is explicitly marked **1/30 epochs, PREVIEW**, not a final-model result.

Descriptive relative L1 against the stored D6 reference: C0 frame 0.01991 /
reflection 0.07590; first-epoch 8DNA frame 0.20237 / reflection 0.69499.
These include MC noise and the earlier disclosed path-depth difference.
They do not establish a converged 8DNA limitation or a method winner.

Preview PNG SHA256:
`f3fb8ab932d6e537c01ebd1c535d879c3472917b9cecca36ff8e516ffaec8734`.
Checkpoint SHA256:
`f0936a6d186ea11fe0f406aead7b6ab1d75d258b1bea4060ddd7f6b146472a24`.

Local review image:
`eval_comparison/8DNA/ck2_references_and_candidates_epoch01_preview.png`;
an identical numbered copy and actual step/metric metadata are in
`evaluation/21B/`. The hidden collector initially lacked Get-FileHash in its
older PowerShell environment. Its error log was retained, the PNG/metadata
were fetched and verified directly, and the collector now uses .NET SHA256.
The corrected collector remains responsible for the eventual fixed-final
`eval_comparison/8DNA/ck2_references_and_candidates.png` (guard >=245760 steps).
That final file is not claimed to exist at this preview checkpoint. Training
continues on LabServer63; no later architecture or comparative verdict follows.

## Amendment 1 execution — takeover by a second session (2026-11-16)

**Why.** The session that started Amendment 1 ended after it paused the native job. The user then asked a different session (`rest-gi-ef`) to take over the work running on LabServer63.

**State found at takeover (07:20 UTC).**
- The native training PID 3610740 had been held with SIGSTOP at the epoch-4 boundary since 05:46 UTC.
- The anchor `resume_anchor/epoch=4-step=40960.ckpt` (sha256 `3c101c00…df56`) was in place.
- `benchmark_pipeline.py` and `packed_pipeline.py` were uploaded, but no benchmark had been run.
- No migration script existed.

**Benchmark (predeclared admission rule, run unchanged).**
- Launcher: `scripts/run_benchmark21b.sh`. Output: `results/pipeline_benchmark.json`, 07:21–07:23 UTC, with the native job paused.
- Correctness PASS: ordered fields, masks and dtypes are bit-equal. Paired optimizer steps on the anchor agree:
  - loss |Δ| = 0;
  - gradient relative L2 ≤ 4.3e-6;
  - parameter max |Δ| ≤ 6e-8.
- Complete-step timing (fixture):
  - baseline 0.0881 / 0.0749 s;
  - prefetch 0.0538 / 0.0627 s;
  - median speedup **1.40× → admitted** (rule ≥ 1.20×).
- The profiler tables are in `results/profile_{baseline,prefetch}.txt`.

**Migration** (as the amendment specifies; nothing in the native code, checkpoints or logs was edited):
- The original launcher (PID 3608014) and the paused job (PID 3610740) were stopped with SIGKILL. SIGKILL was used so that the launcher's EXIT trap would not release the held public-asset dispatcher during the resumed fit.
- A new launcher, `scripts/launch_packed21b.sh`, takes the same lock, uses the same environment and keeps the same hold. It runs `scripts/train_packed21b.py` (local copy: `experiments/8dna_ck21b/train_packed21b.py`).
- The run resumes from the verified anchor in a separate directory, `training_packed/CK2_bx38/`. The anchor is copied there as `last.ckpt`, and the resume uses upstream `train.py --resume`.
- The packed dataset and the prefetch loader replace only the host data path. The per-epoch uniform permutation stays on the CPU and its hash is logged.
- The data-generating random stream after the execution boundary restarts from a **fresh seed, 21105**. The native checkpoint holds no RNG state, and reusing 21101 would replay the epoch-0 data stream. The continuation is therefore not bit-identical.
- Everything else is unchanged:
  - the model and FP32 arithmetic;
  - the objectives, Adam, clip 5 and batch 32768;
  - 128^4 tuples per epoch and dataset_reload 1;
  - 30 epochs / 245,760 steps;
  - the scene, ER states, reference and figure guards.
- The first launch attempt exited silently at `flock -n`, because the killed job was still releasing the lock. The relaunch at 07:25:03 UTC succeeded.
- `Restoring states from …/training_packed/CK2_bx38/last.ckpt` was logged, epoch 5 started, and the first host permutation hash was `f6bbf062…11a3` (33.8 s).

**Observed live rate (epoch 5).**
- The first ≈ 340 steps were slow (≈ 3.4 it/s over 101 s, cause not isolated).
- After that, the steady state is ≈ 15.5 it/s (≈ 0.064 s/step). The native epoch 4 ran at 12.67 it/s, so the live gain is about **1.2× in steady state**, below the fixture's 1.40×.
- GPU utilisation stays around 30–40 %. The step therefore remains CPU / launch-bound and was not made GPU-bound by this change.
- ptrace_scope = 1 prevents attaching a sampling profiler to the running job. The cause of the start-up slowdown and of the remaining CPU bound is unattributed.
- These timings are execution observations, not method results.

**Status.** Training continues on LabServer63 at `training_packed/CK2_bx38/`. The final figure guard (≥ 245,760 steps) and the Windows collector rules are unchanged. The public-asset batch stays held until the resumed fit exits.

**Utilisation decision (user, 2026-11-16).**
- The profiler attribution of the remaining bound was reported to the user. The benchmark profile shows ≈ 1,500 kernel launches per step, ≈ 21 ms of GPU work in a ≈ 64 ms step, and a ≈ 34 % utilisation ceiling.
- Three options were presented: kernel fusion as an Amendment 2, running the held public batch concurrently, and a larger batch.
- The user chose to keep the current Amendment 1 execution unchanged. No Amendment 2 was made and no concurrent job was started.

## User-requested 12-epoch single-frame preview (2026-11-16)

The user requested one intermediate image from the completed 12-epoch fit.
`render_epoch12_preview.py` reads the immutable `epoch=11-step=98304.ckpt`
(SHA256 `49d1ad1ae2b4580c8f6c607fa2782e8d3ff9849da204fd8d7042aeaf28f33a21`)
without changing training state. This one-off preview ran concurrently with
the continuing fit; temporary contention is excluded from performance claims.
The first attempt failed before rendering because the standalone launcher
lacked the virtualenv PATH required by Ninja. Its error log was preserved;
the corrected launcher uses the training environment.

The 512-square image uses native neuralpath max_depth 10 / rr_depth 5 and
128 one-spp passes with seeds 21201 through 21328, matching the first
replicate of the earlier preview. Display remains exposure 1, Reinhard
`x/(1+x)`, gamma 2.2. Rendering took 140.7 seconds. Finite HDR values and
the transferred PNG hash were checked, and the actual local PNG was opened:
the checker box and green panel are present, with substantial visible noise.
This is a 12/30-epoch preview, not a convergence or final-quality result.

The single PNG and provenance JSON are in
`eval_comparison/8DNA/8dna_ck2_epoch12_preview.*`; identical copies plus
linear NPY / EXR are in `evaluation/21B/`. PNG SHA256:
`952e005a3c4f8c5c4ec86a4ace8c95520de6c75667a1ca4e3e20ef898049289c`.
The fixed 30-epoch selection and final collector remain unchanged.

## User-requested noise attribution on the frozen 12-epoch model

The user asked whether the strong visible grain follows from 8DNA's structure.
This is a read-only diagnostic extension of the intermediate preview, not a
new architecture, training intervention, or baseline verdict. The ongoing fit
continued throughout; concurrent diagnostic timings are not performance claims.

Primary source: [8DNA, arXiv 2604.25129v1](https://arxiv.org/html/2604.25129v1),
Sections 3.3, 4.2, 4.4 and Appendix B Algorithm 3. Section 4.4 explicitly
identifies missing MIS over incident location as a source of high variance
under small/point emitters. Section 4.2 reports similar or sometimes worse
variance than PT on surface-only assets; variance reduction is not universal.
Normalizing flows also struggle with high-frequency specular interreflection,
which is a separate representation-fidelity concern. The main qualitative
comparison uses 8192 spp, whereas the variance table uses 128 spp.

Implementation check: upstream `models/eight_dna.py:299` samples the boundary
location from the learned transport distribution before `sample_emitter_direction`
at line 324. Directional MIS exists in `models/integrator.py`; it does not
supply an emitter-aware alternative boundary-location proposal. CK2 uses a
single point emitter. The resulting vulnerability therefore matches the paper's
declared limitation. Even exact transport-proportional sampling would leave
variance from the current illumination/visibility product; more training does
not guarantee its removal. This explanation is supported by paper and code,
but the fraction of measured variance caused specifically by missing spatial
MIS has not been isolated by a component counterfactual.

`diagnose_epoch12_noise.py` fixes the same checkpoint at step 98304 (SHA256
`49d1ad1ae2b4580c8f6c607fa2782e8d3ff9849da204fd8d7042aeaf28f33a21`).
It renders the original scene/camera at 256-square resolution, native neuralpath
max_depth 10 / rr_depth 5, with four independent 128-spp blocks. Seed bases
221000, 222000, 223000, 224000 each advance by one per one-spp pass. For each
block, the first 32-spp average is also retained in memory. Unbiased sample
variance across the four independent blocks is averaged over RGB and pixels.
Masks conservatively require all four original 512-square child pixels to
belong to the region. No physical reference, denoiser, training change or
image-based checkpoint selection is used.

Measured mean linear-HDR variances, 32 spp -> 128 spp:

| Region | 32 spp | 128 spp | Ratio |
| --- | ---: | ---: | ---: |
| Frame | 4.33636 | 1.12385 | 3.8585 |
| Reflection ROI_T | 13.55935 | 3.65919 | 3.7056 |
| Panel | 7.48103 | 2.02242 | 3.6991 |
| target_direct mask | 0.05803 | 0.01708 | 3.3986 |

The approximately inverse-spp variance scaling verifies a substantial runtime
Monte Carlo noise component with the learned weights fixed. It does not prove
that all visible error is variance or isolate every stochastic term. Four
replicates provide an estimate, not a precision confidence bound. Increasing
128 -> 512 spp predicts half the RMS noise under independent finite-variance
sampling; the recorded 512-spp RMS values are explicitly estimates from the
128-spp variance, not measurements across independent 512-spp replicates.

The four blocks are averaged for an actual 512-spp image. The side-by-side
PNG is left 128 spp, right 512 spp with nested samples, identical exposure 1,
Reinhard and gamma 2.2. It was opened locally; grain is reduced but remains
substantial, especially on the panel. No convergence or physical-bias verdict
follows. Increasing spp reduces sampling noise, while learned blur/mean error
requires separate high-spp reference comparison. Adding incident-location MIS
would modify the baseline renderer and is not implemented here.

All block HDR arrays, comparison PNGs and metrics are under
`evaluation/21B/noise_epoch12/`; the review copy is
`eval_comparison/8DNA/ck2_epoch12_noise_128_vs_512.png`. The transferred comparison
hash matches the server: SHA256
`727b36c3f4f76c3e3c5ed6ca6e291c256eb9e480faab18aea21681a077e8a678`.
The diagnostic took 409.1 seconds. The original fit was still advancing in
epoch 13 (zero-based) after the diagnostic completed.

## User-requested current-checkpoint 4096-spp integrated comparison

At 09:41 UTC on 2026-11-16 the latest completed fit was 18/30 epochs,
`epoch=17-step=147456.ckpt`; epoch 18 (zero-based, the 19th epoch) was running.
The user's initial 8192-spp request was replaced by **4096 spp before any
8192-spp job was launched**. The 18-epoch checkpoint is copied into
`results/8dna_ck2_epoch18_4096spp/checkpoint.ckpt` so normal checkpoint pruning
cannot change or delete the selected asset during rendering. Its SHA256 is
`f5a7c4db6faed2cc31b419888db7e53f585bf1e5481323cdc363b20b59c56b86`.

`render_highspp.py` renders the same 512-square CK2 camera/material/light with
native neuralpath depth 10 / rr_depth 5, fixed exposure and no denoiser.
The requested total is one 4096-spp image, not two 4096-spp replicates.
Execution batching was probed without using probe samples in the output:
1 spp 1.228 s, 4 spp 2.459 s; 8 spp exceeded available GPU memory and was
rejected. The actual render uses 4 spp per call, seeds `23101 + batch_index`,
finite-value checks and atomic float64 accumulators every 128 spp. This
batching changes sample grouping only. The ongoing fit is not paused.
Initial production rate: 256 spp in 162.84 seconds, about 0.636 s/spp.

The user additionally requested an integrated reference / ReST-GI / 8DNA /
ReST-GI error / 8DNA error figure, including the fixed reflection crop.
`figure_highspp_matched.py` is queued after render completion. It uses the
stored C0 s8601 **r0 alone**, seed 86600, whose source metadata confirms
4096 spp; the earlier preview's two-replicate average is preserved in its
historical files. ReST-GI retains its inherited high-spp reference LOW plus
4096-spp predicted-H composition, explicitly labelled. Thus the candidates'
learned-render spp labels match, while the full pipelines and path-depth
contracts remain different. All candidates share exposure 1, Reinhard,
gamma 2.2; both HDR-error maps use one scale and colorbar. The 18/30 label
is explicit, and no final convergence or method-win verdict is implied.

Execution observed: renderer PID 3816811, server figure waiter PID 3821097,
hidden Windows collector PID 32508 (identities may become stale after exit).
The collector waits for `matched4096_complete.json`, verifies epoch / step /
spp / r0 identity and every downloaded file's SHA256, then places the single
PNG and integrated comparison under `eval_comparison/8DNA/`, with HDR arrays
and provenance under `evaluation/21B/8dna_ck2_epoch18_4096spp/`. Final image:
`ck2_references_and_candidates_epoch18_matched4096spp.png`.
The figure postprocessor and collector passed syntax checks. At this entry
the render was still in progress; no completed 4096-spp image is claimed.

## 4096-spp completion and validation OOM recovery (19:28 KST check)

The 18-epoch render completed all 4096 spp in 2525.4 seconds (2528.3 seconds
including its first figure export). The matched single-r0 integrated figure
completed on the server at approximately 19:25 KST; the Windows collector
downloaded and SHA256-verified all four files at 19:26:12 KST. The actual local
integrated PNG was opened and inspected: captions, the fixed reflection crop,
both shared-scale error maps and the colorbar are visible. Residual grain and
reflection-shape differences remain in the 8DNA panel image. Descriptive
relative L1 is ReST-GI 0.0199919 / 0.0759056 (frame / reflection), 8DNA
0.0664655 / 0.1989928. These are an intermediate fit with residual MC noise,
different path-depth regimes and ReST-GI's inherited reference LOW contract;
they do not establish a final converged method verdict.

Integrated PNG SHA256:
`57311f71754cb1391ea4b727b5b3bebbc4a3a4fd9fc2f0f62bd201d91da96c43`.
Single-image PNG SHA256:
`377343b0c512b478a94aca30c5f0294f9c8ec05b1148e67671eff56649cfce63`.
Files are present under `eval_comparison/8DNA/` and raw HDR / full provenance
under `evaluation/21B/8dna_ck2_epoch18_4096spp/`.

The concurrent fit **failed at 19:24:38 KST**, after finishing epoch 21
(zero-based) optimizer updates and entering validation. The traceback is a
CUDA OOM in native `render_simple -> model.sample -> normflow.vrqs`, attempting
another 26 MiB. The high-spp renderer was still running. Scheduling that
renderer alongside training without sufficient validation-peak headroom was
an execution-management error; the failure is preserved, not a numerical
method failure or a completed 30-epoch result. The launcher's EXIT trap
released the previously held public-asset dispatcher as designed.

Latest recoverable weights and Adam are epoch 18 / step 155648 (19 completed
epochs). Checkpoint internals and finite model tensors were verified. The
native last-v1 symlink still pointed to this saved epoch while the later
epochs were not retained as new top-k checkpoints. Thus 24576 completed but
unsaved optimizer updates from epochs 19--21 cannot be recovered. They count
as additional historical compute; recovery does not pretend they never ran.
The standalone initial `last.ckpt` is still the five-epoch migration anchor
and must not be blindly reused.

`recover_training19.py` reuses the admitted packed pipeline and exact native
model/loss/Adam/batch/tuple count/validation settings, copies the verified
19-epoch checkpoint into the distinct `training_recovery19/CK2_bx38/` folder,
and resumes toward the same global_step 245760 fixed final target. Fresh
continuation seed 21119 is disclosed because the checkpoint has no RNG state.
The sole checkpoint-retention correction is keeping every completed epoch
(`save_top_k=-1`), rather than relying on a last link to an older retained
best checkpoint. No held-out/image tuning or new learned mechanism is added.

Recovery launcher PID 3863454 holds only this project's public dispatcher and
waits for its already-dispatched Hair neural render to finish. Unrelated
`rest_x99` workloads are left untouched. No further diagnostic/high-spp render
will be scheduled concurrently by this recovery chain. The public dispatcher
is restored when recovery exits. At this record the recovery was queued;
actual restoration/start must be verified from `chain.status` and
`logs/train_recovery19.log`, not inferred from launcher creation alone.

## Recovery actually resumed after user priority request (19:46 KST)

At 19:43 KST recovery was still waiting for the public Hair r0 128-spp render,
which had been running for over 18 minutes without a saved 128-spp checkpoint.
The user explicitly asked to resume training. `preempt_public_hair.py` verified
the exact argv and process-tree identity of PID 3857935 and 32 compiler-worker
descendants before terminating only those 33 processes. Their identities and
the intentional preemption reason are recorded in `public_hair_preemption.json`.
The public dispatcher remains held. Its original later 4096-spp phase revisits
the same Hair/model/rep key and can regenerate the missing 128-spp checkpoint.
This is a deferred public evaluation, not a failed asset-quality result.
Unrelated `rest_x99` / `bench99b.py` processes were left untouched.

The existing recovery launcher advanced to `training_recovery19` at 19:44:42
KST. Actual training PID 3873830 logged `Restored all states` from the copied
19-epoch checkpoint, whose SHA256 is
`1f2d106f6d92dfb4b5232c7aa3aa35295a9c038c2079b8cc724c434b7767439c`.
The restored source includes one Adam optimizer state, finite learned tensors,
epoch 18 and global_step 155648. Native sanity validation and the next physical
data-buffer generation/permutation completed. At 19:46:23 KST epoch 19
(zero-based, the 20th epoch) had advanced through 210/8192 real training steps
with finite logged loss. Thus recovery is observed running, not merely queued.
GPU snapshot: 39% utilisation, 2788/12288 MiB across all active projects.
The resumed run retains the fixed 30-epoch target and durable per-epoch
checkpoint configuration described above.

## User-requested Git snapshot of 8DNA changes and outputs

The user explicitly requested that this agent commit its 8DNA modifications
and artifacts directly. This snapshot includes only WL21B's experiment code,
the actual server launchers, protocol/execution accommodations, completed
review images, linear HDR arrays, provenance and this chronological worklog.
The earlier public-release WL21A work is already committed separately. Other
ongoing batches, including WL93E and WL99, are outside this commit.

`evaluation/21B/manifest_ck21b_snapshot.json` inventories 43 artifact/provenance
files (26,057,568 bytes) with SHA256s and labels the 30-epoch recovery as ongoing.
Full checkpoints and large training caches remain at the recorded server
paths, with checkpoint hashes retained in the metadata. Collector logs and
temporary download files remain ignored. Exact-byte Git attributes preserve
the exported measurements' hashes across Windows/Linux checkouts.

Validation before staging: 15 local Python sources parsed; all server Python
and shell sources passed syntax checks; all three PowerShell scripts parsed;
artifact JSONs parsed; first/12/18-epoch PNG provenance and every epoch18
manifest hash checked; all seven local NPY image arrays have finite values
and expected dimensions. The packed-pipeline admission JSON confirms paired
correctness and the predeclared >=1.2x timing gate. Completed images were
visually reviewed in the earlier entries. These checks do not turn the
ongoing fit into a completed 30-epoch result.

## Fixed 30-epoch fit completed and automatic preview sampling correction

The check at 01:01 KST on 2026-11-17 verified that the fixed 30-epoch fit
completed at 21:43 KST on 2026-11-16. `Trainer.fit` reached max_epochs 30,
global_step 245760, with all recovery epochs retained. The final checkpoint
is `training_recovery19/CK2_bx38/final.ckpt`, SHA256
`d107c982ce9546774acf2c0d575c0f151c482a50af1f6f6a1b92e15c7fafd077`.
The chain exited successfully at 21:52:16 KST after producing its automatic
comparison figure. The Windows collector verified and downloaded it at
21:52:23 KST. The actual local figure was opened and checked.

The automatic figure used two 256-spp calls to the helper, with base seeds
21201 and 21202. Inspection exposed an execution defect in this helper:
`render(chunk=1)` increments the seed for every one-spp call. Consequently,
the two ranges are 21201--21456 and 21202--21457, sharing 255 seeds. There
are 257 unique seeds with duplicated weights, giving an effective independent
sample count of approximately 256.5 under the usual independent-seed model.
The earlier first-epoch two-by-128 preview likewise shared 127 seeds. This
corrects its earlier description as two independent replicates. Historical
images, metadata, hashes and measurements are preserved.

The automatic 30-epoch preview therefore retains substantial noise. Its
recorded relative L1 (8DNA frame 0.17568 / reflection 0.63524) describes that
image. Comparing these values with the 18-epoch 4096-spp figure does not
isolate training quality. The independent-seed diagnostic and the single
4096-spp rendering use distinct sampling loops and seed ranges. Their recorded
conditions remain valid.

`figure.py` now reserves non-overlapping ranges with bases 21201 and
21201+spp for future invocations. The original executed source is preserved
in `evaluation/21B/execution_evidence/figure_source_used_final.py`. The corrected
source passed syntax and seed-range checks. Existing renders retain their
original bytes and provenance. At this check the server's public Fabric
renderer occupied about 11.3/12 GiB of GPU memory. The completed final-model
preview is available locally, and a 4096-spp render of the 30-epoch model
requires a subsequent rendering run.

## Final-model 4096-spp tiled render and sequential ECRO follow-up (2026-11-17)

The owner requested the final 30-epoch CK2 image after the Fabric execution
work, followed by **ECRO on the same scene at 4096 spp**, and explicitly chose
the **WL100 near-mirror-trained model** when distinguishing it from the older
ECRO C0 s8601 already included in the historical comparison. This is a static
readout/export follow-up of WL21B, not another architecture, training run or
dynamic recovery comparison. No lifecycle/speed superiority is inferred.

The released native 8DNA integrator and trained parameters remain unchanged.
The WL21A tiled runner now accepts a scene bridge and explicit checkpoint,
with source/checkpoint hashes in its execution identity. The fixed anchor is
the final checkpoint SHA256 `d107c982ce9546774acf2c0d575c0f151c482a50af1f6f6a1b92e15c7fafd077`,
global step 245760. CK2 camera, frozen ER materials, light and 512-square
resolution are preserved. Tile interiors are 128 square with filter-radius
halo; each final pixel receives exactly 4096 samples. The seed schedule and
batch grouping are recorded, and changing the grouping uses a separate cache.

An initial chunk-4 attempt produced its complete 64-spp checkpoint and about
4 percent of the 4096-spp sample budget. It was explicitly terminated by this
agent at its coordinator (exit 143), preserving partial tile caches, because
small Python/Torch/Dr.Jit calls made the readout slow under contention. This
is an execution attempt, not an image/model-selection result. Its log and
status are preserved in `evaluation/21B/epoch30_4096_launch/`.

Bounded center-tile resource probes on this checkpoint measured steady calls
of 1.601 s at 16 spp (2364 MiB/process), and 2.158 s at 32 spp (4294 MiB).
64 spp failed with CUDA OOM while another project held about 7.1 GiB. Probe
samples are excluded from the delivered image. Chunk 8 measured 1364 MiB;
20 percent per-process memory headroom plus a 768 MiB reserve admitted **two**
workers from the 32 logical-CPU ceiling. Other projects were not stopped.
The machine has 16 physical cores / 32 logical CPUs; worker admission is
limited here by independent GPU contexts rather than CPU availability.

The new cache is `~/rest_8dna_ck21b/results/epoch30_4096spp_tiles_c8/`.
Its 64-spp full-frame assembly passed finite/shape checks and was opened:
checker geometry and reflection are visible, with substantial expected
low-spp noise. The **4096-spp run is ongoing**, launched in a detached server
process (launcher PID 4124051, current workers 4126666/4126667 at the check).
Do not interpret the preview as the final image. Final export is gated on
complete per-tile counts and the expected checkpoint, followed by the existing
reference / historical ECRO C0 / 8DNA / error-map figure. The original C0
figure retains its explicitly labelled reference-LOW + learned-H convention.

For the additional ECRO readout, `ecro_followup.json` freezes the selected
WL100 near final checkpoint (24000 steps, seed10001), SHA256
`b3550787e08c8a4d41c975eb485be459591d84cf14a7c010a36984d10ac62e5a`,
the original CK2 scene and the server's frozen renderer/model source hashes.
The model source is byte-identical after LF normalization to the WL100
implementation; CPU strict state loading passes with 735664 learned
parameters and an 8-by-69 material buffer. **Nothing is trained or refitted.**
The original CK2 S1 material features are supplied to the encoder, instead of
silently replacing the scene's panel with WL100's alpha .01 / F0 1 material.
This is material-distribution transfer of that checkpoint, not an in-training
material evaluation; both the figure and metadata disclose it. Learned
parameter equality is checked before and after replacing the feature buffer.

`queue_ecro_after_epoch30.sh` is detached on LabServer63 (PID 4129816 at
launch), currently `waiting_for_8dna_epoch30_4096`. It starts ECRO inference
only after the completed 8DNA 30-epoch/4096-spp manifest exists; the inference
script rechecks the final 8DNA identity and all frozen input hashes. ECRO
builds current geometry state and direct source, evaluates its own response,
then renders **its own physical e1 + e2 + learned hl** at 4096 spp, batch 8,
32 distinct 128-spp seed blocks. Reference data is read only during plotting.
There is no reference-LOW substitution in this additional ECRO result.

That second figure contains REF, 8DNA, WL100 ECRO and both shared-scale HDR
error maps, with full-frame and fixed reflection crops. The stored reference
is still four 8192-spp six-event ER renders; 8DNA's native collapsed-asset
transport is a different depth contract, so measurements are descriptive.
Output is planned under `evaluation/21B/ecro_ck2_wl100_near_4096spp/` and
`eval_comparison/ECRO/`; 8DNA exports remain under `eval_comparison/8DNA/`.
Independent hidden Windows collectors wait for each completion manifest and
verify every downloaded SHA256. Server computation continues without the
SSH session; local collection requires this Windows machine to remain awake.

At this checkpoint both **final images remain pending**. Python and shell
syntax, PowerShell parsing, the fixed ECRO checkpoint load, preview finiteness
and image appearance, 8DNA progress and ECRO waiting state were checked.
The complete ECRO inference path is deliberately not run before 8DNA finishes.
The public Fabric/training queue that the owner stopped remains stopped.
