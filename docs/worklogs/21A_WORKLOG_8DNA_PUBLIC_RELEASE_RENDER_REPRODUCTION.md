# Worklog — 8DNA public-release rendering and paper-claim qualification (2026-11-16)

Status: preflight running. No paper-reproduction verdict yet.

The user authorized rendering the downloaded public 8DNA assets on LabServer63
and checking whether the paper's claimed numerical quality is reproduced.
This is a dependent qualification of the WL19/WL21 released baseline, without a
new learned mechanism, architecture selection, deformation experiment, or
ReST-GI comparison. The suffix preserves historical worklogs and avoids the
unrelated in-progress audit97 files present before this session.

Protocol: `experiments/8dna_public_repro21a/protocol.json`, frozen before main
renders. Code and logs are separate from every historical evaluation.
Server folder: `/home/nd/rest_8dna_public21a/`.

## Public sources and intended measurements

- Paper: https://arxiv.org/html/2604.25129v1, Tables 1–2 and section 4.
- Official code: https://github.com/lwwu2/8dna26 at
  `4a2157ca24e506c5ac0831f27d656ecc50a64f07`.
- Official scene archive SHA256:
  `e60653896a978fb7a386c09c85f477cf98054082f41aa6745ea8d2538e0d1d19`.
- Official weights archive SHA256:
  `ef134a50bfade0431c97a71fd224dd833a56faeb79bf2ae14312f3a0be0bb13c`.

The actual server has all 11 public assets/checkpoints and the upstream pinned
commit. Only untracked downloaded archives appear in upstream Git status.
The paper tables contain 10 assets; Cloud is excluded from the primary table
comparison because it is not a table entry. The notebook renders Seal scene2
at 256 square and 256 spp. This is not the paper's 800-square evaluation.
The published table does not explicitly map scene function names or specify
MSE color space. Main evaluation uses released scene1, linear-HDR whole-image
MSE, and separately reports fixed demo-display MSE. These missing details limit
any exact published-number replication claim.

The primary queue preserves each scene's native physical reference, materials,
camera and lighting, pretrained parameters, and the notebook's neuralpath / 10
bounce / rr-depth 5 / wavefront flags. Accuracy, same-method sampling variance,
and synchronized warm wall time are distinct measurements. Independent seeds
and reference-noise estimates prevent noisy targets being mistaken for model
bias. Table comparisons will remain numeric/descriptive; no arbitrary new
paper acceptance tolerance is invented. Absolute times on RTX 3080 Ti cannot
be equated to the paper's RTX 3090 times. No training claim is tested.

## Preflight observations

- SSH to the requested server succeeds. GPU: RTX 3080 Ti 12GB, driver
  595.71.05, initially idle (1MiB, 0%, no compute processes).
- Reuses the existing isolated Python 3.10 runtime with official Torch 2.3.1 /
  Mitsuba 3.5.1 / DrJit 0.4.6 pins. Driver-matched existing OptiX libraries in
  `/home/nd/nvlibs/optix595` are supplied through the job environment.
- Upstream source/checkpoints are not edited. First model import is rebuilding
  the upstream CUDA extension for SM86. Initial 64-square/1-spp preflight uses
  the notebook's actual active `neuralpath`, not the incorrect historical
  WL19 wrapper's `neuralvolpath` selection.
- Local restricted subprocess startup returns 0xC0000142; authorized shell
  reads and SSH work through sandbox escalation. This is a local tool issue,
  not a server/8DNA result.
- Pre-existing untracked `docs/audits/` and `experiments/audit97/` are preserved.

Execution updates and final evidence will be appended here.

## Initial measured checkpoint — two assets, 800 square, 128 spp per replicate

Project base observed for the execution:
`05cc9da74559cb5b79a991574c0a8dab56c5d442`, branch `Selective_Recompute`.
All 47 extracted scene/archive entries and all 11 extracted checkpoints were
compared byte-for-byte with their downloaded public archives: zero missing or
different files. Upstream tracked Git diff is empty.

The initial 64-square smoke completed in 1.96 seconds with finite output.
The main queue runs in a detached, flock-protected single-GPU process. The
automatic CPU-side watcher regenerates `summary.json`, `analysis.status` and
`exports/` as checkpoints arrive. The scripts preserve raw HDR EXR, float64
accumulators, seeds, chunk timings, hashes and invocation/warmup times. No
runtime flag fallback has been required. Notebook `VCallRecord=false` works
on this GPU with the pinned software.

| Released scene1 | Neural 128 spp wall, 2-seed mean | PT 128 spp wall, 2-seed mean | PT/neural ratio | Paper Table 2 ratio |
|---|---:|---:|---:|---:|
| Seal | 28.630 s | 94.690 s | 3.307 | 3.709 |
| Teaset | 31.727 s | 15.817 s | 0.499 | 1.262 |

Times are synchronized warm `mi.render` wall times summed over 32 four-spp
chunks, excluding loading, warmup, per-chunk host-array copying/accumulation,
and file writing. Invocation totals are stored separately. Hardware and
software timing differences preclude an identical-RTX3090 reproduction claim.
Seal is faster with 8DNA in this execution; Teaset is about twice as fast with
PT. The latter is a measured difference from the paper's speed ratio on the
released scene, not a proved contradiction of its timing configuration.

The two independent 128-spp image differences also give the unbiased
same-method sampling-variance point estimate `MSE(X0,X1)/2`:

| scene | Measured neural 100×variance | Paper neural | Measured PT 100×variance | Paper PT |
|---|---:|---:|---:|---:|
| Seal | 0.06748 | 0.084 | 2.19200 | 2.561 |
| Teaset | 0.10862 | 0.107 | 0.10544 | 0.102 |

These are two-replicate point estimates, not confidence bounds. The Teaset
values closely match the paper's variance scale; Seal shows the same large
variance-reduction behavior (about 32.5× PT/neural here versus 30.5× in the
paper). This supports that the public models and renderer are behaving
plausibly on these two scenes. It is not a full ten-asset verdict.

### Accuracy is still unqualified

Using the means of the two 128-spp images per method, raw whole-image linear
100×MSE is 1.14463 for Seal and 0.14771 for Teaset. Their estimated Monte Carlo
noise floors are 1.12974 and 0.10703. In particular the noisy Seal target alone
overwhelms its published 0.057 reconstruction error: declaring failure from
the apparent 20× raw ratio would be invalid. Independent-residual cross-MSE
point estimates are 0.01748 and 0.04082, respectively, but two replicates
provide no reliable uncertainty interval. They are **not** final accuracy
verdicts. Primary high-spp targets remain unchanged and outstanding.

The source Figure 9 and both generated comparison panels were visually
inspected. The Seal tablecloth/wall/illumination composition and Teaset
checkered-table composition correspond to the released scene1 images shown
there. This establishes those Figure 9 correspondences. It does not resolve
the unpublished Table 1 aggregation or color-space details.

The measured Seal PT throughput projects about 13.4 GPU-hours for its total
65536-spp target alone (extrapolation, not completed timing). The whole queue
is therefore a long-running experiment. The protocol's per-invocation 8-hour
watchdog can leave especially expensive assets incomplete; such cases retain
their latest accumulator and failure status and cannot count as complete
paper qualification. Nothing has been labelled fully reproduced.

`precision_protocol.json` records a prospectively designed extra independent-
seed bias estimator, but its new seed stream has **not been executed**. It is
not an amendment or a replacement for the primary high-spp targets, and no
result relies on it.

Local review snapshots are under `results/evaluation/21A/` with provenance.
The server remains authoritative for ongoing raw outputs; snapshot filenames
explicitly retain the 128-spp stage. Continue by inspecting
`/home/nd/rest_8dna_public21a/{queue.status,current.status,analysis.status,summary.json,failures.log}`
and the per-job logs before any final interpretation.

## User-requested comparison-image relocation (2026-11-16)

The local review folder was found at `evaluation/21A/` after a project-level
relocation outside this step. At the user's explicit request, the generated
Seal and Teaset 128-spp comparison PNGs were moved to
`eval_comparison/8DNA/` in the project root. SHA256 values are unchanged.
`evaluation/21A/snapshot_manifest.json` now records their new paths, and
`eval_comparison/8DNA/manifest.json` links the images to their original run
records. Other evidence and the downloaded paper figure remain in
`evaluation/21A/`. Server outputs and the running evaluation are unchanged.

For the user-requested commit, the initial summary, public-file provenance and
the eight Seal/Teaset per-method/per-seed JSON run records were copied into
`eval_comparison/8DNA/initial_measurements/`. The comparison manifest hashes
these committed evidence copies as well as the two PNGs. The relocated
`evaluation/` tree's existing ignore rules remain unchanged. These are the
initial 128-spp snapshots, not a refresh or a final high-spp accuracy verdict.

## Public evaluation stopped by the user (2026-11-17)

At 01:19:18 KST the user requested stopping the Fabric render. The current
Fabric neural r0 128-spp process had been running for about 3 hours 24 minutes.
Consecutive GPU samples showed 0% compute utilisation, around 26 W power and
11312 MiB attributed to that process, while its main CPU thread consumed
approximately one core. Its log contained initial loading and a Dr.Jit
allocation-cache warning. The exact CPU hotspot was not established.

`stop_public_queue.py` verified the queue and Fabric argv, froze dispatch,
recorded process identities and terminated the verified 35-process queue /
renderer tree. The server records `stopped_by_user` for the queue and current
job. The follow-up check found no GPU compute processes, GPU memory at
1/12288 MiB, and no surviving dispatcher / timeout / Fabric main process.
The active public evaluation is stopped. Its interruption is a user stop.
The process record and original Fabric log are preserved under
`evaluation/21A/execution_evidence/`. The completed CK2 30-epoch fit and
previously collected images remain available at their recorded locations.

## User-authorized Fabric parallel execution follow-up (2026-11-17)

The user requested preventing the one-core Fabric execution and questioned
limiting parallelism to two workers on the 32-logical-CPU server. This
execution follow-up changes batching, process scheduling, observability and
recovery. It uses the same released Fabric scene1, weights and native CUDA
neuralpath integrator. The stopped public evaluation queue stays stopped.
Small probes and one bounded Fabric completion validate the implementation.

### Attribution and batching probe

Independent 32-square 1-spp probes completed in both virtual-call modes.
VCallRecord=false took 1.33 seconds in the render call after loading;
VCallRecord=true took 6.89 seconds. The notebook's false mode remains in use.
The 800-square, 4-spp full-frame probe exceeded its 75-second limit. Repeated
faulthandler snapshots locate the Python caller at upstream
`models/integrator.py:134`, `scene.ray_intersect`. This narrows the observed
large-wavefront stall to that native call. The internal C++ hotspot remains
unresolved. A 128-square crop of the identical 800-square camera completed
the 4-spp render call in 0.91 seconds, with peak PyTorch allocation 293710336
bytes. Batching by camera crop therefore provides a measured execution fix.

### Implemented execution path

`render.py --workers N --tile-size S` dispatches to `parallel_render.py`.
The legacy default invocation retains its existing entry path. Spawned
processes each hold one native CUDA scene/model and render independent tiles.
Crop sensors retain the full image dimensions/camera. Filter-radius overlap
is rendered and discarded around tile interiors before assembly. Explicit
per-tile seed ranges make scheduling independent of random streams.

Per-tile float64 sums and first-128 sums are written atomically every 32 spp
and at the requested target. Configuration identity, dimensions, counts and
finite values are checked on resume. The coordinator reports progress every
five seconds with worker PIDs, CPU use, pixel-sample progress and ETA. Workers
that make no progress within the chosen timeout are terminated with their
process groups. The next invocation can use committed tile caches. Parent
death also terminates the GPU workers. Torch compiler worker proliferation is
limited through TORCHINDUCTOR_COMPILE_THREADS=1 per render worker.

`--workers auto` requests the number of CPUs in the current affinity mask,
32 here (16 physical cores with SMT). It calibrates a representative center
tile and measures process GPU memory. The implementation chooses a 20%
per-worker memory margin and a shared 512-MiB reserve. These are scheduler
headroom choices. During the full-frame validation, calibration measured
2434 MiB per worker and 7984 MiB free, selecting two simultaneous workers.
Thus this implementation does not demonstrate 32 simultaneous GPU-resident
workers. Other project's GPU allocation varied from about 3.6 to 6.2 GiB
during the probes and was left running.

The one-command entry is
`bash ~/rest_8dna_public21a/scripts/launch_fabric_parallel.sh [spp] [workers]`.
Its defaults are 800-square, 128 spp, 128-square tiles and automatic worker
selection. It writes its own execution outputs under `results/fabric_parallel_v1/`.
The original public queue was not resumed by this change.

### Measured validation

On the same 256-square, 32-spp, 64-square-tile workload, one process took
30.069 seconds and two took 20.621 seconds, a 1.458x end-to-end speedup.
Recorded average CPU use was 1.088 versus 2.110 logical cores. Serial and
parallel images are bit-identical. A 16-to-32-spp continuation that changes
from one to two processes is also bit-identical to the uninterrupted result.
Camera-ray differences between full/cropped sensors are <=1.20e-7.
A constant-illumination test has zero seam difference after tile assembly.

The four-worker expansion probe encountered slow initialization in two
workers while the other two committed all 16 tiles. The first coordinator
then treated delayed worker shutdown as failure. The revised coordinator
releases unused workers once all tiles are committed. The completed tile
cache was recovered into a bit-identical image in 0.512 seconds. This cached
recovery time is not a four-worker rendering speed measurement.

A deliberate 0.5-second no-progress timeout produced the expected failed
status and ended the test workers. GPU process checks afterwards showed only
the unrelated project workload. All local HDR arrays are finite. Source
syntax and the legacy/parallel timing interpretation were checked. The
analysis helper now recognizes tile metadata and avoids summing concurrent
tile durations as a full-frame rendering time. Unavailable or differently
scoped timing measurements are labelled in its report.

The actual 800-square Fabric frame completed all 128 spp. Its initial 4-spp
stage took 16.603 seconds in workers; extending 4 to 128 took 208.525 seconds
in workers, 214.973 seconds including that invocation's calibration. This
is a resumed render, so 208.525 seconds alone excludes the initial 4-spp
stage. GPU use reached 90% in the live check and 73% in the saved CSV sample.
The image was opened locally; fabric geometry/shading and the assembled frame
are visible, with residual Monte Carlo grain. This execution qualification
does not add a reconstruction-quality verdict against a converged reference.

The complete logs, stacks, configuration, numerical checks, native HDRs and
review PNG are under `evaluation/21A/fabric_parallel/`, inventoried in its
SHA256 manifest. The selected measurements are summarized in
`execution_summary.json`. The old stopped-run evidence is preserved in the
previous section and its execution_evidence directory.
