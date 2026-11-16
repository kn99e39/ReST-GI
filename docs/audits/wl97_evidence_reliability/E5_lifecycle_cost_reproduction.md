# WL97 audit, track E5: independent reproduction of the lifecycle-cost half of P0-4

Date: 2026-11-16. Auditor: independent agent (track E5). Scope: WL96A Stage 3 (lifecycle cost on CK2) and the WL96 cross-workload
ceiling arithmetic. Nothing was trained, tuned, modified, moved, deleted or committed. No other track's report was opened.

This track was run in two sittings. A first attempt (same instructions) wrote the harness and completed timing runs 1-6 and the
generator sub-step run, then was interrupted before reporting. The second sitting (this report) treated those outputs as unverified
working notes: it re-read the harness, recomputed every quoted ratio from the run files with separate code (`supp_e5.py`), verified
that local copies are byte-identical to the server outputs (sha256), found one contaminated run (run2) and repeated it on an idle
machine (run7), and wrote the report.

## 1. Claim under audit

P0-4 (cost half): "The learned broad higher-order resolvent has no meaningful ... lifecycle-cost headroom over explicit recurrence
on the tested CK2 workload."

Original evidence: `docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md` section B4 (lines 122-162), conclusions at
lines 188 and 205; code `experiments/ecro_util96a/w96a.py` `cost()` (lines 521-685); data `results/ecro_util96a/cost_s1.0.json`
(sha256 103b775ed571); protocol `experiments/ecro_util96a/protocol/ecro_util96a.json` (COST criterion: U(LAM_rho) >= 1.25 on wall
for edit -> frame or source -> frame, frame 512^2 x 64 spp; scaling extension only if U(ZERO) >= 1.15). Also WL96
(`docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md` lines 144-149) ceilings 1.09x / 1.16x from
`results/gm1_c3urx1_91/results/econ/life91.json` (sha256 2f3fc40b5afb).

## 2. Method

Own harness `experiments/audit97/E5/e5_time.py` (sha256 9675abd00f06). It does not import `w96a.py` or `e2e91.py`. It calls the frozen
production components being timed from the read-only checkout `~/rest_x96a/code` (commit a1718f7): `mat94.ctx_ck2`, `pt86.Ctx.set_geom`
/ `render` / `state_rays`, `ecro86.build_state` / `update_state` / `dirty_rays`, `quad87.Quad` / `qtable`, `xport87.XOp`,
`model86.GeoState` / `ECRO.factors` / `apply` through `mat94.load_frozen(R92A, 8601)`.

- Every wall is bracketed by `dr.sync_thread()` + `torch.cuda.synchronize()`.
- Compilation is excluded through the Dr.Jit kernel history (backend time of cache misses) and reported: 743-921 ms in the untimed
  warm events of each process; 0 cache misses and 0 ms compilation in every timed event and every component repetition.
- Events: per arm 2 untimed warm edits + 2 warm sources, then 12 edits (8 in scaling runs) and 16 source updates; one total timer per
  event independent of the part timers (parts sum / total 0.982-1.016, typically 0.994). Event blocks: A = original order
  (R0, LAM24, LAM64, ZERO sequential), B = R0 / ZERO interleaved (ABBA), C = interleaved without frame, D = R0 after the LAM arms,
  E / F / G = LAM24 with a numpy integer cast / one torch thread / default again.
- Components in isolation: 20 repetitions after 1 warm call, first 2 discarded -> n = 18 per process (36 over the two s = 1.0
  processes; this is below the requested >= 20 per process, see finding E5-11); median and quartiles; Dr.Jit kernel execution time from
  the kernel history; torch GPU time from the torch profiler (sum of CUDA event durations); process CPU time and main-thread CPU time.
- Kernel-history side effect check: the same components with the flag off differ by <= 1.7 % at s = 1.0 and <= 2.8 % in the
  scaling runs (s = 1.0: q table 24.10 vs 24.13 ms, render
  58.19 vs 58.40 ms, set_geom 30.17 vs 30.14 ms, update_state 19.11 vs 18.80 ms).
- Cost-probe correctness: after the edit the selective relation update has the same state digest as a full rebuild, and the
  round trip returns the T0 digest (all runs).
- Scaling: own copy of the `econ88.refine` rule (monkeypatch of `scene86.stationary_rects` / `box_faces` / `SPH_NPHI` / `SPH_NZ`) at
  s = 1.0 / 1.5 / 2.0 / 3.0 -> N = 1,270 / 2,802 / 5,080 / 11,430.
- Generator sub-steps: `e5_gen.py` (own split of `ECRO.factors` into tokens / per-bin heads / global heads; the split reproduces
  the production factors exactly, max abs difference 0.0 on c, a, Q, P).
- Consolidation: `analyze_e5.py` -> `e5_summary.json`, `e5_tables.txt`; `supp_e5.py` -> `e5_supplement.json`.

Hardware / software: LabServer63, RTX 3080 Ti 12 GB (driver 595.71.05), 2 x Xeon Silver 4110 (32 logical CPUs), torch 2.3.1+cu121
(OpenMP backend, 16 intra-op threads), Dr.Jit 0.4.6, Mitsuba 3.5.1.

Contamination record (before each run: `uptime` 1-minute load; `nvidia-smi --query-compute-apps`):

| run | s | load before | foreign GPU apps | note |
|---|---|---|---|---|
| run1 | 1.0 | 0.01-0.17 | none | clean |
| run2 (OMP passive) | 1.0 | 2.92 | none | **contaminated**: every part 8-33 % slower than run1 (R0 edit 163.0 ms); cause not the wait policy (run7); used for nothing but a within-run ratio, superseded by run7 |
| run3 | 1.0 | 2.45 | none | load is the decaying footprint of my own preceding runs; agrees with run1 within 2 % |
| run4 / run5 / run6 | 1.5 / 2.0 / 3.0 | 2.39 / 1.65 / 1.30 | none | same remark |
| gen | 1.0 | 0.25 | none | clean |
| run7 (OMP passive, repeat) | 1.0 | 0.31 | none | clean; run in the second sitting |

GPU budget: process wall of all runs together about 6.9 minutes (63 + 20 + 63 + 48 + 63 + 120 + about 15 + 21 s), peak process
memory 6.3 GB at s = 3.0 (torch peak 2.4 GB; above the ~4 GB guideline for 2 minutes, alone on the GPU), <= 3.5 GB otherwise.

## 3. Results

### 3.1 Question 1: are the original component times and lifecycles reproduced?

Yes. R0 (explicit) lifecycles, medians in ms:

| quantity | WL96A | run1 A / B / D | run3 A / B / D | run7 A |
|---|---|---|---|---|
| edit -> frame | 144.1 | 148.9 / 147.6 / 147.7 | 146.1 / 146.0 / 146.0 | 146.6 |
| source -> frame | 88.5 | 90.2 / 90.1 / 89.4 | 89.8 / 89.8 / 89.2 | 90.3 |
| geometry update | 30.2 | 30.7 / 30.2 / 30.3 | 31.1 / 31.1 / 31.2 | 31.4 |
| relation update | 18.1 | 20.0 / 19.2 / 19.4 | 18.2 / 18.1 / 18.1 | 18.0 |
| relation coefficients | 5.0 | 6.0 / 6.0 / 6.0 | 5.0 / 4.9 / 4.9 | 4.9 |
| q table (edit) | 25.7 | 25.6 / 25.7 / 25.6 | 25.6 / 25.6 / 25.4 | 25.8 |
| prefix order | 2.0 | 1.95 / 1.96 / 1.95 | 1.89 / 1.88 / 1.88 | 1.87 |
| three tail orders | 3.3 | 3.28 / 3.26 / 3.26 | 3.32 / 3.32 / 3.31 | 3.33 |
| hand-over | 0.2 | 0.20 / 0.19 / 0.19 | 0.20 / 0.20 / 0.20 | 0.20 |
| render 512^2 x 64 | 59.5 | 60.3 / 60.1 / 60.1 | 60.2 / 60.1 / 60.1 | 60.2 |

Tolerance: lifecycles +1.3 % to +3.3 % (edit) and +0.8 % to +2.0 % (source) of the original; every part within 1.9 ms; the tail
within 0.05 ms. Interquartile ranges of the totals are <= 1.5 ms. Dirty rays per edit 49,936 of 650,240 (identical to the original);
29,881 rays actually change. Isolated components: three tail orders 3.17 / 3.21 ms (orig 3.29), one order 1.10 / 1.11 (1.14),
Lambda24 0.146 / 0.157 (0.16), Lambda64 0.227 (0.24), `factors` 4.35 / 4.39 with a pre-built GeoState, 4.79 in `e5_gen.py`
(orig 5.3, n = 5), ECRO apply 0.76 (0.77), q table 24.1 / 24.0 (24.2), frame 58.4 / 57.8 (57.3).

ZERO arm (three tail orders removed, no generator), U = R0 / ZERO:

| | WL96A | run1 A | run1 B (interleaved) | run3 A | run3 B | run7 A |
|---|---|---|---|---|---|---|
| edit | 1.038 | 1.032 | 1.022 | 1.024 | 1.021 | 1.024 |
| source | 1.043 | 1.046 | 1.045 | 1.041 | 1.040 | 1.040 |

The predeclared COST criterion (U >= 1.25) is not met in any run. The original edit value 1.038 is 1.5 points above the interleaved
measurement and above the Amdahl value from its own parts (3.31 / 144.1 -> 1.0235); with 6 edits per arm in sequential blocks this
is an order / drift effect of about 2 ms. It does not change any decision.

### 3.2 Question 2: Amdahl ceiling from my numbers

U = L / (L - t_tail), R0 events only (`e5_supplement.json` `amdahl`):

| | run1 B | run3 B | run7 A |
|---|---|---|---|
| edit -> frame | 147.6 / (147.6 - 3.26) = **1.023** | **1.023** | **1.023** |
| source -> frame | 90.1 / (90.1 - 3.27) = **1.038** | **1.038** | **1.038** |
| edit, no frame | 87.7 / (87.7 - 3.25) = 1.038 (measured R0 / ZERO 1.041) | 1.040 (1.042) | - |
| source, no frame | 30.1 / (30.1 - 3.27) = 1.122 (measured 1.129) | 1.124 (1.129) | - |

The tail is 2.2 % of an edit and 3.6-3.7 % of a source update (WL96A: 2.3 % / 3.7 %). The worklog's no-frame values 1.05x / 1.12x
(line 162) reproduce as 1.04x / 1.12-1.13x.

### 3.3 Question 3: implementation overhead versus transport work; conventions

Share of each component that is GPU kernel time (run1; run3 within 2 %; `e5_supplement.json` `overhead`):

| component | wall ms | GPU kernel ms | share | where the rest goes (CPU main thread ~ wall in all rows) |
|---|---|---|---|---|
| q table | 24.13 | 1.84 (Dr.Jit exec, 6 kernels) | 7.6 % | Python-side Dr.Jit tracing of the q program on every call; codegen 0.42 ms |
| render 512^2 x 64 | 58.40 | 22.67 | 38.8 % | about 35.7 ms fixed per call (render 256^2 x 1 spp: 35.0 ms wall, 0.18 ms exec) |
| geometry update `set_geom` | 30.14 | 0.04 | 0.1 % | `scene86.Geom` numpy construction 23.9 ms; `params.update()` 2.7 ms (contains the BVH refresh; not separable); `set_tables` 3.1 ms |
| relation update | 18.80 | 0.06 (+ 4.43 torch GPU in `dirty_rays`) | <= 24 % | `dirty_rays` 5.6 ms wall; re-trace of 49,936 rays 8.4 ms wall for 0.05 ms of kernel time; array copies |
| relation coefficients (XOp) | 5.47 | 2.01 (resident W) | 37 % | re-upload of the constant W table each build (3.69 ms with W resident) |
| one order / three tail orders | 1.10 / 3.17 | 0.53 / 1.58 (48 / 144 GPU launches) | 50 % | Python loop over receiver groups in `XOp.scatter` |
| Lambda24 apply | 0.146 | 0.105 | 72 % | |
| `ECRO.factors` | 4.35 | 4.20 | 97 % | |

GPU-kernel totals versus wall: source table path 4.08 ms of 29.0 ms; source -> frame 26.8 of 87.4 ms (31 %); edit -> frame
26.9-36.0 of 141.8 ms (19-25 %; the range is whether the 2.7 ms `params.update` wall, the `dirty_rays` GPU time and the coefficient
build are counted as GPU work). So **69-81 % of the measured lifecycle is CPU-side implementation overhead** (single-threaded
Python / Dr.Jit tracing / numpy), not transport work.

Ceilings for a zero-cost replacement of the three tail orders under frame conventions (run1, N = 1,270; composed from isolated
component medians; run3 agrees to 0.002 on wall and 0.02 on the GPU basis; `e5_tables.txt` section 5). These are conventions,
not claims about a product:

| frame convention | wall: source / edit | GPU-kernel basis: source / edit (range) |
|---|---|---|
| 512^2 x 64 (the predeclared one) | 1.038 / 1.023 | 1.063 / 1.046-1.063 |
| 512^2 x 16 | 1.047 / 1.026 | 1.195 / 1.09-1.19 |
| 512^2 x 4 | 1.050 / 1.027 | **1.40** / 1.12-1.39 |
| 512^2 x 1 | 1.051 / 1.027 | **1.53** / 1.13-1.52 |
| 256^2 x 64 | 1.048 / 1.026 | 1.18 / 1.09-1.18 |
| 256^2 x 16 | 1.051 / 1.027 | **1.39** / 1.12-1.38 |
| 256^2 x 4 | 1.052 / 1.027 | **1.53** / 1.13-1.52 |
| 256^2 x 1 | 1.052 / 1.027 | **1.59** / 1.13-1.57 |
| no frame | 1.123 / 1.039 (measured events 1.129 / 1.041) | **1.63** / 1.14-1.61 |

Answer: **no wall-time convention on this workload reaches 1.25x** (maximum 1.13x, source update without a frame; with any frame
<= 1.06x at N = 1,270 and <= 1.07x at N = 11,430). On a GPU-kernel-time basis (an idealised implementation with no CPU overhead
anywhere) the zero-cost ceiling for source updates is >= 1.25x when the frame is light (<= 4 spp at 512^2, <= 16 spp at 256^2) or
absent; for edits only at the low end of the denominator range. Three qualifications of the GPU-basis numbers:
1. They are ceilings for a free replacement. With the measured replacement costs (Lambda24 0.105 GPU ms; on edits a generator of at
   least 0.97 GPU ms, section 3.4) the source values become 1.36 (512^2 x 4) and 1.57 (no frame), the no-frame edit values 1.04-1.14.
2. The tail's own GPU time is 144 small launches; the same de-overheading that the convention assumes would also shrink the tail.
3. A pure kernel-time lifecycle is not realisable; it bounds what an engineered implementation could approach.

The frame dominates every with-frame convention through its fixed 35 ms per-call overhead, not through its rays.

### 3.4 Question 4: the generator arms

Measured (edit -> frame, LAM24; `e5_supplement.json` `generator_arms`; LAM64 is within 1 ms of LAM24 everywhere):

| condition | LAM edit ms | U vs adjacent R0 | q table ms (CPU ms) | generator part ms (CPU ms) | event CPU ms |
|---|---|---|---|---|---|
| WL96A original | 154.6 / 156.5 | 0.93 / 0.92 | 31-33 | about 9 | 580-727 |
| run1 A, production default (16 threads) | 164.0 | 0.908 | 34.6 (441) | 9.16 (135) | 698 |
| run3 A, production default | 159.1 | 0.918 | 32.9 (444) | 8.37 (124) | 690 |
| run1 / run3 G, default again after the controls | 161.9 / 157.9 | 0.912 / 0.925 | 32.4 (477 / 485) | 9.19 / 8.39 | 738 / 728 |
| run1 / run3 E, integer cast done by numpy | 152.9 / 150.1 | **0.966 / 0.972** | 25.0 / 24.9 (25.6 / 25.5) | 8.65 / 7.91 (8.7 / 7.9) | 157 / 155 |
| run1 / run3 F, `torch.set_num_threads(1)` | 152.5 / 149.9 | **0.968 / 0.974** | 25.1 / 25.0 (25.6) | 8.56 / 7.83 | 157 / 154 |
| run7, `OMP_WAIT_POLICY=passive` (clean) | 151.0 | **0.971** | 25.2 (25.8); R0 arm 25.8 | 8.00 (11.9) | 160 |

Explanation of the anomaly (measured by intervention; the last step is inference):
- `model86.GeoState` (model86.py:54-58) converts the relation tables (`hit`, `dbin`, N x K = 650,240 entries each) and the patch
  arrays with `torch.as_tensor(..., dtype=...)` on the CPU. The dtype conversion runs in torch's OpenMP intra-op pool (16 threads):
  isolated, the production GeoState costs 3.2 ms wall but 48 ms CPU; the numpy variant costs 3.3 ms wall and 3.3 ms CPU.
- After that parallel region the OpenMP workers busy-wait. The CPU time appears in the generator part (124-135 ms CPU in 8-9 ms) and
  in the following q table (441-485 ms CPU in 32-35 ms, i.e. about 14 threads spinning for its whole duration); the render part
  after it is unaffected (CPU ~ wall).
- The q table is single-threaded CPU-bound Dr.Jit tracing (section 3.3); with 14-15 sibling threads spinning it runs 7-10 ms slower.
- Three independent interventions remove both the CPU jump and the q-table slowdown (one torch thread; numpy cast; passive OpenMP
  wait policy), and restoring the default brings them back (block G). The q table itself is unchanged code in all arms.
- Inference, not measured: whether the slowdown is hyper-thread sibling contention or a lower all-core turbo frequency; which OpenMP
  runtime (libgomp or libiomp) provides the spin. Cache or allocator effects are not needed to explain the data: thread count alone
  switches the effect on and off.

So the original **0.92-0.93x is not trustworthy as a cost of the generator**: 7-10 ms of the 10.5-12.4 ms LAM-minus-R0 difference is
a thread-pool side effect of one CPU cast in the harness path. The artifact-free value with the same stand-in generator is
**0.97x** (five measurements, 0.966-0.974).

Is a geometry-conditioned generator intrinsically more expensive than three sweeps at this N? Sub-steps of `ECRO.factors` at
N = 1,270 (`gen_s1.0.json`, n = 20):

| step | wall ms | torch GPU ms |
|---|---|---|
| full production `factors` | 4.79 | 4.22 |
| form factors | 0.93 | 0.51 |
| tokens (3 message rounds, incl. form factors) | 1.98 | 0.85 |
| per-bin heads c, a (single-flight factors; not used by a Lambda-only resolvent) | 2.83 | 2.76 |
| global heads Q, P | 0.26 | 0.12 |
| **global-only generator (tokens + Q, P)** | **2.22** | **0.97** |
| three explicit tail orders (for comparison) | 3.17 | 1.58 |

- The stand-in used by WL96A (full OWN `factors` plus a fresh GeoState upload: 7.6 ms isolated, 7.8-9.2 ms inside an event) costs
  2.4-2.8x the three sweeps. That part of the original finding is real.
- 59 % of `factors` is the per-bin single-flight heads that the timed Lambda arm never uses, and the GeoState upload duplicates the
  relation tables that `XOp` uploads anyway. The part a Lambda-only resolvent needs costs 2.2 ms wall / 0.97 GPU ms, which is below
  the three sweeps on both bases. Derived edit ratios (isolated walls substituted into the run1 R0 event; **not measured inside an
  event**): 1.006 (global-only generator, state already resident), 0.984 (plus GeoState upload), 0.967 (full `factors` plus
  GeoState; agrees with the measured 0.966-0.974).
- Therefore at N = 1,270 a generator of this class is not intrinsically more expensive than three sweeps; the honest statement is
  "about parity (0.97-1.01x), no headroom". The worklog sentences "Any ECRO-class generator makes edits slower (0.92-0.93x)"
  (line 188) and "a geometry-conditioned generator costs more than the sweeps" (line 205) overstate what was measured.
- With N the picture moves against the generator: `factors` 4.35 -> 8.38 -> 15.3 -> 37.5 ms (exponent 0.98), form factors
  0.75 -> 5.5 ms (0.91), while the three sweeps go 3.17 -> 3.90 -> 4.47 -> 6.45 ms wall (0.32) and 1.58 -> 6.85 GPU ms. At
  N = 11,430 the form factors alone cost about as much as the three sweeps. The global-only generator was not timed at larger N.
- Derived cross-check outside the audited arm definition (isolated walls, accuracy out of scope): an ECRO-class full operator
  (GeoState + `factors` + apply = 8.4 ms) in place of XOp coefficients + four orders (9.7 ms) would change the edit lifecycle
  by 1.009x. This alternative accounting does not reveal headroom either.

### 3.5 Question 5: scaling with patch refinement

Medians in ms (`e5_tables.txt` section 6; exponents between N = 1,270 and 11,430 in `e5_supplement.json`):

| N (s) | one order wall / GPU | 3 tail orders | q table wall / exec | relation update | set_geom | XOp build | frame 512^2 x 64 | R0 edit / source | U(ZERO) edit / source (interleaved) | U no frame edit / source |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,270 (1.0) | 1.10 / 0.53 | 3.17 | 24.1 / 1.84 | 18.8 | 30.1 | 5.5 | 58.4 | 147.6 / 90.1 | 1.022 / 1.045 | 1.041 / 1.129 |
| 2,802 (1.5) | 1.35 / 0.69 | 3.90 | 29.4 / 4.09 | 37.1 | 62.1 | 8.1 | 59.8 | 207.8 / 96.9 | 1.020 / 1.047 | 1.027 / 1.124 |
| 5,080 (2.0) | 1.53 / 0.91 | 4.47 | 35.6 / 7.25 | 61.0 | 102.3 | 13.3 | 58.1 | 284.1 / 104.3 | 1.016 / 1.041 | 1.020 / 1.116 |
| 11,430 (3.0) | 2.40 / 2.28 | 6.45 | 55.2 / 16.7 | 184.9 | 249.3 | 55.1 | 60.3 | 640.3 / 128.9 | 1.009 / 1.054 | 1.014 / 1.116 |

- The tail's wall share does not grow toward the criterion: source 3.6 % -> 5.1 %; edit 2.2 % -> 1.1 % (the CPU-side geometry and
  relation updates grow as N^0.96 and N^1.04, the sweeps as N^0.32 in wall). The no-frame source ceiling falls slightly
  (1.129 -> 1.116).
- On the GPU-kernel basis the with-frame source ceiling grows (512^2 x 64: 1.063 -> 1.071 -> 1.086 -> 1.155) because the frame is
  constant, and the no-frame source ceiling falls (1.63 -> 1.41 -> 1.31 -> 1.33) because the q-table kernels grow faster than the
  sweeps.
- WL96A's scaling extension was correctly not triggered by its own rule (U(ZERO) < 1.15); the independent scaling runs show that
  running it would not have met the criterion at 2.2x / 4x / 9x the patches.
- Limits: refinement of the same scene only; s = 3.0 values of `set_geom` / relation update have a larger spread (edit IQR about
  +-6 ms).

### 3.6 Question 6: the timing-only edit

- It is a legitimate cost probe of the explicit update path: the selective update re-traces 49,936 rays (7.7 %), of which 29,881
  change, and yields exactly the state of a full rebuild at T1 (digest equality), and the reverse edit restores T0. The geometry
  update, relation update and coefficient rebuild therefore do the work a correct T1 state needs.
- No T1 quality was referenced in WL96A (stated in its protocol: "no quality is measured on it"), and none was measured here.
  The arms are not quality-matched: ZERO drops three orders, LAM applies random factors. U is therefore a cost ceiling for the
  learned side under the assumption that a learned operator would reach the explicit arm's T1 quality with no per-edit work
  beyond the generator. That assumption favours the learned side, so the negative cost conclusion does not rest on unreferenced
  T1 quality.
- What does rest on an untested premise: that the explicit R0 lifecycle (binned prefix order plus three binned sweeps) is the
  right-quality explicit comparator at T1. The accuracy half of WL96A used other prefixes; a more accurate (sharp) prefix costs
  more for both arms (WL91 C3URX1: 11.5 ms exact first gather + 10.6 ms sweeps), which lowers the tail's share further on wall.
- One rigid translation of one box (dirty fraction 7.7 %) is the only edit; other edit sizes change the relation-update part only,
  which is outside the tail.

### 3.7 Question 7: fairness

To the learned side:
- Generous: the prefix stand-in is one cheap binned order (2.0 ms), which maximises the tail's share; the ZERO arm is free; random
  factors cost the same as learned ones; the generator is charged only on edits.
- Unfavourable: the generator stand-in (full OWN `factors` + GeoState) includes 59 % unused work and triggered the thread-pool
  artifact (section 3.4). This affects the LAM edit numbers only, not the ZERO ceiling and not the criterion outcome.
- Structural: the learned arm is defined as "explicit relation update + coefficients + q table + explicit prefix, learned tail".
  The claim is therefore about the tail-resolvent role only. It is silent on learned operators that would replace other lifecycle
  parts (the q table, relation update and frame together are 72-78 % of the lifecycle and were held fixed for both arms).

To the explicit side:
- The explicit sweeps are an unoptimised Python group loop (48 GPU launches per order, 50 % wall overhead), and XOp re-uploads a
  constant table on each edit (1.8 ms). Both arms share the q table, frame and geometry overheads, so these do not bias U; they
  inflate the denominator for both.
- The recurrence depth (four orders in total) is fixed by the protocol. Sensitivity (run1, one order = 1.10 ms wall, zero-cost
  replacement): U >= 1.25 would need at least 20 tail orders (source -> frame), 33 (edit -> frame), 20 (edit, no frame) or
  **7 (source, no frame)**. The with-frame conclusion is insensitive to plausible depths; the no-frame source convention is not.

### 3.8 WL96 cross-workload ceiling (WL91 anchors)

From `life91.json`, arm C3URX1: edit 124.66 ms with 10.57 ms of sweeps -> 1.093 (WL96: 1.09, confirmed). Source 75.81 ms with
9.40 ms of sweeps in the source event -> **1.142**; WL96's 75.8 / 65.2 = 1.16 subtracts the edit event's 10.57 ms from the source
lifecycle. Likewise "source without render 1.98x" is 1.79x with the source event's own sweep time. Both stay on the same side of
1.25 as stated in WL96. WL96 itself records that the no-frame source convention clears 1.25x on that workload and sets it aside
as not user-visible; that is a convention choice, not a measurement. The WL91 lifecycle was not re-timed here.

## 4. Findings

| id | category | material | summary | original vs independent | verdict | confidence |
|---|---|---|---|---|---|---|
| E5-1 | confirmation | yes | R0 lifecycles, all eight parts and the isolated components reproduce | edit 144.1 vs 146.0-148.9 ms; source 88.5 vs 89.2-90.3 ms; tail 3.31 vs 3.26-3.33 ms | VERIFIED | high |
| E5-2 | confirmation | yes | Zero-cost ceiling far below 1.25 on wall; predeclared COST criterion not met | U(ZERO) 1.038 / 1.043 vs 1.021-1.024 / 1.040-1.045; Amdahl 1.023 / 1.038 | VERIFIED | high |
| E5-3 | evaluation_defect | yes (interpretation only) | Generator-arm 0.92-0.93x is inflated by an OpenMP busy-wait artifact of `GeoState`'s CPU cast; the stand-in generator also carries unused work | 0.92-0.93 vs 0.966-0.974 (artifact removed); derived 0.98-1.01 with a Lambda-only generator | IMPLEMENTATION_DEFECT for the number; CONDITIONALLY_VALID for "no gain on edits" | high (artifact), medium (derived trim) |
| E5-4 | invalid_inference | yes | "Nothing to amortise because sweeps are not the bottleneck" holds for this implementation's wall time, of which 69-81 % is CPU-side overhead; on a GPU-kernel basis with a light or no frame the source-update ceiling is 1.4-1.6 | wall max 1.13 (no frame, source); GPU basis 1.063 at 512^2 x 64, 1.40-1.63 at <= 4 spp or no frame | CONDITIONALLY_VALID (as measured); UNTESTED_INFERENCE (as a property of the transport work) | high (measurement), medium (kernel-time accounting) |
| E5-5 | confirmation | yes | Tail share does not grow toward 1.25 with N up to 9x on wall | U(ZERO) edit 1.022 -> 1.009, source 1.045 -> 1.054, no-frame source 1.129 -> 1.116 | VERIFIED (same scene, refinement only) | medium-high |
| E5-6 | dataset_limitation | no | Recurrence depth fixed at four orders; no-frame source convention would reach 1.25 with >= 7 free tail orders | derived from 1.10 ms per order | CONDITIONALLY_VALID | medium |
| E5-7 | dataset_limitation | no | Timing-only edit is a valid cost probe (selective = full rebuild), but arms are not quality-matched and no T1 reference exists; U is a ceiling under an assumption that favours the learned side | probe checks pass in 7 / 7 runs | CONDITIONALLY_VALID | high |
| E5-8 | statistical_weakness | no | Original: 6 edits per arm, sequential arm blocks, one process; edit U(ZERO) 1.5 points high; "20 warm reps" was 5 for generator and frame | 1.038 vs 1.021-1.024 | CONDITIONALLY_VALID | high |
| E5-9 | evaluation_defect | no | WL96 source ceilings use the edit event's sweep time | 1.16 -> 1.14; 1.98 -> 1.79; edit 1.09 confirmed | CONDITIONALLY_VALID | high |
| E5-10 | confirmation | no | Comparison is generous to the learned side on prefix and ZERO arm; shared overheads do not bias U; claim is limited to the tail-resolvent role | see 3.7 | CONDITIONALLY_VALID | medium-high |
| E5-11 | provenance_gap | no | Own harness stores summaries, not raw rows; components n = 18 per process; run2 contaminated and superseded by run7 | see sections 2 and 7 | NONE | high |

Evidence paths: E5-1/2 `results/audit97/E5/e5_tables.txt` sections 1-2, `e5_supplement.json` `amdahl`, original
`results/ecro_util96a/cost_s1.0.json`, worklog 96A lines 130-162. E5-3 `e5_supplement.json` `generator_arms` / `generator_trim`,
`gen_s1.0.json`, `experiments/ecro86/model86.py:54-58`, `experiments/ecro_util96a/w96a.py:603-608`, worklog 96A lines 133-134,
157, 188, 205. E5-4 `e5_tables.txt` sections 4-5, `e5_supplement.json` `overhead`. E5-5 `e5_tables.txt` section 6. E5-6
`e5_supplement.json` `orders_sensitivity_run1`. E5-7 `probe_check` in every run file; protocol `ecro_util96a.json` (`edit`).
E5-8 `w96a.py:624, 637, 664, 668`. E5-9 `e5_supplement.json` `wl96_anchor_check`, worklog 96 lines 144-147.

Decision-level impact, in one place:
- The acceptance result of the cost half (COST criterion not met; no scaling extension) is unchanged by every finding.
- Two interpretive sentences of WL96A are not supported as written: the generator penalty 0.92-0.93x (E5-3) and the unqualified
  "nothing to amortise in cost" (E5-4). Both should be read with the restrictions in section 5.

## 5. P0 assessment within this lens

**P0-4, cost half: CONDITIONALLY_VALID.** The measurement is VERIFIED: on the tested CK2 workload a free replacement of the three
tail orders gains 1.02x (edit -> frame) and 1.04x (source -> frame); the predeclared criterion 1.25x is not met; this holds up to
9x the patch count. Restrictions that must travel with the claim:
1. It is a statement about wall time in this Python / Dr.Jit / torch implementation, in which 69-81 % of a lifecycle is CPU-side
   overhead and the frame has a fixed 35 ms cost. It is not established as a property of the transport work: on a GPU-kernel
   basis the source-update ceiling is 1.06 with the predeclared frame and 1.4-1.6 with a light or no frame (edits 1.05-1.6 as a
   range, 1.04-1.14 once a minimal generator is charged).
2. It concerns only the role "learned operator replaces the three explicit tail orders after an explicit prefix", with the
   relation update, coefficient build, q table and frame kept explicit and identical in both arms.
3. Fixed recurrence depth (four orders), one scene family, one rigid edit, one GPU / CPU pair.
4. Cost only: arms are not quality-matched and T1 quality is unreferenced; the bound is a ceiling for the learned side.
5. The generator-arm figure 0.92-0.93x must be replaced by 0.97x (same stand-in) or about parity (Lambda-only generator, derived);
   "any generator makes edits slower" and "a generator costs more than the sweeps" hold for the stand-in at N = 1,270 and
   increasingly with N, not intrinsically at N = 1,270.

**P0-5 (within this lens only): CONDITIONALLY_VALID.** The cost evidence supports "no lifecycle-cost reason to learn the tail
resolvent on this workload and implementation". It does not by itself support rejecting ECRO representation mechanisms in general,
and it cannot establish a general cost impossibility, consistent with the second half of P0-5.

P0-1, P0-2, P0-3 and the accuracy half of P0-4: not assessed (outside this track).

## 6. Special checks

| check | result |
|---|---|
| Harness independence (no import of `w96a.py` / `e2e91.py`) | pass (`e5_time.py` imports mat94, fx92, ecro86, model86, quad87, xport87, scene86, pt86 only) |
| Synchronisation and compilation exclusion | pass: 0 kernel cache misses in timed events and components; warm compile 743-921 ms per process reported |
| Kernel-history flag does not distort walls | pass: <= 1.7 % difference with the flag off at s = 1.0, <= 2.8 % in the scaling runs (q table, render, set_geom, relation update) |
| Parts sum versus independent total timer | pass: 0.982-1.016 |
| Uncontended timing | run1, gen, run7 clean; run3-6 at 1-minute load 1.3-2.5 from my own preceding runs, no foreign GPU process, run1 vs run3 within 2 %; run2 contaminated and superseded |
| Cost probe correctness | pass: selective update digest = full rebuild; round trip = T0 (7 / 7 runs) |
| Split generator equals production `factors` | pass: max abs difference 0.0 on c, a, Q, P |
| Local copies identical to server outputs | pass: sha256 equal for all run JSONs and scripts |
| No leftover server process; existing `~/rest_*` untouched | pass: only `~/rest_x97/E5/` written |

## 7. Not verified, and why

- Raw per-event timings: my harness kept medians, quartiles, min and max only, so medians cannot be recomputed from raw rows.
- The micro-mechanism of the q-table slowdown under spinning OpenMP threads (hyper-thread contention versus frequency), and which
  OpenMP runtime spins. Thread count as the switch is measured; the rest is inference.
- A Lambda-only generator inside a real edit event, and its scaling with N (derived from isolated walls only).
- GPU time of the BVH refresh inside `params.update()` (bounded above by its 2.7 ms wall).
- The WL91 lifecycle (only the WL96 arithmetic on its stored numbers was checked) and any other hardware.
- T1 image quality of any arm; the accuracy half of P0-4; whether four orders suffice on CK2 (other tracks).
- Whether an engineered implementation actually approaches the GPU-kernel basis; no optimisation was attempted (audit only).
- The original run's GPU utilisation trace (`results/ecro_util96a/gpu.txt`) was listed but not analysed.

## 8. Reproducibility appendix

Inputs (sha256 prefixes; server paths under `~/rest_x96a/code/experiments/` unless noted):

| file | sha256 |
|---|---|
| ecro86/pt86.py | 912e68acd314 |
| ecro86/ecro86.py | 0ef09a869397 |
| ecro86/model86.py | 72a728b81e49 |
| ecro86/scene86.py | b75a2d9cbae1 |
| gm1_explicit87/quad87.py | cfa9c5c4e854 |
| gm1_explicit87/xport87.py | aba6e5765c22 |
| ecro_mat94/mat94.py | 15268a6a598e |
| ecro_attr92/fx92.py | fec8f4b71c80 |
| ecro_local93d/addendum_checker/scene_ck2.json | 977ba122ab32 |
| ecro_util96a/w96a.py (read, not imported) | afb80dd3200d |
| ~/rest_x92a/results/models/own_s8601.pt | 739e7eb50b25 |
| results/ecro_util96a/cost_s1.0.json (original) | 103b775ed571 |
| results/gm1_c3urx1_91/results/econ/life91.json | 2f3fc40b5afb |

Own scripts (`experiments/audit97/E5/`): e5_time.py 9675abd00f06, e5_gen.py 511f2c6715e6, run_e5.sh 98082d44653c,
chain_e5.sh 0283252cf732, analyze_e5.py d5ab06ad2432, supp_e5.py e77e8c991765.

Outputs (`results/audit97/E5/`, identical on the server under `~/rest_x97/E5/out/`): run1_s1.0.json 064728f31b25,
run2_passive_s1.0.json 1a8557b5f9e2 (contaminated), run3_s1.0.json e50632ef6a4b, run4_s1.5.json 9b953cdecb75,
run5_s2.0.json 6bddb47a8fb3, run6_s3.0.json 0e8ec749378c, run7_passive_s1.0.json 8733438fd9d9, gen_s1.0.json a9a010a6481c,
e5_summary.json 6a8f66ac308c, e5_tables.txt, e5_supplement.json; per run a `.log` (PRE / POST uptime and GPU apps) and a
`.gpu.txt` utilisation trace. `_interrupted_run_log.md` is the coordinator's action log of the first sitting.

Commands (server; `B=~/rest_x97/E5`, scripts copied to `$B/code` with scp, LF verified with `file`):

```
bash $B/code/run_e5.sh run1_s1.0 1.0                      # full plan, s = 1.0
bash $B/code/chain_e5.sh                                  # run2 (OMP_WAIT_POLICY=passive, --lamonly --noprof), run3 (repeat),
                                                          # run4-6 (--quick at s = 1.5 / 2.0 / 3.0), sequential
. ~/rest_x95a/env.sh; $PY $B/code/e5_gen.py $B ~/rest_x96a/code ~/rest_x92a $B/out/gen_s1.0.json
OMP_WAIT_POLICY=passive bash $B/code/run_e5.sh run7_passive_s1.0 1.0 --lamonly --noprof   # second sitting, idle machine
```

`run_e5.sh` expands to `$PY e5_time.py $B ~/rest_x96a/code ~/rest_x92a <s> $B/out/<tag>.json [flags]`. Seeds: source list
rng 96500 (the WL96A rule), random Lambda factors torch seed 97500, render seeds 97810+i / 97830+i, component render seed 7.
Local: `python experiments/audit97/E5/analyze_e5.py > results/audit97/E5/e5_tables.txt`; `python experiments/audit97/E5/supp_e5.py`.
