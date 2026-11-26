# Worklog — Robust Spatial Bias Inference and Frozen ECRO C1 Dynamic Closure (2026-11-23)

**Batch class:** suffix batch of WL109 (owner brief, 2026-11-23). It closes the evidence gap left by WL109 and WL109A. It
adds no estimator, denoiser or architecture.

**Unchanged verdicts.** WL109 stays **NOT_SUPPORTED**. WL109A stays **INCONCLUSIVE**. Their data, code, protocols and
worklogs are not edited.

**Question.** Can frozen ECRO C1 (M = 1) be shown, with a calibrated spatial-bias inference procedure, to preserve the
expected image within the original tolerance and to reduce TOTAL MSE at equal GPU cost, including after the frozen P7
geometry edit?

**What may change.** Only the confidence procedure for Q, its calibration, independent evidence collection, statistical
reporting and dynamic-closure instrumentation.

**What stays the same:**
- Q estimand: mean_{ROI, RGB} μ_D² / mean Var(TOTAL_R0), with D = H_C1 − H_R0.
- Limits: 0.01, and 0.02 for silhouette.
- B1: ±0.5 %.
- C1, M, table, checkpoint, renderer, ROIs and every WL109 gate.

## Stage B — development and the chosen procedure (before any WL109B evidence)

### Development sources (not qualification evidence)

- WL109A synthetic generators with fresh development seeds (77,000,001+): `experiments/ecro_estimator109b/dev109b.py`.
- Raw small-ROI pixels from WL109A's independent R0 null pool.
- WL109A's reviewer JSON.

These sources shaped the choice of procedure. None of them is used to qualify it.

### Why the WL109 bound fails

WL109 computes Q_g = mean(D̄_g² − s_g²/n) per group and puts a t interval on 8 groups.
- D̄_g² is chi-square-like and skewed to the right.
- When a realization happens to be low, the estimated SE is low as well, so the upper bound under-covers.
- With more replicates per group, s_g²/n becomes nearly constant and no longer offsets the skew. This explains why WL109A's
  heteroscedastic under-coverage got worse at larger R.
- With strongly heteroscedastic noise, a few pixels carry the whole statistic. That raises the skew further.

### Development round 1: WL109 Q compared with XJ

XJ is the cross-product U-statistic over distinct groups with a delete-one-group jackknife. 600 realizations per cell.

| Family | WL109 Q: worst Q = 0 non-coverage | XJ: worst Q = 0 non-coverage |
|---|---|---|
| heteroscedastic 1.5 | 0.100 | ≤ 0.043 |
| any family | — | ≤ 0.043 |

- XJ drops the diagonal term, and that removes the Q = 0 under-coverage.
- XJ still under-covers when a bias is injected along the noise (heteroscedastic noise with a noise-proportional bias):
  0.07–0.11.
- Cause: the bias sits on the few highest-variance elements. The jackknife SE then scales with |μ + ē| and collapses
  together with the estimate. This is the usual weakness of a Wald interval for a squared mean.

### Real heteroscedasticity (WL109A null pool, development)

- Per-pixel log-SD standard deviation in the material-boundary and silhouette ROIs is **0.04–0.45**.
- The synthetic stress families use 1.0 and 1.5. At those values one or two pixels effectively carry all the variance.
- Both stress families are nevertheless **kept as hard gates**. They are the families that sank WL109A.

### Development round 2: XJL

- **Change from XJ.** Add the worst-case variance of the linear term, b·q with b = 4λ/(R·X·scale), and invert the bound in
  q.
- **Coverage.** One-sided non-coverage is 0.000–0.002 in every family, shape, Q_true, pattern and design.
- **Cost.** The bound is conservative. In the stress families, with Var(D)/scale = 2, Q = 0 often fails the limit and
  synthetic power is low.
- **Real R0 null noise at R = 512.** Q = 0 fail rate 0; power at 2 × limit 1.0.
- **Bonferroni-level power on real noise, by G:**

| G | Power |
|---|---|
| 8 | 0.71–0.73 |
| 16 | 0.997 |
| 32 | 1.000 |

### Chosen procedure (frozen): XJL with G = 32

**Grouping**
- The R paired frames, in seed order, form R/16 subgroups of 16 pairs.
- These are merged into G = 32 contiguous groups.
- X is the number of ROI pixels × 3 RGB channels.

**Estimator**
- U = Σ_{g≠h} ⟨D̄_g, D̄_h⟩ / (G(G−1)·X·scale).
- U is unbiased for Q because distinct groups are independent.
- No clipping is applied, so U may be negative.

**Variance**
- V(q) = se²_jack + b·q.
- se²_jack is the delete-one-group jackknife variance of U.
- b = 4λ/(R·X·scale). λ is 16 × the largest eigenvalue of the sample covariance of the subgroup means over the ROI elements.
- b·q bounds the variance of the linear term 2⟨μ, ē⟩/(X·scale) for every bias field μ with mean square q·scale. The bound
  also holds under spatially correlated noise, because it uses λ_max.
- λ is biased upward for wide ROIs, which is conservative.

**Bounds**
- Upper bound: sup{q ≥ 0 : U ≥ q − t₀.₉₇₅·√V(q)}.
- Lower bound for the BIASED claim: inf{q ≥ 0 : U ≤ q + t·√V(q)}, with t at Bonferroni family α = 0.05/84.
- Both use Student t with G − 1 degrees of freedom and have closed-form roots.

**Multiplicity**
- EQUIVALENT requires every one-sided 97.5 % upper bound to be at or below its limit. This is an intersection-union test, so
  a false overall equivalence claim has probability at most 2.5 % without adjustment.
- Per-ROI upper bounds are per-ROI claims, not simultaneous intervals.
- BIASED is family-wise 5 % (Bonferroni).

**Assumptions**
- Replicates are independent across seeds.
- Groups are identically distributed within a condition.
- Nothing is assumed about pixel independence, spatial correlation, heteroscedasticity or symmetry of D.

**Expected failure modes**
- Very few groups combined with strongly skewed group means.
- λ underestimated when a rare tail never appears in the R pairs.
- Conservatism, which lowers precision rather than coverage.

The original WL109 Q (G = 8, t₇) is still reported, for historical comparison only.

## Frozen protocol summary

Protocol: `experiments/ecro_estimator109b/protocol109b.json`.

### Stage A
- Check the evaluation/109A hashes.
- Rerun the unchanged `calib109a` pool and hist. Every leaf must be exact.
- Compute the true median and 99th percentile of the per-draw upper bound (WL109A's label defect).
- Reproduce the WL109-design null fail rates.
- STOP on any failure.

### Stage C: sealed qualification

**Synthetic families**
- Families: gauss_iid, mc_like_4, mc_like_32, asym_mc4, heteroscedastic 1.0 and 1.5, het_mc4, correlated.
- Six real ROI shapes; R = 512 and R = 1024.
- 2,000 realizations per cell, with new seeds.
- Injections: Q_true ∈ {0, 0.5, 1, 2} × limit, with four patterns.
- Each entry's non-coverage is classified by its 95 % Clopper–Pearson interval:

| Class | Condition |
|---|---|
| PASS | CP upper ≤ 0.05 |
| FAIL | CP lower > 0.05 |
| BORDERLINE | otherwise |

- If no design qualifies and any entry is BORDERLINE, every cell is escalated uniformly by 6,000 more realizations.
- After escalation, an entry passes if its pooled rate is ≤ 0.05. There is no second escalation.

**GPU null pool**
- New R0 frames: 4,096 per view × spp, stored as 128 blocks of 16 pairs.
- 400 random block re-groupings per design.

**Gates on the null pool**

| Gate | Requirement |
|---|---|
| Coverage | Pooled non-coverage ≤ 0.05 for every Q_true and pattern |
| Precision | Per-setting P(upper > limit \| Q = 0) ≤ 0.01; product over the 84 settings of P(upper ≤ limit) ≥ 0.80 |
| Power | Pooled P(lower_family > limit \| 2 × limit) ≥ 0.80 for every pattern |
| Direct null | No BIASED state in two disjoint R0-versus-R0 decisions |

**Budget rule**
- R = 512 if every gate passes. Otherwise R = 1024 if every gate passes there. Otherwise **STOP INCONCLUSIVE**, and no C1
  frame is rendered.

### Stage D
- Paired R0/C1 at R for all six views × {4, 16, 32} spp, from new seeds.
- Bias decision:
  - any BIASED → Stage E;
  - otherwise any UNRESOLVED → INCONCLUSIVE;
  - otherwise → Stage F.

### Stage E
- Re-capture the flagged units from the replication seed family.
- If the flagged condition persists → BIAS_SUPPORTED, followed by the P versus P′ diagnostic, with its code committed first.
- If it does not persist → INCONCLUSIVE.

### Stage F
- New R0 grid and timing.
- WL109 rules: reference precision, grid bracketing, P2 ≤ 0.85, P3 upper bound < 1.0, P5 and P6.
- Mapping: INCONCLUSIVE, NOT_SUPPORTED (practical utility) or STATIC_PASS.

### Stage G
- P7 edit. D1–D3, D5 and D6 unchanged; D4 uses XJL.
- Mapping as in the brief.

### Seeds

| Family | Base |
|---|---|
| null | 31 M |
| paired | 32 M |
| grid | 33 M |
| timing | 33.6 M |
| dynamic | 34 M |
| dynamic reference | 34.95 M |
| mechanism | 35 M |
| replication | 38 M |
| smoke offset | +70 M |

- Every family is disjoint from WL101–WL109A and from the development seeds. `tests109b.py` checks this.
- A rebase of the replication family from 32.7 M to 38 M was found by this test, before anything was committed.

### GPU ownership

- Every GPU unit runs under the unchanged WL109 guard.
- The guard waits on the CPU, without a GPU library loaded, until no tmux session exists and the GPU has had no compute
  process for 120 s.
- When a foreign compute process appears, the guard kills only its own child's process group.
- Units are atomic and are skipped on restart.
- CPU stages run with `CUDA_VISIBLE_DEVICES=`.

## Execution record

(appended below as the chain runs)

### 2026-11-23 — chain start and smoke

- The chain `run109b.sh evidence cc8fb24` started on LabServer69 at 18:34:06 (+09:00).
  - Server output: `results/ecro_estimator109b/evidence/`.
  - Protocol sha256: `ba1f19a6…`.
- `tests109b.py` passed.
- Two CPU jobs started in the background: Stage A (WL109A reproduction) and synthetic stage 1.
- The guard waited for a short foreign GPU process to end. It then admitted each smoke unit:

| Smoke unit | Duration (s) |
|---|---|
| null | 160 |
| paired | 100 |
| grid | 41 |
| timing | 43 |
| dynamic | 37 |

- Every code path ran to the end:
  - smoke synthetic, pool and decide (outcome `SMOKE`; the frozen rule would have asked for escalation on the 3-realization
    smoke cells);
  - the bias phase and the static phase, which the smoke run forces through;
  - dynamic D1–D6.
- All smoke numbers are smoke data, not evidence. They use R = 64 and G = 4.
- `SMOKE PASS` at 18:52:37. The independent R0 null pool (4,096 frames per view × spp) started under the guard.

### 2026-11-23 / 24 — null pool, Stage A, Stage C qualification, paired capture in progress

**Null pool.** 4,096 R0 frames for each of the 18 view × spp settings. One attempt with no preemption, 18:54–20:34
(6,012 s).

**Stage A: PASS.**
- Artifact hashes match.
- The unchanged `calib109a` pool and hist reruns reproduce every leaf: 0 mismatches each. The budget decision is equal.
- The label defect is confirmed: in every setting, the archived `upper_median` equals the mean of the per-draw upper bounds.
  The true quantiles are in `stageA/hist_upper_quantiles.json`.
- The WL109-design null fail rates for H3/H4 material boundary reproduce.

**Stage C qualification: PROCEED with R = 512.**
- Synthetic stage 1 took 3,614 s. The real-null pool analysis ran 20:34 → 06:27 on the CPU.
- No escalation was needed.

| Gate | R = 512 | R = 1024 |
|---|---|---|
| Synthetic entries | 624 / 624 PASS (0 FAIL, 0 BORDERLINE); worst one-sided non-coverage 0.000 | 624 / 624 PASS; worst 0.0005 |
| Real-null pooled non-coverage | 0.000 for every Q_true and pattern | 0.000 |
| Precision | max per-setting fail 0.000; joint pass probability 1.000 | max per-setting fail 0.000 |
| Pooled Bonferroni power at 2 × limit | ≥ 0.99997 for every pattern | 1.0 |
| Direct null | 56 / 56 states EQUIVALENT, none BIASED | 56 / 56 EQUIVALENT |

- By the budget rule (the smallest design that passes every gate), **R = 512** was selected. No C1 frame existed before
  this decision.
- The measured coverage is far more conservative than the nominal 0.025. This is the expected price of the worst-case
  linear-term variance. It lowers synthetic precision in the stress families. On real noise it did not.

**Paired capture (Stage D), R = 512.**
- The guard waited for foreign GPU work from 06:27, then admitted the unit at 10:49.
- It preempted itself three times, each time killing only its own child when foreign work appeared:

| Time | Foreign work | WL109B run time before preemption |
|---|---|---|
| 10:50 | foreign PID | 33 s |
| 10:56 | foreign PID | 45 s |
| 14:37 | tmux `b28` | 1,332 s |

- At 14:55, 16 of 18 units are complete. H5 at 16 and 32 spp remain.
- The guard is waiting while `b28` and its GPU processes run, and holds no GPU memory.

## WL109B Amendment 1 — owner-directed host move LabServer69 → LabServer70 (2026-11-24)

**Owner direction, 2026-11-24.**
- LabServer69 is held by TransGI reproduction work until the evening.
- Execution therefore moves to LabServer70, which has the same RTX 3090.
- The owner expects no material effect on the final decision.
- LabServer63 is no longer used.

**Classification.** This is a non-scientific execution change inside the same ongoing protocol, so it is an amendment.
- The estimator, the inference procedure, the budget R = 512, every threshold and every decision rule are unchanged.
- Stages A–C and their outputs stand as completed on LabServer69.

**State at the move.**
- On LabServer69 the chain was waiting in the guard behind tmux `b28`. It held no GPU memory.
- 16 of 18 paired units were complete. H5 at 16 and 32 spp were missing.
- The operator stopped the chain processes (`run109b.sh` and its guard, by PID; no foreign process was touched) and
  appended an `OPERATOR STOP` line to `chain.state`.

**What was copied to LabServer70, paths unchanged, nothing on 69 modified or deleted:**
- the Python environment (venv and uv Python), Mitsuba 3.8.0 and `rest_dyn101`;
- the ReST-GI code copy;
- the frozen inputs (WL105/107/108);
- WL109 manifests, dev and metrics;
- the WL109B evidence directory without the null pool and smoke outputs, which later stages do not read.

Transfer script: `results/ecro_estimator109b/xfer70.sh` on LabServer69.

**LabServer70 differences.**
- NVIDIA driver 525.60.13; LabServer69 uses 550.54.15.
- A second GPU, an RTX 3080 Ti. GPU commands are pinned to the RTX 3090 by UUID, but the guard watches every GPU and
  every tmux session on the host.

**Predeclared cross-host rule** (`xcheck109b.py`, run under the guard before any further capture):
1. The unchanged WL109 admission must pass on LabServer70. It checks sources, checkpoint, scene and table hash (which must
   equal the WL109 manifest), and replays the archived 4096-spp V0 frame.
2. The first replicate of each of the 16 completed paired units is re-rendered with the same seed. It is compared with the
   stored `frame0_R0` and `frame0_C1H`.
   - **PASS** if every relative L1 is ≤ 1e-6 (the WL109 structural tolerance). The 16 units are kept, and only the 2
     missing units are captured on LabServer70.
   - **FAIL** otherwise. The LabServer69 units are moved aside and kept as descriptive only, and all 18 units are
     recaptured on LabServer70 with the same seeds.

**Rest of the chain.**
- Grid, timing and dynamic all run on LabServer70 (`run109b_amend1.sh`), so every equal-time comparison stays within one
  host and GPU.
- WL109's 0.792 equal-time ratio was measured on LabServer69. It is compared only descriptively.

### Amendment 1 execution: the frozen admission refuses LabServer70 (2026-11-24)

- **Transfer, 69 → 70.** Finished 16:40–17:23 on LabServer69's clock. LabServer70's clock runs a few minutes behind.
  All 16 transferred paired units match their `unit.json` hashes.
- **`tests109b.py`** passed on LabServer70.
- **`xcheck` stopped in the frozen admission after 6 s.** The guard admitted it after 120 s of free GPU. No GPU work had
  started.
  - Immediate cause: `renderer108.frozen_runtime` parses a single `nvidia-smi` row, but LabServer70 has two GPUs.
  - Underlying cause: the frozen WL108 admission pins LabServer69. It asserts GPU = RTX 3090 with driver **550.54.15**,
    asserts the OptiX library is `libnvoptix.so.550.54.15`, and labels the host `LabServer69`.
  - LabServer70 runs driver 525.60.13, so the frozen admission cannot pass there even with the parse fixed.
- The owner was asked to choose among three options:
  - wait for LabServer69;
  - replace the admission for LabServer70;
  - align LabServer70's driver.

## WL109B Amendment 2 — LabServer70 admission (owner decision, 2026-11-24)

**Decision.** The owner chose to replace the admission for LabServer70.

**What changed.**
- `experiments/ecro_estimator109b/admission70.py` is a copy of `renderer108.frozen_runtime`. Only its environment pins
  differ:
  - The GPU row is selected by the UUID in `CUDA_VISIBLE_DEVICES`. The name must still be RTX 3090.
  - The driver must equal LabServer70's declared driver, 525.60.13.
  - The OptiX library must match that driver.
  - The host label and the OptiX sha256 are recorded as found.
- The diff against the frozen function contains only these lines.

**What is unchanged and still executed.**
- Baseline and prior manifests.
- Renderer source hashes.
- Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, NumPy and Python versions.
- Workload, camera and scene fingerprint.
- Learned HIGH table hash.
- The archived 4096-spp V0 replay within the WL105 tolerance.

**How it is installed.**
- The frozen file is not edited.
- `common109b` installs the copy in place of `renderer108.frozen_runtime`, inside WL109B processes only, when
  `WL109B_ADMISSION=lab70` is set.
- The chain `run109b_amend2.sh` is `run109b_amend1.sh` with that flag set.

**What still guards reproducibility.**
- The Amendment 1 cross-host rule is unchanged: re-render the first replicate of each of the 16 LabServer69 paired units.
  - All relative L1 ≤ 1e-6: keep the 16 units.
  - Otherwise: recapture all 18 units on LabServer70.
- If Mitsuba/Dr.Jit OptiX cannot run on driver 525, the admission fails and nothing is captured. That would be reported.

### Amendment 2 execution and Amendment 3 probe: LabServer70 cannot run the frozen renderer (2026-11-24)

**Amendment 2 chain (`run109b_amend2.sh`, commit 5c4bdeb).**
- The first launch carried a placeholder commit label. It was stopped before any GPU work, recorded in `chain.state`, and
  relaunched.
- `xcheck` then stopped while importing Dr.Jit: `libatomic.so.1` is missing.
  - The WL108 runtime directory holds a symlink to LabServer69's `/usr/local/anaconda3/lib/libatomic.so.1`.
  - LabServer70 has no anaconda installation.
- A copy of the real file, `runtime70/lib`, removed that error. Two further incompatibilities then appeared:
  - LabServer70 runs CentOS 7, with glibc 2.17 and the GCC 4.8 `libstdc++`.
  - The frozen Dr.Jit 1.3.1 binaries need GLIBC 2.27 (`libdrjit-core.so`) and GLIBCXX 3.4.22.

**Amendment 3 probe (owner decision: try Docker).**
- Command: `docker run --rm` with the `python:3.10-bookworm` image (glibc 2.36).
  - GPU: the RTX 3090 only, selected by UUID, with `NVIDIA_DRIVER_CAPABILITIES=all`.
  - User `nd` (uid 1014); `/home/nd` mounted at the same path.
  - Before the run, no foreign GPU process or tmux session was present.
- Inside the container the frozen stack loads with the frozen versions: Dr.Jit 1.3.1, Mitsuba 3.8.0, Torch 2.3.1+cu121,
  NumPy 1.26.4, Python 3.10.21.
- OptiX initialization fails: *"OptiX 8.0 requires driver revision R535 or newer"*. LabServer70's driver is 525.60.13.

**Conclusion.**
- The frozen renderer cannot run on LabServer70 while its driver is older than R535, whether natively or in a container.
- Nothing was captured on LabServer70. No WL109B evidence or decision depends on it.
- Amendments 1–3 are recorded as **not executable on LabServer70**.
- The transferred copies remain on LabServer70 and are unused.
- LabServer69's data and state are unchanged: 16 of 18 paired units done; Stages A–C complete.

## WL109B Amendment 3 — LabServer70 driver 550.54.15 and containerized execution (owner decisions, 2026-11-24)

**Owner decisions.**
- LabServer70 must become a usable shared execution host. Under no circumstances may it go down.
- Prepare a fallback access path first.
- Upgrade the driver without a reboot.
- Run the frozen stack in Docker.

### Fallback access
- A VS Code tunnel `labserver70` runs as nd in a Docker container: `nd-vscode-tunnel`, image `python:3.10-bookworm`,
  restart policy `unless-stopped`, host network. Docker is enabled at boot. GitHub device login was done by the owner.
- Limitation: the tunnel terminal is a shell in the container, not on the host.
  - Host sshd (port 26048) authenticates only through the lab privacyIDEA OTP (`pam_privacyidea`). The owner has no OTP
    token.
  - The tunnel therefore gives non-root access only. Host root recovery still needs tailscale, an OTP, or local/LAN access
    (xrdp 3389 is open).
- An unsanctioned docker-socket route to host root was declined.

### Driver 525.60.13 → 550.54.15
- Executed by the owner, with sudo, from commands prepared and checked here. No reboot.

**Preparation (nd, no root).**
- The NVIDIA online repository is not configured on LabServer70. The 525 driver came from the local
  `cuda-rhel7-12-0-local` repo.
- The 12 RPMs (the 11 installed driver packages plus the new dependency `nvidia-kmod-common`) were downloaded from the NVIDIA
  rhel7 repository into `~/nvidia-550.54.15/`.
- Verification:
  - sha256 against the repository metadata: all 12 match.
  - GPG signatures against NVIDIA key D42D0685 (fingerprint 610C 7B14 E068 A878 070D A4E9 9CD0 A493 D42D 0685), checked
    in a temporary rpmdb: all 12 OK.
  - Epoch 3, as installed.
- No repository was added to the system.
- `yum --disablerepo='*' -x 'kernel*' --assumeno localinstall` dry run: Install 1, Upgrade 11, no other package, no
  dependency error.

**Rollback path (checked, not needed).**
- The local repo still holds the 44 files of 525.60.13: `yum history undo` offline.

**Execution.**
1. Stopped the nine GPU containers and lightdm. lightdm's X server (`/usr/bin/X` on GPU 0) and nd's `btop` held the
   device.
2. With the nvidia module down to its two inter-module references and no holder in `fuser`:
   - `localinstall` completed.
   - `dkms status`: `nvidia/550.54.15, 3.10.0-862.el7.x86_64: installed`.
   - The 550.54.15 modules were loaded with `modprobe`.
3. Result: `NVRM 550.54.15`; `nvidia-smi` shows driver 550.54.15 (CUDA 12.4), with the RTX 3090 and the RTX 3080 Ti.
4. lightdm and the nine containers were restarted. The X server is back on GPU 0.
5. Containers reach the GPU through toolkit 1.11.

### Container probe
- The frozen stack runs in `python:3.10-bookworm`, image ID `sha256:6fd6ebbd05fc…271d`.
- It needs three host files bind-mounted read-only. Toolkit 1.11 predates driver R535 and does not inject them:
  `/usr/share/nvidia/nvoptix.bin`, `libnvidia-gpucomp.so.550.54.15` and `libnvidia-nvvm.so.4`.
  - Without them, OptiX fails with `OPTIX_ERROR_INTERNAL_COMPILER_ERROR` (7299).
- With them in place:
  - Dr.Jit 1.3.1, Mitsuba 3.8.0, Torch 2.3.1+cu121, NumPy 1.26.4, Python 3.10.21; glibc 2.36.
  - An OptiX render runs; `libnvoptix.so.550.54.15` is loaded; torch sees the RTX 3090.
- Host `nvidia-smi` lists a container's compute process by its host PID, the same PID `docker top` reports. The guard can
  therefore see foreign containerized GPU work and identify its own.

### Amendment 3 execution design (`run109b_amend3.sh`)
- **Frozen admission.** The frozen WL108 admission runs **unchanged** inside the container. The container exposes only the
  RTX 3090, with driver 550.54.15 and `libnvoptix.so.550.54.15`, which satisfies its pins. Amendment 2 is not used.
- **Container.** `ctr109b.sh` runs each GPU command in a container: image pinned by ID; RTX 3090 by UUID; uid 1014;
  `/home/nd` and the three OptiX files mounted; the frozen interpreter and venv used as files.
- **Guard.** `guard109c.py` is `guard109.py` with three additions:
  - GPU PIDs of its own named container (`docker top`) count as its own.
  - On preemption or timeout it runs `docker kill` before killing the process group.
  - It enforces a 1-minute load-average gate during the 120-s free window.
- **Load gates.**

| Stages | 1-min load average must be |
|---|---|
| xcheck, paired, grid (data) | ≤ 8 (leave CPU to others) |
| timing, dynamic (timing-bearing) | ≤ 2 (no CPU contention in measured frame times) |

- Load averages are logged at every start, exit and preemption.
- **Unchanged.**
  - The Amendment 1 cross-host rule (keep the 16 LabServer69 units only if every re-rendered first replicate matches within
    1e-6; otherwise recapture all 18).
  - The CPU stages, which run on the host.
  - Every estimator, inference, threshold and decision rule.

## WL109B Amendment 4 — CPU stages on Main WSL (owner decision, 2026-11-24)

**Why.** LabServer70's 16 cores are held by other work: load average about 16 at 18:45. The owner asked that no CPU on
LabServer70 be used, and that the CPU stages be offloaded to Main.

**Main.** The owner's Windows workstation, reached by `ssh Main`. It runs WSL2 Ubuntu 22.04 with 32 cores, 31 GB of memory
and glibc 2.35, and reaches LabServer70 directly over tailscale SSH.

**Mirror.** `pull70_main.sh`, run detached through the repo's `wsl_run_detached.sh`, pulls from LabServer70 to the same
`/home/nd` paths:
- the frozen uv Python and venv;
- `rest_dyn101`;
- the code;
- the WL108 inputs and WL107 capture/reference;
- the WL109 manifests;
- the WL109B evidence, excluding `tmp`, `mpl` and `smoke`.

The analyses run with `HOME=/home/nd`, so every path resolves exactly as on the GPU hosts.

**Split of stages.**

| Host | Script | What runs |
|---|---|---|
| LabServer70 | `run109b_amend4_gpu.sh` | GPU only, in three phases: gpu1 (xcheck + paired); gpu2 (grid + timing, only if `bias109b.json` is EQUIVALENT_ALL); gpu3 (dynamic, only if `static109b.json` is STATIC_PASS) |
| Main WSL | `run109b_main.sh` | CPU only: tests; bias (pull paired + xcheck, `analyze109b --phase bias`, push the bias decision); static (pull grid + timing, `analyze109b --phase static` + `decide109b`, push) |

- Main appends its states to LabServer70's `chain.state`, prefixed `MAIN`. LabServer70 holds the authoritative record.
- The LabServer70 guard polling (nvidia-smi and the load average every 10 s while waiting) is the only work done there
  outside the GPU stages.

**Reproducibility note.** The analyses are deterministic NumPy/SciPy code on the same interpreter and wheels. A different CPU
can change floating-point results only at the last-bit level, through BLAS kernel selection. No decision threshold is near
that scale.

### Amendment 4 withdrawn (2026-11-24)

**Why it was withdrawn.**
- The tailscale path from Main WSL to LabServer70 carried about 3 MB/s.
- Mirroring about 26 GB would take about 2 h.
- The owner chose instead to give WL109B a few LabServer70 threads.

**What had run.** Before the withdrawal, nothing beyond the setup steps had executed:
- The Main pull had copied 1.3 GB. It was stopped by PID; the partial mirror is unused and stays in Main WSL `/home/nd`.
- The LabServer70 gpu1 chain was still waiting at its load gate. It was stopped by PID.
- No GPU stage and no analysis had started.

## WL109B Amendment 5 — capped CPU use on LabServer70 (owner decision, 2026-11-24)

Everything runs on LabServer70 again (`run109b_amend5.sh`). Other work holds the host CPU, so WL109B caps its own use.

**CPU stages on the host.**
- Pinned to 4 cores with `taskset -c 12-15`.
- Run under `nice 10` with 4 BLAS threads.

**GPU containers.** Limited to 4 CPUs (`docker --cpus 4`).

**Load gates.**
- Data stages (xcheck, paired, grid) start without a load gate. Their outputs depend only on the seeds.
- The timing-bearing stages (timing, dynamic) keep the Amendment 3 gate: 1-minute load average ≤ 2. Their measured frame
  times must be free of CPU contention, so they wait until the host CPU is free.

**Unchanged from Amendments 1 and 3.** The frozen admission inside the container, the cross-host rule, and every estimator,
inference, threshold and decision rule.

### 2026-11-24 — Amendment 5 execution: cross-host check, Stage D bias decision, grid

All times are LabServer70's clock.

- **Chain start.** `run109b_amend5.sh` (c77cb86) started at 18:58. `tests109b.py` passed (CPU-capped).
- **Cross-host check: PASS, bitwise** (`xcheck/xcheck109b.json`).
  - The guard admitted it at 19:00:23 after 120 s of free GPU. Load average was 13.9; this stage has no load gate.
  - The frozen WL108 admission passed **unchanged** inside the container: sources, checkpoint, scene, table hash and the
    archived 4096-spp V0 replay.
  - The 16 LabServer69 paired units: the first replicate re-rendered on LabServer70 equals the stored frame **bitwise**
    for both R0 and C1 HIGH (relative L1 0.0). The units are kept.
- **Paired.** The two missing units (H5 at 16 and 32 spp, R = 512) were captured in 237 s.
  - LOW and the original HIGH were bitwise identical in all 512 frames of each unit.
  - All 18 units are now present.
- **Stage D bias decision: EQUIVALENT_ALL** (`metrics/bias109b.json`, CPU-capped analysis, 19:07–19:12).
  - Q (XJL): 28 / 28 conditions EQUIVALENT, at every bias spp.
  - B1: 28 / 28 EQUIVALENT.
  - No condition is flagged.

| Condition | Largest upper bound / limit | Note |
|---|---|---|
| H4 material boundary | 0.138 (16 spp) | point 0.024 × limit |
| H3 material boundary | 0.083 (4 spp) | |
| Every other condition | ≤ 0.05 | |

  - B1 95 % intervals lie within ±0.08 %. The tolerance is ±0.5 %.
  - The original WL109 Q (G = 8; descriptive only) on the same R = 512 data has every upper bound ≤ 0.08 × limit.
- **Stage F grid.** 18 levels × 64 replicates, 675 s.
- **Timing.** Waiting at its load gate (≤ 2). Load average about 14 at 19:26, from other work.

### Amendment 5 addendum — per-GPU foreign detection in the guard (owner decision, 2026-11-24)

**Finding (reported by another process on LabServer70).**
- `guard109c.py` inherits `guard109.py`'s rule: any compute process on any GPU counts as foreign.
- That is correct on single-GPU LabServer69.
- On LabServer70 it means work on the RTX 3080 Ti alone blocks or preempts WL109B on the RTX 3090, although the two share no
  device memory.

**Change.** A `--gpu-uuid U` option.
- Only compute processes on GPU U count, for both the start condition and preemption. They are read from
  `nvidia-smi --query-compute-apps=pid,gpu_uuid`.
- The default is unchanged: every GPU on the host.

**Use.**
- Data stages on a multi-GPU host use the option.
- Timing-bearing stages keep the whole-host rule, because work on another GPU can still perturb measured frame times through
  PCIe or CPU.

**Effect on this batch.**
- WL109B's remaining stages are timing and dynamic, which both keep the whole-host rule. Neither the running chain nor any
  WL109B result is affected.
- The running chain script was deliberately left untouched: bash reads a running script incrementally.
- Only `guard109c.py` was updated on LabServer70.

### 2026-11-24/25 — timing stalls inside the container; WL109B Amendment 6 (in-container exclusivity check)

**Timing attempts.** All times are LabServer70's clock.

| Attempt | Admitted | Load | Ended | Run time | How it ended |
|---|---|---|---|---|---|
| 1 | 01:37 | 1.7 | 03:23 | 6,340 s | Preempted by the guard when foreign PID 13978 appeared |
| 2 | 03:45 | — | 03:46 | 47 s | Preempted by foreign PID 6175 |

Since then the guard waits: GPU work on the RTX 3080 Ti (the timing stage keeps the whole-host rule), and load above 2.

**Defect: attempt 1 never measured anything.**
- `capture_timing.log` repeats *"waiting for foreign GPU process [18326]"* 211 times, every 30 s.
- PID 18326 was the capture process itself.
- Cause: `runtime109.exclusive_block` → `wait_free_gpu` excludes only `os.getpid()`. Inside a container that is the
  namespace PID (7), while `nvidia-smi` reports the process by its host PID. The process therefore waited for itself.
- The smoke run never reaches this path: it uses `require=False`. On LabServer69 no container is involved.
- Consequence: the capture held a CUDA context and about 1.8 GB on the RTX 3090 for 1 h 45 min without measuring. When
  real foreign work appeared, the host guard preempted it correctly. No timing data was written.

**Amendment 6.**
- What changes: inside the LabServer70 container (`WL109B_HOST=LabServer70`, set by `ctr109b.sh`), `common109b` replaces
  `runtime109.foreign_gpu_apps` with an empty list, in WL109B processes only. The frozen file is not edited.
- Where exclusivity comes from instead: the host guard `guard109c.py` checks every GPU on the host every 2 s. On any foreign
  compute process it kills the container and restarts the unit from the beginning.
- What this means for a record:
  - A timing or dynamic unit can complete only if no foreign compute process appeared during its attempt.
  - The guard log for that attempt (start → exit with no preemption) is the exclusivity record.
  - The in-JSON `exclusivity` field is therefore always "clean" and is not used as evidence.
- The load gate for timing-bearing stages (≤ 2) and the whole-host rule are unchanged.

### 2026-11-25/26 — host GPU failure and reboot; chain relaunched (execution record, no protocol change)

All times are LabServer70's clock.

| Time | Event |
|---|---|
| 11-25 13:00–17:56 | Timing still waiting: other users' compute processes on the RTX 3090 (12510, 22343, 13489 …) |
| 11-25 17:57 | `nvidia-smi` starts failing on the host (a GPU failure; the owner confirmed the GPU went down). The guard cannot see the GPUs and keeps waiting (48 logged waits) |
| 11-26 17:53 | Host rebooted. The chain (PID 10041) and the timing guard (PID 10742) are killed at the timing wait. No `STOP` line: the processes were killed, not failed |
| 11-26 18:51 | GPUs recovered (driver 550.54.15 loaded, both idle). The same chain `run109b_amend5.sh` is relaunched at commit d2e255e (code on the host verified identical to the commit) |

- **No WL109B GPU work ran** between attempt 2 (preempted 11-25 03:46) and the reboot. No timing unit was written; only a
  `.tmp_V0` from the preempted attempt remains, and the capture deletes it at the next unit start.
- **On relaunch**, completed stages are verified and skipped: tests rerun, `xcheck109b.json` = PASS is reused, and paired / grid
  units are detected as complete. The bias analysis is a deterministic recomputation from the same paired units.
- The cause of the GPU failure was not investigated by this batch (the host logs need root). No WL109B measurement was
  taken before or during it.

### 2026-11-26 — WL109B Amendment 7 (owner decision): the guard watches only the RTX 3090 in every stage

**Decision.** The owner directed that WL109B should run whenever the RTX 3090 is free, without regard to the RTX 3080 Ti.

**What changes.**
- Every guarded stage passes `--gpu-uuid` of the RTX 3090 (GPU-ab3bc073…) to `guard109c.py`. This now includes the
  timing-bearing stages (timing, dynamic).
- Only compute processes on the 3090 count as foreign. That applies both to the start condition (120 s free) and to
  preemption.
- Work on the 3080 Ti no longer delays or preempts WL109B.

**What is unchanged.**
- The load gate of the timing-bearing stages: 1-minute load average ≤ 2 at admission.
- The container, frozen stack and Amendment 6.
- The protocol, seeds, criteria and analysis.

**Execution.**
- New chain `run109b_amend7.sh`: `run109b_amend5.sh` with the guard flag added, resuming after the completed bias stage.
- On 11-26 the relaunched Amendment 5 chain had recomputed the bias stage (`EQUIVALENT_ALL`, 18:56). It was then stopped by
  PID while its grid guard was waiting for the 3080 Ti. No container was running, and nothing was written.
- The order from there is unchanged: grid (complete units skipped) → timing → static → dynamic.

**Interpretive consequence (to be reported with Stage F/G).**
- Frame times measured while the 3080 Ti runs other users' work can include host-level interference that does not occur
  on an idle host: PCIe/memory bandwidth, CPU and power/thermal.
- R0 and C1 are measured in the same interleaved schedule, so such interference largely affects both, but not necessarily
  equally.
- The guard log records the load at start and exit. The chain also writes both GPUs' utilization and memory every 10 s
  (`evidence/gpu_util_lab70.csv`; `nvidia-smi` queries only), so each timing unit can be classified afterwards by whether
  the 3080 Ti was busy during it.

**Amendment 7 addendum (2026-11-26 19:10): load gate of the timing-bearing stages 2 → 4.**

- **Observation.** After Amendment 7 the grid check ran at 19:02 (27 s, all units skipped). Timing then waited on the load
  gate.
- **Cause.** The host 1-minute load sat at 2.0–2.2. It was driven by the RTX 3080 Ti training process (PID 14270, one core
  at ~94 %). So the ≤ 2 gate re-imposed the 3080 Ti dependence that the owner had removed.
- **Change.** The gate becomes a chain parameter, `WL109B_TIMING_MAXLOAD` (default 2), set to 4 for this relaunch.
- **Why 4 is safe.** The host has 16 cores and WL109B's container is capped at 4 CPUs. A load of ≤ 4 at admission still
  leaves at least 8 idle cores.
- **What is recorded.** The load at start and exit of every attempt (guard log). The 10 s GPU-occupancy record continues.

### 2026-11-26 — WL109B Amendment 8 (owner decision): no GPU guard for the remaining stages

**Decision.** At 19:12 the owner directed that the guard be removed altogether.

**What changes.**
- The remaining GPU stages (timing, then dynamic if STATIC_PASS) start immediately in the same container (`ctr109b.sh`, RTX
  3090 only).
- There is no idle window, no load gate and no preemption.
- This supersedes the brief's GPU-ownership rule for these stages. That rule required a CPU-only wait and preemption on
  foreign GPU work.

**What is unchanged.**
- The container, frozen stack, protocol, seeds and analysis.
- Amendment 6: the in-container foreign check stays disabled.

**What replaces enforcement.**
- Exclusivity is now recorded, not enforced.
- `evidence/gpu_util_lab70.csv` and `evidence/gpu_apps_lab70.csv` hold both GPUs' utilization and compute processes every
  10 s, from `nvidia-smi` queries only.
- `start_unguarded` / `exit_unguarded` events with the load are appended to the guard log.
- A timing or dynamic unit that overlapped foreign compute on the RTX 3090 will be identified from these records and
  reported. It is not silently used.

**Execution.**
- New chain `run109b_amend8.sh`: `run109b_amend7.sh` without the guard, resuming after the completed grid stage.
- The Amendment 7 chain was stopped by PID while its timing guard was in the 120 s idle window.

### 2026-11-26 — Stage F timing, static decision and Stage G dynamic (Amendment 8 chain)

All times are LabServer70's clock. Commit label of the chain: `2504147`.

| Stage | Window | Duration | Outcome |
|---|---|---|---|
| timing | 19:12:37–19:20:44 | 487 s | 6 / 6 views written; admission passed, table sha equal to WL109 |
| static (CPU, 4 cores) | 19:20:44–19:23:13 | 149 s | **STATIC_PASS** |
| dynamic | 19:23:13–19:30:26 | 433 s | **QUALIFIED_STATIC_AND_DYNAMIC** |

**Exclusivity record** (`gpu_apps_lab70.csv`, every 10 s):

| Stage | RTX 3090 samples | Foreign process on the 3090 | RTX 3080 Ti |
|---|---|---|---|
| timing | 49 | none (48 show only WL109B PID 19491; the first is before the container started) | foreign training PID 14270 throughout; mean utilization 33 % |
| dynamic | 43 | none (only WL109B PID 7796) | same foreign training throughout |

- 1-minute load average: 2.05 at timing start, 2.89 at timing exit, 2.93 at dynamic start, 2.72 at dynamic exit.
- The in-JSON `exclusivity` fields are "clean" by construction under Amendment 6 and are not used as evidence. The table
  above is the evidence.

## Final report

### Implementation Fact

- **Procedure.** The frozen inference procedure is XJL with G = 32 groups of 16-pair subgroups at R = 512. It consists of the
  cross-product U-statistic, the delete-one-group jackknife variance and the worst-case linear-term variance b·q, inverted
  with Student t at 31 df.
- **Unchanged inputs.** The estimator (C1, M = 1), table, checkpoint, renderer, ROIs, Q estimand, limits (0.01; silhouette
  0.02) and B1 (±0.5 %) are imported unchanged from WL109.
- **Stages run in protocol order.**
  - Stage A: PASS.
  - Stage C: PROCEED at R = 512.
  - Stage D: EQUIVALENT_ALL.
  - Stage E: not entered, because no condition was BIASED.
  - Stage F: STATIC_PASS.
  - Stage G: QUALIFIED_STATIC_AND_DYNAMIC.
- **Hosts.**
  - LabServer69 ran Stages A–C and 16 of the 18 paired units.
  - LabServer70 ran everything else, inside the pinned container (Amendments 1, 3 and 5).
  - The 16 LabServer69 units passed the bitwise cross-host check.
- **Execution amendments.** Amendments 1–8 are recorded above. Amendment 4 was withdrawn.
- **Changes to GPU-ownership enforcement by owner decision.**
  - Amendments 7 and 8 change only how GPU ownership is enforced.
  - The final timing and dynamic stages ran without a guard. Their exclusivity on the RTX 3090 is shown by the 10 s record.
  - The RTX 3080 Ti ran other users' training during both stages.
- **No change to the rest.** No protocol threshold, seed, estimator, table or decision rule was changed at any point.

### Measurement

**Stage C (qualification, before any C1 frame).**

| Check | Result |
|---|---|
| Synthetic entries, R = 512 | 624 / 624 PASS; worst one-sided non-coverage 0.000 |
| Real-null pooled non-coverage | 0.000 |
| Precision at Q = 0 | max per-setting fail 0.000 |
| Bonferroni power at 2 × limit | ≥ 0.99997 |
| Direct R0-versus-R0 null | 56 / 56 EQUIVALENT |

**Stage D (spatial bias).** 28 ROI × view conditions (primary, SSS interior, silhouette and material boundary for six views;
control for four) at 4, 16 and 32 spp.

- **Q (XJL):** 28 / 28 conditions EQUIVALENT at every spp.

| ROI | Largest upper bound / limit |
|---|---|
| primary | 0.036 |
| SSS interior | 0.040 |
| silhouette | 0.026 |
| material boundary | **0.138** (H4, 16 spp) |
| control | 0.048 |

  - Largest |point| / limit: 0.024 (material boundary).

- **WL109's two failed ROIs, re-measured.**

| ROI | XJL upper bounds (4 / 16 / 32 spp) | Fraction of limit 0.01 | WL109-design Q upper bounds on the same data (descriptive) |
|---|---|---|---|
| H3 material boundary | 0.00083 / 0.00062 / 0.00066 | ≤ 0.083 × | 0.00076 / < 0 / 0.00043 |
| H4 material boundary | 0.00097 / 0.00138 / 0.00097 | ≤ 0.138 × | 0.00077 / 0.00050 / 0.00016 |

  - In WL109 these ROIs had upper bounds of 0.0136 and 0.0268.
  - Over all 84 settings, the WL109-design bound on the new data is ≤ 0.077 × limit.

- **B1:** 28 / 28 EQUIVALENT. The widest half-interval is 5.1e-4 (H3 material boundary).
  - One interval excludes zero: H1 primary, −9.3e-6, CI [−1.7e-5, −1.6e-6].
  - That is about 1/300 of the ±0.5 % band, so it is a resolved but negligible offset.

**Stage F (cost-matched static).** Equal-time MSE ratio, C1 / R0 at C1's frame time. GM over six views, 95 % CI.

| Gate | Rule | Value |
|---|---|---|
| P2 | primary 4 spp ≤ 0.85 | **0.789** |
| P3 | its upper95 < 1.0 | [0.751, 0.828] |
| P5 | primary 16 / 32 spp non-regression | 0.791 [0.756, 0.828] / 0.795 [0.760, 0.832] |
| P6 | silhouette 4–32 spp | 0.844–0.851 (upper95 ≤ 0.902) |
| P6 | SSS interior 4–32 spp | 0.794–0.799 (upper95 ≤ 0.835) |
| Reference precision | | max reference noise / arm RMSE 0.0107 |
| Grid bracketing | | interpolation flag `ok` everywhere |

- **Report only.**
  - Material boundary 0.823–0.853 (CI up to 0.929).
  - Contact 0.760–0.764.
  - Equal-spp primary 0.737 [0.694, 0.784].

- **Per view, primary 4 spp:**

| View | Equal-time ratio | C1 / R0 frame time (ms) | Cost ratio |
|---|---:|---|---:|
| V0 | 0.729 | 12.15 / 10.85 | 1.120 |
| H1 | 0.797 | 14.21 / 13.09 | 1.086 |
| H2 | 0.738 | 19.77 / 19.24 | 1.028 |
| H3 | 0.865 | 26.58 / 24.98 | 1.064 |
| H4 | 0.810 | 22.77 / 22.51 | 1.012 |
| H5 | 0.801 | 15.58 / 14.48 | 1.076 |

- Peak Dr.Jit watermark: 36.09–36.12 MB over all slots. C1 is at most 25 kB above R0 in the same view.

**Stage G (P7 edit: WL100 SSS configuration 0 → 2, light 0, view V0).**

| Check | Result |
|---|---|
| D1 | selective = full = fresh, bitwise; the table changed from static as expected |
| D2 | both replays bitwise; frozen versus eager differences 0 |
| D3 | relative L1 1.2e-8 (not bitwise; within the frozen tolerance); pass |
| D4 Q | EQUIVALENT in all five ROIs; largest upper bound 0.00040 (silhouette, limit 0.02) and ≤ 0.00037 elsewhere (limit 0.01) |
| D4 B1 | EQUIVALENT in all five ROIs |
| D5 | equal-time ratio **0.745**, CI [0.741, 0.748]; equal-spp 0.656; cost ratio 1.121 |

D5 is measured against a fresh 16 × 8,192-spp reference.

D6 costs:

| Cost | Value |
|---|---|
| Update | 14.1 ms total: geometry 3.2, state 4.0, light 0.13, source 1.0, response 5.7 ms |
| Dirty rays | 85,844 (objects A and B moved) |
| First frame after edit | C1 13.4 ms; R0 4 spp 11.0 ms |
| Steady state | C1 12.67 ms; R0 4 spp 11.30 ms |
| Peak watermark | 37.16 MB for both |
| C1 persistent state | none |

### Observation

- **The variance gain matches WL109.**
  - WL109 measured P2 0.792 [0.763, 0.822] on LabServer69 with an exclusive host.
  - WL109B measures 0.789 [0.751, 0.828] on LabServer70 with fresh grid and timing.
  - The per-view ratios agree to within 0.04 (largest difference H3: 0.829 → 0.865).
- **Frame times are slightly higher than on LabServer69.** V0 is 12.15 / 10.85 ms here against 11.74 / 10.43 ms there.
  - The cost ratio is almost unchanged (1.120 against 1.125).
  - Both arms were measured while the RTX 3080 Ti ran foreign training.
- **The bias maps show no spatial structure** (`evaluation/109B/bias_maps.png`). The per-pixel z of mean D looks like noise in
  every view, including the H3 / H4 material boundaries.
- **XJL is conservative.** At Q = 0 its upper bounds sit at 0.02–0.14 × limit. The WL109-design bound on the same data is
  mostly lower, but that bound is the anticonservative one.
- **The dynamic gain is larger than the static GM.** D5 (0.745) is below the static GM (0.789), and close to the static V0
  value (0.729) for the same view before the edit.

### Interpretation

- **Spatial bias.** WL109's NOT_SUPPORTED came from its B2 gate: two material-boundary ROIs failed under an 8-group
  Student-t bound. WL109A showed that this bound under-covers when noise is heteroscedastic.
  - With a procedure that qualified on synthetic and real-noise gates *before* any C1 frame, the independent R = 512 data
    place every bias upper bound at or below 0.14 × the original limits.
  - So the WL109 failure is best read as a property of the bound, not as evidence of a spatial bias in frozen C1.
  - This is an equivalence statement: the spatial bias power is below the tolerance (Q ≤ limit). It is not a proof that
    the bias is zero.
- **Cost-matched variance reduction.** The reduction is about 21 % of TOTAL MSE at equal GPU time on six static views, and
  about 25 % on one edited view. It holds at 4, 16 and 32 spp and in every region class.
- **Dynamic closure.**
  - Under the ordinary ECRO update, C1 needs no state of its own.
  - It stays bitwise consistent with full and fresh rebuilds (D1).
  - It keeps the equivalence after the edit (D4) and the benefit after the edit (D5).
- **Scope.** All of this is measured for one scene family, one table and checkpoint, one edit and one dynamic view. It says
  nothing about other scenes, other edits or other estimators.

### Unresolved

- **Generality.** There is one scene family, one frozen table and one P7 edit, and the dynamic stage has one view. The gain
  depends on the LOW–HIGH covariance of this frozen estimator (as in WL109).
- **The D5 interval is narrow** ([0.741, 0.748]). It resamples frames, grid and timing for that one view and edit only.
  Variation across views or edits is not in it.
- **Host conditions for timing.** Timing and dynamic ran with the RTX 3090 exclusive but with foreign training on the RTX
  3080 Ti and load 2–3 (Amendments 7 and 8). The C1 / R0 cost ratios match WL109's exclusive-host values closely, but
  differential interference was not measured.
- **Conservatism of XJL.** Its precision cost in the synthetic stress families (worst-case linear term) did not matter on
  real noise. A less conservative calibrated procedure was not studied.
- **The contact ROI** is reported for variance only. It is not one of the frozen bias ROIs.
- **The LabServer70 GPU failure** (2026-11-25 17:57 → reboot 11-26 17:53) has an unknown cause. No WL109B GPU work was
  running at the time.
- **Untested.** A denoiser, temporal reuse and production integration were not tested and are not authorised.

### Architecture Verdict

**QUALIFIED_STATIC_AND_DYNAMIC**, by the frozen protocol mapping:
- Stage D EQUIVALENT_ALL;
- Stage F STATIC_PASS (P2, P3, P5, P6, reference precision, grid bracketing);
- Stage G with D1–D3 pass, D4 EQUIVALENT and D5 pass.

What the verdict covers:
- Frozen ECRO C1 (M = 1) preserves the expected image within the original WL109 spatial and mean tolerances.
- It lowers TOTAL MSE at equal GPU cost, statically on six views and after the P7 edit on one view.
- This holds for this scene, table and edit.

What it does not change:
- WL109 stays **NOT_SUPPORTED** and WL109A stays **INCONCLUSIVE**; their records are unchanged.
- This verdict is new evidence from a calibrated procedure, not a rewrite of either.
- No production adoption, deployment or successor batch follows from this report.

### Reproducibility

**Code and amendments**

| Item | Commit |
|---|---|
| Protocol and code frozen before evidence | `cc8fb24` |
| Amendment 1 | `d5fc566` |
| Amendment 2 | `5c4bdeb` |
| Amendment 3 | `88fb2e7` |
| Amendment 4 (withdrawn) | `98abe2b` |
| Amendment 5 | `c77cb86` |
| Amendment 5 addendum | `83c45d3` |
| Amendment 6 | `d2e255e` |
| Amendment 7 | `2390989`, `09418b2`, `a39da83` |
| Amendment 8 | `2504147` |

The final stages ran from `run109b_amend8.sh` at `2504147`.

**Protocol.** `experiments/ecro_estimator109b/protocol109b.json`, sha256 `ba1f19a63fc4…`.

**Environment**
- LabServer69: native frozen stack.
- LabServer70: Docker image `sha256:6fd6ebbd05fc…` (python:3.10-bookworm).
- Driver 550.54.15 on both, Dr.Jit 1.3.1 / Mitsuba 3.8.0. The frozen WL108 admission passed on both hosts.

**Seeds.** As listed above. Every family is disjoint from WL101–WL109A and from the development seeds (`tests109b.py`).

**Evidence**
- Server: `LabServer70:~/rest_sss101a/results/ecro_estimator109b/evidence/`. Stage A–C files were produced on LabServer69.
- Local mirror without the large arrays: `results/ecro_estimator109b/evidence/`. The paired `.npy` (9.7 GB) and the dynamic
  accumulators (409 MB) stay on the server.

**Reviewer package.** `evaluation/109B/`, with `manifest109b.json` (sha256 per file). It contains:
- figures: `calibration.png`, `bias_bounds.png`, `equal_time.png`, `bias_maps.png`;
- decision JSONs and the six timing records;
- chain state, guard logs and the 10 s GPU-occupancy records.

The figures come from `experiments/ecro_estimator109b/export109b.py`, which runs on CPU and reads only finished outputs.
