# Worklog 108 — ECRO Grazing-Rim Input and Learnability (2026-11-20)

The owner authorized WL108 on the existing `Denoise_Reconstructor` branch. Its base is
`69c8f23557e83d69890eeb98a9858524c536b20d`, the completed WL107 Amendment 1 assessment.
The owner designated the regular identifier. The working tree and LR3 branch were checked before
implementation. The two factorial variables are observable grazing geometry and the learning
objective. The operator is the original R3 pair of independently normalized nonnegative 15×15
kernels on LOW and HIGH.

## Preregistration

WL103 motivates paired covariance accounting. WL105 supplies R3, the training sampler and 40/8/6
camera split. WL106 supplies the frozen boundary and contact exclusions. WL107 supplies independently
qualified references, target-informed O1 panels and the negative O2 outcome. The WL107 fit proves
a feasible weight assignment on its panels, leaving runtime prediction open.

The [frozen protocol](../../experiments/ecro_denoise108/protocol108.json) specifies A0, A1, B0 and
B1, three paired seeds, thresholds, reference precision, conditionally sealed confirmation cameras,
compute budgets and stops. A0 and A1 receive a neutral zero channel. B0 and B1 receive the
foreground-conditioned mean squared view-normal cosine from the original 8×8 grid. All models
have 1,056,038 parameters, including 414 weights from the symmetric first-convolution extension.
The original initialized weights are copied and each new channel starts at zero.

L0 is the original component loss. L1 adds one term at coefficient 0.25, the relative squared
TOTAL error on mixed sphere-coverage pixels. A per-observation reconstruction loss and the
original full-frame component losses both remain active. No oracle target enters training.
The four arms use the same 20,000 Adam steps, cosine schedule, crop sequence by seed and final-step
selection. Held-out evaluation is run only after all twelve models are frozen.

## LabServer69 execution destination (owner direction, 2026-11-20)

The owner directed WL108 to LabServer69 after preregistration. LabServer63's first preflight
stopped before data qualification because its historical environment was missing the interpreter
and shared objects. An isolated version-pinned recovery reached the six-test suite. The suite
caught platform-dependent float32 promotion in the analytic synthetic positive control, before
qualification and training. Changing that calculation to explicit float64 reproduces the
predeclared target pair and keeps the same numerical assertion tolerance. The old logs are
preserved on LabServer63. No WL108 fit or reference render ran there.

LabServer69 uses the native RTX 3090 / 550.54.15 stack with Torch 2.3.1+cu121, Triton 2.3.1,
Mitsuba 3.8.0, Dr.Jit 1.3.1 and NumPy 1.26.4. An analytic CUDA sphere intersection returned
the known distance of 2.0 using the native OptiX library. The only local library search addition
is an experiment-scoped link to the installed libatomic. The frozen WL105/WL107 inputs are
copied to the WL108 inputs directory, with the full original identity gates applied there.
The historical scripts are read-only. The WL108 loader admits the requested device by validating
the original renderer/config/checkpoint bytes and canonical scene. Stage A uses only the scene
geometry and checks geometry against frozen per-pixel sphere hit counts. Conditional fresh
radiometric confirmation additionally requires exact archived learned-table identity and paired
historical renderer replay within the original WL105 tolerance before any new reference frames.

The original protocol file and sealed camera list retain their hashes. The host override and
synthetic arithmetic fix precede actual qualification or training. Both factors, all thresholds,
the frozen reference, the seeds and the timing budgets retain their original definitions.
