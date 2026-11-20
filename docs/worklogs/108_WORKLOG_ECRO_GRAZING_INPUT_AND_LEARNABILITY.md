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
