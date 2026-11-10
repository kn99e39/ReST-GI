"""WL83 torch numerics patch for closure82 (WL82 code is imported, never modified).

Refraction into the coat, t(w) = (w_x / eta, w_y / eta, t_z). closure82.refract_up computes t_z = sqrt(1 - (w_x^2 + w_y^2) / eta^2), which
cancels catastrophically in float32 near the horizon when eta_e -> 1 (the radicand rounds to 0; the refracted-lobe pdf Jacobian
cos_o / (eta^2 cos_o') then explodes; WL83 dev smoke, synthetic closure with eta = 1: chi-square expected counts ~1e26). Algebraically
identical, cancellation-free form (w_x^2 + w_y^2 = 1 - w_z^2):  t_z = sqrt(max(eta^2 - 1, 0) + w_z^2) / eta.
install() replaces closure82.refract_up in every WL83 process (training, direct fits, evaluation, tests) unless W83_UNPATCHED is set; the
Dr.Jit runtime closure83_dr uses the same form. Stage A re-evaluates the stored WL82 fits UNPATCHED (faithful reproduction).
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
import closure82 as CL  # noqa: E402

ORIGINAL_REFRACT_UP = CL.refract_up


def refract_up_robust(w, eta):
    tx, ty = w[:, 0] / eta, w[:, 1] / eta
    tz = torch.sqrt((eta * eta - 1).clamp_min(0) + w[:, 2].clamp_min(0) ** 2) / eta
    return torch.stack([tx, ty, tz], 1)


def install():
    if not os.environ.get('W83_UNPATCHED'):
        CL.refract_up = refract_up_robust
    return CL


install()
