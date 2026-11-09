# WL82 candidate mechanisms (not adopted)

Design record for WL82, "Incident-dependent Energy vs. Refraction-conditioned Angular Representation". These are isolated, unadopted representation mechanisms. They are not part of the canonical FNTO architecture.

- **Protocol (authoritative):** `experiments/fnto_compiled82/protocol/compiled82.json`
- **Code:** `experiments/fnto_compiled82/` (imports WL81 `fnto_compiled81`, unchanged)
- **Worklog:** `docs/worklogs/82_WORKLOG_INCIDENT_ENERGY_VS_REFRACTION_ANGULAR.md`

## Mechanisms

Every arm uses the same R = 8 lobes: one Lambert lobe and seven GGX lobes. The lobes are split into two fixed groups:

- inner group: the Lambert lobe and GGX lobes 4–7;
- outer group: GGX lobes 1–3.

| Arm | Definition | Status |
|---|---|---|
| E | Symmetric separable attenuation s(μ_i)·s(μ_o) of the inner group. s is piecewise linear in μ: 5 knots per channel, values in [0, 1]. | Physically admissible (certificate under the softmax energy mode). |
| Einc | Incident-only scale s(μ_i) applied to the whole WL81 closure. | Diagnostic only: not reciprocal. |
| ER | E plus the inner GGX lobes evaluated at coat-refracted directions with the η⁻² solid-angle Jacobian. One fitted η_e ∈ [1, 2.5]. ER with η_e = 1 is exactly E. | Physically admissible (certificate under the softmax energy mode); reciprocal by construction. |

None of these mechanisms is part of the canonical FNTO architecture. Whether any of them is adopted is a user decision.
