# ER-v1-GF — gauge-fixed parameterization of the certified ER family (candidate, not adopted)

Status: experimental candidate of WL104I (2026-11-20). It is not the canonical ER and not adopted; adoption is a human decision.
Implementation: [`experiments/ecro_wl84_104i/gf104i.py`](../../experiments/ecro_wl84_104i/gf104i.py).

## Original parameterization (certified ER, 64 raw parameters)

Per channel c, the response uses

F = O0 + O1 + O2 + S (L + I0 + I1 + I2 + I3),  S = s_c(μ_i) s_c(μ_o)

with amplitudes λ_c,r for slots r = 0 (Lambert), 1..3 (outer GGX), 4..7 (inner GGX). The raw parameters are:

| Block | Size | Decoding |
|---|---:|---|
| Lobe shape | 21 | z[3k], z[3k+1] log-α logits, z[3k+2] F0 logit, k = 0..6 |
| Amplitudes | 27 | λ_c,0..7 = softmax(z_c,0..8)[0..7]; slot 8 is absorption |
| Attenuation knots | 15 | s_c,j = σ(z_c,j), j = 0..4 at μ = 0, .15, .4, .7, 1 |
| Coat index | 1 | η = 1 + 1.5 σ(z) |

The reachable compiled set is: Σ_r λ_c,r < 1 (strictly), λ > 0, s ∈ (0,1)^5, η ∈ (1, 2.5).

## Exact gauges (WL104H)

1. **Softmax shift**, one per channel: z_c,0..8 → z_c,0..8 + t leaves λ unchanged.
2. **Attenuation scale**, one per channel: s_c → κ s_c and λ_c,r → λ_c,r / κ² for r ∈ {0, 4..7}, with the absorption slot absorbing the change, leaves F unchanged. This is valid while κ s_c,j < 1 and the amplitudes stay in the simplex.

## ER-v1-GF (58 raw parameters)

The idea is to remove the absorption slot and let the attenuation knots carry the absorbed energy of the attenuated block. The Lambert + inner block is pinned to the whole budget left by the outer lobes.

Per channel c:

| Block | Size | Decoding |
|---|---:|---|
| Lobe shape | 21 | unchanged (y[0:21] = z[0:21]) |
| Outer amplitudes | 3 | λ_c,k = e^{a_k} / (1 + Σ e^{a}), k = 1..3; B_c = 1 / (1 + Σ e^{a}) |
| Lambert/inner split | 4 | w_L = 1 / (1 + Σ e^{b}), w_Im = e^{b_m} / (1 + Σ e^{b}), m = 0..3 |
| Block amplitudes | 0 | λ_c,0 = B_c w_L, λ_c,4+m = B_c w_Im |
| Attenuation knots | 5 | s'_c,j = σ(y_c,j) |
| Coat index | 1 (global) | unchanged |

Layout: y[0:21] lobes, y[21:30] outer logits (3 × 3), y[30:42] split logits (3 × 4), y[42:57] knots (3 × 5), y[57] η. The parameter count is 21 + 9 + 12 + 15 + 1 = 58 = 64 − 6.

The additive-log-ratio coordinates (one reference logit fixed at 0) are smooth bijections onto the open simplices. There is no max, no tie and no other non-smooth point anywhere in the forward map.

## Equivalence relation

**Original → candidate** (float64, exact). Let O = Σ_{k=1..3} λ_c,k, B = 1 − O, T = λ_c,0 + Σ_m λ_c,4+m and w = λ_{c,{0,4..7}} / T. Then:

- a_k = log(λ_c,k / B)
- b_m = log(λ_c,4+m / λ_c,0)
- s' = s √(T / B)

Since T < B (the absorption slot is positive), s' < s < 1. Each product S λ_r is preserved term by term: s'_i s'_o B w_r = s_i s_o (T / B) B (λ_r / T) = s_i s_o λ_r. The outer terms are unchanged.

**Gauge invariance.** Both gauges leave (a, b, s') unchanged:
- the softmax shift does not change λ;
- the scale gauge maps T → T / κ² and s → κ s, so s √(T / B) is invariant, and B and w are invariant.

The six gauge directions therefore do not exist in the candidate coordinates.

**Candidate → original** (any section, used only for testing). Choose κ ∈ (max_j s'_j, 1); the tests use κ = (1 + max_j s'_j) / 2. Set:

- λ_c,r = B w_r κ² for the block;
- absorption = B (1 − κ²) > 0;
- s = s' / κ < 1.

This is an interior original state with the same response.

**Response-family coverage.** The two maps show that the sets of physical responses reachable by the original and by ER-v1-GF are identical. Both are open: original Σλ < 1 with s < 1 corresponds to candidate s' < 1.

## Constraints, boundaries and limitations

- **Compiled-state boundary.** The canonical compiled ER-v1-GF state has Σ_r λ_c,r = 1 exactly (no absorption slot). This lies on the closure of, but outside, the original raw-reachable compiled set (Σλ < 1). The physical response is that of interior original states, and the runtime certificate a_c(wi) ≤ Σλ ≤ 1 still holds. The 69-float layout and the runtime evaluator are unchanged.
- **Native proposal.** The compiled sampling probabilities are derived from λ and F0 by the unchanged WL80C rule. In the canonical state, λ of the Lambert/inner block is larger by the factor B / T than in an equivalent original state. Equivalent responses therefore have different native proposals; only fixed-proposal comparisons isolate response.
- **Vanishing block.** As the Lambert/inner block's effective energy goes to 0, s' → 0 and its sigmoid slope vanishes. The original has the same behaviour through λ → 0.
- **Reference slots.** The reference logits (B for the outer group, Lambert for the split) are a coordinate choice. They make the optimizer geometry different from the original even at identical physical starts.
- **What is not removed.** Approximate overlaps (near-duplicate lobes, amplitude–Fresnel and amplitude–roughness near-degeneracies, inner vs attenuation response similarity) are not removed. Discrete lobe-permutation symmetries are not removed either.
- **Certificate.** Passivity holds by the WL82 certificate with Σλ ≤ 1 and s' ∈ (0,1). Reciprocity holds because every term remains symmetric.
