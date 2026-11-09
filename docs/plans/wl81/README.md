# WL81 plan — Compiled Neural Material Rules: Learned Representation and Rendering Utility Validation

Status: frozen design for one bounded batch. It is not an adopted architecture. The authoritative machine-readable rules are:
- `experiments/fnto_compiled81/protocol/compiled81.json` (v1);
- `experiments/fnto_compiled81/protocol/amendment1_candidate.json` (the energy candidate, fixed before its first evaluation).

## Question

Does the fixed WL80C compiled-lobe closure offer a physically sound, generalizable and computationally useful way to compress complex local material scattering for ReST-GI? The closure is R = 8: one Lambert lobe and seven anisotropic Schlick-GGX lobes, produced by a shared material compiler.

The brief separates six dimensions and keeps them separate here:
1. representation capacity;
2. learned-compiler generalization;
3. physical and sampling correctness;
4. compression;
5. GPU runtime;
6. rendering.

## Reference (Stage A)

The reference is a layered coated conductor:
- a rough dielectric coat;
- a homogeneous isotropic-scattering slab;
- a rough conductor underneath.

It is evaluated by position-free random walks. The code is a Dr.Jit port of pbrt-v4 `LayeredBxDF` (Guo et al. 2018), with per-lane ports of the Mitsuba 3.5.1 `roughdielectric` and `roughconductor` interfaces (`layered81.py`).

Why this family:
- It is physically interpretable and passive.
- It has strong angular structure: an exact-Fresnel coat lobe plus refracted, attenuated multiple scattering.
- It is not a trivial match for the lobe basis.
- It has a real GPU evaluation path.
- One query costs a random walk of tens of interface and medium events, against a single closed-form closure evaluation. That cost gap is measured in Stage E, not assumed.

Qualification before the freeze. The dev runs below are re-run inside the frozen chain:

| Check | What it compares | Result |
|---|---|---|
| Interface ports | Ports vs the Mitsuba plugins | Bit-identical eval / sample; pdf rel 8e-5 |
| Walk consistency (first run) | Sampled vs eval-integrated albedo, and reciprocity | Exposed a bias in the pbrt-v4 MIS weights of the two exit strategies (z up to 19 / 79) |
| Walk consistency (after correction) | Same checks with corrected partner density | All |z| ≤ 2.6 |
| Independent check | Mitsuba volpath geometric slab | 18 / 18 channels within |z| ≤ 2.15, |rel| ≤ 1 % |

The MIS correction uses the NEE density as the partner density.

## Software fixes (separated from representation changes)

1. **WL80D sampler correction.** `closure80d_dr`, imported unchanged. Stage B re-verifies it on the frozen WL80C parameters with the WL80D tests and the WL80C contract.
2. **Horizon numerics (new, found in WL81).** The WL80C and WL80D runtimes clamp cos² θ ≥ 1e-7 inside Smith Λ but divide by the unclamped cos θ_i. A single GGX lobe of weight 1 therefore has directional albedo:

   | θ_i | μ = cos θ_i | Albedo |
   |---|---|---|
   | > 89.98° | 1e-4 | 3.15 |
   | > 89.98° | 1e-6 | 315 |

   So for θ_i > 89.98° every arm, the control included, is non-passive in implementation.

   The fix uses algebraically identical, clamp-free closed forms:
   - eval_r = F D cos_o / (2 (cos_o √(cos_i² + t_i) + cos_i √(cos_o² + t_o)));
   - pdf_r = D / (2 (cos_i + √(cos_i² + t_i))).

   It is applied identically to every arm (`closure81.py`, `closure81_dr.py`). Stage B6 tests it.

## Energy A/B (Stage C)

- **Control:** the WL80C softmax sum bound, Σλ ≤ 1. It is a certificate because every lobe's albedo is ≤ 1.
- **Candidate (amendment 1):** `C81_limit_complete_sup`. This is a post-WL80D hypothesis revision. It keeps softplus weights with f = g / max(1, M), but M is now the supremum over the **closed** hemisphere, including the exact grazing limit μ = 0, computed with the horizon-safe lobe albedo. The margin is 3 se, plus a refinement margin, plus a local-variation margin.
  - It is labelled **empirically passive, not certified**.
  - Its physical validity is decided by the contract, including the grazing-limit check G81, which goes down to μ = 2e-7 with independent samples.

How the comparison is run:
- Direct fits use the WL80C oracle protocol on an independent `gate` set of 300 coated-conductor materials.
- Test sets are never used for arm selection.
- An arm qualifies only if it passes **both** the physical contract **and** the representation gate R1–R3.
- If no arm qualifies, the batch stops before any compiler training.

## Learned compilation (Stage D)

- Both arms are trained with 2 seeds each, using the WL80C compiler architecture with 10 inputs.
- The primary arm is a qualified arm, the candidate first.
- Held-out sets:
  - `test_iid`;
  - `test_block`: sharp coat over a thick slab over a rough base, a joint block that is about 2.5 % of the box;
  - the held-out incident band 55–65°;
  - fresh directions.
- Gates L1–L6 come from WL80C, with their meaning re-checked for this family (see protocol `gates.justification`). Every threshold is at least 4× the measured reference noise floor of its set.
- Descriptive extras: oracle-P and oracle-U ceilings, and material-count scaling at 750 / 1500 / 3000 training materials.

## Utility (Stages E and F)

- **GPU (E):**
  - Arms: the layered reference at one walk per query (the pbrt nSamples = 1 path) and at the matched-quality walk count N_eq; the control and candidate closures; Mitsuba principled as a sanity comparator.
  - Measurements: kernel-history GPU time, io (memory-bound) and fused16 (compute-bound) workloads, compile latency, state bytes, and amortised cost.
- **Compression:** the 53-float compiled state against isotropic reference tabulations at 8³, 16³ and 32³.
- **Rendering (F):**
  - Uses Mitsuba's existing `path` integrator, with the materials as Python BSDF plugins.
  - Scene: point lights, a Lambert floor and wall for limited indirect light, 2 light placements and 2 rigid poses, the first 3 `test_iid` materials.
  - Accumulation is done in the experiment's own code, because LabServer63's cuda film is defective. A renderer validation must pass first.
  - The verdict uses a time-matched comparison against the reference.

## Not authorised

- Rank, basis, loss, population or threshold changes after results.
- Neural correction networks.
- Material filtering.
- Test-set-based selection.
- Production FNTO changes.
- Nonlocal K or dynamic GI.
- Successor batches.
