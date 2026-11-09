# Worklog — Compiled Neural Material Rules: Learned Representation and Rendering Utility Validation (2026-11-09)

## Origin

- **Brief.** WL81, "Compiled Neural Material Rules: Learned Representation and Rendering Utility Validation", 2026-11-09.
- **User decisions, 2026-11-09:**
  - Option 2.
  - The server is idle.
  - Reuse the WL80D sampling fix, but confirm it with focused regression.
  - The WL80D energy candidate is **not adopted**, because a grazing-limit passivity violation was confirmed.
  - The new candidate is a post-WL80D revision, fixed by a protocol amendment before its first evaluation and judged on independent data.
  - No learned-compiler training unless the physical contract and representation gate pass.
  - Remove duplicate implementation.
  - No hard time or hardware budget.
- **Where.** Branch `Selective_Recompute` on LabServer63 (main server), RTX 3080 Ti 12 GB. Mitsuba 3.5.1, Dr.Jit 0.4.6, torch 2.3.1.
- **Isolated paths.**
  - `experiments/fnto_compiled81/`
  - `results/fnto_compiled81/`
  - `docs/plans/wl81/`
  - server `~/rest_compiled81` (dev runs: `~/rest_compiled81_dev`)
- **Preserved evidence.** WL80C (`9f5b388`, `d219cf9`) and WL80D (`679657b`, `c3577be`) code and evidence are untouched. Their modules are imported, not copied.

## Pre-freeze work (reference selection and debugging; not evidence for any candidate)

1. **Reference implementation.** `layered81.py` is a coated-conductor layered BSDF:
   - a Dr.Jit port of pbrt-v4 `LayeredBxDF::f` and `Sample_f`;
   - per-lane ports of the Mitsuba 3.5.1 rough dielectric and rough conductor interfaces.
2. **Interface audit (dev, `iface`).**
   - 65,536 random pairs × 18 dielectric settings (IOR / α × radiance / importance × component masks) + 3 conductor settings.
   - eval and sample are bit-identical to the plugins; pdf max rel 8.1e-5.
   - One test-harness correction: Mitsuba leaves `bs.pdf` unmasked on rejected conductor samples, so sample validity = nonzero weight.
3. **Walk audit (dev, `walk`), first run.**
   - Sampled albedo vs eval-integrated albedo disagreed: z up to 19, about 5 %.
   - Reciprocity failed: z up to 79.
   - **Cause found.** pbrt-v4 `LayeredBxDF::f` MIS-weights its two exit strategies with `exitInterface.PDF(-w, wi)`. That is the density of the reverse query, not the density with which the NEE strategy generates the inside direction, so the weights do not sum to one.
   - **Correction.** Use `diel_pdf(wi, -w, transmission)`.
4. **Walk audit, after the correction (`walk_b`).** All |z| ≤ 2.6.
   - max_depth 64 truncates up to 1.2 % for a near-white thick slab, so max_depth is frozen at 1024.
5. **Independent slab (dev, `slab`).** Mitsuba volpath (llvm) on a geometric slab: rough dielectric top, interior homogeneous medium, roughconductor floor, directional light, radiance meter.
   - 6 configurations × 3 channels: max |z| 2.15, max |rel| 1.0 %.
   - llvm was used because LabServer63's cuda film is defective (WL53).
6. **WL80C / WL80D horizon defect (found here).**
   - torch `closure80c.eval_closure`: a single GGX lobe of weight 1 has directional albedo 3.15 at μ = 1e-4, 31.4 at 1e-5 and 315 at 1e-6 (α 0.05 and 0.3, F0 1, 2^22 samples).
   - Cause: Λ clamps cos² θ ≥ 1e-7, but the division by cos θ_i is unclamped.
   - Fix: algebraically identical clamp-free closed forms (`closure81.py`, `closure81_dr.py`), the same for every arm. Stage B6 tests it.
7. **Dev smoke of the chain** (`--dev` populations of 16–64 materials; non-evidence; quality numbers not interpreted).
   - Reference noise floor of the dev gate set at 4096 walks: E_L1 median 0.006.
   - Highlight peak half-vs-half: 0.52° at 256 walks and 0.47° at 1024 walks. That is mainly peak quantisation on the 201-grid, not walk noise. `highlight_grid` walks were set to 1024 before the freeze.
   - Renderer validation: Lambert / point light exact to 6e-7, closure plane 8e-7, layered plane |z| ≤ 3.35.

## Freeze

- **`e4b471e` — protocol v1 and all WL81 code.**
  - Pre-freeze changes prompted by the dev smoke (none of them followed any candidate result):
    - `highlight_grid` walks 256 → 1024;
    - C5b supplement on fixed synthetic closures, because the WL80C C5b eligibility set was empty in dev, so the test could not be evaluated;
    - G81 sampling moved to GPU PCG32 / `block_sum`;
    - kernel-history flag enabled in bench / render;
    - Stage C → D rule: train both arms when one qualifies; only a qualified arm can be primary;
    - Stage E precondition;
    - compile-latency repetitions for 1024 materials.
- **`2ac5aed` — amendment 1 (energy candidate `C81_limit_complete_sup`).** Committed before any candidate evaluation on WL81 data. Provenance: post-WL80D hypothesis revision.
- **Server.** `~/rest_compiled81/code` = `git -c core.autocrlf=false archive` of `2ac5aed`.
- **Chain.** One detached chain, `run81.sh`, started 2026-11-09 09:16 UTC; status in `~/rest_compiled81/chain.status`.
  - Order: A0 audits → B regression → A1 populations and targets → C energy A/B → D learned compiler → E GPU → compression → F rendering.
  - Every gate stops downstream stages by the frozen rules.
