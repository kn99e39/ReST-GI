# Worklog — Incident-dependent Energy vs. Refraction-conditioned Angular Representation (2026-11-10)

## Origin

- **Brief.** WL82, "Incident-dependent Energy vs. Refraction-conditioned Angular Representation" (user, 2026-11-10). A bounded representation-mechanism experiment that attributes WL81's representation failure to H1 (incident-dependent energy) or H2 (refraction-conditioned angular shape). Not a production FNTO implementation; no learned compiler; no successor authorised.
- **Where.** Branch `Selective_Recompute` (no new branch / worktree), LabServer63 (main server, idle at start), RTX 3080 Ti 12 GB, Mitsuba 3.5.1 / Dr.Jit 0.4.6 / torch 2.3.1 (WL81 environment, `~/rest_compiled81/env.sh`).
- **Isolated paths.** `experiments/fnto_compiled82/`, `results/fnto_compiled82/`, `results/evaluation/82/`, `docs/plans/wl82/`; server `~/rest_compiled82` (dev smoke `~/rest_compiled82_dev`). WL80C / WL80D / WL81 code, data and evidence are imported or read, never modified.

## Sources read before design

WL80C / WL80D / WL81 worklogs; WL81 protocol v1 + amendments 1–3; `layered81.py`, `closure81.py`, `energy81.py`, `fit81.py`, `eval81.py`, `data81.py`, `contract81.py`, `diag81_posthoc.py`, `viz81.py`; WL81 Stage C results and post hoc JSONs; `results/evaluation/81/`.

## Holdout status (verified before the protocol)

- WL81 generated `train / val / gate / test_iid / test_block / train_eval` in Stage A1. `chain.status` shows no stage after C (D/E/F not reached).
- The post hoc diagnostic and the visuals read only the gate set.
- On LabServer63 the atime of `eval_test_iid.npz` (10:02:34 UTC) is the WL81 highlight generation reading its `phis`. The atimes of `fit_test_iid`, `fit_test_block`, `eval_test_block` and `hl_test_iid` (10:43:11–14 UTC) coincide with the creation of `data_sha256.txt` (10:43:17 UTC), i.e. hashing.
- No fit, candidate, control or visual ever read the holdouts. **test_iid and test_block are the independent evaluation sets of WL82; the gate set is development data** (it was inspected in WL81 and is used here for development and mechanism diagnosis only).

## Design (frozen in `protocol/compiled82.json` before any candidate evaluation)

**Stage A decomposition.** For every incident direction, A(wi) = ∫ f cos dwo and p(wo|wi) = f cos / A.
- E_shape = energy-weighted total variation of p between closure and reference (in [0, 2]). A closure channel with zero albedo counts as TV 1 (no division, no exclusion).
- Oracle (diagnostic only; uses reference albedo at evaluation time, may violate reciprocity): f_or = f_R8 · A_ref^ind / A_R8^ind. The ratio comes from an **independent** albedo set: 4096 fresh wo × 256 fresh walks at every evaluation incidence, with the closure albedo computed on the same independent samples.
- Algebraic identity, checked for every arm: the same-sample oracle has E_L1 = E_shape exactly and E_alb = 0.
- Items:
  - A: the WL81 frozen direct fits (log1p);
  - B: A + oracle;
  - C: shape-focused R = 8 fit (normalised-shape objective, energy free per incidence);
  - D: C + oracle;
  - additionally the WL82-protocol C0 refit + oracle.

**Arms.** All arms keep the WL81 corrected R = 8 lobes. Fixed lobe groups: INNER = Lambert + GGX 4–7, OUTER = GGX 1–3.

| Arm | Definition | Params | State |
|---|---|---|---|
| C0 | WL81 closure | 48 | 53 |
| E | outer lobes + s(μ_i) s(μ_o) · inner lobes; s = per-channel piecewise-linear knots at μ ∈ {0, .15, .4, .7, 1}, values in [0, 1] | 63 | 68 |
| Einc | s(μ_i) · C0 — incident-only, **non-reciprocal diagnostic** (deployable analogue of the oracle) | 63 | 68 |
| ER | E with the inner GGX lobes evaluated at the coat-refracted directions t(w) = (w_x/η, w_y/η, √(1 − (1 − w_z²)/η²)) with the Jacobian η⁻², one fitted η_e ∈ [1, 2.5]; ER(η = 1) ≡ E | 64 | 69 |

- **Energy modes:**
  - cert = WL80C softmax (Σλ ≤ 1);
  - unb = softplus (diagnostic, isolates the energy-constraint confound).
- **Certificate (cert, C0 / E / ER).** Every component albedo is ≤ 1:
  - for the refracted lobe this follows from the solid-angle identity cos θ_o dω_o = η² cos θ_o' dω_o' (the outside albedo is the inside integral over the refraction cone);
  - hence a(wi) ≤ Σλ ≤ 1.
- **Reciprocity** is exact by construction for C0 / E / ER.
- **Grazing.** Horizon-safe forms (WL81) for every lobe; refracted inner directions never graze.
- **H2 specificity.** ER differs from E only by the deterministic refraction remap (one scalar per material). The arms are not exactly capacity-matched: 64 vs 63 parameters and a higher evaluation cost (reported).

**Fitting.**
- New fit sets: 64 wi × 64 wo × 1024 walks per material. This is a deviation from WL81's 64 × 16: per-incidence albedo and shape need many wo per wi.
- Objectives:
  - metric = per-incidence relative L1 + directional-albedo error, equal weights, fixed before any WL82 result from the WL81 post hoc finding about the log1p loss;
  - shape = per-incidence normalised TV.
- Optimiser: the WL81 direct-fit Adam (0.03 → 3e-4 cosine, warm-up 50).
- Steps:
  - 12000 primary;
  - 6000 and a perturbed init on the gate set;
  - predeclared convergence rule |Δ median| ≤ 0.005, with a single escalation to 24000 steps.

**Decision (test_iid).**
- E (invalid): any validity condition fails (reproduction, focused tests, cert contracts, E_shape noise rule, split).
- A: E_cert passes the WL81 gate.
- C: otherwise, if the oracle-corrected R = 8 (C0 or shape fit) reaches R1.
- B: otherwise, if ER_cert passes the gate with ≥ 25 % lower median E_L1 and E_shape than E_cert.
- D: otherwise.
- test_block is a region check (R1 / R2; it has no highlight set).

**Real materials.** 8 predeclared named finishes (gold, copper, chrome, aluminium car paint, candy red over silver, satin brass, milky silver, rough iron) with published metal F0 values. They are fitted and evaluated like the other sets, plus sphere shading images (direct lighting at the surface, no GI).

## Implementation (pre-freeze)

- `closure82.py` (arms, decode, torch sampler / pdf)
- `data82.py` (reproduction, fit sets, independent albedo sets, real set)
- `fit82.py`
- `eval82.py` (decomposition, oracle, highlight, strata)
- `contract82.py` (focused tests T0–T10; physical contract T1–T4)
- `gate82.py` (convergence rule, verdict)
- `viz82.py`
- `run82.sh` (one detached chain, job pool)

WL81 modules are imported unchanged.

## Dev smoke and pre-freeze corrections (non-evidence; dev quality numbers not interpreted)

Dev root `~/rest_compiled82_dev`: 16 materials per WL81 set, 8 real materials, 200 / 400 / 800 steps.

**Run 1–3 results.**
- Stage R reproduced the WL81 `eval_gate` and `fit_gate` prefixes **bit-identically**. Every later stage executed, including the escalation path, the contract, the visuals and the verdict mapping.

**Corrections made before the freeze** (also recorded in `pre_freeze_corrections` of the protocol):
1. **T10 test bug.** The same-sample oracle has E_alb = 0 only where every closure channel has albedo > 0. The synthetic zero-channel group is now excluded from that check.
2. **T5 test design.** The random test closures included near-delta lobes. The uniform-quadrature standard error is unreliable there: the identical sampler code gave C0 z 3.6 against Einc z 12.
   - Random closures are now restricted to sampled lobes with α ≥ 0.1 (the WL80C chi-square eligibility principle).
   - The chi-square threshold is now 1e-4 for 72 tests (multiplicity).
   - Two seeds afterwards: max z ≤ 2.9, chi-square min p ≥ 0.001.
3. **Torch test-sampler pdf.** The dev own-sampler audit showed albedo means up to 1e10 for every arm, including C0, while still passing the formal 4-se rule because se grows with the tail.
   - Cause: `vndf_pdf_wo` required wi·h > 0. That holds mathematically for reflection, but at near-mirror grazing pairs float32 cancellation in wi + wo made it fail. The pdf became 0 where eval was non-zero, and the sample weight reached 3e9.
   - Fix: validity now equals eval's (cos_i, cos_o > 1e-7, |wi + wo| > 1e-6), as in the WL81 Dr.Jit runtime. The eval path was never affected.
   - Afterwards, at full audit sample counts: cert arms max albedo 1.004 with 0 violations; unbounded arms up to 23, mostly adversarial λ (report only).
4. **numpy-bool JSON serialisation** in the test harness.

**Full-scale probe** (300 gate materials, 64 × 64 fit set):
- One fit saturates the GPU: ER 58–63 ms/step, C0 26–32 ms/step, peak 1.6 GB. Four concurrent processes bring no throughput gain, so the chain uses MAXJ = 2.
- `torch.compile` was probed and **rejected**. It is 2.2–2.5× faster, but its forward deviates from eager by about 3 % on the s(μ) arms, and compiled reruns are not bit-identical. All fits are eager.
- Expected main-run wall clock: about 6.5–7 h.
