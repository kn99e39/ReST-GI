# Worklog — Architecture Batch 6: FNTO v0.1 Implementation and Physical-Edit Validation (2026-10-23)

## SESSION QUESTION

User brief "ReST-GI — Architecture Batch 6, FNTO v0.1 Initial Implementation & Physical-Edit Validation" (2026-10-23):

> Can a compressed learned intrinsic transport operator and current-geometry free-space coupling jointly preserve
> complex-material fidelity while recovering geometry-dependent interreflection without retraining reusable learned
> parameters?

**Verdict: NOT SUPPORTED** (predeclared protocol `experiments/fnto_lego/protocol/fnto_v1.json`, amendment 1).

The composed state solve K∘C diverged on **every** held-out view: 64 of 64 view solves, T0 and T1, all arms, after
5–6 Jacobi iterations. Static representability and edit recovery are therefore unmeasurable. Both protocol rules fire:
"divergence on any held-out view" and "static fail".

**What the batch nevertheless shows:**
- **Coupling side.** The current-geometry free-space coupling behaves as designed: exact incremental relation update,
  same-asset exposed paths, and an occluder changing an untouched pair. This holds in the contracts and on the Lego
  relation tables.
- **Kernel side.** The learned intrinsic kernel reproduces first-event transport only coarsely: 22.0 dB vs first-event
  references on held-out T0, 23.7 dB on held-out Stage A probes, against a ≈34 dB noise floor.
- **The cause.** The learned kernel's nonlocal parts are non-passive for incident fields unlike its sun-probe training
  fields: K_mid on 26 % of ports, K_glob on 18 %.

## RELATION TO EARLIER WORK

**Contract.** `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md`, authoritative and unchanged. The implementation mapping is
`experiments/fnto_lego/IMPLEMENTATION_DESIGN.md`, declared before results.

**Historical arms, reused unchanged:**
- RNA_T0 / RNA_T1, Surface-Owned A_full / B_sel / SO_T0 (worklogs 37–38);
- SPT-1 (worklog 39, NOT SUPPORTED; not modified, not used as a starting implementation).

**Shared evaluation regime.** Same held-out T0/T1 sets, display transform and regions as worklogs 37–39.

**Code and data locations:**
- code: `experiments/fnto_lego/`;
- outputs: `results/fnto_lego/`;
- exports: `results/evaluation/41/`.

**Commits:**
- `cc1cc9f`: implementation and declaration, before main training and before any held-out evaluation;
- `4dc186a`, `55715e2`, `4ea5638`: execution-only amendment 1;
- the result commit that adds this worklog.

The worklog-40 architecture review (same day) is a separate session record.

## A. IMPLEMENTED REPRESENTATION

**Reference-renderer semantics, measured first.** Source: `fnto_semantics.py`, `results/fnto_lego/semantics/`.

| question | measurement | consequence |
|---|---|---|
| Burley SSS across disconnected parts of one object | a shadowed yellow slab 2 cm from a lit one glows (0.065, 0.0055, 0.00002) | the Lego's SSS response is object-scoped |
| across separate objects | 0.0 | the box can neither feed nor block it |
| entering through another material of the same object | identical glow from a lit Gray slab | entry is material-agnostic, exit only at SSS surfaces |
| SSS counted as a bounce | no | |
| first intrinsic event | `max 1, diffuse 0, glossy 0, transmission 1` keeps through-glass sunlight (4.008 = full) and hides a lit plane behind glass (0.0316 = surface reflection only) | Stage A renders = K applied to a known incident field |
| free-space share on the isolated Lego | yellow red 0.52 first event vs 0.92 full | C must carry ≈44 % of yellow's red radiance |

**Optical domains (contract §2).**
- A = the Lego object. Its intrinsic part covers local BSDF events, object-wide Burley and glass-solid traversal.
- B = the red box.
- Every free-flight segment is in C, including Lego→Lego segments (a = b).

**Ports.**
- 463,614 mesh-complete ports on A and 1,702 on B, built from 1.2 cm clusters by (voxel, signed normal axis, material).
- Open components are two-sided.
- 38,090 coarse 4 cm ports serve the mid-range kernel.

**Kernel K_a (per domain, owned parameters).** A has 6,531,047 parameters and B 202,335. The kernel is linear in
radiance with all coefficients ≥ 0:
- **K_loc:** learned diffuse + 2 GGX lobes (α ≥ 0.01). The delta sun is exact; indirect light enters per gather
  direction.
- **K_glob:** rank 64 per colour; non-negative p·q; cosine entry, isotropic exit.
- **K_band:** 32 ports within 5 cm, 19 hemispherical source bins, w_o-dependent.
- **K_mid:** 64 coarse ports within 30 cm.

**Coupling C.**
- Ray-cast relations: 128 cosine gathers per port, 59.6 M rays. Hits map to the facing-side cluster; inside faces of
  closed solids are blocked; misses are dark.
- An incremental update after a rigid move re-traces only rays touching the moved domain's old or new bounds.
- **The learned coupling rule C_φ is empty in this implementation** (declared), so contract §6 question 3 is not
  addressed.

**State solve.**
- Jacobi over the per-port incident samples; stop at a relative change < 1e-3, at most 100 iterations; divergence is
  detected.
- Sender outgoing state = isotropic part + exact delta-sun glossy. These are declared approximations.

**Training supervision (Stage A).**
- Isolated-domain first-event probe renders with the reference renderer: 1,400 train, 120 held out, 12 superposition
  triples. All at 512², 512 spp, PC CPU, 3.3 h in total.
- Stimuli:
  - one sun per image, 70 % from the upper and 30 % from the lower hemisphere;
  - 0–10 shadow-only mask spheres;
  - random box placements, never within 0.25 m of T1.
- The incident field of every image is exact (traced port visibility).
- No full-scene image, no free-space interreflection and nothing of T1 entered training.
- 24,000 steps on LabServer63: 51.7 min, peak 5.4 GiB, final loss 0.0017.

## B. SYNTHETIC CONTRACTS (production code; `tests/test_fnto_contracts.py`, `results/fnto_lego/contracts.json`)

| contract | result |
|---|---|
| A local scattering | PASS: analytic BSDF reproduced to 9e-7; response follows the incident direction; parameters bit-identical |
| B nonlocal internal (trained) | PASS: unlit side of a half-lit slab, energy-weighted error 0.112 |
| C transmission (trained) | PASS: top-face exit from bottom-face entry, error 0.165; forward lobe 0.85×, side 1.40× (directional smoothing) |
| D directional sharp | PASS: GGX α 0.09 exact on the delta-sun path (2.7e-5); an SH2 bottleneck of the same lobe errs 94 %; indirect glossy is limited by 128 gather directions (24 % error for a 5° patch; declared) |
| E editable exposed interreflection, one domain | PASS: an occluder between wall and floor of the same domain; updated single bounce within 1.9 % / 0.9 % of a dense reference; stale 37 % off; kernel digest unchanged |
| F linearity / zero / multi-bounce | PASS: superposition 1.4e-7, scaling by −2.5 6.2e-7, zero → 0; a closed ρ = 0.8 cube converges to ρ/(1−ρ) within 0.14 % (rate 0.76) |

**Development history before declaration** (kept in the test file):
- **B, first run:** failed with 70–140 % errors at 3–25 cm. This exposed a gap between K_band and the low-rank K_glob,
  which **K_mid** now fills. The criterion was then changed from per-channel to energy-weighted, because the blue
  reference is ≈0.
- **C, first run:** failed at 0.223. The model was under-fitted (loss still falling), and steps were raised from 1,500
  to 4,000.

## C. REAL-SCENE RESULTS

**Full composition (decisive arms; `results/fnto_lego/eval_main/eval.json`, `verdict.json`).** Every view solve of
S_T0, D_stale, E_upd and E_full diverged: 64/64, non-finite after 5–6 iterations. Kernel parameters were bit-identical
across all arms. Rule values: S, the material screen, F, R_RNA and W are all unmeasurable (NaN).

**Kernel alone, coupling switched off** (declared first-event decomposition; references are `fnto_firstevent_render.py`
at 1,024 spp, same cameras and suns):

| | held-out T0 | held-out T1 |
|---|---|---|
| K[sun] vs first-event reference, display PSNR | 22.04 dB | 22.19 dB |
| predicted / reference first-event energy | 1.14 | 1.14 |
| reference remainder share (full − first) | 25.4 % | 25.4 % |

**Per-material first-event display MSE of K on held-out T0.**
- Yellow is lowest at 0.0027 (B 0.0015).
- Reflective materials are worse: Black 0.0139, Gray 0.0085, Sand 0.0081, Brown 0.0071, Light_brown 0.0065.
- Glass: transparent 0.0078, Red_Glass 0.0130.

**Stage A held-out probes** (`eval_main/stagea/stagea_eval.json`):
- pooled first-event display PSNR 23.7 dB (A 23.6, B 29.3);
- shadow-side yellow MSE 0.0080.

The holdout monitor plateaued from step 8,000 (23.4 → 23.9 dB). The noise floor of these 512-spp probes is ≈34 dB:
Cycles(two suns) vs Cycles(sun 1) + Cycles(sun 2) gives 33.9 dB across three independent renders. The kernel error
therefore sits ≈10 dB above render noise.

**Superposition probes:**
- operator superposition holds to 1.8e-7;
- the renderer itself departs by 8.5 % rel L1 (MC noise and the direct clamp);
- FNTO on two-sun fields: 26.0 dB.

**Comparison with historical arms** (descriptive). On held-out T0 the historical arms score RNA_T0 33.57 dB, SO_T0-PU
33.24 dB and SPT-1 22.23 dB. FNTO has no full-scene image to compare.

## D. EFFICIENCY (measured; another user's process occupied the PC GPU during PC-side runs)

| item | cost |
|---|---|
| Stage A probe renders (PC CPU, 32 threads) | train 10,384 s, holdout 939 s, superposition 286 s |
| Stage A prep (traced port visibility per image) | train 759 s, holdout 65 s |
| kernel training (RTX 3080 Ti, one-off) | 3,105 s, 24 k steps, peak 5.4 GiB |
| kernel tables | 1.2 s |
| relation full build per state (RTX 5080, Mitsuba) | 21 s for 59.6 M rays |
| incremental T0→T1 relation update | 0.83 % of rays selected (494 k; 31 k rows): selection 11.3 s (CPU, numpy), trace + map 1.9 s; equal to a full rebuild except 615 ties (identical hit distances) |
| port sun visibility per view | 24–30 ms |
| solve per view | ≈0.2 s per iteration (server); diverged after 5–6 |
| pixel render per view (≈250 k px) | 0.44 s |
| learned parameters / dynamic state | 26.9 MB (fp32) / 357 MB (59.6 M incident samples × fp16 RGB) |

No end-to-end edit-to-render latency is reported, because no update produced a valid image.

## E. ARCHITECTURE ATTRIBUTION (post-result diagnostics, descriptive; `results/fnto_lego/diag/diag.json`)

**Multi-bounce convergence → caused by learned-kernel non-passivity, not by the solver.** The production solver
converges for a passive kernel (contract F). Measured passivity of the trained kernel:

**D1, effective albedo per port under a uniform unit incident field:**

| component | ports > 1 | Yellow ports > 1 | q99 | max |
|---|---|---|---|---|
| K_loc | 0.8 % | | 0.93 | 4.8 |
| K_band | 0.01 % | | 0.56 | 1.4 |
| K_glob | 17.8 % | 41 % | 7.9 | 33 |
| K_mid | 26.4 % | 64 % | 13.0 | 47 |
| sum | 36.7 % | | | |

**D2, growth of incident energy per Jacobi iteration on the real T0 coupling graph:**
- K_loc alone: ≈1.00 for 7 iterations, then rising to 1.55 (a few high-albedo loops);
- K_loc + K_glob: stable at 2.09 (spectral radius ≈ 2.1);
- any subset containing K_mid: 3.8 → 12 → non-finite.

**D4, one free-space bounce only.** The predicted first-bounce increment carries 4.17 M energy units against the
reference's entire remainder of 0.76 M on held-out T0: 5.5× too much after a single bounce.

**Spatial incident-light identifiability → supported as the mechanism (D3).**
- Amplifying ports are not just the never-excited ones. 36 % of the 313,664 ports never directly lit in Stage A have
  albedo > 1, and so do 32 % of ports lit in more than 100 training images.
- The kernel is linear, so it is identified only on the span of its training incident fields: 1,400 sun-coherent
  fields, each lighting one side of the object.
- The spatially broad, multi-sided indirect fields that C delivers lie largely outside that span. There the
  unconstrained non-negative source weights of K_mid and K_glob amplify instead of conserving energy.
- Linearity was imposed and held (F1, superposition probes), but linearity does not imply passivity outside the
  excited span.

**Intrinsic kernel expressivity → a second, independent shortfall.**
- K reproduces even the excited first-event regime only coarsely: 22.0 dB vs first-event references, 23.7 dB on
  held-out probes, ≈10 dB above render noise, with +14 % energy.
- Reflective materials (Black, Gray, Sand) are worse than yellow in this regime.
- The cause is not isolated. Candidates: K_loc parameterisation on bump- and scratch-detailed surfaces, sun visibility
  at pixel level, and optimisation (the monitor plateaued from step 8,000).

**Path-family ownership, coupling coverage and accuracy.**
- **Not implicated.** The K/C path partition is exact by construction: first-event training data versus explicit free
  flight.
- **Coupling semantics** passed contract E, including the same-domain occluder case.
- **Lego relations:** hit→port q99 9 mm; inside-solid blocking; incremental update tie-equivalent.
- **Not measured on the Lego:** coupling *accuracy*, because no composed image exists.

**Directional bandwidth.** It did not decide this outcome. The delta sun is exact; indirect glossy is limited by 128
gathers and by the sender-side isotropic folding (declared). Contracts C and D quantify the limits.

**Runtime cost.** It did not decide this outcome. Relation update and visibility take seconds and milliseconds; the
incremental selection step (11.3 s on CPU) dominates.

## F. ARCHITECTURE VERDICT

**NOT SUPPORTED.** The implemented FNTO v0.1 cannot produce a full-GI result under its own reuse contract: the
composition of the learned intrinsic operator with current-geometry free-space coupling diverges on every held-out
view.

Neither part of the decisive question was answered positively:
- **Complex-material fidelity:** unmeasurable for the composition; the kernel alone gives 22 dB on first-event
  references.
- **Edit recovery without retraining:** unmeasurable.

Parameters stayed bit-identical, and the relation update behaved as designed. This is reported as such, not as a
partial success of the architecture.

## G. ARCHITECTURE CONCLUSION SUPPORTED BY THIS BATCH

**The decisive missing property is energy behaviour, not linearity.** A learned intrinsic operator for this kind of
factorization must be **passive (energy-bounded) on the incident fields the coupling actually produces**, not only
accurate on its training stimuli. Two facts from this batch support this:
- isolated-domain probes with one sun and masks identified a linear operator only on a span that excludes the
  indirect fields of the composed scene;
- the nonlocal components amplified those fields, ρ(KC) ≈ 2 for K_glob and much larger with K_mid.

The contract v0.1 lists positivity, passivity and energy bounds as "require tests". In this batch they were the
failing property, and the K/C composition could not be evaluated without them.

Open questions, not authorised and not implemented:
- whether an intrinsic operator with an enforced energy bound, and/or training stimuli that span indirect-like
  incident fields, keeps first-event fidelity;
- whether the ≈10 dB first-event shortfall of K is representational or optimisational.

Worklog 39 (SPT-1) is not reinterpreted. No later batch is started.
