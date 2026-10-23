# FNTO v0.1 — Implementation Design (Batch 6, worklog 41)

Implementation notes for `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` (authoritative; not rewritten here). Written before
the main Stage A training and before any FNTO evaluation on the held-out Lego sets; the predeclared numbers are in
`protocol/fnto_v1.json`. Everything below is an implementation choice unless it is quoted from the contract.

## 0. Reference-renderer semantics (measured first, `fnto_semantics.py`, `results/fnto_lego/semantics/`)

| question | measurement (Blender 3.5.1 Cycles, the generator's settings) | consequence |
|---|---|---|
| Does Christensen-Burley SSS cross disconnected parts of ONE object? | yes: a shadowed yellow slab 2 cm from a lit one glows (0.065, 0.0055, 0.00002) | the Lego's SSS response is object-scoped, not component-scoped |
| ... across separate objects? | no (0.0 exactly) | the box can neither feed nor block the Lego's SSS |
| ... entering through a non-SSS material of the same object? | yes: identical glow when the lit slab is Gray | Burley entry irradiance is taken anywhere on the object; exit only at SSS surfaces |
| Is SSS counted as a bounce? | no (max_bounces 0 keeps it) | |
| First intrinsic event isolation | `max 1, diffuse 0, glossy 0, transmission 1`: keeps sunlight transmitted through a glass solid (4.008 = full) and does NOT see a lit plane through glass (0.0316 = surface reflection only) | Stage A renders = K applied to the known incident field, no free-space segment |
| Free-space share on the isolated Lego | yellow red: 0.52 first event vs 0.92 full | C must carry ~44 % of yellow's red radiance (exposed same-asset interreflection) |

Domain mapping (contract §2: domains are not object IDs): the renderer's intrinsic Burley operator is object-wide, so the
Lego's intrinsic domain A contains every Lego surface interface with Burley (object-scoped, material-agnostic entry),
local BSDF events and glass-solid traversal; every free-flight segment between Lego surfaces is in C (a = b), never in
K. The box is domain B. This is the reference semantics, documented, not changed.

## 1. Optical domains and boundary ports — `fnto_anchors.py`

Mesh-complete ports (SPT-1 used only camera-observed cells; SSS entry happens on unseen faces): area-proportional
surface samples clustered by (1.2 cm voxel, signed dominant normal axis, material); closed components oriented outward,
open components two-sided. A: 463,614 ports, B: 1,702. Coarse ports (4 cm, 38,090) for the mid-range kernel.
Failure signal: hit -> port distance (q99 9 mm), facing fraction (99.4 %).

## 2. Coordinates

Per-domain canonical frames (T0 placement); edits in this batch are translations, so directions are shared with the
world. Gather directions use a per-port random tangent rotation (anti-aliasing); bins and band geometry use a
deterministic frame. Rotation covariance is supported by construction (all inputs in local frames) but is not tested
by the T0/T1 edit (declared).

## 3. Intrinsic kernel input / output — `fnto_model.py`

K_a maps the incident field on its ports and at a receiving point to outgoing radiance; linear in radiance, all
coefficients non-negative:
- **K_loc** — learned per-point BSDF (diffuse + 2 GGX lobes, alpha >= 0.01), applied at the receiving point to the delta
  sun (exact direction and the pixel's own visibility) and to indirect radiance per gather direction.
- **K_glob** — sum_m p_m(x) q_m(s), rank 64 per colour channel (A), cosine-weighted entry, isotropic exit: long range.
- **K_band** — 32 nearest ports within 5 cm, 19 hemispherical source bins, exit depends on w_o: near-field SSS,
  thin walls, glass entrance -> exit.
- **K_mid** — 64 coarse ports within 30 cm, irradiance entry: the 3-25 cm range. It was added after contract B exposed
  that band + low-rank global leave this range unresolved (70-140 % errors), before declaration.

Each domain owns its parameters (A: TriField 512^2 + 64^2, x8 channels; B: 64^2 + 16^2). No learned component is shared
between domains or with C.

## 4. Directional representation and angular support

- Delta sun: exact on every path (pixels, ports, sender glossy).
- Indirect at camera pixels: 128 cosine directions per port, up to 4 interpolation ports, full GGX lobes per direction.
  The indirect glossy angular resolution is limited by 128 directions (contract D reports a 24 % quadrature error for
  a 5 deg patch; declared).
- Sender outgoing state used by C: isotropic part + exact delta-sun glossy. Glossy reflection of indirect light is
  folded in isotropically with lobe albedo 1; band and mid terms are evaluated at w_o = n_s. These are declared
  sender-side approximations; the Lego's glossy weights are 2-5 %.
- Band source bins: 19 (cap 20 deg, 6 x 20-50 deg, 12 x 50-90 deg). Contract C shows directional transmission is
  smoothed (forward lobe 0.73 / 0.84).

## 5. Material-internal transport

K_glob + K_band + K_mid, trained only from isolated-domain first-event probes. Sharp, clear refraction is not resolved
(bins); the Lego's glass is rough (alpha 0.64). Measurement: Stage A holdout per material, contracts B and C.

## 6. Free-space coupling — `fnto_relations.py`, `fnto_solve.py`

Explicit ray queries of the current geometry (allowed by the brief §2.3): 128 gathers x 465 k ports (59.6 M rays). Hits
are mapped to the hit triangle's facing-side cluster; hits on the inside of a closed solid are blocked; misses carry no
light.

Compression, caching and reuse:
- ports instead of pixels;
- fixed directions;
- per-port aggregates (irradiance, 19 bins, coarse clusters);
- an incremental relation update that re-traces only rays touching the moved domain's old or new bounds (0.83 % of
  rays, 31 k rows on the T1 edit). It equals a full rebuild except for 615 ties (identical hit distance) between
  coincident triangles.

**The learned coupling rule C_phi is empty in this implementation:** free flight has no material degrees of freedom to
learn. Consequently contract §6 question 3 ("does *learned* coupling capture the influence without a dense rebuild")
is not answered by this batch. This is a declared scope decision, not a hidden renderer: the relation build and the
incremental update are measured and reported as coupling cost.

## 7. Multi-bounce solution

Jacobi on the per-port incident samples: L <- (K applied to the sun + current samples) gathered through the relation
table. Stop when the relative L1 change is < 1e-3 (at most 100 iterations). Divergence (non-finite, or growth after 5
iterations) is detected and reported, never clamped away. The residual decay rate estimates the spectral radius of K∘C.
Contract F: a closed rho = 0.8 cube converges to rho/(1 - rho) within 0.14 %, at rate 0.76.

## 8. Training supervision and identifiability — `fnto_stagea_render.py`, `fnto_stagea_prep.py`, `fnto_train.py`

Isolated-domain first-event probe renders (the reference renderer, the generator's light convention). The Lego and the
box are present together; without free-space paths each is only an occluder of the other's sunlight.

Stimuli:
- one sun per image, 70 % upper / 30 % lower hemisphere (excites downward faces);
- 0-10 camera-invisible shadow-only spheres (spatially localised incidence) in 70 % of images;
- random box placements, never within 0.25 m of T1;
- cameras 70 % above / 30 % below.

Splits: 1,400 training images, 120 held-out (unseen masks / suns / cameras / placements), 12 superposition triples.

The incident field of every image is exact (traced port visibility). gamma is estimated per step by unbiased importance
sampling of 32,768 ports; evaluation uses all ports. No full-scene image, no free-space interreflection and nothing of T1
enters training.

## 9. Geometry-edit update procedure

On edit:
1. incremental relation update;
2. port sun visibility in the new geometry;
3. state solve per view (cold start);
4. pixel render.

No learned parameter changes; digests are compared.

## 10. Memory, cost and expected failure modes

Memory and cost:
- Parameters: A 6,531,047, B 202,335.
- Per-port tables: GPU, tens of MB plus the 32 x 19 band evaluation on the fly.
- State: 59.6 M incident samples x 3 x fp16 = 357 MB.

Expected failure modes and their measurements:
- **Yellow SSS fidelity** (object-wide Burley; 41 % of pixels): Stage A holdout and T0 yellow.
- **Sender approximations:** T0 shadowed and reflective error vs the first-event decomposition.
- **Port resolution** at the 3.6-12 mm box-plate gap.
- **K_loc extrapolation** to incident directions the sun never provided: T0 regions lit only indirectly.
- **Passivity of the learned operator:** solve residual rate, divergence.
- **First-event / full-render semantic mismatch:** Stage A holdout vs the T0 first-event decomposition.
