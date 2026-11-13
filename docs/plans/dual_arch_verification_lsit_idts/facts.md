CODE / PHYSICS FACT SHEET: ECRO family (WL86 to WL93C). Read-only check at commit 4ed4ac3. Nothing was run except numpy reads of the stored material table and analytic arithmetic.
Labels: [OBS] = read directly in code or a stored file (path:line given). [ARITH] = computed from code constants. [SUP] = supported inference. [PLAU] = plausible hypothesis. [UNDET] = could not be determined from the code.

## 1. Discretisation (experiments/ecro86/scene86.py, protocol/ecro86.json)

**Bins**
- [OBS] The bins are 64 equal-solid-angle bins of the patch-local hemisphere: NZ=4 rings that are uniform in cosθ, times NPHI=16 azimuth sectors (scene86.py:15-20; protocol `representation.bins_z/bins_phi`).
  - Bin index = iz*16+iphi, with phi measured from -π (scene86.py:131-138; pt86.py:153-156).
  - DOMEGA = 2π/64 = 0.0982 sr (scene86.py:20).
  - [ARITH] Ring boundaries are θ = 90 / 75.5 / 60 / 41.4 / 0°. Each sector is 22.5°.
- [OBS] `bin_of` clamps iz to [0,3]. A direction below the patch-frame horizon is therefore put into ring 0 (pt86.py:153-156).

**Patches**
- [OBS] Layout (scene86.py:29-39, 47-56, 21, 92-100):
  - floor 16x16, ceiling 16x16, back wall 16x12, left wall 16x12, right wall 12x16 (25 cm cells);
  - panel 6x7 (20 cm cells);
  - each box face 3x3, 2 boxes x 6 faces;
  - sphere 8 (azimuth) x 4 (z bands), equal area.
- [ARITH] Total patches: 256+256+192+192+192+42+108+32 = **1270**, matching protocol "1270".
- [ARITH] ER patches: floor 256 (B) + panel 42 (S1) + sphere 32 (S2) = 330. All other patches are Lambert.
- [OBS] Material ids: 0 = B floor, 1 = S1 panel, 2 = S2 sphere, 3 = walls/ceiling (Lambert 0.7), 4 = box A (Lambert 0.5), 5 = box B (Lambert 0.6) (scene86.py:22; mats86.py:4; protocol `lambert_albedo`).
- [OBS] Sphere patches are curved. They have one patch-centre frame (t = Z×n normalised, b, n) (scene86.py:97-98). Bins, state-ray directions, source binning and the camera lookup all use this patch-centre frame (pt86.py:179-181, 192). Ray origins and source points sit on the true sphere surface with the true normal (pt86.py:186-192).
  - [ARITH] One sphere patch spans 45° of azimuth and a 0.5 band in z, so its normals vary by tens of degrees.

**State rays**
- [OBS] There are KAPPA = 8 rays per incident bin, K = 512 per patch (protocol `state_rays_per_bin`; scene86.py:17-18).
- [OBS] Each ray has its own uniform point in the patch (us[0], us[1]) and a uniform direction inside its bin (us[2], us[3]). The offsets come from one RNG with seed 86001 and are **identical for every patch and every geometry** (pt86.py:114-117, 126, 353-361). They are not jittered-stratified within a bin; the stratification is only one bin per group of 8 rays.
- [OBS] Each ray stores:
  - the hit patch (int16; -1 for a miss, a back face, or a direction below the local horizon);
  - the outgoing bin at the hit patch toward the ray origin (uint8), in the hit patch's frame;
  - the segment o/e (6 float32), used only by the dirty test (pt86.py:362-376; ecro86.py:57-67).
- [ARITH] 1270 x 512 = **650,240 rays**.
  - Stored at 2+1+24 = 27 B per ray: 17,556,480 B ≈ 17.6 MB, under the protocol's 20 MB `state_bytes_max`.
  - hit+dbin only: 1.95 MB.
- [OBS] Ray weight rw = cosθ(patch frame) · DOMEGA / KAPPA (model86.py:40-46).
  - For sphere patches this cosine is the patch-centre cosine, not the point-local cosine.

**Source encoding s16**
- [OBS] PSRC = 16 points per patch (pt86.py:378-402). The 16 (u,v) offsets come from the same seed-86001 RNG and are shared by all patches. They are random, not stratified (pt86.py:117, 127).
- [OBS] For each point and each light: E = rgb·cos(true normal)/d², multiplied by a shadow ray and the cone test. The result is binned by the **direction toward the light**, expressed in the patch(-centre) frame, then summed and divided by 16.
  - So s[i, bin, rgb] is the patch-average **direct irradiance** (cosine included) per incident bin.
- [ARITH] s has 243,840 floats (975 KB) per lighting and costs 20,320 shadow rays per light.

**Lights**
- [OBS] `light_arrays` allocates fixed [2,3] arrays, so more than 2 lights is a hard limit (it fails with an index error) (scene86.py:230-237).
- [OBS] Light types:
  - point light: cos_cut = -2;
  - spot: hard cone cut, `dot(-wl, axis) > cos_cut`, with no falloff (pt86.py:250; 390).
- [OBS] Training uses exactly one white point light per lighting (I = 30, 4 lightings per geometry) (scene86.py:218-227; gen86.py:32).
  - Held-out sources: LMIX (2 coloured point lights) and LSPOT (cone 35°, rgb 60).

## 2. Transport conventions and event order (pt86.py:1-10, 202-285; gm1_explicit87.json `contract`)

- [OBS] Estimator: next-event estimation (NEE) to every light at each vertex, then ER BSDF sampling. There is no Russian roulette. D = 6 (pt86.py:26).
- [OBS] The contribution made by NEE at camera vertex k has k scattering events.
  - **e1 = q** read by the camera (k=1).
  - **e2 = Tq** (k=2).
  - h = events 3..6.
  - **LOW = e1 + e2**, path-traced (gm1_explicit87.json `contract.camera`).
- [OBS] In ECRO mode, kmax = 2. At x2 the renderer adds hl = throughput(x1) · B[patch(x2), bin(x2→x1)] (pt86.py:228-233). The throughput includes x1's sampled ER weight.
  - **B holds the radiance leaving x2 made of 2..5 scattering events**, so hl supplies camera events 3..6 without double counting q.
  - The lookup is piecewise constant over patch and bin.
- [OBS] Table target f (field probe, pt86.py:327-351):
  - A random point in the patch and a uniform direction in the bin.
  - A probe ray must re-hit within 1 mm. NEE is skipped at vertex 1. NEE at vertices 2..5 gives events 2..5.
  - The result is averaged over valid samples. For the sphere, "valid" means above the local horizon.
  - Training spp = 256 per (patch, bin) [ARITH: 20.8 M probe paths per lighting]. Val and trainhq use 1024, split into 2 halves.
  - Stored as float16 (gen86.py:44-48).
- [OBS] WL92A split targets: e2_p/e2_o = exactly 2 events, with the last sender on the panel or elsewhere; e3_p/e3_o = 3..5 events (ecro86a/pt86a.py:73-75).
  - t2 = e2_p + e2_o (own92a.py:115).
  - The single-flight target is t2; the global target is t − t2 (ecro_stage93/model93.py:109-115).

## 3. ECRO operator (experiments/ecro86/model86.py)

**Generator**
- [OBS] Generator, which is source-independent (it never sees s) (model86.py:91-107). Patch features:
  - centroid, normal, tangent, log area;
  - MLP of the frozen 69-float ER state;
  - openness = 1 − Σ form factors.
- [OBS] It then runs 3 rounds of message passing over Ft plus a global mean. Ft is the dense N×N matrix of state-ray cosine form factors (model86.py:83-89). [ARITH] Ft has 1,612,900 floats.
- [OBS] Heads (softplus ≥ 0, per colour). Sizes: d = 128, R = 16, ρ = 24.
  - c[N,64,3,16] maps incident bin to channel; a[N,64,3,16] maps channel to outgoing bin (inputs: token + bin embedding).
  - Q[N,3,24,16] and P[N,3,16,24] are per-patch only.

**Application** (model86.py:109-124)
- u = c·s; **Bq = a·u** (learned once-scattered outgoing radiance, approximating q).
- **E1** = Σ over the 8 rays of each incident bin of rw·Bq[hit, dbin]. This is the single learned flight with current visibility; a miss or back face gives 0.
- **single = a·(c·E1)** (2 events).
- **global = a·(P·Σ_j area_j Q_j·u_j)**.
  - The input of the global term is u = c·s (encoded direct light), not Bq or E1.
  - [SUP] Per colour it is a rank ≤ 24 map from the whole-scene source to the whole table.
- B = a·(c·E1 + yg).
- [SUP] The local operator a cᵀ is a 64x64 matrix of rank ≤ 16 per patch and colour. The same a and c are used at the source use, the receiver use and the global decode.
- [OBS] WL86 trained on the sum target (2..5 events), so ownership of the components was not enforced (protocol `training.loss`; model86.py:130-133). WL92A "OWN" added the component losses.

**Sizes**
- [ARITH] Parameter count by hand count of the layer shapes (not instantiated): ≈ **735,664** (budget 1 M).
- [ARITH] Table B and s each have 1270·64·3 = 243,840 floats; this matches the protocol's `table_floats`.
- [ARITH] c and a have 3.90 M floats each; Q and P have 1.46 M floats each.

## 4. WL93 local arms (experiments/ecro_stage93/model93.py)

- [OBS] B = S_θ(s) + G_frozen(s). G_frozen = `own.apply(..., use_single=False)` of WL92A own_s<seed>, using own's own a/c/Q/P (model93.py:100-102). The global heads of the trainable copy are frozen and unused (model93.py:65-67).
- [OBS] Modes (model93.py:69-79):
  - **tied**: c_s = c_r, a_s = a_r.
  - **untied**: duplicated heads head_c2 / head_a2 for the receiver, with a shared trunk.
  - **tiedcm**: c = softplus(head + head2); head2 hidden layers are freshly initialised from a fixed seed and its last layer is zero.
  - All three equal own at step 0.
- [OBS] `single()` returns S = a_r·c_r·E1, Bq = a_s·c_s·s, and E1 (model93.py:81-91).
- [OBS] Losses: l_sum on S+G vs t; l_s on S vs t2; l_g on G vs t−t2; l_q on Bq vs q. Each uses den = bin-mean + 0.05·global mean (model93.py:105-115).

## 5. WL93C oracle (experiments/ecro_stage93c/eval93c.py:43-79; protocol `task_a_oracle`, `physical_local_operator`)

- [OBS] `exact(Mn, E)[i,bo] = Σ_bi E[i,bi]·M64[m_i,bi,bo]`, with Mn = Mq[PMAT] (eval93c.py:53-55, 97, 114).
- [OBS] `flight` = the deployed gather with own.rw (ecro_stage93b/eval93b.py:149-156), identical to the flight inside model93 (test T_FLIGHT).
- [OBS] ORA: Bq = M64(s); E1 = flight(Bq); S = M64(E1); G = frozen own global. **ORA = M64(flight(M64(s16))) + G.**
  - It must equal WL92B X_M12 to 1e-5 (eval93c.py, test T_ORACLE).
- [OBS] Attribution variants:
  - ORA_SRC: exact source use, learned T0 receiver.
  - ORA_RCV: learned T0 source use, exact receiver.
  - ORA_PHYS: S = M64(flight(q_B5 mean)), no encoding.
- [SUP] ORA uses no reference data: s, state rays, the precomputed M64 and the frozen learned G are all runtime-computable. The protocol's "non-deployable diagnostic" label is a role designation, not a data dependency. ORA_PHYS does use the B5 reference q.
- [OBS] The receiver target in Task B is R* = M64(flight(q)) (train93c.py:5).

## 6. Explicit comparator and M64 (experiments/gm1_explicit87/quad87.py, xport87.py; gm1_explicit87.json `explicit`)

**fbar (quad87.py:68-110)**
- [OBS] fbar(m,w)[bo] = (1/DOMEGA)·∫_bo f_m(w,v)dv.
- [OBS] One-sample MIS mixture: 0.5 ER sampling from w plus 0.5 uniform hemisphere, weight f/p_mix. The cosine is removed by dividing by v.z.
- [OBS] Reciprocity is used: sampling is from the light side.

**M64**
- [OBS] Dense table: fbar on 4,096 incident cells (8x8 cell centres per bin), M_dense = 16,384 samples per cell, seed 87002, built in the **flat patch frame** (quad87.py:15-16, 48-56, 223-231).
- [OBS] `enc_matrix()` = Mq[m, bi, bo] = the uniform mean of fbar over the 64 cells of incident bin bi (quad87.py:240-242). This is "M64", [6,64,64,3].
  - Lambert: alb/π exactly (quad87.py:229).
  - Realisation us(86001), M_ray 65,536, seed 87001 (ecro_attr92/diag92.py:133-139).
- [SUP] M64 discards the actual distribution of incident directions inside a bin. Sphere patches use the flat-frame matrix with no horizon or valid-solid-angle correction (protocol: "sphere patches use the flat-frame matrix as WL87 declared").
- [ARITH] M64 has 73,728 floats; the dense table has 4.72 M float64.

**W_ray (used by EXP, not by ORA)**
- [OBS] W_ray = fbar(w_r)·cos(w_r)·DOMEGA/KAPPA, evaluated at each ray's exact direction (quad87.py:188-221).
- [OBS] Sphere rows 6+iz use the per-ray point frame, point cosine and horizon, normalised by the mean valid solid angle.

**q table, the once-scattered estimator (quad87.py:268-338)**
- [OBS] Budget B0 = P 16, Mq 1024; this is the WL87/WL93 budget. Per point:
  - P = 16 jittered-stratified 4x4 points per patch (sphere points carry the point shading frame);
  - per light: E = I·cos/d² with a shadow ray and the cone test;
  - ER: Mq MIS samples about each point's light direction, outgoing samples binned in the **patch** frame and kept only above the local horizon;
  - sphere bins divided by the mean valid solid angle (flat bins: DOMEGA);
  - Lambert: alb/π·E exactly.
- [ARITH] B0: 330 ER patches x 16 x 1024 = 5.41 M MIS samples per light.
- [ARITH] B5 (P 256, Mq 4096; the WL93A evaluation reference): 346 M samples per light.

**enc_q and EXP**
- [OBS] enc_q = Σ_bi s·M64 (xport87.py:107-114).
- [OBS] EXP: L1 = q; L_{k+1} = T_d L_k via the W_ray gather/scatter; B_exp = L2+L3+L4+L5 (xport87.py:49-86).
- [OBS] There is no renormalisation or clamping (protocol `explicit.not_imposed`).

## 7. Frozen ER material closure (fnto_compiled83/closure83_dr.py; semantics in fnto_compiled82/closure82.py; basis in fnto_compiled80c/closure80c.py)

**Provenance**
- [OBS] Path tracing uses `CD.DrClosure83('ER', mats.states, mid)` (pt86.py:236-237).
- [OBS] The states are bit-identical copies of the WL85 fixture (mats86.py:26; data/mats86.json `er_states_equal_wl85: true`).
- [OBS] Those fixture states are WL83 **learned-compiler** outputs `fit_<set>_L_ER_s8301` decoded in 'cert' mode (gm1_char85/census85.py fixture(); naming in fnto_compiled83/compiler83.py:144-156).
  - B = test83_iid #115, S1 = test83_hard #97, S2 = test83_hard #41.

**Form (closure83_dr.py:1-17, 99-155)**
- [OBS] 8 lobes:
  - Lambert (lam index 0);
  - 3 OUTER anisotropic GGX lobes;
  - 4 INNER GGX lobes, evaluated at refracted directions t(w) = (w_x/η, w_y/η, sqrt(η²−1+w_z²)/η), scaled by 1/η².
- [OBS] f = Σ_outer λ g + s(μ_i)s(μ_o)·[λ0/π + Σ_inner λ g(t(wi), t(wo))/η²].
  - s(μ) is a per-RGB piecewise-linear curve, knots at μ = 0, .15, .4, .7, 1, values in [0,1].
- [OBS] g = F·D / (2(cos_o·sqrt(cos_i² + α_u²w_x² + α_v²w_y²) + cos_i·sqrt(...))), i.e. anisotropic GGX D with height-correlated Smith G2/(4 cos_i cos_o).
  - Fresnel: Schlick per lobe with scalar (achromatic) f0, using dot(wi,h).
  - λ is per colour per lobe.
- [OBS] There is no explicit coat-interface Fresnel transmission term; energy is carried by λ and s(μ).
- [OBS] There is no multiple-scattering energy compensation in any GGX lobe.
- [OBS] Anisotropy axes are aligned with the local shading frame.

**Constraints**
- [OBS] Passivity is certified by construction: λ = softmax over 9 slots (8 lobes + absorption), so Σλ ≤ 1 per channel. Each lobe's albedo is ≤ 1 and s ≤ 1. α ∈ [0.004, 1]; η_e ∈ [1, 2.5] (closure80c.py:24-42; closure82.py header and decode at line 45).
- [OBS] Reciprocity is exact by construction because every term is symmetric (closure82.py header; quad87.py:4).
- [OBS] Sampling: lobe chosen with wi-independent probabilities, GGX VNDF (spherical caps), cosine Lambert, an exit-refraction validity check, weight = eval/pdf (closure83_dr.py:164-202).

**Stored states** [OBS] (numpy read of experiments/ecro86/data/mats86.npz; λ column order: Lambert, outer k0..2, inner k3..6)

| | S1 panel | S2 sphere | B floor |
|---|---|---|---|
| α_u range | 0.026–0.075 | 0.074–0.18 | 0.27–0.45 |
| α_v vs α_u | ≈ α_u (within ~10 %) | ≈ α_u | ≈ α_u |
| Lambert λ | ≤ 0.002 | 0.002–0.014 | 0.05–0.09 |
| η_e | 2.4955 | 2.119 | 1.433 |
| Σλ per channel | 1.0 | 1.0 | 0.9987–0.9999 |
| Physical layered descriptor (coat α, coat IOR, slab τ, base α) | 0.0507, 1.3335, 0.024, 0.0518 | 0.131, 1.697, 0.024, 0.114 | 0.434, 1.335, 0.043, 0.332 |

- [OBS] S1 dominant lobes: red is outer lobe k0 (λ 0.57, α 0.044); green/blue are inner lobe k4 (λ 0.44–0.50, α 0.031).
- [OBS] S1's η_e 2.4955 sits at the 2.5 bound, while its physical coat IOR is 1.333.
- [OBS] Descriptor source: fnto_compiled81/layered81.py:1-20, 36. The physical reference "P" is a pbrt-v4-style coated conductor (rough dielectric coat, slab, rough conductor) evaluated by random walk.
- [OBS] Census (results/gm1_char85/results/census/fixture.json; r50 = sqrt(Ω50/π) rad, census85.py:78; incidence 0/30/45/60/80°):
  - closure r50: S1 0.146 / 0.146 / 0.147 / 0.142 / 0.090; S2 0.40 → 0.21; B 0.62 → 0.56.
  - closure albedo: S1 0.68 → 0.54; S2 0.51 → 0.41; B 0.45 → 0.32.
  - S1 sharpest lobe r50 at 45° = 0.049.
- [ARITH] S1 Ω50 ≈ π·0.146² = 0.067 sr at normal incidence and 0.025 sr at 80°. The sharpest lobe has Ω50 ≈ 0.0075 sr. One bin is 0.098 sr, so half of S1's reflected flux fits inside less than one bin.
- [OBS] Docs: WL88 and WL90 state "S1 α ≈ 0.05" (docs/worklogs/88_...md:44; 90_...md:263). This is consistent with the stored α.

**Measured rank of M64** [OBS] (results/ecro_attr92/results/local.json, flux-weighted matrix):

| | rank-16 Eckart–Young residual | rank for 99 % energy | NMF-16 peak retention |
|---|---|---|---|
| S1 | 0.46–0.52 | 46–59 | 0.47–0.54 |
| S2 | 0.20–0.34 | 31–42 | — |
| B | 0.03–0.04 | 8 | — |

- [OBS] Learned WL86 a cᵀ on S1: Frobenius 0.88, peak 0.11–0.14, mass 0.68–0.76.

**Analytic inferences**
- [SUP] Near the normal, refraction maps exterior angle to interior as dθ_out ≈ η·dθ_in. An inner lobe with α 0.031 at η 2.5 is therefore about 2.5x broader in exterior angle near the normal, and the factor grows toward grazing.
- [PLAU] The path tracer uses Mitsuba's si.to_local, whose tangent for a rectangle is the to_world x-axis (= RA1) and for the sphere the φ-direction (= the point-frame tangent ts). This likely matches the patch/point frames of quad87. Not verified by running; its impact is small given α_u ≈ α_v.

## 8. Table B contents by variant (summary)

- **WL86 ECRO:** B = a·c·flight(a·c·s) + a·P·ΣAQ·c·s. Trained to total events 2..5; ownership not enforced.
- **WL92A OWN:** the same form; single-flight term → t2 (events = 2), global term → events 3..5.
- **WL93:** learned S(s) plus the frozen OWN G.
- **ORA:** M64·flight·M64·s + G.
- **EXP:** L2..L5 by W_ray recurrence from the q table.
- [OBS] All of these are read identically by `ecro` mode (ecro86.py:113; xport87.py:7-8).

## 9. Could not determine from code [UNDET]

- What the learned ECRO/OWN/T0/TQ weights look like internally, beyond the stored WL92 local.json metrics.
- How faithful ER is to the layered P BSDF for these three particular materials. WL84 qualified the learned ER only at population level.
- The physical meaning of η_e (it saturates at the bound for S1; WL82 records "η_e not identified").
- Whether the sphere's flat-frame M64 versus the point-frame W/q (horizon normalisation) causes a measurable bias in ORA.
- The magnitude of the error from using a uniform-in-bin mean in M64 versus the true within-bin incidence distribution, for ORA as opposed to EXP.
- Runtime costs (not measured here).
- The WL92A OWN training hyper-parameters beyond the loss form (own92a.py train() not read in full).
- Exact Mitsuba shading-frame tangents (§7).
