# WL95 read-only contract: current ECRO pipeline and its exact-sender control

Scope: I read code and docs only. I edited no files and ran no GPU work.

"Current" deployed operator: the WL92A OWN checkpoints `own_s8601/8602`. They load into the unchanged `model86.ECRO` class (`experiments/ecro_stage93/train93.py:45-51`) [FACT]. WL94 uses them through `mat94.load_frozen` (`experiments/ecro_mat94/mat94.py:209-212`) [FACT].

OWN trains the same architecture with `L = L_sum + ||B_single − t2||² + ||B_global − (t − t2)||²`, using den-normalised squared error (`experiments/ecro_own92a/own92a.py:172-176`) [FACT]. So the single flight is supervised towards the exactly-2-event field. The global term is supervised towards 3..5 events [FACT].

## (1) Data flow with equations

Notation:
- Patch `n` is in [0, N).
- Incident bin `m` and outgoing bin `d` are each in [0, NB).
- Colour channel `c` is in {R, G, B}.
- Lights are batched along `l`.
- Factor tensors are per colour channel. No cross-channel mixing exists anywhere (`experiments/ecro86/model86.py:103-106`) [FACT].

### a. Bins and patch frames
- Each bin is `iz·16 + iφ`. The ring index is `iz = clamp(floor(z·4), 0, 3)`, where z = cos θ in the patch frame (t, b, n). The sector index is `iφ = clamp(floor((atan2(y,x)+π)·16/2π), 0, 15)` (`experiments/ecro86/pt86.py:153-156`) [FACT].
- Bins have equal solid angle, ΔΩ = 2π/64 ≈ 0.0982 sr (`experiments/ecro86/scene86.py:15-20`) [FACT].
- A direction inside a bin is drawn as `z = (iz+a)/4`, `φ = (iφ+b)·2π/16 − π` (`pt86.py:195-199`) [FACT].
- Sphere patches are binned in the patch-centre frame (`scene86.py:92-100`, `pt86.py:180-181`). A z below that frame's horizon is clamped to ring 0 (`pt86.py:154`) [FACT].

### b. Incident source field s ∈ ℝ[N, 64, 3], computed by `Ctx.source`
- Code: `experiments/ecro86/pt86.py:378-402` [FACT].
- `s[n,m,c] = (1/16) Σ_{j=1..16} Σ_l I_l,c · cos_j,l / d²_j,l · V_j,l · 1[bin(w_j,l) = m]`.
- Each patch uses 16 points `p_j = patch_point(n, u0_j, u1_j)`. The offsets are uniform random and are the same for every patch: `rng(86001)` after the state-ray draw (`pt86.py:114-117, 127`) [FACT].
- The cosine is taken against the point normal `nn`. A shadow ray and the spot cone are applied (`pt86.py:386-394`) [FACT].
- Units: W/m², i.e. patch-averaged direct irradiance per incident-direction bin [FACT]. A point light puts each point's whole irradiance into one bin; the exact direction inside that bin is discarded [INF].
- The generator never sees s. s enters only through `apply` (`experiments/ecro86/protocol/ecro86.json:11`) [FACT].

### c. Geometry state, computed by `Ctx.state_rays`
- Code: `pt86.py:353-376` [FACT].
- There are K = 512 rays per patch, indexed `r = n·K + k`. The incident bin is `k // κ`, with κ = 8.
- Origin: `patch_point(n, us0[k], us1[k]) + nn·1e-4`, one uniform random point per ray. Direction: `dir_in_bin(k//8, us2[k], us3[k])`.
- The `us` offsets come from `default_rng(86001)` and are the same for every patch. They are fixed and deterministic [FACT: `pt86.py:114-116, 126, 359-363`]. So each patch reuses one identical pattern of 512 (position, direction) samples, jittered but not stratified in position [FACT].
- Each ray stores:
  - `hit`: the hit patch from `patch_bin`. It is −1 for a miss (open side), a back-face hit, or when `w·nn ≤ 1e-3`. Stored as int16.
  - `dbin`: the bin of −w in the hit patch's frame, i.e. the sender's outgoing bin. Stored as uint8.
  - `seg`: origin and end point, used by the dirty test.
  - Source: `pt86.py:364-375` [FACT].
- `build_state` wraps this (`experiments/ecro86/ecro86.py:43-45`) [FACT].

### d. Ray weights
- `rw_k = z_k·ΔΩ/κ`, where `z_k = (iz(k//8) + us2[k])/4` is the ray's patch-frame cosine. It is regenerated from the same seed (`experiments/ecro86/model86.py:40-46`) [FACT].
- Invalid rays are not renormalised. In particular, sphere rays below the local horizon get `hit = −1` but keep their `rw` [FACT, model86 has no normalisation]. By contrast, the field target averages only over valid samples (`pt86.py:349-350`) [FACT]. This mismatch for curved patches is an [INF].

### e. Generator `factors(gs)`
- Code: `model86.py:91-107` [FACT].
- Form factors: `Ft[i,j] = Σ_{k: hit=j} rw_k/π` (dense N×N, `model86.py:83-89`). Openness is `1 − Σ_j Ft[i,j]` [FACT].
- Patch features (27-d): `[0.5·centre, normal, tangent, log area + 3, mat_mlp(ER state 69→16), openness]` → `inp` → token t ∈ ℝ¹²⁸ (`model86.py:95-96`) [FACT].
- Three rounds of `t ← t + upd([t, Ft·msg(t), mean(glb(t))])` (`model86.py:97-100`) [FACT].
- Bin feature (16-d): `[bin-centre dir (3), learned bin_emb (13)]`. Heads:
  - `c = softplus(head_c([t, bin]))` → [N, 64, 3, R]
  - `a = softplus(head_a([t, bin]))` → [N, 64, 3, R]
  - `Q` → [N, 3, ρ, R]
  - `P` → [N, 3, R, ρ]
  - Source: `model86.py:101-107` [FACT].
- c and a come from separate heads on identical (patch, bin) inputs [FACT].

### f. Application `apply(f, gs, s)`
- Code: `model86.py:109-124` [FACT].
- Sender activations: `u[l,n,c,r] = Σ_m c[n,m,c,r] s[l,n,m,c]` (line 112).
- Learned once-scattered sender radiance: `Bq[l,n,d,c] = Σ_r a[n,d,c,r] u[l,n,c,r]` (line 115). The local operator is `M_n = a cᵀ`, rank ≤ 16 per patch and channel, non-negative, in 1/sr.
- Flight:
  - `E1[l,n,m,c] = Σ_{κ=0..7} 1[hit_{n,mκ} ≥ 0] · Bq[l, hit_{n,mκ}, dbin_{n,mκ}, c] · rw_{mκ}` (`model86.py:116-119`).
  - Identical code is reused as `eval93b.flight` (`experiments/ecro_stage93b/eval93b.py:149-156`) [FACT].
  - E1 is a deterministic 8-sample estimate per incident bin of ∫_bin L cos dω (W/m²). Each sample reads the sender's patch × outgoing-bin table value [FACT/INF].
- Receiver: `u1 = Σ_m c[n,m,c,r] E1[l,n,m,c]` (line 120). This is the same c as the sender, so the local operator is tied and used twice [FACT].
- Global term:
  - `wq[l,c,p] = Σ_n area_n Σ_r Q[n,c,p,r] u[l,n,c,r]`
  - `yg[l,n,c,r] = Σ_p P[n,c,r,p] wq[l,c,p]`
  - Source: lines 122-123.
  - The global term is driven by the direct-source activations u, not by E1 [FACT].
- Output: `B[l,n,d,c] = Σ_r a[n,d,c,r] (u1 + yg)` (line 124). Terms:
  - `B_single = a·c·E1`
  - `B_global = a·P·Σ area·Q·u`
  - Both are obtainable through `use_single` / `use_global` [FACT].
- The function is linear and non-negative in s [FACT, docstring lines 15, 103-106].

### g. Camera readout, `Ctx.trace` mode `'ecro'` with `kmax = 2`
- Code: `pt86.py:228-233, 300-303` [FACT].
- Camera ray → x1: next-event estimation (NEE) gives e1. Then one ER BSDF sample sets the throughput `thr = f·cos/pdf` at x1 (lines 272-284).
- The sampled ray hits x2, after a front-face test with `wi.z > 1e-6` (line 226). Then:
  - `(pid, bn) = patch_bin(si, sid, −ray.d)`
  - `hl = thr · B[pid, bn, :]` (lines 229-233)
- NEE at x2 gives e2 (line 265), and the path stops.
- So the higher-order image is throughput(x1) × B[nearest patch(x2), bin(x2→x1 in that patch's frame)], with no extra ray (`ecro86.json:9`) [FACT].
- B is the 2..5-event outgoing radiance, so hl estimates events 3..6 [FACT, `ecro86.json:7-10`].
- WL94 renders CK2 with exactly this path and saves only `hl` (`mat94.py:318-338`). Its table is `B = S_arm + G` (`mat94.py:281-282`) [FACT].

### h. Targets and reference field
- `Ctx.field` (`pt86.py:327-351`) uses uniform random points on the patch and uniform directions inside each bin. It traces back to the point (must land within 1 mm) and returns the average over valid samples of the 2..5-event radiance [FACT].
- `pt86a.field_src(split=True)` (`experiments/ecro86a/pt86a.py:73-200`) splits each sample by event count k at which NEE fires: k = 2 → e2, k ≥ 3 → e3. It splits by whether the k = 2 vertex has material 1, the S1 panel (`_p`, otherwise `_o`), at lines 175-180 [FACT].
- The four tables sum to `f` [FACT, lines 174-199]. `t2 = e2_p + e2_o` and `t3 = t − t2` (`own92a.py:115`; OWN protocol "targets") [FACT].

## (2) Sizes

| Quantity | Value | Source |
|---|---|---|
| N patches | 1,270 = floor 256 + ceiling 256 + 3 walls × 192 + panel 42 (6×7, 20 cm) + 2 boxes × 6 faces × 9 (3×3 per face, size varies with box) + sphere 32 (8×4) | `scene86.py:29-39, 47-56, 70-71`; `ecro86.json:46` [FACT] |
| NB | 64 = NZ 4 × NPHI 16; ΔΩ = 0.0982 sr | `ecro86.json:47`, `scene86.py:15-20` [FACT] |
| κ | 8 rays per incident bin | `ecro86.json:48` [FACT] |
| K | 512 per patch; N·K = 650,240 state rays | `scene86.py:18` [FACT]; `docs/worklogs/86_…md:128` [MEAS] |
| Source points | 16 per patch | `ecro86.json:48` [FACT] |
| R | 16 channels | `ecro86.json:55` [FACT] |
| ρ (global rank) | 24 | `ecro86.json:55` [FACT] |
| Token dim / rounds | 128 / 3 | `ecro86.json:55` [FACT] |
| Patch input | 27-d | `model86.py:70, 95` [FACT] |
| Bin feature | 16-d | `model86.py:101` [FACT] |
| Parameters | 735,664 | `ecro_own92a` protocol "preserved"; `mat94.py:433` test T_BUFFER [FACT]. My per-layer recount gives the same total [INF]. |
| Output table | 1270·64·3 = 243,840 floats (~1 MB) | `ecro86.json:50` [FACT] |
| Factor tables | c, a 3.90 M floats each; Q, P 1.46 M each; ≈ 43 MB | `86_…md:128` [MEAS] |
| State | 27 B/ray → 17.6 MB | `86_…md:128` [MEAS] |
| Ft | dense 1270² | `model86.py:89` [FACT] |

## (3) Every quantisation, average or aggregation point

1. **Patch spatial support.** B is piecewise constant over 1,270 patches (25 cm / 20 cm / 3×3 per box face / 8×4 sphere). At x2 the readout uses the nearest patch (`pt86.py:158-181`, `ecro86.json:46`) [FACT].
2. **Outgoing direction.** B is constant per 64-bin cell. The target is the average over uniform points in the patch × uniform directions in the bin, counting valid samples only (`pt86.py:337-350`) [FACT].
3. **Source encoding.**
   - Spatial: patch average over 16 fixed random points [FACT].
   - Angular: the incident direction is quantised to its bin and only the bin sum of irradiance is kept (`pt86.py:395-402`) [FACT].
   - The 16-point pattern is a fixed deterministic sampling bias, not temporal noise [INF].
4. **Local operator.** Rank ≤ 16 non-negative `a cᵀ` per patch and channel, on 64 × 64 bins (`model86.py:103-104, 112-115`) [FACT].
5. **Flight.**
   - E1 per (receiver patch, incident bin) is an 8-ray estimate. Each ray has one origin point and one direction from a fixed pattern shared by all patches [FACT].
   - Each ray reads the sender at the quantised sender patch and the sender's outgoing bin. This is a double projection: sender patch average and sender bin average [FACT].
   - Receiver positions are aggregated into one patch-level E1 [FACT].
   - No renormalisation for invalid or below-horizon rays [FACT].
6. **Receiver.** The same rank-16 `c` acts on bin irradiance E1, so incident direction inside a bin is not carried [FACT].
7. **Global term.**
   - A scene-wide, area-weighted sum is compressed to a ρ = 24 latent per channel, then decoded per patch through P and a (`model86.py:122-124`) [FACT].
   - It is driven by the direct u only [FACT].
8. **Generator inputs.**
   - The geometry relation is aggregated to patch-pair cosine-visibility form factors plus a global mean (`model86.py:83-100`) [FACT].
   - Material is the per-patch 69-d ER state compressed to 16 dimensions (`model86.py:95`) [FACT].
9. **Curved patches.** Bins use the patch-centre frame on the sphere. Below-frame directions are clamped to ring 0 (`pt86.py:154, 180`) [FACT].
10. **Camera readout.** x1 is exact (path-traced BSDF sample). x2 is a nearest-patch, bin-of-(x2→x1) lookup, with no interpolation (`pt86.py:229-233`) [FACT].
11. **M64.**
    - Incident bin = mean of fbar over 8×8 cell centres; outgoing = bin average.
    - Monte Carlo with M_dense = 16,384 samples.
    - Flat patch frame, also for the sphere: declared approximation (`experiments/gm1_explicit87/quad87.py:15-16, 223-242`) [FACT]. Details in section (5).
12. **Training targets.** 256-spp per-slot fields (target noise 0.10 relative L1 per lighting) [MEAS, `86_…md:129`].

## (4) Geometry-edit update path

1. **In-place geometry.** `Ctx.set_geom` rewrites the box vertex positions and normals in Mitsuba (`params.update()`, which rebuilds the acceleration structure) and recomputes the patch tables (`pt86.py:130-139, 120-128`). Measured: 31 ms wall [MEAS, `86_…md:83, 214`].
2. **State rays, selective.**
   - `dirty_rays` runs on the GPU over all N·K segments. A ray is dirty if its origin is on a moved box, or its segment crosses the old or new OBB of a moved box with a 1 mm pad (`ecro86.py:57-67`) [FACT].
   - Only dirty rays are re-traced and spliced into the old state (`ecro86.py:70-78`) [FACT].
   - This is exactly equal to a full rebuild, because the offsets are fixed [FACT: contract at `ecro86.py:161-168`; MEAS PASS at `86_…md:165`].
   - Measured: 8.7–10.2 % of rays dirty; 21 ms selective vs 26–28 ms full; saving 5–7 ms, described as "immaterial" [MEAS, `86_…md:131, 214, 229`].
3. **Source encoding.** Full recompute: N·16 shadow rays per light (5 ms) [MEAS, `86_…md:214`].
4. **Generator.**
   - `respond` calls `model.factors(gs)` afresh for the whole scene. No selective or partial factor update exists (`ecro86.py:81-87`) [FACT].
   - Ft (N²), message passing and the global mean propagate any edit to all tokens, so every patch's c, a, Q, P are recomputed [FACT/INF].
   - Cost order:
     - form-factor scatter O(N·K)
     - messages 3 × O(N²·d + N·d²)
     - heads O(N·NB·d²) and O(N·d·3ρR)
     - apply O(L·N·NB·3R) per contraction + flight gather O(L·N·K·3) + global O(L·N·3ρR)
     - [INF from code]
   - Measured: operator 11 ms (torch wall); transfer 1.5 ms; per-edit total 76–78 ms wall [MEAS, `86_…md:214-215`].
   - Scaling with scene size is unmeasured; the worklog states O(patches × rays) for the flight and O(patches²) for Ft [MEAS/FACT, `86_…md:260`].
5. **Reused across an edit.**
   - Frozen parameters, `mat_states`, `rw`, `bin_dirs` [FACT].
   - The fixed patch layout and material map [FACT].
   - Non-dirty state rays [FACT].
   - The second view reuses the state (22 ms GPU for 64 spp) [MEAS, `86_…md:216`].
6. **STALE control.** It uses T0's whole `State` (rays and patch tables) with the current source (`ecro86.py:105-108`) [FACT].

## (5) M64, and what C2_srcS1 substitutes

### M64 definition
- `Mq = DG.local_matrices()[1] = Quad(realization_us(86001), M_ray 65536, seed_ray 87001, M_dense 16384, seed_dense 87002).enc_matrix()` [FACT].
  - `experiments/ecro_attr92/diag92.py:133-139`
  - `experiments/gm1_explicit87/protocol/gm1_explicit87.json:36-39`
  - `experiments/gm1_explicit87/xport87.py:38-41`
- Equation: `Mq[m, bi, bo, c] = (1/64) Σ_{cells j ∈ bi} fbar_m(w_j)[bo, c]` (`quad87.py:240-242`) [FACT].
  - Cell centres: 8 × 8 strata per incident bin, `z = (iz + (sz+½)/8)/4`, `φ = (iφ + (sp+½)/8)·2π/16 − π` (`quad87.py:48-56, 29-30`) [FACT].
- `fbar_m(w)[bo] = (1/ΔΩ) ∫_{v ∈ bo} f_m(w, v) dv`, the bin-averaged BRDF without cosine, in 1/sr [FACT].
  - Estimated with M_dense = 16,384 one-sample MIS draws per cell, seed 87002 + 10m.
  - The MIS mixture is ½ ER-sampling about w + ½ uniform hemisphere; each draw contributes `f / p_mix` into the bin of v (`quad87.py:68-82, 85-110, 223-231`) [FACT].
  - Lambert entries are exactly ρ/π [FACT].
  - Patch-centre flat frame for every material, including sphere material 2. No local-horizon normalisation in this table (`quad87.py:15-16`) [FACT].
- Shape [6, 64, 64, 3], full 64 × 64, no rank truncation [FACT].
- `mq_ext` appends Lambert rows `Mq[3]/0.7 · albedo` for WL94's new ids (`mat94.py:231-235`) [FACT].
- Application: `ex(x)[n, bo] = Σ_bi x[n, bi] · Mq[PMAT_n, bi, bo]`, where x is bin irradiance (s or E1) (`mat94.py:246`; `eval93b.py:220-222`) [FACT].
- `flux_w`: `z_b = (iz + ½)/4`, and `M·z_bi·z_bo` is the flux-transfer form used in the metrics (`diag92.py:114-131`) [FACT].
- Within-bin weighting:
  - M64 weights incident cells uniformly in solid angle, while s and E1 carry the cosine.
  - The two agree only if incident radiance is uniform over the bin and cos varies little within a ring [INF].
  - For a point light, all of a point's irradiance comes from one direction, but M64 spreads it over the whole incident bin. That blurs the S1 mirror peak across outgoing bins [INF].

### Arm definitions in `arm_tables`
- Code: `mat94.py:238-257` [FACT].
- `ap(x) = a·(c·x)` is the deployed learned local operator. `ex(x)` is M64.
- `Bl = ap(s)` (= the model's Bq). `Bm = ex(s)`.
- `S_arm = rcv(flight(src))`. `G = apply(use_single=False)`, the deployed learned global term. `B_arm = S_arm + G` (`mat94.py:244, 282`) [FACT].
- C0 is algebraically the model's own single flight [INF from code]. T_TABLES_REPRO reproduces WL93D tables to ≤ 1e-5 (`mat94.py:516-528`) [FACT].

### What C2_srcS1 substitutes
- **Sender table.**
  - S1 panel patches (`PSID == 5`, `mat94.py:271`) use `Bm = Σ_bi s[n, bi] Mq[1, bi, bo]`; all other patches keep the learned `Bl`.
  - So the exact frozen-ER bin-projected once-scattered sender radiance replaces the learned rank-16 `a cᵀ` only at the panel sender [FACT].
- **Unchanged:**
  - the 16-point, patch-averaged, bin-quantised encoding s (so this is not the true once-scattered q, unlike `xport87.qtable` at `quad87.py:268-338`) [FACT]
  - the 8-ray flight with sender patch × bin lookup [FACT]
  - the learned receiver `ap` [FACT]
  - the learned global term G, which carries the ≥ 3-event panel paths [FACT]
  - the readout [FACT]
- **Linearity check.** `srcS1 + srcNonS1 = src + C0`, tested to ≤ 1e-5 (`mat94.py:526-529`) [FACT].

### Related measurements
- WL94, CK2 untextured ROI_T E_H, C0 → C2_srcS1: 0.047 → 0.031 (s8601) and 0.042 → 0.019 (s8602); noise bound 0.0084 (`docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md:154-159`) [MEAS].
- On the panel face, C2_srcS1 moves the S bias from −0.34 to +0.13, but relative L1 stays 0.40 (`94_…md:234`) [MEAS].
- Flux-norm structure of M64 S1 and the WL86 frozen generators (not the OWN checkpoints): rank-16 residual 0.46–0.52; rank for 99 % energy 46–59; WL86 frozen `a cᵀ` residual 0.88–0.89 with mirror-bin retention 0.11–0.14 (`docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md:137-139`) [MEAS].
- Explicit comparator, not ECRO: the explicit-vs-oracle gap is the angular projection of once-scattered L1 in the first gather (97–102 % isolated; angular ≫ spatial) (`docs/worklogs/90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md:164-178, 183`) [MEAS]. In WL87, the BINQ / ENC / RAYSEND controls did not explain the deficit (E2 T1 C: 0.046 / 0.054 / 0.059 vs EXP 0.049) (`docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md:222-231`) [MEAS]. Whether these apply to ECRO's learned tables is [HYP].
