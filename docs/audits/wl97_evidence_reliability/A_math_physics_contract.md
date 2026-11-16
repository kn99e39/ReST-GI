# WL97 Track A - Mathematical and physical contract audit (independent reviewer report)

Date: 2026-11-16. Repository `c:/Projects/ReST-GI`, branch `Selective_Recompute`, HEAD `efcf86a`. Server checks: LabServer63,
`~/rest_x97/A/` (own files only), frozen code `~/rest_x96a/code` (commit a1718f7).

This report continues an interrupted first attempt of the same track (action log `results/audit97/A/_interrupted_run_log.md`).
Its scripts were re-used after a spot check; one of its working results was found WRONG and is discarded here (section 6, item D1).

Labels: **[MEAS]** = number I produced or re-derived in this audit; **[CODE]** = read from executable code; **[STORED]** = number read
from a batch artifact without recomputation; **[INF]** = my inference.

---

## 1. Scope and method

Track A reconstructs the transport equations that the code actually implements (not the docstrings) and checks them against
physics and against the way WL86-WL96A interpreted them. Seven areas were assigned: (1) ER convention and cosine factors,
(2) source / directional encoding, (3) patch and bin semantics, (4) the ECRO operator and the ownership split, (5) event-order
accounting, (6) passivity claims, (7) visibility handling. Special checks: `an94.W` 8 x 8 vs the true 4 x 16 weights, the M64
teacher vs the exact directional response, e2 / e3 counting and `_p` / `_o` tagging, earlier sharp interactions inside e3_o.

Method: targeted code reading of the frozen modules, plus five independent checks in my own files:

| id | script (experiments/audit97/A/) | what it settles | where it ran |
|---|---|---|---|
| a1 | `a1_bins_m64_deliv.py` | bin layout, exact bin weights, `an94.W` ratio; stored-delivery ratios per receiver group | local numpy |
| a2 / a2b | `a2_m64_er.py`, `a2b_convergence.py` | numpy port of the ER closure (`er_np.py`) vs the stored M64, block by block | local numpy (first attempt) |
| a4 | `a4_m64_ring0.py` | the grazing M64 rows with a quadrature that resolves the lobe; discrete vs physical albedo | local numpy |
| a3 | `a3_ck2_delivery.py` | deterministic recomputation of the CK2 light -> S1 panel -> box A delivery: exact, binned (two bin-mean conventions), stored M64 | local numpy |
| a5 | `a5_learned_passivity.py` | distribution of the learned local operator's directional albedo (WL92A OWN models) on CK2 | server, 1 short GPU job (< 1 min, < 2 GB) |

Nothing was trained or tuned; no existing file was modified; no commit was made.

---

## 2. The implemented equations (reconstructed from code)

Notation: patch-local frame (t, b, n); 64 bins = 4 rings uniform in z = cos(theta) x 16 azimuth sectors, `bin = iz * 16 + iphi`,
`iphi = floor((atan2(y, x) + pi) * 16 / 2pi)`; every bin has solid angle DOMEGA = 2pi / 64; ZB = ring-centre cosine
(0.125, 0.375, 0.625, 0.875); W_T = ZB * DOMEGA.

**ER closure** [CODE `fnto_compiled83/closure83_dr.py:121-155`]. `eval(wi, wo)` returns `f(wi, wo) * cos(wo)`: the cosine of the
SECOND argument (`co_ = max(wo.z, 0)` multiplies every term, lines 126-127, 139, 151). `f` is symmetric (reciprocity of my numpy
port to 6e-15 [MEAS a2]). `sample(wi, ...)` returns `wo` and `eval / pdf`.

**Path tracer** [CODE `ecro86/pt86.py`]. At every vertex `wi = to_local(-ray.d)` (towards the viewer, line 222); next-event estimation
uses `cl.eval(wi, wo_light)`, so the cosine is the light-side cosine, times `I / d^2` (lines 249-259). Continuation multiplies the
throughput by `eval / pdf` of the sampled direction (lines 272-284). No Russian roulette. This is a correct point-light path tracer.

**Field (table target)** [CODE `pt86.py:327-351`]. For every (patch, bin): a random patch point and a direction uniform in solid angle
inside the bin; the probe vertex (k = 1) does NO next-event estimation; vertices k = 2..5 do. `B[patch, bin]` is therefore the
solid-angle bin mean of the outgoing radiance made of 2..5 scattering events (the scattering at the patch itself counts),
normalised by the count of valid probes. It is a piecewise-constant radiance per (patch, bin), NOT cosine-weighted.

**Camera read-out** [CODE `pt86.py:228-233, 260-271, 300-302`]. 'ecro' mode: kmax = 2; e1 = NEE at x1, e2 = NEE at x2,
`hl = throughput(x1) * B[patch(x2), bin(x2 -> x1)]`. Since B holds 2..5 events at x2, hl holds 3..6 events at the camera = the
reference component h (3 <= k <= D = 6). No double count (B excludes the 1-event radiance, which e2 carries) and no omission.

**State rays** [CODE `pt86.py:114-127, 353-376`]. K = 512 = 64 bins x kappa 8. The offsets `u_state` have shape (4, 1, K) and are
broadcast to all patches: every patch uses the SAME 512 local directions and the same patch-point offsets (seed 86001). A ray
records the hit patch (-1 for a miss or a back face) and the bin of the reversed direction in the hit patch's frame.
Flight weight of ray r: `rw[r] = z_r * DOMEGA / kappa` with the ray's own cosine (`model86.py:40-46`, `a95.py:61-64`,
`quad87.py:201-204` identical).

**Source encoding** [CODE `pt86.py:378-402`]. `s[patch, bin_in] = (1 / 16) sum over 16 fixed patch points of I cos / d^2` (shadowed),
accumulated into the incident bin of the light direction: irradiance by incident bin. The 16 point offsets are also shared by all
patches (shape (2, 1, 16), same generator).

**Explicit operators (WL87)** [CODE `gm1_explicit87/quad87.py`].
- `fbar(m, w)[bo] = (1 / DOMEGA) int_bo f(v, w) dv` (lines 68-110): one-sample MIS (ER sampling + uniform hemisphere);
  `val = eval(w, v) / (v.z * p_mix)` (line 81), i.e. the cosine that `eval` carries is divided out. Correct estimator of the
  solid-angle bin mean of f.
- `W_ray[m, r, bo] = fbar(m, w_r)[bo] * cos(w_r) * DOMEGA / kappa` (lines 198-204); Lambert `rho / pi * cos * DOMEGA / kappa`.
- `qtable` (lines 268-338): `q[i, bo] = mean over P patch points of sum_l E_l(x) fbar(m_i, w_l(x))[bo]`, exact light directions.
- **M64** = `enc_matrix` (lines 240-242): `M64[m, bin_in, bo] = mean over the 8 x 8 cell centres of bin_in of fbar(m, w_cell)[bo]`.
  No cosine weight over the incident cells (the cosine is in s).
- Recurrence `L_{k+1} = scatter(gather(L_k))` (`xport87.py:49-73`): gather reads the binned table at (hit, bin) per ray; scatter
  applies W_ray. Sender side is binned; receiver side uses each ray's exact incident direction.

**Exact sender (WL95A X1)** [CODE `ecro_attr95a/a95.py:90-135`]. Re-traces each state ray, evaluates
`eval(wi = towards the receiver, wo = towards the light) * I / d^2` with a shadow ray: `f * E` at the exact hit point and direction.

**ECRO operator** [CODE `ecro86/model86.py:91-124`]. Per patch n and colour channel: `M_n[bin_in, bo] = sum_r c[n, bin_in, ., r] a[n, bo, ., r]`
(R = 16, softplus, non-negative).
`Bq = M s` (latent 1-event radiance); `E1 = flight(Bq)` with `rw`; `B = M E1 + a P (sum_j area_j Q_j c_j s_j)`.
- 'single' = `M_rcv . flight . M_snd . s` = exactly 2 events; 'global' = low-rank term for 3..5 events. The SAME factor pair (a, c)
  of a patch acts as sender operator, receiver operator and decoder of the global term.
- The operator is applied ONCE. There is no recursion over the learned operator anywhere in `apply`.
- WL92A 'OWN' objective [CODE `ecro_own92a/own92a.py:172-176`]: `(Bs + Bg - t)`, `(Bs - t2)`, `(Bg - (t - t2))`, with t = 2..5 events and
  t2 = exactly 2 events. `Bq` (the 1-event sender radiance) is never a supervised quantity; it is identified only through the
  product `M_rcv . flight . M_snd`.

**Event split of the references** [CODE `ecro86a/pt86a.py:73-200`]. Vertices are counted from the receiver: k = 1 probe point,
k = 2 the surface the probe sees. The flag `v2p = (material at vertex 2 == 1)` is set once, at k = 2 (line 177), and applied to
all later contributions (lines 178-180):

| table | content |
|---|---|
| e2_p | exactly 2 events, vertex 2 (the last sender before the receiver) has the S1 material |
| e2_o | exactly 2 events, vertex 2 has any other material (incl. floor ER and sphere S2) |
| e3_p | 3..5 events, vertex 2 has the S1 material; earlier vertices arbitrary |
| e3_o | 3..5 events, vertex 2 not S1; **earlier vertices arbitrary, including the S1 panel, the sphere and the floor** |

With the WL94 per-patch material patch (`ecro_mat94/mat94.py:150-178`) the flag follows the per-patch material id; on CK2 'none'
this is the panel.

---

## 3. Findings

Each finding: id, category, material, claim affected, original vs independent number, evidence, decision-level impact, proposed
verdict, confidence.

### A-01 Cosine / 1/pi / solid-angle / kappa accounting is consistent everywhere I traced it (confirmation)

- Category: confirmation. Material: yes (it removes a whole class of explanations for the negative chain). P0: 1-4.
- Evidence [CODE]: section 2. Every use of `eval` puts the light-side (or sampled-side) direction in the second argument or divides
  the cosine out (`quad87.py:81`); all flights use `cos * DOMEGA / kappa`; exitance uses W_T, which equals the exact integral of
  cos over a bin (max abs difference 0.0, sum = pi) [MEAS a1].
- Evidence [MEAS]: Lambert state through my port: `eval = rho / pi * cos(wo)` to 1.7e-8 (a2); WL95A's own Dr.Jit test T_COS
  1.6e-7 [STORED `results/ecro_attr95a/s0.json`]. My numpy port of the closure reproduces the stored Dr.Jit M64 block sums to
  0.1-0.3 % in all 16 ring blocks (a2b for 15 blocks, a4 for ring0 -> ring0), so port and closure agree, and a Lambert operator has
  albedo rho under W_T (0.7 -> 0.7, a1).
- No missing or doubled cosine, 1/pi, DOMEGA or kappa factor was found in pt86, pt86a, quad87 (fbar, W_ray, qtable, enc_matrix),
  xport87.XOp, model86 (rw, apply), a95 (flight, receive, x1_radiance, exitance), w96a (explicit_op, tail).
- Proposed verdict: VERIFIED. Confidence: high (code + numerical agreement), limited to the modules listed.

### A-02 `an94.W` assumes 8 rings x 8 sectors; the true layout is 4 x 16 (evaluation defect, already disclosed by WL95 / WL95A; effect on grazing sender fields was understated)

- Category: evaluation_defect. Material: yes for S1 sender-field energy numbers of WL94D; no for Lambert-receiver exitance and for
  image metrics.
- Truth [CODE]: `scene86.py:15-20` (NZ, NPHI from the protocol: 4, 16), `pt86.py:153-156, 195-199`. `an94.py:19` hard-codes
  `NB, NPHI, NZ = 64, 8, 8`; `an94.py:32` builds `W = ((b // 8) + 0.5) / 8 * 2pi / 64`.
- [MEAS a1] W94 / W_T per true ring: sectors 0-7 (local y < 0) x0.50, x0.83, x0.90, x0.93; sectors 8-15 x1.50, x1.17, x1.10, x1.07.
  Ring sums are equal. Consequences in kind:
  - exactly unbiased for any field that is constant over azimuth within a ring, hence for every Lambert outgoing table
    (delivered exitance on Lambert receivers, e.g. box A: WL95A measured 1.208 vs 1.211 [STORED]);
  - biased for azimuthally asymmetric fields, strongest in the grazing ring: a field concentrated in ring 0 on one azimuth side is
    weighted x0.5 or x1.5.
- Consumers (every import of `an94.W`, `an94.ex`, `an94.bin_group_amp`, `an94.table_metrics` or a hard-coded copy), found by grep
  over WL86-WL96A code:

  | batch | location | quantity | change in kind |
  |---|---|---|---|
  | WL94 | `ecro_mat94/an94.py:35-37, 76-81, 104, 131-147` (`ex`, `response`, `bin_group_amp`, reference shares) | per-patch exitance of model / reference tables, log response amplitudes, bin-group amplitudes | exact for Lambert patches (ratios on the same mask cancel); biased for ER patches (floor, sphere, panel) |
  | WL94A | `ecro_mat94r/r94.py:69, 118` via `AN.table_metrics` | same | same |
  | WL94B | `ecro_exp94b/b94.py:47, 86` (`W = AN.W`, auxiliary exitance `ex` of the audit files), `:289, 372` | auxiliary exitance; table metrics | same. The G_EXPOSURE gate (1.23x vs 1.25x) uses the loss-normalised masses `nm / nt` (`b94.py:83-85, 158-160, 198-200`), NOT W: **unaffected** |
  | WL94C | `ecro_q94c/q94.py:222` via table metrics / `face_ext` | same | same |
  | WL94D | `ecro_s94d/s94.py:129, 157, 287` | G_PHYS exitance ratio / corr; S1 sender field rel L1, energy ratio; CK2 flight floor | Lambert-receiver numbers stand; **S1 sender field numbers move** (below) |
  | WL95 | `results/wl95_calc/wl95_calc.py:17, 124, 140` (and copy under `ecro_attr95a/wl95calc/`) | W94-weighted reduced-rank fits and field metrics (W_T variants reported next to them) | deliberate bridge to the old numbers; the `rrr*_W94` operators were FIT with the wrong weight |
  | WL95A | `ecro_attr95a/a95.py:28, 198-224, 310` | reproduction of WL94D only; all new numbers use W_T (`a95.py:26-27`) | corrected |
  | WL96A | `ecro_util96a/w96a.py:32` | corrected ZB; `an94` used only for masks / image metrics (no W) | not affected |
  | WL86-WL93D | `ecro86.py:157`, `analyze86a.py:46`, `diag92.py:119`, `xport87.py:131`, `rel88.py`, `rel89.py`, `tests9x` ... | all use `S.NPHI`, `S.NZ` | correct; none imports an94 |

- Magnitude on the CK2 S1 sender fields [STORED `results/ecro_attr95a/main.json`, both weights computed by WL95A on identical fields]:
  learned sender vs M64, energy ratio W_T -> W94: s8601 0.433 -> 0.557, **s8602 0.891 -> 1.123 (the sign of the energy error
  flips)**; rel L1 1.041 -> 1.075 and 1.118 -> 1.221; NMF16 energy 0.458 -> 0.617.
  WL95's statement "the bias is <= 0.02 field L1 and <= 0.04 energy" (worklog 95, line 225) was measured on pooled training /
  validation fields of bound arms and does not hold for grazing learned-sender fields (up to 0.10 L1, 0.23 energy).
- Decision-level impact: no rel-L1-based verdict changes (values stay >= 1). Any WL94D statement about the DIRECTION of the S1
  sender energy error under the old weight is unreliable (the s8602 example flips). The WL94B exposure gate does not use W [CODE]. The
  WL94 / 94A / 94C response-amplitude tables on ER patches were not recomputed (inputs not local).
- Proposed verdict: IMPLEMENTATION_DEFECT (evaluation), decision impact limited; the later batches' disclosure is accurate about the
  defect but understated its size on grazing fields. Confidence: high for the defect and the consumer list (grep-complete for the
  listed directories), medium for "no verdict flips" (not every WL94x number was recomputed).

### A-03 The stored M64 is an accurate bin average of the ER closure, but "bin average" embeds three approximations; it is not the exact directional response and it is not flux-conserving at grazing incidence

- Category: confirmation (accuracy of the stored operator) + dataset_limitation (what it is a teacher of). Material: yes
  (M64 is used as teacher in WL94D, as the yardstick of the WL95 rank bounds and as comparator A4 in WL95A).
- [MEAS a2b, a4] Independent deterministic recomputation of M64[S1] with my numpy port: block sums mine / stored = 0.998-1.001 for
  15 of the 16 (ring_in, ring_out) blocks (midpoint rule); for the ring0 -> ring0 block a lobe-resolving log-azimuth quadrature
  (256 x 1536 per incident direction) gives row sums 0.993-1.002 and per-row rel L1 0.2-0.5 % (incident bins 0, 5, 11; rings 1
  and 3 controls 0.998-1.001). The stored operator (MIS Monte Carlo, 16,384 samples per incident cell) is therefore the quantity
  its definition states. Reciprocity asymmetry of the stored S1 operator 0.5 % (noise level) [MEAS a1].
- What the definition embeds [CODE `quad87.py:48-56, 85-110, 240-242`]:
  1. incident side: unweighted mean of f over 64 cell centres of the incident bin = "irradiance uniformly spread over the bin's
     solid angle". (WL86A's 'enc' / 'ang' probes use the other convention, `f * cos / zb`, `pt86a.py:140-141, 157-158`: radiance
     uniform in the bin. The two encoding controls are not the same operator. For S1 ring-0 rows the physical albedo differs by
     only 0.593 vs 0.603, so this inconsistency is minor.)
  2. outgoing side: SOLID-ANGLE mean of the radiance, later converted to flux with the bin-centre cosine. For a lobe that leaves at
     cos < ZB the discrete system emits more energy than the physical lobe.
  3. patch-constant position.
- [MEAS a4] S1, incident ring 0 (green): M64 row albedo under W_T 0.683 vs physical directional albedo 0.593 (cell mean): the
  discrete operator over-emits by 15 % on average, by 43 % for the most grazing cell (cos 0.016) and by 2 % at cos 0.23.
  Rings 1 and 3: 1.00-1.03 and 0.92-1.02. M64 stays passive under the discrete criterion (max row albedo 0.825; a1 reproduces the
  WL95A number exactly).
- [MEAS a3] For the actual CK2 light on the panel (all irradiance in incident ring 0, cos_i 0.025-0.154, irradiance-weighted mean
  0.047): emitted flux of the binned sender table / physical exitance = **1.30** (exact light directions, infinite samples) and
  **1.22** (stored M64 applied to s). Physical albedo of the panel for this light 0.47 / 0.58 / 0.56 (RGB).
- Decision-level impact: "distance to M64" is not "distance to the physical response" on grazing workloads. The WL94D teacher
  experiment taught a target that over-emits 22 % on CK2 and over-delivers ~1.5x to box A (A-04). Operator-level bounds of WL95
  measured against M64 inherit this.
- Proposed verdict for "M64 = physically exact directional response": INVALID_COMPARISON where M64 is used as ground truth for
  delivery on CK2; VERIFIED for "M64 = exact bin average as defined". Confidence: high.

### A-04 P0-2 reproduced without any path tracer or state ray: the binned sender over-delivers to box A by ~1.4-1.5x deterministically; about half of that excess is the solid-angle-mean convention, the rest is within-bin / within-patch structure

- Category: confirmation (+ invalid_inference risk on the magnitude attribution). Material: yes. P0-2, P0-3.
- Independent computation [MEAS a3]: `E(y) = int_panel L(x -> y) cos cos / r^2 dA` by midpoint quadrature over the panel
  (42 patches x 40^2 points) and box-A patches (3^2 points), light -> panel irradiance analytic, ER through my numpy port.
  Box A is convex and unoccluded from the panel; exitance = rho_A * E; compared with the stored path-traced reference
  (`results/ecro_attr95a/deliv.npz`, ref_a / ref_b = e2_p halves, half-noise 0.059).

  | sender model (infinite samples) | box A flux / path-traced ref | abs rel L1 vs ref | corr | WL95A pooled counterpart [STORED] |
  |---|---|---|---|---|
  | exact direction + exact point | **1.015** | **0.068** | 0.995 | X1: 1.022 / 0.111 (SE 0.041) |
  | binned table, solid-angle bin mean (= `qtable`, `field` convention) | **1.417** | 0.429 | 0.950 | q256: 1.422 / 0.433 |
  | stored M64 applied to s (32^2 source points) | **1.490** | 0.491 | 0.972 | M64 pooled source: 1.495 / 0.500; M64(s16): 1.544 / 0.546 |
  | binned table, cos-weighted (flux-conserving) bin mean - audit counterfactual | 1.210 | 0.239 | 0.973 | not tested in WL95A |

  Resolution check (NS 4, NP 64, NR 4 instead of 5 / 40 / 3): exact 1.015 / 0.068; solid-angle mean 1.411 / 0.423; stored M64
  1.485 / 0.486; cos-weighted mean 1.204 / 0.236. The table is stable to <= 0.006 in ratio.
- Original vs independent: q256 1.422 vs 1.417; M64 1.495 vs 1.490; X1 1.022 vs 1.015. Agreement within 0.5 % on the binned arms.
- Interpretation [INF]: (i) the over-delivery of the binned sender is a deterministic projection bias, not relation sampling:
  confirmed, because my computation has no relation samples at all. (ii) WL95A attributes the loss to "the patch-constant and
  bin-constant outgoing representation" (worklog 95A line 149). With the SAME 64 bins and patches, replacing the solid-angle mean
  by the flux-conserving mean lowers the excess from +40 % to +19 % relative to my exact value (L1 0.41 -> 0.23). So roughly half
  of the reported bias magnitude belongs to the bin-mean convention (energy inflation 1.30x, A-03), and roughly half is a genuine
  resolution limit of piecewise-constant bins / patches for this grazing lobe. WL95A did not separate the two.
- 30 / 44 / 22 / 4 % of the exact box-A irradiance leaves the panel in outgoing rings 0 / 1 / 2 / 3: box A is lit mostly by
  off-specular tails, not by the mirror peak.
- Decision-level impact: P0-2 as worded stands on CK2 box A. The statement "evidence for an angular representation floor of the
  patch x 64-bin carrier" stands qualitatively (a +19 % / 0.24 L1 bias remains under the alternative projection), but its
  magnitude (0.43-0.55 L1, "+40-50 %") is specific to one projection convention. The counterfactual is a measurement on one
  workload, not a repair and not a proposal.
- Scope limits: one scene, one light 5 cm in front of the panel plane (every incident direction at cos < 0.16 on a material with
  GGX alpha 0.026-0.075), one receiver object, the exactly-2-event S1-mediated component only.
- Proposed verdict: P0-2 VERIFIED on the tested workload (CONDITIONALLY_VALID as a statement about "the 64-bin carrier": the
  magnitude is convention-dependent). Confidence: high for the reproduction, medium for the 50 / 50 split (one workload, my
  counterfactual only).

### A-05 P0-3 reproduced: exact direction and hit point close the box-A delivery error to the reference noise level - in expectation; not at the deployed relation budget and not resolved on other receivers

- Category: confirmation with restrictions. Material: yes. P0-3.
- [MEAS a3] exact = 1.015 flux ratio, 0.068 abs rel L1 against the path-traced reference whose own half-split noise is 0.059,
  corr 0.995. This also validates the reference itself (`pt86a.field_src` e2_p on box A) with a method that shares no code with it.
- Restrictions [STORED `results/ecro_attr95a/an.json`, `deliv.npz`; MEAS a1 for the group breakdown]:
  - single relation draw (K = 512, the deployed budget): X1 median abs rel L1 0.62, worst 2.2; the deployed X1 draw (0.627) is
    WORSE than the deployed binned M64 draw (0.524). The closure needs pooling (64 draws = 32,768 rays per receiver patch) or a
    deterministic quadrature such as mine.
  - other receiver groups: walls / ceiling X1 pooled 1.23 +- 0.40, all non-panel 1.14 +- 0.18 (not resolved; binned M64 there
    1.03-1.26 +- 0.03-0.05); sphere 0.91 +- 0.02 and box B 0.88 +- 0.06 (significantly low with an exact sender: receiver-side
    binned response or reference noise, not attributed).
  - only the 2-event S1-mediated component.
- Proposed verdict: CONDITIONALLY_VALID (valid for box A in expectation / converged quadrature). Confidence: high.

### A-06 P0-1 passivity: the criterion is correct and the reported maxima reproduce exactly, but they are a worst-row statistic; on the tested CK2 input the learned S1 sender emits no more energy than the exact binned operator (albedo 0.19-0.65), and the deployed operator is never iterated

- Category: confirmation of the number + invalid_inference risk on its reading. Material: yes. P0-1.
- Criterion [CODE `a95.py:147-152`, derivation]: for an operator mapping irradiance-by-incident-bin to radiance-by-outgoing-bin,
  `alb[bin_in] = sum_bo M[bin_in, bo] W_T[bo]` is the exitance per unit irradiance of the piecewise-constant system (W_T is the exact
  integral of the cosine over a bin). `alb <= 1` is the right discrete passivity criterion; a Lambert operator gives exactly rho
  [MEAS a1]; the exact operators give <= 0.825 (M64 rows) and <= 0.85 for single incident directions (sampled bins of rings 0, 1, 3) [MEAS a4].
  (Discard note: the first attempt's value "true bin-mean operator albedo 1.25 / single direction 4.1" came from an aliased
  quadrature and is wrong; see D1.)
- [MEAS a5, server, production functions] WL92A OWN models, S1 panel patches on CK2:

  | | s8601 | s8602 | M64 | physical (a3 / a4) |
  |---|---|---|---|---|
  | max row albedo (42 patches x 64 bins x 3 ch) | 2.2637 (= WL95A) | 3.0774 (= WL95A) | 0.825 | <= 0.85 (ring 0: <= 0.62) |
  | mean / median row albedo | 0.561 / 0.506 | 0.552 / 0.456 | 0.635 | - |
  | rows with albedo > 1 | 10.4 % | 15.4 % | 0 | 0 |
  | patches with at least one row > 1 | 42 / 42 | 42 / 42 | - | - |
  | rows that carry CK2 irradiance (189, all ring 0): share > 1, max | 4.2 %, 1.44 | 41 %, 1.79 | 0 | 0 |
  | CK2 irradiance-weighted albedo (R / G / B) | 0.19 / 0.35 / 0.30 | 0.46 / 0.65 / 0.64 | 0.60 / 0.68 / 0.68 | 0.47 / 0.58 / 0.56 |

  - The learned S1 albedo depends strongly on the incident AZIMUTH (ring 0, green, mean over patches: 0.04-1.7 across the 16
    sectors) although S1 is azimuthally isotropic (M64: 0.68 in every sector). [INF] The factor pair behaves as a fit to the
    incident-direction distribution of the training lights, not as a material law.
  - Lambert patches (walls, rho 0.7): irradiance-weighted learned albedo 0.699 / 0.697 (both seeds), mean row albedo 0.71 / 0.72,
    rows > 1: 0.1 % / 1.5 %, none on lit rows. Box A (rho 0.5): 0.54-0.58; box B (rho 0.6): 0.56-0.58. The learned local
    operator is energetically right on the simple materials.
  - Every one of the 64 x 3 incident (bin, channel) rows of the panel was excited at least once in the WL94D training audit
    (`train_bins.npy` all true), so the over-unity rows cannot be assigned to never-excited inputs with that mask.
- Structural facts [CODE]: (i) `model86.apply` applies the local operator twice in series (sender, receiver) and never iterates it,
  so over-unity rows cannot cause a divergent solve in the deployed ECRO (unlike FNTO WL41-WL51); (ii) the 1-event operator is a
  latent quantity that no WL92A loss term supervises (`own92a.py:172-176`).
- Decision-level impact: "non-passive" is true row-wise and should not be read as "the deployed operator creates energy on the
  tested workload": for the CK2 light the emitted / incident energy is 0.19-0.35 (s8601: 0.3-0.5x of M64, 0.4-0.6x of the
  physical albedo) and 0.46-0.65 (s8602: 0.8-0.95x of M64, 1.0-1.15x of the physical albedo). The observed over-delivery of
  s8602 to box A (1.63x) is therefore mainly a redistribution in direction, not an energy gain of the sender.
- Proposed verdict: "may violate passivity" VERIFIED as a worst-row property; CONDITIONALLY_VALID as a statement about deployed
  behaviour (restrictions above). Confidence: high (exact reproduction of both maxima).

### A-07 P0-1 "large avoidable error": "large" is supported against the path-traced reference; "avoidable" is not tested, and two of the comparisons use M64, which is itself biased on this workload

- Category: invalid_inference (if P0-1 is asserted as worded) + evaluation_defect (comparator). Material: yes. P0-1 and the WL95A
  RANK16 verdict.
- Large [STORED, checked against the reference in a1]: learned sender delivery to box A 0.517x (s8601) / 1.630x (s8602) of the
  path-traced e2_p reference, abs rel L1 0.569 / 0.965, corr 0.78; exact sender 1.02x / 0.11. Supported.
- Avoidable: WL95A's own text says "Avoidable within rank 16 on this workload? Not shown" (worklog 95A line 143); the phrase in
  P0-1 is the PROPOSITION of WL95A (line 6), not its result. My check: inside the patch x 64-bin carrier the exact bin-averaged
  operator M64 delivers with abs rel L1 0.546 (pooled) to the same receivers - the same size as the learned s8601 sender (0.569).
  No experiment shows a sender in this carrier with a small delivery error on CK2.
- Comparator issue [CODE `a95.py:494-512`]: the RANK16_DELIVERY rule requires a bound arm to reproduce the delivery of A4 = M64
  (flux within 10 %, map L1 <= 0.10) and to have an image error not worse than A4's. A4's own pooled delivery is 1.544x the
  reference. Against the reference [MEAS a1 from `deliv.npz`]: A1 (NMF16) 0.975x / L1 0.655; A3 (rank 32) 1.127x / 0.386;
  A4 (M64) 1.544x / 0.546. Image E_H [STORED]: A1 0.0209, A3 0.0204, A4 0.0245 (floor 0.0118, tolerance 0.0167). The two failed
  legs of the rule (c2, c3) therefore measure "distance to a biased comparator"; the reference-based leg (c4) passes for all arms.
  The operator-level statement "rank-16 non-negative fits to the training distribution do not represent the grazing S1 rows
  (peak <= 0.02)" does not depend on this and I did not re-derive it.
- The latent status of the sender (A-06) matters for the reading: the learned sender error is an identification failure of an
  unsupervised factor as much as a capacity statement; WL94D (teacher on Bq, with the biased M64 as target and the W94 weight in
  its metrics) is the only batch that supervised it.
- Decision-level impact: P0-1 should read "large error source in delivery; avoidability within the tested carrier NOT
  established". RANK16 'INADEQUATE' is valid as "does not reproduce M64's delivery", not as "worse than M64 against the reference".
- Proposed verdict: "large" VERIFIED; "avoidable" UNTESTED_INFERENCE; RANK16 delivery legs INVALID_COMPARISON (comparator), overall
  CONDITIONALLY_VALID. Confidence: high for the comparator facts, medium for the consequences (operator-level rank evidence not
  re-derived by me).

### A-08 Event-order accounting is consistent between references, explicit recurrence, ECRO terms and the camera read-out; e3_o does contain earlier sharp interactions and WL96A says so (confirmation)

- Category: confirmation. Material: yes for the reading of P0-4.
- [CODE] section 2: table target = 2..5 events; camera hl = 3..6 events; e2 = exactly 2; e3 = 3..5; ECRO 'single' has exactly the
  event content of e2 = e2_p + e2_o and 'global' that of e3; WL92A pairs them accordingly; normalisation (solid-angle bin mean of
  radiance) is the same on both sides. WL96A explicit tail: `e3 = T(L2 + L3 + L4)` with L2 = reference e2, the last gather split by
  the material of the hit patch (`w96a.py:101-122`), which matches the `v2p` predicate of the reference.
- e3_o = "last sender is not S1". It contains light -> panel -> X -> receiver paths and paths whose last sender is the glossy
  floor or sphere. WL96A's worklog states exactly this (worklog 96A line 26). No discrepancy between code and text.
- Consequence for P0-4 [INF]: the explicit tail starts from the path-traced exact e2 table of all patches. The sharp first hop
  (light -> panel -> receiver, the quantity of P0-2 / P0-3) is therefore GIVEN to both the explicit recurrence and the
  hypothetical learned resolvent; "no headroom over explicit recurrence" is a statement about propagating a correct 2-event field
  through the binned operator, not about a self-contained explicit method.
- Proposed verdict: event accounting VERIFIED; P0-4 within this lens CONDITIONALLY_VALID (exact e2 prefix; CK2; e3_o domain).
  Confidence: high for the accounting.

### A-09 The 512 state rays and the 16 source points use one offset pattern shared by all patches (observation)

- Category: statistical_weakness (design property, disclosed implicitly). Material: no on its own.
- [CODE `pt86.py:114-127`]: `u_state` (4, 1, K) and `u_src` (2, 1, 16) are broadcast over patches. All patches of a receiver face
  sample the same local directions, so a relation draw is a coherent (not independent) quadrature across patches: per-draw
  delivery errors are spatially correlated, which is consistent with the heavy per-draw spread WL95A reports (X1 single-draw
  L1 0.62; deployed draw of A4 1.21x vs pooled 1.54x). The CK2 light is 5 cm in front of the panel plane and 0.3 m off its edge,
  so 16 shared points per 20 cm panel patch give a noisy s (WL95A reports this separately as 'source_draw_var').
- Impact: single-draw ("deployed") numbers on CK2 are draws of a high-variance, spatially coherent estimator; conclusions should
  rest on pooled numbers, as WL95A's do. No verdict proposed (NONE).

### A-10 Visibility handling (confirmation)

- [CODE] state rays start at `p + n * 1e-4`, test `dot(w, n_point) > 1e-3`, record a hit only for front faces
  (`pt86.py:359-369`); the outgoing bin is taken in the hit patch's frame (`patch_bin`, lines 158-181). qtable and source use shadow
  rays (`quad87.py:288-292`, `pt86.py:391-394`). X1 re-traces the stored segment and requires the panel shape (`a95.py:101-110`).
  Sphere patches use the patch-centre frame with per-point horizon tests and valid-solid-angle normalisation, the same in the
  field and in qtable (`pt86.py:342-345`, `quad87.py:298-307, 329-330`).
- I found no visibility inconsistency relevant to CK2 box A. Verdict: VERIFIED by code reading only (no numerical test of the
  sphere path). Confidence: medium.

---

## 4. Priority-0 assessment within the Track A lens

| claim | assessment | restrictions |
|---|---|---|
| P0-1 learned S1 sender has large avoidable error; deployed operator may violate passivity | "large": VERIFIED vs the path-traced reference. "avoidable": UNTESTED_INFERENCE (WL95A itself: "not shown"; exact binned M64 delivers with the same L1). "may violate passivity": VERIFIED as worst-row albedo (2.26 / 3.08 reproduced exactly); CONDITIONALLY_VALID as deployed behaviour | row-max over 8,064 rows; 10-15 % of rows; CK2 irradiance-weighted albedo 0.19-0.65 (<= the exact binned operator's 0.60-0.68); operator never iterated; sender is an unsupervised latent factor; M64 comparisons are against a biased comparator |
| P0-2 M64 / 64-bin sender introduces systematic sharp-delivery bias not eliminated by relation sampling | VERIFIED on CK2 box A by a computation without relation samples (1.42 / 1.49 vs WL95A 1.42 / 1.49-1.54) | one scene / light / receiver; about half of the excess is the solid-angle-mean bin convention (energy inflation 1.30x), not the bin resolution |
| P0-3 exact-direction and hit-point evaluation substantially closes the CK2 delivery error | CONDITIONALLY_VALID: reproduced deterministically (1.015 / L1 0.068 at reference noise 0.059) | in expectation / converged quadrature only; a single K = 512 draw is worse than the binned deployed draw; resolved for box A only |
| P0-4 learned broad higher-order resolvent has no meaningful headroom over explicit recurrence on CK2 | Event accounting VERIFIED; claim CONDITIONALLY_VALID within this lens | explicit recurrence is seeded with the path-traced exact e2 table; domain e3_o = last sender not S1 (still contains earlier sharp vertices). Accuracy / cost numbers not re-derived by this track |
| P0-5 negative results justify rejecting the tested ECRO mechanisms, not a general impossibility | Consistent with what the contracts show: every sharp-delivery number above is specific to (a) one piecewise-constant projection convention, (b) rank-16 tied factors with an unsupervised sender, (c) one grazing workload. Nothing in the equations supports a general impossibility statement | within this lens only; the "rejecting the tested mechanisms" half depends on findings A-03, A-04, A-07 being carried as restrictions |

---

## 5. Special checks

| check | result | evidence |
|---|---|---|
| `an94.W` 8 x 8 vs true 4 x 16 | Defect confirmed; ratio x0.5 / x1.5 in the grazing ring by azimuth half, ring sums equal; consumer list in A-02; Lambert-receiver exitance unaffected; S1 sender-field energy moves by up to 0.23 (one sign flip) | a1 JSON; `an94.py:19, 32`; `main.json` sender block |
| M64 teacher vs exact directional response | Stored M64 = accurate bin average (<= 0.5 % per grazing row). It averages f uniformly over 8 x 8 incident cell centres (no cosine) and over the outgoing bin's solid angle; it is not flux-conserving at grazing incidence (+15 % row albedo in ring 0; +22 % emitted flux for the CK2 light) and over-delivers 1.49x to box A | a2b, a4, a3 |
| e2 / e3 counting, `_p` / `_o` tagging | Consistent: e2 = exactly 2 events, e3 = 3..5; tag = material of the vertex adjacent to the receiver, set once at k = 2 | `pt86a.py:102-200` |
| does e3_o contain earlier sharp interactions | Yes (panel, sphere, floor at vertices >= 3; floor / sphere also as last sender). WL96A states it | `pt86a.py:176-180`; worklog 96A line 26 |
| ER cosine convention | `eval(wi, wo) = f cos(wo)` (second argument); every caller consistent | A-01 |
| passivity criterion | `sum_bo M W_T <= 1` is correct for irradiance-by-bin -> radiance-by-bin; Lambert gives rho | A-06, a1 |

---

## 6. Discarded or corrected working results of the first attempt

- **D1 (discarded).** `a2_m64_er.json` reports "mine vs stored M64" ring0 -> ring0 block ratio 2.20, `albedo_max_WT_mine` 1.249,
  `table_albedo_single_direction_max` 4.12 and the `flux_conservation_S1` ratios. These come from a 24 x 24 midpoint rule whose
  nodes coincide with the mirror direction of every incident cell centre, while the grazing S1 lobe is ~1e-3 rad wide in azimuth
  (a2b shows non-convergence at cos 0.016). They are quadrature artefacts. The resolved quadrature (a4) gives 0.993-1.002 and
  albedo 0.68. Only the non-grazing blocks and the convention / reciprocity tests of a2 are used in this report. The file is kept
  unchanged as a record.
- The delivery group table of a1 and the bin-weight results were spot-checked (W ratios re-derived by hand; A4 box A 1.544 and X1
  1.022 equal `an.json`) and are used.

---

## 7. Not verified, and why

- Sphere-patch and floor-receiver paths (curved frames, receiver-side M64): code reading only.
- WL94 / 94A / 94B / 94C numbers under the corrected weight: fields and tables are not in the local results; the direction of the
  change is given in kind (A-02). The WL94B exposure gate (1.23x vs 1.25x) in particular was not recomputed.
- WL95 operator-level rank bounds (peak <= 0.02, op L1 0.233): read, not re-derived.
- WL96A accuracy and cost numbers, image metrics (`an94.eh`), render pipeline timing: outside this track's checks; only the event
  content of the compared tables was verified.
- The learned global term (P, Q) and the full-table energy balance of ECRO: not analysed.
- WL86-WL91 (gm1 relation batches) beyond the shared modules quad87 / xport87: bin weights grep-checked only.
- The cos-weighted counterfactual of A-04 was evaluated on one workload only; no claim is made about other scenes.
- Dr.Jit closure itself was not re-executed for the convention test; agreement of my port with the stored Dr.Jit operator
  (0.1-0.5 %) and WL95A's stored T_COS are the evidence.

---

## 8. Reproducibility appendix

Inputs (sha256, first 16 hex):

```
d3c5be854d80665b  experiments/ecro86/data/mats86.npz                       (ER states; S1 = state 1)
d4d02e5a2f423ce9  results/ecro_attr95a/ops95a.npz                          (A1..A4; A4 = M64[S1])
f2324a32effc80ca  results/ecro_attr95a/deliv.npz                           (per-draw deliveries, ref_a / ref_b)
d8f4ebdc72d78472  results/ecro_attr95a/an.json
28d104991bcad6f7  results/ecro_attr95a/main.json
977ba122ab3232e7  experiments/ecro_local93d/addendum_checker/scene_ck2.json
7110b5da503d4638  experiments/ecro_mat94/an94.py
df3a14c22e05e37c  experiments/ecro_attr95a/a95.py
cfa9c5c4e854526c  experiments/gm1_explicit87/quad87.py
912e68acd314d79e  experiments/ecro86/pt86.py
cfc109cb61c76eb2  experiments/ecro86a/pt86a.py
72a728b81e490fb7  experiments/ecro86/model86.py
3eee59095fc77fdb  experiments/fnto_compiled83/closure83_dr.py
739e7eb50b25d329  LabServer63:~/rest_x92a/results/models/own_s8601.pt
cee9a09a289fca8c  LabServer63:~/rest_x92a/results/models/own_s8602.pt
d2462ddb0f6064c6  LabServer63:~/rest_x94d/results/audit94d/train_bins.npy
```

Outputs of this audit (results/audit97/A/):

```
43ce085c3ff7dcee  a1_bins_m64_deliv.json
2bcc48fcd024e350  a2_m64_er.json                (ring0 block and flux_conservation_S1 entries DISCARDED, see D1)
49c01f91eafd2622  a2b_convergence.json
755fb1452a06493b  a4_m64_ring0_z256_p1536.json
2fc03fb491c6e823  a3_ck2_delivery_ns5_np40_nr3.json
12a084ad4412a9e7  a3_ck2_delivery_ns4_np64_nr4.json
7ea05899aad36eb5  a5_learned_passivity.json     (server copy identical)
```

Commands (repository root, local python 3.11 + numpy):

```
python experiments/audit97/A/a1_bins_m64_deliv.py                 # -> results/audit97/A/a1_bins_m64_deliv.json
python experiments/audit97/A/a2_m64_er.py                         # -> a2_m64_er.json, a2_m64_mine.npz (ring0 block: see D1)
python experiments/audit97/A/a2b_convergence.py                   # -> a2b_convergence.json
python experiments/audit97/A/a4_m64_ring0.py 256 1536             # -> a4_m64_ring0_z256_p1536.json (bins 0, 5, 11, 19, 55)
python experiments/audit97/A/a3_ck2_delivery.py 5 40 3            # -> a3_ck2_delivery_ns5_np40_nr3.{json,npz}  (~7 min)
python experiments/audit97/A/a3_ck2_delivery.py 4 64 4            # resolution check
```

Server (LabServer63, one GPU process, < 1 min, < 2 GB; nothing written outside `~/rest_x97/A`):

```
scp experiments/audit97/A/a5_learned_passivity.py LabServer63:rest_x97/A/
cd ~/rest_x97/A && mkdir -p R/results out && . ~/rest_x95a/env.sh
$PY a5_learned_passivity.py ~/rest_x97/A/R ~/rest_x92a ~/rest_x94d/results/audit94d/train_bins.npy ~/rest_x97/A/out/a5_learned_passivity.json
# copied back to results/audit97/A/a5_learned_passivity.json and a5_server.log
```

Seeds: a2 `default_rng(97001)` (convention test directions); a3 / a4 are deterministic quadratures; a5 uses the frozen WL92A OWN
checkpoints `own_s8601.pt`, `own_s8602.pt` and the deployed state realization (table_seed 86001).
Quadrature parameters: a3 sender tables 256 (z) x 1536 (log-azimuth, scale 3e-4 rad, centred on the mirror azimuth) per light
direction, NS^2 light directions per panel patch; delivery NP^2 panel points per patch, NR^2 points per box-A patch.
